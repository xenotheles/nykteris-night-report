"""Cause rules. Each rule is a small, explainable heuristic.

A frame gets a verdict (``ok``, ``reject`` or ``check``) and a list of
causes, most likely first. Every cause carries a sentence with the numbers
that triggered it, so the report can say why.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import Thresholds
from .timeline import Night

CAUSES = {
    "clouds": "Clouds or haze",
    "guiding": "Guiding error",
    "elongation": "Elongated stars, guiding normal",
    "seeing": "Seeing or wind",
    "focus": "Focus drift",
    "altitude": "Low altitude",
    "trail": "Satellite or plane trail",
    "flip": "Meridian flip",
    "blur": "Blurred, cause unclear",
    "no_stars": "No stars detected",
}

ADVICE = {
    "clouds": "Clouds cannot be fixed at the rig. If Ekos aborted nothing, consider the Scheduler's "
              "weather or HFR/star-count based aborts so cloudy subs are not saved.",
    "guiding": "Check balance, cable drag and backlash, and look at the guide graph around these times. "
               "If spikes repeat at a fixed period, periodic error correction may help.",
    "elongation": "Stars were elongated in one direction while guiding looked normal. Typical causes are "
                  "differential flexure between guide scope and main scope, wind gusts, or a loose focuser.",
    "seeing": "Short-lived blur with round stars and normal guiding points to seeing or wind. "
              "Nothing to change at the rig; these subs can be dropped or down-weighted.",
    "focus": "Focus drifted as the temperature changed. Enable an autofocus trigger on temperature change "
             "(for example every 1 degree C) or every N subs in the Ekos Focus module.",
    "altitude": "The target was low. Plan the session to end earlier, or start the target closer to transit.",
    "trail": "Single trails are removed by sigma-clipping during stacking; these subs are usually fine to keep.",
    "flip": "The first sub after the meridian flip was degraded. Increase the guiding settle time after the flip, "
            "or add a short delay before capture resumes.",
    "blur": "Stars were blurred without a clear pattern. Look at the sub by eye.",
    "no_stars": "No stars were found. The sub may be fully clouded, the dew cap may have failed, or the mount "
                "may have lost the target.",
}


@dataclass
class Baseline:
    fwhm: float
    stars: float
    background: float
    guide_rms: float | None
    airmass: float | None


def _median(values):
    v = [x for x in values if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else None


def baselines(night: Night) -> dict[tuple, Baseline]:
    groups: dict[tuple, list] = {}
    for f in night.frames:
        groups.setdefault(f.group, []).append(f)
    out = {}
    for key, fs in groups.items():
        out[key] = Baseline(
            fwhm=_median(f.metrics.get("fwhm_px") for f in fs) or float("nan"),
            stars=_median(f.metrics.get("stars") for f in fs) or float("nan"),
            background=_median(f.metrics.get("background") for f in fs) or float("nan"),
            guide_rms=_median(f.ctx.get("guide_rms") for f in night.frames),
            airmass=_median(f.ctx.get("airmass") for f in fs),
        )
    return out


def _fmt(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def classify(night: Night, th: Thresholds) -> dict[tuple, Baseline]:
    base = baselines(night)
    frames = night.frames

    # Focus reference: the last autofocus, or the last focuser move seen in FITS headers.
    last_ref_idx = None
    last_ref_temp = None
    fwhm_since_ref: list[float] = []

    for idx, f in enumerate(frames):
        m, c = f.metrics, f.ctx
        b = base[f.group]
        causes: list[tuple[str, str, str]] = []  # (cause, confidence, explanation)
        scale = f.scale

        stars = m.get("stars", 0)
        fwhm = m.get("fwhm_px", float("nan"))
        ecc = m.get("eccentricity", float("nan"))
        coh = m.get("ecc_coherence", float("nan"))
        bg = m.get("background", float("nan"))

        rel_fwhm = fwhm / b.fwhm if b.fwhm and np.isfinite(fwhm) else float("nan")
        rel_stars = stars / b.stars if b.stars else float("nan")
        rel_bg = bg / b.background if b.background else float("nan")
        X = c.get("airmass")
        alt = c.get("altitude")
        alt_factor = (X / b.airmass) ** th.airmass_fwhm_exponent if X and b.airmass else 1.0
        corr_fwhm = rel_fwhm / alt_factor if np.isfinite(rel_fwhm) else rel_fwhm

        g_rms = c.get("guide_rms")
        g_limit = th.guide_rms_max_arcsec
        if b.guide_rms:
            # Bad when above the absolute limit or well above this night's typical RMS.
            g_limit = min(g_limit, th.guide_rms_ratio * b.guide_rms)
        guide_bad = g_rms is not None and g_rms > g_limit
        sampled = np.isfinite(fwhm) and fwhm >= th.eccentricity_min_fwhm_px
        elongated = sampled and np.isfinite(ecc) and ecc > th.eccentricity_max and \
            (not np.isfinite(coh) or coh >= th.elongation_coherence)
        blurred = np.isfinite(rel_fwhm) and rel_fwhm > th.fwhm_ratio
        starved = np.isfinite(rel_stars) and rel_stars < th.cloud_star_ratio
        bright = np.isfinite(rel_bg) and rel_bg > th.background_ratio
        degraded = blurred or elongated or starved or guide_bad

        c.update(rel_fwhm=rel_fwhm, rel_stars=rel_stars, rel_bg=rel_bg, corr_fwhm=corr_fwhm,
                 guide_limit=g_limit, elongated=bool(elongated), sampled=bool(sampled))
        fwhm_as = f" ({fwhm * scale:.1f}\")" if scale and np.isfinite(fwhm) else ""

        # Refocus bookkeeping (before judging this frame).
        af_t = c.get("af_time")
        new_af = af_t is not None and (idx == 0 or frames[idx - 1].ctx.get("af_time") != af_t)
        if idx == 0 or c.get("focus_moved") or new_af:
            last_ref_idx = idx
            last_ref_temp = c.get("af_temperature") if af_t is not None else c.get("temperature")
            fwhm_since_ref = []
        fwhm_since_ref.append(fwhm)

        if stars == 0 or not np.isfinite(fwhm):
            causes.append(("no_stars", "high", "No usable stars were detected in this sub."))
        else:
            # Meridian flip: degraded sub right after a flip or a pier side change.
            since_flip = c.get("since_flip_s")
            after_flip = (since_flip is not None and since_flip <= th.flip_settle_s) or c.get("pier_changed")
            if after_flip and degraded:
                when = f"{since_flip / 60:.0f} min after the flip completed" if since_flip is not None else \
                    "first sub on the new pier side"
                detail = []
                if guide_bad:
                    detail.append(f"guide RMS {_fmt(g_rms)}\" had not settled")
                if elongated:
                    detail.append(f"stars elongated (eccentricity {_fmt(ecc)})")
                if blurred:
                    detail.append(f"FWHM {rel_fwhm:.2f}x the median")
                causes.append(("flip", "high", f"Started {when}; " + ", ".join(detail or ["quality dropped"]) + "."))

            # Low altitude explains blur and background when the airmass correction absorbs it.
            low = alt is not None and alt < th.altitude_min_deg
            if (blurred or bright or starved) and low and corr_fwhm <= th.fwhm_ratio:
                causes.append(("altitude", "high",
                               f"Altitude {alt:.0f} deg (airmass {X:.2f}); FWHM {rel_fwhm:.2f}x and background "
                               f"{rel_bg:.2f}x the median, as expected this low."))

            # Clouds: stars lost while FWHM stays normal.
            if starved and corr_fwhm <= th.fwhm_ratio and not any(k == "altitude" for k, *_ in causes):
                ev = [f"star count {rel_stars:.0%} of the median", f"FWHM normal ({rel_fwhm:.2f}x)"]
                conf = "medium"
                if bright:
                    ev.append(f"background up {rel_bg - 1:.0%}")
                    conf = "high"
                gs = c.get("guide_snr")
                if gs is not None and c.get("guide_dropped"):
                    ev.append(f"guide star lost {c['guide_dropped']} times")
                    conf = "high"
                text = "; ".join(ev)
                causes.append(("clouds", conf, text[0].upper() + text[1:] + ": thin cloud or haze passing."))

            # Guiding.
            if not any(k == "flip" for k, *_ in causes):
                if elongated and guide_bad:
                    causes.append(("guiding", "high",
                                   f"Stars elongated (eccentricity {ecc:.2f}, angle {m.get('elong_angle_deg', 0):.0f} deg) "
                                   f"and guide RMS {g_rms:.2f}\" during the exposure (limit {g_limit:.2f}\")."))
                elif guide_bad and blurred:
                    causes.append(("guiding", "medium",
                                   f"Guide RMS {g_rms:.2f}\" (limit {g_limit:.2f}\") with FWHM {rel_fwhm:.2f}x "
                                   f"the median: guiding error blurred the stars."))
                elif elongated:
                    gtxt = f"guide RMS {g_rms:.2f}\" looked normal" if g_rms is not None else "no guide data"
                    causes.append(("elongation", "medium",
                                   f"Stars elongated in one direction (eccentricity {ecc:.2f}, "
                                   f"angle {m.get('elong_angle_deg', 0):.0f} deg), {gtxt}."))

            # Blur with round stars: altitude, focus drift or seeing.
            taken = {k for k, *_ in causes}
            if blurred and not taken & {"flip", "guiding", "elongation", "altitude"}:
                if corr_fwhm <= th.fwhm_ratio:
                    causes.append(("altitude", "medium",
                                   f"FWHM {rel_fwhm:.2f}x the median{fwhm_as}, explained by airmass {X:.2f} "
                                   f"at altitude {alt:.0f} deg."))
                else:
                    temp = c.get("temperature")
                    dtemp = (temp - last_ref_temp) if temp is not None and last_ref_temp is not None else None
                    k = th.focus_trend_frames
                    trend = False
                    if len(fwhm_since_ref) > k:
                        # A trend is a steady climb: no step down, no single big step.
                        recent = np.array(fwhm_since_ref[-(k + 1):])
                        steps = np.diff(recent) / recent[:-1]
                        trend = bool(np.all(steps > -0.03) and np.all(steps < 0.15)) and recent[-1] > 1.1 * recent[0]
                    previous = [frames[j].metrics.get("fwhm_px") for j in range(max(0, idx - 3), idx)]
                    previous = [n for n in previous if n is not None and np.isfinite(n)]
                    jump = bool(previous) and fwhm > 1.15 * float(np.median(previous)) and not trend
                    if (dtemp is not None and abs(dtemp) >= th.focus_temp_delta_c and not jump) or trend:
                        bits = [f"FWHM {rel_fwhm:.2f}x the median{fwhm_as}, stars round"]
                        if dtemp is not None:
                            bits.append(f"temperature {dtemp:+.1f} C since the last refocus")
                        if trend:
                            bits.append(f"FWHM rising over the last {k + 1} subs")
                        causes.append(("focus", "high" if (trend and dtemp is not None and abs(dtemp) >= th.focus_temp_delta_c) else "medium",
                                       "; ".join(bits) + "."))
                    elif g_rms is None or not guide_bad:
                        gtxt = f"guiding normal ({g_rms:.2f}\")" if g_rms is not None else "no guide data"
                        causes.append(("seeing", "medium" if jump else "low",
                                       f"FWHM {rel_fwhm:.2f}x the median{fwhm_as} with round stars and {gtxt}"
                                       + ("; a short jump, not a trend." if jump else ".")))
                    else:
                        causes.append(("blur", "low", f"FWHM {rel_fwhm:.2f}x the median."))

            # Elevated background alone (moon, dawn, light pollution) is reported but not rejected.
            if bright and not causes and rel_bg > 1.5 * th.background_ratio:
                causes.append(("altitude" if (alt is not None and alt < th.altitude_min_deg) else "clouds", "low",
                               f"Background {rel_bg:.2f}x the median with normal stars."))

        # Trails are independent of everything else.
        if m.get("trail_length_px", 0) >= th.trail_min_length_px and m.get("trail_axis_ratio", 0) >= th.trail_min_axis_ratio:
            causes.append(("trail", "high", f"A linear feature {m['trail_length_px']:.0f} px long crosses the sub."))

        f.causes = [{"cause": k, "label": CAUSES[k], "confidence": conf, "text": txt} for k, conf, txt in causes]
        if not causes:
            f.verdict = "ok"
        elif all(x["cause"] == "trail" for x in f.causes) or all(x["confidence"] == "low" for x in f.causes):
            f.verdict = "check"
        else:
            f.verdict = "reject"
    return base
