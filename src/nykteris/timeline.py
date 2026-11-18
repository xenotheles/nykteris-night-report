"""Put frames, the .analyze session and the guide log on one UTC timeline,
then attach per-exposure context to every frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .ekos_analyze import AnalyzeSession, tz_hours_from_abbrev
from .frames import Frame
from .guidelog import GuideLog


@dataclass
class Night:
    frames: list[Frame]
    sessions: list[AnalyzeSession] = field(default_factory=list)
    guide: GuideLog | None = None
    utc_offset_h: float | None = None
    offset_source: str = ""
    notes: list[str] = field(default_factory=list)
    autofocus: list = field(default_factory=list)  # (t_end, temperature, success)
    flips: list[float] = field(default_factory=list)  # completion times
    dithers: list[float] = field(default_factory=list)


def _round_quarter(hours: float) -> float:
    return round(hours * 4.0) / 4.0


def match_captures(frames: list[Frame], sessions: list[AnalyzeSession]) -> dict[str, object]:
    by_name = {}
    for s in sessions:
        for c in s.captures:
            by_name[c.basename] = c
    return {f.name: by_name[f.name] for f in frames if f.name in by_name}


def align(frames: list[Frame], sessions: list[AnalyzeSession], guide: GuideLog | None,
          utc_offset_h: float | None = None) -> Night:
    night = Night(frames=frames, sessions=sessions, guide=guide)
    matches = match_captures(frames, sessions)

    if utc_offset_h is not None:
        night.utc_offset_h = utc_offset_h
        night.offset_source = "given on the command line"
    elif matches:
        diffs = [c.t - frames_by_name.date_end for frames_by_name, c in
                 ((f, matches[f.name]) for f in frames if f.name in matches)]
        night.utc_offset_h = _round_quarter(float(np.median(diffs)) / 3600.0)
        night.offset_source = f"derived from {len(matches)} captures matched to FITS DATE-OBS"
    else:
        for s in sessions:
            h = tz_hours_from_abbrev(s.tz_abbrev)
            if h is not None:
                night.utc_offset_h = h
                night.offset_source = f"from the .analyze time zone '{s.tz_abbrev}'"
                break
    if night.utc_offset_h is None:
        night.utc_offset_h = 0.0
        night.offset_source = "unknown, assumed UTC"
        if guide is not None:
            night.notes.append("Could not determine the local time offset; guide log times assumed to be UTC. "
                               "Use --utc-offset if the guide data looks shifted.")

    shift = -night.utc_offset_h * 3600.0
    for s in sessions:
        s.shift(shift)
    if guide is not None:
        guide.shift(shift)

    for s in sessions:
        for af in s.autofocus:
            if af.success:
                night.autofocus.append((af.t_end, af.temperature))
        for t, state in s.flip_states:
            if state == "MOUNT_FLIP_COMPLETED":
                night.flips.append(t)
        night.dithers += [t for t, st in s.guide_states if st == "Dithering"]
    if guide is not None:
        night.dithers += guide.dithers
    night.autofocus.sort()
    night.flips.sort()
    night.dithers.sort()

    if sessions and not matches:
        night.notes.append("No CaptureComplete filename in the .analyze file matched a FITS file; "
                           "Ekos measurements were not attached.")
    for f in frames:
        c = matches.get(f.name)
        if c is not None:
            f.ctx["ekos_hfr"] = c.hfr
            f.ctx["ekos_stars"] = c.stars
            f.ctx["ekos_ecc"] = c.eccentricity
    _attach_context(night)
    return night


def _interp(t, ts, vs):
    if len(ts) == 0:
        return None
    if t <= ts[0]:
        return float(vs[0]) if t >= ts[0] - 1800 else None
    if t >= ts[-1]:
        return float(vs[-1]) if t <= ts[-1] + 1800 else None
    return float(np.interp(t, ts, vs))


def _step(t, ts, vs):
    """Value of a step function (last value at or before t)."""
    if len(ts) == 0:
        return None
    i = int(np.searchsorted(ts, t, side="right")) - 1
    return vs[i] if i >= 0 else None


def _computed_altitude(f: Frame, t: float) -> float | None:
    if None in (f.ra_deg, f.dec_deg, f.site_lat, f.site_lon):
        return None
    jd = t / 86400.0 + 2440587.5
    gmst = (280.46061837 + 360.98564736629 * (jd - 2451545.0)) % 360.0
    ha = math.radians(((gmst + f.site_lon - f.ra_deg) + 540.0) % 360.0 - 180.0)
    lat, dec = math.radians(f.site_lat), math.radians(f.dec_deg)
    s = math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(ha)
    return math.degrees(math.asin(max(-1.0, min(1.0, s))))


def airmass(alt_deg: float | None) -> float | None:
    if alt_deg is None or alt_deg <= 0:
        return None
    return 1.0 / (math.sin(math.radians(alt_deg)) + 0.50572 * (alt_deg + 6.07995) ** -1.6364)


def _attach_context(night: Night) -> None:
    sess = night.sessions
    g_t = np.concatenate([np.asarray(s.guide_t) for s in sess]) if sess else np.zeros(0)
    g_ra = np.concatenate([np.asarray(s.guide_ra) for s in sess]) if sess else np.zeros(0)
    g_dec = np.concatenate([np.asarray(s.guide_dec) for s in sess]) if sess else np.zeros(0)
    g_snr = np.concatenate([np.asarray(s.guide_snr) for s in sess]) if sess else np.zeros(0)
    order = np.argsort(g_t)
    g_t, g_ra, g_dec, g_snr = g_t[order], g_ra[order], g_dec[order], g_snr[order]

    c_t = np.concatenate([np.asarray(s.coords_t) for s in sess]) if sess else np.zeros(0)
    c_alt = np.concatenate([np.asarray(s.coords_alt) for s in sess]) if sess else np.zeros(0)
    c_side = np.concatenate([np.asarray(s.coords_side, int) for s in sess]) if sess else np.zeros(0, int)
    o = np.argsort(c_t)
    c_t, c_alt, c_side = c_t[o], c_alt[o], c_side[o]

    tp_t = np.concatenate([np.asarray(s.temp_t) for s in sess]) if sess else np.zeros(0)
    tp_v = np.concatenate([np.asarray(s.temp_v) for s in sess]) if sess else np.zeros(0)
    o = np.argsort(tp_t)
    tp_t, tp_v = tp_t[o], tp_v[o]

    guide = night.guide
    source = None
    if guide is not None and len(guide):
        source = "guide log"
    elif len(g_t):
        source = ".analyze GuideStats"

    prev_focus_pos = None
    af_times = [a[0] for a in night.autofocus]
    for f in night.frames:
        t0, t1 = f.date_obs, f.date_end
        mid = 0.5 * (t0 + t1)
        ctx = f.ctx

        # Guiding during the exposure.
        ctx["guide_source"] = source
        if source == "guide log":
            m = (guide.t >= t0) & (guide.t <= t1)
            ok = m & ~guide.dropped & np.isfinite(guide.ra) & np.isfinite(guide.dec)
            ra, dec, snr = guide.ra[ok], guide.dec[ok], guide.snr[m]
            n_drop = int((m & guide.dropped).sum())
        elif source:
            m = (g_t >= t0) & (g_t <= t1)
            ra, dec, snr = g_ra[m], g_dec[m], g_snr[m]
            n_drop = 0
        else:
            ra = dec = snr = np.zeros(0)
            n_drop = 0
        ctx["guide_n"] = int(len(ra))
        ctx["guide_dropped"] = n_drop
        if len(ra) >= 3:
            ctx["guide_rms_ra"] = float(np.std(ra))
            ctx["guide_rms_dec"] = float(np.std(dec))
            ctx["guide_rms"] = float(math.hypot(ctx["guide_rms_ra"], ctx["guide_rms_dec"]))
            ctx["guide_peak"] = float(np.max(np.hypot(ra - ra.mean(), dec - dec.mean())))
        else:
            ctx["guide_rms"] = None
        snr = snr[np.isfinite(snr)] if len(snr) else snr
        ctx["guide_snr"] = float(np.median(snr)) if len(snr) else None

        # Altitude and pier side.
        alt = _interp(mid, c_t, c_alt) if len(c_t) else None
        alt_src = "mount"
        if alt is None:
            alt = _computed_altitude(f, mid)
            alt_src = "computed"
        if alt is None and f.header_alt is not None:
            alt, alt_src = f.header_alt, "header"
        ctx["altitude"] = alt
        ctx["altitude_source"] = alt_src if alt is not None else None
        ctx["airmass"] = airmass(alt)
        side = _step(mid, c_t, c_side) if len(c_t) else None
        ctx["pier_side"] = {0: "WEST", 1: "EAST"}.get(int(side)) if side is not None else f.pier_side

        # Focuser temperature.
        temp = _interp(mid, tp_t, tp_v) if len(tp_t) else None
        if temp is None:
            temp = f.focus_temp
        ctx["temperature"] = temp

        # Focus history.
        i = int(np.searchsorted(af_times, t0, side="right")) - 1
        if i >= 0:
            ctx["since_af_s"] = t0 - night.autofocus[i][0]
            ctx["af_temperature"] = night.autofocus[i][1]
            ctx["af_time"] = night.autofocus[i][0]
        else:
            ctx["since_af_s"] = None
            ctx["af_temperature"] = None
            ctx["af_time"] = None
        ctx["focus_moved"] = prev_focus_pos is not None and f.focus_pos is not None and f.focus_pos != prev_focus_pos
        prev_focus_pos = f.focus_pos if f.focus_pos is not None else prev_focus_pos

        # Meridian flip.
        j = int(np.searchsorted(night.flips, t0, side="right")) - 1
        ctx["since_flip_s"] = t0 - night.flips[j] if j >= 0 else None

    # Pier side change relative to the previous frame (FITS-only fallback for flips).
    prev = None
    for f in night.frames:
        side = f.ctx.get("pier_side")
        f.ctx["pier_changed"] = prev is not None and side is not None and side != prev
        if side is not None:
            prev = side
