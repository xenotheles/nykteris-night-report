"""Render the night as one self-contained HTML page (inline SVG, no external files)."""

from __future__ import annotations

import csv
import io
import math
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from jinja2 import Environment, PackageLoader, select_autoescape  # noqa: E402

from . import __version__  # noqa: E402
from .rules import ADVICE, CAUSES  # noqa: E402
from .timeline import Night  # noqa: E402

# Colour per cause, chosen to stay distinguishable on light and dark backgrounds.
COLORS = {
    "ok": "#8a94a6",
    "clouds": "#4c8bf5",
    "guiding": "#e8590c",
    "elongation": "#f59f00",
    "seeing": "#2f9e44",
    "focus": "#ae3ec9",
    "altitude": "#868e96",
    "trail": "#15aabf",
    "flip": "#e03131",
    "blur": "#a61e4d",
    "no_stars": "#212529",
}


def _local(night: Night, t: float) -> datetime:
    return datetime.fromtimestamp(t, tz=timezone.utc).replace(tzinfo=None) + timedelta(hours=night.utc_offset_h or 0)


def _primary(f) -> str:
    return f.causes[0]["cause"] if f.causes else "ok"


def timeline_svg(night: Night) -> str:
    frames = night.frames
    x = [_local(night, 0.5 * (f.date_obs + f.date_end)) for f in frames]
    scale = np.array([f.scale or np.nan for f in frames])
    have_scale = np.isfinite(scale).all()
    fwhm = np.array([f.metrics.get("fwhm_px", np.nan) for f in frames]) * (scale if have_scale else 1)
    panels = [
        ("FWHM (\")" if have_scale else "FWHM (px)", fwhm),
        ("Stars", np.array([f.metrics.get("stars", np.nan) for f in frames], float)),
        ("Background (ADU)", np.array([f.metrics.get("background", np.nan) for f in frames])),
        ("Eccentricity", np.array([f.metrics.get("eccentricity", np.nan) for f in frames])),
    ]
    g = np.array([f.ctx.get("guide_rms") if f.ctx.get("guide_rms") is not None else np.nan for f in frames], float)
    if np.isfinite(g).any():
        panels.append(("Guide RMS (\")", g))
    alt = np.array([f.ctx.get("altitude") if f.ctx.get("altitude") is not None else np.nan for f in frames], float)
    if np.isfinite(alt).any():
        panels.append(("Altitude (deg)", alt))
    temp = np.array([f.ctx.get("temperature") if f.ctx.get("temperature") is not None else np.nan for f in frames], float)
    if np.isfinite(temp).any():
        panels.append(("Focuser temp (C)", temp))

    plt.rcParams.update({"font.size": 8, "svg.fonttype": "none", "font.family": "sans-serif"})
    fig, axes = plt.subplots(len(panels), 1, sharex=True, figsize=(9.5, 1.35 * len(panels) + 0.5))
    fig.patch.set_alpha(0)
    causes = [_primary(f) for f in frames]
    filters = sorted({f.filter or "-" for f in frames})
    markers = dict(zip(filters, "os^Dvp*"))
    for ax, (label, y) in zip(axes, panels):
        ax.set_facecolor("none")
        ax.plot(x, y, color="#8a94a6", lw=0.8, zorder=1)
        for flt in filters:
            for cause in sorted(set(causes)):
                idx = [i for i, f in enumerate(frames) if (f.filter or "-") == flt and causes[i] == cause]
                if not idx:
                    continue
                ax.scatter([x[i] for i in idx], y[idx], s=14 if cause == "ok" else 26, c=COLORS[cause],
                           marker=markers[flt], zorder=3, edgecolors="none")
        for t, _ in night.autofocus:
            ax.axvline(_local(night, t), color="#ae3ec9", lw=0.8, ls=":", zorder=0)
        for t in night.flips:
            ax.axvline(_local(night, t), color="#e03131", lw=0.9, ls="--", zorder=0)
        ax.set_ylabel(label, rotation=0, ha="right", va="center")
        ax.grid(alpha=0.25, lw=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#8a94a6")
        ax.tick_params(colors="#8a94a6")
        ax.yaxis.label.set_color("#8a94a6")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    off = night.utc_offset_h or 0
    axes[-1].set_xlabel(f"Local time (UTC{off:+g})", color="#8a94a6")
    fig.align_ylabels(axes)
    fig.tight_layout(h_pad=0.4)
    buf = io.StringIO()
    fig.savefig(buf, format="svg", transparent=True)
    plt.close(fig)
    svg = buf.getvalue()
    return svg[svg.index("<svg"):]


def _fmt(v, nd=2, unit=""):
    if v is None:
        return ""
    try:
        if not math.isfinite(v):
            return ""
    except TypeError:
        return str(v)
    return f"{v:.{nd}f}{unit}"


def build_context(night: Night) -> dict:
    frames = night.frames
    counts = Counter(_primary(f) for f in frames if f.verdict != "ok")
    total_s = sum(f.exptime for f in frames)
    kept_s = sum(f.exptime for f in frames if f.verdict != "reject")
    filters = Counter(f.filter or "-" for f in frames)
    t0, t1 = frames[0].date_obs, frames[-1].date_end
    objects = sorted({f.object for f in frames if f.object})

    findings = []
    for cause, n in counts.most_common():
        fs = [f for f in frames if f.verdict != "ok" and _primary(f) == cause]
        times = sorted(_local(night, f.date_obs) for f in fs)
        span = times[0].strftime("%H:%M") if len(times) == 1 else f"{times[0]:%H:%M} to {times[-1]:%H:%M}"
        findings.append({
            "cause": cause, "label": CAUSES[cause], "count": n, "span": span,
            "color": COLORS[cause], "advice": ADVICE[cause],
            "example": fs[0].causes[0]["text"],
        })

    rows = []
    for i, f in enumerate(frames):
        m, c = f.metrics, f.ctx
        rows.append({
            "n": i + 1,
            "time": _local(night, f.date_obs).strftime("%H:%M:%S"),
            "name": f.name,
            "filter": f.filter or "-",
            "exp": _fmt(f.exptime, 0),
            "stars": m.get("stars", ""),
            "fwhm": _fmt(m.get("fwhm_px", float("nan")) * f.scale, 2) if f.scale else _fmt(m.get("fwhm_px"), 2),
            "fwhm_px": _fmt(m.get("fwhm_px"), 2),
            "ecc": _fmt(m.get("eccentricity"), 2),
            "bg": _fmt(m.get("background"), 0),
            "guide": _fmt(c.get("guide_rms"), 2),
            "alt": _fmt(c.get("altitude"), 0),
            "temp": _fmt(c.get("temperature"), 1),
            "ekos_hfr": _fmt(c.get("ekos_hfr"), 2),
            "verdict": f.verdict,
            "cause": _primary(f),
            "color": COLORS[_primary(f)],
            "causes": [dict(x, color=COLORS[x["cause"]]) for x in f.causes],
        })

    inputs = getattr(night, "inputs", None)
    th = getattr(night, "thresholds", None)
    return {
        "version": __version__,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "objects": objects,
        "date": _local(night, t0).strftime("%Y-%m-%d"),
        "start": _local(night, t0).strftime("%H:%M"),
        "end": _local(night, t1).strftime("%H:%M"),
        "utc_offset": night.utc_offset_h,
        "offset_source": night.offset_source,
        "n_frames": len(frames),
        "n_reject": sum(f.verdict == "reject" for f in frames),
        "n_check": sum(f.verdict == "check" for f in frames),
        "total_h": total_s / 3600.0,
        "kept_h": kept_s / 3600.0,
        "filters": ", ".join(f"{k} x{v}" for k, v in sorted(filters.items())),
        "findings": findings,
        "svg": timeline_svg(night),
        "rows": rows,
        "flagged": [r for r in rows if r["verdict"] != "ok"],
        "legend": [(k, CAUSES.get(k, "Good"), COLORS[k]) for k in ["ok"] + [c for c in COLORS if c in counts]],
        "notes": night.notes,
        "inputs": {
            "lights": str(inputs.lights) if inputs else "",
            "analyze": [str(p) for p in inputs.analyze] if inputs else [],
            "guide": [str(p) for p in inputs.guide_logs] if inputs else [],
        },
        "guide_source": next((f.ctx.get("guide_source") for f in frames if f.ctx.get("guide_source")), None),
        "thresholds": asdict(th) if th else {},
        "n_autofocus": len(night.autofocus),
        "n_flips": len(night.flips),
    }


def render_html(night: Night) -> str:
    env = Environment(loader=PackageLoader("nykteris", "templates"), autoescape=select_autoescape(["html", "j2"]))
    return env.get_template("report.html.j2").render(**build_context(night))


def write_csv(night: Night, path: Path) -> None:
    cols = ["file", "date_obs_utc", "filter", "exptime", "stars", "fwhm_px", "hfr_px", "eccentricity",
            "ecc_coherence", "background", "noise", "guide_rms", "guide_peak", "altitude", "pier_side",
            "temperature", "ekos_hfr", "verdict", "causes"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for f in night.frames:
            m, c = f.metrics, f.ctx
            w.writerow([
                f.name, datetime.fromtimestamp(f.date_obs, tz=timezone.utc).isoformat(), f.filter, f.exptime,
                m.get("stars"), _fmt(m.get("fwhm_px"), 3), _fmt(m.get("hfr_px"), 3), _fmt(m.get("eccentricity"), 3),
                _fmt(m.get("ecc_coherence"), 3), _fmt(m.get("background"), 1), _fmt(m.get("noise"), 2),
                _fmt(c.get("guide_rms"), 3), _fmt(c.get("guide_peak"), 3), _fmt(c.get("altitude"), 2),
                c.get("pier_side") or "", _fmt(c.get("temperature"), 2), _fmt(c.get("ekos_hfr"), 3),
                f.verdict, ";".join(x["cause"] for x in f.causes),
            ])


def write_rejects(night: Night, path: Path) -> None:
    lines = [f"{f.path}\t{_primary(f)}" for f in night.frames if f.verdict == "reject"]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))
