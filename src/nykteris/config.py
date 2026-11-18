"""Thresholds for the cause rules.

All values are educated guesses until they have been checked against real
nights. Every value can be overridden from a TOML file with a
``[thresholds]`` table, see ``nykteris thresholds``.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


@dataclass
class Thresholds:
    # Blur. A sub is "blurred" when its FWHM exceeds this multiple of the
    # median FWHM of its group (same filter, exposure and binning).
    fwhm_ratio: float = 1.25

    # Clouds or haze: star count below this fraction of the group median
    # while FWHM stays normal.
    cloud_star_ratio: float = 0.6
    # Background above this multiple of the group median supports the
    # cloud verdict (moonlight or light pollution reflected by cloud).
    background_ratio: float = 1.15

    # Elongation. Median star eccentricity above this value counts as
    # elongated, but only when stars are well sampled (see next value).
    eccentricity_max: float = 0.55
    # Below this FWHM in pixels the eccentricity is not judged at all:
    # undersampled stars show high eccentricity on perfectly good nights.
    eccentricity_min_fwhm_px: float = 2.0
    # Elongation angles must agree to this degree (0 = random, 1 = all
    # the same) before the elongation is treated as tracking error.
    elongation_coherence: float = 0.5

    # Guiding. Total RMS error during the exposure (arcsec) is bad when it
    # exceeds the absolute limit or this multiple of the night median.
    guide_rms_max_arcsec: float = 1.5
    guide_rms_ratio: float = 2.0

    # Altitude below which degraded subs are attributed to the target
    # being low.
    altitude_min_deg: float = 30.0
    # Seeing FWHM grows roughly with airmass to this power.
    airmass_fwhm_exponent: float = 0.6

    # Focus drift: focuser temperature change since the last autofocus
    # (degrees C) that makes focus drift a likely cause.
    focus_temp_delta_c: float = 1.0
    # Or: FWHM rising over at least this many consecutive subs since the
    # last autofocus.
    focus_trend_frames: int = 3

    # Meridian flip: degraded subs starting within this many seconds after
    # a completed flip are attributed to the flip.
    flip_settle_s: float = 600.0

    # Satellite or plane trails: a detected object at least this long
    # (px) with at least this axis ratio.
    trail_min_length_px: float = 40.0
    trail_min_axis_ratio: float = 6.0

    @classmethod
    def from_toml(cls, path: str | Path) -> "Thresholds":
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
        table = data.get("thresholds", data)
        known = {f.name: f for f in fields(cls)}
        unknown = sorted(set(table) - set(known))
        if unknown:
            raise ValueError(f"Unknown threshold(s) in {path}: {', '.join(unknown)}")
        values = {}
        for key, value in table.items():
            ftype = known[key].type
            values[key] = int(value) if ftype in ("int", int) else float(value)
        return cls(**values)

    def to_toml(self) -> str:
        lines = [
            "# Nykteris thresholds. Values are provisional; tune them on your own nights.",
            "[thresholds]",
        ]
        for key, value in asdict(self).items():
            lines.append(f"{key} = {value}")
        return "\n".join(lines) + "\n"
