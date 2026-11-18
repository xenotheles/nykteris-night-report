"""N.I.N.A. inputs: Session Metadata plugin files and autofocus reports.

Session Metadata plugin (tcpalmer/nina.plugin.sessionmetadata) writes
``ImageMetaData.csv`` and/or ``ImageMetaData.json`` next to the images, one
record per image. Fields used here: ``FilePath``, ``ExposureStartUTC``
(ISO 8601, UTC), ``ExposureStart`` (local, to the minute), ``Duration``,
``DetectedStars``, ``HFR``, ``FWHM``, ``Eccentricity``,
``GuidingRMSArcSec`` (+ RA/DEC; 0 when not guiding), ``FocuserPosition``,
``FocuserTemp``, ``PierSide`` (East/West), ``Airmass``, ``ImageType``.

Autofocus reports (``%LOCALAPPDATA%\\NINA\\AutoFocus\\*.json``, written by
``AutoFocusReport`` in NINA.WPF.Base): ``Timestamp`` (local, with offset),
``Temperature``, ``Filter``, ``CalculatedFocusPoint.Position``,
``Duration``.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .frames import _iso


@dataclass
class ImageRecord:
    basename: str
    t_start: float | None  # UTC unix seconds
    local_start: float | None  # local wall time "as if UTC", minute resolution
    duration: float | None
    stars: int | None = None
    hfr: float | None = None
    fwhm: float | None = None
    eccentricity: float | None = None
    guide_rms: float | None = None
    guide_rms_ra: float | None = None
    guide_rms_dec: float | None = None
    focuser_pos: float | None = None
    focuser_temp: float | None = None
    pier_side: str | None = None
    airmass: float | None = None
    image_type: str = ""


@dataclass
class NinaSession:
    paths: list[Path] = field(default_factory=list)
    images: list[ImageRecord] = field(default_factory=list)

    def by_name(self) -> dict[str, ImageRecord]:
        return {r.basename: r for r in self.images}


def _f(v) -> float | None:
    if v is None or v == "":
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _basename(p: str) -> str:
    return str(p).replace("\\", "/").rsplit("/", 1)[-1]


def _record(row: dict) -> ImageRecord | None:
    path = row.get("FilePath") or ""
    if not path:
        return None
    t = None
    if row.get("ExposureStartUTC"):
        try:
            t = _iso(row["ExposureStartUTC"]).timestamp()
        except ValueError:
            t = None
    loc = None
    if row.get("ExposureStart"):
        try:
            loc = _iso(row["ExposureStart"]).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            loc = None
    rms = _f(row.get("GuidingRMSArcSec"))
    stars = _f(row.get("DetectedStars"))
    pier = (row.get("PierSide") or "").strip().upper() or None
    return ImageRecord(
        basename=_basename(path),
        t_start=t,
        local_start=loc,
        duration=_f(row.get("Duration")),
        stars=int(stars) if stars is not None else None,
        hfr=_f(row.get("HFR")),
        fwhm=_f(row.get("FWHM")),
        eccentricity=_f(row.get("Eccentricity")),
        # The plugin writes 0 when no guiding data was recorded.
        guide_rms=rms if rms else None,
        guide_rms_ra=_f(row.get("GuidingRMSRAArcSec")) or None,
        guide_rms_dec=_f(row.get("GuidingRMSDECArcSec")) or None,
        focuser_pos=_f(row.get("FocuserPosition")),
        focuser_temp=_f(row.get("FocuserTemp")),
        pier_side=pier if pier in ("EAST", "WEST") else None,
        airmass=_f(row.get("Airmass")),
        image_type=(row.get("ImageType") or "").strip(),
    )


def parse_image_metadata(paths) -> NinaSession:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    s = NinaSession(paths=[Path(p) for p in paths])
    seen = set()
    for p in s.paths:
        if p.suffix.lower() == ".json":
            rows = json.loads(p.read_text(encoding="utf-8-sig"))
            if isinstance(rows, dict):
                rows = [rows]
        else:
            with open(p, newline="", encoding="utf-8-sig") as fh:
                rows = list(csv.DictReader(fh))
        for row in rows:
            r = _record(row)
            if r is not None and r.basename not in seen:
                seen.add(r.basename)
                s.images.append(r)
    return s


@dataclass
class AutofocusReport:
    t: float  # UTC unix seconds (report written when the run finished)
    temperature: float | None
    filter: str
    position: float | None
    path: Path


def parse_autofocus_reports(paths) -> list[AutofocusReport]:
    out = []
    for p in paths:
        try:
            d = json.loads(Path(p).read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        if not isinstance(d, dict) or "Timestamp" not in d:
            continue
        try:
            ts = _iso(d["Timestamp"])
        except ValueError:
            continue
        if ts.tzinfo is None:  # pragma: no cover
            ts = ts.astimezone()
        cfp = d.get("CalculatedFocusPoint") or {}
        out.append(AutofocusReport(
            t=ts.timestamp(),
            temperature=_f(d.get("Temperature")),
            filter=str(d.get("Filter") or ""),
            position=_f(cfp.get("Position")) if isinstance(cfp, dict) else None,
            path=Path(p),
        ))
    out.sort(key=lambda a: a.t)
    return out


def utc_offset_hours(session: NinaSession) -> float | None:
    """Offset from records holding both local and UTC start times."""
    diffs = [r.local_start - r.t_start for r in session.images if r.local_start is not None and r.t_start is not None]
    if not diffs:
        return None
    diffs.sort()
    med = diffs[len(diffs) // 2]
    # ExposureStart is truncated to the minute; add half a minute before rounding.
    return round((med + 30.0) / 900.0) / 4.0


def is_report(path: Path) -> bool:
    try:
        with open(path, encoding="utf-8-sig") as fh:
            head = fh.read(400)
    except OSError:
        return False
    return '"Timestamp"' in head and ('"CalculatedFocusPoint"' in head or '"AutoFocuserName"' in head or '"Method"' in head)


