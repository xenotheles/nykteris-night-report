"""Parser for KStars/Ekos ``.analyze`` session files.

Format (from KStars ``ekos/analyze/analyze.cpp``): plain text, one event per
line, comma separated. Lines starting with ``#`` are comments. The first
event is ``AnalyzeStartTime,<yyyy-MM-dd hh:mm:ss.zzz>,<tz abbreviation>`` in
local time. Every other line is ``<EventType>,<seconds since start>,...``.

Fields used here:

- ``CaptureStarting,t,exposure,filter[,device]``
- ``CaptureComplete,t,exposure,filter,hfr,filename[,stars,median,ecc,device]``
- ``CaptureAborted,t,exposure[,device]``
- ``GuideState,t,state``
- ``GuideStats,t,raErr,decErr,raPulse,decPulse,snr,skyBg,numStars`` (arcsec)
- ``Temperature,t,temp[,device]`` (focuser temperature)
- ``MountState,t,state``
- ``MountCoords,t,ra,dec,az,alt,pierSide[,ha]`` (degrees, pier -1/0/1)
- ``MeridianFlipState,t,MOUNT_FLIP_*``
- ``AutofocusStarting,t,filter,temp,reason,reasonInfo[,device]``
- ``AutofocusComplete,t,temp,reason,reasonInfo,filter,points,weights,...`` (v2)
- ``AutofocusComplete,t,filter,points[,curve,title]`` (v1)
- ``AutofocusAborted,t,temp,reason,reasonInfo,filter,...`` (v2)
- ``AdaptiveFocusComplete,t,filter,temp,...``
- ``SchedulerJobStart,t,name`` / ``SchedulerJobEnd,t,name,reason``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

TIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"

PIER_SIDES = {-1: None, 0: "WEST", 1: "EAST"}

# Common abbreviations Qt prints for AnalyzeStartTime. Only a fallback: the
# UTC offset is normally derived by matching captures to FITS DATE-OBS.
TZ_ABBREV_HOURS = {
    "UTC": 0, "GMT": 0, "WET": 0, "WEST": 1, "BST": 1, "IST": 1, "CET": 1,
    "CEST": 2, "EET": 2, "EEST": 3, "MSK": 3, "EST": -5, "EDT": -4,
    "CST": -6, "CDT": -5, "MST": -7, "MDT": -6, "PST": -8, "PDT": -7,
    "AKST": -9, "AKDT": -8, "HST": -10, "AEST": 10, "AEDT": 11,
    "ACST": 9.5, "ACDT": 10.5, "AWST": 8, "NZST": 12, "NZDT": 13, "JST": 9,
}


@dataclass
class Capture:
    t: float  # local unix-like seconds (naive local time treated as UTC)
    exposure: float
    filter: str
    hfr: float
    filename: str
    stars: int | None = None
    median: float | None = None
    eccentricity: float | None = None

    @property
    def basename(self) -> str:
        return self.filename.replace("\\", "/").rsplit("/", 1)[-1]


@dataclass
class Autofocus:
    t_start: float | None
    t_end: float
    success: bool
    temperature: float | None
    filter: str = ""
    reason: str = ""


@dataclass
class AnalyzeSession:
    path: Path
    start_local: datetime | None = None
    tz_abbrev: str = ""
    captures: list[Capture] = field(default_factory=list)
    aborted: list[float] = field(default_factory=list)
    guide_states: list[tuple[float, str]] = field(default_factory=list)
    guide_t: list[float] = field(default_factory=list)
    guide_ra: list[float] = field(default_factory=list)
    guide_dec: list[float] = field(default_factory=list)
    guide_snr: list[float] = field(default_factory=list)
    temp_t: list[float] = field(default_factory=list)
    temp_v: list[float] = field(default_factory=list)
    mount_states: list[tuple[float, str]] = field(default_factory=list)
    coords_t: list[float] = field(default_factory=list)
    coords_alt: list[float] = field(default_factory=list)
    coords_az: list[float] = field(default_factory=list)
    coords_ha: list[float] = field(default_factory=list)
    coords_side: list[int] = field(default_factory=list)
    flip_states: list[tuple[float, str]] = field(default_factory=list)
    autofocus: list[Autofocus] = field(default_factory=list)
    scheduler: list[tuple[float, str, str]] = field(default_factory=list)
    unknown_events: dict[str, int] = field(default_factory=dict)

    def shift(self, seconds: float) -> None:
        """Add ``seconds`` to every timestamp (used to move to UTC)."""

        def sh(lst):
            return [t + seconds for t in lst]

        for c in self.captures:
            c.t += seconds
        self.aborted = sh(self.aborted)
        self.guide_states = [(t + seconds, s) for t, s in self.guide_states]
        self.guide_t = sh(self.guide_t)
        self.temp_t = sh(self.temp_t)
        self.mount_states = [(t + seconds, s) for t, s in self.mount_states]
        self.coords_t = sh(self.coords_t)
        self.flip_states = [(t + seconds, s) for t, s in self.flip_states]
        for a in self.autofocus:
            a.t_end += seconds
            if a.t_start is not None:
                a.t_start += seconds
        self.scheduler = [(t + seconds, a, b) for t, a, b in self.scheduler]

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "guide_t": np.asarray(self.guide_t, float),
            "guide_ra": np.asarray(self.guide_ra, float),
            "guide_dec": np.asarray(self.guide_dec, float),
            "guide_snr": np.asarray(self.guide_snr, float),
            "temp_t": np.asarray(self.temp_t, float),
            "temp_v": np.asarray(self.temp_v, float),
            "coords_t": np.asarray(self.coords_t, float),
            "coords_alt": np.asarray(self.coords_alt, float),
            "coords_side": np.asarray(self.coords_side, int),
        }


def _f(s: str) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def parse_analyze(path: str | Path) -> AnalyzeSession:
    path = Path(path)
    s = AnalyzeSession(path=path)
    base = 0.0
    af_start: dict[str, tuple[float, float | None, str, str]] = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split(",")
            ev = p[0]
            if ev == "AnalyzeStartTime" and len(p) >= 2:
                try:
                    s.start_local = datetime.strptime(p[1], TIME_FORMAT)
                except ValueError:
                    s.start_local = datetime.strptime(p[1][:19], "%Y-%m-%d %H:%M:%S")
                s.tz_abbrev = p[2] if len(p) > 2 else ""
                # Naive local wall time treated as if it were UTC. The real
                # offset is applied later with AnalyzeSession.shift().
                base = s.start_local.replace(tzinfo=timezone.utc).timestamp()
                continue
            if len(p) < 2:
                continue
            rel = _f(p[1])
            if rel is None:
                continue
            t = base + rel
            if ev == "CaptureComplete" and len(p) >= 6:
                s.captures.append(Capture(
                    t=t, exposure=_f(p[2]) or 0.0, filter=p[3], hfr=_f(p[4]) or 0.0,
                    filename=p[5],
                    stars=int(_f(p[6])) if len(p) > 6 and _f(p[6]) is not None else None,
                    median=_f(p[7]) if len(p) > 7 else None,
                    eccentricity=_f(p[8]) if len(p) > 8 else None,
                ))
            elif ev == "CaptureAborted":
                s.aborted.append(t)
            elif ev == "GuideState" and len(p) >= 3:
                s.guide_states.append((t, p[2]))
            elif ev == "GuideStats" and len(p) >= 9:
                ra, dec, snr = _f(p[2]), _f(p[3]), _f(p[6])
                if ra is not None and dec is not None:
                    s.guide_t.append(t)
                    s.guide_ra.append(ra)
                    s.guide_dec.append(dec)
                    s.guide_snr.append(snr if snr is not None else np.nan)
            elif ev == "Temperature" and len(p) >= 3:
                v = _f(p[2])
                if v is not None:
                    s.temp_t.append(t)
                    s.temp_v.append(v)
            elif ev == "MountState" and len(p) >= 3:
                s.mount_states.append((t, p[2]))
            elif ev == "MountCoords" and len(p) >= 7:
                alt, az, side = _f(p[5]), _f(p[4]), _f(p[6])
                if alt is not None:
                    s.coords_t.append(t)
                    s.coords_alt.append(alt)
                    s.coords_az.append(az if az is not None else np.nan)
                    s.coords_side.append(int(side) if side is not None else -1)
                    s.coords_ha.append(_f(p[7]) if len(p) > 7 and _f(p[7]) is not None else np.nan)
            elif ev == "MeridianFlipState" and len(p) >= 3:
                s.flip_states.append((t, p[2]))
            elif ev == "AutofocusStarting" and len(p) >= 4:
                dev = p[6] if len(p) > 6 else ""
                af_start[dev] = (t, _f(p[3]), p[2], p[4] if len(p) > 4 else "")
            elif ev in ("AutofocusComplete", "AutofocusAborted"):
                ok = ev == "AutofocusComplete"
                if len(p) >= 8 and _f(p[2]) is not None:
                    temp, flt = _f(p[2]), p[5]
                    dev = p[10] if len(p) > 10 else ""
                else:  # version 1 complete: filter, points
                    temp, flt, dev = None, p[2] if len(p) > 2 else "", ""
                start = af_start.pop(dev, None) or (af_start.popitem()[1] if af_start else None)
                if temp is None and start is not None:
                    temp = start[1]
                s.autofocus.append(Autofocus(
                    t_start=start[0] if start else None, t_end=t, success=ok,
                    temperature=temp, filter=flt, reason=start[3] if start else "",
                ))
            elif ev == "AdaptiveFocusComplete":
                pass  # small focuser nudges; tracked through FOCUSPOS instead
            elif ev == "SchedulerJobStart" and len(p) >= 3:
                s.scheduler.append((t, "start", p[2]))
            elif ev == "SchedulerJobEnd" and len(p) >= 3:
                s.scheduler.append((t, "end", p[2]))
            elif ev in ("CaptureStarting", "AlignState", "TargetDistance"):
                pass
            else:
                s.unknown_events[ev] = s.unknown_events.get(ev, 0) + 1
    return s


def tz_hours_from_abbrev(abbrev: str) -> float | None:
    return TZ_ABBREV_HOURS.get(abbrev.strip().upper())
