"""Parser for PHD2 format guide logs (PHD2 itself and the Ekos internal guider).

The Ekos internal guider writes ``guide_log-<date>.txt`` in PHD2 log format
version 2.5 (KStars ``ekos/guide/internalguide/guidelog.cpp``). A guiding
section looks like::

    Guiding Begins at 2026-11-18 21:15:03
    Pixel scale = 3.87 arc-sec/px, Binning = 1, Focal length = 200 mm
    ...
    Frame,Time,mount,dx,dy,RARawDistance,DECRawDistance,...,SNR,ErrorCode
    1,0.000,"Mount",0.120,-0.030,0.101,0.050,...
    INFO: DITHER by 3.000, -2.000, new lock pos = ...
    Guiding Ends at 2026-11-18 23:01:10

Timestamps are local wall time without a zone. ``Time`` is seconds since
the "Guiding Begins" line. Distances are in guide camera pixels.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_BEGIN = re.compile(r"^Guiding Begins at (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
_END = re.compile(r"^Guiding Ends at")
_SCALE = re.compile(r"Pixel scale = ([0-9.]+) arc-sec/px")
_DITHER = re.compile(r"^INFO: DITHER")
_SETTLE_DONE = re.compile(r"^INFO: SETTLING STATE CHANGE, Settling complete")


@dataclass
class GuideLog:
    paths: list[Path] = field(default_factory=list)
    t: np.ndarray = field(default_factory=lambda: np.zeros(0))  # wall time as if UTC
    ra: np.ndarray = field(default_factory=lambda: np.zeros(0))  # arcsec
    dec: np.ndarray = field(default_factory=lambda: np.zeros(0))  # arcsec
    snr: np.ndarray = field(default_factory=lambda: np.zeros(0))
    dropped: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    dithers: list[float] = field(default_factory=list)
    sections: list[tuple[float, float]] = field(default_factory=list)

    def shift(self, seconds: float) -> None:
        self.t = self.t + seconds
        self.dithers = [d + seconds for d in self.dithers]
        self.sections = [(a + seconds, b + seconds) for a, b in self.sections]

    def __len__(self) -> int:
        return len(self.t)


def _wall(s: str) -> float:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()


def _num(row: dict, key: str) -> float:
    v = row.get(key)
    if v is None or v == "":
        return np.nan
    try:
        return float(v)
    except ValueError:
        return np.nan


def parse_guidelog(paths) -> GuideLog:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    out = GuideLog(paths=[Path(p) for p in paths])
    ts, ras, decs, snrs, drops = [], [], [], [], []
    for path in out.paths:
        begin = None
        scale = 1.0
        header = None
        last_t = None
        with open(path, encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                line = raw.rstrip("\n").rstrip("\r")
                m = _BEGIN.match(line)
                if m:
                    begin = _wall(m.group(1))
                    header = None
                    last_t = begin
                    continue
                if begin is None:
                    continue
                if _END.match(line):
                    out.sections.append((begin, last_t if last_t is not None else begin))
                    begin = None
                    continue
                m = _SCALE.search(line)
                if m:
                    scale = float(m.group(1))
                    continue
                if line.startswith("Frame,Time,"):
                    header = next(csv.reader([line]))
                    continue
                if _DITHER.match(line):
                    out.dithers.append(last_t if last_t is not None else begin)
                    continue
                if header is None or not line[:1].isdigit():
                    continue
                values = next(csv.reader([line]))
                row = dict(zip(header, values))
                rel = _num(row, "Time")
                if np.isnan(rel):
                    continue
                t = begin + rel
                last_t = t
                mount = row.get("mount", "")
                err = _num(row, "ErrorCode")
                dropped = mount.strip('"').upper() == "DROP" or (not np.isnan(err) and err != 0)
                ts.append(t)
                ras.append(_num(row, "RARawDistance") * scale)
                decs.append(_num(row, "DECRawDistance") * scale)
                snrs.append(_num(row, "SNR"))
                drops.append(bool(dropped))
        if begin is not None:
            out.sections.append((begin, last_t if last_t is not None else begin))
    order = np.argsort(ts, kind="stable")
    out.t = np.asarray(ts, float)[order]
    out.ra = np.asarray(ras, float)[order]
    out.dec = np.asarray(decs, float)[order]
    out.snr = np.asarray(snrs, float)[order]
    out.dropped = np.asarray(drops, bool)[order]
    out.dithers.sort()
    return out
