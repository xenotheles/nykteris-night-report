"""Glue: discover inputs, measure, align, classify."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import Thresholds
from .ekos_analyze import parse_analyze
from .frames import discover
from .guidelog import parse_guidelog
from .metrics import MetricsCache, measure_all
from .rules import classify
from .timeline import Night, align


@dataclass
class Inputs:
    lights: Path
    analyze: list[Path]
    guide_logs: list[Path]


def find_inputs(lights: Path, analyze=None, guide_logs=None) -> Inputs:
    """Use the given files, or look next to the lights folder and one level up."""
    lights = Path(lights)
    search = [lights, lights.parent, lights.parent.parent]
    if analyze is None:
        analyze = sorted({p for d in search for p in d.glob("*.analyze")})
    if guide_logs is None:
        guide_logs = sorted({p for d in search for pat in ("guide_log-*.txt", "PHD2_GuideLog*.txt")
                             for p in d.glob(pat)})
    return Inputs(lights=lights, analyze=[Path(p) for p in analyze], guide_logs=[Path(p) for p in guide_logs])


def run(inputs: Inputs, thresholds: Thresholds, utc_offset_h: float | None = None,
        cache_path: Path | None = None, jobs: int | None = None, progress=None) -> Night:
    frames = discover(inputs.lights)
    if not frames:
        raise SystemExit(f"No light frames found below {inputs.lights}")
    cache = MetricsCache(cache_path)
    for f, m in zip(frames, measure_all([f.path for f in frames], cache, jobs=jobs, progress=progress)):
        f.metrics = m
    sessions = [parse_analyze(p) for p in inputs.analyze]
    guide = parse_guidelog(inputs.guide_logs) if inputs.guide_logs else None
    # Keep only sessions that overlap the frames (an analyze folder may hold many nights).
    night = align(frames, sessions, guide, utc_offset_h=utc_offset_h)
    night.thresholds = thresholds
    night.inputs = inputs
    night.baselines = classify(night, thresholds)
    return night


def move_rejects(night: Night) -> int:
    n = 0
    for f in night.frames:
        if f.verdict == "reject":
            dest = f.path.parent / "rejected"
            dest.mkdir(exist_ok=True)
            shutil.move(str(f.path), dest / f.name)
            n += 1
    return n
