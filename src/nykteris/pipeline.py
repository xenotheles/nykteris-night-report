"""Glue: discover inputs, measure, align, classify."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .config import Thresholds
from .ekos_analyze import parse_analyze
from .frames import discover
from .guidelog import parse_guidelog
from .metrics import MetricsCache, measure_all
from .nina import is_report, parse_autofocus_reports, parse_image_metadata
from .rules import classify
from .timeline import Night, align


@dataclass
class Inputs:
    lights: Path
    analyze: list[Path]
    guide_logs: list[Path]
    nina_metadata: list[Path] = field(default_factory=list)
    af_reports: list[Path] = field(default_factory=list)


def _metadata_files(folders) -> list[Path]:
    """NINA Session Metadata files; prefer JSON over CSV for the same folder."""
    out = []
    for d in folders:
        js = sorted(d.glob("ImageMetaData*.json"))
        out += js or sorted(d.glob("ImageMetaData*.csv"))
    return out


def find_inputs(lights: Path, analyze=None, guide_logs=None, nina_metadata=None, af_reports=None) -> Inputs:
    """Use the given files, or look next to the lights folder and up to two levels up."""
    lights = Path(lights)
    search = [lights, lights.parent, lights.parent.parent]
    if analyze is None:
        analyze = sorted({p for d in search for p in d.glob("*.analyze")})
    if guide_logs is None:
        guide_logs = sorted({p for d in search for pat in ("guide_log-*.txt", "PHD2_GuideLog*.txt")
                             for p in d.glob(pat)})
    if nina_metadata is None:
        # The plugin writes one file per image folder, so also look below the lights folder.
        folders = sorted({lights, lights.parent, *(p.parent for p in lights.rglob("ImageMetaData*"))})
        nina_metadata = _metadata_files(folders)
    if af_reports is None:
        af_reports = sorted({p for d in search for p in (d / "AutoFocus").glob("*.json") if is_report(p)})
    else:
        expanded = []
        for p in map(Path, af_reports):
            expanded += sorted(p.glob("*.json")) if p.is_dir() else [p]
        af_reports = expanded
    return Inputs(lights=lights, analyze=[Path(p) for p in analyze], guide_logs=[Path(p) for p in guide_logs],
                  nina_metadata=[Path(p) for p in nina_metadata], af_reports=[Path(p) for p in af_reports])


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
    nina = parse_image_metadata(inputs.nina_metadata) if inputs.nina_metadata else None
    reports = parse_autofocus_reports(inputs.af_reports) if inputs.af_reports else []
    night = align(frames, sessions, guide, utc_offset_h=utc_offset_h, nina=nina, af_reports=reports)
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
