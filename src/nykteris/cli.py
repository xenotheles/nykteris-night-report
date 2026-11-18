"""Command line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import Thresholds


def _cmd_report(args) -> int:
    from .pipeline import find_inputs, move_rejects, run
    from .report import render_html, write_csv, write_rejects

    th = Thresholds.from_toml(args.config) if args.config else Thresholds()
    inputs = find_inputs(Path(args.lights), analyze=args.analyze, guide_logs=args.guide_log,
                         nina_metadata=args.nina_metadata, af_reports=args.af_reports)
    out = Path(args.output) if args.output else Path(args.lights) / "nykteris-report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    cache = None if args.no_cache else out.parent / ".nykteris-cache.json"

    def progress(n, total):
        if sys.stderr.isatty():
            print(f"\rMeasuring subs: {n}/{total}", end="" if n < total else "\n", file=sys.stderr)

    print(f"Lights:    {inputs.lights}", file=sys.stderr)
    print(f"Analyze:   {', '.join(map(str, inputs.analyze)) or 'none'}", file=sys.stderr)
    print(f"Guide log: {', '.join(map(str, inputs.guide_logs)) or 'none'}", file=sys.stderr)
    if inputs.nina_metadata:
        print(f"NINA meta: {', '.join(map(str, inputs.nina_metadata))}", file=sys.stderr)
    if inputs.af_reports:
        print(f"NINA AF:   {len(inputs.af_reports)} autofocus reports", file=sys.stderr)
    night = run(inputs, th, utc_offset_h=args.utc_offset, cache_path=cache, jobs=args.jobs, progress=progress)
    out.write_text(render_html(night), encoding="utf-8")
    write_csv(night, out.with_suffix(".csv"))
    write_rejects(night, out.parent / (out.stem + "-rejects.txt"))
    n_rej = sum(f.verdict == "reject" for f in night.frames)
    n_chk = sum(f.verdict == "check" for f in night.frames)
    print(f"{len(night.frames)} subs, {n_rej} flagged to reject, {n_chk} worth a look.", file=sys.stderr)
    print(f"Report:    {out}", file=sys.stderr)
    if args.move_rejects:
        moved = move_rejects(night)
        print(f"Moved {moved} subs into 'rejected' folders (nothing was deleted).", file=sys.stderr)
    return 0


def _cmd_synth(args) -> int:
    from .synth import generate

    r = generate(args.outdir, n_frames=args.frames, width=args.width, height=args.height, seed=args.seed,
                 flavor=args.flavor, image_format=args.format)
    print(f"Synthetic night written to {args.outdir}")
    print(f"  lights:    {r['lights']}")
    for key in ("analyze", "guide_log", "nina_metadata", "af_reports"):
        if r.get(key):
            print(f"  {key}: {r[key]}")
    print(f"  truth:     {Path(args.outdir) / 'truth.json'}")
    return 0


def _cmd_thresholds(args) -> int:
    print(Thresholds().to_toml(), end="")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nykteris", description="Night report for Ekos/KStars imaging sessions.")
    p.add_argument("--version", action="version", version=f"nykteris {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("report", help="analyse a night and write an HTML report")
    r.add_argument("lights", help="folder with the night's light frames, FITS or XISF (searched recursively)")
    r.add_argument("--analyze", action="append", metavar="FILE",
                   help="Ekos .analyze file (repeatable). Default: *.analyze next to the lights or one or two levels up")
    r.add_argument("--guide-log", action="append", metavar="FILE",
                   help="PHD2 format guide log (repeatable). Default: guide_log-*.txt or PHD2_GuideLog*.txt nearby")
    r.add_argument("--nina-metadata", action="append", metavar="FILE",
                   help="NINA Session Metadata ImageMetaData.csv/.json (repeatable). Default: found next to the lights")
    r.add_argument("--af-reports", action="append", metavar="PATH",
                   help="NINA autofocus report JSON file or folder (repeatable). "
                        "Default: an AutoFocus folder next to the lights or up to two levels up")
    r.add_argument("-o", "--output", help="report path (default: <lights>/nykteris-report.html)")
    r.add_argument("--config", help="TOML file with a [thresholds] table")
    r.add_argument("--utc-offset", type=float, help="local time offset of the logs in hours, e.g. 1 for CET")
    r.add_argument("-j", "--jobs", type=int, help="parallel workers for measuring (default: CPU count, max 8)")
    r.add_argument("--no-cache", action="store_true", help="re-measure every sub")
    r.add_argument("--move-rejects", action="store_true",
                   help="move rejected subs into a 'rejected' subfolder (never deletes)")
    r.set_defaults(func=_cmd_report)

    s = sub.add_parser("synth", help="write a synthetic night with labelled faults")
    s.add_argument("outdir")
    s.add_argument("--frames", type=int, default=100)
    s.add_argument("--width", type=int, default=512)
    s.add_argument("--height", type=int, default=384)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--flavor", choices=["ekos", "nina"], default="ekos", help="which capture software to imitate")
    s.add_argument("--format", choices=["fits", "xisf"], help="image format (default: fits for ekos, xisf for nina)")
    s.set_defaults(func=_cmd_synth)

    t = sub.add_parser("thresholds", help="print the default thresholds as TOML")
    t.set_defaults(func=_cmd_thresholds)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
