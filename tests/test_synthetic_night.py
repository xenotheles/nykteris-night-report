"""End-to-end checks against the synthetic night's ground truth."""

import json
from collections import defaultdict
from pathlib import Path

import pytest

from nykteris.cli import main
from nykteris.config import Thresholds
from nykteris.pipeline import Inputs, find_inputs, run


def _score(night_result, truth):
    """Per label: (true positives, labelled, predicted). Ambiguous frames are ignored."""
    stats = defaultdict(lambda: [0, 0, 0])
    confusion = []
    for f in night_result.frames:
        label = truth[f.name]["label"]
        if label == "ambiguous":
            continue
        got = f.causes[0]["cause"] if f.verdict != "ok" else "ok"
        stats[label][1] += 1
        stats[got][2] += 1
        if got == label:
            stats[label][0] += 1
        else:
            confusion.append((truth[f.name]["index"], label, got))
    return stats, confusion


@pytest.fixture(scope="module")
def full(night):
    inputs = find_inputs(Path(night["lights"]))
    return run(inputs, Thresholds(), jobs=1)


def test_inputs_discovered(night):
    inputs = find_inputs(Path(night["lights"]))
    assert inputs.analyze == [Path(night["analyze"])]
    assert inputs.guide_logs == [Path(night["guide_log"])]


def test_utc_offset_derived(full):
    assert full.utc_offset_h == 1.0
    assert "matched" in full.offset_source
    assert len(full.flips) == 1 and len(full.autofocus) == 2


def test_every_injected_fault_found(full, night):
    stats, confusion = _score(full, night["truth"])
    for label in ["clouds", "guiding", "seeing", "focus", "flip", "trail", "altitude"]:
        tp, n, _ = stats[label]
        assert n > 0, label
        assert tp / n >= 0.75, (label, stats[label], confusion)
    # Good frames stay good.
    tp, n, _ = stats["ok"]
    assert tp / n >= 0.95, confusion
    assert len(confusion) <= 4, confusion


def test_explanations_carry_numbers(full, night):
    by_label = {night["truth"][f.name]["label"]: f for f in full.frames}
    assert "guide RMS" in by_label["guiding"].causes[0]["text"]
    assert "temperature" in by_label["focus"].causes[0]["text"]
    assert "flip" in by_label["flip"].causes[0]["text"]
    assert by_label["trail"].verdict == "check"


def test_fits_only_fallback(night):
    """Without .analyze and guide log, the header path still explains most faults."""
    result = run(Inputs(Path(night["lights"]), [], []), Thresholds(), jobs=1)
    stats, confusion = _score(result, night["truth"])
    for label in ["clouds", "seeing", "flip", "trail", "altitude"]:
        tp, n, _ = stats[label]
        assert tp / n >= 0.75, (label, stats[label], confusion)
    # Without guide data, guiding faults show up as elongation.
    guided = [f for f in result.frames if night["truth"][f.name]["label"] == "guiding"]
    assert all(f.causes and f.causes[0]["cause"] == "elongation" for f in guided)


def test_cli_writes_report(night, tmp_path):
    out = tmp_path / "r" / "report.html"
    assert main(["report", str(night["lights"]), "-o", str(out), "-j", "1"]) == 0
    html = out.read_text()
    assert "Night report" in html and "<svg" in html and "Meridian flip" in html
    assert out.with_suffix(".csv").exists()
    rejects = (out.parent / "report-rejects.txt").read_text().splitlines()
    assert len(rejects) > 5
