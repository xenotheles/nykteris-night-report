"""N.I.N.A. support: XISF, Session Metadata, autofocus reports, PHD2 log."""

from pathlib import Path

import numpy as np
import pytest

from nykteris import xisf
from nykteris.cli import main
from nykteris.config import Thresholds
from nykteris.frames import read_frame
from nykteris.nina import parse_autofocus_reports, parse_image_metadata, utc_offset_hours
from nykteris.pipeline import Inputs, find_inputs, run

from test_synthetic_night import _score


def test_xisf_roundtrip(tmp_path: Path):
    data = (np.arange(60 * 40, dtype=np.uint16).reshape(40, 60) * 7) % 65535
    for compress in (True, False):
        p = tmp_path / f"x{compress}.xisf"
        xisf.write(p, data, [("DATE-OBS", "2026-11-18T20:05:57.6890000", ""), ("EXPTIME", 240.0, ""),
                             ("OBJECT", "M 33 'core'", ""), ("XBINNING", 1, "")])
        np.testing.assert_array_equal(xisf.read_data(p), data)
        h = xisf.read_header(p)
        assert h["EXPTIME"] == 240.0 and h["XBINNING"] == 1 and h["OBJECT"] == "M 33 'core'"


def test_shuffle_inverse():
    raw = bytes(range(250)) + b"\x07"
    assert xisf.unshuffle(xisf.shuffle(raw, 2), 2) == raw


def test_nina_headers(nina_night):
    path = sorted(Path(nina_night["lights"]).glob("*.xisf"))[0]
    f = read_frame(path)
    assert f.software.startswith("N.I.N.A.")
    assert f.exptime == 240 and f.filter == "L" and f.pier_side == "WEST"
    assert abs(f.scale - 0.9694) < 1e-3  # from XPIXSZ and FOCALLEN, no SCALE key
    assert f.focus_pos == 31250 and f.focus_temp is not None
    assert f.header_alt is not None  # CENTALT
    assert round((f.date_loc - f.date_obs) / 3600, 2) == 1.0  # DATE-LOC vs DATE-OBS


def test_session_metadata_csv_and_json(nina_night):
    folder = Path(nina_night["lights"])
    js = parse_image_metadata(folder / "ImageMetaData.json")
    cs = parse_image_metadata(folder / "ImageMetaData.csv")
    assert len(js.images) == len(cs.images) == 100
    a, b = js.images[3], cs.images[3]
    assert a.basename == b.basename and a.basename.endswith(".xisf")  # Windows path stripped
    assert a.t_start == pytest.approx(b.t_start) and a.guide_rms == pytest.approx(b.guide_rms)
    assert a.fwhm is None  # NaN without Hocus Focus
    assert utc_offset_hours(js) == 1.0


def test_autofocus_reports(nina_night):
    reports = parse_autofocus_reports(sorted(Path(nina_night["af_reports"]).glob("*.json")))
    assert len(reports) == 2
    assert reports[0].temperature == pytest.approx(4.0)
    assert reports[1].position == 31310


def test_inputs_discovered(nina_night):
    inp = find_inputs(Path(nina_night["lights"]))
    assert inp.analyze == []
    assert inp.guide_logs == [Path(nina_night["guide_log"])]
    assert inp.nina_metadata == [Path(nina_night["lights"]) / "ImageMetaData.json"]
    assert len(inp.af_reports) == 2


@pytest.mark.parametrize("variant", ["full", "metadata_only", "csv_only"])
def test_nina_night_faults(nina_night, variant):
    full = find_inputs(Path(nina_night["lights"]))
    inputs = {
        "full": full,
        "metadata_only": Inputs(full.lights, [], [], full.nina_metadata, full.af_reports),
        "csv_only": Inputs(full.lights, [], [], [full.lights / "ImageMetaData.csv"], []),
    }[variant]
    night = run(inputs, Thresholds(), jobs=1)
    assert night.capture_software == "N.I.N.A."
    assert night.utc_offset_h == 1.0
    stats, confusion = _score(night, nina_night["truth"])
    for label in ["clouds", "guiding", "seeing", "focus", "flip", "trail", "altitude"]:
        tp, n, _ = stats[label]
        assert tp / n >= 0.75, (label, stats[label], confusion)
    assert stats["ok"][0] / stats["ok"][1] >= 0.95, confusion
    if variant != "csv_only":
        assert len(night.autofocus) == 2


def test_cli_nina(nina_night, tmp_path):
    out = tmp_path / "nina.html"
    assert main(["report", str(nina_night["lights"]), "-o", str(out), "-j", "1"]) == 0
    html = out.read_text()
    assert "N.I.N.A." in html and "NINA image metadata" in html


@pytest.mark.parametrize("codec", ["lz4", "lz4hc", "zstd"])
def test_xisf_other_codecs(tmp_path: Path, codec):
    data = (np.arange(30 * 20, dtype=np.uint16).reshape(20, 30) * 13) % 4000
    raw = data.astype("<u2").tobytes()
    shuffled = xisf.shuffle(raw, 2)
    if codec.startswith("lz4"):
        lz4_block = pytest.importorskip("lz4.block")
        payload = lz4_block.compress(shuffled, store_size=False)
    else:
        zstandard = pytest.importorskip("zstandard")
        payload = zstandard.ZstdCompressor().compress(shuffled)
    p = tmp_path / "c.xisf"
    xisf.write(p, data, [("EXPTIME", 1.0, "")], compress=False)
    # Swap the payload and the compression attribute in place of the raw block.
    blob = p.read_bytes()
    hlen = int.from_bytes(blob[8:12], "little")
    header = blob[16:16 + hlen].decode()
    offset = int(header.split('location="attachment:')[1].split(":")[0])
    header = header.replace(f'location="attachment:{offset}:{len(raw)}"',
                            f'location="attachment:{offset}:{len(payload)}" compression="{codec}+sh:{len(raw)}:2"')
    hb = header.encode()
    assert 16 + len(hb) <= offset
    p.write_bytes(blob[:8] + len(hb).to_bytes(4, "little") + b"\0" * 4 + hb + b"\0" * (offset - 16 - len(hb)) + payload)
    np.testing.assert_array_equal(xisf.read_data(p), data)
