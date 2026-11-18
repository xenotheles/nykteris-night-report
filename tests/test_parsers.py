from pathlib import Path

import numpy as np

from nykteris.config import Thresholds
from nykteris.ekos_analyze import parse_analyze
from nykteris.frames import read_frame
from nykteris.guidelog import parse_guidelog

ANALYZE = """#KStars version 3.7.5. Analyze log version 2.0.

AnalyzeStartTime,2026-11-18 21:00:00.000,CET
MountCoords,10.000,23.4621,30.6599,143.5258,67.3824,0,-15.4126
AutofocusStarting,20.000,L,4.0,0,
AutofocusComplete,200.000,4.0,0,,L,1|2|3,0,,,
AutofocusComplete,300.000,L,1|2|3
GuideStats,400.000,0.378,-0.313,30,25,39.113,120.000,1
CaptureComplete,700.000,240.000,L,1.300,C:\\Users\\me\\Pictures\\M_33_Light_001.fits,180,940,0.250,CCD
MeridianFlipState,800.000,MOUNT_FLIP_COMPLETED
Temperature,900.000,3.990,Focuser
SomeFutureEvent,950.000,x
"""

GUIDELOG = """KStars version 3.7.5. PHD2 log version 2.5. Log enabled at 2026-11-18 21:00:00

Guiding Begins at 2026-11-18 21:05:00
Pixel scale = 2.00 arc-sec/px, Binning = 1, Focal length = 200 mm
Frame,Time,mount,dx,dy,RARawDistance,DECRawDistance,RAGuideDistance,DECGuideDistance,RADuration,RADirection,DECDuration,DECDirection,XStep,YStep,StarMass,SNR,ErrorCode
1,1.000,"Mount",0.1,0.2,0.500,-0.250,0,0,10,E,10,N,,,1000,30.00,0
INFO: DITHER by 3.000, -2.000, new lock pos = 400.000, 300.000
2,3.000,"DROP",,,,,,,0,,0,,,,0,2.00,1
3,5.000,"Mount",0.1,0.2,1.000,0.000,0,0,10,E,10,N,,,1000,30.00,0
Guiding Ends at 2026-11-18 21:06:00
"""


def test_parse_analyze(tmp_path: Path):
    p = tmp_path / "x.analyze"
    p.write_text(ANALYZE)
    s = parse_analyze(p)
    assert s.tz_abbrev == "CET"
    assert len(s.captures) == 1
    c = s.captures[0]
    assert c.basename == "M_33_Light_001.fits"  # Windows path handled
    assert c.hfr == 1.3 and c.stars == 180
    assert len(s.autofocus) == 2 and s.autofocus[0].temperature == 4.0
    assert s.flip_states[0][1] == "MOUNT_FLIP_COMPLETED"
    assert s.coords_alt == [67.3824] and s.coords_side == [0]
    assert s.guide_ra == [0.378]
    assert s.unknown_events == {"SomeFutureEvent": 1}
    # Times are wall clock seconds; capture is 700 s after start.
    assert s.captures[0].t - s.coords_t[0] == 690


def test_parse_guidelog(tmp_path: Path):
    p = tmp_path / "guide_log.txt"
    p.write_text(GUIDELOG)
    g = parse_guidelog(p)
    assert len(g) == 3
    assert g.dropped.tolist() == [False, True, False]
    np.testing.assert_allclose(g.ra[[0, 2]], [1.0, 2.0])  # px * 2"/px
    np.testing.assert_allclose(g.dec[0], -0.5)
    assert len(g.dithers) == 1
    assert g.t[2] - g.t[0] == 4.0
    assert len(g.sections) == 1


def test_read_frame_headers(night):
    path = sorted(Path(night["lights"]).glob("*.fits"))[0]
    f = read_frame(path)
    assert f.filter == "L" and f.exptime == 240
    assert abs(f.scale - 0.9694) < 1e-3
    assert f.pier_side == "WEST"
    assert abs(f.ra_deg - 23.4621) < 1e-3 and abs(f.site_lat - 50.11) < 1e-6


def test_thresholds_toml_roundtrip(tmp_path: Path):
    th = Thresholds(fwhm_ratio=1.4, focus_trend_frames=5)
    p = tmp_path / "t.toml"
    p.write_text(th.to_toml())
    assert Thresholds.from_toml(p) == th


def test_thresholds_unknown_key(tmp_path: Path):
    p = tmp_path / "t.toml"
    p.write_text("[thresholds]\nfwhm_ration = 1.3\n")
    try:
        Thresholds.from_toml(p)
    except ValueError as e:
        assert "fwhm_ration" in str(e)
    else:
        raise AssertionError("expected ValueError")
