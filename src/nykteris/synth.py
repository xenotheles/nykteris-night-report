"""Synthetic Ekos night: FITS lights, a matching .analyze file and guide log.

Every fault is injected on purpose and every frame carries a ground-truth
label in ``truth.json``, so the cause rules can be tested automatically.
The physics is deliberately simple (Gaussian stars, Gaussian noise); the
point is a consistent timeline across the three inputs, not realism.

Labels: ``ok``, ``clouds``, ``guiding``, ``seeing``, ``focus``,
``altitude``, ``trail``, ``flip``. Frames near a threshold are labelled
``ambiguous`` and ignored when scoring.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from astropy.io import fits

from . import xisf

# Site and target. Generic central European site, M 33 in November.
SITE_LAT = 50.11
SITE_LON = 8.68
TZ_HOURS = 1  # CET
TARGET = "M 33"
TARGET_RA = 23.4621  # deg
TARGET_DEC = 30.6599
FOCAL_MM = 800.0
PIX_UM = 3.76
SCALE = 206.265 * PIX_UM / FOCAL_MM  # arcsec/px
GUIDE_SCALE = 3.87  # arcsec/px
SEEING_ZENITH = 2.5  # arcsec FWHM
EXPOSURE = 240.0
DOWNLOAD = 6.0
DITHER_EVERY = 3
DITHER_SETTLE = 20.0
GUIDE_CADENCE = 2.0
FLIP_DURATION = 300.0
CAPTURE_ROOT = "/home/stellarmate/Pictures/M_33/Light/L"


@dataclass
class Plan:
    clouds: dict
    guiding: dict
    seeing: dict
    focus_start: int
    focus_end: int  # last drifting frame; autofocus runs before focus_end + 1
    trail: int


DEFAULT_PLAN = Plan(
    clouds={6: 0.25, 7: 0.18, 8: 0.32},
    guiding={24: 7.0, 25: 6.0},
    seeing={33: 1.7, 34: 1.6},
    focus_start=42,
    focus_end=50,
    trail=58,
)


def _gmst_deg(unix: float) -> float:
    jd = unix / 86400.0 + 2440587.5
    d = jd - 2451545.0
    return (280.46061837 + 360.98564736629 * d) % 360.0


def hour_angle_h(unix: float) -> float:
    lst = (_gmst_deg(unix) + SITE_LON) % 360.0
    ha = (lst - TARGET_RA + 540.0) % 360.0 - 180.0
    return ha / 15.0


def altaz(unix: float) -> tuple[float, float]:
    ha = math.radians(hour_angle_h(unix) * 15.0)
    lat, dec = math.radians(SITE_LAT), math.radians(TARGET_DEC)
    sin_alt = math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(ha)
    alt = math.asin(sin_alt)
    cos_az = (math.sin(dec) - sin_alt * math.sin(lat)) / (math.cos(alt) * math.cos(lat))
    az = math.degrees(math.acos(max(-1.0, min(1.0, cos_az))))
    if math.sin(ha) > 0:
        az = 360.0 - az
    return math.degrees(alt), az


def airmass(alt_deg: float) -> float:
    # Kasten and Young (1989)
    return 1.0 / (math.sin(math.radians(alt_deg)) + 0.50572 * (alt_deg + 6.07995) ** -1.6364)


def _transit_after(unix: float) -> float:
    ha = hour_angle_h(unix)
    return unix - ha * 3600.0 / 1.0027379


def _render(rng, w, h, stars_xy, flux, sigma_x, sigma_y, theta, sky, gradient, transmission, trail):
    img = np.zeros((h, w), np.float64)
    yy, xx = np.mgrid[0:h, 0:w]
    img += sky * (1.0 + gradient * (xx / w - 0.5))
    c, s = math.cos(theta), math.sin(theta)
    # Inverse covariance of a rotated elliptical Gaussian.
    a_ = c * c / (2 * sigma_x**2) + s * s / (2 * sigma_y**2)
    b_ = s * c * (1 / (2 * sigma_x**2) - 1 / (2 * sigma_y**2))
    c_ = s * s / (2 * sigma_x**2) + c * c / (2 * sigma_y**2)
    norm = 1.0 / (2 * math.pi * sigma_x * sigma_y)
    r = int(math.ceil(4 * max(sigma_x, sigma_y))) + 1
    oy, ox = np.mgrid[-r : r + 1, -r : r + 1]
    for (x, y), f in zip(stars_xy, flux):
        ix, iy = int(round(x)), int(round(y))
        if ix < -r or iy < -r or ix >= w + r or iy >= h + r:
            continue
        dx = ox + ix - x
        dy = oy + iy - y
        stamp = f * transmission * norm * np.exp(-(a_ * dx * dx + 2 * b_ * dx * dy + c_ * dy * dy))
        x0, x1 = max(ix - r, 0), min(ix + r + 1, w)
        y0, y1 = max(iy - r, 0), min(iy + r + 1, h)
        img[y0:y1, x0:x1] += stamp[y0 - (iy - r) : y1 - (iy - r), x0 - (ix - r) : x1 - (ix - r)]
    if trail:
        (x0, y0), (x1, y1), peak = trail
        dist = np.abs((y1 - y0) * xx - (x1 - x0) * yy + x1 * y0 - y1 * x0) / math.hypot(x1 - x0, y1 - y0)
        img += peak * np.exp(-0.5 * (dist / 1.3) ** 2)
    noisy = rng.normal(img, np.sqrt(np.maximum(img, 1.0)) + 3.0)
    return np.clip(noisy + 300.0, 0, 65535).astype(np.uint16)


def _local(unix: float) -> datetime:
    return datetime.fromtimestamp(unix, tz=timezone.utc) + timedelta(hours=TZ_HOURS)


def generate(outdir: str | Path, n_frames: int = 100, width: int = 512, height: int = 384,
             n_stars: int = 700, seed: int = 1, plan: Plan = DEFAULT_PLAN, flavor: str = "ekos",
             image_format: str | None = None) -> dict:
    """Write a synthetic night. ``flavor`` is "ekos" or "nina"."""
    if flavor not in ("ekos", "nina"):
        raise ValueError("flavor must be 'ekos' or 'nina'")
    image_format = image_format or ("xisf" if flavor == "nina" else "fits")
    outdir = Path(outdir)
    lights = outdir / "Light" / "L" if flavor == "ekos" else outdir / "2026-11-18" / "LIGHT"
    lights.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    af_events: list[tuple[float, float, float]] = []  # t_end, temperature, position
    nina_rows: list[dict] = []

    # Place the first meridian flip after frame 14.
    cadence = EXPOSURE + DOWNLOAD + DITHER_SETTLE / DITHER_EVERY
    anchor = datetime(2026, 11, 18, 20, 0, tzinfo=timezone.utc).timestamp()
    transit = _transit_after(anchor)
    start = transit - 14.6 * cadence
    session_start = start - 1200.0

    # Fixed star field, power-law fluxes.
    fov_pad = 40
    sx = rng.uniform(-fov_pad, width + fov_pad, n_stars)
    sy = rng.uniform(-fov_pad, height + fov_pad, n_stars)
    flux = np.minimum(400.0 * rng.uniform(0, 1, n_stars) ** (-1 / 0.7), 3.0e5)

    analyze: list[tuple[float, str]] = []
    guide_rows: list[tuple[float, float, float, float, bool]] = []  # t, ra", dec", snr, dropped
    guide_sections: list[tuple[float, float]] = []
    dithers: list[float] = []

    def ev(t, name, *fields):
        analyze.append((t, name + "," + "{:.3f}".format(t - session_start) + ("," + ",".join(str(f) for f in fields) if fields else "")))

    # Session preamble.
    ev(session_start + 5, "MountState", "Slewing")
    ev(session_start + 60, "MountState", "Tracking")
    ev(session_start + 70, "AlignState", "In Progress")
    ev(session_start + 160, "AlignState", "Complete")
    ev(session_start + 200, "GuideState", "Calibrating")
    ev(session_start + 380, "GuideState", "Calibrated")
    ev(session_start + 390, "GuideState", "Guiding")
    guide_on = session_start + 390

    temp0 = 4.0
    af_temp = temp0
    ev(session_start + 500, "AutofocusStarting", "L", f"{temp0:.1f}", 0, "")
    ev(session_start + 800, "AutofocusComplete", f"{temp0:.1f}", 0, "", "L", "31200|3.1|31250|2.6|31300|2.9", 0, "", "", "")
    focus_pos = 31250
    af_events.append((session_start + 800, temp0, focus_pos))

    truth: dict = {}
    t = start
    pier = 0  # WEST, looking east, before the meridian
    flipped = False
    last_temp_written = None
    dx = dy = 0.0
    alts = []
    frame_info = []
    for i in range(n_frames):
        # Meridian flip before this frame once the target has crossed.
        flip_frame = False
        if not flipped and hour_angle_h(t) > 0.02:
            ev(t - 900, "MeridianFlipState", "MOUNT_FLIP_PLANNED")
            ev(t, "MeridianFlipState", "MOUNT_FLIP_WAITING")
            ev(t + 1, "MeridianFlipState", "MOUNT_FLIP_ACCEPTED")
            ev(t + 2, "GuideState", "Aborted")
            guide_sections.append((guide_on, t + 2))
            ev(t + 3, "MeridianFlipState", "MOUNT_FLIP_RUNNING")
            ev(t + 3, "MountState", "Slewing")
            ev(t + 120, "MountState", "Tracking")
            ev(t + 121, "MeridianFlipState", "MOUNT_FLIP_COMPLETED")
            ev(t + 122, "MeridianFlipState", "MOUNT_FLIP_NONE")
            ev(t + 125, "AlignState", "In Progress")
            ev(t + 220, "AlignState", "Complete")
            ev(t + 260, "GuideState", "Guiding")
            guide_on = t + 260
            t += FLIP_DURATION
            pier = 1
            flipped = True
            flip_frame = True

        # Autofocus after the focus drift.
        if i == plan.focus_end + 1:
            ev(t, "AutofocusStarting", "L", f"{current_temp:.1f}", 3, "")
            ev(t + 240, "AutofocusComplete", f"{current_temp:.1f}", 3, "", "L", "31260|3.4|31310|2.6|31360|3.0", 0, "", "", "")
            af_temp = current_temp
            focus_pos = 31310
            af_events.append((t + 240, current_temp, focus_pos))
            t += 250

        if i > 0 and i % DITHER_EVERY == 0:
            ev(t, "GuideState", "Dithering")
            dithers.append(t)
            dx += rng.uniform(-4, 4)
            dy += rng.uniform(-4, 4)
            ev(t + DITHER_SETTLE - 2, "GuideState", "Dithering successful")
            ev(t + DITHER_SETTLE - 1, "GuideState", "Guiding")
            t += DITHER_SETTLE

        t0, t1 = t, t + EXPOSURE
        mid = 0.5 * (t0 + t1)
        alt, az = altaz(mid)
        X = airmass(alt)
        alts.append(alt)

        # Focuser temperature: slow decline, a fast drop during the drift.
        hours = (t0 - start) / 3600.0
        current_temp = temp0 - 0.15 * hours
        if plan.focus_start <= i:
            k = min(i, plan.focus_end) - plan.focus_start + 1
            current_temp -= 2.5 * k / (plan.focus_end - plan.focus_start + 1)
        if i > plan.focus_end:
            current_temp = current_temp  # stays low after the drop
        for tt in np.arange(t0, t1, 60.0):
            v = round(current_temp + rng.normal(0, 0.02), 2)
            if v != last_temp_written:
                ev(tt, "Temperature", f"{v:.3f}", "Focuser Simulator")
                last_temp_written = v

        label = "ok"
        transmission = 1.0
        sky = 900.0 * (1.0 + 0.6 * (X - 1.0))
        seeing_factor = 1.0
        focus_factor = 1.0
        trail_len = 0.0
        theta = 0.0 if pier == 0 else math.pi
        guide_sigma = 0.45
        guide_extra = None
        guide_snr = 40.0
        trail = None

        if i in plan.clouds:
            transmission = plan.clouds[i]
            sky *= 1.6 + (1 - transmission)
            guide_snr = 40.0 * transmission
            label = "clouds"
        if i in plan.guiding:
            trail_len = plan.guiding[i]
            guide_extra = ("spike", 4.0)
            label = "guiding"
        if i in plan.seeing:
            seeing_factor = plan.seeing[i]
            guide_sigma = 0.6
            label = "seeing"
        if plan.focus_start <= i <= plan.focus_end:
            focus_factor = 1.0 + 0.7 * (i - plan.focus_start + 1) / (plan.focus_end - plan.focus_start + 1)
            if focus_factor >= 1.4:
                label = "focus"
            elif focus_factor >= 1.12:
                label = "ambiguous"
        if flip_frame:
            trail_len = 5.5
            theta = math.radians(35)
            guide_extra = ("settle", 3.0)
            label = "flip"
        if i == plan.trail:
            trail = ((-10, rng.uniform(0, height)), (width + 10, rng.uniform(0, height)), 260.0)
            label = "trail"
        if label == "ok":
            if alt < 27.0:
                label = "altitude"
            elif alt < 33.0:
                label = "ambiguous"
        if alt < 33:
            guide_sigma *= 1.2

        fwhm_arcsec = SEEING_ZENITH * X**0.6 * seeing_factor * focus_factor * math.exp(rng.normal(0, 0.03))
        sigma = fwhm_arcsec / SCALE / 2.3548
        sigma_major = math.sqrt(sigma**2 + trail_len**2 / 12.0)

        xy = np.column_stack([sx + dx, sy + dy])
        if pier == 1:
            xy = np.column_stack([width - 1 - (sx + dx), height - 1 - (sy + dy)])
        img = _render(rng, width, height, xy, flux * EXPOSURE / 240.0, sigma_major, sigma, theta,
                      sky, 0.08 + 0.1 * max(0.0, X - 1.5), transmission, trail)

        # Guide samples during the exposure (and the settle time before it).
        for tg in np.arange(t0 - DITHER_SETTLE + 1, t1, GUIDE_CADENCE):
            ra = rng.normal(0, guide_sigma)
            dec = rng.normal(0, guide_sigma * 0.8)
            if guide_extra and t0 <= tg:
                kind, amp = guide_extra
                frac = (tg - t0) / EXPOSURE
                if kind == "spike" and 0.3 < frac < 0.75:
                    ra += amp * math.sin(math.pi * (frac - 0.3) / 0.45)
                elif kind == "settle":
                    ra += amp * math.exp(-frac / 0.25) * math.cos(12 * frac)
                    dec += 0.6 * amp * math.exp(-frac / 0.25)
            snr = max(1.0, rng.normal(guide_snr, 2.0))
            dropped = snr < 6.0 and rng.uniform() < 0.5
            guide_rows.append((tg, ra, dec, snr, dropped))

        local = _local(t0)
        alt0, az0 = altaz(t0)
        if flavor == "ekos":
            name = f"M_33_Light_L_{int(EXPOSURE)}_secs_{local:%Y-%m-%dT%H-%M-%S}_{i + 1:03d}.{image_format}"
            date_obs = datetime.fromtimestamp(t0, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
            cards = [
                ("ROWORDER", "TOP-DOWN"), ("INSTRUME", "CCD Simulator"), ("TELESCOP", "Telescope Simulator"),
                ("OBSERVER", "Unknown"), ("OBJECT", TARGET), ("EXPTIME", EXPOSURE), ("DARKTIME", EXPOSURE),
                ("CCD-TEMP", -10.0), ("PIXSIZE1", PIX_UM), ("PIXSIZE2", PIX_UM), ("XBINNING", 1),
                ("YBINNING", 1), ("XPIXSZ", PIX_UM), ("YPIXSZ", PIX_UM), ("FRAME", "Light"),
                ("IMAGETYP", "Light Frame"), ("FILTER", "L"), ("GAIN", 100.0), ("OFFSET", 30.0),
                ("FOCALLEN", FOCAL_MM), ("APTDIA", 160.0), ("FOCUSPOS", focus_pos),
                ("FOCUSTEM", round(current_temp, 2)), ("SCALE", round(SCALE, 6)), ("SITELAT", SITE_LAT),
                ("SITELONG", SITE_LON), ("AIRMASS", round(airmass(alt0), 4)),
                ("OBJCTRA", _sexa(TARGET_RA / 15.0)), ("OBJCTDEC", _sexa(TARGET_DEC)),
                ("RA", TARGET_RA), ("DEC", TARGET_DEC), ("OBJCTALT", round(alt0, 4)),
                ("OBJCTAZ", round(az0, 4)), ("PIERSIDE", "WEST" if pier == 0 else "EAST"),
                ("EQUINOX", 2000), ("DATE-OBS", date_obs),
            ]
        else:
            name = f"{local:%Y-%m-%d_%H-%M-%S}_L_-10.00_{EXPOSURE:.2f}s_{i:04d}.{image_format}"
            utc = datetime.fromtimestamp(t0, tz=timezone.utc)
            cards = [
                ("IMAGETYP", "LIGHT"), ("EXPOSURE", EXPOSURE), ("EXPTIME", EXPOSURE),
                ("DATE-LOC", local.strftime("%Y-%m-%dT%H:%M:%S.%f") + "0"),
                ("DATE-OBS", utc.strftime("%Y-%m-%dT%H:%M:%S.%f") + "0"),
                ("XBINNING", 1), ("YBINNING", 1), ("GAIN", 100), ("OFFSET", 30), ("EGAIN", 1.0),
                ("XPIXSZ", PIX_UM), ("YPIXSZ", PIX_UM), ("INSTRUME", "Camera Simulator"),
                ("SET-TEMP", -10.0), ("CCD-TEMP", -10.0), ("TELESCOP", "Telescope Simulator"),
                ("FOCALLEN", FOCAL_MM), ("FOCRATIO", 5.0), ("RA", TARGET_RA), ("DEC", TARGET_DEC),
                ("CENTALT", round(alt0, 4)), ("CENTAZ", round(az0, 4)), ("AIRMASS", round(airmass(alt0), 4)),
                ("PIERSIDE", "West" if pier == 0 else "East"), ("SITEELEV", 120.0),
                ("SITELAT", SITE_LAT), ("SITELONG", SITE_LON), ("OBJECT", TARGET),
                ("OBJCTRA", _sexa(TARGET_RA / 15.0)), ("OBJCTDEC", _sexa(TARGET_DEC)),
                ("FOCNAME", "Focuser Simulator"), ("FOCPOS", focus_pos), ("FOCUSPOS", focus_pos),
                ("FOCTEMP", round(current_temp, 2)), ("FOCUSTEM", round(current_temp, 2)),
                ("FWHEEL", "Filter Wheel Simulator"), ("FILTER", "L"), ("ROWORDER", "TOP-DOWN"),
                ("EQUINOX", 2000.0), ("SWCREATE", "N.I.N.A. 3.1.2.9001 (x64)"),
            ]
        if image_format == "xisf":
            xisf.write(lights / name, img, [(k, v, "") for k, v in cards])
        else:
            hdr = fits.Header()
            for key, val in cards:
                hdr[key] = val
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fits.PrimaryHDU(img, header=hdr).writeto(lights / name, overwrite=True)

        if flavor == "nina":
            win = [r for r in guide_rows if t0 <= r[0] <= t1 and not r[4]]
            g_ra = float(np.std([r[1] for r in win])) if len(win) > 2 else 0.0
            g_dec = float(np.std([r[2] for r in win])) if len(win) > 2 else 0.0
            nina_hfr = fwhm_arcsec / SCALE / 2.0 * (1 + 0.15 * trail_len / 7) * math.exp(rng.normal(0, 0.03))
            nina_rows.append({
                "ExposureNumber": i, "FilePath": "C:\\Users\\astro\\Documents\\N.I.N.A\\2026-11-18\\LIGHT\\" + name,
                "FilterName": "L", "ExposureStart": local.strftime("%Y-%m-%d %H:%M"), "Duration": EXPOSURE,
                "Binning": "1x1", "CameraTemp": -10.0, "CameraTargetTemp": -10.0, "Gain": 100, "Offset": 30,
                "ADUStDev": 40.0, "ADUMean": round(sky + 300, 1), "ADUMedian": round(sky + 290, 1), "ADUMin": 250,
                "ADUMax": 65535, "DetectedStars": int(180 * transmission), "HFR": round(nina_hfr, 4),
                "HFRStDev": 0.3, "FWHM": "NaN", "Eccentricity": "NaN",
                "GuidingRMS": round(math.hypot(g_ra, g_dec) / GUIDE_SCALE, 4),
                "GuidingRMSArcSec": round(math.hypot(g_ra, g_dec), 4),
                "GuidingRMSRA": round(g_ra / GUIDE_SCALE, 4), "GuidingRMSRAArcSec": round(g_ra, 4),
                "GuidingRMSDEC": round(g_dec / GUIDE_SCALE, 4), "GuidingRMSDECArcSec": round(g_dec, 4),
                "FocuserPosition": focus_pos, "FocuserTemp": round(current_temp, 2), "RotatorPosition": 0,
                "PierSide": "West" if pier == 0 else "East", "Airmass": round(airmass(alt0), 4),
                "ExposureStartUTC": datetime.fromtimestamp(t0, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "0Z",
                "MountRA": TARGET_RA, "MountDec": TARGET_DEC, "ImageType": "LIGHT",
            })

        # Ekos' own view of the frame.
        ev(t0, "CaptureStarting", f"{EXPOSURE:.3f}", "L", "CCD Simulator")
        mount_alt, mount_az = altaz(t0)
        ev(t0, "MountCoords", f"{TARGET_RA:.4f}", f"{TARGET_DEC:.4f}", f"{mount_az:.4f}", f"{mount_alt:.4f}",
           pier, f"{hour_angle_h(t0) * 15:.4f}")
        hfr = fwhm_arcsec / SCALE / 2.0 * (1 + 0.15 * trail_len / 7) * math.exp(rng.normal(0, 0.03))
        ev(t1 + DOWNLOAD - 1, "CaptureComplete", f"{EXPOSURE:.3f}", "L", f"{hfr:.3f}",
           f"{CAPTURE_ROOT}/{name}", int(180 * transmission), int(sky), f"{min(0.9, 0.25 + trail_len / 12):.3f}",
           "CCD Simulator")
        truth[name] = {"index": i, "label": label, "alt": round(alt, 2), "fwhm_arcsec": round(fwhm_arcsec, 3),
                       "transmission": transmission, "pier": "WEST" if pier == 0 else "EAST"}
        frame_info.append((t0, t1))
        t = t1 + DOWNLOAD

    ev(t + 10, "GuideState", "Idle")
    guide_sections.append((guide_on, t + 10))
    ev(t + 20, "MountState", "Parking")
    ev(t + 80, "MountState", "Parked")

    out = {"lights": lights, "truth": truth}
    rows = sorted(guide_rows, key=lambda r: r[0])
    if flavor == "ekos":
        out["analyze"] = _write_analyze(outdir, session_start, analyze, guide_rows)
        out["guide_log"] = _write_guide_log(
            outdir / f"guide_log-{_local(session_start + 150):%Y-%m-%dT%H-%M-%S}.txt",
            f"KStars version 3.8.0. PHD2 log version 2.5. Log enabled at {_local(session_start + 150):%Y-%m-%d %H:%M:%S}",
            rows, guide_sections, dithers, t, phd2=False)
    else:
        out["guide_log"] = _write_guide_log(
            outdir / f"PHD2_GuideLog_{_local(session_start + 150):%Y-%m-%d_%H%M%S}.txt",
            f"PHD2 version 2.6.13, Log version 2.5. Log enabled at {_local(session_start + 150):%Y-%m-%d %H:%M:%S}",
            rows, guide_sections, dithers, t, phd2=True)
        out["nina_metadata"] = _write_nina_metadata(lights, nina_rows)
        out["af_reports"] = _write_af_reports(outdir / "AutoFocus", af_events)

    meta = {
        "generator": "nykteris.synth",
        "flavor": flavor,
        "image_format": image_format,
        "seed": seed,
        "site": {"lat": SITE_LAT, "lon": SITE_LON, "tz_hours": TZ_HOURS},
        "target": TARGET,
        "scale_arcsec_px": SCALE,
        "lights": str(lights.relative_to(outdir)),
        "alt_range": [round(min(alts), 1), round(max(alts), 1)],
    }
    (outdir / "truth.json").write_text(json.dumps({"meta": meta, "frames": truth}, indent=1))
    out["meta"] = meta
    return out


def _write_analyze(outdir: Path, session_start: float, analyze, guide_rows) -> Path:
    stamp = _local(session_start)
    path = outdir / f"ekos-{stamp:%Y-%m-%dT%H-%M-%S}.analyze"
    with open(path, "w") as fh:
        fh.write("#KStars version 3.8.0. Analyze log version 2.0.\n\n")
        fh.write(f"AnalyzeStartTime,{stamp:%Y-%m-%d %H:%M:%S}.000,CET\n")
        guide_lines = [(r[0], "GuideStats,{:.3f},{:.3f},{:.3f},{},{},{:.3f},{:.3f},{}".format(
            r[0] - session_start, r[1], r[2], int(abs(r[1]) * 80), int(abs(r[2]) * 80), r[3], 120.0, 1))
            for r in guide_rows if not r[4]]
        for _, line in sorted(analyze + guide_lines, key=lambda x: x[0]):
            fh.write(line + "\n")
    return path


def _write_guide_log(path: Path, first_line: str, rows, sections, dithers, t_end: float, phd2: bool) -> Path:
    """Ekos internal guider or PHD2 log. PHD2 adds an ErrorDescription column."""
    cols = ("Frame,Time,mount,dx,dy,RARawDistance,DECRawDistance,RAGuideDistance,DECGuideDistance,"
            "RADuration,RADirection,DECDuration,DECDirection,XStep,YStep,StarMass,SNR,ErrorCode")
    if phd2:
        cols += ",ErrorDescription"
    with open(path, "w") as fh:
        fh.write(first_line + "\n\n")
        di = 0
        for a, b in sections:
            begin = math.floor(a)
            fh.write(f"Guiding Begins at {_local(begin):%Y-%m-%d %H:%M:%S}\n")
            if phd2:
                fh.write("Dither = both axes, Dither scale = 1.000, Image noise reduction = none, "
                         "Guide-frame time lapse = 0, Server enabled\n")
            fh.write(f"Pixel scale = {GUIDE_SCALE:.2f} arc-sec/px, Binning = 1, Focal length = 200 mm\n")
            fh.write(f"RA = {TARGET_RA / 15:.2f} hr, Dec = {TARGET_DEC:.1f} deg, Hour angle = N/A hr, Pier side = West, "
                     "Rotator pos = N/A, Alt = 50.0 deg, Az = 120.0 deg\n")
            fh.write("Mount = mount, xAngle = 90.0, xRate = 7.500, yAngle = 0.0, yRate = 7.500\n")
            fh.write(cols + "\n")
            n = 1
            for r in rows:
                if not (a <= r[0] < b):
                    continue
                while di < len(dithers) and dithers[di] <= r[0]:
                    fh.write("INFO: DITHER by 3.000, -2.000, new lock pos = 400.000, 300.000\n")
                    fh.write("INFO: SETTLING STATE CHANGE, Settling started\n")
                    di += 1
                ra_px, dec_px = r[1] / GUIDE_SCALE, r[2] / GUIDE_SCALE
                if r[4]:
                    tail = ',"Star lost - low SNR"' if phd2 else ""
                    fh.write(f'{n},{r[0] - begin:.3f},"DROP",,,,,,,0,,0,,,,0,{r[3]:.2f},{2 if phd2 else 1}{tail}\n')
                else:
                    fh.write(f'{n},{r[0] - begin:.3f},"Mount",{ra_px:.3f},{dec_px:.3f},{ra_px:.3f},{dec_px:.3f},'
                             f'{-0.7 * ra_px:.3f},{-0.7 * dec_px:.3f},{int(abs(ra_px) * 300)},{"W" if ra_px < 0 else "E"},'
                             f'{int(abs(dec_px) * 300)},{"N" if dec_px < 0 else "S"},,,{int(r[3] * 120)},{r[3]:.2f},0'
                             + (',""' if phd2 else "") + "\n")
                n += 1
            fh.write(f"Guiding Ends at {_local(b):%Y-%m-%d %H:%M:%S}\n\n")
        fh.write(f"Log closed at {_local(t_end + 100):%Y-%m-%d %H:%M:%S}\n")
    return path


def _write_nina_metadata(folder: Path, rows: list[dict]) -> Path:
    """ImageMetaData.csv and .json, as the Session Metadata plugin writes them."""
    import csv

    with open(folder / "ImageMetaData.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    js = [{k: (float("nan") if v == "NaN" else v) for k, v in r.items()} for r in rows]
    (folder / "ImageMetaData.json").write_text(json.dumps(js, indent=2))
    return folder / "ImageMetaData.json"


def _write_af_reports(folder: Path, events) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for t_end, temp, pos in events:
        local = _local(t_end)
        report = {
            "Version": 2, "Filter": "L", "AutoFocuserName": "NINA", "StarDetectorName": "NINA",
            "Timestamp": local.strftime("%Y-%m-%dT%H:%M:%S.%f") + f"0+{TZ_HOURS:02d}:00",
            "Temperature": round(temp, 2), "Method": "STARHFR", "Fitting": "HYPERBOLIC",
            "InitialFocusPoint": {"Position": pos - 40, "Value": 3.1, "Error": 0.1},
            "CalculatedFocusPoint": {"Position": pos, "Value": 2.6, "Error": 0.0},
            "PreviousFocusPoint": {"Position": pos - 60, "Value": 0.0, "Error": 0.0},
            "MeasurePoints": [{"Position": pos + d, "Value": 2.6 + abs(d) / 60, "Error": 0.1} for d in (-90, -45, 0, 45, 90)],
            "Duration": "00:04:00",
        }
        (folder / f"{local:%Y-%m-%d--%H-%M-%S}.json").write_text(json.dumps(report, indent=2))
    return folder


def _sexa(v: float) -> str:
    sign = "-" if v < 0 else ""
    v = abs(v)
    d = int(v)
    m = int((v - d) * 60)
    s = (v - d - m / 60) * 3600
    return f"{sign}{d:02d} {m:02d} {s:05.2f}"
