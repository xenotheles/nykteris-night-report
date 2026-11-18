"""Per-frame image metrics measured with sep (Source Extractor as a library).

Measured for every light:

- ``stars``: number of star-like detections
- ``hfr_px``: median half-flux radius
- ``fwhm_px``: 2 x HFR, the FWHM of a Gaussian with that HFR
- ``eccentricity``: median sqrt(1 - (b/a)^2) of the brighter stars
- ``ecc_coherence``: how well the elongation angles agree, 0 (random) to 1
- ``elong_angle_deg``: mean elongation angle (image x axis = 0)
- ``background``, ``noise``: global sky level and RMS (ADU)
- ``gradient_pct``: 5 to 95 percentile spread of the background map
- ``trail_length_px``, ``trail_axis_ratio``: the most line-like detection

The results do not depend on thresholds, so they can be cached.
"""

from __future__ import annotations

import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import sep
from astropy.io import fits

from . import xisf

METRICS_VERSION = 1

sep.set_extract_pixstack(3_000_000)
sep.set_sub_object_limit(4096)


def _load(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".xisf":
        data = np.asarray(xisf.read_data(path), dtype=np.float64)
        bayer = xisf.read_header(path).get("BAYERPAT")
        return _prepare(data, bayer)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with fits.open(path, memmap=False) as hdul:
            hdu = hdul[0] if hdul[0].data is not None else hdul[1]
            data = np.asarray(hdu.data, dtype=np.float64)
            bayer = hdu.header.get("BAYERPAT")
    return _prepare(data, bayer)


def _prepare(data: np.ndarray, bayer) -> np.ndarray:
    if data.ndim == 3:
        data = data.mean(axis=0) if data.shape[0] <= 4 else data[0]
    if bayer:
        # Super-pixel 2x2 sum: a luminance image without demosaicing artefacts.
        h, w = (data.shape[0] // 2) * 2, (data.shape[1] // 2) * 2
        d = data[:h, :w]
        data = d[0::2, 0::2] + d[1::2, 0::2] + d[0::2, 1::2] + d[1::2, 1::2]
    return np.ascontiguousarray(data)


def measure(path: str | Path) -> dict:
    path = Path(path)
    data = _load(path)
    sat = float(np.nanmax(data))
    bkg = sep.Background(data, bw=64, bh=64, fw=3, fh=3)
    back = bkg.back()
    sub = data - back
    rms = float(bkg.globalrms)
    objs = sep.extract(sub, 4.0, err=rms, minarea=5, deblend_cont=0.005)

    ny, nx = data.shape
    out = {
        "version": METRICS_VERSION,
        "width": nx,
        "height": ny,
        "background": float(bkg.globalback),
        "noise": rms,
        "gradient_pct": float(
            100.0 * (np.percentile(back, 95) - np.percentile(back, 5)) / max(abs(bkg.globalback), 1e-9)
        ),
        "stars": 0,
        "hfr_px": float("nan"),
        "fwhm_px": float("nan"),
        "eccentricity": float("nan"),
        "ecc_coherence": float("nan"),
        "elong_angle_deg": float("nan"),
        "trail_length_px": 0.0,
        "trail_axis_ratio": 0.0,
        "saturated_stars": 0,
    }
    if len(objs) == 0:
        return out

    a = objs["a"]
    b = objs["b"]
    ratio = a / np.maximum(b, 1e-6)

    # Line-like objects: satellites, planes, meteors. sigma_major of a line
    # of length L is L / sqrt(12).
    length = a * np.sqrt(12.0)
    linear = ratio >= 3.0
    if np.any(linear):
        k = int(np.argmax(np.where(linear, length, 0)))
        out["trail_length_px"] = float(length[k])
        out["trail_axis_ratio"] = float(ratio[k])

    edge = 8
    peak_total = objs["peak"] + back[objs["y"].astype(int).clip(0, ny - 1), objs["x"].astype(int).clip(0, nx - 1)]
    saturated = peak_total >= 0.95 * sat if sat > 1000 else np.zeros(len(objs), bool)
    good = (
        (ratio < 3.0)
        & (objs["npix"] >= 5)
        & (objs["x"] > edge) & (objs["x"] < nx - edge)
        & (objs["y"] > edge) & (objs["y"] < ny - edge)
        & ~saturated
        & (objs["flag"] & sep.OBJ_TRUNC == 0)
    )
    out["saturated_stars"] = int(saturated.sum())
    stars = objs[good]
    out["stars"] = int(len(stars))
    if len(stars) < 3:
        return out

    # One aperture for all stars, sized from the brighter stars' moments, so
    # faint stars are not truncated by their small isophotal footprint.
    a_bright = np.median(stars["a"][stars["flux"] >= np.median(stars["flux"])])
    rmax = np.full(len(stars), float(np.clip(3.0 * 2.3548 * a_bright, 5.0, 40.0)))
    hfr, flag = sep.flux_radius(sub, stars["x"], stars["y"], rmax, 0.5, subpix=5)
    ok = (flag == 0) & np.isfinite(hfr) & (hfr > 0.3)
    if np.any(ok):
        out["hfr_px"] = float(np.median(hfr[ok]))
        out["fwhm_px"] = 2.0 * out["hfr_px"]

    # Shape from the brighter half, where isophotal moments are reliable.
    bright = stars[stars["flux"] >= np.median(stars["flux"])]
    sa, sb = bright["a"], bright["b"]
    ecc = np.sqrt(np.clip(1.0 - (sb / np.maximum(sa, 1e-6)) ** 2, 0.0, 1.0))
    out["eccentricity"] = float(np.median(ecc))
    z = np.exp(2j * bright["theta"])
    w = ecc
    if w.sum() > 0:
        mean = np.sum(w * z) / w.sum()
        out["ecc_coherence"] = float(abs(mean))
        out["elong_angle_deg"] = float(np.degrees(0.5 * np.angle(mean)))
    return out


class MetricsCache:
    def __init__(self, path: Path | None):
        self.path = path
        self.data: dict = {}
        if path and path.exists():
            try:
                self.data = json.loads(path.read_text())
            except (OSError, ValueError):
                self.data = {}

    @staticmethod
    def key(p: Path) -> str:
        st = p.stat()
        return f"{p.resolve()}|{st.st_size}|{st.st_mtime_ns}|{METRICS_VERSION}"

    def get(self, p: Path):
        return self.data.get(self.key(p))

    def put(self, p: Path, m: dict) -> None:
        self.data[self.key(p)] = m

    def save(self) -> None:
        if self.path:
            self.path.write_text(json.dumps(self.data))


def measure_all(paths: list[Path], cache: MetricsCache, jobs: int | None = None, progress=None) -> list[dict]:
    results: list[dict | None] = [cache.get(p) for p in paths]
    todo = [i for i, r in enumerate(results) if r is None]
    if todo:
        jobs = jobs or min(os.cpu_count() or 1, 8)
        if jobs > 1 and len(todo) > 4:
            with ProcessPoolExecutor(max_workers=jobs) as ex:
                for n, (i, m) in enumerate(zip(todo, ex.map(measure, [paths[i] for i in todo], chunksize=4))):
                    results[i] = m
                    cache.put(paths[i], m)
                    if progress:
                        progress(n + 1, len(todo))
        else:
            for n, i in enumerate(todo):
                results[i] = measure(paths[i])
                cache.put(paths[i], results[i])
                if progress:
                    progress(n + 1, len(todo))
        cache.save()
    return results  # type: ignore[return-value]
