"""Discover light frames and read the FITS header keys that INDI writes."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from astropy.io import fits

FITS_SUFFIXES = {".fits", ".fit", ".fts"}


@dataclass
class Frame:
    path: Path
    name: str
    date_obs: float  # exposure start, UTC unix seconds
    exptime: float
    filter: str = ""
    binning: int = 1
    imagetyp: str = ""
    object: str = ""
    ra_deg: float | None = None
    dec_deg: float | None = None
    site_lat: float | None = None
    site_lon: float | None = None
    scale: float | None = None  # arcsec per (binned) pixel
    ccd_temp: float | None = None
    focus_pos: float | None = None
    focus_temp: float | None = None
    pier_side: str | None = None  # "EAST" / "WEST"
    header_alt: float | None = None
    gain: float | None = None
    instrument: str = ""
    telescope: str = ""
    bayer: str | None = None
    # Filled later
    metrics: dict = field(default_factory=dict)
    ctx: dict = field(default_factory=dict)
    verdict: str = "ok"
    causes: list = field(default_factory=list)

    @property
    def date_end(self) -> float:
        return self.date_obs + self.exptime

    @property
    def group(self) -> tuple:
        return (self.filter or "-", round(self.exptime, 3), self.binning)


def _num(h, key):
    v = h.get(key)
    if v is None or isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _sexagesimal(value: str, hours: bool) -> float | None:
    try:
        parts = [float(p) for p in str(value).replace(":", " ").split()]
    except ValueError:
        return None
    if not parts:
        return None
    sign = -1.0 if str(value).strip().startswith("-") else 1.0
    parts[0] = abs(parts[0])
    while len(parts) < 3:
        parts.append(0.0)
    v = sign * (parts[0] + parts[1] / 60 + parts[2] / 3600)
    return v * 15.0 if hours else v


def parse_date_obs(value: str) -> float:
    s = str(value).strip().rstrip("Z")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def read_frame(path: Path) -> Frame | None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with fits.open(path, memmap=False) as hdul:
            h = hdul[0].header
            if hdul[0].data is None and len(hdul) > 1:
                h = hdul[1].header
    if "DATE-OBS" not in h:
        return None
    exptime = _num(h, "EXPTIME")
    if exptime is None:
        exptime = _num(h, "EXPOSURE") or 0.0
    binning = int(_num(h, "XBINNING") or 1)
    scale = _num(h, "SCALE")
    if scale is None:
        pix = _num(h, "PIXSIZE1") or _num(h, "XPIXSZ")
        fl = _num(h, "FOCALLEN")
        if pix and fl:
            # PIXSIZE1 is the unbinned pixel size; XPIXSZ already includes binning.
            if _num(h, "PIXSIZE1"):
                pix = pix * binning
            scale = 206.265 * pix / fl
    ra = _num(h, "RA")
    dec = _num(h, "DEC")
    if ra is None and h.get("OBJCTRA"):
        ra = _sexagesimal(h["OBJCTRA"], hours=True)
    if dec is None and h.get("OBJCTDEC"):
        dec = _sexagesimal(h["OBJCTDEC"], hours=False)
    lat = _num(h, "SITELAT")
    lon = _num(h, "SITELONG")
    if lat is None and isinstance(h.get("SITELAT"), str):
        lat = _sexagesimal(h["SITELAT"], hours=False)
    if lon is None and isinstance(h.get("SITELONG"), str):
        lon = _sexagesimal(h["SITELONG"], hours=False)
    imagetyp = str(h.get("IMAGETYP") or h.get("FRAME") or "")
    pier = h.get("PIERSIDE")
    return Frame(
        path=path,
        name=path.name,
        date_obs=parse_date_obs(h["DATE-OBS"]),
        exptime=float(exptime),
        filter=str(h.get("FILTER", "") or "").strip(),
        binning=binning,
        imagetyp=imagetyp,
        object=str(h.get("OBJECT", "") or "").strip(),
        ra_deg=ra,
        dec_deg=dec,
        site_lat=lat,
        site_lon=lon,
        scale=scale,
        ccd_temp=_num(h, "CCD-TEMP"),
        focus_pos=_num(h, "FOCUSPOS"),
        focus_temp=_num(h, "FOCUSTEM"),
        pier_side=str(pier).strip().upper() if pier else None,
        header_alt=_num(h, "OBJCTALT"),
        gain=_num(h, "GAIN"),
        instrument=str(h.get("INSTRUME", "") or ""),
        telescope=str(h.get("TELESCOP", "") or ""),
        bayer=str(h["BAYERPAT"]).strip() if h.get("BAYERPAT") else None,
    )


def is_light(frame: Frame) -> bool:
    t = frame.imagetyp.lower()
    return (not t) or ("light" in t)


def discover(folder: Path) -> list[Frame]:
    """All light frames below ``folder``, sorted by exposure start."""
    frames = []
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() in FITS_SUFFIXES and p.is_file() and "rejected" not in p.relative_to(folder).parts:
            f = read_frame(p)
            if f is not None and is_light(f):
                frames.append(f)
    frames.sort(key=lambda f: f.date_obs)
    return frames
