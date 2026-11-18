"""Discover light frames and read the FITS header keys that INDI writes."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from astropy.io import fits

from . import xisf

FITS_SUFFIXES = {".fits", ".fit", ".fts"}
IMAGE_SUFFIXES = FITS_SUFFIXES | {".xisf"}


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
    date_loc: float | None = None  # DATE-LOC as wall time "as if UTC" (NINA)
    software: str = ""
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


def _iso(value: str) -> datetime:
    """ISO 8601 with up to 7 fractional digits (NINA) and an optional zone."""
    s = str(value).strip().replace(" ", "T")
    tz = None
    if s.endswith("Z"):
        s, tz = s[:-1], timezone.utc
    m = re.match(r"^(.*?T\d{2}:\d{2}(?::\d{2})?)(\.\d+)?([+-]\d{2}:?\d{2})?$", s)
    if m:
        frac = m.group(2) or ""
        s = m.group(1) + (("." + frac[1:7].ljust(6, "0")) if frac else "") + (m.group(3) or "")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz or timezone.utc)
    return dt


def parse_date_obs(value: str) -> float:
    return _iso(value).timestamp()


def _read_header(path: Path):
    if path.suffix.lower() == ".xisf":
        return xisf.read_header(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with fits.open(path, memmap=False) as hdul:
            h = hdul[0].header
            if hdul[0].data is None and len(hdul) > 1:
                h = hdul[1].header
            return h.copy()


def read_frame(path: Path) -> Frame | None:
    try:
        h = _read_header(path)
    except (OSError, ValueError) as e:
        warnings.warn(f"Skipping {path.name}: {e}")
        return None
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
            # PIXSIZE1 (INDI) is the unbinned pixel size; XPIXSZ (NINA) already includes binning.
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
    date_loc = None
    if h.get("DATE-LOC"):
        try:
            date_loc = _iso(h["DATE-LOC"]).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            date_loc = None
    alt = _num(h, "OBJCTALT")
    if alt is None:
        alt = _num(h, "CENTALT")
    focus_pos = _num(h, "FOCUSPOS")
    if focus_pos is None:
        focus_pos = _num(h, "FOCPOS")
    focus_temp = _num(h, "FOCUSTEM")
    if focus_temp is None:
        focus_temp = _num(h, "FOCTEMP")
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
        focus_pos=focus_pos,
        focus_temp=focus_temp,
        pier_side=str(pier).strip().upper() if pier else None,
        header_alt=alt,
        gain=_num(h, "GAIN"),
        instrument=str(h.get("INSTRUME", "") or ""),
        telescope=str(h.get("TELESCOP", "") or ""),
        bayer=str(h["BAYERPAT"]).strip() if h.get("BAYERPAT") else None,
        date_loc=date_loc,
        software=str(h.get("SWCREATE", "") or "").strip(),
    )


def is_light(frame: Frame) -> bool:
    t = frame.imagetyp.lower()
    return (not t) or ("light" in t)


def discover(folder: Path) -> list[Frame]:
    """All light frames below ``folder``, sorted by exposure start."""
    frames = []
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file() and "rejected" not in p.relative_to(folder).parts:
            f = read_frame(p)
            if f is not None and is_light(f):
                frames.append(f)
    frames.sort(key=lambda f: f.date_obs)
    return frames
