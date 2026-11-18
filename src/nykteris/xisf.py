"""Minimal XISF 1.0 reader and writer (the default N.I.N.A. file format).

Layout: the signature ``XISF0100``, a little-endian uint32 header length, four
reserved bytes, then an XML header. Each ``<Image>`` element gives
``geometry="width:height:channels"``, ``sampleFormat`` and
``location="attachment:<offset>:<size>"``; pixel data is planar and
little-endian. ``compression="<codec>[+sh]:<uncompressed size>[:<item size>]"``
names the codec (zlib, lz4, lz4hc, zstd) and optional byte shuffling.
N.I.N.A. copies its FITS keywords into ``<FITSKeyword name value comment>``
elements, so the same header logic works for both formats.

zlib is in the standard library. lz4 and zstd need the optional ``lz4`` and
``zstandard`` packages.
"""

from __future__ import annotations

import base64
import struct
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

import numpy as np

SIGNATURE = b"XISF0100"
NS = "{http://www.pixinsight.com/xisf}"
DTYPES = {
    "UInt8": "<u1", "UInt16": "<u2", "UInt32": "<u4",
    "Float32": "<f4", "Float64": "<f8",
}


class XISFError(ValueError):
    pass


def _header_xml(fh) -> ET.Element:
    sig = fh.read(8)
    if sig != SIGNATURE:
        raise XISFError("not an XISF 1.0 file")
    (length,) = struct.unpack("<I", fh.read(4))
    fh.read(4)
    return ET.fromstring(fh.read(length).decode("utf-8"))


def _image_element(root: ET.Element) -> ET.Element:
    img = root.find(f"{NS}Image")
    if img is None:
        img = root.find("Image")
    if img is None:
        raise XISFError("no Image element")
    return img


def _fits_value(raw: str):
    v = raw.strip()
    if v.startswith("'"):
        return v[1:-1].replace("''", "'").strip() if v.endswith("'") else v[1:].strip()
    if v in ("T", "F"):
        return v == "T"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def read_header(path: str | Path) -> dict:
    """FITS keywords of the first image as a plain dict (first occurrence wins)."""
    with open(path, "rb") as fh:
        root = _header_xml(fh)
    img = _image_element(root)
    out: dict = {}
    for kw in list(img.iter(f"{NS}FITSKeyword")) + list(img.iter("FITSKeyword")):
        name = kw.get("name", "").strip().upper()
        if name and name not in out and name not in ("COMMENT", "HISTORY"):
            out[name] = _fits_value(kw.get("value", ""))
    # XISF properties as fallbacks for keys a writer may have left out.
    props = {p.get("id"): p.get("value", p.text) for p in list(img.iter(f"{NS}Property")) + list(img.iter("Property"))}
    if "DATE-OBS" not in out and props.get("Observation:Time:Start"):
        out["DATE-OBS"] = props["Observation:Time:Start"]
    if "EXPTIME" not in out and props.get("Instrument:ExposureTime"):
        out["EXPTIME"] = float(props["Instrument:ExposureTime"])
    if "IMAGETYP" not in out and img.get("imageType"):
        out["IMAGETYP"] = img.get("imageType")
    return out


def _decompress(data: bytes, spec: str) -> bytes:
    parts = spec.split(":")
    codec = parts[0].lower()
    shuffled = codec.endswith("+sh")
    codec = codec.replace("+sh", "")
    size = int(parts[1])
    if codec == "zlib":
        out = zlib.decompress(data)
    elif codec in ("lz4", "lz4hc"):
        try:
            import lz4.block
        except ImportError as e:  # pragma: no cover
            raise XISFError("LZ4-compressed XISF needs the 'lz4' package (pip install lz4)") from e
        out = lz4.block.decompress(data, uncompressed_size=size)
    elif codec == "zstd":
        try:
            import zstandard
        except ImportError as e:  # pragma: no cover
            raise XISFError("Zstandard-compressed XISF needs the 'zstandard' package") from e
        out = zstandard.ZstdDecompressor().decompress(data, max_output_size=size)
    else:
        raise XISFError(f"unsupported XISF compression codec '{codec}'")
    if shuffled:
        item = int(parts[2]) if len(parts) > 2 else 1
        out = unshuffle(out, item)
    return out


def shuffle(data: bytes, item: int) -> bytes:
    n = len(data) // item
    head = np.frombuffer(data, np.uint8, count=n * item).reshape(n, item).T.tobytes()
    return head + data[n * item:]


def unshuffle(data: bytes, item: int) -> bytes:
    n = len(data) // item
    head = np.frombuffer(data, np.uint8, count=n * item).reshape(item, n).T.tobytes()
    return head + data[n * item:]


def read_data(path: str | Path) -> np.ndarray:
    """Pixel data of the first image: (height, width) or (channels, height, width)."""
    path = Path(path)
    with open(path, "rb") as fh:
        root = _header_xml(fh)
        img = _image_element(root)
        geom = [int(x) for x in img.get("geometry", "").split(":")]
        if len(geom) < 2:
            raise XISFError("bad geometry")
        width, height = geom[0], geom[1]
        channels = geom[2] if len(geom) > 2 else 1
        fmt = img.get("sampleFormat", "UInt16")
        if fmt not in DTYPES:
            raise XISFError(f"unsupported sample format {fmt}")
        loc = img.get("location", "").split(":")
        if loc[0] == "attachment":
            fh.seek(int(loc[1]))
            raw = fh.read(int(loc[2]))
        elif loc[0] == "inline" and img.text:
            raw = base64.b64decode(img.text.strip())
        else:
            raise XISFError(f"unsupported data location '{img.get('location')}'")
    if img.get("compression"):
        raw = _decompress(raw, img.get("compression"))
    data = np.frombuffer(raw, DTYPES[fmt])
    if img.get("pixelStorage", "Planar").lower() == "normal" and channels > 1:
        data = data.reshape(height, width, channels).transpose(2, 0, 1)
    else:
        data = data.reshape(channels, height, width)
    return data[0] if channels == 1 else data


def _fits_text(value) -> str:
    if isinstance(value, bool):
        return "T" if value else "F"
    if isinstance(value, (int, float)):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def write(path: str | Path, data: np.ndarray, keywords: list[tuple[str, object, str]],
          image_type: str = "Light", compress: bool = True) -> None:
    """Write a single-channel UInt16 image the way N.I.N.A. does (zlib + byte shuffle)."""
    data = np.ascontiguousarray(data, dtype="<u2")
    h, w = data.shape
    raw = data.tobytes()
    if compress:
        payload = zlib.compress(shuffle(raw, 2), 6)
        compression = f' compression="zlib+sh:{len(raw)}:2"'
    else:
        payload = raw
        compression = ""
    kws = "".join(
        f'<FITSKeyword name="{k}" value="{_xml(_fits_text(v))}" comment="{_xml(c)}"/>' for k, v, c in keywords
    )

    def header(offset: int) -> bytes:
        xml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<xisf version="1.0" xmlns="http://www.pixinsight.com/xisf">'
            f'<Image geometry="{w}:{h}:1" sampleFormat="UInt16" bounds="0:65535" colorSpace="Gray" '
            f'imageType="{image_type}" location="attachment:{offset}:{len(payload)}"{compression}>'
            f"{kws}</Image>"
            '<Metadata><Property id="XISF:CreatorApplication" type="String">Nykteris synthetic night</Property></Metadata>'
            "</xisf>"
        )
        return xml.encode("utf-8")

    # The offset is part of the header, so size the header first and pad to a block.
    probe = header(0)
    offset = ((16 + len(probe) + 64) // 4096 + 1) * 4096
    hdr = header(offset)
    with open(path, "wb") as fh:
        fh.write(SIGNATURE + struct.pack("<I", len(hdr)) + b"\0\0\0\0" + hdr)
        fh.write(b"\0" * (offset - 16 - len(hdr)))
        fh.write(payload)


def _xml(s: str) -> str:
    return s.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")
