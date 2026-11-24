"""Strict read-only, Hebrew/Unicode-safe image & EXIF I/O for Windows."""

from __future__ import annotations

import datetime as dt
import io
import logging
import re
import struct
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple

import cv2
import numpy as np
import rawpy
from PIL import ExifTags, Image

from .config import IMAGE_EXTENSIONS, RAW_EXTENSIONS

logger = logging.getLogger(__name__)

# Canon CR3 metadata UUID inside moov box
_CR3_META_UUID = bytes.fromhex("85c0b687820f11e08111f4ce462b6a48")
# Adobe XMP UUID inside ISOBMFF (CR3)
_CR3_XMP_UUID = bytes.fromhex("be7acfcb97a942e89c71999491e3afac")

# Standard TIFF IFD tag IDs
_TAG_MODEL = 0x0110
_TAG_ORIENTATION = 0x0112
_TAG_DATETIME = 0x0132
_TAG_RATING = 0x4746
_TAG_EXPOSURE_TIME = 0x829A
_TAG_FNUMBER = 0x829D
_TAG_ISO = 0x8827
_TAG_DATETIME_ORIGINAL = 0x9003
_TAG_DATETIME_DIGITIZED = 0x9004
_TAG_FOCAL_LENGTH = 0x920A
_TAG_SUBSEC_ORIGINAL = 0x9291
_TAG_SUBSEC_DIGITIZED = 0x9292
_TAG_LENS_MAKE = 0xA433
_TAG_LENS_MODEL = 0xA434

_TIFF_TYPE_SIZES = {
    1: 1,   # BYTE
    2: 1,   # ASCII
    3: 2,   # SHORT
    4: 4,   # LONG
    5: 8,   # RATIONAL
    7: 1,   # UNDEFINED
    9: 4,   # SLONG
    10: 8,  # SRATIONAL
}


def discover_images(source_dir: Path, limit: int = 0) -> List[Tuple[Path, str]]:
    """
    Recursively discover supported image files in source_dir (strictly read-only).
    Returns sorted list of (file_path, subfolder_name).
    """
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    results: List[Tuple[Path, str]] = []
    for p in sorted(source_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        if p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            rel = p.relative_to(source_dir)
            subfolder = rel.parts[0] if len(rel.parts) > 1 else "root"
        except Exception:
            subfolder = p.parent.name
        results.append((p, subfolder))

    if limit and limit > 0:
        # Take balanced sample across subfolders if multiple exist
        subfolders: Dict[str, List[Tuple[Path, str]]] = {}
        for item in results:
            subfolders.setdefault(item[1], []).append(item)
        if len(subfolders) > 1:
            sampled: List[Tuple[Path, str]] = []
            per_sub = max(1, limit // len(subfolders))
            for sf in sorted(subfolders.keys()):
                sampled.extend(subfolders[sf][:per_sub])
            # Fill remainder if odd
            idx = per_sub
            sf_keys = sorted(subfolders.keys())
            while len(sampled) < limit:
                added = False
                for sf in sf_keys:
                    if idx < len(subfolders[sf]) and len(sampled) < limit:
                        sampled.append(subfolders[sf][idx])
                        added = True
                if not added:
                    break
                idx += 1
            return sampled[:limit]
        return results[:limit]

    return results


def _parse_tiff_ifd(blob: bytes) -> Dict[int, Any]:
    """Parse a standalone TIFF IFD blob (such as Canon CR3 CMT1/CMT2)."""
    if len(blob) < 8:
        return {}
    endian = blob[:2]
    if endian == b"II":
        bo = "<"
    elif endian == b"MM":
        bo = ">"
    else:
        return {}

    magic, ifd_offset = struct.unpack_from(f"{bo}HI", blob, 2)
    if magic != 42 or ifd_offset + 2 > len(blob):
        return {}

    num_entries = struct.unpack_from(f"{bo}H", blob, ifd_offset)[0]
    tags: Dict[int, Any] = {}
    pos = ifd_offset + 2

    for _ in range(num_entries):
        if pos + 12 > len(blob):
            break
        tag_id, dtype, count, val_offset = struct.unpack_from(f"{bo}HHII", blob, pos)
        val_raw_bytes = blob[pos + 8 : pos + 12]
        pos += 12

        unit_size = _TIFF_TYPE_SIZES.get(dtype)
        if not unit_size:
            continue
        total_bytes = unit_size * count
        if total_bytes <= 4:
            data = val_raw_bytes[:total_bytes]
        else:
            if val_offset + total_bytes > len(blob):
                continue
            data = blob[val_offset : val_offset + total_bytes]

        try:
            if dtype == 2:  # ASCII
                tags[tag_id] = data.split(b"\x00", 1)[0].decode("ascii", errors="ignore").strip()
            elif dtype == 3:  # SHORT
                vals = struct.unpack(f"{bo}{count}H", data)
                tags[tag_id] = vals[0] if count == 1 else vals
            elif dtype == 4:  # LONG
                vals = struct.unpack(f"{bo}{count}I", data)
                tags[tag_id] = vals[0] if count == 1 else vals
            elif dtype == 5:  # RATIONAL
                vals = []
                for i in range(count):
                    num, den = struct.unpack_from(f"{bo}II", data, i * 8)
                    vals.append((num, den))
                tags[tag_id] = vals[0] if count == 1 else vals
            elif dtype == 10:  # SRATIONAL
                vals = []
                for i in range(count):
                    num, den = struct.unpack_from(f"{bo}ii", data, i * 8)
                    vals.append((num, den))
                tags[tag_id] = vals[0] if count == 1 else vals
        except Exception:
            continue

    return tags


def _iter_isobmff_boxes(data: bytes, start: int = 0, end: Optional[int] = None):
    """Yield (box_type_str, payload_start, payload_end) from an ISOBMFF byte slice."""
    if end is None:
        end = len(data)
    pos = start
    while pos + 8 <= end:
        size, btype_bytes = struct.unpack_from(">I4s", data, pos)
        hdr = 8
        if size == 1:
            if pos + 16 > end:
                break
            size = struct.unpack_from(">Q", data, pos + 8)[0]
            hdr = 16
        elif size == 0:
            size = end - pos
        if size < hdr or pos + size > end:
            break
        btype = btype_bytes.decode("latin1", errors="ignore")
        yield btype, pos + hdr, pos + size
        pos += size


def _parse_datetime_with_subsec(dt_str: Optional[str], subsec: Optional[str]) -> Tuple[Optional[float], str]:
    if not dt_str:
        return None, ""
    clean = dt_str.strip()
    try:
        base = dt.datetime.strptime(clean, "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None, clean
    frac = 0.0
    sub_clean = ""
    if subsec:
        digits = "".join(ch for ch in str(subsec).strip() if ch.isdigit())
        if digits:
            sub_clean = digits
            frac = float(f"0.{digits}")
    ts = base.timestamp() + frac
    iso = base.strftime("%Y-%m-%d %H:%M:%S") + (f".{sub_clean}" if sub_clean else "")
    return ts, iso


def _format_exposure(rat: Any) -> str:
    if isinstance(rat, tuple) and len(rat) == 2 and rat[1]:
        num, den = rat
        if num == 0:
            return ""
        val = num / den
        if val >= 1.0:
            return f"{val:.1f}s"
        denom = round(1.0 / val)
        return f"1/{denom}s"
    if isinstance(rat, (int, float)) and rat > 0:
        if rat >= 1.0:
            return f"{rat:.1f}s"
        return f"1/{round(1.0 / rat)}s"
    return ""


def _format_rational_float(rat: Any) -> Optional[float]:
    if isinstance(rat, tuple) and len(rat) == 2 and rat[1]:
        return round(rat[0] / rat[1], 2)
    if isinstance(rat, (int, float)):
        return round(float(rat), 2)
    return None


def parse_exposure_seconds(exp_str: str) -> Optional[float]:
    """Parse formatted exposure string like '1/2000s' or '0.5s' into float seconds."""
    if not exp_str:
        return None
    s = str(exp_str).strip().lower().rstrip("s")
    try:
        if "/" in s:
            num, den = s.split("/", 1)
            d = float(den)
            return float(num) / d if d > 0 else None
        val = float(s)
        return val if val > 0 else None
    except Exception:
        return None


_RE_EMB_RATING = re.compile(rb'xmp:Rating\s*=\s*"(-?\d+)"|<xmp:Rating>\s*(-?\d+)\s*</xmp:Rating>', re.IGNORECASE)
_RE_EMB_LABEL = re.compile(rb'xmp:Label\s*=\s*"([^"]*)"|<xmp:Label>\s*([^<]*)\s*</xmp:Label>', re.IGNORECASE)


def _extract_xmp_rating_label_bytes(blob: bytes) -> Tuple[Optional[int], Optional[str]]:
    rating: Optional[int] = None
    label: Optional[str] = None
    m = _RE_EMB_RATING.search(blob)
    if m:
        raw = m.group(1) or m.group(2)
        try:
            r = int(raw.decode("ascii", errors="ignore"))
            rating = max(0, min(5, r)) if r >= 0 else 0
        except Exception:
            pass
    m2 = _RE_EMB_LABEL.search(blob)
    if m2:
        raw = m2.group(1) or m2.group(2)
        try:
            label = raw.decode("utf-8", errors="ignore").strip()
        except Exception:
            pass
    return rating, label


def _orient_norm_box(x: float, y: float, w: float, h: float, orientation: int) -> Tuple[float, float, float, float]:
    """Transform a normalized [0..1] sensor AF box into displayed image orientation."""
    if orientation == 6:  # 90 CW
        return (1.0 - (y + h), x, h, w)
    if orientation == 8:  # 90 CCW
        return (y, 1.0 - (x + w), h, w)
    if orientation == 3:  # 180
        return (1.0 - (x + w), 1.0 - (y + h), w, h)
    return (x, y, w, h)


def _parse_canon_af_info2(vals: Any, orientation: int = 1) -> List[Dict[str, Any]]:
    """
    Parse Canon MakerNote tag 0x0026 (AFInfo2 / AFInfo3 on EOS R5 / R5m2 / DSLR)
    into normalized [0..1] display-oriented AF boxes.
    Returns active/in-focus and selected AF boxes (typically 1..5 boxes).
    """
    if not isinstance(vals, (tuple, list)) or len(vals) < 12:
        return []
    num_pts = int(vals[2])
    valid_pts = int(vals[3])
    af_w = int(vals[6] or vals[4] or 0)
    af_h = int(vals[7] or vals[5] or 0)
    if num_pts <= 0 or af_w <= 0 or af_h <= 0 or len(vals) < 8 + 4 * num_pts:
        return []

    widths = vals[8 : 8 + num_pts]
    heights = vals[8 + num_pts : 8 + 2 * num_pts]
    xs_raw = vals[8 + 2 * num_pts : 8 + 3 * num_pts]
    ys_raw = vals[8 + 3 * num_pts : 8 + 4 * num_pts]

    m = (num_pts + 15) // 16
    bit_start = 8 + 4 * num_pts
    in_focus_mask = vals[bit_start : bit_start + m] if len(vals) >= bit_start + m else ()
    selected_mask = vals[bit_start + m : bit_start + 2 * m] if len(vals) >= bit_start + 2 * m else ()

    def _mask_set(mask: Any) -> set:
        out = set()
        for wi, word in enumerate(mask):
            if word:
                for b in range(16):
                    if word & (1 << b):
                        idx = wi * 16 + b
                        if idx < num_pts:
                            out.add(idx)
        return out

    in_focus = _mask_set(in_focus_mask)
    selected = _mask_set(selected_mask)

    # Keep only active/selected or valid tracked boxes (avoid dumping 1,053 empty grid zones)
    limit = min(valid_pts if valid_pts > 0 else num_pts, num_pts)
    candidate_indices = sorted(in_focus | selected) if (in_focus or selected) else list(range(min(limit, 8)))
    boxes: List[Dict[str, Any]] = []
    for i in candidate_indices:
        if i >= num_pts:
            continue
        bw = int(widths[i])
        bh = int(heights[i])
        if bw <= 0 or bh <= 0:
            continue
        sx = int(xs_raw[i])
        if sx >= 32768:
            sx -= 65536
        sy = int(ys_raw[i])
        if sy >= 32768:
            sy -= 65536
        cx_s = (af_w / 2.0 + sx) / af_w
        cy_s = (af_h / 2.0 - sy) / af_h  # Canon +Y is upward
        nw_s = bw / af_w
        nh_s = bh / af_h
        ox, oy, ow, oh = _orient_norm_box(cx_s - nw_s / 2.0, cy_s - nh_s / 2.0, nw_s, nh_s, orientation)
        boxes.append(
            {
                "x": round(max(0.0, min(1.0, ox)), 4),
                "y": round(max(0.0, min(1.0, oy)), 4),
                "w": round(max(0.0, min(1.0, ow)), 4),
                "h": round(max(0.0, min(1.0, oh)), 4),
                "cx": round(max(0.0, min(1.0, ox + ow / 2.0)), 4),
                "cy": round(max(0.0, min(1.0, oy + oh / 2.0)), 4),
                "in_focus": i in in_focus,
                "selected": i in selected,
            }
        )
    # Sort in-focus boxes first
    boxes.sort(key=lambda b: (not b["in_focus"], not b["selected"]))
    return boxes


def parse_cr3_exif_from_stream(fh: BinaryIO) -> Dict[str, Any]:
    """
    Fast header-only EXIF, Canon AF points & embedded XMP extraction from a Canon .CR3 stream.
    Reads only moov + XMP uuid boxes (typically <600 KB) without loading the full RAW payload.
    """
    meta: Dict[str, Any] = {
        "timestamp": None,
        "timestamp_str": "",
        "orientation": 1,
        "focal_length": None,
        "f_number": None,
        "exposure_time": "",
        "exposure_seconds": None,
        "iso": None,
        "camera_model": "",
        "lens_model": "",
        "camera_rating": 0,
        "camera_label": "",
        "af_points": [],
    }
    try:
        fh.seek(0)
        got_moov = False
        got_xmp = False
        # Walk top-level boxes to find 'moov' and embedded XMP 'uuid' box
        while True:
            hdr = fh.read(8)
            if len(hdr) < 8:
                break
            size, btype = struct.unpack(">I4s", hdr)
            header_len = 8
            if size == 1:
                ext = fh.read(8)
                if len(ext) < 8:
                    break
                size = struct.unpack(">Q", ext)[0]
                header_len = 16
            if size < header_len:
                break
            payload_size = size - header_len
            if btype == b"moov":
                moov_bytes = fh.read(min(payload_size, 4 * 1024 * 1024))
                cmt1_tags: Dict[int, Any] = {}
                cmt2_tags: Dict[int, Any] = {}
                cmt3_tags: Dict[int, Any] = {}
                for sub_type, s_start, s_end in _iter_isobmff_boxes(moov_bytes):
                    if sub_type == "uuid" and (s_end - s_start) >= 16:
                        uid = moov_bytes[s_start : s_start + 16]
                        if uid == _CR3_META_UUID:
                            for m_type, m_start, m_end in _iter_isobmff_boxes(
                                moov_bytes, s_start + 16, s_end
                            ):
                                if m_type == "CMT1":
                                    cmt1_tags = _parse_tiff_ifd(moov_bytes[m_start:m_end])
                                elif m_type == "CMT2":
                                    cmt2_tags = _parse_tiff_ifd(moov_bytes[m_start:m_end])
                                elif m_type == "CMT3":
                                    cmt3_tags = _parse_tiff_ifd(moov_bytes[m_start:m_end])
                        elif uid == _CR3_XMP_UUID:
                            r, lbl = _extract_xmp_rating_label_bytes(moov_bytes[s_start + 16 : s_end])
                            if r is not None:
                                meta["camera_rating"] = r
                            if lbl:
                                meta["camera_label"] = lbl
                            got_xmp = True
                if cmt1_tags or cmt2_tags:
                    meta["orientation"] = int(cmt1_tags.get(_TAG_ORIENTATION, 1) or 1)
                    meta["camera_model"] = str(cmt1_tags.get(_TAG_MODEL, "") or "")
                    meta["lens_model"] = str(cmt2_tags.get(_TAG_LENS_MODEL, "") or "").strip()
                    if not meta["camera_rating"] and _TAG_RATING in cmt1_tags:
                        try:
                            meta["camera_rating"] = max(0, min(5, int(cmt1_tags[_TAG_RATING])))
                        except Exception:
                            pass
                    dt_str = (
                        cmt2_tags.get(_TAG_DATETIME_ORIGINAL)
                        or cmt2_tags.get(_TAG_DATETIME_DIGITIZED)
                        or cmt1_tags.get(_TAG_DATETIME)
                    )
                    subsec = (
                        cmt2_tags.get(_TAG_SUBSEC_ORIGINAL)
                        or cmt2_tags.get(_TAG_SUBSEC_DIGITIZED)
                    )
                    ts, iso_str = _parse_datetime_with_subsec(dt_str, subsec)
                    meta["timestamp"] = ts
                    meta["timestamp_str"] = iso_str
                    exp_raw = cmt2_tags.get(_TAG_EXPOSURE_TIME)
                    meta["exposure_time"] = _format_exposure(exp_raw)
                    meta["exposure_seconds"] = parse_exposure_seconds(meta["exposure_time"])
                    meta["f_number"] = _format_rational_float(cmt2_tags.get(_TAG_FNUMBER))
                    meta["focal_length"] = _format_rational_float(cmt2_tags.get(_TAG_FOCAL_LENGTH))
                    iso_val = cmt2_tags.get(_TAG_ISO)
                    if isinstance(iso_val, tuple):
                        iso_val = iso_val[0]
                    meta["iso"] = int(iso_val) if iso_val is not None else None
                if cmt3_tags and 0x0026 in cmt3_tags:
                    meta["af_points"] = _parse_canon_af_info2(cmt3_tags[0x0026], meta["orientation"])
                got_moov = True
                if got_moov and got_xmp:
                    break
            elif btype == b"uuid" and payload_size >= 16 and payload_size <= 2 * 1024 * 1024:
                uid = fh.read(16)
                rem = payload_size - 16
                if uid == _CR3_XMP_UUID:
                    xmp_blob = fh.read(rem)
                    r, lbl = _extract_xmp_rating_label_bytes(xmp_blob)
                    if r is not None:
                        meta["camera_rating"] = r
                    if lbl:
                        meta["camera_label"] = lbl
                    got_xmp = True
                    if got_moov and got_xmp:
                        break
                else:
                    fh.seek(rem, io.SEEK_CUR)
            elif btype == b"mdat":
                # mdat is the huge RAW payload; skip past it (XMP usually precedes mdat)
                if got_moov:
                    break
                fh.seek(payload_size, io.SEEK_CUR)
            else:
                fh.seek(payload_size, io.SEEK_CUR)
    except Exception as exc:
        logger.debug("CR3 EXIF parse fallback: %s", exc)
    return meta


def parse_jpeg_exif_from_bytes(jpeg_bytes: bytes) -> Dict[str, Any]:
    """Extract EXIF + embedded XMP metadata from JPEG bytes via PIL."""
    meta: Dict[str, Any] = {
        "timestamp": None,
        "timestamp_str": "",
        "orientation": 1,
        "focal_length": None,
        "f_number": None,
        "exposure_time": "",
        "exposure_seconds": None,
        "iso": None,
        "camera_model": "",
        "lens_model": "",
        "camera_rating": 0,
        "camera_label": "",
    }
    try:
        # Check embedded XMP packet in first 256 KB of JPEG header
        head = jpeg_bytes[:262144]
        xmp_idx = head.find(b"<x:xmpmeta")
        if xmp_idx != -1:
            xmp_end = head.find(b"</x:xmpmeta>", xmp_idx)
            xmp_slice = head[xmp_idx : (xmp_end + 12 if xmp_end != -1 else xmp_idx + 16384)]
            r, lbl = _extract_xmp_rating_label_bytes(xmp_slice)
            if r is not None:
                meta["camera_rating"] = r
            if lbl:
                meta["camera_label"] = lbl

        with Image.open(io.BytesIO(jpeg_bytes)) as img:
            exif = img.getexif()
            if not exif:
                return meta
            meta["orientation"] = int(exif.get(_TAG_ORIENTATION, 1) or 1)
            meta["camera_model"] = str(exif.get(_TAG_MODEL, "") or "")
            if not meta["camera_rating"] and _TAG_RATING in exif:
                try:
                    meta["camera_rating"] = max(0, min(5, int(exif[_TAG_RATING])))
                except Exception:
                    pass
            exif_ifd = exif.get_ifd(0x8769) or {}
            meta["lens_model"] = str(exif_ifd.get(_TAG_LENS_MODEL, "") or "").strip()
            dt_str = (
                exif_ifd.get(_TAG_DATETIME_ORIGINAL)
                or exif_ifd.get(_TAG_DATETIME_DIGITIZED)
                or exif.get(_TAG_DATETIME)
            )
            subsec = (
                exif_ifd.get(_TAG_SUBSEC_ORIGINAL)
                or exif_ifd.get(_TAG_SUBSEC_DIGITIZED)
            )
            ts, iso_str = _parse_datetime_with_subsec(dt_str, subsec)
            meta["timestamp"] = ts
            meta["timestamp_str"] = iso_str
            exp = exif_ifd.get(_TAG_EXPOSURE_TIME)
            if exp is not None:
                try:
                    exp_f = float(exp)
                    meta["exposure_time"] = _format_exposure(exp_f)
                    meta["exposure_seconds"] = exp_f
                except Exception:
                    pass
            fn = exif_ifd.get(_TAG_FNUMBER)
            if fn is not None:
                try:
                    meta["f_number"] = round(float(fn), 2)
                except Exception:
                    pass
            fl = exif_ifd.get(_TAG_FOCAL_LENGTH)
            if fl is not None:
                try:
                    meta["focal_length"] = round(float(fl), 1)
                except Exception:
                    pass
            iso_val = exif_ifd.get(_TAG_ISO)
            if iso_val is not None:
                try:
                    meta["iso"] = int(iso_val[0] if isinstance(iso_val, (tuple, list)) else iso_val)
                except Exception:
                    pass
    except Exception as exc:
        logger.debug("JPEG EXIF parse warning: %s", exc)
    return meta


def parse_raw_exif_from_stream(fh: BinaryIO) -> Dict[str, Any]:
    """Header-only EXIF & embedded XMP reader for general rawpy-supported RAW formats (NEF, ARW, CR2, DNG, RAF, ORF, RW2, etc.)."""
    meta: Dict[str, Any] = {
        "timestamp": None,
        "timestamp_str": "",
        "orientation": 1,
        "focal_length": None,
        "f_number": None,
        "exposure_time": "",
        "exposure_seconds": None,
        "iso": None,
        "camera_model": "",
        "lens_model": "",
        "camera_rating": 0,
        "camera_label": "",
    }
    try:
        fh.seek(0)
        head = fh.read(524288)
        xmp_idx = head.find(b"<x:xmpmeta")
        if xmp_idx != -1:
            xmp_end = head.find(b"</x:xmpmeta>", xmp_idx)
            xmp_slice = head[xmp_idx : (xmp_end + 12 if xmp_end != -1 else xmp_idx + 16384)]
            r, lbl = _extract_xmp_rating_label_bytes(xmp_slice)
            if r is not None:
                meta["camera_rating"] = r
            if lbl:
                meta["camera_label"] = lbl
    except Exception:
        pass

    try:
        import exifread

        fh.seek(0)
        tags = exifread.process_file(fh, details=False, strict=False)
        if not tags:
            return meta

        orient_tag = tags.get("Image Orientation")
        if orient_tag is not None and getattr(orient_tag, "values", None):
            try:
                meta["orientation"] = int(orient_tag.values[0])
            except Exception:
                pass

        model_tag = tags.get("Image Model")
        if model_tag is not None:
            meta["camera_model"] = str(model_tag).strip()

        lens_tag = tags.get("EXIF LensModel") or tags.get("MakerNote LensModel")
        if lens_tag is not None:
            meta["lens_model"] = str(lens_tag).strip()

        dt_tag = (
            tags.get("EXIF DateTimeOriginal")
            or tags.get("EXIF DateTimeDigitized")
            or tags.get("Image DateTime")
        )
        sub_tag = tags.get("EXIF SubSecTimeOriginal") or tags.get("EXIF SubSecTimeDigitized")
        if dt_tag is not None:
            ts, iso_str = _parse_datetime_with_subsec(str(dt_tag), str(sub_tag) if sub_tag else None)
            meta["timestamp"] = ts
            meta["timestamp_str"] = iso_str

        def _tag_float(tag_obj: Any) -> Optional[float]:
            if tag_obj is None:
                return None
            vals = getattr(tag_obj, "values", None)
            if vals:
                v = vals[0]
                if hasattr(v, "num") and hasattr(v, "den") and v.den:
                    return float(v.num) / float(v.den)
                try:
                    return float(v)
                except Exception:
                    pass
            try:
                return float(str(tag_obj))
            except Exception:
                return None

        exp_f = _tag_float(tags.get("EXIF ExposureTime"))
        if exp_f is not None and exp_f > 0:
            meta["exposure_time"] = _format_exposure(exp_f)
            meta["exposure_seconds"] = exp_f

        fn = _tag_float(tags.get("EXIF FNumber"))
        if fn is not None and fn > 0:
            meta["f_number"] = round(fn, 2)

        fl = _tag_float(tags.get("EXIF FocalLength"))
        if fl is not None and fl > 0:
            meta["focal_length"] = round(fl, 1)

        iso_tag = tags.get("EXIF ISOSpeedRatings") or tags.get("EXIF PhotographicSensitivity")
        if iso_tag is not None and getattr(iso_tag, "values", None):
            try:
                val = iso_tag.values[0]
                meta["iso"] = int(val)
            except Exception:
                pass

        if not meta["camera_rating"]:
            rating_tag = tags.get("Image Rating")
            if rating_tag is not None and getattr(rating_tag, "values", None):
                try:
                    meta["camera_rating"] = max(0, min(5, int(rating_tag.values[0])))
                except Exception:
                    pass
    except Exception as exc:
        logger.debug("General RAW EXIF parse warning: %s", exc)
    return meta


def read_fast_exif_and_camera_meta(path: Path) -> Dict[str, Any]:
    """Header-only fast metadata reader (no pixel decode) for directory scanning & XMP/EXIF refresh."""
    suffix = path.suffix.lower()
    meta: Dict[str, Any] = {
        "timestamp": None,
        "timestamp_str": "",
        "orientation": 1,
        "focal_length": None,
        "f_number": None,
        "exposure_time": "",
        "exposure_seconds": None,
        "iso": None,
        "camera_model": "",
        "lens_model": "",
        "camera_rating": 0,
        "camera_label": "",
        "file_size": 0,
    }
    try:
        st = path.stat()
        meta["file_size"] = int(st.st_size)
        if suffix == ".cr3":
            with open(path, "rb") as fh:
                meta.update(parse_cr3_exif_from_stream(fh))
        elif suffix in RAW_EXTENSIONS:
            with open(path, "rb") as fh:
                meta.update(parse_raw_exif_from_stream(fh))
        else:
            with open(path, "rb") as fh:
                head_bytes = fh.read(262144)
            meta.update(parse_jpeg_exif_from_bytes(head_bytes))
        if meta["timestamp"] is None:
            meta["timestamp"] = float(st.st_mtime)
            meta["timestamp_str"] = dt.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    except Exception as exc:
        logger.debug("Fast EXIF scan failed for %s: %s", path.name, exc)
    return meta


def apply_orientation_bgr(img_bgr: np.ndarray, orientation: int) -> np.ndarray:
    """Apply EXIF orientation (1..8) to a BGR numpy array."""
    if orientation == 1 or orientation is None:
        return img_bgr
    if orientation == 2:
        return cv2.flip(img_bgr, 1)
    if orientation == 3:
        return cv2.rotate(img_bgr, cv2.ROTATE_180)
    if orientation == 4:
        return cv2.flip(img_bgr, 0)
    if orientation == 5:
        return cv2.flip(cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE), 1)
    if orientation == 6:
        return cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
    if orientation == 7:
        return cv2.flip(cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE), 1)
    if orientation == 8:
        return cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img_bgr


def _decode_jpeg_fast_bgr(jpeg_bytes: bytes) -> Optional[np.ndarray]:
    """
    Decode JPEG bytes via cv2.imdecode. For ultra-high-res 45MP embedded previews (>4800px),
    uses hardware-fast DCT 1/2 scaling (IMREAD_REDUCED_COLOR_2 -> ~4096x2732, 11.2MP)
    which is ~4x faster, uses 4x less RAM, reduces high-ISO single-pixel noise, and still
    preserves full 11MP resolution for sharpness scoring, crops, and 2048px exports.
    """
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    use_half = False
    try:
        with Image.open(io.BytesIO(jpeg_bytes)) as im:
            w, h = im.size
            if max(w, h) >= 4800:
                use_half = True
    except Exception:
        pass
    flag = cv2.IMREAD_REDUCED_COLOR_2 if use_half else cv2.IMREAD_COLOR
    return cv2.imdecode(arr, flag)


def load_preview_bgr_and_meta(path: Path) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Strictly read-only loader for rawpy-supported RAWs (.CR3, .NEF, .ARW, .CR2, .DNG, .RAF, .ORF, .RW2, etc.)
    and .JPG/.JPEG using Python binary file handles (safe for Unicode paths).
    Returns (image_bgr, metadata_dict).
    """
    suffix = path.suffix.lower()
    meta: Dict[str, Any] = {
        "timestamp": None,
        "timestamp_str": "",
        "orientation": 1,
        "focal_length": None,
        "f_number": None,
        "exposure_time": "",
        "iso": None,
        "camera_model": "",
        "preview_width": 0,
        "preview_height": 0,
    }

    try:
        if suffix in RAW_EXTENSIONS:
            thumb = None
            img_bgr: Optional[np.ndarray] = None
            with open(path, "rb") as fh:
                if suffix == ".cr3":
                    meta.update(parse_cr3_exif_from_stream(fh))
                else:
                    meta.update(parse_raw_exif_from_stream(fh))
                fh.seek(0)
                with rawpy.imread(fh) as raw:
                    try:
                        thumb = raw.extract_thumb()
                    except Exception:
                        thumb = None
                    if thumb is None:
                        rgb = raw.postprocess(half_size=True, use_camera_wb=True, output_bps=8)
                        img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            if img_bgr is None and thumb is not None:
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    thumb_bytes = thumb.data
                    if meta["timestamp"] is None:
                        thumb_meta = parse_jpeg_exif_from_bytes(thumb_bytes)
                        if thumb_meta["timestamp"] is not None:
                            meta["timestamp"] = thumb_meta["timestamp"]
                            meta["timestamp_str"] = thumb_meta["timestamp_str"]
                        if meta.get("orientation", 1) == 1 and thumb_meta.get("orientation", 1) != 1:
                            meta["orientation"] = thumb_meta["orientation"]

                    img_bgr = _decode_jpeg_fast_bgr(thumb_bytes)
                    if img_bgr is None:
                        return None, meta
                    orient = meta.get("orientation", 1)
                    h, w = img_bgr.shape[:2]
                    if orient in (6, 8) and w > h:
                        img_bgr = apply_orientation_bgr(img_bgr, orient)
                    elif orient == 3:
                        img_bgr = apply_orientation_bgr(img_bgr, orient)
                elif thumb.format == rawpy.ThumbFormat.BITMAP:
                    img_bgr = cv2.cvtColor(thumb.data, cv2.COLOR_RGB2BGR)
                    orient = meta.get("orientation", 1)
                    h, w = img_bgr.shape[:2]
                    if orient in (6, 8) and w > h:
                        img_bgr = apply_orientation_bgr(img_bgr, orient)
                    elif orient == 3:
                        img_bgr = apply_orientation_bgr(img_bgr, orient)
                else:
                    with open(path, "rb") as fh:
                        with rawpy.imread(fh) as raw:
                            rgb = raw.postprocess(half_size=True, use_camera_wb=True, output_bps=8)
                            img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            if img_bgr is None:
                return None, meta

        else:
            with open(path, "rb") as fh:
                raw_bytes = fh.read()
            meta.update(parse_jpeg_exif_from_bytes(raw_bytes))
            img_bgr = _decode_jpeg_fast_bgr(raw_bytes)
            if img_bgr is None:
                return None, meta
            img_bgr = apply_orientation_bgr(img_bgr, meta.get("orientation", 1))

        # Fallback timestamp from file mtime if EXIF lacked DateTimeOriginal
        if meta["timestamp"] is None:
            mtime = path.stat().st_mtime
            meta["timestamp"] = float(mtime)
            meta["timestamp_str"] = dt.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")

        h, w = img_bgr.shape[:2]
        meta["preview_width"] = int(w)
        meta["preview_height"] = int(h)
        return img_bgr, meta

    except Exception as exc:
        logger.warning("Failed to read %s: %s", path.name, exc)
        return None, meta


def compute_framing_histogram(img_bgr: np.ndarray) -> np.ndarray:
    """Compute a compact normalized HSV histogram (float32, 64 bins) for burst framing similarity."""
    small = cv2.resize(img_bgr, (96, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [24], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [20], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [20], [0, 256]).flatten()
    hist = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)
    norm = np.linalg.norm(hist)
    if norm > 1e-6:
        hist /= norm
    return hist


def save_jpeg_bgr(
    img_bgr: np.ndarray,
    out_path: Path,
    quality: int = 88,
    max_dim: Optional[int] = None,
) -> None:
    """Unicode-safe JPEG writer using cv2.imencode + Python binary file handle."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    to_save = img_bgr
    if max_dim is not None:
        h, w = img_bgr.shape[:2]
        longest = max(h, w)
        if longest > max_dim:
            scale = max_dim / float(longest)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            to_save = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(".jpg", to_save, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError(f"Failed to encode JPEG for {out_path}")
    with open(out_path, "wb") as fh:
        fh.write(buf.tobytes())
