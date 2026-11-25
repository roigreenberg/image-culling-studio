"""Lightroom-compatible XMP sidecar reader and real-time atomic writer."""

from __future__ import annotations

import datetime as dt
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Standard Lightroom Classic color labels (English default)
VALID_LABELS = ("", "Red", "Yellow", "Green", "Blue", "Purple")
LABEL_NORMALIZE = {
    "": "",
    "none": "",
    "red": "Red",
    "yellow": "Yellow",
    "green": "Green",
    "blue": "Blue",
    "purple": "Purple",
    # Common Lightroom localized or numeric shortcuts
    "6": "Red",
    "7": "Yellow",
    "8": "Green",
    "9": "Blue",
}

_RE_ATTR_RATING = re.compile(r'\bxmp:Rating\s*=\s*"(-?\d+)"', re.IGNORECASE)
_RE_ELEM_RATING = re.compile(r"<xmp:Rating>\s*(-?\d+)\s*</xmp:Rating>", re.IGNORECASE)
_RE_ATTR_LABEL = re.compile(r'\bxmp:Label\s*=\s*"([^"]*)"', re.IGNORECASE)
_RE_ELEM_LABEL = re.compile(r"<xmp:Label>\s*([^<]*)\s*</xmp:Label>", re.IGNORECASE)
_RE_ATTR_PM_COLOR = re.compile(r'\bphotomechanic:ColorClass\s*=\s*"(\d+)"', re.IGNORECASE)
_RE_ATTR_META_DATE = re.compile(r'\bxmp:MetadataDate\s*=\s*"([^"]*)"', re.IGNORECASE)
_RE_RDF_DESC_OPEN = re.compile(r"(<rdf:Description\b[^>]*?)(/?>)", re.DOTALL)

_CRS_NS = 'xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"'
_CRS_CROP_ATTRS = (
    "HasCrop",
    "CropTop",
    "CropLeft",
    "CropBottom",
    "CropRight",
    "CropAngle",
    "CropConstrainToWarp",
)
_RE_CRS_ATTRS = {
    name: re.compile(rf'\bcrs:{name}\s*=\s*"([^"]*)"', re.IGNORECASE)
    for name in _CRS_CROP_ATTRS
}

_PM_COLOR_TO_LABEL = {"0": "", "1": "Red", "2": "Yellow", "3": "Green", "4": "Blue", "5": "Purple"}
_LABEL_TO_PM_COLOR = {v: k for k, v in _PM_COLOR_TO_LABEL.items()}


def normalize_label(label: Optional[str]) -> str:
    if not label:
        return ""
    s = str(label).strip()
    return LABEL_NORMALIZE.get(s.lower(), s if s in VALID_LABELS else "")


def normalize_rating(rating: Any) -> Optional[int]:
    if rating is None or rating == "":
        return None
    try:
        val = int(rating)
        if val < 0:
            return 0
        if val > 5:
            return 5
        return val
    except (ValueError, TypeError):
        return None


def get_xmp_path(image_path: Path) -> Path:
    """Return the Lightroom sidecar .xmp path for an image (same stem, .xmp extension)."""
    return image_path.with_suffix(".xmp")


def parse_xmp_text(xml_text: str) -> Dict[str, Any]:
    """Parse rating and color label from an XMP XML string."""
    rating: Optional[int] = None
    label: Optional[str] = None

    m = _RE_ATTR_RATING.search(xml_text) or _RE_ELEM_RATING.search(xml_text)
    if m:
        try:
            r = int(m.group(1))
            rating = max(0, min(5, r)) if r >= 0 else 0
        except ValueError:
            pass

    m2 = _RE_ATTR_LABEL.search(xml_text) or _RE_ELEM_LABEL.search(xml_text)
    if m2:
        label = normalize_label(m2.group(1))
    else:
        m3 = _RE_ATTR_PM_COLOR.search(xml_text)
        if m3:
            label = _PM_COLOR_TO_LABEL.get(m3.group(1), "")

    return {"rating": rating, "label": label}


def parse_xmp_has_crop(xml_text: str) -> bool:
    """Return True if an XMP string declares crs:HasCrop="True"."""
    m_hc = _RE_CRS_ATTRS["HasCrop"].search(xml_text or "")
    return bool(m_hc and m_hc.group(1).strip().lower() == "true")


def read_xmp_sidecar(image_path: Path) -> Dict[str, Any]:
    """Read an image's .xmp sidecar if it exists."""
    xmp_path = get_xmp_path(image_path)
    if not xmp_path.exists():
        return {"exists": False, "path": str(xmp_path), "rating": None, "label": None, "raw_xml": None}
    try:
        raw_xml = xmp_path.read_text(encoding="utf-8", errors="replace")
        parsed = parse_xmp_text(raw_xml)
        return {
            "exists": True,
            "path": str(xmp_path),
            "rating": parsed["rating"],
            "label": parsed["label"],
            "raw_xml": raw_xml,
        }
    except Exception as exc:
        logger.warning("Failed to read XMP sidecar %s: %s", xmp_path, exc)
        return {"exists": False, "path": str(xmp_path), "rating": None, "label": None, "raw_xml": None}


def _set_crs_attr(xml_text: str, name: str, value: str) -> str:
    pat = _RE_CRS_ATTRS[name]
    if pat.search(xml_text):
        return pat.sub(f'crs:{name}="{value}"', xml_text, count=1)
    m = _RE_RDF_DESC_OPEN.search(xml_text)
    if m:
        return xml_text[: m.start(2)] + f'\n   crs:{name}="{value}"' + xml_text[m.start(2) :]
    return xml_text


def _apply_crop_to_xmp(xml_text: str, crop: Optional[Dict[str, Any]]) -> str:
    if crop is None:
        return xml_text
    updated = xml_text
    if "xmlns:crs=" not in updated:
        m = _RE_RDF_DESC_OPEN.search(updated)
        if m:
            updated = updated[: m.start(2)] + f"\n   {_CRS_NS}" + updated[m.start(2) :]

    applied = bool(crop.get("applied", False))
    if applied:
        left = max(0.0, min(1.0, float(crop.get("left", 0.0))))
        top = max(0.0, min(1.0, float(crop.get("top", 0.0))))
        right = max(0.0, min(1.0, float(crop.get("right", 1.0))))
        bottom = max(0.0, min(1.0, float(crop.get("bottom", 1.0))))
        angle = max(-45.0, min(45.0, float(crop.get("angle", 0.0))))
        updated = _set_crs_attr(updated, "HasCrop", "True")
        updated = _set_crs_attr(updated, "CropTop", f"{top:.6f}")
        updated = _set_crs_attr(updated, "CropLeft", f"{left:.6f}")
        updated = _set_crs_attr(updated, "CropBottom", f"{bottom:.6f}")
        updated = _set_crs_attr(updated, "CropRight", f"{right:.6f}")
        updated = _set_crs_attr(updated, "CropAngle", f"{angle:.4f}")
        updated = _set_crs_attr(updated, "CropConstrainToWarp", "0")
    else:
        updated = _set_crs_attr(updated, "HasCrop", "False")
        updated = _set_crs_attr(updated, "CropTop", "0")
        updated = _set_crs_attr(updated, "CropLeft", "0")
        updated = _set_crs_attr(updated, "CropBottom", "1")
        updated = _set_crs_attr(updated, "CropRight", "1")
        updated = _set_crs_attr(updated, "CropAngle", "0")
    return updated


def _create_minimal_xmp(
    rating: Optional[int],
    label: Optional[str],
    now_iso: str,
    crop: Optional[Dict[str, Any]] = None,
) -> str:
    attrs = [
        'rdf:about=""',
        'xmlns:xmp="http://ns.adobe.com/xap/1.0/"',
        f'xmp:MetadataDate="{now_iso}"',
    ]
    if rating is not None:
        attrs.append(f'xmp:Rating="{int(rating)}"')
    if label:
        attrs.append(f'xmp:Label="{label}"')
    attr_block = "\n   ".join(attrs)
    base = (
        '<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Antigravity ImageCulling Studio 1.0">\n'
        ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        f"  <rdf:Description\n   {attr_block}/>\n"
        " </rdf:RDF>\n"
        "</x:xmpmeta>\n"
    )
    return _apply_crop_to_xmp(base, crop)


def _update_existing_xmp(
    xml_text: str,
    rating: Optional[int],
    label: Optional[str],
    now_iso: str,
    crop: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Update ONLY xmp:Rating, xmp:Label, and xmp:MetadataDate in an existing XMP file,
    preserving all Lightroom Develop settings, crops, masks, keywords, etc.
    """
    updated = xml_text

    # Ensure xmlns:xmp namespace is declared on rdf:Description if missing
    if "xmlns:xmp=" not in updated:
        m = _RE_RDF_DESC_OPEN.search(updated)
        if m:
            updated = (
                updated[: m.start(2)]
                + '\n   xmlns:xmp="http://ns.adobe.com/xap/1.0/"'
                + updated[m.start(2) :]
            )

    # Update or insert xmp:Rating
    if rating is not None:
        r_str = str(int(rating))
        if _RE_ATTR_RATING.search(updated):
            updated = _RE_ATTR_RATING.sub(f'xmp:Rating="{r_str}"', updated, count=1)
        elif _RE_ELEM_RATING.search(updated):
            updated = _RE_ELEM_RATING.sub(f"<xmp:Rating>{r_str}</xmp:Rating>", updated, count=1)
        else:
            m = _RE_RDF_DESC_OPEN.search(updated)
            if m:
                updated = updated[: m.start(2)] + f'\n   xmp:Rating="{r_str}"' + updated[m.start(2) :]

    # Update, insert, or remove xmp:Label
    if label is not None:
        clean_label = normalize_label(label)
        if clean_label:
            if _RE_ATTR_LABEL.search(updated):
                updated = _RE_ATTR_LABEL.sub(f'xmp:Label="{clean_label}"', updated, count=1)
            elif _RE_ELEM_LABEL.search(updated):
                updated = _RE_ELEM_LABEL.sub(f"<xmp:Label>{clean_label}</xmp:Label>", updated, count=1)
            else:
                m = _RE_RDF_DESC_OPEN.search(updated)
                if m:
                    updated = (
                        updated[: m.start(2)]
                        + f'\n   xmp:Label="{clean_label}"'
                        + updated[m.start(2) :]
                    )
        else:
            # Clear label attribute/element if set to empty
            updated = re.sub(r'\s*\bxmp:Label\s*=\s*"[^"]*"', "", updated, flags=re.IGNORECASE)
            updated = re.sub(r"\s*<xmp:Label>[^<]*</xmp:Label>", "", updated, flags=re.IGNORECASE)
        # Keep Photo Mechanic ColorClass in sync if present in the XMP
        if _RE_ATTR_PM_COLOR.search(updated):
            pm_code = _LABEL_TO_PM_COLOR.get(clean_label, "0")
            updated = _RE_ATTR_PM_COLOR.sub(f'photomechanic:ColorClass="{pm_code}"', updated, count=1)

    # Update xmp:MetadataDate so Lightroom detects the file changed
    if _RE_ATTR_META_DATE.search(updated):
        updated = _RE_ATTR_META_DATE.sub(f'xmp:MetadataDate="{now_iso}"', updated, count=1)
    else:
        m = _RE_RDF_DESC_OPEN.search(updated)
        if m:
            updated = (
                updated[: m.start(2)]
                + f'\n   xmp:MetadataDate="{now_iso}"'
                + updated[m.start(2) :]
            )

    updated = _apply_crop_to_xmp(updated, crop)
    return updated


def write_xmp_sidecar(
    image_path: Path,
    rating: Optional[int] = None,
    label: Optional[str] = None,
    crop: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
    """
    Atomically write rating/label/crop to the .xmp sidecar alongside image_path.
    Returns (success, previous_xml_or_none, new_xml_or_none, error_message_or_none).
    """
    xmp_path = get_xmp_path(image_path)
    if not image_path.parent.exists():
        return False, None, None, f"Parent directory not accessible: {image_path.parent}"

    now_iso = dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")
    prev_xml: Optional[str] = None
    try:
        if xmp_path.exists():
            prev_xml = xmp_path.read_text(encoding="utf-8", errors="replace")
            new_xml = _update_existing_xmp(prev_xml, rating, label, now_iso, crop=crop)
        else:
            new_xml = _create_minimal_xmp(rating, normalize_label(label) if label else None, now_iso, crop=crop)

        tmp_path = xmp_path.with_suffix(".xmp.tmp")
        tmp_path.write_text(new_xml, encoding="utf-8")
        os.replace(tmp_path, xmp_path)
        return True, prev_xml, new_xml, None
    except Exception as exc:
        logger.warning("Failed to write XMP sidecar %s: %s", xmp_path, exc)
        return False, prev_xml, None, str(exc)


def restore_xmp_sidecar(image_path: Path, previous_xml: Optional[str]) -> Tuple[bool, Optional[str]]:
    """
    Restore an image's .xmp sidecar to its previous XML content.
    If previous_xml is None (meaning no .xmp existed before), removes the created .xmp file.
    """
    xmp_path = get_xmp_path(image_path)
    try:
        if previous_xml is None:
            if xmp_path.exists():
                xmp_path.unlink()
            return True, None
        tmp_path = xmp_path.with_suffix(".xmp.tmp")
        tmp_path.write_text(previous_xml, encoding="utf-8")
        os.replace(tmp_path, xmp_path)
        return True, None
    except Exception as exc:
        logger.warning("Failed to restore XMP sidecar %s: %s", xmp_path, exc)
        return False, str(exc)
