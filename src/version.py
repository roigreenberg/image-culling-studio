"""Application version, changelog reader, and GitHub Releases configuration."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Tuple

APP_NAME = "ImageCulling Studio"
GITHUB_REPO = os.environ.get("IMAGE_CULLING_GITHUB_REPO", "roigreenberg/image-culling-studio")

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = WORKSPACE_ROOT / "VERSION"
CHANGELOG_FILE = WORKSPACE_ROOT / "CHANGELOG.md"


def get_local_version() -> str:
    """Read semantic version string from the root VERSION file."""
    try:
        v = VERSION_FILE.read_text(encoding="utf-8").strip()
        if v:
            return v
    except OSError:
        pass
    return "0.0.0"


def parse_version(version_str: str) -> Tuple[int, ...]:
    """Parse a semantic version like 'v1.2.3' or '1.2.3-beta' into a comparable int tuple."""
    clean = str(version_str or "").strip().lstrip("vV")
    nums = [int(m) for m in re.findall(r"\d+", clean.split("-", 1)[0])]
    return tuple(nums) if nums else (0,)


def is_newer_version(latest: str, current: str) -> bool:
    """Return True if `latest` is strictly newer than `current`."""
    a = parse_version(latest)
    b = parse_version(current)
    max_len = max(len(a), len(b))
    a_pad = a + (0,) * (max_len - len(a))
    b_pad = b + (0,) * (max_len - len(b))
    return a_pad > b_pad


def read_local_changelog(max_chars: int = 12000) -> str:
    """Return the local CHANGELOG.md contents."""
    try:
        text = CHANGELOG_FILE.read_text(encoding="utf-8")
        return text[:max_chars]
    except OSError:
        return ""
