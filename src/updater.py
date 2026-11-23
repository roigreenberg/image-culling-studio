"""Safe self-updater backed by GitHub Releases (never touches user data or weights)."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from .version import (
    APP_NAME,
    GITHUB_REPO,
    WORKSPACE_ROOT,
    get_local_version,
    is_newer_version,
    read_local_changelog,
)

logger = logging.getLogger(__name__)

# Paths that the updater MUST NEVER overwrite or delete
PROTECTED_PREFIXES = (
    ".venv/",
    "venv/",
    ".git/",
    "cull_output/",
    "studio_data/",
    "models/",
    "dist/",
    ".update_backup/",
    "vendor/ProjectKestrel/analyzer/models/",
)

PROTECTED_EXACT = {
    ".env",
    "my_picks.json",
    "embeddings.npz",
}


def _is_protected(rel_posix: str) -> bool:
    norm = rel_posix.lstrip("./")
    if norm in PROTECTED_EXACT or norm.startswith(".env.") and norm != ".env.example":
        return True
    for pref in PROTECTED_PREFIXES:
        if norm == pref.rstrip("/") or norm.startswith(pref):
            return True
    return False


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_version_info() -> Dict[str, Any]:
    """Return local version, repository slug, and local changelog."""
    return {
        "app_name": APP_NAME,
        "current_version": get_local_version(),
        "github_repo": GITHUB_REPO,
        "releases_url": f"https://github.com/{GITHUB_REPO}/releases",
        "changelog": read_local_changelog(),
    }


def check_for_updates(timeout: float = 6.0, repo: Optional[str] = None) -> Dict[str, Any]:
    """
    Check GitHub Releases API (`releases/latest`) for a newer version.
    Fails gracefully when offline or if no release has been published yet.
    """
    target_repo = (repo or GITHUB_REPO).strip()
    current = get_local_version()
    api_url = f"https://api.github.com/repos/{target_repo}/releases/latest"

    result: Dict[str, Any] = {
        "ok": False,
        "current_version": current,
        "latest_version": current,
        "update_available": False,
        "release_name": "",
        "published_at": "",
        "changelog": "",
        "html_url": f"https://github.com/{target_repo}/releases",
        "download_url": "",
        "asset_name": "",
        "github_repo": target_repo,
        "error": "",
    }

    req = urllib.request.Request(
        api_url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": f"ImageCulling-Updater/{current}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            result["ok"] = True
            result["error"] = "No published GitHub release found yet."
        else:
            result["error"] = f"GitHub API HTTP {exc.code}: {exc.reason}"
        return result
    except Exception as exc:
        result["error"] = f"Could not reach GitHub Releases: {exc}"
        return result

    tag = str(payload.get("tag_name") or "").strip()
    latest = tag.lstrip("vV") or current
    assets = payload.get("assets") or []

    # Prefer a release ZIP asset named ImageCulling-v*.zip (or any .zip asset except models pack)
    chosen_url = ""
    chosen_name = ""
    for a in assets:
        name = str(a.get("name") or "")
        dl = str(a.get("browser_download_url") or "")
        low = name.lower()
        if low.endswith(".zip") and "model" not in low:
            chosen_url = dl
            chosen_name = name
            break
    if not chosen_url and payload.get("zipball_url"):
        chosen_url = str(payload["zipball_url"])
        chosen_name = f"{target_repo.split('/')[-1]}-{tag}.zip"

    result.update(
        {
            "ok": True,
            "latest_version": latest,
            "update_available": is_newer_version(latest, current),
            "release_name": str(payload.get("name") or tag),
            "published_at": str(payload.get("published_at") or ""),
            "changelog": str(payload.get("body") or "").strip(),
            "html_url": str(payload.get("html_url") or result["html_url"]),
            "download_url": chosen_url,
            "asset_name": chosen_name,
        }
    )
    return result


def _strip_common_root(names: list[str]) -> str:
    """Detect if all ZIP entries live under a single top-level folder (like GitHub zipballs)."""
    top_dirs = set()
    for n in names:
        clean = n.replace("\\", "/").strip("/")
        if not clean:
            continue
        parts = clean.split("/")
        if len(parts) == 1 and not n.endswith("/"):
            return ""
        top_dirs.add(parts[0])
    if len(top_dirs) == 1:
        return next(iter(top_dirs)) + "/"
    return ""


def _install_updated_deps() -> Tuple[bool, str]:
    """Install/update cross-platform base dependencies into the active Python environment."""
    req_base = WORKSPACE_ROOT / "requirements-base.txt"
    req_file = req_base if req_base.is_file() else (WORKSPACE_ROOT / "requirements.txt")
    if not req_file.is_file():
        return True, "No requirements file found."
    try:
        cmd = [sys.executable, "-m", "pip", "install", "-r", str(req_file)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            return False, (proc.stderr or proc.stdout)[-800:]
        return True, f"Updated dependencies from {req_file.name}."
    except Exception as exc:
        return False, str(exc)


def download_and_apply_update(
    download_url: Optional[str] = None,
    repo: Optional[str] = None,
    progress_cb: Optional[Callable[[str, float], None]] = None,
) -> Dict[str, Any]:
    """
    Download a release ZIP and overlay code files onto WORKSPACE_ROOT while strictly
    preserving all user data, .env, .venv, culling outputs, and model weights.
    """
    prev_version = get_local_version()

    def _emit(msg: str, pct: float) -> None:
        logger.info("[updater] (%.0f%%) %s", pct, msg)
        if progress_cb:
            try:
                progress_cb(msg, pct)
            except Exception:
                pass

    if not download_url:
        _emit("Checking latest release metadata...", 5.0)
        info = check_for_updates(repo=repo)
        if not info.get("ok"):
            return {"ok": False, "error": info.get("error") or "Update check failed."}
        if not info.get("update_available"):
            return {
                "ok": True,
                "updated": False,
                "previous_version": prev_version,
                "new_version": prev_version,
                "message": f"Already up to date (v{prev_version}).",
            }
        download_url = info.get("download_url")
        if not download_url:
            return {"ok": False, "error": "Release has no downloadable ZIP archive."}

    req_base_before = _file_sha256(WORKSPACE_ROOT / "requirements-base.txt")
    req_full_before = _file_sha256(WORKSPACE_ROOT / "requirements.txt")

    with tempfile.TemporaryDirectory(prefix="imageculling_upd_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        zip_path = tmp_dir / "release.zip"

        _emit("Downloading release archive...", 15.0)
        req = urllib.request.Request(
            download_url,
            headers={"User-Agent": f"ImageCulling-Updater/{prev_version}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                total = int(resp.headers.get("Content-Length") or 0)
                read_bytes = 0
                with open(zip_path, "wb") as out_fh:
                    while True:
                        chunk = resp.read(256 * 1024)
                        if not chunk:
                            break
                        out_fh.write(chunk)
                        read_bytes += len(chunk)
                        if total > 0:
                            pct = 15.0 + 45.0 * (read_bytes / total)
                            _emit(f"Downloading ({read_bytes / 1e6:.1f} MB)...", pct)
        except Exception as exc:
            return {"ok": False, "error": f"Download failed: {exc}"}

        _emit("Validating archive...", 62.0)
        if not zipfile.is_zipfile(zip_path):
            return {"ok": False, "error": "Downloaded file is not a valid ZIP archive."}

        backup_stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = WORKSPACE_ROOT / ".update_backup" / f"v{prev_version}_{backup_stamp}"
        files_written = 0

        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            root_prefix = _strip_common_root(names)

            for member in zf.infolist():
                if member.is_dir():
                    continue
                raw_name = member.filename.replace("\\", "/")
                if root_prefix and raw_name.startswith(root_prefix):
                    rel = raw_name[len(root_prefix) :]
                else:
                    rel = raw_name
                rel = rel.lstrip("/")
                if not rel or ".." in Path(rel).parts:
                    continue
                if _is_protected(rel):
                    continue

                dest = (WORKSPACE_ROOT / rel).resolve()
                # Strict ZipSlip containment check
                if not str(dest).startswith(str(WORKSPACE_ROOT.resolve())):
                    continue

                if dest.is_file():
                    b_dest = backup_dir / rel
                    b_dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(dest, b_dest)

                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member, "r") as src_fh, open(dest, "wb") as dst_fh:
                    shutil.copyfileobj(src_fh, dst_fh)
                # Preserve executable bit on macOS/Linux scripts
                if sys.platform != "win32" and (rel.endswith(".sh") or rel.endswith(".command")):
                    try:
                        os.chmod(dest, 0o755)
                    except OSError:
                        pass
                files_written += 1

    _emit(f"Applied {files_written} updated files.", 82.0)

    req_changed = (
        _file_sha256(WORKSPACE_ROOT / "requirements-base.txt") != req_base_before
        or _file_sha256(WORKSPACE_ROOT / "requirements.txt") != req_full_before
    )
    deps_updated = False
    deps_msg = ""
    if req_changed:
        _emit("Requirements changed — updating Python packages...", 88.0)
        deps_updated, deps_msg = _install_updated_deps()

    new_version = get_local_version()
    _emit(f"Update complete: v{prev_version} -> v{new_version}. Restart the app to use the new version.", 100.0)
    return {
        "ok": True,
        "updated": True,
        "previous_version": prev_version,
        "new_version": new_version,
        "files_written": files_written,
        "deps_updated": deps_updated,
        "deps_message": deps_msg,
        "backup_dir": str(backup_dir.relative_to(WORKSPACE_ROOT)),
        "restart_required": True,
        "message": f"Updated from v{prev_version} to v{new_version}. Please restart ImageCulling Studio.",
    }
