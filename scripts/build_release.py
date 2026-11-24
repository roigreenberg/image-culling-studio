"""Build a clean, zero-secrets release ZIP ready for GitHub Releases.

Usage:
    python scripts/build_release.py
    python scripts/build_release.py --out-dir ./dist
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import zipfile
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.version import get_local_version  # noqa: E402

SECRET_PATTERNS = [
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"gh[po]_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9\-_]{20,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]
USER_PATH_PATTERN = re.compile(r"(C:\\Users\\|/Users/|/home/)", re.IGNORECASE)
PATH_DOC_ALLOWLIST = {"SECURITY.md", "scripts/pre-commit", "scripts/build_release.py"}
EXEC_FILES = {"Install_MacOS.command", "Start_MacOS.command"}

# Strict allowlist: ONLY files an end user needs to install, run, update, and comply with AGPL-3.0.
# Excludes tests, docs/ (GitHub Pages), promo_video/, scripts/, git/node configs, and security/dev files.
ALLOWED_EXACT_FILES = {
    "CHANGELOG.md",
    "Install_MacOS.command",
    "Install_Windows.bat",
    "LICENSE",
    "README.md",
    "Start_MacOS.command",
    "Start_Windows.bat",
    "THIRD_PARTY_NOTICES.md",
    "VERSION",
    "requirements-base.txt",
    "requirements.txt",
    "run_cull.py",
    "run_webapp.py",
    "setup_models.py",
    "train_ranker.py",
    "update.py",
    "vendor/ProjectKestrel/LICENSE",
    "vendor/ProjectKestrel/VERSION.txt",
}
ALLOWED_DIR_PREFIXES = (
    "src/",
    "webapp/",
    "vendor/ProjectKestrel/analyzer/kestrel_analyzer/",
    "vendor/ProjectKestrel/analyzer/models/",
)


def _is_release_file(norm: str) -> bool:
    if any(part.startswith(".") for part in norm.split("/")):
        return False
    if norm in ALLOWED_EXACT_FILES:
        return True
    return any(norm.startswith(prefix) for prefix in ALLOWED_DIR_PREFIXES)


def _tracked_files() -> list[str]:
    out = subprocess.check_output(["git", "ls-files", "-z"], cwd=WORKSPACE_ROOT)
    return [p for p in out.decode("utf-8", errors="replace").split("\x00") if p]


def build_release_zip(out_dir: Path) -> Path:
    version = get_local_version()
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_name = f"ImageCulling-v{version}.zip"
    zip_path = out_dir / zip_name
    root_folder = f"ImageCulling-v{version}"

    files = _tracked_files()
    if not files:
        raise RuntimeError("No tracked git files found. Commit your files before building a release.")

    count = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for rel in files:
            norm = rel.replace("\\", "/")
            if not _is_release_file(norm):
                continue
            full = WORKSPACE_ROOT / norm
            if not full.is_file():
                continue
            raw = full.read_bytes()

            # Zero-secrets & zero-personal-path safety check
            if b"\x00" not in raw[:4096]:
                text = raw.decode("utf-8", errors="ignore")
                for pat in SECRET_PATTERNS:
                    if pat.search(text):
                        raise RuntimeError(f"Secret pattern detected in {norm}; aborting release build.")
                if norm not in PATH_DOC_ALLOWLIST and USER_PATH_PATTERN.search(text):
                    raise RuntimeError(f"Hardcoded user path detected in {norm}; aborting release build.")

            arcname = f"{root_folder}/{norm}"
            zi = zipfile.ZipInfo(arcname)
            zi.compress_type = zipfile.ZIP_DEFLATED
            # Preserve executable Unix permissions for macOS .command and hook scripts
            if norm in EXEC_FILES or norm.endswith(".sh") or norm.endswith(".command"):
                zi.external_attr = 0o100755 << 16
            else:
                zi.external_attr = 0o100644 << 16
            zf.writestr(zi, raw)
            count += 1

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"Built clean release archive: {zip_path} ({count} files, {size_mb:.2f} MB)")
    return zip_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build zero-secrets ImageCulling Studio release ZIP")
    parser.add_argument("--out-dir", type=str, default="./dist")
    args = parser.parse_args()
    build_release_zip(Path(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
