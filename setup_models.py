"""Download and verify Project Kestrel AI model weights required for ImageCulling.

Usage:
    python setup_models.py                 # Download default models (~480 MB)
    python setup_models.py --check         # Verify all required files are present
    python setup_models.py --include-mdv5a # Also download optional mdv5a detector (~565 MB)
    python setup_models.py --preload-siglip # Also pre-download Hugging Face SigLIP 2 weights
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

WORKSPACE_ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS_DIR = WORKSPACE_ROOT / "vendor" / "ProjectKestrel" / "analyzer" / "models"

UPSTREAM_REPO = "SanjaySoniLV/ProjectKestrel"
UPSTREAM_REF = "main"

# (relative path under analyzer/models, is_lfs_binary, approx_mb)
CORE_MODEL_FILES: List[Tuple[str, bool, float]] = [
    # Small lookup tables & calibration metadata
    ("quality_normalization_data.csv", False, 0.01),
    ("birds/birds_global.csv", False, 1.0),
    ("birds/NOTICES.md", False, 0.01),
    ("bioclip/NOTICES.md", False, 0.01),
    ("bioclip/gate_index.json", False, 0.01),
    ("bioclip/gate_probe.json", False, 0.01),
    ("bioclip/species_index.json", False, 0.3),
    ("bioclip/wildlife_index.json", False, 0.2),
    # ONNX & NumPy model weights (Git LFS in upstream ProjectKestrel)
    ("quality.onnx", True, 1.2),
    ("speciesnet/mdv1000-cedar.onnx", True, 102.0),
    ("speciesnet/sam_hq_vit_tiny_encoder.onnx", True, 28.2),
    ("speciesnet/sam_hq_vit_tiny_decoder.onnx", True, 18.3),
    ("bioclip/bioclip2_image_int8.onnx", True, 5.1),
    ("bioclip/bioclip2_image_wdq16.onnx", True, 3.0),
    ("bioclip/bioclip2_image.data", True, 303.2),
    ("bioclip/gate_text_emb.npy", True, 0.1),
    ("bioclip/species_text_emb.npy", True, 17.3),
    ("bioclip/wildlife_text_emb.npy", True, 1.8),
]

MDV5A_MODEL_FILES: List[Tuple[str, bool, float]] = [
    ("speciesnet/mdv5a.onnx", True, 1.5),
    ("speciesnet/mdv5a.onnx.data", True, 563.6),
]

_LFS_POINTER_HEADER = b"version https://git-lfs.github.com/spec/v1"


def is_valid_asset(path: Path) -> bool:
    """Return True if path exists, is non-empty, and is not an unresolved Git LFS pointer."""
    if not path.is_file():
        return False
    size = path.stat().st_size
    if size == 0:
        return False
    if size < 512:
        try:
            head = path.read_bytes()[:64]
            if head.startswith(_LFS_POINTER_HEADER):
                return False
        except OSError:
            return False
    return True


def required_files_for_detector(detector_name: str = "mdv1000-cedar") -> List[Tuple[str, bool, float]]:
    files = list(CORE_MODEL_FILES)
    if detector_name == "mdv5a":
        files.extend(MDV5A_MODEL_FILES)
    return files


def find_missing_kestrel_models(
    kestrel_dir: Path = WORKSPACE_ROOT / "vendor" / "ProjectKestrel",
    detector_name: str = "mdv1000-cedar",
) -> List[str]:
    """Return relative paths of any missing or corrupted Project Kestrel model assets."""
    models_dir = kestrel_dir / "analyzer" / "models"
    missing: List[str] = []
    for rel, _, _ in required_files_for_detector(detector_name):
        if not is_valid_asset(models_dir / rel):
            missing.append(rel)
    return missing


def _candidate_urls(rel_path: str, is_lfs: bool, ref: str) -> List[str]:
    upstream_path = f"analyzer/models/{rel_path}"
    media_url = f"https://media.githubusercontent.com/media/{UPSTREAM_REPO}/{ref}/{upstream_path}"
    raw_url = f"https://github.com/{UPSTREAM_REPO}/raw/{ref}/{upstream_path}"
    raw_content_url = f"https://raw.githubusercontent.com/{UPSTREAM_REPO}/{ref}/{upstream_path}"
    if is_lfs:
        return [media_url, raw_url]
    return [raw_content_url, raw_url]


def _download_file(urls: List[str], dest: Path, label: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".download")
    last_err: Optional[Exception] = None

    for url in urls:
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "ImageCulling-ModelSetup/1.0"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                total = int(resp.headers.get("Content-Length") or 0)
                downloaded = 0
                first_chunk = True
                with open(tmp, "wb") as out_fh:
                    while True:
                        chunk = resp.read(1024 * 256)
                        if not chunk:
                            break
                        if first_chunk:
                            first_chunk = False
                            if len(chunk) < 512 and chunk.startswith(_LFS_POINTER_HEADER):
                                raise RuntimeError(f"Received Git LFS pointer instead of binary from {url}")
                        out_fh.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            pct = min(100.0, 100.0 * downloaded / total)
                            mb_done = downloaded / (1024 * 1024)
                            mb_tot = total / (1024 * 1024)
                            sys.stdout.write(f"\r  -> {label}: {mb_done:6.1f} / {mb_tot:6.1f} MB ({pct:5.1f}%)")
                        else:
                            mb_done = downloaded / (1024 * 1024)
                            sys.stdout.write(f"\r  -> {label}: {mb_done:6.1f} MB")
                        sys.stdout.flush()
                sys.stdout.write("\n")

            if not is_valid_asset(tmp):
                raise RuntimeError(f"Downloaded file failed validation: {dest.name}")
            os.replace(tmp, dest)
            return
        except Exception as exc:
            last_err = exc
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            sys.stdout.write("\n")

    raise RuntimeError(f"Failed to download {label} from {urls}: {last_err}")


def preload_siglip2() -> None:
    """Optional helper to pre-download Hugging Face SigLIP 2 weights before first run."""
    from transformers import AutoModel, AutoProcessor

    model_id = "google/siglip2-so400m-patch14-384"
    print(f"\nPre-downloading Hugging Face SigLIP 2 ({model_id})...")
    AutoProcessor.from_pretrained(model_id)
    AutoModel.from_pretrained(model_id)
    print("SigLIP 2 cached successfully.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and verify Project Kestrel AI model weights for ImageCulling."
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help=f"Target Project Kestrel models directory (default: {DEFAULT_MODELS_DIR})",
    )
    parser.add_argument(
        "--ref",
        default=UPSTREAM_REF,
        help=f"Upstream {UPSTREAM_REPO} branch/tag/commit to download from (default: {UPSTREAM_REF})",
    )
    parser.add_argument(
        "--include-mdv5a",
        action="store_true",
        help="Also download the optional larger MegaDetector v5a weights (~565 MB)",
    )
    parser.add_argument(
        "--preload-siglip",
        action="store_true",
        help="Also pre-download Hugging Face SigLIP 2 weights (~1.7 GB)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only verify whether all required model files are present (no downloads)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist locally",
    )
    args = parser.parse_args()

    files = list(CORE_MODEL_FILES)
    if args.include_mdv5a:
        files.extend(MDV5A_MODEL_FILES)

    models_dir: Path = args.models_dir
    missing = [(rel, is_lfs, mb) for (rel, is_lfs, mb) in files if args.force or not is_valid_asset(models_dir / rel)]

    if args.check:
        if not missing:
            print(f"OK: All {len(files)} required Project Kestrel model files are present in {models_dir}")
            return 0
        print(f"Missing {len(missing)} Project Kestrel model file(s) in {models_dir}:")
        for rel, _, mb in missing:
            print(f"  - {rel} (~{mb:.1f} MB)")
        print("\nRun `python setup_models.py` to download them automatically.")
        return 1

    if not missing:
        print(f"All {len(files)} Project Kestrel model files are already present in {models_dir}.")
    else:
        total_mb = sum(mb for _, _, mb in missing)
        print(
            f"Downloading {len(missing)} Project Kestrel model file(s) (~{total_mb:.0f} MB) "
            f"from github.com/{UPSTREAM_REPO} ({args.ref})..."
        )
        for idx, (rel, is_lfs, _) in enumerate(missing, start=1):
            dest = models_dir / rel
            urls = _candidate_urls(rel, is_lfs, args.ref)
            print(f"[{idx}/{len(missing)}] {rel}")
            _download_file(urls, dest, rel)
        print(f"\nDone! Project Kestrel models are ready in {models_dir}.")

    if args.preload_siglip:
        preload_siglip2()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
