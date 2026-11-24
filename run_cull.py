"""Headless GPU-accelerated Project Kestrel + SigLIP 2 Two-Track Culling Pipeline."""

from __future__ import annotations

import argparse
import collections
import gc
import json
import logging
import os
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

from src.config import PipelineConfig, load_local_env
from src.deliverables import (
    export_preview_folders,
    write_structured_data_and_embeddings,
)
from src.io_utils import discover_images, load_preview_bgr_and_meta, save_jpeg_bgr
from src.kestrel_bridge import KestrelCoreRunner, extract_crops_bgr
from src.scene_burst_and_select import (
    assign_scenes_and_bursts,
    route_and_score_all,
    select_layers_final,
)
from src.siglip2_scorer import SigLIP2Scorer


def _setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def _format_duration(seconds: float) -> str:
    mins, secs = divmod(int(round(seconds)), 60)
    hrs, mins = divmod(mins, 60)
    if hrs > 0:
        return f"{hrs}h {mins}m {secs}s"
    if mins > 0:
        return f"{mins}m {secs}s"
    return f"{secs}s"


def _make_image_id(subfolder: str, filename: str) -> str:
    safe_sf = "".join(c if c.isalnum() or c in "-_" else "_" for c in subfolder)
    safe_fn = "".join(c if c.isalnum() or c in "-_." else "_" for c in filename)
    return f"{safe_sf}__{safe_fn}"


def _run_step2a_kestrel(
    image_items: List[Tuple[Path, str]],
    cfg: PipelineConfig,
) -> Tuple[List[Dict[str, Any]], np.ndarray, int]:
    """
    Step 2A: Run Project Kestrel (MegaDetector + SAM-HQ + BioCLIP 2 + AnimalGate +
    QualityClassifier) across 100% of images with strict read-only Hebrew-safe I/O.
    """
    logger = logging.getLogger("pipeline.step2a")
    cache_path = cfg.output_dir / "kestrel_cache.jsonl"
    bio_emb_dir = cfg.output_dir / "gallery_assets" / "bioclip_embs"
    bio_emb_dir.mkdir(parents=True, exist_ok=True)

    cached_by_id: Dict[str, Dict[str, Any]] = {}
    if cfg.resume and cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    cached_by_id[rec["image_id"]] = rec
                except Exception:
                    continue
        logger.info("Loaded %d cached Step 2A Kestrel records.", len(cached_by_id))

    records: List[Dict[str, Any]] = []
    todo_items: List[Tuple[Path, str, str]] = []
    for path, subfolder in image_items:
        img_id = _make_image_id(subfolder, path.name)
        if img_id in cached_by_id:
            rec = cached_by_id[img_id]
            ok_thumb = (cfg.output_dir / rec["thumb_rel_path"]).exists()
            ok_crop = (cfg.output_dir / rec["crop_rel_path"]).exists()
            ok_emb = (bio_emb_dir / f"{img_id}.npy").exists()
            ok_sc = (
                not rec.get("has_wildlife")
                or (cfg.output_dir / rec.get("_siglip_crop_rel_path", "")).exists()
            )
            if ok_thumb and ok_crop and ok_emb and ok_sc:
                records.append(rec)
                continue
        todo_items.append((path, subfolder, img_id))

    skipped_errors = 0
    if todo_items:
        kestrel = KestrelCoreRunner(cfg)

        # Bounded producer queue keeps RAM strictly bounded
        max_inflight = 12
        work_q: queue.Queue = queue.Queue(maxsize=12)
        sentinel = object()

        def _load_one(item: Tuple[Path, str, str]) -> Optional[Dict[str, Any]]:
            path, subfolder, img_id = item
            img_bgr, meta = load_preview_bgr_and_meta(path)
            if img_bgr is None:
                return None
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            return {
                "path": path,
                "subfolder": subfolder,
                "image_id": img_id,
                "img_bgr": img_bgr,
                "img_rgb": img_rgb,
                "meta": meta,
            }

        def _producer() -> None:
            nonlocal skipped_errors
            inflight: collections.deque = collections.deque()
            with ThreadPoolExecutor(max_workers=cfg.io_workers) as pool:
                it = iter(todo_items)
                for _ in range(max_inflight):
                    try:
                        inflight.append(pool.submit(_load_one, next(it)))
                    except StopIteration:
                        break
                while inflight:
                    fut = inflight.popleft()
                    try:
                        inflight.append(pool.submit(_load_one, next(it)))
                    except StopIteration:
                        pass
                    try:
                        res = fut.result()
                    except Exception as exc:
                        logger.warning("Read error: %s", exc)
                        res = None
                    if res is None:
                        skipped_errors += 1
                        continue
                    work_q.put(res)
            work_q.put(sentinel)

        prod = threading.Thread(target=_producer, daemon=True)
        prod.start()

        save_pool = ThreadPoolExecutor(max_workers=4)
        pbar = tqdm(total=len(todo_items), desc="Step 2A: Kestrel (MD+SAM+BioCLIP2)", unit="img")

        with open(cache_path, "a", encoding="utf-8") as cache_fh:
            count = 0
            while True:
                item = work_q.get()
                if item is sentinel:
                    break

                img_id = item["image_id"]
                img_bgr = item["img_bgr"]
                img_rgb = item["img_rgb"]
                meta = item["meta"]
                path: Path = item["path"]
                subfolder: str = item["subfolder"]

                kres, bio_emb = kestrel.analyze_image(img_rgb, ascii_name=f"{img_id}.jpg")

                subj_box = tuple(kres["bbox"]) if kres["has_wildlife"] else None
                head_box = tuple(kres["head_bbox"]) if kres["has_wildlife"] else None
                zoom_crop, tight_crop = extract_crops_bgr(
                    img_bgr, subj_box, head_box, output_size=cfg.crop_output_size
                )

                thumb_rel = f"gallery_assets/thumbs/{img_id}.jpg"
                crop_rel = f"gallery_assets/crops/{img_id}.jpg"
                scrop_rel = f"gallery_assets/siglip_crops/{img_id}.jpg" if tight_crop is not None else ""

                # Save JPEGs + BioCLIP embedding
                np.save(bio_emb_dir / f"{img_id}.npy", bio_emb)
                save_pool.submit(
                    save_jpeg_bgr,
                    img_bgr,
                    cfg.output_dir / thumb_rel,
                    cfg.gallery_thumb_quality,
                    cfg.gallery_thumb_max_dim,
                )
                save_pool.submit(
                    save_jpeg_bgr,
                    zoom_crop,
                    cfg.output_dir / crop_rel,
                    cfg.crop_quality,
                    None,
                )
                if tight_crop is not None:
                    save_pool.submit(
                        save_jpeg_bgr,
                        tight_crop,
                        cfg.output_dir / scrop_rel,
                        cfg.crop_quality,
                        None,
                    )

                rec = {
                    "image_id": img_id,
                    "source_path": str(path),
                    "subfolder": subfolder,
                    "filename": path.name,
                    "stem": path.stem,
                    "timestamp": float(meta["timestamp"]),
                    "timestamp_str": meta["timestamp_str"],
                    "focal_length": meta.get("focal_length"),
                    "f_number": meta.get("f_number"),
                    "exposure_time": meta.get("exposure_time", ""),
                    "iso": meta.get("iso"),
                    "camera_model": meta.get("camera_model", ""),
                    "preview_width": meta.get("preview_width", 0),
                    "preview_height": meta.get("preview_height", 0),
                    "thumb_rel_path": thumb_rel,
                    "crop_rel_path": crop_rel,
                    "_siglip_crop_rel_path": scrop_rel,
                }
                rec.update(kres)
                records.append(rec)
                cache_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

                item["img_bgr"] = None
                item["img_rgb"] = None
                count += 1
                if count % 25 == 0:
                    cache_fh.flush()
                if count % 200 == 0:
                    gc.collect()
                pbar.update(1)

            cache_fh.flush()

        pbar.close()
        save_pool.shutdown(wait=True)
        prod.join()
        del kestrel
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Load all BioCLIP 2 embeddings aligned with `records`
    bio_embs = np.stack(
        [np.load(bio_emb_dir / f"{r['image_id']}.npy") for r in records]
    ).astype(np.float16)
    logger.info("Step 2A finished: %d processed, %d skipped.", len(records), skipped_errors)
    return records, bio_embs, skipped_errors


def run_pipeline(cfg: PipelineConfig, reselect_only: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()
    cfg.ensure_output_dirs()
    _setup_logging(cfg.output_dir / "pipeline.log")
    logger = logging.getLogger("pipeline")

    logger.info("=" * 74)
    logger.info("Project Kestrel + SigLIP 2 Two-Track Culling Pipeline")
    logger.info("Source (STRICT READ-ONLY): %s", cfg.source_dir)
    logger.info("Output directory:          %s", cfg.output_dir.resolve())

    if reselect_only:
        res_path = cfg.output_dir / "results.json"
        npz_path = cfg.output_dir / "embeddings.npz"
        if not res_path.is_file() or not npz_path.is_file():
            raise FileNotFoundError("--reselect-only requires existing results.json and embeddings.npz")
        logger.info("Reselect-only mode: loading cached scores & embeddings from %s", cfg.output_dir)
        records = json.loads(res_path.read_text(encoding="utf-8"))
        npz = np.load(npz_path)
        bio_embs = npz["bioclip_embs"]
        siglip_embs = npz["siglip_embs"]
        skipped_errors = 0
    if not reselect_only:
        if cfg.source_dir is None:
            raise ValueError(
                "Source directory is required. Pass --input-dir / --source or set CULL_SOURCE_DIR in .env."
            )
        has_cuda = torch.cuda.is_available()
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        logger.info("PyTorch: %s | CUDA: %s | MPS: %s", torch.__version__, has_cuda, has_mps)
        if has_cuda:
            props = torch.cuda.get_device_properties(0)
            logger.info("GPU: %s (%.2f GB VRAM)", props.name, props.total_memory / (1024**3))
        elif has_mps:
            logger.info("GPU: Apple Silicon MPS")
        else:
            logger.warning("No CUDA/MPS GPU detected — running on CPU (slower).")

        image_items = discover_images(cfg.source_dir, limit=cfg.limit)
        if not image_items:
            raise RuntimeError(f"No supported images found in {cfg.source_dir}")
        logger.info("Discovered %d images across source subfolders.", len(image_items))

        # Step 2A: Project Kestrel Core Extraction on 100% of images
        records, bio_embs, skipped_errors = _run_step2a_kestrel(image_items, cfg)

        # Step 2B: Batched SigLIP 2 Scoring on 100% of images (Full Frame + Wildlife Head Crops)
        siglip2 = SigLIP2Scorer(cfg)
        siglip_embs = siglip2.score_all_images(records, cfg.output_dir)
        siglip2.close()

    # Step 2A.4: Scene & Burst Grouping across 100% of images (with >= 10% zoom split)
    records = assign_scenes_and_bursts(records, siglip_embs, cfg)

    # Step 2C: Two-Track Routing + Hybrid +15% Bonus + Optional Personal Ranker
    records = route_and_score_all(records, cfg, bioclip_embs=bio_embs, siglip_embs=siglip_embs)

    # Step 2D: Quality- & Scenario-Driven Selection (no hardcoded counts)
    records = select_layers_final(records, cfg, siglip_embs=siglip_embs)

    # Step 3: Export 2048px JPEGs and CSV/JSON/embeddings.npz
    export_preview_folders(records, cfg)
    write_structured_data_and_embeddings(records, bio_embs, siglip_embs, cfg)

    elapsed = time.perf_counter() - t0
    w_count = sum(1 for r in records if r["track"] == "Wildlife")
    l_count = sum(1 for r in records if r["track"] == "Landscape")
    o_count = sum(1 for r in records if r["track"] == "Other")
    h_count = sum(1 for r in records if r.get("hybrid_bonus"))
    l2_count = sum(1 for r in records if r["layer"] == "layer2_to_edit")
    l1_total = sum(1 for r in records if r["layer"] in ("layer2_to_edit", "layer1_keep"))
    rej_count = sum(1 for r in records if r["layer"] == "rejected")

    summary = {
        "total_processed": len(records),
        "skipped_errors": skipped_errors,
        "wildlife_count": w_count,
        "landscape_count": l_count,
        "other_count": o_count,
        "hybrid_light_count": h_count,
        "layer1_total_count": l1_total,
        "layer1_pct": round(100.0 * l1_total / max(1, len(records)), 2),
        "layer2_count": l2_count,
        "layer2_pct": round(100.0 * l2_count / max(1, len(records)), 2),
        "rejected_count": rej_count,
        "runtime_seconds": round(elapsed, 2),
        "runtime_str": _format_duration(elapsed),
        "output_dir": str(cfg.output_dir.resolve()),
    }

    logger.info("=" * 74)
    logger.info("PIPELINE COMPLETE SUMMARY")
    logger.info("  Total images processed  : %d (skipped: %d)", summary["total_processed"], skipped_errors)
    logger.info("  Wildlife / Birds (A)    : %d (Hybrid Light +15%%: %d)", w_count, h_count)
    logger.info("  Sun / Landscape (B)     : %d", l_count)
    logger.info("  Other / Misfires        : %d", o_count)
    logger.info("  Layer 1 (Keep total)    : %d (%.2f%%)", l1_total, summary["layer1_pct"])
    logger.info("  Layer 2 (To Edit)       : %d (%.2f%%)", l2_count, summary["layer2_pct"])
    logger.info("  Total runtime           : %s", summary["runtime_str"])
    logger.info("  Output directory        : %s", summary["output_dir"])
    logger.info("=" * 74)
    return summary


def main() -> None:
    load_local_env()
    parser = argparse.ArgumentParser(description="Project Kestrel + SigLIP 2 Culling Pipeline")
    parser.add_argument(
        "--input-dir",
        "--source",
        dest="source",
        type=str,
        default=os.environ.get("CULL_SOURCE_DIR", ""),
        help="Strict read-only source photo directory (or set CULL_SOURCE_DIR in .env)",
    )
    parser.add_argument(
        "--output-dir",
        "--output",
        dest="output",
        type=str,
        default=os.environ.get("CULL_OUTPUT_DIR", "./cull_output"),
        help="Output directory for previews, CSV/JSON, and embeddings.npz",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--detector", type=str, default="mdv1000-cedar", choices=["mdv1000-cedar", "mdv5a"])
    parser.add_argument("--siglip-batch", type=int, default=16)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--reselect-only", action="store_true", help="Re-run grouping, routing, selection & exports from cached results.json + embeddings.npz")
    args = parser.parse_args()

    if not args.reselect_only and not args.source:
        parser.error("Provide --input-dir <path> (or set CULL_SOURCE_DIR in an untracked .env file).")

    cfg = PipelineConfig(
        source_dir=Path(args.source) if args.source else None,
        output_dir=Path(args.output),
        limit=args.limit,
        kestrel_detector=args.detector,
        siglip2_batch_size=args.siglip_batch,
        resume=not args.no_resume,
    )
    run_pipeline(cfg, reselect_only=args.reselect_only)


if __name__ == "__main__":
    main()
