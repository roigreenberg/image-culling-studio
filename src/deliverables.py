"""Deliverables generator: 2048px JPEG exports, CSV/JSON results, and embeddings.npz."""

from __future__ import annotations

import csv
import json
import logging
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

from .config import PipelineConfig
from .io_utils import load_preview_bgr_and_meta, save_jpeg_bgr

logger = logging.getLogger(__name__)


def _safe_label_slug(rec: Dict[str, Any]) -> str:
    if rec.get("track") == "Landscape":
        return "SunriseLandscape"
    sp = str(rec.get("species") or "Wildlife").strip()
    slug = re.sub(r"[^A-Za-z0-9]+", "", sp.title())
    return slug[:28] or "Wildlife"


def _export_single_preview(
    source_path: Path, out_paths: List[Path], max_dim: int, quality: int
) -> bool:
    img_bgr, _ = load_preview_bgr_and_meta(source_path)
    if img_bgr is None:
        return False
    for out_p in out_paths:
        save_jpeg_bgr(img_bgr, out_p, quality=quality, max_dim=max_dim)
    return True


def export_preview_folders(records: List[Dict[str, Any]], cfg: PipelineConfig) -> None:
    """
    Generate 2048px JPEG previews inside:
      - ./cull_output/layer2_to_edit/<subfolder>/<rank>_score<score>_<SpeciesOrLandscape>_<stem>.jpg
      - ./cull_output/layer1_keep/<subfolder>/<rank>_score<score>_<SpeciesOrLandscape>_<stem>.jpg
    """
    l2_dir = cfg.output_dir / "layer2_to_edit"
    l1_dir = cfg.output_dir / "layer1_keep"
    if l2_dir.exists():
        shutil.rmtree(l2_dir)
    if l1_dir.exists():
        shutil.rmtree(l1_dir)
    l2_dir.mkdir(parents=True, exist_ok=True)
    l1_dir.mkdir(parents=True, exist_ok=True)

    l2_recs = sorted([r for r in records if r["layer"] == "layer2_to_edit"], key=lambda r: r["overall_rank"])
    l1_all = sorted(
        [r for r in records if r["layer"] in ("layer2_to_edit", "layer1_keep")],
        key=lambda r: r["overall_rank"],
    )

    l2_sub_rank: Dict[str, int] = {}
    l2_path_map: Dict[str, Path] = {}
    for rec in l2_recs:
        sf = rec["subfolder"]
        l2_sub_rank[sf] = l2_sub_rank.get(sf, 0) + 1
        sc = int(round(float(rec["overall_score"])))
        slug = _safe_label_slug(rec)
        fname = f"{l2_sub_rank[sf]:03d}_score{sc:02d}_{slug}_{rec['stem']}.jpg"
        p = l2_dir / sf / fname
        l2_path_map[rec["image_id"]] = p
        rec["export_layer2_path"] = str(p.relative_to(cfg.output_dir)).replace("\\", "/")

    l1_sub_rank: Dict[str, int] = {}
    tasks = []
    for rec in l1_all:
        sf = rec["subfolder"]
        l1_sub_rank[sf] = l1_sub_rank.get(sf, 0) + 1
        sc = int(round(float(rec["overall_score"])))
        slug = _safe_label_slug(rec)
        fname = f"{l1_sub_rank[sf]:04d}_score{sc:02d}_{slug}_{rec['stem']}.jpg"
        p1 = l1_dir / sf / fname
        rec["export_layer1_path"] = str(p1.relative_to(cfg.output_dir)).replace("\\", "/")
        outs = [p1]
        if rec["image_id"] in l2_path_map:
            outs.append(l2_path_map[rec["image_id"]])
        tasks.append((Path(rec["source_path"]), outs))

    failed = 0
    with ThreadPoolExecutor(max_workers=cfg.io_workers) as pool:
        futs = [
            pool.submit(_export_single_preview, src, outs, cfg.export_max_dim, cfg.export_quality)
            for src, outs in tasks
        ]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Exporting 2048px JPEGs", unit="img"):
            try:
                if not fut.result():
                    failed += 1
            except Exception as exc:
                logger.warning("Export error: %s", exc)
                failed += 1
    if failed:
        logger.warning("Failed to export %d preview JPEGs.", failed)


def write_structured_data_and_embeddings(
    records: List[Dict[str, Any]],
    bioclip_embs: np.ndarray,
    siglip_embs: np.ndarray,
    cfg: PipelineConfig,
) -> None:
    """Save results.json, results.csv, and embeddings.npz (aligned by image_id)."""
    ordered = sorted(records, key=lambda r: int(r.get("overall_rank", 999999)))
    clean_records = [{k: v for k, v in r.items() if not k.startswith("_")} for r in ordered]

    json_path = cfg.output_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(clean_records, fh, ensure_ascii=False, indent=2)

    csv_fields = [
        "overall_rank",
        "layer",
        "track",
        "species",
        "species_conf",
        "family",
        "animal_group",
        "hybrid_bonus",
        "subfolder",
        "filename",
        "stem",
        "overall_score",
        "sharpness_score",
        "eye_contact_score",
        "action_score",
        "scenic_score",
        "kestrel_quality",
        "head_sharp_raw",
        "global_sharp_raw",
        "tonal_balance",
        "md_conf",
        "animal_gate_score",
        "burst_id",
        "burst_size",
        "burst_rank",
        "timestamp_str",
        "focal_length",
        "f_number",
        "exposure_time",
        "iso",
        "bbox",
        "head_bbox",
        "source_path",
        "export_layer2_path",
        "export_layer1_path",
        "thumb_rel_path",
        "crop_rel_path",
    ]
    csv_path = cfg.output_dir / "results.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for rec in clean_records:
            row = dict(rec)
            row["bbox"] = json.dumps(rec.get("bbox", []))
            row["head_bbox"] = json.dumps(rec.get("head_bbox", []))
            writer.writerow(row)

    # Save aligned embeddings.npz ordered by overall_rank
    emb_indices = [int(r["_emb_idx"]) for r in ordered]
    npz_path = cfg.output_dir / "embeddings.npz"
    np.savez_compressed(
        npz_path,
        image_ids=np.array([r["image_id"] for r in ordered]),
        filenames=np.array([r["filename"] for r in ordered]),
        tracks=np.array([r["track"] for r in ordered]),
        layers=np.array([r["layer"] for r in ordered]),
        overall_scores=np.array([float(r["overall_score"]) for r in ordered], dtype=np.float32),
        bioclip_embs=bioclip_embs[emb_indices].astype(np.float16),
        siglip_embs=siglip_embs[emb_indices].astype(np.float16),
    )
    logger.info("Saved %s, %s, and %s", csv_path, json_path, npz_path)
