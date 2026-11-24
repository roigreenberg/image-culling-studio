"""Isolated E2E test server with temporary SQLite DB and synthetic images."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import uvicorn

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import webapp.server as srv
from src.io_utils import save_jpeg_bgr
from webapp.db import CullDatabase, make_image_id
from webapp.pipeline_runner import TaskManager


def _seed_and_patch() -> tempfile.TemporaryDirectory:
    tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    root = Path(tmp.name)
    src_dir = root / "photos"
    out_dir = root / "out"
    (src_dir / "day1").mkdir(parents=True)
    (src_dir / "day2").mkdir(parents=True)
    (out_dir / "gallery_assets" / "thumbs").mkdir(parents=True)
    (out_dir / "gallery_assets" / "crops").mkdir(parents=True)

    test_db = CullDatabase(root / "e2e_studio.db")
    test_tm = TaskManager(test_db, broadcast=srv.broadcast_event)

    dir_rec = test_db.register_directory(src_dir, out_dir)
    dir_id = dir_rec["dir_id"]

    # Create 4 distinct synthetic images across 2 subfolders and 2 bursts
    specs = [
        ("day1", "KESTREL_001.jpg", "layer2_to_edit", "Wildlife", "Eurasian Kestrel", "S0001", 2, 1, 1, 91.5, 1, (30, 140, 200), 5, "Green"),
        ("day1", "KESTREL_002.jpg", "layer1_keep", "Wildlife", "Eurasian Kestrel", "S0001", 2, 2, 2, 74.0, 2, (40, 120, 180), 0, "Yellow"),
        ("day2", "SUNRISE_003.jpg", "layer1_keep", "Landscape", "None", "S0002", 2, 1, 1, 68.0, 3, (200, 120, 40), 0, "Yellow"),
        ("day2", "BLURRY_004.jpg", "to_delete", "Other", "None", "S0002", 2, 2, 2, 22.0, 4, (80, 80, 80), 0, "Red"),
    ]

    scanned = []
    ai_recs = []
    for idx, (sf, fn, layer, track, species, bid, bsize, bidx, brk, score, rank, color, cam_r, cam_lbl) in enumerate(specs):
        src_path = src_dir / sf / fn
        img = np.full((180, 240, 3), color, dtype=np.uint8)
        save_jpeg_bgr(img, src_path)

        img_id = make_image_id(dir_id, sf, fn)
        thumb_rel = f"gallery_assets/thumbs/{img_id}.jpg"
        crop_rel = f"gallery_assets/crops/{img_id}.jpg"
        save_jpeg_bgr(img, out_dir / thumb_rel)
        save_jpeg_bgr(img[:120, :120], out_dir / crop_rel)

        scanned.append({
            "image_id": img_id,
            "source_path": str(src_path),
            "subfolder": sf,
            "filename": fn,
            "stem": Path(fn).stem,
            "file_ext": ".jpg",
            "file_size": src_path.stat().st_size,
            "timestamp": 1700000000.0 + idx * 10,
            "timestamp_str": f"2025-05-01 06:00:{idx * 10:02d}",
            "focal_length": 500.0 if track == "Wildlife" else 24.0,
            "f_number": 5.6,
            "exposure_time": "1/2000s" if track == "Wildlife" else "1/125s",
            "exposure_seconds": 0.0005 if track == "Wildlife" else 0.008,
            "iso": 400,
            "camera_model": "Canon EOS R5",
            "lens_model": "RF100-500mm" if track == "Wildlife" else "RF24-70mm",
            "camera_rating": cam_r,
            "camera_label": cam_lbl,
            "xmp_exists": False,
            "xmp_rating": None,
            "xmp_label": None,
            "af_points": [{"x": 0.4, "y": 0.35, "w": 0.1, "h": 0.1, "cx": 0.45, "cy": 0.4, "in_focus": True}] if idx == 0 else [],
        })
        ai_recs.append({
            "image_id": img_id,
            "source_path": str(src_path),
            "layer": layer,
            "track": track,
            "species": species,
            "species_conf": 0.92 if track == "Wildlife" else 0.0,
            "burst_id": bid,
            "burst_size": bsize,
            "burst_index": bidx,
            "burst_rank": brk,
            "overall_score": score,
            "sharpness_score": score - 2.0,
            "eye_contact_score": 80.0 if track == "Wildlife" else 0.0,
            "action_score": 75.0 if track == "Wildlife" else 0.0,
            "scenic_score": 82.0 if track == "Landscape" else 15.0,
            "overall_rank": rank,
            "thumb_rel_path": thumb_rel,
            "crop_rel_path": crop_rel,
            "bbox": [20, 20, 180, 150] if track == "Wildlife" else [],
            "head_bbox": [60, 30, 120, 90] if track == "Wildlife" else [],
            "scenario_type": "action_sequence" if track == "Wildlife" else ("static_group" if track == "Landscape" else "single"),
            "hybrid_bonus": 1 if idx == 0 else 0,
        })

    test_db.upsert_scanned_images(dir_id, scanned)
    test_db.update_ai_results(dir_id, "e2e_run_1", ai_recs)

    srv.db = test_db
    srv.tasks = test_tm
    srv.DEFAULT_OUTPUT_DIR = out_dir
    srv._thumb_path_cache.clear()
    srv._crop_path_cache.clear()
    return tmp


def main() -> None:
    _tmp_keepalive = _seed_and_patch()
    port = int(os.environ.get("E2E_PORT", "8799"))
    uvicorn.run(srv.app, host="127.0.0.1", port=port, log_level="warning")
    _tmp_keepalive.cleanup()


if __name__ == "__main__":
    main()
