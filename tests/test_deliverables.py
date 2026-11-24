import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.config import PipelineConfig
from src.deliverables import (
    _safe_label_slug,
    export_preview_folders,
    write_structured_data_and_embeddings,
)
from src.io_utils import save_jpeg_bgr


def _sample_record(idx: int, src_path: Path, layer: str = "layer2_to_edit", track: str = "Wildlife") -> dict:
    return {
        "_emb_idx": idx,
        "_siglip_crop_rel_path": f"gallery_assets/siglip_crops/img_{idx}.jpg",
        "image_id": f"day1__img_{idx}.jpg",
        "overall_rank": idx + 1,
        "layer": layer,
        "track": track,
        "species": "Eurasian Kestrel" if track == "Wildlife" else "None",
        "species_conf": 0.91,
        "family": "Falconidae",
        "animal_group": "Bird",
        "hybrid_bonus": False,
        "subfolder": "day1",
        "filename": f"img_{idx}.jpg",
        "stem": f"img_{idx}",
        "overall_score": 88.5 - idx * 5,
        "sharpness_score": 82.0,
        "eye_contact_score": 76.0,
        "action_score": 70.0,
        "scenic_score": 65.0,
        "kestrel_quality": 0.85,
        "head_sharp_raw": 2.1,
        "global_sharp_raw": 1.8,
        "tonal_balance": 0.72,
        "md_conf": 0.95,
        "animal_gate_score": 0.98,
        "burst_id": "S0001",
        "burst_size": 2,
        "burst_rank": idx + 1,
        "burst_index": idx + 1,
        "timestamp": 1700000000.0 + idx,
        "timestamp_str": "2025-05-01 06:00:00",
        "focal_length": 500.0,
        "f_number": 5.6,
        "exposure_time": "1/2000s",
        "iso": 400,
        "bbox": [10, 20, 100, 120],
        "head_bbox": [30, 30, 60, 60],
        "source_path": str(src_path),
        "thumb_rel_path": f"gallery_assets/thumbs/img_{idx}.jpg",
        "crop_rel_path": f"gallery_assets/crops/img_{idx}.jpg",
    }


class TestDeliverables(unittest.TestCase):
    def test_safe_label_slug(self) -> None:
        self.assertEqual(_safe_label_slug({"track": "Landscape"}), "SunriseLandscape")
        self.assertEqual(_safe_label_slug({"track": "Wildlife", "species": "Common Kingfisher!"}), "CommonKingfisher")
        self.assertEqual(_safe_label_slug({"track": "Wildlife", "species": ""}), "Wildlife")

    def test_write_structured_data_embeddings_and_html_and_exports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src_dir = tmp_path / "src"
            src_dir.mkdir()
            out_dir = tmp_path / "out"
            cfg = PipelineConfig(source_dir=src_dir, output_dir=out_dir, io_workers=2, export_max_dim=120)
            cfg.ensure_output_dirs()

            img = np.full((160, 240, 3), 120, dtype=np.uint8)
            src1 = src_dir / "img_0.jpg"
            src2 = src_dir / "img_1.jpg"
            save_jpeg_bgr(img, src1)
            save_jpeg_bgr(img, src2)

            records = [
                _sample_record(0, src1, layer="layer2_to_edit", track="Wildlife"),
                _sample_record(1, src2, layer="layer1_keep", track="Landscape"),
            ]
            bioclip_embs = np.ones((2, 16), dtype=np.float32)
            siglip_embs = np.ones((2, 32), dtype=np.float32) * 0.5

            export_preview_folders(records, cfg)
            self.assertTrue((out_dir / records[0]["export_layer2_path"]).is_file())
            self.assertTrue((out_dir / records[0]["export_layer1_path"]).is_file())
            self.assertTrue((out_dir / records[1]["export_layer1_path"]).is_file())
            self.assertNotIn("export_layer2_path", records[1])

            write_structured_data_and_embeddings(records, bioclip_embs, siglip_embs, cfg)

            results_json = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(len(results_json), 2)
            self.assertNotIn("_emb_idx", results_json[0])
            self.assertNotIn("_siglip_crop_rel_path", results_json[0])

            with open(out_dir / "results.csv", "r", encoding="utf-8-sig", newline="") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["filename"], "img_0.jpg")
            self.assertEqual(json.loads(rows[0]["bbox"]), [10, 20, 100, 120])

            with np.load(out_dir / "embeddings.npz") as npz:
                self.assertEqual(npz["bioclip_embs"].shape, (2, 16))
                self.assertEqual(npz["siglip_embs"].shape, (2, 32))
                self.assertEqual(list(npz["image_ids"]), ["day1__img_0.jpg", "day1__img_1.jpg"])


if __name__ == "__main__":
    unittest.main()
