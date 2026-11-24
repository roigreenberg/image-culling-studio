import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from src.io_utils import save_jpeg_bgr
from webapp.db import CullDatabase, make_image_id
from webapp.pipeline_runner import TaskManager
from webapp.xmp_sync import read_xmp_sidecar


class TestPipelineRunner(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self._tmp.name)
        self.db = CullDatabase(self.root / "studio.db")
        self.events = []
        self.tm = TaskManager(self.db, broadcast=self.events.append)

        self.src_dir = self.root / "photos"
        self.out_dir = self.root / "out"
        (self.src_dir / "day1").mkdir(parents=True)
        self.out_dir.mkdir(parents=True)

        img = np.full((120, 160, 3), 100, dtype=np.uint8)
        for i in range(6):
            save_jpeg_bgr(img, self.src_dir / "day1" / f"IMG_{i:03d}.jpg")

        self.dir_rec = self.db.register_directory(self.src_dir, self.out_dir)
        self.dir_id = self.dir_rec["dir_id"]

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_directory_scan_and_previews(self) -> None:
        ok, msg = self.tm.start_directory_scan(self.dir_id)
        self.assertTrue(ok, msg)

        # Wait up to 5s for background scan thread to finish
        deadline = time.time() + 5.0
        while self.tm.get_status()["active"] and time.time() < deadline:
            time.sleep(0.05)

        st = self.tm.get_status()
        self.assertFalse(st["active"])
        self.assertEqual(st["stage"], "done")

        q = self.db.query_images(self.dir_id)
        self.assertEqual(q["total"], 6)

        first_img = q["items"][0]
        hq_path = self.tm.ensure_single_hq_preview(first_img, self.out_dir)
        self.assertIsNotNone(hq_path)
        self.assertTrue(hq_path.is_file())

        full_path = self.tm.ensure_full_res_jpeg(first_img, self.out_dir, force_raw_full=False)
        self.assertIsNotNone(full_path)
        self.assertTrue(full_path.is_file())

    def test_apply_ai_mapping_to_xmp(self) -> None:
        img_path = self.src_dir / "day1" / "IMG_000.jpg"
        img_id = make_image_id(self.dir_id, "day1", "IMG_000.jpg")
        self.db.upsert_scanned_images(
            self.dir_id,
            [
                {
                    "image_id": img_id,
                    "source_path": str(img_path),
                    "subfolder": "day1",
                    "filename": "IMG_000.jpg",
                    "stem": "IMG_000",
                }
            ],
        )
        self.db.update_ai_results(
            self.dir_id,
            "run_1",
            [{"image_id": img_id, "source_path": str(img_path), "layer": "layer2_to_edit", "track": "Wildlife", "overall_score": 90.0}],
        )

        dry = self.tm.apply_ai_mapping_to_xmp(self.dir_id, dry_run=True)
        self.assertEqual(dry["would_update"], 1)
        self.assertEqual(dry["synced"], 0)
        self.assertFalse(read_xmp_sidecar(img_path)["exists"])

        real = self.tm.apply_ai_mapping_to_xmp(self.dir_id, dry_run=False)
        self.assertEqual(real["synced"], 1)
        xmp = read_xmp_sidecar(img_path)
        self.assertTrue(xmp["exists"])
        self.assertEqual(xmp["label"], "Green")

    def test_train_personal_ranker_with_mocked_output_path(self) -> None:
        # Seed 6 culled images (3 keep/edit, 3 reject) + results.json + embeddings.npz
        ids = []
        recs = []
        for i in range(6):
            fn = f"IMG_{i:03d}.jpg"
            img_id = make_image_id(self.dir_id, "day1", fn)
            ids.append(img_id)
            self.db.upsert_scanned_images(
                self.dir_id,
                [
                    {
                        "image_id": img_id,
                        "source_path": str(self.src_dir / "day1" / fn),
                        "subfolder": "day1",
                        "filename": fn,
                        "stem": f"IMG_{i:03d}",
                    }
                ],
            )
            recs.append(
                {
                    "image_id": img_id,
                    "source_path": str(self.src_dir / "day1" / fn),
                    "layer": "layer2_to_edit" if i < 3 else "rejected",
                    "track": "Wildlife",
                    "overall_score": 85.0 if i < 3 else 25.0,
                    "sharpness_score": 80.0 if i < 3 else 20.0,
                    "eye_contact_score": 75.0 if i < 3 else 15.0,
                    "action_score": 70.0 if i < 3 else 10.0,
                    "scenic_score": 30.0,
                    "tonal_balance": 0.6,
                }
            )
        self.db.update_ai_results(self.dir_id, "run_1", recs)
        for i, img_id in enumerate(ids):
            self.db.record_metadata_change(img_id, rating=5 if i < 3 else 1, source="user")

        (self.out_dir / "results.json").write_text(json.dumps(recs), encoding="utf-8")
        rng = np.random.default_rng(42)
        np.savez_compressed(
            self.out_dir / "embeddings.npz",
            image_ids=np.array(ids),
            siglip_embs=rng.normal(size=(6, 8)).astype(np.float16),
            bioclip_embs=rng.normal(size=(6, 8)).astype(np.float16),
        )

        import webapp.pipeline_runner as pr_mod
        with (
            patch.object(pr_mod, "__file__", str(self.root / "webapp" / "pipeline_runner.py")),
            patch("train_ranker.train_from_picks") as mock_train,
        ):
            res = self.tm.train_personal_ranker(self.dir_id)
        self.assertTrue(res["ok"])
        self.assertEqual(res["keep_edit_count"], 3)
        self.assertEqual(res["reject_count"], 3)
        self.assertTrue((self.root / "studio_data" / "my_picks.json").is_file())
        mock_train.assert_called_once()

    def _wait_idle(self, timeout: float = 10.0) -> dict:
        deadline = time.time() + timeout
        while self.tm.get_status()["active"] and time.time() < deadline:
            time.sleep(0.05)
        return self.tm.get_status()

    def test_cancel_when_idle_and_hq_preview_batch(self) -> None:
        self.assertFalse(self.tm.cancel_current())
        # Scan first so images are in DB
        self.tm.start_directory_scan(self.dir_id)
        self._wait_idle()

        ok, msg = self.tm.start_hq_preview_batch(self.dir_id)
        self.assertTrue(ok, msg)
        st = self._wait_idle()
        self.assertFalse(st["active"])
        self.assertEqual(st["stage"], "done")

    def test_start_culling_and_reselect_with_mocked_models(self) -> None:
        self.tm.start_directory_scan(self.dir_id)
        self._wait_idle()

        fake_kestrel = MagicMock()
        fake_kestrel.analyze_image.return_value = (
            {
                "has_wildlife": True,
                "animal_group": "bird",
                "species": "Eurasian Kestrel",
                "species_conf": 0.9,
                "family": "Falconidae",
                "md_conf": 0.92,
                "animal_gate_score": 0.95,
                "kestrel_quality": 0.85,
                "head_sharp_raw": 2.2,
                "bbox": [10, 10, 100, 100],
                "head_bbox": [30, 20, 70, 60],
                "subject_area_frac": 0.25,
                "num_verified_animals": 1,
                "global_sharp_raw": 1.8,
                "tonal_balance": 0.7,
                "blown_frac": 0.01,
            },
            np.ones(8, dtype=np.float16),
        )

        def _fake_score_all(records, _out_dir):
            for idx, r in enumerate(records):
                r["siglip_action_raw"] = 0.12 - idx * 0.01
                r["siglip_scenic_raw"] = 0.05
                r["siglip_scenic_prob"] = 0.3
                r["siglip_light_prob"] = 0.1
                r["siglip_eye_raw"] = 0.10
            return np.ones((len(records), 8), dtype=np.float16)

        fake_siglip = MagicMock()
        fake_siglip.score_all_images.side_effect = _fake_score_all

        with (
            patch("src.kestrel_bridge.KestrelCoreRunner", return_value=fake_kestrel),
            patch("src.siglip2_scorer.SigLIP2Scorer", return_value=fake_siglip),
            patch("src.scene_burst_and_select._apply_personal_ranker_if_available"),
        ):
            ok, msg = self.tm.start_culling(self.dir_id, mode="full")
            self.assertTrue(ok, msg)
            st = self._wait_idle(timeout=15.0)

        self.assertFalse(st["active"])
        self.assertEqual(st["stage"], "done")
        self.assertTrue((self.out_dir / "results.json").is_file())
        self.assertTrue((self.out_dir / "embeddings.npz").is_file())

        # Now test start_reselect using the generated embeddings.npz + results.json
        with patch("src.scene_burst_and_select._apply_personal_ranker_if_available"):
            ok_rs, msg_rs = self.tm.start_reselect(self.dir_id)
            self.assertTrue(ok_rs, msg_rs)
            st_rs = self._wait_idle(timeout=10.0)
        self.assertFalse(st_rs["active"])
        self.assertEqual(st_rs["stage"], "done")


if __name__ == "__main__":
    unittest.main()
