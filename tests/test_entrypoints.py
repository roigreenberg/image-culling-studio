import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np

import run_cull
import run_webapp
import setup_models
import train_ranker
from src.io_utils import save_jpeg_bgr


class TestEntrypoints(unittest.TestCase):
    def test_setup_models_validation_and_urls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing_file = root / "missing.onnx"
            self.assertFalse(setup_models.is_valid_asset(missing_file))

            empty_file = root / "empty.onnx"
            empty_file.write_bytes(b"")
            self.assertFalse(setup_models.is_valid_asset(empty_file))

            lfs_file = root / "lfs.onnx"
            lfs_file.write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\n")
            self.assertFalse(setup_models.is_valid_asset(lfs_file))

            valid_file = root / "valid.json"
            valid_file.write_bytes(b'{"ok": true}')
            self.assertTrue(setup_models.is_valid_asset(valid_file))

            core_only = setup_models.required_files_for_detector("mdv1000-cedar")
            with_mdv5a = setup_models.required_files_for_detector("mdv5a")
            self.assertGreater(len(with_mdv5a), len(core_only))

            missing_list = setup_models.find_missing_kestrel_models(root, "mdv1000-cedar")
            self.assertEqual(len(missing_list), len(core_only))

            lfs_urls = setup_models._candidate_urls("quality.onnx", True, "main")
            self.assertIn("media.githubusercontent.com", lfs_urls[0])
            raw_urls = setup_models._candidate_urls("birds/NOTICES.md", False, "main")
            self.assertIn("raw.githubusercontent.com", raw_urls[0])

            # CLI --check returns 1 when models are missing
            with patch.object(sys, "argv", ["setup_models.py", "--models-dir", str(root), "--check"]):
                self.assertEqual(setup_models.main(), 1)

    def test_train_ranker_from_picks_and_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            picks_path = root / "my_picks.json"
            results_path = root / "results.json"
            npz_path = root / "embeddings.npz"
            model_out = root / "models" / "personal_ranker.pkl"
            history_jsonl = root / "models" / "feedback_history.jsonl"

            with self.assertRaises(FileNotFoundError):
                train_ranker.train_from_picks(picks_path, results_path, npz_path, model_out, history_jsonl)

            picks_path.write_text(json.dumps({"picks": []}), encoding="utf-8")
            with self.assertRaises(ValueError):
                train_ranker.train_from_picks(picks_path, results_path, npz_path, model_out, history_jsonl)

            # Build 10 samples (5 keep/edit, 5 reject) so StratifiedKFold CV also runs
            ids = [f"day1__img_{i}.jpg" for i in range(10)]
            picks = [
                {"image_id": ids[i], "user_decision": "layer2_edit" if i < 5 else "reject"}
                for i in range(10)
            ]
            picks_path.write_text(json.dumps({"picks": picks}), encoding="utf-8")

            records = [
                {
                    "image_id": ids[i],
                    "sharpness_score": 85.0 if i < 5 else 20.0,
                    "eye_contact_score": 80.0 if i < 5 else 15.0,
                    "action_score": 75.0 if i < 5 else 10.0,
                    "scenic_score": 30.0,
                    "tonal_balance": 0.7,
                    "track": "Wildlife",
                }
                for i in range(10)
            ]
            results_path.write_text(json.dumps(records), encoding="utf-8")

            rng = np.random.default_rng(7)
            np.savez_compressed(
                npz_path,
                image_ids=np.array(ids),
                siglip_embs=rng.normal(size=(10, 8)).astype(np.float16),
                bioclip_embs=rng.normal(size=(10, 8)).astype(np.float16),
            )

            train_ranker.train_from_picks(picks_path, results_path, npz_path, model_out, history_jsonl)
            self.assertTrue(model_out.is_file())
            self.assertTrue(history_jsonl.is_file())
            bundle = joblib.load(model_out)
            self.assertEqual(bundle["n_samples"], 10)
            self.assertEqual(bundle["n_pos"], 5)
            self.assertEqual(bundle["n_neg"], 5)

    def test_run_cull_helpers(self) -> None:
        self.assertEqual(run_cull._format_duration(45), "45s")
        self.assertEqual(run_cull._format_duration(125), "2m 5s")
        self.assertEqual(run_cull._format_duration(3665), "1h 1m 5s")
        self.assertEqual(run_cull._make_image_id("day 1", "IMG_01.CR3"), "day_1__IMG_01.CR3")

    def test_setup_models_download_and_preload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "weights.bin"

            class _FakeResp:
                def __init__(self, payload: bytes):
                    self._stream = io.BytesIO(payload)
                    self.headers = {"Content-Length": str(len(payload))}

                def read(self, n: int = -1) -> bytes:
                    return self._stream.read(n)

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

            # First URL returns LFS pointer, second URL returns valid binary
            lfs_bytes = b"version https://git-lfs.github.com/spec/v1\noid sha256:123\n"
            good_bytes = b"VALID_BINARY_PAYLOAD_" * 40
            side_effects = [_FakeResp(lfs_bytes), _FakeResp(good_bytes)]

            with patch("setup_models.urllib.request.urlopen", side_effect=side_effects):
                setup_models._download_file(["http://bad/1", "http://good/2"], dest, "weights.bin")

            self.assertTrue(dest.is_file())
            self.assertEqual(dest.read_bytes(), good_bytes)

        with (
            patch("transformers.AutoProcessor.from_pretrained") as m_proc,
            patch("transformers.AutoModel.from_pretrained") as m_mod,
        ):
            setup_models.preload_siglip2()
            m_proc.assert_called_once()
            m_mod.assert_called_once()

    def test_run_webapp_main(self) -> None:
        with (
            patch.object(sys, "argv", ["run_webapp.py", "--host", "0.0.0.0", "--port", "9001"]),
            patch("run_webapp.uvicorn.run") as mock_uv,
        ):
            run_webapp.main()
        mock_uv.assert_called_once_with(
            "webapp.server:app",
            host="0.0.0.0",
            port=9001,
            log_level="info",
        )

    def test_run_cull_main_end_to_end_mocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src_dir = root / "photos"
            out_dir = root / "out"
            (src_dir / "day1").mkdir(parents=True)
            img = np.full((120, 160, 3), 100, dtype=np.uint8)
            save_jpeg_bgr(img, src_dir / "day1" / "IMG_001.jpg")
            save_jpeg_bgr(img, src_dir / "day1" / "IMG_002.jpg")

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
                patch.object(sys, "argv", ["run_cull.py", "--source", str(src_dir), "--output", str(out_dir), "--no-resume"]),
                patch("run_cull.KestrelCoreRunner", return_value=fake_kestrel),
                patch("run_cull.SigLIP2Scorer", return_value=fake_siglip),
                patch("run_cull._setup_logging"),
                patch("src.scene_burst_and_select._apply_personal_ranker_if_available"),
            ):
                run_cull.main()

            self.assertTrue((out_dir / "results.json").is_file())
            self.assertTrue((out_dir / "results.csv").is_file())
            self.assertTrue((out_dir / "embeddings.npz").is_file())


if __name__ == "__main__":
    unittest.main()
