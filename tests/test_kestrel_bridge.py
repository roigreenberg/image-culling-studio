import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from src.config import PipelineConfig
from src.kestrel_bridge import (
    KestrelCoreRunner,
    compute_global_landscape_metrics,
    compute_masked_head_sharpness,
    extract_crops_bgr,
)


def _make_textured_rgb(w: int = 320, h: int = 240, sharp: bool = True) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Draw crisp checkerboard / edges for sharp, or smooth gradient for blurry
    if sharp:
        for y in range(0, h, 8):
            for x in range(0, w, 8):
                if ((x // 8) + (y // 8)) % 2 == 0:
                    img[y : y + 8, x : x + 8] = (220, 210, 200)
                else:
                    img[y : y + 8, x : x + 8] = (40, 45, 50)
    else:
        img[:, :] = (120, 120, 120)
        cv2.circle(img, (w // 2, h // 2), 60, (130, 130, 130), -1)
        img = cv2.GaussianBlur(img, (21, 21), 5.0)
    return img


class TestKestrelBridge(unittest.TestCase):
    def test_compute_masked_head_sharpness_sharp_vs_blurry(self) -> None:
        sharp_img = _make_textured_rgb(320, 240, sharp=True)
        blurry_img = _make_textured_rgb(320, 240, sharp=False)
        mask = np.zeros((240, 320), dtype=bool)
        mask[40:200, 80:240] = True
        bbox = (80, 40, 240, 200)

        sharp_val, head_box = compute_masked_head_sharpness(sharp_img, mask, bbox)
        blurry_val, _ = compute_masked_head_sharpness(blurry_img, mask, bbox)

        self.assertGreater(sharp_val, blurry_val)
        self.assertEqual(len(head_box), 4)
        self.assertGreater(head_box[2], head_box[0])
        self.assertGreater(head_box[3], head_box[1])

    def test_compute_masked_head_sharpness_empty_or_tiny_mask(self) -> None:
        img = _make_textured_rgb(100, 100, sharp=True)
        empty_mask = np.zeros((100, 100), dtype=bool)
        val, box = compute_masked_head_sharpness(img, empty_mask, (10, 10, 90, 90))
        self.assertEqual(val, 0.0)
        self.assertEqual(box, (10, 10, 90, 90))

        # Tiny bbox (<16px)
        mask = np.ones((100, 100), dtype=bool)
        val_tiny, _ = compute_masked_head_sharpness(img, mask, (10, 10, 20, 20))
        self.assertEqual(val_tiny, 0.0)

    def test_compute_masked_head_sharpness_large_crop_cap(self) -> None:
        img = _make_textured_rgb(900, 900, sharp=True)
        mask = np.ones((900, 900), dtype=bool)
        val, head_box = compute_masked_head_sharpness(img, mask, (0, 0, 900, 900))
        self.assertGreater(val, 0.0)
        self.assertEqual(len(head_box), 4)

    def test_compute_global_landscape_metrics(self) -> None:
        sharp_land = _make_textured_rgb(400, 300, sharp=True)
        blurry_land = _make_textured_rgb(400, 300, sharp=False)

        m_sharp = compute_global_landscape_metrics(sharp_land)
        m_blurry = compute_global_landscape_metrics(blurry_land)

        self.assertGreater(m_sharp["global_sharp_raw"], m_blurry["global_sharp_raw"])
        self.assertGreaterEqual(m_sharp["tonal_balance"], 0.0)
        self.assertLessEqual(m_sharp["tonal_balance"], 1.0)
        self.assertEqual(m_sharp["blown_frac"], 0.0)

        # Blown-out white image should have blown_frac ~ 1.0 and tonal_balance = 0.0
        blown_img = np.full((200, 200, 3), 255, dtype=np.uint8)
        m_blown = compute_global_landscape_metrics(blown_img)
        self.assertEqual(m_blown["blown_frac"], 1.0)
        self.assertEqual(m_blown["tonal_balance"], 0.0)

    def test_extract_crops_bgr_landscape_and_wildlife(self) -> None:
        img_bgr = np.zeros((600, 800, 3), dtype=np.uint8)
        # Landscape (no bbox)
        zoom_land, tight_land = extract_crops_bgr(img_bgr, None, None, output_size=384)
        self.assertEqual(zoom_land.shape, (384, 384, 3))
        self.assertIsNone(tight_land)

        # Wildlife (with bbox and head_bbox)
        zoom_wild, tight_wild = extract_crops_bgr(
            img_bgr,
            subject_bbox=(100, 100, 500, 450),
            head_bbox=(220, 120, 340, 240),
            output_size=384,
        )
        self.assertEqual(zoom_wild.shape, (384, 384, 3))
        self.assertIsNotNone(tight_wild)
        self.assertEqual(tight_wild.shape, (384, 384, 3))

    def test_kestrel_core_runner_missing_models_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = PipelineConfig(kestrel_dir=Path(tmp))
            with self.assertRaises(FileNotFoundError):
                KestrelCoreRunner(cfg)

    def test_kestrel_core_runner_analyze_image_wildlife_and_landscape(self) -> None:
        # Build fake kestrel_analyzer package in sys.modules so KestrelCoreRunner runs completely offline
        fake_cfg_mod = types.ModuleType("kestrel_analyzer.config")
        fake_cfg_mod.MODELS_DIR = Path(".")
        fake_cfg_mod.QUALITYCLASSIFIER_PATH = Path("q.onnx")
        fake_cfg_mod.QUALITY_NORMALIZATION_DATA_PATH = Path("qn.csv")

        fake_det_sam_cls = MagicMock()
        fake_det_sam = MagicMock()
        fake_det_sam.detector.device = "CPU"
        fake_det_sam.predictor.device = "CPU"
        fake_det_sam.run_with_resilience.side_effect = lambda fn: fn()
        fake_det_sam.get_square_crop.return_value = (
            np.zeros((64, 64, 3), dtype=np.uint8),
            np.ones((64, 64), dtype=bool),
        )
        fake_det_sam_cls.return_value = fake_det_sam

        fake_bio = MagicMock()
        fake_bio.session.get_providers.return_value = ["CPUExecutionProvider"]
        fake_bio.embed.return_value = [np.ones(768, dtype=np.float32)]
        fake_bio.route.return_value = {"route": "bird"}
        fake_bio.classify_embedding.return_value = {
            "top_species_labels": ["Eurasian Kestrel"],
            "top_species_scores": [0.92],
            "top_family_labels": ["Falconidae"],
        }

        fake_gate = MagicMock()
        fake_gate.animal_score.return_value = 0.95
        fake_gate.keep_score.return_value = True

        fake_qual = MagicMock()
        fake_qual.providers_used = ["CPUExecutionProvider"]
        fake_qual.classify.return_value = 0.88

        fake_modules = {
            "kestrel_analyzer": types.ModuleType("kestrel_analyzer"),
            "kestrel_analyzer.config": fake_cfg_mod,
            "kestrel_analyzer.ml": types.ModuleType("kestrel_analyzer.ml"),
            "kestrel_analyzer.ml.animal_gate": types.SimpleNamespace(AnimalGate=MagicMock(return_value=fake_gate)),
            "kestrel_analyzer.ml.bioclip_species": types.SimpleNamespace(BioClipSpeciesClassifier=MagicMock(return_value=fake_bio)),
            "kestrel_analyzer.ml.quality": types.SimpleNamespace(QualityClassifier=MagicMock(return_value=fake_qual)),
            "kestrel_analyzer.ml.speciesnet_sam_hq": types.SimpleNamespace(DetectorSAMHQWrapper=fake_det_sam_cls),
        }

        cfg = PipelineConfig()
        with (
            patch("setup_models.find_missing_kestrel_models", return_value=[]),
            patch("src.kestrel_bridge._init_kestrel_imports", return_value=(False, "CPU")),
            patch.dict(sys.modules, fake_modules),
        ):
            runner = KestrelCoreRunner(cfg)

            # 1) Wildlife detected
            img_rgb = _make_textured_rgb(200, 160, sharp=True)
            mask = np.zeros((160, 200), dtype=bool)
            mask[20:140, 40:160] = True
            fake_det_sam.get_prediction.return_value = (
                [mask],
                [((40, 20), (160, 140))],
                ["animal"],
                [0.96],
            )
            res_w, emb_w = runner.analyze_image(img_rgb, "bird.jpg")
            self.assertTrue(res_w["has_wildlife"])
            self.assertEqual(res_w["species"], "Eurasian Kestrel")
            self.assertEqual(emb_w.shape, (768,))
            self.assertEqual(emb_w.dtype, np.float16)

            # 2) No wildlife detected -> landscape fallback
            fake_det_sam.get_prediction.return_value = ([], [], [], [])
            res_l, emb_l = runner.analyze_image(img_rgb, "land.jpg")
            self.assertFalse(res_l["has_wildlife"])
            self.assertEqual(res_l["species"], "None")
            self.assertEqual(emb_l.shape, (768,))


if __name__ == "__main__":
    unittest.main()
