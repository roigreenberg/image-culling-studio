import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from src.config import PipelineConfig
from src.siglip2_scorer import SigLIP2Scorer, _extract_tensor, _load_pil_rgb_safe


class _FakeModel:
    def __init__(self, dim: int = 8):
        self.dim = dim
        self.logit_scale = torch.tensor(np.log(10.0), dtype=torch.float32)
        self.logit_bias = torch.tensor(-1.0, dtype=torch.float32)

    def to(self, _device):
        return self

    def eval(self):
        return self

    def get_text_features(self, **kwargs):
        n = kwargs["input_ids"].shape[0]
        feats = torch.zeros((n, self.dim), dtype=torch.float32)
        feats[:, 0] = 1.0
        return feats

    def get_image_features(self, pixel_values):
        n = pixel_values.shape[0]
        feats = torch.zeros((n, self.dim), dtype=torch.float32)
        feats[:, 0] = 1.0
        return feats


class _FakeProcessor:
    def __call__(self, text=None, images=None, **_kwargs):
        if text is not None:
            return {"input_ids": torch.zeros((len(text), 4), dtype=torch.long)}
        return {"pixel_values": torch.zeros((len(images), 3, 16, 16), dtype=torch.float32)}


class TestSigLIP2Scorer(unittest.TestCase):
    def test_extract_tensor_variants(self) -> None:
        t = torch.ones((2, 4))
        self.assertTrue(torch.equal(_extract_tensor(t), t))
        self.assertTrue(torch.equal(_extract_tensor(SimpleNamespace(pooler_output=t)), t))
        seq = torch.arange(12).reshape(2, 3, 2)
        self.assertTrue(torch.equal(_extract_tensor(SimpleNamespace(pooler_output=None, last_hidden_state=seq)), seq[:, 0]))
        with self.assertRaises(TypeError):
            _extract_tensor({"unexpected": True})

    def test_load_pil_rgb_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "test.png"
            Image.new("L", (16, 16), color=128).save(p)
            rgb = _load_pil_rgb_safe(p)
            self.assertEqual(rgb.mode, "RGB")
            self.assertEqual(rgb.size, (16, 16))

    def test_scorer_fallback_and_score_all_images(self) -> None:
        cfg = PipelineConfig(siglip2_batch_size=2)

        def _fake_from_pretrained(model_id, **_kwargs):
            if model_id == cfg.siglip2_model:
                raise RuntimeError("Simulated primary failure")
            return _FakeModel(dim=8)

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            (out_dir / "thumbs").mkdir()
            (out_dir / "scrops").mkdir()
            Image.new("RGB", (32, 32), color=(10, 20, 30)).save(out_dir / "thumbs" / "1.jpg")
            Image.new("RGB", (32, 32), color=(40, 50, 60)).save(out_dir / "thumbs" / "2.jpg")
            Image.new("RGB", (32, 32), color=(70, 80, 90)).save(out_dir / "scrops" / "1.jpg")

            records = [
                {
                    "image_id": "1",
                    "has_wildlife": True,
                    "thumb_rel_path": "thumbs/1.jpg",
                    "_siglip_crop_rel_path": "scrops/1.jpg",
                },
                {
                    "image_id": "2",
                    "has_wildlife": False,
                    "thumb_rel_path": "thumbs/2.jpg",
                },
            ]

            with (
                patch("src.siglip2_scorer.torch.cuda.is_available", return_value=False),
                patch("src.siglip2_scorer.AutoProcessor.from_pretrained", return_value=_FakeProcessor()),
                patch("src.siglip2_scorer.AutoModel.from_pretrained", side_effect=_fake_from_pretrained),
            ):
                scorer = SigLIP2Scorer(cfg)
                self.assertEqual(scorer.model_id, cfg.siglip2_fallback_model)
                embs = scorer.score_all_images(records, out_dir)
                scorer.close()

            self.assertEqual(embs.shape, (2, 8))
            self.assertEqual(embs.dtype, np.float16)
            for rec in records:
                self.assertIn("siglip_action_raw", rec)
                self.assertIn("siglip_scenic_raw", rec)
                self.assertIn("siglip_scenic_prob", rec)
                self.assertIn("siglip_light_prob", rec)
                self.assertIn("siglip_eye_raw", rec)
            self.assertEqual(records[1]["siglip_eye_raw"], 0.0)
            self.assertNotEqual(records[0]["siglip_eye_raw"], 0.0)


if __name__ == "__main__":
    unittest.main()
