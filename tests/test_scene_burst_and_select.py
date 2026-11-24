import tempfile
import unittest
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np

from src.config import PipelineConfig
from src.scene_burst_and_select import (
    _apply_personal_ranker_if_available,
    _is_good,
    _is_long_action_sequence,
    _is_not_bad,
    _robust_percentile_0_100,
    assign_scenes_and_bursts,
    route_and_score_all,
    select_layers_final,
)


def _unit_vec(dim: int, idx: int) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[idx] = 1.0
    return v


class TestSceneBurstAndSelect(unittest.TestCase):
    def test_robust_percentile_0_100(self) -> None:
        self.assertEqual(_robust_percentile_0_100(np.array([], dtype=np.float32)).size, 0)
        np.testing.assert_allclose(_robust_percentile_0_100(np.array([5.0])), [75.0])
        np.testing.assert_allclose(_robust_percentile_0_100(np.array([3.0, 3.0, 3.0])), [50.0, 50.0, 50.0])
        res = _robust_percentile_0_100(np.array([0.0, 50.0, 100.0]), low_p=0.0, high_p=100.0)
        np.testing.assert_allclose(res, [0.0, 50.0, 100.0])

    def test_assign_scenes_and_bursts_zoom_time_and_sim(self) -> None:
        cfg = PipelineConfig()
        records = [
            {"subfolder": "s1", "timestamp": 100.0, "filename": "01.jpg", "focal_length": 400.0},
            {"subfolder": "s1", "timestamp": 100.5, "filename": "02.jpg", "focal_length": 400.0},  # same burst
            {"subfolder": "s1", "timestamp": 101.0, "filename": "03.jpg", "focal_length": 500.0},  # >=10% zoom -> split
            {"subfolder": "s1", "timestamp": 110.0, "filename": "04.jpg", "focal_length": 500.0},  # >4s gap -> split
            {"subfolder": "s2", "timestamp": 110.2, "filename": "05.jpg", "focal_length": 500.0},  # diff subfolder -> split
        ]
        embs = np.vstack([
            _unit_vec(8, 0),
            _unit_vec(8, 0),
            _unit_vec(8, 0),
            _unit_vec(8, 0),
            _unit_vec(8, 0),
        ])
        ordered = assign_scenes_and_bursts(records, embs, cfg)
        burst_ids = [r["burst_id"] for r in ordered]
        self.assertEqual(burst_ids, ["S0001", "S0001", "S0002", "S0003", "S0004"])
        self.assertEqual(ordered[0]["burst_size"], 2)
        self.assertEqual(ordered[1]["burst_index"], 2)
        self.assertEqual(ordered[2]["burst_size"], 1)

    def test_route_and_score_all_tracks_and_hybrid_bonus(self) -> None:
        cfg = PipelineConfig()
        records = [
            {
                "subfolder": "s1",
                "has_wildlife": True,
                "head_sharp_raw": 2.5,
                "kestrel_quality": 0.9,
                "siglip_eye_raw": 0.12,
                "siglip_action_raw": 0.15,
                "siglip_light_prob": 0.20,
                "siglip_scenic_raw": 0.05,
            },
            {
                "subfolder": "s1",
                "has_wildlife": True,
                "head_sharp_raw": 1.0,
                "kestrel_quality": 0.4,
                "siglip_eye_raw": -0.05,
                "siglip_action_raw": -0.05,
                "siglip_light_prob": 0.01,
                "siglip_scenic_raw": -0.05,
            },
            {
                "subfolder": "s1",
                "has_wildlife": False,
                "global_sharp_raw": 2.0,
                "tonal_balance": 0.8,
                "siglip_scenic_raw": 0.08,
                "blown_frac": 0.02,
            },
            {
                "subfolder": "s1",
                "has_wildlife": False,
                "global_sharp_raw": 0.5,
                "tonal_balance": 0.1,
                "siglip_scenic_raw": -0.05,
                "blown_frac": 0.50,
            },
        ]
        route_and_score_all(records, cfg)
        self.assertEqual(records[0]["track"], "Wildlife")
        self.assertTrue(records[0]["hybrid_bonus"])
        self.assertEqual(records[1]["track"], "Wildlife")
        self.assertFalse(records[1]["hybrid_bonus"])
        self.assertGreater(records[0]["overall_score"], records[1]["overall_score"])
        self.assertEqual(records[2]["track"], "Landscape")
        self.assertEqual(records[3]["track"], "Other")
        self.assertLessEqual(records[3]["overall_score"], 35.0)

    def test_not_bad_and_good_helpers(self) -> None:
        cfg = PipelineConfig()
        good_w = {"track": "Wildlife", "overall_score": 80.0, "sharpness_score": 70.0}
        self.assertTrue(_is_not_bad(good_w, cfg))
        self.assertTrue(_is_good(good_w, cfg))

        blurry_w = {"track": "Wildlife", "overall_score": 60.0, "sharpness_score": 30.0}
        self.assertFalse(_is_not_bad(blurry_w, cfg))

        good_l = {"track": "Landscape", "overall_score": 70.0, "scenic_score": 75.0, "tonal_balance": 0.5}
        self.assertTrue(_is_not_bad(good_l, cfg))
        self.assertTrue(_is_good(good_l, cfg))
        self.assertFalse(_is_not_bad({"track": "Other", "overall_score": 90.0}, cfg))

    def test_select_layers_final_single_static_action_and_artistic(self) -> None:
        cfg = PipelineConfig()
        records = [
            # Single good wildlife -> layer2_to_edit
            {
                "burst_id": "S0001", "track": "Wildlife", "overall_score": 85.0, "sharpness_score": 75.0,
                "action_score": 60.0, "scenic_score": 20.0, "kestrel_quality": 0.8, "tonal_balance": 0.5,
                "timestamp": 1.0,
            },
            # Single bad Other -> to_delete
            {
                "burst_id": "S0002", "track": "Other", "overall_score": 15.0, "sharpness_score": 10.0,
                "action_score": 0.0, "scenic_score": 20.0, "kestrel_quality": 0.0, "tonal_balance": 0.1,
                "exposure_seconds": 0.001, "timestamp": 2.0,
            },
            # Single slow-shutter artistic wildlife -> protected as 'rejected' instead of 'to_delete'
            {
                "burst_id": "S0003", "track": "Wildlife", "overall_score": 30.0, "sharpness_score": 25.0,
                "action_score": 55.0, "scenic_score": 30.0, "kestrel_quality": 0.3, "tonal_balance": 0.6,
                "exposure_seconds": 0.02, "timestamp": 3.0,
            },
        ]
        # Static group of 4 frames (keep cap = 2, edit = 1-2)
        for i in range(4):
            records.append({
                "burst_id": "S0004", "track": "Wildlife",
                "overall_score": 80.0 - i * 3.0, "sharpness_score": 70.0,
                "action_score": 40.0, "scenic_score": 20.0, "kestrel_quality": 0.8, "tonal_balance": 0.5,
                "timestamp": 10.0 + i * 0.5,
            })

        # Long action sequence of 6 flying frames (all high action) -> keep ALL 6 in layer1/layer2
        action_embs = []
        for i in range(6):
            records.append({
                "burst_id": "S0005", "track": "Wildlife",
                "overall_score": 82.0 if i < 5 else 40.0,
                "sharpness_score": 72.0 if i < 5 else 30.0,
                "action_score": 85.0, "scenic_score": 20.0, "kestrel_quality": 0.8, "tonal_balance": 0.5,
                "timestamp": 20.0 + i * 1.0,
                "_emb_idx": i,
            })
            action_embs.append(_unit_vec(8, i % 6))

        self.assertTrue(_is_long_action_sequence([r for r in records if r["burst_id"] == "S0005"]))

        select_layers_final(records, cfg, siglip_embs=np.vstack(action_embs))

        by_burst = {}
        for r in records:
            by_burst.setdefault(r["burst_id"], []).append(r)

        self.assertEqual(by_burst["S0001"][0]["layer"], "layer2_to_edit")
        self.assertEqual(by_burst["S0002"][0]["layer"], "to_delete")
        self.assertEqual(by_burst["S0003"][0]["layer"], "rejected")  # artistic blur protected

        s4_layers = [r["layer"] for r in by_burst["S0004"]]
        self.assertEqual(s4_layers.count("layer2_to_edit"), 2)
        self.assertEqual(s4_layers.count("rejected"), 2)

        s5_layers = [r["layer"] for r in by_burst["S0005"]]
        # Even the 6th frame (overall_score=40) is kept in layer1_keep because >=60% of action seq is not bad
        self.assertTrue(all(ly in ("layer1_keep", "layer2_to_edit") for ly in s5_layers))
        self.assertEqual(records[0]["overall_rank"] > 0, True)

    def test_apply_personal_ranker_if_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "personal_ranker.pkl"
            rng = np.random.default_rng(123)
            X = rng.normal(size=(12, 22)).astype(np.float32)  # 8 siglip + 8 bioclip + 6 tech
            y = np.array([1] * 6 + [0] * 6, dtype=np.int32)
            clf = lgb.LGBMClassifier(n_estimators=10, min_child_samples=2, verbose=-1, random_state=42)
            clf.fit(X, y)
            joblib.dump({"model": clf}, model_path)

            cfg = PipelineConfig(personal_ranker_path=model_path, personal_blend_weight=0.5)
            records = [
                {
                    "_emb_idx": 0,
                    "track": "Wildlife",
                    "overall_score": 60.0,
                    "sharpness_score": 70.0,
                    "eye_contact_score": 65.0,
                    "action_score": 60.0,
                    "scenic_score": 20.0,
                    "tonal_balance": 0.5,
                },
                {
                    "_emb_idx": 1,
                    "track": "Other",
                    "overall_score": 20.0,
                    "sharpness_score": 20.0,
                    "eye_contact_score": 0.0,
                    "action_score": 0.0,
                    "scenic_score": 10.0,
                    "tonal_balance": 0.2,
                },
            ]
            sig = rng.normal(size=(2, 8)).astype(np.float16)
            bio = rng.normal(size=(2, 8)).astype(np.float16)
            _apply_personal_ranker_if_available(records, bio, sig, cfg)
            self.assertIn("personal_score", records[0])
            self.assertNotIn("personal_score", records[1])  # Other track untouched


if __name__ == "__main__":
    unittest.main()
