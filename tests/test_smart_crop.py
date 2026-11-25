import math
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.smart_crop import clamp_crop_to_image, compute_smart_crop, detect_horizons
from webapp.db import CullDatabase
from webapp.xmp_sync import parse_xmp_has_crop, write_xmp_sidecar


def _make_tilted_horizon_image(angle_deg: float = 2.5, width: int = 800, height: int = 533) -> np.ndarray:
    """Create a synthetic sky-over-water landscape image with a crisp tilted horizon."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    slope = math.tan(math.radians(angle_deg))
    y0 = height * 0.55
    xs = np.arange(width, dtype=np.float64)
    ys = y0 + slope * (xs - width * 0.5)
    for x in range(width):
        split_y = int(max(1, min(height - 1, round(ys[x]))))
        # Bright warm sky above, dark blue-gray water below
        img[:split_y, x] = (230, 210, 190)
        img[split_y:, x] = (65, 50, 35)
    return img


class TestSmartCrop(unittest.TestCase):
    def test_clamp_crop_to_image_keeps_rotated_corners_inside(self) -> None:
        left, top, right, bottom = clamp_crop_to_image(
            0.0, 0.0, 1.0, 1.0, angle_deg=5.0, aspect_w_over_h=1.5, locked_ratio=1.5
        )
        # Verify locked aspect ratio preserved
        w = right - left
        h = bottom - top
        self.assertAlmostEqual((w * 1.5) / h, 1.5, places=3)
        # Verify all 4 rotated corners sit inside [0, 1] x [0, 1]
        cx = 0.5 * (left + right)
        cy = 0.5 * (top + bottom)
        rad = math.radians(5.0)
        c, s = math.cos(rad), math.sin(rad)
        for sx in (-0.5, 0.5):
            for sy in (-0.5, 0.5):
                dx = sx * w * 1.5
                dy = sy * h
                rx = (cx * 1.5 + dx * c - dy * s) / 1.5
                ry = cy + dx * s + dy * c
                self.assertGreaterEqual(rx, -1e-4)
                self.assertLessEqual(rx, 1.0 + 1e-4)
                self.assertGreaterEqual(ry, -1e-4)
                self.assertLessEqual(ry, 1.0 + 1e-4)

    def test_detect_horizons_finds_tilted_horizon(self) -> None:
        img = _make_tilted_horizon_image(angle_deg=2.5)
        horizons, sel_idx = detect_horizons(img)
        self.assertGreaterEqual(len(horizons), 1)
        self.assertEqual(sel_idx, 0)
        self.assertAlmostEqual(horizons[0]["angle"], 2.5, delta=0.5)

    def test_small_bird_enforces_2700px_minimum_long_side(self) -> None:
        # 45MP Canon R5 sensor (8192x5464) with a tiny bird (5% of frame)
        img = np.full((400, 600, 3), 120, dtype=np.uint8)
        res = compute_smart_crop(
            preview_bgr=img,
            bbox=[280, 180, 310, 210],  # 5% wide bird
            head_bbox=[282, 182, 294, 194],
            preview_width=600,
            preview_height=400,
            orig_width=8192,
            orig_height=5464,
            track="Wildlife",
            min_long_px=2700,
            aspect_locked=True,
        )
        self.assertGreaterEqual(max(res["crop_width_px"], res["crop_height_px"]), 2700)
        self.assertTrue(res["aspect_locked"])
        self.assertFalse(res["applied"])
        # Aspect ratio stays locked to 8192/5464 (~1.5)
        self.assertAlmostEqual(res["crop_width_px"] / res["crop_height_px"], 8192 / 5464, delta=0.02)

    def test_large_bird_does_not_clip_subject(self) -> None:
        img = np.full((400, 600, 3), 90, dtype=np.uint8)
        res = compute_smart_crop(
            preview_bgr=img,
            bbox=[100, 60, 500, 340],  # large bird
            head_bbox=[260, 70, 340, 140],
            preview_width=600,
            preview_height=400,
            orig_width=8192,
            orig_height=5464,
            track="Wildlife",
            min_long_px=2700,
        )
        # Entire bird bbox (normalized [0.167, 0.15, 0.833, 0.85]) stays inside crop
        self.assertLessEqual(res["left"], 100 / 600.0 + 1e-4)
        self.assertGreaterEqual(res["right"], 500 / 600.0 - 1e-4)
        self.assertLessEqual(res["top"], 60 / 400.0 + 1e-4)
        self.assertGreaterEqual(res["bottom"], 340 / 400.0 - 1e-4)

    def test_xmp_crop_write_preserves_develop_and_clears_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "RG5_0001.CR3"
            img_path.write_bytes(b"fake-raw")
            xmp_path = img_path.with_suffix(".xmp")
            xmp_path.write_text(
                '<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
                ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
                '  <rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/" '
                'xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/" '
                'crs:Exposure2012="+0.50" xmp:Rating="4"/>\n'
                " </rdf:RDF>\n"
                "</x:xmpmeta>\n",
                encoding="utf-8",
            )

            ok, prev_xml, new_xml, err = write_xmp_sidecar(
                img_path,
                crop={"applied": True, "left": 0.1, "top": 0.15, "right": 0.9, "bottom": 0.85, "angle": -1.75},
            )
            self.assertTrue(ok, err)
            self.assertIn('crs:Exposure2012="+0.50"', new_xml)
            self.assertIn('xmp:Rating="4"', new_xml)
            self.assertIn('crs:HasCrop="True"', new_xml)
            self.assertIn('crs:CropLeft="0.100000"', new_xml)
            self.assertIn('crs:CropAngle="-1.7500"', new_xml)
            self.assertTrue(parse_xmp_has_crop(new_xml))

            # Now reset crop
            ok2, _, cleared_xml, err2 = write_xmp_sidecar(img_path, crop={"applied": False})
            self.assertTrue(ok2, err2)
            self.assertIn('crs:HasCrop="False"', cleared_xml)
            self.assertIn('crs:Exposure2012="+0.50"', cleared_xml)
            self.assertFalse(parse_xmp_has_crop(cleared_xml))

    def test_db_save_and_undo_crop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = CullDatabase(Path(tmp) / "test.db")
            src_dir = Path(tmp) / "photos"
            src_dir.mkdir()
            out_dir = Path(tmp) / "out"
            out_dir.mkdir()
            d = db.register_directory(src_dir, out_dir)
            img_file = src_dir / "IMG_1.jpg"
            img_file.write_bytes(b"jpg")
            iid = f"{d['dir_id']}__root__IMG_1.jpg"
            db.upsert_scanned_images(
                d["dir_id"],
                [
                    {
                        "image_id": iid,
                        "source_path": str(img_file),
                        "subfolder": "root",
                        "filename": "IMG_1.jpg",
                        "stem": "IMG_1",
                        "file_ext": ".jpg",
                        "file_size": 3,
                    }
                ],
            )

            crop_obj = {
                "left": 0.1, "top": 0.1, "right": 0.9, "bottom": 0.9, "angle": 1.2,
                "aspect_locked": True, "aspect_ratio": 1.5, "orig_width": 6000, "orig_height": 4000,
                "crop_width_px": 4800, "crop_height_px": 3200, "applied": True, "horizons": [],
            }
            updated = db.save_image_crop(iid, crop_obj, record_change=True, action_type="smart_crop")
            self.assertTrue(updated["crop_applied"])
            self.assertEqual(updated["crop"]["angle"], 1.2)

            # Revert via change_log
            latest = db.get_latest_unreverted_change(d["dir_id"])
            self.assertIsNotNone(latest)
            self.assertEqual(latest["field"], "crop")
            db.mark_change_reverted(latest["id"], latest["field"], latest["old_value"], iid)
            reverted = db.get_image(iid)
            self.assertFalse(reverted["crop_applied"])
            self.assertIsNone(reverted["crop"])


if __name__ == "__main__":
    unittest.main()
