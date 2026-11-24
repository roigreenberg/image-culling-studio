import json
import tempfile
import unittest
from pathlib import Path

from webapp.db import CullDatabase, make_dir_id, make_image_id


class TestCullDatabase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self._tmp.name)
        self.db = CullDatabase(self.root / "studio.db")
        self.src_dir = self.root / "photos"
        self.out_dir = self.root / "out"
        self.src_dir.mkdir()
        self.out_dir.mkdir()
        self.dir_rec = self.db.register_directory(self.src_dir, self.out_dir)
        self.dir_id = self.dir_rec["dir_id"]

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _seed_images(self) -> list[str]:
        items = []
        for idx, (sf, fn, fl, ts) in enumerate(
            [
                ("day1", "IMG_001.CR3", 500.0, 100.0),
                ("day1", "IMG_002.CR3", 500.0, 101.0),
                ("day2", "IMG_003.CR3", 200.0, 200.0),
            ]
        ):
            img_id = make_image_id(self.dir_id, sf, fn)
            items.append(
                {
                    "image_id": img_id,
                    "source_path": str(self.src_dir / sf / fn),
                    "subfolder": sf,
                    "filename": fn,
                    "stem": Path(fn).stem,
                    "file_ext": ".cr3",
                    "file_size": 1024 + idx,
                    "timestamp": ts,
                    "timestamp_str": f"2025-05-01 06:00:0{idx}",
                    "focal_length": fl,
                    "f_number": 5.6,
                    "exposure_time": "1/2000s",
                    "exposure_seconds": 0.0005,
                    "iso": 400,
                    "camera_model": "Canon EOS R5",
                    "lens_model": "RF100-500mm",
                    "camera_rating": 5 if idx == 0 else 0,
                    "camera_label": "Green" if idx == 0 else "",
                    "xmp_exists": False,
                    "xmp_rating": None,
                    "xmp_label": None,
                }
            )
        ins, upd = self.db.upsert_scanned_images(self.dir_id, items)
        self.assertEqual((ins, upd), (3, 0))

        # Re-upserting updates without duplicating
        ins2, upd2 = self.db.upsert_scanned_images(self.dir_id, items)
        self.assertEqual((ins2, upd2), (0, 3))
        return [it["image_id"] for it in items]

    def test_id_helpers_and_settings_session_logs(self) -> None:
        self.assertEqual(len(make_dir_id(str(self.src_dir))), 12)
        self.assertEqual(make_image_id("abc", "s1", "a.CR3", legacy=True), "s1__a.CR3")
        self.assertEqual(make_image_id("abc", "s1", "a.CR3", legacy=False), "abc__s1__a.CR3")

        s = self.db.get_settings()
        self.assertTrue(s["realtime_xmp_write"])
        s2 = self.db.update_settings({"grid_card_size": 320})
        self.assertEqual(s2["grid_card_size"], 320)

        self.db.save_session_state({"dir_id": self.dir_id, "tab": "layer2"})
        self.assertEqual(self.db.get_session_state()["tab"], "layer2")

        entry = self.db.add_event_log("info", "test", "hello", {"k": 1}, run_id="r1")
        self.assertEqual(entry["level"], "INFO")
        logs = self.db.get_event_logs(limit=10)
        self.assertGreaterEqual(len(logs), 1)
        self.assertEqual(logs[0]["message"], "hello")

    def test_ai_results_query_facets_and_burst_splits(self) -> None:
        ids = self._seed_images()
        self.db.update_ai_results(
            self.dir_id,
            "run_1",
            [
                {
                    "image_id": ids[0],
                    "source_path": str(self.src_dir / "day1" / "IMG_001.CR3"),
                    "layer": "layer2_to_edit",
                    "track": "Wildlife",
                    "species": "Eurasian Kestrel",
                    "species_conf": 0.94,
                    "burst_id": "S0001",
                    "burst_size": 2,
                    "burst_index": 1,
                    "burst_rank": 1,
                    "overall_score": 88.0,
                    "sharpness_score": 85.0,
                    "overall_rank": 1,
                },
                {
                    "image_id": ids[1],
                    "source_path": str(self.src_dir / "day1" / "IMG_002.CR3"),
                    "layer": "layer1_keep",
                    "track": "Wildlife",
                    "species": "Eurasian Kestrel",
                    "species_conf": 0.90,
                    "burst_id": "S0001",
                    "burst_size": 2,
                    "burst_index": 2,
                    "burst_rank": 2,
                    "overall_score": 70.0,
                    "sharpness_score": 72.0,
                    "overall_rank": 2,
                },
                {
                    "image_id": ids[2],
                    "source_path": str(self.src_dir / "day2" / "IMG_003.CR3"),
                    "layer": "rejected",
                    "track": "Landscape",
                    "species": "None",
                    "burst_id": "S0002",
                    "burst_size": 1,
                    "burst_index": 1,
                    "burst_rank": 1,
                    "overall_score": 40.0,
                    "sharpness_score": 50.0,
                    "overall_rank": 3,
                },
            ],
        )

        # Query filters
        res_all = self.db.query_images(self.dir_id)
        self.assertEqual(res_all["total"], 3)

        res_l2 = self.db.query_images(self.dir_id, ai_layer="layer2_to_edit", use_ai_layer=True)
        self.assertEqual(res_l2["total"], 1)
        self.assertEqual(res_l2["items"][0]["image_id"], ids[0])

        res_rated = self.db.query_images(self.dir_id, rating_val=5, rating_op="gte")
        self.assertEqual(res_rated["total"], 1)

        facets = self.db.get_filter_facets(self.dir_id)
        self.assertEqual(facets["tracks"].get("Wildlife"), 2)
        self.assertIn("RF100-500mm", facets["lenses"])

        # Burst split & unsplit
        split_res = self.db.split_burst_at(ids[1])
        self.assertTrue(split_res["ok"])
        img1 = self.db.get_image(ids[1])
        self.assertNotEqual(img1["burst_id"], "S0001")
        self.assertEqual(len(self.db.list_burst_splits(self.dir_id)), 1)

        unsplit_res = self.db.unsplit_burst(ids[1])
        self.assertTrue(unsplit_res["ok"])
        self.assertEqual(self.db.get_image(ids[1])["burst_id"], "S0001")

        # Siblings, position info, reapply splits, AF points, and search/sort filters
        siblings = self.db.get_burst_siblings(self.dir_id, "S0001")
        self.assertEqual(len(siblings), 2)
        pos_info = self.db.get_burst_position_info(self.dir_id, "S0001")
        self.assertGreaterEqual(pos_info["burst_pos"], 1)

        self.db.split_burst_at(ids[1], record_split=True)
        # Simulate AI overwrite restoring S0001, then reapply_manual_burst_splits
        self.db.update_ai_results(
            self.dir_id,
            "run_2",
            [{"image_id": ids[1], "source_path": str(self.src_dir / "day1" / "IMG_002.CR3"), "burst_id": "S0001", "layer": "layer1_keep"}],
        )
        reapplied = self.db.reapply_manual_burst_splits(self.dir_id)
        self.assertEqual(reapplied, 1)

        self.db.set_af_points(ids[0], [{"x": 0.4, "y": 0.4, "w": 0.1, "h": 0.1, "in_focus": True}])
        self.assertEqual(len(self.db.get_image(ids[0])["af_points"]), 1)

        q_search = self.db.query_images(self.dir_id, search="Kestrel", sort_by="overall_score", sort_dir="desc")
        self.assertEqual(q_search["total"], 1)

        bursts = self.db.query_bursts(self.dir_id)
        self.assertGreaterEqual(bursts["total"], 2)

    def test_metadata_change_undo_and_learning_stats(self) -> None:
        ids = self._seed_images()
        self.db.update_ai_results(
            self.dir_id,
            "run_1",
            [
                {"image_id": ids[0], "source_path": str(self.src_dir / "day1" / "IMG_001.CR3"), "layer": "layer2_to_edit", "track": "Wildlife", "overall_score": 85.0},
                {"image_id": ids[1], "source_path": str(self.src_dir / "day1" / "IMG_002.CR3"), "layer": "rejected", "track": "Wildlife", "overall_score": 30.0},
            ],
        )

        # Record user rating change with batch_id
        updated_img = self.db.record_metadata_change(
            ids[1], rating=1, label="Red", source="user", action_type="edit", xmp_synced=True, batch_id="b_1"
        )
        self.assertIsNotNone(updated_img)
        self.assertEqual(self.db.get_image(ids[1])["effective_rating"], 1)
        self.assertEqual(self.db.get_image(ids[1])["effective_label"], "Red")
        batch_entries = self.db.get_batch_changes("b_1")
        self.assertEqual(len(batch_entries), 2)
        self.assertGreaterEqual(len(self.db.get_sibling_unreverted_changes(batch_entries[0])), 1)

        # Learning stats & training picks
        stats = self.db.get_learning_stats(self.dir_id)
        self.assertIn("by_rating", stats)
        picks = self.db.get_training_picks(self.dir_id)
        self.assertEqual(len(picks), 2)  # ids[0] has camera_rating=5, ids[1] has user rating=1

        # Undo via mark_change_reverted
        latest = self.db.get_latest_unreverted_change(self.dir_id)
        self.assertIsNotNone(latest)
        self.db.mark_change_reverted(latest["id"], latest["field"], latest["old_value"], latest["image_id"])
        self.assertTrue(self.db.get_change_entry(latest["id"])["reverted"])

    def test_pipeline_runs_and_import_results_json(self) -> None:
        self.db.create_pipeline_run("r_001", self.dir_id, "full", 10)
        self.db.finish_pipeline_run("r_001", "done", 10, 0, {"kept": 4})
        runs = self.db.list_pipeline_runs()
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["status"], "done")

        # Import legacy results.json
        src_img = self.root / "legacy_src" / "sub1" / "IMG_999.CR3"
        src_img.parent.mkdir(parents=True)
        src_img.write_bytes(b"raw")
        rj = self.out_dir / "results.json"
        rj.write_text(
            json.dumps(
                [
                    {
                        "image_id": "sub1__IMG_999.CR3",
                        "source_path": str(src_img),
                        "subfolder": "sub1",
                        "filename": "IMG_999.CR3",
                        "stem": "IMG_999",
                        "layer": "layer1_keep",
                        "track": "Wildlife",
                        "overall_score": 77.0,
                        "overall_rank": 1,
                    }
                ]
            ),
            encoding="utf-8",
        )
        imported = self.db.import_existing_results_json(rj, self.out_dir, force=True)
        self.assertEqual(imported, 1)
        self.assertIsNotNone(self.db.get_image("sub1__IMG_999.CR3"))


if __name__ == "__main__":
    unittest.main()
