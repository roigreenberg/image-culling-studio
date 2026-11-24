import os
import tempfile
import unittest
from pathlib import Path

from src.config import IMAGE_EXTENSIONS, RAW_EXTENSIONS, PipelineConfig, load_local_env


class TestConfig(unittest.TestCase):
    def test_extensions_membership(self) -> None:
        self.assertIn(".cr3", RAW_EXTENSIONS)
        self.assertIn(".nef", RAW_EXTENSIONS)
        self.assertIn(".arw", RAW_EXTENSIONS)
        self.assertNotIn(".jpg", RAW_EXTENSIONS)
        self.assertIn(".jpg", IMAGE_EXTENSIONS)
        self.assertIn(".jpeg", IMAGE_EXTENSIONS)
        self.assertTrue(RAW_EXTENSIONS.issubset(IMAGE_EXTENSIONS))

    def test_load_local_env(self) -> None:
        key_new = "IMAGECULLING_TEST_VAR_NEW"
        key_existing = "IMAGECULLING_TEST_VAR_EXISTING"
        os.environ.pop(key_new, None)
        os.environ[key_existing] = "original"

        try:
            with tempfile.TemporaryDirectory() as tmp:
                env_file = Path(tmp) / ".env"
                env_file.write_text(
                    "\n".join(
                        [
                            "# Comment line",
                            "",
                            "INVALID_LINE_WITHOUT_EQUALS",
                            f'{key_new} = "hello_world"',
                            f"{key_existing}=overwritten",
                            "IMAGECULLING_TEST_SINGLE='single_quoted'",
                        ]
                    ),
                    encoding="utf-8",
                )
                load_local_env(env_file)
                self.assertEqual(os.environ.get(key_new), "hello_world")
                self.assertEqual(os.environ.get(key_existing), "original")
                self.assertEqual(os.environ.get("IMAGECULLING_TEST_SINGLE"), "single_quoted")
        finally:
            os.environ.pop(key_new, None)
            os.environ.pop(key_existing, None)
            os.environ.pop("IMAGECULLING_TEST_SINGLE", None)

    def test_load_local_env_missing_file(self) -> None:
        load_local_env(Path("nonexistent_env_file_12345.env"))

    def test_ensure_output_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "cull_out"
            cfg = PipelineConfig(output_dir=out_dir)
            cfg.ensure_output_dirs()
            expected = [
                out_dir,
                out_dir / "gallery_assets" / "thumbs",
                out_dir / "gallery_assets" / "crops",
                out_dir / "gallery_assets" / "siglip_crops",
                out_dir / "layer1_keep",
                out_dir / "layer2_to_edit",
            ]
            for d in expected:
                self.assertTrue(d.is_dir(), f"Expected directory {d} to exist")


if __name__ == "__main__":
    unittest.main()
