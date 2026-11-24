import io
import struct
import tempfile
import unittest
from pathlib import Path

from unittest.mock import MagicMock, patch

import numpy as np
import rawpy
from PIL import Image

from src.io_utils import (
    _CR3_META_UUID,
    _CR3_XMP_UUID,
    _extract_xmp_rating_label_bytes,
    _format_exposure,
    _format_rational_float,
    _iter_isobmff_boxes,
    _orient_norm_box,
    _parse_canon_af_info2,
    _parse_datetime_with_subsec,
    _parse_tiff_ifd,
    apply_orientation_bgr,
    compute_framing_histogram,
    discover_images,
    load_preview_bgr_and_meta,
    parse_cr3_exif_from_stream,
    parse_exposure_seconds,
    parse_jpeg_exif_from_bytes,
    parse_raw_exif_from_stream,
    read_fast_exif_and_camera_meta,
    save_jpeg_bgr,
)


def _build_tiff_ifd(entries, endian="<") -> bytes:
    """Build a minimal valid TIFF IFD blob for testing _parse_tiff_ifd."""
    header = (b"II" if endian == "<" else b"MM") + struct.pack(f"{endian}HI", 42, 8)
    num = len(entries)
    ifd_header = struct.pack(f"{endian}H", num)
    data_offset = 8 + 2 + num * 12 + 4
    ifd_records = bytearray()
    extra_data = bytearray()

    for tag_id, dtype, count, raw_val in entries:
        if dtype == 2:  # ASCII
            b = raw_val.encode("ascii") + b"\x00"
            count = len(b)
            payload = b
        elif dtype == 3:  # SHORT
            vals = raw_val if isinstance(raw_val, (tuple, list)) else (raw_val,)
            count = len(vals)
            payload = struct.pack(f"{endian}{count}H", *vals)
        elif dtype == 4:  # LONG
            vals = raw_val if isinstance(raw_val, (tuple, list)) else (raw_val,)
            count = len(vals)
            payload = struct.pack(f"{endian}{count}I", *vals)
        elif dtype == 5:  # RATIONAL
            vals = raw_val if isinstance(raw_val[0], (tuple, list)) else (raw_val,)
            count = len(vals)
            payload = b"".join(struct.pack(f"{endian}II", n, d) for n, d in vals)
        elif dtype == 10:  # SRATIONAL
            vals = raw_val if isinstance(raw_val[0], (tuple, list)) else (raw_val,)
            count = len(vals)
            payload = b"".join(struct.pack(f"{endian}ii", n, d) for n, d in vals)
        else:
            raise ValueError(f"Unsupported dtype {dtype}")

        if len(payload) <= 4:
            padded = payload.ljust(4, b"\x00")
            ifd_records.extend(struct.pack(f"{endian}HHII", tag_id, dtype, count, 0)[:8] + padded)
        else:
            off = data_offset + len(extra_data)
            ifd_records.extend(struct.pack(f"{endian}HHII", tag_id, dtype, count, off))
            extra_data.extend(payload)

    next_ifd = struct.pack(f"{endian}I", 0)
    return header + ifd_header + bytes(ifd_records) + next_ifd + bytes(extra_data)


def _box(btype: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", 8 + len(payload), btype) + payload


class TestIoUtils(unittest.TestCase):
    def test_discover_images_and_balanced_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "day1").mkdir()
            (root / "day2").mkdir()
            (root / "day1" / "a.jpg").write_bytes(b"x")
            (root / "day1" / "b.CR3").write_bytes(b"x")
            (root / "day1" / ".hidden.jpg").write_bytes(b"x")
            (root / "day1" / "notes.txt").write_bytes(b"x")
            (root / "day2" / "c.NEF").write_bytes(b"x")
            (root / "day2" / "d.jpeg").write_bytes(b"x")
            (root / "top.dng").write_bytes(b"x")

            all_imgs = discover_images(root)
            self.assertEqual(len(all_imgs), 5)
            subfolders = {p.name: sf for p, sf in all_imgs}
            self.assertEqual(subfolders["top.dng"], "root")
            self.assertEqual(subfolders["a.jpg"], "day1")
            self.assertEqual(subfolders["c.NEF"], "day2")

            limited = discover_images(root, limit=3)
            self.assertEqual(len(limited), 3)
            # Balanced across subfolders
            self.assertEqual(len({sf for _, sf in limited}), 3)

    def test_discover_images_missing_dir(self) -> None:
        with self.assertRaises(FileNotFoundError):
            discover_images(Path("nonexistent_dir_xyz_999"))

    def test_parse_tiff_ifd_little_and_big_endian(self) -> None:
        entries = [
            (0x0110, 2, 0, "Canon EOS R5"),
            (0x0112, 3, 0, 6),
            (0x8827, 3, 0, (400, 0)),
            (0x829A, 5, 0, (1, 2000)),
            (0x920A, 5, 0, (600, 1)),
            (0x9003, 2, 0, "2025:05:01 06:30:15"),
        ]
        for endian in ("<", ">"):
            blob = _build_tiff_ifd(entries, endian=endian)
            tags = _parse_tiff_ifd(blob)
            self.assertEqual(tags[0x0110], "Canon EOS R5")
            self.assertEqual(tags[0x0112], 6)
            self.assertEqual(tags[0x8827], (400, 0))
            self.assertEqual(tags[0x829A], (1, 2000))
            self.assertEqual(tags[0x920A], (600, 1))
            self.assertEqual(tags[0x9003], "2025:05:01 06:30:15")

        self.assertEqual(_parse_tiff_ifd(b"bad"), {})
        self.assertEqual(_parse_tiff_ifd(b"XX\x2a\x00\x08\x00\x00\x00"), {})

    def test_iter_isobmff_boxes(self) -> None:
        b1 = _box(b"ftyp", b"crx ")
        b2_payload = b"hello"
        b2 = struct.pack(">I4sQ", 1, b"uuid", 16 + len(b2_payload)) + b2_payload
        b3 = struct.pack(">I4s", 0, b"mdat") + b"tail"
        boxes = list(_iter_isobmff_boxes(b1 + b2 + b3))
        self.assertEqual([b[0] for b in boxes], ["ftyp", "uuid", "mdat"])
        self.assertEqual((b1 + b2 + b3)[boxes[1][1] : boxes[1][2]], b2_payload)
        self.assertEqual((b1 + b2 + b3)[boxes[2][1] : boxes[2][2]], b"tail")

    def test_datetime_and_exposure_helpers(self) -> None:
        ts, iso = _parse_datetime_with_subsec("2025:06:15 07:12:34", "45")
        self.assertIsNotNone(ts)
        self.assertEqual(iso, "2025-06-15 07:12:34.45")

        ts_bad, iso_bad = _parse_datetime_with_subsec("invalid", None)
        self.assertIsNone(ts_bad)
        self.assertEqual(iso_bad, "invalid")

        self.assertEqual(_format_exposure((1, 2000)), "1/2000s")
        self.assertEqual(_format_exposure((2, 1)), "2.0s")
        self.assertEqual(_format_exposure(0.0005), "1/2000s")
        self.assertEqual(_format_exposure((0, 1)), "")
        self.assertEqual(_format_rational_float((28, 10)), 2.8)
        self.assertAlmostEqual(parse_exposure_seconds("1/2000s"), 0.0005)
        self.assertAlmostEqual(parse_exposure_seconds("1.5s"), 1.5)
        self.assertIsNone(parse_exposure_seconds(""))
        self.assertIsNone(parse_exposure_seconds("1/0s"))

    def test_extract_xmp_rating_label_bytes(self) -> None:
        blob1 = b'<rdf:Description xmp:Rating="4" xmp:Label="Green" />'
        self.assertEqual(_extract_xmp_rating_label_bytes(blob1), (4, "Green"))

        blob2 = b"<xmp:Rating>9</xmp:Rating><xmp:Label> Yellow </xmp:Label>"
        self.assertEqual(_extract_xmp_rating_label_bytes(blob2), (5, "Yellow"))

        blob3 = b'<rdf:Description xmp:Rating="-1" />'
        self.assertEqual(_extract_xmp_rating_label_bytes(blob3), (0, None))

    def test_orient_norm_box_and_canon_af_info2(self) -> None:
        np.testing.assert_allclose(_orient_norm_box(0.1, 0.2, 0.3, 0.4, 1), (0.1, 0.2, 0.3, 0.4))
        np.testing.assert_allclose(_orient_norm_box(0.1, 0.2, 0.3, 0.4, 6), (0.4, 0.1, 0.4, 0.3))
        np.testing.assert_allclose(_orient_norm_box(0.1, 0.2, 0.3, 0.4, 8), (0.2, 0.6, 0.4, 0.3))
        np.testing.assert_allclose(_orient_norm_box(0.1, 0.2, 0.3, 0.4, 3), (0.6, 0.4, 0.3, 0.4))

        # Build synthetic AFInfo2 array with 2 points, point 0 in focus, point 1 selected
        vals = [
            0, 0,
            2,    # num_pts
            2,    # valid_pts
            1000, # af_w fallback
            1000, # af_h fallback
            1000, # af_w
            1000, # af_h
            100, 100,  # widths
            100, 100,  # heights
            0, 65436,  # xs_raw (0, -100)
            100, 0,    # ys_raw (100, 0)
            0b01,      # in_focus_mask (pt 0)
            0b10,      # selected_mask (pt 1)
        ]
        boxes = _parse_canon_af_info2(vals, orientation=1)
        self.assertEqual(len(boxes), 2)
        self.assertTrue(boxes[0]["in_focus"])
        self.assertFalse(boxes[0]["selected"])
        self.assertAlmostEqual(boxes[0]["cx"], 0.5, places=3)
        self.assertAlmostEqual(boxes[0]["cy"], 0.4, places=3)
        self.assertTrue(boxes[1]["selected"])

    def test_parse_cr3_exif_from_stream(self) -> None:
        cmt1 = _build_tiff_ifd([(0x0110, 2, 0, "Canon EOS R5"), (0x0112, 3, 0, 1)])
        cmt2 = _build_tiff_ifd(
            [
                (0x9003, 2, 0, "2025:05:10 05:45:00"),
                (0x9291, 2, 0, "12"),
                (0x829A, 5, 0, (1, 1600)),
                (0x829D, 5, 0, (56, 10)),
                (0x920A, 5, 0, (500, 1)),
                (0x8827, 3, 0, 800),
                (0xA434, 2, 0, "RF100-500mm F4.5-7.1 L IS USM"),
            ]
        )
        meta_uuid_payload = _CR3_META_UUID + _box(b"CMT1", cmt1) + _box(b"CMT2", cmt2)
        xmp_uuid_payload = _CR3_XMP_UUID + b'<x:xmpmeta><rdf:Description xmp:Rating="5" xmp:Label="Green"/></x:xmpmeta>'
        moov = _box(b"moov", _box(b"uuid", meta_uuid_payload) + _box(b"uuid", xmp_uuid_payload))
        cr3_bytes = _box(b"ftyp", b"crx ") + moov + _box(b"mdat", b"\x00" * 16)

        meta = parse_cr3_exif_from_stream(io.BytesIO(cr3_bytes))
        self.assertEqual(meta["camera_model"], "Canon EOS R5")
        self.assertEqual(meta["lens_model"], "RF100-500mm F4.5-7.1 L IS USM")
        self.assertEqual(meta["timestamp_str"], "2025-05-10 05:45:00.12")
        self.assertEqual(meta["exposure_time"], "1/1600s")
        self.assertAlmostEqual(meta["f_number"], 5.6)
        self.assertAlmostEqual(meta["focal_length"], 500.0)
        self.assertEqual(meta["iso"], 800)
        self.assertEqual(meta["camera_rating"], 5)
        self.assertEqual(meta["camera_label"], "Green")

    def test_parse_jpeg_exif_and_unicode_io_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            unicode_dir = Path(tmp) / "שמורה_2025"
            unicode_dir.mkdir()
            jpg_path = unicode_dir / "תמונה_01.jpg"

            img_bgr = np.zeros((120, 200, 3), dtype=np.uint8)
            img_bgr[:, :, 2] = 200  # red channel in BGR
            save_jpeg_bgr(img_bgr, jpg_path, quality=90, max_dim=100)

            loaded_bgr, meta = load_preview_bgr_and_meta(jpg_path)
            self.assertIsNotNone(loaded_bgr)
            self.assertEqual(loaded_bgr.shape[:2], (60, 100))
            self.assertEqual(meta["preview_width"], 100)
            self.assertEqual(meta["preview_height"], 60)

            fast_meta = read_fast_exif_and_camera_meta(jpg_path)
            self.assertGreater(fast_meta["file_size"], 0)
            self.assertIsNotNone(fast_meta["timestamp"])

            hist = compute_framing_histogram(loaded_bgr)
            self.assertEqual(hist.shape, (64,))
            self.assertEqual(hist.dtype, np.float32)
            self.assertAlmostEqual(float(np.linalg.norm(hist)), 1.0, places=4)

        # Test PIL EXIF parsing with synthetic JPEG bytes
        pil_img = Image.new("RGB", (40, 20), color=(10, 20, 30))
        exif = pil_img.getexif()
        exif[0x0110] = "TestCam"
        exif[0x0112] = 6
        exif[0x0132] = "2025:01:02 03:04:05"
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", exif=exif)
        parsed = parse_jpeg_exif_from_bytes(buf.getvalue())
        self.assertEqual(parsed["camera_model"], "TestCam")
        self.assertEqual(parsed["orientation"], 6)
        self.assertEqual(parsed["timestamp_str"], "2025-01-02 03:04:05")

    def test_apply_orientation_bgr(self) -> None:
        arr = np.zeros((10, 20, 3), dtype=np.uint8)
        self.assertEqual(apply_orientation_bgr(arr, 1).shape, (10, 20, 3))
        self.assertEqual(apply_orientation_bgr(arr, 3).shape, (10, 20, 3))
        self.assertEqual(apply_orientation_bgr(arr, 6).shape, (20, 10, 3))
        self.assertEqual(apply_orientation_bgr(arr, 8).shape, (20, 10, 3))
        self.assertEqual(apply_orientation_bgr(arr, 5).shape, (20, 10, 3))
        self.assertEqual(apply_orientation_bgr(arr, 7).shape, (20, 10, 3))

    def test_parse_raw_exif_from_stream_and_load_preview_raw_paths(self) -> None:
        # Mock exifread tags for parse_raw_exif_from_stream
        class _Ratio:
            def __init__(self, num, den):
                self.num = num
                self.den = den

        class _Tag:
            def __init__(self, values, text=""):
                self.values = values
                self._text = text

            def __str__(self) -> str:
                return self._text

        fake_tags = {
            "Image Orientation": _Tag([6]),
            "Image Model": _Tag([], "Nikon Z9"),
            "EXIF LensModel": _Tag([], "NIKKOR Z 400mm"),
            "EXIF DateTimeOriginal": _Tag([], "2025:07:01 08:15:30"),
            "EXIF SubSecTimeOriginal": _Tag([], "25"),
            "EXIF ExposureTime": _Tag([_Ratio(1, 1000)]),
            "EXIF FNumber": _Tag([_Ratio(45, 10)]),
            "EXIF FocalLength": _Tag([_Ratio(400, 1)]),
            "EXIF ISOSpeedRatings": _Tag([640]),
        }
        raw_stream = io.BytesIO(b"<x:xmpmeta><rdf:Description xmp:Rating=\"3\" xmp:Label=\"Blue\"/></x:xmpmeta>")
        with patch("exifread.process_file", return_value=fake_tags):
            meta = parse_raw_exif_from_stream(raw_stream)
        self.assertEqual(meta["camera_model"], "Nikon Z9")
        self.assertEqual(meta["lens_model"], "NIKKOR Z 400mm")
        self.assertEqual(meta["orientation"], 6)
        self.assertEqual(meta["exposure_time"], "1/1000s")
        self.assertAlmostEqual(meta["f_number"], 4.5)
        self.assertAlmostEqual(meta["focal_length"], 400.0)
        self.assertEqual(meta["iso"], 640)
        self.assertEqual(meta["camera_rating"], 3)
        self.assertEqual(meta["camera_label"], "Blue")

        # Mock rawpy.imread for JPEG thumb, BITMAP thumb, and postprocess fallback
        pil_img = Image.new("RGB", (60, 40), color=(20, 40, 60))
        jbuf = io.BytesIO()
        pil_img.save(jbuf, format="JPEG")
        jpeg_bytes = jbuf.getvalue()

        with tempfile.TemporaryDirectory() as tmp:
            nef_path = Path(tmp) / "test.nef"
            nef_path.write_bytes(b"fake_nef")

            # 1) JPEG thumb
            mock_raw_jpg = MagicMock()
            mock_raw_jpg.__enter__.return_value = mock_raw_jpg
            mock_raw_jpg.extract_thumb.return_value = MagicMock(format=rawpy.ThumbFormat.JPEG, data=jpeg_bytes)
            with (
                patch("src.io_utils.parse_raw_exif_from_stream", return_value={"orientation": 1}),
                patch("src.io_utils.rawpy.imread", return_value=mock_raw_jpg),
            ):
                bgr, rmeta = load_preview_bgr_and_meta(nef_path)
            self.assertIsNotNone(bgr)
            self.assertEqual((rmeta["preview_width"], rmeta["preview_height"]), (60, 40))

            # 2) BITMAP thumb
            mock_raw_bmp = MagicMock()
            mock_raw_bmp.__enter__.return_value = mock_raw_bmp
            mock_raw_bmp.extract_thumb.return_value = MagicMock(
                format=rawpy.ThumbFormat.BITMAP,
                data=np.zeros((30, 50, 3), dtype=np.uint8),
            )
            with (
                patch("src.io_utils.parse_raw_exif_from_stream", return_value={"orientation": 6}),
                patch("src.io_utils.rawpy.imread", return_value=mock_raw_bmp),
            ):
                bgr_bmp, _ = load_preview_bgr_and_meta(nef_path)
            self.assertEqual(bgr_bmp.shape[:2], (50, 30))

            # 3) No thumb -> raw.postprocess
            mock_raw_pp = MagicMock()
            mock_raw_pp.__enter__.return_value = mock_raw_pp
            mock_raw_pp.extract_thumb.side_effect = rawpy.LibRawNoThumbnailError("no thumb")
            mock_raw_pp.postprocess.return_value = np.zeros((24, 32, 3), dtype=np.uint8)
            with (
                patch("src.io_utils.parse_raw_exif_from_stream", return_value={"orientation": 1}),
                patch("src.io_utils.rawpy.imread", return_value=mock_raw_pp),
            ):
                bgr_pp, _ = load_preview_bgr_and_meta(nef_path)
            self.assertEqual(bgr_pp.shape[:2], (24, 32))


if __name__ == "__main__":
    unittest.main()
