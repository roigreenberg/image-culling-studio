import tempfile
import unittest
from pathlib import Path

from webapp.xmp_sync import (
    get_xmp_path,
    normalize_label,
    normalize_rating,
    parse_xmp_text,
    read_xmp_sidecar,
    restore_xmp_sidecar,
    write_xmp_sidecar,
)


class TestXmpSync(unittest.TestCase):
    def test_normalize_label_and_rating(self) -> None:
        self.assertEqual(normalize_label("red"), "Red")
        self.assertEqual(normalize_label("GREEN"), "Green")
        self.assertEqual(normalize_label("7"), "Yellow")
        self.assertEqual(normalize_label("none"), "")
        self.assertEqual(normalize_label(None), "")
        self.assertEqual(normalize_label("UnknownColor"), "")

        self.assertIsNone(normalize_rating(None))
        self.assertIsNone(normalize_rating(""))
        self.assertIsNone(normalize_rating("abc"))
        self.assertEqual(normalize_rating(-3), 0)
        self.assertEqual(normalize_rating(9), 5)
        self.assertEqual(normalize_rating("4"), 4)

    def test_parse_xmp_text_variants(self) -> None:
        attr_xml = '<rdf:Description xmp:Rating="4" xmp:Label="Blue" />'
        self.assertEqual(parse_xmp_text(attr_xml), {"rating": 4, "label": "Blue"})

        elem_xml = "<rdf:Description><xmp:Rating>5</xmp:Rating><xmp:Label>Purple</xmp:Label></rdf:Description>"
        self.assertEqual(parse_xmp_text(elem_xml), {"rating": 5, "label": "Purple"})

        pm_xml = '<rdf:Description photomechanic:ColorClass="3" />'
        self.assertEqual(parse_xmp_text(pm_xml), {"rating": None, "label": "Green"})

    def test_write_read_preserve_and_restore_xmp_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "IMG_0001.CR3"
            img_path.write_bytes(b"raw")
            xmp_path = get_xmp_path(img_path)
            self.assertEqual(xmp_path.name, "IMG_0001.xmp")

            # 1) Read non-existent
            res0 = read_xmp_sidecar(img_path)
            self.assertFalse(res0["exists"])

            # 2) Create new XMP
            ok, prev, new_xml, err = write_xmp_sidecar(img_path, rating=5, label="Green")
            self.assertTrue(ok, err)
            self.assertIsNone(prev)
            self.assertIn('xmp:Rating="5"', new_xml)
            self.assertIn('xmp:Label="Green"', new_xml)

            res1 = read_xmp_sidecar(img_path)
            self.assertTrue(res1["exists"])
            self.assertEqual(res1["rating"], 5)
            self.assertEqual(res1["label"], "Green")

            # 3) Restore when prev was None -> deletes XMP
            ok_rest, err_rest = restore_xmp_sidecar(img_path, None)
            self.assertTrue(ok_rest, err_rest)
            self.assertFalse(xmp_path.exists())

            # 4) Existing Lightroom XMP with Develop settings must be preserved
            lr_xml = (
                '<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
                ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
                '  <rdf:Description rdf:about="" xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"\n'
                '   crs:Exposure2012="+0.50"\n'
                '   photomechanic:ColorClass="1"\n'
                '   xmp:Rating="2"\n'
                '   xmp:Label="Red">\n'
                '  </rdf:Description>\n'
                " </rdf:RDF>\n"
                "</x:xmpmeta>\n"
            )
            xmp_path.write_text(lr_xml, encoding="utf-8")

            ok2, prev2, new_xml2, err2 = write_xmp_sidecar(img_path, rating=4, label="Yellow")
            self.assertTrue(ok2, err2)
            self.assertEqual(prev2, lr_xml)
            self.assertIn('crs:Exposure2012="+0.50"', new_xml2)
            self.assertIn('xmp:Rating="4"', new_xml2)
            self.assertIn('xmp:Label="Yellow"', new_xml2)
            self.assertIn('photomechanic:ColorClass="2"', new_xml2)
            self.assertIn('xmlns:xmp="http://ns.adobe.com/xap/1.0/"', new_xml2)

            # Clear label
            ok3, _, new_xml3, _ = write_xmp_sidecar(img_path, rating=None, label="")
            self.assertTrue(ok3)
            self.assertNotIn("xmp:Label", new_xml3)
            self.assertIn('photomechanic:ColorClass="0"', new_xml3)

            # Restore to previous LR XML
            ok_rest2, _ = restore_xmp_sidecar(img_path, prev2)
            self.assertTrue(ok_rest2)
            self.assertEqual(xmp_path.read_text(encoding="utf-8"), lr_xml)


if __name__ == "__main__":
    unittest.main()
