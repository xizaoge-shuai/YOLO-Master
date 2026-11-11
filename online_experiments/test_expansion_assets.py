"""CPU-only contracts for full official-asset preparation; no downloads or GPU."""

import importlib.util
import stat
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).with_name("prepare_expansion_assets.py")
SPEC = importlib.util.spec_from_file_location("expansion_assets_contract", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AssetContracts(unittest.TestCase):
    def test_pinned_backbone_identity(self):
        self.assertEqual(MODULE.MODEL_ID, "timm/vit_base_patch16_dinov3.lvd1689m")
        self.assertEqual(len(MODULE.REVISION), 40)

    def test_exact_official_archive_inventory(self):
        self.assertEqual(
            set(MODULE.COCO_FILES),
            {"train2017.zip", "val2017.zip", "annotations_trainval2017.zip"},
        )
        self.assertTrue(all("images.cocodataset.org/" in url for url in MODULE.COCO_FILES.values()))

    def test_reject_traversal_absolute_and_symlink_members(self):
        for name in ("../outside", "/outside", "a/../../outside", "a\\outside"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                MODULE.safe_member(name, 0)
        with self.assertRaises(ValueError):
            MODULE.safe_member("train2017/image.jpg", stat.S_IFLNK)

    def test_reject_existing_symlink_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "raw"
            outside = Path(directory) / "outside"
            root.mkdir()
            outside.mkdir()
            (root / "train2017").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                MODULE.contained_path(root, "train2017/image.jpg")

    def test_valid_image_and_annotation_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(
                MODULE.contained_path(root, "train2017/000000000001.jpg"),
                root / "train2017/000000000001.jpg",
            )
            self.assertEqual(
                MODULE.contained_path(root, "annotations/instances_val2017.json"),
                root / "annotations/instances_val2017.json",
            )

    def test_manifest_seal_detects_mutated_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "assets.json"
            MODULE.write(path, {"original_area": 17.25, "iscrowd": 1})
            expected = MODULE.sha(path)
            self.assertEqual(MODULE.read(path), {"original_area": 17.25, "iscrowd": 1})
            path.write_bytes(path.read_bytes() + b" ")
            self.assertNotEqual(expected, MODULE.sha(path))


if __name__ == "__main__":
    unittest.main()
