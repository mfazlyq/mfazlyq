import tempfile
import unittest
from pathlib import Path

from colorectal_seg.manifest import (
    case_id,
    load_manifest,
    pair_image_and_label_files,
    split_pairs,
    write_manifest,
)


class ManifestTests(unittest.TestCase):
    def test_case_id_removes_nifti_and_channel_suffix(self):
        self.assertEqual(case_id(Path("colon_001_0000.nii.gz")), "colon_001")
        self.assertEqual(case_id(Path("colon_001.nii")), "colon_001")

    def test_pairs_nii_gz_images_with_nii_labels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images = root / "imagesTr"
            labels = root / "labelsTr"
            images.mkdir()
            labels.mkdir()
            (images / "case_001_0000.nii.gz").touch()
            (labels / "case_001.nii").touch()

            pairs = pair_image_and_label_files(images, labels)
            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0]["case_id"], "case_001")
            self.assertEqual(pairs[0]["image"], "imagesTr/case_001_0000.nii.gz")
            self.assertEqual(pairs[0]["label"], "labelsTr/case_001.nii")

    def test_raises_for_missing_label(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images = root / "imagesTr"
            labels = root / "labelsTr"
            images.mkdir()
            labels.mkdir()
            (images / "case_001.nii.gz").touch()
            with self.assertRaisesRegex(ValueError, "pairing is incomplete"):
                pair_image_and_label_files(images, labels)

    def test_split_is_reproducible_and_disjoint(self):
        pairs = [{"case_id": f"case_{i:03d}", "image": f"i{i}", "label": f"l{i}"} for i in range(10)]
        first = split_pairs(pairs, validation_fraction=0.2, seed=17)
        second = split_pairs(pairs, validation_fraction=0.2, seed=17)
        self.assertEqual(first, second)
        train_ids = {item["case_id"] for item in first["train"]}
        val_ids = {item["case_id"] for item in first["validation"]}
        self.assertFalse(train_ids.intersection(val_ids))
        self.assertEqual(len(val_ids), 2)

    def test_manifest_load_checks_paths_and_split_leakage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "images").mkdir()
            (root / "labels").mkdir()
            (root / "images" / "i.nii.gz").touch()
            (root / "labels" / "l.nii.gz").touch()
            payload = {
                "train": [{"case_id": "A", "image": "images/i.nii.gz", "label": "labels/l.nii.gz"}],
                "validation": [{"case_id": "A", "image": "images/i.nii.gz", "label": "labels/l.nii.gz"}],
            }
            manifest_path = write_manifest(payload, root / "manifest.json")
            with self.assertRaisesRegex(ValueError, "leakage"):
                load_manifest(manifest_path, root)


if __name__ == "__main__":
    unittest.main()
