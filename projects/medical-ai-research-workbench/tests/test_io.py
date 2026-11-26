import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from medworkbench.io import load_nifti_bytes, validate_same_geometry


def create_nifti_bytes(data, affine=None, suffix=".nii.gz"):
    affine = np.eye(4) if affine is None else affine
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / f"volume{suffix}"
        nib.save(nib.Nifti1Image(data, affine), str(path))
        return path.read_bytes()


class NiftiIoTests(unittest.TestCase):
    def test_load_3d_nifti(self):
        data = np.zeros((8, 9, 10), dtype=np.int16)
        data[3:5, 3:5, 3:5] = 100
        payload = create_nifti_bytes(data)

        volume = load_nifti_bytes(payload, "demo.nii.gz")
        self.assertEqual(volume.shape, (8, 9, 10))
        self.assertEqual(volume.orientation, "RAS")
        self.assertEqual(volume.spacing_mm, (1.0, 1.0, 1.0))

    def test_reject_non_nifti_extension(self):
        with self.assertRaises(ValueError):
            load_nifti_bytes(b"not-a-volume", "scan.dcm")

    def test_reject_4d_volume(self):
        data = np.zeros((4, 4, 4, 2), dtype=np.float32)
        payload = create_nifti_bytes(data)
        with self.assertRaises(ValueError):
            load_nifti_bytes(payload, "timeseries.nii.gz")

    def test_geometry_mismatch_detected(self):
        data = np.zeros((5, 5, 5), dtype=np.uint8)
        reference = load_nifti_bytes(create_nifti_bytes(data), "ct.nii.gz")
        shifted_affine = np.eye(4)
        shifted_affine[0, 3] = 4.0
        candidate = load_nifti_bytes(
            create_nifti_bytes(data, shifted_affine),
            "mask.nii.gz",
        )

        with self.assertRaisesRegex(ValueError, "Affine mismatch"):
            validate_same_geometry(reference, candidate)

    def test_shape_mismatch_detected(self):
        reference_data = np.zeros((5, 5, 5), dtype=np.uint8)
        candidate_data = np.zeros((5, 5, 6), dtype=np.uint8)
        reference = load_nifti_bytes(create_nifti_bytes(reference_data), "ct.nii.gz")
        candidate = load_nifti_bytes(create_nifti_bytes(candidate_data), "mask.nii.gz")

        with self.assertRaisesRegex(ValueError, "Shape mismatch"):
            validate_same_geometry(reference, candidate)


if __name__ == "__main__":
    unittest.main()
