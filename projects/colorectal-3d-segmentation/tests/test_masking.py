"""Tests for anatomical-prior-guided hierarchical masking."""

import unittest

import numpy as np

from colorectal_seg.masking import apply_hierarchical_anatomical_masking


class HierarchicalMaskingTests(unittest.TestCase):
    def test_prediction_is_restricted_to_colon_prior(self):
        probability = np.zeros((12, 12, 12), dtype=np.float32)
        prior = np.zeros_like(probability, dtype=np.uint8)
        prior[2:8, 2:8, 2:8] = 1
        probability[3:5, 3:5, 3:5] = 0.9
        probability[9:11, 9:11, 9:11] = 1.0  # stronger but outside the prior

        prediction, details = apply_hierarchical_anatomical_masking(probability, prior)

        self.assertEqual(details["status"], "ok")
        self.assertTrue(np.all(prediction[prior == 0] == 0))
        self.assertGreater(int(prediction.sum()), 0)

    def test_empty_prior_is_not_silently_accepted(self):
        probability = np.ones((4, 4, 4), dtype=np.float32) * 0.8
        prior = np.zeros((4, 4, 4), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "prior is empty"):
            apply_hierarchical_anatomical_masking(probability, prior)

    def test_margin_is_applied_in_physical_spacing(self):
        probability = np.zeros((9, 9, 9), dtype=np.float32)
        probability[4, 4, 4] = 0.9
        prior = np.zeros_like(probability, dtype=np.uint8)
        prior[4, 4, 5] = 1

        strict, _ = apply_hierarchical_anatomical_masking(probability, prior)
        expanded, _ = apply_hierarchical_anatomical_masking(
            probability, prior, prior_margin_mm=2.0, spacing_mm=(1.0, 1.0, 1.0)
        )
        self.assertEqual(int(strict.sum()), 0)
        self.assertGreater(int(expanded.sum()), 0)


if __name__ == "__main__":
    unittest.main()
