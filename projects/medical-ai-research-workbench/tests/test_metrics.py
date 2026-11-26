import unittest

import numpy as np

from medworkbench.metrics import compute_binary_metrics, compute_prior_coverage


class BinaryMetricsTests(unittest.TestCase):
    def test_perfect_overlap(self):
        gt = np.zeros((4, 4, 4), dtype=np.uint8)
        gt[1:3, 1:3, 1:3] = 1
        result = compute_binary_metrics(gt, gt.copy())

        self.assertEqual(result["dice"], 1.0)
        self.assertEqual(result["iou"], 1.0)
        self.assertEqual(result["precision"], 1.0)
        self.assertEqual(result["recall"], 1.0)

    def test_partial_overlap_counts(self):
        gt = np.zeros((3, 3, 3), dtype=np.uint8)
        pred = np.zeros_like(gt)
        gt[0, 0, 0] = 1
        gt[0, 0, 1] = 1
        pred[0, 0, 0] = 1
        pred[1, 1, 1] = 1

        result = compute_binary_metrics(gt, pred)
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fp"], 1)
        self.assertEqual(result["fn"], 1)
        self.assertAlmostEqual(result["dice"], 0.5)
        self.assertAlmostEqual(result["iou"], 1 / 3)

    def test_empty_masks(self):
        empty = np.zeros((2, 2, 2), dtype=np.uint8)
        result = compute_binary_metrics(empty, empty)
        self.assertEqual(result["dice"], 1.0)
        self.assertEqual(result["iou"], 1.0)
        self.assertEqual(result["precision"], 0.0)
        self.assertEqual(result["recall"], 0.0)

    def test_probability_threshold(self):
        gt = np.zeros((2, 2, 2), dtype=np.uint8)
        gt[0, 0, 0] = 1
        probabilities = np.zeros((2, 2, 2), dtype=np.float32)
        probabilities[0, 0, 0] = 0.6
        probabilities[1, 1, 1] = 0.4

        result = compute_binary_metrics(gt, probabilities, probability_threshold=0.5)
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fp"], 0)
        self.assertEqual(result["fn"], 0)

    def test_prior_coverage(self):
        gt = np.zeros((3, 3, 3), dtype=np.uint8)
        prior = np.zeros_like(gt)
        gt[0, 0, 0] = 1
        gt[0, 0, 1] = 1
        prior[0, 0, 0] = 1

        result = compute_prior_coverage(gt, prior)
        self.assertAlmostEqual(result["coverage"], 0.5)
        self.assertEqual(result["covered_voxels"], 1)
        self.assertEqual(result["missed_voxels"], 1)

    def test_shape_mismatch_fails(self):
        with self.assertRaises(ValueError):
            compute_binary_metrics(np.zeros((2, 2, 2)), np.zeros((2, 2, 3)))


if __name__ == "__main__":
    unittest.main()
