"""Binary segmentation metrics and anatomical-prior coverage."""

from __future__ import annotations

import numpy as np

from .io import binarize_volume


def compute_binary_metrics(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    *,
    probability_threshold: float = 0.5,
) -> dict[str, float | int]:
    """Compute voxel-wise binary metrics for two aligned 3D volumes."""
    if np.shape(ground_truth) != np.shape(prediction):
        raise ValueError(
            f"Ground Truth and prediction shapes differ: "
            f"{np.shape(ground_truth)} vs {np.shape(prediction)}."
        )

    gt = binarize_volume(ground_truth, probability_threshold)
    pred = binarize_volume(prediction, probability_threshold)

    tp = int(np.count_nonzero(gt & pred))
    fp = int(np.count_nonzero(~gt & pred))
    fn = int(np.count_nonzero(gt & ~pred))
    tn = int(np.count_nonzero(~gt & ~pred))

    dice_denominator = 2 * tp + fp + fn
    iou_denominator = tp + fp + fn
    total = tp + fp + fn + tn

    dice = (2.0 * tp / dice_denominator) if dice_denominator else 1.0
    iou = (tp / iou_denominator) if iou_denominator else 1.0
    precision = (tp / (tp + fp)) if (tp + fp) else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) else 0.0
    accuracy = ((tp + tn) / total) if total else 0.0

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "ground_truth_voxels": int(np.count_nonzero(gt)),
        "prediction_voxels": int(np.count_nonzero(pred)),
    }


def compute_prior_coverage(
    ground_truth: np.ndarray,
    prior: np.ndarray,
    *,
    probability_threshold: float = 0.5,
) -> dict[str, float | int]:
    """Measure the fraction of GT foreground voxels contained in a prior."""
    if np.shape(ground_truth) != np.shape(prior):
        raise ValueError(
            f"Ground Truth and prior shapes differ: "
            f"{np.shape(ground_truth)} vs {np.shape(prior)}."
        )

    gt = binarize_volume(ground_truth, probability_threshold)
    prior_mask = binarize_volume(prior, probability_threshold)

    gt_count = int(np.count_nonzero(gt))
    covered_count = int(np.count_nonzero(gt & prior_mask))
    prior_count = int(np.count_nonzero(prior_mask))

    coverage = covered_count / gt_count if gt_count else 0.0
    return {
        "coverage": float(coverage),
        "covered_voxels": covered_count,
        "ground_truth_voxels": gt_count,
        "prior_voxels": prior_count,
        "missed_voxels": gt_count - covered_count,
    }
