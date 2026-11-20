"""Hierarchical masking with an independently generated colon anatomical prior."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import ndimage


def apply_hierarchical_anatomical_masking(
    tumor_probability: np.ndarray,
    colon_prior: np.ndarray,
    *,
    threshold_multipliers: Sequence[float] = (0.5, 0.3, 0.1),
    prior_margin_mm: float = 0.0,
    spacing_mm: Sequence[float] = (1.0, 1.0, 1.0),
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Constrain tumor probabilities to a separately produced colon prior.

    The adaptive threshold follows the original notebook's 50/30/10 percent
    fallback sequence, with the maximum probability computed over the full
    volume. A candidate is then intersected with the colon prior (optionally
    dilated by a margin in millimeters), and only its largest 3D component is
    retained.

    An empty colon prior is a hard error: silently returning an empty tumor
    mask would hide a failed anatomical-prior stage and create false negatives.
    """
    probability = np.asarray(tumor_probability, dtype=np.float32)
    prior = np.asarray(colon_prior)

    if probability.ndim != 3 or prior.ndim != 3:
        raise ValueError("tumor_probability and colon_prior must be 3D arrays.")
    if probability.shape != prior.shape:
        raise ValueError(
            f"Shape mismatch: tumor probability {probability.shape}, colon prior {prior.shape}."
        )
    if not np.isfinite(probability).all():
        raise ValueError("tumor_probability contains NaN or infinite values.")
    if np.any(probability < 0.0) or np.any(probability > 1.0):
        raise ValueError("tumor_probability must contain values in [0, 1].")
    if prior_margin_mm < 0:
        raise ValueError("prior_margin_mm cannot be negative.")
    if not threshold_multipliers or any(
        multiplier <= 0 or multiplier > 1 for multiplier in threshold_multipliers
    ):
        raise ValueError("threshold_multipliers must contain values in (0, 1].")

    region = prior > 0
    if not region.any():
        raise ValueError("Colon anatomical prior is empty; inspect the external segmentation.")

    if prior_margin_mm > 0:
        spacing = np.asarray(spacing_mm, dtype=np.float64)
        if spacing.shape != (3,) or np.any(spacing <= 0):
            raise ValueError("spacing_mm must contain three positive voxel spacings.")
        distance_mm = ndimage.distance_transform_edt(~region, sampling=spacing)
        region = region | (distance_mm <= float(prior_margin_mm))

    max_confidence = float(probability.max())
    if max_confidence <= 0.0:
        return np.zeros(probability.shape, dtype=np.uint8), {
            "status": "no_tumor_confidence",
            "threshold": 0.0,
            "threshold_multiplier": 0.0,
            "prior_voxels": int(region.sum()),
            "prediction_voxels": 0,
        }

    structure = np.ones((3, 3, 3), dtype=np.uint8)
    for multiplier in threshold_multipliers:
        threshold = max_confidence * float(multiplier)
        candidate = (probability >= threshold) & region
        if not candidate.any():
            continue

        components, component_count = ndimage.label(candidate, structure=structure)
        if component_count > 1:
            sizes = np.bincount(components.ravel())
            sizes[0] = 0
            largest_component = int(sizes.argmax())
            candidate = components == largest_component

        prediction = candidate.astype(np.uint8)
        return prediction, {
            "status": "ok",
            "threshold": float(threshold),
            "threshold_multiplier": float(multiplier),
            "prior_voxels": int(region.sum()),
            "prediction_voxels": int(prediction.sum()),
        }

    return np.zeros(probability.shape, dtype=np.uint8), {
        "status": "no_candidate_inside_prior",
        "threshold": float(max_confidence * float(threshold_multipliers[-1])),
        "threshold_multiplier": float(threshold_multipliers[-1]),
        "prior_voxels": int(region.sum()),
        "prediction_voxels": 0,
    }


__all__ = ["apply_hierarchical_anatomical_masking"]
