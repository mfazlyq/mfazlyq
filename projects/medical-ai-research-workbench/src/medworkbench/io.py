"""NIfTI input helpers with explicit geometry validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile

import nibabel as nib
import numpy as np


@dataclass(frozen=True)
class NiftiVolume:
    """A canonicalized 3D NIfTI volume and its spatial metadata."""

    name: str
    data: np.ndarray
    affine: np.ndarray
    spacing_mm: tuple[float, float, float]
    orientation: str

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(size) for size in self.data.shape)


def load_nifti_bytes(file_bytes: bytes, filename: str) -> NiftiVolume:
    """Load a .nii or .nii.gz byte stream into closest canonical orientation.

    The temporary file is removed even if nibabel fails to parse the upload.
    Only 3D scalar volumes are supported in this first release.
    """
    lower_name = filename.lower()
    if not lower_name.endswith((".nii", ".nii.gz")):
        raise ValueError("File must have a .nii or .nii.gz extension.")
    if not file_bytes:
        raise ValueError("Uploaded file is empty.")
    if len(file_bytes) > 200 * 1024 * 1024:
        raise ValueError("File exceeds the 200 MB per-file upload limit.")

    suffix = ".nii.gz" if lower_name.endswith(".nii.gz") else ".nii"
    temp_path: str | None = None

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name

        image = nib.load(temp_path)
        canonical = nib.as_closest_canonical(image)

        if len(canonical.shape) != 3:
            raise ValueError(
                f"Only 3D scalar NIfTI volumes are supported; received shape {canonical.shape}."
            )
        voxel_count = int(np.prod(canonical.shape, dtype=np.int64))
        if voxel_count > 60_000_000:
            raise ValueError(
                f"Volume has {voxel_count:,} voxels; the current local prototype limit is "
                "60 million voxels per file to reduce memory exhaustion risk."
            )

        data = np.asarray(canonical.dataobj, dtype=np.float32)
        if data.ndim != 3:
            raise ValueError("NIfTI data could not be interpreted as a 3D volume.")
        if not np.isfinite(data).all():
            raise ValueError("Volume contains NaN or infinite values.")
        if any(size < 2 for size in data.shape):
            raise ValueError(f"Volume dimensions are too small: {data.shape}.")

        affine = np.asarray(canonical.affine, dtype=np.float64).copy()
        spacing = tuple(float(value) for value in canonical.header.get_zooms()[:3])
        orientation = "".join(nib.aff2axcodes(affine))

        if len(spacing) != 3 or any(value <= 0 for value in spacing):
            raise ValueError("Invalid voxel spacing in NIfTI header.")

        return NiftiVolume(
            name=Path(filename).name,
            data=data,
            affine=affine,
            spacing_mm=spacing,
            orientation=orientation,
        )
    except ValueError:
        raise
    except Exception as exc:  # nibabel exposes several format-specific exceptions
        raise ValueError(f"Could not read NIfTI file '{filename}': {exc}") from exc
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)


def validate_same_geometry(
    reference: NiftiVolume,
    candidate: NiftiVolume,
    *,
    affine_atol: float = 1e-3,
) -> None:
    """Raise ValueError unless shape and voxel-to-world affine are compatible."""
    if reference.shape != candidate.shape:
        raise ValueError(
            f"Shape mismatch: CT {reference.shape} vs {candidate.name} {candidate.shape}. "
            "Resample or align the files in a documented preprocessing step."
        )

    if not np.allclose(reference.affine, candidate.affine, rtol=0.0, atol=affine_atol):
        max_delta = float(np.max(np.abs(reference.affine - candidate.affine)))
        raise ValueError(
            f"Affine mismatch for {candidate.name} (maximum absolute difference "
            f"{max_delta:.6g}). The app will not silently resample or register this file."
        )


def binarize_volume(data: np.ndarray, probability_threshold: float = 0.5) -> np.ndarray:
    """Convert a binary label map or [0, 1] probability map to a boolean mask.

    Integer/multiclass label maps use non-zero as foreground. Values fully within
    [0, 1] use the supplied threshold.
    """
    array = np.asarray(data)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got shape {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError("Mask contains NaN or infinite values.")
    if not 0.0 <= probability_threshold <= 1.0:
        raise ValueError("Probability threshold must be between 0 and 1.")

    if array.size == 0:
        return np.zeros(array.shape, dtype=bool)

    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if minimum >= 0.0 and maximum <= 1.0:
        return array >= probability_threshold
    return array != 0
