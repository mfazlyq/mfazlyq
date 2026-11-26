"""Generate synthetic 3D NIfTI files for local UI testing.

The volumes are geometric phantoms, not patient data and not clinical CT scans.
"""

from __future__ import annotations

from pathlib import Path
import sys

import nibabel as nib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "demo_data"


def ellipsoid(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    center: tuple[float, float, float],
    radii: tuple[float, float, float],
) -> np.ndarray:
    cx, cy, cz = center
    rx, ry, rz = radii
    return (
        ((x - cx) / rx) ** 2
        + ((y - cy) / ry) ** 2
        + ((z - cz) / rz) ** 2
    ) <= 1.0


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    nx, ny, nz = 96, 96, 64
    x, y, z = np.meshgrid(
        np.arange(nx, dtype=np.float32),
        np.arange(ny, dtype=np.float32),
        np.arange(nz, dtype=np.float32),
        indexing="ij",
    )

    body = ellipsoid(x, y, z, (48, 48, 32), (39, 43, 29))
    ct = np.full((nx, ny, nz), -950.0, dtype=np.float32)
    soft_texture = 28.0 + 12.0 * np.sin(x / 10.0) + 8.0 * np.cos(y / 13.0)
    ct[body] = soft_texture[body]

    liver = ellipsoid(x, y, z, (31, 39, 34), (13, 19, 17))
    spine = ellipsoid(x, y, z, (49, 65, 31), (8, 8, 10))
    kidney = ellipsoid(x, y, z, (68, 37, 31), (8, 12, 9))
    ct[liver] = 105.0
    ct[spine] = 420.0
    ct[kidney] = 72.0

    # Synthetic tumor-like target and independent prior; these are arbitrary
    # geometric objects solely to exercise upload, overlay, and metric code.
    ground_truth = ellipsoid(x, y, z, (67, 50, 31), (7, 9, 6))
    colon_prior = ellipsoid(x, y, z, (67, 50, 31), (12, 14, 11))
    prediction = ellipsoid(x, y, z, (68, 50, 32), (7, 8, 6))
    false_positive = ellipsoid(x, y, z, (56, 45, 25), (2, 2, 2))
    prediction = prediction | false_positive

    affine = np.eye(4, dtype=np.float64)  # Synthetic 1 mm isotropic voxel grid.
    volumes = {
        "demo_ct.nii.gz": ct.astype(np.int16),
        "demo_ground_truth.nii.gz": ground_truth.astype(np.uint8),
        "demo_colon_prior.nii.gz": colon_prior.astype(np.uint8),
        "demo_prediction.nii.gz": prediction.astype(np.uint8),
    }

    for filename, data in volumes.items():
        output_path = OUTPUT_DIR / filename
        image = nib.Nifti1Image(data, affine)
        nib.save(image, str(output_path))
        print(f"Created {output_path.relative_to(PROJECT_ROOT)} · shape={data.shape}")

    print("\nSynthetic demo volumes created. They are not real CT images or patient data.")
    print("Upload them in the Streamlit app to exercise the viewer and metrics.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Could not generate demo data: {exc}", file=sys.stderr)
        raise
