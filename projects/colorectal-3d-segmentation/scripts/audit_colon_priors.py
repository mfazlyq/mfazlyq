"""Check external TotalSegmentator colon priors against their source CT grids."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from generate_colon_priors import list_ct_volumes
from colorectal_seg.manifest import case_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--priors-dir", type=Path, required=True)
    args = parser.parse_args()

    images = list_ct_volumes(args.images_dir)
    if not images:
        raise FileNotFoundError(f"No CT NIfTI volumes found in {args.images_dir}")

    missing = []
    empty = []
    mismatched = []
    total_voxels = 0
    total_prior_voxels = 0

    for image_path in images:
        identifier = case_id(image_path)
        prior_path = args.priors_dir / identifier / "colon.nii.gz"
        if not prior_path.is_file():
            missing.append(identifier)
            continue

        image = nib.load(str(image_path))
        prior = nib.load(str(prior_path))
        if image.shape[:3] != prior.shape[:3] or not np.allclose(
            image.affine, prior.affine, rtol=1e-4, atol=1e-3
        ):
            mismatched.append(identifier)
            continue

        mask = np.asanyarray(prior.dataobj) > 0
        if not mask.any():
            empty.append(identifier)
        total_voxels += int(mask.size)
        total_prior_voxels += int(mask.sum())

    print(f"CT volumes: {len(images)}")
    print(f"Missing priors: {len(missing)} {missing[:10]}")
    print(f"Empty priors: {len(empty)} {empty[:10]}")
    print(f"Grid mismatches: {len(mismatched)} {mismatched[:10]}")
    if total_voxels:
        print(f"Mean prior occupancy: {100.0 * total_prior_voxels / total_voxels:.4f}%")
    if missing or empty or mismatched:
        raise SystemExit("Prior audit failed. Review the listed cases before training/evaluation.")
    print("Prior file/grid audit passed. Visual anatomical review is still required.")


if __name__ == "__main__":
    main()
