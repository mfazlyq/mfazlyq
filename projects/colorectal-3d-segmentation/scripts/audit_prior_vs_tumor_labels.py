"""Measure the fraction of annotated tumor voxels covered by each colon prior."""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def case_id(path: Path) -> str:
    name = path.name
    for suffix in (".nii.gz", ".nii"):
        if name.lower().endswith(suffix):
            return name[:-len(suffix)].removesuffix("_0000")
    return path.stem


def list_nifti(folder: Path) -> list[Path]:
    return sorted(
        item for item in folder.iterdir()
        if item.is_file() and not item.name.startswith("._")
        and item.name.lower().endswith((".nii.gz", ".nii"))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--priors-dir", type=Path, required=True)
    args = parser.parse_args()

    images = {case_id(path): path for path in list_nifti(args.images_dir)}
    labels = {case_id(path): path for path in list_nifti(args.labels_dir)}
    if not images or not labels:
        raise FileNotFoundError("No paired CT images/labels found.")
    if set(images) != set(labels):
        raise ValueError("CT/label case IDs do not match.")

    coverage_values = []
    missing = []
    empty_gt = []
    for identifier in sorted(images):
        prior_path = args.priors_dir / identifier / "colon.nii.gz"
        if not prior_path.is_file():
            missing.append(identifier)
            continue

        image = nib.load(str(images[identifier]))
        label = nib.load(str(labels[identifier]))
        prior = nib.load(str(prior_path))
        for role, candidate in (("label", label), ("prior", prior)):
            if candidate.shape[:3] != image.shape[:3] or not np.allclose(
                candidate.affine, image.affine, rtol=1e-4, atol=1e-3
            ):
                raise ValueError(f"{role} grid mismatch for case {identifier}")

        tumor = np.asanyarray(label.dataobj) == 1
        colon = np.asanyarray(prior.dataobj) > 0
        tumor_voxels = int(tumor.sum())
        if tumor_voxels == 0:
            empty_gt.append(identifier)
            continue
        covered = int(np.logical_and(tumor, colon).sum())
        coverage = covered / tumor_voxels
        coverage_values.append(coverage)
        print(
            f"{identifier}: tumor_voxels={tumor_voxels:,}, "
            f"inside_prior={covered:,}, coverage={coverage:.4%}"
        )

    print("\nSummary")
    print(f"Cases with tumor GT: {len(coverage_values)}")
    print(f"Missing priors: {len(missing)} {missing[:10]}")
    print(f"Cases with empty tumor GT: {len(empty_gt)} {empty_gt[:10]}")
    if coverage_values:
        print(f"Mean tumor coverage by colon prior: {np.mean(coverage_values):.4%}")
        print(f"Minimum tumor coverage: {np.min(coverage_values):.4%}")
    if missing:
        raise SystemExit("Some colon priors are missing.")
    print("Coverage is a screening metric; inspect low-coverage cases manually.")


if __name__ == "__main__":
    main()
