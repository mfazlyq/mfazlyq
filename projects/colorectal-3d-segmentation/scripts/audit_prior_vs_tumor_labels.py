"""Measure tumor ground-truth coverage by colon priors at physical dilation margins."""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


def case_id(path: Path) -> str:
    name = path.name
    for suffix in (".nii.gz", ".nii"):
        if name.lower().endswith(suffix):
            return name[:-len(suffix)].removesuffix("_0000")
    return path.stem


def list_nifti(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise NotADirectoryError(f"Directory does not exist: {folder}")
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
    parser.add_argument(
        "--margins-mm",
        type=float,
        nargs="+",
        default=[0.0, 5.0, 10.0],
        help="Physical dilation margins to compare (default: 0 5 10 mm).",
    )
    args = parser.parse_args()

    if any(margin < 0 for margin in args.margins_mm):
        raise ValueError("Margins cannot be negative.")

    images = {case_id(path): path for path in list_nifti(args.images_dir)}
    labels = {case_id(path): path for path in list_nifti(args.labels_dir)}
    if not images or not labels:
        raise FileNotFoundError("No paired CT images/labels found.")
    if set(images) != set(labels):
        raise ValueError("CT/label case IDs do not match.")

    per_margin: dict[float, list[float]] = {m: [] for m in args.margins_mm}
    pooled_tumor: dict[float, int] = {m: 0 for m in args.margins_mm}
    pooled_covered: dict[float, int] = {m: 0 for m in args.margins_mm}
    missing: list[str] = []
    empty_gt: list[str] = []

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

        # NIfTI voxel sizes correspond to the three spatial array axes.
        spacing_mm = tuple(float(v) for v in image.header.get_zooms()[:3])
        distance_mm = ndimage.distance_transform_edt(~colon, sampling=spacing_mm)

        for margin in args.margins_mm:
            allowed_region = colon if margin == 0 else (colon | (distance_mm <= margin))
            covered = int(np.logical_and(tumor, allowed_region).sum())
            coverage = covered / tumor_voxels
            per_margin[margin].append(coverage)
            pooled_tumor[margin] += tumor_voxels
            pooled_covered[margin] += covered
            print(
                f"{identifier} | margin={margin:g} mm | "
                f"tumor_voxels={tumor_voxels:,} | "
                f"inside_prior={covered:,} | coverage={coverage:.4%} | "
                f"prior_voxels={int(allowed_region.sum()):,}"
            )

    print("\nSummary")
    print(f"Cases with tumor GT: {sum(bool(v) for v in per_margin.values()) if per_margin else 0}")
    print(f"Missing priors: {len(missing)} {missing[:10]}")
    print(f"Cases with empty tumor GT: {len(empty_gt)} {empty_gt[:10]}")
    for margin in args.margins_mm:
        values = per_margin[margin]
        if values:
            macro = float(np.mean(values))
            pooled = pooled_covered[margin] / pooled_tumor[margin]
            print(
                f"Margin {margin:g} mm | macro mean coverage={macro:.4%} | "
                f"pooled voxel coverage={pooled:.4%} | minimum case coverage={min(values):.4%}"
            )
    if missing:
        raise SystemExit("Some colon priors are missing.")
    print(
        "Coverage is a screening metric, not segmentation accuracy. "
        "Review masks visually, especially cases with low tumor coverage."
    )


if __name__ == "__main__":
    main()
