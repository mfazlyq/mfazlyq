"""Create visual CT/tumor/colon-prior overlays for manual anatomical QC."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import nibabel as nib
import numpy as np
from scipy import ndimage


def find_case(folder: Path, case: str) -> Path:
    for suffix in (".nii.gz", ".nii"):
        candidate = folder / f"{case}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No NIfTI file found for {case} in {folder}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--priors-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/prior_visual_qc"))
    parser.add_argument("--cases", nargs="+", default=["colon_001", "colon_005", "colon_006"])
    parser.add_argument("--window-min", type=float, default=-200)
    parser.add_argument("--window-max", type=float, default=200)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for case in args.cases:
        image_path = find_case(args.images_dir, case)
        label_path = find_case(args.labels_dir, case)
        prior_path = args.priors_dir / case / "colon.nii.gz"
        if not prior_path.is_file():
            raise FileNotFoundError(f"Colon prior not found: {prior_path}")

        image = nib.load(str(image_path))
        label = nib.load(str(label_path))
        prior = nib.load(str(prior_path))
        for name, other in (("label", label), ("prior", prior)):
            if other.shape[:3] != image.shape[:3] or not np.allclose(
                other.affine, image.affine, rtol=1e-4, atol=1e-3
            ):
                raise ValueError(f"{case}: {name} grid does not match CT.")

        volume = np.asarray(image.dataobj, dtype=np.float32)
        tumor = np.asarray(label.dataobj) == 1
        colon = np.asarray(prior.dataobj) > 0
        if not tumor.any():
            raise ValueError(f"{case}: tumor ground-truth mask is empty.")
        if not colon.any():
            raise ValueError(f"{case}: colon prior is empty.")

        spacing = tuple(float(v) for v in image.header.get_zooms()[:3])
        distance = ndimage.distance_transform_edt(~colon, sampling=spacing)
        colon_5mm = colon | (distance <= 5.0)
        colon_10mm = colon | (distance <= 10.0)

        tumor_voxels = int(tumor.sum())
        coverage = {
            0: np.logical_and(tumor, colon).sum() / tumor_voxels * 100,
            5: np.logical_and(tumor, colon_5mm).sum() / tumor_voxels * 100,
            10: np.logical_and(tumor, colon_10mm).sum() / tumor_voxels * 100,
        }

        slice_counts = tumor.sum(axis=(0, 1))
        z_index = int(np.argmax(slice_counts))
        ct_slice = np.rot90(volume[:, :, z_index])
        tumor_slice = np.rot90(tumor[:, :, z_index])
        prior_slices = [
            np.rot90(colon[:, :, z_index]),
            np.rot90(colon_5mm[:, :, z_index]),
            np.rot90(colon_10mm[:, :, z_index]),
        ]

        fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
        panels = [
            ("Tumor GT", None, None),
            ("Colon prior 0 mm", prior_slices[0], 0),
            ("Colon prior +5 mm", prior_slices[1], 5),
            ("Colon prior +10 mm", prior_slices[2], 10),
        ]

        for axis, (title, prior_slice, margin) in zip(axes, panels):
            axis.imshow(
                ct_slice,
                cmap="gray",
                vmin=args.window_min,
                vmax=args.window_max,
                origin="lower",
            )
            if tumor_slice.any():
                axis.contour(
                    tumor_slice.astype(np.uint8),
                    levels=[0.5],
                    colors=["red"],
                    linewidths=1.2,
                    origin="lower",
                )
            if prior_slice is not None and prior_slice.any():
                axis.contour(
                    prior_slice.astype(np.uint8),
                    levels=[0.5],
                    colors=["cyan"],
                    linewidths=1.0,
                    origin="lower",
                )
            if margin is None:
                subtitle = f"axial slice {z_index}"
            else:
                subtitle = f"tumor coverage: {coverage[margin]:.2f}%"
            axis.set_title(f"{title}\n{subtitle}")
            axis.axis("off")

        legend = [
            Line2D([0], [0], color="red", lw=2, label="Tumor ground truth"),
            Line2D([0], [0], color="cyan", lw=2, label="Colon prior boundary"),
        ]
        fig.legend(handles=legend, loc="lower center", ncol=2)
        fig.suptitle(
            f"{case} — CT slice with maximum tumor voxels "
            f"(displayed in native voxel grid)",
            fontsize=13,
        )
        output_path = args.output_dir / f"{case}_prior_review.png"
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(
            f"{case}: axial slice={z_index}; "
            f"coverage 0/5/10 mm={coverage[0]:.2f}%/"
            f"{coverage[5]:.2f}%/{coverage[10]:.2f}%"
        )
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
