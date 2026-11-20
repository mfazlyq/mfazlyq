"""Generate one colon prior per Task10_Colon CT using TotalSegmentator.

Expected output: <output-dir>/<case_id>/colon.nii.gz
The generated priors are predictions, not ground-truth annotations.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys

import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from colorectal_seg.manifest import case_id


def list_ct_volumes(images_dir: Path) -> list[Path]:
    return sorted(
        path for path in images_dir.iterdir()
        if path.is_file()
        and not path.name.startswith("._")
        and path.name.lower().endswith((".nii", ".nii.gz"))
    )


def verify_geometry(image_path: Path, prior_path: Path) -> None:
    image = nib.load(str(image_path))
    prior = nib.load(str(prior_path))
    if image.shape[:3] != prior.shape[:3]:
        raise ValueError(
            f"Grid shape mismatch for {image_path.name}: "
            f"image={image.shape[:3]}, prior={prior.shape[:3]}"
        )
    if not np.allclose(image.affine, prior.affine, rtol=1e-4, atol=1e-3):
        raise ValueError(f"Image/prior affine mismatch for {image_path.name}")
    values = np.unique(np.asanyarray(prior.dataobj))
    if not np.any(values > 0):
        raise ValueError(f"Generated colon prior is empty for {image_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("gpu", "cpu", "gpu:0"), default="gpu")
    parser.add_argument("--fast", action="store_true", help="Use TotalSegmentator's lower-resolution fast model.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.images_dir.is_dir():
        raise NotADirectoryError(f"CT image directory does not exist: {args.images_dir}")
    if not shutil_which("TotalSegmentator"):
        raise RuntimeError("TotalSegmentator CLI not found. Install it with: pip install TotalSegmentator")

    image_files = list_ct_volumes(args.images_dir)
    if not image_files:
        raise FileNotFoundError(f"No NIfTI CT volumes found in {args.images_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(image_files)} CT volumes.")
    for index, image_path in enumerate(image_files, start=1):
        identifier = case_id(image_path)
        case_output = args.output_dir / identifier
        expected_prior = case_output / "colon.nii.gz"

        if expected_prior.is_file() and not args.overwrite:
            verify_geometry(image_path, expected_prior)
            print(f"[{index}/{len(image_files)}] {identifier}: existing prior verified; skip")
            continue

        case_output.mkdir(parents=True, exist_ok=True)
        command = [
            "TotalSegmentator",
            "-i", str(image_path),
            "-o", str(case_output),
            "--roi_subset", "colon",
            "--device", args.device,
        ]
        if args.fast:
            command.append("--fast")

        print(f"[{index}/{len(image_files)}] Generating colon prior: {identifier}")
        subprocess.run(command, check=True)

        if not expected_prior.is_file():
            available = sorted(path.name for path in case_output.iterdir())
            raise FileNotFoundError(
                f"Expected {expected_prior} was not produced. Output files: {available}"
            )
        verify_geometry(image_path, expected_prior)
        print(f"    Saved and geometry-checked: {expected_prior}")

    print(f"Finished. Colon priors are under: {args.output_dir.resolve()}")


def shutil_which(command: str) -> str | None:
    import shutil
    return shutil.which(command)


if __name__ == "__main__":
    main()
