"""Inspect unique label values in NIfTI segmentation masks.

Run locally or in Colab against the original labels directory. The script
reports class IDs and voxel counts; it does not upload or modify patient data.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np


def find_masks(directory: Path) -> list[Path]:
    files = list(directory.glob("*.nii")) + list(directory.glob("*.nii.gz"))
    return sorted(set(files))


def inspect_masks(directory: Path) -> list[dict[str, object]]:
    if not directory.is_dir():
        raise NotADirectoryError(f"Label directory does not exist: {directory}")

    mask_files = find_masks(directory)
    if not mask_files:
        raise FileNotFoundError(f"No .nii or .nii.gz masks found in: {directory}")

    rows: list[dict[str, object]] = []
    global_counts: Counter[str] = Counter()

    for mask_path in mask_files:
        image = nib.load(str(mask_path))
        array = np.asanyarray(image.dataobj)
        values, counts = np.unique(array, return_counts=True)
        value_counts = {
            str(value.item() if hasattr(value, "item") else value): int(count)
            for value, count in zip(values, counts)
        }
        global_counts.update(value_counts)
        rows.append(
            {
                "file": mask_path.name,
                "shape": "x".join(str(int(size)) for size in array.shape),
                "voxel_count": int(array.size),
                "unique_values": json.dumps(value_counts, sort_keys=True),
                "class_count": int(len(values)),
            }
        )

    print(f"Scanned masks: {len(rows)}")
    print("Global label-value voxel counts:")
    for value, count in sorted(global_counts.items(), key=lambda item: float(item[0])):
        print(f"  value {value}: {count:,} voxels")
    print("\nPer-mask summary:")
    for row in rows:
        print(f"  {row['file']}: {row['unique_values']}")

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit unique integer/class values in NIfTI segmentation masks."
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        required=True,
        help="Directory containing .nii or .nii.gz label masks.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV output path (keep it outside the public Git repository).",
    )
    args = parser.parse_args()

    rows = inspect_masks(args.labels_dir)
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV written to: {args.output_csv}")


if __name__ == "__main__":
    main()
