"""Create reproducible train/validation JSON from paired NIfTI files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from colorectal_seg.manifest import pair_image_and_label_files, split_pairs, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--images-dir", default="imagesTr")
    parser.add_argument("--labels-dir", default="labelsTr")
    parser.add_argument("--output", type=Path, default=Path("data/manifest.json"))
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--colon-priors-dir",
        type=Path,
        default=None,
        help="Optional directory with <case_id>/colon.nii.gz outputs from TotalSegmentator.",
    )
    args = parser.parse_args()

    root = args.data_root.expanduser().resolve()
    pairs = pair_image_and_label_files(
        root / args.images_dir,
        root / args.labels_dir,
        colon_priors_dir=args.colon_priors_dir,
    )
    manifest = split_pairs(pairs, args.validation_fraction, args.seed)
    destination = write_manifest(manifest, args.output)

    print(f"Paired cases: {len(pairs)}")
    print(f"Train cases: {len(manifest['train'])}")
    print(f"Validation cases: {len(manifest['validation'])}")
    print(f"Manifest saved to: {destination.resolve()}")


if __name__ == "__main__":
    main()
