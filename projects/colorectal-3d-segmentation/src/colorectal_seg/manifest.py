"""Dataset pairing and reproducible train/validation manifest utilities."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


NIFTI_SUFFIXES = (".nii.gz", ".nii")


def case_id(path: Path) -> str:
    """Return a case identifier, removing NIfTI and common CT channel suffixes."""
    name = path.name
    for suffix in NIFTI_SUFFIXES:
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    if name.endswith("_0000"):
        name = name[:-5]
    return name


def pair_image_and_label_files(
    images_dir: str | Path,
    labels_dir: str | Path,
) -> list[dict[str, str]]:
    """Pair images and labels by case ID and return paths relative to data root.

    The images directory may contain single-channel names such as
    case_001_0000.nii.gz. Label files are expected to use case_001.nii.gz
    or case_001.nii.
    """
    images_path = Path(images_dir).expanduser().resolve()
    labels_path = Path(labels_dir).expanduser().resolve()
    if not images_path.is_dir():
        raise FileNotFoundError(f"Images directory does not exist: {images_path}")
    if not labels_path.is_dir():
        raise FileNotFoundError(f"Labels directory does not exist: {labels_path}")

    image_files = sorted(
        p for p in images_path.iterdir()
        if p.is_file() and p.name.lower().endswith(NIFTI_SUFFIXES)
    )
    label_files = sorted(
        p for p in labels_path.iterdir()
        if p.is_file() and p.name.lower().endswith(NIFTI_SUFFIXES)
    )

    images_by_id = {case_id(p): p for p in image_files}
    labels_by_id = {case_id(p): p for p in label_files}

    if len(images_by_id) != len(image_files):
        raise ValueError("Duplicate image case IDs were found.")
    if len(labels_by_id) != len(label_files):
        raise ValueError("Duplicate label case IDs were found.")

    missing_labels = sorted(set(images_by_id) - set(labels_by_id))
    missing_images = sorted(set(labels_by_id) - set(images_by_id))
    if missing_labels or missing_images:
        details = []
        if missing_labels:
            details.append(f"images without labels: {missing_labels[:10]}")
        if missing_images:
            details.append(f"labels without images: {missing_images[:10]}")
        raise ValueError("Image/label pairing is incomplete; " + "; ".join(details))

    if not images_by_id:
        raise ValueError("No .nii or .nii.gz image/label pairs were found.")

    common_root = images_path.parent
    pairs = []
    for identifier in sorted(images_by_id):
        image = images_by_id[identifier]
        label = labels_by_id[identifier]
        pairs.append({
            "case_id": identifier,
            "image": image.relative_to(common_root).as_posix(),
            "label": label.relative_to(common_root).as_posix(),
        })
    return pairs


def split_pairs(
    pairs: list[dict[str, str]],
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, list[dict[str, str]]]:
    """Create deterministic, case-level train and validation splits."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1.")
    if len(pairs) < 2:
        raise ValueError("At least two paired cases are required for a split.")

    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    validation_count = max(
        1,
        min(len(shuffled) - 1, round(len(shuffled) * validation_fraction)),
    )
    validation = shuffled[:validation_count]
    train = shuffled[validation_count:]
    return {"train": train, "validation": validation}


def write_manifest(manifest: dict[str, Any], output_path: str | Path) -> Path:
    """Write manifest JSON, creating parent directories when needed."""
    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination


def load_manifest(
    manifest_path: str | Path,
    data_root: str | Path,
) -> dict[str, list[dict[str, str]]]:
    """Load manifest and resolve its relative image/label paths."""
    source = Path(manifest_path).expanduser()
    root = Path(data_root).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    result: dict[str, list[dict[str, str]]] = {}

    for split_name in ("train", "validation"):
        records = payload.get(split_name)
        if not isinstance(records, list) or not records:
            raise ValueError(f"Manifest split '{split_name}' must be a non-empty list.")
        resolved_records = []
        for record in records:
            if not isinstance(record, dict) or "image" not in record or "label" not in record:
                raise ValueError(f"Invalid record in '{split_name}' split: {record!r}")
            image = Path(record["image"])
            label = Path(record["label"])
            image_path = image if image.is_absolute() else root / image
            label_path = label if label.is_absolute() else root / label
            if not image_path.is_file():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            if not label_path.is_file():
                raise FileNotFoundError(f"Label file not found: {label_path}")
            resolved_records.append({
                **record,
                "image": str(image_path.resolve()),
                "label": str(label_path.resolve()),
            })
        result[split_name] = resolved_records

    train_ids = {record.get("case_id") for record in result["train"]}
    validation_ids = {record.get("case_id") for record in result["validation"]}
    overlap = train_ids.intersection(validation_ids) - {None}
    if overlap:
        raise ValueError(f"Case leakage between train and validation splits: {sorted(overlap)}")
    return result
