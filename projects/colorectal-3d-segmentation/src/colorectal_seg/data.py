"""MONAI preprocessing and data-loader construction."""

from __future__ import annotations

from typing import Any

from monai.data import CacheDataset, DataLoader, list_data_collate
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
)


def build_transforms(config: dict[str, Any], training: bool) -> Compose:
    """Build a 3D CT transform chain for images and binary segmentation masks."""
    preprocessing = config["preprocessing"]
    training_config = config["training"]
    spatial_size = tuple(int(v) for v in training_config["patch_size"])
    keys = ["image", "label"]

    transforms = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
        Orientationd(keys=keys, axcodes=preprocessing.get("orientation", "RAS")),
        Spacingd(
            keys=keys,
            pixdim=tuple(float(v) for v in preprocessing["target_spacing"]),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=float(preprocessing["ct_window"]["min"]),
            a_max=float(preprocessing["ct_window"]["max"]),
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        EnsureTyped(keys=keys, track_meta=True),
    ]

    if training:
        transforms.append(
            RandCropByPosNegLabeld(
                keys=keys,
                label_key="label",
                spatial_size=spatial_size,
                pos=1,
                neg=1,
                num_samples=int(training_config["samples_per_volume"]),
                image_key="image",
                image_threshold=0,
            )
        )
    return Compose(transforms)


def build_dataloaders(
    manifest: dict[str, list[dict[str, str]]],
    config: dict[str, Any],
    device_has_cuda: bool,
) -> tuple[DataLoader, DataLoader]:
    """Create cached MONAI loaders for training and full-volume validation."""
    data_config = config["data"]
    batch_size = int(config["training"]["batch_size"])
    workers = int(data_config.get("num_workers", 0))
    cache_rate = float(data_config.get("cache_rate", 0.0))

    train_dataset = CacheDataset(
        data=manifest["train"],
        transform=build_transforms(config, training=True),
        cache_rate=cache_rate,
        num_workers=workers,
        progress=True,
    )
    validation_dataset = CacheDataset(
        data=manifest["validation"],
        transform=build_transforms(config, training=False),
        cache_rate=cache_rate,
        num_workers=workers,
        progress=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=list_data_collate,
        pin_memory=device_has_cuda,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        collate_fn=list_data_collate,
        pin_memory=device_has_cuda,
    )
    return train_loader, validation_loader
