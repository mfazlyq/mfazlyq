"""Model definitions for baseline and future research architectures."""

from __future__ import annotations

from typing import Any

from monai.networks.nets import UNet


def build_model(config: dict[str, Any]):
    """Build the documented 3D U-Net baseline.

    The CBAM-U2Net research architecture is intentionally not represented here
    until its exact blocks, skip paths, attention placement, and masking rules
    are implemented and independently tested.
    """
    model_config = config["model"]
    if model_config.get("name") != "unet_baseline":
        raise ValueError(
            "Only 'unet_baseline' is currently implemented. "
            "Add and validate research architectures as separate model classes."
        )

    return UNet(
        spatial_dims=3,
        in_channels=int(model_config["in_channels"]),
        out_channels=int(model_config["num_classes"]),
        channels=tuple(int(v) for v in model_config["channels"]),
        strides=tuple(int(v) for v in model_config["strides"]),
        num_res_units=int(model_config["num_res_units"]),
    )
