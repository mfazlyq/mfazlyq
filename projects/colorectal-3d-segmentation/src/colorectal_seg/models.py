"""Model factory for baseline and notebook-derived research architectures."""

from __future__ import annotations

from typing import Any

from monai.networks.nets import UNet

from colorectal_seg.cbam_rsu2net import build_cbam_rsu2net


def build_model(config: dict[str, Any]):
    """Build the configured 3D segmentation model."""
    model_config = config["model"]
    model_name = model_config.get("name")

    if model_name == "unet_baseline":
        return UNet(
            spatial_dims=3,
            in_channels=int(model_config["in_channels"]),
            out_channels=int(model_config["num_classes"]),
            channels=tuple(int(v) for v in model_config["channels"]),
            strides=tuple(int(v) for v in model_config["strides"]),
            num_res_units=int(model_config["num_res_units"]),
        )

    if model_name == "cbam_rsu2net":
        return build_cbam_rsu2net(
            in_channels=int(model_config["in_channels"]),
            num_classes=int(model_config["num_classes"]),
        )

    raise ValueError(
        "Unsupported model name. Choose 'unet_baseline' or 'cbam_rsu2net'."
    )
