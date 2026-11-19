"""CBAM-RSU2Net 3D architecture ported from the supplied Colab notebook.

This module preserves the notebook's layer topology and channel widths so an
existing state_dict can be checked for compatibility. It is a research
implementation, not a clinically validated model.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ChannelAttention3D(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16) -> None:
        super().__init__()
        hidden = max(in_planes // ratio, 1)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(in_planes, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, in_planes, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError("kernel_size must be 3 or 7")
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        pooled = torch.cat((avg_out, max_out), dim=1)
        return self.sigmoid(self.conv1(pooled))


class CBAM3D(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        self.ca = ChannelAttention3D(in_planes, ratio)
        self.sa = SpatialAttention3D(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.ca(x)
        return x * self.sa(x)


class REBNCONV3D(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 3, dirate: int = 1) -> None:
        super().__init__()
        self.conv_s1 = nn.Conv3d(
            in_ch, out_ch, kernel_size=3, padding=dirate, dilation=dirate
        )
        self.bn_s1 = nn.BatchNorm3d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu_s1(self.bn_s1(self.conv_s1(x)))


def _upsample_like(src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.interpolate(
        src, size=target.shape[2:], mode="trilinear", align_corners=True
    )


class RSU4_CBAM3D(nn.Module):
    """Two-level residual U-block with CBAM on the decoded feature map."""

    def __init__(self, in_ch: int = 3, mid_ch: int = 12, out_ch: int = 3) -> None:
        super().__init__()
        self.rebnconvin = REBNCONV3D(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV3D(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool3d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV3D(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool3d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV3D(mid_ch, mid_ch, dirate=1)
        self.rebnconv4 = REBNCONV3D(mid_ch, mid_ch, dirate=2)
        self.rebnconv5 = REBNCONV3D(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv6 = REBNCONV3D(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv7 = REBNCONV3D(mid_ch * 2, out_ch, dirate=1)
        self.cbam = CBAM3D(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.rebnconvin(x)
        enc1 = self.rebnconv1(residual)
        enc2 = self.rebnconv2(self.pool1(enc1))
        enc3 = self.rebnconv3(self.pool2(enc2))
        center = self.rebnconv4(enc3)
        dec2 = self.rebnconv5(torch.cat((center, enc3), dim=1))
        dec2 = _upsample_like(dec2, enc2)
        dec1 = self.rebnconv6(torch.cat((dec2, enc2), dim=1))
        dec1 = _upsample_like(dec1, enc1)
        decoded = self.rebnconv7(torch.cat((dec1, enc1), dim=1))
        return self.cbam(decoded) + residual


class CBAM_RSU2Net(nn.Module):
    """3D CBAM-RSU2Net with a binary head by default.

    Two output channels match the audited Task10_Colon label values (0=background,
    1=tumor). The legacy 3-channel head is available only to inspect historical
    checkpoints; its third channel is not supervised by the supplied masks and
    must never be interpreted as a learned bowel segmentation.
    """

    def __init__(self, in_ch: int = 1, out_ch: int = 2) -> None:
        super().__init__()
        if in_ch != 1:
            raise ValueError("The supplied notebook uses a single-channel CT input.")
        if out_ch not in (2, 3):
            raise ValueError("out_ch must be 2 (binary Task10_Colon) or 3 (legacy notebook inspection).")
        self.stage1 = RSU4_CBAM3D(in_ch, 16, 64)
        self.pool12 = nn.MaxPool3d(2, stride=2, ceil_mode=True)
        self.stage2 = RSU4_CBAM3D(64, 32, 128)
        self.pool23 = nn.MaxPool3d(2, stride=2, ceil_mode=True)
        self.stage3 = RSU4_CBAM3D(128, 64, 256)
        self.stage4 = RSU4_CBAM3D(256 + 128, 32, 128)
        self.stage5 = RSU4_CBAM3D(128 + 64, 16, 64)
        self.outconv = nn.Conv3d(64, out_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.stage1(x)
        enc2 = self.stage2(self.pool12(enc1))
        enc3 = self.stage3(self.pool23(enc2))
        dec2 = _upsample_like(enc3, enc2)
        dec2 = self.stage4(torch.cat((dec2, enc2), dim=1))
        dec1 = _upsample_like(dec2, enc1)
        dec1 = self.stage5(torch.cat((dec1, enc1), dim=1))
        return self.outconv(dec1)


def build_cbam_rsu2net(in_channels: int = 1, num_classes: int = 2) -> CBAM_RSU2Net:
    """Factory matching the notebook's default model configuration."""
    return CBAM_RSU2Net(in_ch=in_channels, out_ch=num_classes)


__all__ = [
    "CBAM_RSU2Net", "CBAM3D", "ChannelAttention3D", "REBNCONV3D",
    "RSU4_CBAM3D", "SpatialAttention3D", "build_cbam_rsu2net",
]
