"""Train the 3D MONAI U-Net baseline."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from monai.data import decollate_batch
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete

from colorectal_seg.data import build_dataloaders
from colorectal_seg.manifest import load_manifest
from colorectal_seg.models import build_model
from colorectal_seg.masking import apply_hierarchical_anatomical_masking


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(configured: str) -> torch.device:
    if configured == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(configured)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this PyTorch environment.")
    return device


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    required = {"data", "preprocessing", "model", "training"}
    missing = required.difference(config or {})
    if missing:
        raise ValueError(f"Config is missing sections: {sorted(missing)}")
    return config


def validate_patch_size(patch_size: tuple[int, ...], strides: tuple[int, ...]) -> None:
    if len(patch_size) != 3 or len(strides) != 4:
        raise ValueError("This project expects a 3D patch and four U-Net downsampling strides.")
    divisor = int(np.prod(strides))
    if any(size < divisor or size % divisor != 0 for size in patch_size):
        raise ValueError(
            f"Each patch dimension must be at least and divisible by {divisor}; got {patch_size}."
        )


def validate_config(config: dict[str, Any]) -> None:
    model_config = config["model"]
    training_config = config["training"]
    if int(model_config["in_channels"]) != 1:
        raise ValueError("The current CT data pipeline expects one input channel.")
    if int(model_config["num_classes"]) != 2:
        raise ValueError("The starter currently supports binary background/tumor segmentation only.")
    if len(model_config["channels"]) != len(model_config["strides"]) + 1:
        raise ValueError("UNet channels must contain one more entry than strides.")
    patch_size = tuple(int(v) for v in training_config["patch_size"])
    strides = tuple(int(v) for v in model_config["strides"])
    validate_patch_size(patch_size, strides)
    if int(training_config["epochs"]) < 1:
        raise ValueError("epochs must be at least 1.")
    if int(training_config["batch_size"]) < 1:
        raise ValueError("batch_size must be at least 1.")


def evaluate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    patch_size: tuple[int, int, int],
    overlap: float,
    num_classes: int,
    use_colon_prior: bool = False,
    prior_margin_mm: float = 0.0,
    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """Compute mean foreground Dice on full volumes.

    When enabled, each tumor prediction is constrained by an independently
    generated colon prior. The prior is never inferred from the unsupervised
    third channel of the legacy notebook model.
    """
    inferer = SlidingWindowInferer(
        roi_size=patch_size,
        sw_batch_size=1,
        overlap=overlap,
    )
    post_prediction = AsDiscrete(argmax=True, to_onehot=num_classes)
    post_label = AsDiscrete(to_onehot=num_classes)
    metric = DiceMetric(include_background=False, reduction="mean")

    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = inferer(images, model)

            if use_colon_prior:
                if "colon_prior" not in batch:
                    raise RuntimeError(
                        "Anatomical-prior evaluation requested, but batch has no 'colon_prior'. "
                        "Generate and attach TotalSegmentator priors before evaluation."
                    )
                probabilities = torch.softmax(logits, dim=1)
                prior_batch = batch["colon_prior"]
                predictions = []
                targets = []
                for index in range(images.shape[0]):
                    prediction_np, _details = apply_hierarchical_anatomical_masking(
                        probabilities[index, 1].detach().cpu().numpy(),
                        prior_batch[index, 0].detach().cpu().numpy(),
                        prior_margin_mm=prior_margin_mm,
                        spacing_mm=spacing_mm,
                    )
                    prediction_index = torch.as_tensor(
                        prediction_np.astype(np.int64), device=device
                    )
                    target_index = labels[index, 0].long()
                    prediction_onehot = torch.nn.functional.one_hot(
                        prediction_index, num_classes=num_classes
                    ).movedim(-1, 0).float()
                    target_onehot = torch.nn.functional.one_hot(
                        target_index, num_classes=num_classes
                    ).movedim(-1, 0).float()
                    predictions.append(prediction_onehot)
                    targets.append(target_onehot)
                metric(y_pred=predictions, y=targets)
            else:
                predictions = [post_prediction(item) for item in decollate_batch(logits)]
                targets = [post_label(item) for item in decollate_batch(labels)]
                metric(y_pred=predictions, y=targets)

    result = float(metric.aggregate().item())
    metric.reset()
    return result


def train(config: dict[str, Any]) -> Path:
    validate_config(config)
    seed = int(config["training"]["seed"])
    set_seed(seed)
    device = choose_device(str(config["training"].get("device", "auto")))
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(config["data"]["manifest"], config["data"]["root"])
    train_loader, validation_loader = build_dataloaders(
        manifest,
        config,
        device_has_cuda=(device.type == "cuda"),
    )

    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    patch_size = tuple(int(v) for v in config["training"]["patch_size"])
    num_classes = int(config["model"]["num_classes"])
    epochs = int(config["training"]["epochs"])
    validation_interval = max(1, int(config["training"]["validation_interval"]))
    best_dice = float("-inf")
    best_checkpoint = output_dir / "best_model.pt"
    log_path = output_dir / "training_log.csv"

    with log_path.open("w", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["epoch", "train_loss", "validation_foreground_dice"])

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            batches = 0
            for batch in train_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = loss_function(logits, labels)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite training loss at epoch {epoch}.")
                loss.backward()
                optimizer.step()
                running_loss += float(loss.detach().cpu().item())
                batches += 1

            if batches == 0:
                raise RuntimeError("Training loader produced no batches.")
            mean_loss = running_loss / batches
            validation_dice = ""

            if epoch % validation_interval == 0 or epoch == epochs:
                validation_dice = evaluate(
                    model=model,
                    loader=validation_loader,
                    device=device,
                    patch_size=patch_size,
                    overlap=float(config["training"]["sliding_window_overlap"]),
                    num_classes=num_classes,
                    use_colon_prior=bool(config["data"].get("use_colon_prior", False)),
                    prior_margin_mm=float(config["preprocessing"].get("colon_prior_margin_mm", 0.0)),
                    spacing_mm=tuple(
                        float(v) for v in config["preprocessing"]["target_spacing"]
                    ),
                )
                if validation_dice > best_dice:
                    best_dice = validation_dice
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "validation_foreground_dice": validation_dice,
                            "config": config,
                            "seed": seed,
                        },
                        best_checkpoint,
                    )

            writer.writerow([epoch, f"{mean_loss:.8f}", validation_dice])
            log_file.flush()
            print(
                f"Epoch {epoch:03d}/{epochs:03d} | "
                f"train_loss={mean_loss:.6f} | val_foreground_dice={validation_dice}"
            )

    print(f"Best checkpoint: {best_checkpoint} (Dice={best_dice:.6f})")
    print(f"Training log: {log_path}")
    return best_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the 3D colorectal segmentation baseline.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML experiment config.")
    args = parser.parse_args()
    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
