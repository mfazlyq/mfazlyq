# 3D Colorectal Tumor Segmentation — Research Starter

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-3D%20Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)
![MONAI](https://img.shields.io/badge/MONAI-1.6.1-0054A6)
![Status](https://img.shields.io/badge/Status-Research%20Starter-orange)

A reproducible starter project for **3D colorectal tumor segmentation from CT volumes** using NIfTI data, PyTorch, and MONAI.

This project is designed as an engineering foundation for further experiments related to CBAM-based feature attention, U²-Net / Residual U-block architectures, and hierarchical masking strategies.

> **Important scope note:** The current implementation provides a MONAI 3D U-Net baseline and a configurable data pipeline. It does **not** claim to implement the complete CBAM-U²Net with adaptive hierarchical masking from the author's thesis. That research architecture should be added and validated as a separate model implementation. This is a research/education project, not a clinical diagnostic tool.

## Research context

The project is aligned with the research direction:

**Adaptive Hierarchical Masking Approach in a CBAM-U²Net Architecture for 3D Colorectal Cancer Segmentation.**

The starter establishes a consistent experiment path before introducing the proposed architecture:

1. Pair CT volumes and segmentation masks.
2. Create deterministic train/validation manifests.
3. Apply configurable 3D orientation, resampling, and CT intensity windowing.
4. Train a reproducible 3D U-Net baseline.
5. Evaluate foreground Dice on validation volumes using sliding-window inference.
6. Record experiment settings and checkpoints for comparison.

## Current capabilities

- NIfTI .nii and .nii.gz image/mask pairing by case identifier.
- Deterministic train/validation manifest generation.
- MONAI dictionary transforms for 3D CT and label volumes.
- Configurable target voxel spacing, CT window, crop size, batch size, and training parameters.
- 3D MONAI U-Net baseline with Dice + Cross-Entropy loss.
- Foreground Dice validation with sliding-window inference.
- Best-checkpoint saving and CSV training log.
- Unit tests for manifest pairing and split reproducibility.
- Dataset and model weights are excluded from Git by default.

## Repository structure

~~~text
colorectal-3d-segmentation/
├── configs/
│   └── default.yaml
├── docs/
│   └── RESEARCH_PROTOCOL.md
├── scripts/
│   └── create_manifest.py
├── src/
│   └── colorectal_seg/
│       ├── __init__.py
│       ├── data.py
│       ├── manifest.py
│       ├── models.py
│       └── train.py
├── tests/
│   └── test_manifest.py
├── .gitignore
├── pyproject.toml
└── requirements.txt
~~~

## Dataset layout

The project expects paired CT images and masks. Keep patient data outside the Git repository.

~~~text
data/
└── Task10_Colon/
    ├── imagesTr/
    │   ├── case_001_0000.nii.gz
    │   ├── case_002_0000.nii.gz
    │   └── ...
    └── labelsTr/
        ├── case_001.nii.gz
        ├── case_002.nii.gz
        └── ...
~~~

The pairing logic removes the common _0000 image-channel suffix. If your dataset uses a different naming convention, adapt the matching function before creating a manifest.

The starter assumes a **single-channel CT image** and a **binary mask** where 0 is background and 1 is tumor. Confirm your label values and convert them if necessary. Do not infer this from filenames.

## Installation

Create an isolated Python environment first.

~~~bash
python -m venv .venv
~~~

Activate it:

**Windows PowerShell**
~~~powershell
.venv\Scripts\Activate.ps1
~~~

**macOS / Linux**
~~~bash
source .venv/bin/activate
~~~

Install PyTorch using the official selector for your operating system and CUDA version, if applicable: https://pytorch.org/get-started/locally/

Then install this project's dependencies:

~~~bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
~~~

## Prepare the manifest

Place the dataset under data/Task10_Colon/, then run:

~~~bash
python scripts/create_manifest.py --data-root data/Task10_Colon --images-dir imagesTr --labels-dir labelsTr --output data/manifest.json --validation-fraction 0.2 --seed 42
~~~

The manifest stores relative paths, not image contents. Inspect the generated split and ensure that no patient/case leakage exists between train and validation.

## Configure an experiment

Edit configs/default.yaml before training. Important settings include:

- target_spacing: voxel spacing in millimeters; select this based on the dataset and research protocol.
- ct_window: CT intensity window in Hounsfield units; verify this for the actual data.
- patch_size: default 96 × 96 × 96 voxels.
- num_classes: currently 2 for background and tumor.
- seed: experiment reproducibility seed.

These defaults are starting points, not validated optimal settings.

## Train the baseline

~~~bash
python -m colorectal_seg.train --config configs/default.yaml
~~~

The trainer writes checkpoints and a CSV log to the configured output directory (default: outputs/baseline_unet/).

## Run unit tests

~~~bash
python -m unittest discover -s tests -v
~~~

The manifest tests use only Python's standard library. Model training requires the packages in requirements.txt, a correctly prepared dataset, and suitable compute resources.

## Evaluation and reporting

The baseline trainer computes foreground Dice on the validation set with sliding-window inference. Dice alone is insufficient for a complete segmentation study. A thesis-level evaluation should also report, where appropriate, Dice Similarity Coefficient (DSC), Intersection over Union (IoU), sensitivity/recall, precision, Hausdorff Distance (HD95), and volume difference or other justified measures.

Report the number of evaluated cases, empty-mask handling, thresholding/argmax strategy, resampling policy, and aggregation method. Never compare runs that use different splits or preprocessing without clearly documenting the differences.

## Research limitations and responsible use

- No patient dataset is included. Do not commit identifiable or restricted medical data.
- This is not a clinical product and must not be used to make diagnosis or treatment decisions.
- The U-Net is a baseline, not the proposed CBAM-U²Net research contribution.
- The current pipeline assumes binary labels and one CT channel.
- Resampling and CT window parameters must be verified for the source dataset.
- Results must be validated on an appropriate held-out cohort and reviewed by domain experts.
- Reproducibility requires recording dataset version, split manifest, code revision, configuration, hardware, and random seed.

## Author

**M. Fazly Qusyairy**  
GitHub: https://github.com/mfazlyq

Related portfolio: https://github.com/mfazlyq

---

Research starter for reproducible 3D medical image segmentation experiments.
