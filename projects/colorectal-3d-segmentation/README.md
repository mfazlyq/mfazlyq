# 3D Colorectal Tumor Segmentation

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-3D%20Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)
![MONAI](https://img.shields.io/badge/MONAI-1.6.1-0054A6)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-orange)

A research codebase for **3D colorectal tumor segmentation from CT volumes** using NIfTI, PyTorch, and MONAI.

Research direction: **Adaptive Hierarchical Masking Approach in a CBAM-U²Net Architecture for 3D Colorectal Cancer Segmentation.**

> This repository is research software, not a clinical diagnostic product. The baseline trainer and the notebook-derived CBAM-RSU2Net architecture are kept separate until the original label semantics and checkpoint can be verified.

## Implementation status

| Component | Status |
|---|---|
| MONAI 3D U-Net binary baseline | Starter trainer available |
| CBAM-RSU2Net 3D architecture from author's Colab notebook | Ported to `src/colorectal_seg/cbam_rsu2net.py`; local forward and parameter-count smoke tests pass |
| Three-class label mapping (background / tumor / bowel) | **Needs verification against source masks** |
| Hierarchical Anatomical Masking | Described and audited; canonical reusable implementation pending verification |
| Original `.pth` checkpoint compatibility | Not yet checked; checkpoint was not included |
| Full-volume benchmark and metric reproduction | Not yet independently reproduced |

The CBAM-RSU2Net port preserves the notebook's stage/channel topology and reports the same **6,209,705 parameters** in a local smoke test. It is not yet wired into the trainer: the notebook does not show an explicit mapping step that establishes how class index 2 (bowel) is represented in the ground-truth masks.

## Repository structure

~~~text
colorectal-3d-segmentation/
├── configs/
│   └── default.yaml
├── docs/
│   ├── COLAB_NOTEBOOK_AUDIT.md
│   └── RESEARCH_PROTOCOL.md
├── scripts/
│   └── create_manifest.py
├── src/
│   └── colorectal_seg/
│       ├── cbam_rsu2net.py
│       ├── data.py
│       ├── manifest.py
│       ├── models.py
│       └── train.py
├── tests/
│   ├── test_cbam_rsu2net.py
│   └── test_manifest.py
├── .gitignore
├── pyproject.toml
└── requirements.txt
~~~

## Dataset layout

Keep patient data outside Git. The baseline starter expects paired CT images and masks:

~~~text
data/
└── Task10_Colon/
    ├── imagesTr/
    │   ├── case_001_0000.nii.gz
    │   └── ...
    └── labelsTr/
        ├── case_001.nii.gz
        └── ...
~~~

The pairing code removes the common `_0000` image-channel suffix. Confirm the actual naming convention and label values before training.

The existing baseline trainer assumes one CT channel and a **binary** mask (0 = background, 1 = tumor). The notebook-derived CBAM-RSU2Net is configured for **three output channels**. Do not use it for training or interpret its bowel channel until the masks' unique values and the full class mapping have been verified.

## Install

Create and activate a virtual environment:

~~~bash
python -m venv .venv
~~~

On Windows PowerShell:

~~~powershell
.venv\Scripts\Activate.ps1
~~~

Install PyTorch for your operating system/CUDA version using the [official PyTorch selector](https://pytorch.org/get-started/locally/), then install the project:

~~~bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
~~~

## Prepare the baseline manifest

~~~bash
python scripts/create_manifest.py --data-root data/Task10_Colon --images-dir imagesTr --labels-dir labelsTr --output data/manifest.json --validation-fraction 0.2 --seed 42
~~~

Inspect the resulting train/validation split and ensure that no patient/case leakage exists.

## Train the binary baseline

~~~bash
python -m colorectal_seg.train --config configs/default.yaml
~~~

The starter writes a best checkpoint and CSV training log to the configured output directory (default: `outputs/baseline_unet/`).

## Run tests

~~~bash
PYTHONPATH=src python -m unittest discover -s tests -v
~~~

The manifest tests use the Python standard library. The CBAM-RSU2Net smoke tests require PyTorch and are skipped in lightweight environments where PyTorch is absent.

## Notebook findings and reproduction

See [Colab Notebook Audit](docs/COLAB_NOTEBOOK_AUDIT.md) for the extracted preprocessing/training settings, architecture port status, reported notebook metrics, and issues that must be resolved before reproducing results.

Key items to verify before the proposed method is treated as reproducible:
1. Unique label values and the exact mapping for background, tumor, and bowel.
2. State-dictionary compatibility with the original `.pth` file.
3. One consistent softmax/argmax or sigmoid evaluation convention, matching the target semantics.
4. Per-case and aggregate Dice, precision, recall, and case-level detection definitions.
5. One canonical hierarchical-mask implementation reused for evaluation, visualization, and timing.
6. Full-volume inference latency using sliding-window inference and the complete post-processing path.

The settings reported in the notebook (RAS, 1 mm isotropic spacing, CT window −200 to +200 HU, and 96³ training crops) are preserved as source-derived values in the audit. They should be treated as that experiment's configuration, not universal defaults.

## Responsible use

- Do not commit patient scans, labels, identifiable health information, or model weights.
- This repository is not intended for diagnosis or treatment decisions.
- Report dataset version, split manifest, preprocessing, configuration, random seed, hardware, checkpoint provenance, and metric aggregation when publishing experiments.
- Validate research results with appropriate held-out data and qualified domain experts.

## Author

**M. Fazly Qusyairy**  
GitHub: https://github.com/mfazlyq
