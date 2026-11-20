# 3D Colorectal Tumor Segmentation

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-3D%20Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)
![MONAI](https://img.shields.io/badge/MONAI-1.6.1-0054A6)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-orange)

Research code for **3D colorectal tumor segmentation from CT volumes** using NIfTI, PyTorch, and MONAI.

Research direction: **Adaptive Hierarchical Masking Approach in a CBAM-U²Net Architecture for 3D Colorectal Cancer Segmentation.**

> This is research software, not a clinical diagnostic product. Patient data, NIfTI masks, and model weights are not stored in this repository.

## Dataset finding and selected method

The author ran a label-value audit on 126 Task10_Colon masks. The scan found only:
- 0 = background: 3,533,465,548 voxels
- 1 = tumor foreground: 1,808,436 voxels
- No label 2 was present.

Therefore, the implementation direction is a **two-output CBAM-RSU2Net (background/tumor) plus a separate external colon anatomical prior**.

The third output channel in the original Colab notebook was not supervised as bowel by the supplied binary labels. It must not be reused as a bowel mask.

The candidate external prior generator is TotalSegmentator, whose default CT task includes a colon class. Its upstream documentation warns that colon and small bowel may sometimes be confused. Consequently, generated colon masks are predictions, not ground truth, and must pass spatial checks, tumor-coverage analysis, and visual review before they are used to constrain tumor predictions.

On the three sample cases, the author measured macro mean tumor coverage of 83.5954% (0 mm), 98.4212% (5 mm), and 99.9162% (10 mm). The 5 mm setting is only a candidate for validation: the prior volume expands as the margin grows, and these three cases are not enough to select a final margin. See [Anatomical Prior Protocol](docs/ANATOMICAL_PRIOR_PROTOCOL.md) for per-case results and the validation plan.

## Implementation status

| Component | Status |
|---|---|
| MONAI 3D U-Net binary baseline | Starter trainer available |
| CBAM-RSU2Net architecture | Ported from Colab; two-class head added for the audited labels |
| External colon prior generator | Successfully run by the author on 3 sample CT volumes |
| Prior geometry/completeness audit | Passed for the 3 sample CT volumes |
| Tumor coverage by colon prior | 0/5/10 mm comparison completed on 3 cases; validation on the full cohort remains pending |
| Hierarchical masking | Reuses 50/30/10 adaptive threshold and largest-component cleanup with independent prior |
| Real-data reproduction and metrics | Not yet run or independently reproduced |

## Repository structure

~~~text
colorectal-3d-segmentation/
├── configs/
│   ├── cbam_rsu2net_binary.yaml
│   └── default.yaml
├── docs/
│   ├── ANATOMICAL_PRIOR_PROTOCOL.md
│   ├── COLAB_NOTEBOOK_AUDIT.md
│   └── RESEARCH_PROTOCOL.md
├── scripts/
│   ├── audit_label_values.py
│   ├── audit_colon_priors.py
│   ├── audit_prior_vs_tumor_labels.py
│   ├── create_manifest.py
│   └── generate_colon_priors.py
├── src/
│   └── colorectal_seg/
│       ├── cbam_rsu2net.py
│       ├── data.py
│       ├── manifest.py
│       ├── masking.py
│       ├── models.py
│       └── train.py
├── tests/
│   ├── test_cbam_rsu2net.py
│   ├── test_manifest.py
│   └── test_masking.py
├── .gitignore
├── pyproject.toml
└── requirements.txt
~~~

## Dataset layout

Keep patient data outside Git.

~~~text
data/Task10_Colon/
├── imagesTr/
│   ├── colon_001.nii.gz
│   └── ...
├── labelsTr/
│   ├── colon_001.nii.gz
│   └── ...
└── colon_priors/
    └── colon_001/
        └── colon.nii.gz
~~~

The manifest pairs images, binary tumor labels, and optional colon priors by case identifier. macOS AppleDouble sidecars whose names begin with ._ are ignored.

## Install

Create and activate a Python virtual environment:

~~~bash
python -m venv .venv
~~~

Install PyTorch for the operating system/CUDA version using the [official PyTorch selector](https://pytorch.org/get-started/locally/), then:

~~~bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
~~~

Install TotalSegmentator separately for the anatomical prior workflow:

~~~bash
pip install TotalSegmentator
~~~

## Workflow

1. Audit the source label values:

~~~bash
python scripts/audit_label_values.py --labels-dir data/Task10_Colon/labelsTr
~~~

2. Generate TotalSegmentator colon priors:

~~~bash
python scripts/generate_colon_priors.py --images-dir data/Task10_Colon/imagesTr --output-dir data/Task10_Colon/colon_priors --device gpu
~~~

3. Check generated priors for missing/empty files and CT grid mismatch:

~~~bash
python scripts/audit_colon_priors.py --images-dir data/Task10_Colon/imagesTr --priors-dir data/Task10_Colon/colon_priors
~~~

4. Check how much of each tumor ground-truth mask is covered by the generated colon prior:

~~~bash
python scripts/audit_prior_vs_tumor_labels.py --images-dir data/Task10_Colon/imagesTr --labels-dir data/Task10_Colon/labelsTr --priors-dir data/Task10_Colon/colon_priors --margins-mm 0 5 10
~~~

5. Inspect representative overlays and low-coverage cases. The preliminary 3-case test suggests 5 mm is a candidate to validate, not a final setting. Do not use the prior as a hard constraint until the anatomical segmentations have been visually reviewed and coverage has been measured on all cases.

6. Create a reproducible manifest with all three sources:

~~~bash
python scripts/create_manifest.py --data-root data/Task10_Colon --images-dir imagesTr --labels-dir labelsTr --colon-priors-dir colon_priors --output data/cbam_rsu2net_binary_manifest.json --validation-fraction 0.2 --seed 42
~~~

7. Train the binary CBAM-RSU2Net with external-prior post-processing:

~~~bash
python -m colorectal_seg.train --config configs/cbam_rsu2net_binary.yaml
~~~

See [Anatomical Prior Protocol](docs/ANATOMICAL_PRIOR_PROTOCOL.md) for the complete method, validation requirements, margin ablation, and checkpoint implications.

## Tests

~~~bash
PYTHONPATH=src python -m unittest discover -s tests -v
~~~

The model smoke tests require PyTorch, and the masking tests require NumPy/SciPy. Full data processing additionally requires MONAI, nibabel, TotalSegmentator, source CT volumes, and label masks.

## Research limitations

- TotalSegmentator colon masks are model predictions, may confuse colon with small bowel, and may omit tumor-invaded anatomy.
- The prior is not a ground-truth label.
- A strict prior can remove true tumor voxels; report tumor coverage by prior and compare with unmasked predictions.
- The legacy three-output checkpoint cannot be loaded strictly into the two-output model; retraining/fine-tuning is required.
- Report dataset version, train/validation split, preprocessing, TotalSegmentator version/task, prior margin, thresholds, random seed, hardware, checkpoint provenance, and metric aggregation.
- The system is not intended for clinical diagnosis or treatment decisions.

## Author

**M. Fazly Qusyairy**  
GitHub: https://github.com/mfazlyq
