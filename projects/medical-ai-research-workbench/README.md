# Medical AI Research Workbench

A local-first research application for inspecting 3D NIfTI CT volumes, reviewing segmentation overlays, validating image geometry, and calculating binary segmentation metrics.

> **Research software only.** This application is not a medical device, diagnostic system, or clinical decision-support tool. Demo files are synthetic. Do not upload identifiable patient information to a public deployment. Use real medical data only when authorized and under the applicable institutional privacy and ethics requirements.

## What is included

- NIfTI CT volume upload: `.nii` and `.nii.gz`.
- Optional binary mask uploads for Ground Truth, external Colon Prior, and AI Prediction.
- Axial, coronal, and sagittal slice viewers.
- Overlay contours with a consistent legend:
  - Red: Ground Truth
  - Cyan: Colon Prior
  - Violet: AI Prediction
- NIfTI geometry validation: canonical orientation, volume shape, and voxel-to-world affine comparison.
- Binary segmentation metrics: Dice, IoU, precision, recall, and accuracy.
- Ground-truth tumor coverage by an external anatomical prior.
- CSV export of calculated metrics.
- Synthetic 3D phantom generator for testing the application without patient data.
- Unit tests and a GitHub Actions workflow.

## Current scope

This first release is a **viewer and evaluation workbench**. It does not run CBAM-RSU2Net inference, load model checkpoints, train a network, or automatically generate TotalSegmentator priors. Those capabilities can be added as separate, validated modules later.

The app expects segmentation files to be binary masks or probability maps. For probability maps with values between 0 and 1, the selected threshold is applied. For integer label maps, every non-zero value is treated as foreground. This first version is designed for the binary Task10_Colon tumor-label convention.

## Run locally in VS Code

### 1. Clone this branch

From PowerShell or the VS Code terminal:

```powershell
git clone --branch feature/medical-ai-research-workbench https://github.com/mfazlyq/mfazlyq.git
cd mfazlyq
code .
```

### 2. Open the project folder

From the repository root, run:

```powershell
cd projects/medical-ai-research-workbench
```

### 3. Create a virtual environment

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install the workbench

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 5. Generate synthetic demo volumes

From the repository root:

```bash
python projects/medical-ai-research-workbench/scripts/create_demo_data.py
```

The script writes four synthetic NIfTI files to `demo_data/`. They are generated locally and are not committed to Git.

### 6. Start Streamlit

```bash
streamlit run app.py
```

Streamlit prints the local URL (normally `http://localhost:8501`). Upload the four generated files from `demo_data/` to explore the interface.

## Project structure

```text
medical-ai-research-workbench/
├── .streamlit/
│   └── config.toml
├── docs/
│   ├── ARCHITECTURE.md
│   └── DATA_AND_SAFETY.md
├── scripts/
│   └── create_demo_data.py
├── src/
│   └── medworkbench/
│       ├── __init__.py
│       ├── io.py
│       ├── metrics.py
│       └── visualization.py
├── tests/
│   ├── test_io.py
│   └── test_metrics.py
├── app.py
├── pyproject.toml
├── requirements.txt
└── .gitignore
```

## Metrics and definitions

For binary Ground Truth (GT) and Prediction (P):

- Dice = `2TP / (2TP + FP + FN)`
- IoU = `TP / (TP + FP + FN)`
- Precision = `TP / (TP + FP)`
- Recall = `TP / (TP + FN)`
- Accuracy = `(TP + TN) / (TP + TN + FP + FN)`
- Prior coverage = `|GT ∩ Prior| / |GT|`

If the denominator for precision or recall is zero, that metric is reported as 0. If both GT and Prediction are empty, Dice and IoU are defined as 1 for this implementation; interpret empty-mask cases separately in research reporting.

**Important:** Prior coverage is not Dice and is not model accuracy. It only reports how many ground-truth foreground voxels fall inside the supplied prior.

## Data compatibility and geometry checks

All uploaded volumes are converted to closest canonical orientation (RAS+) before display. Overlays and metrics are enabled only when the candidate mask and CT have matching 3D shapes and matching voxel-to-world affines within a small numerical tolerance. The app does not silently resample or register masks. A geometry mismatch must be corrected in a separately documented preprocessing step.

NiBabel uses NIfTI affine metadata to map voxel coordinates into world coordinates. See the [NiBabel NIfTI documentation](https://nipy.org/nibabel/nifti_images.html).

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Roadmap

1. Interactive three-plane NIfTI viewer and binary metrics — included in this first release.
2. 3D surface rendering and connected-component inspection.
3. Import model predictions from CBAM-RSU2Net checkpoints through an explicitly configured inference adapter.
4. Experiment comparison and reproducibility report.
5. Optional MONAI integration for a local inference workflow.

## Technology

Python · Streamlit · NiBabel · NumPy · Plotly · NIfTI

## Author

**M. Fazly Qusyairy**  
GitHub: https://github.com/mfazlyq
