from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medworkbench.io import NiftiVolume, binarize_volume, load_nifti_bytes, validate_same_geometry
from medworkbench.metrics import compute_binary_metrics, compute_prior_coverage
from medworkbench.visualization import PLANE_AXES, make_slice_figure


st.set_page_config(
    page_title="Medical AI Research Workbench",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1600px;}
    .hero {padding: 1.15rem 1.4rem; border: 1px solid rgba(85,221,224,.22);
           border-radius: 16px; background: linear-gradient(115deg, rgba(15,42,61,.92), rgba(8,19,33,.94));}
    .eyebrow {font-size: .72rem; letter-spacing: .17em; color: #55DDE0; font-weight: 700;}
    .hero h1 {font-size: clamp(2rem, 3.8vw, 3rem); line-height: 1.1; margin: .35rem 0 .7rem;}
    .hero p {color: #A6B9CC; max-width: 850px; margin-bottom: .2rem;}
    .metric-card {border: 1px solid rgba(130,170,205,.18); background: rgba(12,27,45,.68);
                  border-radius: 12px; padding: .85rem 1rem;}
    .metric-label {font-size: .72rem; color: #93A9BE; letter-spacing: .08em; text-transform: uppercase;}
    .small-note {color: #94A9BC; font-size: .82rem;}
    div[data-testid="stFileUploader"] {border: 1px dashed rgba(85,221,224,.25); border-radius: 10px; padding: .45rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">MEDICAL IMAGING · RESEARCH TOOLING · 3D SEGMENTATION</div>
      <h1>Medical AI Research Workbench</h1>
      <p>Inspect NIfTI volumes, compare segmentation masks, validate spatial geometry,
      and calculate transparent binary segmentation metrics — in one local research workspace.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

st.warning(
    "Research prototype only — not for diagnosis or clinical decisions. "
    "Use synthetic or properly de-identified/authorized data. This version does not run AI inference."
)

with st.sidebar:
    st.markdown("## 01 · Data input")
    st.caption("Upload one CT volume and any aligned masks you want to inspect.")
    ct_upload = st.file_uploader(
        "CT volume · required",
        type=["nii", "gz"],
        key="ct_upload",
        help="3D NIfTI only: .nii or .nii.gz (maximum 200 MB).",
        max_upload_size=200,
    )
    gt_upload = st.file_uploader(
        "Ground Truth · optional",
        type=["nii", "gz"],
        key="gt_upload",
        max_upload_size=200,
    )
    prior_upload = st.file_uploader(
        "Colon Prior · optional",
        type=["nii", "gz"],
        key="prior_upload",
        max_upload_size=200,
    )
    pred_upload = st.file_uploader(
        "AI Prediction · optional",
        type=["nii", "gz"],
        key="pred_upload",
        max_upload_size=200,
    )
    st.markdown("---")
    st.markdown("## 02 · Display")
    probability_threshold = st.slider(
        "Probability threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Applied only when the uploaded mask values are between 0 and 1. Integer label maps use non-zero as foreground.",
    )
    st.caption("The threshold changes visualization and metrics only; it does not modify uploaded files.")
    st.markdown("---")
    st.markdown("## 03 · About")
    st.caption("Version 0.1.0 · Local-first · NIfTI · Binary masks")

if not ct_upload:
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.markdown("### Start with a CT volume")
        st.write(
            "Upload a 3D CT NIfTI file. Add Ground Truth, Colon Prior, and AI Prediction "
            "to activate overlays and evaluation metrics."
        )
        st.markdown(
            """
            **Supported in this release**
            - Axial, coronal, and sagittal slice navigation
            - Ground Truth / Prior / Prediction overlays
            - Spatial geometry checks
            - Dice, IoU, precision, recall, accuracy, and prior coverage
            """
        )
    with right:
        st.markdown("### Quick demo")
        st.code(
            "python projects/medical-ai-research-workbench/scripts/create_demo_data.py\n"
            "streamlit run projects/medical-ai-research-workbench/app.py",
            language="bash",
        )
        st.caption("The demo generator creates synthetic NIfTI files locally. Upload them here after generation.")
    st.stop()


@st.cache_data(show_spinner="Reading NIfTI volume...")
def cached_load(file_bytes: bytes, filename: str) -> NiftiVolume:
    return load_nifti_bytes(file_bytes, filename)


def read_upload(uploaded_file, role: str) -> NiftiVolume | None:
    if uploaded_file is None:
        return None
    try:
        volume = cached_load(uploaded_file.getvalue(), uploaded_file.name)
        st.sidebar.success(f"{role}: loaded · {volume.shape}")
        return volume
    except Exception as exc:
        st.sidebar.error(f"{role}: {exc}")
        return None


with st.spinner("Loading CT volume..."):
    try:
        ct_volume = cached_load(ct_upload.getvalue(), ct_upload.name)
    except Exception as exc:
        st.error(f"Could not load CT volume: {exc}")
        st.stop()

gt_volume = read_upload(gt_upload, "Ground Truth")
prior_volume = read_upload(prior_upload, "Colon Prior")
pred_volume = read_upload(pred_upload, "AI Prediction")

candidate_volumes: dict[str, NiftiVolume | None] = {
    "Ground Truth": gt_volume,
    "Colon Prior": prior_volume,
    "AI Prediction": pred_volume,
}
valid_volumes: dict[str, NiftiVolume] = {}
invalid_volumes: dict[str, str] = {}

for label, candidate in candidate_volumes.items():
    if candidate is None:
        continue
    try:
        validate_same_geometry(ct_volume, candidate)
        valid_volumes[label] = candidate
    except ValueError as exc:
        invalid_volumes[label] = str(exc)

if invalid_volumes:
    with st.expander("Spatial compatibility warnings", expanded=True):
        st.error(
            "One or more masks do not share the CT grid. Those masks are excluded from overlays and metrics. "
            "The app does not silently resample or register medical images."
        )
        for label, message in invalid_volumes.items():
            st.write(f"**{label}:** {message}")

ct_min = float(np.nanpercentile(ct_volume.data, 1))
ct_max = float(np.nanpercentile(ct_volume.data, 99))
if not np.isfinite(ct_min) or not np.isfinite(ct_max) or ct_min >= ct_max:
    ct_min = float(np.min(ct_volume.data))
    ct_max = float(np.max(ct_volume.data))
if ct_min >= ct_max:
    ct_max = ct_min + 1.0

st.markdown("## Volume overview")
overview = st.columns(4)
overview[0].metric("Volume shape", " × ".join(str(v) for v in ct_volume.shape))
overview[1].metric("Voxel spacing", " × ".join(f"{v:.2f}" for v in ct_volume.spacing_mm) + " mm")
overview[2].metric("Orientation", ct_volume.orientation)
overview[3].metric("Display window · P1–P99", f"{ct_min:.0f} to {ct_max:.0f}")
st.caption(
    f"Loaded: {ct_volume.name} · Closest canonical orientation · "
    "Display intensity is clipped to the 1st–99th percentile for visualization only."
)

st.markdown("## 3D volume slice viewer")
st.caption(
    "Each panel is a 2D slice through the same canonical 3D volume. "
    "Use the slider below each panel to navigate independently."
)

overlay_arrays: dict[str, np.ndarray] = {}
for label, volume in valid_volumes.items():
    overlay_arrays[label] = binarize_volume(
        volume.data,
        probability_threshold=probability_threshold,
    )

cols = st.columns(3, gap="medium")
plane_order = ["Axial", "Coronal", "Sagittal"]
for col, plane in zip(cols, plane_order):
    axis = PLANE_AXES[plane]
    max_index = ct_volume.shape[axis] - 1
    initial_index = max_index // 2
    with col:
        st.markdown(f"### {plane}")
        slice_index = st.slider(
            f"{plane} slice",
            min_value=0,
            max_value=max_index,
            value=initial_index,
            key=f"slice_{plane.lower()}",
        )
        figure = make_slice_figure(
            ct_volume.data,
            overlay_arrays,
            plane,
            slice_index,
            display_min=ct_min,
            display_max=ct_max,
        )
        st.plotly_chart(
            figure,
            use_container_width=True,
            config={"displaylogo": False, "scrollZoom": True},
        )

if valid_volumes:
    legend_items = []
    if "Ground Truth" in valid_volumes:
        legend_items.append('<span style="color:#FF5B67">● Ground Truth</span>')
    if "Colon Prior" in valid_volumes:
        legend_items.append('<span style="color:#55DDE0">● Colon Prior</span>')
    if "AI Prediction" in valid_volumes:
        legend_items.append('<span style="color:#B49AFF">● AI Prediction</span>')
    st.markdown("Overlay legend: " + " &nbsp; · &nbsp; ".join(legend_items), unsafe_allow_html=True)

st.markdown("---")
st.markdown("## Segmentation evaluation")

gt_data = valid_volumes.get("Ground Truth")
prior_data = valid_volumes.get("Colon Prior")
pred_data = valid_volumes.get("AI Prediction")

has_metrics = False
csv_rows: list[tuple[str, str]] = []

if gt_data is not None and pred_data is not None:
    metrics = compute_binary_metrics(
        gt_data.data,
        pred_data.data,
        probability_threshold=probability_threshold,
    )
    has_metrics = True
    metric_cols = st.columns(5)
    metric_items = [
        ("Dice", metrics["dice"]),
        ("IoU", metrics["iou"]),
        ("Precision", metrics["precision"]),
        ("Recall", metrics["recall"]),
        ("Accuracy", metrics["accuracy"]),
    ]
    for column, (label, value) in zip(metric_cols, metric_items):
        column.metric(label, f"{float(value) * 100:.2f}%")
        csv_rows.append((label, f"{float(value):.8f}"))

    with st.expander("Voxel-level confusion counts", expanded=False):
        counts = st.columns(4)
        counts[0].metric("TP", f"{metrics['tp']:,}")
        counts[1].metric("FP", f"{metrics['fp']:,}")
        counts[2].metric("FN", f"{metrics['fn']:,}")
        counts[3].metric("TN", f"{metrics['tn']:,}")
        st.caption(
            f"GT foreground: {metrics['ground_truth_voxels']:,} voxels · "
            f"Prediction foreground: {metrics['prediction_voxels']:,} voxels"
        )
else:
    st.info("Upload both Ground Truth and AI Prediction on the same CT grid to calculate Dice, IoU, precision, recall, and accuracy.")

if gt_data is not None and prior_data is not None:
    coverage = compute_prior_coverage(
        gt_data.data,
        prior_data.data,
        probability_threshold=probability_threshold,
    )
    has_metrics = True
    st.markdown("### Anatomical prior coverage")
    cover_cols = st.columns(4)
    cover_cols[0].metric("Tumor coverage", f"{float(coverage['coverage']) * 100:.2f}%")
    cover_cols[1].metric("GT voxels covered", f"{coverage['covered_voxels']:,}")
    cover_cols[2].metric("GT voxels missed", f"{coverage['missed_voxels']:,}")
    cover_cols[3].metric("Prior foreground", f"{coverage['prior_voxels']:,}")
    st.caption(
        "Prior coverage = GT foreground voxels inside the prior / all GT foreground voxels. "
        "This is not Dice and does not measure prior precision."
    )
    for key, value in coverage.items():
        csv_rows.append((f"prior_{key}", str(value)))

if has_metrics:
    report_buffer = io.StringIO()
    writer = csv.writer(report_buffer)
    writer.writerow(["metric", "value"])
    writer.writerows(csv_rows)
    st.download_button(
        "Download evaluation summary · CSV",
        data=report_buffer.getvalue().encode("utf-8"),
        file_name="medical_ai_workbench_metrics.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption(
    "Medical AI Research Workbench · Research prototype · "
    "No inference is performed in this version. Verify all preprocessing, mask alignment, "
    "and metric conventions before using results in a research report."
)
