"""Plotly helpers for 2D slice inspection of 3D NIfTI volumes."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


PLANE_AXES = {
    "Axial": 2,
    "Coronal": 1,
    "Sagittal": 0,
}

OVERLAY_COLORS = {
    "Ground Truth": "#FF5B67",
    "Colon Prior": "#55DDE0",
    "AI Prediction": "#B49AFF",
}


def slice_2d(volume: np.ndarray, plane: str, index: int) -> np.ndarray:
    """Return a display-oriented 2D slice from a canonical 3D array."""
    if plane not in PLANE_AXES:
        raise ValueError(f"Unsupported plane '{plane}'.")
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume.shape}.")

    axis = PLANE_AXES[plane]
    if not 0 <= index < volume.shape[axis]:
        raise IndexError(f"Slice index {index} is out of range for {plane}.")

    if plane == "Axial":
        result = volume[:, :, index].T
    elif plane == "Coronal":
        result = volume[:, index, :].T
    else:
        result = volume[index, :, :].T

    return np.asarray(result)


def make_slice_figure(
    ct_volume: np.ndarray,
    overlays: dict[str, np.ndarray],
    plane: str,
    index: int,
    *,
    display_min: float,
    display_max: float,
) -> go.Figure:
    """Create a CT heatmap with optional binary mask boundaries."""
    image_slice = slice_2d(ct_volume, plane, index)
    figure = go.Figure()
    figure.add_trace(
        go.Heatmap(
            z=image_slice,
            colorscale="Gray",
            zmin=display_min,
            zmax=display_max,
            showscale=False,
            hovertemplate="Voxel value: %{z:.1f}<extra></extra>",
            name="CT",
        )
    )

    for label, mask in overlays.items():
        if label not in OVERLAY_COLORS:
            continue
        mask_slice = slice_2d(mask.astype(np.uint8), plane, index)
        if not np.any(mask_slice):
            continue

        figure.add_trace(
            go.Contour(
                z=mask_slice.astype(np.float32),
                contours={
                    "start": 0.5,
                    "end": 0.5,
                    "size": 1,
                    "coloring": "lines",
                    "showlabels": False,
                },
                line={"color": OVERLAY_COLORS[label], "width": 2},
                showscale=False,
                hoverinfo="skip",
                name=label,
            )
        )

    figure.update_layout(
        template="plotly_dark",
        title={"text": f"{plane} · slice {index}", "x": 0.02, "font": {"size": 13}},
        margin={"l": 4, "r": 4, "t": 38, "b": 4},
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "x": 0,
            "font": {"size": 9},
        },
    )
    figure.update_xaxes(visible=False, constrain="domain")
    figure.update_yaxes(
        visible=False,
        scaleanchor="x",
        scaleratio=1,
        autorange="reversed",
    )
    return figure
