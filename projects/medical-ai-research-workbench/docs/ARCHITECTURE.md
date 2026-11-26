# Architecture

## Application boundary

The Workbench is a local-first Streamlit application. It is an inspection and evaluation interface, not a model-serving platform.

```text
User uploads
  ├── CT NIfTI (required)
  ├── Ground Truth mask (optional)
  ├── Colon prior (optional)
  └── AI prediction (optional)
          |
          v
   NiBabel NIfTI loader
          |
          v
 Closest canonical orientation
          |
          v
 Shape + affine validation
          |
          +----------------------+
          |                      |
          v                      v
 Three-plane viewer        Binary metrics
 axial / coronal / sagittal Dice / IoU / precision / recall
                          prior coverage
          |                      |
          +-----------+----------+
                      |
                      v
                CSV summary
```

## Modules

- `app.py`: Streamlit interface, upload lifecycle, layout, and metric export.
- `src/medworkbench/io.py`: NIfTI loading, canonical orientation, spacing/affine metadata, geometry validation, and binary mask conversion.
- `src/medworkbench/metrics.py`: voxel-level overlap metrics and anatomical-prior coverage.
- `src/medworkbench/visualization.py`: Plotly slice extraction and overlay boundaries.
- `scripts/create_demo_data.py`: deterministic synthetic volume generation for UI testing.
- `tests/`: unit tests for metric definitions and spatial input safeguards.

## Geometry policy

All volumes are converted to closest canonical orientation before display. A mask is compatible only if its 3D array shape matches the CT and its affine is numerically equal within the configured tolerance. The Workbench does not silently interpolate, register, or resample data. Mismatches block that mask from overlay and metric calculation.

NiBabel's NIfTI affine maps voxel coordinates to world coordinates. Geometry must be considered in addition to matching array dimensions.

## Metric conventions

All evaluation masks are converted to binary foreground:
- Values entirely in [0, 1] use the probability threshold from the sidebar.
- Other integer/label volumes use non-zero as foreground.

The current implementation computes global voxel-wise metrics (micro aggregation over voxels), not per-case macro averages. A future batch-evaluation page should report both aggregation conventions and per-case values.

Prior coverage is evaluated separately from prediction metrics:
`coverage = count(GT AND prior) / count(GT)`.

The current app does not calculate Hausdorff distance, surface Dice, connected components, or lesion-level detection rate.

## Extension points

Future modules may provide:
- A local CBAM-RSU2Net inference adapter with explicit checkpoint and preprocessing configuration.
- TotalSegmentator prior generation as an optional external step.
- 3D surface rendering and linked crosshairs.
- Per-case batch reports and experiment tracking.
