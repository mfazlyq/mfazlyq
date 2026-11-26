# Data, Privacy, and Research Scope

## Intended use

This software is a research prototype for viewing 3D NIfTI data and evaluating binary masks. It is not a medical device, clinical decision support system, diagnostic tool, or treatment recommendation system.

## Data handling

- Run locally when working with any non-public research data.
- Do not commit patient CT volumes, masks, predictions, checkpoints, DICOM files, or exported reports to Git.
- Do not upload identifiable or sensitive patient data to a public app or a deployment whose data-handling terms have not been reviewed.
- Use de-identified data only when permitted by the dataset license, institution, and applicable ethics approval.
- The included demo generator produces synthetic geometric phantoms and does not contain patient-derived data.

## Geometry and metric safeguards

- Shape equality alone is not sufficient for spatial alignment; compare NIfTI affines too.
- The Workbench converts data to closest canonical orientation for consistent viewing.
- The app does not silently resample, register, or deform masks to fit a CT grid.
- Metric calculations are enabled only for masks with matching shape and affine.
- Review preprocessing, label convention, thresholding, and aggregation before reporting results.
- Prior coverage measures containment of the Ground Truth by the prior; it is not a prediction metric and does not measure prior precision.

## Interpretation limits

- Dice, IoU, precision, recall, and accuracy are voxel-wise binary metrics in this release.
- Accuracy can be misleading for highly imbalanced tumor/background volumes; report overlap and class-sensitive metrics too.
- Empty-mask cases need separate interpretation. This implementation defines Dice and IoU as 1 when both masks are empty, while precision and recall are 0 when their respective denominators are zero.
- A visual overlay is a quality-control aid and does not establish clinical validity.
- The application does not execute AI inference or verify the provenance of supplied prediction files.

## Reporting

Record dataset release, case IDs (using an approved anonymized convention), preprocessing, label semantics, affine/spacing, threshold, model/checkpoint provenance, hardware, and metric aggregation. Do not publish protected health information.
