# External Colon Anatomical Prior Protocol

## Decision and scope

The audited Task10_Colon label masks contain only two values: 0 (background) and 1 (tumor). They do not provide a ground-truth colon/bowel class. The proposed implementation therefore separates two sources:

1. **CBAM-RSU2Net binary tumor model:** takes the CT volume and predicts background/tumor (two output channels).
2. **External colon anatomical prior:** a separate CT anatomy segmentation model generates a colon mask. This mask is an inference-time anatomical constraint, not a target channel and not a claim that the tumor model learned colon anatomy.

The legacy three-channel head remains available only for historical inspection. Its third channel is not a valid bowel segmentation and must not be used as an anatomical prior.

## Candidate prior generator: TotalSegmentator

The default TotalSegmentator CT task includes a colon class (class index 20 in the current published class map). It supports selecting only requested structures with --roi_subset colon and writing outputs to NIfTI. The project pins no TotalSegmentator package version; record the installed version and model weights with each experiment.

Official upstream documentation:
- Repository and usage: https://github.com/wasserth/TotalSegmentator
- Class details: https://github.com/wasserth/TotalSegmentator#class-details
- Paper: Wasserthal et al. (2023), TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images, Radiology: Artificial Intelligence, https://doi.org/10.1148/ryai.230024

The upstream documentation notes that colon and small_bowel are sometimes mixed up. Treat its output as a **candidate anatomical prior**, not ground truth. Manually review overlay/surface visualizations on representative Task10_Colon cases, including tumors near or invading the bowel wall. If the prior misses the tumor-bearing colon segment, strict masking may remove a correct tumor prediction.

The default total CT task is documented as Apache-2.0. Check the current upstream license and task notes before redistributing model weights or using a different TotalSegmentator subtask.

## Preliminary three-case coverage result

The author generated external colon masks for three sample CT volumes and ran the tumor-label coverage audit. At the original 0 mm margin:

| Case | Tumor voxels | Tumor voxels inside prior | Coverage |
|---|---:|---:|---:|
| colon_001 | 1,783 | 1,464 | 82.1088% |
| colon_005 | 15,212 | 11,046 | 72.6137% |
| colon_006 | 1,194 | 1,147 | 96.0637% |

The macro mean (equal weight per case) is **83.5954%**. The pooled voxel coverage (all tumor voxels combined) is **75.0838%**. The lowest-coverage case, colon_005, has **27.3863%** of annotated tumor voxels outside the prior.

This is a preliminary three-case sample only. It is not an estimate of performance across all 126 cases. It is already sufficient to show that an unexpanded prior must **not** be applied as an unexamined hard constraint: it would exclude a substantial fraction of tumor ground-truth voxels in these samples.

The coverage audit script now compares the prior at multiple physical margins (default 0, 5, and 10 mm):

~~~bash
python scripts/audit_prior_vs_tumor_labels.py \
  --images-dir data/Task10_Colon/imagesTr \
  --labels-dir data/Task10_Colon/labelsTr \
  --priors-dir data/Task10_Colon/colon_priors \
  --margins-mm 0 5 10
~~~

The test should show whether dilation improves coverage, but **higher coverage alone does not prove the anatomical prior is correct**. Larger margins also make the prior less selective. Inspect the overlays, record the prior volume/occupancy, and compare masked versus unmasked tumor metrics using a fixed validation split. Select margin settings on validation data only, then report them unchanged on a held-out test set.

## Author-run margin comparison on the three sample cases

The author subsequently ran the updated coverage audit at physical dilation margins of 0, 5, and 10 mm. The results are:

| Case | Tumor voxels | Covered at 0 mm | Coverage 0 mm | Covered at 5 mm | Coverage 5 mm | Covered at 10 mm | Coverage 10 mm |
|---|---:|---:|---:|---:|---:|---:|---:|
| colon_001 | 1,783 | 1,464 | 82.1088% | 1,743 | 97.7566% | 1,783 | 100.0000% |
| colon_005 | 15,212 | 11,046 | 72.6137% | 14,871 | 97.7583% | 15,212 | 100.0000% |
| colon_006 | 1,194 | 1,147 | 96.0637% | 1,191 | 99.7487% | 1,191 | 99.7487% |

| Margin | Macro mean across cases | Pooled tumor-voxel coverage | Minimum case coverage |
|---|---:|---:|---:|
| 0 mm | 83.5954% | 75.0838% | 72.6137% |
| 5 mm | 98.4212% | 97.8888% | 97.7566% |
| 10 mm | 99.9162% | 99.9835% | 99.7487% |

At 5 mm, 384 of 18,189 annotated tumor voxels remain outside the dilated prior. At 10 mm, 3 of 18,189 remain outside. The anatomical-prior voxel counts increased as follows:

| Case | Prior voxels at 0 mm | At 5 mm | At 10 mm |
|---|---:|---:|---:|
| colon_001 | 369,566 | 546,684 | 794,553 |
| colon_005 | 245,180 | 427,971 | 701,302 |
| colon_006 | 229,081 | 394,115 | 684,674 |

The coverage improves substantially with dilation, but the anatomical prior also expands markedly. This makes 5 mm a **candidate validation setting**, not a final selected parameter. The 10 mm result approaches complete tumor coverage in this tiny sample but produces a considerably broader prior and may reduce anatomical selectivity. A larger prior can admit more false-positive regions and therefore cannot be selected solely by maximizing tumor coverage.

**Decision at this stage:** do not hard-mask the tumor predictions yet. Visually inspect CT, tumor ground truth, original colon prior, and 5/10 mm dilated priors for the three cases, especially colon_005. Then run the same coverage and overlay QC on the full cohort, and compare raw versus post-processed tumor metrics on a fixed validation split. Select the margin using validation data only; do not tune on the held-out test set. These three cases are a pilot sample, not evidence of cohort-wide performance.

## Generate prior masks

Install TotalSegmentator in the same Colab environment as the dataset:

~~~bash
pip install TotalSegmentator
~~~

Start with a small sample of CT volumes (for example, copy 3-5 files to a temporary folder) to inspect model behavior. Then generate all priors:

~~~bash
python scripts/generate_colon_priors.py \
  --images-dir data/Task10_Colon/imagesTr \
  --output-dir data/Task10_Colon/colon_priors \
  --device gpu
~~~

The expected output for each case is:

~~~text
data/Task10_Colon/colon_priors/
└── colon_001/
    └── colon.nii.gz
~~~

The generator checks that each output is non-empty and that its voxel grid and affine match the source CT. It can skip existing valid outputs; pass --overwrite to regenerate them. --fast is optional but reduces model resolution and should be treated as a different experiment configuration.

## Audit and validate priors before training

First check missing, empty, or geometry-mismatched masks:

~~~bash
python scripts/audit_colon_priors.py \
  --images-dir data/Task10_Colon/imagesTr \
  --priors-dir data/Task10_Colon/colon_priors
~~~

Then measure how much of each ground-truth tumor mask lies inside the colon prior:

~~~bash
python scripts/audit_prior_vs_tumor_labels.py \
  --images-dir data/Task10_Colon/imagesTr \
  --labels-dir data/Task10_Colon/labelsTr \
  --priors-dir data/Task10_Colon/colon_priors
~~~

This is a critical safety check for the research pipeline: a prior that excludes annotated tumor voxels can force false negatives. There is no universal acceptable coverage threshold; inspect low-coverage cases and decide the protocol before evaluating the model.

## Build the prior-aware manifest

~~~bash
python scripts/create_manifest.py \
  --data-root data/Task10_Colon \
  --images-dir imagesTr \
  --labels-dir labelsTr \
  --colon-priors-dir colon_priors \
  --output data/cbam_rsu2net_binary_manifest.json \
  --validation-fraction 0.2 \
  --seed 42
~~~

The manifest associates image, tumor label, and colon prior using the same case ID. Each prior is transformed with the CT and label: orientation is aligned, the image is resampled linearly, and labels/priors use nearest-neighbor interpolation. The validation loader keeps the full volume for sliding-window inference.

## Visual quality control

Create side-by-side axial overlays for the tumor ground truth and the original, 5 mm, and 10 mm colon priors. The script chooses the axial slice with the largest number of tumor voxels for each requested case:

~~~bash
python scripts/preview_prior_overlays.py \\
  --images-dir data/Task10_Colon/imagesTr \\
  --labels-dir data/Task10_Colon/labelsTr \\
  --priors-dir data/Task10_Colon/colon_priors \\
  --output-dir outputs/prior_visual_qc \\
  --cases colon_001 colon_005 colon_006
~~~

Red contours show the annotated tumor; cyan contours show the colon prior at the specified margin. Review the entire tumor region in all three dimensions, not only this representative axial slice. The PNGs are a screening aid and not a substitute for volumetric review.

## Train and evaluate

Use configs/cbam_rsu2net_binary.yaml. The network predicts two classes; the independent prior is only used by validation/inference post-processing. The hierarchical masking implementation retains the notebook's 50%, 30%, 10% relative-maximum threshold fallback and largest-component cleanup, but intersects candidates with the external colon prior rather than the unsupervised third network channel.

The strict margin is 0 mm by default. Compare 0, 5, and 10 mm margins on the validation split only, alongside unmasked binary predictions. Do not select a margin on the test set. Keep the same split and checkpoint for all ablation comparisons.

~~~bash
python -m colorectal_seg.train --config configs/cbam_rsu2net_binary.yaml
~~~

## Checkpoint implications

A legacy three-output checkpoint cannot be loaded strictly into the new two-output model because the final convolution has a different shape. The old output channel 2 is not supervised as bowel by the audited labels. Do not load that third channel as an anatomical prior. For the binary model, initialize a fresh two-class head and train/fine-tune with the binary labels; record whether any compatible backbone weights were transferred.

## Required ablation and reporting

At minimum compare on identical validation/test cases:
- Binary CBAM-RSU2Net without anatomical masking.
- Binary CBAM-RSU2Net with external colon prior at 0 mm margin.
- Binary CBAM-RSU2Net with prior dilated by candidate margins selected on validation only.

Report both raw and post-processed metrics: DSC, precision, recall/sensitivity, IoU, HD95 where justified, case-level hit rate, number of prior failures, and ground-truth tumor coverage by the prior. Record the TotalSegmentator version/task/device, output grid, post-processing parameters, and manual QC protocol.

## Limitations

- TotalSegmentator colon masks are predictions and may confuse colon with small bowel; visual/quantitative QC is mandatory.
- The colon prior is not a ground-truth annotation and should not be described as one.
- The pipeline is research software, not a clinical diagnostic device.
