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
