# Pilot Validation Report — External Colon Anatomical Prior

## Objective

Evaluate whether a separately generated TotalSegmentator colon mask can serve as an external anatomical prior for the binary CBAM-RSU2Net research prototype. The pilot compares physical dilation margins of 0, 5, and 10 mm on three selected Task10_Colon sample cases.

## Scope and method

- Sample cases: colon_001, colon_005, and colon_006.
- Source labels: binary tumor ground truth (background = 0, tumor = 1).
- Prior generator: TotalSegmentator colon output.
- Geometry/completeness checks: passed for all three generated sample priors.
- Coverage definition: tumor voxels that fall inside the dilated prior divided by all tumor ground-truth voxels for that case.
- Visual screening: manually reviewed the representative axial slice with the maximum number of tumor voxels in each sample. Red = tumor ground truth; cyan = colon prior boundary.
- Margins were applied in physical millimeters.

This is a three-case proof-of-concept, not a cohort-wide performance estimate.

## Tumor-label coverage

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

At 5 mm, 384 of 18,189 tumor voxels remain outside the prior. At 10 mm, 3 of 18,189 remain outside.

## Prior expansion

| Case | Prior voxels at 0 mm | At 5 mm | At 10 mm | 5 mm / 0 mm | 10 mm / 0 mm |
|---|---:|---:|---:|---:|---:|
| colon_001 | 369,566 | 546,684 | 794,553 | 1.48× | 2.15× |
| colon_005 | 245,180 | 427,971 | 701,302 | 1.75× | 2.86× |
| colon_006 | 229,081 | 394,115 | 684,674 | 1.72× | 2.99× |

The 10 mm prior is considerably larger. In colon_006, tumor coverage is identical at 5 and 10 mm (99.7487%), while the 10 mm prior contains approximately 1.74 times as many voxels as the 5 mm prior.

## Visual screening observations

The representative overlays for all three sample cases show that the 5 mm boundary expands around the tumor ground-truth contours and increases their coverage compared with the unexpanded prior. The 10 mm boundary is visibly broader. In colon_006, the 10 mm margin does not improve tumor coverage over 5 mm despite the additional expansion. The overlays also show prior regions outside the tumor contour; this is expected for an anatomical prior and means coverage alone cannot establish anatomical precision.

This visual check is limited to one representative axial slice per case. It does not constitute full 3D volumetric verification or a formal assessment of prior precision.

## Prototype decision

**Selected demonstration setting: 5 mm physical dilation.**

Rationale:
1. It achieves 97.8888% pooled and 98.4212% macro mean tumor-label coverage in the three samples.
2. Its minimum per-case coverage is 97.7566%.
3. The 10 mm setting adds only about 2.09 percentage points of pooled coverage compared with 5 mm, while increasing prior volume substantially.
4. In colon_006, 10 mm provides no coverage improvement over 5 mm.

The 5 mm setting is therefore used as a practical sample-level prototype default in `configs/cbam_rsu2net_binary.yaml`. It is not claimed to be globally optimal or validated across all 126 cases.

## What is and is not validated

**Completed for the three selected samples**
- TotalSegmentator prior generation.
- Non-empty output and CT/prior geometry checks.
- Quantitative coverage comparison at 0, 5, and 10 mm.
- Representative maximum-tumor axial overlay review.

**Not performed**
- Prior generation and coverage audit across all 126 cases.
- Complete 3D volumetric manual review.
- End-to-end CBAM-RSU2Net training and checkpoint reproduction.
- Comparison of raw versus masked tumor predictions using Dice, precision, recall, or other prediction metrics.

The coverage values in this report describe how much ground-truth tumor lies inside the prior. They are **not model segmentation metrics**.

## Reproducibility

Run the sample scripts with the three selected CT volumes, labels, and generated priors in separate sample directories:

```bash
python scripts/generate_colon_priors.py \
  --images-dir data/sample/imagesTr \
  --output-dir data/sample/colon_priors \
  --device gpu

python scripts/audit_colon_priors.py \
  --images-dir data/sample/imagesTr \
  --priors-dir data/sample/colon_priors

python scripts/audit_prior_vs_tumor_labels.py \
  --images-dir data/sample/imagesTr \
  --labels-dir data/sample/labelsTr \
  --priors-dir data/sample/colon_priors \
  --margins-mm 0 5 10

python scripts/preview_prior_overlays.py \
  --images-dir data/sample/imagesTr \
  --labels-dir data/sample/labelsTr \
  --priors-dir data/sample/colon_priors \
  --output-dir outputs/prior_visual_qc \
  --cases colon_001 colon_005 colon_006
```

Patient CT volumes, label masks, and model weights are intentionally excluded from this repository.

## Research and use limitation

This report documents a technical portfolio prototype and a small-sample method check. It must not be interpreted as clinical validation, cohort-wide generalization, or evidence of diagnostic performance.
