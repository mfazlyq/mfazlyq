# Colab Notebook Audit — Colorectal_Cancer_Segmentation_fix.ipynb

**Review scope:** Static review of the 43-cell notebook supplied by the author, including source cells and saved notebook outputs. The original notebook is retained by the author and is not copied into this public repository because it is large and contains rendered outputs. No CT images, segmentation masks, or model checkpoint were supplied with the notebook.

## 1. Method extracted from the notebook

| Component | Notebook implementation |
|---|---|
| Dataset | Task10_Colon; saved cell output reports 126 image/label pairs, split into 100 training and 26 validation cases |
| Split | Lexicographically sorted file list; first 80% for training and final 20% for validation |
| Image orientation | RAS |
| Resampling | 1.0 × 1.0 × 1.0 mm |
| CT intensity | Clip/window −200 to +200 HU and scale to [0, 1] |
| Foreground handling | CropForeground, then pad to at least 96 × 96 × 96 |
| Training sampling | Positive/negative label-guided crops; 4 crops per volume |
| Input and output | One CT channel; model configured for 3 output channels |
| Architecture | Custom 3D CBAM + RSU4 residual U-block stages, named CBAM_RSU2Net |
| Training recipe | DiceCELoss (softmax, one-hot target), AdamW (LR 1e-4, weight decay 1e-5), CosineAnnealingLR (T_max=100), 100 epochs, validation every 2 epochs, AMP |
| Checkpoint | best_cbam_rsu2net.pth loaded from Google Drive in evaluation cells |
| Proposed post-processing | Combine tumor and bowel probabilities to form an anatomical region; keep largest component; try tumor thresholds at 50%, 30%, then 10% of maximum tumor probability; constrain tumor prediction to the region; keep largest tumor component |
| Visual output | 2D slices, static 3D meshes, Plotly 3D overlays, per-case scores, learning-curve and comparison charts |

The saved notebook outputs report 6,209,705 trainable parameters. The notebook reports a baseline Dice/precision/recall of 0.3971/0.4401/0.4653 and, for the masked evaluation, a 21/26 case-level overlap hit count, conditional Dice 0.4757, precision 0.5808, and recall 0.4347. These are **reported notebook outputs**, not independently reproduced results.

## 2. Architecture port

src/colorectal_seg/cbam_rsu2net.py ports the actual model topology from the notebook, retaining the layer names and stage/channel definitions used by its model code. A local CPU smoke test completed with input shape (1, 1, 16, 16, 16), output shape (1, 3, 16, 16, 16), and exactly 6,209,705 parameters.

This smoke test verifies that the ported architecture executes and matches the notebook's reported parameter count. It does **not** verify checkpoint compatibility: the .pth checkpoint was not included, so a real load_state_dict comparison remains outstanding.

## 3. Reproducibility and correctness issues to resolve

### P0 — Three output classes are not defined by the visible data pipeline

The model is configured for three classes described in comments as background, tumor, and bowel. However, the data transforms only load, orient, resample, intensity-scale, crop, pad, and tensor-convert the label. The notebook contains no explicit label remapping step and no printed unique() audit of the masks.

Therefore, the source shown does not establish that class index 2 (bowel) is present in the training targets. If source masks contain only 0 and 1, the model's bowel output channel receives no positive supervision, and the subsequent bowel-probability mask is not validated. Before wiring this model into the training pipeline, record the unique label values and voxel counts for every source mask, then define and document the class mapping. Do not infer the mapping from the dataset folder name.

### P0 — Evaluation code uses incompatible prediction conventions

The notebook uses more than one prediction convention:
- A per-case evaluation cell thresholds sigmoid(logits) > 0.5 across all output channels.
- Another evaluation cell uses argmax and one-hot conversion with to_onehot=2.
- The hierarchical evaluation explicitly reads softmax channels 1 and 2 as tumor and bowel.

These conventions are not interchangeable for mutually exclusive multiclass segmentation. The evaluation protocol must be made consistent with the verified class mapping (normally softmax + argmax for exclusive classes, or sigmoid + per-channel thresholds only for a genuinely multilabel target).

### P0 — Reported overall masked Dice does not reconcile with the per-case table

The saved output lists 21 non-zero per-case Dice scores, whose arithmetic mean is 0.4757. Their arithmetic mean across all 26 cases, assigning zero to the five misses, is approximately 0.3842. The same cell reports “Mean Dice (Semua Pasien)” as 0.1759. These values do not reconcile under the displayed calculation and need to be recomputed from a single saved prediction/target set with a documented aggregation convention.

The 80.77% figure is calculated as the fraction of cases with Dice > 0; describe it as a case-level overlap hit rate unless a separately defined lesion-detection criterion is implemented. It is not the same quantity as voxel-level recall/sensitivity.

### P1 — Hierarchical masking differs between evaluation, visualization, and timing cells

- The principal evaluation uses 50%/30%/10% adaptive threshold fallback and largest-connected-component filtering.
- The five-case visualization uses only a 50% maximum-confidence threshold and a different KeepLargestConnectedComponent setup.
- The end-to-end timing cell uses a simple anatomical mask and 50% threshold, without the full fallback procedure or connected-component cleanup.

Consequently, the reported timing is not a benchmark of the exact same post-processing routine used in the principal evaluation. The implementation should be centralized in one tested function and reused by evaluation, visualization, and timing.

### P1 — Training completion and checkpoint provenance are not included

The saved output for the 100-epoch training cell ends in KeyboardInterrupt during epoch 1. Separate cells load best_cbam_rsu2net.pth and produce evaluation results, so the notebook does not establish that the displayed checkpoint was created by the interrupted training run. Keep the checkpoint's source run, epoch, validation metric, configuration, and code revision with the experiment record.

### P1 — Data split and preprocessing should be made explicit

The current split is based on sorted filenames, not a seeded randomized or stratified split. This is deterministic but can be biased if case ordering reflects acquisition or another structure. Save a manifest of train/validation case IDs and review class/lesion distribution across both splits. Also note that 1 mm isotropic spacing and the −200/+200 HU window are notebook settings, not universal defaults.

### P1 — Inference timing is patch-level, not full-volume patient latency

The timing cells use a synthetic 96³ input. The notebook separately shows validation volumes such as 250 × 229 × 296 voxels and uses sliding-window inference in some visualization cells. A single 96³ forward pass is a patch benchmark, not full-volume inference latency. Full-volume latency must include sliding-window stitching and the complete masking post-processing, with GPU model, precision mode, ROI, overlap, and warm-up procedure reported.

### P2 — Notebook outputs are not one clean execution record

There are duplicate data-loading cells and repeated model definitions. At least one downstream cell reports that df_results is unavailable even though an earlier cell defines it, and the training output is interrupted while later result cells show completed evaluation. This suggests saved outputs came from different or partially reset runtime states. A clean, ordered run should be generated after the code is modularized.

## 4. Recommended implementation sequence

1. Verify label values and establish the exact three-class mapping (or change the model to the true number of classes).
2. Load the supplied checkpoint against the ported architecture and confirm all keys and tensor shapes match.
3. Centralize post-processing in a single function with explicit parameters and unit tests.
4. Rebuild evaluation so each case's prediction, target, Dice, precision, recall, and detection status are computed from the same tensors; publish macro and micro aggregation definitions separately.
5. Reproduce full-volume inference with sliding windows and benchmark the full pipeline.
6. Run a clean training experiment with a saved split manifest, config, random seeds, epoch log, and checkpoint metadata.

## 5. What has and has not been reproduced

**Completed in this repository:** exact model topology port; local CPU forward smoke test; parameter-count comparison.

**Not yet verified:** three-class target semantics, .pth checkpoint compatibility, training reproduction, masked metrics, full-volume inference latency, and clinical validity.

No patient image data or checkpoint is included. This project remains research software and is not intended for clinical diagnosis or treatment decisions.
