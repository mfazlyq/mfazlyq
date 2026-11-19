# Research protocol and experiment reporting

## Purpose

This document defines the minimum metadata to record when comparing 3D colorectal tumor segmentation experiments.

## Dataset record

Record the dataset name and version, inclusion/exclusion criteria, number of cases, image modality, acquisition characteristics, label provenance, and any licensing or access restrictions. Keep patient-identifiable data out of this repository.

## Split integrity

- Split by patient/case, not by slices or patches.
- Keep an immutable copy of the train/validation/test manifest for each experiment.
- Do not tune hyperparameters against the final test set.
- Document external validation separately from internal validation.

## Preprocessing

For every run, record input voxel spacing and target spacing, orientation convention, CT intensity window and normalization, resampling interpolation for images and masks, crop/patch dimensions, label-value mapping and treatment of empty masks, and any augmentation used only on training data.

The starter defaults are placeholders for experimentation. They are not a medical-imaging standard and must be justified using dataset properties and the research protocol.

## Model and optimization

Record the exact architecture revision, feature-channel configuration, attention modules, hierarchical masking rules, loss function, optimizer, learning rate, batch size, epoch count, random seed, software versions, and hardware.

The baseline in this repository is a conventional 3D MONAI U-Net. The proposed CBAM-U²Net and adaptive hierarchical masking should be implemented as a distinct, documented model, with an ablation plan.

## Evaluation

At minimum, report Dice Similarity Coefficient (DSC), Intersection over Union (IoU), precision and sensitivity, and HD95 when clinically and computationally appropriate.

Specify whether metrics are calculated per volume, per class, per patient, or globally; explain how empty ground-truth and empty-prediction cases are handled. Report mean and dispersion, not only the best case.

## Reproducibility checklist

- [ ] Dataset version and data-access conditions recorded.
- [ ] Patient-level split manifest archived.
- [ ] Preprocessing configuration versioned.
- [ ] Model architecture and parameter count reported.
- [ ] Random seeds and software versions recorded.
- [ ] Checkpoints and logs stored outside Git.
- [ ] Baselines and ablations run under comparable conditions.
- [ ] Final results reviewed by a domain expert.
