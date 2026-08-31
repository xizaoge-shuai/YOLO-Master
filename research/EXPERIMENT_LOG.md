# Research Experiment Log

## A2. Exact-layer task sensitivity

Dataset: VisDrone500  
Calibration / evaluation split: 20 / 100  
Backbone: DINOv3-S/16, layers 3/7/11  
Detector: frozen-cache LatentMixture + Detect

### Horizontal flip

| Restored layers | mAP50-95 | AP-gap recovery |
|---|---:|---:|
| Transport only | 0.001113 | 0% |
| Layer 3 | 0.008442 | 98.48% |
| Layer 7 | 0.001116 | 0.04% |
| Layer 11 | 0.001117 | 0.05% |
| Layer 3+7 | 0.008448 | 98.56% |
| Layer 3+11 | 0.008533 | 99.70% |
| All layers | 0.008555 | 100% |

### Zoom-in 15%

| Restored layers | mAP50-95 | AP-gap recovery |
|---|---:|---:|
| Transport only | 0.002848 | 0% |
| Layer 3 | 0.010688 | 100.30% |
| Layer 7 | 0.002853 | 0.07% |
| Layer 11 | 0.002822 | -0.33% |
| Layer 3+7 | 0.010712 | 100.61% |
| Layer 3+11 | 0.010670 | 100.07% |
| All layers | 0.010665 | 100% |

### Observation

For both horizontal flipping and zoom-in augmentation, almost the entire downstream detection gap is determined by layer 3. Exact recovery of layers 7 and 11 alone provides essentially no task recovery. This motivates task-sensitive correction rather than indiscriminate reconstruction of all representation defects.

## A3. Task-sensitive low-rank defect subspace

VisDrone500 uses the fixed 400/100 train-validation split. All 100 held-out validation samples are evaluated.

### Main findings

1. The downstream task sensitivity is highly concentrated at DINOv3 layer 3.

For horizontal flip, the mean first-order impacts are:

- layer 3: 2.7379e-03
- layer 7: 3.2978e-06
- layer 11: 1.1655e-05

For 15% zoom-in:

- layer 3: 2.4670e-03
- layer 7: 5.9732e-06
- layer 11: 1.1321e-05

2. Task-sensitive subspaces provide the largest advantage under small rank budgets.

Horizontal flip AP-gap recovery:

- rank 4: PCA 0.1674, task-sensitive 0.3326
- rank 8: PCA 0.3166, task-sensitive 0.6487
- rank 16: PCA 0.7205, task-sensitive 0.7732
- rank 64: PCA 0.8729, task-sensitive 0.9524

Zoom-in AP-gap recovery:

- rank 1: PCA 0.1664, task-sensitive 0.4171
- rank 2: PCA 0.2623, task-sensitive 0.4395
- rank 4: PCA 0.4116, task-sensitive 0.5486
- rank 64: PCA 1.0641, task-sensitive 0.9591

3. Minimum rank for at least 90% AP-gap recovery:

- horizontal flip: PCA 96, task-sensitive 64
- zoom-in: PCA 64, task-sensitive 64

The evidence therefore supports task-sensitive low-rank correction primarily as a low-budget representation mechanism rather than a universally superior subspace at every rank.

## A4. Calibration stability

Three independently sampled calibration sets were evaluated on the same complete 100-image held-out VisDrone500 validation split.

At rank 64:

- 5 calibration samples:
  - hflip: 0.9220 ± 0.0326 AP-gap recovery
  - zoom: 0.9333 ± 0.0604

- 20 calibration samples:
  - hflip: 0.9079 ± 0.0324
  - zoom: 0.9323 ± 0.0801

- 50 calibration samples:
  - hflip: 0.9637 ± 0.0336
  - zoom: 0.9379 ± 0.0504

Rank 64 is therefore used as the default correction budget for the first deployable predictor.
