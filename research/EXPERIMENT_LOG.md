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
