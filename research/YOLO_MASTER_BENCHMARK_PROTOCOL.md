# YOLO-Master CVPR 2026 benchmark alignment

Source: user-supplied Lin_YOLO-Master_MOE-Accelerated_with_Specialized_Transformers_for_Enhanced_Real-time_Detection_CVPR_2026_paper.pdf,
page 6, Tables 1 and 2. The requested paper is YOLO-Master, not FroFA.

## What the paper states

- Five datasets: COCO, PASCAL VOC, VisDrone, KITTI, SKU-110K.
- Table 2 reports 600 epochs, 640 resolution, batch size 256, SGD.
- Augmentations: Mosaic 1.0, Copy-Paste 0.1, Random Affine, HSV Jitter.
- Table 1 YOLO-Master-N: COCO AP/AP50 42.4/59.2; VisDrone AP/AP50 19.6/33.7.
- These are literature reference numbers, not scores reproduced in this project.
- Dataset split details, ignore handling, exact release configuration, and hardware
  must be verified before claiming exact reproduction. Repository v0-v15 configs
  must not be assumed equivalent to the paper release.
- Paper latency is inference latency; the D1 extension studies adaptation training
  cost. Cached training does not remove DINO extraction for new raw test images.

## Current study versus full evaluation

Current: fixed 400 train / 100 validation images, three training seeds,
100 epochs, 640 input, batch 8, AdamW, frozen DINOv3 + mean_value detection head.
All 500 source images are from VisDrone images/train (verified from all500.txt);
400/100 is an internal split of this training subset. Validation is the pilot
split and D1 evaluator, not the official 548-image validation or independent test-dev.

Use full VisDrone first: 6,471 train, 548 validation, and 1,610 test-dev.
Choose configurations on validation; reserve test-dev for the finalized candidate.
Official VisDrone test-dev annotations are available per the dataset authors:
https://github.com/VisDrone/VisDrone-Dataset
Keep official ignored-region handling via the raw annotation archives/toolkit.
Report the official detection metric, and separately labeled COCO-style AP,
AP50, AP75, AP-small/medium/large with explicit area and maxDet definitions.
Do not relabel a generic pycocotools conversion as the official VisDrone protocol.

Server inventory: research/benchmark-readiness-20260925.json.
VisDrone image and label files are present in all three full splits. One training
label has zero height (9999985_00000_d_0000020.txt, row 11). Preserve the source
and explicitly exclude/log the degenerate box in a derived full-data annotation
set; do not silently alter historical pilot input files.
COCO has 5,000 validation images and val JSON; train images and train JSON are absent.

## Baseline families

1. Paper reference: pinned YOLO-Master-N trained from scratch on official splits
   with the documented recipe. Disclose any batch/optimizer/augmentation deviations.
   Gradient accumulation is not identical to paper batch 256 (e.g. BatchNorm).
2. D1 practical baseline: same YOLO-Master-N scratch model and our frozen methods
   under the same total measured GPU-time budget, on the same split/evaluator.
3. Mechanism controls: C1, C2, a small discrete-scale cache, full-online frozen
   DINO, C2-Mix, C2-Transport and C2-Correct under matched teacher-image budgets.
4. Literature adaptations: existing FroFA/LOFF-TA detection adaptations, explicitly
   labeled as idea implementations, not their original paper reproductions.

## Scaling requirements

Full VisDrone C2 raw FP16 training features are about 44.43 GiB; val C1 about
1.88 GiB before serialization/allocator/model costs. Do not preload them all
using the pilot's GPU-resident bank. Implement disk/CPU streaming and measure I/O.
Full COCO C1 alone is about 406.11 GiB before labels, serialization and additional
views, exceeding current free disk. Plan storage/precision before building it.

For the full comparison, account for cache generation, setup/parity forwards,
online forwards, corrector updates, detection training and validation. Publish
both single-run total and an explicitly amortized repeated-training scenario.
Use a second full dataset and a separate source-to-target generalization protocol
before making broad efficiency or domain-generalization claims.
