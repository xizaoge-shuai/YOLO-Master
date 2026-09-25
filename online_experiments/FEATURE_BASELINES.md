# D1 feature augmentation baselines

This campaign implements selected ideas from FroFA (CVPR 2024) and LOFF-TA
(NeurIPS 2024) inside the existing frozen-DINOv3 detection pilot. It does not
reproduce either paper's classifier, complete augmentation search or results.
It does not implement the separate D1 from-scratch YOLO-Master comparison.

Sources consulted:
- https://arxiv.org/html/2403.10519v2 (Section 3.3 and Appendix A2)
- https://arxiv.org/html/2410.02527v1 (Section 3.3)

## Fixed configuration

The successful runs/paper/flip-cache/latest-job.txt job supplies the actual
original/flip feature cache, extraction provenance, dataset, train/validation
split, optimizer settings and epoch count. The current experiment is 400/100
images, 100 epochs, seeds 0/1/2, fixed mean-value fusion. GPU0 is identified by
UUID and all new runs execute serially. No model or dataset download is needed.
The original trainer/helper source files are not edited.

## Ten configurations, three seeds each

| Name | Cache views | Feature augmentation |
| --- | --- | --- |
| cache1 | original | none |
| cache2 | original + real image flip | choose one with probability 1/2 |
| frofa-c1-d01 | original | channel-squared brightness, delta 0.1 |
| frofa-c1-d03 | original | channel-squared brightness, delta 0.3 |
| frofa-c2-d01 | original + real flip | real view selection + brightness 0.1 |
| frofa-c2-d03 | original + real flip | real view selection + brightness 0.3 |
| transport-c1 | original | feature flip + continuous center scale |
| loffta-c1-n005 | original | same geometry + relative Gaussian noise 0.05 |
| transport-c2 | original + real flip | real flip selection + feature center scale |
| loffta-c2-n005 | original + real flip | same geometry + relative Gaussian noise 0.05 |

FroFA applies per-image/per-channel spatial min/max normalization, adds a
uniform channel offset in [-delta, delta], clips the normalized values to
[0,1], then restores the original channel range. Constant channels stay
constant. Different DINO layers draw independently. Detector/label geometry
is unchanged by brightness.

LOFF-TA-style augmentation retains the existing center-scale interval
(currently [0.85,1.15]) and 0.5 horizontal-flip probability. This is zooming
on a fixed 640 canvas, not changing model input resolution. Features use
bilinear interpolation, align_corners=False, raw zero padding and no range
clipping. Boxes follow the same transform, clipping and visibility rules as
the image-space pilot. Minimum box size is tested at image resolution, not
at the feature grid resolution. Gaussian noise has standard deviation 0.05
times each warped channel's spatial population standard deviation.

These choices deliberately retain the D1 detector rather than LOFF-TA's
classifier/CLS-token/projection-normalization architecture. Rotation, shear,
TrivialAugment, pooling and the original paper's full recipe are not included.
C2 combinations are additional stronger controls, not methods claimed by
the papers. Raw zero padding is an approximation; a poor geometric result
alone does not establish a general impossibility of feature augmentation.

## Integrity and cost

- Feature draws use an independent generator; detector initialization and
  image shuffle remain paired by seed.
- Spatial variants share flip/scale draws. Non-spatial C2 variants share the
  old cache2 flip sequence. Different augmentation families do not claim
  identical realized image-space augmentations.
- Augmentation is batch-local and does not modify stored cache tensors.
- All validation uses the original unaugmented validation cache.
- Thirty full runs are preceded by ten one-epoch smoke runs. The first
  failure stops the queue. Partial successful results are kept and reported.
- Cache files and source hashes are checked. Torch version is checked
  against extraction. Data/label identity is inherited and checked.
- All training runs use zero online backbone calls. Cache construction is
  reused, so actual additional build cost is zero. build+wall charges the
  archived original training-cache build once per run as a cold-start estimate;
  existing validation-cache build is excluded. This is labeled in the report.
- The CSV times cover feature augmentation as part of training. Setup, training,
  validation and total times remain available in per-run summaries. Feature
  caches are GPU-resident: memory results are specific to this small pilot.
- Strengths are fixed before the sweep; all are reported. Best validation AP
  and selection among strengths on 100 images are exploratory, not held-out
  test performance. Report final AP as well and lock settings before larger data.

## Deliverable

Run install_feature_baselines.sh in the server's yolomaster environment.
It adds the new helper and this note, then starts the queue via nohup.
Status, logs, comparison.json and comparison.txt live under
runs/paper/feature-baselines/job-*; latest-job.txt points to that job.

Local verification covers feature/label transformations, small-object
retention, input immutability, RNG isolation, validation isolation and source
patch guards. Actual YOLO/DINO GPU training is verified by the server smoke
runs, not by the local CPU tests.
