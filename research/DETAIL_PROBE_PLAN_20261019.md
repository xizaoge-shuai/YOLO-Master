# Full-split cache and spatial-detail screening (2026-10-19)

Status (2026-10-22): implementation and 49 CPU contract tests completed; GPU smoke and full campaign pending launch. No new accuracy result yet. The interrupted implementation was preserved by the user's server backup commit e83ba00.

## Question and evidence boundary

Can a small trainable image-detail pathway use part of the compute saved by frozen DINOv3 caching to improve detection? This is a gain probe and stronger baseline, not a novelty claim. Side-Tuning (ECCV 2020), ViT-Adapter (ICLR 2023), and ViT-CoMer (CVPR 2024) already establish related side-network/spatial-prior ideas. The previous reconstruction and native gradient correction results remain negative; they are not evidence that this new branch works.

## Frozen screening protocol

- VisDrone training 6,471 images, development validation 548 images, seed 0 first. No test-dev use.
- Existing DINOv3-S/16, image size 640, layers 3/7/11; frozen teacher and mean_value detection head.
- 100 epochs, batch 8, AdamW lr 0.001 and weight decay 0.0005, cosine decay to 0.00001. Same initialized head and deterministic sample/augmentation/query schedules.
- All arms request continuous uniform scales [0.85, 1.15] and 50% horizontal flip. Exactly 25% teacher image queries cumulatively, including tail batches. Query images and transforms paired across arms.
- Primary endpoint: fixed final epoch, pinned official-reference VisDrone AP (0-100), AP50, AP75, AR500. Existing custom AP is explicitly diagnostic. Final validation uses all 548 images; validate diagnostically every 10 epochs and at the final epoch. No best-epoch selection.
- Report query images/calls, train and validation seconds, wall time, cache construction time and images, physical cache bytes, added parameters, peak VRAM, and raw-image loading overhead. Full RGB detail and lowpass controls both receive equally augmented images at training and original images at evaluation.
- GPU1 failed a real CUDA allocation probe. GPU0 is shared with an existing task: record telemetry; these timings cannot establish an uncontended speedup. Strict equal-time experiments remain a later gate.

| Arm | Cached training views | Approximation on non-query images | Additional input |
|---|---|---|---|
| transport25 | original + flip | spatial transport | none |
| blank25 | original + flip | transport plus true blank-feature padding | none |
| views25 | 3 scales x 2 flips | nearest cached scale, residual transport, blank padding | none |
| detail25 | original + flip | same as blank25 | small RGB branch fused into P3 |
| lowpass25 | original + flip | same as blank25 | identical branch, lowpass RGB |

Six-view scales are {0.85, 1, 1.15}. Non-anchor requests remain continuous and approximate: relative feature warp = requested scale / anchor scale. This does not restore image context cropped out of an anchor. Label transforms always follow the requested image transform, never the anchor transform.

The branch uses stride-2 convolution, two depthwise/pointwise stages, GroupNorm and SiLU, and a zero-initialized 1x1 P3 projection. It adds no labels or teacher features beyond the paired budget. Lowpass reduces the image to 40x40 by area filtering then restores 640x640 bilinearly. Parameter count and initialization match the RGB branch. Equal initial predictions, nonzero learning gradients, save/load, and geometry/query contracts require tests.

The implemented branch adds 4,672 parameters. Both branch modes have identical initial head/branch states. Final-checkpoint sidecar recovery and archive retry against a newer pushed commit are covered by regression tests.

## Execution and recovery

1. Implement separate model/data/runner files and tests; preserve old training code and results.
2. Check original cache, image, label, manifest identities. Build a 17-image smoke cache and run all five arms for one epoch on 17/8 images. This includes a non-full tail batch. Do not interpret smoke AP.
3. Build four new full training banks while referencing the existing two. FP16 storage is about 89 GiB additional. Require actual required bytes plus 20 GiB free reserve before building. Preserve partial output on failure. Record actual construction cost and blank-image extraction.
4. Run the five arms serially. Atomic last checkpoint allows exact epoch-boundary resume; old completed runs cannot be overwritten. Snapshot and hash loaded source dependencies before starting. Save per-epoch paired schedules and monitor GPU utilization.
5. Verify final predictions cover exactly 548 images, evaluation matches the pinned reference protocol, image/query totals match, and paired schedule hashes agree. Archive only lightweight configs, metrics, provenance and hashes. Commit and push scoped progress, including failures.

Do not count all four new view banks against the two-view arms. Charge source train and validation cache build once per run (not amortized over seeds), plus each arm's required new views/blank; report data preparation separately. Any later reused cache run must still report its charged construction cost and physical footprint.

## Interpretation and next gate

Compare detail25 against blank25 and lowpass25 first, then against views25 and transport25. A single seed is exploratory, not a significant improvement. Continue to paired seeds 1/2 and strict cost measurements only if the enhancement plausibly improves final official AP at acceptable cost. Report regressions and parameter/inference overhead. Full-data cross-backbone/dataset tests follow a stable candidate, not a smoke or oracle-gradient result. The equal-token/equal-energy spatial intervention from the previous diagnostic is still a separate pending mechanism experiment.

## Implementation tasks

- Data: cache builder, approximate multi-view sampling, aligned RGB/labels, query and validation loader; contract tests.
- Model: zero-init P3 side branch and lowpass control; initialization/gradient/state tests.
- Runner: frozen protocol, resumable training, official final scoring, costs and telemetry, snapshot validation, background launcher and scoped archive/push.
- Review: independent protocol and code review, targeted tests/lint, required repository checks with pre-existing failures reported separately.
