# Completed held-out feature/task diagnostic

Run: /data/users/zhjia/YOLO-Master/runs/paper/heldout-corrector/job-7c6i3o4w

SUCCEEDED: 548 validation images, 5,480 teacher-image evaluations, 10,960 paired feature rows; 136.69 seconds (diagnostic timing only). No parameter updates; final100epoch seed0 checkpoints; batch4.

## Nonidentity transforms

Rows average the two flip conditions. Positive percentages mean reduction relative to Transport. These are feature/loss metrics, not AP gains.

| Scale | Foreground relative-MSE reduction | Background relative-MSE reduction | Frozen-head loss reduction | Transport / Exact loss | Rec / Exact loss |
|---|---:|---:|---:|---:|---:|
| 0.85 | 10.40% | 15.37% | 8.89% | 1.3863 | 1.2631 |
| 1.15 | 10.25% | 14.77% | 4.38% | 1.4224 | 1.3601 |
| 0.75 | 12.95% | 18.51% | 16.27% | 1.5338 | 1.2842 |
| 1.25 | 12.62% | 19.33% | 5.38% | 1.5142 | 1.4328 |

## Identity-anchor behavior

At scale1, foreground relative MSE changes from 0.0000000003 to 0.007096. Background changes to 0.005527. Mean loss ratio changes from 0.999998 to 1.002531. Do not report relative MSE percentage change with this near-zero baseline.

The true-flip identity teacher is also its anchor by construction. The unflipped cache/fresh maximum relative L2 is 0.0001103. The unnecessary identity residual is measurable, but this finite diagnostic does not prove it causes the AP gap in randomly scaled training.

## What this supports

- The reconstruction corrector reduces foreground/background feature error and one fixed Online head's loss on held-out images under the eight nonidentity cases. Describing it as complete held-out failure is inaccurate.
- Larger background than foreground improvement motivates a foreground-aware comparison; it does not prove foreground supervision improves AP.
- Scales0.75/1.25 lie outside the training scale interval0.85--1.15; this is transform-range transfer within VisDrone, not cross-dataset transfer.
- Corrected features still incur loss above exact online features. Full-data reference AP remains Correct25 9.7656, Transport25 9.6350, Mix25 11.3526, Online11.3982, seed0 only.
- Next bounded ablation: exact-anchor bypass/identity-preserving correction and foreground-aware or task-aware supervision, each against unchanged reconstruction correction, Transport and Mix with paired queries. Do not infer AP improvements from diagnostic losses.
- Since these validation images inform method design, later claims need untouched target/test evaluation and multiple training seeds; no test-dev images used here.

## Verification and failures

Four feature metric tests pass, including two regressions observed failing before the mask fix. Review verified transforms, frozen state, loss normalization and corrected mask path. Foreground uses all positive-area visible YOLO boxes; detection loss keeps original training filters. Fractional maximum overlap is not exact union area.

Initial startup failed before evaluation because HF_HOME was omitted; failure artifacts are preserved. Retry used the existing offline model cache. GPU1 was unavailable to CUDA; GPU0 used. Global lint has known pre-existing findings, recorded in the reference-evaluation archive. No efficiency benchmark is inferred.
