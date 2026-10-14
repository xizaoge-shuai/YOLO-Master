# Frozen-head feature/assignment intervention, 2026-10-14

Status: protocol fixed before new diagnostic results. This is mechanism exploration,
not a new training method or an AP benchmark.

## Question and design

Does transporting cached features change task-aligned assignments and detection
gradients? Does the completed reconstruction corrector repair those changes?

Use the completed full VisDrone job `job-start-eDmH0odX`, final epoch-100
Online100 seed-0 head, original DINOv3-S/16 identity, mean_value fusion, and the
final Correct25 seed-0 reconstruction network. Freeze all parameters and BatchNorm.
Do not alter training, checkpoint, cache, or old provenance files.

For each image, flip in {false,true} and centered scale in {1,.85,1.15}, compute
fresh exact features and two approximations: transport from the real corresponding
flip anchor, and the existing learned residual correction. Preserve the existing
FP16 feature interface. Match images, transformed labels, head and anchor grid.

For each branch (one2many and one2one), evaluate all four cells:

| Prediction features | Assignment donor |
|---|---|
| Exact | Exact |
| Exact | Approximate |
| Approximate | Exact |
| Approximate | Approximate |

Two distinct target protocols must remain separate:

1. `native_soft`: transplant the complete native assignment, including target
   confidence weights. Use that donor's native score-sum denominator. This measures
   the combined matching/weighting effect, not isolated hard matching.
2. `hard_common`: transplant the foreground mask, matched GT boxes and class;
   use one-hot unit target scores on positives and a common denominator equal to
   max(1, exact-assignment positive count). This controlled surrogate separates
   hard matching from quality-score changes. It is not the native training loss.

Loss components: box, classification, DFL, and sum, computed per image (batch=1).
The present reg_max=1 head has no DFL term; report the zero, do not infer a DFL
mechanism. Logit gradients are with respect to raw box-regression outputs and
classification logits, separately. They are not parameter/feature gradients.
Compare their cosine, relative L2 error and norm ratio to Exact/Exact in the same
target protocol. Do not combine gradient coordinates with incompatible units.

Assignment metrics: foreground Jaccard; GT-switch fraction on shared positives;
positive counts; GT coverage. Size strata use visible transformed boxes in the
640-pixel input (small <=32^2, medium <=96^2, large >96^2), not official AP_S.
Empty unions/zero-norm cosine are missing, not arbitrary perfect scores.
Keep all per-image measurements and aggregate images equally, retaining flip/scale.

## Fixed stages and scope

1. CPU tests: native-loss and gradient parity for reg_max 1/4, empty GT, identical
   four cells, target-label selection, soft-score independence under hard_common,
   finite outputs and matched-GT metrics. First observe missing-code test failure.
2. GPU smoke: first two training examples from a deterministic seed-20261014
   permutation. Six views, both approximation methods. Check fresh/cache identity
   relative L2 <=.01 and native-loss component parity <=2e-5 absolute/relative.
3. Development: first 128 training images from that same permutation; no tuning or
   checkpoint update. Training-image diagnostic is explicitly not held-out evidence.
4. Confirmation: all 548 previously used validation images, same fixed protocol.
   This validation set has already informed research; do not call it untouched test.

The smoke controls execution; development results do not select images, views,
checkpoints or metrics for confirmation. Start with the Online100 head only; a
second frozen-head replication is a later explicitly recorded experiment.

## Integrity, runtime and artifacts

- Snapshot all loaded repository Python sources, manifest hashes, checkpoint hashes,
  Git revision, environment, selected sample IDs, teacher image count and seeds.
- Verify original cache manifest/data identity; check used cache shapes and sizes;
  freshly compare real identity views. Record that whole large cache hashes are
  inherited from the original manifest rather than recomputed. Never rewrite them.
- Zero optimizer updates; assert model/corrector state unchanged and source/checkpoint
  hashes unchanged at completion. Independent assignment tensors must be detached.
- GPU1 allocation currently fails. Use GPU0 batch=1, four CPU threads, existing
  caches only, small CSV/JSON outputs; no performance/speedup claim under contention.
- Independent code review before full execution. Preserve failed outputs, new job
  directories for retries. A launcher provides start/status and never duplicates an
  active job. Commit explicit code and lightweight results, no weights/caches or push.

## Interpretation gates

Assignment disagreement alone is not proof it caused the AP gap. Lower surrogate
loss or improved gradient agreement is not AP recovery. If teacher-assignment
intervention improves gradients only under native_soft, investigate quality weights
before claiming hard-assignment repair. If it also improves hard_common, it supports
a controlled local mechanism; a matched-budget training intervention is still
required. A null result is retained and should redirect the method, not be hidden.
