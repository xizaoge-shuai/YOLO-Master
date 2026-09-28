# Full-data identity/foreground ablation plan

Execution authorized by user on 2026-09-28. Three new full VisDrone arms,
6471 train / 548 val, seed0, batch8, epochs100, exactly161775 online images.
Reuse completed full-data Mix25, Transport25 and Rec25 as labelled historical
accuracy references. Do not infer controlled speedups from historical timings.

Reuse the candidate code already present on the server (pilot job-start-LIz9kPc6):
gate(scale)=min(1, abs(log(scale))/abs(log(1.15))). No new gate tuning.
- gate25: original global reconstruction loss, identity-gated residual.
- fg25: original ungated residual, 0.5 global + 0.5 foreground relative MSE.
- gatefg25: both changes.
All use50560 parameters. Empty foreground falls back to global loss.
Use all positive-area visible training-box fragments for foreground masks;
loss labels retain historical target filters. This differs from the existing
candidate pilot's use of filtered boxes and is recorded explicitly.

Keep existing trainer/evaluator source untouched. Add a full-data batch adapter,
fixed variant specification, serial background campaign and per-arm reference
evaluation/archive. The current server budget_corrector.py changes must be
reviewed and recorded as a new source version; do not overwrite historical
provenance to make its old hashes pass. New campaign snapshots actual sources.

Checks before formal launch:
- Focused gate/loss tests, paired schedule test, empty foreground, gradients.
- 17/8 smoke for all new arms.
- New rec25 adapter vs original default two-epoch weights must match exactly.
- Interrupted gatefg25 must resume to the same two-epoch weights/counters.
- Reference evaluator hashes must match the previously verified port.
- Source and dataset/cache hashes, complete raw validation coverage.
- Scoped code checks, required global checks, read-only review, scoped commits.

Each formal arm finishes100 epochs, exports final/custom-best predictions,
and scores both under the pinned reference protocol. Final reference AP is
primary; custom-best is secondary. Archive lightweight evidence and failures;
exclude weights/caches/datasets. Preserve background telemetry and checkpoint
resume. No test-dev or extra validation queries for corrector training.

No AP improvement is assumed. One-seed full-data results are method screening,
not a stable advantage. Current pilot gate and gate+foreground final AP means
4.0630/4.2384 did not exceed original Rec25 4.2809.
