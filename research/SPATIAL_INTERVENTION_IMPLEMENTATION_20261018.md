# Spatial intervention implementation plan

The user approved `NEXT_INTERVENTION_PLAN_20261018.md` and explicitly authorized push on 2026-10-18. Implement the first-priority spatial intervention now. The matched-energy study depends on its results and is not silently added to this campaign.

**Goal:** Run a paired six-arm, two-head diagnostic on the same fixed 128 training images, including parameter gradients, with immutable provenance and verified result archiving.

**Architecture:** New pure helpers define disjoint spatial masks, oracle replacements and parameter-gradient summaries. A separate runner reuses the existing cache, preprocessing, checkpoint identity checks and native loss utilities; existing experiments remain unchanged. A launcher runs one two-image integration smoke and then the fixed 128-image development study, with scoped archive/commit/push after successful verification.

**Stack:** Existing yolomaster Python/PyTorch environment, CSV/JSON, Bash/nohup, Git origin feature branch.

## Implementation tasks

- [x] Add failing contracts in `online_experiments/test_spatial_intervention.py`: masks partition every feature token; identity produces no added padding; flip and clipped visible boxes; patch-edge mixing; oracle mask locality; cheap padding preserves constant features/identity; parameter gradients agree with direct autograd, including disconnected one2one shared parameters.
- [x] Implement `online_experiments/spatial_intervention.py`: 640-pixel valid-content and canvas support warped by the same affine rule, area-pooled to 40 tokens; compare pixel and feature support to mark mixed boundaries separately. Valid foreground means patch overlap with any visible GT fragment. Output five disjoint masks: new padding, old letterbox, foreground, background, mixed. Unit tests must pass before GPU work.
- [x] Add `online_experiments/diagnose_spatial_intervention.py`: same seed 20261014 training IDs, 6 transforms, heads online-geom and mix25, six arms exact/transport/rec/padding_oracle/content_oracle/blank_fill. Extract once per image/view for both heads. A separate gray-image forward creates the blank template and is charged explicitly. Save regional SSE/energy, native loss/matching, hard-common exact-assignment output and parameter gradient metrics.
- [x] Preserve eval mode and one2one detachment. Request parameter gradients without optimizer steps; divide parameter groups into p3/p4/p5 and each detection branch/task. Emit NA for zero reference gradients with norms/counts. Keep the existing reg_max=1 auxiliary L1 in localization loss. Check native-loss parity and unchanged state hashes.
- [x] Add `online_experiments/run_spatial_intervention.sh` and `archive_spatial_intervention.py`: unique job, lock and latest pointer, smoke then development, hash/row/grid checks, scoped small archives, no raw images/caches/weights. Stop after development; do not auto-run validation or a new candidate.
- [ ] Run tests, targeted lint/format/spelling, repository-required checks; obtain focused reviewer feedback and resolve important issues. Run a two-image smoke before the full 128-image diagnostic. Capture failures without overwriting them.
- [ ] Commit explicit paths, launch background work, verify a real processed image and log the job path in PROGRESS. Push the existing branch to origin only, without force. Record result-push errors separately from computation and archive state.

## Exact statistical scope

The natural-size spatial oracles do not control region area or energy; those are reported explicitly, and causal localization claims await the separate matched-energy stage. The six views contain two exact anchors; keep identity rows. Parameter-gradient comparisons are conditional eval-state diagnostics, not actual optimizer trajectories. The test set is not touched. No new AP or efficiency claim is produced.

## Pre-launch verification and limits

- Targeted unit/regression/archive tests: 23 passed, with four temporary-directory cleanup warnings. Region partition, identity support, oracle locality and disconnected gradients are covered; full detector integration is the two-image smoke gate.
- All four new Python files pass Ruff check/format and targeted codespell; Bash syntax passes. Repository-wide checks retain the previously recorded 2768 Ruff failures and five formatting files plus parse errors, outside these new files. Full codespell is bounded to 45 seconds; exact result is retained with the verification archive.
- A separate review agent was requested; it inspected initial code and equal-image aggregation but stopped on an account usage limit before delivering a final review. This is not an independent audit pass. Manual review checked native-loss reuse, parameter graph connectivity, one2one detachment, separate blank-query charging and source/grid integrity.
- Runtime smoke and development status are recorded separately in PROGRESS and the job archive. This protocol is frozen before launch; do not edit it merely to update completion checkboxes after the run starts.
