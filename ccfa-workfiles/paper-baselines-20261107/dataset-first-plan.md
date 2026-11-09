# Dataset-first scheduling implementation plan

> **For agentic workers:** implement the independent coordinator and review the process handoff in parallel. Preserve every frozen training source and existing result identity.

**Goal:** Complete each dataset's seed0 comparisons and necessary ablations before returning to seed1/2 repeats; keep full COCO last.

**Architecture:** Add one external coordinator. Read the immutable28-case queue, run its10 seed0 cases through the original training/evaluation entry, and save18 remaining repetitions as deferred. Adopt the already-running seed0 child without restarting it; no edits to original queue, provenance, model, optimizer or evaluation implementation.

**Tech stack:** Python standard library subprocess/procfs/flock and existing frozen training/archive helpers.

## Authorized design and scope

- User explicitly selected dataset experiments before repeated seeds. No new confirmation or changed scientific hypothesis is needed.
- Keep current adapter training alive. Inspect exact UID, argv, PID starttime and inherited file descriptors before stopping only the old supervisor and launcher.
- New coordinator acquires both original campaign locks and its own phase lock. No process-group signals, pkill or SIGINT. Record the handoff and old checkpoint.
- A completed child is accepted only after DONE plus full frozen validation, never by PID disappearance alone. Stop phase on failure; never start a seed1/2 run in this phase.
- Original28-case completion/archive is not produced by the10-case phase. Status must say dataset-first seed0 phase, with repetitions deferred.
- VOC, SKU and COCO training interfaces are not yet fully wired. The coordinator must report that boundary honestly; no fake automatic full-dataset training or success marker.

## Implementation tasks

- [x] New online_experiments/dataset_first_campaign.py: immutable plan selection, strict child identity/adoption, lock ownership, seed0-only dispatch, existing validation/quantization/archive reuse, explicit phase status.
- [x] CPU tests for seed filtering, PID/starttime/argv verification, DONE/validation acceptance and lock refusal; AST/CLI/Ruff. No GPU smoke or subset experiment. Frozen810 files unchanged; source SHA b335c28eb5379246a6d0821047e5fa8f0359c847b0012902b5b663598203ee5f.
- [x] Independent coordinator and supervisor-only handoff review. Recheck live descriptors/identities at the actual handoff, not against historical adapter PID.
- [ ] Commit code and records before changing process ownership. Controlled handoff, then verify current child remains same PID and epoch counters continue.
- [ ] Update operations/dataset plan: seed0 across VisDrone,VOC,SKU,fullCOCO first, then repetitions. Preserve already-completed seeds.
- [ ] Scoped commit/push handoff progress and exact phase status.

## AP interpretation check during implementation

User asks whether low AP is abnormal. Literature YOLO-Master uses600 epochs,batch256,SGD,strong image augmentation; project detector references use100,batch8,AdamW,Mosaic/HSV/Copy-Paste disabled. DINO cached candidate is a different frozen-backbone/head architecture. These explain a real comparability gap, not proof that the entire observed accuracy loss is harmless. Keep published/reference and project matched-protocol numbers separate; no model retuning or600-epoch run is authorized merely by this check.

## Latest execution constraint

User explicitly says to finish experiment data first, then add epochs/augmentation. Keep100 epochs, the existing augmentation distribution, optimizer, learning rate and final-epoch endpoint unchanged. At the fresh read,7 seed0 cases were DONE and detail0-s0 was running; the completed adapter process must not be adopted using its obsolete PID.
