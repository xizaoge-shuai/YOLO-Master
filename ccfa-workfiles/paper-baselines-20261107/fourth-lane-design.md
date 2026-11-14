# Fourth training lane: VOC LOFF-TA

User authorization, 2026-11-14: add one or two concurrent existing baseline/candidate jobs and collect all derivable metrics together. Add one job now, subject to admission; keep the existing three workers alive. This supersedes the previous three-worker scheduling cap only. Full splits, seed0, final100, batch8 and640 remain unchanged.

## Resource evidence and choice

At11:04UTC, GPU0 had14409MiB free and100%utilization; host MemAvailable19342MiB, swap8191MiB used. VOC FroFA46/100, SKU INT837/100 and SKU Detail25 8/100 were advancing. Subsequent vmstat showed swap-in32–36KiB/s and no swap-out. Most VOC resident pages were file-backed; adding a pure cached VOC method can share the existing feature bank. This supports a guarded fourth lane, not two additional workers.

The fourth worker is the already-authorized VOC `loffta-c2-n005`. Alternatives were another online-backbone method, which adds more private memory/teacher work, or five workers immediately, which leaves too little host reserve. No new method, seed, hyperparameter, GPU1 use or change to another user's process is authorized here.

## Scheduling contract

- Add `online_experiments/priority_voc_overlap.py` and its CPU-only contract tests. Import established process identity/pidfd helpers; do not modify the running three-lane controller, original trainer, dataset, cache or frozen source anchors.
- Verify the exact VOC supervisor644301 and FroFA child1467410 using UID, boot ID, start ticks, argv, canonical job/output and frozen plan. Signal only the exact old supervisor through pidfd. Adopt the existing child; never signal a process group or restart training.
- Preserve an immutable handoff record before signaling. Recovery may adopt that same child only after proving the old supervisor has exited; recovery sends no signal.
- New supervisor owns the canonical VOC pipeline lock. Adopted FroFA finishes, validates and archives. The extra LOFF-TA worker runs in parallel, validates and archives once. Wait for both before continuing the remaining original INT8/Adapter/Detail sequence, so no duplicate LOFF-TA launch is possible.
- Extra admission uses the existing brief global GPU gate, new slot3, a maximum of four held slots/workers, free GPU>=12288MiB and host MemAvailable>=18432MiB. Pass the slot FD to the trainer. Existing SKU/VOC baseline slots0/1 and SKU Detail slot2 are preserved.
- Extra output is the original canonical VOC `train/loffta-c2-n005-s0`; logs/status live in a separate fourth-lane sidecar directory. Existing CPU metric watcher discovers this output without retraining or inference.
- On error preserve checkpoints, children and logs. Do not automatically restart failed training or launch replacement seeds. Shared-GPU elapsed time remains observational.

## Implementation and verification plan

- [x] Check-only command proves frozen plan/process identities without signaling or training.
- [x] CPU contracts cover PID/argv mismatch, supervisor-only signal, recovery without signaling, slot/resource guards and duplicate-method avoidance. All27Linux tests passed.
- [x] Independently review new scheduler; new CPU tests and targeted lint/format passed on the server. Repository-wide existing failures and missing codespell logged; no unrelated fixes.
- [x] Commit/push new source and design before starting the detached supervisor:068d9f2.
- [x] Verify the original three PID/start identities remain unchanged, the fourth canonical output has exactly one owner, slots0/1/2/3 are held, and real training activity begins. Original1366-source anchor unchanged; actual CPUtick/pagefault growth and pmon activity recorded at11:20UTC.
- [x] Archive launch/resource evidence and refresh handoff/progress for scoped commit/push through the existing archive lock.

## Single prediction, complete metrics

Keep the existing CPU-only single-pass watcher. Each trainer saves its final full-test predictions once. Derive native primary AP/mAP and labeled auxiliary AP/AP50/AP75/AP-small/medium/large, AR/maxDet, per-class/area results and P-R tensors from those predictions. Reuse actual training curves, query journals, measured resource/storage/parameter sources. Inspect for missing error analysis and add only separately versioned saved-prediction CPU analysis when needed. Do not invent latency, energy, controlled GPU-hours or equal-time comparisons from shared-GPU logs.

Actual launch19:17CST:supervisor2300793 and VOC LOFF-TA child2300847,alongside the original three PIDs. Final live/source/lease evidence is saved under `research/checks/fourth-lane-20261114/launch-verification.json`; runtime snapshot under `research/experiments/priority-four-lane/job-20261114-voc-loffta/`. First epoch/final AP are not completed results.
