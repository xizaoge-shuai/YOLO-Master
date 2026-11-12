# Experiment completion and dataset expansion

## Superseding authorization:2026-11-12 minimal single-run paper baselines

User explicitly cancels seed repeats and broad extra baseline/ablation sweeps under deadline. Prioritize real YOLO-Master-N and YOLO12-N, plus the four supplied-paper idea adaptations (FroFA, LOFFTA, ViT-Adapter, Cache and Reuse INT8), then Detail25. One run per dataset/configuration, full current splits,100epochs,seed0 only. Run concurrently on healthyGPU0 when admission allows. Reuse completed VisDrone results; do not repeat already measured cases.

- [x] Stop old seed/VOC/SKU supervisors; preserve every result and checkpoint. Pause SKU C2 at retained72epoch checkpoint; let the already98epoch VOC Mix25 worker finish without launching further old cases.
- [ ] Add direct full YOLO-Master-N/YOLO12-N training and native final-evaluation worker using verified real architecture configurations.
- [ ] Add a minimal seven-case/dataset coordinator:two direct detectors,four supplied-paper adaptations,Detail25; shared two-slot GPU0 admission.
- [ ] CPU contract/source review,scope commit+push,then launch actual full-data background work and verify its first real training log.
- [ ] Replace handoff/status instructions with the seven-case queues and record canceled extras. Full COCO preparation may continue; cross-backbone/diagnostic expansions remain deferred.

## Active authorization:2026-11-10 complete remaining experiments

User requests all missing measurements,parallel execution when resources permit,and baselines before the candidate when stages must serialize. Existing100epoch/augmentation recipes and complete-split execution remain fixed. New source files only:activeVisDrone810/VOC1366 source snapshots must stay byte-identical.

- [ ] Recover SKU preparation using author-loader coordinate policy,retain rawCSV/bboxes and per-row audit; restore unchanged full12-case pipeline.
- [ ] Complete CPU qualitative panels from27GT-selected images and existing predictions/108heatmap arrays,with fixed threshold and failures retained.
- [ ] Add CPU-tested18-case seed continuation waiting for original10seed0 phase and locks;8baseline repeats before10candidate/ablation repeats,using commonGPU0 admission.
- [ ] Add full548 transformed-view detectionAP diagnostic with transformed ignore regions/GT and explicit auxiliary protocol;baseline arms before candidate at inference,teacher calls separately charged.
- [ ] Prepare second-backbone weights and a versioned fullVisDrone adapter without changing running sources;within-backbone baseline-first comparisons.
- [ ] Prepare completeCOCO from official sources,then queue its full-data comparison afterVOC/SKU,with a disk-space gate before large banks;do not duplicate completed datasets or claim subsets as fullCOCO.
- [ ] Queue controlled equal-time and bottleneck measurements behind an actualGPU-isolation gate;keep sharedGPU observations separate.
- [ ] Review sources/contracts and real output,commit+push each concrete milestone;updateHOME handoff with all active/queued statuses and limitations.

## Active task:2026-11-10 parallel preparation and evidence analysis

User now authorizes starting VOC and SKU together and requests claim-relevant additional experiments. Keep100 epochs/current augmentation,baseline-first then the existing candidate,seed0 before independent repeats,fullCOCO last. Do not alter manuscript or frozen VisDrone sources.

- [x] Read supplied recommendations and map against existing writing/figure plans.
- [x] Fresh server audit:VisDrone7 seed0 DONE,Detail0 epoch38;109GiB availableRAM,data disk1.2TiB. GPU1 reports requires-reset;GPU0 is shared with another user's job.
- [x] Isolate pinned pycocotools2.0.11/TIDE1.0.1 dependencies under research worktree; existing training environment untouched.
- [x] VOC07+12 full data sealed; SKUfixed download runs in background; official splits/raw annotations and download hashes preserved.
- [x] Implement new manifest-driven/mmap training and correct VOC/SKU evaluation without editing frozen VisDrone trainer.
- [x] CPU contracts and independent review passed; initial implementation committed36e96ac, guarded parallel admission2b521ae. NoGPU smoke or subset experiment.
- [x] Start two dataset pipelines with explicit resource admission; VOCsnapshot passed thenWAITING_GPU,SKUWAITING_DATA. BrokenGPU1 never selected. Full runs remain asynchronous.
- [x] Run full548-image size/perclass/TIDE analysis for8 arms and extract existing convergence curves; source prediction/checkpoint hashes bound and results committed36e96ac.
- [x] Implement full548 geometry/feature-frequency/full-forward inference diagnostics; final CPU input/GT/source gates and wrapper contracts passed. Execution remains separately tracked.
- [x] Current evidence plan/operations/PROGRESS updated; CPU results and diagnostic sources archived, scoped commits36e96ac/2b521ae/23b9b54 pushed. GPU diagnostic/new dataset results will archive on completion.

## Historical stages (superseded scheduling retained below)

User order: finish existing metrics and controls, archive, reclaim unused storage, then add formal datasets; keep the candidate method unchanged during evidence collection. No GPU smoke or subset training.

- [x] Verify existing full-data evidence and active queue.
- [x] Inspect YOLO-Master and primary benchmark sources.
- [x] Inventory project storage and verify duplicate extraction contents.
- [x] Remove only byte-identical recoverable temporary data.
- [x] Record cleanup and the staged dataset protocol; commit and push.
- [ ] Existing asynchronous work: finish the 28-case full VisDrone queue.
- [ ] Existing evidence still needed: comparable end-to-end costs, equal-time runs, inference overhead, mechanism controls.
- [ ] Latest2026-11-09 order: VOC07+12, SKU-110K, complete COCO2017 last; one dataset at a time on AutoDL.
- [ ] On each new dataset finish baselines before Detail25/ablations; archive evidence before reclaiming unused banks and moving on. Method tuning remains later.

Current implementation scope is storage cleanup and experiment scheduling documentation. New dataset training has not started. No running training source is edited.

Latest status check could not connect: SSH connection reset, then banner timeout on bounded retry. Last confirmed state remains3/28 complete and adapter59/100; this is a historical snapshot, not a fresh liveness claim. Baseline-first plan change saved locally, remote commit/push pending connectivity.

2026-11-09: AutoDL migration is tracked in .planning/autodl-migration-20261109. Source cloned on a separate branch; isolated environment installation and complete VOC preparation run in background. No new training or baseline AP. Campus source checkpoints remain unavailable; current50GiB disk must be expanded before full VOC cache construction.

Superseding user direction2026-11-09: original yang109 is reachable again and is the execution platform. AutoDL experiments no longer advance. Resume audit verified810 frozen sources,3 completed runs and adapter59/100 checkpoint. Original full-data queue resumed at epoch60. Finish necessary VisDrone evidence, then VOC, SKU, complete COCO last, with one dataset active at a time. Root disk34GiB free versus data volume1.3TiB; no data cleanup in this recovery.
