# Experiment completion and staged dataset expansion

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
