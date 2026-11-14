# Experiment completion and dataset expansion

## 2026-11-14 user-authorized third GPU lane

- [x] Inspect actual resources:GPU0free18865MiB,hostMemAvailable~18GiB;onlytwoourGPUworkers,otheruseruntouched. Select existingSKU Detail25 as thirdlane;no new method/seed/epoch.
- [x] Design separateSKU-supervisor takeover that adopts liveINT8 unchanged,keepsVOC untouched,runsDetail concurrently,andjoinsitsresultinstead ofduplicateDetail. Preserve frozen training sources;briefglobaladmissiongate andnewslot2.
- [x] Implement independentSKU-supervisoradoption andthirdslot;17serverCPUchecks andtargetedRuffpassed,actualreadonlypreflight/recoveryverifiedoldsupervisor641892/INT8child1881349 andfrozenplan. NativeCondaos.pidfd_openmissingresolvedusinglibc+realselfsignal0check;independentreviewpassed. Firsthandoffexit-argv racefixedwithstrictno-signalrecovery;originalfailurepreserved. Sourcepublicationccc0e22/642301b precedeslaunch.
- [x] Actualthirdlane started: newcoordinator2105130 andDetailGPUchild2105170;oldVOC1467410/start42814294 andINT81881349/start44524918unchanged. Teacherqueryjournalstarted;originaltrainers,automaticvalidation/archive/CPUmetrics remainwired. Record changedconcurrency andpublishhandoff;firstepoch/finalmetrics pending.

## 2026-11-14 cached-training throughput diagnosis

- [x] Verify actual100epochs,batch8,640 andfull16551/8219 unique-sample plans;VOC32/SKU14-15continue,no training deadlock,repeatedcachebuild orper-epochfulltest.
- [x] Sampleper-processGPU,pidstat,I/O,memory andswap rates. Correct prior inference fromwholeGPU100%:otheruserSM61-84%,ours~12-24%;diskandmemorynotseverelyblockedinsample.
- [x] Independentlyauditfrozeninputandtrainingpaths:syncmmap/CPUcopy/H2D,noDataLoaderprefetch;INT8CPUdequantabout15%inclusiveelapsed,perbatchGPUCPUsynchronization. Relativecausalweightsnotprofiled.
- [x] Recorddiagnosis,rawobservations,sourcehashesandproposedpatchboundary. No training source/recipe/process change ornewGPUwork;no performancefixclaimed.

## 2026-11-14 16:05 live status refresh

- [x] Confirm activeVOC FroFA31/100 andSKU INT813/100;GPU0healthy100%utilization,approximately856GiBdiskfree. Current active training logs have no new failure.
- [x] Verify newly completedSKU LOFFTAfinal100 and nativeAP@10032.4844/denseAP@50042.3587;sealedCPUmetrics6/14complete,0failed. VisDrone7/7,VOC2/7,SKU4/7corecasescomplete.
- [x] Fix cost exporter's fixed old completion list to discover sealed results across the authorized7methods×2datasets;require100epochs and retainallSHAchecks. Actual19rowexport preservesprevious18rows exactly;targetedRuff and independentreviewpassed. No training/inference/evaluatorcall.
- [x] Refreshcost/mAPexports,handoff,currentcaseETAandscoped publication;fullCOCOdeferred andcurrentmethodCOCO800/200notstarted. Active frozen training sources unchanged.

## 2026-11-14 COCO subset scope update

- [x] User defers full COCO under deadline; preserve prepared assets and existing VOC/SKU queues. No full-COCO training process is running.
- [x] Re-read original COCO800/200 split, cached-latent summary/100-row curve and YOLO26n-scratch100-row curve. Keep final100 and validation-selected best separate; historical COCO128 predictor is another method/split.
- [x] Record fixed800/200 as the reusable COCO subset identity, not a result-selected redraw. Current Detail25 and seven core methods have no completed results on this split; no old-method score is substituted.
- [ ] Run current core methods on the fixed subset with matched data/metrics after integration; this is pending work, not a completed external validation.

## 2026-11-14 cost-accuracy evidence and D1 requirements

User asks whether training cost is measured and how to demonstrate an efficient accuracy tradeoff. Preserve the current single-seed full-data queue; no training restart, extra seed, or new evaluation forward pass.

- [x] Read original D1 task clauses:accuracy/VRAM/GPUhours,atleast2datasets,and>=50%trainingcost reduction onatleast1dataset;separate oldsubset results from full-data evidence.
- [x] Verify latest queues:VOC2/7complete,FroFA7/100;SKU3/7complete,LOFFTA68/100;CPUmetrics5/14complete,0failed;VisDronecore7/7andINT8preparation2/2complete,COCOrawdata prepared.
- [x] Audit current costs and strong baselines:18completed configurations available;retainAdapter0 andAdapter25,rawsharedGPUtime,queryimagesANDcallcounts,andcold/warm bookkeeping scopes. No controlledequal-time results yet.
- [x] Add CPU-only cost exporter and realCSV/provenance;math,missingvalue,andindependentdefinitionchecks passed. DefineAPgap/costratio/CostAtGapwithfixedreference/thresholds,nocustomscore chosen to guaranteevictory.
- [x] Final source/realdata verification passed; running frozen sources unchanged,18realrows and missing controlled costs checked. Handoff and lightweight exports refreshed; scoped publication is recorded in research/PROGRESS.md. Controlledcost/equaltime remainfuturework,notclaimedcomplete.

## 2026-11-13 baseline comparison and CPU parallel completion

User requests the remaining baseline comparisons and all derivable metrics together, with parallel work wherever resources permit. Retain the authorized seven-method single-seed scope, full splits, final100 protocol, and current workers. Do not alter frozen sources or duplicate training/inference.

- [x] Verify current GPU/CPU/RAM/I/O resources and completed baseline coverage: two active GPU0 jobs,100%GPU utilization, GPU1 unhealthy; CPU~85%idle, MemAvailable~47GiB, I/Owait0–1%. Keep current two GPU leases; existing jobs continue.
- [x] Recover full548 VisDrone seven-method AP table and honest Detail differences; identify missing auxiliary evidence for FroFA/LOFFTA/INT8/realMaster/YOLO12 beyond the older eight-arm analysis.
- [x] Add and launch a CPU-only low-priority, singleton INT8 preparation sidecar for the two existing dataset banks; preserve original banks/seals and include build_seconds in future cold costs. PID1276050 verified PREPARING_CPU;VOC offset876/16551 atfirst check,then SKU automatically.
- [x] Add and launch full548 CPU metric completion from the existing seven-method predictions; export complete tensors/PR/class/area evidence, compute only missing TIDE results and reuse completed analyses. PID1276051 verified RUNNING_CPU on realMaster;no GPU inference or retraining.
- [x] Verify unchanged running-source hashes, actual sidecar processes/progress, runtime autoarchive/commit/push, and refresh the existing handoff/status report. FourCPUcontracts+Ruff,1366/1370anchor checks and independent reviews passed;source pushed1f8fa1f. Runtime autoarchives sharepriority/archive.lock;full asynchronous outputs remain pending.

## 2026-11-12 single-pass metric completion

User requests all derivable metrics together, avoiding second training/evaluation passes under deadline. Current direct/cached training and priority coordinator source hashes remain fixed. New CPU-only sidecars consume the one persisted final full-test prediction set per configuration.

- [x] Export native primary consistency, auxiliaryCOCO AP/AP50/AP75/size/perclass/AR,rawPR/score tensors and actualtrain/lr/time curves from saved predictions/logs only. Full4952-image completed VOC C1 reference verified without inference; nativeAP50 matches exactly and100training rows exported.
- [x] Reuse measured cost/storage/VRAM/query/parameter sources with explicit scopes; filter dummy zero validation columns and do not use cached validation as full-imageFPS. Identity-bound frozen teacher21,596,544parameters counted once on CPU; actualhead/native-nc sources checked.
- [x] Start an idempotent CPU-only watcher for the14 authorized VOC/SKU endpoints,archive metric bundles without changing training jobs or triggering GPU passes. PID661685 verified alive;14cases WAITING_RESULTS,0failed atlaunch. NativeVOCMaster/SKUYOLO12 continued epoch3/13 during verification.
- [x] CPU contracts,independent metric/source review,scoped commit+push,and update handoff with one-command bundle status.16CPUcontracts+targetedRuff passed;fullVOCsaved-prediction check passed;sources/docs/evidence pushed asf1f4956;launch evidence archived separately. Metric results remain asynchronous,not claimed complete.

## Superseding authorization:2026-11-12 minimal single-run paper baselines

User explicitly cancels seed repeats and broad extra baseline/ablation sweeps under deadline. Prioritize real YOLO-Master-N and YOLO12-N, plus the four supplied-paper idea adaptations (FroFA, LOFFTA, ViT-Adapter, Cache and Reuse INT8), then Detail25. One run per dataset/configuration, full current splits,100epochs,seed0 only. Run concurrently on healthyGPU0 when admission allows. Reuse completed VisDrone results; do not repeat already measured cases.

- [x] Stop old seed/VOC/SKU supervisors; preserve every result and checkpoint. Pause SKU C2 at retained72epoch checkpoint; let the already98epoch VOC Mix25 worker finish without launching further old cases.
- [x] Add direct full YOLO-Master-N/YOLO12-N training and native final-evaluation worker using verified real architecture configurations.
- [x] Add a minimal seven-case/dataset coordinator:two direct detectors,four supplied-paper adaptations,Detail25; shared two-slot GPU0 admission.
- [x] CPU contract/source review,scope commit+push,then launch actual full-data background work and verify its first real training log:VOC realMaster epoch1,SKU realYOLO12 epoch2,parallelGPU0; twelveCPUcontracts passed. A zero-step native initialization retry was preserved after font/network waiting; no repeated training or source change.
- [x] Replace handoff/status instructions with the seven-case queues and record canceled extras. Full COCO preparation may continue; cross-backbone/diagnostic expansions remain deferred.

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
