# 实验展示计划速览

当前：400/100 小规模 baseline 与固定预算试验已完成；全量 VisDrone 6471/548 的六组、单种子实验正在运行。正式评估、离散缩放缓存、最终校正方法和跨域全量验证仍待补齐。

正文建议 4 张实验图（10 个子图）+ 3 张表；补充材料 4 张实验图（16 个子图）。另设 1 张方法示意图。所有拟展示数据必须来自真实运行；Final candidate 只是待确定的方法标签。

主结论依次回答：精度—总成本、固定预算收益、任务相关机制、跨数据集与骨干。先冻结指标与评估，再复用 checkpoint 补评估；本计划不修改正在运行的训练源代码。

# D1 cached-feature detection: experiment display contract v1

Date: 2026-09-26. Scope: prospective experiment and reporting plan; not a claim that the proposed method succeeds.
Status was checked directly on the server. Historical results keep their original protocol labels.
This plan changes documentation only. The running campaign's training source and configuration are unchanged.

## 1. Current state and evidence

We have completed small-scale baseline screening and the first fixed-query correction experiment.
We have started full VisDrone training; we have not completed a full benchmark or fixed the final method.

| Stage | Actual evidence | Remaining work |
|---|---|---|
| Real multi-view cache | C1 original and C2 original + real horizontal flip, 3 seeds | C6 real scale views not run |
| Matched-query pilot | Mix / Transport / reconstruction-only Correct at q=10%,25%, 18 complete runs | Correct has no stable AP advantage over Transport; held-out mechanism diagnosis needed |
| Full data | 6 arms, 6471 train / 548 val, seed 0, 100 epochs, currently running | Official evaluator, 3-seed final comparison, additional baselines |
| Equal total time | No completed controlled experiment | Include cache build, I/O, evaluation and checkpoint overhead |
| Generalization | Early COCO128 and VisDrone small experiments | Full COCO; source-frozen correction transfer; second backbone |

Active job: runs/paper/full-visdrone/job-start-eDmH0odX.
At verification: RUNNING train/cache1 epochs=100. Six one-epoch profiles finished; their AP is only profiling output.
Cache build: 303.6 seconds; parity maximum 0.000205; integrity check: 7019 images.
Preflight estimate: approximately 14.54 hours for the six 100-epoch training/validation arms under the profiled load. This is not a completion guarantee.
Arms: cache1, cache2, mix25, transport25, correct25, online-geom.
All use DINOv3-S/16, layers [3,7,11], 640 input, mean_value head. The full campaign uses streamed FP16 feature storage.
The present corrector has 50,560 parameters. Do not describe it as the early 231,040-parameter predictor.
Evaluation is currently custom DetMetrics, confidence 0.001, max_det 500, without official ignored-region handling.
The current checkpoint selected as "best" uses this custom AP.

### Data actually measured

| Dataset / experiment | Training / validation or diagnostic scale | What it establishes |
|---|---|---|
| Recent VisDrone pilots | 400 / 100, from a 500-image subset of the official training split; seeds 0,1,2 | Controlled preliminary comparisons, not full benchmark AP |
| Early COCO128 predictor experiments | 80 / 20; calibration 10; seeds 0,1,2 | Small per-dataset experiments only |
| Early VisDrone predictor experiments | 400 / 100; calibration 50; seeds 0,1,2 | Small per-dataset experiments only |
| Equivariance diagnostics | 100 images per dataset for COCO128 and VisDrone | Feature error / residual structure, not full detection performance |
| Oracle / task-sensitive diagnostics | Typically 100 evaluation images; calibration budgets 5,10,20,50 across experiments | Mechanism and oracle evidence, not deployable performance |
| Current full VisDrone campaign | Official train 6471 / val 548; seed 0 | Full-data matched-head comparison in progress; evaluator still diagnostic |
| VisDrone test-dev | 1610 available; unused in current method development | Reserved holdout after configuration freeze |
| COCO train2017 / val2017 | No full experiment completed | 2026-09-25 inventory: train images and train annotation JSON absent; 5000 val images present |

Directory names such as "unified-crossdataset" do not prove that source-trained weights were frozen and transferred.
The early unified predictor has 409,984 parameters; the earlier hidden-256 L3 predictor has 231,040. Preserve these implementation distinctions.
Old "Transport near random" results used a fixed-head feature substitution diagnostic. Current Transport trains the head with transported views.
They answer different questions; do not join their AP into one comparison curve.

### Completed baseline families

- Frozen cache-only C1; real-view C2; online-none / online-hflip / online-geom.
- Mixture with online queries q=10%,25%,50% in the mean-sweep/retry campaign.
- FroFA-inspired detection adaptations: C1/C2, strengths 0.1 and 0.3.
- LOFF-TA-inspired detection adaptations: C1/C2, noise 0.05.
- Transport C1/C2; matched-query Mix / Transport / reconstruction Correct at q=10%,25%.
- Head fusion probe: router_only, deepest_value, mean_value (one seed).
- Earlier residual rank, calibration size, predictor width and positional-input diagnostics.

The FroFA / LOFF-TA arms reproduce ideas in this detector, not the original paper's complete experimental protocol.
No completed full YOLO-Master-N reference, C6 scale cache, full COCO, cross-backbone method comparison, or controlled equal-time run is present.

Budget pilot evidence (AP points, mean of three seeds):

| q | Mix best | Transport best | Correct best | Correct minus Transport best | Correct minus Transport final |
|---|---:|---:|---:|---:|---:|
| 10% | 3.6920 | 4.0984 | 3.8714 | -0.2270 | -0.1067 |
| 25% | 4.1635 | 4.5306 | 4.4190 | -0.1116 | +0.0391 |

A reduction in training-query reconstruction error does not establish held-out AP improvement.
Transport-p25-s1 has an anomalous slowdown affecting training and validation. Preserve it; remeasure controlled costs instead of deleting it or claiming a speedup.
Evidence: runs/paper/budget-corrector/job-start-7U9Q7DSc and research/experiments/budget-corrector-job-start-7U9Q7DSc.

## 2. Benchmark comparison rules

YOLO-Master's supplied CVPR 2026 PDF, page 6, reports YOLO-Master-N COCO AP 42.4 and VisDrone AP 19.6.
Its described recipe uses 600 epochs, 640 input, batch 256, SGD and image augmentations including Mosaic, CopyPaste, RandomAffine and HSV.
These are author-reported values, not results measured by the current D1 scripts.

A different backbone or training recipe does NOT prohibit an end-to-end detector comparison.
Direct numeric ranking requires the same dataset split, category mapping, annotation/ignore policy and evaluator.
Different training recipes may be compared as complete systems with their real data, pretraining and compute disclosed.
To attribute improvement to the correction module, additionally hold backbone, head, data, initialization, augmentation/query schedule and optimization budget fixed.

The current pilot is 400/100 and uses a custom evaluator; full current training still does not implement official VisDrone ignored regions.
Therefore subtracting current AP from the paper's 19.6 is not a valid measure of our method's improvement or deficit.
"Paper-reported AP" and "AP from an official dataset evaluator" are separate concepts.
The PDF alone does not establish the exact evaluator revision or ignored-region treatment used for its VisDrone number.

Two reference rows are planned:
1. YOLO-Master-N as a locally measured full detector under the common evaluation protocol, with complete recipe and cost.
2. Author-reported YOLO-Master-N, separately marked as reported, with no paired delta or significance test against our runs unless protocol equivalence is verified.

Fresh-image inference latency must include DINO preprocessing + backbone + head + postprocessing.
Cached-feature validation latency is not deployable end-to-end inference latency.
DINO pretraining cost is outside downstream adaptation cost, but checkpoint provenance and pretraining data must be disclosed.

Primary sources:
- VisDrone evaluator: https://github.com/VisDrone/VisDrone2018-DET-toolkit
- Ignored regions: https://github.com/VisDrone/VisDrone2018-DET-toolkit/blob/master/utils/dropObjectsInIgr.m
- COCO evaluator: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
- Local paper: Lin_YOLO-Master_MOE-Accelerated_with_Specialized_Transformers_for_Enhanced_Real-time_Detection_CVPR_2026_paper.pdf

## 3. Fixed reporting and metric contract

### Accuracy

- Primary endpoint: final checkpoint at the predeclared training budget, dataset bbox AP averaged over IoU 0.50:0.05:0.95, expressed on 0-100 scale.
- Secondary: AP50, AP75, AP by object size; best-validation AP and last-ten-epoch mean as stability diagnostics.
- The change to a final-checkpoint primary endpoint is declared before current full-run results are available. Historical pilot best/final values remain unchanged.
- VisDrone primary: official 10-class evaluator, original annotations and ignored regions, maximum 500 detections. Validate any port against the reference implementation before calling it official-equivalent.
- COCO primary: pycocotools bbox evaluation, maxDets 100; AP_S / AP_M / AP_L use original-pixel area thresholds 32^2 and 96^2.
- VisDrone small-object extension: explicitly name "COCO-style AP_S on VisDrone, maxDets=500". It is not an output of the official VisDrone toolkit. Freeze conversion, ignored-region treatment and area definition; test them before reporting.
- A later official re-score of the saved custom-best checkpoint is "official AP of the custom-selected checkpoint", not "best official AP over all epochs".
- Re-evaluate saved final predictions or checkpoints without retraining when possible. Predictions with a different threshold or required output set may require inference again, not training again.
- Test-dev is used only after method/hyperparameters are frozen. Do not optimize against test-dev AP.

### Cost

- Primary cost: first-use allocated-device hours for the full adaptation job = allocated GPUs x elapsed seconds / 3600, from cache/model setup through training, scheduled validation and artifact writes; include I/O and correction overhead, exclude scheduler queue.
- Single-GPU wall time and allocated-device hours are both retained. GPU utilization is supplementary telemetry, not a multiplier defining billed time.
- Report cache build, validation and total wall separately; train time already contains correction time, so do not add correction_s a second time.
- If caches are reused in practice, report both cold per-run charged cost and actual shared campaign cost. Do not call an attributed cold total a new measurement of rebuilding for every seed.
- Amortized reuse is a separate plot with explicit K and formula. Never hide construction in the headline efficiency claim.
- Storage: actual file bytes in GiB, train/val caches separately. Resources: peak CUDA allocated/reserved memory and host RAM with definitions.
- Teacher query budget q = actual training-query images / actual training image presentations. Also record cache-build, calibration and diagnostic backbone forwards separately.
- Frozen parameters, trainable head parameters and trainable corrector parameters are separate fields.
- Optional energy only if sampled power integration is implemented and background load is accounted for; no energy claim from runtime alone.

### Mechanism and statistics

- Main causal comparator: Transport at identical query budget. Report paired AP differences by seed.
- Held-out diagnostics: per-layer relative MSE and cosine, foreground/background split, plus detection loss under the SAME frozen reference head.
- Foreground weighting uses box-to-feature-cell area overlap, so tiny boxes do not silently create empty masks. Document empty-region exclusions.
- Equal image and layer weighting; identical transformations and samples across methods. Background pixels must not dominate an alleged task-aware improvement.
- Training-query MSE is a fit diagnostic only. Held-out labels may be used for frozen diagnostic masks and loss, never to update model weights.
- Three fixed seeds (0,1,2); run is the experimental unit. Curves show mean and sample SD across runs, not epoch-to-epoch spread.
- Use paired initializations, sample order, transform schedules and query identities. Block/interleave method execution across seeds; preserve GPU telemetry.
- Do not claim statistical significance from three seeds by default. Do not stop collecting seeds because a desired p-value appears.
- For descriptive gap recovery, use (AP_method - AP_C2)/(AP_online - AP_C2) only when the denominator is positive; distinguish this from AP_method/AP_online.
- Missing metrics are NA with reason, never zero. Never pool pilot custom AP and official full-data AP.

## 4. Paper layout: 8 experimental figures, 3 tables

Proposed layout: four main experimental figures (10 panels), four supplementary experimental figures (16 panels), three main tables.
A separate method diagram is not counted as an experimental figure.
This is a content plan, not a CVPR formatting requirement. No simulated results or assumed-winning curves will be drawn.

Unified visual mapping:
- C1: gray #7F7F7F, square; C2: gray #4D4D4D, diamond; C6: black #222222, triangle.
- Mix: blue #0072B2, circle; Transport: orange #E69F00, square.
- Reconstruction-only Correct (Rec): green #009E73, triangle.
- Final candidate: purple #CC79A7, diamond; this name is provisional until the method is fixed.
- Online exact-feature reference: dark #333333, star; do not call it a theoretical upper bound.
- Same method keeps color and marker across every figure. q is an annotation, not a new color.
- Shared legend outside panels, one or two rows. Restrict each panel to scientifically necessary curves.
- Report AP in points. Use no dual y-axis, broken axes or smoothed curves hiding variability.
- Use mean +/- sample SD, n=3, stated in captions. Mechanism plots average images within each seed first.
- All panel IDs, axes, legends and data dependencies are also stored in paper_figure_spec_20260926.json.

### E1: Does caching with correction improve the accuracy-cost tradeoff? (2 panels)

(a) Full VisDrone val. (b) Full COCO val2017.
- x: first-use total adaptation cost, allocated GPU-hours, log scale.
- y: dataset primary final AP (0-100 units; readable tick range shared only within the same dataset).
- Form: measured scatter with x/y error bars; lines connect only the same method family at different q.
- Legend: C1, C2, C6 (VisDrone only initially), Mix, Transport, Rec, final candidate, Online.
- Annotate q=10%,25%,50% when run; do not invent missing COCO q50.
- Pareto interpretation includes construction and I/O, not just cached-head training seconds.
- YOLO-Master practical reference is in T1/T3 rather than overcrowding this matched-frozen-model plot.

### E2: Is correction better than simply buying more online features? (3 panels)

(a) x: q (%) = 0,10,25,50,100; y: final VisDrone AP. Line + markers + SD.
(b) x: q (%) = 10,25,50; y: paired delta in VisDrone COCO-style AP_S versus same-q Transport. Zero reference line.
(c) x: fixed first-use total-time budget (GPU-hours); y: final VisDrone AP. Measured budget-limited training points + SD.
- Legend for (a,b): Mix, Transport, Rec, final candidate; C2/Online endpoints as references where applicable.
- Legend for (c): C2, Mix25, Transport25, final candidate25, Online.
- Lock caps at 0.25T, 0.5T, 1.0T, where T is the controlled preflight estimate for the 100-epoch Online job, before looking at comparative AP.
- Equal-time runs use a budget-normalized LR schedule and identical evaluation frequency policy; stop within the cap accounting for final evaluation/checkpoint reserve.
- q=0 correction has no free calibration; use identity bypass. q=100 uses true features for the head; no unnecessary correction update. Shared endpoints may be reused only after implementation equivalence is checked.
- Equal-epoch curves from old runs do not substitute for (c).

### E3: Does lower feature error improve task-relevant representations? (3 panels)

(a) x: held-out transform; y: foreground relative MSE.
(b) x: same transform; y: background relative MSE.
(c) x: same transform; y: frozen reference-head detection loss / exact-feature detection loss.
- Categories: hflip, scale 0.85, 1.15 (inside training range), scale 0.75, 1.25 (outside range).
- Form: grouped points with SD; optional thin connecting lines only to guide matching methods, not a continuous x trend.
- Legend: Transport, Rec, final candidate. Exact reference: horizontal 0 error or 1 loss-ratio line.
- Fixed frozen Online reference head, identical transformed boxes, no model updates on val.
- Prefer all 548 VisDrone val images; record diagnostic teacher compute separately from training budget.
- Current reconstruction-only training-query MSE cannot fill these panels.

### E4: Does the method generalize? (2 panels)

(a) Transfer heatmap.
- x: correction trained on target; correction trained on source and frozen; source correction adapted with target q25.
- y: VisDrone -> COCO, COCO -> VisDrone.
- Cell: delta in target final AP versus target Transport25; annotate mean and SD.
- Diverging colorbar centered at zero, same symmetric range for all cells; no categorical legend.
- A new target detector head is trained in every arm with the same target labels and q25 queries. Only corrector training/transfer policy changes.
- This is transfer of correction, not zero-shot object detection. Target-trained controls reuse existing q25 runs.
- Disclose source pretraining/correction cost separately; target-only and combined costs are both recorded.

(b) Backbone robustness.
- x: DINOv3-S/16, DINOv3-B/16, DINOv2-S/14.
- y: paired delta in full VisDrone final AP versus Transport25 for each backbone.
- Form: paired points with SD and zero line. Legend: Rec, final candidate.
- Same image protocol and detector head design; document token-grid/padding and feature-dimension adaptations.
- Retrain correction separately for each backbone. This tests method robustness, not copying correction weights across incompatible feature spaces.
- S -> B alone establishes within-family scaling; the second family supports the broader backbone claim.

### Supplementary S1: Optimization and stability (2 panels)
(a) x epoch; y train loss. (b) x epoch; y validation AP.
- Lines mean +/- SD; same method legend; no smoothing.
- Mark final and best epochs in (b), retain all seeds including unfavorable ones.
- Existing logs supply (a); use official re-evaluated checkpoints where available for (b), otherwise explicitly label custom diagnostic AP and never splice metric versions.

### Supplementary S2: Cache capacity and amortization (3 panels)
(a) x train-cache GiB; y final AP; points C1/C2/C6 and final25 annotated with query budget.
(b) x reuse count K={1,2,3,5,10}; y cumulative allocated GPU-hours; cache methods vs Online.
(c) x method; y seconds; stacked non-overlapping setup/cache, head+teacher+correction training, validation, remaining I/O/artifact overhead.
- (b) measured first-use points distinguished from dashed analytical amortization.
- Corrector subtime is an annotation inside training, not added on top.
- C6 = fixed-640 real feature views at centered scales {0.85,1.0,1.15} x {no flip,flip}.
- These are discrete geometric views, not random input-resolution training. Online and approximate arms use matching image/label geometry.

### Supplementary S3: Capacity and representation structure (3 panels)
(a) x corrector bottleneck width {16,32,64,128}; y final AP; final candidate only, q25.
(b) x residual output gain {0,0.25,0.5,1}; y final AP; frozen recipe except gain, q25.
(c) x SVD rank {8,16,32,64,128}; y cumulative held-out residual energy explained (%).
- (a,b): line + points + SD; Transport reference line; actual trainable parameter count in caption/table.
- (c): separate lines for flip and scale residuals, per-layer distinction by line style or split supplemental table.
- Network width is not SVD rank. Oracle decomposition uses exact target features and is only a diagnostic.
- Run capacity/gain sensitivity only after candidate and optimization schedule are frozen; do not use an unrestricted grid.

### Supplementary S4: Qualitative inspection (8 image panels: 2 rows x 4 columns)
- Columns: GT, Transport25, Rec25, final candidate25.
- Rows: a dense small-object scene and a difficult scene selected by predeclared annotation criteria, not by largest method gain.
- x/y: original-image pixel coordinates, ticks hidden; shared crop and scale in every column.
- Overlay legend: TP green #009E73, FP vermilion #D55E00, FN orange #E69F00; line styles distinguish boxes when printed grayscale.
- Pin IoU matching and display confidence for all methods. Include observed failure behavior even if the new method loses.
- Seed 0 is preselected for visualizations; IDs and crop coordinates saved in the manifest.

### Tables

T1: Full-dataset accuracy.
Rows: C1,C2,C6,Online,Mix25,Transport25,Rec25,final25, selected FroFA/LOFF-TA adaptations and measured YOLO-Master-N.
Columns: dataset/split, model and pretraining, AP/AP50/AP75, size AP (with evaluator label), n and SD.
Paper-reported references appear in a separate marked block, not in paired tests or an undifferentiated best-value ranking.
Main development comparisons are on val; reserve VisDrone test-dev for the frozen selected methods. Never merge val and test-dev numbers.

T2: Matched-q25 component ablation.
Rows: Transport, reconstruction-only, foreground-balanced reconstruction, foreground+task loss, and condition/gating removals if those components exist in the final candidate.
Columns: final AP, size AP, held-out FG error, trainable parameters, query images, total time.
These are proposed hypotheses, not implemented features or a promised winning architecture.
Reuse identical baseline runs; do not repeat them for each row. Start with two bounded candidate changes.

T3: Resource accounting and practical comparison.
Columns: cache build, train, validation, first-use total GPU-hours, cache GiB, peak GPU/CPU memory, teacher calls by purpose, parameters and fresh-image inference latency.
YOLO-Master runs and frozen DINO runs share measurement hardware/precision/batch/latency protocol, with model differences disclosed.

## 5. Run order and reuse

P0 (already running): finish six full VisDrone seed-0 diagnostic arms. Do not modify their source mid-run.
P1: freeze and validate official evaluation + size-metric conversion; re-score existing final/best artifacts without training again.
P2: held-out paired mechanism diagnostic; test at most foreground balancing and one task-loss variant initially.
P3: C6, selected feature-augmentation adaptations, and YOLO-Master-N reference; finalize candidate and extend matched full runs to seeds 1,2.
P4: q10/q50 and equal-time runs; reuse q25 endpoints, checkpoints and cost records across E1/E2/T1/T2.
P5: full COCO, DINOv3-B and DINOv2-S; transfer of the frozen candidate. Resource preparation may run concurrently with analysis.
P6: frozen-method test-dev evaluation, qualitative panels, final resource/inference measurements.

Do not run the Cartesian product of every dataset x backbone x augmentation strength x query x seed.
Use VisDrone/DINOv3-S for selection; confirm selected configurations on COCO and other backbones.
A revised method need not wait for every supplemental sweep before an exploratory cross-dataset check, but do not make final generalization claims with an unfrozen recipe.
If correction remains worse than Transport, report that result and reconsider the mechanism; adding datasets alone does not create method novelty.

Resource gate: the 2026-09-25 inventory lacked COCO train2017 images and train JSON.
Raw FP16 three-layer COCO C1 estimate is about 406.1 GiB, C2 about 812.2 GiB before overhead; available disk then was about 327.8 GiB before the new VisDrone cache.
Prepare storage/data first. A compression/projection change is a new method/configuration and needs its own accuracy validation; do not silently substitute it.

## 6. Required artifacts per run

Identity: experiment_id, protocol_version, dataset/split manifests and hashes, evaluator revision, code snapshot, method/backbone/pretraining revision, layers/input size, seed.
Schedule: head/corrector init hashes, image order, transforms, query identities or deterministic schedule hashes, optimizer/LR/epochs/time cap.
Accuracy: final and custom-selected-best AP, official selection/evaluation labels, AP50/AP75, size AP, predictions with original coordinates; explicit NA reasons.
Costs: cache purpose/build time/bytes, setup/train/validation/wall seconds, query image counts by purpose, actual presentations, GPU count, GPU telemetry and host RAM.
Checkpoint: final model/optimizer/corrector/RNG; selected best checkpoint with selection metric; no claim of unretained epoch results.
Mechanism: held-out image/transform IDs, layer errors, foreground masks/aggregation rules, frozen head hash.
Statistics: raw seed values first; means/SD and paired deltas derived without rounding intermediate values.

One result can serve multiple panels only when its data, optimization and evaluation contracts match.
No copying pilot AP into full-data tables, no treating profiling AP as a trained baseline, no treating oracle correction as a deployable method.
