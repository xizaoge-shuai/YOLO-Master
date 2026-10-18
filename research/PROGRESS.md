# D1 experiment progress

## 2026-09-25: cached-feature augmentation baselines

- Completed: 10 configurations, 3 seeds each, 100 epochs, fixed 400/100 split.
- Actual head: mean_value; identical unaugmented validation cache.
- Strong cheap reference: two real cached views + geometric feature transport.
- Mean best AP: C1 2.5694; C2 3.6405; Transport C1 3.1775;
  Transport C2 3.7666; LOFF-TA-style C2 3.7675.
- Noise adds approximately 0.0009 mean best AP over Transport C2 in this pilot;
  this is not evidence of a reliable improvement.
- Next: 10%/25% matched-backbone-query C2-Mix / C2-Transport / C2-Correct.
- Source code and experimental evidence are recorded in separate commits.
- [Evidence and analysis](experiments/feature-baselines-20260925-job-6JnMcLAf/README.md).

Earlier online, fusion, mean-sweep and flip-cache runs remain historical
references. Their helper scripts are retained as run; this entry does not
claim that their weights/predictions have been independently re-evaluated.

## 2026-09-25: fixed-budget correction implementation and full-data audit

- Added six C2 cases: Mix / Transport / reconstruction-only Correct at 10% and 25%.
  Formal campaign: 100 epochs, seeds 0/1/2; fixed 400/100 pilot.
- Training queries have paired identities, transforms and masks across arms.
  Actual teacher images, setup parity, correction time, checkpoints and schedule hashes are logged.
- Background launcher runs smoke before training and archives/commits lightweight progress on completion or failure.
- Validation: 17 related tests passed; new-file Ruff and launcher syntax checks passed.
  Repository-wide checks still report 2,768 existing Ruff issues, five format mismatches;
  codespell is not installed. Evidence: validation/budget-corrector-20260925/.
- GPU smoke/full campaign had not been run at this implementation commit.
- User confirmed target reference is YOLO-Master CVPR 2026. Paper recipe and differences:
  [benchmark protocol](YOLO_MASTER_BENCHMARK_PROTOCOL.md).
- Full VisDrone files: 6,471 train / 548 val / 1,610 test-dev; one zero-height training box
  requires an explicit derived-annotation exclusion before full training.
- COCO: 5,000 val images and annotations present; training images/JSON absent despite train list.
  Full VisDrone caches require streaming; COCO C1 raw features alone require about 406 GiB.

## Fixed-budget correction: job-smoke-1bpNRqmT

- Status: FAILED exit=1; inspect job.log and worker log
- Evidence: [archive](experiments/budget-corrector-job-smoke-1bpNRqmT/README.md)
- Source job: /data/users/zhjia/YOLO-Master/runs/paper/budget-corrector/job-smoke-1bpNRqmT
- Report best/final AP, paired differences, actual query images and total charged time.
- No full-dataset or independent-test claim; no conclusion inferred from smoke AP.

## Fixed-budget correction: job-smoke-ajfAh00N

- Status: SMOKE_SUCCEEDED
- Evidence: [archive](experiments/budget-corrector-job-smoke-ajfAh00N/README.md)
- Source job: /data/users/zhjia/YOLO-Master/runs/paper/budget-corrector/job-smoke-ajfAh00N
- Report best/final AP, paired differences, actual query images and total charged time.
- No full-dataset or independent-test claim; no conclusion inferred from smoke AP.

- Post-smoke verification: all six raw schedules match their saved hashes; query schedules
  match across arms. Corrector checkpoints contain trained weights and optimizer state
  (50,560 parameters; 5/12 updates at p10/p25). Formal 18-run campaign not launched here.
- Pilot provenance: all 500 images come from official VisDrone training, internally split 400/100.

## Fixed-budget correction: job-start-7U9Q7DSc

- Status: SUCCEEDED
- Evidence: [archive](experiments/budget-corrector-job-start-7U9Q7DSc/README.md)
- Source job: /data/users/zhjia/YOLO-Master/runs/paper/budget-corrector/job-start-7U9Q7DSc
- Report best/final AP, paired differences, actual query images and total charged time.
- No full-dataset or independent-test claim; no conclusion inferred from smoke AP.

## 2026-09-25: paired-budget results audited; full-data roadmap fixed

- Source job: runs/paper/budget-corrector/job-start-7U9Q7DSc; raw archive commit 8d01693.
- Verified 201 archived evidence hashes and 18 complete 100-epoch runs. Local/server analysis
  matches in 256 numeric fields to absolute/relative tolerance 1e-12. No AP rerun from predictions.
- Correct minus Transport best AP: p10 -0.2270, p25 -0.1116;
  final AP: p10 -0.1067, p25 +0.0391. Current reconstruction-only correction has no
  demonstrated stable AP advantage over same-budget Transport.
- Last-ten-epoch training-query relative MSE reductions: about 52-54% (p10), 62% (p25).
  This establishes training-query fit only; held-out feature utility remains unmeasured.
- Transport-p25-s1 slowdown affects training and validation; keep all original results.
  No algorithmic speedup claim from the aggregate 0.902 ratio. Remeasure with telemetry.
- Fresh GPT-5.6-Sol ultra reviewer: same-family/provisional WARN, not independent replication.
  AP GT and query pairing pass; custom pilot scope, best-on-validation selection,
  timing anomaly and explicit metrics.py hash omission require qualifiers/actions.
- Analysis, statistical appendix, figures and audit:
  [budget analysis](experiments/budget-corrector-job-start-7U9Q7DSc/analysis-report.md).
- [Next experiments](NEXT_EXPERIMENTS_AFTER_BUDGET_CORRECTOR.md): start full VisDrone
  streaming/evaluation/baseline preparation now; first full seed includes the current corrector
  as a diagnostic control. One bounded held-out/object-aware diagnosis, then selected
  COCO and second-backbone studies. No new GPU training launched in this analysis turn.
- New analysis Ruff checks pass. Repository-wide pre-existing 2,768 Ruff issues and five
  format mismatches persist; codespell remains unavailable. Evidence in analysis-validation/.

## Full VisDrone: job-smoke-2TKOvomC

- Status: SUCCEEDED
- Evidence: [archive](experiments/full-visdrone-job-smoke-2TKOvomC-succeeded/README.md)
- Diagnostic AP only; official ignored-region evaluation pending. No test-dev tuning.

## Full VisDrone background experiment entry (2026-09-26)

- Added six matched-head arms on full 6471 train / 548 val, batch-streamed FP16 cache and exact image query budgets.
- Test-dev untouched; raw ignored-region annotations and best/last VisDrone prediction exports retained. Diagnostic AP is not official VisDrone AP.
- Full dataset preparation and six-arm 17/8 smoke passed; two-epoch interrupted/continuous corrector weights match exactly; final-epoch commit interruption recovery passed.
- Independent code review findings fixed; details: FULL_VISDRONE_IMPLEMENTATION_CHECKS.md. Global lint retains 2768 pre-existing findings and five format mismatches; codespell absent.
- Start/status/resume commands: online_experiments/FULL_VISDRONE.md. Full 100-epoch campaign not launched by assistant.

## Experiment display and metric contract (2026-09-26)

- Verified active full-data job job-start-eDmH0odX: cache build and 7019-image integrity passed; six profiles complete; train/cache1 at 100 epochs is running. Profile AP is not a final result.
- Completed pilots include C1/C2, online/mixed-query controls, Transport, FroFA/LOFF-TA idea adaptations and matched-budget reconstruction correction. No stable correction gain over same-budget Transport yet.
- Confirmed early COCO128 80/20 and VisDrone 400/100 predictor runs; neither establishes full COCO or source-frozen transfer. Official evaluation, C6 scale cache, equal-time and cross-backbone tests remain pending.
- Added PAPER_EXPERIMENT_DISPLAY_PLAN_20260926.md plus figure, experiment-matrix and metric contracts: 4 main figures / 10 panels, 4 supplementary figures / 16 panels, and 3 main tables. Final method remains a hypothesis.
- Prospective primary endpoints: final-budget dataset AP and first-use total GPU-hours including cache construction. Best AP remains secondary; official VisDrone and COCO-style size metrics are distinguished.
- Existing training source hashes verified unchanged. No additional GPU experiment started and no old results overwritten. Specification/schema checks pass. Required repository checks and their limitations are recorded in paper_display_validation_20260926.json.

## Full VisDrone: job-start-eDmH0odX

- Status: SUCCEEDED
- Evidence: [archive](experiments/full-visdrone-job-start-eDmH0odX-succeeded/README.md)
- Diagnostic AP only; official ignored-region evaluation pending. No test-dev tuning.

## 2026-09-27: full VisDrone pinned-reference evaluation completed

- Evidence: [reference evaluation](experiments/visdrone-reference-job-full-20260927/README.md).
- Twelve full-548-image prediction sets rescored; native Octave parity on 24 synthetic and two real-image cases passed (max 4.41e-13 AP points). Fixed reference commit and actual source bytes verified.
- Final reference AP: C1 9.1186, C2 10.6568, Mix25 11.3526, Transport25 9.6350, Correct25 9.7656, Online 11.3982. One seed only. Reference class-image weighting and ignored-region behavior preserved; old custom AP unchanged.
- Corrector gains +0.1306 over Transport but trails Mix by 1.5870 AP. No final-method or stable-benefit claim. Best checkpoint was selected by old custom AP. Held-out feature/task diagnostic pending.
- Ten reference tests and two feature metric tests pass; global pre-existing Ruff/format findings preserved. Validation evidence included; no push.

## 2026-09-27: frozen validation feature/task diagnostic launched

- Run: runs/paper/heldout-corrector/job-7c6i3o4w; [protocol](../online_experiments/HELDOUT_CORRECTOR.md), [evidence](experiments/heldout-corrector-job-7c6i3o4w/README.md).
- Full548 validation, final100epoch corrector and one frozen Online head, 10 augmentations, no updates. Separate foreground masks preserve all visible box fragments while loss keeps original target filtering. Four targeted tests and review pass.
- First start job-z1m6vokw failed before evaluation: missing HF_HOME. Retry inherits original offline cache. GPU1 CUDA unavailable; using GPU0 batch4. Diagnostic time is not an efficiency benchmark.

## 2026-09-27: frozen validation diagnostic completed

- [Results and analysis](experiments/heldout-corrector-job-7c6i3o4w/analysis-report.md): 548 images, 5480 teacher-image evaluations, 10960 paired rows, 20 condition summaries; 136.69 seconds; source/per-image hashes verified; no updates.
- Nonidentity scales reduce foreground relative MSE by 10-13%, background by 15-19%, and fixed-head loss by 4-16% versus Transport. These are one-seed validation diagnostics, not AP improvements or cross-dataset transfer.
- Scale1 exact anchors are unnecessarily changed: foreground relative MSE ~0 -> 0.00710 and mean loss ratio ~1 -> 1.00253. Identity-preserving behavior is a concrete candidate ablation, not an established AP fix.
- Current method remains unfinished. Preserve C2/Mix/Transport controls and matched queries; next finite variants can test exact-anchor preservation and foreground/task supervision. Do not launch an unrestricted parameter sweep.

## 2026-09-28: recover completed candidate pilot

- [Evidence](experiments/budget-corrector-candidate-job-start-LIz9kPc6/README.md). Six400/100 seed0/1/2 runs complete; all14 recorded source hashes verified. Gate final AP mean4.0630, gate+foreground4.2384; original Rec25 was4.2809. No gain claim.
- Candidate code already present was preserved. Full-data factorial gate/foreground study follows with separate source provenance and visible-fragment masks.

## Correction ablation job-smoke-ibKknuJd: succeeded

- [Evidence](experiments/correction-ablation-job-smoke-ibKknuJd/succeeded/README.md).
- Fixed gate/foreground factors, paired query schedules; no test-dev or stable-advantage claim.


## 2026-10-13: experiment evidence and CVPR story review

- [Review and next diagnostic plan](CVPR_STORY_REVIEW.md); [read-only artifact audit](experiments/story-review/verified-results.json), with 18 source-file hashes. No new training/evaluation or cache deletion in this review.
- Full VisDrone final reference AP over seeds 0/1/2: C2 10.3847 +/- 0.2518, Mix25 11.0168 +/- 0.2934, Online100 11.4961 +/- 0.1693 (sample SD). Mix25 improves C2 by 0.6321 AP and trails Online100 by 0.4792 AP.
- Completed seed0 Anchor25/Task25 reference AP is 10.1128/10.0809, below paired Mix25 11.3526. All 100 candidate schedule hashes match Mix25; candidate source and three relevant dependency hashes match as-run provenance. Method remains exploratory.
- VOC is a 2000/400 trainval subset with diagnostic AP, not official test or frozen source-to-target transfer. Its query reduction has not established wall-time speedup. New VisDrone timings also contain large time-varying validation overhead.
- Raw Master-N CSV has 198 rows and 98 inconsistent duplicated epoch groups; checkpoint AP is retained but its strict training-budget comparison requires provenance repair. YOLO11/12 CSVs each have 100 unique rows.
- DiKE/REVIVE paper exemplars motivate paired and equal-magnitude causal interventions. Existing cache-augmentation literature prevents a novelty claim based solely on D1 recency. Next decision: separate feature approximation and assignment effects before extending the method or launching more full campaigns.

## 2026-10-14: frozen-head assignment intervention implementation

- Added [fixed diagnostic protocol](ASSIGNMENT_INTERVENTION_PLAN.md), native-soft and hard/common-denominator 2x2 target transplantation, separate box/classification output gradients, GT-switch/coverage metrics, frozen-state and source/checkpoint audits. No optimizer updates or AP claims.
- 11 CPU contracts pass; target lint/format/spelling, Bash syntax and CLI help pass. Independent reviewer findings on checkpoint identity and image weighting fixed. Global Ruff has 2768 pre-existing issues, five format files plus parse errors; full codespell timed out after 45 seconds.
- [Preflight and check record](experiments/assignment-intervention-preflight-20261014/VERIFICATION.md) preserves two initialization failures; reused historical HF_HOME and handled nested extra model state. GPU1 CUDA allocation fails; planned batch-1 GPU0 run is shared and excluded from speed claims.
- Background sequence: 2 training-image smoke, fixed 128 training-image development, all 548 prior validation images. Two approximations, six views, two loss branches, two target protocols, four feature/target cells. This remains a mechanism diagnostic, not a settled training method.

## 2026-10-14: live assignment diagnostic and verified result archiving

- [Run details and commands](ASSIGNMENT_INTERVENTION_RUN_20261014.md). Worker job-20261014-EgMkSelF has completed its 2-image smoke (12 teacher images, 384 cells), exact native loss parity, identity relative L2 0.000034996, unchanged model states and zero updates. Fixed 128-image development is running, followed by all 548 validation images.
- Added a separate result watcher; it does not edit running diagnostic sources. It verifies stage counts, finite parity and five output hashes, archives small summaries, and commits explicitly scoped paths without push. It rejects unrelated progress/staged changes, preserves manual archive edits, and resumes its own interrupted commit without duplicate progress.
- Independent review concerns addressed; seven archive tests pass, including real temporary-Git failed-commit recovery. Target lint/format/spelling pass. The diagnostic retains its previously passing eleven mathematical contracts.

<!-- assignment-intervention:job-20261014-EgMkSelF:smoke -->
## Assignment intervention job-20261014-EgMkSelF: smoke complete

- [Verified result archive](experiments/assignment-intervention-job-20261014-EgMkSelF/smoke/README.md); 2 images, 12 teacher images, 384 cells.
- Native loss parity max absolute difference 0; identity relative L2 3.4995974e-05; model states unchanged; zero updates. No AP claim.

<!-- assignment-intervention:job-20261014-EgMkSelF:development -->
## Assignment intervention job-20261014-EgMkSelF: development complete

- [Verified result archive](experiments/assignment-intervention-job-20261014-EgMkSelF/development/README.md); 128 images, 768 teacher images, 24576 cells.
- Native loss parity max absolute difference 0; identity relative L2 9.0877351e-05; model states unchanged; zero updates. No AP claim.

<!-- assignment-intervention:job-20261014-EgMkSelF:confirmation -->
## Assignment intervention job-20261014-EgMkSelF: confirmation complete

- [Verified result archive](experiments/assignment-intervention-job-20261014-EgMkSelF/confirmation/README.md); 548 images, 3288 teacher images, 105216 cells.
- Native loss parity max absolute difference 0; identity relative L2 0.00013656419; model states unchanged; zero updates. No AP claim.

## 2026-10-16: completed assignment analysis and frozen Mix25 replication preparation

- [Analysis report](analysis/assignment-intervention-20261015/analysis-report.md), paired image-level tables and two figures are based on verified original CSV hashes. Original campaign completed 2/128/548 images, with zero updates and exact native-loss parity. Small stage archives are committed as 0b04440, d65a3e7 and aaa2db5; the CRLF check fix preserved raw bytes.
- On the 548-image existing validation set, global relative feature MSE decreased from 0.212400 to 0.039714 (81.302% ratio-of-means reduction). Native-soft one2many box-output gradient cosine increased only from 0.139712 to 0.178602. Teacher matching improves classification alignment more strongly than box alignment; this does not establish an AP cause, a novel mechanism or a successful training method.
- Intervals are descriptive image resampling and filename-prefix cluster sensitivity, not independent training-seed or population inference. Existing validation is not an untouched test. Global MSE may be dominated by padding/background; this explanation has not been excluded.
- Erratum: this repository's reg_max=1 BboxLoss computes auxiliary normalized ltrb L1 in the historical loss_dfl slot. The old zero-term assumption was wrong. Actual diagnostic computation already matched native loss; no raw results were changed.
- [Next fixed protocol](ASSIGNMENT_REPLICATION_PLAN_20261015.md): repeat the identical full 548-image diagnostic with the existing Mix25 seed-0 epoch-100 head, holding the corrector and all other settings fixed. Launcher is prepared; actual job path will be recorded after launch. This is a head replication, not a new model/dataset/AP experiment.
- Verification: 19 targeted tests passed (4 temporary-directory cleanup warnings), analyzer/archiver Ruff check and format passed, Bash syntax passed, analysis source hash verified and diagnostic source unchanged from ad972524. Figure QA: 11 pass, 3 documented warnings, 0 fail. Global checks retain 2768 pre-existing Ruff errors, five formatting files plus parse errors; full codespell timed out after 45 seconds. Additional review covered supplied critical snippets only. No push.

## 2026-10-16: Mix25 frozen-head replication launched

- Background job `/data/users/zhjia/YOLO-Master/runs/paper/assignment-intervention/job-replication-mix25-20261016-mOtQ5Tzb`; worker PID 2329985; implementation/analysis commit e34f2e827a7fdd6ed4c2233a18c1207bce5da0b5. Provenance confirms the Mix25 head. Full 548-image existing validation set, unchanged diagnostic and corrector, zero planned optimizer updates.
- Live verified progress at recording: {"completed_images": 1, "elapsed_seconds": 22.849116755649447, "teacher_images": 6, "total_images": 548}. This is an in-progress run, not a result or AP claim. Previous full diagnostic took 10452 seconds; runtime will vary with GPU load.
- Monitor: `bash online_experiments/run_assignment_replication.sh status`. Raw logs and per-image data remain in the job directory. On successful computation, the launcher verifies and commits the small confirmation archive; archiving has a separate status and never rewrites raw results. Original campaign pointer is preserved. No push.

<!-- assignment-intervention:job-replication-mix25-20261016-mOtQ5Tzb:confirmation -->
## Assignment intervention job-replication-mix25-20261016-mOtQ5Tzb: confirmation complete

- [Verified result archive](experiments/assignment-intervention-job-replication-mix25-20261016-mOtQ5Tzb/confirmation/README.md); 548 images, 3288 teacher images, 105216 cells.
- Native loss parity max absolute difference 0; identity relative L2 0.00013656419; model states unchanged; zero updates. No AP claim.

## 2026-10-18: Mix25-head replication analyzed; next spatial intervention specified

- Job job-replication-mix25-20261016-mOtQ5Tzb completed all 548 validation images, 3288 teacher queries and 105216 cells in 10791.83 seconds. Result archive commit f2039af9669d2691ac7e6948c6478c61b37266fa. All output hashes verified; exact native-loss parity, unchanged states and zero training updates.
- [Paired analysis and two figures](analysis/assignment-replication-20261018/analysis-report.md): sources, views, cache and corrector are identical to the Online100 diagnostic; per-row feature-MSE difference is exactly zero. Only the frozen detector head changes. This is one seed per recipe, not a second backbone/dataset or new AP evaluation.
- Hard-common Rec one2many classification AA/AE is 0.842393/0.970054 with Mix25 versus 0.839975/0.968404 with Online100; box AA/AE is 0.377584/0.432631 versus 0.382021/0.435595. Limited localization-output-gradient recovery persists across the two heads.
- Scale breakdown changes the classification interpretation: at scale 0.85, Mix25 Transport classification AA/AE is 0.451275/0.520120, versus Online100 0.833684/0.965585. Mix25 Transport gradient norm ratio is 2.264354; Rec restores classification AA to 0.847839 but box AA is 0.384942. Padding is a specific untested explanation, not an established cause. Rec has diagnostic utility here; no new AP gain claim.
- [Next protocol](NEXT_INTERVENTION_PLAN_20261018.md): separate new padding, original letterbox, valid foreground/background; compare padding/content oracles and a cheap padding control, then equal-token/equal-norm residual interventions. Add head-parameter-gradient diagnostics before treating output gradients as training evidence. No new GPU job launched. Preserve earlier Anchor/TaskAware negative results.
- Analysis uses paired image means and descriptive 2000-draw image/prefix bootstrap (76 filename-prefix groups), not independent training-seed inference. All primary gradient metrics finite. New script Ruff, format and targeted codespell pass; repository-wide Ruff/format retain pre-existing failures and codespell times out after 45 seconds. Two plots visually checked. Explicitly scoped commit only; no push.
