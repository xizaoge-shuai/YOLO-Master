# Fixed-budget C2 feature correction

Run inside the existing yolomaster environment at ~/YOLO-Master:

```bash
bash online_experiments/run_budget_corrector.sh start
bash online_experiments/run_budget_corrector.sh status
bash online_experiments/run_budget_corrector.sh watch
```

Start detaches with nohup. Ctrl-C in watch stops the viewer only.
The launcher uses the existing flip-cache latest-job marker, performs six
one-epoch smoke runs, then 18 formal runs (three arms x two budgets x three seeds).
The campaign runs serially on GPU0, stops on first failed worker, and never
overwrites a prior job. Completed formal results, including partial failure
evidence, are archived and committed automatically. An archive failure has a
separate archive-status.txt and does not falsely change training status.

## Fixed protocol

- Same 400/100 pilot split, frozen DINOv3 ViT-S/16, layers 3/7/11, 640 input,
  mean_value detection head, AdamW and 100-epoch cosine schedule as prior runs.
- C2 anchor cache: real original and horizontal-flip features.
- Budgets: 10% / 25% of training images sent to the frozen backbone. Every
  method uses identical shuffle, augmentation proposals and query positions
  within each seed/budget. Query images are reused for supervision and detection.
- This pilot requires equal-size batches. Cumulative integer batch quotas give
  exactly 4,000 / 10,000 queried images over 40,000 training-image presentations.
  A one-epoch 25% smoke uses 96/400 (24%); the second epoch balances the fraction.
- Mix: exact augmented features on queries; original/flip anchors on nonqueries.
- Transport: same exact queries; center-scale warped anchors on nonqueries.
- Correct: same exact queries; transported anchors plus a predicted residual
  on nonqueries. No detection gradient enters the corrector.
- Corrector: shared 1x1 -> depthwise 3x3 -> 1x1 network, rank 64, SiLU,
  conditioned on scale, flip, layer and spatial coordinates, zero output init.
  AdamW lr .001 and weight decay .0001. No hyperparameter search in this plan.
  Loss: mean per-image/layer feature MSE normalized by detached teacher energy.
- Separate initialization and routing RNGs. Validation receives the same original
  cache as all prior runs; no augmentation or corrector and no calibration on val.

## Evidence and costs

results.csv: unchanged detector metrics. correction.jsonl: pre-update query
reconstruction MSE, transport MSE, residual magnitude, cumulative steps and time.
schedule.jsonl: raw sample IDs, geometry and query mask, with hashes matched by
the summary across methods. Correction losses are training-query diagnostics,
not held-out reconstruction accuracy.

Teacher image count excludes four setup parity images in training_online_percent;
both setup and total counts are reported separately. Historical C2 build cost is
charged once per run; actual new build is zero. Validation-cache construction
is excluded. Corrector train and prediction time is included in training time;
the additional field isolates its synchronized interval. Equal query count does
not imply equal GPU time. Tiny pilot caches are GPU resident; full-data work
requires a streaming loader and a fresh timing study.

Use best and final AP paired differences against both Mix and Transport.
Do not select only favorable seeds or treat three seeds as cross-domain evidence.
The detector evaluator remains the existing D1 COCO-style implementation,
not the official VisDrone toolkit or pycocotools. Results cannot be compared
numerically to the YOLO-Master paper until the full evaluation protocol is aligned.
