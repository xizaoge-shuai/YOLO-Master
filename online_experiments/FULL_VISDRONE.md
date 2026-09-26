# Full VisDrone stage (v1)

This is the next diagnostic stage after the 400/100 budget pilot, not a paper benchmark claim.

- Official train: 6,471 images; official validation: 548 images. No test-dev access.
- DINOv3-S/16, layers 3/7/11, 640, original pinned teacher revision, FP32 extraction/FP16 cache.
- Mean-value head, AdamW lr=0.001, weight_decay=0.0005, batch=8, 100 epochs, seed 0.
- Six arms: cache1, cache2, mix25, transport25, correct25, online-geom.
- Correct25 keeps the current reconstruction-only correction. It is a diagnostic control, not an assumed improvement.
- C2 uses real horizontal-flip features. Mixed/transport/correct share proposed flips/scales and exact query image IDs.
- Per-image cumulative budgets include the last seven images of each epoch. At 25%/100 epochs: 161,775 queried images.
- Read-only FP16 memmaps stream batches from disk/OS page cache. Approximately 46.3 GiB for train C2 + val C1; reserve another 30 GiB for versioned best checkpoints/predictions.
- Derived labels filter/log zero-area boxes. Original images and annotations are never overwritten.

## Run on the server

Activate yolomaster, then:

~~~bash
cd ~/YOLO-Master
bash online_experiments/run_full_visdrone.sh start
bash online_experiments/run_full_visdrone.sh status
~~~

Start builds the full cache, measures six full one-epoch profiles, reports an estimated remaining runtime,
then serially trains all six methods on GPU0. It uses existing locally cached teacher weights.
Do not use GPU1. No concurrent duplicate campaign is supported.

~~~bash
JOB="$(cat runs/paper/full-visdrone/latest-job.txt)"
tail -n 30 -f "$JOB/job.log"
# Current case's detailed epoch log, e.g.:
tail -n 5 "$JOB"/train-*.log
~~~

Ctrl+C in tail stops viewing, not the background worker.

After a training interruption with unchanged source and a completed cache:

~~~bash
bash online_experiments/run_full_visdrone.sh resume
~~~

Resume skips DONE cases and restores unfinished detector/corrector optimizer, scheduler, RNG and epoch.
An interrupted cache build is not resumable in v1; retain its failed evidence and start a new job.
Do not change training sources while a campaign is running. Source fingerprints are checked before each case. Cache and raw-data integrity is verified once per campaign launch/resume.
Status prints each case's last completed epoch. Completed/failed campaigns automatically commit lightweight
evidence under research/experiments, provided the Git index and progress file have no unrelated staged/edit work.
No weights, image data or feature cache are committed; no automatic push.

## Evaluation boundary

Validation uses all 548 validation images. Current AP uses the existing custom DetMetrics, conf=0.001,
max_det=500, converted YOLO labels. It does not implement VisDrone ignored-region handling.
It is NOT official VisDrone AP and must not be compared directly with the paper's AP table.
Raw official validation annotations are retained; best and last predictions are exported as eight-column
VisDrone files in original image coordinates for later official toolkit scoring.
MATLAB/Octave is absent on the server. Official evaluation and COCO-style area AP remain pending.
The 1,610 test-dev images are reserved until configuration selection is complete.

YOLO-Master-N scratch and the paper's 600-epoch/batch-256 recipe are separate experiments, not included
in this six-arm queue. The current head is the project's cached-feature head, not the paper detector.

## Cost and evidence

Each per-run total charges the applicable training cache, validation cache, preparation and run wall time.
Actual shared campaign wall time (including profiling) is recorded separately in campaign-cost.json;
do not sum per-run charged cache costs and call that actual campaign runtime.
GPU telemetry and host peak RSS/I/O counters help identify contention and data-loading costs.
Training-query counts exclude cache construction, which is listed separately in cache/manifest.json.
The 100-epoch comparisons are matched-step, not equal-time comparisons.

Smoke is a separate 17-train/8-val test with a one-image tail, one epoch per arm:
~~~bash
bash online_experiments/run_full_visdrone.sh smoke
~~~
Smoke completion does not establish accuracy or full-data throughput.
