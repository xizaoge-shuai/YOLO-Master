# Single-pass metric collection, 2026-11-12

The authorized queues contain seven methods on each of full VOC0712 and SKU110K, one seed0 run per method, fixed final epoch100. Existing VisDrone evidence is reused. The new metric sidecar consumes saved final predictions on CPU; it never trains, reads images, runs inference, or reserves a GPU. Running trainer and coordinator sources remain unchanged.

The scorer exports native-metric consistency, explicitly auxiliary COCO-style AP/AP50/AP75/size metrics, AR@1/10/100/500, class and area tables, GT counts, interpolated P-R/score tensors, real training curves, resource scopes, and input/output hashes. VOC native eleven-point AP50 stays primary. SKU maxDet100/500 remain distinct. Native validation CSV zeros are unmeasured placeholders and are excluded.

`cpu-reference-voc-cache1/` was derived from the already completed full4952-image VOC test predictions. Native AP50 equals the original 67.59999942289588 exactly; the bundle contains100 actual training rows. Identity-bound frozen teacher parameters are21,596,544; the VOC cache head has1,295,519, giving22,892,063 pipeline parameters. These are an existing C1 reference endpoint, not newly trained priority results.

`frozen-teacher-parameters.json` records an offline CPU parameter count using the extractor's actual DINOv3 backbone class, with no forward pass. Contract evidence is in the adjacent JSON files. The complete saved prediction and training checkpoint remain on the server; the light derived evidence is archived here.

Shared-GPU time is observational. Cached-feature validation is not full-image inference latency. Missing P95, energy or phase measurements remain unavailable. Checkpoint bytes describe a training checkpoint, not deployment-only weights. No independent-seed uncertainty is inferred from a single run.

Production bundles: `runs/paper/metric-bundles/job-20261112-single-pass/<dataset>/<method>/bundle/`. Completed light bundles are committed under `research/experiments/single-pass-metrics-20261112/`. See the root server handoff for status commands.
