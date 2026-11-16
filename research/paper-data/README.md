# Paper data inventory

All project training and metric observers were stopped at the user request on2026-11-17. No automatic restart is authorized. Weights/checkpoints, caches, raw predictions and results are retained.

Use core-results.csv for21 planned core pairs,16 with final100 metrics. Blank cells are unmeasured. historical-visdrone-final-seeds.csv and historical-visdrone-seed-summary.csv retain the earlier campaign separately. Weighted and class-macro AP are separate columns. data-index.csv maps18 data families to their existing sources, fields and availability. See the repository-root Chinese D1 relationship and paper-data guide.

No new training, inference or detection evaluation. Derived tables read saved metrics; provenance and output hashes are in sources.json. Historical shared-GPU resource values are observations, not controlled speedups.
