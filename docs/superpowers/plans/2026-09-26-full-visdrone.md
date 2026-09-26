# Full VisDrone first-stage implementation

User-approved design: research/NEXT_EXPERIMENTS_AFTER_BUDGET_CORRECTOR.md.
Deliver tested background commands, preserve historical sources, commit scoped progress.

1. Test exact per-image budgets (including 6471 % 8 tail), deterministic pairing,
   label filtering, split isolation, inverse letterbox and disk-backed cache reads.
2. Add a versioned standalone data/cache module: official train/val, derived labels,
   raw validation annotations, FP16 memory-mapped features, teacher identity/hashes.
3. Add six matched-head arms: C1/C2, Mix/Transport/Correct at 25%, full online geometry.
   Use original mean_value detector/optimizer; retain current reconstruction-only corrector.
   Save actual image-query counts, best/last weights, metrics, predictions and costs.
4. Add serial GPU0 campaign: separate smoke; full cache build; full one-epoch profile;
   six 100-epoch seed-0 runs; resource telemetry; failure propagation; scoped auto-archive.
5. Provide a separate YOLO-Master-N 100-epoch scratch reference command using derived
   labels. It is not the paper's 600-epoch/batch-256 reproduction.
6. Run unit/integration smoke and required repository checks; commit code and evidence.

Evaluation boundary: full 6471/548 split, 1610 test-dev untouched. Current Python
DetMetrics reports diagnostic AP with 500 detections/image. Export original-coordinate
VisDrone prediction files. No MATLAB/Octave is installed: official toolkit execution,
ignore-aware score verification and COCO-style area metrics remain a subsequent stage.
Never label this AP official or directly comparable to the paper's table.

Avoid restarting or extending historical pilots. Full-data source hashes include
metrics/loss/decode and dirty-tree status. No datasets, tensors or weights in Git.
