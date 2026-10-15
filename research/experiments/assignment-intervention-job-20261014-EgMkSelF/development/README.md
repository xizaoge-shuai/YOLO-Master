# Assignment intervention: development

Source: `/data/users/zhjia/YOLO-Master/runs/paper/assignment-intervention/job-20261014-EgMkSelF/development`. Images: 128; teacher images: 768; rows: 24576.

Batch-1 frozen epoch-100 seed-0 Online100 head; Transport/Rec; six views; one2many/one2one; native_soft/hard_common; EE/EA/AE/AA cells (feature donor, assignment donor). Full per-image CSVs stay in the run directory and their hashes are in summary.json. No weights, feature caches, optimizer updates or AP evaluation in this archive. Shared GPU elapsed time is not an efficiency comparison.

Smoke is only an execution check. Development uses training images. Confirmation reuses the existing 548-image validation split and is not an untouched test. Cross-view summaries average each image first. Gradient metrics refer to raw-output coordinates, not feature or parameter gradients. Exact-assignment interventions use oracle information; improvements do not establish AP recovery.
