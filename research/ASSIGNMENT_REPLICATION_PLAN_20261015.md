# Frozen-head replication, 2026-10-15

The completed Online100-head intervention shows substantial global feature-MSE
improvement without comparable localization-gradient alignment. This is a local
diagnostic and may depend on which head was frozen. Before a new training method,
replicate the identical four-cell diagnostic using the already completed Mix25
seed-0 epoch-100 head.

- Exactly the same 548 validation images, order, labels, six transforms, two
  approximation methods, soft/hard protocols and output metrics. No subset selection.
- Change only `--head mix25`; reuse the identical diagnostic source and corrector.
- All native-loss/identity/state/source checks remain enabled. No weight updates.
- Same batch=1, GPU0; no throughput comparison under shared compute.
- Primary comparison: persistence/direction of hard-common AE-AA classification
  versus box-output cosine changes; report all four cells and both branches.
- This is a head replication, not a second backbone, dataset, seed replication or
  official AP evaluation. No method/model selection based on the 548-image metrics.
- Existing validation has already informed research. A positive replication still
  requires spatial/equal-energy residual interventions and matched-budget AP training.
- Native reg_max=1 BboxLoss returns normalized ltrb L1 in the historical DFL slot;
  the old protocol's expectation that this slot should be zero was mistaken.
  Native computation and all raw results remain unchanged; interpretation corrected.
- Unique job directory and `latest-replication.txt`; preserve original campaign pointer.
  On success verify and archive the confirmation stage with its actual head metadata.
