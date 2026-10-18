# Joint spatial/native-gradient intervention: development

Source job `job-20261018-pJHRhd15`; 128 fixed training images, 768 paired teacher queries plus one blank-template query. Both seed-0 frozen heads share each exact feature. Eight arms and six views; no optimizer updates or AP evaluation. Natural-area oracles do not control residual energy. Hard exact-assignment output/parameter gradients plus separate actual native-loss gradients. The p=0.25 estimator is analytically enumerated; all views are queried for this diagnostic. This is a batch1 frozen-eval screen, not a training cost or AP result. Passing only nominates a later minibatch probe; full training is not launched. Regional errors and all failed screening cells are retained. Zero-reference parameter gradients are undefined, especially one2one shared blocks.

Full row data and source snapshots stay under runs/paper/joint-gradient; measured hashes are retained here. No weights or feature cache in Git.
