# Minimal single-run baseline priority

User2026-11-12 removes seed repetitions and extra control/ablation sweeps under deadline. Per full dataset,run actualYOLO-Master-N andYOLO12-N plus the four supplied-paper idea adaptations,thenDetail25. Every configuration runs once withseed0,100epochs. Both GPU0 slots may train concurrently when memory allows. Completed VisDrone seven-case endpoints are hash-bound in reused-visdrone-core-results.csv and will not be retrained.

Old three supervisors were stopped. SKU extraC2 retained checkpoint72; existing VOC Mix25 epoch98 worker may finish,without scheduling further old cases. Original source,rawdata,featurebanks,checkpoints and logs remain intact. New cached-side baselines reuse sealed full banks and frozen existing trainer. The four paper-derived rows remain explicit detection idea adaptations,not the original papers' code or reported benchmarks. SharedGPU costs are observational.
