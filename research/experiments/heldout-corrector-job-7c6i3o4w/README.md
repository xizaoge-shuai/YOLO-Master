# Held-out full VisDrone feature/task diagnostic

Source run: /data/users/zhjia/YOLO-Master/runs/paper/heldout-corrector/job-7c6i3o4w

- All 548 official validation images, five scales, two flip states; final epoch100 checkpoints, seed0, frozen parameters. No test-dev use or parameter updates.
- Method-development diagnostic, not independent test or AP scoring. Fixed Online head, batch4, identical ordering across arms. No batch-invariance or runtime-speedup claim.
- Foreground mask retains all positive-area visible YOLO-box fragments; detector loss retains historical target filters. Maximum per-box overlap is not exact union area.
- Review found target-filter leakage into masks; fixed before execution with two failing-then-passing tests. Four feature-metric tests pass; scoped Ruff passes. Second read-only review found no remaining actionable defects.
- GPU1 basic CUDA allocation failed as busy/unavailable. GPU0 used. First start failed before teacher evaluation because HF_HOME was omitted; logs preserved. Retry uses original training cache at ~/cache/huggingface, fully offline.
- Current job was launched successfully; final completion/results are recorded separately when available.
