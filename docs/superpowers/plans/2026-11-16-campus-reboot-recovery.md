# Campus reboot recovery implementation plan

User approved returning the interrupted AutoDL assignments to the campus server on 2026-11-16, prioritizing completed results. Both campus GPUs now report healthy and idle. Keep all frozen trainers, datasets, feature banks, augmentation schedules, seed0, 100 epochs, batch8 and640 unchanged.

1. Verify the current boot, absence of old workers, original plans and complete checkpoint states. Preserve byte-identical checkpoint backups for VOC INT8 epoch78, VOC LOFF-TA epoch9 and SKU Detail25 epoch17. This step is complete; origin-verification.json stays beside the new recovery job.
2. Add only a recovery scheduler and CPU tests. Use original command generation, native checkpoint resume, existing validation and archival functions. Hold queue/output/GPU leases, inherit leases into workers, and merge completed records under a mutex so concurrent VOC endpoints cannot overwrite each other. Never signal other users' jobs.
3. Run two full-data lanes: GPU0 VOC INT8 then VOC Adapter; GPU1 VOC LOFF-TA then SKU Detail25 then VOC Detail25. Check actual RESUME messages, process identities and GPU allocations. Restore the original CPU saved-prediction observer; no new smoke training or repeated inference.
4. Record the new boot, physical GPU mapping, AutoDL cancellation decision, observed progress, commands and code checks. Commit and push only new scheduler, tests and lightweight evidence. Preserve original provenance and all old artifacts. Mixed runtime conditions remain observational costs.

Source worktree: /mnt/data/users/zhjia/YOLO-Master-paper-baselines-20261107.
Recovery job: runs/paper/reboot-recovery/job-20261116-campus-two-gpu.
Original priorities: runs/paper/priority/job-20261112-single-seed.
