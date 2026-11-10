# Session progress

2026-11-10:
- User requests concurrentVOC/SKU plus targeted new evidence. Dispatched three independent tasks:verified full raw dataset preparation;newmanifest/mmap training adapter;CPU-only full548prediction and learning-curve analysis.
- Read current progress and supplied suggestions. Seven seed0 complete;Detail0 epoch38. CurrentGPU1 is unusable(requiresreset);CPU/data preparation continues inparallel,GPUtraining requireshealthy-deviceadmission.
- Installed pinned metric packages in task-specific target directory,without modifying active training environment. NoGPU smoke/subset or method tuning.

2026-11-08:

- Read relevant literature/design instructions and kept the existing method frozen.
- Inspected official YOLO-Master dataset settings and benchmark maintainers' sources.
- Read-only inventory found no full COCO training data.
- Verified every COCO temporary extraction file against a retained copy or ZIP.
- Removed only the recoverable temporary extraction; preserved all datasets, checkpoints, predictions, archives and active feature banks.
- Existing full VisDrone queue continues, with no failed cases at the last check.
- Dataset expansion is staged after existing metrics, following the user's latest direction.
- Archived the cleanup evidence and dataset/evidence plan, updated guide/ledger/PROGRESS; commit81b2d0f03ec1a26f98b98903daf7d79fce455469 pushed and remote HEAD verified.
- Final read: adapter q25 epoch59/100, existing full queue continues. New dataset downloads and training have not started.

Later user direction: finish original metrics; on the three new formal datasets complete baselines before our method and ablations.

- Attempted fresh read-only SSH query: server reset the connection.
- Bounded retry: banner exchange timeout; no current server status obtained.
- Updated the existing dataset plan locally to baseline-first. No remote mutation, new training, cleanup or method changes in this follow-up.
- New plan change is not yet committed/pushed; retain the last verified remote commit81b2d0f03ec1a26f98b98903daf7d79fce455469.

2026-11-09 AutoDL continuation:
- User changed execution to one dataset at a time, VOC07+12 then SKU-110K, full COCO last.
- Password SSH and loopback reverse proxy work. Existing Miniconda reused, isolated Python3.10.20 setup runs in background. Persistent root public-key access was rejected by auto-review and was not installed.
- User repository cloned at81b2d0f03ec1a26f98b98903daf7d79fce455469 into a separate migration branch. Full VOC raw-data preparation started; no new training or AP.
- Data disk remains50GiB, insufficient for full feature banks. Old campus host remains inaccessible, so old checkpoint recovery is pending.
- Added explicit command guide and migrated the baseline-first plan to sequential dataset execution. Publication commit/push status is recorded by the current task, not inferred from old success records.

Later2026-11-09 returned to original server:
- Read-only capacity/process audit distinguishes root-disk shortage from free RAM/VRAM. Existing RUNNING marker was stale after reboot; no worker existed.
- Verified original810 source hashes, environment, manifests/protocol,3 completed runs,548-prediction bindings and adapter59-epoch checkpoint including optimizer/scheduler/RNG/query counters.
- Preserved original checkpoint/summary/environment/rows and reboot identity outside Git; resumed original launcher at13:34Asia/Shanghai. Log confirms epoch60 onGPU0, completed rows skipped.
- Current order is VisDrone necessary evidence closure, VOC, SKU, fullCOCO last. AutoDL preparations are historical; no new AutoDL experiment is scheduled. No dataset/cache deletion, model change, smoke/subset trial or new AP claimed.
