# Resume two interrupted full-data runs on AutoDL

Scope: VOC LOFF-TA seed0 and SKU Detail25 seed0 only, both fixed final epoch100.
Use each frozen trainer's native complete-epoch checkpoint recovery. Preserve the
origin checkpoint/config/source/data bytes and history. Do not restart from zero,
extend epochs, introduce new seeds, or claim identical numerical results across GPUs.

The transfer manifest (`migrated-crossdataset-transfer-v1`) contains `origin`
(`job`, `output`, `method`, `seed`, `epoch`, `checkpoint_sha256`) and `files` entries
(`path`, `sha256`, `bytes`, `origin_stat`). Optional `source`, `target`, and `dataset`
fields are accepted. The old absolute path must resolve through the explicitly
created AutoDL alias to its listed target inside the permitted target root.

`seal` checks every transferred byte by SHA256, validates source/config/provenance
coverage and the checkpoint's original hash, and writes a host relocation seal.
Each immutable input records its original stat, new realpath, actual target stat,
and byte hash. The seal binds the wrapper and frozen Python sources. Runtime
outputs are recorded as origin evidence but are excluded from the stat map.

`run` installs a narrow adapter for the frozen suite's `stamp`: a request must be
in that verified immutable whitelist; its realpath and current real stat must
still match the seal before returning the origin stamp. Unlisted files fail.
Frozen provenance/config/source checks remain active. The checkpoint is inspected
on the CPU for unchanged config and a legal epoch; the origin checkpoint hash is
required until the first recorded resume, not after legitimate new epochs.

Preserve `environment.json` as byte-identical `environment-origin.json`, then let
the unchanged trainer write the real AutoDL environment. Record both hardware
phases and mixed-host cost limitations. Never relabel one GPU/environment as
another. Final endpoint receives migration-origin metadata, with `validate`
running through the same seal and stat adapter.

Every complete epoch retains optimizer, scheduler, Python/NumPy/CPU/CUDA RNG,
query counters and history. Incomplete epoch work is replayed. Keep the physical
teacher-query journal so replay costs remain visible. Migration transfer time and
discarded partial-epoch time are separate from checkpoint-accounted train time.

This module and its CPU tests are new files. No frozen trainer/source file is
edited. No GPU smoke run is part of implementation verification.

## Campus continuation

The separate `continue_campus_after_migration.py` driver adopts only the exact
currently running baseline child (VOC FroFA or SKU INT8), preserving its PID.
The caller first verifies checkpoint/transfer backups and retires the paused
delegated child explicitly; the driver refuses to run while that child remains.
Only the old supervisor receives the reviewed pidfd handoff signal. Main-child
ownership is proved by recorded worker/lane/argv/lifetime, including legitimate
reparenting from an earlier handoff.

After the adopted child finishes, validation/archiving and all other unchanged
methods continue through the original slot0/slot1 admission. The AutoDL method
is explicitly skipped, marked `DELEGATED_AUTODL`, and never receives a fake DONE.
Each completed-list update re-reads the current file under the shared archive
lock, preserving independently returned remote results. Once local work ends,
state becomes `LOCAL_COMPLETE_REMOTE_PENDING` until the real remote endpoint
returns. Optional `--recover` uses the persisted exact-child/source-sealed
handoff and sends no signal. The original paused remote process is never killed
by this scheduling module.
