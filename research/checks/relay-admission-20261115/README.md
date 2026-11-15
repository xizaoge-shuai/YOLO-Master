# AutoDL relay and campus admission: 2026-11-16

User authorized reverse forwarding and lower VOC admission. GPU/host free thresholds are 12 GiB with at most two project workers and two leases. Original seed0, full data, 100 epochs, batch8 and 640 recipes remain unchanged. VOC INT8 has completed real epochs; SKU Adapter continues. No new final AP is claimed.

Native SSH forwards disconnected after the first probe. Detached Windows relay PID25632 now retains its password only in memory and reconnects, binds loopback ports only, pins the original Ed25519 key before authentication, and serves AutoDL proxy17897 plus campus SSH19822. Copy and return still require Windows/VPN/tunnels; already failed transfer clients must be relaunched separately. Old clients were confirmed exited; new3439657/3439658 resume unchanged SHA-bound partials.

Actual 25-second large-file window: approximately0.959 MiB/s, not a whole-transfer estimate or controlled training speed. Completed tar reused and train-v0.npy resumes from5665357824 bytes. AutoDL still WAITING_TRANSFER.

Six changed/new Python files pass targeted Ruff and format checks; 121 CPU regression tests pass. These are scheduler/transport checks, not GPU smoke training. Immutable experiment sources, caches, original manifests, checkpoints and teacher journals are preserved.
