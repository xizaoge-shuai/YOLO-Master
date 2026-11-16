"""CPU contracts for a reboot recovery that preserves the frozen experiments."""

import contextlib
import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    from online_experiments import resume_campus_after_reboot as recovery
except ImportError:
    recovery = None


class RecoveryContracts(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(recovery, "reboot recovery driver is missing")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args = SimpleNamespace(
            dataset="voc0712", job=self.root / "voc", dataset_job=self.root / "data", first_detector="master-n"
        )
        self.args.job.mkdir()

    def test_two_lanes_cover_each_authorized_pending_case_once(self):
        self.assertEqual(
            recovery.LANES,
            {
                0: (("voc0712", "cache-reuse-c1-int8"), ("voc0712", "vit-adapter-cache25")),
                1: (("voc0712", "loffta-c2-n005"), ("sku110k", "detail25"), ("voc0712", "detail25")),
            },
        )

    def test_frozen_plan_is_checked_without_rewriting(self):
        path = self.args.job / "pipeline-plan.json"
        recovery.original.write(path, {"gpu": 0, "methods": ["frozen"]})
        before = path.read_bytes()
        with patch.object(recovery.original, "plan", return_value={"gpu": 0, "methods": ["frozen"]}):
            recovery.check_frozen_plan(self.args)
        self.assertEqual(before, path.read_bytes())
        with patch.object(recovery.original, "plan", return_value={"gpu": 1, "methods": ["changed"]}):  # noqa: SIM117
            with self.assertRaisesRegex(RuntimeError, "Frozen"):
                recovery.check_frozen_plan(self.args)
        self.assertEqual(before, path.read_bytes())

    def test_exact_output_workers_refuse_orphan_even_if_saved_pid_is_stale(self):
        output = str(recovery.original.output_for(self.args, "detail25"))
        workers = [{"pid": 9, "argv": ["python", "train", "--output", output]}]
        with patch.object(recovery.shared, "training_processes", return_value=workers):
            with self.assertRaisesRegex(RuntimeError, "already owns"):
                recovery.require_no_output_owner(self.args, "detail25")
            recovery.require_no_output_owner(self.args, "loffta-c2-n005")

    def test_completion_merge_retains_parallel_lane_and_frozen_order(self):
        path = self.args.job / "completed-methods.json"
        recovery.original.write(path, [{"method": "detail25", "archive": {"pushed": True}}])
        with patch.object(recovery.original, "archive_lock", contextlib.nullcontext):
            recovery.record_complete(self.args, "loffta-c2-n005", {"pushed": False})
        rows = recovery.original.read(path)
        self.assertEqual([row["method"] for row in rows], ["loffta-c2-n005", "detail25"])
        self.assertTrue(rows[1]["archive"]["pushed"])

    def test_checkpoint_evidence_rejects_epoch_or_config_disagreement(self):
        state = {
            "args": {"method": "detail25", "provenance_sha256": "abc"},
            "epoch": 2,
            "rows": [{"epoch": 1}, {"epoch": 2}],
            "summary": {"completed_epochs": 2, "epochs": [{"epoch": 1}, {"epoch": 2}], "provenance_sha256": "abc"},
        }
        recovery.validate_checkpoint_state(state, state["args"])
        altered = copy.deepcopy(state)
        altered["epoch"] = 3
        with self.assertRaisesRegex(RuntimeError, "checkpoint"):
            recovery.validate_checkpoint_state(altered, state["args"])
        with self.assertRaisesRegex(RuntimeError, "checkpoint"):
            recovery.validate_checkpoint_state(state, {"method": "other"})

    @unittest.skipUnless(importlib.util.find_spec("fcntl"), "Linux flock required")
    def test_gpu_lease_blocks_second_worker_on_same_gpu_and_holds_child_fd(self):
        import fcntl

        lockroot = self.root / "locks"
        with patch.object(recovery, "GPU_LOCK_ROOT", lockroot):
            with recovery.try_gpu_lease(0, 12288, gpu_free=lambda _: 20000, host_free=lambda: 20000) as first:
                self.assertIsNotNone(first)
                with recovery.try_gpu_lease(0, 12288, gpu_free=lambda _: 20000, host_free=lambda: 20000) as second:
                    self.assertIsNone(second)
                with recovery.try_gpu_lease(1, 12288, gpu_free=lambda _: 20000, host_free=lambda: 20000) as other_gpu:
                    self.assertIsNotNone(other_gpu)
                with (lockroot / "gpu0-slot0.lock").open("a") as contender:  # noqa: SIM117
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with recovery.try_gpu_lease(0, 12288, gpu_free=lambda _: 20000, host_free=lambda: 20000) as next_owner:
                self.assertIsNotNone(next_owner)

    @unittest.skipUnless(importlib.util.find_spec("fcntl"), "Linux flock required")
    def test_legacy_other_slot_prevents_new_slot0_admission(self):
        import fcntl

        lockroot = self.root / "locks"
        lockroot.mkdir()
        with (lockroot / "gpu1-slot2.lock").open("a") as old:
            fcntl.flock(old.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch.object(recovery, "GPU_LOCK_ROOT", lockroot):  # noqa: SIM117
                with recovery.try_gpu_lease(1, 12288, gpu_free=lambda _: 20000, host_free=lambda: 20000) as lease:
                    self.assertIsNone(lease)


if __name__ == "__main__":
    unittest.main()
