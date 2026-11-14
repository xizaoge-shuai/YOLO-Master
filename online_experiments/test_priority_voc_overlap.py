"""CPU-only safety checks for one additional VOC baseline lane."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

try:
    from online_experiments import priority_voc_overlap as lane
except ImportError:
    lane = None


class Safety(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(lane, "VOC overlap scheduler is missing")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.args = SimpleNamespace(
            job=Path(self.temp.name),
            dataset_job=Path(self.temp.name) / "old",
            sidecar=Path(self.temp.name) / "sidecar",
            dataset="voc0712",
        )
        self.args.sidecar.mkdir()
        lane.original.write(self.args.job / "completed-methods.json", [])

    def test_extra_admission_limits_four_workers_and_reserves(self):
        good = {"occupied": 3, "workers": 3, "free_mib": 12288, "host_mib": 18432}
        self.assertTrue(lane.can_admit(**good))
        for update in ({"occupied": 4}, {"workers": 4}, {"free_mib": 12287}, {"host_mib": 18431}):
            self.assertFalse(lane.can_admit(**dict(good, **update)))

    def test_adopted_frofa_is_never_restarted_or_signalled(self):
        output = self.args.job / "train/frofa-c2-d01-s0"
        output.mkdir(parents=True)
        (output / "DONE").write_text("SUCCEEDED\n")
        child = {"pid": 20, "uid": 7, "start_ticks": 12, "argv": ["python", "train.py"]}
        with patch.object(lane.shared, "process", side_effect=[dict(child, ppid=1, state="S"), None]), patch.object(
            lane.time, "sleep"
        ), patch.object(lane.shared, "signal_one") as send:
            lane.wait_adopted(self.args, child)
        send.assert_not_called()

    def test_adopted_exit_without_done_is_failure(self):
        with patch.object(lane.shared, "process", return_value=None), self.assertRaisesRegex(RuntimeError, "DONE"):
            lane.wait_adopted(self.args, {"pid": 20})

    def test_baseline_waits_overlap_then_skips_duplicate_loff(self):
        events = []
        state = SimpleNamespace(
            update=lambda *a, **k: None, overlap_done=SimpleNamespace(wait=lambda: events.append("wait:loff"))
        )
        with patch.object(lane, "wait_adopted", side_effect=lambda *a: events.append("adopt")), patch.object(
            lane, "finish", side_effect=lambda a, m, s, l: events.append("finish:" + m)
        ), patch.object(lane, "train_one", side_effect=lambda a, m, l, s: events.append("train:" + m)):
            lane.baseline(self.args, {"pid": 20}, state)
        self.assertEqual(
            events,
            [
                "adopt",
                "finish:frofa-c2-d01",
                "wait:loff",
                "train:cache-reuse-c1-int8",
                "train:vit-adapter-cache25",
                "train:detail25",
            ],
        )

    def test_failed_overlap_sets_event_and_preserves_other_lane(self):
        state = SimpleNamespace(lock=__import__("threading").RLock(), errors=[], update=Mock(), overlap_done=Mock())
        with patch.object(lane, "train_one", side_effect=RuntimeError("worker failure")):
            lane.run_thread(self.args, "extra", {}, state)
        self.assertEqual(len(state.errors), 1)
        state.overlap_done.set.assert_called_once_with()
        self.assertEqual(state.update.call_args.args[:2], ("extra", "FAILED"))

    def test_recovery_verifies_exact_child_and_never_signals(self):
        old = {"pid": 10, "uid": 7, "start_ticks": 12, "argv": ["python", "old.py"]}
        child = dict(old, pid=20, argv=["python", "train.py"], state="S")
        with patch.object(lane.shared, "process", side_effect=[None, dict(child, ppid=1)]), patch.object(
            lane.shared, "signal_one"
        ) as send:
            lane.shared.verify_recovery_processes({"supervisor": old, "child": child})
        send.assert_not_called()

    def test_completed_order_and_lane_status_are_canonical(self):
        state = lane.State(self.args)
        state.update("extra", "TRAIN", method=lane.LOFF, child_pid=123)
        status = lane.original.read(self.args.job / "pipeline-status.json")
        self.assertEqual(status["lanes"]["extra"]["method"], lane.LOFF)
        self.assertEqual(status["state"], "RUNNING_VOC_OVERLAP")

    def test_existing_regular_training_log_is_preserved(self):
        alias = self.args.job / f"train-{lane.LOFF}.log"
        alias.write_text("old log\n")
        with self.assertRaisesRegex(RuntimeError, "Refuse overwriting"):
            lane.training_log_alias(self.args, lane.LOFF)
        self.assertEqual(alias.read_text(), "old log\n")

    @unittest.skipUnless(__import__("sys").platform == "linux", "POSIX symlink")
    def test_training_log_alias_is_idempotent_and_rejects_changed_target(self):
        lane.training_log_alias(self.args, lane.LOFF)
        lane.training_log_alias(self.args, lane.LOFF)
        alias = self.args.job / f"train-{lane.LOFF}.log"
        target = self.args.sidecar / f"train-{lane.LOFF}.log"
        target.write_text("epoch 1\n")
        self.assertEqual(alias.read_text(), "epoch 1\n")
        alias.unlink()
        alias.symlink_to(self.args.sidecar / "other.log")
        with self.assertRaisesRegex(RuntimeError, "different target"):
            lane.training_log_alias(self.args, lane.LOFF)

    def test_extra_lease_holds_slot3_and_passes_descriptor(self):
        try:
            import fcntl
        except ImportError:
            self.skipTest("POSIX flock only")
        lock_root = self.args.job / "locks"
        state = SimpleNamespace(update=Mock())
        with patch.object(lane, "GPU_LOCK_ROOT", lock_root), patch.object(
            lane.shared, "training_processes", return_value=[{}, {}, {}]
        ), patch.object(lane.shared, "host_memory", return_value=20000), patch.object(
            lane, "gpu_memory", return_value=14000
        ), lane.extra_lease(self.args, state):
            self.assertGreaterEqual(self.args.gpu_lock_fd, 0)
            with (lock_root / "gpu0-slot3.lock").open("a") as probe, self.assertRaises(BlockingIOError):
                fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.assertFalse(hasattr(self.args, "gpu_lock_fd"))


if __name__ == "__main__":
    unittest.main()
