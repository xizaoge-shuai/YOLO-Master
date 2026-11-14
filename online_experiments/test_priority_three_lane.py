"""CPU-only safety checks for a supervisor-only three-lane handoff."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

try:
    from online_experiments import priority_three_lane as lane
except ImportError:
    lane = None


class Safety(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(lane, "three-lane handoff implementation is missing")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.args = SimpleNamespace(job=Path(self.temp.name), dataset_job=Path(self.temp.name) / "old", sidecar=Path(self.temp.name) / "sidecar")
        self.args.sidecar.mkdir()

    def test_pid_or_argv_mismatch_refuses_signal(self):
        expected = {"pid": 10, "uid": 7, "start_ticks": 12, "argv": ["python", "old.py"]}
        for update in ({"pid": 11}, {"argv": ["python", "other.py"]}, {"start_ticks": 13}, {"uid": 8}):
            with self.subTest(update=update), self.assertRaises(RuntimeError):
                lane.same_process(expected, dict(expected, **update))

    def test_real_unbuffered_supervisor_argv_matches_sys_argv_metadata(self):
        import sys
        script = str(lane.ROOT / "online_experiments/priority_single_seed_campaign.py")
        metadata = [script, "run", "--job", "/exact/sku110k"]
        self.assertTrue(lane.supervisor_argv([sys.executable, "-u", *metadata], metadata))
        self.assertTrue(lane.supervisor_argv([sys.executable, *metadata], metadata))
        self.assertFalse(lane.supervisor_argv([sys.executable, "-u", "other.py", *metadata[1:]], metadata))

    def test_missing_python_pidfd_binding_uses_libc(self):
        with patch.object(lane.os, "pidfd_open", None, create=True), patch.object(lane, "libc_pidfd", return_value=71) as libc:
            self.assertEqual(lane.open_pidfd(123), 71)
        libc.assert_called_once_with("pidfd_open", 123)

    def test_pidfd_capability_uses_signal_zero_and_closes_descriptor(self):
        with patch.object(lane, "open_pidfd", return_value=71), patch.object(lane, "send_pidfd") as send, patch.object(lane.os, "close") as close:
            lane.pidfd_capability()
        send.assert_called_once_with(71, 0)
        close.assert_called_once_with(71)

    def test_pidfd_sender_has_libc_fallback_without_raw_pid_kill(self):
        with patch.object(lane.signal, "pidfd_send_signal", None, create=True), patch.object(lane, "libc_pidfd", return_value=0) as libc:
            lane.send_pidfd(71, 0)
        libc.assert_called_once_with("pidfd_send_signal", 71, 0)

    @unittest.skipUnless(__import__("sys").platform == "linux", "requires Linux pidfds")
    def test_real_self_pidfd_signal_zero(self):
        lane.pidfd_capability()

    def test_only_verified_supervisor_is_signalled(self):
        old = {"pid": 10, "uid": 7, "start_ticks": 12, "argv": ["python", "old.py"], "state": "S"}
        child = dict(old, pid=20, ppid=10, argv=["python", "train.py"])
        snapshots = {10: [old, None], 20: [child]}
        def probe(pid):
            rows = snapshots[pid]
            return rows.pop(0) if len(rows) > 1 else rows[0]
        signal_old = Mock()
        with patch.object(lane, "process", side_effect=probe), patch.object(lane, "signal_one", signal_old):
            lane.handoff(self.args, {"supervisor": old, "child": child})
        signal_old.assert_called_once_with(old)
        self.assertTrue((self.args.sidecar / "handoff.json").is_file())

    def test_identity_changed_before_handoff_never_signals(self):
        old = {"pid": 10, "uid": 7, "start_ticks": 12, "argv": ["python", "old.py"]}
        with patch.object(lane, "process", return_value=dict(old, start_ticks=99)), patch.object(lane, "signal_one") as send:
            with self.assertRaises(RuntimeError):
                lane.handoff(self.args, {"supervisor": old, "child": dict(old, pid=20)})
        send.assert_not_called()

    def test_adopted_child_can_be_reparented_without_restart(self):
        output = self.args.job / "train/cache-reuse-c1-int8-s0"
        output.mkdir(parents=True)
        (output / "DONE").write_text("SUCCEEDED\n")
        child = {"pid": 20, "uid": 7, "start_ticks": 12, "argv": ["python", "train.py"]}
        with patch.object(lane, "process", side_effect=[dict(child, ppid=1, state="S"), None]), patch.object(lane.time, "sleep"):
            lane.wait_adopted(self.args, child)

    def test_unsealed_adopted_endpoint_is_failure(self):
        with patch.object(lane, "process", return_value=None), self.assertRaisesRegex(RuntimeError, "DONE"):
            lane.wait_adopted(self.args, {"pid": 20})

    def test_extra_admission_enforces_three_workers_memory_and_slot(self):
        good = dict(slot=2, occupied=2, workers=2, free_mib=14336, host_mib=12288)
        self.assertTrue(lane.can_admit(**good))
        for update in ({"workers": 3}, {"free_mib": 14335}, {"host_mib": 12287}, {"occupied": 3}):
            with self.subTest(update=update):
                self.assertFalse(lane.can_admit(**dict(good, **update)))
        self.assertFalse(lane.can_admit(slot=1, occupied=2, workers=2, free_mib=15000, host_mib=20000))

    def test_baseline_waits_detail_and_does_not_train_it(self):
        events = []
        state = SimpleNamespace(update=lambda *a, **k: None)
        with patch.object(lane, "wait_adopted", side_effect=lambda *a: events.append("adopt")), patch.object(lane, "finish", side_effect=lambda a, m, s: events.append("finish:" + m)), patch.object(lane, "train_one", side_effect=lambda a, m, l, s: events.append("train:" + m)), patch.object(lane, "wait_detail", side_effect=lambda *a: events.append("wait:detail25")):
            lane.baseline(self.args, {"pid": 20}, state)
        self.assertEqual(events, ["adopt", "finish:cache-reuse-c1-int8", "train:vit-adapter-cache25", "wait:detail25"])

    @unittest.skipUnless(__import__("os").name == "posix", "requires POSIX flock")
    def test_duplicate_job_lock_is_rejected(self):
        from online_experiments.crossdataset_campaign import Busy, job_lock
        with job_lock(self.args.job), self.assertRaises(Busy), job_lock(self.args.job):
            pass


if __name__ == "__main__":
    unittest.main()
