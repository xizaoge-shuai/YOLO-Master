"""CPU scheduling contracts for continuing campus work around delegated runs."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    from online_experiments import continue_campus_after_migration as campus
except ImportError:
    campus = None


class CampusSafety(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(campus, "Campus continuation driver is missing")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args = SimpleNamespace(
            job=self.root / "priority",
            dataset_job=self.root / "dataset",
            sidecar=self.root / "sidecar",
            dataset="voc0712",
            delegated_method="loffta-c2-n005",
            first_detector="master-n",
            old_supervisor=10,
            adopt_child=20,
        )
        self.args.job.mkdir()
        self.args.sidecar.mkdir()
        campus.original.write(
            self.args.job / "completed-methods.json", [{"method": "master-n"}, {"method": "yolo12-n"}]
        )

    def test_only_exact_dataset_supervisor_script_and_job_are_accepted(self):
        script = str(campus.ROOT / "online_experiments/priority_voc_overlap.py")
        metadata = [script, "run", "--job", str(self.args.job)]
        for argv in ([sys.executable, "-u", *metadata], [sys.executable, *metadata]):
            self.assertTrue(campus.supervisor_matches(self.args, argv, metadata))
        self.assertFalse(campus.supervisor_matches(self.args, [sys.executable, "other.py"], metadata))
        self.args.job = self.root / "different"
        self.assertFalse(campus.supervisor_matches(self.args, [sys.executable, *metadata], metadata))

    def make_check_fixture(self):
        script = str(campus.ROOT / "online_experiments/priority_voc_overlap.py")
        metadata_argv = [script, "run", "--job", str(self.args.job)]
        old = {"pid": 10, "uid": 7, "state": "S", "start_ticks": 11, "argv": [sys.executable, "-u", *metadata_argv]}
        child = {
            "pid": 20,
            "uid": 7,
            "state": "S",
            "start_ticks": 12,
            "ppid": 1,
            "argv": campus.original.command_for(self.args, "train", "frofa-c2-d01"),
        }
        campus.original.write(
            self.args.job / "worker.json",
            {"pid": 10, "boot_id": "boot", "scheduler": "priority-voc-overlap-v1", "argv": metadata_argv},
        )
        campus.original.write(
            self.args.job / "active-child.json",
            {"pid": 20, "action": "train", "method": "frofa-c2-d01", "argv": child["argv"]},
        )
        campus.original.write(
            self.args.job / "pipeline-status.json", {"lanes": {"baseline": {"method": "frofa-c2-d01", "child_pid": 20}}}
        )
        campus.original.write(self.args.job / "pipeline-plan.json", {"frozen": True})
        for method in ("master-n", "yolo12-n"):
            output = campus.original.output_for(self.args, method)
            output.mkdir(parents=True)
            (output / "DONE.json").write_text("{}")
        return old, child

    def test_check_accepts_reparented_main_child_and_never_sends_signal(self):
        old, child = self.make_check_fixture()
        actual_read = Path.read_text

        def read_text(path, *args, **kwargs):
            return (
                "boot\n" if path.as_posix() == "/proc/sys/kernel/random/boot_id" else actual_read(path, *args, **kwargs)
            )

        with patch.object(campus.sys, "platform", "linux"), patch.object(
            campus.os, "getuid", return_value=7, create=True
        ), patch.object(campus.shared, "pidfd_capability"), patch.object(
            campus.original, "plan", return_value={"frozen": True}
        ), patch.object(campus.shared, "process", side_effect=lambda pid: {10: old, 20: child}.get(pid)), patch.object(
            campus.shared, "training_processes", return_value=[child]
        ), patch.object(Path, "read_text", read_text), patch.object(campus.shared, "signal_one") as send:
            result = campus.check(self.args)
        self.assertEqual(result["child"]["ppid"], 1)
        send.assert_not_called()

    def test_completed_merge_reloads_current_file_and_preserves_remote_result(self):
        records = campus.original.read(self.args.job / "completed-methods.json")
        records.append({"method": self.args.delegated_method, "archive": {"remote": True}})
        campus.original.write(self.args.job / "completed-methods.json", records)
        with patch.object(campus.original, "archive_lock", return_value=__import__("contextlib").nullcontext()):
            campus.record_complete(self.args, "frofa-c2-d01", {"commit": "local"})
        merged = campus.original.read(self.args.job / "completed-methods.json")
        self.assertEqual([r["method"] for r in merged], ["master-n", "yolo12-n", "frofa-c2-d01", "loffta-c2-n005"])
        self.assertEqual(merged[-1]["archive"], {"remote": True})

    def test_remaining_queue_skips_delegated_run_without_creating_done(self):
        events = []
        with patch.object(campus, "wait_adopted", side_effect=lambda *a: events.append("wait")), patch.object(
            campus, "finish", side_effect=lambda a, m: events.append("finish:" + m)
        ), patch.object(campus, "train_one", side_effect=lambda a, m: events.append("train:" + m)):
            campus.continue_queue(self.args, {"child": {"pid": 20}})
        self.assertEqual(
            events,
            ["wait", "finish:frofa-c2-d01", "train:cache-reuse-c1-int8", "train:vit-adapter-cache25", "train:detail25"],
        )
        self.assertFalse((campus.original.output_for(self.args, self.args.delegated_method) / "DONE").exists())
        status = campus.original.read(self.args.job / "pipeline-status.json")
        self.assertEqual(status["state"], "LOCAL_COMPLETE_REMOTE_PENDING")

    def test_adopted_child_never_restarted_or_signalled(self):
        output = campus.original.output_for(self.args, "frofa-c2-d01")
        output.mkdir(parents=True)
        (output / "DONE").write_text("SUCCEEDED\n")
        child = {"pid": 20, "uid": 7, "start_ticks": 12, "argv": ["python", "train.py"]}
        with patch.object(campus.shared, "process", side_effect=[dict(child, ppid=1, state="S"), None]), patch.object(
            campus.time, "sleep"
        ), patch.object(campus.shared, "signal_one") as send:
            campus.wait_adopted(self.args, child)
        send.assert_not_called()

    def test_failed_adopted_child_is_not_retried(self):
        with patch.object(campus.shared, "process", return_value=None), self.assertRaisesRegex(RuntimeError, "DONE"):
            campus.wait_adopted(self.args, {"pid": 20})

    def test_status_has_explicit_delegation_lane(self):
        campus.emit(self.args, "ADOPTED_TRAINING", method="frofa-c2-d01", child_pid=20)
        status = campus.original.read(self.args.job / "pipeline-status.json")
        self.assertEqual(status["lanes"]["remote"]["state"], "DELEGATED_AUTODL")
        self.assertEqual(status["lanes"]["baseline"]["child_pid"], 20)

    def test_live_delegated_worker_blocks_new_driver_run(self):
        argv = [
            "python",
            "suite.py",
            "train",
            "--output",
            str(campus.original.output_for(self.args, self.args.delegated_method)),
        ]
        with patch.object(
            campus.shared, "training_processes", return_value=[{"pid": 30, "argv": argv}]
        ), self.assertRaisesRegex(RuntimeError, "retired"):
            campus.require_delegated_retired(self.args)


if __name__ == "__main__":
    unittest.main()
