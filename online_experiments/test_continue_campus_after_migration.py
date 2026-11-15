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
        for method in ("master-n", "yolo12-n"):
            output = campus.original.output_for(self.args, method)
            output.mkdir(parents=True)
            (output / "DONE.json").write_text("{}")

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
            output.mkdir(parents=True, exist_ok=True)
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

    def test_waiting_voc_after_frofa_is_a_valid_no_child_snapshot(self):
        self.make_check_fixture()
        self.args.adopt_child = 0
        campus.original.write(
            self.args.job / "pipeline-status.json",
            {"lanes": {"baseline": {"state": "WAITING_OVERLAP", "method": "frofa-c2-d01"}}},
        )
        completed = campus.original.read(self.args.job / "completed-methods.json")
        completed.append({"method": "frofa-c2-d01"})
        campus.original.write(self.args.job / "completed-methods.json", completed)
        output = campus.original.output_for(self.args, "frofa-c2-d01")
        output.mkdir(parents=True)
        (output / "DONE").write_text("SUCCEEDED\n")
        self.assertIsNone(
            campus.resolve_adopted_method(
                self.args, None, {"method": "frofa-c2-d01", "state": "WAITING_OVERLAP"}, completed
            )
        )

    def test_idle_snapshot_rejects_unsealed_frofa(self):
        self.args.adopt_child = 0
        completed = [{"method": m} for m in campus.original.ordered_methods(self.args)[:3]]
        with self.assertRaisesRegex(RuntimeError, "DONE"):
            campus.resolve_adopted_method(
                self.args, None, {"method": "frofa-c2-d01", "state": "WAITING_OVERLAP"}, completed
            )

    def test_sku_later_adapter_requires_five_sealed_predecessors(self):
        self.args.dataset, self.args.first_detector, self.args.delegated_method = "sku110k", "yolo12-n", "detail25"
        completed = [{"method": m} for m in campus.original.ordered_methods(self.args)[:5]]
        for row in completed:
            output = campus.original.output_for(self.args, row["method"])
            output.mkdir(parents=True, exist_ok=True)
            (output / ("DONE.json" if row["method"] in campus.original.DIRECT else "DONE")).write_text("{}")
        method = campus.resolve_adopted_method(
            self.args, {"pid": 20}, {"method": "vit-adapter-cache25", "child_pid": 20}, completed
        )
        self.assertEqual(method, "vit-adapter-cache25")
        with self.assertRaisesRegex(RuntimeError, "prefix"):
            campus.resolve_adopted_method(
                self.args, {"pid": 20}, {"method": "vit-adapter-cache25", "child_pid": 20}, completed[:-1]
            )

    def test_remaining_queue_skips_delegated_run_without_creating_done(self):
        events = []
        output = campus.original.output_for(self.args, "frofa-c2-d01")
        output.mkdir(parents=True)
        (output / "DONE").write_text("SUCCEEDED\n")
        completed = campus.original.read(self.args.job / "completed-methods.json")
        completed.append({"method": "frofa-c2-d01"})
        campus.original.write(self.args.job / "completed-methods.json", completed)
        with patch.object(campus.shared, "training_processes", return_value=[]), patch.object(
            campus, "finish", side_effect=lambda a, m: events.append("finish:" + m)
        ), patch.object(campus, "train_one", side_effect=lambda a, m: events.append("train:" + m)):
            campus.continue_queue(self.args, {"child": {"pid": 20}})
        self.assertEqual(events, ["train:cache-reuse-c1-int8", "train:vit-adapter-cache25", "train:detail25"])
        self.assertFalse((campus.original.output_for(self.args, self.args.delegated_method) / "DONE").exists())
        status = campus.original.read(self.args.job / "pipeline-status.json")
        self.assertEqual(status["state"], "LOCAL_COMPLETE_REMOTE_PENDING")

    def test_wait_for_natural_supervisor_exit_never_signals(self):
        old, _ = self.make_check_fixture()
        with patch.object(campus.shared, "process", side_effect=[old, None]), patch.object(
            campus.time, "sleep"
        ), patch.object(campus.shared, "signal_one") as send, patch.object(campus.shared, "handoff") as handoff:
            campus.wait_previous_supervisor(self.args, {"supervisor": old, "child": None})
        send.assert_not_called()
        handoff.assert_not_called()
        self.assertEqual(
            campus.original.read(self.args.sidecar / "handoff.json")["state"], "PREVIOUS_SUPERVISOR_EXITED_NATURALLY"
        )

    def test_wait_refuses_reused_supervisor_pid(self):
        old, _ = self.make_check_fixture()
        with patch.object(campus.shared, "process", return_value=dict(old, start_ticks=999)), self.assertRaisesRegex(
            RuntimeError, "lifetime"
        ):
            campus.wait_previous_supervisor(self.args, {"supervisor": old, "child": None})

    @unittest.skipUnless(sys.platform == "linux", "Requires a real Linux flock")
    def test_canonical_lock_waits_for_real_busy_flock_then_releases(self):
        import fcntl

        path = self.args.job / "pipeline.lock"
        with path.open("a") as held:
            fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

            def release(delay):
                self.assertEqual(delay, 10)
                self.assertEqual(
                    campus.original.read(self.args.sidecar / "status.json")["state"], "WAITING_CANONICAL_LOCK"
                )
                fcntl.flock(held.fileno(), fcntl.LOCK_UN)

            with patch.object(campus.time, "sleep", side_effect=release) as sleep:
                with campus.canonical_queue_lock(self.args), self.assertRaises(BlockingIOError):
                    fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                sleep.assert_called_once_with(10)
            # The continuation's finally block released its actual flock.
            fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(held.fileno(), fcntl.LOCK_UN)

    def test_queue_rejects_remaining_live_training_owner(self):
        argv = ["python", "train", "--output", str(campus.original.output_for(self.args, "frofa-c2-d01"))]
        with patch.object(
            campus.shared, "training_processes", return_value=[{"pid": 30, "argv": argv}]
        ), self.assertRaisesRegex(RuntimeError, "remains alive"):
            campus.continue_queue(self.args, {"child": None})

    def test_retirement_snapshot_accepts_naturally_exited_supervisor_without_signal(self):
        from online_experiments import return_autodl_results as receiver

        old, _ = self.make_check_fixture()
        self.args.adopt_child = 0
        self.args.retirement = self.root / "retirement.json"
        self.args.bundle_root = self.root / "bundle"
        campus.original.write(
            self.args.retirement, {"schema": "autodl-migration-retirement-v1", "cases": {"voc0712": {"retired": True}}}
        )
        metadata = campus.original.read(self.args.job / "worker.json")
        case = {"queue": self.args.job, "method": self.args.delegated_method}
        with patch.object(receiver, "load_case", return_value=case), patch.object(
            receiver, "validate_retirement", return_value={"retired": True}
        ), patch.object(receiver, "retired_worker_inactive", return_value=True), patch.object(
            campus.shared, "training_processes", return_value=[]
        ), patch.object(campus.os, "getuid", return_value=7, create=True), patch.object(
            campus.shared, "signal_one"
        ) as send:
            evidence = campus.retirement_snapshot(self.args, metadata, {"state": "FAILED"}, "boot", None)
        self.assertEqual(evidence["supervisor"]["state"], "EXITED_AT_SNAPSHOT")
        self.assertEqual(evidence["supervisor"]["pid"], old["pid"])
        self.assertIsNone(evidence["child"])
        send.assert_not_called()

    def test_retirement_check_uses_verified_sidecar_when_admission_overwrites_status(self):
        old, _ = self.make_check_fixture()
        self.args.adopt_child = 0
        self.args.retirement = self.root / "retirement.json"
        prior = self.root / "prior-sidecar"
        prior.mkdir()
        metadata = campus.original.read(self.args.job / "worker.json")
        metadata["argv"] += ["--sidecar", str(prior)]
        old["argv"] = [sys.executable, "-u", *metadata["argv"]]
        campus.original.write(self.args.job / "worker.json", metadata)
        baseline = {"state": "WAITING_GPU", "method": "cache-reuse-c1-int8"}
        campus.original.write(
            prior / "status.json", {"schema": "priority-voc-overlap-v1", "pid": 10, "lanes": {"baseline": baseline}}
        )
        campus.original.write(
            self.args.job / "pipeline-status.json", {"state": "WAITING_GPU", "message": "GPU0 free=16142 need16384"}
        )
        actual_read = Path.read_text

        def read_text(path, *args, **kwargs):
            return (
                "boot\n" if path.as_posix() == "/proc/sys/kernel/random/boot_id" else actual_read(path, *args, **kwargs)
            )

        with patch.object(campus.sys, "platform", "linux"), patch.object(
            campus.os, "getuid", return_value=7, create=True
        ), patch.object(campus.shared, "pidfd_capability"), patch.object(
            campus.original, "plan", return_value={"frozen": True}
        ), patch.object(campus.shared, "process", return_value=old), patch.object(
            Path, "read_text", read_text
        ), patch.object(campus, "retirement_snapshot", return_value={"verified": True}) as snapshot:
            self.assertEqual(campus.check(self.args), {"verified": True})
        self.assertEqual(snapshot.call_args.args[2], baseline)

    def test_prior_sidecar_status_rejects_different_owner(self):
        self.make_check_fixture()
        prior = self.root / "prior-sidecar"
        prior.mkdir()
        metadata = campus.original.read(self.args.job / "worker.json")
        metadata["argv"] += ["--sidecar", str(prior)]
        campus.original.write(
            prior / "status.json", {"schema": "priority-voc-overlap-v1", "pid": 999, "lanes": {"baseline": {}}}
        )
        with self.assertRaisesRegex(RuntimeError, "another supervisor"):
            campus.prior_sidecar_baseline(self.args, metadata)

    def test_retirement_snapshot_rejects_still_live_delegated_process(self):
        from online_experiments import return_autodl_results as receiver

        self.make_check_fixture()
        self.args.retirement = self.root / "retirement.json"
        self.args.bundle_root = self.root / "bundle"
        campus.original.write(
            self.args.retirement, {"schema": "autodl-migration-retirement-v1", "cases": {"voc0712": {}}}
        )
        case = {"queue": self.args.job, "method": self.args.delegated_method}
        with patch.object(receiver, "load_case", return_value=case), patch.object(
            receiver, "validate_retirement", return_value={}
        ), patch.object(receiver, "retired_worker_inactive", return_value=False), self.assertRaisesRegex(
            RuntimeError, "still alive"
        ):
            campus.retirement_snapshot(self.args, {}, {}, "boot", None)

    def test_retirement_snapshot_rejects_live_worker_after_supervisor_exit(self):
        from online_experiments import return_autodl_results as receiver

        self.make_check_fixture()
        self.args.retirement = self.root / "retirement.json"
        self.args.bundle_root = self.root / "bundle"
        campus.original.write(
            self.args.retirement, {"schema": "autodl-migration-retirement-v1", "cases": {"voc0712": {}}}
        )
        case = {"queue": self.args.job, "method": self.args.delegated_method}
        worker = {"pid": 99, "argv": campus.original.command_for(self.args, "train", "frofa-c2-d01")}
        with patch.object(receiver, "load_case", return_value=case), patch.object(
            receiver, "validate_retirement", return_value={}
        ), patch.object(receiver, "retired_worker_inactive", return_value=True), patch.object(
            campus.shared, "training_processes", return_value=[worker]
        ), self.assertRaisesRegex(RuntimeError, "exact old supervisor"):
            campus.retirement_snapshot(self.args, {}, {"method": "frofa-c2-d01", "child_pid": 99}, "boot", None)

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


class CampusAdmission(unittest.TestCase):
    def test_16142_mib_admits_second_worker_without_double_reservation(self):
        self.assertIsNone(campus.admission_reason(12288, 1, 1, 16142, 16384))

    def test_gpu_and_host_memory_each_block_below_required_floor(self):
        self.assertEqual(campus.admission_reason(12288, 1, 1, 12287, 16384), "insufficient_free_gpu_memory")
        self.assertEqual(campus.admission_reason(12288, 1, 1, 16142, 12287), "insufficient_available_host_memory")
        self.assertIsNone(campus.admission_reason(12288, 1, 1, 12288, 12288))

    def test_two_existing_project_workers_block_even_when_memory_is_free(self):
        self.assertEqual(campus.admission_reason(12288, 0, 2, 24000, 64000), "two_project_training_workers_active")

    def test_two_held_stage_leases_block_even_when_workers_not_visible(self):
        self.assertEqual(campus.admission_reason(12288, 2, 0, 24000, 64000), "two_stage_leases_occupied")

    def test_gpu_safety_floor_cannot_be_disabled(self):
        with self.assertRaisesRegex(ValueError, "8192"):
            campus.admission_reason(8191, 0, 0, 24000, 64000)

    @unittest.skipUnless(sys.platform == "linux", "Requires real Linux flock")
    def test_lease_waits_and_releases_fd_after_training_exception(self):
        import fcntl

        with tempfile.TemporaryDirectory() as directory:
            lock_root = Path(directory)
            args = SimpleNamespace(gpu=0, min_free_mib=12288)
            workers = [{"pid": 1, "state": "T"}]
            with (lock_root / "gpu0-slot0.lock").open("a") as held:
                fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch.object(campus, "GPU_LOCK_ROOT", lock_root), patch.object(
                    campus, "gpu_memory", side_effect=[12287, 16142]
                ), patch.object(campus.shared, "host_memory", return_value=16384), patch.object(
                    campus.shared, "training_processes", return_value=workers
                ), patch.object(campus.time, "sleep") as sleep, patch.object(campus, "emit") as emit:
                    with self.assertRaisesRegex(RuntimeError, "training failed"):  # noqa: SIM117 - explicit failure scope
                        with campus.gpu_admission(args, "VOC INT8") as free:
                            self.assertEqual(free, 16142)
                            self.assertIsInstance(args.gpu_lock_fd, int)
                            with (lock_root / "gpu0-slot1.lock").open("a") as competing:  # noqa: SIM117 - inspect the live lease
                                with self.assertRaises(BlockingIOError):
                                    fcntl.flock(competing.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                            raise RuntimeError("training failed")
                    sleep.assert_called_once_with(15)
                    self.assertEqual(emit.call_args.args[1], "WAITING_RESOURCES")
                    self.assertEqual(emit.call_args.kwargs["admission"]["reason"], "insufficient_free_gpu_memory")
                self.assertFalse(hasattr(args, "gpu_lock_fd"))
                with (lock_root / "gpu0-slot1.lock").open("a") as available:
                    fcntl.flock(available.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


if __name__ == "__main__":
    unittest.main()
