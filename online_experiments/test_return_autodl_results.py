"""CPU-only contracts for bounded download and last-visible completion seals."""

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

try:
    from online_experiments import return_autodl_results as back
except ImportError:
    back = None


def digest(value):
    return hashlib.sha256(value).hexdigest()


class SFTP:
    def __init__(self, data):
        self.data = data

    def open(self, path, mode="rb"):
        return io.BytesIO(self.data[path])

    def lstat(self, path):
        import stat

        return SimpleNamespace(st_size=len(self.data[path]), st_mode=stat.S_IFREG | 0o600)


class Contracts(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(back, "return implementation is missing")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.case = {
            "dataset": "sku110k",
            "method": "detail25",
            "origin": {
                "output": "/mnt/data/users/zhjia/repo/runs/train/detail25-s0",
                "job": "/mnt/data/users/zhjia/repo/runs/sku",
                "method": "detail25",
                "epoch": 17,
                "seed": 0,
                "checkpoint_sha256": "a" * 64,
            },
        }
        self.remote = "/root/autodl-tmp/campus-mirror/repo/runs/train/detail25-s0/summary.json"
        self.row = {"relative": "summary.json", "remote": self.remote, "bytes": 5, "sha256": digest(b"hello")}
        self.manifest = {
            "schema": "autodl-resume-return-v1",
            "dataset": "sku110k",
            "origin": self.case["origin"],
            "files": [self.row],
        }

    def test_manifest_binds_origin_and_exact_remote_mapping(self):
        self.assertEqual(back.validate_return(self.manifest, self.case), [self.row])
        for update in (
            {"relative": "../escape"},
            {"remote": self.remote.replace("repo/runs", "other/runs")},
            {"bytes": -1},
            {"sha256": "bad"},
        ):
            with self.subTest(update=update), self.assertRaises(ValueError):
                back.validate_return(dict(self.manifest, files=[dict(self.row, **update)]), self.case)
        with self.assertRaises(ValueError):
            back.validate_return(dict(self.manifest, origin=dict(self.case["origin"], epoch=1)), self.case)

    def test_manifest_rejects_duplicate_and_unreasonable_size(self):
        with self.assertRaises(ValueError):
            back.validate_return(dict(self.manifest, files=[self.row, self.row]), self.case)
        with self.assertRaises(ValueError):
            back.validate_return(dict(self.manifest, files=[dict(self.row, bytes=32 * 1024**3)]), self.case)

    def test_download_checks_hash_before_atomic_final(self):
        target = self.root / "nested/result.json"
        back.download_file(SFTP({self.remote: b"hello"}), self.row, target, lambda _: None)
        self.assertEqual(target.read_bytes(), b"hello")
        self.assertFalse(list(target.parent.glob("*.partial.*")))

    def test_existing_wrong_final_is_preserved(self):
        target = self.root / "result.json"
        target.write_bytes(b"wrong")
        with self.assertRaises(RuntimeError):
            back.download_file(SFTP({self.remote: b"hello"}), self.row, target, lambda _: None)
        self.assertEqual(target.read_bytes(), b"wrong")

    def test_matching_partial_resumes_and_wrong_partial_refuses(self):
        target = self.root / "result.json"
        partial = back.partial_path(target, self.row)
        partial.write_bytes(b"he")
        back.download_file(SFTP({self.remote: b"hello"}), self.row, target, lambda _: None)
        self.assertEqual(target.read_bytes(), b"hello")
        target.unlink()
        partial.write_bytes(b"XX")
        with self.assertRaises(RuntimeError):
            back.download_file(SFTP({self.remote: b"hello"}), self.row, target, lambda _: None)
        self.assertEqual(partial.read_bytes(), b"XX")

    def test_empty_partial_can_resume(self):
        target = self.root / "result.json"
        back.partial_path(target, self.row).touch()
        back.download_file(SFTP({self.remote: b"hello"}), self.row, target, lambda _: None)
        self.assertEqual(target.read_bytes(), b"hello")

    def test_bad_download_never_publishes_final(self):
        target = self.root / "result.json"
        with self.assertRaises(RuntimeError):
            back.download_file(SFTP({self.remote: b"wrong"}), self.row, target, lambda _: None)
        self.assertFalse(target.exists())

    def test_stopped_original_worker_is_still_active(self):
        expected = {
            "pid": 20,
            "uid": 1010,
            "start_ticks": 30,
            "boot_id": "boot",
            "retired": True,
            "checkpoint_sha256": "a" * 64,
        }
        case = dict(self.case, retirement=expected)
        for state in ("T", "S", "R"):
            process = {"pid": 20, "uid": 1010, "start_ticks": 30, "state": state}
            self.assertFalse(back.retired_worker_inactive(case, "boot", lambda _, process=process: process))
        self.assertTrue(back.retired_worker_inactive(case, "boot", lambda _: None))
        self.assertTrue(
            back.retired_worker_inactive(
                case, "boot", lambda _: {"pid": 20, "uid": 1010, "start_ticks": 31, "state": "S"}
            )
        )

    def test_retirement_hash_and_explicit_retired_flag_are_required(self):
        for record in ({"retired": False}, {"retired": True, "checkpoint_sha256": "b" * 64}):
            with self.assertRaises(ValueError):
                back.validate_retirement(record, self.case)

    def test_preserve_origin_then_install_without_done(self):
        old, stage, archive = self.root / "canonical", self.root / "stage", self.root / "old"
        old.mkdir()
        stage.mkdir()
        (old / "weights").mkdir()
        (old / "weights/last.pt").write_bytes(b"old checkpoint")
        (stage / "summary.json").write_bytes(b"new")
        (stage / "DONE").write_text("SUCCEEDED\n")
        hold = self.root / "held/DONE"
        back.install_moves(old, stage, archive, hold, digest(b"old checkpoint"))
        self.assertEqual((archive / "weights/last.pt").read_bytes(), b"old checkpoint")
        self.assertTrue((old / "summary.json").exists())
        self.assertFalse((old / "DONE").exists())
        self.assertEqual(hold.read_text(), "SUCCEEDED\n")

    def test_origin_checkpoint_change_refuses_any_move(self):
        old, stage, archive = self.root / "canonical", self.root / "stage", self.root / "old"
        (old / "weights").mkdir(parents=True)
        stage.mkdir()
        (old / "weights/last.pt").write_bytes(b"changed")
        (stage / "DONE").write_text("SUCCEEDED\n")
        with self.assertRaises(RuntimeError):
            back.install_moves(old, stage, archive, self.root / "held/DONE", "a" * 64)
        self.assertTrue(old.exists())
        self.assertTrue(stage.exists())
        self.assertFalse(archive.exists())

    def test_install_recovers_after_origin_move_and_keeps_done_hidden(self):
        output, stage, archive = self.root / "canonical", self.root / "stage", self.root / "old"
        (archive / "weights").mkdir(parents=True)
        stage.mkdir()
        (archive / "weights/last.pt").write_bytes(b"old checkpoint")
        (stage / "summary.json").write_bytes(b"new")
        hold = self.root / "held/DONE"
        hold.parent.mkdir()
        hold.write_text("SUCCEEDED\n")
        back.install_moves(output, stage, archive, hold, digest(b"old checkpoint"))
        self.assertTrue(output.exists())
        self.assertFalse((output / "DONE").exists())
        self.assertTrue(hold.exists())

    def test_install_recognizes_last_publication_after_crash(self):
        output, stage, archive = self.root / "canonical", self.root / "stage", self.root / "old"
        (archive / "weights").mkdir(parents=True)
        output.mkdir()
        (archive / "weights/last.pt").write_bytes(b"old checkpoint")
        (output / "DONE").write_text("SUCCEEDED\n")
        back.install_moves(output, stage, archive, self.root / "held/DONE", digest(b"old checkpoint"))
        self.assertEqual((output / "DONE").read_text(), "SUCCEEDED\n")

    def test_completed_record_merges_fresh_list_and_sorts(self):
        job = self.root / "queue"
        job.mkdir()
        (job / "completed-methods.json").write_text(json.dumps([{"method": "master-n"}, {"method": "loffta-c2-n005"}]))
        (job / "pipeline-plan.json").write_text(json.dumps({"methods": ["master-n", "frofa-c2-d01", "loffta-c2-n005"]}))
        with patch.object(back.original, "archive_lock", contextlib.nullcontext):
            back.merge_completed(SimpleNamespace(job=job), "frofa-c2-d01", self.root / "final", {"commit": "x"})
        rows = json.loads((job / "completed-methods.json").read_text())
        self.assertEqual([row["method"] for row in rows], ["master-n", "frofa-c2-d01", "loffta-c2-n005"])

    def test_validation_uses_private_job_and_canonical_command(self):
        queue = self.root / "queue"
        queue.mkdir()
        (queue / "active-child.json").write_text("original training child")
        native = SimpleNamespace(job=queue)
        command = ["python", "validate", "--output", "/exact/canonical"]

        def execute(args, action, method):
            self.assertNotEqual(args.job, queue)
            self.assertEqual(back.original.command_for(args, action, method), command)
            self.assertEqual((queue / "active-child.json").read_text(), "original training child")

        with patch.object(back.original, "command_for", return_value=command), patch.object(
            back.original, "execute", side_effect=execute
        ):
            back.execute_validation(native, "detail25", self.root / "private-validation")

    def test_real_execute_adapter_hides_cuda_and_preserves_queue_metadata(self):
        queue = self.root / "queue"
        queue.mkdir()
        marker = queue / "active-child.json"
        marker.write_text("live campus training")
        native = SimpleNamespace(job=queue, dependencies=self.root / "deps")
        process = SimpleNamespace(pid=42, wait=lambda: 0)
        popen = Mock(return_value=process)
        with patch.object(back.original, "command_for", return_value=["python", "validate"]), patch.object(
            back.original.subprocess, "Popen", popen
        ):
            back.execute_validation(native, "detail25", self.root / "private-validation")
        self.assertEqual(popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(marker.read_text(), "live campus training")


if __name__ == "__main__":
    unittest.main()
