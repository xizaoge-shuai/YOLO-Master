"""CPU safety checks for the destination migration coordinator."""

import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

try:
    from online_experiments import run_autodl_migrated_pair as runner
except ImportError:
    runner = None


class DestinationSafety(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(runner, "AutoDL coordinator missing")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.target = self.root / "target"
        self.target.mkdir()

    def completed_output(self):
        output = self.target / "output"
        output.mkdir()
        (output / "weights").mkdir()
        (output / "weights/origin-migration.pt").write_bytes(b"origin checkpoint")
        origin = {
            "output": str(output),
            "job": "/origin/job",
            "method": "detail25",
            "epoch": 17,
            "seed": 0,
            "checkpoint_sha256": runner.sha(output / "weights/origin-migration.pt"),
        }
        names = (
            "args.json",
            "environment.json",
            "initialization.json",
            "summary.json",
            "metrics.json",
            "predictions.json",
            "weights/last.pt",
            "teacher-queries.jsonl",
        )
        for name in names:
            (output / name).write_text(json.dumps({"completed_epochs": 100}))
        for name in ("migration-origin.json", "environment-origin.json", "migration-resume.json"):
            (output / name).write_text("{}")
        (output / "schedules").mkdir()
        for number in range(1, 101):
            (output / "schedules" / f"{number:03d}.json").write_text(json.dumps({"epoch": number}))
        (output / "endpoint.json").write_text(
            json.dumps(
                {
                    "fixed_epoch": 100,
                    "sha256": {name: runner.sha(output / name) for name in names},
                    "schedule_sha256": {path.name: runner.sha(path) for path in (output / "schedules").glob("*.json")},
                    "migration": {"origin": origin},
                }
            )
        )
        (output / "DONE").write_text("SUCCEEDED\n")
        return output, origin

    def archive(self, name="repo/code.py", kind=None, data=b"verified code"):
        path = self.root / "input.tar"
        with tarfile.open(path, "w") as stream:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            if kind:
                member.type, member.linkname, member.size = kind, "outside", 0
            stream.addfile(member, None if kind else io.BytesIO(data))
        record = {
            "target": str(self.target / name),
            "target_relative": name,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        return path, [record]

    def test_verified_regular_archive_reuses_identical_destination(self):
        path, rows = self.archive()
        runner.extract_verified(path, self.target, rows)
        destination = self.target / "repo/code.py"
        first = destination.stat()
        runner.extract_verified(path, self.target, rows)
        self.assertEqual(destination.read_bytes(), b"verified code")
        self.assertEqual(destination.stat().st_mtime_ns, first.st_mtime_ns)

    def test_conflicting_destination_is_preserved(self):
        path, rows = self.archive()
        destination = self.target / "repo/code.py"
        destination.parent.mkdir()
        destination.write_bytes(b"existing differing source")
        with self.assertRaisesRegex(RuntimeError, "conflict"):
            runner.extract_verified(path, self.target, rows)
        self.assertEqual(destination.read_bytes(), b"existing differing source")

    def test_tar_links_and_traversal_are_rejected(self):
        path, rows = self.archive(kind=tarfile.SYMTYPE)
        with self.assertRaisesRegex(RuntimeError, "regular"):
            runner.extract_verified(path, self.target, rows)
        path, rows = self.archive("../outside")
        with self.assertRaisesRegex(RuntimeError, "relative"):
            runner.extract_verified(path, self.target, rows)

    def test_unlisted_and_hash_corrupted_members_are_rejected(self):
        path, rows = self.archive()
        with self.assertRaisesRegex(RuntimeError, "coverage"):
            runner.extract_verified(path, self.target, [])
        rows[0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "SHA256"):
            runner.extract_verified(path, self.target, rows)
        self.assertFalse((self.target / "repo/code.py").exists())

    def test_dataset_marker_is_bound_to_exact_global_manifest(self):
        rows = [{"dataset": "sku110k", "target": "/target/file", "bytes": 4, "sha256": "a" * 64}]
        marker = runner.expected_marker("sku110k", rows)
        self.assertEqual(
            marker["dataset_manifest_sha256"],
            runner.canonical_hash([{key: rows[0][key] for key in ("target", "bytes", "sha256")}]),
        )
        changed = [{**rows[0], "sha256": "b" * 64}]
        self.assertNotEqual(marker, runner.expected_marker("sku110k", changed))

    def test_waiter_rejects_an_unexpected_global_manifest_hash(self):
        from types import SimpleNamespace

        marker = self.root / "markers"
        marker.mkdir()
        (marker / "manifest.json").write_text(json.dumps({"schema": "autodl-resume-transfer-v1", "files": []}))
        (marker / "sku110k.DONE.json").write_text("{}")
        case = self.root / "case"
        case.mkdir()
        args = SimpleNamespace(marker_dir=marker, expected_manifest_sha256="0" * 64, poll_seconds=1)
        with self.assertRaisesRegex(RuntimeError, "SHA256"):
            runner.wait_transfer(args, "sku110k", case)

    def test_existing_destination_parent_symlink_is_rejected(self):
        outside = self.root / "outside"
        outside.mkdir()
        try:
            (self.target / "repo").symlink_to(outside, target_is_directory=True)
        except OSError:
            self.skipTest("Creating symlinks requires OS privilege")
        path, rows = self.archive()
        with self.assertRaisesRegex(RuntimeError, "symlink"):
            runner.extract_verified(path, self.target, rows)

    def test_return_manifest_requires_completed_endpoint(self):
        output = self.target / "output"
        output.mkdir()
        origin = {"output": str(output), "job": "/origin/job", "method": "detail25", "epoch": 17}
        with self.assertRaisesRegex(RuntimeError, "complete"):
            runner.result_manifest("sku110k", origin, self.target)
        output.rmdir()
        output, origin = self.completed_output()
        (output / "pipeline.lock").write_text("lock")
        (output / "partial.tmp").write_text("partial")
        result = runner.result_manifest("sku110k", origin, self.target)
        self.assertEqual(result["schema"], "autodl-resume-return-v1")
        names = {row["relative"] for row in result["files"]}
        self.assertTrue(
            {"DONE", "endpoint.json", "metrics.json", "summary.json", "weights/origin-migration.pt"} <= names
        )
        self.assertNotIn("pipeline.lock", names)
        self.assertNotIn("partial.tmp", names)

    def test_return_manifest_rejects_changed_endpoint_file(self):
        output, origin = self.completed_output()
        (output / "args.json").write_text("changed configuration")
        with self.assertRaisesRegex(RuntimeError, "endpoint SHA256"):
            runner.result_manifest("sku110k", origin, self.target)

    def test_return_manifest_rejects_missing_schedule(self):
        output, origin = self.completed_output()
        (output / "schedules/001.json").unlink()
        with self.assertRaisesRegex(RuntimeError, "schedule"):
            runner.result_manifest("sku110k", origin, self.target)

    def test_full_resume_log_is_preserved_inside_return_output(self):
        output, _ = self.completed_output()
        case = self.root / "case"
        case.mkdir()
        (case / "resume.log").write_bytes(b"verification\nresume epoch18\nfinal full-test done\n")
        runner.preserve_resume_log(case, output)
        self.assertEqual((output / "migration-resume.log").read_bytes(), (case / "resume.log").read_bytes())


if __name__ == "__main__":
    unittest.main()
