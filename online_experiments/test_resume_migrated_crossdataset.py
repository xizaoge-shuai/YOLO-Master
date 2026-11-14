"""CPU tests for narrow, content-verified migration guards."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    from online_experiments import resume_migrated_crossdataset as migration
except ImportError:
    migration = None


class MigrationGuards(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(migration, "Migration entry point is missing")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "mirror"
        self.source.mkdir()
        self.file = self.source / "data.bin"
        self.file.write_bytes(b"verified immutable bytes")
        self.record = {
            "path": str(self.file),
            "target": str(self.file),
            "bytes": self.file.stat().st_size,
            "sha256": hashlib.sha256(self.file.read_bytes()).hexdigest(),
            "origin_stat": {"bytes": self.file.stat().st_size, "mtime_ns": 100, "ctime_ns": 200},
        }

    def verified(self):
        return migration.verify_file(self.record, self.source, self.root)

    def test_verified_bytes_get_origin_stamp_without_changing_target(self):
        record = self.verified()
        mapper = migration.verified_stamp({str(self.file): record})
        self.assertEqual(mapper(self.file), self.record["origin_stat"])
        self.assertEqual(self.file.read_bytes(), b"verified immutable bytes")

    def test_unknown_stamp_request_fails_closed(self):
        mapper = migration.verified_stamp({str(self.file): self.verified()})
        with self.assertRaisesRegex(RuntimeError, "not sealed"):
            mapper(self.root / "unverified.bin")

    def test_changed_target_is_rejected_even_if_origin_metadata_matches(self):
        mapper = migration.verified_stamp({str(self.file): self.verified()})
        self.file.write_bytes(b"corrupt")
        with self.assertRaisesRegex(RuntimeError, "changed"):
            mapper(self.file)

    def test_wrong_transfer_hash_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "SHA256"):
            migration.verify_file(dict(self.record, sha256="0" * 64), self.source, self.root)

    def test_wrong_declared_byte_count_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "bytes"):
            migration.verify_file(dict(self.record, bytes=1), self.source, self.root)

    def test_source_path_outside_explicit_root_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "source root"):
            migration.verify_file(self.record, self.root / "other", self.root)

    def test_source_path_with_parent_traversal_is_rejected(self):
        escaped = self.source / ".." / "mirror" / self.file.name
        with self.assertRaisesRegex(RuntimeError, "source root"):
            migration.verify_file(dict(self.record, path=str(escaped)), self.source, self.root)

    def test_target_outside_explicit_root_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "target root"):
            migration.verify_file(self.record, self.source, self.root / "other")

    def test_target_field_cannot_name_another_copy(self):
        other = self.source / "other.bin"
        other.write_bytes(self.file.read_bytes())
        with self.assertRaisesRegex(RuntimeError, "target"):
            migration.verify_file(dict(self.record, target=str(other)), self.source, self.root)

    def test_runtime_outputs_are_not_immutable_stamp_whitelist(self):
        output = self.source / "run"
        self.assertTrue(migration.is_runtime_input(output / "weights/last.pt", output))
        self.assertFalse(migration.is_runtime_input(self.file, output))

    def test_environment_archive_preserves_bytes_and_allows_real_new_identity(self):
        output = self.source / "run"
        output.mkdir()
        original = b'{"identity":{"gpu":"origin GPU"}}\n'
        (output / "environment.json").write_bytes(original)
        migration.prepare_environment(output, hashlib.sha256(original).hexdigest())
        self.assertEqual((output / "environment-origin.json").read_bytes(), original)
        self.assertFalse((output / "environment.json").exists())
        (output / "environment.json").write_text('{"identity":{"gpu":"new GPU"}}')
        migration.prepare_environment(output, hashlib.sha256(original).hexdigest())
        self.assertIn("new GPU", (output / "environment.json").read_text())

    def test_environment_archive_refuses_conflicting_origin(self):
        output = self.source / "run"
        output.mkdir()
        (output / "environment-origin.json").write_text("wrong")
        with self.assertRaisesRegex(RuntimeError, "environment"):
            migration.prepare_environment(output, "0" * 64)

    def test_environment_archive_recovers_interruption_between_copy_and_unlink(self):
        output = self.source / "run"
        output.mkdir()
        original = b'{"identity":{"gpu":"origin GPU"}}\n'
        (output / "environment.json").write_bytes(original)
        (output / "environment-origin.json").write_bytes(original)
        migration.prepare_environment(output, hashlib.sha256(original).hexdigest())
        self.assertFalse((output / "environment.json").exists())
        self.assertEqual((output / "environment-origin.json").read_bytes(), original)

    def test_checkpoint_origin_hash_only_required_before_first_resume(self):
        checkpoint = self.source / "last.pt"
        checkpoint.write_bytes(b"new epoch")
        state = {"epoch": 18, "args": {"method": "detail25"}, "rows": [{}] * 18}
        origin = {"epoch": 17, "checkpoint_sha256": "0" * 64}
        with patch.object(migration, "read_checkpoint", return_value=state):
            migration.check_checkpoint(checkpoint, origin, state["args"], started=True)
            with self.assertRaisesRegex(RuntimeError, "origin checkpoint"):
                migration.check_checkpoint(checkpoint, origin, state["args"], started=False)

    def test_origin_checkpoint_copy_remains_when_last_checkpoint_advances(self):
        output = self.source / "run"
        (output / "weights").mkdir(parents=True)
        last = output / "weights/last.pt"
        last.write_bytes(b"origin epoch17")
        digest = migration.sha(last)
        record = migration.preserve_checkpoint(output, digest)
        last.write_bytes(b"new epoch18")
        self.assertEqual(Path(record["path"]).read_bytes(), b"origin epoch17")
        migration.verify_current(record)
        self.assertEqual(migration.preserve_checkpoint(output, digest), record)

    def test_checkpoint_config_epoch_and_history_cannot_change(self):
        checkpoint = self.source / "last.pt"
        checkpoint.write_bytes(b"checkpoint")
        origin = {"epoch": 9, "checkpoint_sha256": migration.sha(checkpoint)}
        for state in (
            {"epoch": 8, "args": {}, "rows": [{}] * 8},
            {"epoch": 101, "args": {}, "rows": [{}] * 101},
            {"epoch": 9, "args": {"changed": True}, "rows": [{}] * 9},
            {"epoch": 9, "args": {}, "rows": []},
        ):
            with patch.object(migration, "read_checkpoint", return_value=state), self.assertRaises(RuntimeError):
                migration.check_checkpoint(checkpoint, origin, {}, started=True)

    def test_migration_metadata_is_added_without_rewriting_endpoint_hashes(self):
        output = self.source / "run"
        output.mkdir()
        endpoint = {"sha256": {"weights/last.pt": "original"}, "fixed_epoch": 100}
        (output / "endpoint.json").write_text(json.dumps(endpoint))
        migration.annotate_endpoint(output, {"origin": {"epoch": 17}, "schema": migration.SCHEMA}, "seal-hash")
        result = json.loads((output / "endpoint.json").read_text())
        self.assertEqual(result["sha256"], endpoint["sha256"])
        self.assertEqual(result["migration"]["origin"]["epoch"], 17)

    def test_seal_checks_coverage_and_excludes_mutable_runtime_outputs(self):
        repo = self.source / "repo"
        job = repo / "runs/paper/crossdataset/job"
        output = repo / "runs/paper/priority/job/voc/train/loffta-c2-n005-s0"
        (job / "cache").mkdir(parents=True)
        (output / "weights").mkdir(parents=True)
        (repo / "frozen.py").write_text("# frozen\n")
        manifest, identity = self.source / "manifest.json", self.source / "identity.json"
        manifest.write_text("{}")
        identity.write_text("{}")
        migration.write(
            job / "input-integrity.json",
            {str(self.file): {"sha256": migration.sha(self.file), **migration.actual_stat(self.file)}},
        )
        cache_file = job / "cache/features.npy"
        cache_file.write_bytes(b"test shard")
        migration.write(
            job / "cache/manifest.json",
            {
                "parts": {
                    "test-v0": {
                        "file": str(cache_file),
                        "sha256": migration.sha(cache_file),
                        "stat": migration.actual_stat(cache_file),
                    }
                }
            },
        )
        migration.write(job / "cache/complete.json", {})
        (job / "cache/teacher-queries.jsonl").write_text("")
        migration.write(
            job / "provenance.json",
            {
                "manifest": str(manifest),
                "manifest_sha256": migration.sha(manifest),
                "identity_file": str(identity),
                "identity_file_sha256": migration.sha(identity),
                "source_sha256": {"frozen.py": migration.sha(repo / "frozen.py")},
            },
        )
        config = {"method": "loffta-c2-n005", "dataset": "voc0712", "seed": 0, "epochs": 100, "batch": 8}
        migration.write(output / "args.json", config)
        migration.write(output / "environment.json", {"identity": {"gpu": "origin"}})
        migration.write(output / "initialization.json", {})
        (output / "weights/last.pt").write_bytes(b"origin checkpoint")
        origin = {
            "job": str(job),
            "output": str(output),
            "method": "loffta-c2-n005",
            "seed": 0,
            "epoch": 9,
            "checkpoint_sha256": migration.sha(output / "weights/last.pt"),
        }
        files = [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": migration.sha(path),
                "origin_stat": migration.actual_stat(path),
            }
            for path in self.source.rglob("*")
            if path.is_file()
        ]
        transfer = self.root / "transfer.json"
        migration.write(transfer, {"schema": migration.TRANSFER_SCHEMA, "origin": origin, "files": files})
        args = SimpleNamespace(
            transfer_list=transfer, source_root=self.source, target_root=self.root, seal=self.root / "seal.json"
        )
        state = {
            "args": config,
            "epoch": 9,
            "rows": [{}] * 9,
            **dict.fromkeys(
                (
                    "model",
                    "optimizer",
                    "scheduler",
                    "python_rng",
                    "numpy_rng",
                    "cpu_rng",
                    "cuda_rng",
                    "totals",
                    "summary",
                    "query_images",
                    "query_calls",
                )
            ),
        }
        with patch.object(migration, "ROOT", repo.resolve()), patch.object(
            migration, "read_checkpoint", return_value=state
        ):
            migration.make_seal(args)
            seal = migration.load_seal(args.seal)
        self.assertNotIn(str(output / "weights/last.pt"), seal["immutable"])
        self.assertIn(str(cache_file), seal["immutable"])
        self.assertEqual(migration.sha(output / "weights/origin-migration.pt"), origin["checkpoint_sha256"])


if __name__ == "__main__":
    unittest.main()
