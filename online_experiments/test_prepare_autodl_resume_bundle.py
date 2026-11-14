"""CPU contracts for content-bound, regular-member migration archives."""

import hashlib
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

try:
    from online_experiments import prepare_autodl_resume_bundle as bundle
except ImportError:
    bundle = None


class BundleSafety(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(bundle, "Migration bundle generator is missing")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "campus"
        self.source.mkdir()
        self.target = self.root / "autodl"

    def test_archive_dereferences_only_safe_in_root_files(self):
        source = self.source / "data.txt"
        source.write_text("verified source\n")
        alias = self.source / "link.txt"
        try:
            alias.symlink_to(source)
        except OSError:
            self.skipTest("Creating symlinks requires OS privilege")
        files = {}
        bundle.add_file(files, alias, self.source, self.target)
        archive = self.root / "safe.tar"
        bundle.write_archive(archive, files, self.source)
        with tarfile.open(archive) as stream:
            members = stream.getmembers()
            self.assertEqual([member.name for member in members], ["link.txt"])
            self.assertTrue(members[0].isfile())
            self.assertEqual(stream.extractfile(members[0]).read(), source.read_bytes())
        self.assertEqual(files[str(alias)]["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())

    def test_original_hash_is_not_replaced_after_corruption(self):
        source = self.source / "source.py"
        source.write_text("changed content")
        files = {}
        expected = hashlib.sha256(b"original content").hexdigest()
        bundle.add_file(files, source, self.source, self.target, expected_sha=expected)
        with self.assertRaisesRegex(RuntimeError, "SHA256"):
            bundle.write_archive(self.root / "corrupt.tar", files, self.source)
        self.assertEqual(files[str(source)]["sha256"], expected)

    def test_original_stat_change_rejected_before_archiving(self):
        source = self.source / "input.jpg"
        source.write_bytes(b"actual")
        expected = dict(bundle.stamp(source), bytes=1)
        with self.assertRaisesRegex(RuntimeError, "stat"):
            bundle.add_file({}, source, self.source, self.target, expected_stat=expected)

    def test_escaping_symlink_and_parent_path_are_rejected(self):
        outside = self.root / "outside.txt"
        outside.write_text("outside")
        with self.assertRaisesRegex(RuntimeError, "root"):
            bundle.add_file({}, outside, self.source, self.target)
        with self.assertRaisesRegex(RuntimeError, "canonical"):
            bundle.add_file({}, self.source / ".." / "outside.txt", self.source, self.target)

    def test_duplicate_conflicting_source_hash_is_rejected(self):
        source = self.source / "same.txt"
        source.write_text("same")
        files = {}
        bundle.add_file(files, source, self.source, self.target, expected_sha="a" * 64)
        with self.assertRaisesRegex(RuntimeError, "Conflicting"):
            bundle.add_file(files, source, self.source, self.target, expected_sha="b" * 64)

    def test_archive_refuses_existing_destination(self):
        destination = self.root / "existing.tar"
        destination.write_bytes(b"preserve")
        with self.assertRaises(FileExistsError):
            bundle.write_archive(destination, {}, self.source)
        self.assertEqual(destination.read_bytes(), b"preserve")

    def test_capacity_counts_both_archive_and_unpacked_members(self):
        rows = [{"bytes": 100}, {"bytes": 200}]
        result = bundle.capacity(rows, archive_member_bytes=150, target_free_bytes=1000, reserve_bytes=50)
        self.assertEqual(result["required_target_bytes"], 500)
        with self.assertRaisesRegex(RuntimeError, "space"):
            bundle.capacity(rows, archive_member_bytes=150, target_free_bytes=499, reserve_bytes=50)

    def test_collect_covers_frozen_inputs_anchor_cache_and_resume_state(self):
        from types import SimpleNamespace

        def put(relative, payload):
            path = self.source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload) if isinstance(payload, dict) else payload)
            return path

        repo = self.source / "repo"
        source = put("repo/frozen.py", "frozen source")
        anchor = put("repo/anchor.json", {"anchor": True})
        for name in ("prepare_autodl_resume_bundle", "transfer_autodl_resume", "resume_migrated_crossdataset"):
            put(f"repo/online_experiments/{name}.py", "migration source")
        image, label = put("data/image.jpg", "image"), put("data/label.txt", "label")
        manifest = put("data/dataset-manifest.json", {"evaluation_file": "evaluation/test.json"})
        evaluation = put("data/evaluation/test.json", {"images": []})
        put(
            "data/complete.json", {"manifest_sha256": bundle.sha(manifest), "evaluation_sha256": bundle.sha(evaluation)}
        )
        identity = put("identity.json", {"model_id": "unused", "revision": "unused"})
        inputs = put(
            "job/input-integrity.json",
            {str(path): {"sha256": bundle.sha(path), **bundle.stamp(path)} for path in (image, label)},
        )
        provenance = put(
            "job/provenance.json",
            {
                "dataset": "voc0712",
                "input_integrity_sha256": bundle.sha(inputs),
                "manifest": str(manifest),
                "manifest_sha256": bundle.sha(manifest),
                "identity_file": str(identity),
                "identity_file_sha256": bundle.sha(identity),
                "source_sha256": {"frozen.py": bundle.sha(source)},
                "source_anchor": {"file": str(anchor), "sha256": bundle.sha(anchor)},
            },
        )
        query = put("job/cache/teacher-queries.jsonl", "")
        parts = {}
        for name in ("train-v0", "train-v1", "test-v0", "blank"):
            path = put(f"job/cache/{name}.npy", "fake test shard")
            parts[name] = {"file": str(path), "sha256": bundle.sha(path), "stat": bundle.stamp(path)}
        cache = put(
            "job/cache/manifest.json",
            {
                "parts": parts,
                "provenance_sha256": bundle.sha(provenance),
                "teacher_query_journal_sha256": bundle.sha(query),
            },
        )
        put("job/cache/complete.json", {"manifest_sha256": bundle.sha(cache)})
        output = self.source / "priority/train/loffta-c2-n005-s0"
        for name in ("environment.json", "initialization.json", "summary.json", "weights/last.pt"):
            put(f"priority/train/loffta-c2-n005-s0/{name}", {"method": "loffta-c2-n005"})
        put(
            "priority/train/loffta-c2-n005-s0/args.json",
            {
                "method": "loffta-c2-n005",
                "dataset": "voc0712",
                "seed": 0,
                "epochs": 100,
                "batch": 8,
                "provenance_sha256": bundle.sha(provenance),
                "cache_manifest_sha256": bundle.sha(cache),
            },
        )
        args = SimpleNamespace(source_root=self.source, target_root=self.target, repo=repo, extra_source_files=[])
        files = bundle.collect(args, "voc0712", self.source / "job", output)
        self.assertTrue(
            {str(source), str(anchor), str(image), str(label), str(evaluation), str(output / "weights/last.pt")}
            <= files.keys()
        )
        self.assertEqual(sum(not row["archive"] for row in files.values()), 4)
        archive = self.root / "fixture.tar"
        bundle.write_archive(archive, files, self.source)
        with tarfile.open(archive) as stream:
            self.assertTrue(all(member.isfile() for member in stream.getmembers()))
            self.assertNotIn("job/cache/train-v0.npy", stream.getnames())


if __name__ == "__main__":
    unittest.main()
