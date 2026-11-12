"""CPU contracts: one saved-prediction bundle, exact inputs, no new GPU work."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from online_experiments import metric_bundle_campaign as worker


class Contracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.priority = self.root / "priority"
        self.old = self.root / "old"
        self.plan = {"dataset_job": str(self.old)}

    def test_exact_seven_methods_and_baselines_precede_ours(self):
        self.assertEqual(len(worker.METHODS), 7)
        self.assertEqual(worker.METHODS[:2], ("master-n", "yolo12-n"))
        self.assertEqual(worker.METHODS[-1], "detail25")
        self.assertNotIn("blank25", worker.METHODS)
        plans = {data: {"plan": {"methods": list(worker.METHODS)}} for data in worker.DATASETS}
        order = list(worker.case_order(plans))
        self.assertEqual(len(order), 14)
        self.assertEqual(order[-2:], [("voc0712", "detail25"), ("sku110k", "detail25")])

    def test_native_done_json_and_cached_done_are_distinct(self):
        run = self.root / "run"
        run.mkdir()
        self.assertFalse(worker.ready(run, "master-n"))
        (run / "DONE").write_text("SUCCEEDED\n")
        self.assertFalse(worker.ready(run, "master-n"))
        self.assertTrue(worker.ready(run, "detail25"))
        (run / "DONE.json").write_text("{}")
        self.assertTrue(worker.ready(run, "master-n"))

    def test_adopt_old_completed_cache_only_not_partial_or_native(self):
        old = self.old / "train/detail25-s0"
        old.mkdir(parents=True)
        new = (self.priority / "voc0712/train/detail25-s0").resolve()
        self.assertEqual(worker.resolve_output(self.priority, "voc0712", self.plan, "detail25"), new)
        (old / "DONE").write_text("SUCCEEDED")
        self.assertEqual(worker.resolve_output(self.priority, "voc0712", self.plan, "detail25"), old.resolve())
        native_old = self.old / "train/master-n-s0"
        native_old.mkdir()
        (native_old / "DONE.json").write_text("{}")
        self.assertEqual(
            worker.resolve_output(self.priority, "voc0712", self.plan, "master-n"),
            (self.priority / "voc0712/train/master-n-s0").resolve(),
        )

    def test_coordinator_resolved_output_is_authoritative_and_contained(self):
        new = (self.priority / "voc0712/train/detail25-s0").resolve()
        worker.write(self.priority / "voc0712/completed-methods.json", [{"method": "detail25", "output": str(new)}])
        self.assertEqual(worker.resolve_output(self.priority, "voc0712", self.plan, "detail25"), new)
        worker.write(
            self.priority / "voc0712/completed-methods.json",
            [{"method": "detail25", "output": str(self.root / "unrelated")}],
        )
        with self.assertRaisesRegex(ValueError, "escapes"):
            worker.resolve_output(self.priority, "voc0712", self.plan, "detail25")

    def test_validators_only_have_validate_action_and_cpu_environment(self):
        for dataset in worker.DATASETS:
            for method in worker.METHODS:
                command = worker.validation_command(dataset, self.plan, method, self.root / "run")
                self.assertIn("validate", command)
                self.assertNotIn("train", command)
                self.assertNotIn("--device", command)
                self.assertNotIn("--gpu", command)
                if method not in worker.DIRECT:
                    self.assertEqual(command[command.index("--seed") + 1], "0")
        env = worker.cpu_environment(self.root / "deps")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            self.assertEqual(env[name], "1")

    def test_bundle_integrity_accepts_seal_and_rejects_tamper(self):
        bundle = self.root / "bundle"
        bundle.mkdir()
        (bundle / "status.txt").write_text("SUCCEEDED\n")
        for name in ("metrics.json", "provenance.json", "resource-metrics.json"):
            worker.write(bundle / name, {})
        names = [path.name for path in bundle.iterdir()]
        worker.write(bundle / "completion.json", {"output_sha256": {name: worker.sha(bundle / name) for name in names}})
        worker.verify_bundle(bundle)
        worker.write(bundle / "metrics.json", {"tampered": True})
        with self.assertRaisesRegex(ValueError, "Changed"):
            worker.verify_bundle(bundle)

    def test_bundle_integrity_rejects_escape(self):
        bundle = self.root / "bundle"
        bundle.mkdir()
        (bundle / "status.txt").write_text("SUCCEEDED")
        for name in ("metrics.json", "provenance.json", "resource-metrics.json"):
            worker.write(bundle / name, {})
        worker.write(self.root / "outside.json", {})
        hashes = {
            name: worker.sha(bundle / name) for name in ("metrics.json", "provenance.json", "resource-metrics.json")
        }
        hashes["../outside.json"] = worker.sha(self.root / "outside.json")
        worker.write(bundle / "completion.json", {"output_sha256": hashes})
        with self.assertRaisesRegex(ValueError, "escaping"):
            worker.verify_bundle(bundle)

    def test_native_identity_rejects_wrong_method_even_allowed_architecture(self):
        run = self.root / "run"
        worker.write(
            run / "args.json",
            {"method": "yolo12-n", "seed": 0, "epochs": 100, "dataset": "voc0712", "manifest_sha256": "correct"},
        )
        with self.assertRaisesRegex(ValueError, "method"):
            worker.validate_identity("voc0712", {"manifest_sha256": "correct"}, "master-n", run)

    def test_cached_metric_command_binds_original_job_provenance_only(self):
        plan = {"dataset_job": str(self.old), "manifest": str(self.root / "manifest.json")}
        for method in worker.METHODS:
            command = worker.metric_command(plan, method, self.root / "relocated-run", self.root / "bundle")
            self.assertEqual(command[command.index("--run") + 1], str(self.root / "relocated-run"))
            if method in worker.DIRECT:
                self.assertNotIn("--provenance", command)
            else:
                self.assertEqual(command[command.index("--provenance") + 1], str(self.old / "provenance.json"))

    def test_existing_bundle_is_not_recomputed_or_revalidated(self):
        case = self.root / "case"
        source = self.root / "run"
        source.mkdir()
        worker.write(source / "endpoint.json", {})
        worker.write(source / "DONE.json", {})
        plan = {"manifest": str(self.root / "manifest"), "manifest_sha256": "manifest-digest"}
        from types import SimpleNamespace

        args = SimpleNamespace(job=self.root / "job", priority=self.priority, dependencies=self.root / "deps")
        case = args.job / "voc0712/master-n"
        (case / "bundle").mkdir(parents=True)
        worker.write(case / "bundle/completion.json", {})
        with patch.object(worker, "sha", return_value="digest"), patch.object(
            worker, "execute"
        ) as execute, patch.object(worker, "validate_identity"), patch.object(worker, "verify_bundle"), patch.object(
            worker, "archive_case", return_value={"commit": "test"}
        ):
            worker.process_case(args, "voc0712", "master-n", plan, source)
            self.assertEqual(execute.call_count, 1)
            execute.reset_mock()
            worker.process_case(args, "voc0712", "master-n", plan, source)
            execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
