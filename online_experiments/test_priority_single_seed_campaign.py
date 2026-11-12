"""CPU contracts for the urgent seven-method single-seed campaign."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from online_experiments import priority_single_seed_campaign as worker


class Contracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.args = SimpleNamespace(job=base / "priority/voc0712", dataset_job=base / "old", dataset="voc0712")
        self.args.dataset_job.mkdir()
        worker.write(self.args.dataset_job / "provenance.json", {"manifest": str(base / "full-manifest.json")})

    def test_only_seven_cases_direct_then_papers_then_ours(self):
        self.assertEqual(
            worker.METHODS,
            (
                "master-n",
                "yolo12-n",
                "frofa-c2-d01",
                "loffta-c2-n005",
                "cache-reuse-c1-int8",
                "vit-adapter-cache25",
                "detail25",
            ),
        )
        self.assertEqual(len(set(worker.METHODS)), 7)
        self.assertFalse(
            any(method in worker.METHODS for method in ("mix25", "cache1", "cache2", "blank25", "lowpass25"))
        )

    def test_cached_commands_reuse_sealed_job_but_separate_output(self):
        command = worker.command_for(self.args, "train", "frofa-c2-d01")
        self.assertEqual(Path(command[2]).name, "crossdataset_suite.py")
        self.assertEqual(command[command.index("--job") + 1], str(self.args.dataset_job))
        self.assertEqual(command[command.index("--output") + 1], str(worker.output_for(self.args, "frofa-c2-d01")))
        self.assertEqual(command[command.index("--seed") + 1], "0")

    def test_sku_uses_unchanged_tolerant_protocol_wrapper(self):
        self.args.dataset = "sku110k"
        command = worker.command_for(self.args, "train", "detail25")
        self.assertEqual(Path(command[2]).name, "sku_official_loader_suite.py")

    def test_direct_prepare_and_train_bind_full_manifest(self):
        command = worker.command_for(self.args, "prepare", "master-n")
        self.assertIn("--dataset-manifest", command)
        self.assertEqual(command[-1], str(self.args.job.parent / "datasets/voc0712"))
        command = worker.command_for(self.args, "train", "yolo12-n")
        self.assertEqual(command[command.index("--method") + 1], "yolo12-n")
        self.assertEqual(command[command.index("--device") + 1], "0")
        self.assertNotIn("--epochs", command)

    def test_direct_validation_has_no_training_or_gpu_argument(self):
        command = worker.command_for(self.args, "validate", "master-n")
        self.assertEqual(command[3:], ["validate", "--output", str(worker.output_for(self.args, "master-n"))])

    def test_incomplete_old_output_is_not_adopted(self):
        old = self.args.dataset_job / "train/frofa-c2-d01-s0"
        old.mkdir(parents=True)
        worker.write(old / "summary.json", {"epochs": 100, "completed_epochs": 5})
        self.assertEqual(worker.output_for(self.args, "frofa-c2-d01"), self.args.job / "train/frofa-c2-d01-s0")
        (old / "DONE").write_text("SUCCEEDED\n")
        self.assertEqual(worker.output_for(self.args, "frofa-c2-d01"), old)


if __name__ == "__main__":
    unittest.main()
