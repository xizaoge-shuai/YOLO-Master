"""CPU contracts for legacy coverage, historical audit and immutable bundles."""

import csv
import tempfile
import unittest
from pathlib import Path

from online_experiments import complete_visdrone_core_metrics as core


class CoreEvidenceContracts(unittest.TestCase):
    def test_completion_detects_changed_sources_and_outputs(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            source = folder / "source.txt"
            target = folder / "metrics.txt"
            source.write_text("original")
            target.write_text("measured")
            core.dump(folder / "completion.json", {
                "input_sha256": {str(source): core.sha(source)},
                "output_sha256": {target.name: core.sha(target)},
            })
            core.verify_completion(folder)
            target.write_text("edited")
            with self.assertRaises(ValueError):
                core.verify_completion(folder)

    def test_legacy_checkpoint_and_prediction_coverage_are_checked(self):
        with tempfile.TemporaryDirectory() as name:
            job = Path(name)
            evaluation = job / "evaluation/master-n-last.json"
            evaluation.parent.mkdir()
            checkpoint = job / "train/master-n/weights/last.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"not deserialized")
            core.dump(evaluation, {"checkpoint_sha256": core.sha(checkpoint)})
            predictions = job / "predictions/master-n/last"
            predictions.mkdir(parents=True)
            (predictions / "one.txt").write_text("1,1,2,2,0.8,1,-1,-1\n")
            records = [{"id": "one", "shape": [10, 10], "gt": core.np.empty((0, 8))}]
            _, cases, _, bindings = core.native_cases(evaluation, "master-n", records)
            self.assertEqual(len(cases), 1)
            self.assertEqual(len(bindings), 2)
            (predictions / "extra.txt").write_text("")
            with self.assertRaises(ValueError):
                core.native_cases(evaluation, "master-n", records)

    def test_legacy_duplicate_epochs_remain_visible_without_validation_zeros(self):
        with tempfile.TemporaryDirectory() as name:
            job = Path(name)
            run = job / "train/master-n"
            run.mkdir(parents=True)
            with (run / "results.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, ["epoch", "train/box_loss", "metrics/mAP50(B)"])
                writer.writeheader()
                writer.writerows([{"epoch": 1, "train/box_loss": 2, "metrics/mAP50(B)": 0}] * 2)
            curves, resources, _ = core.curves_and_resources(run, "master-n", job)
            self.assertEqual(resources["legacy_history_audit"]["duplicate_epoch_rows"], 1)
            self.assertEqual(len(curves), 2)
            self.assertNotIn("metrics/mAP50(B)", curves[0])
            self.assertIsNone(resources["parameters"])

    def test_existing_unsealed_case_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "master-n").mkdir()
            args = type("Args", (), {"original": root, "output": root})()
            row = {"evaluation_source": "user-data:/YOLO-Master/evaluation.json", "method": "YOLO-Master-N"}
            with self.assertRaises(FileExistsError):
                core.score_case(args, row, [], {})


if __name__ == "__main__":
    unittest.main()
