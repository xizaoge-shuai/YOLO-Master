"""CPU-only contracts for real full native detector baselines; no GPU trial."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from online_experiments import direct_baseline_suite as suite


class Contracts(unittest.TestCase):
    def test_exact_original_box_conversion_and_ignored_targets(self):
        sample = {
            "width": 100,
            "height": 50,
            "objects": [
                {"category_id": 2, "bbox_xyxy0_half_open": [10, 5, 30, 25]},
                {"category_id": 3, "bbox_xyxy0_half_open": [0, 0, 50, 50], "difficult": True},
                {"category_id": 4, "bbox_xyxy0_half_open": [0, 0, 50, 50], "iscrowd": 1},
            ],
        }
        labels = suite.normalized_objects(sample)
        np.testing.assert_allclose(labels, [[2, 0.2, 0.3, 0.2, 0.4]])
        self.assertEqual(labels.dtype, np.float32)
        self.assertEqual(len(sample["objects"]), 3)

    def test_empty_targets_stay_two_dimensional(self):
        self.assertEqual(suite.normalized_objects({"width": 10, "height": 10, "objects": []}).shape, (0, 5))

    def test_exact_model_names_and_recipe(self):
        self.assertEqual(set(suite.MODELS), {"master-n", "yolo12-n"})
        self.assertEqual(suite.MODELS["master-n"], "ultralytics/cfg/models/master/v0/det/yolo-master-n.yaml")
        config = suite.recipe("master-n", Path("dataset"), Path("run"), "0")
        for name, expected in {
            "epochs": 100,
            "seed": 0,
            "batch": 8,
            "imgsz": 640,
            "optimizer": "AdamW",
            "pretrained": False,
            "amp": False,
            "val": False,
            "fraction": 1.0,
        }.items():
            self.assertEqual(config[name], expected)
        self.assertEqual(config["scale"], 0.15)
        self.assertEqual(config["fliplr"], 0.5)

    def test_repeat_or_gap_epochs_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "results.csv"
            for epochs in ([1, 2, 2], [1, 3], [2]):
                with target.open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["epoch"])
                    writer.writerows([[number] for number in epochs])
                with self.assertRaises(RuntimeError):
                    suite.epoch_rows(target)
            target.write_text("epoch\n1\n2\n3\n")
            self.assertEqual(len(suite.epoch_rows(target)), 3)

    def test_heldout_voc_native_endpoint_keeps_difficult(self):
        samples = [
            {
                "sample_id": "one",
                "objects": [
                    {
                        "category_id": 0,
                        "bbox_xyxy0_half_open": [0, 0, 10, 10],
                        "bbox_voc_1based_inclusive": [1, 1, 10, 10],
                    },
                    {
                        "category_id": 0,
                        "bbox_xyxy0_half_open": [20, 20, 30, 30],
                        "bbox_voc_1based_inclusive": [21, 21, 30, 30],
                        "difficult": True,
                    },
                ],
            }
        ]
        predictions = [
            {"sample_id": "one", "category_id": 0, "bbox_xyxy0_half_open": [20, 20, 30, 30], "score": 0.99},
            {"sample_id": "one", "category_id": 0, "bbox_xyxy0_half_open": [0, 0, 10, 10], "score": 0.9},
        ]
        result = suite.scoring.evaluate_voc07(samples, predictions, ["object"])
        self.assertAlmostEqual(result["mAP50_11point"], 100)
        self.assertEqual(result["per_class"]["object"]["ignored_detections"], 1)

    def test_prepare_seal_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = root / "manifest.json"
            manifest.write_text("original")
            asset = root / "train.txt"
            asset.write_text("image")
            suite.dump(
                root / "complete.json",
                {
                    "version": suite.VERSION,
                    "manifest_sha256": suite.sha(manifest),
                    "sha256": {"train.txt": suite.sha(asset)},
                },
            )
            suite.check_prepared(manifest, root)
            asset.write_text("changed")
            with self.assertRaises(RuntimeError):
                suite.check_prepared(manifest, root)


if __name__ == "__main__":
    unittest.main()
