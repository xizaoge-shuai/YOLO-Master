"""CPU contracts for one-prediction evidence export; no training/inference."""

import csv
import tempfile
import unittest
from pathlib import Path

from online_experiments import complete_metric_bundle as bundle


def sample(objects):
    return {"sample_id": "one", "width": 2000, "height": 2000, "objects": objects}


def obj(box, **kwargs):
    return {"category_id": 0, "bbox_xyxy0_half_open": box, **kwargs}


def pred(box, score=0.9):
    return {"sample_id": "one", "category_id": 0, "bbox_xyxy0_half_open": box, "score": score}


class MetricContracts(unittest.TestCase):
    def metrics(self, samples, predictions):
        evaluator, ground = bundle.coco_evidence(samples, predictions, ["object"])
        return evaluator, ground, bundle.extract(evaluator, ["object"])[0]

    def test_perfect_box_and_primary_voc(self):
        from online_experiments.crossdataset_suite import evaluate_voc07

        samples, predictions = [sample([obj([10, 10, 40, 40])])], [pred([10, 10, 40, 40])]
        evaluator, _, values = self.metrics(samples, predictions)
        self.assertAlmostEqual(values["100"]["AP"], 100)
        self.assertEqual(evaluator.eval["precision"].shape, (10, 101, 1, 4, 4))
        native = evaluate_voc07(samples, predictions, ["object"])
        self.assertAlmostEqual(native["mAP50_11point"], 100)
        bundle.check_native("voc0712", native, samples, predictions, ["object"], values)
        bad = {**native, "mAP50_11point": 99}
        with self.assertRaises(RuntimeError):
            bundle.check_native("voc0712", bad, samples, predictions, ["object"], values)

    def test_difficult_and_crowd_ignored(self):
        samples = [sample([obj([10, 10, 40, 40]), obj([100, 100, 150, 150], difficult=True),
                           obj([200, 200, 250, 250], iscrowd=1)])]
        predictions = [pred([100, 100, 150, 150], 0.99), pred([200, 200, 250, 250], 0.98), pred([10, 10, 40, 40])]
        _, ground, values = self.metrics(samples, predictions)
        self.assertAlmostEqual(values["500"]["AP"], 100)
        self.assertEqual([item["iscrowd"] for item in ground["annotations"]], [0, 1, 1])

    def test_empty_detections_zero_ap_undefined_area(self):
        _, _, values = self.metrics([sample([obj([10, 10, 40, 40])])], [])
        self.assertEqual(values["100"]["AP"], 0)
        self.assertIsNone(values["100"]["AP_large"])

    def test_dense_recall_more_than_100_detections(self):
        boxes = [[10 + (i % 20) * 60, 10 + (i // 20) * 60, 30 + (i % 20) * 60, 30 + (i // 20) * 60]
                 for i in range(150)]
        _, _, values = self.metrics([sample([obj(box) for box in boxes])], [pred(box, 1 - i / 1000) for i, box in enumerate(boxes)])
        self.assertAlmostEqual(values["100"]["AR"], 100 * 100 / 150)
        self.assertAlmostEqual(values["500"]["AR"], 100)
        native = {"by_max_detections": {str(n): values[str(n)] for n in (100, 500)}}
        bundle.check_native("sku110k", native, [], [], ["object"], values)

    def test_curve_placeholders_excluded_and_actual_nc_log(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (root / "results.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["epoch", "time", "train/box_loss", "lr/pg0", "metrics/mAP50(B)", "val/box_loss"])
                writer.writeheader()
                writer.writerow({"epoch": 1, "time": 9, "train/box_loss": 2, "lr/pg0": .001, "metrics/mAP50(B)": 0, "val/box_loss": 0})
            log = root / "train.log"
            log.write_text("YOLO12n summary: 272 layers, 2,568,243 parameters, 2,568,227 gradients, 6.5 GFLOPs\n"
                           "YOLO12n summary (fused): 200 layers, 2,560,000 parameters, 0 gradients, 6.4 GFLOPs\n")
            resources, curves = bundle.resource_evidence(root, {"version": "direct-full-detection-v1"}, log)
            self.assertEqual(resources["parameters"], 2568243)
            self.assertEqual(resources["trainable_parameters"], 2568227)
            self.assertEqual(set(curves[0]), {"epoch", "time", "train/box_loss", "lr/pg0"})

    def test_cached_epoch_curves_and_composite_counts_not_log(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle.dump(root / "initialization.json", {"parameters": 12000, "trainable_parameters": 12000, "detail_parameters": 4672})
            bundle.dump(root / "args.json", {})
            log = root / "train.log"
            log.write_text("Internal YOLO summary: 100 layers, 9,999 parameters, 9,999 gradients\n")
            resources, curves = bundle.resource_evidence(root, {"version": "crossdataset-detection-v1",
                "epochs": [{"epoch": 1, "train_loss": 4, "train_seconds": 3, "lr": .001}]}, log)
            self.assertEqual(resources["parameters"], 12000)
            self.assertEqual(resources["trainable_parameters"], 12000)
            self.assertEqual(curves[0]["train_loss"], 4)


if __name__ == "__main__":
    unittest.main()
