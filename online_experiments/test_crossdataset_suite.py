"""CPU contracts for split integrity, native coordinates, and fixed endpoints."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from online_experiments import crossdataset_suite as suite


def record(name, objects):
    return {"sample_id": name, "width": 100, "height": 80, "objects": objects}


def obj(box, difficult=False, category=0):
    return {"category_id": category, "bbox_xyxy0_half_open": box, "difficult": difficult, "iscrowd": 0}


def prediction(name, box, score=0.9, category=0):
    return {"sample_id": name, "bbox_xyxy0_half_open": box, "score": score, "category_id": category}


class NativeEvaluationContracts(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("pycocotools"), "isolated pycocotools dependency is not available here"
    )
    def test_dense_coco_ap500_does_not_reuse_hardcoded_ap100_summary(self):
        objects, detections = [], []
        for index in range(101):
            x0, y0 = (index % 20) * 3, (index // 20) * 3
            box = [x0, y0, x0 + 1, y0 + 1]
            objects.append(obj(box))
            detections.append(prediction("dense", box, 0.99 - index * 0.001))
        with tempfile.TemporaryDirectory() as folder:
            result = suite.evaluate_coco([record("dense", objects)], detections, ["object"], Path(folder))
        self.assertLess(result["by_max_detections"]["100"]["AP"], 100)
        self.assertAlmostEqual(result["by_max_detections"]["500"]["AP"], 100)

    def test_voc_one_pixel_box_and_difficult_are_not_false_positives(self):
        samples = [record("a", [obj([0, 0, 1, 1])]), record("b", [obj([10, 10, 20, 20], True)])]
        detections = [prediction("b", [10, 10, 20, 20], 0.99), prediction("a", [0, 0, 1, 1])]
        result = suite.evaluate_voc07(samples, detections, ["object"])
        self.assertAlmostEqual(result["mAP50_11point"], 100)
        self.assertEqual(result["per_class"]["object"]["ignored_detections"], 1)

    def test_voc_exact_iou_threshold_is_strict(self):
        result = suite.evaluate_voc07(
            [record("a", [obj([0, 0, 10, 10])])], [prediction("a", [0, 0, 5, 10])], ["object"]
        )
        self.assertEqual(result["mAP50_11point"], 0)

    def test_voc_duplicate_detection_is_false_positive(self):
        result = suite.evaluate_voc07(
            [record("a", [obj([0, 0, 10, 10]), obj([20, 0, 30, 10])])],
            [prediction("a", [0, 0, 10, 10], 0.9), prediction("a", [0, 0, 10, 10], 0.8)],
            ["object"],
        )
        self.assertAlmostEqual(result["mAP50_11point"], 600 / 11)

    def test_inverse_letterbox_does_not_add_category_or_change_xyxy(self):
        info = {"original_shape": [50, 100], "resized_shape": [320, 640], "pad": [0, 160]}
        value = suite.inverse_predictions(np.asarray([[64, 224, 128, 288, 0.8, 0]]), info)
        np.testing.assert_allclose(value[0, :4], [10, 10, 20, 20])
        self.assertEqual(value[0, 5], 0)

    def test_coco_preserves_empty_images_and_uses_native_pixel_area(self):
        samples = [record("a", [obj([2, 3, 12, 23])]), record("b", [])]
        ground, detections = suite.coco_payload(samples, [prediction("a", [2, 3, 12, 23])], ["object"])
        self.assertEqual(len(ground["images"]), 2)
        self.assertEqual(ground["annotations"][0]["area"], 200)
        self.assertEqual(ground["annotations"][0]["bbox"], [2, 3, 10, 20])
        self.assertEqual(detections[0]["category_id"], 1)


class InputContracts(unittest.TestCase):
    def test_partial_cache_resume_cannot_recreate_missing_prefix_as_zeros(self):
        with tempfile.TemporaryDirectory() as folder:
            missing = Path(folder) / "missing.partial.npy"
            with self.assertRaisesRegex(RuntimeError, "prefix is missing"):
                suite.open_partial_array(missing, np.float16, (4, 2), 3)
            self.assertFalse(missing.exists())

    def test_partial_cache_resume_rejects_wrong_dtype(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "partial.npy"
            np.save(target, np.ones((4, 2), dtype=np.uint8))
            with self.assertRaisesRegex(RuntimeError, "dtype/shape mismatch"):
                suite.open_partial_array(target, np.float16, (4, 2), 3)

    def test_checkpoint100_recovers_missing_summary_before_final_evaluation(self):
        rows = [{"epoch": index} for index in range(1, 101)]
        summary = {"completed_epochs": 100, "epochs": rows, "provenance_sha256": "frozen"}
        state = {"epoch": 100, "rows": rows, "args": {"provenance_sha256": "frozen"}, "summary": summary}
        with tempfile.TemporaryDirectory() as folder:
            suite.restore_epoch_evidence(Path(folder), state)
            self.assertEqual(suite.read(Path(folder) / "summary.json"), summary)

    def test_physical_teacher_count_retains_uncertain_and_replayed_calls(self):
        with tempfile.TemporaryDirectory() as folder:
            journal = Path(folder) / "queries.jsonl"
            for event in [
                {"attempt_id": "one", "event": "attempt", "images": 2},
                {"attempt_id": "one", "event": "completed", "images": 2},
                {"attempt_id": "two", "event": "attempt", "images": 2},
            ]:
                suite.append_jsonl(journal, event)
            result = suite.query_journal(journal)
            self.assertEqual(result["attempted_images"], 4)
            self.assertEqual(result["completed_images"], 2)
            self.assertEqual(result["uncertain_images"], 2)

    def test_manifest_rejects_pilot_even_if_claimed_expected(self):
        with self.assertRaisesRegex(ValueError, "official full split"):
            suite.validate_manifest(
                {
                    "schema": "detection-dataset-v1",
                    "dataset": "voc0712",
                    "nc": 20,
                    "class_names": [str(index) for index in range(20)],
                    "splits": {"train": [], "test": []},
                }
            )

    def test_letterboxed_targets_exclude_difficult(self):
        sample = record("a", [obj([0, 0, 100, 80]), obj([1, 1, 3, 3], True)])
        classes, boxes, info = suite.prepare_targets(sample)
        self.assertEqual(classes, [[0]])
        np.testing.assert_allclose(boxes, [[0.5, 0.5, 1.0, 0.8]])
        self.assertEqual(info["pad"], [0, 64])

    def test_training_cache_selection_respects_method_budget(self):
        self.assertEqual(suite.consumed_parts("online100"), ["test-v0"])
        self.assertEqual(suite.consumed_parts("cache1"), ["test-v0", "train-v0"])
        self.assertEqual(suite.consumed_parts("detail25"), ["test-v0", "train-v0", "train-v1", "blank"])

    def test_quantizer_packed_int4_roundtrip_and_constants(self):
        array = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
        codes, low, step = suite.quantize_array(array, 4)
        restored = suite.dequantize_array(codes, low, step, 4, 4)
        np.testing.assert_allclose(restored, array, atol=0.03)
        constant = np.full((1, 1, 4, 4), 3.5, dtype=np.float32)
        codes, low, step = suite.quantize_array(constant, 8)
        np.testing.assert_array_equal(suite.dequantize_array(codes, low, step, 8, 4), constant)


if __name__ == "__main__":
    unittest.main()
