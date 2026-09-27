"""Reference-protocol tests independent of detector training."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PORT = Path(__file__).with_name("visdrone_reference_eval.py")


def module():
    assert PORT.is_file(), "VisDrone reference evaluator is not implemented"
    spec = importlib.util.spec_from_file_location("visdrone_reference_eval", PORT)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def rows(values):
    return np.asarray(values, dtype=np.float64).reshape(-1, 8)


class Availability(unittest.TestCase):
    def test_evaluator_exists(self):
        self.assertTrue(PORT.is_file(), "Missing official-protocol evaluator")


class Protocol(unittest.TestCase):
    def test_native_colon_threshold_at_exact_point_85(self):
        ev = module()
        gt = np.array([[30, 20, 20, 20, 0]], dtype=float)
        dt = np.array([[30, 20, 17, 20, 0.8]], dtype=float)
        self.assertEqual(ev.THRESHOLDS[7], np.nextafter(0.85, np.inf))
        self.assertEqual(ev.match_thresholds(gt, dt)[7, 0], 0)

    def test_duplicate_or_missing_annotation_ids_rejected(self):
        sys.path.insert(0, str(PORT.parent))
        from run_visdrone_reference_eval import validate_annotation_identity

        samples = [{"image_path": f"{i}.jpg"} for i in range(548)]
        annotations = {str(i): str(i) for i in range(548)}
        hashes = {f"{i}.txt": "hash" for i in range(548)}
        self.assertEqual(len(validate_annotation_identity(samples, annotations, hashes)), 548)
        duplicate = samples[:-1] + samples[:1]
        with self.assertRaises(ValueError):
            validate_annotation_identity(duplicate, annotations, hashes)
        with self.assertRaises(ValueError):
            validate_annotation_identity(samples, annotations, {})

    def test_last_equal_overlap_gt_wins_and_normal_precedes_ignore(self):
        ev = module()
        gt = np.array([[0, 0, 10, 10, 0], [5, 0, 10, 10, 0]], dtype=float)
        dt = np.array([[2.5, 0, 10, 10, 0.9], [0, 0, 10, 10, 0.8]], dtype=float)
        np.testing.assert_array_equal(ev.match_thresholds(gt, dt)[0], [1, 1])
        gt = np.array([[0, 0, 10, 10, 0], [-1, -1, 20, 20, 1]], dtype=float)
        dt = np.array([[0, 0, 10, 10, 0.9], [0, 0, 10, 10, 0.8], [0, 0, 10, 10, 0.7]], dtype=float)
        np.testing.assert_array_equal(ev.match_thresholds(gt, dt)[0], [1, -1, -1])

    def test_loader_preserves_invalid_predictions_for_reference_scoring(self):
        sys.path.insert(0, str(PORT.parent))
        from run_visdrone_reference_eval import load_cases

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            path.joinpath("sample.txt").write_text("10,10,-1,5,0.9,1,-1,-1\n")
            cases, hashes = load_cases(path, [("sample", rows([[10, 10, 5, 5, 1, 1, 0, 0]]), (100, 100))])
            self.assertEqual(cases[0][1][0, 2], -1)
            self.assertIn("sample", hashes)

    def test_perfect_and_high_score_false_positive(self):
        gt = rows([[10, 10, 10, 10, 1, 1, 0, 0]])
        tp = [10, 10, 10, 10, 0.8, 1, -1, -1]
        fp = [60, 60, 10, 10, 0.9, 1, -1, -1]
        ev = module()
        self.assertAlmostEqual(ev.evaluate_raw([(gt, rows([tp]), (100, 100))])["AP"], 100)
        self.assertAlmostEqual(ev.evaluate_raw([(gt, rows([fp, tp]), (100, 100))])["AP"], 50)

    def test_empty_detections_and_duplicate(self):
        gt = rows([[10, 10, 10, 10, 1, 1, 0, 0]])
        ev = module()
        self.assertEqual(ev.evaluate_raw([(gt, rows([]), (100, 100))])["AP"], 0)
        det = rows([[10, 10, 10, 10, 0.9, 1, -1, -1], [10, 10, 10, 10, 0.8, 1, -1, -1]])
        self.assertEqual(ev.evaluate_raw([(gt, det, (100, 100))])["AP"], 100)

    def test_reference_class_occurrence_weighting(self):
        # Official calcAccuracy repeats class IDs once for each image containing the class.
        a = rows([[10, 10, 10, 10, 1, 1, 0, 0]])
        b = rows([[10, 10, 10, 10, 1, 1, 0, 0], [50, 50, 10, 10, 1, 2, 0, 0]])
        det = rows([[50, 50, 10, 10, 0.9, 2, -1, -1]])
        result = module().evaluate_raw([(a, rows([]), (100, 100)), (b, det, (100, 100))])
        self.assertAlmostEqual(result["AP"], 100 / 3)
        self.assertAlmostEqual(result["macro_AP_diagnostic"], 50)

    def test_global_max_detections_before_class_filter(self):
        gt = rows([[10, 10, 10, 10, 1, 1, 0, 0]])
        det = rows([[60, 60, 10, 10, 0.9, 2, -1, -1], [10, 10, 10, 10, 0.8, 1, -1, -1]])
        result = module().evaluate_raw([(gt, det, (100, 100))])
        self.assertEqual(result["AR1"], 0)
        self.assertEqual(result["AR10"], 100)

    def test_ignored_region_union_and_score_flag(self):
        ev = module()
        gt = rows(
            [
                [1, 1, 20, 40, 0, 0, 0, 0],
                [21, 1, 20, 40, 0, 0, 0, 0],
                [10, 10, 20, 20, 1, 1, 0, 0],
                [60, 60, 10, 10, 1, 1, 0, 0],
            ]
        )
        det = rows([[10, 10, 20, 20, 0.9, 1, -1, -1], [60, 60, 10, 10, 0.8, 1, -1, -1]])
        prepared_gt, prepared_det = ev.prepare_arrays(gt, det, (100, 100))
        self.assertEqual(len(prepared_gt), 1)
        self.assertEqual(len(prepared_det), 1)
        self.assertEqual(prepared_gt[0, 4], 0)
        self.assertEqual(ev.evaluate_raw([(gt, det, (100, 100))])["AP"], 100)
        # The reference denominator includes an ignored per-object GT entry.
        gt2 = rows([[10, 10, 10, 10, 1, 1, 0, 0], [60, 60, 10, 10, 0, 1, 0, 0]])
        det2 = rows([[60, 60, 10, 10, 0.9, 1, -1, -1], [10, 10, 10, 10, 0.8, 1, -1, -1]])
        self.assertEqual(ev.evaluate_raw([(gt2, det2, (100, 100))])["AP"], 50)


if __name__ == "__main__":
    unittest.main()
