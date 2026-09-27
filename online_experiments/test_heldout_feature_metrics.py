"""Geometric foreground metrics must retain tiny boxes and handle empty regions."""

import importlib.util
import unittest
from pathlib import Path

import torch

PATH = Path(__file__).with_name("diagnose_full_corrector.py")


class FeatureMetrics(unittest.TestCase):
    def load(self):
        self.assertTrue(PATH.exists(), "Held-out diagnostics not implemented")
        spec = importlib.util.spec_from_file_location("diagnose_full_corrector", PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_tiny_box_has_nonzero_fractional_weight(self):
        m = self.load()
        boxes = torch.tensor([[0.501, 0.501, 0.002, 0.002]])
        weight = m.box_cell_weights(boxes, 40, 40)
        self.assertGreater(float(weight.sum()), 0)
        self.assertAlmostEqual(float(weight.sum()), 0.002**2 * 40**2, places=5)

    def test_full_image_and_empty_region(self):
        m = self.load()
        weights = m.box_cell_weights(torch.tensor([[0.5, 0.5, 1.0, 1.0]]), 4, 4)
        torch.testing.assert_close(weights, torch.ones(4, 4))
        target = torch.ones(2, 4, 4)
        self.assertAlmostEqual(m.region_relative_mse(target * 2, target, weights), 1)
        self.assertEqual(m.region_relative_mse(target, target, weights), 0)
        self.assertIsNone(m.region_relative_mse(target, target, torch.zeros_like(weights)))

    def test_visible_subpixel_box_survives_mask_geometry(self):
        m = self.load()
        self.assertTrue(
            hasattr(m, "mask_geometry"), "Mask geometry must retain visible boxes independently of loss filtering"
        )
        tiny = torch.tensor([[0.6, 0.4, 0.001, 0.001]])
        result = m.mask_geometry(tiny, True, 0.75)
        torch.testing.assert_close(result, torch.tensor([[0.425, 0.425, 0.00075, 0.00075]]))
        self.assertGreater(float(m.box_cell_weights(result, 40, 40).sum()), 0)

    def test_mask_clips_visible_sliver_and_drops_only_invisible_box(self):
        m = self.load()
        self.assertTrue(hasattr(m, "mask_geometry"), "Mask geometry is missing")
        boxes = torch.tensor([[0.996, 0.5, 0.04, 0.1], [0.99, 0.5, 0.001, 0.1]])
        # Scaling the first box leaves <10% visible. The second is fully outside.
        result = m.mask_geometry(boxes, False, 1.05)
        self.assertEqual(tuple(result.shape), (1, 4))
        self.assertGreater(float(result[0, 2]), 0)
        self.assertLess(float(result[0, 2]), 0.04 * 1.05 * 0.1)
        self.assertGreater(float(m.box_cell_weights(result, 40, 40).sum()), 0)


if __name__ == "__main__":
    unittest.main()
