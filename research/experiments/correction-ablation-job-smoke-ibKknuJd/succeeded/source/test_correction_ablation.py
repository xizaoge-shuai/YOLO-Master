"""Contracts for the fixed full-data correction ablation."""

import importlib.util
import unittest
from pathlib import Path

import torch


class AblationTests(unittest.TestCase):
    def module(self):
        path = Path(__file__).with_name("correction_ablation.py")
        self.assertTrue(path.exists(), "Full-data ablation adapter is missing")
        spec = importlib.util.spec_from_file_location("correction_ablation", path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def test_fixed_factorial_variants(self):
        m = self.module()
        self.assertEqual(
            m.VARIANTS,
            {"rec25": (False, False), "gate25": (True, False), "fg25": (False, True), "gatefg25": (True, True)},
        )

    def test_visible_tiny_mask_survives(self):
        m = self.module()
        boxes = torch.tensor([[0.6, 0.4, 0.001, 0.001]])
        changed = m.mask_geometry(boxes, True, 0.75)
        torch.testing.assert_close(changed, torch.tensor([[0.425, 0.425, 0.00075, 0.00075]]))

    def test_foreground_changes_spatial_gradient(self):
        m = self.module()
        p = torch.full((1, 1, 2, 2), 2.0, requires_grad=True)
        loss, _, _ = m.foreground_correction_loss((p,), (torch.ones_like(p),), [torch.tensor([[0.25, 0.25, 0.5, 0.5]])])
        self.assertAlmostEqual(float(loss.detach()), 1.0)
        loss.backward()
        self.assertGreater(float(p.grad[0, 0, 0, 0]), float(p.grad[0, 0, 1, 1]))

    def test_gate_identity_and_parameter_count(self):
        m = self.module()
        net = m.ConditionalCorrector(384, 64, 0, use_identity_gate=True)
        self.assertEqual(sum(p.numel() for p in net.parameters()), 50560)
        with torch.no_grad():
            net.net[-1].bias.fill_(2.0)
        x = torch.randn(2, 384, 2, 2)
        y = net((x,), [1.0, 1.0], [False, True])[0]
        self.assertTrue(torch.equal(x, y))


if __name__ == "__main__":
    unittest.main()
