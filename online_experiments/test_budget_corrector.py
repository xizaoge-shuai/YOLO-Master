"""Contract tests for fixed-budget cached feature correction."""

import ast
import random
from pathlib import Path

import pytest
import torch

from online_experiments.budget_corrector import (
    ConditionalCorrector,
    correction_loss,
    foreground_correction_loss,
    identity_gate,
    patch_pilot,
    query_batches,
    severity_query_batches,
)


def test_budget_counts_and_pairing():
    for pct in (10, 25):
        plans = [query_batches(50, pct, epoch, 7) for epoch in range(100)]
        assert sum(map(len, plans)) * 8 == 40000 * pct // 100
        assert plans == [query_batches(50, pct, e, 7) for e in range(100)]
        assert plans != [query_batches(50, pct, e, 8) for e in range(100)]
        assert all(len(x) == len(set(x)) and all(0 <= i < 50 for i in x) for x in plans)
    with pytest.raises(ValueError):
        query_batches(50, 0, 0, 7)


def test_corrector_starts_at_transport_and_learns_without_teacher_gradients():
    torch.manual_seed(91)
    state = torch.random.get_rng_state().clone()
    model = ConditionalCorrector(channels=8, rank=4, seed=0)
    assert torch.equal(state, torch.random.get_rng_state())
    source = tuple(torch.randn(2, 8, 5, 5) for _ in range(3))
    target = tuple(x + 0.4 for x in source)
    scale, flip = [0.9, 1.1], [False, True]
    pred = model(source, scale, flip)
    assert all(torch.equal(x, y) for x, y in zip(source, pred))
    before = correction_loss(pred, target).item()
    opt = torch.optim.AdamW(model.parameters(), lr=0.02)
    for _ in range(20):
        opt.zero_grad()
        loss = correction_loss(model(source, scale, flip), target)
        loss.backward()
        opt.step()
    assert correction_loss(model(source, scale, flip), target).item() < before
    assert all(x.grad is None for x in source + target)
    assert all(torch.equal(x, y) for x, y in zip(source, pred))


def test_patch_keeps_validation_and_adds_corrector_checkpoint():
    source = Path("online_experiments/d1_online_compare.py").read_text()
    tree = patch_pilot(source)
    code = ast.unparse(tree)
    assert "yield d.collate_cached([val_bank[i] for i in ids])" in code
    assert "'corrector': correction_checkpoint()" in code
    assert "cached_bank(train, device)" in code
    compile(tree, "<budget-pilot>", "exec")


def test_global_rng_unaffected_by_query_schedule():
    random.seed(9)
    state = random.getstate()
    query_batches(50, 25, 5, 0)
    assert state == random.getstate()



def test_identity_gate_preserves_both_c2_anchors():
    model = ConditionalCorrector(channels=8, rank=4, seed=3, use_identity_gate=True)

    # Force a nonzero residual so equality genuinely comes from the gate.
    with torch.no_grad():
        model.net[-1].bias.fill_(1.0)

    source = tuple(torch.randn(2, 8, 5, 5) for _ in range(3))
    output = model(source, [1.0, 1.0], [False, True])

    assert all(torch.equal(x, y) for x, y in zip(source, output))

    g = identity_gate([1.0, 0.85, 1.15], device=torch.device("cpu"), dtype=torch.float32)
    assert g[0].item() == 0.0
    assert torch.isclose(g[1], torch.tensor(1.0), atol=1e-6)
    assert torch.isclose(g[2], torch.tensor(1.0), atol=1e-6)


def test_identity_gated_corrector_has_gradient_away_from_anchor():
    model = ConditionalCorrector(channels=8, rank=4, seed=5, use_identity_gate=True)
    source = tuple(torch.randn(2, 8, 5, 5) for _ in range(3))
    target = tuple(x + 0.3 for x in source)

    loss = correction_loss(model(source, [0.9, 1.1], [False, True]), target)
    loss.backward()

    assert any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in model.parameters()
    )


def test_foreground_loss_empty_region_falls_back_to_global():
    source = tuple(torch.randn(2, 8, 5, 5) for _ in range(3))
    target = tuple(x + 0.2 for x in source)
    empty = [torch.empty((0, 4)), torch.empty((0, 4))]

    combined, global_loss, foreground_loss = foreground_correction_loss(source, target, empty)

    expected = correction_loss(source, target)
    assert torch.allclose(global_loss, expected, atol=1e-7, rtol=1e-6)
    assert torch.allclose(foreground_loss, expected, atol=1e-7, rtol=1e-6)
    assert torch.allclose(combined, expected, atol=1e-7, rtol=1e-6)


def test_foreground_loss_uses_fractional_box_weights():
    prediction = (torch.zeros(1, 4, 4, 4),) * 3
    target = tuple(torch.ones(1, 4, 4, 4) for _ in range(3))
    boxes = [torch.tensor([[0.5, 0.5, 0.10, 0.10]], dtype=torch.float32)]

    combined, global_loss, foreground_loss = foreground_correction_loss(prediction, target, boxes)

    assert torch.isfinite(combined)
    assert torch.isfinite(global_loss)
    assert torch.isfinite(foreground_loss)
    assert combined.requires_grad is False



def test_scale_severity_query_policy_preserves_exact_budget_and_rng():
    rng = random.Random(123456)
    before = rng.getstate()

    selected = severity_query_batches(
        n_batches=50,
        pct=25,
        epoch=7,
        seed=0,
        aug_rng=rng,
        batch_size=8,
        scale_min=0.85,
        scale_max=1.15,
    )

    # Policy inspection must not perturb the real augmentation RNG.
    assert rng.getstate() == before

    # Same exact query budget as the historical random Mix25.
    random_selected = query_batches(50, 25, 7, 0)
    assert len(selected) == len(random_selected)

    # Reconstruct the batch severity ranking independently.
    probe = random.Random()
    probe.setstate(before)
    scores = {}
    for batch_index in range(50):
        values = []
        for _ in range(8):
            probe.random()
            scale = probe.uniform(0.85, 1.15)
            values.append(abs(__import__("math").log(scale)))
        scores[batch_index] = sum(values) / len(values)

    unselected = set(scores) - selected
    assert min(scores[i] for i in selected) >= max(scores[i] for i in unselected)
