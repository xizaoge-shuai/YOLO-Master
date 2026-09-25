"""Contract tests for fixed-budget cached feature correction."""

import ast
import random
from pathlib import Path

import pytest
import torch

from online_experiments.budget_corrector import (
    ConditionalCorrector,
    correction_loss,
    patch_pilot,
    query_batches,
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
