"""CPU contracts for assignment transplantation; no data or checkpoints required."""

from types import SimpleNamespace

import pytest
import torch

import ultralytics.nn  # noqa: F401 -- initialize tasks before this repository's loss module
from online_experiments.assignment_intervention import (
    assignment_metrics,
    capture_assignment,
    fixed_loss,
    gradient_comparison,
    loss_and_gradient,
    state_hash,
)
from ultralytics.utils.loss import v8DetectionLoss


def fixture(reg_max=1, empty=False):
    torch.manual_seed(7)
    model = torch.nn.Linear(1, 1)
    model.model = [SimpleNamespace(stride=torch.tensor([8.0, 16.0, 32.0]), nc=2, reg_max=reg_max)]
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    criterion = v8DetectionLoss(model)
    predictions = {
        "boxes": (torch.rand(1, 4 * reg_max, 84) + 0.5).requires_grad_(),
        "scores": torch.randn(1, 2, 84).requires_grad_(),
        "feats": [torch.zeros(1, 4, n, n) for n in (8, 4, 2)],
    }
    batch = {
        "batch_idx": torch.zeros(0 if empty else 2),
        "cls": torch.empty(0, 1) if empty else torch.tensor([[0.0], [1.0]]),
        "bboxes": torch.empty(0, 4) if empty else torch.tensor([[0.4, 0.4, 0.3, 0.3], [0.7, 0.7, 0.2, 0.25]]),
    }
    return criterion, predictions, batch


def test_state_hash_tracks_tensor_buffers_and_extra_state():
    class WithExtraState(torch.nn.BatchNorm1d):
        def get_extra_state(self):
            return {"schema": 1, "nested": {"active": self.training}}

    model = WithExtraState(3).eval()
    first = state_hash(model)
    assert first == state_hash(model)
    model.running_mean[0] += 1
    assert first != state_hash(model)
    second = state_hash(model)
    model.train()
    assert second != state_hash(model)


def test_aggregation_equal_image_weights_with_missing_views():
    from online_experiments.diagnose_assignment_intervention import aggregate

    rows = [
        {"method": "a", "image_id": 1, "cosine": 1.0},
        {"method": "a", "image_id": 1, "cosine": None},
        {"method": "a", "image_id": 2, "cosine": 0.1},
        {"method": "a", "image_id": 2, "cosine": 0.3},
    ]
    result = aggregate(rows, ["method"])[0]
    assert result["images"] == 2
    assert result["observations"] == 4
    assert result["cosine_mean"] == pytest.approx(0.6)
    assert result["cosine_n"] == 2
    assert result["cosine_views_n"] == 3


@pytest.mark.parametrize("reg_max", [1, 4])
@pytest.mark.parametrize("empty", [False, True])
def test_native_loss_and_gradient_parity(reg_max, empty):
    criterion, predictions, batch = fixture(reg_max, empty)
    assignment, reference = capture_assignment(criterion, predictions, batch)
    assert all(not x.requires_grad for x in assignment)
    actual = fixed_loss(criterion, predictions, assignment, "native_soft")
    torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-6)
    native = criterion(predictions, batch)[0].sum()
    variables = [predictions["scores"], predictions["boxes"]]
    native_grad = torch.autograd.grad(native, variables, allow_unused=True)
    check_grad = torch.autograd.grad(actual.sum(), variables, allow_unused=True)
    for expected, found in zip(native_grad, check_grad):
        if expected is None:
            assert found is None
        else:
            torch.testing.assert_close(found, expected, atol=2e-6, rtol=2e-6)


def test_identity_four_cells_and_zero_gradient_handling():
    criterion, predictions, batch = fixture()
    a, _ = capture_assignment(criterion, predictions, batch)
    denom = max(int(a[3].sum()), 1)
    for protocol in ("native_soft", "hard_common"):
        outputs = [loss_and_gradient(criterion, predictions, a, protocol, denom) for _ in range(4)]
        for loss, gradients in outputs:
            torch.testing.assert_close(loss, outputs[0][0])
            for name in ("cls", "box"):
                stats = gradient_comparison(gradients[name], outputs[0][1][name])
                assert stats["relative_l2"] == 0
                assert stats["cosine"] == pytest.approx(1, abs=1e-6)
    stats = gradient_comparison(torch.zeros(3), torch.zeros(3))
    assert stats == {"cosine": None, "relative_l2": None, "norm_ratio": None}


@pytest.mark.parametrize("reg_max", [1, 4])
def test_foreign_assignment_matches_native_loss_with_fixed_assigner(reg_max):
    criterion, predictions, batch = fixture(reg_max)
    donor_predictions = {**predictions, "scores": -predictions["scores"], "boxes": predictions["boxes"] * 1.7}
    donor, _ = capture_assignment(criterion, donor_predictions, batch)
    original = criterion.assigner
    try:
        criterion.assigner = lambda *args: donor
        reference = criterion(predictions, batch)[0]
    finally:
        criterion.assigner = original
    actual = fixed_loss(criterion, predictions, donor, "native_soft")
    torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-6)


def test_hard_targets_use_labels_and_ignore_soft_quality():
    criterion, predictions, batch = fixture()
    a, _ = capture_assignment(criterion, predictions, batch)
    assert a[3].sum() > 0
    b = tuple(t.clone() for t in a)
    b[2].mul_(0.01)
    denom = max(int(a[3].sum()), 1)
    torch.testing.assert_close(
        fixed_loss(criterion, predictions, a, "hard_common", denom),
        fixed_loss(criterion, predictions, b, "hard_common", denom),
    )
    b[0][b[3].bool()] = 1 - b[0][b[3].bool()]
    assert not torch.allclose(
        fixed_loss(criterion, predictions, a, "hard_common", denom),
        fixed_loss(criterion, predictions, b, "hard_common", denom),
    )
    with pytest.raises(ValueError):
        fixed_loss(criterion, predictions, a, "hard_common", 0)


def test_assignment_switches_empty_regions_and_size_coverage():
    labels = torch.tensor([[0, 1, 0]])
    boxes = torch.tensor([[[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 80.0, 80.0], [0.0, 0.0, 0.0, 0.0]]])
    a = (labels, boxes, torch.zeros(1, 3, 2), torch.tensor([[True, True, False]]), torch.tensor([[0, 1, 0]]))
    b = tuple(t.clone() for t in a)
    b[4][0, 1] = 0
    b[1][0, 1] = boxes[0, 0]
    result = assignment_metrics(a, b, torch.tensor([100.0, 6400.0, 20000.0]))
    assert result["fg_jaccard"] == 1
    assert result["gt_switch_fraction"] == 0.5
    assert result["exact_coverage_small"] == 1
    assert result["approx_coverage_medium"] == 0
    assert result["fg_jaccard_large"] is None
    empty = tuple(t.clone() for t in a)
    empty[3].zero_()
    assert assignment_metrics(empty, empty, torch.empty(0))["fg_jaccard"] is None
