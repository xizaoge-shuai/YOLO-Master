"""Independent native-loss and enumerated Bernoulli estimator contracts."""

import math

import pytest
import torch

from online_experiments import gradient_calibration as g
from online_experiments.assignment_intervention import state_hash
from online_experiments.spatial_intervention import parameter_groups


def test_joint_oracles_combine_regions_and_preserve_unselected_tokens():
    transported = torch.arange(10.0).reshape(1, 2, 1, 5)
    exact = transported + 100
    masks = {
        "new_padding": torch.tensor([True, False, False, False, False]).reshape(1, 1, 1, 5),
        "foreground": torch.tensor([False, True, False, False, False]).reshape(1, 1, 1, 5),
        "background": torch.tensor([False, False, True, False, False]).reshape(1, 1, 1, 5),
        "mixed": torch.tensor([False, False, False, True, False]).reshape(1, 1, 1, 5),
        "letterbox": torch.tensor([False, False, False, False, True]).reshape(1, 1, 1, 5),
    }
    support = torch.tensor([0.0, 1.0, 1.0, 0.5, 1.0]).reshape(1, 1, 1, 5)
    blank = torch.full_like(transported, 10)
    old_transport, old_exact = transported.clone(), exact.clone()
    arms = g.joint_spatial_arms(transported, exact, masks, support, blank)
    joint = transported.clone()
    joint[..., :3] = exact[..., :3]
    cheap = transported.clone()
    cheap[..., 0] += 10
    cheap[..., 1:3] = exact[..., 1:3]
    cheap[..., 3] += 5
    torch.testing.assert_close(arms["joint_oracle"], joint)
    torch.testing.assert_close(arms["blank_content_oracle"], cheap)
    assert torch.equal(transported, old_transport)
    assert torch.equal(exact, old_exact)


def test_joint_oracles_accept_three_feature_levels_and_reject_truncation():
    masks = {
        "new_padding": torch.tensor([True, False, False]).reshape(1, 1, 1, 3),
        "foreground": torch.tensor([False, True, False]).reshape(1, 1, 1, 3),
        "background": torch.zeros(1, 1, 1, 3, dtype=torch.bool),
    }
    support = torch.tensor([0.0, 1.0, 0.5]).reshape(1, 1, 1, 3)
    transported = tuple(torch.full((1, channels, 1, 3), float(channels)) for channels in (2, 3, 4))
    exact = tuple(level + 100 for level in transported)
    blank = tuple(torch.full_like(level, 10) for level in transported)
    actual = g.joint_spatial_arms(transported, list(exact), masks, support, blank)
    assert all(isinstance(levels, tuple) and len(levels) == 3 for levels in actual.values())
    for index in range(3):
        expected = g.joint_spatial_arms(transported[index], exact[index], masks, support, blank[index])
        for name in expected:
            torch.testing.assert_close(actual[name][index], expected[name])
    for wrong in (exact[:2], (), exact[0]):
        with pytest.raises(ValueError, match="level"):
            g.joint_spatial_arms(transported, wrong, masks, support, blank)
    with pytest.raises(ValueError, match="level"):
        g.joint_spatial_arms(transported, exact, masks, support, blank[:2])


@pytest.mark.parametrize("p", [0.1, 0.25, 1.0])
def test_estimator_matches_enumerated_autograd_gradient_distribution(p):
    parameter = torch.tensor([0.4, -0.7, 1.2], dtype=torch.float64, requires_grad=True)
    low_loss = ((parameter * torch.tensor([2.0, 1.0, -1.0]) - 0.3) ** 2).sum()
    high_loss = ((parameter * torch.tensor([-1.0, 3.0, 0.5]) + 0.8) ** 2).sum()
    low = torch.autograd.grad(low_loss, parameter)[0]
    high = torch.autograd.grad(high_loss, parameter)[0]
    approximate = {"a": low[:2], "b": low[2:]}
    exact = {"a": high[:2], "b": high[2:]}
    actual = g.estimator_statistics(approximate, exact, p)
    # Enumerate a four-query population containing one exact query at p=.25;
    # the general two-outcome enumeration also checks p=.1 and the p=1 limit.
    outcomes = [low, low + (high - low) / p]
    probabilities = [1 - p, p]
    expectation = sum(weight * value for weight, value in zip(probabilities, outcomes))
    variance = sum(weight * (value - expectation).square().sum() for weight, value in zip(probabilities, outcomes))
    query_outcomes = [torch.zeros_like(high), high / p]
    query_variance = sum(weight * (value - high).square().sum() for weight, value in zip(probabilities, query_outcomes))
    row = actual["__all__"]
    assert row["parameters"] == 3
    assert row["variance_trace"] == pytest.approx(float(variance), abs=1e-12)
    assert row["query_only_variance_trace"] == pytest.approx(float(query_variance), abs=1e-12)
    assert row["residual_squared_norm"] == pytest.approx(float((high - low).square().sum()))
    assert row["exact_squared_norm"] == pytest.approx(float(high.square().sum()))
    assert row["expectation_relative_l2"] < 1e-14
    for name, value in zip(("unselected", "selected"), outcomes):
        assert row[f"{name}_cosine"] == pytest.approx(float(torch.nn.functional.cosine_similarity(value, high, dim=0)))
        assert row[f"{name}_norm_ratio"] == pytest.approx(float(value.norm() / high.norm()))
        assert row[f"{name}_relative_l2"] == pytest.approx(float((value - high).norm() / high.norm()))
    if p == 1:
        assert row["variance_ratio_vs_query_only"] is None
    else:
        assert row["variance_ratio_vs_query_only"] == pytest.approx(float(variance / query_variance))
    assert row["variance_trace"] == pytest.approx(sum(actual[key]["variance_trace"] for key in approximate))


def test_estimator_zero_reference_and_anticorrelated_control_are_explicit():
    bad = g.estimator_statistics({"x": torch.tensor([-1.0])}, {"x": torch.tensor([1.0])})["__all__"]
    assert bad["variance_ratio_vs_query_only"] == 4
    assert bad["unselected_cosine"] == -1
    assert bad["selected_norm_ratio"] == 7
    zero = g.estimator_statistics({"x": torch.zeros(3)}, {"x": torch.zeros(3)})["__all__"]
    assert zero["variance_trace"] == 0
    assert zero["variance_ratio_vs_query_only"] is None
    assert zero["expectation_relative_l2"] is None
    for name in ("unselected", "selected"):
        assert zero[f"{name}_cosine"] is None
        assert zero[f"{name}_norm_ratio"] is None
        assert zero[f"{name}_relative_l2"] is None


@pytest.mark.parametrize("p", [0, -0.1, 1.1, math.nan, math.inf, True])
def test_estimator_rejects_invalid_query_probability(p):
    with pytest.raises(ValueError, match="probability"):
        g.estimator_statistics({"x": torch.ones(3)}, {"x": torch.ones(3)}, p)


@pytest.mark.parametrize(
    "approximate,exact",
    [
        ({"x": torch.ones(3)}, {"y": torch.ones(3)}),
        ({"x": torch.ones(3)}, {"x": torch.ones(2)}),
        ({"x": torch.ones(1, 3)}, {"x": torch.ones(1, 3)}),
        ({"x": torch.tensor([math.nan])}, {"x": torch.ones(1)}),
        ({}, {}),
        ({"__all__": torch.ones(3)}, {"__all__": torch.ones(3)}),
    ],
)
def test_estimator_rejects_invalid_gradient_blocks(approximate, exact):
    with pytest.raises(ValueError):
        g.estimator_statistics(approximate, exact)


def test_native_gradient_matches_direct_actual_detection_loss_and_detachment():
    from scripts.d1_train_cached_detector import CachedLatentDetector
    from ultralytics.utils.loss import E2EDetectLoss

    torch.manual_seed(19)
    model = CachedLatentDetector(in_channels=(4, 4, 4), nc=2, imgsz=64, epochs=1).eval()
    criterion = E2EDetectLoss(model)
    groups = parameter_groups(model)
    levels = tuple(torch.randn(2, 4, 4, 4) for _ in range(3))
    targets = {
        "batch_idx": torch.tensor([0.0, 1.0]),
        "cls": torch.tensor([[0.0], [1.0]]),
        "bboxes": torch.tensor([[0.4, 0.4, 0.3, 0.3], [0.7, 0.7, 0.2, 0.25]]),
    }
    pairs = [pair for block in groups.values() for pair in block]
    parameters = [parameter for _, parameter in pairs]
    # Existing accumulated gradients must not be cleared or replaced by diagnostics.
    parameters[0].grad = torch.full_like(parameters[0], 7)
    saved_grad = parameters[0].grad.clone()
    before = state_hash(model)
    predictions = model.training_predictions(levels)
    direct_loss = criterion(predictions, targets)[0].sum()
    expected = torch.autograd.grad(direct_loss, parameters, allow_unused=True, retain_graph=True)
    branch_loss = criterion.one2many(predictions["one2many"], targets)[0].sum()
    branch_expected = torch.autograd.grad(branch_loss, parameters, allow_unused=True)
    actual, loss_value = g.native_parameter_gradients(model, criterion, groups, levels, targets)
    assert loss_value == pytest.approx(float(direct_loss.detach()))
    index = 0
    for block, members in groups.items():
        reference = []
        for _, parameter in members:
            gradient = expected[index]
            reference.append(torch.zeros_like(parameter).flatten() if gradient is None else gradient.flatten())
            if block in ("p3", "p4", "p5"):
                one2many = branch_expected[index]
                if gradient is None:
                    assert one2many is None
                else:
                    torch.testing.assert_close(gradient, one2many)
            index += 1
        torch.testing.assert_close(actual[block], torch.cat(reference))
        assert not actual[block].requires_grad
    assert before == state_hash(model)
    assert torch.equal(parameters[0].grad, saved_grad)
    assert all(parameter.grad is None for parameter in parameters[1:])
    assert actual["one2one_box"].norm() > 0
    assert actual["one2one_cls"].norm() > 0
    model.train()
    with pytest.raises(ValueError, match="eval"):
        g.native_parameter_gradients(model, criterion, groups, levels, targets)
