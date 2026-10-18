"""Contracts for spatial support and actual parameter-gradient diagnostics."""

import pytest
import torch

from online_experiments import spatial_intervention as s


def test_regions_partition_identity_and_letterbox():
    info = {"resized_shape": [8, 16], "pad": [0, 4]}
    boxes = torch.tensor([[0.25, 0.5, 0.25, 0.5]])
    masks, support = s.spatial_regions(info, boxes, False, 1.0, imgsz=16, grid_size=4)
    assert set(masks) == {"new_padding", "letterbox", "foreground", "background", "mixed"}
    assert torch.equal(sum(m.int() for m in masks.values()), torch.ones(1, 1, 4, 4, dtype=torch.int))
    assert masks["new_padding"].sum() == 0
    assert masks["letterbox"].sum() == 8
    assert masks["foreground"].sum() == 4
    assert masks["background"].sum() == 4
    assert masks["mixed"].sum() == 0
    assert torch.equal(support, torch.ones_like(support))


def test_regions_flip_clipping_and_mixed_boundaries():
    info = {"resized_shape": [16, 16], "pad": [0, 0]}
    boxes = torch.tensor([[0.0, 0.5, 0.4, 0.4]])
    a, _ = s.spatial_regions(info, boxes, False, 1.15, imgsz=16, grid_size=4)
    b, _ = s.spatial_regions(info, boxes, True, 1.15, imgsz=16, grid_size=4)
    assert a["foreground"].sum() > 0
    assert torch.equal(a["foreground"].flip(-1), b["foreground"])
    for scale in (0.85, 0.5):
        masks, support = s.spatial_regions(info, boxes, False, scale, imgsz=16, grid_size=4)
        assert torch.equal(sum(m.int() for m in masks.values()), torch.ones(1, 1, 4, 4, dtype=torch.int))
        assert torch.all((support >= 0) & (support <= 1))
        if scale == 0.85:
            assert masks["mixed"].sum() > 0
        else:
            assert masks["new_padding"].sum() > 0


def test_oracle_is_local_and_cheap_padding_preserves_constant_and_identity():
    transport = torch.arange(16.0).reshape(1, 1, 4, 4)
    exact = transport + 10
    mask = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    mask[..., 0, :] = True
    out = s.restore_region(transport, exact, mask)
    assert torch.equal(out[mask], exact[mask])
    assert torch.equal(out[~mask], transport[~mask])
    support = torch.tensor([0.0, 0.2, 0.6, 1.0]).reshape(1, 1, 1, 4)
    blank = torch.full((1, 3, 4, 4), 7.0)
    torch.testing.assert_close(s.blank_fill(blank * support, blank, support), blank)
    assert torch.equal(s.blank_fill(transport, exact, torch.ones_like(transport)), transport)


def test_gradient_blocks_match_direct_autograd_and_keep_unused_zero():
    shared = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    branch = torch.nn.Parameter(torch.tensor([4.0, 5.0]))
    unused = torch.nn.Parameter(torch.tensor([6.0]))
    loss = (shared.square() * branch).sum()
    groups = {"shared": [("shared", shared)], "branch": [("branch", branch)], "unused": [("unused", unused)]}
    actual = s.parameter_gradients(loss, groups, retain_graph=True)
    expected = torch.autograd.grad(loss, (shared, branch, unused), allow_unused=True)
    torch.testing.assert_close(actual["shared"], expected[0])
    torch.testing.assert_close(actual["branch"], expected[1])
    assert torch.equal(actual["unused"], torch.zeros(1))
    assert all(p.grad is None for p in (shared, branch, unused))
    detached = (shared.detach() * branch).sum()
    found = s.parameter_gradients(detached, groups)
    assert torch.equal(found["shared"], torch.zeros(2))
    stats = s.compare_gradient_blocks(found, found)
    assert stats["shared"]["cosine"] is None
    assert stats["shared"]["reference_norm"] == 0
    assert stats["branch"]["cosine"] == pytest.approx(1)
