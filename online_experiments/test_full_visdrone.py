"""Contracts that differ from the earlier fixed-size pilot."""

import random

import numpy as np
import pytest

from online_experiments.full_visdrone_data import (
    clean_labels,
    epoch_plan,
    inverse_letterbox,
    read_feature_batch,
    validate_splits,
)


def test_full_split_budget_keeps_every_tail_image_and_pairs_arms():
    total = 0
    for epoch in range(100):
        order, flips, scales, queries = epoch_plan(6471, 25, epoch, 0)
        assert sorted(order) == list(range(6471))
        assert len(flips) == len(scales) == len(queries) == 6471
        assert len(order[-(6471 % 8) :]) == 7
        total += sum(queries)
        assert total == ((epoch + 1) * 6471 * 25) // 100
    assert total == 161775
    state = random.getstate()
    assert epoch_plan(17, 25, 4, 2) == epoch_plan(17, 25, 4, 2)
    assert random.getstate() == state
    online = epoch_plan(17, 100, 4, 2)
    mixed = epoch_plan(17, 25, 4, 2)
    assert online[:3] == mixed[:3]
    assert all(online[3])
    assert not any(epoch_plan(17, 0, 4, 2)[3])


def test_only_zero_area_labels_are_filtered_and_logged():
    rows, removed = clean_labels("3 0.4 0.2 0.1 0\n1 0.5 0.5 0.2 0.3\n")
    assert rows == [[1.0, 0.5, 0.5, 0.2, 0.3]]
    assert removed == [1]
    for bad in ["10 0.5 0.5 0.1 0.1", "1 nan 0.5 0.1 0.1", "1 0 0 -1 0.1"]:
        with pytest.raises(ValueError):
            clean_labels(bad)


def test_split_overlap_rejected_even_with_different_paths():
    a = {"sample_id": "train/a", "image_sha256": "same"}
    b = {"sample_id": "val/b", "image_sha256": "same"}
    with pytest.raises(ValueError, match="overlap"):
        validate_splits([a], [b])


def test_inverse_letterbox_uses_actual_rounded_xy_scales():
    info = {"original_shape": [100, 201], "resized_shape": [318, 640], "pad": [0, 161]}
    box = np.array([[0.0, 161.0, 640.0, 479.0, 0.8, 3.0]])
    result = inverse_letterbox(box, info)
    np.testing.assert_allclose(result, [[0.0, 0.0, 201.0, 100.0, 0.8, 4.0, -1.0, -1.0]])


def test_memmap_gathers_requested_views_without_mutating_disk(tmp_path):
    arrays = []
    for view in range(2):
        x = np.arange(5 * 3 * 2 * 2 * 2, dtype=np.float16).reshape(5, 3, 2, 2, 2) + view * 200
        path = tmp_path / f"v{view}.npy"
        np.save(path, x)
        arrays.append(np.load(path, mmap_mode="r"))
    out = read_feature_batch(arrays, [4, 0, 2], [1, 0, 1])
    assert out.shape == (3, 3, 2, 2, 2)
    np.testing.assert_array_equal(out[0], arrays[1][4])
    before = float(arrays[1][4, 0, 0, 0, 0])
    out[0, 0, 0, 0, 0] = -1
    assert arrays[1][4, 0, 0, 0, 0] == before


def test_cache_corruption_is_rejected(tmp_path):
    from online_experiments.full_visdrone_data import sha, verify_cache_part

    path = tmp_path / "cache.npy"
    data = np.ones((2, 3, 2, 2), dtype=np.float16)
    np.save(path, data)
    expected = {"sha256": sha(path), "bytes": path.stat().st_size}
    verify_cache_part(path, expected, data.shape)
    changed = np.load(path, mmap_mode="r+")
    changed[0, 0, 0, 0] = 2
    changed.flush()
    del changed
    with pytest.raises(RuntimeError, match="Cache"):
        verify_cache_part(path, expected, data.shape)


def test_smoke_cannot_be_resumed_as_full():
    from online_experiments.full_visdrone_data import validate_mode

    meta = {"smoke": True, "splits": {"train": [{}] * 17, "val": [{}] * 8}}
    validate_mode(meta, True)
    with pytest.raises(ValueError, match="mode"):
        validate_mode(meta, False)



def test_full_split_pareto_query_budgets_are_exact():
    expected = {
        10: 64710,
        50: 323550,
    }

    for pct, target in expected.items():
        total = 0

        for epoch in range(100):
            _, _, _, queries = epoch_plan(
                6471,
                pct,
                epoch,
                0,
            )
            total += sum(queries)

            assert total == (
                ((epoch + 1) * 6471 * pct)
                // 100
            )

        assert total == target
