"""Screening must preserve failed cells and distinguish smoke from development."""

import pytest

from online_experiments.diagnose_joint_gradient import screening_decision


def records(count, ratio=0.5):
    return [
        {
            "arm": arm,
            "head": head,
            "scale": scale,
            "block": "__all__",
            "exact_squared_norm": 2.0,
            "residual_squared_norm": 2.0 * ratio,
        }
        for arm in ("transport", "blank_fill", "rec")
        for head in ("online-geom", "mix25")
        for scale in (0.85, 1.15)
        for _ in range(count * 2)
    ]


def test_smoke_does_not_qualify_for_training():
    result = screening_decision(records(2), 2)
    assert not result["eligible_for_minibatch_probe"]
    assert result["full_training_authorized_by_this_gate"] is False


def test_one_failing_head_scale_excludes_candidate():
    rows = records(128)
    for row in rows:
        if row["arm"] == "rec" and row["head"] == "mix25" and row["scale"] == 0.85:
            row["residual_squared_norm"] = 2.0
    result = screening_decision(rows, 128)
    assert result["eligible_for_minibatch_probe"] == ["transport", "blank_fill"]
    assert len(result["cells"]) == 12
    assert result["full_training_authorized_by_this_gate"] is False


def test_screen_uses_pooled_energy_and_rejects_incomplete_grid():
    rows = records(128)
    rows[0]["exact_squared_norm"] = 1000.0
    rows[0]["residual_squared_norm"] = 100.0
    result = screening_decision(rows, 128)
    assert result["cells"][0]["variance_ratio"] == pytest.approx(355.0 / 1510.0)
    with pytest.raises(ValueError, match="Incomplete"):
        screening_decision(rows[1:], 128)


def test_zero_reference_cannot_pass_gate():
    rows = records(128)
    for row in rows:
        row["exact_squared_norm"] = 0.0
    result = screening_decision(rows, 128)
    assert result["eligible_for_minibatch_probe"] == []
    assert all(cell["variance_ratio"] is None for cell in result["cells"])
