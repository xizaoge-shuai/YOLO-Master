"""Tests for budget accounting and matched experiment evidence gates."""

import json
import subprocess

import pytest

from online_experiments.detail_probe import (
    check_counts,
    compare_schedules,
    publish_epoch_evidence,
    push_verified_archive,
    sha,
    validate_completed,
)


def test_query_budget_includes_tail_cumulative_rounding():
    check_counts(1, 17, 17, 4)
    check_counts(4, 17, 17, 17)
    check_counts(100, 6471, 6471, 161775)
    with pytest.raises(RuntimeError):
        check_counts(4, 17, 17, 16)
    with pytest.raises(RuntimeError):
        check_counts(1, 16, 17, 4)


def test_pairing_gate_rejects_changed_augmentation_schedule(tmp_path):
    outputs = [tmp_path / "transport", tmp_path / "detail"]
    for output in outputs:
        output.mkdir()
        (output / "summary.json").write_text(json.dumps({"schedule_sha256": {"001.json": "abc"}}))
    assert compare_schedules(outputs) == {"001.json": "abc"}
    (outputs[1] / "summary.json").write_text(json.dumps({"schedule_sha256": {"001.json": "different"}}))
    with pytest.raises(RuntimeError):
        compare_schedules(outputs)


def test_completion_gate_rejects_a_changed_schedule_even_with_done(tmp_path):
    (tmp_path / "schedules").mkdir()
    schedule = tmp_path / "schedules/001.json"
    schedule.write_text('{"queries": [1, 2, 3, 4]}')
    args = {"epochs": 1, "train_images": 17, "val_images": 8}
    summary = {"completed_epochs": 1, "query_images": 4, "schedule_sha256": {"001.json": sha(schedule)}}
    for name, value in (("args.json", args), ("summary.json", summary), ("metrics.json", [{"epoch": 1}])):
        (tmp_path / name).write_text(json.dumps(value))
    (tmp_path / "DONE").write_text("SUCCEEDED\n")
    validate_completed(tmp_path, True)
    schedule.write_text('{"queries": [5, 6, 7, 8]}')
    with pytest.raises(RuntimeError, match="schedules changed"):
        validate_completed(tmp_path, True)


def test_final_checkpoint_republishes_missing_and_stale_sidecars(tmp_path):
    state = {"epoch": 100, "rows": [{"epoch": i} for i in range(1, 101)], "summary": {"completed_epochs": 100}}
    (tmp_path / "summary.json").write_text('{"completed_epochs": 99}')
    publish_epoch_evidence(tmp_path, state)
    assert json.loads((tmp_path / "summary.json").read_text())["completed_epochs"] == 100
    assert len(json.loads((tmp_path / "metrics.json").read_text())) == 100
    state["rows"].pop()
    with pytest.raises(RuntimeError, match="evidence is incomplete"):
        publish_epoch_evidence(tmp_path, state)


def test_archive_retry_preserves_a_pushed_descendant(tmp_path):
    remote, repo = tmp_path / "remote.git", tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()

    git("init", "--bare", str(remote))
    git("init", "-b", "main")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Test")
    git("remote", "add", "origin", str(remote))
    (repo / "result.txt").write_text("archive A")
    git("add", "result.txt")
    git("commit", "-m", "archive A")
    old = git("rev-parse", "HEAD")
    git("push", "origin", "main")
    (repo / "result.txt").write_text("descendant B")
    git("commit", "-am", "descendant B")
    new = git("rev-parse", "HEAD")
    git("push", "origin", "main")
    assert push_verified_archive(repo, old) == new
    assert git("ls-remote", "origin", "refs/heads/main").split()[0] == new
