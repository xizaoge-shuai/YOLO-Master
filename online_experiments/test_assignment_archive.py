"""Integrity gates before automatic diagnostic result commits."""

import hashlib
import json
import subprocess

import pytest

from online_experiments.archive_assignment_intervention import commit_files, verify_stage, write_preserving


def test_generated_archive_cannot_overwrite_manual_edits(tmp_path):
    path = tmp_path / "README.md"
    path.write_text("Existing manual note\n")
    with pytest.raises(ValueError, match="overwrite"):
        write_preserving(path, "Generated note\n")
    assert path.read_text() == "Existing manual note\n"


def make_stage(path):
    path.mkdir()
    (path / "status.txt").write_text("SUCCEEDED\n")
    names = ["per-image.csv", "assignments.csv", "summary.csv", "assignment-summary.csv", "selection.json"]
    for name in names:
        (path / name).write_text("value\n1\n")
    payload = path / names[0]
    summary = {
        "images": 2,
        "teacher_images": 12,
        "rows": 384,
        "training_updates": 0,
        "states_unchanged": True,
        "native_loss_parity_max_absolute": 0,
        "parity_max_relative_l2": 0.001,
        "outputs_sha256": {name: hashlib.sha256((path / name).read_bytes()).hexdigest() for name in names},
    }
    (path / "summary.json").write_text(json.dumps(summary))
    return payload


def test_archive_rejects_modified_measurements(tmp_path):
    stage = tmp_path / "smoke"
    payload = make_stage(stage)
    assert verify_stage(stage, 2)["images"] == 2
    payload.write_text("value\n999\n")
    with pytest.raises(ValueError, match="hash"):
        verify_stage(stage, 2)


def test_archive_rejects_wrong_counts_or_unfinished_run(tmp_path):
    stage = tmp_path / "smoke"
    make_stage(stage)
    with pytest.raises(ValueError, match="count"):
        verify_stage(stage, 128)
    (stage / "status.txt").write_text("RUNNING\n")
    with pytest.raises(ValueError, match="complete"):
        verify_stage(stage, 2)


@pytest.mark.parametrize("corruption", ["missing_digest", "nan"])
def test_archive_rejects_incomplete_or_nonfinite_integrity_record(tmp_path, corruption):
    stage = tmp_path / "smoke"
    make_stage(stage)
    path = stage / "summary.json"
    record = json.loads(path.read_text())
    if corruption == "missing_digest":
        record["outputs_sha256"] = {}
    else:
        record["parity_max_relative_l2"] = float("nan")
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        verify_stage(stage, 2)


def make_repo(path):
    subprocess.run(["git", "init", str(path)], check=True, capture_output=True)
    for key, value in (("user.name", "Archive Test"), ("user.email", "archive@example.invalid")):
        subprocess.run(["git", "config", key, value], cwd=path, check=True)
    (path / "research").mkdir()
    (path / "research/PROGRESS.md").write_text("Original progress\n")
    subprocess.run(["git", "add", "research/PROGRESS.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "Initialize test"], cwd=path, check=True, capture_output=True)
    (path / "result.md").write_text("Verified result\n")
    return "\n<!-- stage:test -->\nNew progress\n"


def test_commit_rejects_unrelated_progress_changes(tmp_path):
    block = make_repo(tmp_path)
    progress = tmp_path / "research/PROGRESS.md"
    progress.write_text(progress.read_text() + "Unrelated pending research\n")
    with pytest.raises(RuntimeError, match="progress"):
        commit_files(tmp_path, ["result.md"], block, "<!-- stage:test -->", "result.md", "Record test")
    assert "Unrelated pending research" in progress.read_text()
    assert "New progress" not in progress.read_text()


def test_commit_recovers_staging_and_does_not_duplicate_after_commit(tmp_path, monkeypatch):
    block = make_repo(tmp_path)
    real_run = subprocess.run

    def fail_commit(command, *args, **kwargs):
        if command[:2] == ["git", "commit"]:
            raise subprocess.CalledProcessError(1, command)
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", fail_commit)
    with pytest.raises(subprocess.CalledProcessError):
        commit_files(tmp_path, ["result.md"], block, "<!-- stage:test -->", "result.md", "Record test")
    monkeypatch.setattr(subprocess, "run", real_run)
    commit = commit_files(tmp_path, ["result.md"], block, "<!-- stage:test -->", "result.md", "Record test")
    again = commit_files(tmp_path, ["result.md"], block, "<!-- stage:test -->", "result.md", "Record test")
    assert commit == again
    assert (tmp_path / "research/PROGRESS.md").read_text().count("New progress") == 1
