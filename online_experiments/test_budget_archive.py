"""Git archive must preserve evidence and avoid unrelated staged work."""

import json
import subprocess

import pytest

from online_experiments import archive_budget_corrector as archiver
from online_experiments.budget_corrector import ConditionalCorrector


def test_corrector_initialization_does_not_seed_cuda(monkeypatch):
    import torch

    def forbidden(*args, **kwargs):
        raise AssertionError("Corrector constructor touched CUDA RNG")

    monkeypatch.setattr(torch.cuda, "manual_seed_all", forbidden)
    ConditionalCorrector(channels=8, rank=4, seed=0)


def test_archive_scopes_commit_and_preserves_raw_bytes(tmp_path, monkeypatch):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()

    git("init", "-q")
    git("config", "user.name", "Archive test")
    git("config", "user.email", "archive-test@example.invalid")
    research = tmp_path / "research"
    research.mkdir()
    (research / "PROGRESS.md").write_text("# Progress\n")
    git("add", "research/PROGRESS.md")
    git("commit", "-qm", "initial test")
    job = tmp_path / "runs/job-smoke-test"
    job.mkdir(parents=True)
    (job / "status.txt").write_text("SMOKE_SUCCEEDED\n")
    run = job / "smoke/c2-mix-p10-s0"
    run.mkdir(parents=True)
    original = b"epoch,map50_95\r\n1,0.001\r\n"
    (run / "results.csv").write_bytes(original)
    monkeypatch.setattr(archiver, "ROOT", tmp_path)
    (tmp_path / "unrelated.txt").write_text("leave me alone")
    git("add", "unrelated.txt")
    with pytest.raises(RuntimeError, match="staged changes"):
        archiver.archive(job)
    git("reset", "-q", "HEAD", "--", "unrelated.txt")
    archiver.archive(job)
    after = git("rev-parse", "HEAD")
    archive = research / "experiments/budget-corrector-job-smoke-test"
    assert (archive / "original/smoke/c2-mix-p10-s0/metrics.csv").read_bytes() == original
    manifest = json.loads((archive / "archive-manifest.json").read_text())
    assert manifest["status"] == "SMOKE_SUCCEEDED"
    assert "?? unrelated.txt" in git("status", "--short")
    assert not git("diff", "--cached", "--name-only")
    archiver.archive(job)
    assert after == git("rev-parse", "HEAD")
