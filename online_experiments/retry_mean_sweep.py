#!/usr/bin/env python3
"""Retry missing mean-sweep runs on physical GPU 0, stopping on first failure."""
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def completed(folder, epochs, method, seed, require_done=True):
    if require_done and not (folder / "DONE").exists():
        return False
    summary = json.loads((folder / "summary.json").read_text())
    with (folder / "results.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    if (summary["completed_epochs"] != epochs or summary["method"] != method
            or summary["seed"] != seed
            or [int(r["epoch"]) for r in rows] != list(range(1, epochs + 1))):
        raise RuntimeError(f"Incomplete or mislabeled result: {folder}")
    return True

def plan_runs(old, tasks, epochs):
    reuse, pending = [], []
    for method, seed in tasks:
        folder = old / "train" / f"{method}-s{seed}"
        if completed(folder, epochs, method, seed):
            reuse.append(folder)
        else:
            pending.append((method, seed))
    return reuse, pending

def verify_sources(folder, root):
    hashes = json.loads((folder / "args.json").read_text())["source_hashes"]
    if not hashes:
        raise RuntimeError(f"Missing source hashes: {folder}")
    for rel, expected in hashes.items():
        path = root / rel
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Source changed: {rel}; do not mix old and new runs")

def run_checked(command, log, env, cwd):
    with log.open("w") as stream:
        result = subprocess.run(command, env=env, cwd=cwd,
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        print("\n".join(log.read_text(errors="replace").splitlines()[-60:]), flush=True)
        raise RuntimeError(f"exit={result.returncode}; stopped queue; log={log}")

def refresh(sweep, job, epochs):
    sweep.collect(job, epochs)
    path = job / "comparison.txt"
    lines = path.read_text().splitlines()
    lines[0] = ("AP: 0-100. Reused old runs + serial GPU0 retries; "
                "timings are mixed execution conditions. Cache construction excluded.")
    path.write_text("\n".join(lines) + "\n")

def recover(old, job):
    sys.path.insert(0, str(ROOT))
    from online_experiments import mean_sweep as sweep

    if (old / "status.txt").read_text().startswith(("RUNNING", "QUEUED")):
        raise RuntimeError("Source job is still running")
    if (job / "reference_args.json").exists():
        raise RuntimeError("Use a fresh recovery directory")

    plan = json.loads((old / "plan.json").read_text())
    epochs, threads, tasks = plan["epochs"], plan["threads"], plan["tasks"]
    if any(m not in sweep.CASES or not isinstance(s, int) for m, s in tasks):
        raise RuntimeError("Invalid experiment plan")
    if len({(m, s) for m, s in tasks}) != len(tasks):
        raise RuntimeError("Duplicate experiment tasks")

    reuse, pending = plan_runs(old, tasks, epochs)
    for folder in reuse:
        verify_sources(folder, ROOT)

    train = job / "train"
    train.mkdir()
    for folder in reuse:
        (train / folder.name).symlink_to(folder.resolve(), target_is_directory=True)

    reference = job / "reference_args.json"
    reference.write_text((old / "reference_args.json").read_text())
    plan["gpus"] = ["0"]
    (job / "plan.json").write_text(json.dumps(plan, indent=2))

    uuid = subprocess.check_output(
        ["nvidia-smi", "-i", "0", "--query-gpu=uuid", "--format=csv,noheader"],
        text=True).strip()
    if not uuid.startswith("GPU-") or len(uuid.splitlines()) != 1:
        raise RuntimeError(f"Cannot resolve physical GPU 0 UUID: {uuid}")

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, HF_HUB_OFFLINE="1",
               PYTHONUNBUFFERED="1", CUBLAS_WORKSPACE_CONFIG=":4096:8",
               OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))

    provenance = dict(
        source_job=str(old), gpu_index=0, gpu_uuid=uuid,
        reused=[f.name for f in reuse], rerun_from_epoch_1=pending,
        orchestrator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (job / "recovery.json").write_text(json.dumps(provenance, indent=2))
    refresh(sweep, job, epochs)
    print(f"REUSE={len(reuse)} RETRY={len(pending)} GPU0={uuid}", flush=True)

    status = job / "status.txt"
    status.write_text("RUNNING GPU0 basic CUDA check\n")
    probe = """import torch
assert torch.cuda.is_available()
x = torch.ones((512, 512), device="cuda:0")
for _ in range(20):
    y = x @ x
    assert torch.isfinite(y).all().item()
    assert (y == 512).all().item()
torch.cuda.synchronize()
print("GPU0 CUDA basic check PASS:", torch.cuda.get_device_name(0), flush=True)
"""
    run_checked([sys.executable, "-u", "-c", probe],
                job / "gpu0-check.log", env, ROOT)

    for index, (method, seed) in enumerate(pending):
        for folder in reuse:
            verify_sources(folder, ROOT)

        name = f"{method}-s{seed}"
        status.write_text(
            f"RUNNING GPU=0 {name} completed={len(reuse)+index}/{len(tasks)}\n")
        print(f"START {name} GPU=0", flush=True)

        log = train / f"{name}.log"
        command = [
            sys.executable, "-u", str(ROOT / "online_experiments/mean_sweep.py"),
            "--reference", str(reference), "--job", str(job),
            "--worker", method, "--seed", str(seed), "--phase", "train",
            "--epochs", str(epochs), "--threads", str(threads)]
        run_checked(command, log, env, ROOT)

        folder = train / name
        completed(folder, epochs, method, seed, require_done=False)

        config_path = folder / "args.json"
        config = json.loads(config_path.read_text())
        config.update(
            parallel_sweep=False, recovery_source_job=str(old),
            recovery_orchestrator_sha256=provenance["orchestrator_sha256"])
        config_path.write_text(json.dumps(config, indent=2))

        (folder / "DONE").write_text("exit=0\n")
        refresh(sweep, job, epochs)
        print(f"END {name} completed={len(reuse)+index+1}/{len(tasks)}", flush=True)

    status.write_text("SUCCEEDED\n")
    print((job / "comparison.txt").read_text(), flush=True)

if __name__ == "__main__":
    old, job = (Path(p).resolve() for p in sys.argv[1:3])
    job.mkdir(parents=True, exist_ok=True)
    try:
        recover(old, job)
    except BaseException as exc:
        (job / "status.txt").write_text(f"FAILED: {exc}\n")
        raise
