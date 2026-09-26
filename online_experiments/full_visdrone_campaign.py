"""Serial full-split campaign, telemetry and scoped Git evidence archive."""

import argparse
import fcntl
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from online_experiments.full_visdrone_data import (
    build,
    dump,
    prepare,
    read,
    sha,
    source_snapshot,
    validate_mode,
    verify_inputs_and_cache,
    verify_sources,
)

METHODS = ("cache1", "cache2", "mix25", "transport25", "correct25", "online-geom")


def status(job, message):
    (job / "status.txt").write_text(message + "\n")
    print(message, flush=True)


def comparison(job, phase):
    lines = [
        "Diagnostic AP 0-100; NOT official VisDrone AP. Smoke is not a benchmark.",
        "Cache/data preparation charged once per run, including validation cache.",
        "method seed epochs best_AP last_AP train_s val_s build+wall_s queries cache_GiB VRAM_GiB",
    ]
    values = []
    for method in METHODS:
        run = job / phase / f"{method}-s0"
        if not (run / "DONE").exists():
            continue
        s = read(run / "summary.json")
        values.append(s)
        lines.append(
            f"{method} 0 {s['completed_epochs']} {s['best_AP']:.4f} {s['last_AP']:.4f} "
            f"{s['train_seconds']:.1f} {s['val_seconds']:.1f} {s['build_plus_wall_seconds']:.1f} "
            f"{s['query_images']} {s['cache_bytes'] / 2**30:.3f} {s['peak_vram_gib']:.3f}"
        )
    prefix = "comparison" if phase == "train" else f"{phase}-comparison"
    (job / f"{prefix}.txt").write_text("\n".join(lines) + "\n")
    dump(job / f"{prefix}.json", values)
    return values


def run_phase(job, phase, epochs):
    (job / phase).mkdir(exist_ok=True)
    for method in METHODS:
        run = job / phase / f"{method}-s0"
        if (run / "DONE").exists():
            print(f"SKIP completed {phase}/{method}", flush=True)
            continue
        verify_sources(read(job / "provenance.json"))
        status(job, f"RUNNING {phase}/{method} epochs={epochs}")
        command = [
            sys.executable,
            "-u",
            "online_experiments/full_visdrone_train.py",
            "--job",
            str(job),
            "--output",
            str(run),
            "--method",
            method,
            "--epochs",
            str(epochs),
        ]
        logfile = job / f"{phase}-{method}.log"
        with logfile.open("a") as log:
            code = subprocess.call(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        comparison(job, phase)
        if code:
            print("\n".join(logfile.read_text(errors="replace").splitlines()[-25:]), flush=True)
            raise RuntimeError(f"{phase}/{method} exit={code}; inspect {logfile.name}")
        s = read(run / "summary.json")
        print(f"END {phase}/{method} AP={s['best_AP']:.4f} train_s={s['train_seconds']:.1f}", flush=True)
    summaries = [read(job / phase / f"{m}-s0/summary.json") for m in ("mix25", "transport25", "correct25")]
    if any(s["schedule_sha256"] != summaries[0]["schedule_sha256"] for s in summaries[1:]):
        raise RuntimeError("Paired query/augmentation schedule mismatch")
    if len({s["query_images"] for s in summaries}) != 1:
        raise RuntimeError("Paired query image count mismatch")


def archive(job):
    text = (job / "status.txt").read_text().strip()
    if text.startswith("RUNNING"):
        raise RuntimeError("Cannot archive a running job")
    name = "full-visdrone-" + job.name + "-" + text.split()[0].lower()
    dest = ROOT / "research/experiments" / name
    rel = dest.relative_to(ROOT).as_posix()
    if (dest / "manifest.json").exists():
        commit = subprocess.check_output(["git", "log", "-1", "--format=%H", "--", rel], cwd=ROOT, text=True).strip()
        if not commit:
            raise RuntimeError("Existing uncommitted archive; inspect it")
        print(f"COMMIT={commit} ARCHIVE={rel}")
        return
    if subprocess.call(["git", "diff", "--cached", "--quiet"], cwd=ROOT):
        raise RuntimeError("Git index contains staged work")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", "research/PROGRESS.md"], cwd=ROOT, text=True
    ).strip():
        raise RuntimeError("Commit existing progress edits before auto-archive")
    paths = [p for p in job.glob("*.json") if p.name != "dataset-manifest.json"]
    paths += list(job.glob("*comparison.txt")) + [job / "status.txt"] + list((job / "cache").glob("*.json"))
    for phase in ("train", "profile", "smoke"):
        for folder in (job / phase).glob("*"):
            paths += [
                folder / f
                for f in ("args.json", "summary.json", "trainability.json", "metrics.csv", "DONE")
                if (folder / f).is_file()
            ]
    if text.startswith("FAILED"):
        paths += [p for p in job.glob("*.log") if p.name != "archive.log" and p.stat().st_size < 2 * 1024**2]
    dest.mkdir(parents=True)
    hashes = {}
    for src in paths:
        relative = src.relative_to(job)
        if relative.suffix == ".log":
            relative = relative.with_suffix(".txt")
        dst = dest / "original" / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        if sha(src) != sha(dst):
            raise RuntimeError("Archive copy mismatch")
        hashes[dst.relative_to(dest).as_posix()] = sha(dst)
    (dest / ".gitattributes").write_text("original/** -text -whitespace\n")
    meta = read(job / "dataset-manifest.json") if (job / "dataset-manifest.json").exists() else {}
    dump(
        dest / "manifest.json",
        {
            "source_job": str(job),
            "status": text,
            "files": hashes,
            "split_counts": {k: len(v) for k, v in meta.get("splits", {}).items()},
            "dataset_manifest_sha256": sha(job / "dataset-manifest.json") if meta else None,
            "note": "Diagnostic metrics; official evaluation pending; test-dev untouched.",
        },
    )
    (dest / "README.md").write_text(
        f"# Full VisDrone pipeline evidence\n\nStatus: {text}\n\n"
        "No official VisDrone AP claim. Smoke is not a benchmark.\n"
    )
    with (ROOT / "research/PROGRESS.md").open("a") as f:
        f.write(
            f"\n## Full VisDrone: {job.name}\n\n- Status: {text}\n"
            f"- Evidence: [archive](experiments/{name}/README.md)\n"
            "- Diagnostic AP only; official ignored-region evaluation pending. No test-dev tuning.\n"
        )
    subprocess.run(["git", "add", "--", rel, "research/PROGRESS.md"], cwd=ROOT, check=True)
    staged = subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True).splitlines()
    if any(p != "research/PROGRESS.md" and not p.startswith(rel + "/") for p in staged):
        raise RuntimeError("Unexpected staged paths")
    subprocess.run(["git", "diff", "--cached", "--check"], cwd=ROOT, check=True)
    subprocess.run(["git", "commit", "-m", f"research(d1): archive {job.name} {text.split()[0]}"], cwd=ROOT, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    print(f"COMMIT={commit} ARCHIVE={rel}")


def campaign(job, smoke):
    attempt_start = time.time()
    lock = (ROOT / "runs/paper/full-visdrone/.gpu0.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (job / "provenance.json").exists():
        verify_sources(read(job / "provenance.json"))
    else:
        dump(job / "provenance.json", source_snapshot())
        dump(
            job / "plan.json",
            {
                "methods": METHODS,
                "epochs": 1 if smoke else 100,
                "seeds": [0],
                "train_images": 17 if smoke else 6471,
                "val_images": 8 if smoke else 548,
                "smoke": smoke,
                "device": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "official_evaluation": False,
                "test_dev_used": False,
            },
        )
    plan = read(job / "plan.json")
    if plan["smoke"] != smoke:
        raise ValueError("Campaign mode differs from saved plan")
    log = (job / "gpu-telemetry.csv").open("a")
    monitor = subprocess.Popen(
        [
            "nvidia-smi",
            "-i",
            "0",
            "--query-gpu=timestamp,uuid,utilization.gpu,memory.used,power.draw,temperature.gpu",
            "--format=csv,noheader",
            "-l",
            "5",
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    try:
        if not (job / "dataset-manifest.json").exists():
            status(job, "RUNNING prepare official train/val")
            prepare(job, smoke)
        validate_mode(read(job / "dataset-manifest.json"), smoke)
        if not (job / "cache/manifest.json").exists():
            status(job, "RUNNING build streamed cache")
            build(job)
        status(job, "RUNNING verify cache and raw data integrity")
        verify_inputs_and_cache(job)
        if smoke:
            run_phase(job, "smoke", 1)
        else:
            run_phase(job, "profile", 1)
            values = comparison(job, "profile")
            hours = sum(x["train_seconds"] + x["val_seconds"] for x in values) * 100 / 3600
            dump(
                job / "eta.json",
                {
                    "estimated_training_validation_hours": hours,
                    "basis": "six full-split one-epoch profiles x100; excludes checkpoint I/O variation",
                },
            )
            print(f"PROFILE estimated remaining training+validation={hours:.2f} hours", flush=True)
            run_phase(job, "train", 100)
        status(job, "SUCCEEDED")
    except BaseException as exc:
        status(job, f"FAILED {type(exc).__name__}: {exc}")
        raise
    finally:
        monitor.terminate()
        monitor.wait(timeout=10)
        log.close()
        cost_path = job / "campaign-cost.json"
        costs = read(cost_path) if cost_path.exists() else {"attempts": []}
        costs["attempts"].append(
            {
                "start_unix": attempt_start,
                "end_unix": time.time(),
                "wall_seconds": time.time() - attempt_start,
                "status": (job / "status.txt").read_text().strip(),
            }
        )
        costs["actual_campaign_wall_seconds"] = sum(x["wall_seconds"] for x in costs["attempts"])
        costs["note"] = (
            "Actual campaign includes prepare/cache/profiles/training and failed attempts; excludes Git archival and downtime. Per-run charged build costs are separate, not summed here."
        )
        dump(cost_path, costs)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--archive", action="store_true")
    a = p.parse_args()
    archive(a.job.resolve()) if a.archive else campaign(a.job.resolve(), a.smoke)
