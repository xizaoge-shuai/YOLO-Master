"""Serial full-data correction ablation with paired smoke and automatic archival."""

import argparse
import fcntl
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "online_experiments"))

from online_experiments.full_visdrone_data import (
    dump,
    read,
    sha,
    source_snapshot,
    verify_inputs_and_cache,
    verify_sources,
)

METHODS = ("gate25", "fg25", "gatefg25")
OWN = (
    "correction_ablation.py",
    "correction_ablation_campaign.py",
    "run_correction_ablation.sh",
    "test_correction_ablation.py",
)


def status(job, value):
    (job / "status.txt").write_text(value + "\n")
    print(value, flush=True)


def initialize(job, source, smoke):
    job.mkdir(parents=True, exist_ok=True)
    if (job / "plan.json").exists():
        plan = read(job / "plan.json")
        if plan["smoke"] != smoke or Path(plan["source_job"]) != source:
            raise RuntimeError("Resume source/mode changed")
        verify_sources(read(job / "provenance.json"))
        return plan
    meta = read(source / "dataset-manifest.json")
    counts = tuple(len(meta["splits"][s]) for s in ("train", "val"))
    if counts != ((17, 8) if smoke else (6471, 548)) or meta["smoke"] != smoke:
        raise RuntimeError("Source split differs from requested protocol")
    for name in ("dataset-manifest.json", "cache", "raw-val-annotations"):
        (job / name).symlink_to(source / name, target_is_directory=(source / name).is_dir())
    snapshot = source_snapshot()
    snapshot["files"].update({"online_experiments/" + name: sha(ROOT / "online_experiments" / name) for name in OWN})
    dump(job / "provenance.json", snapshot)
    plan = {
        "source_job": str(source),
        "smoke": smoke,
        "methods": METHODS,
        "seed": 0,
        "epochs": 2 if smoke else 100,
        "batch": 8,
        "query_percent": 25,
        "actual_new_cache_build_seconds": 0,
        "primary": "final reference AP",
        "test_dev_used": False,
        "source_dataset_sha256": sha(source / "dataset-manifest.json"),
        "source_cache_manifest_sha256": sha(source / "cache/manifest.json"),
    }
    dump(job / "plan.json", plan)
    return plan


def run_case(job, method, *, label=None, legacy=False, stop_after=0):
    plan = read(job / "plan.json")
    label = label or method
    output = job / "train" / f"{label}-s0"
    if (output / "DONE").exists():
        print(f"SKIP complete {label}", flush=True)
        return output
    verify_sources(read(job / "provenance.json"))
    status(job, f"RUNNING {label} epochs={plan['epochs']}")
    cmd = [
        sys.executable,
        "-u",
        "online_experiments/correction_ablation.py",
        "--job",
        str(job),
        "--output",
        str(output),
        "--method",
        method,
        "--epochs",
        str(plan["epochs"]),
        "--batch",
        "8",
    ]
    if legacy:
        cmd += ["--legacy"]
    if stop_after:
        cmd += ["--stop-after", str(stop_after)]
    with (job / f"{label}.log").open("a") as log:
        code = subprocess.call(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if code != (77 if stop_after else 0):
        raise RuntimeError(f"{label} exit={code}: see {label}.log")
    if not stop_after:
        s = read(output / "summary.json")
        if s["completed_epochs"] != plan["epochs"] or s["query_images"] != (plan["epochs"] * s["train_images"]) // 4:
            raise RuntimeError("Incomplete epochs or wrong online query count")
        print(f"END {label} custom_last_AP={s['last_AP']:.4f}", flush=True)
    return output


def smoke_checks(job):
    import torch

    legacy = run_case(job, "rec25", label="legacy", legacy=True)
    adapted = run_case(job, "rec25")
    for method in METHODS:
        run_case(job, method)
    run_case(job, "gatefg25", label="resume", stop_after=1)
    resumed = run_case(job, "gatefg25", label="resume")

    def equal_state(a, b):
        if torch.is_tensor(a):
            return torch.equal(a, b)
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(equal_state(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple)):
            return len(a) == len(b) and all(equal_state(x, y) for x, y in zip(a, b))
        return a == b

    checks = {}
    for name, left, right in [
        ("original_vs_adapter", legacy, adapted),
        ("continuous_vs_resumed", job / "train/gatefg25-s0", resumed),
    ]:
        a = torch.load(left / "weights/last.pt", map_location="cpu", weights_only=False)
        b = torch.load(right / "weights/last.pt", map_location="cpu", weights_only=False)
        for key in ("model", "corrector", "optimizer", "corrector_optimizer", "scheduler", "cpu_rng", "cuda_rng"):
            if not equal_state(a[key], b[key]):
                raise RuntimeError(f"Smoke equality failed: {name}/{key}")
        for key in ("query_images", "query_calls", "correction_steps"):
            assert a["counters"][key] == b["counters"][key], (name, key)
        checks[name] = "exact state equality"
    summaries = [read(p / "summary.json") for p in (job / "train").iterdir() if p.is_dir()]
    if any(s["schedule_sha256"] != summaries[0]["schedule_sha256"] for s in summaries):
        raise RuntimeError("Smoke query/transform schedules differ")
    checks["paired_schedules"] = "all seven cases match"
    dump(job / "smoke-checks.json", checks)


def reference_eval(job, method):
    from online_experiments.run_visdrone_reference_eval import load_cases, load_ground_truth
    from online_experiments.visdrone_reference_eval import evaluate_raw

    reference = Path((ROOT / "runs/paper/visdrone-eval/latest-job.txt").read_text().strip())
    evidence = read(reference / "provenance.json")
    if (reference / "status.txt").read_text().strip() != "SUCCEEDED":
        raise RuntimeError("Verified reference evaluation unavailable")
    for name, digest in evidence["evaluator_source_hashes"].items():
        if sha(ROOT / "online_experiments" / name) != digest:
            raise RuntimeError("Reference evaluator differs from the native-parity-verified implementation")
    if any(x["max_absolute_AP_point_error"] > 1e-8 for x in read(reference / "parity.json")):
        raise RuntimeError("Previous native parity did not pass")
    records, annotations, manifest_hash = load_ground_truth(job)
    if annotations != evidence["annotation_hashes"] or manifest_hash != evidence["manifest_sha256"]:
        raise RuntimeError("Reference validation data changed")
    dump(
        job / "evaluation/provenance.json",
        {
            "reference_job": str(reference),
            "parity_sha256": sha(reference / "parity.json"),
            "reference_provenance_sha256": sha(reference / "provenance.json"),
            "evaluator_source_hashes": evidence["evaluator_source_hashes"],
            "manifest_sha256": manifest_hash,
            "checkpoint_policy": "last primary; best selected by custom AP",
        },
    )
    run = job / "train" / f"{method}-s0"
    expected = read(Path(read(job / "plan.json")["source_job"]) / "train/correct25-s0/summary.json")
    summary = read(run / "summary.json")
    if summary["schedule_sha256"] != expected["schedule_sha256"] or summary["query_images"] != expected["query_images"]:
        raise RuntimeError("Full schedules/queries differ from historical paired baseline")
    for checkpoint in ("last", "best"):
        cases, fingerprints = load_cases(run / "predictions" / checkpoint, records)
        value = evaluate_raw(cases)
        value.update(method=method, checkpoint=checkpoint, seed=0, images=548)
        dump(job / "evaluation" / f"{method}-{checkpoint}.json", value)
        dump(job / "evaluation" / f"{method}-{checkpoint}-inputs.json", fingerprints)
    lines = [
        "AP 0-100; pinned reference. Final checkpoint primary. One seed only.",
        "Historical baselines are accuracy references; timings are not a controlled speed comparison.",
        "method origin final_AP best_checkpoint_AP train_s build+wall_s query_images",
    ]
    source = Path(read(job / "plan.json")["source_job"])
    for label in ("mix25", "transport25", "correct25", *METHODS):
        folder = reference if label not in METHODS else job / "evaluation"
        final = folder / f"{label}-last.json"
        best = folder / f"{label}-best.json"
        if not final.exists() or not best.exists():
            continue
        s = read((source if label not in METHODS else job) / "train" / f"{label}-s0/summary.json")
        lines.append(
            f"{label} {'historical' if label not in METHODS else 'new'} {read(final)['AP']:.4f} "
            f"{read(best)['AP']:.4f} {s['train_seconds']:.1f} {s['build_plus_wall_seconds']:.1f} {s['query_images']}"
        )
    (job / "comparison.txt").write_text("\n".join(lines) + "\n")


def archive(job, label):
    """Scoped, append-only per-stage evidence; never include caches or weights."""
    dest = ROOT / "research/experiments" / ("correction-ablation-" + job.name) / label
    if dest.exists():
        raise RuntimeError(f"Archive already exists: {dest}; preserve it")
    if subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True).strip():
        raise RuntimeError("Unrelated staged changes prevent automatic commit")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", "research/PROGRESS.md"], cwd=ROOT, text=True
    ).strip():
        raise RuntimeError("Uncommitted progress edits prevent automatic append")
    paths = [p for p in job.glob("*.json") if p.name != "dataset-manifest.json"]
    paths += [p for p in (job / "status.txt", job / "comparison.txt", job / "job.log") if p.exists()]
    paths += list((job / "evaluation").glob("*.json"))
    for run in (job / "train").glob("*"):
        paths += [
            p
            for p in run.iterdir()
            if p.is_file()
            and p.name in ("args.json", "summary.json", "trainability.json", "variant.json", "metrics.csv", "DONE")
        ]
    if label.startswith("failed"):
        paths += list(job.glob("*.log"))
    dest.mkdir(parents=True)
    hashes = {}
    for src in set(paths):
        dst = dest / "original" / src.relative_to(job)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        hashes[dst.relative_to(dest).as_posix()] = sha(dst)
    for name in (*OWN, "budget_corrector.py", "diagnose_full_corrector.py"):
        dst = dest / "source" / name
        dst.parent.mkdir(exist_ok=True)
        shutil.copyfile(ROOT / "online_experiments" / name, dst)
        hashes[dst.relative_to(dest).as_posix()] = sha(dst)
    (dest / ".gitattributes").write_text("original/** -text -whitespace\n")
    dump(dest / "manifest.json", {"job": str(job), "stage": label, "files": hashes})
    (dest / "README.md").write_text(
        f"# Correction ablation: {label}\n\nSource: {job}\n\n"
        "One-seed method screening; final reference AP primary. See plan and original evidence.\n"
    )
    with (ROOT / "research/PROGRESS.md").open("a") as f:
        f.write(
            f"\n## Correction ablation {job.name}: {label}\n\n"
            f"- [Evidence](experiments/correction-ablation-{job.name}/{label}/README.md).\n"
            "- Fixed gate/foreground factors, paired query schedules; no test-dev or stable-advantage claim.\n"
        )
    explicit = [str(p.relative_to(ROOT)) for p in dest.rglob("*") if p.is_file()] + ["research/PROGRESS.md"]
    subprocess.run(["git", "add", "-f", "--", *explicit], cwd=ROOT, check=True)
    subprocess.run(["git", "diff", "--cached", "--check"], cwd=ROOT, check=True)
    subprocess.run(["git", "commit", "-m", f"research(d1): correction ablation {label}"], cwd=ROOT, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    print(f"COMMIT={commit}", flush=True)
    (job / "archive-status.txt").write_text(f"COMMITTED {label} {commit}\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    job, source = a.job.resolve(), a.source.resolve()
    lock = (ROOT / "runs/paper/full-visdrone/.gpu0.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    initialize(job, source, a.smoke)
    with (job / "gpu-telemetry.csv").open("a") as log:
        monitor = subprocess.Popen(
            [
                "nvidia-smi",
                "-i",
                "0",
                "--query-gpu=timestamp,uuid,utilization.gpu,memory.used,power.draw",
                "--format=csv,noheader",
                "-l",
                "5",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            status(job, "RUNNING verify source, cache and raw inputs")
            verify_inputs_and_cache(job)
            if a.smoke:
                smoke_checks(job)
            else:
                for method in METHODS:
                    run_case(job, method)
                    status(job, f"RUNNING reference evaluation {method}")
                    reference_eval(job, method)
                    if not (job / f"archived-{method}").exists():
                        archive(job, method)
                        (job / f"archived-{method}").write_text("OK\n")
            status(job, "SUCCEEDED")
            archive(job, "succeeded")
        except BaseException as exc:
            status(job, f"FAILED {type(exc).__name__}: {exc}")
            try:
                archive(job, "failed-" + str(time.time_ns()))
            except Exception as err:  # noqa: BLE001 - preserve the original training failure if archival also fails
                (job / "archive-status.txt").write_text(f"ARCHIVE_FAILED {err}\n")
            raise
        finally:
            monitor.terminate()
            monitor.wait(timeout=10)


if __name__ == "__main__":
    main()
