"""Urgent full-data queue: two native detectors, four paper adaptations, then Detail25.

Scheduling replacement only: reuse the sealed VOC/SKU dataset, feature banks,
teacher identity, final-epoch protocol, and unmodified trainers. Never enqueue
seed repeats, extra cache controls, budget sweeps, or ablations in this campaign.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
METHODS = (
    "master-n",
    "yolo12-n",
    "frofa-c2-d01",
    "loffta-c2-n005",
    "cache-reuse-c1-int8",
    "vit-adapter-cache25",
    "detail25",
)
DIRECT = frozenset(METHODS[:2])


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    from online_experiments.crossdataset_campaign import write as atomic_write

    atomic_write(path, value)


def sha(path):
    from online_experiments.crossdataset_campaign import sha as digest

    return digest(path)


def status(args, state, message, **extra):
    from online_experiments.crossdataset_campaign import status as emit

    emit(args.job, state, message, dataset=args.dataset, **extra)


def output_for(args, method):
    """Adopt only a previously completed endpoint, never a partial old case."""
    if method not in DIRECT:
        old = args.dataset_job / "train" / f"{method}-s0"
        if (old / "DONE").is_file():
            return old
    return args.job / "train" / f"{method}-s0"


def command_for(args, action, method):
    if method not in METHODS:
        raise ValueError("Method is outside the seven authorized single-seed cases")
    output = output_for(args, method)
    if method in DIRECT:
        command = [sys.executable, "-u", str(ROOT / "online_experiments/direct_baseline_suite.py"), action]
        if action == "validate":
            return command + ["--output", str(output)]
        manifest = read(args.dataset_job / "provenance.json")["manifest"]
        prepared = args.job.parent / "datasets" / args.dataset
        command += ["--dataset-manifest", manifest]
        if action == "prepare":
            return command + ["--output", str(prepared)]
        return command + ["--prepared", str(prepared), "--method", method, "--output", str(output), "--device", "0"]
    if action not in {"train", "validate"}:
        raise ValueError("Existing sealed feature banks require no new preparation")
    trainer = "crossdataset_suite.py" if args.dataset == "voc0712" else "sku_official_loader_suite.py"
    return [
        sys.executable,
        "-u",
        str(ROOT / "online_experiments" / trainer),
        action,
        "--job",
        str(args.dataset_job),
        "--method",
        method,
        "--seed",
        "0",
        "--output",
        str(output),
    ]


def live_output_workers(args):
    """Detect orphans by exact output argument before restarting a supervisor."""
    outputs = {str(output_for(args, method)) for method in METHODS}
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            command = (entry / "cmdline").read_bytes().decode(errors="surrogateescape").split("\0")
            if "--output" in command and command[command.index("--output") + 1] in outputs:
                found.append(int(entry.name))
        except (OSError, IndexError):
            continue
    return found


def execute(args, action, method):
    command = command_for(args, action, method)
    label = f"{action}-{method}"
    write(args.job / "commands" / f"{label}.json", {"argv": command, "cwd": str(ROOT)})
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES="0",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HOME=str(ROOT.parent / "cache/huggingface"),
        GH_CONFIG_DIR=str(ROOT.parent / ".config/gh"),
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        PYTHONUNBUFFERED="1",
        TMPDIR=str(ROOT.parent / "tmp/paper-baselines"),
        PYTHONPATH=str(args.dependencies) + os.pathsep + str(ROOT),
    )
    with (args.job / f"{label}.log").open("a", encoding="utf-8") as stream:
        child = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            pass_fds=(args.gpu_lock_fd,) if hasattr(args, "gpu_lock_fd") else (),
        )
        write(args.job / "active-child.json", {"pid": child.pid, "action": action, "method": method, "argv": command})
        result = child.wait()
    write(args.job / "last-child.json", {"pid": child.pid, "action": action, "method": method, "returncode": result})
    if result:
        raise RuntimeError(f"{label} exited {result}; preserve logs/checkpoints; no automatic retry")


def validate_inputs(args):
    from online_experiments import crossdataset_suite

    if args.dataset == "sku110k":
        from online_experiments.sku_official_loader_suite import install

        suite = install()
    else:
        suite = crossdataset_suite
    provenance = suite.verify_provenance(args.dataset_job)
    if provenance["dataset"] != args.dataset:
        raise ValueError("Sealed dataset job does not match selected dataset")
    suite.cache_manifest(args.dataset_job)
    expected = {"voc0712": {"train": 16551, "test": 4952}, "sku110k": {"train": 8219, "test": 2936}}[args.dataset]
    if any(provenance["counts"].get(key) != count for key, count in expected.items()):
        raise ValueError("Expected complete official train/test splits; subsets are forbidden")
    return provenance


def plan(args):
    provenance = validate_inputs(args)
    sources = (
        "online_experiments/priority_single_seed_campaign.py",
        "online_experiments/direct_baseline_suite.py",
        "online_experiments/crossdataset_campaign.py",
    )
    return {
        "schema": "urgent-seven-method-single-seed-v1",
        "dataset": args.dataset,
        "dataset_job": str(args.dataset_job),
        "provenance_sha256": sha(args.dataset_job / "provenance.json"),
        "cache_manifest_sha256": sha(args.dataset_job / "cache/manifest.json"),
        "manifest": provenance["manifest"],
        "manifest_sha256": provenance["manifest_sha256"],
        "counts": provenance["counts"],
        "methods": list(METHODS),
        "source_sha256": {name: sha(ROOT / name) for name in sources},
        "seed": 0,
        "epochs": 100,
        "batch": 8,
        "imgsz": 640,
        "optimizer": "AdamW",
        "final_checkpoint_only": True,
        "gpu": 0,
        "concurrency": "Up to two existing shared GPU0 leases; 8GiB first,16GiB second admission reserve",
        "timing": "Shared-GPU observations; not controlled efficiency or equal-time measurements",
        "paper_baseline_scope": "Detection adaptations of supplied paper ideas, not claims of original-paper reproduction",
        "excluded_by_user_priority": ["seed1", "seed2", "extra cache/mix controls", "ablations", "cross-backbone"],
    }


@contextlib.contextmanager
def archive_lock():
    import fcntl

    folder = ROOT / "runs/paper/priority"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "archive.lock").open("a") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        yield


def archive_case(args, method):
    from online_experiments.archive_assignment_intervention import commit_files
    from online_experiments.replication_campaign import push, sanitized

    marker_path = args.job / "archives" / f"{method}-s0.json"
    if marker_path.is_file():
        previous = read(marker_path)
        if args.publish and not previous.get("pushed"):
            previous.update(pushed=True, pushed_head=push())
            write(marker_path, previous)
        return previous
    source = output_for(args, method)
    destination = ROOT / "research/experiments/priority-single-seed-20261112" / args.dataset / f"{method}-s0"
    destination.mkdir(parents=True, exist_ok=True)
    required = (
        ("summary.json", "metrics.json", "DONE.json")
        if method in DIRECT
        else ("args.json", "summary.json", "metrics.json", "environment.json", "initialization.json", "endpoint.json")
    )
    names = []
    for name in required:
        if not (source / name).is_file():
            raise RuntimeError("Validated endpoint lacks required archive evidence: " + name)
    available = set(required) | {"args.json", "environment.json", "initialization.json", "endpoint.json"}
    for name in sorted(available):
        if not (source / name).is_file():
            continue
        target = destination / name
        content = json.dumps(sanitized(read(source / name)), indent=2, allow_nan=False) + "\n"
        if target.exists() and target.read_text(encoding="utf-8") != content:
            raise RuntimeError("Refuse changed archived endpoint: " + str(target))
        target.write_text(content, encoding="utf-8")
        names.append(target.relative_to(ROOT).as_posix())
    reference = destination / "source-reference.json"
    write(
        reference,
        {
            "source": str(source),
            "dataset_job": str(args.dataset_job),
            "pipeline_plan_sha256": sha(args.job / "pipeline-plan.json"),
        },
    )
    names.append(reference.relative_to(ROOT).as_posix())
    readme = destination / "README.md"
    readme.write_text(
        f"# {args.dataset}: {method}, seed 0\n\n"
        "Full official splits; fixed final epoch100; batch8,640,AdamW. "
        "Native YOLO-Master-N/YOLO12-N are directly trained detector configurations. "
        "Other named paper baselines are detection idea adaptations. "
        "Shared-GPU timings are observational. Exact metric definition is retained in metrics.json. "
        "This is one seed, not a seed mean.\n",
        encoding="utf-8",
    )
    names.append(readme.relative_to(ROOT).as_posix())
    marker = f"priority-single-seed-20261112:{args.dataset}:{method}"
    block = f"\n\n<!-- {marker} -->\n## Priority {args.dataset}: {method} completed\n\nFull seed0/final100 endpoint archived: [{method}]({readme.relative_to(ROOT / 'research').as_posix()}). No seed repeats or ablations queued.\n"
    with archive_lock():
        commit = commit_files(
            ROOT,
            names,
            block,
            marker,
            readme.relative_to(ROOT).as_posix(),
            f"research: record priority {args.dataset} {method} seed0",
        )
        result = {"commit": commit, "pushed": False}
        if args.publish:
            try:
                result.update(pushed=True, pushed_head=push())
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                result["push_error"] = str(error)
        write(marker_path, result)
    return result


def run(args):
    from online_experiments.crossdataset_campaign import Busy, gpu_admission, job_lock

    args.job.mkdir(parents=True, exist_ok=True)
    with job_lock(args.job):
        if live_output_workers(args):
            raise Busy("A worker already owns one of these exact output paths")
        contract = plan(args)
        path = args.job / "pipeline-plan.json"
        if path.exists() and read(path) != contract:
            raise RuntimeError("Plan/source changed; preserve this job and use a new version")
        write(path, contract)
        write(
            args.job / "worker.json",
            {
                "pid": os.getpid(),
                "argv": sys.argv,
                "started_unix": time.time(),
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            },
        )
        status(args, "PREPARING_NATIVE_DATA", "Full official split labels for both direct detectors")
        execute(args, "prepare", "master-n")
        completed = []
        for index, method in enumerate(METHODS, 1):
            output = output_for(args, method)
            done = output / ("DONE.json" if method in DIRECT else "DONE")
            if not done.is_file():
                with gpu_admission(args, f"priority {method} seed0 {index}/7"):
                    status(
                        args,
                        "TRAINING",
                        f"{method}: full data,seed0,100 epochs",
                        method=method,
                        completed=len(completed),
                        planned=7,
                    )
                    execute(args, "train", method)
            status(args, "VALIDATING", method, method=method, completed=len(completed), planned=7)
            execute(args, "validate", method)
            if not done.is_file():
                raise RuntimeError("Validator returned without endpoint completion marker")
            result = archive_case(args, method)
            completed.append({"method": method, "output": str(output), "archive": result})
            write(args.job / "completed-methods.json", completed)
        status(
            args,
            "SUCCEEDED",
            "All seven priority methods completed at seed0/final100",
            completed=7,
            planned=7,
            all_archives_pushed=all(item["archive"]["pushed"] for item in completed),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "status"))
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--dataset", choices=("voc0712", "sku110k"))
    parser.add_argument("--dataset-job", type=Path)
    parser.add_argument("--dependencies", type=Path, default=ROOT / ".research-deps/evidence-20261110/site")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    args.job = args.job.resolve()
    if args.action == "status":
        path = args.job / "pipeline-status.json"
        print(json.dumps(read(path) if path.is_file() else {"state": "NOT_STARTED", "job": str(args.job)}, indent=2))
        return
    if args.dataset is None or args.dataset_job is None:
        parser.error("run requires --dataset and --dataset-job")
    args.dataset_job, args.dependencies = args.dataset_job.resolve(), args.dependencies.resolve()
    args.gpu, args.min_free_mib = 0, 8192
    from online_experiments.crossdataset_campaign import Busy

    try:
        run(args)
    except Busy as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from error
    except Exception:
        args.job.mkdir(parents=True, exist_ok=True)
        write(
            args.job / f"failure-{time.time_ns()}.json",
            {
                "traceback": traceback.format_exc(),
                "time_unix": time.time(),
                "policy": "Preserve all data/logs/checkpoints; no automatic retry or new method",
            },
        )
        status(args, "FAILED", "Inspect failure record and exact stage log")
        raise


if __name__ == "__main__":
    main()
