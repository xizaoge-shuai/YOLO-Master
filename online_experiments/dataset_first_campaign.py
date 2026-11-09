"""Finish the existing VisDrone seed-zero cases; defer all independent seed repeats.

This external coordinator leaves the frozen trainers, queue and provenance
unchanged. It never sends process signals and never starts a new dataset,
smoke run or subset experiment. Run it after the old supervisors release their
locks, optionally adopting their still-running seed-zero training child.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PHASE = "dataset-first"


class CoordinationBusy(RuntimeError):
    """A different supervisor owns a lock; leave its progress status untouched."""


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def log(phase, message):
    with (phase / "job.log").open("a", encoding="utf-8") as stream:
        stream.write(time.strftime("%Y-%m-%dT%H:%M:%S%z ") + message + "\n")


def state(job, name, message, **extra):
    phase = job / PHASE
    record = {"state": name, "message": message, "time_unix": time.time(), **extra}
    write(phase / "status.json", record)
    target = job / "status.txt"
    temporary = target.with_name("status.txt.dataset-first.tmp")
    temporary.write_text(f"DATASET_FIRST {name}: {message}\n", encoding="utf-8")
    temporary.replace(target)
    log(phase, f"{name}: {message}")


def partition(queue):
    """Retain the actual frozen ordering, with no duplicate or invented cases."""
    names, first, deferred = set(), [], []
    for case in queue:
        seed, method, name = case.get("seed"), case.get("method"), case.get("name")
        if type(seed) is not int or seed not in (0, 1, 2) or not isinstance(method, str):
            raise ValueError("Malformed original method/seed identity")
        if name != f"{method}-s{seed}" or name in names:
            raise ValueError("Original queue has inconsistent or duplicate case names")
        names.add(name)
        (first if seed == 0 else deferred).append(case)
    if (len(queue), len(first), len(deferred)) != (28, 10, 18):
        raise ValueError("Expected the existing 28-case queue: 10 seed-zero, 18 seed repeats")
    if any(case.get("resume_original") for case in first):
        raise ValueError("Unexpected legacy-output seed-zero case")
    return first, deferred


def process(pid):
    """Read a stable Linux process identity, including kernel start-time ticks."""
    path = Path("/proc") / str(pid)
    try:
        before = (path / "stat").read_text().rsplit(") ", 1)[1].split()
        uid = path.stat().st_uid
        argv = (path / "cmdline").read_bytes().decode("utf-8", errors="surrogateescape").split("\0")
        if argv and argv[-1] == "":
            argv.pop()
        after = (path / "stat").read_text().rsplit(") ", 1)[1].split()
        if before[19] != after[19]:
            raise RuntimeError("Process ID was recycled during identity inspection")
        cwd = str((path / "cwd").resolve(strict=True)) if after[0] != "Z" else None
        return {"pid": pid, "uid": uid, "starttime": int(after[19]), "state": after[0], "argv": argv, "cwd": cwd}
    except (FileNotFoundError, ProcessLookupError):
        return None


def require_adopt_identity(observed, expected_argv, starttime):
    if observed["uid"] != os.getuid() or observed["starttime"] != starttime:
        raise RuntimeError("Adoption process UID/start time mismatch; refusing to attach")
    if observed["state"] != "Z" and (observed["argv"] != expected_argv or observed["cwd"] != str(ROOT)):
        raise RuntimeError("Adoption process complete argv/working directory mismatch")


def live_trainers(job, original_job):
    """Find own training actors for these jobs; validators do not use a GPU."""
    result = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            item = process(int(entry.name))
        except (PermissionError, RuntimeError):
            continue
        if item is None or item["uid"] != os.getuid() or item["state"] == "Z":
            continue
        argv = item["argv"]
        if not any(Path(arg).name in ("paper_baseline_suite.py", "replication_suite.py") for arg in argv):
            continue
        if "--job" not in argv:
            continue
        index = argv.index("--job")
        if index + 1 >= len(argv) or argv[index + 1] not in (str(job), str(original_job)):
            continue
        if "validate" in argv or "snapshot" in argv:
            continue
        result.append(item)
    return result


@contextlib.contextmanager
def locks(job, original_job):
    import fcntl

    with contextlib.ExitStack() as stack:
        paths = (job / PHASE / "campaign.lock", job.parent / "campaign.lock", original_job.parent / "campaign.lock")
        for path in dict.fromkeys(paths):
            stream = stack.enter_context(path.open("a"))
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise CoordinationBusy(f"Another supervisor still holds {path}; no experiment was started") from error
        yield


def trainer_command(job, source, cache, action, case, quantized=None):
    command = [
        sys.executable,
        "-u",
        str(ROOT / "online_experiments/paper_baseline_suite.py"),
        action,
        "--job",
        str(job),
        "--source",
        str(source),
        "--cache",
        str(cache),
        "--method",
        case["method"],
        "--seed",
        str(case["seed"]),
    ]
    if quantized is not None:
        command += ["--quantized", str(quantized)]
    return command


def quantized_manifest(job, cache, case, *, build):
    if not case["method"].startswith("cache-reuse-"):
        return None
    bits = int(case["method"].rsplit("int", 1)[1])
    pointer = job / f"quantized-int{bits}-location.json"
    destination = Path(read(pointer)["directory"]) if pointer.exists() else job / f"quantized-int{bits}"
    if (destination / "complete.json").is_file():
        return destination / "manifest.json"
    if not build:
        raise RuntimeError("Completed/adopted quantized case is missing its sealed physical cache")
    if destination.exists() and any(destination.iterdir()):
        destination = job / f"quantized-int{bits}-retry-{uuid.uuid4().hex[:8]}"
    write(pointer, {"directory": str(destination)})
    from online_experiments.paper_quantized_cache import build_quantized_cache

    return build_quantized_cache(cache, destination, bits=bits)


def validate(job, source, cache, case):
    if not (job / "train" / case["name"] / "DONE").is_file():
        raise RuntimeError(f"{case['name']} has no DONE marker; process exit is not success")
    quantized = quantized_manifest(job, cache, case, build=False)
    command = trainer_command(job, source, cache, "validate", case, quantized)
    with (job / PHASE / f"validate-{case['name']}.log").open("a") as stream:
        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)


def adopt(args, case, source, cache):
    job, phase = args.job, args.job / PHASE
    expected = trainer_command(job, source, cache, "train", case, quantized_manifest(job, cache, case, build=False))
    record = {
        "pid": args.adopt_pid,
        "starttime": args.adopt_starttime,
        "case": case["name"],
        "expected_argv": expected,
        "expected_uid": os.getuid(),
        "signal_delivery": "none",
    }
    observed, verified = process(args.adopt_pid), False
    while observed is not None:
        require_adopt_identity(observed, expected, args.adopt_starttime)
        if observed["state"] == "Z":
            break
        verified = True
        actors = live_trainers(job, args.original_job)
        if any(item["pid"] != args.adopt_pid for item in actors):
            raise RuntimeError("Another training actor appeared while adopting the seed-zero run")
        summary_path = job / "train" / case["name"] / "summary.json"
        epoch = read(summary_path).get("completed_epochs") if summary_path.exists() else None
        state(
            job,
            "ADOPTING",
            f"{case['name']} epoch={epoch}/100; no seed repeats",
            case=case["name"],
            pid=args.adopt_pid,
            completed_epochs=epoch,
        )
        record.update(verified_live_identity=verified, last_observed_epoch=epoch, last_observed_time_unix=time.time())
        write(phase / "adoption.json", record)
        # Check identity frequently; keep monitoring logs compact.
        for _ in range(max(1, int(30 / args.poll_seconds))):
            time.sleep(args.poll_seconds)
            observed = process(args.adopt_pid)
            if observed is None or observed["state"] == "Z":
                break
            require_adopt_identity(observed, expected, args.adopt_starttime)
        if observed is None or observed["state"] == "Z":
            break
    if observed is not None:
        require_adopt_identity(observed, expected, args.adopt_starttime)
    validate(job, source, cache, case)
    record.update(
        verified_live_identity=verified,
        state="FULL_DONE_VALIDATED",
        validated_time_unix=time.time(),
        success_evidence="DONE plus frozen full-split/checkpoint/prediction/provenance validator",
    )
    write(phase / "adoption.json", record)


def archive_stage(job, name, paths):
    from online_experiments.replication_campaign import archive

    stage = f"dataset-first-{name}"
    marker = job / f"archive-{stage}.json"
    if not marker.exists():
        archive(job, stage, paths, True)
    result = read(marker)
    write(job / PHASE / "archives" / f"{name}.json", result)
    return result


def run(args):
    job, phase = args.job, args.job / PHASE
    phase.mkdir(parents=True, exist_ok=True)
    with locks(job, args.original_job):
        switch = read(job / "original-switch.json")
        if Path(switch["original_job"]).resolve() != args.original_job or switch.get("phase") != "stopped_all":
            raise RuntimeError("Original supervisor switch is incomplete or refers to another original job")
        queue_path = job / "queue.json"
        first, deferred = partition(read(queue_path))
        provenance_path = job / "provenance.json"
        provenance = read(provenance_path)
        source, cache = Path(provenance["source_job"]), Path(provenance["cache_manifest"])
        plan = {
            "schema": "dataset-first-seed-zero-v1",
            "job": str(job),
            "origin_root": str(args.origin_root),
            "original_job": str(args.original_job),
            "original_queue_sha256": digest(queue_path),
            "provenance_sha256": digest(provenance_path),
            "coordinator_sha256": digest(Path(__file__)),
            "source": str(source),
            "cache_manifest": str(cache),
            "cases": first,
            "policy": "VisDrone seed0 first; all independent seed1/2 repeats deferred until datasets finish",
            "next_datasets": [
                {"dataset": name, "training_integration": "pending", "launched": False}
                for name in ("VOC07+12", "SKU-110K", "COCO2017")
            ],
            "no_smoke_or_subset_runs": True,
            "no_process_signals": True,
        }
        plan_path = phase / "plan.json"
        if plan_path.exists() and read(plan_path) != plan:
            raise RuntimeError("Existing dataset-first plan/input identity changed")
        write(plan_path, plan)
        write(
            phase / "deferred-seeds.json",
            {
                "count": len(deferred),
                "cases": deferred,
                "policy": "Deferred, not cancelled; run only after all selected datasets complete seed0 work",
            },
        )
        (phase / "pid.txt").write_text(str(os.getpid()) + "\n", encoding="utf-8")
        write(phase / "worker.json", process(os.getpid()))
        actors = live_trainers(job, args.original_job)
        if any(item["pid"] != args.adopt_pid for item in actors):
            raise RuntimeError("Unexpected existing trainer; refusing to create concurrent GPU work")
        if args.adopt_pid is not None and args.adopt_method not in {case["method"] for case in first}:
            raise ValueError("Adopted method is not an existing seed-zero phase case")
        archive_stage(job, "plan", [plan_path, phase / "deferred-seeds.json"])
        completed = []
        for number, case in enumerate(first, 1):
            state(
                job,
                "RUNNING",
                f"seed0 {number}/{len(first)} {case['name']}; seed1/2={len(deferred)} deferred",
                case=case["name"],
                completed_seed_zero=len(completed),
                planned_seed_zero=len(first),
                deferred_seed_repeats=len(deferred),
            )
            prior_adoption = read(phase / "adoption.json") if (phase / "adoption.json").is_file() else {}
            adoption_complete = (
                prior_adoption.get("state") == "FULL_DONE_VALIDATED"
                and prior_adoption.get("case") == case["name"]
                and prior_adoption.get("pid") == args.adopt_pid
                and prior_adoption.get("starttime") == args.adopt_starttime
            )
            if args.adopt_pid is not None and case["method"] == args.adopt_method and not adoption_complete:
                adopt(args, case, source, cache)
            elif (job / "train" / case["name"] / "DONE").is_file():
                validate(job, source, cache, case)
            else:
                if live_trainers(job, args.original_job):
                    raise RuntimeError("Existing trainer must finish before any next seed-zero case")
                quantized = quantized_manifest(job, cache, case, build=True)
                command = trainer_command(job, source, cache, "train", case, quantized)
                write(phase / f"command-{case['name']}.json", {"argv": command, "cwd": str(ROOT)})
                with (phase / f"train-{case['name']}.log").open("a") as stream:
                    subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
                validate(job, source, cache, case)
            completed.append(job / "train" / case["name"])
            from online_experiments.paper_baseline_campaign import table3_and_resources
            from online_experiments.replication_campaign import check_pairing

            table3_and_resources(job, args.original_job)
            write(phase / "pairing.json", check_pairing(completed))
            output = completed[-1]
            archive_stage(
                job,
                case["name"],
                [
                    output / name
                    for name in (
                        "args.json",
                        "summary.json",
                        "initialization.json",
                        "environment.json",
                        "metrics.json",
                        "official-final.json",
                    )
                ],
            )
            if (
                digest(queue_path) != plan["original_queue_sha256"]
                or digest(provenance_path) != plan["provenance_sha256"]
            ):
                raise RuntimeError("Frozen original queue/provenance changed during external coordination")
        completion = {
            "state": "PHASE_COMPLETE",
            "completed_seed_zero": len(completed),
            "planned_seed_zero": len(first),
            "deferred_seed_repeats": len(deferred),
            "next_action": "READY_FOR_NEXT_DATASET",
            "next_dataset": "VOC07+12",
            "next_dataset_training_integration": "pending",
            "new_dataset_launched": False,
            "original_28_case_queue_complete": False,
        }
        write(phase / "completion.json", completion)
        result = archive_stage(
            job,
            "phase-complete",
            [
                phase / "completion.json",
                phase / "pairing.json",
                plan_path,
                phase / "deferred-seeds.json",
                *sorted((job / "tables").glob("*.csv")),
            ],
        )
        state(
            job,
            "PHASE_COMPLETE",
            "seed0=10/10; seed1/2=18 deferred; READY_FOR_NEXT_DATASET VOC integration pending",
            **{key: value for key, value in completion.items() if key != "state"},
            archive_commit=result["commit"],
            pushed=result["pushed"],
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    monitor = subparsers.add_parser("status")
    monitor.add_argument("--job", type=Path, required=True)
    worker = subparsers.add_parser("run")
    worker.add_argument("--job", type=Path, required=True)
    worker.add_argument("--origin-root", type=Path, required=True)
    worker.add_argument("--original-job", type=Path, required=True)
    worker.add_argument("--adopt-pid", type=int)
    worker.add_argument("--adopt-starttime", type=int)
    worker.add_argument("--adopt-method", default="vit-adapter-cache25")
    worker.add_argument("--poll-seconds", type=float, default=2.0)
    args = parser.parse_args()
    args.job = args.job.resolve()
    if args.action == "status":
        status = args.job / PHASE / "status.json"
        print(json.dumps(read(status), indent=2) if status.is_file() else "Dataset-first phase has not started")
        return
    args.origin_root, args.original_job = args.origin_root.resolve(), args.original_job.resolve()
    if (args.adopt_pid is None) != (args.adopt_starttime is None):
        parser.error("--adopt-pid and --adopt-starttime must be provided together")
    if not 0.1 <= args.poll_seconds <= 30 or (
        args.adopt_pid is not None and (args.adopt_pid <= 0 or args.adopt_starttime <= 0)
    ):
        parser.error("Use a positive adoption PID and polling interval between 0.1 and 30 seconds")
    os.environ.setdefault("HF_HOME", str(args.origin_root.parent / "cache/huggingface"))
    os.environ.setdefault("GH_CONFIG_DIR", str(args.origin_root.parent / ".config/gh"))
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONUNBUFFERED="1",
        CUDA_VISIBLE_DEVICES="0",
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
    )
    try:
        run(args)
    except CoordinationBusy as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from error
    except Exception:
        phase = args.job / PHASE
        phase.mkdir(parents=True, exist_ok=True)
        failure = phase / f"failure-{time.time_ns()}.json"
        previous = read(phase / "status.json") if (phase / "status.json").is_file() else {}
        write(
            failure,
            {
                "traceback": traceback.format_exc(),
                "last_status": previous,
                "policy": "Stop this phase on failure; do not start seed repeats or a new dataset",
            },
        )
        state(
            args.job,
            "FAILED",
            "dataset-first phase stopped; inspect failure JSON and stage logs",
            failure_file=str(failure),
            last_case=previous.get("case"),
        )
        try:
            archive_stage(args.job, failure.stem, [failure])
        except (RuntimeError, OSError, ValueError, subprocess.SubprocessError):
            log(phase, "Failure archive remains pending: " + traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
