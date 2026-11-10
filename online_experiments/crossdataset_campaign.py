"""Run full VOC/SKU seed-zero pipelines with healthy-GPU admission and explicit evidence."""

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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
GPU_LOCK_ROOT = ROOT / "runs/paper/crossdataset"
METHODS = (
    "cache1",
    "cache2",
    "mix25",
    "online100",
    "frofa-c2-d01",
    "loffta-c2-n005",
    "cache-reuse-c1-int8",
    "vit-adapter-cache25",
    "transport25",
    "blank25",
    "detail25",
    "lowpass25",
)


class Busy(RuntimeError):
    """A different pipeline already owns this job."""


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def status(job, state, message, **extra):
    record = {"state": state, "message": message, "time_unix": time.time(), **extra}
    write(job / "pipeline-status.json", record)
    temporary = job / "status.txt.tmp"
    temporary.write_text(f"{state}: {message}\n", encoding="utf-8")
    temporary.replace(job / "status.txt")
    with (job / "pipeline.log").open("a", encoding="utf-8") as stream:
        stream.write(time.strftime("%Y-%m-%dT%H:%M:%S%z ") + state + ": " + message + "\n")


@contextlib.contextmanager
def job_lock(job):
    import fcntl

    with (job / "pipeline.lock").open("a") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise Busy("Another pipeline owns this job; existing status is preserved") from error
        yield


def gpu_memory(index):
    command = ["nvidia-smi", "-i", str(index), "--query-gpu=memory.free,pstate", "--format=csv,noheader,nounits"]
    output = subprocess.check_output(command, text=True, timeout=15).strip()
    fields = [field.strip() for field in output.split(",")]
    if len(fields) != 2 or not fields[0].isdigit() or not fields[1].startswith("P"):
        raise RuntimeError("Selected GPU is unavailable or requires reset: " + output)
    return int(fields[0])


def live_children(job):
    """Refuse duplicate cache/training workers after a supervisor exits unexpectedly."""
    result = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            arguments = (entry / "cmdline").read_bytes().decode(errors="surrogateescape").split("\0")
            if not any(Path(value).name == "crossdataset_suite.py" for value in arguments):
                continue
            if "--job" not in arguments:
                continue
            position = arguments.index("--job")
            if position + 1 < len(arguments) and arguments[position + 1] == str(job):
                result.append({"pid": int(entry.name), "argv": arguments[:-1]})
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return result


@contextlib.contextmanager
def gpu_admission(args, message):
    """Admit at most two new stages, reserving extra memory for the second lease."""
    import fcntl

    GPU_LOCK_ROOT.mkdir(parents=True, exist_ok=True)
    lock = GPU_LOCK_ROOT / f"gpu{args.gpu}-stages.lock"
    stream = None
    while stream is None:
        candidates, selected, admitted = [], None, False
        state_message = f"{message}; waiting for GPU{args.gpu} admission gate"
        try:
            with lock.open("a") as gate:
                try:
                    fcntl.flock(gate.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                except BlockingIOError:
                    acquired = False
                if acquired:
                    busy = 0
                    for number in range(2):
                        candidate = (GPU_LOCK_ROOT / f"gpu{args.gpu}-slot{number}.lock").open("a")
                        candidates.append(candidate)
                        try:
                            fcntl.flock(candidate.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            busy += 1
                        else:
                            if selected is None:
                                selected = candidate
                            else:
                                fcntl.flock(candidate.fileno(), fcntl.LOCK_UN)
                    if selected is not None:
                        free = gpu_memory(args.gpu)
                        required = args.min_free_mib + (8192 if busy else 0)
                        if free >= required:
                            admitted, stream = True, selected
                        state_message = f"{message}; GPU{args.gpu} free={free}MiB, need>={required}MiB"
                    else:
                        state_message = f"{message}; both new GPU{args.gpu} stage slots are occupied"
        finally:
            for candidate in candidates:
                if not admitted or candidate is not selected:
                    candidate.close()
        if stream is None:
            status(args.job, "WAITING_GPU", state_message, dataset=args.dataset)
            time.sleep(30)
    try:
        args.gpu_lock_fd = stream.fileno()
        yield free
    finally:
        del args.gpu_lock_fd
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def execute(args, action, *, method=None):
    command = [
        sys.executable,
        "-u",
        str(ROOT / "online_experiments/crossdataset_suite.py"),
        action,
        "--job",
        str(args.job),
    ]
    if action == "snapshot":
        command += ["--manifest", str(args.data_root / "dataset-manifest.json"), "--identity", str(args.identity)]
    elif action == "cache":
        command += ["--build-batch", "8"]
    if method is not None:
        command += ["--method", method, "--seed", "0"]
    label = action + ("-" + method if method else "")
    write(args.job / "commands" / f"{label}.json", {"argv": command, "cwd": str(ROOT)})
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HOME=str(ROOT.parent / "cache/huggingface"),
        GH_CONFIG_DIR=str(ROOT.parent / ".config/gh"),
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        PYTHONUNBUFFERED="1",
        TMPDIR=str(ROOT.parent / "tmp/paper-baselines"),
    )
    env["PYTHONPATH"] = str(args.dependencies) + os.pathsep + str(ROOT)
    with (args.job / f"{label}.log").open("a") as stream:
        child = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            pass_fds=(args.gpu_lock_fd,) if hasattr(args, "gpu_lock_fd") else (),
        )
        write(
            args.job / "active-child.json",
            {"pid": child.pid, "action": action, "method": method, "started_unix": time.time(), "command": command},
        )
        result = child.wait()
    write(
        args.job / "last-child.json",
        {"pid": child.pid, "action": action, "method": method, "returncode": result, "finished_unix": time.time()},
    )
    if result:
        raise RuntimeError(f"{label} exited{result}; preserve log/checkpoint, no automatic retry")


def archive_case(args, method):
    from online_experiments.archive_assignment_intervention import commit_files
    from online_experiments.replication_campaign import push, sanitized

    stage = method + "-s0"
    marker_file = args.job / "archives" / f"{stage}.json"
    if marker_file.exists():
        return read(marker_file)
    output = args.job / "train" / stage
    destination = ROOT / "research/experiments/crossdataset" / args.job.name / stage
    destination.mkdir(parents=True, exist_ok=True)
    names = []
    for name in (
        "args.json",
        "summary.json",
        "metrics.json",
        "initialization.json",
        "environment.json",
        "endpoint.json",
    ):
        path = output / name
        if not path.is_file():
            raise RuntimeError("Completed method is missing required archive evidence: " + name)
        target = destination / name
        content = json.dumps(sanitized(read(path)), indent=2, allow_nan=False) + "\n"
        if target.exists() and target.read_text() != content:
            raise RuntimeError("Refuse changed archived result: " + str(target))
        target.write_text(content, encoding="utf-8")
        names.append(target.relative_to(ROOT).as_posix())
    readme = destination / "README.md"
    readme.write_text(
        f"# {args.dataset}: {stage}\n\nFull official splits,100 epochs,batch8,seed0. "
        "Fixed final checkpoint; evaluator/budget/implementation in args and score records. "
        "Related-paper methods are detection idea adaptations. SharedGPU timing is observational.\n",
        encoding="utf-8",
    )
    names.append(readme.relative_to(ROOT).as_posix())
    marker = f"crossdataset:{args.job.name}:{stage}"
    block = f"\n\n<!-- {marker} -->\n## {args.dataset} full seed0: {method}\n\n"
    block += f"Archived [{stage}]({readme.relative_to(ROOT / 'research').as_posix()}); final epoch,explicit metric.\n"
    commit = commit_files(
        ROOT,
        names,
        block,
        marker,
        readme.relative_to(ROOT).as_posix(),
        f"research: record {args.dataset} {stage} full endpoint",
    )
    result = {"commit": commit, "pushed": False}
    if args.publish:
        try:
            result.update(pushed=True, pushed_head=push())
        except (RuntimeError, OSError, subprocess.SubprocessError) as error:
            result["push_error"] = str(error)
    write(marker_file, result)
    return result


def run(args):
    args.job.mkdir(parents=True, exist_ok=True)
    with job_lock(args.job):
        if live_children(args.job):
            raise Busy("A cache/training/validation child is still running for this job; inspect it before restarting")
        plan = {
            "schema": "crossdataset-seed-zero-plan-v1",
            "dataset": args.dataset,
            "data_root": str(args.data_root),
            "identity_sha256": sha(args.identity),
            "methods": list(METHODS),
            "seed": 0,
            "epochs": 100,
            "batch": 8,
            "coordinator_sha256": sha(Path(__file__)),
            "gpu_index": args.gpu,
            "gpu_stages": "up to two admitted new stages; second requires16GiB free; VisDrone preserved; noGPU1reset",
            "no_smoke_or_subset": True,
            "test_selection": "fixed final epoch only",
            "deferred_extensions": ["views25", "vit-adapter-cache0", "cache-reuse-c1-int4", "seed1", "seed2"],
            "existing_method_unchanged": True,
        }
        if (args.job / "pipeline-plan.json").exists() and read(args.job / "pipeline-plan.json") != plan:
            raise RuntimeError("Pipeline plan changed; use a new job for a changed experiment")
        write(args.job / "pipeline-plan.json", plan)
        write(
            args.job / "worker.json",
            {
                "pid": os.getpid(),
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "argv": sys.argv,
                "started_unix": time.time(),
            },
        )
        while not (args.data_root / "complete.json").is_file():
            preparation = read(args.data_root / "status.json") if (args.data_root / "status.json").exists() else {}
            if str(preparation.get("state", preparation.get("status", ""))).startswith("FAILED"):
                raise RuntimeError("Dataset preparation failed; inspect its own failure record")
            status(
                args.job,
                "WAITING_DATA",
                f"{args.dataset}: full data and immutable seal pending",
                dataset=args.dataset,
                preparation=preparation,
            )
            time.sleep(30)
        seal = read(args.data_root / "complete.json")
        if seal["dataset"] != args.dataset or sha(args.data_root / "dataset-manifest.json") != seal["manifest_sha256"]:
            raise RuntimeError("Dataset seal/name/manifest mismatch")
        status(args.job, "SNAPSHOT", "Seal source, full splits and fixed recipe", dataset=args.dataset)
        execute(args, "snapshot")
        with gpu_admission(args, "Build complete C2/test/blank bank"):
            status(args.job, "BUILDING_CACHE", "Full bank; memory-mapped, no subset run", dataset=args.dataset)
            execute(args, "cache")
        completed = []
        for index, method in enumerate(METHODS, 1):
            output = args.job / "train" / f"{method}-s0"
            if not (output / "DONE").exists():
                with gpu_admission(args, f"{method} seed0 {index}/{len(METHODS)}"):
                    status(
                        args.job,
                        "TRAINING",
                        f"{method}:100 epochs,current recipe",
                        dataset=args.dataset,
                        method=method,
                        completed=len(completed),
                        planned=len(METHODS),
                    )
                    execute(args, "train", method=method)
            status(args.job, "VALIDATING", method, dataset=args.dataset, method=method)
            execute(args, "validate", method=method)
            if not (output / "DONE").is_file():
                raise RuntimeError("Validator returned without a complete DONE endpoint")
            result = archive_case(args, method)
            completed.append({"method": method, "archive": result})
            write(args.job / "completed-methods.json", completed)
        status(
            args.job,
            "SUCCEEDED",
            f"{args.dataset}:12 full seed0 cases validated; repeats deferred",
            dataset=args.dataset,
            completed=len(completed),
            planned=len(METHODS),
            all_archives_pushed=all(row["archive"]["pushed"] for row in completed),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "status"))
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--dataset", choices=("voc0712", "sku110k"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--identity", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--min-free-mib", type=int, default=8192)
    parser.add_argument("--dependencies", type=Path, default=ROOT / ".research-deps/evidence-20261110/site")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    args.job = args.job.resolve()
    if args.action == "status":
        print(json.dumps(read(args.job / "pipeline-status.json"), indent=2))
        return
    if args.dataset is None or args.data_root is None or args.identity is None:
        parser.error("run requires --dataset,--data-root,--identity")
    if args.gpu < 0 or args.min_free_mib < 8192:
        parser.error("Use a healthy nonnegative GPU index and at least8192MiB admission reserve")
    args.data_root, args.identity = args.data_root.resolve(), args.identity.resolve()
    try:
        run(args)
    except Busy as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from error
    except Exception:
        failure = {
            "traceback": traceback.format_exc(),
            "time_unix": time.time(),
            "policy": "Preserve all data,logs and checkpoints; no automatic retry or method change",
        }
        write(args.job / f"failure-{time.time_ns()}.json", failure)
        status(args.job, "FAILED", "Inspect failure record and stage log; no success inferred from process exit")
        raise


if __name__ == "__main__":
    main()
