"""Resume the five authorized remaining endpoints after the campus reboot.

Only scheduling changes: original plans, commands, datasets, caches and native
checkpoint restoration remain authoritative. One project lane per physical GPU.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from online_experiments import priority_single_seed_campaign as original
from online_experiments import priority_three_lane as shared
from online_experiments.crossdataset_campaign import GPU_LOCK_ROOT, gpu_memory, job_lock

SCHEMA = "campus-reboot-two-gpu-recovery-v1"
LANES = {
    0: (("voc0712", "cache-reuse-c1-int8"), ("voc0712", "vit-adapter-cache25")),
    1: (("voc0712", "loffta-c2-n005"), ("sku110k", "detail25"), ("voc0712", "detail25")),
}
DATASET_JOBS = {
    "voc0712": "job-20261110-voc0712-s0-v2",
    "sku110k": "job-20261110-sku110k-official-loader-v2",
}


def case_args(dataset, dependencies, publish):
    job = ROOT / "runs/paper/priority/job-20261112-single-seed" / dataset
    frozen = original.read(job / "pipeline-plan.json")
    return SimpleNamespace(
        dataset=dataset,
        job=job,
        dataset_job=ROOT / "runs/paper/crossdataset" / DATASET_JOBS[dataset],
        dependencies=dependencies,
        publish=publish,
        first_detector=frozen["methods"][0],
    )


def check_frozen_plan(args):
    frozen = original.read(args.job / "pipeline-plan.json")
    if original.plan(args) != frozen:
        raise RuntimeError("Frozen seven-method plan/source/cache differs; preserve original evidence")
    return frozen


def require_no_output_owner(args, method):
    output = str(original.output_for(args, method))
    owners = [
        worker["pid"]
        for worker in shared.training_processes()
        if "--output" in worker["argv"] and worker["argv"][worker["argv"].index("--output") + 1] == output
    ]
    if owners:
        raise RuntimeError(f"Live training already owns exact output {output}: {owners}")


def validate_checkpoint_state(state, config):
    epoch, rows, summary = state.get("epoch"), state.get("rows"), state.get("summary", {})
    if (
        state.get("args") != config
        or not isinstance(epoch, int)
        or not isinstance(rows, list)
        or not 0 <= epoch <= 100
        or epoch != len(rows)
        or [row.get("epoch") for row in rows] != list(range(1, epoch + 1))
        or summary.get("completed_epochs") != epoch
        or summary.get("epochs") != rows
        or summary.get("provenance_sha256") != config.get("provenance_sha256")
    ):
        raise RuntimeError("Native checkpoint config/epoch/summary evidence differs")


def checkpoint_record(args, method):
    output = original.output_for(args, method)
    checkpoint = output / "weights/last.pt"
    if not checkpoint.is_file():
        if (output / "args.json").exists():
            raise RuntimeError("Partial output has config but no native checkpoint; inspect before restarting")
        return {"mode": "NEW", "checkpoint_epoch": 0}
    import torch

    config = original.read(output / "args.json")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint_state(state, config)
    required = {
        "model",
        "optimizer",
        "scheduler",
        "totals",
        "python_rng",
        "numpy_rng",
        "cpu_rng",
        "cuda_rng",
        "query_images",
        "query_calls",
    }
    if not required.issubset(state):
        raise RuntimeError("Native checkpoint lacks full optimizer/scheduler/RNG/query state")
    return {
        "mode": "NATIVE_RESUME",
        "checkpoint_epoch": state["epoch"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": original.sha(checkpoint),
    }


def done(args, method):
    return (original.output_for(args, method) / ("DONE.json" if method in original.DIRECT else "DONE")).is_file()


def record_complete(args, method, archive):
    with original.archive_lock():
        path = args.job / "completed-methods.json"
        rows = original.read(path) if path.exists() else []
        rows = [row for row in rows if row["method"] != method]
        rows.append({"method": method, "output": str(original.output_for(args, method)), "archive": archive})
        order = list(original.ordered_methods(args))
        rows.sort(key=lambda row: order.index(row["method"]))
        original.write(path, rows)


@contextlib.contextmanager
def try_gpu_lease(gpu, min_free_mib, *, gpu_free=gpu_memory, host_free=shared.host_memory):
    """Slot0 only; any occupied historical slot rejects admission on this GPU."""
    import fcntl

    GPU_LOCK_ROOT.mkdir(parents=True, exist_ok=True)
    streams, selected = [], None
    try:
        with (GPU_LOCK_ROOT / f"gpu{gpu}-stages.lock").open("a") as gate:
            try:
                fcntl.flock(gate.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                occupied = False
                for slot in range(4):
                    stream = (GPU_LOCK_ROOT / f"gpu{gpu}-slot{slot}.lock").open("a")
                    streams.append(stream)
                    try:
                        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        occupied = True
                if not occupied and gpu_free(gpu) >= min_free_mib and host_free() >= 12288:
                    selected = streams[0]
        for stream in streams:
            if stream is not selected:
                stream.close()
        yield selected.fileno() if selected is not None else None
    finally:
        for stream in streams:
            stream.close()


@contextlib.contextmanager
def output_lock(output):
    """Inherited by the child, so an orphan remains protected after parent loss."""
    import fcntl

    output.mkdir(parents=True, exist_ok=True)
    with (output / "reboot-recovery.lock").open("a") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("A recovery child owns this exact output lock") from error
        yield stream.fileno()


class State:
    def __init__(self, args, cases):
        self.args, self.cases = args, cases
        self.lock, self.lanes, self.errors = threading.RLock(), {}, []
        self.boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()

    def update(self, gpu, state, **extra):
        with self.lock:
            self.lanes[str(gpu)] = {"gpu": gpu, "state": state, "time_unix": time.time(), **extra}
            self.emit()

    def emit(self):
        record = {
            "schema": SCHEMA,
            "state": "FAILED" if self.errors else "RUNNING",
            "pid": os.getpid(),
            "boot_id": self.boot_id,
            "time_unix": time.time(),
            "lanes": self.lanes,
            "errors": self.errors,
        }
        original.write(self.args.job / "status.json", record)
        for dataset, args in self.cases.items():
            completed = original.read(args.job / "completed-methods.json")
            relevant = {key: row for key, row in self.lanes.items() if row.get("dataset") == dataset}
            active = [row for row in relevant.values() if row["state"] not in {"COMPLETED", "FAILED"}]
            failures = [row for row in self.errors if row["dataset"] == dataset]
            status = "FAILED" if failures else ("SUCCEEDED" if len(completed) == 7 else "RUNNING_RECOVERY")
            original.status(
                args,
                status,
                "Campus reboot recovery; original frozen training recipes",
                completed=len(completed),
                planned=7,
                method=active[0].get("method") if active else None,
                lanes=relevant,
                recovery_job=str(self.args.job),
            )


def child_environment(args, gpu):
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
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
    return env


def execute(args, action, method, gpu, state, inherited_fds):
    require_no_output_owner(args, method)
    command = original.command_for(args, action, method)
    label = f"{action}-{method}"
    record = {"argv": command, "cwd": str(ROOT), "CUDA_VISIBLE_DEVICES": str(gpu), "boot_id": state.boot_id}
    original.write(state.args.job / "commands" / f"{args.dataset}-{label}.json", record)
    with (args.job / f"{label}.log").open("a", encoding="utf-8") as stream:
        child = subprocess.Popen(
            command,
            cwd=ROOT,
            env=child_environment(args, gpu),
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            pass_fds=tuple(inherited_fds),
        )
        record.update(pid=child.pid, action=action, method=method, started_unix=time.time())
        original.write(args.job / "active-child.json", record)
        while True:
            state.update(
                gpu,
                action.upper(),
                dataset=args.dataset,
                method=method,
                child_pid=child.pid,
                output=str(original.output_for(args, method)),
            )
            try:
                result = child.wait(timeout=10)
                break
            except subprocess.TimeoutExpired:
                continue
    original.write(
        state.args.job / "children" / f"{args.dataset}-{label}.json",
        dict(record, returncode=result, finished_unix=time.time()),
    )
    if result:
        raise RuntimeError(f"{args.dataset}/{label} exited {result}; preserve checkpoint, no automatic restart")


def run_case(args, method, gpu, state):
    require_no_output_owner(args, method)
    with output_lock(original.output_for(args, method)) as output_fd:
        if not done(args, method):
            resume = checkpoint_record(args, method)
            resume.update(
                dataset=args.dataset,
                method=method,
                gpu=gpu,
                boot_id=state.boot_id,
                time_unix=time.time(),
                pipeline_plan_sha256=original.sha(args.job / "pipeline-plan.json"),
            )
            resume["gpu_runtime"] = subprocess.check_output(
                ["nvidia-smi", "-i", str(gpu), "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
                text=True,
                timeout=15,
            ).strip()
            original.write(state.args.job / "resume-records" / f"{args.dataset}-{method}-{time.time_ns()}.json", resume)
            while True:
                with try_gpu_lease(gpu, state.args.min_free_mib) as lease_fd:
                    if lease_fd is not None:
                        execute(args, "train", method, gpu, state, (output_fd, lease_fd))
                        execute(args, "validate", method, gpu, state, (output_fd, lease_fd))
                        break
                state.update(gpu, "WAITING_GPU", dataset=args.dataset, method=method)
                time.sleep(10)
        else:
            execute(args, "validate", method, gpu, state, (output_fd,))
        if not done(args, method):
            raise RuntimeError("Native validator returned without final completion marker")
        state.update(gpu, "ARCHIVING", dataset=args.dataset, method=method)
        try:
            archive = original.archive_case(args, method)
        except Exception:  # noqa: BLE001 - preserve finished training when publishing fails
            archive = {"pushed": False, "archive_pending": True, "error": traceback.format_exc()}
            original.write(state.args.job / "pending-archives" / f"{args.dataset}-{method}.json", archive)
        record_complete(args, method, archive)
    state.update(gpu, "COMPLETED", dataset=args.dataset, method=method, archive=archive)


def run_lane(gpu, state):
    for dataset, method in LANES[gpu]:
        try:
            run_case(state.cases[dataset], method, gpu, state)
        except Exception:  # noqa: BLE001 - record one lane's failure while the other continues
            error = {
                "gpu": gpu,
                "dataset": dataset,
                "method": method,
                "traceback": traceback.format_exc(),
                "time_unix": time.time(),
            }
            with state.lock:
                state.errors.append(error)
                original.write(state.args.job / f"failure-gpu{gpu}-{time.time_ns()}.json", error)
                state.update(gpu, "FAILED", dataset=dataset, method=method)
            return


def run(args):
    if sys.platform != "linux":
        raise RuntimeError("Recovery requires Linux process ownership and flock")
    args.job.mkdir(parents=True, exist_ok=True)
    cases = {dataset: case_args(dataset, args.dependencies, args.publish) for dataset in DATASET_JOBS}
    with contextlib.ExitStack() as stack:
        stack.enter_context(job_lock(args.job))
        for case in cases.values():
            stack.enter_context(job_lock(case.job))
            if original.live_output_workers(case):
                raise RuntimeError("A live exact-output worker exists; no process is signaled")
        frozen = {dataset: check_frozen_plan(case) for dataset, case in cases.items()}
        plan = {
            "schema": SCHEMA,
            "lanes": {str(key): value for key, value in LANES.items()},
            "canonical_plan_sha256": {
                dataset: original.sha(case.job / "pipeline-plan.json") for dataset, case in cases.items()
            },
            "original_contracts": frozen,
            "gpu_min_free_mib": args.min_free_mib,
            "host_min_available_mib": 12288,
            "concurrency": "one project worker per physical GPU; GPUs 0 and 1",
            "training_changes": [],
            "timing": "Reboot/driver/GPU mapping changes recorded per resume; descriptive mixed-condition timings",
        }
        plan = json.loads(json.dumps(plan))
        path = args.job / "plan.json"
        if path.exists() and original.read(path) != plan:
            raise RuntimeError("Recovery plan changed; preserve this recovery directory")
        original.write(path, plan)
        if args.action == "check":
            print(json.dumps(plan, indent=2))
            return
        state = State(args, cases)
        original.write(
            args.job / "worker.json",
            {
                "schema": SCHEMA,
                "pid": os.getpid(),
                "boot_id": state.boot_id,
                "argv": sys.argv,
                "time_unix": time.time(),
            },
        )
        for dataset, case in cases.items():
            prior = case.job / "worker.json"
            saved = args.job / f"original-{dataset}-worker.json"
            if prior.is_file() and not saved.exists():
                original.write(saved, original.read(prior))
            original.write(
                prior,
                {
                    "schema": SCHEMA,
                    "scheduler": SCHEMA,
                    "pid": os.getpid(),
                    "boot_id": state.boot_id,
                    "argv": sys.argv,
                    "recovery_job": str(args.job),
                },
            )
        threads = [threading.Thread(target=run_lane, args=(gpu, state), name=f"gpu{gpu}") for gpu in LANES]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        state.emit()
        rows = [row for case in cases.values() for row in original.read(case.job / "completed-methods.json")]
        all_done = len(rows) == 14
        all_pushed = all(row["archive"].get("pushed") for row in rows)
        completion = {
            "state": "FAILED"
            if state.errors
            else ("SUCCEEDED" if all_done and all_pushed else "COMPLETED_PENDING_ARCHIVE"),
            "completed": len(rows),
            "planned": 14,
            "all_archives_pushed": all_pushed,
            "time_unix": time.time(),
            "errors": state.errors,
        }
        original.write(args.job / "completion.json", completion)
        original.write(
            args.job / "status.json",
            dict(completion, schema=SCHEMA, pid=os.getpid(), boot_id=state.boot_id, lanes=state.lanes),
        )
        if state.errors:
            raise RuntimeError("One or more recovery lanes failed; inspect explicit failure records")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "check", "status"))
    parser.add_argument("--job", type=Path, default=ROOT / "runs/paper/reboot-recovery/job-20261116-campus-two-gpu")
    parser.add_argument("--dependencies", type=Path, default=ROOT / ".research-deps/evidence-20261110/site")
    parser.add_argument("--min-free-mib", type=int, default=12288)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    args.job, args.dependencies = args.job.resolve(), args.dependencies.resolve()
    if args.action == "status":
        print(json.dumps(original.read(args.job / "status.json"), indent=2))
        return
    if args.min_free_mib < 12288:
        parser.error("Recovery retains at least 12288 MiB GPU admission")
    run(args)


if __name__ == "__main__":
    main()
