"""Adopt the VOC supervisor and overlap its unchanged FroFA with cached LOFF-TA.

Only one extra slot (3), at most four project training workers. Frozen trainers,
datasets, output paths and training recipes remain unchanged. Linux only.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from online_experiments import priority_single_seed_campaign as original
from online_experiments import priority_three_lane as shared
from online_experiments.crossdataset_campaign import GPU_LOCK_ROOT, gpu_admission, gpu_memory, job_lock

FROFA, LOFF = "frofa-c2-d01", "loffta-c2-n005"
SCHEMA = "priority-voc-overlap-v1"


def check_pending(args, child):
    completed = original.read(args.job / "completed-methods.json")
    if [item["method"] for item in completed] != list(original.ordered_methods(args)[:2]):
        raise RuntimeError("Expected exactly the two completed native VOC detectors")
    if any(not (original.output_for(args, item["method"]) / "DONE.json").is_file() for item in completed):
        raise RuntimeError("Completed detector endpoint marker missing")
    for method in original.ordered_methods(args)[3:]:
        if original.output_for(args, method).exists():
            raise RuntimeError("Refuse existing pending output: " + method)
    expected = str(original.output_for(args, FROFA))
    found = [
        p["pid"]
        for p in shared.training_processes()
        if "--output" in p["argv"] and p["argv"][p["argv"].index("--output") + 1] == expected
    ]
    if found != [child["pid"]]:
        raise RuntimeError("Adopted FroFA output must have exactly one owner")


def check(args):
    if args.dataset != "voc0712" or sys.platform != "linux":
        raise RuntimeError("VOC Linux scheduler required")
    shared.pidfd_capability()
    if original.plan(args) != original.read(args.job / "pipeline-plan.json"):
        raise RuntimeError("Frozen VOC source/data plan changed")
    metadata, active = original.read(args.job / "worker.json"), original.read(args.job / "active-child.json")
    old, child = shared.process(args.old_supervisor), shared.process(args.adopt_child)
    if old is None or child is None or metadata["pid"] != old["pid"] or active["pid"] != child["pid"]:
        raise RuntimeError("Exact old supervisor and adopted child required")
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if metadata["boot_id"] != boot or old["uid"] != os.getuid() or child["uid"] != os.getuid():
        raise RuntimeError("Process boot or ownership differs")
    if not shared.supervisor_argv(old["argv"], metadata["argv"]) or old["argv"][old["argv"].index("--job") + 1] != str(
        args.job
    ):
        raise RuntimeError("Old supervisor is not this canonical VOC queue")
    if child["ppid"] != old["pid"] or active["action"] != "train" or active["method"] != FROFA:
        raise RuntimeError("Only live FroFA child can be adopted")
    if child["argv"] != original.command_for(args, "train", FROFA) or active["argv"] != child["argv"]:
        raise RuntimeError("FroFA command differs from frozen command")
    if "Z" in (old["state"], child["state"]):
        raise RuntimeError("A handoff process already exited")
    check_pending(args, child)
    return {
        "supervisor": old,
        "child": child,
        "boot_id": boot,
        "plan_sha256": original.sha(args.job / "pipeline-plan.json"),
        "rollback_argv": [sys.executable, *metadata["argv"]],
        "policy": "Signal exact supervisor only, never child/group; no automatic retries",
    }


def check_recovery(args):
    if args.dataset != "voc0712" or sys.platform != "linux":
        raise RuntimeError("VOC Linux recovery required")
    if original.plan(args) != original.read(args.job / "pipeline-plan.json"):
        raise RuntimeError("Frozen VOC plan changed")
    evidence, previous = original.read(args.sidecar / "handoff.json"), original.read(args.sidecar / "plan.json")
    digest = original.sha(args.job / "pipeline-plan.json")
    if (
        evidence["plan_sha256"] != digest
        or previous["frozen_pipeline_plan_sha256"] != digest
        or previous["schema"] != SCHEMA
    ):
        raise RuntimeError("Recovery handoff does not match sealed VOC plan")
    if evidence["supervisor"]["pid"] != args.old_supervisor or evidence["child"]["pid"] != args.adopt_child:
        raise RuntimeError("Recovery PID arguments differ from persisted handoff")
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if evidence["boot_id"] != boot or evidence["child"]["uid"] != os.getuid():
        raise RuntimeError("Recovery boot/owner changed")
    shared.verify_recovery_processes(evidence)
    metadata, active = original.read(args.job / "worker.json"), original.read(args.job / "active-child.json")
    if metadata["pid"] == args.old_supervisor:
        if not shared.supervisor_argv(evidence["supervisor"]["argv"], metadata["argv"]):
            raise RuntimeError("Old worker metadata differs from persisted handoff")
    elif metadata.get("scheduler") != SCHEMA or metadata.get("handoff_file") != str(args.sidecar / "handoff.json"):
        raise RuntimeError("Canonical queue belongs to another supervisor")
    elif (current := shared.process(metadata["pid"])) is not None and current.get("state") != "Z":
        raise RuntimeError("Adopting supervisor is still alive")
    child = evidence["child"]
    if active["pid"] != child["pid"] or active["action"] != "train" or active["method"] != FROFA:
        raise RuntimeError("Canonical active child differs from persisted FroFA")
    if active["argv"] != child["argv"] or child["argv"] != original.command_for(args, "train", FROFA):
        raise RuntimeError("Recovery child command changed")
    check_pending(args, child)
    return evidence


def scheduler_plan(args, evidence):
    return {
        "schema": SCHEMA,
        "dataset": args.dataset,
        "job": str(args.job),
        "frozen_pipeline_plan_sha256": evidence["plan_sha256"],
        "adopted_child": evidence["child"],
        "methods": {"adopt": FROFA, "extra": LOFF, "remaining": list(original.ordered_methods(args)[4:])},
        "epochs": 100,
        "batch": 8,
        "imgsz": 640,
        "seed": 0,
        "gpu": 0,
        "extra_admission": {
            "slot": 3,
            "free_gpu_mib": 12288,
            "available_host_mib": 18432,
            "max_our_training_processes": 4,
        },
        "source_sha256": {
            name: original.sha(ROOT / name)
            for name in (
                "online_experiments/priority_voc_overlap.py",
                "online_experiments/test_priority_voc_overlap.py",
                "online_experiments/priority_three_lane.py",
            )
        },
        "policy": "Scheduling only; unchanged frozen recipes; preserve workers/results, no retry; shared timings observational",
    }


def wait_adopted(args, child):
    while (current := shared.process(child["pid"])) is not None and current.get("state") != "Z":
        shared.same_lifetime(child, current)
        if current.get("argv") != [""]:
            shared.same_process(child, current)
        time.sleep(10)
    if not (original.output_for(args, FROFA) / "DONE").is_file():
        raise RuntimeError("Adopted FroFA exited without sealed DONE; no restart")


def can_admit(occupied, workers, free_mib, host_mib):
    return occupied < 4 and workers < 4 and free_mib >= 12288 and host_mib >= 18432


@contextlib.contextmanager
def extra_lease(args, state):
    import fcntl

    GPU_LOCK_ROOT.mkdir(parents=True, exist_ok=True)
    selected = None
    while selected is None:
        candidates = []
        with (GPU_LOCK_ROOT / "gpu0-stages.lock").open("a") as gate:
            try:
                fcntl.flock(gate.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                busy, available = 0, {}
                for number in range(4):
                    stream = (GPU_LOCK_ROOT / f"gpu0-slot{number}.lock").open("a")
                    candidates.append(stream)
                    try:
                        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        busy += 1
                    else:
                        available[number] = stream
                if 3 in available and can_admit(
                    busy, len(shared.training_processes()), gpu_memory(0), shared.host_memory()
                ):
                    selected = available[3]
        for stream in candidates:
            if stream is not selected:
                stream.close()
        if selected is None:
            state.update("extra", "WAITING_RESOURCES", method=LOFF)
            time.sleep(15)
    try:
        args.gpu_lock_fd = selected.fileno()
        yield
    finally:
        del args.gpu_lock_fd
        selected.close()


class State(shared.State):
    def __init__(self, args):
        super().__init__(args)
        self.overlap_done = threading.Event()

    def update(self, lane, state, **extra):
        with self.lock:
            self.lanes[lane] = {"state": state, "time_unix": time.time(), **extra}
            original.write(
                self.args.sidecar / "status.json",
                {
                    "schema": SCHEMA,
                    "pid": os.getpid(),
                    "lanes": self.lanes,
                    "errors": self.errors,
                    "timing": "Shared GPU; observational only",
                },
            )
            active = [
                value["method"]
                for value in self.lanes.values()
                if value.get("method") and value["state"] not in ("COMPLETED", "FAILED")
            ]
            original.status(
                self.args,
                "PARTIAL_FAILURE" if self.errors else "RUNNING_VOC_OVERLAP",
                "VOC adopted FroFA + parallel cached LOFF-TA",
                lanes=self.lanes,
                completed=len(self.completed),
                planned=7,
                sidecar=str(self.args.sidecar),
                method=self.lanes.get("baseline", {}).get("method"),
                parallel_methods=active,
            )


def finish(args, method, state, lane):
    shared.execute(args, "validate", method, state, lane)
    if not (original.output_for(args, method) / "DONE").is_file():
        raise RuntimeError("Validator lacks sealed DONE")
    state.complete(method, original.archive_case(args, method))


def training_log_alias(args, method):
    """Expose the complete sidecar log to the unchanged CPU metrics watcher."""
    target = args.sidecar / f"train-{method}.log"
    alias = args.job / f"train-{method}.log"
    if alias.is_symlink():
        if alias.resolve() != target.resolve():
            raise RuntimeError("Existing canonical log has a different target: " + str(alias))
    elif alias.exists():
        raise RuntimeError("Refuse overwriting existing canonical training log: " + str(alias))
    else:
        alias.symlink_to(target)


def train_one(args, method, lane, state):
    admission = extra_lease(args, state) if lane == "extra" else gpu_admission(args, "VOC " + method)
    with admission:
        training_log_alias(args, method)
        shared.execute(args, "train", method, state, lane)
    finish(args, method, state, lane)
    state.update(lane, "COMPLETED", method=method)


def baseline(args, child, state):
    state.update("baseline", "ADOPTED_TRAINING", method=FROFA, child_pid=child["pid"])
    wait_adopted(args, child)
    finish(args, FROFA, state, "baseline")
    state.update("baseline", "WAITING_OVERLAP", method=FROFA)
    state.overlap_done.wait()
    # LOFF was executed exactly once by the extra lane, even if it failed.
    # A failed extra lane must not prevent the other untouched methods running.
    for method in original.ordered_methods(args)[4:]:
        train_one(args, method, "baseline", state)


def run_thread(args, lane, evidence, state):
    try:
        if lane == "baseline":
            baseline(args, evidence["child"], state)
        else:
            train_one(args, LOFF, lane, state)
    except Exception as error:  # noqa: BLE001 - preserve every lane failure while the other lane continues.
        with state.lock:
            state.errors.append({"lane": lane, "traceback": traceback.format_exc()})
        state.update(lane, "FAILED", error=str(error), policy="Other lane continues; preserve output, no retry")
    finally:
        if lane == "extra":
            state.overlap_done.set()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "run", "status"))
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--dataset-job", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--old-supervisor", type=int, default=644301)
    parser.add_argument("--adopt-child", type=int, default=1467410)
    parser.add_argument("--dependencies", type=Path, default=ROOT / ".research-deps/evidence-20261110/site")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--recover", action="store_true", help="Persisted handoff only; never signal")
    args = parser.parse_args()
    args.job, args.dataset_job, args.sidecar, args.dependencies = (
        getattr(args, key).resolve() for key in ("job", "dataset_job", "sidecar", "dependencies")
    )
    args.dataset, args.first_detector, args.gpu, args.min_free_mib = "voc0712", "master-n", 0, 8192
    if args.action == "status":
        print(json.dumps(original.read(args.sidecar / "status.json"), indent=2))
        return
    checker = check_recovery if args.recover else check
    evidence = checker(args)
    if args.action == "check":
        print(json.dumps(evidence, indent=2))
        return
    args.sidecar.mkdir(parents=True, exist_ok=True)
    with job_lock(args.sidecar):
        evidence = checker(args)
        original.write(
            args.sidecar / ("recovery-plan.json" if args.recover else "plan.json"), scheduler_plan(args, evidence)
        )
        try:
            if not args.recover:
                shared.handoff(args, evidence)
            with job_lock(args.job):
                if args.recover:
                    evidence = check_recovery(args)
                    original.write(
                        args.sidecar / "recovery-handoff.json",
                        dict(evidence, state="ADOPTED_WITHOUT_SIGNAL", time_unix=time.time()),
                    )
                state = State(args)
                worker = {
                    "pid": os.getpid(),
                    "argv": sys.argv,
                    "started_unix": time.time(),
                    "boot_id": evidence["boot_id"],
                    "handoff_file": str(args.sidecar / "handoff.json"),
                    "scheduler": SCHEMA,
                }
                original.write(args.sidecar / "worker.json", dict(worker, handoff=evidence))
                original.write(args.job / "worker.json", worker)
                threads = [
                    threading.Thread(target=run_thread, args=(copy.copy(args), name, evidence, state), name=name)
                    for name in ("baseline", "extra")
                ]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
                if state.errors:
                    raise RuntimeError("One or more lanes failed; inspect preserved outputs and status")
                original.status(
                    args,
                    "SUCCEEDED",
                    "All seven VOC seed0/final100 endpoints completed; LOFF-TA overlapped",
                    completed=7,
                    planned=7,
                )
        except Exception:
            original.write(
                args.sidecar / f"recovery-required-{time.time_ns()}.json",
                {
                    "traceback": traceback.format_exc(),
                    "adopted_child": evidence["child"],
                    "rollback_argv": evidence["rollback_argv"],
                    "policy": "Inspect live workers/DONE; never restart original queue with live sidecar children",
                },
            )
            raise


if __name__ == "__main__":
    main()
