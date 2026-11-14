"""Preserve the live campus baseline and delegate exactly one method to AutoDL.

Scheduling only: no frozen source edits, no seed/epoch changes, no child kills.
The caller must separately retire the paused delegated child after sealing its
checkpoint/transfer evidence. Returned remote endpoints are never fabricated.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from online_experiments import priority_single_seed_campaign as original
from online_experiments import priority_three_lane as shared
from online_experiments.crossdataset_campaign import gpu_admission, job_lock

SCHEMA = "campus-after-autodl-delegation-v1"
CONTRACT = {
    "voc0712": ("frofa-c2-d01", "loffta-c2-n005", "priority_voc_overlap.py", "priority-voc-overlap-v1"),
    "sku110k": ("cache-reuse-c1-int8", "detail25", "priority_three_lane.py", "priority-three-lane-v1"),
}


def supervisor_matches(args, argv, metadata_argv):
    script = str(ROOT / "online_experiments" / CONTRACT[args.dataset][2])
    return bool(
        metadata_argv
        and metadata_argv[0] == script
        and "--job" in metadata_argv
        and metadata_argv[metadata_argv.index("--job") + 1] == str(args.job)
        and argv in ([sys.executable, "-u", *metadata_argv], [sys.executable, *metadata_argv])
    )


def check(args):
    if sys.platform != "linux" or args.delegated_method != CONTRACT[args.dataset][1]:
        raise RuntimeError("Only the two explicit Linux campus delegations are supported")
    shared.pidfd_capability()
    if original.plan(args) != original.read(args.job / "pipeline-plan.json"):
        raise RuntimeError("Frozen source/cache/dataset plan changed")
    metadata = original.read(args.job / "worker.json")
    active = original.read(args.job / "active-child.json")
    status = original.read(args.job / "pipeline-status.json")
    old, child = shared.process(args.old_supervisor), shared.process(args.adopt_child)
    if old is None or child is None or metadata["pid"] != old["pid"] or active["pid"] != child["pid"]:
        raise RuntimeError("Exact recorded supervisor and main child required")
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if metadata["boot_id"] != boot or any(p["uid"] != os.getuid() or p["state"] == "Z" for p in (old, child)):
        raise RuntimeError("Process ownership, boot or lifetime differs")
    if metadata.get("scheduler") != CONTRACT[args.dataset][3] or not supervisor_matches(
        args, old["argv"], metadata["argv"]
    ):
        raise RuntimeError("Supervisor identity is not the exact prior sidecar")
    method = CONTRACT[args.dataset][0]
    if active["action"] != "train" or active["method"] != method or active["argv"] != child["argv"]:
        raise RuntimeError("Canonical active main child differs")
    if child["argv"] != original.command_for(args, "train", method):
        raise RuntimeError("Adopted main child command differs from frozen recipe")
    baseline = status["lanes"]["baseline"]
    if baseline.get("child_pid") != child["pid"] or baseline.get("method") != method:
        raise RuntimeError("Prior sidecar baseline lane differs from adopted child")
    completed = original.read(args.job / "completed-methods.json")
    expected = list(original.ordered_methods(args))[: list(original.ordered_methods(args)).index(method)]
    if [row["method"] for row in completed] != expected:
        raise RuntimeError("Prior completed method prefix differs from frozen queue")
    if any(not done(args, row["method"]) for row in completed):
        raise RuntimeError("Prior completed endpoint lacks its DONE marker")
    output = str(original.output_for(args, method))
    owners = [
        p["pid"]
        for p in shared.training_processes()
        if "--output" in p["argv"] and p["argv"][p["argv"].index("--output") + 1] == output
    ]
    if owners != [child["pid"]]:
        raise RuntimeError("Adopted output must have exactly one live training owner")
    return {
        "supervisor": old,
        "child": child,
        "boot_id": boot,
        "plan_sha256": original.sha(args.job / "pipeline-plan.json"),
        "rollback_argv": [sys.executable, *metadata["argv"]],
        "delegated_method": args.delegated_method,
    }


def check_recovery(args):
    if sys.platform != "linux" or args.delegated_method != CONTRACT[args.dataset][1]:
        raise RuntimeError("Only the two explicit Linux delegations may be recovered")
    plan, evidence = original.read(args.sidecar / "plan.json"), original.read(args.sidecar / "handoff.json")
    if any(
        evidence[key] != plan["evidence"][key]
        for key in ("supervisor", "child", "boot_id", "plan_sha256", "delegated_method")
    ):
        raise RuntimeError("Persisted handoff differs from its source-sealed plan")
    if any(original.sha(ROOT / name) != value for name, value in plan["source_sha256"].items()):
        raise RuntimeError("Continuation source changed after handoff")
    if original.plan(args) != original.read(args.job / "pipeline-plan.json") or evidence["plan_sha256"] != original.sha(
        args.job / "pipeline-plan.json"
    ):
        raise RuntimeError("Frozen pipeline differs from persisted handoff")
    if evidence["supervisor"]["pid"] != args.old_supervisor or evidence["child"]["pid"] != args.adopt_child:
        raise RuntimeError("Recovery PID arguments differ from persisted handoff")
    if (
        evidence["boot_id"] != Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        or evidence["child"]["uid"] != os.getuid()
    ):
        raise RuntimeError("Recovery process boot/owner changed")
    shared.verify_recovery_processes(evidence)
    metadata, active = original.read(args.job / "worker.json"), original.read(args.job / "active-child.json")
    if metadata["pid"] != args.old_supervisor:
        if metadata.get("scheduler") != SCHEMA or metadata.get("handoff_file") != str(args.sidecar / "handoff.json"):
            raise RuntimeError("Canonical job belongs to another continuation")
        if (current := shared.process(metadata["pid"])) is not None and current.get("state") != "Z":
            raise RuntimeError("A continuation supervisor is still alive")
    child = evidence["child"]
    if (
        active["pid"] != child["pid"]
        or active["argv"] != child["argv"]
        or child["argv"] != original.command_for(args, "train", CONTRACT[args.dataset][0])
    ):
        raise RuntimeError("Canonical child no longer matches handoff")
    return evidence


def require_delegated_retired(args):
    output = str(original.output_for(args, args.delegated_method))
    if any(
        "--output" in p["argv"] and p["argv"][p["argv"].index("--output") + 1] == output
        for p in shared.training_processes()
    ):
        raise RuntimeError(
            "Paused delegated child must be explicitly retired by the caller after backup; driver never signals it"
        )


def emit(args, state, **extra):
    completed = original.read(args.job / "completed-methods.json")
    remote_done = any(row["method"] == args.delegated_method for row in completed)
    lanes = {
        "baseline": {"state": state, **extra},
        "remote": {"state": "COMPLETED" if remote_done else "DELEGATED_AUTODL", "method": args.delegated_method},
    }
    record = {
        "schema": SCHEMA,
        "state": state,
        "pid": os.getpid(),
        "lanes": lanes,
        "completed": len(completed),
        "planned": 7,
        "time_unix": time.time(),
    }
    original.write(args.sidecar / "status.json", record)
    original.status(
        args,
        state,
        "Campus baseline continuation; one explicitly delegated AutoDL endpoint",
        lanes=lanes,
        completed=len(completed),
        planned=7,
        sidecar=str(args.sidecar),
        **extra,
    )


def record_complete(args, method, archive):
    # A remote-return handler may add another endpoint while this queue runs.
    # Read under the shared archive lock on each write; never use a stale list.
    with original.archive_lock():
        completed = original.read(args.job / "completed-methods.json")
        completed = [row for row in completed if row["method"] != method]
        completed.append({"method": method, "output": str(original.output_for(args, method)), "archive": archive})
        order = list(original.ordered_methods(args))
        completed.sort(key=lambda row: order.index(row["method"]))
        original.write(args.job / "completed-methods.json", completed)


def done(args, method):
    return (original.output_for(args, method) / ("DONE.json" if method in original.DIRECT else "DONE")).is_file()


def wait_adopted(args, child):
    while (current := shared.process(child["pid"])) is not None and current.get("state") != "Z":
        shared.same_lifetime(child, current)
        if current.get("argv") != [""]:
            shared.same_process(child, current)
        time.sleep(10)
    if not done(args, CONTRACT[args.dataset][0]):
        raise RuntimeError("Adopted main child exited without DONE; preserve checkpoint, no retry")


def finish(args, method):
    emit(args, "VALIDATING", method=method)
    original.execute(args, "validate", method)
    if not done(args, method):
        raise RuntimeError("Validator returned without sealed endpoint")
    record_complete(args, method, original.archive_case(args, method))


def train_one(args, method):
    output = str(original.output_for(args, method))
    if any(
        "--output" in p["argv"] and p["argv"][p["argv"].index("--output") + 1] == output
        for p in shared.training_processes()
    ):
        raise RuntimeError("Exact output already has a live owner; no duplicate training")
    with gpu_admission(args, "Campus remaining " + method):
        emit(args, "TRAINING", method=method)
        original.execute(args, "train", method)
    finish(args, method)


def continue_queue(args, evidence):
    adopted = CONTRACT[args.dataset][0]
    emit(args, "ADOPTED_TRAINING", method=adopted, child_pid=evidence["child"]["pid"])
    wait_adopted(args, evidence["child"])
    finish(args, adopted)
    for method in original.ordered_methods(args):
        completed = original.read(args.job / "completed-methods.json")
        if method == adopted or any(row["method"] == method for row in completed):
            continue
        if method == args.delegated_method:
            emit(args, "DELEGATED_AUTODL", method=method)
            if done(args, method):
                finish(args, method)
            continue
        if done(args, method):
            finish(args, method)
        else:
            train_one(args, method)
    completed = original.read(args.job / "completed-methods.json")
    if not any(row["method"] == args.delegated_method for row in completed) and done(args, args.delegated_method):
        finish(args, args.delegated_method)
    completed = original.read(args.job / "completed-methods.json")
    emit(args, "SUCCEEDED" if len(completed) == 7 else "LOCAL_COMPLETE_REMOTE_PENDING")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "run", "status"))
    parser.add_argument("--dataset", choices=tuple(CONTRACT), required=True)
    for name in ("job", "dataset-job", "sidecar"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--old-supervisor", type=int, required=True)
    parser.add_argument("--adopt-child", type=int, required=True)
    parser.add_argument("--delegated-method", required=True)
    parser.add_argument("--dependencies", type=Path, default=ROOT / ".research-deps/evidence-20261110/site")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--recover", action="store_true", help="Persisted exact-child handoff only; never signal")
    args = parser.parse_args()
    args.job, args.dataset_job, args.sidecar, args.dependencies = (
        getattr(args, key).resolve() for key in ("job", "dataset_job", "sidecar", "dependencies")
    )
    args.first_detector = "master-n" if args.dataset == "voc0712" else "yolo12-n"
    args.gpu, args.min_free_mib = 0, 8192
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
        require_delegated_retired(args)
        original.write(
            args.sidecar / ("recovery-plan.json" if args.recover else "plan.json"),
            {
                "schema": SCHEMA,
                "evidence": evidence,
                "source_sha256": {
                    name: original.sha(ROOT / name)
                    for name in (
                        "online_experiments/continue_campus_after_migration.py",
                        "online_experiments/test_continue_campus_after_migration.py",
                    )
                },
            },
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
                worker = {
                    "pid": os.getpid(),
                    "argv": sys.argv,
                    "boot_id": evidence["boot_id"],
                    "started_unix": time.time(),
                    "scheduler": SCHEMA,
                    "handoff_file": str(args.sidecar / "handoff.json"),
                }
                original.write(args.job / "worker.json", worker)
                original.write(args.sidecar / "worker.json", worker)
                continue_queue(args, evidence)
        except Exception:
            original.write(
                args.sidecar / "failure.json",
                {
                    "traceback": traceback.format_exc(),
                    "evidence": evidence,
                    "policy": "Preserve all live children/checkpoints; no automatic retry or signal to delegated child",
                },
            )
            raise


if __name__ == "__main__":
    main()
