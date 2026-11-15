"""Wait for the existing campus supervisor and skip one AutoDL delegation.

Scheduling only: no frozen source edits, no seed/epoch changes, no child kills.
The caller must separately retire the paused delegated child after sealing its
checkpoint/transfer evidence. Returned remote endpoints are never fabricated.
"""

from __future__ import annotations

import argparse
import contextlib
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


def completed_local_prefix(args, completed):
    """Remote completion may create one explicit hole in the local queue."""
    names = [row["method"] for row in completed]
    local = [method for method in original.ordered_methods(args) if method != args.delegated_method]
    local_names = [method for method in names if method != args.delegated_method]
    if len(names) != len(set(names)) or local_names != local[: len(local_names)]:
        raise RuntimeError("Completed local method prefix differs from frozen queue")
    if any(not done(args, method) for method in names):
        raise RuntimeError("Prior completed endpoint lacks its DONE marker")
    return local_names


def resolve_adopted_method(args, child, baseline, completed):
    names = completed_local_prefix(args, completed)
    order = list(original.ordered_methods(args))
    if child is None:
        if (
            args.dataset != "voc0712"
            or names != order[:3]
            or baseline.get("state") != "WAITING_OVERLAP"
            or baseline.get("method") != order[2]
        ):
            raise RuntimeError("No-child snapshot requires completed VOC FroFA and WAITING_OVERLAP")
        return None
    method = baseline.get("method")
    local = [item for item in order if item != args.delegated_method]
    if (
        method not in (CONTRACT[args.dataset][0], "vit-adapter-cache25")
        or method == args.delegated_method
        or names != local[: local.index(method)]
    ):
        raise RuntimeError("Observed baseline does not follow the frozen completed prefix")
    if baseline.get("child_pid") != child["pid"]:
        raise RuntimeError("Observed baseline lane child differs")
    return method


def retirement_snapshot(args, metadata, baseline, boot, old):
    """Read-only continuation after the two explicit training retirements."""
    from online_experiments import return_autodl_results as receiver

    receipt = original.read(args.retirement)
    if receipt.get("schema") != "autodl-migration-retirement-v1":
        raise RuntimeError("Unknown retirement receipt")
    case = receiver.load_case(args, args.dataset)
    if case["queue"] != args.job or case["method"] != args.delegated_method:
        raise RuntimeError("Retirement receipt belongs to another campus queue")
    case["retirement"] = receiver.validate_retirement(receipt.get("cases", {}).get(args.dataset, {}), case)
    if not receiver.retired_worker_inactive(case, boot):
        raise RuntimeError("Delegated original worker is still alive")
    completed_local_prefix(args, original.read(args.job / "completed-methods.json"))
    workers = shared.training_processes()
    child, method = None, None
    for candidate in original.ordered_methods(args):
        output = str(original.output_for(args, candidate))
        owners = [
            worker
            for worker in workers
            if "--output" in worker["argv"] and worker["argv"][worker["argv"].index("--output") + 1] == output
        ]
        if not owners:
            continue
        if candidate == args.delegated_method or len(owners) != 1 or child is not None or done(args, candidate):
            raise RuntimeError("Retirement continuation found an unexpected live output owner")
        child, method = owners[0], candidate
        if (
            old is None
            or old.get("state") == "Z"
            or child["argv"] != original.command_for(args, "train", method)
            or baseline.get("method") != method
            or baseline.get("child_pid") != child["pid"]
        ):
            raise RuntimeError("Live baseline must belong to the exact old supervisor lane and frozen command")
    if args.adopt_child and (child is None or child["pid"] != args.adopt_child):
        raise RuntimeError("Explicit observed child differs from retirement snapshot")
    expected_argv = [sys.executable, *metadata["argv"]]
    if not supervisor_matches(args, expected_argv, metadata["argv"]):
        raise RuntimeError("Exited supervisor metadata is outside the exact old job")
    return {
        "supervisor": old
        or {
            "pid": args.old_supervisor,
            "uid": os.getuid(),
            "state": "EXITED_AT_SNAPSHOT",
            "start_ticks": None,
            "argv": expected_argv,
        },
        "child": child,
        "boot_id": boot,
        "plan_sha256": original.sha(args.job / "pipeline-plan.json"),
        "rollback_argv": expected_argv,
        "delegated_method": args.delegated_method,
        "adopted_method": method,
        "retirement_sha256": original.sha(args.retirement),
    }


def prior_sidecar_baseline(args, metadata):
    """Resource admission overwrites canonical status; use the verified owner."""
    argv = metadata["argv"]
    if not supervisor_matches(args, [sys.executable, *argv], argv) or argv.count("--sidecar") != 1:
        raise RuntimeError("Prior supervisor has no unique verified sidecar argument")
    index = argv.index("--sidecar")
    if index + 1 >= len(argv):
        raise RuntimeError("Prior supervisor sidecar argument has no value")
    sidecar = Path(argv[index + 1])
    if not sidecar.is_absolute():
        raise RuntimeError("Prior supervisor sidecar must be absolute")
    status = original.read(sidecar / "status.json")
    if status.get("schema") != CONTRACT[args.dataset][3] or status.get("pid") != metadata["pid"]:
        raise RuntimeError("Prior sidecar status belongs to another supervisor")
    baseline = status.get("lanes", {}).get("baseline")
    if not isinstance(baseline, dict):
        raise RuntimeError("Verified prior sidecar has no baseline lane")  # noqa: TRY004 - invalid saved scheduler state
    return baseline


def check(args):
    if sys.platform != "linux" or args.delegated_method != CONTRACT[args.dataset][1]:
        raise RuntimeError("Only the two explicit Linux campus delegations are supported")
    shared.pidfd_capability()
    if original.plan(args) != original.read(args.job / "pipeline-plan.json"):
        raise RuntimeError("Frozen source/cache/dataset plan changed")
    metadata = original.read(args.job / "worker.json")
    active = original.read(args.job / "active-child.json")
    status = original.read(args.job / "pipeline-status.json")
    old = shared.process(args.old_supervisor)
    child = shared.process(args.adopt_child) if args.adopt_child else None
    if metadata["pid"] != args.old_supervisor:
        raise RuntimeError("Canonical metadata belongs to another supervisor")
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if getattr(args, "retirement", None):
        if metadata["boot_id"] != boot or metadata.get("scheduler") != CONTRACT[args.dataset][3]:
            raise RuntimeError("Retirement snapshot supervisor boot/scheduler differs")
        if old is not None and (
            old["uid"] != os.getuid()
            or (old.get("state") != "Z" and not supervisor_matches(args, old["argv"], metadata["argv"]))
        ):
            raise RuntimeError("Retirement snapshot live supervisor identity differs")
        return retirement_snapshot(args, metadata, prior_sidecar_baseline(args, metadata), boot, old)
    if old is None or (args.adopt_child and child is None):
        raise RuntimeError("Exact recorded supervisor and requested baseline required")
    if metadata["boot_id"] != boot or any(p["uid"] != os.getuid() or p["state"] == "Z" for p in (old, child) if p):
        raise RuntimeError("Process ownership, boot or lifetime differs")
    if metadata.get("scheduler") != CONTRACT[args.dataset][3] or not supervisor_matches(
        args, old["argv"], metadata["argv"]
    ):
        raise RuntimeError("Supervisor identity is not the exact prior sidecar")
    baseline = status["lanes"]["baseline"]
    completed = original.read(args.job / "completed-methods.json")
    method = resolve_adopted_method(args, child, baseline, completed)
    if child and child["argv"] != original.command_for(args, "train", method):
        raise RuntimeError("Observed baseline command differs from frozen recipe")
    if not child or active["pid"] != child["pid"]:
        previous = shared.process(active["pid"])
        if (
            active.get("method") not in [row["method"] for row in completed]
            or active.get("action") != "train"
            or active.get("argv") != original.command_for(args, "train", active["method"])
            or (previous and previous.get("state") != "Z")
        ):
            raise RuntimeError("Stale canonical child must be a sealed exited predecessor")
    elif active.get("action") != "train" or active.get("method") != method or active.get("argv") != child["argv"]:
        raise RuntimeError("Canonical active baseline differs")
    workers = shared.training_processes()
    for candidate in original.ordered_methods(args):
        if candidate == args.delegated_method:
            continue
        output = str(original.output_for(args, candidate))
        owners = [
            p["pid"]
            for p in workers
            if "--output" in p["argv"] and p["argv"][p["argv"].index("--output") + 1] == output
        ]
        if owners != ([child["pid"]] if candidate == method else []):
            raise RuntimeError("Nondelegated output has an unexpected live training owner")
    return {
        "supervisor": old,
        "child": child,
        "boot_id": boot,
        "plan_sha256": original.sha(args.job / "pipeline-plan.json"),
        "rollback_argv": [sys.executable, *metadata["argv"]],
        "delegated_method": args.delegated_method,
        "adopted_method": method,
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
    if evidence["supervisor"]["pid"] != args.old_supervisor or (
        args.adopt_child and (evidence["child"] is None or evidence["child"]["pid"] != args.adopt_child)
    ):
        raise RuntimeError("Recovery PID arguments differ from persisted handoff")
    if getattr(args, "retirement", None) and evidence.get("retirement_sha256") != original.sha(args.retirement):
        raise RuntimeError("Retirement receipt changed after the observation plan")
    if (
        evidence["boot_id"] != Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        or evidence["supervisor"]["uid"] != os.getuid()
    ):
        raise RuntimeError("Recovery process boot/owner changed")
    previous = shared.process(evidence["supervisor"]["pid"])
    if previous is not None and previous.get("state") != "Z":
        shared.same_process(evidence["supervisor"], previous)
    metadata = original.read(args.job / "worker.json")
    if metadata["pid"] != args.old_supervisor:
        if metadata.get("scheduler") != SCHEMA or metadata.get("handoff_file") != str(args.sidecar / "handoff.json"):
            raise RuntimeError("Canonical job belongs to another continuation")
        if (current := shared.process(metadata["pid"])) is not None and current.get("state") != "Z":
            raise RuntimeError("A continuation supervisor is still alive")
    completed_local_prefix(args, original.read(args.job / "completed-methods.json"))
    return evidence


def wait_previous_supervisor(args, evidence):
    """The old queue exits naturally; this controller never signals a process."""
    old = evidence["supervisor"]
    while (current := shared.process(old["pid"])) is not None and current.get("state") != "Z":
        shared.same_lifetime(old, current)
        if current.get("argv") != [""]:
            shared.same_process(old, current)
        original.write(
            args.sidecar / "status.json",
            {
                "schema": SCHEMA,
                "state": "WAITING_PREVIOUS_SUPERVISOR",
                "pid": os.getpid(),
                "old_supervisor_pid": old["pid"],
                "time_unix": time.time(),
                "policy": "No signals; existing baseline continues unchanged",
            },
        )
        time.sleep(10)
    original.write(
        args.sidecar / "handoff.json",
        dict(evidence, state="PREVIOUS_SUPERVISOR_EXITED_NATURALLY", time_unix=time.time()),
    )


@contextlib.contextmanager
def canonical_queue_lock(args):
    """Wait for a concurrent verified result installation on the same lock."""
    import fcntl

    with (args.job / "pipeline.lock").open("a") as stream:
        while True:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                original.write(
                    args.sidecar / "status.json",
                    {
                        "schema": SCHEMA,
                        "state": "WAITING_CANONICAL_LOCK",
                        "pid": os.getpid(),
                        "time_unix": time.time(),
                        "job": str(args.job),
                    },
                )
                time.sleep(10)
            else:
                break
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


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
    if Path(output).exists():
        raise RuntimeError("Pending local output already exists without DONE; preserve checkpoint, no automatic retry")
    with gpu_admission(args, "Campus remaining " + method):
        emit(args, "TRAINING", method=method)
        original.execute(args, "train", method)
    finish(args, method)


def continue_queue(args, evidence):
    # The previous supervisor may finish more cases while we wait. Re-read its
    # real sealed endpoints; never restart an existing DONE or a live output.
    completed_local_prefix(args, original.read(args.job / "completed-methods.json"))
    local_outputs = {
        str(original.output_for(args, method))
        for method in original.ordered_methods(args)
        if method != args.delegated_method
    }
    if any(
        "--output" in worker["argv"] and worker["argv"][worker["argv"].index("--output") + 1] in local_outputs
        for worker in shared.training_processes()
    ):
        raise RuntimeError("A previous local training owner remains alive; no duplicate queue")
    for method in original.ordered_methods(args):
        completed = original.read(args.job / "completed-methods.json")
        if any(row["method"] == method for row in completed):
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
    parser.add_argument(
        "--retirement", type=Path, help="Explicit retirement receipt; observe the old queue without signals"
    )
    parser.add_argument("--bundle-root", type=Path, help="Sealed transfer plans required with --retirement")
    parser.add_argument("--dependencies", type=Path, default=ROOT / ".research-deps/evidence-20261110/site")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--recover", action="store_true", help="Persisted exact-child handoff only; never signal")
    args = parser.parse_args()
    if bool(args.retirement) != bool(args.bundle_root):
        parser.error("--retirement and --bundle-root must be supplied together")
    if args.retirement:
        args.retirement, args.bundle_root = args.retirement.resolve(), args.bundle_root.resolve()
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
            wait_previous_supervisor(args, evidence)
            require_delegated_retired(args)
            with canonical_queue_lock(args):
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
