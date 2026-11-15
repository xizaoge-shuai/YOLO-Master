"""CPU-only campus receiver for exactly two delegated AutoDL training runs.

Download first. Preserve the paused origin directory, install only after an
explicit retirement receipt and a fresh liveness check, validate/archive, then
publish DONE last. Credentials use the transfer helper's single stdin line.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import time
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from online_experiments import priority_single_seed_campaign as original
from online_experiments import transfer_autodl_resume as transfer

SOURCE_ROOT = Path("/mnt/data/users/zhjia")
REMOTE_ROOT = PurePosixPath("/root/autodl-tmp/campus-mirror")
DATASETS = ("sku110k", "voc0712")
METHODS = {"sku110k": "detail25", "voc0712": "loffta-c2-n005"}
LIFETIMES = {"sku110k": (2105170, 45384841), "voc0712": (2300847, 46203056)}
OLD_JOBS = {"sku110k": "job-20261110-sku110k-official-loader-v2", "voc0712": "job-20261110-voc0712-s0-v2"}
PRIORITY = Path("runs/paper/priority/job-20261112-single-seed")
SCHEMA = "autodl-resume-return-v1"
CHUNK = 8 * 1024**2
GIB = 1024**3


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    original.write(Path(path), value)


def bounded(path, parent):
    path, parent = Path(path), Path(parent).resolve()
    if ".." in path.parts or not path.is_absolute() or path == parent or not path.resolve().is_relative_to(parent):
        raise ValueError("Local result path escapes its authorized root")
    return path


def load_case(args, dataset):
    plan_path = args.bundle_root / (dataset + "-plan.json")
    plan = read(plan_path)
    if plan.get("schema") != "migrated-crossdataset-transfer-v1":
        raise ValueError("Unknown original bundle plan")
    origin = plan["origin"]
    method = METHODS[dataset]
    expected_output = ROOT / PRIORITY / dataset / "train" / (method + "-s0")
    expected_job = ROOT / "runs/paper/crossdataset" / OLD_JOBS[dataset]
    if (origin.get("output"), origin.get("job"), origin.get("method"), origin.get("seed"), origin.get("repo")) != (
        str(expected_output),
        str(expected_job),
        method,
        0,
        str(ROOT),
    ):
        raise ValueError("Bundle origin is outside the two authorized canonical cases")
    if origin.get("epoch") not in range(1, 100) or not re.fullmatch(
        r"[0-9a-f]{64}", origin.get("checkpoint_sha256", "")
    ):
        raise ValueError("Bundle origin checkpoint/epoch is invalid")
    bounded(expected_output, ROOT)
    return {
        "dataset": dataset,
        "method": method,
        "origin": origin,
        "plan_sha256": sha(plan_path),
        "output": expected_output,
        "dataset_job": expected_job,
        "queue": ROOT / PRIORITY / dataset,
    }


def validate_return(payload, case):
    if payload.get("schema") != SCHEMA or payload.get("dataset") != case["dataset"]:
        raise ValueError("Remote result dataset/schema differs")
    origin = payload.get("origin", {})
    for key in ("output", "job", "method", "epoch", "seed", "checkpoint_sha256"):
        if origin.get(key) != case["origin"].get(key):
            raise ValueError("Remote result original checkpoint identity differs")
    files = payload.get("files")
    if not isinstance(files, list) or not 0 < len(files) <= 10000:
        raise ValueError("Invalid result file count")
    output = PurePosixPath(case["origin"]["output"])
    source = PurePosixPath(str(SOURCE_ROOT).replace("\\", "/"))
    if not output.is_relative_to(source):
        raise ValueError("Original output escapes campus source root")
    remote_output = REMOTE_ROOT / output.relative_to(source)
    total, seen, rows = 0, set(), []
    for row in files:
        relative = row.get("relative")
        if not isinstance(relative, str) or any(char in relative for char in ("\x00", "\\", "\n", "\r")):
            raise ValueError("Invalid result relative path")
        path = PurePosixPath(relative)
        if (
            path.is_absolute()
            or ".." in path.parts
            or str(path) != relative
            or relative in ("", ".")
            or relative in seen
        ):
            raise ValueError("Escaping or duplicate result relative path")
        seen.add(relative)
        if row.get("remote") != str(remote_output / path):
            raise ValueError("Remote result path is outside its exact mirrored origin output")
        if type(row.get("bytes")) is not int or not 0 <= row["bytes"] <= 8 * GIB:
            raise ValueError("Unreasonable result file size")
        if not re.fullmatch(r"[0-9a-f]{64}", row.get("sha256", "")):
            raise ValueError("Invalid result SHA256")
        total += row["bytes"]
        rows.append({key: row[key] for key in ("relative", "remote", "bytes", "sha256")})
    if total > 16 * GIB:
        raise ValueError("Result directory exceeds the authorized lightweight-output limit")
    return rows


def partial_path(path, row):
    return Path(str(path) + ".partial." + row["sha256"][:16])


def download_file(sftp, row, target, throttle):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise RuntimeError("Local result symlink refused")
    if target.exists():
        if not target.is_file() or target.stat().st_size != row["bytes"] or sha(target) != row["sha256"]:
            raise RuntimeError("Existing local result conflicts; preserved")
        return
    partial = partial_path(target, row)
    if partial.is_symlink() or (partial.exists() and (not partial.is_file() or partial.stat().st_size > row["bytes"])):
        raise RuntimeError("Invalid local partial result")
    info = sftp.lstat(row["remote"])
    if not stat.S_ISREG(info.st_mode) or info.st_size != row["bytes"]:
        raise RuntimeError("Remote result size/type changed")
    offset = partial.stat().st_size if partial.exists() else 0
    digest, remote_digest = hashlib.sha256(), hashlib.sha256()
    with sftp.open(row["remote"], "rb") as source:
        if hasattr(source, "prefetch"):
            source.prefetch(file_size=row["bytes"], max_concurrent_requests=64)
        if offset:
            remaining = offset
            with partial.open("rb") as prefix:
                while remaining:
                    size = min(CHUNK, remaining)
                    local_block, remote_block = prefix.read(size), source.read(size)
                    if len(local_block) != size or len(remote_block) != size:
                        raise RuntimeError("Partial prefix ended before its declared size")
                    digest.update(local_block)
                    remote_digest.update(remote_block)
                    throttle(size)
                    remaining -= size
            if digest.hexdigest() != remote_digest.hexdigest():
                raise RuntimeError("Local partial prefix differs from remote result; preserved")
        with partial.open("ab" if partial.exists() else "xb") as destination:
            remaining = row["bytes"] - offset
            while remaining:
                block = source.read(min(CHUNK, remaining))
                if not block:
                    raise RuntimeError("Remote result ended unexpectedly")
                destination.write(block)
                digest.update(block)
                remaining -= len(block)
                throttle(len(block))
            destination.flush()
            os.fsync(destination.fileno())
    if digest.hexdigest() != row["sha256"] or partial.stat().st_size != row["bytes"]:
        raise RuntimeError("Downloaded result hash differs; partial preserved")
    if target.exists():
        raise RuntimeError("Result appeared concurrently; refusing overwrite")
    partial.rename(target)


def validate_retirement(record, case):
    expected = LIFETIMES[case["dataset"]]
    if record.get("retired") is not True or record.get("checkpoint_sha256") != case["origin"]["checkpoint_sha256"]:
        raise ValueError("Explicit retired checkpoint receipt is required")
    if (record.get("pid"), record.get("start_ticks")) != expected or record.get("uid") != 1010:
        raise ValueError("Retirement receipt belongs to another original process")
    if not isinstance(record.get("boot_id"), str) or not record["boot_id"] or not isinstance(record.get("argv"), list):
        raise ValueError("Retirement receipt lacks the original lifetime evidence")
    command = record["argv"]
    if (
        "--output" not in command
        or command[command.index("--output") + 1 :] == []
        or command[command.index("--output") + 1] != case["origin"]["output"]
    ):
        raise ValueError("Retirement receipt command belongs to another output")
    return record


def process(pid):
    try:
        folder = Path("/proc") / str(pid)
        fields = (folder / "stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": pid, "uid": folder.stat().st_uid, "state": fields[0], "start_ticks": int(fields[19])}
    except FileNotFoundError:
        return None


def retired_worker_inactive(case, boot, probe=process):
    record = case["retirement"]
    current = probe(record["pid"])
    if current is None or record["boot_id"] != boot:
        return True
    same = all(current.get(key) == record.get(key) for key in ("pid", "uid", "start_ticks"))
    return not same or current.get("state") in {"Z", "X"}


def output_workers(output):
    matches = []
    for folder in Path("/proc").iterdir():
        if not folder.name.isdecimal():
            continue
        try:
            fields = (folder / "stat").read_text().rsplit(")", 1)[1].split()
            if fields[0] in {"Z", "X"} or folder.stat().st_uid != os.getuid():
                continue
            command = (folder / "cmdline").read_bytes().decode(errors="surrogateescape").rstrip("\0").split("\0")
            if "--output" in command and command[command.index("--output") + 1] == str(output):
                matches.append(int(folder.name))
        except FileNotFoundError:
            continue
    return matches


def install_moves(output, stage, backup, held_done, expected_checkpoint):
    """Preserve both directories; no deletion and no completion seal exposed."""
    output, stage, backup, held_done = map(Path, (output, stage, backup, held_done))
    source = backup if backup.exists() else output
    if sha(source / "weights/last.pt") != expected_checkpoint:
        raise RuntimeError("Original canonical checkpoint changed; installation refused")
    if stage.exists():
        if backup.exists() and output.exists():
            raise RuntimeError("Both origin backup and canonical output exist before stage installation")
        done = stage / "DONE"
        if done.exists():
            if done.read_text(encoding="utf-8").strip() != "SUCCEEDED":
                raise RuntimeError("Returned DONE is invalid")
            held_done.parent.mkdir(parents=True, exist_ok=True)
            if held_done.exists():
                if sha(held_done) != sha(done):
                    raise RuntimeError("Held completion seal conflicts")
                raise RuntimeError("Both staged and held completion seals exist; preserve for review")
            done.rename(held_done)
        elif not held_done.is_file():
            raise RuntimeError("Returned completion seal is missing")
        if not backup.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            output.rename(backup)
        stage.rename(output)
    elif not backup.exists() or not output.exists() or not (held_done.exists() or (output / "DONE").is_file()):
        raise RuntimeError("Incomplete prior install cannot be resumed automatically")


def verify_received(folder, rows, held_done=None):
    for row in rows:
        path = (
            Path(held_done)
            if row["relative"] == "DONE" and held_done and not (folder / "DONE").exists()
            else folder / row["relative"]
        )
        if path.is_symlink() or not path.is_file() or path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise RuntimeError("Returned file no longer matches its sealed manifest")


def verify_endpoint(folder, rows, expected_origin=None):
    lookup = {row["relative"]: row["sha256"] for row in rows}
    required = {
        "args.json",
        "summary.json",
        "metrics.json",
        "environment.json",
        "initialization.json",
        "endpoint.json",
        "predictions.json",
        "teacher-queries.jsonl",
        "weights/last.pt",
        "DONE",
    }
    if not required.issubset(lookup):
        raise RuntimeError("Returned directory lacks required final endpoint files")
    endpoint, summary = read(folder / "endpoint.json"), read(folder / "summary.json")
    if summary.get("completed_epochs") != 100 or endpoint.get("fixed_epoch") != 100:
        raise RuntimeError("Returned result is not final epoch100")
    for name, value in endpoint["sha256"].items():
        if lookup.get(name) != value:
            raise RuntimeError("Returned manifest does not cover every endpoint hash")
    schedules = endpoint["schedule_sha256"]
    if len(schedules) != 100 or any(lookup.get("schedules/" + name) != value for name, value in schedules.items()):
        raise RuntimeError("Returned manifest does not cover all100 epoch schedules")
    if expected_origin is not None:
        preserved = {
            "migration-origin.json",
            "environment-origin.json",
            "weights/origin-migration.pt",
            "migration-resume.json",
        }
        if not preserved.issubset(lookup) or endpoint.get("migration", {}).get("origin") != expected_origin:
            raise RuntimeError("Returned endpoint lacks its exact mixed-host migration origin")
        if lookup["weights/origin-migration.pt"] != expected_origin["checkpoint_sha256"]:
            raise RuntimeError("Returned origin checkpoint differs from the original bundle")


def merge_completed(args, method, output, archive):
    with original.archive_lock():
        path = args.job / "completed-methods.json"
        rows = read(path) if path.exists() else []
        rows = [row for row in rows if row["method"] != method]
        rows.append({"method": method, "output": str(output), "archive": archive})
        order = read(args.job / "pipeline-plan.json")["methods"]
        if any(row["method"] not in order for row in rows) or len({row["method"] for row in rows}) != len(rows):
            raise RuntimeError("Completed queue contains unexpected or duplicate methods")
        rows.sort(key=lambda row: order.index(row["method"]))
        write(path, rows)


def execute_validation(native, method, private_job):
    """Reuse execute without touching the live campus queue's active-child."""
    command = original.command_for(native, "validate", method)
    private_job = Path(private_job)
    private_job.mkdir(parents=True, exist_ok=True)
    auxiliary = SimpleNamespace(**{**vars(native), "job": private_job})
    planner, processes = original.command_for, original.subprocess

    def fixed_command(args, action, chosen):
        if args is auxiliary and action == "validate" and chosen == method:
            return command
        return planner(args, action, chosen)

    class CPUProcesses:
        def __getattr__(self, name):
            return getattr(processes, name)

        def Popen(self, argv, **kwargs):
            environment = dict(kwargs.get("env", os.environ))
            environment.update(
                CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"
            )
            kwargs["env"] = environment
            return processes.Popen(argv, **kwargs)

    # This adapter changes this receiver's module references only. Frozen source
    # files and concurrently running campus processes remain untouched.
    original.command_for, original.subprocess = fixed_command, CPUProcesses()
    try:
        original.execute(auxiliary, "validate", method)
    finally:
        original.command_for, original.subprocess = planner, processes


def remote_json(sftp, path, *, maximum=16 * 1024**2):
    info = transfer.remote_stat(sftp, path)
    if info is None:
        return None
    if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
        raise RuntimeError("Remote JSON is not a bounded regular file")
    with sftp.open(path, "rb") as stream:
        data = stream.read(maximum + 1)
    if len(data) > maximum:
        raise RuntimeError("Remote JSON grew beyond its limit")
    return json.loads(data)


@contextlib.contextmanager
def local_supervisor_guard(queue):
    """Own the canonical supervisor lock before publishing any imported result."""
    import fcntl

    with (Path(queue) / "pipeline.lock").open("a") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def handle_case(args, sftp, dataset, state, throttle):
    plan_path = args.bundle_root / (dataset + "-plan.json")
    if not plan_path.is_file():
        return {"state": "WAITING_REMOTE", "reason": "WAITING_LOCAL_BUNDLE"}
    case = load_case(args, dataset)
    remote_case = PurePosixPath(args.remote_job) / dataset
    status = remote_json(sftp, str(remote_case / "status.json"), maximum=1024**2)
    if status is None or status.get("state") != "SUCCEEDED":
        if status and status.get("state") == "FAILED":
            raise RuntimeError("Delegated target run failed; preserve its evidence")
        return {"state": "WAITING_REMOTE", "remote_state": status.get("state") if status else "NOT_STARTED"}
    payload = remote_json(sftp, str(remote_case / "return-manifest.json"))
    if payload is None:
        return {"state": "WAITING_REMOTE", "reason": "WAITING_RETURN_MANIFEST"}
    rows = validate_return(payload, case)
    receipt = args.job / "return-receipts" / (dataset + ".json")
    identity = {"manifest_sha256": transfer.canonical_hash(payload), "origin_plan_sha256": case["plan_sha256"]}
    prior = read(receipt) if receipt.is_file() else None
    if prior and any(prior.get(key) != value for key, value in identity.items()):
        raise RuntimeError("Previously observed return manifest changed")
    if prior and prior.get("state") == "SUCCEEDED":
        return prior
    stage = bounded(args.job / "returned" / dataset / (case["method"] + "-s0"), ROOT)
    backup = bounded(args.job / "origin-output" / (dataset + "-" + case["method"] + "-s0"), ROOT)
    held = bounded(args.job / "return-seals" / dataset / "DONE", ROOT)
    installed = bool(
        prior
        and prior.get("state") == "INSTALLING"
        and backup.is_dir()
        and case["output"].is_dir()
        and not stage.exists()
        and (held.is_file() or (case["output"] / "DONE").is_file())
    )
    downloaded = bool(prior and prior.get("state") in {"DOWNLOADED", "INSTALLING"} and stage.is_dir())
    if downloaded:
        # A busy local supervisor may last for hours. Recheck the sealed stage,
        # but do not download it again or change an INSTALLING crash receipt.
        verify_received(stage, rows, held)
        verify_endpoint(stage, rows, case["origin"])
    elif not installed:
        stage.mkdir(parents=True, exist_ok=True)
        required = sum(row["bytes"] for row in rows) + GIB
        if shutil.disk_usage(stage).free < required:
            raise RuntimeError("Insufficient campus disk space for a non-destructive result stage")
        for index, row in enumerate(rows, 1):
            if row["relative"] == "DONE" and held.is_file() and prior and prior.get("state") == "INSTALLING":
                if held.stat().st_size != row["bytes"] or sha(held) != row["sha256"]:
                    raise RuntimeError("Held completion seal differs from the return manifest")
                continue
            target = bounded(stage / row["relative"], stage)
            write(
                args.job / "return-status.json",
                {
                    "state": "WAITING_REMOTE",
                    "action": "DOWNLOADING",
                    "dataset": dataset,
                    "file_index": index,
                    "files": len(rows),
                    "relative": row["relative"],
                    "gpu_used": False,
                    "cases": state,
                    "updated_unix": time.time(),
                },
            )
            download_file(sftp, row, target, throttle)
        verify_received(stage, rows, held)
        verify_endpoint(stage, rows, case["origin"])
        write(receipt, {**identity, "state": "DOWNLOADED", "dataset": dataset})
    with local_supervisor_guard(case["queue"]) as acquired:
        if not acquired:
            return {"state": "WAITING_LOCAL_SUPERVISOR", "reason": "CANONICAL_PIPELINE_LOCK_BUSY", **identity}
        return install_received_case(args, case, rows, identity, prior, stage, backup, held, receipt)


def install_received_case(args, case, rows, identity, prior, stage, backup, held, receipt):
    """Caller holds pipeline.lock through retirement checks and final DONE."""
    dataset = case["dataset"]
    retirement_path = args.job / "retirement.json"
    if not retirement_path.is_file():
        return {"state": "WAITING_REMOTE", "reason": "WAITING_EXPLICIT_ORIGIN_RETIREMENT", **identity}
    retirement = read(retirement_path)
    if retirement.get("schema") != "autodl-migration-retirement-v1":
        raise ValueError("Unknown origin retirement receipt schema")
    record = retirement.get("cases", {}).get(dataset)
    if record is None:
        return {"state": "WAITING_REMOTE", "reason": "WAITING_DATASET_RETIREMENT", **identity}
    case["retirement"] = validate_retirement(record, case)
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if not retired_worker_inactive(case, boot) or output_workers(case["output"]):
        return {"state": "WAITING_REMOTE", "reason": "ORIGINAL_OUTPUT_WORKER_STILL_ACTIVE", **identity}
    write(receipt, {**identity, "state": "INSTALLING", "dataset": dataset, "retirement_sha256": sha(retirement_path)})
    install_moves(case["output"], stage, backup, held, case["origin"]["checkpoint_sha256"])
    verify_received(case["output"], rows, held)
    verify_endpoint(case["output"], rows, case["origin"])
    native = SimpleNamespace(
        job=case["queue"],
        dataset=dataset,
        dataset_job=case["dataset_job"],
        dependencies=args.dependencies,
        publish=args.publish,
    )
    execute_validation(native, case["method"], args.job / "return-validation" / dataset)
    # archive_case takes archive_lock internally; never nest that lock here.
    archive = original.archive_case(native, case["method"])
    merge_completed(native, case["method"], case["output"], archive)
    published = case["output"] / "DONE"
    if published.exists():
        expected = next(row for row in rows if row["relative"] == "DONE")
        if held.exists() or sha(published) != expected["sha256"] or not prior or prior.get("state") != "INSTALLING":
            raise RuntimeError("Canonical completion marker appeared before final publication")
    else:
        held.rename(published)
    result = {
        **identity,
        "state": "SUCCEEDED",
        "dataset": dataset,
        "method": case["method"],
        "archive": archive,
        "output": str(case["output"]),
        "preserved_origin": str(backup),
        "gpu_used": False,
    }
    write(receipt, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--bundle-root", required=True, type=Path)
    parser.add_argument("--remote-job", default=str(REMOTE_ROOT / ".migration/job-20261114-resume"))
    parser.add_argument("--dependencies", type=Path, default=ROOT / ".research-deps/evidence-20261110/site")
    parser.add_argument("--host", default="connect.weste.seetacloud.com")
    parser.add_argument("--port", type=int, default=37540)
    parser.add_argument("--username", default="root")
    parser.add_argument("--host-key-sha256", required=True)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--rate-mib", type=float, default=32)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    args.job, args.bundle_root = bounded(args.job.absolute(), ROOT), bounded(args.bundle_root.absolute(), ROOT)
    if not 10 <= args.poll_seconds <= 60 or not 1 <= args.rate_mib <= 40:
        parser.error("poll-seconds must be10..60; rate-mib must be1..40")
    if PurePosixPath(args.remote_job) != REMOTE_ROOT / ".migration/job-20261114-resume":
        parser.error("Only the authorized target resume job is permitted")
    acquired, transport, sftp, state = False, None, None, {}
    try:
        with transfer.owner_lock(args.job / "return.lock"):
            acquired = True
            if hasattr(os, "nice"):
                os.nice(10)
            password = transfer.read_password()
            try:
                transport, sftp = transfer.connect(args, password)
            finally:
                del password
            throttle = transfer.RateLimit(args.rate_mib)
            while True:
                for dataset in DATASETS:
                    if state.get(dataset, {}).get("state") != "SUCCEEDED":
                        state[dataset] = handle_case(args, sftp, dataset, state, throttle)
                completed = sum(value["state"] == "SUCCEEDED" for value in state.values())
                write(
                    args.job / "return-status.json",
                    {
                        "state": "SUCCEEDED" if completed == 2 else "WAITING_REMOTE",
                        "completed": completed,
                        "planned": 2,
                        "cases": state,
                        "gpu_used": False,
                        "updated_unix": time.time(),
                    },
                )
                if completed == 2:
                    return 0
                time.sleep(args.poll_seconds)
    except Exception as error:  # noqa: BLE001 - redact all third-party credential diagnostics
        if acquired:
            write(
                args.job / "return-status.json",
                {
                    "state": "FAILED",
                    "error_type": type(error).__name__,
                    "cases": state,
                    "gpu_used": False,
                    "updated_unix": time.time(),
                },
            )
        print(json.dumps({"state": "FAILED" if acquired else "BUSY", "error_type": type(error).__name__}), flush=True)
        return 1
    finally:
        if sftp is not None:
            sftp.close()
        if transport is not None:
            transport.close()


if __name__ == "__main__":
    raise SystemExit(main())
