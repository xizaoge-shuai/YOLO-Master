"""Wait for verified SKU/VOC transfers, resume both fixed runs, seal returns.

Standalone standard-library coordinator. No new seeds, recipes, smoke training,
or second prediction pass. Source aliases must already point into target_root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import threading
import time
import traceback
import uuid
from pathlib import Path, PurePosixPath

TARGET = Path("/root/autodl-tmp/campus-mirror")
SOURCE = Path("/mnt/data/users/zhjia")
EXTRACTION_LOCK = threading.Lock()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def expected_marker(dataset, rows):
    files = [{key: row[key] for key in ("target", "bytes", "sha256")} for row in rows if row["dataset"] == dataset]
    return {
        "schema": "autodl-resume-transfer-v1",
        "state": "DONE",
        "dataset": dataset,
        "files": len(files),
        "bytes": sum(row["bytes"] for row in files),
        "dataset_manifest_sha256": canonical_hash(files),
    }


def relative_name(value):
    name = PurePosixPath(value)
    if not value or name.is_absolute() or ".." in name.parts or str(name) != value or "\\" in value or "\x00" in value:
        raise RuntimeError("Archive member must be a canonical relative path")
    return name


def destination(root, relative):
    relative_name(relative)
    root = Path(root)
    current = root
    if root.is_symlink():
        raise RuntimeError("Target root must not be a symlink")
    for part in PurePosixPath(relative).parts:
        current /= part
        if current.is_symlink():
            raise RuntimeError("Destination must not traverse a symlink")
    return current


def extract_verified(archive_path, target_root, records):
    expected = {}
    for row in records:
        name = row["target_relative"]
        if destination(target_root, name) != Path(row["target"]) or name in expected:
            raise RuntimeError("Archive target coverage differs from transfer plan")
        expected[name] = row
    with tarfile.open(archive_path, "r:") as archive:
        members = archive.getmembers()
        for member in members:
            relative_name(member.name)
            if not member.isfile():
                raise RuntimeError("Archive members must be regular files")
        if len(members) != len(expected) or {member.name for member in members} != set(expected):
            raise RuntimeError("Archive member coverage differs from transfer plan")
        for member in members:
            row = expected[member.name]
            target = destination(target_root, member.name)
            if member.size != row["bytes"]:
                raise RuntimeError("Archive member byte size differs")
            if target.exists():
                if not target.is_file() or target.stat().st_size != row["bytes"] or sha(target) != row["sha256"]:
                    raise RuntimeError("Existing destination conflict: " + str(target))
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            destination(target_root, member.name)
            temporary = target.with_name(target.name + ".extracting." + uuid.uuid4().hex)
            digest = hashlib.sha256()
            try:
                with archive.extractfile(member) as source, temporary.open("xb") as stream:
                    for block in iter(lambda: source.read(8 * 1024**2), b""):
                        stream.write(block)
                        digest.update(block)
                if digest.hexdigest() != row["sha256"]:
                    raise RuntimeError("Archive member SHA256 differs")
                # link is atomic and refuses clobbering a concurrently created file.
                os.link(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)


def result_manifest(dataset, origin, target_root):
    output = Path(origin["output"])
    if not (output / "DONE").is_file() or any(
        not (output / name).is_file() for name in ("endpoint.json", "metrics.json", "summary.json")
    ):
        raise RuntimeError("Migration endpoint is not complete")
    if read(output / "summary.json").get("completed_epochs") != 100:
        raise RuntimeError("Migration endpoint has not completed100epochs")
    files = []
    root = Path(target_root).resolve(strict=True)
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name.endswith((".lock", ".tmp")):
            continue
        if path.is_symlink() or root not in path.resolve(strict=True).parents:
            raise RuntimeError("Return file must be regular and inside mirror")
        files.append(
            {
                "relative": path.relative_to(output).as_posix(),
                "remote": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha(path),
            }
        )
    lookup = {row["relative"]: row["sha256"] for row in files}
    endpoint = read(output / "endpoint.json")
    required = {
        "args.json",
        "environment.json",
        "initialization.json",
        "summary.json",
        "metrics.json",
        "predictions.json",
        "weights/last.pt",
        "teacher-queries.jsonl",
    }
    if endpoint.get("fixed_epoch") != 100 or set(endpoint.get("sha256", {})) != required:
        raise RuntimeError("Final endpoint SHA256 coverage or epoch differs")
    if any(lookup.get(name) != value for name, value in endpoint["sha256"].items()):
        raise RuntimeError("Final endpoint SHA256 differs from sealed return files")
    schedules = {f"{number:03d}.json" for number in range(1, 101)}
    observed = {
        name.removeprefix("schedules/"): value
        for name, value in lookup.items()
        if name.startswith("schedules/") and name.endswith(".json")
    }
    if set(observed) != schedules or endpoint.get("schedule_sha256") != observed:
        raise RuntimeError("Final schedule coverage or SHA256 differs")
    preserved = {
        "migration-origin.json",
        "environment-origin.json",
        "weights/origin-migration.pt",
        "migration-resume.json",
    }
    if not preserved <= lookup.keys() or endpoint.get("migration", {}).get("origin") != origin:
        raise RuntimeError("Final endpoint migration origin evidence differs")
    if lookup["weights/origin-migration.pt"] != origin["checkpoint_sha256"]:
        raise RuntimeError("Preserved origin checkpoint SHA256 differs")
    return {"schema": "autodl-resume-return-v1", "dataset": dataset, "origin": origin, "files": files}


def preserve_resume_log(case, output):
    source, target = Path(case) / "resume.log", Path(output) / "migration-resume.log"
    if source.is_symlink() or not source.is_file() or target.is_symlink():
        raise RuntimeError("Migration log must be a regular file")
    digest = sha(source)
    if target.exists():
        if sha(target) != digest:
            raise RuntimeError("Existing preserved migration log conflicts")
        return
    with source.open("rb") as reader, target.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=8 * 1024**2)
    if sha(target) != digest:
        raise RuntimeError("Preserved migration log SHA256 differs")


def state(case, dataset, value, **extra):
    write(case / "status.json", {"dataset": dataset, "state": value, "time_unix": time.time(), **extra})


def wait_transfer(args, dataset, case):
    state(case, dataset, "WAITING_TRANSFER")
    manifest_path, marker = args.marker_dir / "manifest.json", args.marker_dir / f"{dataset}.DONE.json"
    while not manifest_path.is_file() or not marker.is_file():
        time.sleep(args.poll_seconds)
    if sha(manifest_path) != args.expected_manifest_sha256:
        raise RuntimeError("Global transfer manifest SHA256 differs from explicit expected digest")
    manifest = read(manifest_path)
    if manifest.get("schema") != "autodl-resume-transfer-v1":
        raise RuntimeError("Unexpected global transfer schema")
    rows = [row for row in manifest["files"] if row["dataset"] == dataset]
    if not rows or read(marker) != expected_marker(dataset, rows):
        raise RuntimeError("Dataset completion marker differs from exact manifest")
    return rows


def wait_environment(args, dataset, case):
    state(case, dataset, "VERIFYING", message="Waiting for explicit verified environment readiness")
    while not args.environment_ready.is_file():
        time.sleep(args.poll_seconds)
    ready = read(args.environment_ready)
    if (
        ready.get("state") != "READY"
        or ready.get("imports_verified") is not True
        or Path(ready["python"]).resolve() != args.python.resolve()
    ):
        raise RuntimeError("Environment readiness does not match requested Python")


def run_case(args, dataset):
    import fcntl

    case = args.run_root / dataset
    case.mkdir(parents=True, exist_ok=True)
    acquired = False
    try:
        with (case / "owner.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
            rows = wait_transfer(args, dataset, case)
            plan_path = args.target_root / ".migration" / f"{dataset}-plan.json"
            archive = args.target_root / ".migration" / f"{dataset}-inputs-and-code.tar"
            for path in (plan_path, archive):
                row = next(row for row in rows if Path(row["target"]) == path)
                if path.is_symlink() or path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
                    raise RuntimeError("Verified transfer metadata/archive changed")
            plan = read(plan_path)
            if plan.get("schema") != "migrated-crossdataset-transfer-v1":
                raise RuntimeError("Unexpected per-dataset transfer schema")
            origin = plan["origin"]
            banks = {row["target"] for row in rows if row["target"] not in (str(plan_path), str(archive))}
            members = [row for row in plan["files"] if row["target"] not in banks]
            state(case, dataset, "UNPACKING", origin=origin)
            with EXTRACTION_LOCK, (args.target_root / ".migration/extraction.lock").open("a") as extraction_lock:
                fcntl.flock(extraction_lock, fcntl.LOCK_EX)
                extract_verified(archive, args.target_root, members)
            wait_environment(args, dataset, case)
            output, seal = Path(origin["output"]), case / "relocation-seal.json"
            if args.target_root.resolve() not in output.resolve(strict=True).parents:
                raise RuntimeError("Origin output alias does not resolve into mirror")
            wrapper = Path(origin["repo"]) / "online_experiments/resume_migrated_crossdataset.py"
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES="0",
                HF_HUB_OFFLINE="1",
                TRANSFORMERS_OFFLINE="1",
                HF_HOME=str(args.source_root / "cache/huggingface"),
                PYTHONUNBUFFERED="1",
                OMP_NUM_THREADS="2",
                OPENBLAS_NUM_THREADS="2",
                MKL_NUM_THREADS="2",
            )
            env["PYTHONPATH"] = (
                str(Path(origin["repo"]) / ".research-deps/evidence-20261110/site")
                + os.pathsep
                + str(Path(origin["repo"]))
            )
            with (case / "resume.log").open("a") as log:
                if not seal.exists():
                    state(case, dataset, "VERIFYING", message="Checking every transferred file SHA256", origin=origin)
                    subprocess.run(
                        [
                            str(args.python),
                            str(wrapper),
                            "seal",
                            "--seal",
                            str(seal),
                            "--transfer-list",
                            str(plan_path),
                            "--source-root",
                            str(args.source_root),
                            "--target-root",
                            str(args.target_root),
                        ],
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        pass_fds=(lock.fileno(),),
                    )
                elif read(seal)["transfer_manifest_sha256"] != sha(plan_path):
                    raise RuntimeError("Existing relocation seal belongs to another transfer")
                command = [str(args.python), str(wrapper), "run", "--seal", str(seal)]
                write(case / "command.json", {"argv": command, "origin": origin})
                child = subprocess.Popen(
                    command,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    pass_fds=(lock.fileno(),),
                )
                while child.poll() is None:
                    summary = read(output / "summary.json")
                    state(
                        case,
                        dataset,
                        "TRAINING",
                        child_pid=child.pid,
                        completed_epochs=summary["completed_epochs"],
                        target_epochs=100,
                        origin_epoch=origin["epoch"],
                        origin=origin,
                    )
                    time.sleep(args.poll_seconds)
                if child.returncode:
                    raise RuntimeError(f"Migration trainer exited {child.returncode}; preserved, no automatic retry")
            preserve_resume_log(case, output)
            result = result_manifest(dataset, origin, args.target_root)
            write(case / "return-manifest.json", result)
            state(
                case,
                dataset,
                "SUCCEEDED",
                completed_epochs=100,
                origin=origin,
                return_manifest=str(case / "return-manifest.json"),
                return_manifest_sha256=sha(case / "return-manifest.json"),
            )
            return True
    except Exception:  # noqa: BLE001 - preserve failed cases while the other dataset continues.
        if acquired:
            state(
                case,
                dataset,
                "FAILED",
                traceback=traceback.format_exc(),
                policy="Preserve all files and checkpoints; no retry",
            )
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-root", type=Path, default=TARGET)
    parser.add_argument("--source-root", type=Path, default=SOURCE)
    parser.add_argument("--marker-dir", type=Path, default=TARGET / ".transfer-markers/job-20261114-two-lanes")
    parser.add_argument("--run-root", type=Path, default=TARGET / ".migration/job-20261114-resume")
    parser.add_argument("--environment-ready", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if (
        args.target_root != TARGET
        or args.source_root != SOURCE
        or len(args.expected_manifest_sha256) != 64
        or args.poll_seconds < 1
    ):
        parser.error("Use the authorized roots, explicit SHA256, and a positive poll interval")
    args.run_root.mkdir(parents=True, exist_ok=True)
    results = {}
    threads = [
        threading.Thread(
            target=lambda name: results.update({name: run_case(args, name)}), args=(dataset,), name=dataset
        )
        for dataset in ("sku110k", "voc0712")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return 0 if all(results.values()) and len(results) == 2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
