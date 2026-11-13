"""Prepare only the two already-authorized INT8 banks on an idle-priority CPU.

This sidecar does not train, predict, open images, allocate a GPU lease, or change
the frozen trainers. Later baseline training reuses the existing quantizer's seal
and retains its original build_seconds in the cold-cost accounting.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATASETS = ("voc0712", "sku110k")
METHOD = "cache-reuse-c1-int8"
METHODS = {"master-n", "yolo12-n", "frofa-c2-d01", "loffta-c2-n005", METHOD, "vit-adapter-cache25", "detail25"}
ANCHOR_SHA256 = "9564e537b32f18fe51a949888a6656a047c2a02797fd9ce863247e1e0b37d932"
MEMORY_RESERVE = 24 * 2**30


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def emit(args, state, message, **extra):
    record = {"state": state, "message": message, "time_unix": time.time(), **extra}
    write(args.job / "status.json", record)
    print(json.dumps(record, allow_nan=False), flush=True)


def inside_root(path):
    path = Path(path).resolve(strict=True)
    if ROOT != path and ROOT not in path.parents:
        raise ValueError("Input must be inside the existing research checkout: " + str(path))
    return path


def check_plan(plan, provenance, dataset):
    expected = {"voc0712": {"train": 16551, "test": 4952}, "sku110k": {"train": 8219, "val": 588, "test": 2936}}[dataset]
    if plan.get("schema") != "urgent-seven-method-single-seed-v1" or plan.get("dataset") != dataset:
        raise ValueError("Expected the frozen seven-method priority plan")
    if len(plan.get("methods", [])) != 7 or set(plan["methods"]) != METHODS:
        raise ValueError("The plan must contain exactly the seven authorized baselines/candidate")
    if any(plan.get(key) != value for key, value in {"seed": 0, "epochs": 100, "batch": 8, "imgsz": 640}.items()):
        raise ValueError("Only the existing seed0/final100/batch8/640 protocol is authorized")
    if not plan.get("final_checkpoint_only") or plan.get("counts") != expected:
        raise ValueError("The fixed final checkpoint and full official split counts are required")
    if provenance.get("dataset") != dataset or provenance.get("counts") != expected:
        raise ValueError("Provenance dataset/full split does not match the plan")
    if provenance.get("protocol", {}).get("epochs") != 100:
        raise ValueError("Frozen provenance must retain the original 100-epoch protocol")


def contracts(priority):
    result = {}
    for dataset in DATASETS:
        plan_path = inside_root(priority / dataset / "pipeline-plan.json")
        plan = read(plan_path)
        dataset_job = inside_root(plan["dataset_job"])
        provenance_path = dataset_job / "provenance.json"
        provenance = read(provenance_path)
        check_plan(plan, provenance, dataset)
        bindings = {
            "provenance_sha256": provenance_path,
            "cache_manifest_sha256": dataset_job / "cache/manifest.json",
            "manifest_sha256": Path(plan["manifest"]),
        }
        for key, path in bindings.items():
            if sha(path) != plan[key]:
                raise RuntimeError("Frozen priority input changed: " + str(path))
        if provenance["manifest"] != plan["manifest"] or provenance["manifest_sha256"] != plan["manifest_sha256"]:
            raise RuntimeError("Dataset manifest binding differs between plan and provenance")
        for name, digest in {**provenance["source_sha256"], **plan["source_sha256"]}.items():
            if sha(inside_root(ROOT / name)) != digest:
                raise RuntimeError("A frozen source anchor changed: " + name)
        expected_count = 1366 if dataset == "voc0712" else 1370
        if len(provenance["source_sha256"]) != expected_count:
            raise RuntimeError("Unexpected frozen source-anchor coverage")
        if dataset == "voc0712" and sha(provenance_path) != ANCHOR_SHA256:
            raise RuntimeError("VOC source anchor differs from the reviewed snapshot")
        if dataset == "sku110k":
            anchor = provenance["source_anchor"]
            if anchor["sha256"] != ANCHOR_SHA256 or sha(anchor["file"]) != ANCHOR_SHA256:
                raise RuntimeError("SKU must retain the identical reviewed VOC source anchor")
        cache = read(dataset_job / "cache/manifest.json")
        result[dataset] = {
            "dataset": dataset,
            "dataset_job": str(dataset_job),
            "pipeline_plan": str(plan_path),
            "pipeline_plan_sha256": sha(plan_path),
            "provenance_sha256": sha(provenance_path),
            "cache_manifest_sha256": sha(dataset_job / "cache/manifest.json"),
            "source_anchor_files": expected_count,
            "source_bank": cache["parts"]["train-v0"],
            "request": {"bits": 8, "quantile": 0.001, "source_sha256": cache["parts"]["train-v0"]["sha256"],
                        "cache_manifest_sha256": sha(dataset_job / "cache/manifest.json")},
        }
    return result


def active_int8_workers(dataset_job):
    found = []
    for process in Path("/proc").iterdir():
        if not process.name.isdecimal():
            continue
        try:
            if process.stat().st_uid != os.getuid():
                continue
            command = (process / "cmdline").read_bytes().decode(errors="surrogateescape").split("\0")
            scripts = {Path(value).name for value in command}
            if not scripts.intersection({"crossdataset_suite.py", "sku_official_loader_suite.py"}) or "train" not in command:
                continue
            pairs = {key: command[command.index(key) + 1] for key in ("--method", "--job") if key in command}
            if pairs.get("--method") == METHOD and Path(pairs.get("--job", "/")).resolve() == dataset_job:
                found.append(int(process.name))
        except (OSError, IndexError):
            continue
    return found


def memory_available():
    values = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return int(values["MemAvailable"].split()[0]) * 1024


@contextlib.contextmanager
def exclusive(path):
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def configure_cpu():
    if sys.platform != "linux":
        raise RuntimeError("Production preparation is Linux CPU-only")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[key] = "1"
    current = os.getpriority(os.PRIO_PROCESS, 0)
    if current < 10:
        os.nice(10 - current)
    subprocess.run(["ionice", "-c", "3", "-p", str(os.getpid())], check=True)
    return {"CUDA_VISIBLE_DEVICES": "", "cpu_threads": 1, "nice": os.getpriority(os.PRIO_PROCESS, 0),
            "io_priority": subprocess.check_output(["ionice", "-p", str(os.getpid())], text=True).strip(),
            "gpu_leases": 0, "teacher_queries": 0, "image_reads": 0, "training_runs": 0, "prediction_runs": 0}


def suite_for(dataset):
    if dataset == "sku110k":
        from online_experiments.sku_official_loader_suite import install

        return install()
    from online_experiments import crossdataset_suite

    return crossdataset_suite


def preserve(path, data):
    if path.is_file() and path.read_bytes() != data:
        raise RuntimeError("Refuse changed archived evidence: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def archive(args, dataset):
    from online_experiments.archive_assignment_intervention import commit_files
    from online_experiments.priority_single_seed_campaign import archive_lock
    from online_experiments.replication_campaign import push

    marker_path = args.job / dataset / "archive.json"
    if marker_path.is_file():
        return read(marker_path)
    source = args.job / dataset
    destination = ROOT / "research/experiments/priority-int8-20261113" / dataset
    names = []
    for filename in ("bank-manifest.json", "bank-complete.json", "cpu-evidence.json", "DONE.json"):
        target = destination / filename
        preserve(target, (source / filename).read_bytes())
        names.append(target.relative_to(ROOT).as_posix())
    readme = destination / "README.md"
    preserve(readme, (f"# {dataset}: CPU INT8 preparation\n\n"
        "Existing Cache and Reuse detection adaptation; seed0/final100 training protocol unchanged. "
        "The original quantized_cache(job,8) implementation produces the physical bank. "
        "Original FP16 banks and manifests are unchanged. Later training reuses this bank; "
        "the quantizer's original build_seconds remains charged to cold cost. "
        "No image decoding, GPU lease, teacher call, training, or prediction is performed here. "
        "Only small manifests and CPU evidence are versioned; physical banks are not committed.\n").encode())
    names.append(readme.relative_to(ROOT).as_posix())
    marker = "priority-int8-20261113:" + dataset
    block = (f"\n\n<!-- {marker} -->\n## {dataset}: existing INT8 baseline bank ready\n\n"
             f"CPU preparation evidence: [{dataset}]({readme.relative_to(ROOT / 'research').as_posix()}). "
             "No retraining or GPU inference; original compression build cost retained.\n")
    with archive_lock():
        commit = commit_files(ROOT, names, block, marker, readme.relative_to(ROOT).as_posix(),
                              f"research: prepare existing {dataset} INT8 bank on CPU")
        receipt = {"commit": commit, "pushed": False}
        try:
            receipt.update(pushed=True, pushed_head=push())
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            receipt["push_error"] = str(error)
        write(marker_path, receipt)
    return receipt


def prepare_one(args, dataset, contract, cpu):
    suite = suite_for(dataset)
    dataset_job = Path(contract["dataset_job"])
    suite.verify_provenance(dataset_job)
    suite.cache_manifest(dataset_job)
    destination = dataset_job / "cache/quantized-int8"
    evidence = args.job / dataset
    done = evidence / "DONE.json"
    if done.is_file():
        seal = read(done)
        for name, expected in seal["evidence_sha256"].items():
            if sha(evidence / name) != expected:
                raise RuntimeError("Previously completed CPU evidence changed")
        if sha(destination / "manifest.json") != seal["bank_manifest_sha256"]:
            raise RuntimeError("Previously prepared INT8 manifest changed")
        suite.quantized_cache(dataset_job, 8)  # Validated complete-bank reuse only.
        return {"dataset": dataset, "reused_sidecar_completion": True}
    while True:
        live = active_int8_workers(dataset_job)
        complete = (destination / "manifest.json").is_file() and (destination / "complete.json").is_file()
        available = memory_available()
        if live and not complete:
            emit(args, "WAITING_EXISTING_WRITER", "Existing INT8 trainer owns preparation; wait, then reuse its seal",
                 dataset=dataset, worker_pids=live)
        elif available < MEMORY_RESERVE:
            emit(args, "WAITING_MEMORY", "CPU preparation requires at least 24GiB MemAvailable", dataset=dataset,
                 mem_available_bytes=available)
        else:
            break
        time.sleep(args.poll_seconds)
    # New wrappers share this lock. Frozen trainers do not; the exact worker
    # check above deliberately happens before calling the unchanged quantizer.
    with exclusive(destination / ".cpu-preparation.lock"):
        if active_int8_workers(dataset_job) and not (destination / "complete.json").is_file():
            raise RuntimeError("INT8 trainer appeared during admission; preserve files and stop rather than duplicate writes")
        emit(args, "PREPARING_CPU", "Existing full INT8 baseline bank, low CPU and I/O priority", dataset=dataset)
        started = time.perf_counter()
        before = {"cache_manifest_sha256": sha(dataset_job / "cache/manifest.json"),
                  "source_bank_stat": suite.stamp(contract["source_bank"]["file"])}
        manifest = suite.quantized_cache(dataset_job, 8)
        elapsed = time.perf_counter() - started
        after = {"cache_manifest_sha256": sha(dataset_job / "cache/manifest.json"),
                 "source_bank_stat": suite.stamp(contract["source_bank"]["file"])}
        if before != after or manifest["request"] != contract["request"]:
            raise RuntimeError("Source bank/request changed; preserve output and do not retry")
        for filename in ("manifest.json", "complete.json"):
            shutil.copyfile(destination / filename, evidence / ("bank-manifest.json" if filename == "manifest.json" else "bank-complete.json"))
        record = {"schema": "priority-int8-cpu-preparation-v1", "contract": contract, "runtime": cpu,
                  "sidecar_sha256": sha(Path(__file__)), "cpu_wall_seconds_this_invocation": elapsed,
                  "quantizer_construction_seconds_this_invocation": manifest["construction_seconds_this_invocation"],
                  "cold_compression_build_seconds_retained": manifest["build_seconds"],
                  "cache_bytes": manifest["cache_bytes"], "source_before": before, "source_after": after,
                  "zero_gpu_scope": "No torch/CUDA import or GPU admission; existing quantizer reads only FP16 npy arrays",
                  "writer_lock_scope": "Advisory lock shared by new wrappers; legacy trainer checked before admission",
                  "completed_unix": time.time()}
        write(evidence / "cpu-evidence.json", record)
        write(done, {"bank_manifest_sha256": sha(destination / "manifest.json"),
                     "evidence_sha256": {name: sha(evidence / name) for name in
                                         ("bank-manifest.json", "bank-complete.json", "cpu-evidence.json")}})
    return {"dataset": dataset, "cache_bytes": manifest["cache_bytes"], "build_seconds": manifest["build_seconds"]}


def run(args):
    cpu = configure_cpu()
    args.job.mkdir(parents=True, exist_ok=True)
    with exclusive(args.job / "pipeline.lock"):
        contract = contracts(args.priority)
        plan = {"schema": "priority-int8-cpu-sidecar-v1", "priority": str(args.priority), "datasets": contract,
                "sidecar_sha256": sha(Path(__file__)), "bits": 8, "minimum_mem_available_bytes": MEMORY_RESERVE}
        plan_path = args.job / "plan.json"
        if plan_path.is_file() and read(plan_path) != plan:
            raise RuntimeError("Sidecar source/input changed; use a new job and preserve this job")
        write(plan_path, plan)
        write(args.job / "worker.json", {"pid": os.getpid(), "argv": sys.argv, "runtime": cpu,
                                        "started_unix": time.time(),
                                        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()})
        completed, failed = [], []
        for dataset in DATASETS:
            (args.job / dataset).mkdir(parents=True, exist_ok=True)
            try:
                item = prepare_one(args, dataset, contract[dataset], cpu)
                if args.publish:
                    item["archive"] = archive(args, dataset)
                completed.append(item)
                write(args.job / "completed.json", completed)
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                failure = {"dataset": dataset, "error": str(error), "traceback": traceback.format_exc()}
                write(args.job / dataset / "failure.json", failure)
                failed.append(failure)
                emit(args, "DATASET_FAILED", "Preserve partial files; no automatic retry; continue other dataset",
                     dataset=dataset, error=str(error))
        emit(args, "FAILED" if failed else "SUCCEEDED", "CPU bank preparation finished", completed=completed, failed=failed)
        if failed:
            raise RuntimeError("CPU preparation failed; inspect preserved per-dataset failure evidence")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "check", "status"))
    parser.add_argument("--priority", type=Path, default=ROOT / "runs/paper/priority/job-20261112-single-seed")
    parser.add_argument("--job", type=Path, default=ROOT / "runs/paper/priority-int8/job-20261113-cpu")
    parser.add_argument("--publish", action="store_true", help="Commit/push only small completed bank evidence")
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    args.priority, args.job = args.priority.resolve(), args.job.resolve()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    if ROOT not in args.job.parents:
        parser.error("--job must stay in this research checkout")
    if args.action == "status":
        path = args.job / "status.json"
        print(json.dumps(read(path) if path.is_file() else {"state": "NOT_STARTED"}, indent=2))
    elif args.action == "check":
        print(json.dumps({"state": "INPUTS_VERIFIED_ONLY", "datasets": contracts(args.priority)}, indent=2))
    else:
        run(args)


if __name__ == "__main__":
    main()
