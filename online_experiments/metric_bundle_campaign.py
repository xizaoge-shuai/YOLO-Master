"""Complete saved-prediction metrics for the two fixed single-seed queues.

This CPU sidecar never trains, infers, queries a backbone, or reserves a GPU.
It waits for each original trainer's sealed endpoint, validates it once, then
derives the complete metric bundle from the predictions already on disk.
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
TERMINAL = frozenset(("SUCCEEDED", "FAILED"))


class Busy(RuntimeError):
    """Another watcher or its surviving child still owns this exact job."""


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
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def emit(job, state, message, **extra):
    record = {"state": state, "message": message, "time_unix": time.time(), **extra}
    write(job / "pipeline-status.json", record)
    (job / "status.txt").write_text(f"{state}: {message}\n", encoding="utf-8")
    with (job / "pipeline.log").open("a", encoding="utf-8") as stream:
        stream.write(time.strftime("%Y-%m-%dT%H:%M:%S%z ") + state + ": " + message + "\n")


@contextlib.contextmanager
def lock(path, *, blocking=True):
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(stream.fileno(), flags)
        except BlockingIOError as error:
            raise Busy("Another metric sidecar/child already owns this job") from error
        yield stream.fileno()


def contract(args):
    plans = {}
    for dataset in DATASETS:
        path = args.priority / dataset / "pipeline-plan.json"
        value = read(path)
        methods = value["methods"]
        if (
            value["schema"] != "urgent-seven-method-single-seed-v1"
            or value["dataset"] != dataset
            or len(methods) != 7
            or set(methods) != set(METHODS)
            or set(methods[:2]) != DIRECT
            or methods[2:] != list(METHODS[2:])
            or value["seed"] != 0
            or value["epochs"] != 100
            or value["final_checkpoint_only"] is not True
        ):
            raise ValueError("Only the two authorized seven-method, seed0/final100 queues are permitted")
        expected = {"voc0712": (16551, 4952), "sku110k": (8219, 2936)}[dataset]
        if (value["counts"]["train"], value["counts"]["test"]) != expected:
            raise ValueError("Full official dataset counts differ")
        for name, digest in value["source_sha256"].items():
            if sha(ROOT / name) != digest:
                raise ValueError("Original queue source changed: " + name)
        if sha(value["manifest"]) != value["manifest_sha256"]:
            raise ValueError("Original full dataset manifest changed")
        plans[dataset] = {"plan_sha256": sha(path), "plan": value}
    sources = ("online_experiments/metric_bundle_campaign.py", "online_experiments/complete_metric_bundle.py")
    return {
        "schema": "single-pass-saved-prediction-metric-campaign-v1",
        "priority": str(args.priority),
        "plans": plans,
        "source_sha256": {name: sha(ROOT / name) for name in sources},
        "training_updates": 0,
        "inference_images": 0,
        "teacher_queries": 0,
        "gpu_used": False,
        "policy": "One CPU bundle per sealed seed0/final100 endpoint; no retraining, inference or automatic retries",
    }


def case_order(plans):
    for index in range(7):
        for dataset in DATASETS:
            yield dataset, plans[dataset]["plan"]["methods"][index]


def resolve_output(priority, dataset, plan, method):
    """Resolve completed reuse through the exact coordinator record when present."""
    new = (priority / dataset / "train" / f"{method}-s0").resolve()
    old = (Path(plan["dataset_job"]) / "train" / f"{method}-s0").resolve()
    path = priority / dataset / "completed-methods.json"
    if path.is_file():
        records = [item for item in read(path) if item["method"] == method]
        if len(records) > 1:
            raise ValueError("Duplicate completed method records")
        if records:
            output = Path(records[0]["output"]).resolve()
            if output not in ({new} if method in DIRECT else {new, old}):
                raise ValueError("Completed-method output escapes the two authorized exact run paths")
            return output
    if method not in DIRECT and (old / "DONE").is_file():
        return old
    return new


def ready(output, method):
    return (output / ("DONE.json" if method in DIRECT else "DONE")).is_file()


def validation_command(dataset, plan, method, output):
    if method in DIRECT:
        return [
            sys.executable,
            "-u",
            str(ROOT / "online_experiments/direct_baseline_suite.py"),
            "validate",
            "--output",
            str(output),
        ]
    script = "crossdataset_suite.py" if dataset == "voc0712" else "sku_official_loader_suite.py"
    return [
        sys.executable,
        "-u",
        str(ROOT / "online_experiments" / script),
        "validate",
        "--job",
        plan["dataset_job"],
        "--method",
        method,
        "--seed",
        "0",
        "--output",
        str(output),
    ]


def validate_identity(dataset, plan, method, output):
    config = read(output / "args.json")
    if config["method"] != method or config["seed"] != 0 or config["epochs"] != 100:
        raise ValueError("Completed endpoint method, seed or epochs differ")
    if method in DIRECT:
        if config["dataset"] != dataset or config["manifest_sha256"] != plan["manifest_sha256"]:
            raise ValueError("Native endpoint belongs to another full dataset")
    elif config["provenance_sha256"] != plan["provenance_sha256"]:
        raise ValueError("Cached endpoint belongs to another sealed dataset job")


def train_log(priority, dataset, plan, method, output):
    if output.parent.parent == (priority / dataset).resolve():
        candidates = [priority / dataset / f"train-{method}.log"]
    else:
        candidates = [
            Path(plan["dataset_job"]) / f"train-{method}-s0.log",
            Path(plan["dataset_job"]) / f"train-{method}.log",
        ]
    return next((path for path in candidates if path.is_file()), None)


def cpu_environment(dependencies):
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONPATH=str(dependencies) + os.pathsep + str(ROOT),
        GH_CONFIG_DIR=str(ROOT.parent / ".config/gh"),
    )
    return env


def execute(args, command, case, action):
    write(case / f"{action}-command.json", {"argv": command, "cuda_visible_devices": "", "threads": 1})
    with (case / f"{action}.log").open("a", encoding="utf-8") as stream:
        child = subprocess.Popen(
            command,
            cwd=ROOT,
            env=cpu_environment(args.dependencies),
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            pass_fds=(args.owner_fd,),
        )
        write(args.job / "active-child.json", {"pid": child.pid, "argv": command, "case": str(case), "action": action})
        result = child.wait()
    write(args.job / "last-child.json", {"pid": child.pid, "returncode": result, "case": str(case), "action": action})
    if result:
        raise RuntimeError(f"CPU {action} exited {result}; inspect {case / (action + '.log')}; no automatic retry")


def verify_bundle(output):
    if (output / "status.txt").read_text(encoding="utf-8").strip() != "SUCCEEDED":
        raise ValueError("Metric bundle is not sealed SUCCEEDED")
    seal = read(output / "completion.json")
    hashes = seal["output_sha256"]
    if not hashes or not {"metrics.json", "provenance.json", "resource-metrics.json"}.issubset(hashes):
        raise ValueError("Metric bundle lacks required outputs")
    for name, digest in hashes.items():
        path = (output / name).resolve()
        if path.parent != output.resolve() or sha(path) != digest:
            raise ValueError("Changed or escaping metric bundle artifact: " + name)
    return seal


def archive_case(args, dataset, method, case):
    from online_experiments.archive_assignment_intervention import commit_files
    from online_experiments.replication_campaign import push

    marker_path = case / "archive.json"
    with lock(ROOT / "runs/paper/priority/archive.lock"):
        if marker_path.is_file():
            result = read(marker_path)
            if args.publish and not result.get("pushed"):
                result.update(pushed=True, pushed_head=push())
                write(marker_path, result)
            return result
        bundle = case / "bundle"
        seal = verify_bundle(bundle)
        names = sorted(set(seal["output_sha256"]) | {"completion.json"})
        files = [bundle / name for name in names]
        if any(
            (path.suffix not in {".json", ".csv", ".npz"} and path.name != "status.txt")
            or path.name in {"predictions.json", "coco-predictions.json"}
            for path in files
        ):
            raise ValueError("Only lightweight derived CSV/JSON/NPZ bundles may be archived")
        if (
            any(path.stat().st_size > 50 * 1024**2 for path in files)
            or sum(path.stat().st_size for path in files) > 200 * 1024**2
        ):
            raise ValueError("Derived metric archive exceeds lightweight artifact budget")
        destination = ROOT / "research/experiments/single-pass-metrics-20261112" / dataset / f"{method}-s0"
        destination.mkdir(parents=True, exist_ok=True)
        paths = []
        for source in files:
            target = destination / source.name
            if target.exists() and sha(target) != sha(source):
                raise ValueError("Refuse changed archived metric bundle")
            if not target.exists():
                shutil.copyfile(source, target)
            paths.append(target.relative_to(ROOT).as_posix())
        reference = destination / "source-reference.json"
        value = {
            "source_case": str(case),
            "case_contract_sha256": sha(case / "case-contract.json"),
            "campaign_plan_sha256": sha(args.job / "pipeline-plan.json"),
        }
        if reference.is_file() and read(reference) != value:
            raise ValueError("Archived source reference changed")
        write(reference, value)
        paths.append(reference.relative_to(ROOT).as_posix())
        readme = destination / "README.md"
        text = (
            f"# {dataset}: {method}, seed 0 metric bundle\n\n"
            "CPU calculations from the original sealed final100 predictions. No training, image decoding, "
            "inference, teacher queries, GPU reservation, or additional seeds. Native dataset metrics remain "
            "primary; COCO-style auxiliary metrics and saved-output PR curves are explicitly labeled. "
            "Resource measurements are shared-GPU observations. Missing measurements stay unavailable.\n"
        )
        if readme.is_file() and readme.read_text(encoding="utf-8") != text:
            raise ValueError("Archived metric README changed")
        readme.write_text(text, encoding="utf-8")
        paths.append(readme.relative_to(ROOT).as_posix())
        marker = f"single-pass-metrics-20261112:{dataset}:{method}"
        block = f"\n\n<!-- {marker} -->\n## Single-pass metrics: {dataset}, {method}\n\nFull-test final100 saved predictions scored once on CPU; [bundle]({readme.relative_to(ROOT / 'research').as_posix()}). No extra training or inference.\n"
        commit = commit_files(
            ROOT,
            paths,
            block,
            marker,
            readme.relative_to(ROOT).as_posix(),
            f"research: complete saved-prediction metrics {dataset} {method}",
        )
        result = {"commit": commit, "pushed": False}
        write(marker_path, result)
        if args.publish:
            try:
                result.update(pushed=True, pushed_head=push())
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                result["push_error"] = str(error)
            write(marker_path, result)
        return result


def metric_command(plan, method, source, bundle):
    command = [
        sys.executable,
        "-u",
        str(ROOT / "online_experiments/complete_metric_bundle.py"),
        "--manifest",
        plan["manifest"],
        "--run",
        str(source),
        "--output",
        str(bundle),
    ]
    if method not in DIRECT:
        command += ["--provenance", str(Path(plan["dataset_job"]) / "provenance.json")]
    return command


def process_case(args, dataset, method, plan, source):
    case = args.job / dataset / method
    case.mkdir(parents=True, exist_ok=True)
    value = {
        "dataset": dataset,
        "method": method,
        "run": str(source),
        "manifest": plan["manifest"],
        "manifest_sha256": plan["manifest_sha256"],
        "endpoint_sha256": sha(source / "endpoint.json"),
        "done_sha256": sha(source / ("DONE.json" if method in DIRECT else "DONE")),
        "scorer_sha256": sha(ROOT / "online_experiments/complete_metric_bundle.py"),
        "seed": 0,
        "epochs": 100,
        "training_updates": 0,
        "inference_images": 0,
        "teacher_queries": 0,
    }
    path = case / "case-contract.json"
    if path.exists() and read(path) != value:
        raise ValueError("Input endpoint changed; preserve the original bundle and use a new job version")
    write(path, value)
    validated = case / "validated.json"
    if validated.exists():
        if read(validated)["case_contract_sha256"] != sha(path):
            raise ValueError("Original validation receipt differs")
    else:
        execute(args, validation_command(dataset, plan, method, source), case, "validate")
        validate_identity(dataset, plan, method, source)
        write(validated, {"case_contract_sha256": sha(path), "time_unix": time.time()})
    bundle = case / "bundle"
    if not (bundle / "completion.json").is_file():
        command = metric_command(plan, method, source, bundle)
        log = train_log(args.priority, dataset, plan, method, source)
        if log is not None:
            command += ["--train-log", str(log)]
        execute(args, command, case, "metrics")
    verify_bundle(bundle)
    archive = archive_case(args, dataset, method, case)
    write(
        case / "pipeline-status.json",
        {"state": "SUCCEEDED", "run": str(source), "archive": archive, "time_unix": time.time()},
    )


def run(args):
    args.job.mkdir(parents=True, exist_ok=True)
    with lock(args.job / "pipeline.lock", blocking=False) as args.owner_fd:
        plan = contract(args)
        path = args.job / "pipeline-plan.json"
        if path.exists() and read(path) != plan:
            raise ValueError("Metric sidecar plan/source changed; use a new versioned job")
        write(path, plan)
        write(args.job / "worker.json", {"pid": os.getpid(), "argv": sys.argv, "started_unix": time.time()})
        try:
            os.nice(10)
        except (AttributeError, PermissionError):
            pass
        last_overview = None
        while True:
            records = []
            for dataset, method in case_order(plan["plans"]):
                case = args.job / dataset / method
                state_path = case / "pipeline-status.json"
                previous = read(state_path) if state_path.is_file() else {}
                state = previous.get("state")
                if state in TERMINAL:
                    records.append({"dataset": dataset, "method": method, **previous})
                    continue
                value = plan["plans"][dataset]["plan"]
                try:
                    source = resolve_output(args.priority, dataset, value, method)
                    if ready(source, method):
                        emit(
                            args.job,
                            "RUNNING_CPU",
                            f"{dataset}/{method}: derive saved-prediction metrics",
                            dataset=dataset,
                            method=method,
                        )
                        write(state_path, {"state": "RUNNING_CPU", "run": str(source), "time_unix": time.time()})
                        process_case(args, dataset, method, value, source)
                    else:
                        upstream_path = args.priority / dataset / "pipeline-status.json"
                        upstream = read(upstream_path) if upstream_path.is_file() else {}
                        if upstream.get("state") in {"FAILED", "SUCCEEDED"}:
                            raise RuntimeError(
                                "Original queue is terminal but this required endpoint has no completion seal"
                            )
                        write(state_path, {"state": "WAITING_RESULTS", "run": str(source), "time_unix": time.time()})
                except (OSError, ValueError, TypeError, KeyError, RuntimeError, subprocess.SubprocessError):
                    write(
                        case / f"failure-{time.time_ns()}.json",
                        {
                            "traceback": traceback.format_exc(),
                            "time_unix": time.time(),
                            "policy": "Preserve partial evidence; continue other cases; no automatic training, inference or retry",
                        },
                    )
                    write(
                        state_path,
                        {
                            "state": "FAILED",
                            "time_unix": time.time(),
                            "message": "Inspect preserved case failure/log; other methods continue",
                        },
                    )
                records.append({"dataset": dataset, "method": method, **read(state_path)})
            write(args.job / "cases.json", records)
            completed = sum(item["state"] == "SUCCEEDED" for item in records)
            failed = sum(item["state"] == "FAILED" for item in records)
            if completed + failed == 14:
                emit(
                    args.job,
                    "FAILED" if failed else "SUCCEEDED",
                    "All available metric bundles processed",
                    completed=completed,
                    failed=failed,
                    planned=14,
                )
                return
            overview = (completed, failed)
            if overview != last_overview:
                emit(
                    args.job,
                    "WAITING_RESULTS",
                    "CPU sidecar waiting for original sealed final endpoints",
                    completed=completed,
                    failed=failed,
                    planned=14,
                )
                last_overview = overview
            if args.once:
                return
            time.sleep(args.poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "status"))
    parser.add_argument("--priority", type=Path, default=ROOT / "runs/paper/priority/job-20261112-single-seed")
    parser.add_argument("--job", type=Path, default=ROOT / "runs/paper/metric-bundles/job-20261112-single-pass")
    parser.add_argument("--dependencies", type=Path, default=ROOT / ".research-deps/evidence-20261110/site")
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument(
        "--once", action="store_true", help="Scan all14 cases once without waiting for future endpoints"
    )
    args = parser.parse_args()
    args.job, args.priority, args.dependencies = (
        args.job.resolve(),
        args.priority.resolve(),
        args.dependencies.resolve(),
    )
    if args.action == "status":
        path = args.job / "pipeline-status.json"
        print(json.dumps(read(path) if path.is_file() else {"state": "NOT_STARTED"}, indent=2))
        cases = args.job / "cases.json"
        if cases.is_file():
            for item in read(cases):
                print(item["dataset"], item["method"], item["state"])
        return
    if not 10 <= args.poll_seconds <= 60:
        parser.error("poll-seconds must be between10 and60")
    try:
        run(args)
    except Busy as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from error
    except Exception:
        args.job.mkdir(parents=True, exist_ok=True)
        write(
            args.job / f"failure-{time.time_ns()}.json", {"traceback": traceback.format_exc(), "time_unix": time.time()}
        )
        emit(args.job, "FAILED", "Metric sidecar stopped; original training queues are unaffected")
        raise


if __name__ == "__main__":
    main()
