"""Finish the 18 frozen full-VisDrone repeats after the seed-zero owner releases its locks.

This external supervisor changes scheduling only. Frozen trainer sources, method
contracts, queue, epoch count and saved legacy checkpoints remain authoritative.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PHASE = "seed-repeats"
METADATA = (
    "args.json",
    "summary.json",
    "initialization.json",
    "environment.json",
    "metrics.json",
    "official-final.json",
)
BASELINES = ("frofa-c2-d01", "loffta-c2-n005", "cache-reuse-c1-int8", "vit-adapter-cache25")
SPATIAL = ("blank25", "detail25", "lowpass25", "transport25", "views25")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    from online_experiments.dataset_first_campaign import write as atomic_write

    atomic_write(path, value)


def sha(path):
    from online_experiments.dataset_first_campaign import digest

    return digest(path)


def status(args, state, message, **extra):
    from online_experiments.crossdataset_campaign import status as emit

    emit(args.job / PHASE, state, message, **extra)


def repeats(queue):
    """Use every existing deferred case exactly once, with paper controls first."""
    from online_experiments.dataset_first_campaign import partition

    first, deferred = partition(queue)
    expected = {(method, seed) for seed in (1, 2) for method in (*BASELINES, *SPATIAL)}
    actual = {(case["method"], case["seed"]) for case in deferred}
    if actual != expected or sum(bool(case.get("resume_original")) for case in deferred) != 1:
        raise ValueError("Deferred cases differ from the existing eight baseline and ten spatial repeats")
    legacy = next(case for case in deferred if case.get("resume_original"))
    if (legacy["method"], legacy["seed"]) != ("blank25", 1):
        raise ValueError("Only the preserved Blank25 seed-one checkpoint may use the original trainer")
    ordered = [case for case in deferred if case["method"] in BASELINES]
    ordered += [case for case in deferred if case["method"] in SPATIAL]
    return first, ordered


def output_for(args, case):
    job = args.original_job if case["resume_original"] else args.job
    return job / "train" / case["name"]


def inputs(args, case):
    job = args.original_job if case["resume_original"] else args.job
    provenance = read(job / "provenance.json")
    return job, Path(provenance["source_job"]), Path(provenance["cache_manifest"])


def command_for(args, case, action):
    from online_experiments.dataset_first_campaign import quantized_manifest, trainer_command

    job, source, cache = inputs(args, case)
    if case["resume_original"]:
        return [
            sys.executable,
            "-u",
            str(args.origin_root / "online_experiments/replication_suite.py"),
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
        ], args.origin_root
    quantized = quantized_manifest(job, cache, case, build=False)
    return trainer_command(job, source, cache, action, case, quantized), ROOT


def ready(args):
    """Configuration epochs=100 is never evidence that training finished."""
    phase = args.job / "dataset-first"
    paths = (phase / "completion.json", phase / "status.json")
    if not all(path.is_file() for path in paths):
        return False
    completion, current = (read(path) for path in paths)
    return (
        completion.get("state") == "PHASE_COMPLETE"
        and current.get("state") == "PHASE_COMPLETE"
        and completion.get("completed_seed_zero") == 10
        and completion.get("planned_seed_zero") == 10
        and completion.get("deferred_seed_repeats") == 18
    )


@contextlib.contextmanager
def campaign_locks(args):
    import fcntl

    from online_experiments.dataset_first_campaign import CoordinationBusy

    with contextlib.ExitStack() as stack:
        streams = []
        paths = (
            args.job / "dataset-first/campaign.lock",
            args.job.parent / "campaign.lock",
            args.original_job.parent / "campaign.lock",
        )
        for path in dict.fromkeys(paths):
            stream = stack.enter_context(path.open("a"))
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise CoordinationBusy(f"Seed-zero owner still holds {path}") from error
            streams.append(stream)
        yield tuple(stream.fileno() for stream in streams)


@contextlib.contextmanager
def wait_seed_zero(args):
    """Never send signals, adopt an active trainer, or race the three old owners."""
    from online_experiments.dataset_first_campaign import CoordinationBusy, live_trainers

    while True:
        if ready(args):
            manager = campaign_locks(args)
            try:
                descriptors = manager.__enter__()
            except CoordinationBusy:
                pass
            else:
                try:
                    if not ready(args) or live_trainers(args.job, args.original_job):
                        raise RuntimeError("A trainer exists after seed-zero completion; refusing duplicate GPU work")
                    args.campaign_lock_fds = descriptors
                    yield
                finally:
                    if hasattr(args, "campaign_lock_fds"):
                        del args.campaign_lock_fds
                    manager.__exit__(*sys.exc_info())
                return
        status(args, "WAITING_SEED0", "Waiting for all ten seed-zero cases and release of all three old locks")
        time.sleep(args.poll_seconds)


def plan(args):
    _, ordered = repeats(read(args.job / "queue.json"))
    return {
        "schema": "frozen-visdrone-seed-repeats-v1",
        "job": str(args.job),
        "original_job": str(args.original_job),
        "origin_root": str(args.origin_root),
        "original_queue_sha256": sha(args.job / "queue.json"),
        "provenance_sha256": sha(args.job / "provenance.json"),
        "original_provenance_sha256": sha(args.original_job / "provenance.json"),
        "coordinator_sha256": sha(Path(__file__)),
        "gpu_admission_sha256": sha(ROOT / "online_experiments/crossdataset_campaign.py"),
        "cases": ordered,
        "baseline_repeats": 8,
        "spatial_repeats": 10,
        "epochs": 100,
        "batch": 8,
        "train_images": 6471,
        "val_images": 548,
        "seed_zero_gate": "dataset-first PHASE_COMPLETE plus three exclusive locks and no live trainer",
        "scheduling": "baseline repeats first; GPU0 up to two resource-admitted stages across all new pipelines",
        "user_authorization": "2026-11-10 user requests all missing experiments, parallel where feasible, baselines first otherwise",
        "supersedes_scheduling_only": "Earlier wait-for-all-datasets rule; frozen deferred-seeds.json is retained unchanged",
        "legacy_resume": "Blank25-s1 continues the existing original checkpoint, optimizer, scheduler and RNG",
        "no_gpu_smoke_or_subset": True,
        "no_process_signals": True,
        "timing_conditions": "Shared GPU; observed costs, not controlled speed comparisons",
    }


def verify_plan(args, expected):
    if plan(args) != expected:
        raise RuntimeError("Frozen queue, source policy, or provenance changed during this supervisor")


def legacy_resume_contract(args, case):
    """Read the real CPU checkpoint before allowing a legacy continuation."""
    if not case["resume_original"]:
        return None
    output = output_for(args, case)
    configuration, summary = read(output / "args.json"), read(output / "summary.json")
    identity = (
        configuration.get("method"),
        configuration.get("seed"),
        configuration.get("epochs"),
        configuration.get("train_images"),
        configuration.get("val_images"),
        configuration.get("batch"),
    )
    if identity != ("blank25", 1, 100, 6471, 548, 8):
        raise RuntimeError("Preserved Blank25 checkpoint has a different full-data recipe")
    checkpoint = output / "weights/last.pt"
    if not checkpoint.is_file():
        raise RuntimeError("Preserved Blank25 checkpoint missing; never restart this case from scratch")
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    epoch = state["epoch"]
    if (
        state["args"] != configuration
        or epoch != summary["completed_epochs"]
        or state["summary"]["completed_epochs"] != epoch
        or len(state["rows"]) != epoch
        or not 1 <= epoch <= 100
        or state["query_images"] != epoch * 6471 * 25 // 100
    ):
        raise RuntimeError("Preserved checkpoint/summary epoch or query evidence is inconsistent")
    for name in ("optimizer", "scheduler", "cpu_rng", "cuda_rng", "model", "totals"):
        if name not in state:
            raise RuntimeError(f"Preserved checkpoint lacks {name}")
    switch = read(args.job / "original-switch.json")
    checksum = sha(checkpoint)
    if epoch == switch["preserved_epoch"] and checksum != switch["checkpoint_sha256"]:
        raise RuntimeError("Original checkpoint changed before its first continuation")
    return {
        "epoch": epoch,
        "checkpoint_sha256": checksum,
        "query_images": state["query_images"],
        "source_root": str(args.origin_root),
        "resume_from_scratch": False,
        "cuda_initialized_during_cpu_contract": torch.cuda.is_initialized(),
    }


def validate(args, case):
    output = output_for(args, case)
    if not (output / "DONE").is_file():
        raise RuntimeError(f"Missing DONE for {case['name']}; exit zero is insufficient")
    command, cwd = command_for(args, case, "validate")
    with (args.job / PHASE / f"validate-{case['name']}.log").open("a", encoding="utf-8") as stream:
        subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, check=True)


def archive(args, stage, paths):
    from online_experiments.replication_campaign import archive as save

    name = f"seed-repeats-{stage}"
    marker = args.job / f"archive-{name}.json"
    if not marker.exists():
        save(args.job, name, paths, True)
    result = read(marker)
    if not result.get("pushed"):
        from online_experiments.replication_campaign import publish_pending

        publish_pending(args.job)
        result = read(marker)
    write(args.job / PHASE / "archives" / f"{stage}.json", result)
    return result


def run(args):
    from online_experiments.crossdataset_campaign import gpu_admission, job_lock
    from online_experiments.dataset_first_campaign import live_trainers, process

    phase = args.job / PHASE
    phase.mkdir(parents=True, exist_ok=True)
    with job_lock(phase):
        expected = plan(args)
        plan_path = phase / "plan.json"
        if plan_path.exists() and read(plan_path) != expected:
            raise RuntimeError("Existing repeat plan changed; preserve the old run")
        write(plan_path, expected)
        write(phase / "worker.json", process(os.getpid()))
        (phase / "pid.txt").write_text(f"{os.getpid()}\n", encoding="utf-8")
        archive(args, "plan", [plan_path])
        with wait_seed_zero(args):
            # Child ownership is inherited through every trainer process, so a
            # supervisor crash cannot silently release the old campaign locks.
            completed = []
            for index, case in enumerate(expected["cases"], 1):
                verify_plan(args, expected)
                output = output_for(args, case)
                status(
                    args,
                    "RUNNING",
                    f"{index}/18 {case['name']}; paper controls precede spatial arms",
                    case=case["name"],
                    completed=len(completed),
                    planned=18,
                )
                if (output / "DONE").is_file():
                    validate(args, case)
                else:
                    if live_trainers(args.job, args.original_job):
                        raise RuntimeError("An existing trainer must finish before starting another repeat")
                    resumed = legacy_resume_contract(args, case)
                    if resumed is not None:
                        write(phase / f"resume-{case['name']}-{time.time_ns()}.json", resumed)
                    command, cwd = command_for(args, case, "train")
                    write(phase / f"command-{case['name']}.json", {"argv": command, "cwd": str(cwd)})
                    admission = SimpleNamespace(job=phase, dataset="visdrone-seed-repeats", gpu=0, min_free_mib=8192)
                    with gpu_admission(admission, f"VisDrone full repeat {case['name']}"):
                        status(
                            args,
                            "TRAINING",
                            f"{index}/18 {case['name']}; admitted GPU0 shared slot",
                            case=case["name"],
                            completed=len(completed),
                            planned=18,
                        )
                        with (phase / f"train-{case['name']}.log").open("a", encoding="utf-8") as stream:
                            subprocess.run(
                                command,
                                cwd=cwd,
                                stdout=stream,
                                stderr=subprocess.STDOUT,
                                check=True,
                                pass_fds=(admission.gpu_lock_fd, *args.campaign_lock_fds),
                            )
                    validate(args, case)
                from online_experiments.paper_baseline_campaign import table3_and_resources
                from online_experiments.replication_campaign import check_pairing

                completed.append(output)
                table3_and_resources(args.job, args.original_job)
                # The resumed case has a distinct source snapshot but must share
                # initial head, geometry schedule, and requested query masks.
                write(phase / "pairing.json", check_pairing(completed))
                archived = output
                if case["resume_original"]:
                    archived = phase / "resumed-original" / case["name"]
                    archived.mkdir(parents=True, exist_ok=True)
                    for name in METADATA:
                        target, content = archived / name, (output / name).read_bytes()
                        if target.exists() and target.read_bytes() != content:
                            raise RuntimeError("Legacy result metadata changed after copying")
                        target.write_bytes(content)
                archive(args, case["name"], [*(archived / name for name in METADATA), phase / "pairing.json"])
            verify_plan(args, expected)
            completion = {
                "state": "SUCCEEDED",
                "completed": len(completed),
                "planned": 18,
                "baseline_repeats": 8,
                "spatial_repeats": 10,
                "full_original_queue_complete": True,
            }
            write(phase / "completion.json", completion)
            result = archive(
                args,
                "complete",
                [
                    phase / "completion.json",
                    phase / "pairing.json",
                    *sorted((args.job / "tables").glob("*.csv")),
                    args.job / "comparison.txt",
                ],
            )
            status(
                args,
                "SUCCEEDED",
                "All 18 full-data repeats validated, archived and pushed",
                **{key: value for key, value in completion.items() if key != "state"},
                archive_commit=result["commit"],
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "status", "plan"))
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--original-job", type=Path)
    parser.add_argument("--origin-root", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    args.job = args.job.resolve()
    if args.action == "status":
        path = args.job / PHASE / "pipeline-status.json"
        print(json.dumps(read(path), indent=2) if path.is_file() else "Seed repeats supervisor has not started")
        return
    if not args.original_job or not args.origin_root or not 1 <= args.poll_seconds <= 60:
        parser.error("run/plan require original job/root and a polling interval from 1 to 60 seconds")
    args.original_job, args.origin_root = args.original_job.resolve(), args.origin_root.resolve()
    if args.action == "plan":
        print(json.dumps(plan(args), indent=2, allow_nan=False))
        return
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
    from online_experiments.crossdataset_campaign import Busy

    try:
        run(args)
    except Busy as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from error
    except Exception:
        phase = args.job / PHASE
        phase.mkdir(parents=True, exist_ok=True)
        failure = phase / f"failure-{time.time_ns()}.json"
        write(
            failure,
            {
                "traceback": traceback.format_exc(),
                "policy": "Preserve failure and checkpoint; stop rather than skip baseline controls",
            },
        )
        status(args, "FAILED", "Inspect preserved failure and case logs; no next case started", failure=str(failure))
        raise


if __name__ == "__main__":
    main()
