"""Full-split, paired-budget screening of extra cache views and a small RGB branch."""

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "online_experiments"))

METHODS = ("transport25", "blank25", "views25", "detail25", "lowpass25")
SOURCE = ROOT / "runs/paper/full-visdrone/job-start-eDmH0odX"
PARITY_JOB = ROOT / "runs/paper/visdrone-eval/job-full-20260927"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha(path):
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_identity():
    names = subprocess.check_output(
        ["git", "ls-files", "--", "*.py", "online_experiments/*.sh"], cwd=ROOT, text=True
    ).splitlines()
    return {name: sha(ROOT / name) for name in names}


def verify_source(job):
    provenance = read(job / "provenance.json")
    expected = provenance["source_sha256"]
    for name, value in expected.items():
        if not (ROOT / name).is_file() or sha(ROOT / name) != value:
            raise RuntimeError(f"Training source changed after protocol snapshot: {name}")
    if sha(ROOT / "research/DETAIL_PROBE_PLAN_20261019.md") != provenance["protocol_sha256"]:
        raise RuntimeError("Experiment protocol changed after snapshot")


def telemetry():
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=15, check=False)
    except (subprocess.TimeoutExpired, OSError) as error:
        return {"time_unix": time.time(), "error": str(error)}
    return {"time_unix": time.time(), "returncode": result.returncode, "gpu": result.stdout.strip()}


def check_counts(epoch, seen, n, queries):
    if seen != n or queries != epoch * n * 25 // 100:
        raise RuntimeError("Incomplete image coverage or mismatched 25-percent query budget")


def compare_schedules(outputs):
    reference = None
    for output in outputs:
        current = read(Path(output) / "summary.json")["schedule_sha256"]
        if not current or (reference is not None and current != reference):
            raise RuntimeError("Augmentation/query schedules differ across paired arms")
        reference = current
    return reference


def validate_completed(output, smoke):
    """Do not trust a DONE marker without its complete protocol evidence."""
    output = Path(output)
    args, summary = read(output / "args.json"), read(output / "summary.json")
    rows = read(output / "metrics.json")
    epochs, n, nv = (1, 17, 8) if smoke else (100, 6471, 548)
    if (args["epochs"], summary["completed_epochs"], args["train_images"], args["val_images"]) != (
        epochs,
        epochs,
        n,
        nv,
    ):
        raise RuntimeError("Incomplete or incompatible completed run")
    if [row["epoch"] for row in rows] != list(range(1, epochs + 1)):
        raise RuntimeError("Completed epoch records are missing or duplicated")
    check_counts(epochs, n, n, summary["query_images"])
    schedules = summary["schedule_sha256"]
    if len(schedules) != epochs or any(sha(output / "schedules" / name) != value for name, value in schedules.items()):
        raise RuntimeError("Completed augmentation schedules changed or incomplete")
    if not smoke:
        official = read(output / "official-final.json")
        if len(official["prediction_sha256"]) != nv or official["checkpoint_sha256"] != sha(output / "weights/last.pt"):
            raise RuntimeError("Official result checkpoint or prediction coverage mismatch")
        if summary["official_final"] != official["metrics"]:
            raise RuntimeError("Official summary mismatch")
        for name, value in official["prediction_sha256"].items():
            if sha(output / "predictions/last" / f"{name}.txt") != value:
                raise RuntimeError("Scored predictions changed after final evaluation")


def publish_epoch_evidence(output, state):
    """Recover sidecars from the atomic checkpoint, including a final-epoch crash."""
    if state["summary"]["completed_epochs"] != state["epoch"] or [r["epoch"] for r in state["rows"]] != list(
        range(1, state["epoch"] + 1)
    ):
        raise RuntimeError("Checkpoint epoch evidence is incomplete")
    dump(output / "metrics.json", state["rows"])
    dump(output / "summary.json", state["summary"])


def score(source, output):
    from run_visdrone_reference_eval import load_cases, load_ground_truth
    from visdrone_reference_eval import REFERENCE_COMMIT, evaluate_raw

    evidence = reference_evidence()
    records, annotations, manifest_hash = load_ground_truth(source)
    cases, predictions = load_cases(output / "predictions/last", records)
    started = time.perf_counter()
    metrics = evaluate_raw(cases)
    result = {
        "metrics": metrics,
        "seconds": time.perf_counter() - started,
        "reference_commit": REFERENCE_COMMIT,
        "reference_parity_evidence": evidence,
        "reference_port_sha256": sha(ROOT / "online_experiments/visdrone_reference_eval.py"),
        "dataset_manifest_sha256": manifest_hash,
        "annotation_sha256": annotations,
        "prediction_sha256": predictions,
        "checkpoint_sha256": sha(output / "weights/last.pt"),
        "selection": "fixed final epoch; full 548-image development validation, not test-dev",
    }
    dump(output / "official-final.json", result)
    return result


def reference_evidence():
    """Tie final scores to the previously checked native official-toolkit parity."""
    previous = read(PARITY_JOB / "provenance.json")
    for name, value in previous["evaluator_source_hashes"].items():
        if sha(ROOT / "online_experiments" / name) != value:
            raise RuntimeError("Reference evaluator changed; native parity must be re-established")
    parity = read(PARITY_JOB / "parity.json")
    if len(parity) != 2 or any(
        not row["preprocessed_arrays_exact"] or not 0 <= row["max_absolute_AP_point_error"] <= 1e-9 for row in parity
    ):
        raise RuntimeError("Missing successful native-reference parity")
    return {
        "job": str(PARITY_JOB),
        "parity_sha256": sha(PARITY_JOB / "parity.json"),
        "provenance_sha256": sha(PARITY_JOB / "provenance.json"),
        "reference_commit": previous["official_reference_commit"],
        "checks": parity,
    }


def train(a):
    import importlib.metadata
    import resource
    from types import SimpleNamespace

    import torch

    from online_experiments.detail_probe_data import DetailBatches, cost_for_method
    from online_experiments.detail_probe_model import DetailDetector
    from online_experiments.full_visdrone_train import atomic_checkpoint, save_predictions, validate
    from online_experiments.fusion_probe import configure_fusion
    from scripts import d1_train_cached_detector as d

    verify_source(a.job)
    output = a.output
    if (output / "DONE").exists():
        raise RuntimeError("Refusing to overwrite a completed run")
    output.mkdir(parents=True, exist_ok=True)
    (output / "weights").mkdir(exist_ok=True)
    torch.set_num_threads(4)
    d.seed_everything(a.seed)
    started = time.perf_counter()
    batches = DetailBatches(
        a.source,
        a.cache,
        a.method,
        a.seed,
        train_limit=17 if a.smoke else None,
        val_limit=8 if a.smoke else None,
    )
    n, nv = len(batches.train), len(batches.val)
    if (n, nv) != ((17, 8) if a.smoke else (6471, 548)):
        raise RuntimeError(f"Unexpected data split {(n, nv)}")
    d.seed_everything(a.seed)
    mode = {"detail25": "rgb", "lowpass25": "lowpass"}.get(a.method, "none")
    model = DetailDetector(in_channels=(384,) * 3, nc=10, imgsz=640, epochs=a.epochs, detail_mode=mode).cuda()
    configure_fusion(model, "mean_value")
    model.detect.max_det = 500
    criterion = d.E2EDetectLoss(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.epochs, eta_min=0.00001)
    args = {
        "method": a.method,
        "seed": a.seed,
        "epochs": a.epochs,
        "batch": a.batch,
        "train_images": n,
        "val_images": nv,
        "smoke": a.smoke,
        "query_pct": 25,
        "fusion": "mean_value",
        "imgsz": 640,
        "max_det": 500,
        "conf": 0.001,
        "lr": 0.001,
        "weight_decay": 0.0005,
        "detail_mode": mode,
        "validation_interval": 10,
        "dataset_manifest_sha256": sha(a.source / "dataset-manifest.json"),
        "cache_manifest_sha256": sha(a.cache),
        "provenance_sha256": sha(a.job / "provenance.json"),
        "primary_endpoint": "final official-reference AP, no best-epoch selection",
        "timing_conditions": "shared GPU0; not a controlled speed benchmark",
    }
    if (output / "args.json").exists() and read(output / "args.json") != args:
        raise RuntimeError("Resume configuration changed")
    dump(output / "args.json", args)
    dump(
        output / "environment.json",
        {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "transformers")},
            "gpu": torch.cuda.get_device_name(),
            "telemetry": telemetry(),
        },
    )
    from online_experiments.assignment_intervention import state_hash

    dump(
        output / "initialization.json",
        {
            "head_state_sha256": state_hash(
                SimpleNamespace(
                    state_dict=lambda: {
                        k: v for k, v in model.state_dict().items() if not k.startswith("detail_branch.")
                    }
                )
            ),
            "parameters": sum(p.numel() for p in model.parameters()),
            "detail_parameters": model.detail_parameter_count,
        },
    )
    begin, rows, totals = (
        0,
        [],
        {
            "train": 0.0,
            "val": 0.0,
            "wall": 0.0,
            "raw_train_seconds": 0.0,
            "raw_val_seconds": 0.0,
            "raw_train_images": 0,
            "raw_val_images": 0,
        },
    )
    last = output / "weights/last.pt"
    if last.exists():
        state = torch.load(last, map_location="cpu", weights_only=False)
        if state["args"] != args:
            raise RuntimeError("Checkpoint configuration mismatch")
        publish_epoch_evidence(output, state)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        begin, rows, totals = state["epoch"], state["rows"], state["totals"]
        batches.query_images, batches.query_calls = state["query_images"], state["query_calls"]
        batches.raw_image_seconds = totals["raw_train_seconds"] + totals["raw_val_seconds"]
        batches.raw_image_images = totals["raw_train_images"] + totals["raw_val_images"]
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        print(f"RESUME {a.method} epoch={begin + 1}", flush=True)
    prior_wall = totals["wall"]
    costs = cost_for_method(read(a.cache), a.method)
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(begin, a.epochs):
        model.train()
        torch.cuda.synchronize()
        tick = time.perf_counter()
        loss_sum, seen = 0.0, 0
        raw_tick, raw_count = batches.raw_image_seconds, batches.raw_image_images
        for batch in batches.training(epoch, a.batch, output):
            features, targets = d.move_batch(batch, torch.device("cuda:0"))
            optimizer.zero_grad(set_to_none=True)
            vector, _ = criterion(model.training_predictions(features), targets)
            loss = vector.sum()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss epoch={epoch + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            loss_sum += float(loss.detach())
            seen += len(batch["sample_ids"])
        torch.cuda.synchronize()
        seconds = time.perf_counter() - tick
        check_counts(epoch + 1, seen, n, batches.query_images)
        if any(p.requires_grad or p.grad is not None for p in batches.extractor.model.parameters()):
            raise RuntimeError("Teacher is no longer frozen")
        row = {
            "epoch": epoch + 1,
            "train_loss": loss_sum / seen,
            "train_seconds": seconds,
            "query_images": batches.query_images,
            "query_calls": batches.query_calls,
            "lr": optimizer.param_groups[0]["lr"],
            "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
        totals["train"] += seconds
        totals["raw_train_seconds"] += batches.raw_image_seconds - raw_tick
        totals["raw_train_images"] += batches.raw_image_images - raw_count
        if (epoch + 1) % 10 == 0 or epoch + 1 == a.epochs:
            tick = time.perf_counter()
            raw_tick, raw_count = batches.raw_image_seconds, batches.raw_image_images
            metrics, val_loss, exported = validate(model, batches, criterion, a.batch)
            torch.cuda.synchronize()
            val_seconds = time.perf_counter() - tick
            totals["val"] += val_seconds
            totals["raw_val_seconds"] += batches.raw_image_seconds - raw_tick
            totals["raw_val_images"] += batches.raw_image_images - raw_count
            row.update(val_loss=val_loss, diagnostic_AP=100 * metrics["metrics/mAP50-95(B)"], val_seconds=val_seconds)
            save_predictions(output / "predictions/last", exported)
        row["peak_vram_gib"] = torch.cuda.max_memory_allocated() / 2**30
        if not all(math.isfinite(float(value)) for value in row.values()):
            raise RuntimeError("Nonfinite training metric")
        rows.append(row)
        scheduler.step()
        totals["wall"] = prior_wall + time.perf_counter() - started
        state = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rows": rows,
            "args": args,
            "totals": totals.copy(),
            "query_images": batches.query_images,
            "query_calls": batches.query_calls,
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
        }
        with (output / "telemetry.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(telemetry()) + "\n")
        summary = {
            **args,
            **costs,
            "completed_epochs": epoch + 1,
            "train_seconds": totals["train"],
            "validation_seconds": totals["val"],
            "wall_seconds": prior_wall + time.perf_counter() - started,
            "query_images": batches.query_images,
            "query_calls": batches.query_calls,
            "peak_vram_gib": max(r["peak_vram_gib"] for r in rows),
            "peak_host_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "schedule_sha256": {p.name: sha(p) for p in sorted((output / "schedules").glob("*.json"))},
        }
        summary["build_plus_wall_seconds"] = summary["wall_seconds"] + costs["cache_build_charged_seconds"]
        summary["data_prepare_charged_seconds"] = batches.meta["prepare_seconds"]
        summary.update({key: value for key, value in totals.items() if key.startswith("raw_")})
        summary["raw_loading_note"] = "CPU image read/decode/letterbox time; already included in train/val wall times"
        state["summary"] = summary
        # The checkpoint commits one complete epoch; sidecars can always be restored from it.
        atomic_checkpoint(last, state)
        publish_epoch_evidence(output, state)
        print(json.dumps(row), flush=True)
    if not a.smoke:
        official = score(a.source, output)
        summary = read(output / "summary.json")
        summary["official_final"] = official["metrics"]
        summary["official_evaluation_seconds"] = official["seconds"]
        summary["wall_seconds"] = prior_wall + time.perf_counter() - started
        summary["build_plus_wall_seconds"] = summary["wall_seconds"] + costs["cache_build_charged_seconds"]
        dump(output / "summary.json", summary)
    verify_source(a.job)
    validate_completed(output, a.smoke)
    (output / "DONE").write_text("SUCCEEDED\n")


def push_verified_archive(root, archive_commit):
    """Push the current descendant tip, never rewind a branch when retrying an old archive."""
    from online_experiments.archive_assignment_intervention import git

    branch = git(root, "branch", "--show-current")
    if not branch:
        raise RuntimeError("Cannot publish an archive from a detached HEAD")
    head = git(root, "rev-parse", "HEAD")
    subprocess.run(["git", "merge-base", "--is-ancestor", archive_commit, head], cwd=root, check=True)
    subprocess.run(["git", "push", "origin", f"{head}:refs/heads/{branch}"], cwd=root, check=True)
    remote = git(root, "ls-remote", "origin", f"refs/heads/{branch}").split()[0]
    if remote != head:
        raise RuntimeError("Push verification mismatch")
    return head


def archive(job, stage, paths, push):
    from online_experiments.archive_assignment_intervention import commit_files

    destination = ROOT / "research/experiments/detail-probe" / job.name / stage
    destination.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for source in paths:
        relative = source.relative_to(job)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise RuntimeError(f"Archive conflict: {target}")
        shutil.copyfile(source, target)
        artifacts.append(str(target.relative_to(ROOT)))
    readme = destination / "README.md"
    content = (
        f"# Detail probe: {stage}\n\nSource job: `{job}`.\n\n"
        "Exploratory full-split seed-0 campaign; shared GPU timings. Smoke is not an accuracy result. "
        "No novelty or successful enhancement claim is implied. Cache and weights excluded.\n"
    )
    readme.write_text(content)
    artifacts.append(str(readme.relative_to(ROOT)))
    marker = f"detail-probe {job.name} {stage}"
    block = f"\n\n## {marker}\n\nArchived [{stage}]({readme.relative_to(ROOT / 'research').as_posix()}). "
    block += "Frozen 25% query screening; official final AP only after full training. Shared-GPU costs provisional.\n"
    commit = commit_files(ROOT, artifacts, block, marker, str(readme.relative_to(ROOT)), f"research: archive {marker}")
    record = {"commit": commit, "stage": stage, "pushed": False}
    if push:
        record["pushed_head"] = push_verified_archive(ROOT, commit)
        record["pushed"] = True
    dump(job / f"archive-{stage}.json", record)


def light_files(output):
    return [
        output / name
        for name in (
            "args.json",
            "initialization.json",
            "environment.json",
            "summary.json",
            "metrics.json",
            "official-final.json",
        )
        if (output / name).is_file()
    ]


def comparison(job):
    rows = []
    for method in METHODS:
        output = job / "train" / f"{method}-s0"
        if not (output / "DONE").exists():
            continue
        summary = read(output / "summary.json")
        rows.append(
            {
                "method": method,
                **{k: summary["official_final"][k] for k in ("AP", "AP50", "AP75", "AR500")},
                "train_s": summary["train_seconds"],
                "build_wall_s": summary["build_plus_wall_seconds"],
                "cache_GiB": summary["cache_bytes"] / 2**30,
                "query_images": summary["query_images"],
                "VRAM_GiB": summary["peak_vram_gib"],
            }
        )
    if rows:
        with (job / "comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        text = "Final epoch official-reference AP (0-100), seed 0. Shared GPU timings; no speedup claim.\n"
        text += "method AP AP50 AP75 AR500 train_s build+wall_s cache_GiB queries VRAM_GiB\n"
        for row in rows:
            text += (
                " ".join(f"{value:.4f}" if isinstance(value, float) else str(value) for value in row.values()) + "\n"
            )
        (job / "comparison.txt").write_text(text)


def campaign(a):
    job = a.job
    job.mkdir(parents=True, exist_ok=True)
    reference_evidence()
    if (job / "provenance.json").exists():
        verify_source(job)
    else:
        names = source_identity()
        required = [
            f"online_experiments/{name}"
            for name in ("detail_probe.py", "detail_probe_data.py", "detail_probe_model.py")
        ]
        if not all(name in names for name in required):
            raise RuntimeError("Commit the new source before starting a campaign")
        if subprocess.check_output(["git", "diff", "HEAD", "--name-only", "--", *names], cwd=ROOT, text=True).strip():
            raise RuntimeError("Commit source changes before running")
        dump(
            job / "provenance.json",
            {
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "source_sha256": names,
                "source_job": str(a.source),
                "telemetry": telemetry(),
                "protocol_sha256": sha(ROOT / "research/DETAIL_PROBE_PLAN_20261019.md"),
            },
        )
    try:
        for phase in ("smoke", "train"):
            cache_dir = job / ("smoke-cache" if phase == "smoke" else "cache")
            status = f"RUNNING {phase} cache"
            (job / "status.txt").write_text(status + "\n")
            print(status, flush=True)
            if not (cache_dir / "manifest.json").exists():
                command = [
                    sys.executable,
                    "-u",
                    __file__,
                    "build",
                    "--job",
                    str(job),
                    "--source",
                    str(a.source),
                    "--cache",
                    str(cache_dir),
                    "--batch",
                    str(a.batch),
                ]
                if phase == "smoke":
                    command.append("--smoke")
                with (job / f"build-{phase}.log").open("a") as log:
                    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            if phase == "train" and not (job / "archive-cache.json").exists():
                archive(
                    job,
                    "cache",
                    [
                        job / "provenance.json",
                        cache_dir / "manifest.json",
                        cache_dir / "integrity.json",
                        cache_dir / "complete.json",
                    ],
                    a.push,
                )
            outputs = []
            for method in METHODS:
                output = job / phase / f"{method}-s0"
                outputs.append(output)
                (job / "status.txt").write_text(f"RUNNING {phase} {method}\n")
                if not (output / "DONE").exists():
                    command = [
                        sys.executable,
                        "-u",
                        __file__,
                        "train",
                        "--job",
                        str(job),
                        "--source",
                        str(a.source),
                        "--cache",
                        str(cache_dir / "manifest.json"),
                        "--output",
                        str(output),
                        "--method",
                        method,
                        "--epochs",
                        "1" if phase == "smoke" else "100",
                        "--batch",
                        str(a.batch),
                    ]
                    if phase == "smoke":
                        command.append("--smoke")
                    with (job / f"{phase}-{method}.log").open("a") as log:
                        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                validate_completed(output, phase == "smoke")
                if phase == "train":
                    comparison(job)
                    if not (job / f"archive-{method}.json").exists():
                        archive(job, method, light_files(output), a.push)
                print(f"COMPLETED {phase} {method}", flush=True)
            schedules = compare_schedules(outputs)
            heads = {read(path / "initialization.json")["head_state_sha256"] for path in outputs}
            if len(heads) != 1:
                raise RuntimeError("Shared initial head states differ across arms")
            dump(
                job / f"{phase}-paired-check.json",
                {"methods": METHODS, "equal_head_initialization": True, "schedule_sha256": schedules},
            )
            stage = "smoke" if phase == "smoke" else "complete"
            if not (job / f"archive-{stage}.json").exists():
                paths = [
                    job / "provenance.json",
                    job / f"{phase}-paired-check.json",
                    cache_dir / "manifest.json",
                    cache_dir / "integrity.json",
                    cache_dir / "complete.json",
                ]
                if phase == "smoke":
                    paths += [p for output in outputs for p in light_files(output)]
                else:
                    paths += [job / "comparison.csv", job / "comparison.txt"]
                archive(job, stage, paths, a.push)
        (job / "status.txt").write_text("SUCCEEDED\n")
    except Exception:
        failure = traceback.format_exc()
        phase_status = (job / "status.txt").read_text() if (job / "status.txt").exists() else "unknown"
        dump(
            job / "failure-details.json",
            {
                "phase_status": phase_status,
                "logs": {
                    path.name: {"sha256": sha(path), "tail": path.read_text(errors="replace")[-6000:]}
                    for path in job.glob("*.log")
                    if path.name != "job.log"
                },
            },
        )
        (job / "status.txt").write_text("FAILED; inspect job.log and case/build logs\n")
        (job / "failure.txt").write_text(failure)
        # Keep the failure visible even if archive/push itself is unavailable.
        try:
            stamp = f"failure-{int(time.time())}"
            archive(job, stamp, [job / "provenance.json", job / "failure.txt", job / "failure-details.json"], a.push)
        except Exception as error:  # noqa: BLE001 - preserve the original experiment failure if archiving also fails.
            print(f"FAILURE ARCHIVE NOT COMPLETED: {error}", flush=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("campaign", "build", "train"))
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--push", action="store_true")
    a = parser.parse_args()
    if a.batch < 1 or a.epochs < 1:
        parser.error("batch and epochs must be positive")
    if a.action == "campaign":
        campaign(a)
    elif a.action == "build":
        verify_source(a.job)
        from online_experiments.detail_probe_data import prepare_cache

        prepare_cache(a.source, a.cache, batch_size=a.batch, limit=17 if a.smoke else None)
    else:
        train(a)


if __name__ == "__main__":
    main()
