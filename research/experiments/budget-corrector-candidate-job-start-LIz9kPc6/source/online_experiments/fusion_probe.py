#!/usr/bin/env python3
"""Offline spatial-feature ablation using the existing training pipeline."""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = ("router_only", "deepest_value", "mean_value")


def configure_fusion(model, case):
    if case not in CASES:
        raise ValueError(case)
    for name in ("p3", "p4", "p5"):
        layer = getattr(model, name)
        if layer.value_fusion_mode != "router_only" or len(layer.in_channels) != 3:
            raise RuntimeError("Expected the original three-input router_only detector")
        if case != "router_only":
            layer.value_fusion_mode = "weighted_sum"
            weights = layer.value_fusion_weights
            weights.fill_(1.0 / 3.0) if case == "mean_value" else weights.zero_()
            if case == "deepest_value":
                weights[-1] = 1.0


def worker(a):
    sys.path.insert(0, str(ROOT))
    from scripts import d1_train_cached_detector as d
    from online_experiments import d1_online_compare as pilot

    cfg = json.loads(a.reference.read_text())
    cfg.update(
        mode="offline", augment="none", parity_only=False,
        seed=a.seed, epochs=a.epochs,
        output=a.job / f"{a.case}-s{a.seed}",
    )
    cfg["cache"] = Path(cfg["cache"])
    layers = cfg["cache_identity"]["layers"]
    if layers != [3, 7, 11]:
        raise RuntimeError(f"Expected layers [3, 7, 11], got {layers}")

    original_detector = d.CachedLatentDetector
    original_write = d.atomic_write_json

    class ProbeDetector(original_detector):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            configure_fusion(self, a.case)

    def write(path, value):
        value = dict(value)
        if Path(path).name == "args.json":
            value["detector_fusion"] = a.case
            value["fusion_probe"] = a.case
            value["reference_args"] = str(a.reference)
            value["source_hashes"] = dict(value["source_hashes"])
            for source in (
                Path(__file__).resolve(),
                Path(pilot.__file__).resolve(),
            ):
                value["source_hashes"][str(source.relative_to(ROOT))] = (
                    hashlib.sha256(source.read_bytes()).hexdigest()
                )
        elif Path(path).name == "summary.json":
            value["fusion_probe"] = a.case
        return original_write(path, value)

    d.CachedLatentDetector = ProbeDetector
    d.atomic_write_json = write
    pilot.arguments = lambda: argparse.Namespace(**cfg)
    try:
        pilot.main()
    finally:
        d.CachedLatentDetector = original_detector
        d.atomic_write_json = original_write


def campaign(a):
    a.job.mkdir(parents=True, exist_ok=True)
    status = a.job / "status.txt"
    status.write_text("RUNNING preflight\n")
    try:
        cfg = json.loads(a.reference.read_text())
        if cfg["mode"] != "offline" or cfg["augment"] != "none":
            raise RuntimeError("Reference must be the completed offline-none run")

        snapshot = a.job / "reference_args.json"
        if snapshot.exists():
            raise FileExistsError("Use a fresh job directory")
        snapshot.write_text(json.dumps(cfg, indent=2))

        gpu = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        (a.job / "nvidia-smi-start.txt").write_text(gpu.stdout + gpu.stderr)

        results, comparable = [], None
        keys = (
            "cache_identity", "epochs", "batch", "lr", "weight_decay",
            "aux_weight", "teacher_dtype", "train_count", "val_count",
            "split_seed", "source_hashes",
        )
        for seed in a.seeds:
            for case in CASES:
                status.write_text(f"RUNNING {case} seed={seed}\n")
                command = [
                    sys.executable, "-u", str(Path(__file__).resolve()),
                    "--reference", str(snapshot), "--job", str(a.job),
                    "--case", case, "--seed", str(seed),
                    "--epochs", str(a.epochs),
                ]
                log = a.job / f"{case}-s{seed}.log"
                print(f"START {case} seed={seed}; log={log}", flush=True)
                with log.open("w") as stream:
                    subprocess.run(
                        command, check=True, stdout=stream,
                        stderr=subprocess.STDOUT, cwd=ROOT,
                    )

                folder = a.job / f"{case}-s{seed}"
                summary = json.loads((folder / "summary.json").read_text())
                config = json.loads((folder / "args.json").read_text())
                split = json.loads((folder / "split.json").read_text())
                signature = {key: config[key] for key in keys}
                signature["split"] = {
                    key: split[key]
                    for key in ("train_sample_ids", "val_sample_ids")
                }
                if comparable is not None and signature != comparable:
                    raise RuntimeError("Configuration, sources or split changed across runs")
                comparable = signature

                if (
                    summary["completed_epochs"] != a.epochs
                    or summary["fusion_probe"] != case
                    or config["detector_fusion"] != case
                ):
                    raise RuntimeError(f"Incomplete or mislabeled result: {folder}")

                with (folder / "results.csv").open(newline="") as stream:
                    rows = list(csv.DictReader(stream))
                if len(rows) != a.epochs:
                    raise RuntimeError(f"Unexpected CSV length: {folder}")
                best = max(rows, key=lambda row: float(row["map50_95"]))

                record = dict(
                    fusion=case, seed=seed,
                    best_AP=100 * summary["best_map50_95"],
                    last_AP=100 * summary["final_map50_95"],
                    best_epoch=int(best["epoch"]),
                    train_s=summary["train_seconds"],
                    VRAM_GiB=summary["peak_vram_gib"],
                )
                results.append(record)
                (a.job / "comparison.json").write_text(
                    json.dumps(results, indent=2)
                )
                text = "fusion seed best_AP last_AP best_epoch train_s VRAM_GiB\n"
                text += "\n".join(
                    f"{r['fusion']} {r['seed']} "
                    f"{r['best_AP']:.4f} {r['last_AP']:.4f} "
                    f"{r['best_epoch']} {r['train_s']:.1f} {r['VRAM_GiB']:.3f}"
                    for r in results
                ) + "\n"
                (a.job / "comparison.txt").write_text(text)
                print(text, flush=True)

        status.write_text("SUCCEEDED\n")
    except BaseException:
        status.write_text("FAILED: inspect job.log and the last case log\n")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    args.reference = args.reference.resolve()
    args.job = args.job.resolve()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    os.chdir(ROOT)
    worker(args) if args.case else campaign(args)
