"""Paired full-VisDrone replications and explicitly named baseline adaptations.

Reuse immutable FP16 banks through a separately verified cache manifest. Native
25-percent arms retain the existing detail-probe loader and model unchanged.
FroFA and LOFF-TA are detection adaptations of the local feature operators.
"""

import argparse
import functools
import hashlib
import importlib.metadata
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "online_experiments"))

from online_experiments.detail_probe import (
    dump,
    publish_epoch_evidence,
    read,
    reference_evidence,
    score,
    sha,
    source_identity,
    telemetry,
)

METHODS = (
    "cache1",
    "cache2",
    "frofa-c2-d01",
    "loffta-c2-n005",
    "mix25",
    "online100",
    "transport25",
    "blank25",
    "views25",
    "detail25",
    "lowpass25",
    "blank0",
    "detail0",
    "detail10",
)
PROTOCOL = ROOT / "research/REPLICATION_ABLATION_PLAN_20261106.md"


def method_spec(method):
    """Give every method an explicit query, input, and geometry contract."""
    if method not in METHODS:
        raise ValueError(f"Unknown suite method: {method}")
    native = (
        method
        if method in ("transport25", "blank25", "views25", "detail25", "lowpass25")
        else {
            "blank0": "blank25",
            "detail0": "detail25",
            "detail10": "detail25",
        }.get(method)
    )
    pct = 100 if method == "online100" else 10 if method == "detail10" else 25 if method.endswith("25") else 0
    detail = "lowpass" if method == "lowpass25" else "rgb" if method in ("detail25", "detail0", "detail10") else "none"
    geometry = (
        "unaugmented"
        if method == "cache1"
        else "true flipped C2; scale 1"
        if method in ("cache2", "frofa-c2-d01")
        else "requested scale on queried images; scale 1 on nonquery C2"
        if method == "mix25"
        else "continuous requested geometry"
    )
    return {
        "native_method": native,
        "query_pct": pct,
        "detail_mode": detail,
        "effective_geometry": geometry,
        "brightness": 0.1 if method == "frofa-c2-d01" else 0.0,
        "noise": 0.05 if method == "loffta-c2-n005" else 0.0,
    }


def check_counts(epoch, seen, n, queries, pct):
    if seen != n or queries != epoch * n * pct // 100:
        raise RuntimeError("Incomplete image coverage or mismatched cumulative query budget")


def cost_for_method(manifest, method):
    """Charge each run's consumed physical banks, including validation and blank."""
    spec = method_spec(method)
    keys = ["val-v0"]
    if method != "online100":
        keys += ["train-v0"] if method == "cache1" else ["train-v0", "train-v1"]
    seconds = float(manifest["source_setup_seconds"])
    if spec["native_method"] in ("blank25", "views25", "detail25", "lowpass25"):
        keys.append("blank")
        seconds += float(manifest["setup_seconds"])
    if method == "views25":
        keys += ["s085-v0", "s085-v1", "s115-v0", "s115-v1"]
    parts = [manifest["parts"][key] for key in keys]
    return {
        "cache_build_charged_seconds": seconds + sum(float(part["seconds"]) for part in parts),
        "cache_bytes": sum(int(part["bytes"]) for part in parts),
        "cache_build_query_images": sum(
            int(part["images"]) + int(part.get("extra_parity_images", 0)) for part in parts
        ),
        "consumed_cache_parts": keys,
    }


@functools.lru_cache(maxsize=1)
def batch_class():
    """Load training dependencies only when data are actually requested."""
    import numpy as np
    import torch

    from online_experiments.d1_online_compare import training_feature
    from online_experiments.detail_probe_data import DetailBatches, _freeze, _manifest_check
    from online_experiments.feature_baselines import add_noise, frofa, warp_features
    from online_experiments.full_visdrone_data import epoch_plan, make_extractor, read_feature_batch
    from scripts import d1_train_cached_detector as d

    class SuiteBatches(DetailBatches):
        """Use the native loader where possible and keep zero-query arms teacher-free."""

        def __init__(self, source, cache_manifest, method, seed, train_limit=None, val_limit=None):
            self.suite_method, self.seed, self.spec = method, seed, method_spec(method)
            self.method = self.spec["native_method"] or method
            self.pct = self.spec["query_pct"]
            source, cache_manifest = Path(source).resolve(), Path(cache_manifest).resolve()
            if cache_manifest.name != "manifest.json":
                raise ValueError("Expected the completed cache manifest.json")
            self.cache = _manifest_check(source, cache_manifest.parent)
            self.meta = read(source / "dataset-manifest.json")
            self.train, self.val = self.meta["splits"]["train"], self.meta["splits"]["val"]
            for name, limit, count in (("train", train_limit, len(self.train)), ("val", val_limit, len(self.val))):
                if limit is not None and not 1 <= limit <= count:
                    raise ValueError(f"Invalid {name} limit")
            self.train = self.train[:train_limit] if train_limit is not None else self.train
            self.val = self.val[:val_limit] if val_limit is not None else self.val
            if len(self.train) > self.cache["train_images"]:
                raise RuntimeError("Training split exceeds built cache")
            views = () if method == "online100" else (0,) if method == "cache1" else (0, 1)
            self.arrays = [np.load(self.cache["parts"][f"train-v{view}"]["file"], mmap_mode="r") for view in views]
            self.val_array = np.load(self.cache["parts"]["val-v0"]["file"], mmap_mode="r")
            self.view_arrays = {}
            if method == "views25":
                from online_experiments.detail_probe_data import NEW_VIEWS

                self.view_arrays = {
                    (scale, int(flip)): np.load(self.cache["parts"][key]["file"], mmap_mode="r")
                    for key, (scale, flip) in NEW_VIEWS.items()
                }
            self.device = torch.device("cuda:0")
            self.extractor = make_extractor(self.meta["identity"]) if self.pct else None
            if self.extractor is not None:
                _freeze(self.extractor)
            needs_blank = self.spec["native_method"] in ("blank25", "views25", "detail25", "lowpass25")
            self.blank = (
                torch.from_numpy(np.load(self.cache["parts"]["blank"]["file"])[0].copy()).to(self.device).float()
                if needs_blank
                else None
            )
            self.query_images = self.query_calls = self.correction_steps = 0
            self.correction_seconds = self.raw_image_seconds = 0.0
            self.raw_image_images = 0
            self.reconstruction = []
            self.corrector = self.optimizer = None

        def training(self, epoch, batch_size, output):
            if self.spec["native_method"] is not None:
                yield from super().training(epoch, batch_size, output)
                return
            if batch_size < 1:
                raise ValueError("Batch size must be positive")
            order, flips, scales, query = epoch_plan(len(self.train), self.pct, epoch, self.seed)
            schedule = {"epoch": epoch + 1, "order": order, "flip": flips, "scale": scales, "query": query}
            path = Path(output) / "schedules" / f"{epoch + 1:03d}.json"
            if path.exists() and read(path) != schedule:
                raise RuntimeError("Existing paired schedule mismatch")
            if not path.exists():
                dump(path, schedule)
            generator = torch.Generator(device=self.device).manual_seed(self.seed + 49979687 + epoch * 1000003)
            for start in range(0, len(order), batch_size):
                ids, fs, ss, qs = (values[start : start + batch_size] for values in (order, flips, scales, query))
                if self.suite_method == "cache1":
                    fs = [False] * len(ids)
                effective = (
                    [scale if chosen else 1.0 for scale, chosen in zip(ss, qs)]
                    if self.suite_method == "mix25"
                    else [1.0] * len(ids)
                    if self.suite_method in ("cache1", "cache2", "frofa-c2-d01")
                    else ss
                )
                items = [self.item(self.train[index], flip, scale) for index, flip, scale in zip(ids, fs, effective)]
                levels = None
                if self.arrays:
                    values = (
                        torch.from_numpy(read_feature_batch(self.arrays, ids, [int(flip) for flip in fs]))
                        .to(self.device)
                        .float()
                    )
                    levels = tuple(values[:, level] for level in range(3))
                    if self.suite_method == "loffta-c2-n005":
                        levels = tuple(
                            add_noise(warp_features(level, [False] * len(ids), ss), generator, 0.05) for level in levels
                        )
                    elif self.suite_method == "frofa-c2-d01":
                        levels = tuple(frofa(level, generator, 0.1) for level in levels)
                queried = [index for index, chosen in enumerate(qs) if chosen]
                if queried:
                    images = torch.stack(
                        [self._image(self.train[ids[index]], fs[index], ss[index]) for index in queried]
                    )
                    with torch.no_grad():
                        result = self.extractor(images)
                    exact = tuple(
                        training_feature(result[f"layer_{layer}"]).to(self.device)
                        for layer in self.meta["identity"]["layers"]
                    )
                    if not all(torch.isfinite(level).all() for level in exact):
                        raise RuntimeError("Nonfinite queried teacher feature")
                    if self.suite_method == "online100":
                        levels = exact
                    else:
                        for level, teacher in zip(levels, exact):
                            level[queried] = teacher
                    self.query_images += len(queried)
                    self.query_calls += 1
                if levels is None:
                    raise RuntimeError("No cached or queried features for batch")
                for index, item in enumerate(items):
                    item["features"] = tuple(level[index] for level in levels)
                yield d.collate_cached(items)

    return SuiteBatches


def snapshot(a):
    """Create a fresh protocol and code identity without modifying old jobs."""
    a.job.mkdir(parents=True, exist_ok=True)
    if (a.job / "provenance.json").exists():
        verify_provenance(a)
        return
    sources = source_identity()
    required = (
        "online_experiments/replication_suite.py",
        "online_experiments/detail_probe.py",
        "online_experiments/detail_probe_data.py",
        "online_experiments/detail_probe_model.py",
    )
    if any(name not in sources for name in required):
        raise RuntimeError("Commit suite source before taking its protocol snapshot")
    if subprocess.check_output(["git", "diff", "HEAD", "--name-only", "--", *sources], cwd=ROOT, text=True).strip():
        raise RuntimeError("Commit training source changes before taking the snapshot")
    evidence = reference_evidence()
    dump(
        a.job / "provenance.json",
        {
            "schema": "replication-suite-v1",
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_sha256": sources,
            "source_job": str(a.source.resolve()),
            "dataset_manifest_sha256": sha(a.source / "dataset-manifest.json"),
            "cache_manifest": str(a.cache.resolve()),
            "cache_manifest_sha256": sha(a.cache),
            "protocol_file": str(a.protocol.resolve()),
            "protocol_sha256": sha(a.protocol),
            "methods": {method: method_spec(method) for method in METHODS},
            "reference_parity_evidence": evidence,
            "telemetry": telemetry(),
        },
    )


def verify_provenance(a):
    provenance = read(a.job / "provenance.json")
    if provenance.get("schema") != "replication-suite-v1":
        raise RuntimeError("Expected a fresh replication-suite-v1 provenance snapshot")
    if provenance["methods"] != {method: method_spec(method) for method in METHODS}:
        raise RuntimeError("Suite method contracts changed after snapshot")
    for name, expected in provenance["source_sha256"].items():
        if not (ROOT / name).is_file() or sha(ROOT / name) != expected:
            raise RuntimeError(f"Training source changed after snapshot: {name}")
    checks = (
        (a.source / "dataset-manifest.json", provenance["dataset_manifest_sha256"]),
        (a.cache, provenance["cache_manifest_sha256"]),
        (Path(provenance["protocol_file"]), provenance["protocol_sha256"]),
    )
    if str(a.source.resolve()) != provenance["source_job"] or str(a.cache.resolve()) != provenance["cache_manifest"]:
        raise RuntimeError("Suite source/cache path changed after snapshot")
    for path, expected in checks:
        if sha(path) != expected:
            raise RuntimeError(f"Protocol or input manifest changed after snapshot: {path}")


def geometry_schedules(output):
    """Geometry pairing remains comparable when query percentages differ."""
    result = {}
    for path in sorted((Path(output) / "schedules").glob("*.json")):
        schedule = read(path)
        geometry = {key: schedule[key] for key in ("epoch", "order", "flip", "scale")}
        result[path.name] = hashlib.sha256(
            json.dumps(geometry, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    return result


def validate_completed(output, smoke, method=None, seed=None, *, request=None):
    """Verify full epoch, budget, schedule, checkpoint, and official-score evidence."""
    output = Path(output)
    args, summary, rows = read(output / "args.json"), read(output / "summary.json"), read(output / "metrics.json")
    spec = method_spec(args["method"])
    if (method is not None and args["method"] != method) or (seed is not None and args["seed"] != seed):
        raise RuntimeError("Completed run method/seed identity mismatch")
    if args.get("batch", 0) < 1 or (not smoke and args["batch"] != 8):
        raise RuntimeError("Completed run batch violates the fixed batch-8 full-run protocol")
    if request is not None:
        expected_identity = {
            "batch": request.batch,
            "provenance_sha256": sha(request.job / "provenance.json"),
            "dataset_manifest_sha256": sha(request.source / "dataset-manifest.json"),
            "cache_manifest_sha256": sha(request.cache),
        }
        for name, expected in expected_identity.items():
            if args.get(name) != expected:
                raise RuntimeError(f"Completed run identity mismatch: {name}")
    epochs, n, nv = (1, 17, 8) if smoke else (100, 6471, 548)
    if (args["epochs"], summary["completed_epochs"], args["train_images"], args["val_images"]) != (
        epochs,
        epochs,
        n,
        nv,
    ):
        raise RuntimeError("Incomplete or incompatible completed run")
    if args["query_pct"] != spec["query_pct"] or args["detail_mode"] != spec["detail_mode"]:
        raise RuntimeError("Completed run method contract mismatch")
    for name in (
        "method",
        "seed",
        "epochs",
        "batch",
        "train_images",
        "val_images",
        "query_pct",
        "detail_mode",
        "provenance_sha256",
        "dataset_manifest_sha256",
        "cache_manifest_sha256",
    ):
        if summary.get(name) != args[name]:
            raise RuntimeError(f"Summary identity mismatch: {name}")
    if [row["epoch"] for row in rows] != list(range(1, epochs + 1)):
        raise RuntimeError("Completed epoch records are missing or duplicated")
    check_counts(epochs, n, n, summary["query_images"], spec["query_pct"])
    schedules = summary["schedule_sha256"]
    if len(schedules) != epochs or any(sha(output / "schedules" / name) != value for name, value in schedules.items()):
        raise RuntimeError("Completed augmentation schedules changed or incomplete")
    if "geometry_schedule_sha256" in summary and summary["geometry_schedule_sha256"] != geometry_schedules(output):
        raise RuntimeError("Completed geometry schedules mismatch")
    if not smoke:
        official = read(output / "official-final.json")
        if len(official["prediction_sha256"]) != nv or official["checkpoint_sha256"] != sha(output / "weights/last.pt"):
            raise RuntimeError("Official checkpoint or prediction coverage mismatch")
        if summary["official_final"] != official["metrics"]:
            raise RuntimeError("Official summary mismatch")
        for name, expected in official["prediction_sha256"].items():
            if sha(output / "predictions/last" / f"{name}.txt") != expected:
                raise RuntimeError("Scored predictions changed after final evaluation")


def train(a):
    import resource

    import torch

    from online_experiments.assignment_intervention import state_hash
    from online_experiments.detail_probe_model import DetailDetector
    from online_experiments.full_visdrone_train import atomic_checkpoint, save_predictions, validate
    from online_experiments.fusion_probe import configure_fusion
    from scripts import d1_train_cached_detector as d

    verify_provenance(a)
    output, spec = a.output, method_spec(a.method)
    if (output / "DONE").exists():
        validate_completed(output, a.smoke, a.method, a.seed, request=a)
        print(f"SKIP verified completed {a.method} seed={a.seed}", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    (output / "weights").mkdir(exist_ok=True)
    torch.set_num_threads(4)
    d.seed_everything(a.seed)
    started = time.perf_counter()
    batches = batch_class()(a.source, a.cache, a.method, a.seed, 17 if a.smoke else None, 8 if a.smoke else None)
    n, nv = len(batches.train), len(batches.val)
    if (n, nv) != ((17, 8) if a.smoke else (6471, 548)):
        raise RuntimeError(f"Unexpected data split {(n, nv)}")
    d.seed_everything(a.seed)
    model = DetailDetector(
        in_channels=(384,) * 3, nc=10, imgsz=640, epochs=a.epochs, detail_mode=spec["detail_mode"]
    ).cuda()
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
        "query_pct": spec["query_pct"],
        "detail_mode": spec["detail_mode"],
        "effective_geometry": spec["effective_geometry"],
        "brightness": spec["brightness"],
        "noise": spec["noise"],
        "fusion": "mean_value",
        "imgsz": 640,
        "max_det": 500,
        "conf": 0.001,
        "lr": 0.001,
        "weight_decay": 0.0005,
        "validation_interval": 10,
        "dataset_manifest_sha256": sha(a.source / "dataset-manifest.json"),
        "cache_manifest_sha256": sha(a.cache),
        "provenance_sha256": sha(a.job / "provenance.json"),
        "primary_endpoint": "fixed final epoch official-reference AP; no best-epoch selection",
        "timing_conditions": "shared GPU; not a controlled speed benchmark",
        "baseline_scope": "local detection idea adaptation"
        if a.method.startswith(("frofa", "loffta"))
        else "matched-head full-split control",
        "feature_augmentation_rng": "separate per-epoch generator: seed + 49979687 + epoch * 1000003",
        "teacher_loaded": batches.extractor is not None,
    }
    if (output / "args.json").exists() and read(output / "args.json") != args:
        raise RuntimeError("Resume configuration changed")
    dump(output / "args.json", args)
    environment = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "transformers")},
        "gpu": torch.cuda.get_device_name(),
    }
    if (output / "environment.json").exists():
        previous = read(output / "environment.json")
        if any(previous[key] != value for key, value in environment.items()):
            raise RuntimeError("Resume environment changed")
    else:
        dump(output / "environment.json", {**environment, "telemetry": telemetry()})
    initialization = {
        "head_state_sha256": state_hash(
            SimpleNamespace(
                state_dict=lambda: {
                    key: value for key, value in model.state_dict().items() if not key.startswith("detail_branch.")
                }
            )
        ),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "detail_parameters": model.detail_parameter_count,
        "teacher_loaded": batches.extractor is not None,
    }
    if (output / "initialization.json").exists() and read(output / "initialization.json") != initialization:
        raise RuntimeError("Initial head or model capacity changed on resume")
    dump(output / "initialization.json", initialization)
    begin, rows = 0, []
    totals = {
        "train": 0.0,
        "val": 0.0,
        "wall": 0.0,
        "raw_train_seconds": 0.0,
        "raw_val_seconds": 0.0,
        "raw_train_images": 0,
        "raw_val_images": 0,
    }
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
        print(f"RESUME {a.method} seed={a.seed} epoch={begin + 1}", flush=True)
    prior_wall = totals["wall"]
    costs = cost_for_method(batches.cache, a.method)
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(begin, a.epochs):
        model.train()
        torch.cuda.synchronize()
        tick, loss_sum, seen = time.perf_counter(), 0.0, 0
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
        check_counts(epoch + 1, seen, n, batches.query_images, spec["query_pct"])
        if batches.extractor is not None and any(
            parameter.requires_grad or parameter.grad is not None for parameter in batches.extractor.model.parameters()
        ):
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
        with (output / "telemetry.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(telemetry()) + "\n")
        summary = {
            **args,
            **costs,
            "completed_epochs": epoch + 1,
            "train_seconds": totals["train"],
            "validation_seconds": totals["val"],
            "wall_seconds": totals["wall"],
            "query_images": batches.query_images,
            "query_calls": batches.query_calls,
            "peak_vram_gib": max(record["peak_vram_gib"] for record in rows),
            "peak_host_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "schedule_sha256": {path.name: sha(path) for path in sorted((output / "schedules").glob("*.json"))},
            "geometry_schedule_sha256": geometry_schedules(output),
            "data_prepare_charged_seconds": batches.meta["prepare_seconds"],
            **{key: value for key, value in totals.items() if key.startswith("raw_")},
            "raw_loading_note": "CPU image read/decode/letterbox time; included in training and validation wall times",
        }
        summary["build_plus_wall_seconds"] = summary["wall_seconds"] + costs["cache_build_charged_seconds"]
        summary["total_charged_seconds"] = summary["build_plus_wall_seconds"] + summary["data_prepare_charged_seconds"]
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
            "summary": summary,
        }
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
        summary["total_charged_seconds"] = summary["build_plus_wall_seconds"] + summary["data_prepare_charged_seconds"]
        dump(output / "summary.json", summary)
    verify_provenance(a)
    validate_completed(output, a.smoke, a.method, a.seed, request=a)
    (output / "DONE").write_text("SUCCEEDED\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("snapshot", "train", "validate"), default="train")
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    a = parser.parse_args()
    if a.batch < 1 or a.seed < 0 or a.epochs != (1 if a.smoke else 100):
        parser.error("Use nonnegative seed, positive batch, and exactly 100 epochs (1 with --smoke)")
    if not a.smoke and a.batch != 8:
        parser.error("Full runs require the fixed protocol batch size of 8")
    if a.action == "snapshot":
        snapshot(a)
        return
    if a.method is None:
        parser.error("--method is required for training or validation")
    a.output = a.output or a.job / ("smoke" if a.smoke else "train") / f"{a.method}-s{a.seed}"
    if a.action == "validate":
        verify_provenance(a)
        validate_completed(a.output, a.smoke, a.method, a.seed, request=a)
    else:
        train(a)


if __name__ == "__main__":
    main()
