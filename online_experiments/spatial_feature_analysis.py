"""Queued full-548 P3 spectra or fresh-RGB inference; never train or run a subset.

`inspect` and `selfcheck` are CPU-only. `features` and `latency` are separate
explicit GPU stages for an external single-slot scheduler. No stage is launched
automatically by inspection. Spectral energy is descriptive, not an AP surrogate.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from online_experiments.analyze_detection_evidence import (
    digest,
    dump,
    geometry,
    load_annotations,
    read,
    sha,
    write_csv,
)

METHODS = ("blank25", "detail25", "lowpass25", "vit-adapter-cache25")
STAGES = ("base_p3", "final_p3", "rgb_residual_p3")
FREQUENCY_EDGES = np.linspace(0, np.sqrt(0.5) + 1e-12, 17)


def spectral_metrics(values):
    """Channel-mean, DC-removed, unit-energy radial power at native P3 resolution."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("Finite CxHxW feature required")
    centered = values - values.mean(axis=(-2, -1), keepdims=True)
    power = np.square(np.abs(np.fft.fft2(centered, axes=(-2, -1), norm="ortho"))).mean(axis=0)
    y, x = np.meshgrid(np.fft.fftfreq(values.shape[-2]), np.fft.fftfreq(values.shape[-1]), indexing="ij")
    radius = np.sqrt(x * x + y * y)
    total = float(power.sum())
    radial = np.histogram(radius, bins=FREQUENCY_EDGES, weights=power)[0]
    radial = radial / total if total else np.zeros_like(radial)
    squared = float(np.square(values).mean())
    edge = float((np.square(np.diff(values, axis=-2)).mean() + np.square(np.diff(values, axis=-1)).mean()) / 2)
    return {
        "high_frequency_fraction": float(power[radius >= 0.25].sum() / total) if total else None,
        "centered_power": total,
        "squared_mean": squared,
        "edge_squared_mean": edge,
        "radial_normalized_power": radial.tolist(),
    }


def rectangle_union_area(rectangles):
    """Exact continuous union area of axis-aligned rectangles using a vertical sweep."""
    boundaries = sorted({value for x0, _, x1, _ in rectangles for value in (x0, x1)})
    area = 0.0
    for left, right in zip(boundaries, boundaries[1:]):
        intervals = sorted((y0, y1) for x0, y0, x1, y1 in rectangles if x0 < right and x1 > left)
        if not intervals:
            continue
        bottom, top = intervals[0]
        length = 0.0
        for y0, y1 in intervals[1:]:
            if y0 > top:
                length += top - bottom
                bottom, top = y0, y1
            else:
                top = max(top, y1)
        area += (right - left) * (length + top - bottom)
    return area


def foreground_weights(gt, letterbox, size=80):
    """Exact GT-union intersection fraction of every P3 cell, including sub-cell boxes."""
    from online_experiments.visdrone_reference_eval import prepare_arrays

    gt, _ = prepare_arrays(gt, np.empty((0, 8)), letterbox["original_shape"])
    gt = gt[(gt[:, 4] == 0) & np.isin(gt[:, 5], np.arange(1, 11))]
    height, width = letterbox["original_shape"]
    rh, rw = letterbox["resized_shape"]
    left, top = letterbox["pad"]
    cell = 640 / size
    cells = {}
    for x, y, w, h in gt[:, :4]:
        x0, x1 = np.clip([x * rw / width + left, (x + w) * rw / width + left], left, left + rw)
        y0, y1 = np.clip([y * rh / height + top, (y + h) * rh / height + top], top, top + rh)
        if x1 <= x0 or y1 <= y0:
            continue
        for cy in range(max(0, int(y0 // cell)), min(size, int(np.ceil(y1 / cell)))):
            for cx in range(max(0, int(x0 // cell)), min(size, int(np.ceil(x1 / cell)))):
                rect = (max(x0, cx * cell), max(y0, cy * cell), min(x1, (cx + 1) * cell), min(y1, (cy + 1) * cell))
                cells.setdefault((cy, cx), []).append(rect)
    weights = np.zeros((size, size), dtype=np.float64)
    for (cy, cx), rectangles in cells.items():
        weights[cy, cx] = rectangle_union_area(rectangles) / cell**2
    if np.any(weights < -1e-10) or np.any(weights > 1 + 1e-10):
        raise ValueError("Foreground rectangle-union coverage outside [0,1]")
    return weights.clip(0, 1)


def original_image_weights(letterbox, size=80):
    """Exact cell fraction within resized original image, excluding letterbox padding."""
    rh, rw = letterbox["resized_shape"]
    left, top = letterbox["pad"]
    cell = 640 / size
    low = np.arange(size) * cell
    x = np.maximum(0, np.minimum(low + cell, left + rw) - np.maximum(low, left))
    y = np.maximum(0, np.minimum(low + cell, top + rh) - np.maximum(low, top))
    return np.outer(y, x) / cell**2


def existing_path(value):
    """Resolve documented mount migration without rewriting archived provenance."""
    path = Path(value)
    if path.exists():
        return path
    old = Path("/data/users/zhjia")
    if path.is_relative_to(old):
        migrated = Path("/mnt/data/users/zhjia") / path.relative_to(old)
        if migrated.exists():
            return migrated
    raise FileNotFoundError(path)


def content_order(samples):
    result = []
    for sample in samples:
        classes = np.asarray(sample["cls"], dtype=np.float64).reshape(-1)
        boxes = np.asarray(sample["bboxes"], dtype=np.float64).reshape(-1, 4)
        if len(classes) != len(boxes) or not np.isfinite(classes).all() or not np.isfinite(boxes).all():
            raise ValueError("Manifest semantic GT classes/boxes are malformed")
        if np.any(classes != np.floor(classes)) or np.any((classes < 0) | (classes >= 10)):
            raise ValueError("Manifest semantic GT class outside VisDrone0..9")
        # Annotation row order and textual float formatting carry no semantic
        # meaning. Preserve image/bank order, but sort per-image class+box tuples.
        semantic_gt = sorted([[int(category), *map(float, box)] for category, box in zip(classes, boxes)])
        result.append(
            {
                **{key: sample[key] for key in ("sample_id", "image_sha256", "letterbox")},
                "semantic_gt_cls_bboxes": semantic_gt,
            }
        )
    return result


def run_paths(args):
    return {
        name: (args.campaign if name.startswith("vit-") else args.original_detail_job) / "train" / f"{name}-s0"
        for name in METHODS
    }


def validate_inputs(args, *, include_bank=False):
    provenance = read(args.campaign / "provenance.json")
    source = Path(provenance["source_job"])
    manifest_path = source / "dataset-manifest.json"
    if sha(manifest_path) != provenance["dataset_manifest_sha256"]:
        raise ValueError("Frozen full-split manifest changed")
    manifest = read(manifest_path)
    if len(manifest["splits"]["val"]) != 548 or len(manifest["splits"]["train"]) != 6471:
        raise ValueError("Only full VisDrone6471/548 is supported; subset switches are absent")
    cache_path = Path(provenance["cache_manifest"])
    if sha(cache_path) != provenance["cache_manifest_sha256"]:
        raise ValueError("Frozen cache manifest changed")
    cache = read(cache_path)
    if manifest["identity"] != cache["identity"]:
        raise ValueError("Teacher identity differs between dataset and feature bank")
    records, annotation_hashes = load_annotations(source, args.archive)
    bindings = {}
    for method, run in run_paths(args).items():
        if not (run / "DONE").exists():
            raise ValueError(f"Incomplete final run: {run}")
        config, official = read(run / "args.json"), read(run / "official-final.json")
        if config["epochs"] != 100 or config["seed"] != 0 or config["imgsz"] != 640:
            raise ValueError("Expected fixed final epoch100 seed0 imgsz640")
        if (config["train_images"], config["val_images"]) != (6471, 548):
            raise ValueError("Run did not train/evaluate the full split")
        if official["annotation_sha256"] != annotation_hashes:
            raise ValueError("Raw GT identity mismatch between methods")
        if sha(run / "weights/last.pt") != official["checkpoint_sha256"]:
            raise ValueError(f"Final checkpoint hash mismatch: {method}")
        run_provenance_path = run.parents[1] / "provenance.json"
        if config["provenance_sha256"] != sha(run_provenance_path):
            raise ValueError(f"Run arguments no longer bind its provenance: {method}")
        run_provenance = read(run_provenance_path)
        run_manifest_path = existing_path(run_provenance["source_job"]) / "dataset-manifest.json"
        run_cache_path = existing_path(run_provenance.get("cache_manifest", run.parents[1] / "cache/manifest.json"))
        if config["dataset_manifest_sha256"] != sha(run_manifest_path):
            raise ValueError(f"Run dataset-manifest hash no longer binds its actual source: {method}")
        if config["cache_manifest_sha256"] != sha(run_cache_path):
            raise ValueError(f"Run cache-manifest hash no longer binds its actual cache: {method}")
        run_manifest, run_cache = read(run_manifest_path), read(run_cache_path)
        if run_cache["dataset_manifest_sha256"] != config["dataset_manifest_sha256"]:
            raise ValueError(f"Run cache belongs to a different dataset manifest: {method}")
        if run_manifest["identity"] != manifest["identity"] or run_cache["identity"] != manifest["identity"]:
            raise ValueError(f"Run teacher revision/preprocessing/layers/image size differ: {method}")
        if run_cache["parts"]["val-v0"]["sha256"] != cache["parts"]["val-v0"]["sha256"]:
            raise ValueError(f"Run uses a different validation feature bank: {method}")
        if run_manifest["smoke"] or run_manifest["test_dev_used"]:
            raise ValueError(f"Run source is a subset or test-dev: {method}")
        if content_order(run_manifest["splits"]["val"]) != content_order(manifest["splits"]["val"]):
            raise ValueError(f"Run validation image content, bank order or geometry differs: {method}")
        critical = {
            name: value
            for name, value in run_provenance["source_sha256"].items()
            if name.startswith("ultralytics/nn/")
            or name
            in (
                "online_experiments/detail_probe_model.py",
                "online_experiments/paper_adapter_model.py",
                "online_experiments/fusion_probe.py",
                "scripts/d1_train_cached_detector.py",
                "scripts/d1_build_feature_cache.py",
                "ultralytics/data/foundation_cache.py",
                "ultralytics/nn/foundation/preprocessing.py",
            )
        }
        required = {
            "online_experiments/detail_probe_model.py",
            "scripts/d1_train_cached_detector.py",
            "online_experiments/fusion_probe.py",
        }
        if method.startswith("vit-"):
            required.add("online_experiments/paper_adapter_model.py")
        if not critical or not required.issubset(critical):
            raise ValueError(f"Run provenance lacks required frozen model source: {method}")
        changed = [name for name, value in critical.items() if not (ROOT / name).is_file() or sha(ROOT / name) != value]
        if changed:
            raise ValueError(f"Frozen model implementation changed for {method}: {changed[:8]}")
        bindings[method] = {
            "run": str(run),
            "checkpoint_sha256": official["checkpoint_sha256"],
            "args_sha256": sha(run / "args.json"),
            "official_final_sha256": sha(run / "official-final.json"),
            "critical_model_source_files": len(critical),
            "model_source_sha256": digest(critical),
            "provenance_sha256": sha(run_provenance_path),
            "dataset_manifest_sha256": sha(run_manifest_path),
            "cache_manifest_sha256": sha(run_cache_path),
            "validation_bank_sha256": run_cache["parts"]["val-v0"]["sha256"],
            "validation_image_order_sha256": digest(content_order(run_manifest["splits"]["val"])),
            "validation_semantic_gt_sha256": digest(
                [row["semantic_gt_cls_bboxes"] for row in content_order(run_manifest["splits"]["val"])]
            ),
            "source_reference_as_saved": run_provenance["source_job"],
            "source_resolved_after_mount_migration": str(run_manifest_path.parent),
        }
    bank = None
    if include_bank:
        part = cache["parts"]["val-v0"]
        bank_path = Path(part["file"])
        if bank_path.stat().st_size != part["bytes"] or sha(bank_path) != part["sha256"]:
            raise ValueError("Full validation feature bank hash mismatch")
        bank = np.load(bank_path, mmap_mode="r", allow_pickle=False)
        if bank.shape != (548, 3, 384, 40, 40) or bank.dtype != np.float16:
            raise ValueError("Unexpected fixed three-layer FP16 bank shape/dtype")
    return manifest, cache, records, bindings, bank


def contract(args, bindings, manifest, cache):
    return {
        "schema": "full548-spatial-evidence-v1",
        "action": args.action,
        "images": 548,
        "training_seed": 0,
        "selection": "fixed final epoch100",
        "methods": METHODS,
        "source_sha256": sha(__file__),
        "offline_analysis_source_sha256": sha(ROOT / "online_experiments/analyze_detection_evidence.py"),
        "bindings": bindings,
        "teacher": manifest["identity"],
        "batch": 1,
        "spectrum": {
            "stage": "P3 stride8 64x80x80",
            "remove_per_channel_spatial_mean": True,
            "power": "mean_channel(abs(fft2(centered,norm=ortho))**2)",
            "radial_normalization": "divide each image/stage radial power by its own total non-DC power",
            "frequency_units": "cycles per native P3 feature-grid pixel",
            "high_frequency_min_radius": 0.25,
            "radial_edges": FREQUENCY_EDGES.tolist(),
            "windowing": "none; original letterbox boundary retained; identical coordinates for all methods",
            "interpretation": "Descriptive frequency/energy measures, not proof of causal accuracy benefit",
        },
        "foreground": "Exact continuous rectangle-union intersection area /stride8 cell area, "
        "including sub-cell boxes and counting overlaps once; FG/BG energy means use "
        "coverage weights; BG is valid original-image coverage minus FG; letterbox padding "
        "is separately1-valid coverage. Each mean uses its own weight-sum denominator. "
        "Empty foreground/background is NA; this is not object-ROI resampling.",
        "latency": {
            "input": "fresh original RGB read+JPEG decode+640 letterbox for every measured call",
            "DINO": "FP32 model, every image and every method independently recomputed, no val-cache read",
            "feature_interface": "reuse frozen training_feature FP32->FP16->FP32 detached clone",
            "warmup_images_per_method": args.warmup,
            "measured_images_per_method": 548,
            "timers": "synchronized host monotonic stage timers; branch CUDA events nested inside inclusive pyramid",
            "postprocessing": "E2E Detect decode/topK in head; conf>=.001 then CPU inverse-letterbox",
            "NMS": "not applicable: frozen Detect(end2end=True), no added NMS",
            "setup_excluded_and_reported_separately": True,
            "prediction_file_write_excluded": True,
            "cache_state": "OS filesystem warm state uncontrolled; GPU shared conditions observational",
            "statistics": "per-image P50/P95/mean, no deployment speedup claim from shared GPU",
        },
        "bank_sha256": cache["parts"]["val-v0"]["sha256"] if args.action == "features" else None,
        "lowpass": "same trained 4672-parameter RGB branch, its own checkpoint, frozen area40 then bilinear640 input",
    }


def gpu():
    import torch

    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("Queued GPU stage requires CUDA; no CPU/model fallback or subset")
    free, total = torch.cuda.mem_get_info(0)
    if free < 8 * 1024**3:
        raise RuntimeError(f"Queued GPU stage needs >=8GiB free, observed {free / 1024**3:.3f}GiB")
    return torch, {
        "device": torch.cuda.get_device_name(0),
        "free_bytes_start": free,
        "total_bytes": total,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def gpu_snapshot():
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            text=True,
            timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError) as error:
        return f"snapshot unavailable: {error}"


def load_model(method, run, torch):
    from online_experiments.detail_probe_model import DetailDetector
    from online_experiments.fusion_probe import configure_fusion
    from online_experiments.paper_adapter_model import PaperAdapterDetector

    config = read(run / "args.json")
    model_class = PaperAdapterDetector if method.startswith("vit-") else DetailDetector
    model = model_class(in_channels=(384, 384, 384), nc=10, imgsz=640, epochs=100, detail_mode=config["detail_mode"])
    configure_fusion(model, "mean_value")
    model.detect.max_det = 500
    checkpoint = torch.load(run / "weights/last.pt", map_location="cpu", weights_only=False)
    if checkpoint["epoch"] != 100 or checkpoint["args"] != config:
        raise ValueError(f"Final checkpoint configuration/epoch differs: {method}")
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    return model.to("cuda:0").eval()


def verify_image(sample):
    if sha(sample["image_path"]) != sample["image_sha256"]:
        raise ValueError(f"Original validation RGB changed: {sample['sample_id']}")


def features(args, manifest, cache, records, bindings, bank):
    torch, hardware = gpu()
    from ultralytics.data.foundation_cache import load_letterboxed_tensor

    dump(args.output / "hardware-start.json", {**hardware, "compute_apps": gpu_snapshot()})
    profiles = [geometry(record) for record in records]
    selected = {
        entry["image_id"]
        for key in ("small_gt", "objects_per_megapixel", "heavy_occluded_gt", "truncated_gt")
        for entry in sorted(profiles, key=lambda item: (-item[key], item["image_id"]))[:8]
    }
    dump(
        args.output / "heatmap-selection.json",
        {
            "selection": "GT-only top8 each difficulty before feature inspection",
            "sample_ids": sorted(selected),
            "union_count": len(selected),
        },
    )
    all_rows, radial_rows = [], []
    for method, run in run_paths(args).items():
        model = load_model(method, run, torch)
        captured = {}
        handles = [
            model.p3.register_forward_hook(
                lambda module, inputs, output, target=captured: target.update(base_p3=output)
            )
        ]
        if model.detail_branch is not None:

            def capture_branch(module, inputs, output, target=captured):
                target["rgb_residual_p3"] = output[0] if isinstance(output, (list, tuple)) else output

            handles.append(model.detail_branch.register_forward_hook(capture_branch))
        with torch.inference_mode():
            for index, (sample, record) in enumerate(zip(manifest["splits"]["val"], records)):
                verify_image(sample)
                image, info = load_letterboxed_tensor(sample["image_path"], imgsz=640)
                if info != sample["letterbox"]:
                    raise ValueError("Frozen letterbox geometry changed")
                raw = torch.from_numpy(np.array(bank[index], copy=True)).to("cuda:0").float()
                rgb = image[None].to("cuda:0")
                inputs = tuple(raw[level][None] for level in range(3))
                if model.detail_branch is not None:
                    inputs += (rgb,)
                captured.clear()
                pyramid = model.pyramid(inputs)
                base = captured["base_p3"][0].cpu().numpy()
                final = pyramid[0][0].cpu().numpy()
                residual = (
                    captured["rgb_residual_p3"][0].cpu().numpy()
                    if "rgb_residual_p3" in captured
                    else np.zeros_like(base)
                )
                if base.shape != (64, 80, 80) or final.shape != base.shape:
                    raise ValueError("P3 dimensions differ from the fixed comparison stage")
                if not np.allclose(final, base + residual, rtol=1e-5, atol=1e-5):
                    raise ValueError("Captured P3 branch does not reconstruct actual frozen pyramid output")
                weights = foreground_weights(record["gt"], info)
                valid_weights = original_image_weights(info)
                if np.any(weights > valid_weights + 1e-9):
                    raise ValueError("Foreground GT union extends into letterbox padding")
                background_weights = (valid_weights - weights).clip(0, 1)
                padding_weights = (1 - valid_weights).clip(0, 1)
                foreground_total, background_total = float(weights.sum()), float(background_weights.sum())
                padding_total = float(padding_weights.sum())
                heatmaps = {
                    "foreground_area_fraction": weights.astype(np.float32),
                    "original_image_area_fraction": valid_weights.astype(np.float32),
                }
                for stage, values in zip(STAGES, (base, final, residual)):
                    metrics = spectral_metrics(values)
                    spatial = np.square(values.astype(np.float64)).mean(axis=0)
                    heatmaps[stage + "_mean_squared"] = spatial.astype(np.float32)
                    all_rows.append(
                        {
                            "method": method,
                            "image_id": record["id"],
                            "stage": stage,
                            **{key: value for key, value in metrics.items() if key != "radial_normalized_power"},
                            "foreground_equivalent_cells": foreground_total,
                            "foreground_nonzero_cells": int(np.count_nonzero(weights)),
                            "background_equivalent_cells": background_total,
                            "padding_equivalent_cells": padding_total,
                            "foreground_squared_mean": float((spatial * weights).sum() / foreground_total)
                            if foreground_total
                            else None,
                            "background_squared_mean": float((spatial * background_weights).sum() / background_total)
                            if background_total
                            else None,
                            "padding_squared_mean": float((spatial * padding_weights).sum() / padding_total)
                            if padding_total
                            else None,
                            "residual_to_base_squared_energy_ratio": float(
                                np.square(residual).sum() / max(np.square(base).sum(), 1e-20)
                            ),
                        }
                    )
                    radial_rows.extend(
                        {
                            "method": method,
                            "image_id": record["id"],
                            "stage": stage,
                            "bin": bin_index,
                            "radius_low": float(FREQUENCY_EDGES[bin_index]),
                            "radius_high": float(FREQUENCY_EDGES[bin_index + 1]),
                            "normalized_power": value,
                        }
                        for bin_index, value in enumerate(metrics["radial_normalized_power"])
                    )
                if record["id"] in selected:
                    folder = args.output / "heatmaps" / method
                    folder.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(folder / f"{record['id']}.npz", **heatmaps)
                if (index + 1) % 64 == 0 or index == 547:
                    print(f"FEATURES {method} {index + 1}/548", flush=True)
        for handle in handles:
            handle.remove()
        del model
        torch.cuda.empty_cache()
    write_csv(args.output / "p3-image-metrics.csv", all_rows)
    write_csv(args.output / "p3-radial-power.csv", radial_rows)
    summary = []
    for method in METHODS:
        for stage in STAGES:
            rows_for_stage = [row for row in all_rows if row["method"] == method and row["stage"] == stage]
            for metric in (
                "high_frequency_fraction",
                "centered_power",
                "squared_mean",
                "edge_squared_mean",
                "foreground_squared_mean",
                "background_squared_mean",
                "padding_squared_mean",
                "residual_to_base_squared_energy_ratio",
            ):
                values = np.array([row[metric] for row in rows_for_stage if row[metric] is not None])
                summary.append(
                    {
                        "method": method,
                        "stage": stage,
                        "metric": metric,
                        "images_total": 548,
                        "images_with_metric": len(values),
                        "mean": float(values.mean()) if len(values) else None,
                        "sample_sd_across_images_not_training_seeds": float(values.std(ddof=1))
                        if len(values) > 1
                        else None,
                    }
                )
    write_csv(args.output / "p3-summary.csv", summary)
    return {
        "images_each": 548,
        "methods": METHODS,
        "feature_image_pairs": 4 * 548,
        "teacher_forward_images": 0,
        "heatmap_n_images_per_method": len(selected),
        "note": "Full548 quantitative features; GT-selected images only for qualitative heatmaps, not AP estimation",
    }


def timed(torch, function, *args, **kwargs):
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = function(*args, **kwargs)
    torch.cuda.synchronize()
    return value, (time.perf_counter() - start) * 1000


def teacher_interface(extracted, layers, convert):
    return tuple(convert(extracted[f"layer_{layer}"]) for layer in layers)


def postprocess_decoded(decoded, info, inverse):
    predictions = decoded[0][0]
    predictions = predictions[predictions[:, 4] >= 0.001]
    return inverse(predictions.cpu().numpy(), info)


def latency(args, manifest, cache, records, bindings):
    torch, hardware = gpu()
    from online_experiments.d1_online_compare import training_feature
    from online_experiments.full_visdrone_data import inverse_letterbox, make_extractor
    from ultralytics.data.foundation_cache import load_letterboxed_tensor

    dump(args.output / "hardware-start.json", {**hardware, "compute_apps": gpu_snapshot()})
    all_rows, all_summary = [], []
    for method, run in run_paths(args).items():
        setup_start = time.perf_counter()
        model = load_model(method, run, torch)
        # Each method is charged its own teacher setup and teacher invocation for
        # every warmup/measured image. No cached feature bank is opened here.
        extractor = make_extractor(manifest["identity"])
        torch.cuda.synchronize()
        setup_seconds = time.perf_counter() - setup_start
        branch_events = []
        handles = []
        if model.detail_branch is not None:

            def branch_before(module, inputs, events=branch_events):
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                events.append([event, None])

            def branch_after(module, inputs, output, events=branch_events):
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                events[-1][1] = event

            handles = [
                model.detail_branch.register_forward_pre_hook(branch_before),
                model.detail_branch.register_forward_hook(branch_after),
            ]
        method_rows = []
        samples = manifest["splits"]["val"]
        for sample in samples:
            verify_image(sample)
        with torch.inference_mode():
            for call in range(args.warmup + 548):
                measured = call >= args.warmup
                index = call - args.warmup if measured else call % 548
                sample, record = samples[index], records[index]
                torch.cuda.synchronize()
                total_start = time.perf_counter()
                (image, info), load_ms = timed(torch, load_letterboxed_tensor, sample["image_path"], imgsz=640)
                if info != sample["letterbox"]:
                    raise ValueError("Frozen letterbox changed")
                rgb, transfer_ms = timed(torch, image[None].to, "cuda:0")
                extracted, teacher_ms = timed(torch, extractor, rgb)
                raw, interface_ms = timed(
                    torch,
                    teacher_interface,
                    extracted,
                    manifest["identity"]["layers"],
                    training_feature,
                )
                inputs = raw + ((rgb,) if model.detail_branch is not None else ())
                branch_events.clear()
                pyramid, pyramid_ms = timed(torch, model.pyramid, inputs)
                decoded, head_ms = timed(torch, model.detect, pyramid)
                branch_ms = sum(start.elapsed_time(end) for start, end in branch_events)

                predictions, post_ms = timed(torch, postprocess_decoded, decoded, info, inverse_letterbox)
                total_ms = (time.perf_counter() - total_start) * 1000
                if measured:
                    row = {
                        "method": method,
                        "image_id": record["id"],
                        "load_letterbox_ms": load_ms,
                        "host_to_device_ms": transfer_ms,
                        "DINO_forward_and_preprocessing_ms": teacher_ms,
                        "FP16_interface_roundtrip_ms": interface_ms,
                        "pyramid_inclusive_ms": pyramid_ms,
                        "RGB_branch_nested_ms": branch_ms,
                        "head_decode_topk_ms": head_ms,
                        "confidence_and_inverse_letterbox_ms": post_ms,
                        "total_ms": total_ms,
                        "detections": len(predictions),
                    }
                    method_rows.append(row)
                    folder = args.output / "predictions" / method
                    folder.mkdir(parents=True, exist_ok=True)
                    np.savetxt(folder / f"{record['id']}.txt", predictions, fmt="%.6f,%.6f,%.6f,%.6f,%.8f,%d,%d,%d")
                    if (index + 1) % 64 == 0 or index == 547:
                        print(f"FRESH_RGB_LATENCY {method} {index + 1}/548", flush=True)
        all_rows.extend(method_rows)
        for metric in method_rows[0]:
            if not metric.endswith("_ms"):
                continue
            values = np.array([row[metric] for row in method_rows])
            all_summary.append(
                {
                    "method": method,
                    "stage": metric,
                    "images": 548,
                    "mean_ms": float(values.mean()),
                    "p50_ms": float(np.quantile(values, 0.5)),
                    "p95_ms": float(np.quantile(values, 0.95)),
                    "teacher_setup_and_model_load_seconds": setup_seconds,
                    "teacher_measured_images": 548,
                    "teacher_warmup_images": args.warmup,
                    "nested_branch_do_not_add_to_pyramid": True,
                }
            )
        dump(
            args.output / f"{method}-latency-bindings.json",
            {
                "run": str(run),
                "bindings": bindings[method],
                "setup_seconds": setup_seconds,
                "teacher_queries": 548 + args.warmup,
                "cache_feature_reads": 0,
                "compute_apps_end": gpu_snapshot(),
            },
        )
        for handle in handles:
            handle.remove()
        del model, extractor
        torch.cuda.empty_cache()
    write_csv(args.output / "latency-per-image.csv", all_rows)
    write_csv(args.output / "latency-summary.csv", all_summary)
    return {
        "images_each": 548,
        "methods": METHODS,
        "teacher_measured_images": 4 * 548,
        "teacher_warmup_images": len(METHODS) * args.warmup,
        "feature_bank_reads": 0,
        "resource_claim": "observational shared-GPU stage timings, not a controlled speed benchmark",
    }


def selfcheck():
    constant = spectral_metrics(np.ones((2, 80, 80)))
    assert constant["high_frequency_fraction"] is None
    signal = np.broadcast_to(np.cos(2 * np.pi * 0.3 * np.arange(80)), (2, 80, 80)).copy()
    wave = spectral_metrics(signal)
    assert abs(wave["high_frequency_fraction"] - 1) < 1e-12
    assert abs(sum(wave["radial_normalized_power"]) - 1) < 1e-12
    scaled = spectral_metrics(signal * 7)
    assert np.allclose(wave["radial_normalized_power"], scaled["radial_normalized_power"])
    assert rectangle_union_area([(1, 1, 3, 3), (2, 2, 4, 4)]) == 7
    info = {"original_shape": [640, 640], "resized_shape": [640, 640], "pad": [0, 0]}
    tiny = np.array([[1, 1, 1, 1, 1, 1, 0, 0]], dtype=float)
    coverage = foreground_weights(tiny, info)
    assert coverage[0, 0] == 1 / 64 and coverage.sum() == 1 / 64
    assert np.array_equal(coverage, foreground_weights(np.concatenate([tiny, tiny]), info))
    padded = {"original_shape": [100, 200], "resized_shape": [320, 640], "pad": [0, 160]}
    valid = original_image_weights(padded)
    assert valid.sum() == 80 * 40
    assert np.all(foreground_weights(tiny, padded) <= valid + 1e-10)
    assert "torch" not in sys.modules
    return {
        "status": "PASS",
        "contracts": [
            "DC removal zero is undefined fraction not zero benefit",
            "known .3cycles/pixel sinusoid in high-frequency range",
            "radial total1",
            "amplitude scale invariance",
            "1px GT retained without containing any P3 cell center",
            "overlapping GT rectangle union counts area once",
            "content background excludes separately measured letterbox padding",
            "no torch/GPU import in selfcheck",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inspect", "selfcheck", "features", "latency"))
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--original-detail-job", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--archive", type=Path, default=ROOT / "datasets/VisDrone/VisDrone2019-DET-val.zip")
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()
    if args.action == "selfcheck":
        print(json.dumps(selfcheck(), indent=2))
        return
    if not all((args.campaign, args.original_detail_job)):
        parser.error("--campaign and --original-detail-job are required")
    if not 0 <= args.warmup <= 548:
        parser.error("warmup must lie in0..548; it does not reduce full measured image coverage")
    manifest, cache, records, bindings, bank = validate_inputs(args, include_bank=args.action == "features")
    protocol = contract(args, bindings, manifest, cache)
    if args.action == "inspect":
        assert "torch" not in sys.modules
        print(json.dumps({"status": "READY_NOT_LAUNCHED", "protocol": protocol, "GPU_execution": False}, indent=2))
        return
    if args.output is None or args.output.exists():
        parser.error("GPU stages require a fresh --output directory; no previous results are overwritten")
    args.output.mkdir(parents=True)
    dump(args.output / "protocol.json", protocol)
    (args.output / "status.txt").write_text(f"RUNNING full548 {args.action}\n")
    started = time.monotonic()
    try:
        result = (
            features(args, manifest, cache, records, bindings, bank)
            if args.action == "features"
            else latency(args, manifest, cache, records, bindings)
        )
        dump(
            args.output / "completion.json",
            {"status": "SUCCEEDED", "action": args.action, "seconds": time.monotonic() - started, **result},
        )
        (args.output / "status.txt").write_text(f"SUCCEEDED full548 {args.action}\n")
    except Exception as error:
        dump(args.output / "failure.json", {"type": type(error).__name__, "message": str(error)})
        (args.output / "status.txt").write_text(f"FAILED {error}\n")
        raise


if __name__ == "__main__":
    main()
