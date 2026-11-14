"""Export saved cost/accuracy evidence with no training, inference or rescoring."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            value.update(block)
    return value.hexdigest()


def read(path, sources):
    path = Path(path)
    result = json.loads(path.read_text(encoding="utf-8"))
    sources[str(path)] = sha(path)
    return result


def checked(path, expected, sources):
    actual = sha(path)
    if actual != expected:
        raise RuntimeError(f"SHA256 mismatch: {path}")
    sources[str(path)] = actual


def done(run, sources):
    path = run / "DONE"
    if path.read_text(encoding="utf-8").strip() != "SUCCEEDED":
        raise RuntimeError(f"Unfinished run: {run}")
    sources[str(path)] = sha(path)


def divide(value, divisor):
    return None if value is None else value / divisor


def base_row(dataset, method, scope, run):
    return {
        "dataset": dataset, "method": method, "scope": scope, "seed": 0,
        "run": str(run), "primary_metric": None, "primary_value": None,
        "mAP50_95": None, "mAP50": None, "mAP75": None,
        "dense_mAP50_95_max500": None, "reference_weighted_AP": None,
        "reference_weighted_AP50": None, "metric_protocol": None,
        "train_images": None, "test_images": None, "completed_epochs": None,
        "train_elapsed_seconds": None, "train_elapsed_h": None,
        "native_model_train_wall_seconds": None, "warm_process_seconds": None,
        "warm_process_h": None, "cold_accounted_seconds": None,
        "cold_accounted_h": None, "cache_build_charged_seconds": None,
        "data_prepare_charged_seconds": None, "final_evaluation_seconds": None,
        "query_images": None, "query_calls": None,
        "native_backbone_training_images": None,
        "cache_build_logical_teacher_images": None,
        "consumed_cache_bytes": None, "consumed_cache_GiB": None,
        "retained_total_cache_bytes": None, "peak_allocated_vram_gib": None,
        "peak_vram_scope": None, "trainable_parameters": None,
        "head_parameters": None, "pipeline_parameters": None,
        "raw_train_seconds": None, "raw_image_seconds": None,
        "controlled_gpu_hours": None, "energy_kwh": None,
        "measured_training_flops": None, "training_only_cold_seconds": None, "time_scope": None,
        "cold_cost_scope": None, "parameter_scope": None,
        "comparability_note": None, "source_verification": None,
    }


def times(row, summary, native=False):
    if native:
        row["train_elapsed_seconds"] = summary.get("training_wall_seconds")
        row["native_model_train_wall_seconds"] = row["train_elapsed_seconds"]
        row["final_evaluation_seconds"] = summary.get("evaluation_wall_seconds")
        row["time_scope"] = (
            "Native model.train() elapsed wall incl trainer setup/overhead; "
            "not the cached trainer sum of epoch train phases. Complete warm/cold process cost unavailable."
        )
    else:
        row["train_elapsed_seconds"] = summary.get("train_seconds")
        row["warm_process_seconds"] = summary.get("wall_seconds")
        row["cache_build_charged_seconds"] = summary.get("cache_build_charged_seconds")
        row["data_prepare_charged_seconds"] = summary.get("data_prepare_charged_seconds")
        parts = [row[key] for key in (
            "warm_process_seconds", "cache_build_charged_seconds", "data_prepare_charged_seconds"
        )]
        if all(value is not None for value in parts):
            row["cold_accounted_seconds"] = sum(parts)
        row["final_evaluation_seconds"] = summary.get(
            "final_inference_seconds", summary.get("official_evaluation_seconds")
        )
        row["time_scope"] = (
            "Checkpoint-accounted sum of synchronized epoch training elapsed wall; "
            "warm process includes validation/evaluation and overhead; shared GPU, not dedicated compute time. "
            "Uncheckpointed work lost on interruption is not recovered."
        )
    for seconds, hours in (
        ("train_elapsed_seconds", "train_elapsed_h"),
        ("warm_process_seconds", "warm_process_h"),
        ("cold_accounted_seconds", "cold_accounted_h"),
    ):
        row[hours] = divide(row[seconds], 3600)
    row["cold_cost_scope"] = (
        "Unavailable: native full warm/cold process cost was not recorded"
        if native else
        "Accounted cold = saved warm wall + consumed feature-bank build incl validation/test features "
        "+ saved data/snapshot prepare. Dataset downloads/extraction and foundation pretraining excluded. "
        "Historical shared cache build is charged once per run; not a fresh controlled cold measurement."
    )


def resources(row, summary, initialization, native=False):
    row["consumed_cache_bytes"] = summary.get("cache_bytes")
    row["consumed_cache_GiB"] = divide(row["consumed_cache_bytes"], 2**30)
    row["retained_total_cache_bytes"] = summary.get("retained_total_cache_bytes")
    row["peak_allocated_vram_gib"] = summary.get("peak_vram_gib")
    row["peak_vram_scope"] = summary.get("peak_vram_scope") or (
        "torch.cuda.max_memory_allocated run peak; train/eval not independently reset. "
        "Not pure-train VRAM, reserved memory or whole-device usage."
    )
    for field in ("query_images", "query_calls", "raw_train_seconds", "raw_image_seconds"):
        row[field] = summary.get(field)
    row["native_backbone_training_images"] = summary.get("native_backbone_training_images")
    if native:
        row["query_images"] = summary.get("teacher_query_images")
    row["cache_build_logical_teacher_images"] = summary.get(
        "cache_build_logical_teacher_images", summary.get("cache_build_query_images")
    )
    row["trainable_parameters"] = initialization.get("trainable_parameters")
    if row["trainable_parameters"] is None and not native:
        row["trainable_parameters"] = initialization.get("parameters")
    row["head_parameters"] = None if native else initialization.get("parameters")
    row["parameter_scope"] = (
        "Native dataset-nc training-model counts from sealed metric bundle; pre-nc initialization ignored."
        if native else
        "Actual initialized cached detection head/branch only; frozen teacher excluded. "
        "No teacher total inferred; see separate frozen-teacher proof."
    )
    if native:
        row["pipeline_parameters"] = initialization.get("pipeline_parameters")


def vis_rows(root, original, sources):
    paper = root / "runs/paper/paper-baselines/job-20261107-Imo9Yocz/train"
    detail = original / "runs/paper/detail-probe/job-20261022-hjP4y3Lf/train"
    old = original / "runs/paper/full-visdrone/job-start-eDmH0odX/train"
    specs = [
        ("detail25", detail / "detail25-s0"),
        ("vit-adapter-cache25", paper / "vit-adapter-cache25-s0"),
        ("vit-adapter-cache0", paper / "vit-adapter-cache0-s0"),
        ("mix25", paper / "mix25-s0"), ("online100", paper / "online100-s0"),
        ("cache2", old / "cache2-s0"),
        ("frofa-c2-d01", paper / "frofa-c2-d01-s0"),
        ("loffta-c2-n005", paper / "loffta-c2-n005-s0"),
        ("cache-reuse-c1-int8", paper / "cache-reuse-c1-int8-s0"),
    ]
    rows = []
    for method, run in specs:
        done(run, sources)
        summary = read(run / "summary.json", sources)
        if summary["completed_epochs"] != 100:
            raise RuntimeError(f"Incomplete final endpoint: {run}")
        read(run / "args.json", sources)
        init_path = run / "initialization.json"
        initialization = read(init_path, sources) if init_path.exists() else {}
        metric_path = (
            original / "runs/paper/visdrone-eval/job-full-20260927/cache2-last.json"
            if method == "cache2" else run / "official-final.json"
        )
        envelope = read(metric_path, sources)
        metrics = envelope.get("metrics", envelope)
        macro = metrics.get("macro_AP_diagnostic")
        if macro is None:
            macro = sum(metrics["class_AP"]) / len(metrics["class_AP"])
        row = base_row("VisDrone", method, "fixed100-full548", run)
        row.update(
            primary_metric="reference_category_macro_mAP50_95", primary_value=macro,
            mAP50_95=macro,
            mAP50=sum(metrics["class_AP50"]) / len(metrics["class_AP50"]),
            mAP75=sum(metrics["class_AP75"]) / len(metrics["class_AP75"]),
            reference_weighted_AP=metrics["AP"], reference_weighted_AP50=metrics["AP50"],
            metric_protocol="Pinned VisDrone reference matching/interpolation; equal class macro and weighted AP separate",
            train_images=summary["train_images"], test_images=548, completed_epochs=100,
            comparability_note=(
                "Same full test set and final100; source trainers/dates and held-out evaluation frequency differ. "
                "Shared-GPU times are observational; no strict speedup or equal-time claim."
            ),
            source_verification="DONE and completed100 checked; consumed JSON SHA recorded; no new inference/evaluation",
        )
        times(row, summary)
        resources(row, summary, initialization)
        rows.append(row)
    return rows


def metric_fields(row, metrics, dataset, bundle):
    native = metrics["native_primary"] if bundle else metrics["primary"]
    if dataset == "voc0712":
        row["primary_metric"] = "VOC2007_native_mAP50_11point"
        row["primary_value"] = row["mAP50"] = native["mAP50_11point"]
        row["metric_protocol"] = "Native VOC2007 eleven-point AP50; no native mAP50_95"
    else:
        groups = native["by_max_detections"]
        row["primary_metric"] = "COCO_native_mAP50_95_max100"
        row["primary_value"] = row["mAP50_95"] = groups["100"]["AP"]
        row["mAP50"] = groups["100"]["AP50"]
        row["mAP75"] = groups["100"]["AP75"]
        row["dense_mAP50_95_max500"] = groups["500"]["AP"]
        row["metric_protocol"] = "COCOeval native bbox macro AP; max100 and dense max500 separate"


def cross_rows(root, sources):
    rows = []
    bundle_root = root / "runs/paper/metric-bundles/job-20261112-single-pass"
    cases = [
        ("voc0712", "master-n"), ("voc0712", "yolo12-n"),
        ("sku110k", "master-n"), ("sku110k", "yolo12-n"),
        ("sku110k", "frofa-c2-d01"),
    ]
    for dataset, method in cases:
        folder = bundle_root / dataset / method / "bundle"
        completion = read(folder / "completion.json", sources)
        for name, expected in completion["output_sha256"].items():
            checked(folder / name, expected, sources)
        metrics = read(folder / "metrics.json", sources)
        if metrics["native_consistency"] != "PASS":
            raise RuntimeError(f"Native score consistency failed: {folder}")
        measured = read(folder / "resource-metrics.json", sources)
        provenance = read(folder / "provenance.json", sources)
        run = Path(provenance["run"])
        for name in ("summary.json", "args.json", "initialization.json"):
            path = run / name
            checked(path, completion["input_sha256"][str(path)], sources)
        summary = read(run / "summary.json", sources)
        read(run / "args.json", sources)
        init = read(run / "initialization.json", sources)
        native = method in ("master-n", "yolo12-n")
        row = base_row(dataset, method, "priority-single-seed", run)
        metric_fields(row, metrics, dataset, True)
        row.update(
            train_images=summary.get("train_images"), test_images=metrics["evaluated_images"],
            completed_epochs=summary.get("completed_epochs"),
            comparability_note="Shared-GPU observation; native model.train wall and cached train-phase sum differ.",
            source_verification="All bundle output SHA checked; consumed original summary/args/init SHA match completion inputs",
        )
        times(row, measured, native)
        resources(row, measured, measured if native else init, native)
        rows.append(row)
    controls = [
        ("voc0712", "job-20261110-voc0712-s0-v2", "cache1"),
        ("voc0712", "job-20261110-voc0712-s0-v2", "cache2"),
        ("voc0712", "job-20261110-voc0712-s0-v2", "mix25"),
        ("sku110k", "job-20261110-sku110k-official-loader-v2", "cache1"),
    ]
    for dataset, job, method in controls:
        run = root / "runs/paper/crossdataset" / job / "train" / f"{method}-s0"
        done(run, sources)
        endpoint = read(run / "endpoint.json", sources)
        for name in ("summary.json", "metrics.json", "args.json", "initialization.json"):
            checked(run / name, endpoint["sha256"][name], sources)
        summary = read(run / "summary.json", sources)
        init = read(run / "initialization.json", sources)
        metrics = read(run / "metrics.json", sources)
        if summary["completed_epochs"] != 100 or endpoint["fixed_epoch"] != 100:
            raise RuntimeError(f"Control is not fixed100: {run}")
        row = base_row(dataset, method, "historical-control", run)
        metric_fields(row, metrics, dataset, False)
        row.update(
            train_images=summary["train_images"], test_images=summary["test_images"], completed_epochs=100,
            comparability_note="Historical completed control, not an additional core-queue completion; shared-GPU time.",
            source_verification="DONE and fixed100 checked; summary/metrics/args/init SHA match endpoint",
        )
        times(row, summary)
        resources(row, summary, init)
        rows.append(row)
    return rows


def comparisons(rows):
    baseline = next(row for row in rows if row["method"] == "online100")
    result = []
    for row in rows:
        result.append({
            "dataset": "VisDrone", "method": row["method"], "reference_method": "online100",
            "macro_mAP_gap_points": row["mAP50_95"] - baseline["mAP50_95"],
            "weighted_AP_gap_points": row["reference_weighted_AP"] - baseline["reference_weighted_AP"],
            "observed_train_elapsed_ratio": divide(row["train_elapsed_seconds"], baseline["train_elapsed_seconds"]),
            "observed_cold_accounted_ratio": divide(row["cold_accounted_seconds"], baseline["cold_accounted_seconds"]),
            "online_query_images_saving_pct": 100 * (1 - row["query_images"] / baseline["query_images"]),
            "online_query_calls_saving_pct": 100 * (1 - row["query_calls"] / baseline["query_calls"]),
            "interpretation": "Observational wall-time ratios only; not controlled speedup, equal-time or FLOP savings.",
        })
    return result


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export(root, original, output):
    sources = {str(Path(__file__).resolve()): sha(__file__)}
    vis = vis_rows(root, original, sources)
    rows = vis + cross_rows(root, sources)
    proof = root / "research/checks/single-pass-metrics-20261112/frozen-teacher-parameters.json"
    if proof.exists():
        read(proof, sources)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "measurements.csv", rows)
    write_csv(output / "vs-online100-observed.csv", comparisons(vis))
    manifest = {
        "schema": "cost-accuracy-saved-evidence-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(rows), "visdrone_comparison_rows": len(vis),
        "input_sha256": sources,
        "output_sha256": {name: sha(output / name) for name in ("measurements.csv", "vs-online100-observed.csv")},
        "training_calls": 0, "inference_calls": 0, "metric_evaluation_calls": 0,
        "missing_policy": "Empty CSV / JSON null means unavailable; never zero-filled or estimated.",
        "cost_policy": "Elapsed hours are wall-clock observations, not precise GPU compute hours. No energy or training FLOPs measured.",
        "cache_policy": "Consumed storage includes held-out feature banks. Original FP16 banks may remain; consumed bytes are not total disk occupancy.",
        "cold_formula": "cold_accounted_seconds = warm_process_seconds + cache_build_charged_seconds + data_prepare_charged_seconds",
        "cold_policy": "Native complete process/cold fields remain missing. Existing cache build measurements include held-out features, are historically charged, not controlled reruns.",
        "vram_policy": "Saved allocated-memory peaks and their original scopes; not pure training VRAM or total device usage.",
        "parameter_policy": "Cached head/branch actual initialization only; no teacher total added without per-run identity audit. Separate proof is recorded when present.",
        "strict_speedup_claim_supported": False,
        "controlled_equal_time_results_available": False,
    }
    (output / "sources.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"rows": len(rows), "visdrone_rows": len(vis), "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--original", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    print(json.dumps(export(
        root, args.original or root.parent / "YOLO-Master",
        args.output or root / "research/metrics/cost-accuracy",
    )))


if __name__ == "__main__":
    main()
