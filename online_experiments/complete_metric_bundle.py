"""CPU evidence bundle from an already completed, sealed prediction pass.

Never imports torch, opens images, calls a detector, or changes the training run.
VOC's native eleven-point AP50 remains the primary endpoint. COCO-style VOC
results are explicitly auxiliary and map difficult objects to crowd, matching
the frozen scorer's conversion rather than claiming native VOC semantics.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import re
import time
from pathlib import Path

import numpy as np

VERSION = "complete-cpu-metrics-v1"
MAX_DETS = (1, 10, 100, 500)
AREAS = ("all", "small", "medium", "large")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows, fields=None):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_valid(values):
    valid = np.asarray(values)[np.asarray(values) > -1]
    return 100 * float(valid.mean()) if valid.size else None


def coco_evidence(samples, predictions, names):
    """One COCOeval evaluation/accumulation gives all requested maxDet outputs."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    from online_experiments.crossdataset_suite import coco_payload

    ground, detections = coco_payload(samples, predictions, names)
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = ground
        gt.createIndex()
        if detections:
            dt = gt.loadRes(detections)
        else:
            dt = COCO()
            dt.dataset = {**ground, "annotations": []}
            dt.createIndex()
        evaluator = COCOeval(gt, dt, "bbox")
        evaluator.params.maxDets = list(MAX_DETS)
        evaluator.evaluate()
        evaluator.accumulate()
    return evaluator, ground


def extract(evaluator, names):
    precision, recall = evaluator.eval["precision"], evaluator.eval["recall"]
    iou50 = int(np.flatnonzero(np.isclose(evaluator.params.iouThrs, 0.5))[0])
    iou75 = int(np.flatnonzero(np.isclose(evaluator.params.iouThrs, 0.75))[0])
    global_metrics, class_rows, area_rows, pr_rows = {}, [], [], []
    for mi, maximum in enumerate(MAX_DETS):
        metrics = {
            "AP": mean_valid(precision[:, :, :, 0, mi]),
            "AP50": mean_valid(precision[iou50, :, :, 0, mi]),
            "AP75": mean_valid(precision[iou75, :, :, 0, mi]),
            "AP_small": mean_valid(precision[:, :, :, 1, mi]),
            "AP_medium": mean_valid(precision[:, :, :, 2, mi]),
            "AP_large": mean_valid(precision[:, :, :, 3, mi]),
            "AR": mean_valid(recall[:, :, 0, mi]),
        }
        global_metrics[str(maximum)] = metrics
        for ai, area in enumerate(AREAS):
            area_rows.append({
                "max_det": maximum, "area": area,
                "AP": mean_valid(precision[:, :, :, ai, mi]),
                "AP50": mean_valid(precision[iou50, :, :, ai, mi]),
                "AP75": mean_valid(precision[iou75, :, :, ai, mi]),
                "AR": mean_valid(recall[:, :, ai, mi]),
            })
        for ki, name in enumerate(names):
            class_rows.append({
                "category_id_zero_based": ki, "class_name": name, "max_det": maximum,
                "AP": mean_valid(precision[:, :, ki, 0, mi]),
                "AP50": mean_valid(precision[iou50, :, ki, 0, mi]),
                "AP75": mean_valid(precision[iou75, :, ki, 0, mi]),
                "AR": mean_valid(recall[:, ki, 0, mi]),
                **{f"AP_{area}": mean_valid(precision[:, :, ki, ai, mi])
                   for ai, area in enumerate(AREAS) if ai},
            })
            if maximum in (100, 500):
                for ti, iou in ((iou50, 0.5), (iou75, 0.75)):
                    for ri, rec in enumerate(evaluator.params.recThrs):
                        value = float(precision[ti, ri, ki, 0, mi])
                        score = float(evaluator.eval["scores"][ti, ri, ki, 0, mi])
                        pr_rows.append({
                            "class_name": name, "max_det": maximum, "iou": iou,
                            "recall_grid": float(rec), "precision": value if value >= 0 else None,
                            "interpolated_score": score if value >= 0 else None,
                        })
    return global_metrics, class_rows, area_rows, pr_rows


def check_native(dataset, native, samples, predictions, names, coco):
    if dataset == "voc0712":
        from online_experiments.crossdataset_suite import evaluate_voc07

        recomputed = evaluate_voc07(samples, predictions, names)
        if recomputed != native:
            raise RuntimeError("Native VOC endpoint differs from sealed final predictions")
    elif dataset == "sku110k":
        for maximum in (100, 500):
            original = native["by_max_detections"][str(maximum)]
            for key, value in original.items():
                actual = coco[str(maximum)][key]
                if value is None and actual is None:
                    continue
                if value is None or actual is None or not math.isclose(value, actual, rel_tol=1e-10, abs_tol=1e-9):
                    raise RuntimeError(f"Native SKU endpoint mismatch: maxDet{maximum} {key}")
        recomputed = native
    else:
        raise ValueError("This protocol supports full VOC0712 or SKU110K only")
    return recomputed


def counts(ground, names, evaluator):
    rows = []
    for ki, name in enumerate(names):
        objects = [item for item in ground["annotations"] if item["category_id"] == ki + 1]
        for area, (low, high) in zip(AREAS, evaluator.params.areaRng, strict=True):
            selected = [item for item in objects if low <= item["area"] <= high]
            rows.append({
                "class_name": name, "area": area, "area_lower_inclusive": low,
                "area_upper_inclusive": high, "all_objects": len(selected),
                "non_crowd_objects": sum(not item["iscrowd"] for item in selected),
                "crowd_or_difficult_objects": sum(bool(item["iscrowd"]) for item in selected),
            })
    return rows


def resource_evidence(run, summary, train_log, provenance=None):
    """Copy measurements; absent fields stay null, never infer FPS from cache."""
    keys = (
        "training_wall_seconds", "evaluation_wall_seconds", "train_seconds", "wall_seconds",
        "cold_seconds", "warm_seconds", "peak_vram_gib", "cache_bytes", "cache_build_charged_seconds",
        "data_prepare_charged_seconds", "training_cache_build_seconds", "teacher_query_images",
        "native_backbone_training_images", "physical_training_teacher_queries", "query_images",
        "query_calls", "parameters", "trainable_parameters", "detail_parameters",
        "observed_mean_predict_wall_ms", "raw_image_seconds", "raw_image_images", "final_inference_seconds",
        "consumed_cache_parts", "retained_total_cache_bytes", "cache_build_logical_teacher_images",
        "quantization_seconds", "compressed_training_bytes",
    )
    result = {key: summary.get(key) for key in keys}
    checkpoint = run / "weights/last.pt"
    result["checkpoint_bytes"] = checkpoint.stat().st_size if checkpoint.is_file() else None
    result["resource_note"] = "Copied saved measurements; shared-GPU observational time. Null means not measured. No fresh latency/FPS inferred from cached-feature evaluation."
    result["training_validation_note"] = "No held-out validation during native training. Native metrics/* and val/* CSV zeros are placeholders and are excluded from curve exports."
    result["pipeline_parameters"] = result["frozen_backbone_parameters"] = None
    if summary.get("version", "").startswith("direct-"):
        result["parameters"] = result["trainable_parameters"] = None
        result["parameter_note"] = "Pre-nc initialization counts and EMA requires_grad flags are not valid trained-model parameter counts."
        result["peak_vram_scope"] = "Process peak through training and final prediction; no separate train/eval reset."
        result["prediction_time_scope"] = "Saved per-image mean includes PIL read, model.predict, warmup and output CPU transfer; P95 unavailable."
    else:
        initialization = read(run / "initialization.json")
        result.update({key: initialization.get(key) for key in ("parameters", "trainable_parameters", "detail_parameters")})
        result["parameter_note"] = "Parameters field is cache detection head/branch only. Pipeline total adds frozen foundation backbone only when identity-bound audit is available."
        result["peak_vram_scope"] = "Maximum epoch training peak; may differ from native train+evaluation process peak."
        result["prediction_time_scope"] = "Cached-feature head/branch inference excludes a fresh foundation backbone forward; cannot report as full-pipeline FPS."
        proof = Path(__file__).resolve().parents[1] / "research/checks/single-pass-metrics-20261112/frozen-teacher-parameters.json"
        provenance_path = Path(provenance) if provenance else run.parent.parent / "provenance.json"
        config = read(run / "args.json")
        if proof.is_file() and provenance_path.is_file():
            evidence, provenance = read(proof), read(provenance_path)
            if sha(provenance_path) != config["provenance_sha256"]:
                raise RuntimeError("Cache run teacher provenance hash differs")
            if evidence["identity"] == provenance["identity"]:
                result["frozen_backbone_parameters"] = evidence["total_parameters"]
                result["pipeline_parameters"] = evidence["total_parameters"] + initialization["parameters"]
                result["backbone_parameter_evidence_sha256"] = sha(proof)
    if train_log and summary.get("version", "").startswith("direct-"):
        content = Path(train_log).read_text(encoding="utf-8", errors="replace")
        matches = re.findall(r"([^\n\r]+?summary[^\n\r]*?):\s*([\d,]+) layers,\s*([\d,]+) parameters,\s*([\d,]+) gradients(?:,\s*([\d.]+) GFLOPs)?", content)
        matches = [match for match in matches if int(match[3].replace(",", "")) > 0 and "fused" not in match[0].lower()]
        if matches:
            model, layers, params, grads, flops = matches[-1]
            result.update({"parameters": int(params.replace(",", "")), "trainable_parameters": int(grads.replace(",", "")),
                           "model_layers": int(layers.replace(",", "")), "reported_GFLOPs": float(flops) if flops else None,
                           "parameter_source": "Dataset-nc training model summary in saved log", "model_summary": model})
            if summary.get("version", "").startswith("direct-"):
                result["pipeline_parameters"] = result["parameters"]
                result["frozen_backbone_parameters"] = 0
    source = run / "results.csv"
    curve_rows = []
    if source.is_file():
        with source.open(newline="", encoding="utf-8-sig") as stream:
            original = [{key.strip(): value for key, value in row.items()} for row in csv.DictReader(stream)]
        if original:
            fields = [key for key in original[0] if key in ("epoch", "time", "train_seconds", "train_loss", "lr")
                      or key.startswith(("train/", "lr/"))]
            curve_rows = [{key: row[key] for key in fields} for row in original]
    elif isinstance(summary.get("epochs"), list) and summary["epochs"]:
        original = summary["epochs"]
        if [row["epoch"] for row in original] != list(range(1, len(original) + 1)):
            raise RuntimeError("Saved epoch history is not contiguous")
        fields = [key for key in original[0] if key in (
            "epoch", "train_loss", "train_seconds", "query_images", "query_calls", "lr", "peak_vram_gib")]
        curve_rows = [{key: row[key] for key in fields} for row in original]
    return result, curve_rows


def run_bundle(args):
    run, output, manifest = args.run.resolve(), args.output.resolve(), args.manifest.resolve()
    if not ((run / "DONE").is_file() or (run / "DONE.json").is_file()):
        raise RuntimeError("Only final completed training runs are eligible")
    if (run / "DONE.json").is_file():
        done = read(run / "DONE.json")
        if done["endpoint_sha256"] != sha(run / "endpoint.json") or done["completed_epochs"] != 100:
            raise RuntimeError("Native DONE endpoint hash/epoch differs")
    elif (run / "DONE").read_text(encoding="utf-8").strip() != "SUCCEEDED":
        raise RuntimeError("Cached DONE marker is not SUCCEEDED")
    if sha(manifest) != read(run / "args.json")["manifest_sha256"]:
        raise RuntimeError("Dataset manifest does not match completed run")
    if read(run / "summary.json")["completed_epochs"] != 100:
        raise RuntimeError("Only fixed final100-epoch completed runs are eligible")
    endpoint = read(run / "endpoint.json")
    for name, expected in endpoint["sha256"].items():
        path = (run / name).resolve()
        if not path.is_relative_to(run) or sha(path) != expected:
            raise RuntimeError(f"Completed run endpoint hash mismatch: {name}")
    sources = {str(path): sha(path) for path in (
        manifest, run / "endpoint.json", run / "predictions.json", run / "metrics.json", run / "summary.json",
        Path(__file__), Path(__file__).with_name("crossdataset_suite.py"),
    )}
    for path in (run / "initialization.json", run / "args.json", run.parent.parent / "provenance.json"):
        if path.is_file():
            sources[str(path)] = sha(path)
    if args.provenance:
        sources[str(args.provenance.resolve())] = sha(args.provenance)
    for path in (run / "DONE", run / "DONE.json"):
        if path.is_file():
            sources[str(path)] = sha(path)
    proof = Path(__file__).resolve().parents[1] / "research/checks/single-pass-metrics-20261112/frozen-teacher-parameters.json"
    if proof.is_file():
        sources[str(proof)] = sha(proof)
    if args.train_log:
        sources[str(args.train_log.resolve())] = sha(args.train_log)
    if (run / "results.csv").is_file():
        sources[str(run / "results.csv")] = sha(run / "results.csv")
    if (output / "completion.json").is_file():
        seal = read(output / "completion.json")
        if seal["input_sha256"] != sources or any(sha(output / name) != value for name, value in seal["output_sha256"].items()):
            raise RuntimeError("Existing metric bundle input/output seal changed")
        print("ALREADY_COMPLETE", output)
        return
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Unsealed output exists; preserve it and choose a new output path")
    meta, saved = read(manifest), read(run / "predictions.json")
    samples, names = meta["splits"]["test"], meta["class_names"]
    expected_count = {"voc0712": 4952, "sku110k": 2936}[meta["dataset"]]
    if len(samples) != expected_count or len({item["sample_id"] for item in samples}) != expected_count:
        raise RuntimeError("Full official test split required")
    if saved["coverage"] != [sample["sample_id"] for sample in samples] or saved["max_det"] < 500:
        raise RuntimeError("Full ordered prediction coverage and maxDet500 required")
    for item in saved["predictions"]:
        if not all(math.isfinite(float(value)) for value in (*item["bbox_xyxy0_half_open"], item["score"])):
            raise RuntimeError("Non-finite saved prediction")
    evaluator, ground = coco_evidence(samples, saved["predictions"], names)
    metrics, classes, areas, curves = extract(evaluator, names)
    native = check_native(meta["dataset"], read(run / "metrics.json")["primary"], samples, saved["predictions"], names, metrics)
    resources, train_curves = resource_evidence(run, read(run / "summary.json"), args.train_log, args.provenance)
    output.mkdir(parents=True, exist_ok=True)
    dump(output / "metrics.json", {
        "version": VERSION, "dataset": meta["dataset"], "evaluated_images": len(samples), "scale": "0-100",
        "native_primary": native, "native_consistency": "PASS",
        "saved_prediction_protocol": {key: value for key, value in saved.items() if key not in ("predictions", "coverage")},
        "auxiliary_coco_by_max_detections": metrics,
        "definitions": {
            "COCO_AP": "101-point interpolated AP averaged over IoU0.50:0.05:0.95, area=all; equal category weighting, not micro AP",
            "AR": "Maximum recall averaged over IoU0.50:0.05:0.95 at selected maxDet; equal category weighting",
            "VOC_auxiliary_only": "COCO-style VOC results use half-open boxes and map difficult to crowd. Native VOC eleven-point AP50 remains unchanged and primary.",
            "area": "Original-image bbox area. COCO ranges: small[0,1024], medium[1024,9216], large[9216,1e10]; boundaries overlap per COCOeval.",
            "PR": "COCOeval interpolated precision/score tensors at 101 recall grid points; -1/CSV empty means undefined. This is not confidence-sweep P/R/F1.",
            "missing": "No new inference; predictions below frozen score0.001 or beyond frozen top500 cannot be recovered.",
        },
    })
    write_csv(output / "per-class.csv", classes)
    write_csv(output / "per-area.csv", areas)
    write_csv(output / "pr-curves.csv", curves)
    write_csv(output / "object-counts.csv", counts(ground, names, evaluator))
    np.savez_compressed(output / "tensors.npz", precision=evaluator.eval["precision"], recall=evaluator.eval["recall"],
                        scores=evaluator.eval["scores"], iou_thresholds=evaluator.params.iouThrs,
                        recall_thresholds=evaluator.params.recThrs, max_detections=np.asarray(MAX_DETS),
                        class_names=np.asarray(names), area_labels=np.asarray(AREAS), area_ranges=np.asarray(evaluator.params.areaRng))
    dump(output / "resource-metrics.json", resources)
    if train_curves:
        write_csv(output / "training-curves.csv", train_curves)
    dump(output / "provenance.json", {"version": VERSION, "input_sha256": sources, "run": str(run), "manifest": str(manifest),
        "pycocotools_version": importlib.metadata.version("pycocotools"), "CPU_only": True,
        "additional_training_images": 0, "additional_prediction_images": 0,
        "tensor_dimensions": {"precision_and_scores": "IoU,recall_grid,class,area,maxDet", "recall": "IoU,class,area,maxDet"}})
    (output / "status.txt").write_text("SUCCEEDED\n", encoding="utf-8")
    for name, value in sources.items():
        if sha(name) != value:
            raise RuntimeError("Input/source changed during CPU scoring")
    dump(output / "completion.json", {"version": VERSION, "input_sha256": sources,
        "output_sha256": {path.name: sha(path) for path in sorted(output.iterdir()) if path.is_file()},
        "completed_unix": time.time()})
    print("SUCCEEDED", output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-log", type=Path)
    parser.add_argument("--provenance", type=Path, help="Sealed original dataset job provenance for relocated cached runs")
    run_bundle(parser.parse_args())


if __name__ == "__main__":
    main()
