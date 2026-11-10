"""CPU-only, final-checkpoint VisDrone evidence; never train or import torch.

Official-reference per-class scores remain separate from COCO-style size metrics
and TIDE AP50 error diagnostics. Original predictions and annotations are read-only.
"""

import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from online_experiments.visdrone_reference_eval import REFERENCE_COMMIT, evaluate_prepared, prepare_arrays

NAMES = ("pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor")
AREA_NAMES = ("all", "small", "medium", "large")
AREAS = ((0, 1e10), (0, 32**2), (32**2, 96**2), (96**2, 1e10))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def rows(raw):
    if not raw.strip():
        return np.empty((0, 8), dtype=np.float64)
    result = np.loadtxt(io.BytesIO(raw), delimiter=",", ndmin=2)
    if result.shape[1] != 8 or not np.isfinite(result).all():
        raise ValueError("Expected finite original-coordinate eight-column rows")
    return result


def load_annotations(source, archive):
    manifest = read(source / "dataset-manifest.json")
    samples = manifest["splits"]["val"]
    if manifest["smoke"] or manifest["test_dev_used"] or len(samples) != 548:
        raise ValueError("Only the complete 548-image development split is supported")
    expected = {Path(name).stem: value for name, value in manifest["raw_val_annotation_sha256"].items()}
    if len(expected) != 548:
        raise ValueError("Original annotation hashes must cover all 548 images")
    records = []
    with zipfile.ZipFile(archive) as stream:
        annotations = {Path(n).stem: n for n in stream.namelist() if "/annotations/" in n and n.endswith(".txt")}
        if set(annotations) != set(expected):
            raise ValueError("Raw archive and frozen annotation identity differ")
        for sample in samples:
            key = Path(sample["image_path"]).stem
            raw = stream.read(annotations[key])
            if hashlib.sha256(raw).hexdigest() != expected[key]:
                raise ValueError(f"Changed raw annotation: {key}")
            records.append({"id": key, "gt": rows(raw), "shape": sample["letterbox"]["original_shape"]})
    return records, expected


def load_predictions(run, records, annotation_hashes):
    official = read(run / "official-final.json")
    if official["annotation_sha256"] != annotation_hashes:
        raise ValueError(f"Run was scored on different raw annotations: {run}")
    if official["checkpoint_sha256"] != sha(run / "weights/last.pt"):
        raise ValueError(f"Final checkpoint hash mismatch: {run}")
    expected = {record["id"] for record in records}
    files = {p.stem: p for p in (run / "predictions/last").glob("*.txt")}
    if set(files) != expected or set(official["prediction_sha256"]) != expected:
        raise ValueError(f"Prediction coverage is not exactly 548: {run}")
    cases = []
    for record in records:
        key = record["id"]
        if sha(files[key]) != official["prediction_sha256"][key]:
            raise ValueError(f"Changed prediction: {key}")
        det = rows(files[key].read_bytes())
        if len(det) > 500 or np.any(np.diff(det[:, 4]) > 0) or np.any(~np.isin(det[:, 5], np.arange(1, 11))):
            raise ValueError(f"Invalid score ordering, maxDets, or classes: {key}")
        cases.append(prepare_arrays(record["gt"], det, record["shape"]))
    return cases, official


def coco_data(records, cases):
    images, annotations, detections = [], [], []
    annotation_id = 1
    for image_id, (record, (gt, det)) in enumerate(zip(records, cases)):
        height, width = record["shape"]
        images.append({"id": image_id, "height": height, "width": width, "file_name": record["id"] + ".jpg"})
        for box in gt:
            if not 1 <= box[5] <= 10:
                continue
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(box[5]),
                    "bbox": box[:4].tolist(),
                    "area": float(box[2] * box[3]),
                    "iscrowd": int(box[4] != 0),
                }
            )
            annotation_id += 1
        detections.extend(
            {"image_id": image_id, "category_id": int(box[5]), "bbox": box[:4].tolist(), "score": float(box[4])}
            for box in det
        )
    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i + 1, "name": name} for i, name in enumerate(NAMES)],
    }, detections


def positive_mean(values):
    selected = values[values >= 0]
    return float(selected.mean() * 100) if selected.size else None


def auxiliary_cases(cases):
    """Keep invalid boxes as zero-area FP candidates; never silently drop them.

    The frozen VisDrone reference permits negative prediction width/height. COCO
    instead excludes negative-area detections even in its all-area result. Clamp
    negative sizes to zero only for auxiliary backends, preserving scores,
    coordinates, categories, number of boxes and the unmodified official inputs.
    """
    result, count, scores = [], 0, []
    for gt, det in cases:
        det = det.copy()
        invalid = np.any(det[:, 2:4] < 0, axis=1)
        count += int(invalid.sum())
        scores.extend(det[invalid, 4].tolist())
        det[:, 2:4] = np.maximum(det[:, 2:4], 0)
        result.append((gt, det))
    return result, {
        "negative_size_predictions": count,
        "maximum_score": max(scores, default=None),
        "auxiliary_handling": "negative sizes clamped to zero, no prediction discarded",
    }


def evaluate_coco(records, cases):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    dataset, detections = coco_data(records, cases)
    with contextlib.redirect_stdout(io.StringIO()):
        truth = COCO()
        truth.dataset = dataset
        truth.createIndex()
        predictions = truth.loadRes(detections)
        evaluator = COCOeval(truth, predictions, "bbox")
        evaluator.params.imgIds = list(range(len(records)))
        # Export already imposed a GLOBAL 500-box image limit; COCOeval also has a per-category cap.
        evaluator.params.maxDets = [1, 10, 500]
        evaluator.params.areaRng = [list(pair) for pair in AREAS]
        evaluator.params.areaRngLbl = list(AREA_NAMES)
        evaluator.evaluate()
        evaluator.accumulate()
    precision, recall = evaluator.eval["precision"], evaluator.eval["recall"]
    sizes = []
    for index, name in enumerate(AREA_NAMES):
        valid_gt = [
            a for a in dataset["annotations"] if not a["iscrowd"] and AREAS[index][0] <= a["area"] <= AREAS[index][1]
        ]
        sizes.append(
            {
                "area": name,
                "AP": positive_mean(precision[:, :, :, index, -1]),
                "AP50": positive_mean(precision[0, :, :, index, -1]),
                "AP75": positive_mean(precision[5, :, :, index, -1]),
                "AR500": positive_mean(recall[:, :, index, -1]),
                "positive_gt_count": len(valid_gt),
            }
        )
    per_class = []
    for index, name in enumerate(NAMES):
        per_class.append(
            {
                "class_id": index + 1,
                "class_name": name,
                "AP": positive_mean(precision[:, :, index, 0, -1]),
                "AP50": positive_mean(precision[0, :, index, 0, -1]),
                "AP75": positive_mean(precision[5, :, index, 0, -1]),
                "AR500": positive_mean(recall[:, index, 0, -1]),
            }
        )
    return {"sizes": sizes, "per_class": per_class}


def tide_data(records, cases):
    from tidecv import Data

    truth, predictions = Data("VisDrone-prepared", max_dets=500), Data("final-predictions", max_dets=500)
    for index, name in enumerate(NAMES):
        truth.add_class(index + 1, name)
        predictions.add_class(index + 1, name)
    for image_id, (record, (gt, det)) in enumerate(zip(records, cases)):
        truth.add_image(image_id, record["id"])
        predictions.add_image(image_id, record["id"])
        for box in gt:
            if not 1 <= box[5] <= 10:
                continue
            if box[4]:
                truth.add_ignore_region(image_id, class_id=int(box[5]), box=box[:4].tolist())
            else:
                truth.add_ground_truth(image_id, int(box[5]), box=box[:4].tolist())
        for box in det:
            predictions.add_detection(image_id, int(box[5]), float(box[4]), box=box[:4].tolist())
    return truth, predictions


def evaluate_tide(records, cases, name):
    from tidecv import TIDE

    truth, predictions = tide_data(records, cases)
    tide = TIDE(pos_threshold=0.5, background_threshold=0.1, mode=TIDE.BOX)
    run = tide.evaluate(truth, predictions, name=name)
    errors = tide.get_all_errors()
    # TIDE intentionally retains ignored predictions in error_dict for its oracle
    # machinery. Their pred.used=None and original/fixed=(None,None) cannot
    # affect AP. Report nonignored counts and preserve library totals separately.
    counts = {
        kind.short_name: sum(getattr(error, "pred", {}).get("used", False) is not None for error in items)
        for kind, items in run.error_dict.items()
    }
    library_counts = {kind.short_name: len(items) for kind, items in run.error_dict.items()}
    return {
        "auxiliary_AP50": float(run.ap_data.get_mAP()),
        "main_oracle_dAP50": errors["main"][name],
        "special_oracle_dAP50": errors["special"][name],
        "raw_error_counts": counts,
        "library_error_counts_including_ignored_predictions": library_counts,
    }


def geometry(record):
    gt, _ = prepare_arrays(record["gt"], np.empty((0, 8)), record["shape"])
    valid = gt[(gt[:, 4] == 0) & np.isin(gt[:, 5], np.arange(1, 11))]
    area = valid[:, 2] * valid[:, 3]
    height, width = record["shape"]
    return {
        "image_id": record["id"],
        "valid_gt": len(valid),
        "small_gt": int(np.count_nonzero(area <= 32**2)),
        "medium_gt": int(np.count_nonzero((area > 32**2) & (area <= 96**2))),
        "large_gt": int(np.count_nonzero(area > 96**2)),
        "heavy_occluded_gt": int(np.count_nonzero(valid[:, 7] == 2)),
        "partial_occluded_gt": int(np.count_nonzero(valid[:, 7] == 1)),
        "truncated_gt": int(np.count_nonzero(valid[:, 6] > 0)),
        "objects_per_megapixel": float(len(valid) * 1e6 / (height * width)),
    }


def curves(run, name):
    values = read(run / "metrics.json")
    result, train_seconds, val_seconds = [], 0.0, 0.0
    for row in values:
        train_seconds += row["train_seconds"]
        val_seconds += row.get("val_seconds", 0)
        result.append(
            {
                "method": name,
                "epoch": row["epoch"],
                "train_loss": row["train_loss"],
                "val_loss": row.get("val_loss"),
                "diagnostic_AP": row.get("diagnostic_AP"),
                "cumulative_train_seconds": train_seconds,
                "cumulative_validation_seconds": val_seconds,
                "train_AP": None,
                "official_reference_AP": None,
            }
        )
    return result


def make_figures(output, methods, size_rows, class_rows, curve_rows, tide_rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    colors = plt.get_cmap("tab10").colors
    panels = ("AP", "AR500")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for axis, metric in zip(axes, panels):
        for index, name in enumerate(methods):
            selected = [r for r in size_rows if r["method"] == name and r["area"] != "all"]
            axis.plot(
                [r["area"] for r in selected],
                [r[metric] for r in selected],
                marker="o",
                label=name,
                color=colors[index],
            )
        axis.set(ylabel=f"Auxiliary COCO-style {metric} (0–100)", xlabel="Original-pixel bounding-box area group")
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.2)
    axes[1].legend(fontsize=7)
    fig.savefig(output / "size-comparison.pdf")
    fig.savefig(output / "size-comparison.svg")
    fig.savefig(output / "size-comparison.png", dpi=300)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for index, name in enumerate(methods):
        selected = [r for r in curve_rows if r["method"] == name]
        for axis, metric in zip(axes, ("train_loss", "val_loss", "diagnostic_AP")):
            present = [r for r in selected if r[metric] is not None]
            axis.plot([r["epoch"] for r in present], [r[metric] for r in present], label=name, color=colors[index])
            axis.set(xlabel="Epoch", ylabel=metric)
            axis.grid(alpha=0.2)
    axes[-1].legend(fontsize=7)
    fig.savefig(output / "training-curves.pdf")
    fig.savefig(output / "training-curves.svg")
    fig.savefig(output / "training-curves.png", dpi=300)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(11, 4), constrained_layout=True)
    values = np.array(
        [
            [
                next(r["official_AP"] for r in class_rows if r["method"] == name and r["class_name"] == cls)
                for cls in NAMES
            ]
            for name in methods
        ]
    )
    im = axis.imshow(values, vmin=0, aspect="auto", cmap="viridis")
    axis.set_xticks(range(10), NAMES, rotation=35, ha="right")
    axis.set_yticks(range(len(methods)), methods)
    fig.colorbar(im, ax=axis, label="Official-reference per-class AP (0–100)")
    fig.savefig(output / "per-class-ap.pdf")
    fig.savefig(output / "per-class-ap.svg")
    fig.savefig(output / "per-class-ap.png", dpi=300)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    categories = ("Cls", "Loc", "Both", "Dupe", "Bkg", "Miss")
    width = 0.8 / len(methods)
    for index, name in enumerate(methods):
        x = np.arange(len(categories)) - 0.4 + width * (index + 0.5)
        axes[0].bar(
            x,
            [r["dAP50"] for kind in categories for r in tide_rows if r["method"] == name and r["error"] == kind],
            width,
            label=name,
            color=colors[index],
        )
    axes[0].set_xticks(range(len(categories)), categories)
    axes[0].set(ylabel="TIDE oracle dAP50 (AP points)", xlabel="Error type")
    axes[0].legend(fontsize=7)
    # Raw counts are not dAP and are therefore presented in a separate panel.
    for index, name in enumerate(methods):
        axes[1].plot(
            categories,
            [r["count"] for kind in categories for r in tide_rows if r["method"] == name and r["error"] == kind],
            marker="o",
            label=name,
            color=colors[index],
        )
    axes[1].set(ylabel="Raw TIDE errors, score ≥ export threshold", xlabel="Error type", yscale="symlog")
    fig.savefig(output / "tide-errors.pdf")
    fig.savefig(output / "tide-errors.svg")
    fig.savefig(output / "tide-errors.png", dpi=300)
    plt.close(fig)


def selfcheck():
    """CPU contracts: ignored-region dropping, ignored-object matching and scale grouping."""
    record = {
        "id": "contract",
        "shape": [100, 100],
        "gt": np.array(
            [[10, 10, 20, 20, 1, 1, 0, 0], [50, 50, 10, 10, 0, 1, 0, 2], [70, 70, 20, 20, 0, 0, 0, 0]], dtype=float
        ),
    }
    det = np.array(
        [[10, 10, 20, 20, 0.9, 1, -1, -1], [50, 50, 10, 10, 0.8, 1, -1, -1], [72, 72, 5, 5, 0.7, 1, -1, -1]],
        dtype=float,
    )
    prepared = prepare_arrays(record["gt"], det, record["shape"])
    assert len(prepared[0]) == 2 and len(prepared[1]) == 2
    assert sorted(prepared[0][:, 4]) == [0, 1]
    coco = evaluate_coco([record], [prepared])
    assert abs(coco["sizes"][0]["AP"] - 100) < 1e-8
    assert coco["sizes"][1]["positive_gt_count"] == 1
    assert coco["sizes"][3]["AP"] is None
    tide = evaluate_tide([record], [prepared], "contract")
    assert abs(tide["auxiliary_AP50"] - 100) < 1e-8
    assert all(value == 0 for value in tide["raw_error_counts"].values())
    invalid = np.array([[10, 10, -5, 5, 0.8, 1, -1, -1]], dtype=float)
    auxiliary, info = auxiliary_cases([(prepared[0], invalid)])
    assert len(auxiliary[0][1]) == 1 and info["negative_size_predictions"] == 1
    assert auxiliary[0][1][0, 2] == 0 and auxiliary[0][1][0, 4] == 0.8
    invalid_tide = evaluate_tide([record], auxiliary, "zero-area-fp")
    invalid_coco = evaluate_coco([record], auxiliary)
    assert invalid_tide["raw_error_counts"]["Bkg"] == 1
    assert invalid_tide["auxiliary_AP50"] == 0
    assert invalid_coco["sizes"][0]["AP"] == 0
    assert "torch" not in sys.modules
    return {
        "status": "PASS",
        "contracts": [
            "ignored-region union prefilter",
            "score-zero GT ignored not negative",
            "class-specific ignored object consumes detection",
            "missing size is null, not zero",
            "negative-size detections retained as zero-area auxiliary FP candidates",
            "no torch import",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("selfcheck", "run"))
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--original-detail-job", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--archive", type=Path, default=ROOT / "datasets/VisDrone/VisDrone2019-DET-val.zip")
    args = parser.parse_args()
    if args.action == "selfcheck":
        print(json.dumps(selfcheck(), indent=2))
        return
    if not all((args.campaign, args.original_detail_job, args.output)):
        parser.error("run requires --campaign, --original-detail-job and --output")
    if args.output.exists():
        raise FileExistsError("Analysis output must be a new directory; preserve all prior analyses")
    args.output.mkdir(parents=True)
    (args.output / "status.txt").write_text("RUNNING offline full548\n")
    started = time.monotonic()
    try:
        provenance = read(args.campaign / "provenance.json")
        source = Path(provenance["source_job"])
        records, annotation_hashes = load_annotations(source, args.archive)
        runs = {
            **{
                name: args.original_detail_job / "train" / f"{name}-s0"
                for name in ("transport25", "blank25", "views25", "detail25", "lowpass25")
            },
            **{name: args.campaign / "train" / f"{name}-s0" for name in ("mix25", "online100", "vit-adapter-cache25")},
        }
        protocol = {
            "schema": "visdrone-offline-evidence-v2",
            "images": 548,
            "training_seeds": [0],
            "checkpoint_selection": "fixed final epoch100, not best",
            "scale": "AP/AR/dAP points 0-100",
            "official": {
                "reference_commit": REFERENCE_COMMIT,
                "aggregation": "image-occurrence-weighted",
                "parity": "recompute each final score and compare saved bound official-final.json",
            },
            "size_auxiliary": {
                "implementation": "pycocotools.COCOeval",
                "area_ranges_original_pixels": AREAS,
                "area_labels": AREA_NAMES,
                "IoUs": np.linspace(0.5, 0.95, 10).tolist(),
                "maxDets": [1, 10, 500],
                "global_export_cap": 500,
                "aggregation": "COCO uniform category macro and 101-point sampled precision",
                "not_official_VisDrone_metric": True,
                "boundary": "COCO inclusive bounds; exact1024/9216 can occur in adjacent groups",
            },
            "ignored_semantics": "Reference union ignored-region rasterization removes GT/dets with >=0.5 overlap. "
            "Remaining score-zero same-class GT become crowd/ignore; category11 excluded. "
            "Truncation/occlusion retained in source/counts and not grounds for removal.",
            "tide": {
                "implementation": "tidecv1.0.1 public APIs",
                "foreground_IoU": 0.5,
                "background_IoU": 0.1,
                "max_dets": 500,
                "not_official_VisDrone_metric": True,
                "dAP50": "independent error-fixing oracles, not additive",
            },
            "curves": "diagnostic_AP is internal macro validation only, never official AP; training AP unavailable",
            "resources": "CPU only; no feature/model inference or checkpoint torch deserialization",
            "negative_size_auxiliary_policy": "Clamp negative prediction width/height to zero for COCO/TIDE only; "
            "keep every box, score and class; original official input unchanged.",
            "ground_truth_profile_boundary": "Descriptive profile counts are disjoint small<=1024, "
            "1024<medium<=9216, large>9216; metric ranges use COCO inclusive bounds.",
            "input_source": str(source),
            "dataset_manifest_sha256": sha(source / "dataset-manifest.json"),
            "annotation_hashes_sha256": digest(annotation_hashes),
            "analysis_source_sha256": sha(__file__),
            "dependencies": {
                name: importlib.metadata.version(name) for name in ("numpy", "pycocotools", "tidecv", "matplotlib")
            },
            "environment_threads": {
                key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
            },
        }
        dump(args.output / "protocol.json", protocol)
        dump(args.output / "contracts.json", selfcheck())
        image_geometry = [geometry(record) for record in records]
        write_csv(args.output / "image-ground-truth-profile.csv", image_geometry)
        cohorts = {
            key: [r["image_id"] for r in sorted(image_geometry, key=lambda row: (-row[key], row["image_id"]))[:8]]
            for key in ("small_gt", "objects_per_megapixel", "heavy_occluded_gt", "truncated_gt")
        }
        dump(
            args.output / "qualitative-cohorts-gt-only.json",
            {"selection_before_prediction_inspection": True, "n_per_cohort": 8, "cohorts": cohorts},
        )
        size_rows, class_rows, curve_rows, tide_rows, summary = [], [], [], [], []
        for name, run in runs.items():
            (args.output / "status.txt").write_text(f"RUNNING full548 {name}\n")
            print(f"START {name} images=548", flush=True)
            cases, saved = load_predictions(run, records, annotation_hashes)
            official = evaluate_prepared(cases)
            for key in ("AP", "AP50", "AP75", "AR1", "AR10", "AR100", "AR500"):
                if abs(official[key] - saved["metrics"][key]) > 1e-9:
                    raise ValueError(f"Recomputed official metric changed: {name} {key}")
            auxiliary, invalid_boxes = auxiliary_cases(cases)
            sizes = evaluate_coco(records, auxiliary)
            tide = evaluate_tide(records, auxiliary, name)
            result = {
                "method": name,
                "seed": 0,
                "run": str(run),
                "images": 548,
                "official_recomputed": official,
                "coco_style_auxiliary": sizes,
                "tide_auxiliary": tide,
                "invalid_prediction_geometry": invalid_boxes,
                "bindings": {
                    "official_final_sha256": sha(run / "official-final.json"),
                    "checkpoint_sha256": saved["checkpoint_sha256"],
                    "prediction_hashes_sha256": digest(saved["prediction_sha256"]),
                    "annotation_hashes_sha256": digest(saved["annotation_sha256"]),
                    "metrics_json_sha256": sha(run / "metrics.json"),
                },
            }
            dump(args.output / f"{name}.json", result)
            size_rows.extend({"method": name, **row} for row in sizes["sizes"])
            class_rows.extend(
                {
                    "method": name,
                    **row,
                    "official_AP": official["class_AP"][row["class_id"] - 1],
                    "official_AP50": official["class_AP50"][row["class_id"] - 1],
                    "official_AP75": official["class_AP75"][row["class_id"] - 1],
                }
                for row in sizes["per_class"]
            )
            curve_rows.extend(curves(run, name))
            tide_rows.extend(
                {"method": name, "error": kind, "dAP50": value, "count": tide["raw_error_counts"][kind]}
                for kind, value in tide["main_oracle_dAP50"].items()
            )
            summary.append(
                {
                    "method": name,
                    "official_AP": official["AP"],
                    "official_AP50": official["AP50"],
                    "official_AP75": official["AP75"],
                    "official_AR500": official["AR500"],
                    **{f"aux_AP_{row['area']}": row["AP"] for row in sizes["sizes"]},
                    **{f"aux_AR500_{row['area']}": row["AR500"] for row in sizes["sizes"]},
                    "tide_auxiliary_AP50": tide["auxiliary_AP50"],
                }
            )
            print(
                f"DONE {name} official_AP={official['AP']:.6f} auxiliary_AP_small={sizes['sizes'][1]['AP']:.6f}",
                flush=True,
            )
        for filename, values in (
            ("summary.csv", summary),
            ("size-metrics-auxiliary.csv", size_rows),
            ("per-class-metrics.csv", class_rows),
            ("training-curves.csv", curve_rows),
            ("tide-errors-auxiliary.csv", tide_rows),
        ):
            write_csv(args.output / filename, values)
        make_figures(args.output, list(runs), size_rows, class_rows, curve_rows, tide_rows)
        dump(
            args.output / "completion.json",
            {
                "status": "SUCCEEDED",
                "methods": list(runs),
                "images_each": 548,
                "method_image_pairs": 548 * len(runs),
                "seconds": time.monotonic() - started,
                "torch_imported": "torch" in sys.modules,
                "protocol_sha256": sha(args.output / "protocol.json"),
            },
        )
        (args.output / "status.txt").write_text(f"SUCCEEDED methods={len(runs)} images_each=548 CPU_only\n")
    except Exception as error:
        dump(args.output / "failure.json", {"type": type(error).__name__, "message": str(error)})
        (args.output / "status.txt").write_text(f"FAILED {type(error).__name__}: {error}\n")
        raise


if __name__ == "__main__":
    main()
