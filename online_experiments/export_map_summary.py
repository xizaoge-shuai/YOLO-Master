"""Export explicit mAP names from sealed results; never evaluate a model again."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    "master-n", "yolo12-n", "frofa-c2-d01", "loffta-c2-n005",
    "cache-reuse-c1-int8", "vit-adapter-cache25", "detail25",
)


def read(path):
    return path.read_text(encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sealed_metrics(path, sources):
    completion = path.parent / "completion.json"
    seal = json.loads(read(completion))
    digest = sha(path)
    if seal["output_sha256"][path.name] != digest:
        raise ValueError("Saved metric seal differs: " + str(path))
    sources[str(path)] = digest
    sources[str(completion)] = sha(completion)
    return json.loads(read(path))


def close(a, b):
    if not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-9):
        raise ValueError(f"Saved class average is inconsistent: {a} versus {b}")


def historical_primary(run, dataset, images, sources):
    """Read old completed controls, keeping them distinct from the core queue."""
    if read(run / "DONE").strip() != "SUCCEEDED":
        raise ValueError("Historical control has no successful completion marker")
    endpoint = json.loads(read(run / "endpoint.json"))
    sources[str(run / "endpoint.json")] = sha(run / "endpoint.json")
    sources[str(run / "DONE")] = sha(run / "DONE")
    for name in ("args.json", "summary.json", "metrics.json"):
        path = run / name
        digest = sha(path)
        if endpoint["sha256"][name] != digest:
            raise ValueError("Historical control endpoint differs: " + str(path))
        sources[str(path)] = digest
    args, summary = (json.loads(read(run / name)) for name in ("args.json", "summary.json"))
    if (args["dataset"], args["seed"], args["test_images"], summary["completed_epochs"]) != (
        dataset, 0, images, 100
    ):
        raise ValueError("Historical control dataset/split/final100 differs")
    return json.loads(read(run / "metrics.json"))["primary"]


def export(output):
    rows, sources, available, core_available = [], {}, [], set()

    def row(dataset, method, protocol, images, maximum, ap, ap50, ap75, path, scope="core", **extra):
        rows.append({"dataset": dataset, "method": method, "scope": scope, "protocol": protocol,
                     "test_images": images, "max_detections": maximum,
                     "mAP50_95": ap, "mAP50": ap50, "mAP75": ap75,
                     "scale": "0-100", "source": str(path), **extra})

    base = ROOT / "runs/paper/visdrone-core-metrics/job-20261113-single-pass"
    for method in METHODS:
        path = base / method / "metrics.json"
        if not (path.parent / "completion.json").is_file():
            continue
        result = sealed_metrics(path, sources)
        if result["images"] != 548 or result["seed"] != 0:
            raise ValueError("Expected complete seed0 VisDrone evidence")
        official = result["official_primary"]
        if any(len(official[key]) != 10 for key in
               ("class_AP", "class_AP50", "class_AP75", "class_image_occurrences")):
            raise ValueError("VisDrone must retain exactly ten category entries")
        if not all(official["class_image_occurrences"]) or not all(
            math.isfinite(value) for key in ("class_AP", "class_AP50", "class_AP75")
            for value in official[key]
        ):
            raise ValueError("All ten VisDrone categories must be evaluable")
        macro = mean(official["class_AP"])
        close(macro, official["macro_AP_diagnostic"])
        row("VisDrone", method, "reference-class-macro", 548, 500, macro,
            mean(official["class_AP50"]), mean(official["class_AP75"]), path,
            reference_weighted_AP=official["AP"], reference_weighted_AP50=official["AP50"])
        for maximum in (100, 500):
            metric = result["coco_style_auxiliary"][str(maximum)]
            row("VisDrone", method, "COCO-style-auxiliary", 548, maximum,
                metric["AP"], metric["AP50"], metric["AP75"], path)
        available.append("VisDrone/" + method)
        core_available.add(("VisDrone", method))

    base = ROOT / "runs/paper/metric-bundles/job-20261112-single-pass"
    for dataset, expected in (("voc0712", 4952), ("sku110k", 2936)):
        for method in METHODS:
            path = base / dataset / method / "bundle/metrics.json"
            if not (path.parent / "completion.json").is_file():
                continue
            result = sealed_metrics(path, sources)
            if result["evaluated_images"] != expected or result["native_consistency"] != "PASS":
                raise ValueError("Expected full-test native-consistent results")
            primary = result["native_primary"]
            if dataset == "voc0712":
                classes = list(primary["per_class"].values())
                if len(classes) != 20:
                    raise ValueError("VOC must include all twenty categories")
                close(mean(x["AP50_11point"] for x in classes), primary["mAP50_11point"])
                row(dataset, method, "VOC2007-native-11point", expected, 500,
                    None, primary["mAP50_11point"], None, path)
            for maximum in (100, 500):
                metric = result["auxiliary_coco_by_max_detections"][str(maximum)]
                protocol = "COCO-native" if dataset == "sku110k" else "COCO-style-auxiliary"
                if dataset == "sku110k":
                    native = primary["by_max_detections"][str(maximum)]
                    for key in ("AP", "AP50", "AP75"):
                        close(metric[key], native[key])
                row(dataset, method, protocol, expected, maximum,
                    metric["AP"], metric["AP50"], metric["AP75"], path)
            available.append(dataset + "/" + method)
            core_available.add((dataset, method))

    historical = (
        ("voc0712", "job-20261110-voc0712-s0-v2", ("cache1", "cache2", "mix25"), 4952),
        ("sku110k", "job-20261110-sku110k-official-loader-v2", ("cache1",), 2936),
    )
    for dataset, job, methods, images in historical:
        for method in methods:
            run = ROOT / "runs/paper/crossdataset" / job / "train" / (method + "-s0")
            if not (run / "DONE").is_file():
                continue
            primary = historical_primary(run, dataset, images, sources)
            path = run / "metrics.json"
            if dataset == "voc0712":
                classes = list(primary["per_class"].values())
                if len(classes) != 20:
                    raise ValueError("Historical VOC control must cover twenty categories")
                close(mean(x["AP50_11point"] for x in classes), primary["mAP50_11point"])
                row(dataset, method, "VOC2007-native-11point", images, 500,
                    None, primary["mAP50_11point"], None, path, scope="historical-control")
                reference = ROOT / "runs/paper/metric-bundles/job-20261112-single-pass/cpu-reference-voc-cache1"
                if method == "cache1" and (reference / "completion.json").is_file():
                    auxiliary = sealed_metrics(reference / "metrics.json", sources)
                    if auxiliary["native_primary"] != primary or auxiliary["evaluated_images"] != images:
                        raise ValueError("Existing VOC auxiliary reference belongs to another control")
                    for maximum in (100, 500):
                        metric = auxiliary["auxiliary_coco_by_max_detections"][str(maximum)]
                        row(dataset, method, "COCO-style-auxiliary", images, maximum,
                            metric["AP"], metric["AP50"], metric["AP75"], reference / "metrics.json",
                            scope="historical-control")
            else:
                for maximum in (100, 500):
                    metric = primary["by_max_detections"][str(maximum)]
                    row(dataset, method, "COCO-native", images, maximum,
                        metric["AP"], metric["AP50"], metric["AP75"], path, scope="historical-control")
            available.append(dataset + "/historical-control/" + method)

    coverage = []
    priority = ROOT / "runs/paper/priority/job-20261112-single-seed"
    for dataset in ("VisDrone", "voc0712", "sku110k"):
        status = {}
        if dataset != "VisDrone":
            path = priority / dataset / "pipeline-status.json"
            status = json.loads(read(path))
            sources[str(path)] = sha(path)
        for method in METHODS:
            state = "COMPLETE" if (dataset, method) in core_available else (
                status.get("state", "PENDING") if status.get("method") == method else "PENDING"
            )
            coverage.append({"dataset": dataset, "method": method, "scope": "core",
                             "metric_state": state, "final_mAP_available": (dataset, method) in core_available})

    for path, digest in sources.items():
        if sha(Path(path)) != digest:
            raise ValueError("An input changed during export: " + path)
    output.mkdir(parents=True, exist_ok=True)
    fields = ("dataset", "method", "scope", "protocol", "test_images", "max_detections",
              "mAP50_95", "mAP50", "mAP75", "scale", "reference_weighted_AP",
              "reference_weighted_AP50", "source")
    temporary = output / "map-results.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output / "map-results.csv")
    temporary = output / "coverage.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(coverage[0]))
        writer.writeheader()
        writer.writerows(coverage)
    temporary.replace(output / "coverage.csv")
    metadata = {"available_runs": available, "rows": len(rows), "input_sha256": sources,
                "exporter_sha256": sha(Path(__file__)), "training_calls": 0,
                "prediction_calls": 0, "metric_evaluation_calls": 0,
                "historical_controls_are_not_core_queue_completions": True,
                "definition": "mAP50_95 averages category AP across IoU0.50:0.05:0.95. "
                "Reference-class-macro retains the VisDrone reference matcher/AP interpolation; "
                "COCO-style auxiliary retains COCO matching/101-point interpolation. "
                "Weighted reference scores are preserved separately, never renamed uniform mAP. "
                "VOC native eleven-point mAP50 does not supply native mAP50_95; empty means unavailable. "
                "The native VOC max_detections field records the saved global500 prediction cap, "
                "not a separate VOC maxDet sweep."}
    (output / "sources.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "available_runs": len(available), "rows": len(rows)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "research/metrics/map-summary")
    export(parser.parse_args().output.resolve())
