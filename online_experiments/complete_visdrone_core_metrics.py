"""Complete CPU-only evidence for seven already trained full548 VisDrone runs.

No image decoding, model loading, training, inference, or additional seeds.
Native occurrence-weighted endpoints stay primary; COCO/TIDE are auxiliary.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.metadata
import io
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from online_experiments import analyze_detection_evidence as old
from online_experiments import complete_metric_bundle as bundle
from online_experiments.complete_metric_bundle import (
    MAX_DETS,
    counts,
    dump,
    extract,
    read,
    sha,
    write_csv,
)
from online_experiments.visdrone_reference_eval import REFERENCE_COMMIT, evaluate_prepared, prepare_arrays

VERSION = "visdrone-core-cpu-v1"
METRICS = ("AP", "AP50", "AP75", "AR1", "AR10", "AR100", "AR500")
NATIVE = {"YOLO-Master-N": "master-n", "YOLO12-N": "yolo12-n"}


def resolve_reference(value, original):
    for prefix, root in (("repo:/", ROOT), ("user-data:/YOLO-Master/", original)):
        if value.startswith(prefix):
            return root / value.removeprefix(prefix)
    raise ValueError("Unsupported core-result source reference")


def native_cases(evaluation, method, records):
    """Legacy files lacked prediction seals: bind current files and verify AP now."""
    saved = read(evaluation)
    job = evaluation.parent.parent
    run = job / "train" / method
    checkpoint = run / "weights/last.pt"
    if sha(checkpoint) != saved["checkpoint_sha256"]:
        raise ValueError("Legacy detector final checkpoint differs from saved endpoint")
    files = {p.stem: p for p in (job / "predictions" / method / "last").glob("*.txt")}
    if set(files) != {r["id"] for r in records}:
        raise ValueError("Legacy prediction coverage must be exactly full548")
    cases, bindings = [], {str(checkpoint): sha(checkpoint)}
    for record in records:
        path = files[record["id"]]
        prediction = old.rows(path.read_bytes())
        if len(prediction) > 500 or np.any(np.diff(prediction[:, 4]) > 0):
            raise ValueError("Legacy export ordering or global500 cap differs")
        if not np.isin(prediction[:, 5], np.arange(1, 11)).all():
            raise ValueError("Legacy prediction categories differ")
        cases.append(prepare_arrays(record["gt"], prediction, record["shape"]))
        bindings[str(path)] = sha(path)
    return run, cases, saved, bindings


def one_coco(records, cases):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    ground, predictions = old.coco_data(records, cases)
    with contextlib.redirect_stdout(io.StringIO()):
        truth = COCO()
        truth.dataset = ground
        truth.createIndex()
        if predictions:
            detected = truth.loadRes(predictions)
        else:
            detected = COCO()
            detected.dataset = {**ground, "annotations": []}
            detected.createIndex()
        evaluator = COCOeval(truth, detected, "bbox")
        evaluator.params.imgIds = list(range(len(records)))
        evaluator.params.maxDets = list(MAX_DETS)
        evaluator.params.areaRng = [list(pair) for pair in old.AREAS]
        evaluator.params.areaRngLbl = list(old.AREA_NAMES)
        evaluator.evaluate()
        evaluator.accumulate()
    return evaluator, ground


def curves_and_resources(run, method, native_job=None):
    sources, curves = {}, []
    fields = ("train_seconds", "validation_seconds", "wall_seconds", "peak_vram_gib",
              "cache_bytes", "cache_build_charged_seconds", "data_prepare_charged_seconds",
              "build_plus_wall_seconds", "total_charged_seconds", "query_images", "query_calls",
              "raw_train_seconds", "raw_val_seconds", "quantization_seconds", "compressed_training_bytes")
    resources = {key: None for key in fields}
    resources.update(parameters=None, trainable_parameters=None, fresh_inference_p95_ms=None,
                     note="Saved shared-GPU observations only; missing values are unmeasured. Cache validation is not fresh-pipeline inference latency.")
    if native_job:
        config = native_job / "configs" / f"{method}-training-result.json"
        if config.is_file():
            value = read(config)
            resources["training_wall_seconds"] = value.get("training_wall_seconds")
            resources["training_protocol"] = value.get("protocol")
            sources[str(config)] = sha(config)
        source = native_job / "analysis" / f"{method}-epochs-deduplicated.csv"
        original = run / "results.csv"
        if original.is_file():
            sources[str(original)] = sha(original)
            with original.open(newline="", encoding="utf-8-sig") as stream:
                history = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(stream)]
            epochs = [int(float(row["epoch"])) for row in history]
            resources["legacy_history_audit"] = {
                "raw_csv_rows": len(history), "unique_epochs": len(set(epochs)),
                "duplicate_epoch_rows": len(history) - len(set(epochs)),
                "minimum_epoch": min(epochs, default=None), "maximum_epoch": max(epochs, default=None),
                "note": "CSV duplication is historical evidence, not proof of repeated optimizer steps. This legacy run is not certified under the new VOC/SKU strict100-unique-epoch contract.",
            }
        if not source.is_file():
            source = original
        if source.is_file():
            sources[str(source)] = sha(source)
            with source.open(newline="", encoding="utf-8-sig") as stream:
                rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(stream)]
            curves = [{k: v for k, v in r.items() if k in ("epoch", "time") or k.startswith(("train/", "lr/"))}
                      for r in rows]
        resources["curve_scope"] = "Existing deduplicated legacy epoch table when available; original raw history is also hash-bound. Epoch/time records do not establish equal optimizer steps or controlled speed."
    else:
        summary_file = run / "summary.json"
        summary = read(summary_file)
        sources[str(summary_file)] = sha(summary_file)
        resources.update({key: summary.get(key) for key in fields})
        metric_file = run / "metrics.json"
        rows = read(metric_file)
        sources[str(metric_file)] = sha(metric_file)
        if not isinstance(rows, list):
            raise ValueError("Expected original per-epoch metric list")
        keys = ("epoch", "train_loss", "train_seconds", "query_images", "query_calls", "lr", "peak_vram_gib")
        curves = [{k: row.get(k) for k in keys} for row in rows]
        resources["curve_scope"] = "Saved training losses, times and learning rates only; no inferred training AP or placeholder validation zeros."
    return curves, resources, sources


def verify_completion(folder):
    seal = read(folder / "completion.json")
    for group in ("input_sha256", "output_sha256"):
        for path, expected in seal[group].items():
            actual = Path(path) if group == "input_sha256" else folder / path
            if not actual.is_file() or sha(actual) != expected:
                raise ValueError(f"Completed evidence changed: {actual}")
    return seal


def score_case(args, row, records, annotation_hashes):
    evaluation = resolve_reference(row["evaluation_source"], args.original)
    method = NATIVE.get(row["method"], evaluation.parent.name.removesuffix("-s0"))
    output = args.output / method
    if (output / "completion.json").is_file():
        verify_completion(output)
        return method, output
    if output.exists():
        raise FileExistsError(f"Unsealed prior output preserved, refusing retry: {output}")
    output.mkdir()
    try:
        inputs = {str(evaluation): sha(evaluation), str(args.core_csv): sha(args.core_csv),
                  str(args.source / "dataset-manifest.json"): sha(args.source / "dataset-manifest.json"),
                  str(Path(__file__).resolve()): sha(__file__),
                  str(Path(old.__file__)): sha(old.__file__),
                  str(Path(bundle.__file__)): sha(bundle.__file__),
                  str(ROOT / "online_experiments/visdrone_reference_eval.py"):
                  sha(ROOT / "online_experiments/visdrone_reference_eval.py")}
        if sha(evaluation) != row["source_sha256"]:
            raise ValueError("Core CSV source endpoint hash differs")
        if row["method"] in NATIVE:
            run, cases, saved, bound = native_cases(evaluation, method, records)
            expected = saved
            seal_note = "Legacy native endpoint had no prediction hash map. Current full548 predictions are newly hash-bound here; all seven official aggregate metrics are recomputed against the saved historical endpoint. This does not claim a historical prediction seal."
        else:
            run = evaluation.parent
            cases, saved = old.load_predictions(run, records, annotation_hashes)
            expected = saved["metrics"]
            bound = {str(run / "weights/last.pt"): sha(run / "weights/last.pt")}
            bound.update({str(run / "predictions/last" / (key + ".txt")): value
                          for key, value in saved["prediction_sha256"].items()})
            seal_note = "Original cached run checkpoint, raw annotations and every saved prediction hash verified."
        inputs.update(bound)
        if saved["reference_commit"] != REFERENCE_COMMIT or len(records) != 548 or int(row["seed"]) != 0:
            raise ValueError("Native reference, complete split or seed identity differs")
        official = evaluate_prepared(cases)
        for key in METRICS:
            if abs(official[key] - expected[key]) > 1e-9:
                raise ValueError(f"Native official endpoint differs: {method} {key}")
        auxiliary, invalid = old.auxiliary_cases(cases)
        evaluator, ground = one_coco(records, auxiliary)
        metrics, classes, areas, pr = extract(evaluator, old.NAMES)
        for item in classes:
            ki = item["category_id_zero_based"]
            item.update(official_AP=official["class_AP"][ki], official_AP50=official["class_AP50"][ki],
                        official_AP75=official["class_AP75"][ki])
        reused = args.previous / f"{method}.json"
        if method in ("vit-adapter-cache25", "detail25") and reused.is_file():
            previous = read(reused)
            if previous["bindings"]["official_final_sha256"] != sha(evaluation):
                raise ValueError("Prior TIDE evidence is bound to a different endpoint")
            tide = previous["tide_auxiliary"]
            inputs[str(reused)] = sha(reused)
            tide_origin = "Reused verified completed full548 TIDE evidence; no repeated TIDE scoring."
        else:
            tide = old.evaluate_tide(records, auxiliary, method)
            tide_origin = "Computed on CPU from the same saved final predictions."
        curves, resources, resource_sources = curves_and_resources(
            run, method, evaluation.parent.parent if row["method"] in NATIVE else None)
        inputs.update(resource_sources)
        if row["method"] in NATIVE:
            resources["observed_mean_predict_wall_ms"] = saved.get("mean_predict_wall_ms")
            resources["prediction_time_scope"] = saved.get("note")
        dump(output / "metrics.json", {"method": method, "seed": 0, "images": 548,
             "official_primary": official, "coco_style_auxiliary": metrics,
             "tide_auxiliary": tide, "tide_origin": tide_origin, "invalid_prediction_geometry": invalid})
        write_csv(output / "per-class.csv", classes)
        write_csv(output / "per-area.csv", areas)
        write_csv(output / "pr-curves.csv", pr)
        write_csv(output / "object-counts.csv", counts(ground, old.NAMES, evaluator))
        if curves:
            write_csv(output / "training-curves.csv", curves)
        dump(output / "resource-metrics.json", resources)
        np.savez_compressed(output / "tensors.npz", precision=evaluator.eval["precision"],
                            recall=evaluator.eval["recall"], scores=evaluator.eval["scores"],
                            iou_thresholds=evaluator.params.iouThrs, recall_thresholds=evaluator.params.recThrs,
                            max_detections=evaluator.params.maxDets, category_ids=evaluator.params.catIds,
                            area_ranges=evaluator.params.areaRng)
        dump(output / "provenance.json", {"version": VERSION, "run": str(run), "seal_note": seal_note,
             "annotations_sha256": annotation_hashes, "reference_commit": REFERENCE_COMMIT,
             "original_manifest_sha256": sha(args.source / "dataset-manifest.json"),
             "auxiliary_protocol": "COCO macro AP at inclusive original-pixel areas; maxDets [1,10,100,500] with original GLOBAL500 export cap and VisDrone ignored-region prefilter. AP/AR scale0-100; PR precision/recall scale0-1. These are auxiliary, not official VisDrone metrics.",
             "pr_scope": "101-point interpolated precision and scores, not a confidence-sweep F1 curve.",
             "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "pycocotools", "tidecv")}})
        (output / "README.md").write_text(
            f"# {method}: full548, seed0, fixed final checkpoint\n\n"
            "CPU evidence from existing saved predictions. No training, image decoding, inference or new seeds. "
            "Official VisDrone occurrence-weighted AP remains primary. COCO area/class/PR and TIDE metrics are auxiliary. "
            "Single-seed results do not provide seed uncertainty. Shared-GPU time is observational. Missing resource values remain null.\n\n"
            + seal_note + "\n", encoding="utf-8")
        if any(sha(Path(path)) != expected for path, expected in inputs.items()):
            raise RuntimeError("Input changed during CPU scoring; refusing a completion seal")
        dump(output / "completion.json", {"version": VERSION, "state": "SUCCEEDED",
             "input_sha256": inputs, "output_sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
             "time_unix": time.time()})
        print(f"DONE {method}: official_AP={official['AP']:.6f}", flush=True)
        return method, output
    except Exception:
        dump(output / "failure.json", {"state": "FAILED", "traceback": traceback.format_exc(), "time_unix": time.time()})
        raise


def publish_case(args, method, output):
    from online_experiments.archive_assignment_intervention import commit_files
    from online_experiments.priority_single_seed_campaign import archive_lock
    from online_experiments.replication_campaign import push

    receipt = args.output / f"archive-{method}.json"
    with archive_lock():
        if receipt.is_file():
            result = read(receipt)
        else:
            seal = verify_completion(output)
            destination = ROOT / "research/experiments/visdrone-core-metrics-20261113" / method
            destination.mkdir(parents=True, exist_ok=True)
            paths = []
            for name in sorted(set(seal["output_sha256"]) | {"completion.json"}):
                source, target = output / name, destination / name
                if source.suffix not in {".json", ".csv", ".npz", ".md"} or source.stat().st_size > 50 * 1024**2:
                    raise ValueError("Only lightweight derived evidence may be archived")
                if target.exists() and sha(target) != sha(source):
                    raise ValueError("Refuse changed existing archive")
                if not target.exists():
                    shutil.copyfile(source, target)
                paths.append(target.relative_to(ROOT).as_posix())
            marker = f"visdrone-core-metrics-20261113:{method}"
            readme = (destination / "README.md").relative_to(ROOT).as_posix()
            block = f"\n\n<!-- {marker} -->\n## VisDrone core metrics: {method}\n\nFull548 saved final predictions scored on CPU. No retraining or inference. [Evidence]({readme.removeprefix('research/')}).\n"
            result = {"commit": commit_files(ROOT, paths, block, marker, readme,
                      f"research: complete saved VisDrone {method} metrics"), "pushed": False}
            dump(receipt, result)
        if not result.get("pushed"):
            try:
                result.update(pushed=True, pushed_head=push())
            except Exception as exc:  # noqa: BLE001 -- preserve publication failures without changing measurements
                result.update(pushed=False, push_error=str(exc))
            dump(receipt, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--original", type=Path, default=ROOT.parent / "YOLO-Master")
    parser.add_argument("--core-csv", type=Path, default=ROOT / "research/checks/minimal-priority-20261112/reused-visdrone-core-results.csv")
    parser.add_argument("--previous", type=Path, default=ROOT / "research/experiments/offline-evidence-job-20261110-004004-cpu548-v2")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if not args.output.is_relative_to((ROOT / "runs").resolve()):
        raise ValueError("Metric campaign output must remain within this repository's runs directory")
    args.output.mkdir(parents=True, exist_ok=True)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES='' for this CPU-only campaign")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise RuntimeError(f"Set {key}=1 before this CPU-only campaign")
    import fcntl

    with (args.output / ".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        dump(args.output / "worker.json", {"pid": os.getpid(), "started_unix": time.time(),
             "version": VERSION, "command": sys.argv})
        dump(args.output / "status.json", {"state": "RUNNING_CPU", "current_method": "loading annotations",
             "completed": 0, "failed": 0, "planned": 7, "time_unix": time.time()})
        records, annotation_hashes = old.load_annotations(args.source, args.archive)
        with args.core_csv.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 7 or len({r["method"] for r in rows}) != 7:
            raise ValueError("Expected exactly seven frozen core methods")
        states = []
        for row in rows:
            dump(args.output / "status.json", {"state": "RUNNING_CPU", "current_method": row["method"],
                 "completed": sum(s["state"] == "SUCCEEDED" for s in states),
                 "failed": sum(s["state"] == "FAILED" for s in states), "planned": 7, "time_unix": time.time()})
            try:
                method, output = score_case(args, row, records, annotation_hashes)
                state = {"method": method, "state": "SUCCEEDED", "output": str(output)}
                if args.publish:
                    state["archive"] = publish_case(args, method, output)
            except Exception:  # noqa: BLE001 -- preserve each failed case and continue other authorized cases
                state = {"method": row["method"], "state": "FAILED", "traceback": traceback.format_exc()}
                print(state["traceback"], flush=True)
            states.append(state)
            dump(args.output / "cases.json", states)
            dump(args.output / "status.json", {"state": "RUNNING_CPU", "current_method": None,
                 "completed": sum(s["state"] == "SUCCEEDED" for s in states),
                 "failed": sum(s["state"] == "FAILED" for s in states), "planned": 7, "time_unix": time.time()})
        failed = sum(s["state"] == "FAILED" for s in states)
        dump(args.output / "status.json", {"state": "FAILED" if failed else "SUCCEEDED",
             "completed": len(states) - failed, "failed": failed, "planned": 7, "time_unix": time.time()})
        if failed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
