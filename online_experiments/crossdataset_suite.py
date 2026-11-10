"""Full VOC0712/SKU110K cached-detector adaptations, with native final endpoints.

This is a new protocol version. Existing VisDrone sources/results are never
modified. No subset, smoke, best-checkpoint, or epoch-tuning mode is provided.
Large feature banks are read through mmap, one selected batch at a time.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VERSION = "crossdataset-detection-v1"
SHAPE = (3, 384, 40, 40)
COUNTS = {"voc0712": {"train": 16551, "test": 4952}, "sku110k": {"train": 8219, "val": 588, "test": 2936}}
METHODS = (
    "cache1",
    "cache2",
    "mix25",
    "online100",
    "frofa-c2-d01",
    "loffta-c2-n005",
    "vit-adapter-cache25",
    "vit-adapter-cache0",
    "cache-reuse-c1-int8",
    "cache-reuse-c1-int4",
    "transport25",
    "blank25",
    "views25",
    "detail25",
    "lowpass25",
    "blank0",
    "detail0",
    "detail10",
)
EXTRA_VIEWS = {"s085-v0": (0.85, False), "s085-v1": (0.85, True), "s115-v0": (1.15, False), "s115-v1": (1.15, True)}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stamp(path):
    stat = Path(path).stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns}


def open_partial_array(path, dtype, shape, offset):
    """A saved positive offset must never create a missing prefix of zeros."""
    path = Path(path)
    if not 0 <= offset <= shape[0]:
        raise RuntimeError("Partial cache offset outside its full shape")
    if offset > 0 and not path.is_file():
        raise RuntimeError("Partial cache prefix is missing; refuse silent zero reconstruction")
    if path.exists():
        array = np.load(path, mmap_mode="r+", allow_pickle=False)
        if array.shape != tuple(shape) or array.dtype != np.dtype(dtype):
            raise RuntimeError("Partial cache dtype/shape mismatch")
        return array
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=tuple(shape))


def resolve_input(root, relative):
    root = Path(root).resolve()
    target = (root / relative).resolve(strict=True)
    if root not in target.parents:
        raise ValueError(f"Dataset input escapes its root: {relative}")
    return target


def validate_manifest(meta):
    dataset = meta.get("dataset")
    if meta.get("schema") != "detection-dataset-v1" or dataset not in COUNTS:
        raise ValueError("Expected a complete detection-dataset-v1 VOC0712/SKU110K manifest")
    expected = COUNTS[dataset]
    if {key: len(value) for key, value in meta["splits"].items()} != expected:
        raise ValueError(f"Expected official full split counts {expected}")
    names = meta["class_names"]
    nc = 20 if dataset == "voc0712" else 1
    if meta.get("nc") != nc or len(names) != nc or len(set(names)) != nc:
        raise ValueError("Dataset class count or ordered names mismatch")
    seen = set()
    image_hashes = {}
    for split, samples in meta["splits"].items():
        for sample in samples:
            identity = sample["sample_id"]
            if identity in seen:
                raise ValueError(f"Duplicate sample ID across official splits: {identity}")
            digest = sample["image_sha256"]
            if digest in image_hashes and image_hashes[digest] != split:
                raise ValueError(f"Identical image across official splits: {identity}")
            seen.add(identity)
            image_hashes[digest] = split
            if sample["width"] < 1 or sample["height"] < 1:
                raise ValueError(f"Invalid image dimensions: {identity}")
            for obj in sample["objects"]:
                x0, y0, x1, y1 = obj["bbox_xyxy0_half_open"]
                if not all(math.isfinite(value) for value in (x0, y0, x1, y1)):
                    raise ValueError(f"Nonfinite annotation: {identity}")
                if not (0 <= x0 < x1 <= sample["width"] and 0 <= y0 < y1 <= sample["height"]):
                    raise ValueError(f"Invalid native annotation bounds: {identity}")
                if not 0 <= obj["category_id"] < nc:
                    raise ValueError(f"Annotation category outside dataset classes: {identity}")
                if split == "test" and dataset == "voc0712" and "bbox_voc_1based_inclusive" not in obj:
                    raise ValueError("VOC test objects must retain official one-based inclusive coordinates")
    return meta


def within_split_duplicate_images(meta):
    """Record official repeated JPEGs without dropping or merging annotated samples."""
    result = []
    for split, samples in meta["splits"].items():
        groups = {}
        for sample in samples:
            groups.setdefault(sample["image_sha256"], []).append(sample["sample_id"])
        result.extend(
            {"split": split, "image_sha256": digest, "sample_ids": identities}
            for digest, identities in groups.items()
            if len(identities) > 1
        )
    return result


def prepare_targets(sample):
    width, height = sample["width"], sample["height"]
    scale = min(640 / height, 640 / width)
    rw, rh = max(1, round(width * scale)), max(1, round(height * scale))
    left, top = (640 - rw) // 2, (640 - rh) // 2
    info = {"original_shape": [height, width], "resized_shape": [rh, rw], "scale": scale, "pad": [left, top]}
    classes, boxes = [], []
    for obj in sample["objects"]:
        if obj.get("difficult", False) or obj.get("iscrowd", 0):
            continue
        x0, y0, x1, y1 = obj["bbox_xyxy0_half_open"]
        classes.append([obj["category_id"]])
        boxes.append(
            [
                ((x0 + x1) * rw / (2 * width) + left) / 640,
                ((y0 + y1) * rh / (2 * height) + top) / 640,
                (x1 - x0) * rw / (width * 640),
                (y1 - y0) * rh / (height * 640),
            ]
        )
    return classes, boxes, info


def inverse_predictions(predictions, info):
    result = np.asarray(predictions, dtype=np.float64).reshape(-1, 6).copy()
    height, width = info["original_shape"]
    rh, rw = info["resized_shape"]
    left, top = info["pad"]
    result[:, [0, 2]] = np.clip((result[:, [0, 2]] - left) * width / rw, 0, width)
    result[:, [1, 3]] = np.clip((result[:, [1, 3]] - top) * height / rh, 0, height)
    return result


def evaluate_voc07(samples, predictions, class_names):
    """Official VOC2007 definition: inclusive pixels, strict IoU>0.5, 11-point AP.

    Predictions are stored zero-based half-open and converted to the official
    one-based inclusive box convention before comparison. Difficult matches are
    ignored, duplicate matches are false positives, classes have equal weight.
    """
    per_class = {}
    sample_map = {sample["sample_id"]: sample for sample in samples}
    for detection in predictions:
        if detection["sample_id"] not in sample_map or not 0 <= detection["category_id"] < len(class_names):
            raise ValueError("Prediction refers to unknown image or category")
    for category, name in enumerate(class_names):
        ground, npos = {}, 0
        for sample in samples:
            objects = [obj for obj in sample["objects"] if obj["category_id"] == category]
            boxes = [
                obj.get(
                    "bbox_voc_1based_inclusive",
                    [
                        obj["bbox_xyxy0_half_open"][0] + 1,
                        obj["bbox_xyxy0_half_open"][1] + 1,
                        *obj["bbox_xyxy0_half_open"][2:],
                    ],
                )
                for obj in objects
            ]
            difficult = np.asarray([obj.get("difficult", False) for obj in objects], dtype=bool)
            ground[sample["sample_id"]] = (
                np.asarray(boxes, dtype=np.float64).reshape(-1, 4),
                difficult,
                np.zeros(len(objects), dtype=bool),
            )
            npos += int((~difficult).sum())
        selected = sorted(
            (item for item in predictions if item["category_id"] == category), key=lambda item: -item["score"]
        )
        tp, fp = np.zeros(len(selected)), np.zeros(len(selected))
        ignored = 0
        for index, item in enumerate(selected):
            box = np.asarray(item["bbox_xyxy0_half_open"], dtype=np.float64).copy()
            box[:2] += 1
            gt, difficult, used = ground[item["sample_id"]]
            match, overlap = -1, -math.inf
            if len(gt):
                low, high = np.maximum(gt[:, :2], box[:2]), np.minimum(gt[:, 2:], box[2:])
                intersection = np.maximum(high - low + 1, 0).prod(axis=1)
                union = (gt[:, 2:] - gt[:, :2] + 1).prod(axis=1) + (box[2:] - box[:2] + 1).prod() - intersection
                iou = intersection / np.maximum(union, np.finfo(np.float64).eps)
                match = int(iou.argmax())
                overlap = float(iou[match])
            if overlap > 0.5:
                if difficult[match]:
                    ignored += 1
                elif not used[match]:
                    tp[index], used[match] = 1, True
                else:
                    fp[index] = 1
            else:
                fp[index] = 1
        cumulative_tp, cumulative_fp = np.cumsum(tp), np.cumsum(fp)
        recall = cumulative_tp / max(npos, 1)
        precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, np.finfo(np.float64).eps)
        ap = sum(float(precision[recall >= threshold].max(initial=0)) / 11 for threshold in np.linspace(0, 1, 11))
        per_class[name] = {
            "AP50_11point": 100 * ap if npos else None,
            "non_difficult_objects": npos,
            "detections": len(selected),
            "ignored_detections": ignored,
        }
    aps = [item["AP50_11point"] for item in per_class.values() if item["AP50_11point"] is not None]
    return {
        "mAP50_11point": float(np.mean(aps)) if aps else 0.0,
        "per_class": per_class,
        "definition": "VOC2007 eleven-point AP; IoU strictly >0.5; difficult ignored; inclusive official boxes",
        "scale": "0-100",
        "evaluated_images": len(samples),
    }


def coco_payload(samples, predictions, class_names):
    image_ids = {sample["sample_id"]: index + 1 for index, sample in enumerate(samples)}
    ground = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": index + 1, "name": name} for index, name in enumerate(class_names)],
    }
    for sample in samples:
        image_id = image_ids[sample["sample_id"]]
        ground["images"].append(
            {"id": image_id, "file_name": sample["sample_id"], "width": sample["width"], "height": sample["height"]}
        )
        for obj in sample["objects"]:
            x0, y0, x1, y1 = obj["bbox_xyxy0_half_open"]
            ground["annotations"].append(
                {
                    "id": len(ground["annotations"]) + 1,
                    "image_id": image_id,
                    "category_id": obj["category_id"] + 1,
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "area": (x1 - x0) * (y1 - y0),
                    "iscrowd": int(obj.get("iscrowd", 0) or obj.get("difficult", False)),
                }
            )
    detections = []
    for item in predictions:
        if item["sample_id"] not in image_ids or not 0 <= item["category_id"] < len(class_names):
            raise ValueError("Prediction refers to unknown image or category")
        x0, y0, x1, y1 = item["bbox_xyxy0_half_open"]
        detections.append(
            {
                "image_id": image_ids[item["sample_id"]],
                "category_id": item["category_id"] + 1,
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "score": item["score"],
            }
        )
    return ground, detections


def evaluate_coco(samples, predictions, names, output):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    ground, detections = coco_payload(samples, predictions, names)
    dump(output / "coco-ground-truth.json", ground)
    dump(output / "coco-predictions.json", detections)
    gt = COCO()
    gt.dataset = ground
    gt.createIndex()
    if detections:
        dt = gt.loadRes(detections)
    else:
        dt = COCO()
        dt.dataset = {**ground, "annotations": []}
        dt.createIndex()
    result = {
        "implementation": "pycocotools.COCOeval bbox",
        "version": importlib.metadata.version("pycocotools"),
        "scale": "0-100",
        "evaluated_images": len(samples),
        "by_max_detections": {},
    }
    for maximum in (100, 500):
        evaluator = COCOeval(gt, dt, "bbox")
        evaluator.params.maxDets = [1, 10, maximum]
        evaluator.evaluate()
        evaluator.accumulate()
        with (output / f"coco-maxdet{maximum}.txt").open("w", encoding="utf-8") as stream, contextlib.redirect_stdout(
            stream
        ):
            evaluator.summarize()
        # COCOeval.stats[0] is hard-coded to maxDets=100, even after maxDets is
        # changed. Extract the complete precision tensor at the requested index.
        precision, recall = evaluator.eval["precision"], evaluator.eval["recall"]

        def average(array):
            values = array[array > -1]
            return 100 * float(values.mean()) if len(values) else None

        metrics = {
            "AP": average(precision[:, :, :, 0, 2]),
            "AP50": average(precision[0, :, :, 0, 2]),
            "AP75": average(precision[5, :, :, 0, 2]),
            "AP_small": average(precision[:, :, :, 1, 2]),
            "AP_medium": average(precision[:, :, :, 2, 2]),
            "AP_large": average(precision[:, :, :, 3, 2]),
            "AR": average(recall[:, :, 0, 2]),
        }
        result["by_max_detections"][str(maximum)] = metrics
    return result


def consumed_parts(method):
    keys = ["test-v0"]
    if method != "online100":
        keys += ["train-v0"] if method == "cache1" or method.startswith("cache-reuse-") else ["train-v0", "train-v1"]
    if method in (
        "blank25",
        "views25",
        "detail25",
        "lowpass25",
        "blank0",
        "detail0",
        "detail10",
        "vit-adapter-cache25",
        "vit-adapter-cache0",
    ):
        keys.append("blank")
    if method == "views25":
        keys += list(EXTRA_VIEWS)
    return keys


def quantize_array(values, bits):
    if bits not in (4, 8):
        raise ValueError("Physical quantizer bits must be 4 or 8")
    values = np.asarray(values, dtype=np.float32)
    low, high = np.quantile(values, (0.001, 0.999), axis=(-2, -1), method="linear").astype(np.float32)
    step = (high - low) / (2**bits - 1)
    codes = np.rint((values - low[..., None, None]) / np.maximum(step[..., None, None], 1e-12))
    codes = np.clip(codes, 0, 2**bits - 1).astype(np.uint8)
    if bits == 4:
        packed = codes[..., ::2].copy()
        packed[..., : codes.shape[-1] // 2] |= codes[..., 1::2] << np.uint8(4)
        codes = packed
    return codes, low, step


def dequantize_array(codes, low, step, bits, width):
    if bits == 4:
        result = np.empty((*codes.shape[:-1], width), dtype=np.float32)
        result[..., ::2] = codes & np.uint8(15)
        result[..., 1::2] = codes[..., : width // 2] >> np.uint8(4)
    else:
        result = codes.astype(np.float32)
    return result * step[..., None, None] + low[..., None, None]


def source_snapshot():
    paths = subprocess.check_output(
        ["git", "ls-files", "*.py", "*.sh", "*.yaml", "*.toml"], cwd=ROOT, text=True
    ).splitlines()
    paths += [Path(__file__).relative_to(ROOT).as_posix()]
    return {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {name: sha(ROOT / name) for name in sorted(set(paths)) if (ROOT / name).is_file()},
    }


def verify_sources(provenance):
    changed = [name for name, digest in provenance["source_sha256"].items() if sha(ROOT / name) != digest]
    if changed:
        raise RuntimeError(f"Training source changed since this protocol snapshot: {changed[:10]}")


def load_dataset(path, *, verify_files=False):
    path = Path(path).resolve(strict=True)
    meta = validate_manifest(read(path))
    root = path.parent
    seal_path = root / "complete.json"
    if not seal_path.is_file() or read(seal_path).get("manifest_sha256") != sha(path):
        raise RuntimeError("Dataset preparation seal is missing or its full manifest hash changed")
    seal = read(seal_path)
    if seal.get("evaluation_sha256") and sha(resolve_input(root, meta["evaluation_file"])) != seal["evaluation_sha256"]:
        raise RuntimeError("Prepared held-out evaluation metadata hash changed")
    signature = {}
    for samples in meta["splits"].values():
        for sample in samples:
            sample["image_path"] = str(resolve_input(root, sample["image"]))
            classes, boxes, info = prepare_targets(sample)
            sample.update(cls=classes, bboxes=boxes, letterbox=info)
            if verify_files:
                for kind in ("image", "label"):
                    target = resolve_input(root, sample[kind])
                    if sha(target) != sample[f"{kind}_sha256"]:
                        raise RuntimeError(f"Dataset {kind} hash mismatch: {sample['sample_id']}")
                    signature[str(target)] = {"sha256": sample[f"{kind}_sha256"], **stamp(target)}
    return meta, signature


def snapshot(args):
    started = time.perf_counter()
    manifest = Path(args.manifest).resolve(strict=True)
    identity = read(args.identity)
    identity = identity.get("identity", identity)
    if identity["imgsz"] != 640 or identity["layers"] != [3, 7, 11] or not identity.get("revision"):
        raise ValueError("Teacher identity must pin revision, 640px, and layers3/7/11")
    provenance_path = args.job / "provenance.json"
    if provenance_path.exists():
        previous = read(provenance_path)
        if previous["manifest_sha256"] != sha(manifest) or previous["identity"] != identity:
            raise RuntimeError("Existing dataset/teacher snapshot changed")
        verify_provenance(args.job)
        return previous
    meta, integrity = load_dataset(manifest, verify_files=True)
    args.job.mkdir(parents=True, exist_ok=True)
    dump(args.job / "input-integrity.json", integrity)
    provenance = {
        "version": VERSION,
        "manifest": str(manifest),
        "manifest_sha256": sha(manifest),
        "dataset": meta["dataset"],
        "counts": COUNTS[meta["dataset"]],
        "within_split_duplicate_images": within_split_duplicate_images(meta),
        "duplicate_image_policy": "Keep official within-split samples and annotations; reject cross-split identical JPEGs and any repeated sample ID.",
        "class_names": meta["class_names"],
        "identity": identity,
        "identity_file": str(Path(args.identity).resolve()),
        "identity_file_sha256": sha(args.identity),
        "input_integrity_sha256": sha(args.job / "input-integrity.json"),
        "snapshot_prepare_seconds": time.perf_counter() - started,
        "protocol": {
            "epochs": 100,
            "batch": 8,
            "imgsz": 640,
            "optimizer": "AdamW",
            "lr": 0.001,
            "weight_decay": 0.0005,
            "cosine_eta_min": 0.00001,
            "flip_probability": 0.5,
            "scale_range": [0.85, 1.15],
            "head_fusion": "mean_value",
            "max_det": 500,
            "confidence": 0.001,
            "endpoint": "fixed final epoch100 on full held-out test; no intermediate test/model selection",
            "baseline_scope": "detection adaptations of paper ideas, not original paper reproductions",
            "shared_gpu_timing": "observed costs only; no controlled speed claim",
            "cost_boundary": "Cold includes snapshot verification, consumed feature-bank build/compression, and warm run; dataset download/extraction excluded.",
        },
        **source_snapshot(),
    }
    dump(provenance_path, provenance)
    return provenance


def verify_provenance(job):
    provenance = read(job / "provenance.json")
    if provenance["version"] != VERSION or sha(provenance["manifest"]) != provenance["manifest_sha256"]:
        raise RuntimeError("Dataset/protocol provenance mismatch")
    if sha(job / "input-integrity.json") != provenance["input_integrity_sha256"]:
        raise RuntimeError("Input integrity record changed")
    if sha(provenance["identity_file"]) != provenance["identity_file_sha256"]:
        raise RuntimeError("Teacher identity source changed")
    verify_sources(provenance)
    for target, record in read(job / "input-integrity.json").items():
        if stamp(target) != {key: record[key] for key in ("bytes", "mtime_ns", "ctime_ns")}:
            raise RuntimeError(f"Dataset input changed after full SHA256 verification: {target}")
    return provenance


def append_jsonl(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def counted_extract(extractor, images, journal, **context):
    import torch

    attempt = {"attempt_id": f"{os.getpid()}-{time.time_ns()}", "images": len(images), **context}
    append_jsonl(journal, {**attempt, "event": "attempt"})
    result = extractor(images)
    torch.cuda.synchronize()
    append_jsonl(journal, {**attempt, "event": "completed"})
    return result


def query_journal(path):
    attempted, completed = {}, set()
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if record["event"] == "attempt":
                    attempted[record["attempt_id"]] = record["images"]
                else:
                    completed.add(record["attempt_id"])
    return {
        "attempted_images": sum(attempted.values()),
        "completed_images": sum(attempted[key] for key in completed),
        "attempted_calls": len(attempted),
        "completed_calls": len(completed),
        "uncertain_images": sum(count for key, count in attempted.items() if key not in completed),
        "note": "Durable teacher-call journal includes replayed work on resume; incomplete attempts are uncertain, not zero.",
    }


def cache_manifest(job, *, full_hash=False):
    path = job / "cache/manifest.json"
    meta = read(path)
    if read(path.parent / "complete.json")["manifest_sha256"] != sha(path):
        raise RuntimeError("Cache completion seal mismatch")
    if meta["provenance_sha256"] != sha(job / "provenance.json"):
        raise RuntimeError("Cache belongs to another dataset/teacher/source snapshot")
    if meta["teacher_query_journal_sha256"] != sha(path.parent / "teacher-queries.jsonl"):
        raise RuntimeError("Cache construction teacher-call journal changed")
    provenance = read(job / "provenance.json")
    allowed = {"train-v0", "train-v1", "test-v0", "blank"} | set(EXTRA_VIEWS)
    if set(meta["parts"]) != set(meta["request"]["parts"]) or not {"train-v0", "train-v1", "test-v0", "blank"} <= set(
        meta["parts"]
    ):
        raise RuntimeError("Cache part coverage does not match its fixed build request")
    if not set(meta["parts"]) <= allowed:
        raise RuntimeError("Unknown feature cache view")
    for key, part in meta["parts"].items():
        if stamp(part["file"]) != part["stat"] or (full_hash and sha(part["file"]) != part["sha256"]):
            raise RuntimeError("Feature cache file integrity mismatch")
        count = 1 if key == "blank" else provenance["counts"]["test" if key.startswith("test-") else "train"]
        values = np.load(part["file"], mmap_mode="r", allow_pickle=False)
        if values.shape != (count, *SHAPE) or values.dtype != np.float16 or part["images"] != count:
            raise RuntimeError("Feature cache coverage/shape/dtype mismatch")
        del values
    return meta


def build_cache(args):
    import torch

    from online_experiments.d1_online_compare import geometry
    from online_experiments.full_visdrone_data import make_extractor
    from ultralytics.data.foundation_cache import load_letterboxed_tensor

    provenance = verify_provenance(args.job)
    meta, _ = load_dataset(provenance["manifest"])
    cache = args.job / "cache"
    cache.mkdir(exist_ok=True)
    keys = ["train-v0", "train-v1", "test-v0", "blank"] + (list(EXTRA_VIEWS) if args.views else [])
    if (cache / "manifest.json").exists():
        result = cache_manifest(args.job, full_hash=True)
        if result["request"] != {"parts": keys, "batch": args.build_batch}:
            raise RuntimeError("Existing cache build request changed")
        return result
    torch.set_num_threads(4)
    shape_bytes = math.prod(SHAPE) * 2
    estimate = sum(
        (1 if key == "blank" else len(meta["splits"]["test" if key.startswith("test-") else "train"])) * shape_bytes
        for key in keys
    )
    retained = sum(path.stat().st_size for path in cache.glob("*.npy"))
    remaining = max(0, estimate - retained)
    if shutil.disk_usage(cache).free < remaining + 20 * 2**30:
        raise RuntimeError(
            f"Need {remaining / 2**30:.2f} GiB remaining feature banks plus20GiB reserve before building"
        )
    begin = time.perf_counter()
    extractor = make_extractor(provenance["identity"])
    extractor.model.requires_grad_(False).eval()
    torch.cuda.synchronize()
    progress_path = cache / "build-progress.json"
    request = {"parts": keys, "batch": args.build_batch}
    if progress_path.exists():
        result = read(progress_path)
        if result["request"] != request or result["provenance_sha256"] != sha(args.job / "provenance.json"):
            raise RuntimeError("Partial cache request/snapshot mismatch")
    else:
        result = {
            "version": VERSION,
            "request": request,
            "provenance_sha256": sha(args.job / "provenance.json"),
            "identity": provenance["identity"],
            "parts": {},
            "setup_seconds": time.perf_counter() - begin,
            "teacher_dtype": "fp32",
            "cache_dtype": "fp16",
            "build_batch": args.build_batch,
        }
    for key in keys:
        if key in result["parts"]:
            part = result["parts"][key]
            if stamp(part["file"]) != part["stat"] or sha(part["file"]) != part["sha256"]:
                raise RuntimeError("Previously completed build part changed")
            continue
        samples = [None] if key == "blank" else meta["splits"]["test" if key.startswith("test-") else "train"]
        target, partial = cache / f"{key}.npy", cache / f"{key}.partial.npy"
        offset = 0
        elapsed = 0.0
        parity = 0.0
        if result.get("partial", {}).get("key") == key:
            offset = result["partial"]["offset"]
            elapsed = result["partial"]["seconds"]
            parity = result["partial"]["parity_max"]
            if target.exists():
                raise RuntimeError(
                    "Interrupted between cache rename and seal; completed bytes retained for manual integrity recovery"
                )
        elif partial.exists() or target.exists():
            raise RuntimeError(f"Unaccounted partial cache retained for inspection: {key}")
        bank = open_partial_array(partial, np.float16, (len(samples), *SHAPE), offset)
        for start in range(offset, len(samples), args.build_batch):
            tick = time.perf_counter()
            if key == "blank":
                images = torch.full((1, 3, 640, 640), 114 / 255, dtype=torch.float32)
            else:
                scale, flip = EXTRA_VIEWS.get(key, (1.0, key.endswith("-v1")))
                images = []
                for sample in samples[start : start + args.build_batch]:
                    image, info = load_letterboxed_tensor(sample["image_path"], imgsz=640)
                    if info != sample["letterbox"]:
                        raise RuntimeError("Precomputed letterbox differs from actual image loading")
                    image, _, _ = geometry(image, torch.empty((0, 1)), torch.empty((0, 4)), flip, scale)
                    images.append(image)
                images = torch.stack(images)
            with torch.no_grad():
                levels = counted_extract(extractor, images, cache / "teacher-queries.jsonl", part=key, offset=start)
                values = torch.stack([levels[f"layer_{layer}"] for layer in provenance["identity"]["layers"]], dim=1)
            if tuple(values.shape[1:]) != SHAPE or not torch.isfinite(values).all():
                raise RuntimeError("Invalid teacher feature extraction")
            saved = values.detach().cpu().half()
            if not torch.isfinite(saved).all():
                raise RuntimeError("FP16 cache overflow")
            relative = float(
                (saved.to(values.device).float() - values.float()).norm() / values.float().norm().clamp_min(1e-8)
            )
            if relative > 0.02:
                raise RuntimeError(f"FP16 storage parity exceeds tolerance: {relative}")
            bank[start : start + len(images)] = saved.numpy()
            bank.flush()
            torch.cuda.synchronize()
            elapsed += time.perf_counter() - tick
            parity = max(parity, relative)
            result["partial"] = {"key": key, "offset": start + len(images), "seconds": elapsed, "parity_max": parity}
            dump(progress_path, result)
            if start % (50 * args.build_batch) == 0:
                print(f"BUILD {meta['dataset']} {key} {start + len(images)}/{len(samples)}", flush=True)
        del bank
        partial.replace(target)
        tick = time.perf_counter()
        digest = sha(target)
        elapsed += time.perf_counter() - tick
        result["parts"][key] = {
            "file": str(target.resolve()),
            "sha256": digest,
            "stat": stamp(target),
            "bytes": target.stat().st_size,
            "images": len(samples),
            "seconds": elapsed,
            "parity_relative_L2": parity,
            "logical_teacher_images": len(samples),
        }
        result.pop("partial", None)
        dump(progress_path, result)
    result["physical_teacher_queries"] = query_journal(cache / "teacher-queries.jsonl")
    result["teacher_query_journal_sha256"] = sha(cache / "teacher-queries.jsonl")
    result["cache_bytes"] = sum(part["bytes"] for part in result["parts"].values())
    result["build_seconds"] = result["setup_seconds"] + sum(part["seconds"] for part in result["parts"].values())
    dump(cache / "manifest.json", result)
    dump(cache / "complete.json", {"manifest_sha256": sha(cache / "manifest.json")})
    print(
        json.dumps({"cache_complete": True, "bytes": result["cache_bytes"], "seconds": result["build_seconds"]}),
        flush=True,
    )
    return result


def quantized_cache(job, bits):
    """Persist packed physical codes; the original FP16 source remains intact."""
    cache = cache_manifest(job)
    source = cache["parts"]["train-v0"]
    destination = job / "cache" / f"quantized-int{bits}"
    destination.mkdir(exist_ok=True)
    manifest_path = destination / "manifest.json"
    request = {
        "bits": bits,
        "quantile": 0.001,
        "source_sha256": source["sha256"],
        "cache_manifest_sha256": sha(job / "cache/manifest.json"),
    }
    if manifest_path.exists():
        manifest = read(manifest_path)
        if manifest["request"] != request or read(destination / "complete.json")["manifest_sha256"] != sha(
            manifest_path
        ):
            raise RuntimeError("Quantized physical bank request/seal mismatch")
        for part in manifest["parts"].values():
            if stamp(part["file"]) != part["stat"]:
                raise RuntimeError("Quantized physical bank changed")
        return {**manifest, "construction_seconds_this_invocation": 0.0}
    started = time.perf_counter()
    original = np.load(source["file"], mmap_mode="r", allow_pickle=False)
    shapes = {
        "codes": (*original.shape[:-1], original.shape[-1] // 2) if bits == 4 else original.shape,
        "low": original.shape[:3],
        "step": original.shape[:3],
    }
    required = math.prod(shapes["codes"]) + (math.prod(shapes["low"]) + math.prod(shapes["step"])) * 4
    if shutil.disk_usage(destination).free < required + 20 * 2**30:
        raise RuntimeError("Insufficient space for physical compressed bank plus20GiB reserve")
    progress_path = destination / "progress.json"
    progress = read(progress_path) if progress_path.exists() else {"request": request, "offset": 0, "seconds": 0.0}
    if progress["request"] != request:
        raise RuntimeError("Partial quantizer source/request changed")
    arrays = {}
    for key, shape in shapes.items():
        path = destination / f"{key}.npy"
        if path.exists() and not progress_path.exists():
            raise RuntimeError("Unaccounted partial quantizer files retained for inspection")
        arrays[key] = open_partial_array(path, np.uint8 if key == "codes" else np.float32, shape, progress["offset"])
    dump(progress_path, progress)
    prior_seconds = progress["seconds"]
    for offset in range(progress["offset"], len(original), 4):
        codes, low, step = quantize_array(original[offset : offset + 4], bits)
        for key, values in (("codes", codes), ("low", low), ("step", step)):
            arrays[key][offset : offset + len(values)] = values
            arrays[key].flush()
        progress.update(offset=offset + len(codes), seconds=prior_seconds + time.perf_counter() - started)
        dump(progress_path, progress)
        if offset % 400 == 0:
            print(f"QUANTIZE int{bits} {progress['offset']}/{len(original)}", flush=True)
    arrays.clear()
    parts = {}
    for key in shapes:
        path = destination / f"{key}.npy"
        parts[key] = {
            "file": str(path.resolve()),
            "sha256": sha(path),
            "stat": stamp(path),
            "bytes": path.stat().st_size,
        }
    manifest = {
        "version": VERSION,
        "request": request,
        "bits": bits,
        "shape": list(original.shape),
        "parts": parts,
        "cache_bytes": sum(part["bytes"] for part in parts.values()),
        "build_seconds": prior_seconds + time.perf_counter() - started,
        "teacher_query_images": 0,
        "packing": "low nibble first along W" if bits == 4 else "uint8",
        "scope": "Cache and Reuse idea adaptation; per-image/layer/channel quantile0.001, original FP16 source preserved",
    }
    dump(manifest_path, manifest)
    dump(destination / "complete.json", {"manifest_sha256": sha(manifest_path)})
    return {**manifest, "construction_seconds_this_invocation": time.perf_counter() - started}


def make_batches(job, method, seed, output):
    """Reuse existing paired augmentation logic with a dataset-specific mmap input."""
    import torch

    from online_experiments import replication_suite
    from online_experiments.feature_baselines import warp_features
    from online_experiments.full_visdrone_data import epoch_plan, make_extractor
    from scripts import d1_train_cached_detector as detector

    class QueryCounter:
        def __init__(self, extractor):
            self.extractor, self.model, self.device = extractor, extractor.model, extractor.device

        def __call__(self, images):
            return counted_extract(self.extractor, images, output / "teacher-queries.jsonl", method=method, seed=seed)

    class DatasetBatches(replication_suite.batch_class()):
        def __init__(self):
            provenance = verify_provenance(job)
            self.provenance = provenance
            self.meta, _ = load_dataset(provenance["manifest"])
            self.meta["identity"] = provenance["identity"]
            self.train, self.val = self.meta["splits"]["train"], self.meta["splits"]["test"]
            self.cache = cache_manifest(job)
            self.suite_method, self.seed, self.spec = method, seed, replication_suite.method_spec(method)
            self.method = self.spec["native_method"] or method
            self.pct, self.device = self.spec["query_pct"], torch.device("cuda:0")
            keys = consumed_parts(method)
            missing = set(keys) - set(self.cache["parts"])
            if missing:
                raise RuntimeError(
                    f"Missing method-specific cache parts: {sorted(missing)}; do not silently build new views"
                )
            compressed = method.startswith("cache-reuse-")
            self.arrays = (
                []
                if compressed
                else [
                    np.load(self.cache["parts"][key]["file"], mmap_mode="r", allow_pickle=False)
                    for key in keys
                    if key in ("train-v0", "train-v1")
                ]
            )
            self.val_array = np.load(self.cache["parts"]["test-v0"]["file"], mmap_mode="r", allow_pickle=False)
            self.view_arrays = (
                {
                    (scale, int(flip)): np.load(self.cache["parts"][key]["file"], mmap_mode="r", allow_pickle=False)
                    for key, (scale, flip) in EXTRA_VIEWS.items()
                }
                if method == "views25"
                else {}
            )
            self.quantized = quantized_cache(job, int(method.rsplit("int", 1)[1])) if compressed else None
            self.codes = self.low = self.step = None
            if compressed:
                self.codes, self.low, self.step = [
                    np.load(self.quantized["parts"][key]["file"], mmap_mode="r", allow_pickle=False)
                    for key in ("codes", "low", "step")
                ]
            self.blank = (
                torch.from_numpy(np.load(self.cache["parts"]["blank"]["file"], mmap_mode="r")[0].copy())
                .to(self.device)
                .float()
                if "blank" in keys
                else None
            )
            self.extractor = None
            if self.pct:
                extractor = make_extractor(provenance["identity"])
                extractor.model.requires_grad_(False).eval()
                self.extractor = QueryCounter(extractor)
            self.query_images = self.query_calls = self.correction_steps = 0
            self.correction_seconds = self.raw_image_seconds = self.dequant_seconds = 0.0
            self.raw_image_images = self.dequant_images = 0
            self.reconstruction = []
            self.corrector = self.optimizer = None

        def training(self, epoch, batch_size, target):
            if self.quantized is None:
                yield from super().training(epoch, batch_size, target)
                return
            order, flips, scales, query = epoch_plan(len(self.train), 0, epoch, self.seed)
            schedule = {"epoch": epoch + 1, "order": order, "flip": flips, "scale": scales, "query": query}
            path = target / "schedules" / f"{epoch + 1:03d}.json"
            if path.exists() and read(path) != schedule:
                raise RuntimeError("Resumed physical-cache schedule changed")
            dump(path, schedule)
            for start in range(0, len(order), batch_size):
                ids, fs, ss = (value[start : start + batch_size] for value in (order, flips, scales))
                tick = time.perf_counter()
                values = dequantize_array(self.codes[ids], self.low[ids], self.step[ids], self.quantized["bits"], 40)
                self.dequant_seconds += time.perf_counter() - tick
                self.dequant_images += len(ids)
                values = torch.from_numpy(values).to(self.device)
                levels = tuple(warp_features(values[:, level], fs, ss) for level in range(3))
                items = [self.item(self.train[index], flip, scale) for index, flip, scale in zip(ids, fs, ss)]
                for index, item in enumerate(items):
                    item["features"] = tuple(level[index] for level in levels)
                yield detector.collate_cached(items)

    return DatasetBatches()


def costs(batches, method):
    cache = batches.cache
    keys = consumed_parts(method)
    result = {
        "consumed_cache_parts": keys,
        "cache_build_charged_seconds": cache["setup_seconds"] + sum(cache["parts"][key]["seconds"] for key in keys),
        "cache_bytes": sum(cache["parts"][key]["bytes"] for key in keys),
        "cache_build_logical_teacher_images": sum(cache["parts"][key]["logical_teacher_images"] for key in keys),
        "retained_total_cache_bytes": cache["cache_bytes"],
        "cache_charge_note": "Per-configuration cold charge includes common test cache; cached files are physically shared between seeds/methods.",
        "data_prepare_charged_seconds": batches.provenance["snapshot_prepare_seconds"],
    }
    if batches.quantized:
        result["cache_bytes"] += batches.quantized["cache_bytes"] - cache["parts"]["train-v0"]["bytes"]
        result["cache_build_charged_seconds"] += batches.quantized["build_seconds"]
        result["quantization_seconds"] = batches.quantized["build_seconds"]
        result["compressed_training_bytes"] = batches.quantized["cache_bytes"]
        result["retained_total_cache_bytes"] += batches.quantized["cache_bytes"]
    return result


def final_predictions(model, batches, output):
    import torch

    from scripts import d1_train_cached_detector as detector

    predictions, coverage = [], []
    model.eval()
    with torch.inference_mode():
        for batch in batches.validation(8):
            features, _ = detector.move_batch(batch, torch.device("cuda:0"))
            decoded, _ = model(features)
            for index, sample_id in enumerate(batch["sample_ids"]):
                sample = batches.val[len(coverage)]
                if sample["sample_id"] != sample_id:
                    raise RuntimeError("Test prediction sample order mismatch")
                values = decoded[index]
                values = values[values[:, 4] >= 0.001].cpu().numpy()
                values = inverse_predictions(values, sample["letterbox"])
                if not np.isfinite(values).all():
                    raise RuntimeError("Nonfinite final test prediction")
                for value in values:
                    category = int(value[5])
                    if value[5] != category or not 0 <= category < batches.meta["nc"]:
                        raise RuntimeError("Invalid predicted dataset category")
                    if value[2] <= value[0] or value[3] <= value[1]:
                        continue
                    predictions.append(
                        {
                            "sample_id": sample_id,
                            "bbox_xyxy0_half_open": value[:4].tolist(),
                            "score": float(value[4]),
                            "category_id": category,
                        }
                    )
                coverage.append(sample_id)
    if coverage != [sample["sample_id"] for sample in batches.val]:
        raise RuntimeError("Incomplete full test prediction coverage")
    dump(
        output / "predictions.json",
        {
            "schema": VERSION,
            "coverage": coverage,
            "predictions": predictions,
            "coordinates": "zero-based half-open original image pixels",
            "max_det": 500,
            "confidence": 0.001,
        },
    )
    return predictions


def score_final(batches, predictions, output):
    samples, names = batches.val, batches.meta["class_names"]
    if batches.meta["dataset"] == "voc0712":
        metrics = {
            "primary": evaluate_voc07(samples, predictions, names),
            "endpoint": "VOC2007test; trainVOC2007+2012trainval; final100epoch; eleven-point AP50",
        }
    else:
        metrics = {
            "primary": evaluate_coco(samples, predictions, names, output),
            "endpoint": "SKU110K officialtest; fixed final100epoch; primary standardCOCO AP@maxDet100; dense maxDet500 secondary",
        }
    dump(output / "metrics.json", metrics)
    return metrics


def validate_completed(args):
    output = args.output or args.job / "train" / f"{args.method}-s{args.seed}"
    provenance = verify_provenance(args.job)
    config, summary, endpoint = (
        read(output / "args.json"),
        read(output / "summary.json"),
        read(output / "endpoint.json"),
    )
    if config["method"] != args.method or config["seed"] != args.seed or config["epochs"] != 100:
        raise RuntimeError("Completed run method/seed/epoch mismatch")
    if config["provenance_sha256"] != sha(args.job / "provenance.json") or summary["completed_epochs"] != 100:
        raise RuntimeError("Completed run source/split or final epoch mismatch")
    if summary["query_images"] != 100 * provenance["counts"]["train"] * config["query_pct"] // 100:
        raise RuntimeError("Completed run logical teacher budget mismatch")
    for name, digest in endpoint["sha256"].items():
        if sha(output / name) != digest:
            raise RuntimeError(f"Final checkpoint/prediction/evaluation hash mismatch: {name}")
    if len(summary["epochs"]) != 100 or [row["epoch"] for row in summary["epochs"]] != list(range(1, 101)):
        raise RuntimeError("Completed run has missing/repeated epochs")
    schedules = {path.name: sha(path) for path in sorted((output / "schedules").glob("*.json"))}
    if schedules != endpoint["schedule_sha256"] or len(schedules) != 100:
        raise RuntimeError("Completed augmentation schedules changed")
    predictions = read(output / "predictions.json")
    if len(predictions["coverage"]) != provenance["counts"]["test"] or len(set(predictions["coverage"])) != len(
        predictions["coverage"]
    ):
        raise RuntimeError("Completed run full held-out test coverage mismatch")
    expected_ids = [sample["sample_id"] for sample in read(provenance["manifest"])["splits"]["test"]]
    if predictions["coverage"] != expected_ids:
        raise RuntimeError("Completed prediction IDs disagree with the exact official split order")
    if query_journal(output / "teacher-queries.jsonl") != summary["physical_training_teacher_queries"]:
        raise RuntimeError("Physical training teacher-call evidence changed")
    cache_manifest(args.job)
    return summary


def restore_epoch_evidence(output, state):
    """Repair the summary sidecar from the last atomic checkpoint after a crash."""
    summary = state["summary"]
    if summary["completed_epochs"] != state["epoch"] or summary["epochs"] != state["rows"]:
        raise RuntimeError("Atomic checkpoint epoch evidence disagrees with its summary")
    if summary["provenance_sha256"] != state["args"]["provenance_sha256"]:
        raise RuntimeError("Atomic checkpoint summary belongs to another protocol")
    dump(output / "summary.json", summary)


def train(args):
    import resource

    import torch

    from online_experiments.assignment_intervention import state_hash
    from online_experiments.detail_probe import telemetry
    from online_experiments.detail_probe_model import DetailDetector
    from online_experiments.full_visdrone_train import atomic_checkpoint
    from online_experiments.fusion_probe import configure_fusion
    from online_experiments.replication_suite import check_counts, method_spec
    from scripts import d1_train_cached_detector as detector

    provenance = verify_provenance(args.job)
    output = args.output or args.job / "train" / f"{args.method}-s{args.seed}"
    if (output / "DONE").exists():
        validate_completed(args)
        print(f"SKIP verified final {args.method}-s{args.seed}", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    (output / "weights").mkdir(exist_ok=True)
    (output / "teacher-queries.jsonl").touch(exist_ok=True)
    torch.set_num_threads(4)
    detector.seed_everything(args.seed)
    started = time.perf_counter()
    batches = make_batches(args.job, args.method, args.seed, output)
    if batches.quantized:
        # Construction belongs to the cold cache charge. Do not count its new
        # physical work a second time inside warm training setup.
        started += batches.quantized["construction_seconds_this_invocation"]
    spec = method_spec(args.method)
    detector.seed_everything(args.seed)
    factory = DetailDetector
    if args.method.startswith("vit-adapter-"):
        from online_experiments.paper_adapter_model import PaperAdapterDetector

        factory = PaperAdapterDetector
    model = factory(
        in_channels=(384,) * 3, nc=batches.meta["nc"], imgsz=640, epochs=100, detail_mode=spec["detail_mode"]
    ).cuda()
    configure_fusion(model, "mean_value")
    model.detect.max_det = 500
    criterion = detector.E2EDetectLoss(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.00001)
    config = {
        "version": VERSION,
        "dataset": batches.meta["dataset"],
        "method": args.method,
        "seed": args.seed,
        "epochs": 100,
        "batch": 8,
        "query_pct": spec["query_pct"],
        "method_contract": spec,
        "nc": batches.meta["nc"],
        "class_names": batches.meta["class_names"],
        "train_images": len(batches.train),
        "test_images": len(batches.val),
        "manifest_sha256": provenance["manifest_sha256"],
        "provenance_sha256": sha(args.job / "provenance.json"),
        "cache_manifest_sha256": sha(args.job / "cache/manifest.json"),
        "protocol": provenance["protocol"],
        "quantized_manifest_sha256": sha(
            args.job / "cache" / f"quantized-int{batches.quantized['bits']}" / "manifest.json"
        )
        if batches.quantized
        else None,
    }
    if (output / "args.json").exists() and read(output / "args.json") != config:
        raise RuntimeError("Run resume config/source/cache changed")
    dump(output / "args.json", config)
    environment = {
        "python": sys.version,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "transformers", "scipy")},
    }
    if (output / "environment.json").exists():
        if read(output / "environment.json")["identity"] != environment:
            raise RuntimeError("Resume package/GPU environment changed")
    else:
        dump(output / "environment.json", {"identity": environment, "telemetry": telemetry()})
    initialization = {
        "parameters": sum(value.numel() for value in model.parameters()),
        "trainable_parameters": sum(value.numel() for value in model.parameters() if value.requires_grad),
        "detail_parameters": model.detail_parameter_count,
        "base_head_sha256": state_hash(
            type(
                "State",
                (),
                {
                    "state_dict": lambda _self: {
                        key: value for key, value in model.state_dict().items() if not key.startswith("detail_branch.")
                    }
                },
            )()
        ),
        "model_contract": getattr(model, "paper_contract", None),
        "teacher_loaded": batches.extractor is not None,
    }
    if (output / "initialization.json").exists() and read(output / "initialization.json") != initialization:
        raise RuntimeError("Paired model initialization changed on resume")
    dump(output / "initialization.json", initialization)
    rows, begin = [], 0
    totals = {
        "train_seconds": 0.0,
        "wall_seconds": 0.0,
        "peak_host_rss_mib": 0.0,
        "raw_image_seconds": 0.0,
        "raw_image_images": 0,
        "dequant_seconds": 0.0,
        "dequant_images": 0,
    }
    checkpoint = output / "weights/last.pt"
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state["args"] != config:
            raise RuntimeError("Checkpoint request/source changed")
        restore_epoch_evidence(output, state)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        rows, begin, totals = state["rows"], state["epoch"], state["totals"]
        batches.query_images, batches.query_calls = state["query_images"], state["query_calls"]
        for key in ("raw_image_seconds", "raw_image_images", "dequant_seconds", "dequant_images"):
            setattr(batches, key, totals[key])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        if begin != len(rows) or not 0 <= begin <= 100:
            raise RuntimeError("Checkpoint epoch evidence is inconsistent")
        print(f"RESUME {args.method}-s{args.seed} epoch={begin + 1}", flush=True)
    prior_wall = totals["wall_seconds"]
    cold = costs(batches, args.method)
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(begin, 100):
        model.train()
        torch.cuda.synchronize()
        tick, loss_sum, seen = time.perf_counter(), 0.0, 0
        for batch in batches.training(epoch, 8, output):
            features, targets = detector.move_batch(batch, torch.device("cuda:0"))
            optimizer.zero_grad(set_to_none=True)
            vector, _ = criterion(model.training_predictions(features), targets)
            loss = vector.sum()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite full-dataset loss epoch={epoch + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            loss_sum += float(loss.detach())
            seen += len(batch["sample_ids"])
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - tick
        check_counts(epoch + 1, seen, len(batches.train), batches.query_images, spec["query_pct"])
        if batches.extractor and any(
            parameter.requires_grad or parameter.grad is not None for parameter in batches.extractor.model.parameters()
        ):
            raise RuntimeError("Frozen teacher trainability changed")
        row = {
            "epoch": epoch + 1,
            "train_loss": loss_sum / seen,
            "train_seconds": elapsed,
            "query_images": batches.query_images,
            "query_calls": batches.query_calls,
            "lr": optimizer.param_groups[0]["lr"],
            "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
        if not all(math.isfinite(value) for value in row.values()):
            raise RuntimeError("Nonfinite epoch evidence")
        rows.append(row)
        scheduler.step()
        totals["train_seconds"] += elapsed
        totals["wall_seconds"] = prior_wall + time.perf_counter() - started
        totals["peak_host_rss_mib"] = max(
            totals["peak_host_rss_mib"], resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        )
        for key in ("raw_image_seconds", "raw_image_images", "dequant_seconds", "dequant_images"):
            totals[key] = getattr(batches, key)
        summary = {
            **config,
            **cold,
            **totals,
            "completed_epochs": epoch + 1,
            "epochs": rows,
            "query_images": batches.query_images,
            "query_calls": batches.query_calls,
            "peak_vram_gib": max(record["peak_vram_gib"] for record in rows),
            "physical_training_teacher_queries": query_journal(output / "teacher-queries.jsonl"),
            "cold_seconds": cold["cache_build_charged_seconds"]
            + cold["data_prepare_charged_seconds"]
            + totals["wall_seconds"],
            "warm_seconds": totals["wall_seconds"],
            "metrics": None,
            "progress_note": "Training only; full held-out test is evaluated after epoch100.",
            "resume_cost_note": "Checkpoint-accounted wall/train times exclude uncheckpointed partial-epoch work lost on interruption; physical teacher journal includes replayed calls.",
        }
        state = {
            "epoch": epoch + 1,
            "args": config,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rows": rows,
            "totals": totals.copy(),
            "query_images": batches.query_images,
            "query_calls": batches.query_calls,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
            "summary": summary,
        }
        atomic_checkpoint(checkpoint, state)
        dump(output / "summary.json", summary)
        append_jsonl(output / "telemetry.jsonl", telemetry())
        print(json.dumps(row), flush=True)
    # No held-out labels/predictions enter model updates or checkpoint selection.
    tick = time.perf_counter()
    predictions = final_predictions(model, batches, output)
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    metrics = score_final(batches, predictions, output)
    evaluation_seconds = time.perf_counter() - tick
    summary = read(output / "summary.json")
    summary.update(
        metrics=metrics,
        final_inference_seconds=inference_seconds,
        evaluation_seconds=evaluation_seconds,
        wall_seconds=prior_wall + time.perf_counter() - started,
        progress_note="Completed fixed final epoch100 and full held-out test.",
    )
    summary["warm_seconds"] = summary["wall_seconds"]
    summary["cold_seconds"] = (
        cold["cache_build_charged_seconds"] + cold["data_prepare_charged_seconds"] + summary["wall_seconds"]
    )
    summary["raw_image_seconds"], summary["raw_image_images"] = batches.raw_image_seconds, batches.raw_image_images
    summary["physical_training_teacher_queries"] = query_journal(output / "teacher-queries.jsonl")
    dump(output / "summary.json", summary)
    endpoint = {
        "fixed_epoch": 100,
        "selected_by": "protocol only; no best-score selection",
        "sha256": {
            name: sha(output / name)
            for name in (
                "args.json",
                "environment.json",
                "initialization.json",
                "summary.json",
                "metrics.json",
                "predictions.json",
                "weights/last.pt",
                "teacher-queries.jsonl",
            )
        },
        "schedule_sha256": {path.name: sha(path) for path in sorted((output / "schedules").glob("*.json"))},
    }
    dump(output / "endpoint.json", endpoint)
    validate_completed(args)
    (output / "DONE").write_text("SUCCEEDED\n", encoding="utf-8")
    print(json.dumps({"completed": args.method, "seed": args.seed, "metrics": metrics}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("snapshot", "cache", "train", "validate"))
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--identity", type=Path)
    parser.add_argument("--build-batch", type=int, default=8)
    parser.add_argument("--views", action="store_true", help="Build four additional real scale banks only if planned")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.job = args.job.resolve()
    if args.seed < 0 or args.build_batch < 1:
        parser.error("seed must be nonnegative and build-batch positive")
    if args.action == "snapshot":
        if args.manifest is None or args.identity is None:
            parser.error("snapshot requires --manifest and --identity")
        snapshot(args)
    elif args.action == "cache":
        build_cache(args)
    else:
        if args.method is None:
            parser.error("train/validate requires --method")
        if args.action == "train":
            train(args)
        else:
            validate_completed(args)
            print(f"VALIDATED {args.method}-s{args.seed}", flush=True)


if __name__ == "__main__":
    main()
