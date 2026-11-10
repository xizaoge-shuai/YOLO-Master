"""Full548 geometric feature and fixed-head loss diagnostic; no official AP.

Six fresh teacher views per image, shared by all four frozen models. Temporary
identity/flip anchors are reused within each image and never become a persistent
validation cache. inspect/selfcheck use CPU only; run is an explicit GPU stage.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from online_experiments import spatial_feature_analysis as spatial
from online_experiments.analyze_detection_evidence import dump, read, sha, write_csv
from online_experiments.visdrone_reference_eval import prepare_arrays

VIEWS = (
    ("identity", False, 1.0),
    ("hflip", True, 1.0),
    ("scale085", False, 0.85),
    ("scale095", False, 0.95),
    ("scale105", False, 1.05),
    ("scale115", False, 1.15),
)
IMAGES = 548
TEACHER_IMAGES = IMAGES * len(VIEWS)
SOURCE_FILES = (
    "online_experiments/geometric_evidence.py",
    "online_experiments/spatial_feature_analysis.py",
    "online_experiments/analyze_detection_evidence.py",
    "online_experiments/visdrone_reference_eval.py",
    "online_experiments/detail_probe_data.py",
    "online_experiments/d1_online_compare.py",
    "online_experiments/feature_baselines.py",
    "online_experiments/full_visdrone_data.py",
    "scripts/d1_build_feature_cache.py",
    "scripts/d1_train_cached_detector.py",
    "ultralytics/utils/loss.py",
    "ultralytics/utils/tal.py",
)


def project_gt(gt, info):
    """Original eight-column GT -> loss targets, after reference ignore processing."""
    gt = np.asarray(gt, dtype=np.float64).reshape(-1, 8)
    prepared, _ = prepare_arrays(gt, np.empty((0, 8)), info["original_shape"])
    target_class = np.isin(prepared[:, 5], np.arange(1, 11))
    evaluable = prepared[(prepared[:, 4] == 0) & target_class]
    valid = (evaluable[:, 2:4] > 0).all(axis=1)
    selected = evaluable[valid]
    height, width = info["original_shape"]
    rh, rw = info["resized_shape"]
    left, top = info["pad"]
    boxes = selected[:, :4].copy()
    boxes[:, 0] = (boxes[:, 0] + boxes[:, 2] / 2) * rw / width + left
    boxes[:, 1] = (boxes[:, 1] + boxes[:, 3] / 2) * rh / height + top
    boxes[:, 2] *= rw / width
    boxes[:, 3] *= rh / height
    boxes /= 640
    counts = {
        "raw_rows": len(gt),
        "raw_ignore_regions": int(np.count_nonzero(gt[:, 5] == 0)),
        "removed_by_reference_ignore_regions": int(np.count_nonzero(gt[:, 5] != 0) - len(prepared)),
        "ignored_gt_after_prepare": int(np.count_nonzero(prepared[:, 4] != 0)),
        "nontarget_class_after_prepare": int(np.count_nonzero(~target_class)),
        "nonpositive_target_boxes": int(np.count_nonzero(~valid)),
        "valid_targets": len(selected),
    }
    return (selected[:, 5:6] - 1).astype(np.float32), boxes.astype(np.float32), counts


def transformed_targets(classes, boxes, flip, scale):
    """NumPy contract of existing center-scale, canvas clip and >=10%/1px keep."""
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Scale must be finite and positive")
    classes, boxes = np.array(classes, copy=True), np.array(boxes, copy=True)
    before_count = len(boxes)
    clipped_count = 0
    if scale != 1.0 and len(boxes):
        lo = (boxes[:, :2] - boxes[:, 2:] / 2 - 0.5) * scale + 0.5
        hi = (boxes[:, :2] + boxes[:, 2:] / 2 - 0.5) * scale + 0.5
        before = (hi - lo).prod(axis=1)
        clipped_count = int(np.count_nonzero(((lo < 0) | (hi > 1)).any(axis=1)))
        lo, hi = lo.clip(0, 1), hi.clip(0, 1)
        size = hi - lo
        keep = (size * 640 >= 1).all(axis=1)
        keep &= size.prod(axis=1) / np.maximum(before, 1e-12) >= 0.1
        boxes = np.concatenate(((hi + lo) / 2, size), axis=1)[keep]
        classes = classes[keep]
    if flip:
        boxes[:, 0] = 1 - boxes[:, 0]
    return (
        classes,
        boxes,
        {
            "targets_before_geometry": before_count,
            "targets_after_geometry": len(boxes),
            "crop_discarded_targets": before_count - len(boxes),
            "clipped_candidates_including_discarded": clipped_count,
        },
    )


def source_hashes():
    return {name: sha(ROOT / name) for name in SOURCE_FILES}


def validate_inputs(args):
    manifest, cache, records, bindings, _ = spatial.validate_inputs(args, include_bank=False)
    if len(records) != IMAGES or len(manifest["splits"]["val"]) != IMAGES:
        raise ValueError("Exactly 548 images required; no subset option")
    for sample, record in zip(manifest["splits"]["val"], records):
        if Path(sample["image_path"]).stem != record["id"]:
            raise ValueError("Raw GT order differs from frozen image order")
        if record["shape"] != sample["letterbox"]["original_shape"]:
            raise ValueError("Raw GT shape differs from image letterbox")
    for method, run in spatial.run_paths(args).items():
        provenance = read(run.parents[1] / "provenance.json")
        for name in SOURCE_FILES[4:]:
            if provenance["source_sha256"].get(name) != sha(ROOT / name):
                raise ValueError(f"Diagnostic dependency changed from frozen training source: {method}/{name}")
    blank_part = cache["parts"]["blank"]
    blank_path = spatial.existing_path(blank_part["file"])
    if blank_path.stat().st_size != blank_part["bytes"] or sha(blank_path) != blank_part["sha256"]:
        raise ValueError("Frozen true-blank feature changed")
    blank = np.load(blank_path, mmap_mode="r", allow_pickle=False)
    if blank.shape != (1, 3, 384, 40, 40) or blank.dtype != np.float16:
        raise ValueError("Expected one true-blank FP16 feature at three teacher layers")
    return manifest, cache, records, bindings, blank


def contract(args, manifest, cache, bindings):
    return {
        "schema": "full548-geometric-evidence-v1",
        "images": IMAGES,
        "views": [{"name": name, "flip": flip, "scale": scale} for name, flip, scale in VIEWS],
        "methods": list(spatial.METHODS),
        "checkpoint_selection": "fixed final epoch100 seed0, no re-selection",
        "bindings": bindings,
        "teacher": manifest["identity"],
        "teacher_images_expected": TEACHER_IMAGES,
        "teacher_blank_images_new": 0,
        "teacher_calls_expected": TEACHER_IMAGES,
        "teacher_interface": "fresh FP32 DINO; layer3/7/11 FP16 roundtrip -> FP32, as in frozen training",
        "temporary_C2": "fresh identity/hflip per image, reused as exact references and C2 anchors; no extra calls",
        "approximation": "choose flip-matched true C2 anchor then existing center-scale transport + true blank fill",
        "blank": {"sha256": cache["parts"]["blank"]["sha256"], "new_teacher_images": 0},
        "geometry_targets": "raw eight-column GT; prepare_arrays then nonignored class1..10, "
        "linear letterbox projection, frozen geometry+transform_targets clipping with >=1pixel and >=10% area",
        "loss_ignore_semantics": "ignored targets/regions are removed before transform; training loss does not mask "
        "ignored background pixels. Cropped ignored regions are not an official transformed-evaluation protocol.",
        "loss": "fixed eval-mode model, no_grad; E2EDetectLoss one2many+one2one with box/cls/dfl gains7.5/.5/1.5; "
        "identical image/GT/RGB for approximate and fresh exact feature inputs within each own frozen model",
        "P3_reference": "same model's P3 on fresh exact teacher features of same view; not ground-truth P3",
        "relative_error": "sqrt(sum((approx-exact)^2)/sum(exact^2)); NA if reference energy is zero",
        "RGB_interpretation": "raw DINO transport error shared and reported once; RGB branches do not directly "
        "restore raw DINO. An additive branch with identical RGB cancels from absolute P3 difference; "
        "relative denominators and loss can differ. No feature-error metric is an AP surrogate.",
        "model_mode": "eval + requires_gradFalse + no_grad, no optimizer or weight/BN-stat updates",
        "source_sha256": source_hashes(),
        "batch": 1,
        "official_AP_computed": False,
        "training_run": False,
        "deployment_or_speed_claim": False,
        "output": str(args.output) if args.output else None,
    }


def feature_error(value, reference, torch):
    difference = value.double() - reference.double()
    error_sum = float(difference.square().sum().item())
    reference_sum = float(reference.double().square().sum().item())
    return {
        "squared_error_sum": error_sum,
        "reference_squared_sum": reference_sum,
        "elements": value.numel(),
        "absolute_l2": math.sqrt(error_sum),
        "rmse": math.sqrt(error_sum / value.numel()),
        "relative_l2": math.sqrt(error_sum / reference_sum) if reference_sum else None,
    }


def fixed_forward(model, inputs, targets, criterion, torch):
    """Reuse one pyramid for fixed raw one2many/one2one loss, without decode/AP."""
    pyramid = model.pyramid(inputs)
    prediction = {
        "one2many": model.detect.forward_head(pyramid, **model.detect.one2many),
        "one2one": model.detect.forward_head([value.detach() for value in pyramid], **model.detect.one2one),
    }
    vector, items = criterion(prediction, targets)
    if not torch.isfinite(vector).all() or not torch.isfinite(items).all():
        raise ValueError("Nonfinite fixed-head loss")
    return pyramid[0].detach().clone(), {
        "total": float(vector.sum().item()),
        **{key: float(value) for key, value in zip(("box", "cls", "dfl"), items.detach().cpu().tolist())},
    }


def summarise(rows, group_keys, metrics):
    result = []
    groups = sorted({tuple(row[key] for key in group_keys) for row in rows})
    for group in groups:
        selected = [row for row in rows if tuple(row[key] for key in group_keys) == group]
        for metric in metrics:
            values = np.array([row[metric] for row in selected if row[metric] is not None], dtype=float)
            result.append(
                {
                    **dict(zip(group_keys, group)),
                    "metric": metric,
                    "images_total": len(selected),
                    "images_with_metric": len(values),
                    "mean": float(values.mean()) if len(values) else None,
                    "sample_sd_across_images_not_training_seeds": float(values.std(ddof=1))
                    if len(values) > 1
                    else None,
                }
            )
    return result


def run(args, manifest, cache, records, bindings, blank):
    torch, hardware = spatial.gpu()
    from online_experiments.d1_online_compare import geometry, training_feature
    from online_experiments.detail_probe_data import transport_with_blank
    from online_experiments.feature_baselines import transform_targets
    from online_experiments.full_visdrone_data import make_extractor
    from ultralytics.data.foundation_cache import load_letterboxed_tensor
    from ultralytics.utils.loss import E2EDetectLoss

    dump(args.output / "hardware-start.json", {**hardware, "compute_apps": spatial.gpu_snapshot()})
    models = {method: spatial.load_model(method, path, torch) for method, path in spatial.run_paths(args).items()}
    for model in models.values():
        model.requires_grad_(False)
        model.eval()
    criteria = {method: E2EDetectLoss(model) for method, model in models.items()}
    extractor = make_extractor(manifest["identity"])
    if hasattr(extractor, "model"):
        extractor.model.requires_grad_(False)
        extractor.model.eval()
    true_blank = torch.from_numpy(np.array(blank[0], copy=True)).to("cuda:0").float()
    teacher_images = teacher_calls = 0
    feature_rows, model_rows, target_rows = [], [], []

    def extract(rgb):
        nonlocal teacher_images, teacher_calls
        extracted = extractor(rgb)
        teacher_images += len(rgb)
        teacher_calls += 1
        values = spatial.teacher_interface(extracted, manifest["identity"]["layers"], training_feature)
        if any(value.shape != (1, 384, 40, 40) or not torch.isfinite(value).all() for value in values):
            raise ValueError("Unexpected teacher feature shape or nonfinite value")
        return values

    with torch.no_grad():
        for index, (sample, record) in enumerate(zip(manifest["splits"]["val"], records)):
            spatial.verify_image(sample)
            image, info = load_letterboxed_tensor(sample["image_path"], imgsz=640)
            if info != sample["letterbox"]:
                raise ValueError("Frozen letterbox geometry changed")
            classes_np, boxes_np, original_counts = project_gt(record["gt"], info)
            classes, boxes = torch.from_numpy(classes_np), torch.from_numpy(boxes_np)
            anchors = {}
            for name, flip, scale in VIEWS:
                augmented, actual_classes, actual_boxes = geometry(image, classes, boxes, flip, scale)
                frozen_classes, frozen_boxes = transform_targets(classes, boxes, flip, scale, 640)
                expected_classes, expected_boxes, counts = transformed_targets(classes_np, boxes_np, flip, scale)
                if not torch.equal(actual_classes, frozen_classes) or not torch.equal(
                    actual_classes, torch.from_numpy(expected_classes)
                ):
                    raise ValueError("Image geometry and loss-target class selection disagree")
                if not torch.allclose(actual_boxes, frozen_boxes, atol=2e-6, rtol=0) or not np.allclose(
                    actual_boxes.numpy(), expected_boxes, atol=2e-6, rtol=0
                ):
                    raise ValueError("Image geometry and loss-target coordinates disagree")
                rgb = augmented[None].to("cuda:0")
                exact = extract(rgb)
                if scale == 1.0:
                    anchors[flip] = exact
                anchor = anchors[flip]
                approximate = tuple(
                    transport_with_blank(value, [scale], true_blank[level][None])[0]
                    for level, value in enumerate(anchor)
                )
                target_rows.append(
                    {"image_id": record["id"], "view": name, "flip": flip, "scale": scale, **original_counts, **counts}
                )
                targets = {
                    "cls": actual_classes.to("cuda:0"),
                    "bboxes": actual_boxes.to("cuda:0"),
                    "batch_idx": torch.zeros(len(actual_classes), dtype=torch.long, device="cuda:0"),
                }
                for level, layer in enumerate(manifest["identity"]["layers"]):
                    feature_rows.append(
                        {
                            "image_id": record["id"],
                            "view": name,
                            "layer": layer,
                            **feature_error(approximate[level], exact[level], torch),
                        }
                    )
                for method, model in models.items():
                    extra = (rgb,) if model.detail_branch is not None else ()
                    exact_p3, exact_loss = fixed_forward(model, exact + extra, targets, criteria[method], torch)
                    approx_p3, approx_loss = fixed_forward(model, approximate + extra, targets, criteria[method], torch)
                    if exact_p3.shape != (1, 64, 80, 80) or approx_p3.shape != exact_p3.shape:
                        raise ValueError("Expected fixed P3 shape1x64x80x80")
                    row = {
                        "method": method,
                        "image_id": record["id"],
                        "view": name,
                        **feature_error(approx_p3, exact_p3, torch),
                    }
                    for key in exact_loss:
                        row[f"exact_loss_{key}"] = exact_loss[key]
                        row[f"approx_loss_{key}"] = approx_loss[key]
                        row[f"paired_loss_delta_{key}"] = approx_loss[key] - exact_loss[key]
                    model_rows.append(row)
            if teacher_images != (index + 1) * len(VIEWS):
                raise ValueError("Teacher count violates six fresh views with reused C2 anchors")
            if (index + 1) % 32 == 0 or index + 1 == IMAGES:
                print(
                    f"GEOMETRY images={index + 1}/{IMAGES} teacher_images={teacher_images}/{TEACHER_IMAGES}", flush=True
                )
                dump(
                    args.output / "progress.json",
                    {"images_complete": index + 1, "teacher_images": teacher_images, "teacher_calls": teacher_calls},
                )
        torch.cuda.synchronize()
    if teacher_images != TEACHER_IMAGES or teacher_calls != TEACHER_IMAGES:
        raise ValueError("Final diagnostic teacher image/call count mismatch")
    if len(feature_rows) != IMAGES * len(VIEWS) * 3 or len(model_rows) != IMAGES * len(VIEWS) * 4:
        raise ValueError("Incomplete full-split diagnostic outputs")
    write_csv(args.output / "raw-dino-error.csv", feature_rows)
    write_csv(args.output / "p3-and-loss.csv", model_rows)
    write_csv(args.output / "gt-geometry-counts.csv", target_rows)
    write_csv(args.output / "raw-dino-summary.csv", summarise(feature_rows, ("view", "layer"), ("relative_l2", "rmse")))
    loss_metrics = tuple(
        f"{prefix}_{key}"
        for prefix in ("exact_loss", "approx_loss", "paired_loss_delta")
        for key in ("total", "box", "cls", "dfl")
    )
    write_csv(
        args.output / "p3-and-loss-summary.csv",
        summarise(model_rows, ("method", "view"), ("relative_l2", "rmse") + loss_metrics),
    )
    return {
        "images": IMAGES,
        "teacher_images": teacher_images,
        "teacher_calls": teacher_calls,
        "teacher_blank_images_new": 0,
        "methods": list(models),
        "method_image_views": len(model_rows),
        "official_AP_computed": False,
        "training_updates": 0,
    }


def selfcheck():
    info = {"original_shape": [100, 200], "resized_shape": [320, 640], "pad": [0, 160]}
    gt = np.array([[20, 10, 40, 20, 1, 1, 0, 0]], dtype=float)
    classes, boxes, counts = project_gt(gt, info)
    assert np.allclose(boxes, [[0.2, 0.35, 0.2, 0.1]])
    assert classes.tolist() == [[0]] and counts["valid_targets"] == 1
    _, flipped, unchanged = transformed_targets(classes, boxes, True, 1)
    assert np.allclose(flipped, [[0.8, 0.35, 0.2, 0.1]]) and unchanged["crop_discarded_targets"] == 0
    c = np.array([[0], [1]], dtype=np.float32)
    b = np.array([[0.01, 0.5, 0.005, 0.1], [0.5, 0.5, 0.2, 0.2]], dtype=np.float32)
    kept, scaled, counts = transformed_targets(c, b, False, 1.15)
    assert kept.tolist() == [[1]] and counts["crop_discarded_targets"] == 1
    assert np.allclose(scaled, [[0.5, 0.5, 0.23, 0.23]], atol=2e-7)
    ignored = np.array([[20, 10, 40, 20, 0, 1, 0, 0], [2, 2, 10, 10, 1, 11, 0, 0]], dtype=float)
    cl, bx, ct = project_gt(np.concatenate([gt, ignored]), info)
    assert len(cl) == len(bx) == 1 and ct["ignored_gt_after_prepare"] == 1
    assert ct["nontarget_class_after_prepare"] == 1
    empty = np.empty((0, 8))
    cl, bx, ct = project_gt(empty, info)
    assert cl.shape == (0, 1) and bx.shape == (0, 4) and ct["valid_targets"] == 0
    assert TEACHER_IMAGES == 3288 and len({name for name, _, _ in VIEWS}) == 6
    assert [(flip, scale) for _, flip, scale in VIEWS[:2]] == [(False, 1.0), (True, 1.0)]
    assert all(not flip for _, flip, _ in VIEWS[2:])
    try:
        transformed_targets(classes, boxes, False, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("Nonpositive scale accepted")
    assert "torch" not in sys.modules
    return {
        "status": "PASS",
        "GPU_execution": False,
        "teacher_images_full_run": TEACHER_IMAGES,
        "contracts": [
            "raw GT letterbox projection",
            "flip",
            "scale clipping/keep",
            "ignore/classes/empty GT",
            "6x548 teacher accounting",
            "temporary anchors reused",
            "no torch import",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inspect", "selfcheck", "run"))
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--original-detail-job", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--archive", type=Path, default=ROOT / "datasets/VisDrone/VisDrone2019-DET-val.zip")
    args = parser.parse_args()
    if args.action == "selfcheck":
        print(json.dumps(selfcheck(), indent=2))
        return
    if not args.campaign or not args.original_detail_job:
        parser.error("--campaign and --original-detail-job required")
    manifest, cache, records, bindings, blank = validate_inputs(args)
    protocol = contract(args, manifest, cache, bindings)
    if args.action == "inspect":
        assert "torch" not in sys.modules
        print(json.dumps({"status": "READY_NOT_LAUNCHED", "GPU_execution": False, "protocol": protocol}, indent=2))
        return
    if args.output is None or args.output.exists():
        parser.error("run requires a fresh --output directory; existing evidence is never overwritten")
    args.output.mkdir(parents=True)
    dump(args.output / "protocol.json", protocol)
    (args.output / "status.txt").write_text("RUNNING geometry full548\n", encoding="utf-8")
    started = time.monotonic()
    try:
        result = run(args, manifest, cache, records, bindings, blank)
        if source_hashes() != protocol["source_sha256"]:
            raise ValueError("Diagnostic source changed during execution")
        dump(args.output / "completion.json", {"status": "SUCCEEDED", "seconds": time.monotonic() - started, **result})
        (args.output / "status.txt").write_text("SUCCEEDED geometry full548\n", encoding="utf-8")
    except Exception as error:
        dump(args.output / "failure.json", {"type": type(error).__name__, "message": str(error)})
        (args.output / "status.txt").write_text(f"FAILED {error}\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
