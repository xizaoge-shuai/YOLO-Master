"""Frozen, held-out feature/task diagnostics for the completed full VisDrone corrector."""

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch


def mask_geometry(boxes, flip, scale):
    """Transform mask boxes, retaining every positive-area visible fragment."""
    lo = ((boxes[:, :2] - boxes[:, 2:] / 2 - 0.5) * scale + 0.5).clamp(0, 1)
    hi = ((boxes[:, :2] + boxes[:, 2:] / 2 - 0.5) * scale + 0.5).clamp(0, 1)
    size = hi - lo
    result = torch.cat(((hi + lo) / 2, size), dim=1)[(size > 0).all(dim=1)]
    if flip:
        result[:, 0] = 1 - result[:, 0]
    return result


def box_cell_weights(boxes, height, width):
    """Maximum per-box fractional cell overlap; preserves tiny objects, not exact union area."""
    boxes = boxes.reshape(-1, 4)
    out = boxes.new_zeros(height, width)
    if not len(boxes):
        return out
    lo = (boxes[:, :2] - boxes[:, 2:] / 2).clamp(0, 1)
    hi = (boxes[:, :2] + boxes[:, 2:] / 2).clamp(0, 1)
    x = torch.arange(width, device=boxes.device, dtype=boxes.dtype) / width
    y = torch.arange(height, device=boxes.device, dtype=boxes.dtype) / height
    ox = (torch.minimum(hi[:, 0, None], x[None] + 1 / width) - torch.maximum(lo[:, 0, None], x[None])).clamp_min(
        0
    ) * width
    oy = (torch.minimum(hi[:, 1, None], y[None] + 1 / height) - torch.maximum(lo[:, 1, None], y[None])).clamp_min(
        0
    ) * height
    return (oy[:, :, None] * ox[:, None, :]).amax(0).clamp(0, 1)


def region_relative_mse(prediction, target, weight):
    """Weighted residual energy / target energy; undefined empty regions remain missing."""
    if float(weight.sum()) <= 1e-8:
        return None
    numerator = ((prediction.float() - target.float()).square().mean(0) * weight).sum()
    denominator = (target.float().square().mean(0) * weight).sum().clamp_min(1e-8)
    return float(numerator / denominator)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import numpy as np

    from online_experiments.d1_online_compare import geometry, training_feature
    from online_experiments.feature_baselines import warp_features
    from online_experiments.full_visdrone_data import dump, read, sha, verify_sources
    from online_experiments.full_visdrone_train import Batches
    from online_experiments.fusion_probe import configure_fusion
    from scripts import d1_train_cached_detector as d
    from ultralytics.data.foundation_cache import load_letterboxed_tensor

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise RuntimeError("Completed diagnostic output exists; preserve it")

    def status(message):
        (output / "status.txt").write_text(message + "\n")
        print(message, flush=True)

    try:
        status("RUNNING load frozen final checkpoints")
        if (args.source_job / "status.txt").read_text().strip() != "SUCCEEDED":
            raise ValueError("Full source campaign is not complete")
        verify_sources(read(args.source_job / "provenance.json"))
        torch.set_num_threads(4)
        d.seed_everything(0)
        batches = Batches(args.source_job, "correct25", 0)
        if len(batches.val) != 548:
            raise ValueError("Held-out diagnostic must use all 548 validation images")
        correct_path = args.source_job / "train/correct25-s0/weights/last.pt"
        head_path = args.source_job / "train/online-geom-s0/weights/last.pt"
        correct_state = torch.load(correct_path, map_location="cpu", weights_only=False)
        head_state = torch.load(head_path, map_location="cpu", weights_only=False)
        if correct_state["epoch"] != 100 or head_state["epoch"] != 100:
            raise ValueError("Expected final 100-epoch checkpoints")
        batches.corrector.load_state_dict(correct_state["corrector"], strict=True)
        model = d.CachedLatentDetector(in_channels=(384,) * 3, nc=10, imgsz=640, epochs=100).cuda()
        configure_fusion(model, "mean_value")
        model.detect.max_det = 500
        model.load_state_dict(head_state["model"], strict=True)
        del correct_state, head_state
        model.eval().requires_grad_(False)
        batches.corrector.eval().requires_grad_(False)
        batches.extractor.model.eval().requires_grad_(False)
        criterion = d.E2EDetectLoss(model)
        source_hash = sha(__file__)
        provenance = {
            "source_job": str(args.source_job),
            "seed": 0,
            "images": 548,
            "batch": args.batch,
            "loss_aggregation": "image-count-weighted mean of batch target-score-normalized E2EDetectLoss components; identical batches and order across methods",
            "head_method": "online-geom",
            "head_checkpoint": "last epoch100",
            "head_sha256": sha(head_path),
            "corrector_sha256": sha(correct_path),
            "source_sha256": source_hash,
            "manifest_sha256": sha(args.source_job / "dataset-manifest.json"),
            "mask": "max per-box fractional cell overlap; background is valid image-content weight minus foreground",
            "mask_boxes": "all positive-area visible transformed YOLO boxes, independent of the >=1 pixel/10% retained-area detection-loss filter",
            "layer_aggregation": "mean of per-layer relative errors within each image; then equal image weights",
            "targets": "same frozen YOLO-format validation targets as the training diagnostic; not official AP",
            "training_updates": 0,
            "test_dev_used": False,
            "transform_cases": [[flip, scale] for flip in [False, True] for scale in [1.0, 0.85, 1.15, 0.75, 1.25]],
        }
        dump(output / "provenance.json", provenance)
        groups, losses = defaultdict(list), defaultdict(list)
        fields = [
            "image_id",
            "flip",
            "scale",
            "method",
            "relative_mse",
            "foreground_relative_mse",
            "background_relative_mse",
        ]
        teacher_images = 0
        parity_max = 0.0
        started = time.perf_counter()
        with (output / "per-image.csv").open("w", newline="") as stream, torch.no_grad():
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for start in range(0, 548, args.batch):
                ids = list(range(start, min(start + args.batch, 548)))
                samples = [batches.val[i] for i in ids]
                images, raw_items = [], []
                for sample in samples:
                    if sha(sample["image_path"]) != sample["image_sha256"]:
                        raise ValueError("Validation image differs from source manifest")
                    image, info = load_letterboxed_tensor(sample["image_path"], imgsz=640)
                    if info != sample["letterbox"]:
                        raise ValueError("Letterbox changed")
                    images.append(image)
                    raw_items.append(batches.item(sample))
                cache = torch.from_numpy(np.array(batches.val_array[ids], copy=True)).cuda().float()
                anchors = {False: tuple(cache[:, layer] for layer in range(3))}
                for flip in [False, True]:
                    for scale in [1.0, 0.85, 1.15, 0.75, 1.25]:
                        transformed, items, contents, mask_boxes = [], [], [], []
                        for sample, image, raw in zip(samples, images, raw_items):
                            im, cls, boxes = geometry(image, raw["cls"], raw["bboxes"], flip, scale)
                            transformed.append(im)
                            items.append({**raw, "cls": cls, "bboxes": boxes})
                            mask_boxes.append(mask_geometry(raw["bboxes"], flip, scale))
                            rh, rw = sample["letterbox"]["resized_shape"]
                            left, top = sample["letterbox"]["pad"]
                            content = boxes.new_tensor(
                                [[(left + rw / 2) / 640, (top + rh / 2) / 640, rw / 640, rh / 640]]
                            )
                            content = mask_geometry(content, flip, scale)
                            contents.append(content)
                        extracted = batches.extractor(torch.stack(transformed))
                        exact = tuple(
                            training_feature(extracted[f"layer_{layer}"])
                            for layer in batches.meta["identity"]["layers"]
                        )
                        teacher_images += len(ids)
                        if scale == 1.0:
                            if flip:
                                anchors[True] = exact
                            else:
                                for cached, fresh in zip(anchors[False], exact):
                                    difference = (
                                        (cached - fresh).square().flatten(1).sum(1)
                                        / fresh.square().flatten(1).sum(1).clamp_min(1e-8)
                                    ).sqrt()
                                    parity_max = max(parity_max, float(difference.max()))
                                if parity_max > 0.01:
                                    raise ValueError(f"Cache identity failure: {parity_max}")
                        transported = tuple(
                            warp_features(x, [False] * len(ids), [scale] * len(ids)) for x in anchors[flip]
                        )
                        corrected = batches.corrector(transported, [scale] * len(ids), [flip] * len(ids))
                        for image_index, item in enumerate(items):
                            item["features"] = tuple(x[image_index] for x in exact)
                        _, targets = d.move_batch(d.collate_cached(items), torch.device("cuda:0"))
                        for method, levels in [("Exact", exact), ("Transport", transported), ("Rec", corrected)]:
                            _, components = criterion(model.training_predictions(levels), targets)
                            loss = float(components.sum())
                            if not np.isfinite(loss):
                                raise ValueError("Nonfinite frozen-head loss")
                            losses[(flip, scale, method)].append((len(ids), loss))
                            if method == "Exact":
                                continue
                            for image_index, sample in enumerate(samples):
                                by_region = defaultdict(list)
                                for predicted, target in zip(levels, exact):
                                    h, w = target.shape[-2:]
                                    fg = box_cell_weights(mask_boxes[image_index].cuda(), h, w)
                                    content = box_cell_weights(contents[image_index].cuda(), h, w)
                                    fg = torch.minimum(fg, content)
                                    for name, weight in [
                                        ("relative_mse", content),
                                        ("foreground_relative_mse", fg),
                                        ("background_relative_mse", (content - fg).clamp_min(0)),
                                    ]:
                                        value = region_relative_mse(predicted[image_index], target[image_index], weight)
                                        if value is not None:
                                            if not np.isfinite(value):
                                                raise ValueError("Nonfinite feature diagnostic")
                                            by_region[name].append(value)
                                row = {"image_id": sample["sample_id"], "flip": flip, "scale": scale, "method": method}
                                row.update(
                                    {
                                        name: float(np.mean(by_region[name])) if by_region[name] else None
                                        for name in fields[4:]
                                    }
                                )
                                writer.writerow(row)
                                groups[(flip, scale, method)].append(row)
                stream.flush()
                status(f"RUNNING held-out images={start + len(ids)}/548 teacher_images={teacher_images}")
        summary = []
        for (flip, scale, method), rows in groups.items():
            exact_loss = sum(n * x for n, x in losses[(flip, scale, "Exact")]) / 548
            method_loss = sum(n * x for n, x in losses[(flip, scale, method)]) / 548
            row = {
                "flip": flip,
                "scale": scale,
                "method": method,
                "images": len(rows),
                "frozen_head_loss": method_loss,
                "exact_head_loss": exact_loss,
                "head_loss_ratio": method_loss / max(exact_loss, 1e-8),
            }
            for name in fields[4:]:
                values = [r[name] for r in rows if r[name] is not None]
                row[name] = float(np.mean(values)) if values else None
                row[f"{name}_n"] = len(values)
            summary.append(row)
        if teacher_images != 5480 or len(summary) != 20 or sha(__file__) != source_hash:
            raise ValueError("Diagnostic coverage or source mismatch")
        if any(
            p.grad is not None
            for module in [model, batches.corrector, batches.extractor.model]
            for p in module.parameters()
        ):
            raise ValueError("Unexpected gradient in frozen diagnostic")
        if sha(head_path) != provenance["head_sha256"] or sha(correct_path) != provenance["corrector_sha256"]:
            raise ValueError("Source checkpoints changed")
        dump(
            output / "summary.json",
            {
                "rows": summary,
                "teacher_images": teacher_images,
                "cache_parity_max": parity_max,
                "seconds": time.perf_counter() - started,
                "training_updates": 0,
                "per_image_sha256": sha(output / "per-image.csv"),
            },
        )
        lines = [
            "Single-seed held-out diagnostic; no weight updates; loss ratio uses ONE frozen Online head.",
            "flip scale method foreground_relMSE background_relMSE loss_ratio",
        ]
        lines += [
            f"{int(r['flip'])} {r['scale']:.2f} {r['method']} {r['foreground_relative_mse']:.6f} "
            f"{r['background_relative_mse']:.6f} {r['head_loss_ratio']:.4f}"
            for r in summary
        ]
        (output / "comparison.txt").write_text("\n".join(lines) + "\n")
        status("SUCCEEDED")
    except BaseException as exc:
        status(f"FAILED {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
