#!/usr/bin/env python3
"""Compare cached and online-frozen DINO training using the existing D1 detector."""
import argparse
import csv
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def geometry(image, classes, boxes, flip, scale):
    """Scale image content around center, then flip; transform labels together."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    boxes = boxes.clone()
    if scale != 1.0:
        theta = image.new_tensor([[[1 / scale, 0, 0], [0, 1 / scale, 0]]])
        grid = F.affine_grid(theta, (1, *image.shape), align_corners=False)
        fill = 114 / 255
        image = F.grid_sample(
            (image - fill)[None], grid, align_corners=False
        )[0] + fill

        if boxes.numel():
            lo = (boxes[:, :2] - boxes[:, 2:] / 2 - 0.5) * scale + 0.5
            hi = (boxes[:, :2] + boxes[:, 2:] / 2 - 0.5) * scale + 0.5
            before = (hi - lo).prod(dim=1)
            lo, hi = lo.clamp(0, 1), hi.clamp(0, 1)
            size = hi - lo

            keep = (size * image.shape[-1] >= 1).all(dim=1)
            keep &= size.prod(dim=1) / before.clamp_min(1e-12) >= 0.1

            boxes = torch.cat(((hi + lo) / 2, size), dim=1)[keep]
            classes = classes[keep]

    if flip:
        image = image.flip(-1)
        boxes[:, 0] = 1 - boxes[:, 0]

    return image.clamp(0.0, 1.0), classes, boxes


def training_feature(feature):
    """Match FP16 cache interface and allow detector backward."""
    with torch.inference_mode(False):
        return feature.detach().to(torch.float16).to(torch.float32).clone()


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--mode", choices=["offline", "online"], required=True)
    p.add_argument("--augment", choices=["none", "hflip", "geom"], default="none")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--train-count", type=int, default=400)
    p.add_argument("--val-count", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--teacher-dtype", choices=["fp32", "fp16", "bf16"], default="fp32"
    )
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--aux-weight", type=float, default=0.0)
    p.add_argument("--scale-min", type=float, default=0.85)
    p.add_argument("--scale-max", type=float, default=1.15)
    p.add_argument("--parity-tolerance", type=float, default=0.02)
    p.add_argument("--parity-only", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main():
    a = arguments()
    if min(a.epochs, a.batch, a.train_count, a.val_count) < 1:
        raise ValueError("epochs, batch and split sizes must be positive")
    if not 0 < a.scale_min <= a.scale_max:
        raise ValueError("invalid scale interval")
    if a.mode == "offline" and (a.augment != "none" or a.parity_only):
        raise ValueError("offline supports only unaugmented cache training")

    root = Path.cwd()
    sys.path.insert(0, str(root))

    from scripts import d1_train_cached_detector as d
    from scripts.d1_build_feature_cache import DINOv3MultiLevelExtractor
    from ultralytics.data.foundation_cache import (
        load_letterboxed_tensor, file_sha256,
    )

    device = torch.device(a.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this GPU cost experiment requires CUDA")
    if a.output.exists() and any(a.output.iterdir()):
        raise FileExistsError(f"nonempty output: {a.output}")

    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "weights").mkdir(exist_ok=True)
    wall_start = time.perf_counter()

    def sync():
        torch.cuda.synchronize(device)

    def dump(name, value):
        d.atomic_write_json(a.output / name, value)

    manifest = json.loads((a.cache / "manifest.json").read_text())
    d.validate_manifest(manifest, cache_root=a.cache, verify_files=False)
    ident = manifest["identity"]
    layers = tuple(int(x) for x in ident["layers"])
    imgsz = int(ident["imgsz"])

    data = d.check_det_dataset(a.dataset, autodownload=False)
    names = data["names"]
    names = (
        {int(k): str(v) for k, v in names.items()}
        if isinstance(names, dict) else dict(enumerate(names))
    )

    train_ids, val_ids = d.split_sample_indices(
        len(manifest["samples"]),
        a.train_count, a.val_count, a.split_seed,
    )
    train = d.CachedDetectionDataset(a.cache, manifest, train_ids, len(names))
    val = d.CachedDetectionDataset(a.cache, manifest, val_ids, len(names))

    dump("split.json", {
        "split_seed": a.split_seed,
        "train_sample_ids": [s["sample_id"] for s in train.samples],
        "val_sample_ids": [s["sample_id"] for s in val.samples],
    })

    cmd = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    )
    config = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(a).items()
    }
    config.update(
        cache_identity=ident,
        git_commit=cmd.stdout.strip(),
        detector_fusion="unchanged router_only",
        validation="unaugmented cached features for EVERY mode",
        teacher_forward_dtype=a.teacher_dtype if a.mode == "online" else None,
        feature_interface="FP16 quantized, FP32 detector; detector AMP disabled",
        evaluator="existing D1 COCO-style evaluator, not pycocotools",
        source_hashes={
            p: file_sha256(root / p)
            for p in [
                "scripts/d1_train_cached_detector.py",
                "scripts/d1_build_feature_cache.py",
                "ultralytics/nn/modules/latent_mixture.py",
            ]
        },
    )
    dump("args.json", config)

    extractor = None
    if a.mode == "online":
        extractor = DINOv3MultiLevelExtractor(
            ident["model_id"],
            revision=ident["revision"],
            layers=layers,
            device=str(device),
            dtype=a.teacher_dtype,
            local_files_only=True,
        )

        parity = []
        for index in range(min(4, len(train))):
            sample = train.samples[index]
            path = train.dataset_root / sample["image_path"]

            if file_sha256(path) != sample["image_sha256"]:
                raise RuntimeError(f"image changed since cache creation: {path}")

            image, metadata = load_letterboxed_tensor(path, imgsz=imgsz)
            if metadata != sample["letterbox"]:
                raise RuntimeError(f"letterbox changed: {path}")

            actual = extractor(image[None])
            cached = train[index]["features"]

            for level, layer in enumerate(layers):
                actual_level = training_feature(actual[f"layer_{layer}"])[0]
                reference = cached[level].to(device).float()
                error = float(
                    (actual_level - reference).norm()
                    / reference.norm().clamp_min(1e-8)
                )
                parity.append({
                    "sample_id": sample["sample_id"],
                    "layer": layer,
                    "relative_l2": error,
                })

        dump("parity.json", parity)
        maximum = max(r["relative_l2"] for r in parity)
        print(
            f"Online/cache identity check: maximum relative L2 = {maximum:.6g}",
            flush=True,
        )
        if not math.isfinite(maximum) or maximum > a.parity_tolerance:
            raise RuntimeError(
                "online/cache mismatch; inspect parity.json and original "
                "extraction dtype before training"
            )

        del actual, cached, actual_level, reference, image
        if a.parity_only:
            print("PARITY PASS; no detector training performed", flush=True)
            return

    # Reset after teacher loading so detector initialization is identical.
    d.seed_everything(a.seed)
    in_channels = tuple(
        int(manifest["samples"][0]["shapes"][f"layer_{layer}"][0])
        for layer in layers
    )
    model = d.CachedLatentDetector(
        in_channels=in_channels,
        nc=len(names),
        imgsz=imgsz,
        epochs=a.epochs,
    ).to(device)

    audit = d.audit_model(model)
    dump("trainability.json", {
        "detector": audit,
        "teacher_loaded": extractor is not None,
        "teacher_trainable_parameters": 0,
    })

    criterion = d.E2EDetectLoss(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=a.lr, weight_decay=a.weight_decay
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=a.epochs, eta_min=a.lr * 0.01
    )

    def bank(dataset):
        items = []
        for index in range(len(dataset)):
            item = dataset[index]
            item["features"] = tuple(x.to(device) for x in item["features"])
            items.append(item)
        return items

    # Same cached validation for all experiments.
    val_bank = bank(val)
    train_bank = bank(train) if a.mode == "offline" else None
    generator = torch.Generator().manual_seed(a.seed)
    aug_rng = random.Random(a.seed + 104729)

    class Batches:
        def __init__(self, training):
            self.training = training

        def __iter__(self):
            ds = train if self.training else val
            indices = (
                torch.randperm(len(ds), generator=generator).tolist()
                if self.training else list(range(len(ds)))
            )
            for start in range(0, len(indices), a.batch):
                ids = indices[start:start + a.batch]
                source = train_bank if self.training else val_bank

                if source is not None:
                    yield d.collate_cached([source[i] for i in ids])
                    continue

                images, items = [], []
                for i in ids:
                    sample = ds.samples[i]
                    image, metadata = load_letterboxed_tensor(
                        ds.dataset_root / sample["image_path"], imgsz=imgsz
                    )
                    if metadata != sample["letterbox"]:
                        raise RuntimeError("image geometry differs from manifest")

                    cls, boxes = d.read_yolo_labels(
                        d.label_path(ds.dataset_root, sample["image_path"]),
                        metadata, imgsz, len(names),
                    )
                    flip = a.augment != "none" and aug_rng.random() < 0.5
                    scale = (
                        aug_rng.uniform(a.scale_min, a.scale_max)
                        if a.augment == "geom" else 1.0
                    )
                    image, cls, boxes = geometry(image, cls, boxes, flip, scale)
                    images.append(image)
                    items.append({
                        "cls": cls,
                        "bboxes": boxes,
                        "sample_id": sample["sample_id"],
                        "image_path": sample["image_path"],
                    })

                result = extractor(torch.stack(images))
                features = tuple(
                    training_feature(result[f"layer_{layer}"])
                    for layer in layers
                )
                for i, item in enumerate(items):
                    item["features"] = tuple(x[i] for x in features)

                yield d.collate_cached(items)

    sync()
    setup_seconds = time.perf_counter() - wall_start
    torch.cuda.reset_peak_memory_stats(device)
    best = -math.inf
    total_train = total_val = 0.0
    fields = [
        "epoch", "train_loss", "val_loss", "map50", "map50_95",
        "train_seconds", "val_seconds", "peak_vram_gib", "lr",
    ]

    with (a.output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()

        for epoch in range(1, a.epochs + 1):
            model.train()
            sync()
            start = time.perf_counter()
            train_loss = 0.0

            for batch in Batches(True):
                features, targets = d.move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                predictions = model.training_predictions(features)
                loss_vector, _ = criterion(predictions, targets)
                aux = d._collect_mixture_aux_loss(
                    model, device, latent_gain=1.0, aux_budget=3.0
                )
                loss = loss_vector.sum() + a.aux_weight * aux

                if not torch.isfinite(loss):
                    raise RuntimeError(f"nonfinite loss at epoch {epoch}")

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
                optimizer.step()
                train_loss += float(loss.detach())

            sync()
            train_seconds = time.perf_counter() - start

            if extractor is not None and any(
                p.requires_grad or p.grad is not None
                for p in extractor.model.parameters()
            ):
                raise RuntimeError("frozen teacher gradient contract violated")

            start = time.perf_counter()
            metrics, val_loss = d.validate(
                model, Batches(False), criterion, device, names, 0.001
            )
            sync()
            val_seconds = time.perf_counter() - start
            total_train += train_seconds
            total_val += val_seconds
            ap = float(metrics["metrics/mAP50-95(B)"])

            row = dict(
                epoch=epoch,
                train_loss=train_loss / len(train),
                val_loss=sum(val_loss),
                map50=float(metrics["metrics/mAP50(B)"]),
                map50_95=ap,
                train_seconds=train_seconds,
                val_seconds=val_seconds,
                peak_vram_gib=torch.cuda.max_memory_allocated(device) / 2**30,
                lr=optimizer.param_groups[0]["lr"],
            )
            if not all(math.isfinite(float(v)) for v in row.values()):
                raise RuntimeError(f"nonfinite metric: {row}")

            writer.writerow(row)
            stream.flush()

            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "metrics": row,
                "teacher_loaded_during_training": extractor is not None,
                "teacher_in_checkpoint": False,
                "cache_identity": ident,
            }
            torch.save(checkpoint, a.output / "weights" / "last.pt")
            if ap >= best:
                best = ap
                torch.save(checkpoint, a.output / "weights" / "best.pt")

            schedule.step()
            sync()

            summary = dict(
                mode=a.mode,
                augment=a.augment,
                seed=a.seed,
                completed_epochs=epoch,
                best_map50_95=best,
                final_map50_95=ap,
                train_seconds=total_train,
                validation_seconds=total_val,
                setup_seconds=setup_seconds,
                wall_seconds=time.perf_counter() - wall_start,
                gpu_hours_training=total_train / 3600,
                peak_vram_gib=row["peak_vram_gib"],
                initial_cache_build_included=False,
                validation_uses_cache=True,
                note="Matched-head pilot; not an official full-dataset benchmark",
            )
            dump("summary.json", summary)
            print(json.dumps(row), flush=True)

    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
