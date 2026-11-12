"""Real YOLO-Master-N / YOLO12-N, full VOC or SKU, fixed seed0/final100.

Uses the repository's actual detector architectures and native training loop.
Only dataset loading and held-out scoring are adapted to the sealed full-split
contract. No subset/smoke, search, seed-repeat or best-test selection is exposed.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from online_experiments import crossdataset_suite as scoring

VERSION = "direct-full-detection-v1"
MODELS = {
    "master-n": "ultralytics/cfg/models/master/v0/det/yolo-master-n.yaml",
    "yolo12-n": "yolo12n.yaml",
}
ANCHOR = ROOT / "runs/paper/crossdataset/job-20261110-voc0712-s0-v2/provenance.json"
ANCHOR_SHA = "9564e537b32f18fe51a949888a6656a047c2a02797fd9ce863247e1e0b37d932"


def read(path):
    return scoring.read(path)


def dump(path, value):
    scoring.dump(path, value)


def sha(path):
    return scoring.sha(path)


def source_snapshot():
    if sha(ANCHOR) != ANCHOR_SHA:
        raise RuntimeError("Frozen native implementation anchor changed")
    sources = dict(read(ANCHOR)["source_sha256"])
    relative = Path(__file__).relative_to(ROOT).as_posix()
    sources[relative] = sha(__file__)
    # SKU wrapper owns its tolerant JPEG policy and its complete decode audit.
    for name in (
        "online_experiments/sku_official_loader_suite.py",
        "online_experiments/sku_official_loader_campaign.py",
        "research/checks/sku-preparation-recovery-20261110/sku-full-image-decode.json",
        "research/checks/sku-preparation-recovery-20261110/sku-full-image-decode-tolerant.json",
    ):
        sources[name] = sha(ROOT / name)
    if len(sources) < 1367:
        raise RuntimeError("Incomplete original implementation snapshot")
    result = {
        "source_sha256": sources,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "policy": "Pinned original1366-file source anchor plus this completed worker only",
    }
    scoring.verify_sources(result)
    return result


def load_dataset(path, *, verify_files=False):
    dataset = read(path)["dataset"]
    if dataset == "sku110k":
        from online_experiments.sku_official_loader_suite import install

        install()
    return scoring.load_dataset(path, verify_files=verify_files)[0]


def normalized_objects(sample):
    """Original-image YOLO xywh, excluding only frozen difficult/crowd targets."""
    labels = []
    for obj in sample["objects"]:
        if obj.get("difficult", False) or obj.get("iscrowd", 0):
            continue
        x0, y0, x1, y1 = obj["bbox_xyxy0_half_open"]
        labels.append(
            [
                obj["category_id"],
                (x0 + x1) / (2 * sample["width"]),
                (y0 + y1) / (2 * sample["height"]),
                (x1 - x0) / sample["width"],
                (y1 - y0) / sample["height"],
            ]
        )
    return np.asarray(labels, dtype=np.float32).reshape(-1, 5)


def prepare(manifest, output):
    manifest, output = manifest.resolve(strict=True), output.resolve()
    meta = load_dataset(manifest, verify_files=True)
    if (output / "complete.json").exists():
        check_prepared(manifest, output)
        print("PREPARED", output, flush=True)
        return
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Incomplete preparation exists; preserve it and use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    files, label_counts = {}, {}
    for split, samples in meta["splits"].items():
        image_lines, objects = [], 0
        folder = output / "labels" / split
        folder.mkdir(parents=True)
        for index, sample in enumerate(samples):
            labels = normalized_objects(sample)
            target = folder / f"{index:06d}.txt"
            np.savetxt(target, labels, fmt=["%d", "%.10g", "%.10g", "%.10g", "%.10g"])
            files[target.relative_to(output).as_posix()] = sha(target)
            image_lines.append(sample["image_path"])
            objects += len(labels)
        target = output / f"{split}.txt"
        target.write_text("\n".join(image_lines) + "\n", encoding="utf-8")
        files[target.name] = sha(target)
        label_counts[split] = objects
    # Native trainer only needs a syntactically valid val entry; held-out test is
    # never evaluated inside training. Real full held-out evaluation follows last.pt.
    data = {
        "path": str(output),
        "train": str(output / "train.txt"),
        "val": str(output / "train.txt"),
        "test": str(output / "test.txt"),
        "nc": meta["nc"],
        "names": meta["class_names"],
    }
    import yaml

    yaml_file = output / "dataset.yaml"
    yaml_file.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    files[yaml_file.name] = sha(yaml_file)
    dump(
        output / "complete.json",
        {
            "version": VERSION,
            "dataset": meta["dataset"],
            "manifest": str(manifest),
            "manifest_sha256": sha(manifest),
            "counts": {k: len(v) for k, v in meta["splits"].items()},
            "training_objects": label_counts,
            "sha256": files,
            "policy": "Original image bytes unchanged; labels derived from sealed full manifest; no dropped images. Native dataset bypasses repair/filter/dedup scanning and reads Pillow RGB for every image.",
        },
    )
    print("PREPARED", output, flush=True)


def check_prepared(manifest, output):
    seal = read(output / "complete.json")
    if seal["version"] != VERSION or seal["manifest_sha256"] != sha(manifest):
        raise RuntimeError("Native dataset seal or original manifest mismatch")
    for name, digest in seal["sha256"].items():
        if sha(output / name) != digest:
            raise RuntimeError("Prepared dataset content changed: " + name)
    return seal


def pil_bgr(path):
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))[:, :, ::-1].copy()


def trainer_class(meta):
    """Use native YOLO transforms/loss/optimizer; retain sealed image/target set."""
    import cv2

    from ultralytics.data.dataset import YOLODataset
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.utils.torch_utils import unwrap_model

    catalog = {sample["image_path"]: sample for rows in meta["splits"].values() for sample in rows}

    class SealedDataset(YOLODataset):
        def get_img_files(self, img_path):
            values = Path(img_path).read_text(encoding="utf-8").splitlines()
            if not values or any(value not in catalog for value in values) or len(values) != len(set(values)):
                raise RuntimeError("Native loader list differs from sealed full dataset")
            return values

        def get_labels(self):
            labels = []
            self.label_files = []
            for filename in self.im_files:
                sample = catalog[filename]
                objects = normalized_objects(sample)
                labels.append(
                    {
                        "im_file": filename,
                        "shape": (sample["height"], sample["width"]),
                        "cls": objects[:, :1],
                        "bboxes": objects[:, 1:],
                        "segments": [],
                        "keypoints": None,
                        "normalized": True,
                        "bbox_format": "xywh",
                    }
                )
            return labels

        def load_image(self, i, rect_mode=True, resize_short=False):
            # Same resize/buffer algorithm as native BaseDataset, with a fixed
            # Pillow decoder instead of stale .npy cache or cv2 JPEG recovery.
            if self.ims[i] is not None:
                return self.ims[i], self.im_hw0[i], self.im_hw[i]
            image = pil_bgr(self.im_files[i])
            height, width = image.shape[:2]
            sample = catalog[self.im_files[i]]
            if (height, width) != (sample["height"], sample["width"]):
                raise RuntimeError("Original image dimension changed")
            if rect_mode:
                ratio = self.imgsz / (min(height, width) if resize_short else max(height, width))
                if ratio != 1:
                    if resize_short:
                        size = (
                            (math.ceil(width * ratio), self.imgsz)
                            if height < width
                            else (self.imgsz, math.ceil(height * ratio))
                        )
                    else:
                        size = (min(math.ceil(width * ratio), self.imgsz), min(math.ceil(height * ratio), self.imgsz))
                    image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
            elif height != self.imgsz or width != self.imgsz:
                image = cv2.resize(image, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = image, (height, width), image.shape[:2]
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length:
                    previous = self.buffer.pop(0)
                    self.ims[previous], self.im_hw0[previous], self.im_hw[previous] = None, None, None
            return image, (height, width), image.shape[:2]

    class FinalOnlyTrainer(DetectionTrainer):
        def build_dataset(self, img_path, mode="train", batch=None):
            stride = max(int(unwrap_model(self.model).stride.max()), 32)
            dataset = SealedDataset(
                img_path=img_path,
                imgsz=self.args.imgsz,
                batch_size=batch,
                augment=mode == "train",
                hyp=self.args,
                rect=False,
                cache=False,
                single_cls=False,
                stride=stride,
                pad=0.0 if mode == "train" else 0.5,
                task="detect",
                classes=None,
                data=self.data,
                fraction=1.0,
            )
            if len(dataset) != len(meta["splits"]["train"]):
                raise RuntimeError("Native dataset unexpectedly dropped original training samples")
            return dataset

        def validate(self):
            # No train/test model selection. The only actual held-out prediction
            # pass below uses the persisted fixed-final checkpoint.
            return {}, 0.0

        def final_eval(self):
            return None

    return FinalOnlyTrainer


def epoch_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [{key.strip(): value for key, value in row.items()} for row in csv.DictReader(stream)]
    epochs = [int(float(row["epoch"])) for row in rows]
    if epochs != list(range(1, len(rows) + 1)):
        raise RuntimeError("Missing/repeated/out-of-order native training epochs")
    return rows


def recipe(method, prepared, output, device):
    return {
        "data": str(prepared / "dataset.yaml"),
        "epochs": 100,
        "batch": 8,
        "imgsz": 640,
        "seed": 0,
        "device": device,
        "workers": 4,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "weight_decay": 0.0005,
        "cos_lr": True,
        "patience": 200,
        "pretrained": False,
        "amp": False,
        "deterministic": True,
        "warmup_epochs": 0,
        "nbs": 8,
        "mosaic": 0,
        "mixup": 0,
        "copy_paste": 0,
        "cutmix": 0,
        "scale": 0.15,
        "fliplr": 0.5,
        "flipud": 0,
        "degrees": 0,
        "translate": 0,
        "shear": 0,
        "perspective": 0,
        "hsv_h": 0,
        "hsv_s": 0,
        "hsv_v": 0,
        "project": str(output.parent),
        "name": output.name,
        "exist_ok": True,
        "plots": False,
        "val": False,
        "save": True,
        "cache": False,
        "fraction": 1.0,
        "rect": False,
        "close_mosaic": 0,
    }


def train(args):
    import torch

    from ultralytics import YOLO

    output = args.output.resolve()
    if (output / "DONE.json").exists():
        validate(output)
        print("SKIP_COMPLETED", output, flush=True)
        return
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Native run output already exists; preserve incomplete artifacts; no implicit restart")
    meta = load_dataset(args.dataset_manifest, verify_files=True)
    seal = check_prepared(args.dataset_manifest, args.prepared)
    snapshot = source_snapshot()
    output.mkdir(parents=True, exist_ok=True)
    dump(output / "snapshot.json", snapshot)
    config = recipe(args.method, args.prepared.resolve(), output, args.device)
    model_yaml = "yolo12n.yaml" if args.method == "yolo12-n" else str(ROOT / MODELS[args.method])
    dump(
        output / "args.json",
        {
            "version": VERSION,
            "method": args.method,
            "model_yaml": model_yaml,
            "seed": 0,
            "epochs": 100,
            "dataset": meta["dataset"],
            "manifest": str(args.dataset_manifest.resolve()),
            "manifest_sha256": sha(args.dataset_manifest),
            "prepared_seal_sha256": sha(args.prepared / "complete.json"),
            "counts": seal["counts"],
            "native_recipe": config,
            "endpoint": "Fixed last.pt after100 unique epochs; no intermediate held-out evaluation or best-test selection",
            "comparison_scope": "Real complete detector; matched full splits/epochs/optimizer settings, native architecture/head/transforms. Not a600epoch YOLO-Master paper reproduction or pixel-identical cached-head training.",
        },
    )
    dump(
        output / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(args.device),
            "gpu": torch.cuda.get_device_name(0),
            "observational_shared_gpu": True,
            "pid": os.getpid(),
        },
    )
    model = YOLO(model_yaml)
    dump(
        output / "initialization.json",
        {
            "architecture": model_yaml,
            "pretrained": False,
            "seed": 0,
            "parameters_before_dataset_class_override": sum(p.numel() for p in model.model.parameters()),
            "note": "Ultralytics deterministic trainer seeds then constructs actual dataset-nc model; native initialization, no loaded pretrained weights.",
        },
    )
    events = []

    def on_epoch(trainer):
        number = int(trainer.epoch) + 1
        if number != len(events) + 1 or int(trainer.batch_size) != 8:
            raise RuntimeError("Native epoch repetition/skip or automatic batch-size reduction")
        events.append(number)
        dump(output / "epoch-state.json", {"completed_epochs": number, "epochs": events})
        print(f"DIRECT_EPOCH method={args.method} dataset={meta['dataset']} epoch={number}/100", flush=True)

    model.add_callback("on_fit_epoch_end", on_epoch)
    started = time.perf_counter()
    model.train(trainer=trainer_class(meta), **config)
    train_seconds = time.perf_counter() - started
    rows = epoch_rows(output / "results.csv")
    if len(rows) != 100 or events != list(range(1, 101)):
        raise RuntimeError("Native training did not complete100 unique full epochs")
    last = output / "weights/last.pt"
    if not last.is_file():
        raise RuntimeError("Fixed final native detector checkpoint missing")
    del model
    torch.cuda.empty_cache()
    model = YOLO(str(last))
    predictions, coverage, seconds = [], [], []
    started_eval = time.perf_counter()
    for sample in meta["splits"]["test"]:
        tick = time.perf_counter()
        result = model.predict(
            source=pil_bgr(sample["image_path"]),
            imgsz=640,
            device=args.device,
            conf=0.001,
            iou=0.7,
            max_det=500,
            rect=False,
            augment=False,
            half=False,
            verbose=False,
        )[0]
        if tuple(result.orig_shape) != (sample["height"], sample["width"]):
            raise RuntimeError("Native final prediction original image dimensions differ")
        values = result.boxes.data.detach().cpu().numpy()
        if values.ndim != 2 or values.shape[1] != 6 or not np.isfinite(values).all() or len(values) > 500:
            raise RuntimeError("Native detector output shape/count/nonfinite error")
        for row in values[np.argsort(-values[:, 4], kind="stable")]:
            category = int(row[5])
            if category != row[5] or not 0 <= category < meta["nc"] or not 0 <= row[4] <= 1:
                raise RuntimeError("Native detector predicted invalid class or probability")
            if row[4] < 0.001 or row[2] <= row[0] or row[3] <= row[1]:
                continue
            predictions.append(
                {
                    "sample_id": sample["sample_id"],
                    "bbox_xyxy0_half_open": row[:4].tolist(),
                    "score": float(row[4]),
                    "category_id": category,
                }
            )
        coverage.append(sample["sample_id"])
        seconds.append(time.perf_counter() - tick)
        if len(coverage) % 100 == 0:
            print(f"FINAL_EVAL {args.method} images={len(coverage)}/{len(meta['splits']['test'])}", flush=True)
    dump(
        output / "predictions.json",
        {
            "version": VERSION,
            "coverage": coverage,
            "predictions": predictions,
            "coordinates": "zero-based half-open original image pixels",
            "confidence": 0.001,
            "max_det": 500,
            "nms_iou": 0.7,
            "native_prediction": True,
        },
    )
    if meta["dataset"] == "voc0712":
        primary = scoring.evaluate_voc07(meta["splits"]["test"], predictions, meta["class_names"])
    else:
        primary = scoring.evaluate_coco(meta["splits"]["test"], predictions, meta["class_names"], output)
    metrics = {
        "primary": primary,
        "endpoint": "Full official test split; fixed final100epoch; same frozen scorer as cache arms",
    }
    dump(output / "metrics.json", metrics)
    dump(
        output / "summary.json",
        {
            "version": VERSION,
            "method": args.method,
            "dataset": meta["dataset"],
            "seed": 0,
            "completed_epochs": 100,
            "evaluated_images": len(coverage),
            "training_wall_seconds": train_seconds,
            "evaluation_wall_seconds": time.perf_counter() - started_eval,
            "observed_mean_predict_wall_ms": 1000 * float(np.mean(seconds)),
            "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "cache_bytes": 0,
            "training_cache_build_seconds": 0,
            "teacher_query_images": 0,
            "native_backbone_training_images": 100 * len(meta["splits"]["train"]),
            "resource_note": "SharedGPU observational wall times, not controlled speed benchmark. Native detector backbones train; pretrained foundation-backbone queries do not apply.",
        },
    )
    scoring.verify_sources(snapshot)
    endpoint_files = [
        "args.json",
        "environment.json",
        "initialization.json",
        "snapshot.json",
        "summary.json",
        "metrics.json",
        "predictions.json",
        "results.csv",
        "epoch-state.json",
        "weights/last.pt",
    ]
    dump(
        output / "endpoint.json", {"version": VERSION, "sha256": {name: sha(output / name) for name in endpoint_files}}
    )
    dump(
        output / "DONE.json",
        {"version": VERSION, "endpoint_sha256": sha(output / "endpoint.json"), "completed_epochs": 100},
    )
    validate(output)
    print("SUCCEEDED", output, flush=True)


def validate(output):
    done, endpoint = read(output / "DONE.json"), read(output / "endpoint.json")
    if done["version"] != VERSION or done["endpoint_sha256"] != sha(output / "endpoint.json"):
        raise RuntimeError("Native completion endpoint seal mismatch")
    for name, digest in endpoint["sha256"].items():
        if sha(output / name) != digest:
            raise RuntimeError("Native endpoint artifact changed: " + name)
    config, summary, predictions = (
        read(output / "args.json"),
        read(output / "summary.json"),
        read(output / "predictions.json"),
    )
    if config["method"] not in MODELS or config["seed"] != 0 or summary["completed_epochs"] != 100:
        raise RuntimeError("Native run method/seed/epoch mismatch")
    if sha(config["manifest"]) != config["manifest_sha256"]:
        raise RuntimeError("Native original dataset manifest changed")
    meta = read(config["manifest"])
    if predictions["coverage"] != [sample["sample_id"] for sample in meta["splits"]["test"]]:
        raise RuntimeError("Native final predictions omit/reorder test images")
    if len(epoch_rows(output / "results.csv")) != 100:
        raise RuntimeError("Native training does not contain100 unique epochs")
    scoring.verify_sources(read(output / "snapshot.json"))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "train", "validate"))
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--method", choices=tuple(MODELS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    if args.action == "validate":
        validate(args.output.resolve())
        print("VALIDATED", args.output, flush=True)
    elif args.dataset_manifest is None:
        parser.error("prepare/train requires --dataset-manifest")
    elif args.action == "prepare":
        prepare(args.dataset_manifest, args.output)
    elif args.method is None or args.prepared is None:
        parser.error("train requires --method and --prepared")
    else:
        train(args)


if __name__ == "__main__":
    main()
