"""Six matched-head full-split arms. AP is diagnostic, not official VisDrone AP."""

import argparse
import csv
import json
import math
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from online_experiments.budget_corrector import ConditionalCorrector, correction_loss
from online_experiments.d1_online_compare import geometry, training_feature
from online_experiments.feature_baselines import transform_targets, warp_features
from online_experiments.full_visdrone_data import (
    NAMES,
    dump,
    epoch_plan,
    inverse_letterbox,
    make_extractor,
    read,
    read_feature_batch,
    sha,
    verify_sources,
)
from online_experiments.fusion_probe import configure_fusion
from scripts import d1_train_cached_detector as d
from ultralytics.data.foundation_cache import load_letterboxed_tensor

METHODS = ("cache1", "cache2", "mix25", "transport25", "correct25", "online-geom")


def sync():
    torch.cuda.synchronize()


def atomic_checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def publish_best(output, epoch):
    """Repair convenient pointers from the committed checkpoint's best epoch."""
    source = output / "best-artifacts" / f"{epoch:03d}"
    if not (source / "checkpoint.pt").is_file() or not (source / "predictions").is_dir():
        raise RuntimeError("Committed best epoch artifacts are incomplete")
    for link, target in (
        (output / "weights/best.pt", source / "checkpoint.pt"),
        (output / "predictions/best", source / "predictions"),
    ):
        link.parent.mkdir(parents=True, exist_ok=True)
        temporary = link.with_name(link.name + ".link-tmp")
        if temporary.is_symlink():
            temporary.unlink()
        temporary.symlink_to(target, target_is_directory=target.is_dir())
        temporary.replace(link)


class Batches:
    def __init__(self, job, method, seed):
        self.job, self.method, self.seed = job, method, seed
        self.meta = read(job / "dataset-manifest.json")
        self.cache = read(job / "cache/manifest.json")
        if sha(job / "dataset-manifest.json") != self.cache["dataset_manifest_sha256"]:
            raise RuntimeError("Dataset manifest changed after caching")
        self.train = self.meta["splits"]["train"]
        self.val = self.meta["splits"]["val"]
        self.pct = 100 if method == "online-geom" else 25 if method.endswith("25") else 0
        self.arrays = (
            []
            if method == "online-geom"
            else [np.load(job / f"cache/train-v{v}.npy", mmap_mode="r") for v in range(1 if method == "cache1" else 2)]
        )
        self.val_array = np.load(job / "cache/val-v0.npy", mmap_mode="r")
        self.extractor = make_extractor(self.meta["identity"]) if self.pct else None
        self.corrector = ConditionalCorrector(384, 64, seed).cuda() if method == "correct25" else None
        self.optimizer = (
            torch.optim.AdamW(self.corrector.parameters(), lr=0.001, weight_decay=1e-4) if self.corrector else None
        )
        self.query_images = self.query_calls = self.correction_steps = 0
        self.correction_seconds = 0.0
        self.reconstruction = []

    def item(self, sample, flip=False, scale=1.0):
        cls = torch.tensor(sample["cls"], dtype=torch.float32).reshape(-1, 1)
        boxes = torch.tensor(sample["bboxes"], dtype=torch.float32).reshape(-1, 4)
        cls, boxes = transform_targets(cls, boxes, flip, scale, 640)
        return {"cls": cls, "bboxes": boxes, "sample_id": sample["sample_id"], "image_path": sample["image_path"]}

    def training(self, epoch, batch_size, output):
        order, all_flips, all_scales, all_queries = epoch_plan(len(self.train), self.pct, epoch, self.seed)
        dump(
            output / "schedules" / f"{epoch + 1:03d}.json",
            {"epoch": epoch + 1, "order": order, "flip": all_flips, "scale": all_scales, "query": all_queries},
        )
        self.reconstruction = []
        for start in range(0, len(order), batch_size):
            ids = order[start : start + batch_size]
            flips = all_flips[start : start + batch_size] if self.method != "cache1" else [False] * len(ids)
            scales = all_scales[start : start + batch_size] if self.pct else [1.0] * len(ids)
            mask = all_queries[start : start + batch_size]
            queried = [i for i, selected in enumerate(mask) if selected]
            effective_scales = [s if q or self.method != "mix25" else 1.0 for s, q in zip(scales, mask)]
            items = [self.item(self.train[i], f, s) for i, f, s in zip(ids, flips, effective_scales)]
            levels = None
            transported = None
            if self.arrays:
                values = torch.from_numpy(read_feature_batch(self.arrays, ids, [int(f) for f in flips])).cuda().float()
                levels = tuple(values[:, l] for l in range(3))
                if self.method in ("transport25", "correct25"):
                    transported = tuple(warp_features(x, [False] * len(ids), scales) for x in levels)
                    levels = transported
            if queried:
                images = []
                for j in queried:
                    sample = self.train[ids[j]]
                    image, info = load_letterboxed_tensor(sample["image_path"], imgsz=640)
                    if info != sample["letterbox"]:
                        raise RuntimeError("Online/cache geometry mismatch")
                    raw = self.item(sample)
                    image, cls, boxes = geometry(image, raw["cls"], raw["bboxes"], flips[j], scales[j])
                    if not torch.equal(cls, items[j]["cls"]) or not torch.allclose(
                        boxes, items[j]["bboxes"], atol=2e-6
                    ):
                        raise RuntimeError("Online/cache target mismatch")
                    images.append(image)
                extracted = self.extractor(torch.stack(images))
                exact = tuple(training_feature(extracted[f"layer_{l}"]) for l in self.meta["identity"]["layers"])
                if not all(torch.isfinite(x).all() for x in exact):
                    raise RuntimeError("Nonfinite exact feature")
                self.query_calls += 1
                self.query_images += len(queried)
                if self.corrector is not None:
                    sync()
                    tick = time.perf_counter()
                    self.optimizer.zero_grad(set_to_none=True)
                    prediction = self.corrector(
                        tuple(x[queried] for x in transported),
                        [scales[j] for j in queried],
                        [flips[j] for j in queried],
                    )
                    loss = correction_loss(prediction, exact)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite correction loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.corrector.parameters(), 10.0)
                    self.optimizer.step()
                    self.reconstruction.append(float(loss.detach()))
                    self.correction_steps += 1
                    sync()
                    self.correction_seconds += time.perf_counter() - tick
                if self.method == "online-geom":
                    levels = exact
            if self.corrector is not None:
                with torch.no_grad():
                    levels = tuple(x.detach() for x in self.corrector(transported, scales, flips))
            if queried and self.method != "online-geom":
                levels = tuple(x.clone() for x in levels)
                for level, teacher in zip(levels, exact):
                    level[queried] = teacher
            if self.corrector is not None and any(x.requires_grad for x in levels):
                raise RuntimeError("Detector must not backpropagate through reconstruction-only corrector")
            for j, item in enumerate(items):
                item["features"] = tuple(x[j] for x in levels)
            yield d.collate_cached(items)

    def validation(self, batch_size):
        for start in range(0, len(self.val), batch_size):
            ids = list(range(start, min(start + batch_size, len(self.val))))
            data = torch.from_numpy(read_feature_batch([self.val_array], ids, [0] * len(ids)))
            items = [self.item(self.val[i]) for i in ids]
            for j, item in enumerate(items):
                item["features"] = tuple(data[j, l] for l in range(3))
            yield d.collate_cached(items)


@torch.no_grad()
def validate(model, batches, criterion, batch_size):
    model.eval()
    metrics = d.DetMetrics(names=dict(enumerate(NAMES)))
    total_loss = torch.zeros(3, device="cuda:0")
    count = 0
    exported = {}
    for batch in batches.validation(batch_size):
        features, targets = d.move_batch(batch, torch.device("cuda:0"))
        _, losses = criterion(model.training_predictions(features), targets)
        total_loss += losses * len(batch["sample_ids"])
        decoded, _ = model(features)
        for j, (classes, boxes) in enumerate(batch["targets"]):
            predictions = decoded[j]
            predictions = predictions[predictions[:, 4] >= 0.001]
            correct = d.match_predictions(predictions, classes.view(-1).cuda(), d.xywh2xyxy(boxes.cuda() * 640))
            metrics.update_stats(
                {
                    "tp": correct.cpu().numpy(),
                    "conf": predictions[:, 4].float().cpu().numpy(),
                    "pred_cls": predictions[:, 5].float().cpu().numpy(),
                    "target_cls": classes.view(-1).numpy(),
                    "target_img": np.full(len(classes), count),
                    "im_name": batch["image_paths"][j],
                }
            )
            sample = batches.val[count]
            exported[Path(sample["image_path"]).stem] = inverse_letterbox(
                predictions.cpu().numpy(), sample["letterbox"]
            )
            count += 1
    metrics.process(plot=False)
    results = {k: float(v) for k, v in metrics.results_dict.items() if k in metrics.keys}
    if count != len(batches.val):
        raise RuntimeError("Incomplete validation split")
    return results, float((total_loss / count).sum()), exported


def save_predictions(folder, exported):
    folder.mkdir(parents=True, exist_ok=True)
    for name, values in exported.items():
        np.savetxt(folder / f"{name}.txt", values, fmt="%.6f,%.6f,%.6f,%.6f,%.8f,%d,%d,%d")


def train(a):
    torch.set_num_threads(4)
    d.seed_everything(a.seed)
    wall_start = time.perf_counter()
    verify_sources(read(a.job / "provenance.json"))
    output = a.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "weights").mkdir(exist_ok=True)
    if (output / "DONE").exists():
        raise RuntimeError("Completed run exists; campaign must skip it, not overwrite")
    batches = Batches(a.job, a.method, a.seed)
    d.seed_everything(a.seed)  # Match head initialization after teacher construction.
    model = d.CachedLatentDetector(in_channels=(384,) * 3, nc=10, imgsz=640, epochs=a.epochs).cuda()
    configure_fusion(model, "mean_value")
    model.detect.max_det = 500
    criterion = d.E2EDetectLoss(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.epochs, eta_min=0.00001)
    args = {
        "method": a.method,
        "seed": a.seed,
        "epochs": a.epochs,
        "batch": a.batch,
        "train_images": len(batches.train),
        "val_images": len(batches.val),
        "imgsz": 640,
        "max_det": 500,
        "conf": 0.001,
        "cache_identity": batches.meta["identity"],
        "fusion": "mean_value",
        "lr": 0.001,
        "weight_decay": 0.0005,
        "aux_weight": 0,
        "query_pct": batches.pct,
        "budget_unit": "image, including tail batches",
        "evaluator": "custom DetMetrics; ignored regions NOT handled; NOT official VisDrone AP",
        "dataset_manifest_sha256": sha(a.job / "dataset-manifest.json"),
        "cache_manifest_sha256": sha(a.job / "cache/manifest.json"),
    }
    previous = read(output / "args.json") if (output / "args.json").exists() else None
    if previous is not None and previous != args:
        raise RuntimeError("Refuse resume with a changed experiment configuration")
    dump(output / "args.json", args)
    dump(
        output / "trainability.json",
        {
            "head_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "corrector_trainable": sum(p.numel() for p in batches.corrector.parameters()) if batches.corrector else 0,
            "teacher_frozen": sum(p.numel() for p in batches.extractor.model.parameters())
            if batches.extractor
            else None,
            "note": "Offline has the same pretrained teacher; it is absent from the training process.",
        },
    )
    begin, best, best_epoch = 0, -1.0, 0
    total_train = total_val = prior_wall = 0.0
    rows = []
    last = output / "weights/last.pt"
    if last.exists():
        state = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        begin, best = state["epoch"], state["best_ap"]
        best_epoch = state["best_epoch"]
        publish_best(output, best_epoch)
        rows = state["rows"]
        total_train, total_val, prior_wall = state["totals"]
        if (output / "summary.json").exists():
            old_summary = read(output / "summary.json")
            if old_summary["completed_epochs"] == begin:
                prior_wall = max(prior_wall, old_summary["wall_seconds"])
        for key, value in state["counters"].items():
            setattr(batches, key, value)
        if batches.corrector is not None:
            batches.corrector.load_state_dict(state["corrector"])
            batches.optimizer.load_state_dict(state["corrector_optimizer"])
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        print(f"RESUME epoch={begin + 1}", flush=True)
    torch.cuda.reset_peak_memory_stats()
    sync()
    setup_seconds = time.perf_counter() - wall_start
    for epoch in range(begin, a.epochs):
        model.train()
        sync()
        tick = time.perf_counter()
        loss_sum, seen = 0.0, 0
        for batch in batches.training(epoch, a.batch, output):
            features, targets = d.move_batch(batch, torch.device("cuda:0"))
            optimizer.zero_grad(set_to_none=True)
            vector, _ = criterion(model.training_predictions(features), targets)
            loss = vector.sum()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite detector loss epoch {epoch + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            loss_sum += float(loss.detach())
            seen += len(batch["sample_ids"])
        sync()
        seconds = time.perf_counter() - tick
        if seen != len(batches.train) or batches.query_images != ((epoch + 1) * seen * batches.pct) // 100:
            raise RuntimeError("Image coverage or cumulative online quota mismatch")
        if batches.extractor and any(
            p.requires_grad or p.grad is not None for p in batches.extractor.model.parameters()
        ):
            raise RuntimeError("Teacher must remain frozen")
        tick = time.perf_counter()
        metrics, val_loss, exported = validate(model, batches, criterion, a.batch)
        sync()
        val_seconds = time.perf_counter() - tick
        total_train += seconds
        total_val += val_seconds
        ap = metrics["metrics/mAP50-95(B)"]
        row = {
            "epoch": epoch + 1,
            "train_loss": loss_sum / seen,
            "val_loss": val_loss,
            "map50": metrics["metrics/mAP50(B)"],
            "map50_95": ap,
            "train_seconds": seconds,
            "val_seconds": val_seconds,
            "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
            "lr": optimizer.param_groups[0]["lr"],
            "query_images": batches.query_images,
            "correction_seconds": batches.correction_seconds,
            "reconstruction_loss": float(np.mean(batches.reconstruction)) if batches.reconstruction else 0.0,
        }
        if not all(math.isfinite(float(x)) for x in row.values()):
            raise RuntimeError("Nonfinite metrics")
        rows.append(row)
        is_best = ap >= best
        best = max(best, ap)
        if is_best:
            best_epoch = epoch + 1
        scheduler.step()
        save_predictions(output / "predictions/last", exported)
        wall = prior_wall + time.perf_counter() - wall_start
        state = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_ap": best,
            "best_epoch": best_epoch,
            "rows": rows,
            "totals": [total_train, total_val, wall],
            "corrector": batches.corrector.state_dict() if batches.corrector else None,
            "corrector_optimizer": batches.optimizer.state_dict() if batches.optimizer else None,
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
            "counters": {
                key: getattr(batches, key)
                for key in ("query_images", "query_calls", "correction_steps", "correction_seconds")
            },
        }
        if is_best:
            artifact = output / "best-artifacts" / f"{best_epoch:03d}"
            save_predictions(artifact / "predictions", exported)
            atomic_checkpoint(artifact / "checkpoint.pt", state)
        wall = prior_wall + time.perf_counter() - wall_start
        with (output / "metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        keys = ["val-v0"] + (
            [] if a.method == "online-geom" else ["train-v0"] if a.method == "cache1" else ["train-v0", "train-v1"]
        )
        build_seconds = batches.cache["setup_seconds"] + sum(batches.cache["parts"][k]["seconds"] for k in keys)
        summary = {
            **args,
            "completed_epochs": epoch + 1,
            "best_AP": 100 * best,
            "best_epoch": best_epoch,
            "last_AP": 100 * ap,
            "train_seconds": total_train,
            "val_seconds": total_val,
            "wall_seconds": wall,
            "setup_seconds_this_process": setup_seconds,
            "peak_host_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "host_io_read_blocks": resource.getrusage(resource.RUSAGE_SELF).ru_inblock,
            "host_io_write_blocks": resource.getrusage(resource.RUSAGE_SELF).ru_oublock,
            "cache_build_charged_seconds": build_seconds,
            "data_prepare_charged_seconds": batches.meta["prepare_seconds"],
            "build_plus_wall_seconds": wall + build_seconds + batches.meta["prepare_seconds"],
            "cache_bytes": sum(batches.cache["parts"][k]["bytes"] for k in keys),
            "query_images": batches.query_images,
            "query_calls": batches.query_calls,
            "correction_seconds": batches.correction_seconds,
            "peak_vram_gib": row["peak_vram_gib"],
            "schedule_sha256": {p.name: sha(p) for p in sorted((output / "schedules").glob("*.json"))},
        }
        dump(output / "summary.json", summary)
        # Commit the epoch only after its metrics, summary and best artifacts exist.
        atomic_checkpoint(last, state)
        publish_best(output, best_epoch)
        summary["wall_seconds"] = prior_wall + time.perf_counter() - wall_start
        summary["build_plus_wall_seconds"] = summary["wall_seconds"] + build_seconds + batches.meta["prepare_seconds"]
        dump(output / "summary.json", summary)
        print(json.dumps(row), flush=True)
    final = read(output / "summary.json")
    if final["completed_epochs"] != a.epochs or len(rows) != a.epochs:
        raise RuntimeError("Refuse DONE with incomplete epoch evidence")
    publish_best(output, best_epoch)
    (output / "DONE").write_text("SUCCEEDED\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=8)
    a = p.parse_args()
    if min(a.epochs, a.batch) < 1:
        p.error("epochs and batch must be positive")
    train(a)


if __name__ == "__main__":
    main()
