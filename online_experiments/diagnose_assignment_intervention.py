"""Frozen-head 2x2 feature/assignment interventions; no optimization or AP evaluation."""

import argparse
import csv
import hashlib
import random
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from online_experiments.assignment_intervention import (
    assignment_metrics,
    capture_assignment,
    fixed_loss,
    gradient_comparison,
    loss_and_gradient,
    state_hash,
)
from online_experiments.budget_corrector import ConditionalCorrector
from online_experiments.d1_online_compare import geometry, training_feature
from online_experiments.feature_baselines import warp_features
from online_experiments.full_visdrone_data import dump, make_extractor, read, sha
from online_experiments.fusion_probe import configure_fusion
from scripts import d1_train_cached_detector as d
from ultralytics.data.foundation_cache import load_letterboxed_tensor


def sources():
    """Record loaded repository Python sources, excluding third-party dependencies."""
    result = {}
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.is_file() and path.suffix == ".py" and path.is_relative_to(ROOT):
            result[str(path.relative_to(ROOT))] = sha(path)
    return dict(sorted(result.items()))


def aggregate(rows, keys):
    """Equal image weights within a fixed method/view/branch/cell."""
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    output = []
    for key, records in sorted(groups.items()):
        result = dict(zip(keys, key))
        result["images"] = len({r["image_id"] for r in records})
        result["observations"] = len(records)
        for field in records[0]:
            if field in keys or field in ("image_id", "index"):
                continue
            per_image = defaultdict(list)
            for record in records:
                if isinstance(record[field], (int, float)):
                    per_image[record["image_id"]].append(record[field])
            values = [statistics.mean(v) for v in per_image.values()]
            if values:
                result[field + "_mean"] = statistics.mean(values)
                result[field + "_n"] = len(values)
                result[field + "_views_n"] = sum(len(v) for v in per_image.values())
        output.append(result)
    return output


def csv_write(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def report(rows):
    lines = [
        "Frozen-head diagnostic. No optimizer updates, no AP claims, shared GPU timing.",
        "Nonidentity views only; equal images and four views per image.",
        "box/cls refer to raw-output gradients, not parameter or feature gradients.",
        "hard_common uses unit-positive targets and a shared exact-positive denominator.",
        "method branch protocol cell loss box_grad_relL2 box_grad_cos cls_grad_relL2 cls_grad_cos",
    ]
    keys = ["method", "branch", "protocol", "cell"]
    for row in aggregate([r for r in rows if r["scale"] != 1.0], keys):
        labels = " ".join(str(row[k]) for k in keys)
        metrics = [
            row.get(k + "_mean")
            for k in (
                "loss_total",
                "box_relative_l2",
                "box_cosine",
                "cls_relative_l2",
                "cls_cosine",
            )
        ]
        lines.append(labels + " " + " ".join("NA" if x is None else f"{x:.6f}" for x in metrics))
    return "\n".join(lines) + "\n"


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    def status(message):
        (output / "status.txt").write_text(message + "\n")
        print(message, flush=True)

    try:
        status("RUNNING initialization")
        torch.set_num_threads(4)
        d.seed_everything(args.seed)
        job = args.source_job.resolve()
        meta = read(job / "dataset-manifest.json")
        manifest = read(job / "cache/manifest.json")
        if sha(job / "dataset-manifest.json") != manifest["dataset_manifest_sha256"]:
            raise ValueError("Original cache/dataset manifest mismatch")
        if meta["identity"] != manifest["identity"]:
            raise ValueError("Teacher identity mismatch")
        if meta["identity"]["imgsz"] != 640 or meta["identity"]["layers"] != [3, 7, 11]:
            raise ValueError("Unexpected source feature dimensions")
        population = meta["splits"][args.split]
        if len(meta["splits"]["train"]) != 6471 or len(meta["splits"]["val"]) != 548:
            raise ValueError("Expected original full VisDrone split")
        if args.count < 0 or args.count > len(population):
            raise ValueError("Invalid image count")
        ids = list(range(len(population)))
        if args.split == "train":
            random.Random(args.seed).shuffle(ids)
        ids = ids[: args.count] if args.count else ids
        selected = [population[i] for i in ids]
        dump(output / "selection.json", {"split": args.split, "seed": args.seed, "indices": ids, "samples": selected})
        parts = [f"{args.split}-v{v}" for v in range(2 if args.split == "train" else 1)]
        arrays = []
        for part in parts:
            path = job / "cache" / (part + ".npy")
            if path.stat().st_size != manifest["parts"][part]["bytes"]:
                raise ValueError("Cache size mismatch")
            array = np.load(path, mmap_mode="r")
            if array.shape != (len(population), 3, 384, 40, 40) or array.dtype != np.float16:
                raise ValueError("Cache shape/dtype mismatch")
            arrays.append(array)
        head_path = job / f"train/{args.head}-s0/weights/last.pt"
        correct_path = job / "train/correct25-s0/weights/last.pt"
        checkpoint_hashes = {str(p): sha(p) for p in (head_path, correct_path)}
        training_metadata = {}
        for method in (args.head, "correct25"):
            path = job / f"train/{method}-s0/args.json"
            training_args = read(path)
            expected = {
                "method": method,
                "seed": 0,
                "epochs": 100,
                "fusion": "mean_value",
                "cache_identity": meta["identity"],
                "train_images": 6471,
                "val_images": 548,
                "dataset_manifest_sha256": sha(job / "dataset-manifest.json"),
                "cache_manifest_sha256": sha(job / "cache/manifest.json"),
            }
            if any(training_args.get(k) != v for k, v in expected.items()):
                raise ValueError(f"Checkpoint training identity mismatch: {method}")
            training_metadata[method] = {"path": str(path), "sha256": sha(path), "args": training_args}
        head_state = torch.load(head_path, map_location="cpu", weights_only=False)
        correct_state = torch.load(correct_path, map_location="cpu", weights_only=False)
        if head_state["epoch"] != 100 or correct_state["epoch"] != 100:
            raise ValueError("Expected epoch-100 checkpoints")
        model = d.CachedLatentDetector(in_channels=(384,) * 3, nc=10, imgsz=640, epochs=100).cuda()
        configure_fusion(model, "mean_value")
        model.load_state_dict(head_state["model"], strict=True)
        corrector = ConditionalCorrector(384, 64, 0).cuda()
        corrector.load_state_dict(correct_state["corrector"], strict=True)
        del head_state, correct_state
        model.eval().requires_grad_(False)
        corrector.eval().requires_grad_(False)
        extractor = make_extractor(meta["identity"])
        extractor.model.eval().requires_grad_(False)
        criterion = d.E2EDetectLoss(model)
        original_states = {"head": state_hash(model), "corrector": state_hash(corrector)}
        source_hashes = sources()
        for relative in source_hashes:
            target = output / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
        provenance = {
            "protocol": "research/ASSIGNMENT_INTERVENTION_PLAN.md",
            "protocol_sha256": sha(ROOT / "research/ASSIGNMENT_INTERVENTION_PLAN.md"),
            "source_job": str(job),
            "source_hashes": source_hashes,
            "checkpoints": checkpoint_hashes,
            "checkpoint_training_metadata": training_metadata,
            "initial_state_hashes": original_states,
            "cache_manifest": manifest,
            "cache_hash_verification": "original manifest hashes inherited; sizes/shapes and used sample slices verified, no full cache rehash",
            "manifest_sha256": sha(job / "dataset-manifest.json"),
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
            "seed": args.seed,
            "split": args.split,
            "count": len(ids),
            "head": args.head,
            "views": [[flip, scale] for flip in (False, True) for scale in (1.0, 0.85, 1.15)],
            "batch": 1,
            "reg_max": model.detect.reg_max,
            "training_updates": 0,
            "environment": {"torch": torch.__version__, "numpy": np.__version__, "gpu": torch.cuda.get_device_name(0)},
            "interpretation": "local frozen-head intervention; not AP recovery, training-gradient estimate or new method; shared GPU, no speed claims",
        }
        dump(output / "provenance.json", provenance)
        rows, match_rows, cache_slices = [], [], []
        teacher_images, parity_max, loss_parity_max = 0, 0.0, 0.0
        started = time.perf_counter()
        for progress, (index, sample) in enumerate(zip(ids, selected), 1):
            if sha(sample["image_path"]) != sample["image_sha256"]:
                raise ValueError("Source image changed")
            image, info = load_letterboxed_tensor(sample["image_path"], imgsz=640)
            if info != sample["letterbox"]:
                raise ValueError("Source letterbox changed")
            raw_cls = torch.tensor(sample["cls"], dtype=torch.float32).reshape(-1, 1)
            raw_boxes = torch.tensor(sample["bboxes"], dtype=torch.float32).reshape(-1, 4)
            for flip in (False, True):
                anchor = None
                if not flip or args.split == "train":
                    raw_cache = np.array(arrays[int(flip)][index : index + 1], copy=True)
                    cache_slices.append(
                        {"index": index, "flip": flip, "sha256": hashlib.sha256(raw_cache.tobytes()).hexdigest()}
                    )
                    tensor = torch.from_numpy(raw_cache).cuda().float()
                    anchor = tuple(tensor[:, layer] for layer in range(3))
                for scale in (1.0, 0.85, 1.15):
                    transformed, cls, boxes = geometry(image, raw_cls, raw_boxes, flip, scale)
                    with torch.no_grad():
                        extracted = extractor(transformed[None])
                        exact = tuple(
                            training_feature(extracted[f"layer_{layer}"]) for layer in meta["identity"]["layers"]
                        )
                        teacher_images += 1
                        if scale == 1.0:
                            if anchor is None:
                                anchor = exact
                            else:
                                for cached, fresh in zip(anchor, exact):
                                    rel = float((cached - fresh).norm() / fresh.norm().clamp_min(1e-8))
                                    parity_max = max(parity_max, rel)
                                if parity_max > 0.01:
                                    raise ValueError(f"Identity parity failed: {parity_max}")
                        transported = tuple(warp_features(x, [False], [scale]) for x in anchor)
                        corrected = corrector(transported, [scale], [flip])
                        exact_predictions = model.training_predictions(exact)
                    targets = {
                        "cls": cls.cuda(),
                        "bboxes": boxes.cuda(),
                        "batch_idx": torch.zeros(len(cls), device="cuda"),
                    }
                    gt_areas = (targets["bboxes"][:, 2:] * 640).prod(1)
                    exact_assignments, reference_gradients = {}, {}
                    for branch, loss_fn in (("one2many", criterion.one2many), ("one2one", criterion.one2one)):
                        exact_a, native = capture_assignment(loss_fn, exact_predictions[branch], targets)
                        actual = fixed_loss(loss_fn, exact_predictions[branch], exact_a, "native_soft")
                        torch.testing.assert_close(actual, native, atol=2e-5, rtol=2e-5)
                        loss_parity_max = max(loss_parity_max, float((actual - native).abs().max()))
                        exact_assignments[branch] = exact_a
                        for protocol in ("native_soft", "hard_common"):
                            _, grads = loss_and_gradient(
                                loss_fn, exact_predictions[branch], exact_a, protocol, max(int(exact_a[3].sum()), 1)
                            )
                            reference_gradients[branch, protocol] = grads
                    for method, levels in (("transport", transported), ("rec", corrected)):
                        if not all(torch.isfinite(x).all() for x in levels):
                            raise ValueError("Nonfinite approximation")
                        feature_mse = statistics.mean(
                            float((a - b).square().sum() / b.square().sum().clamp_min(1e-8))
                            for a, b in zip(levels, exact)
                        )
                        with torch.no_grad():
                            approximate_predictions = model.training_predictions(levels)
                        for branch, loss_fn in (("one2many", criterion.one2many), ("one2one", criterion.one2one)):
                            exact_a = exact_assignments[branch]
                            approximate_a, native = capture_assignment(
                                loss_fn, approximate_predictions[branch], targets
                            )
                            actual = fixed_loss(loss_fn, approximate_predictions[branch], approximate_a, "native_soft")
                            torch.testing.assert_close(actual, native, atol=2e-5, rtol=2e-5)
                            loss_parity_max = max(loss_parity_max, float((actual - native).abs().max()))
                            common = {
                                "image_id": sample["sample_id"],
                                "index": index,
                                "flip": flip,
                                "scale": scale,
                                "method": method,
                                "branch": branch,
                            }
                            match_rows.append({**common, **assignment_metrics(exact_a, approximate_a, gt_areas)})
                            for protocol in ("native_soft", "hard_common"):
                                for feature_key, predictions in (
                                    ("E", exact_predictions),
                                    ("A", approximate_predictions),
                                ):
                                    for donor, assignment in (("E", exact_a), ("A", approximate_a)):
                                        loss, grads = loss_and_gradient(
                                            loss_fn,
                                            predictions[branch],
                                            assignment,
                                            protocol,
                                            max(int(exact_a[3].sum()), 1),
                                        )
                                        row = {
                                            **common,
                                            "protocol": protocol,
                                            "cell": feature_key + donor,
                                            "feature_relative_mse": feature_mse,
                                            "loss_box": float(loss[0]),
                                            "loss_cls": float(loss[1]),
                                            "loss_dfl": float(loss[2]),
                                            "loss_total": float(loss.sum()),
                                        }
                                        for name, gradient in grads.items():
                                            row.update(
                                                {
                                                    name + "_" + k: v
                                                    for k, v in gradient_comparison(
                                                        gradient, reference_gradients[branch, protocol][name]
                                                    ).items()
                                                }
                                            )
                                        rows.append(row)
            if progress == 1 or progress % 8 == 0 or progress == len(ids):
                dump(
                    output / "progress.json",
                    {
                        "completed_images": progress,
                        "total_images": len(ids),
                        "teacher_images": teacher_images,
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                )
                status(f"RUNNING images={progress}/{len(ids)} teacher_images={teacher_images}")
        if teacher_images != len(ids) * 6 or len(rows) != len(ids) * 6 * 2 * 2 * 2 * 4:
            raise ValueError("Incomplete diagnostic grid")
        for relative, fingerprint in source_hashes.items():
            if sha(ROOT / relative) != fingerprint:
                raise ValueError(f"Source changed during run: {relative}")
        for path, fingerprint in checkpoint_hashes.items():
            if sha(path) != fingerprint:
                raise ValueError("Checkpoint changed during run")
        for entry in training_metadata.values():
            if sha(entry["path"]) != entry["sha256"]:
                raise ValueError("Checkpoint training metadata changed during run")
        if original_states != {"head": state_hash(model), "corrector": state_hash(corrector)}:
            raise ValueError("Frozen model state changed")
        csv_write(output / "per-image.csv", rows)
        csv_write(output / "assignments.csv", match_rows)
        csv_write(output / "summary.csv", aggregate(rows, ["method", "branch", "protocol", "flip", "scale", "cell"]))
        csv_write(output / "assignment-summary.csv", aggregate(match_rows, ["method", "branch", "flip", "scale"]))
        dump(output / "cache-slices.json", cache_slices)
        dump(
            output / "summary.json",
            {
                "images": len(ids),
                "split": args.split,
                "teacher_images": teacher_images,
                "rows": len(rows),
                "parity_max_relative_l2": parity_max,
                "native_loss_parity_max_absolute": loss_parity_max,
                "training_updates": 0,
                "states_unchanged": True,
                "elapsed_seconds": time.perf_counter() - started,
                "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
                "outputs_sha256": {
                    name: sha(output / name)
                    for name in (
                        "per-image.csv",
                        "assignments.csv",
                        "summary.csv",
                        "assignment-summary.csv",
                        "selection.json",
                    )
                },
            },
        )
        (output / "comparison.txt").write_text(report(rows))
        status("SUCCEEDED")
        print((output / "comparison.txt").read_text(), flush=True)
    except BaseException:
        (output / "traceback.txt").write_text(traceback.format_exc())
        status("FAILED: inspect traceback.txt")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--count", type=int, default=128, help="0 means the full selected split")
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--head", choices=("online-geom", "mix25"), default="online-geom")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
