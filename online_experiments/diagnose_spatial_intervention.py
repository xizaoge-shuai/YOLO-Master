"""Paired spatial oracles and cheap padding with actual head-parameter gradients."""

import argparse
import hashlib
import random
import shutil
import subprocess
import sys
import time
import traceback
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
from online_experiments.diagnose_assignment_intervention import aggregate, csv_write, sources
from online_experiments.feature_baselines import warp_features
from online_experiments.full_visdrone_data import dump, make_extractor, read, sha
from online_experiments.fusion_probe import configure_fusion
from online_experiments.spatial_intervention import (
    blank_fill,
    compare_gradient_blocks,
    parameter_gradients,
    parameter_groups,
    regional_errors,
    restore_region,
    spatial_regions,
)
from scripts import d1_train_cached_detector as d
from ultralytics.data.foundation_cache import load_letterboxed_tensor

HEADS = ("online-geom", "mix25")
ARMS = ("exact", "transport", "rec", "padding_oracle", "content_oracle", "blank_fill")


def load_resources(job, output, count, seed):
    """Validate the original full-data checkpoint/cache identities before evaluation."""
    meta, manifest = read(job / "dataset-manifest.json"), read(job / "cache/manifest.json")
    if (
        sha(job / "dataset-manifest.json") != manifest["dataset_manifest_sha256"]
        or meta["identity"] != manifest["identity"]
    ):
        raise ValueError("Source cache identity mismatch")
    if meta["identity"]["imgsz"] != 640 or meta["identity"]["layers"] != [3, 7, 11]:
        raise ValueError("Unexpected source feature dimensions")
    if len(meta["splits"]["train"]) != 6471 or len(meta["splits"]["val"]) != 548 or count not in (2, 128):
        raise ValueError("Expected fixed development protocol")
    population = meta["splits"]["train"]
    ids = list(range(len(population)))
    random.Random(seed).shuffle(ids)
    ids = ids[:count]
    selected = [population[i] for i in ids]
    dump(output / "selection.json", {"split": "train", "seed": seed, "indices": ids, "samples": selected})
    arrays = []
    for view in (0, 1):
        name = f"train-v{view}"
        path = job / "cache" / f"{name}.npy"
        if path.stat().st_size != manifest["parts"][name]["bytes"]:
            raise ValueError("Cache file size mismatch")
        array = np.load(path, mmap_mode="r")
        if array.shape != (6471, 3, 384, 40, 40) or array.dtype != np.float16:
            raise ValueError("Cache array shape/dtype mismatch")
        arrays.append(array)
    models, metadata, checkpoints = {}, {}, {}
    for method in (*HEADS, "correct25"):
        args_path = job / f"train/{method}-s0/args.json"
        train_args = read(args_path)
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
        if any(train_args.get(k) != v for k, v in expected.items()):
            raise ValueError(f"Checkpoint training identity mismatch: {method}")
        metadata[method] = {"args": train_args, "sha256": sha(args_path)}
        path = job / f"train/{method}-s0/weights/last.pt"
        checkpoints[str(path)] = sha(path)
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["epoch"] != 100:
            raise ValueError("Expected epoch-100 checkpoint")
        if method == "correct25":
            corrector = ConditionalCorrector(384, 64, 0).cuda()
            corrector.load_state_dict(state["corrector"], strict=True)
            corrector.eval().requires_grad_(False)
        else:
            model = d.CachedLatentDetector(in_channels=(384,) * 3, nc=10, imgsz=640, epochs=100).cuda()
            configure_fusion(model, "mean_value")
            model.load_state_dict(state["model"], strict=True)
            # Keep eval/BN state fixed; enable autograd only to measure head derivatives.
            model.eval().requires_grad_(True)
            models[method] = model
        del state
    extractor = make_extractor(meta["identity"])
    extractor.model.eval().requires_grad_(False)
    return meta, manifest, ids, selected, arrays, models, corrector, extractor, metadata, checkpoints


def measure(model, criterion, groups, levels, targets, common, assignments=None, references=None):
    """Evaluate one arm; references use exact features under the same head and view."""
    predictions = model.training_predictions(levels)
    assignments = {} if assignments is None else assignments
    generated = references is None
    references = {} if generated else references
    output_rows, parameter_rows, matching_rows = [], [], []
    parity = 0.0
    for branch, loss_fn in (("one2many", criterion.one2many), ("one2one", criterion.one2one)):
        pred = predictions[branch]
        donor, native = capture_assignment(loss_fn, pred, targets)
        with torch.no_grad():
            reproduced = fixed_loss(loss_fn, pred, donor, "native_soft")
        torch.testing.assert_close(reproduced, native, atol=2e-5, rtol=2e-5)
        parity = max(parity, float((reproduced - native).abs().max()))
        if generated:
            assignments[branch] = donor
        exact_assignment = assignments[branch]
        denominator = max(int(exact_assignment[3].sum()), 1)
        hard = fixed_loss(loss_fn, pred, exact_assignment, "hard_common", denominator)
        _, output_grad = loss_and_gradient(loss_fn, pred, exact_assignment, "hard_common", denominator)
        parameter_grad = {
            "cls": parameter_gradients(hard[1], groups, retain_graph=True),
            "box": parameter_gradients(hard[0] + hard[2], groups, retain_graph=True),
        }
        if generated:
            references[branch] = {"outputs": output_grad, "parameters": parameter_grad}
        row = common | {
            "branch": branch,
            "protocol": "hard_common_exact",
            "loss_box": float(hard[0].detach()),
            "loss_cls": float(hard[1].detach()),
            "loss_aux_l1": float(hard[2].detach()),
            "native_loss": float(native.sum()),
        }
        for task in ("cls", "box"):
            row.update(
                {
                    f"{task}_{k}": v
                    for k, v in gradient_comparison(output_grad[task], references[branch]["outputs"][task]).items()
                }
            )
            for block, values in compare_gradient_blocks(
                parameter_grad[task], references[branch]["parameters"][task]
            ).items():
                parameter_rows.append(common | {"branch": branch, "task": task, "block": block, **values})
        output_rows.append(row)
        gt_areas = (targets["bboxes"][:, 2:] * 640).prod(-1)
        matching_rows.append(common | {"branch": branch, **assignment_metrics(exact_assignment, donor, gt_areas)})
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError("Unexpected accumulated parameter gradients")
    return output_rows, parameter_rows, matching_rows, assignments, references, parity


def report(rows, parameters):
    lines = [
        "Spatial interventions; frozen eval heads, no training or AP evaluation.",
        "Hard targets from exact features; nonidentity views; equal image weights.",
        "head arm branch output_cls_cos output_box_cos native_loss",
    ]
    for row in aggregate([r for r in rows if r["scale"] != 1], ["head", "arm", "branch"]):
        fields = [row.get(k + "_mean") for k in ("cls_cosine", "box_cosine", "native_loss")]
        lines.append(
            " ".join(str(row[k]) for k in ("head", "arm", "branch"))
            + " "
            + " ".join("NA" if v is None else f"{v:.6f}" for v in fields)
        )
    lines += ["", "head arm branch task block param_cos param_relL2 defined_images"]
    for row in aggregate([r for r in parameters if r["scale"] != 1], ["head", "arm", "branch", "task", "block"]):
        if row["block"] not in ("p3", "one2many_box", "one2one_box"):
            continue
        fields = [row.get(k + "_mean") for k in ("cosine", "relative_l2")]
        lines.append(
            " ".join(str(row[k]) for k in ("head", "arm", "branch", "task", "block"))
            + " "
            + " ".join("NA" if v is None else f"{v:.6f}" for v in fields)
            + f" {row.get('cosine_n', 0)}"
        )
    return "\n".join(lines) + "\n"


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    def status(text):
        (output / "status.txt").write_text(text + "\n")
        print(text, flush=True)

    try:
        status("RUNNING initialization")
        torch.set_num_threads(4)
        d.seed_everything(args.seed)
        meta, manifest, ids, selected, arrays, models, corrector, extractor, metadata, checkpoints = load_resources(
            args.source_job.resolve(), output, args.count, args.seed
        )
        criteria = {head: d.E2EDetectLoss(model) for head, model in models.items()}
        groups = {head: parameter_groups(model) for head, model in models.items()}
        states = {head: state_hash(model) for head, model in models.items()}
        states["corrector"] = state_hash(corrector)
        source_hashes = sources()
        for relative in (
            "online_experiments/archive_spatial_intervention.py",
            "online_experiments/run_spatial_intervention.sh",
            "online_experiments/test_spatial_intervention.py",
        ):
            source_hashes[relative] = sha(ROOT / relative)
        for relative in source_hashes:
            target = output / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
        protocol = "research/SPATIAL_INTERVENTION_IMPLEMENTATION_20261018.md"
        protocol_copy = output / "source" / protocol
        protocol_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / protocol, protocol_copy)
        dump(
            output / "provenance.json",
            {
                "protocol": "research/SPATIAL_INTERVENTION_IMPLEMENTATION_20261018.md",
                "protocol_sha256": sha(ROOT / "research/SPATIAL_INTERVENTION_IMPLEMENTATION_20261018.md"),
                "source_job": str(args.source_job),
                "source_hashes": source_hashes,
                "checkpoints": checkpoints,
                "training_metadata": metadata,
                "cache_manifest": manifest,
                "cache_verification": "sizes/shapes plus used-slice hashes; inherited manifest, no full array rehash",
                "initial_state_hashes": states,
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
                "seed": args.seed,
                "split": "train",
                "count": len(ids),
                "views": [[flip, scale] for flip in (False, True) for scale in (1.0, 0.85, 1.15)],
                "heads": list(HEADS),
                "arms": list(ARMS),
                "parameter_blocks": {
                    h: {
                        name: {"names": [k for k, _ in pairs], "numel": sum(p.numel() for _, p in pairs)}
                        for name, pairs in g.items()
                    }
                    for h, g in groups.items()
                },
                "environment": {
                    "torch": torch.__version__,
                    "numpy": np.__version__,
                    "gpu": torch.cuda.get_device_name(0),
                },
                "training_updates": 0,
                "interpretation": "Conditional frozen-eval diagnostics; parameter autograd enabled without updates; one2one shared detachment preserved; natural-area oracles are not equal-energy interventions.",
            },
        )
        rows, parameters, matches, regions, slices = [], [], [], [], []
        parity, native_parity, teacher_images = 0.0, 0.0, 0
        torch.cuda.synchronize()
        started = time.perf_counter()
        build_started = started
        with torch.no_grad():
            raw_blank = extractor(torch.full((1, 3, 640, 640), 114 / 255))
            blank = tuple(training_feature(raw_blank[f"layer_{l}"]) for l in meta["identity"]["layers"])
        torch.cuda.synchronize()
        blank_seconds = time.perf_counter() - build_started
        blank_bytes = sum(x.numel() * x.element_size() for x in blank)
        dump(
            output / "blank-template.json",
            {
                "teacher_images": 1,
                "seconds": blank_seconds,
                "bytes_fp32": blank_bytes,
                "hashes": [hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest() for x in blank],
            },
        )
        for progress, (index, sample) in enumerate(zip(ids, selected), 1):
            if sha(sample["image_path"]) != sample["image_sha256"]:
                raise ValueError("Image changed")
            image, info = load_letterboxed_tensor(sample["image_path"], imgsz=640)
            if info != sample["letterbox"]:
                raise ValueError("Letterbox changed")
            classes = torch.tensor(sample["cls"], dtype=torch.float32).reshape(-1, 1)
            boxes = torch.tensor(sample["bboxes"], dtype=torch.float32).reshape(-1, 4)
            for flip in (False, True):
                cache = np.array(arrays[int(flip)][index : index + 1], copy=True)
                slices.append({"index": index, "flip": flip, "sha256": hashlib.sha256(cache.tobytes()).hexdigest()})
                tensor = torch.from_numpy(cache).cuda().float()
                anchor = tuple(tensor[:, i] for i in range(3))
                for scale in (1.0, 0.85, 1.15):
                    transformed, cls, bboxes = geometry(image, classes, boxes, flip, scale)
                    with torch.no_grad():
                        extracted = extractor(transformed[None])
                        teacher_images += 1
                        exact = tuple(training_feature(extracted[f"layer_{l}"]) for l in meta["identity"]["layers"])
                        if scale == 1:
                            parity = max(
                                parity,
                                *(float((a - e).norm() / e.norm().clamp_min(1e-8)) for a, e in zip(anchor, exact)),
                            )
                            if parity > 0.01:
                                raise ValueError("Cache identity parity failed")
                        transported = tuple(warp_features(x, [False], [scale]) for x in anchor)
                        corrected = corrector(transported, [scale], [flip])
                        masks, support = spatial_regions(info, boxes, flip, scale)
                        masks = {name: mask.cuda() for name, mask in masks.items()}
                        support = support.cuda()
                        if not torch.equal(
                            sum(mask.int() for mask in masks.values()), torch.ones_like(support, dtype=torch.int)
                        ):
                            raise ValueError("Region partition failed")
                        arms = {
                            "exact": exact,
                            "transport": transported,
                            "rec": corrected,
                            "padding_oracle": tuple(
                                restore_region(a, e, masks["new_padding"]) for a, e in zip(transported, exact)
                            ),
                            "content_oracle": tuple(
                                restore_region(a, e, masks["foreground"] | masks["background"])
                                for a, e in zip(transported, exact)
                            ),
                            "blank_fill": tuple(blank_fill(a, b, support) for a, b in zip(transported, blank)),
                        }
                    targets = {
                        "cls": cls.cuda(),
                        "bboxes": bboxes.cuda(),
                        "batch_idx": torch.zeros(len(cls), device="cuda"),
                    }
                    base = {"image_id": sample["sample_id"], "index": index, "flip": flip, "scale": scale}
                    for arm, levels in arms.items():
                        if not all(torch.isfinite(x).all() for x in levels):
                            raise ValueError("Nonfinite feature")
                        for layer, value, reference in zip(meta["identity"]["layers"], levels, exact):
                            regions.extend(
                                base | {"arm": arm, "layer": layer, **row}
                                for row in regional_errors(value, reference, masks)
                            )
                    for head in HEADS:
                        assignments, references = None, None
                        for arm in ARMS:
                            found, grads, agreement, assignments, references, loss_parity = measure(
                                models[head],
                                criteria[head],
                                groups[head],
                                arms[arm],
                                targets,
                                base | {"head": head, "arm": arm},
                                assignments,
                                references,
                            )
                            rows.extend(found)
                            parameters.extend(grads)
                            matches.extend(agreement)
                            native_parity = max(native_parity, loss_parity)
                        del references
            if progress == 1 or progress % 8 == 0 or progress == len(ids):
                torch.cuda.synchronize()
                dump(
                    output / "progress.json",
                    {
                        "completed_images": progress,
                        "total_images": len(ids),
                        "teacher_images": teacher_images,
                        "blank_teacher_images": 1,
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                )
                status(f"RUNNING images={progress}/{len(ids)} teacher_images={teacher_images}+1")
        final = {head: state_hash(model) for head, model in models.items()}
        final["corrector"] = state_hash(corrector)
        if states != final:
            raise ValueError("Weights/buffers changed")
        expected = {
            "output_rows": len(ids) * 144,
            "parameter_rows": len(ids) * 2016,
            "matching_rows": len(ids) * 144,
            "region_rows": len(ids) * 540,
        }
        found = dict(zip(expected, map(len, (rows, parameters, matches, regions))))
        if found != expected or teacher_images != len(ids) * 6:
            raise ValueError(f"Incomplete output grid: {found}")
        for filename, values in [
            ("per-image.csv", rows),
            ("parameters.csv", parameters),
            ("assignments.csv", matches),
            ("regions.csv", regions),
        ]:
            csv_write(output / filename, values)
        csv_write(output / "summary.csv", aggregate(rows, ["head", "arm", "branch", "flip", "scale"]))
        csv_write(
            output / "parameter-summary.csv",
            aggregate(parameters, ["head", "arm", "branch", "task", "block", "flip", "scale"]),
        )
        csv_write(output / "region-summary.csv", aggregate(regions, ["arm", "layer", "region", "flip", "scale"]))
        csv_write(output / "assignment-summary.csv", aggregate(matches, ["head", "arm", "branch", "flip", "scale"]))
        dump(output / "cache-slices.json", slices)
        (output / "comparison.txt").write_text(report(rows, parameters))
        names = [
            "per-image.csv",
            "parameters.csv",
            "assignments.csv",
            "regions.csv",
            "summary.csv",
            "parameter-summary.csv",
            "region-summary.csv",
            "assignment-summary.csv",
            "selection.json",
            "blank-template.json",
            "cache-slices.json",
            "comparison.txt",
            "provenance.json",
        ]
        torch.cuda.synchronize()
        dump(
            output / "summary.json",
            {
                "images": len(ids),
                "split": "train",
                "heads": list(HEADS),
                "arms": list(ARMS),
                "teacher_images": teacher_images,
                "blank_teacher_images": 1,
                **found,
                "parity_max_relative_l2": parity,
                "native_loss_parity_max_absolute": native_parity,
                "states_unchanged": True,
                "training_updates": 0,
                "elapsed_seconds": time.perf_counter() - started,
                "blank_build_seconds": blank_seconds,
                "blank_storage_bytes_fp32": blank_bytes,
                "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
                "outputs_sha256": {name: sha(output / name) for name in names},
            },
        )
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
    parser.add_argument("--count", type=int, choices=(2, 128), default=128)
    parser.add_argument("--seed", type=int, default=20261014)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
