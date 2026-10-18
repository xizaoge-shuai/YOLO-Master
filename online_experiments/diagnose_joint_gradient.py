"""Joint spatial oracles and native-loss control-variate gradient diagnostics."""

import argparse
import hashlib
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

from online_experiments.assignment_intervention import state_hash
from online_experiments.d1_online_compare import geometry, training_feature
from online_experiments.diagnose_assignment_intervention import aggregate, csv_write, sources
from online_experiments.diagnose_spatial_intervention import HEADS, load_resources, measure, report
from online_experiments.feature_baselines import warp_features
from online_experiments.full_visdrone_data import dump, sha
from online_experiments.gradient_calibration import (
    estimator_statistics,
    joint_spatial_arms,
    native_parameter_gradients,
)
from online_experiments.spatial_intervention import (
    blank_fill,
    parameter_groups,
    regional_errors,
    restore_region,
    spatial_regions,
)
from scripts import d1_train_cached_detector as d
from ultralytics.data.foundation_cache import load_letterboxed_tensor

ARMS = (
    "exact",
    "transport",
    "rec",
    "padding_oracle",
    "content_oracle",
    "blank_fill",
    "joint_oracle",
    "blank_content_oracle",
)
PROBABILITY = 0.25


def screening_decision(rows, count):
    """Compare pooled conditional variance to same-budget query-only updates."""
    cells, passed = [], []
    for arm in ("transport", "blank_fill", "rec"):
        eligible = True
        for head in HEADS:
            for scale in (0.85, 1.15):
                selected = [
                    r
                    for r in rows
                    if r["arm"] == arm and r["head"] == head and r["scale"] == scale and r["block"] == "__all__"
                ]
                if len(selected) != count * 2:
                    raise ValueError("Incomplete screening grid")
                exact = sum(r["exact_squared_norm"] for r in selected)
                residual = sum(r["residual_squared_norm"] for r in selected)
                ratio = residual / exact if exact > 0 else None
                accept = ratio is not None and ratio < 1
                eligible = eligible and accept
                cells.append(
                    {
                        "arm": arm,
                        "head": head,
                        "scale": scale,
                        "observations": len(selected),
                        "exact_squared_norm": exact,
                        "residual_squared_norm": residual,
                        "variance_ratio": ratio,
                        "passes_cell": accept,
                    }
                )
        if eligible:
            passed.append(arm)
    return {
        "stage": "execution_smoke" if count == 2 else "development_screen",
        "probability": PROBABILITY,
        "cells": cells,
        "eligible_for_minibatch_probe": passed if count == 128 else [],
        "full_training_authorized_by_this_gate": False,
        "reason": "Frozen eval, batch1 native-loss screening only; minibatch and train-state checks still required.",
    }


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
            "online_experiments/archive_joint_gradient.py",
            "online_experiments/run_joint_gradient.sh",
            "online_experiments/test_gradient_calibration.py",
            "online_experiments/test_joint_archive.py",
        ):
            source_hashes[relative] = sha(ROOT / relative)
        for relative in source_hashes:
            target = output / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
        protocol = "research/JOINT_GRADIENT_IMPLEMENTATION_20261018.md"
        protocol_copy = output / "source" / protocol
        protocol_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / protocol, protocol_copy)
        dump(
            output / "provenance.json",
            {
                "protocol": "research/JOINT_GRADIENT_IMPLEMENTATION_20261018.md",
                "protocol_sha256": sha(ROOT / "research/JOINT_GRADIENT_IMPLEMENTATION_20261018.md"),
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
                "query_probability_for_enumeration": PROBABILITY,
                "native_gradient_objective": "Actual E2EDetectLoss returned loss.sum(); own assignment for each arm; eval batch1; no clipping or optimizer",
                "interpretation": "Conditional frozen-eval diagnostics; native gradients use own assignments; one2one shared detachment preserved; all views queried; no optimizer updates or AP claims.",
            },
        )
        rows, parameters, matches, regions, slices, native_rows = [], [], [], [], [], []
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
                        arms.update(joint_spatial_arms(transported, exact, masks, support, blank))
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
                        exact_native = None
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
                            native_grad, native_loss = native_parameter_gradients(
                                models[head], criteria[head], groups[head], arms[arm], targets
                            )
                            if arm == "exact":
                                exact_native = native_grad
                            stats = estimator_statistics(native_grad, exact_native, PROBABILITY)
                            native_rows.extend(
                                base | {"head": head, "arm": arm, "block": block, "native_loss": native_loss, **values}
                                for block, values in stats.items()
                            )
                            del native_grad, stats
                        del references, exact_native
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
            "output_rows": len(ids) * 192,
            "parameter_rows": len(ids) * 2688,
            "matching_rows": len(ids) * 192,
            "region_rows": len(ids) * 720,
        }
        found = dict(zip(expected, map(len, (rows, parameters, matches, regions))))
        if len(native_rows) != len(ids) * 6 * 2 * len(ARMS) * 8:
            raise ValueError("Incomplete native-gradient grid")
        if found != expected or teacher_images != len(ids) * 6:
            raise ValueError(f"Incomplete output grid: {found}")
        for filename, values in [
            ("per-image.csv", rows),
            ("parameters.csv", parameters),
            ("assignments.csv", matches),
            ("regions.csv", regions),
            ("native-gradients.csv", native_rows),
        ]:
            csv_write(output / filename, values)
        csv_write(output / "summary.csv", aggregate(rows, ["head", "arm", "branch", "flip", "scale"]))
        csv_write(
            output / "parameter-summary.csv",
            aggregate(parameters, ["head", "arm", "branch", "task", "block", "flip", "scale"]),
        )
        csv_write(output / "region-summary.csv", aggregate(regions, ["arm", "layer", "region", "flip", "scale"]))
        csv_write(output / "assignment-summary.csv", aggregate(matches, ["head", "arm", "branch", "flip", "scale"]))
        csv_write(output / "native-summary.csv", aggregate(native_rows, ["head", "arm", "block", "flip", "scale"]))
        decision = screening_decision(native_rows, len(ids))
        dump(output / "screening.json", decision)
        dump(output / "cache-slices.json", slices)
        native_lines = [
            "",
            "Native-loss batch1 frozen-eval gradient variance screen; p=0.25 (enumeration, not training).",
            "arm head scale calibration/query_only_variance_ratio",
        ]
        for cell in decision["cells"]:
            ratio = cell["variance_ratio"]
            native_lines.append(
                f"{cell['arm']} {cell['head']} {cell['scale']} " + ("NA" if ratio is None else f"{ratio:.6f}")
            )
        native_lines.append("Eligible for minibatch probe: " + str(decision["eligible_for_minibatch_probe"]))
        native_lines.append("No full-training/AP conclusion; all views queried for diagnostics.")
        (output / "comparison.txt").write_text(report(rows, parameters) + "\n".join(native_lines) + "\n")
        names = [
            "per-image.csv",
            "parameters.csv",
            "assignments.csv",
            "regions.csv",
            "native-gradients.csv",
            "native-summary.csv",
            "screening.json",
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
                "native_rows": len(native_rows),
                "expectation_max_relative_l2": max(
                    (r["expectation_relative_l2"] for r in native_rows if r["expectation_relative_l2"] is not None),
                    default=0.0,
                ),
                "parity_max_relative_l2": parity,
                "native_loss_parity_max_absolute": native_parity,
                "states_unchanged": True,
                "training_updates": 0,
                "query_probability_for_enumeration": PROBABILITY,
                "native_gradient_objective": "Actual E2EDetectLoss returned loss.sum(); own assignment for each arm; eval batch1; no clipping or optimizer",
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
