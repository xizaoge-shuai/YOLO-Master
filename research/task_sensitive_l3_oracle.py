#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.d1_build_feature_cache import DINOv3MultiLevelExtractor
from scripts.d1_train_cached_detector import (
    CachedLatentDetector,
    label_path,
    match_predictions,
    read_yolo_labels,
    split_sample_indices,
)
from research.oracle_rank_detection import (
    augment_image,
    transport_feature,
    transform_boxes,
    load_cached_features,
    image_path_from_sample,
    MetricAccumulator,
)
from ultralytics.data.foundation_cache import load_letterboxed_tensor
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.loss import E2EDetectLoss


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--checkpoint", type=Path, required=True)

    p.add_argument(
        "--model",
        default="Tooony133/dinov3-vits16-pretrain-lvd1689m",
    )
    p.add_argument("--revision", required=True)

    p.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=[3, 7, 11],
    )

    p.add_argument(
        "--augmentations",
        nargs="+",
        default=["hflip", "zoom_in15"],
    )

    p.add_argument(
        "--ranks",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 32, 64, 96, 128],
    )

    p.add_argument("--calibration-count", type=int, default=20)
    p.add_argument("--eval-count", type=int, default=100)
    p.add_argument("--positions-per-image", type=int, default=128)
    p.add_argument("--split-seed", type=int, default=0)

    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="fp16")
    p.add_argument("--conf", type=float, default=0.001)

    p.add_argument("--output", type=Path, required=True)

    return p.parse_args()


def load_model(
    checkpoint,
    in_channels,
    nc,
    imgsz,
    device,
):
    model = CachedLatentDetector(
        in_channels=in_channels,
        nc=nc,
        imgsz=imgsz,
        epochs=100,
    ).to(device)

    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    state = payload.get("model") or payload.get("ema")

    if hasattr(state, "state_dict"):
        state = state.state_dict()

    state = dict(state)

    dropped = []

    for key in list(state):
        if (
            key == "_mixture_loss_ema_buf"
            or key.endswith("._mixture_loss_ema_buf")
        ):
            dropped.append(key)
            state.pop(key)

    missing, unexpected = model.load_state_dict(
        state,
        strict=False,
    )

    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    print(
        f"Checkpoint load: PASS; dropped={dropped}",
        flush=True,
    )

    return model


def project_residual(
    residual,
    mean,
    basis,
    rank,
):
    _, channels, height, width = residual.shape

    matrix = (
        residual[0]
        .permute(1, 2, 0)
        .reshape(-1, channels)
        .float()
    )

    mean = mean.to(matrix.device)

    basis = basis[
        :,
        :min(rank, basis.shape[1])
    ].to(matrix.device)

    centered = matrix - mean.unsqueeze(0)

    coeff = centered @ basis

    approx = (
        mean.unsqueeze(0)
        +
        coeff @ basis.T
    )

    return (
        approx
        .reshape(height, width, channels)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .contiguous()
    )


def fit_basis(
    matrix,
    max_rank,
    weights=None,
):
    matrix = matrix.float()

    if weights is None:
        mean = matrix.mean(dim=0)
        centered = matrix - mean.unsqueeze(0)
        fitted = centered
    else:
        weights = weights.float().clamp_min(1e-12)

        weights = (
            weights
            /
            weights.mean().clamp_min(1e-12)
        )

        weights = weights.clamp(
            min=0.02,
            max=20.0,
        )

        mean = (
            (matrix * weights[:, None]).sum(dim=0)
            /
            weights.sum()
        )

        centered = matrix - mean.unsqueeze(0)

        fitted = (
            centered
            *
            torch.sqrt(weights[:, None])
        )

    _, singular, vh = torch.linalg.svd(
        fitted,
        full_matrices=False,
    )

    rank = min(
        max_rank,
        vh.shape[0],
    )

    basis = (
        vh[:rank]
        .T
        .contiguous()
    )

    energy = singular.square()
    total = energy.sum().clamp_min(1e-12)

    def energy_at(r):
        r = min(r, len(energy))
        return float(
            energy[:r].sum() / total
        )

    return {
        "mean": mean.cpu(),
        "basis": basis.cpu(),
        "rank8_energy": energy_at(8),
        "rank16_energy": energy_at(16),
        "rank32_energy": energy_at(32),
        "rank64_energy": energy_at(64),
    }


def calibration(
    *,
    samples,
    indices,
    cache_root,
    dataset_root,
    layers,
    augmentations,
    extractor,
    model,
    criterion,
    nc,
    imgsz,
    device,
    positions_per_image,
    max_rank,
):
    l3_level = layers.index(3)

    residual_vectors = defaultdict(list)
    task_weights = defaultdict(list)

    sensitivity = defaultdict(
        lambda: {
            layer: {
                "grad_norm": [],
                "impact": [],
                "residual_norm": [],
            }
            for layer in layers
        }
    )

    for step, index in enumerate(indices, start=1):
        sample = samples[index]

        base = load_cached_features(
            cache_root,
            sample,
            layers,
            device,
        )

        image_path = image_path_from_sample(
            dataset_root,
            sample,
        )

        image, _ = load_letterboxed_tensor(
            image_path,
            imgsz=imgsz,
        )

        image = image.unsqueeze(0).to(device)

        classes, boxes = read_yolo_labels(
            label_path(
                dataset_root,
                sample["image_path"],
            ),
            sample["letterbox"],
            imgsz,
            nc,
        )

        for aug in augmentations:
            augmented = augment_image(
                image,
                aug,
            )

            with torch.inference_mode():
                true_dict = extractor(
                    augmented
                )

            true_features = tuple(
                true_dict[f"layer_{layer}"].float()
                for layer in layers
            )

            transported = tuple(
                transport_feature(
                    base[level],
                    aug,
                ).float()
                for level, layer in enumerate(layers)
            )

            aug_cls, aug_boxes = transform_boxes(
                classes,
                boxes,
                aug,
            )

            grad_features = tuple(
                feature.detach()
                .clone()
                .requires_grad_(True)
                for feature in transported
            )

            targets = {
                "cls": aug_cls.to(device),
                "bboxes": aug_boxes.to(device),
                "batch_idx": torch.zeros(
                    len(aug_cls),
                    dtype=torch.long,
                    device=device,
                ),
            }

            raw = model.training_predictions(
                grad_features
            )

            loss_vector, _ = criterion(
                raw,
                targets,
            )

            grads = torch.autograd.grad(
                loss_vector.sum(),
                grad_features,
                retain_graph=False,
                create_graph=False,
            )

            for level, layer in enumerate(layers):
                residual = (
                    true_features[level]
                    -
                    transported[level]
                )

                grad = grads[level]

                residual_position = (
                    residual[0]
                    .permute(1, 2, 0)
                    .reshape(
                        -1,
                        residual.shape[1],
                    )
                )

                grad_position = (
                    grad[0]
                    .permute(1, 2, 0)
                    .reshape(
                        -1,
                        grad.shape[1],
                    )
                )

                grad_norm = (
                    grad_position
                    .float()
                    .pow(2)
                    .sum(dim=1)
                    .sqrt()
                )

                residual_norm = (
                    residual_position
                    .float()
                    .pow(2)
                    .sum(dim=1)
                    .sqrt()
                )

                impact = (
                    (
                        grad_position.float()
                        *
                        residual_position.float()
                    )
                    .sum(dim=1)
                    .abs()
                )

                sensitivity[aug][layer][
                    "grad_norm"
                ].append(
                    float(grad_norm.mean())
                )

                sensitivity[aug][layer][
                    "impact"
                ].append(
                    float(impact.mean())
                )

                sensitivity[aug][layer][
                    "residual_norm"
                ].append(
                    float(residual_norm.mean())
                )

                if level != l3_level:
                    continue

                positions = residual_position.shape[0]

                count = min(
                    positions_per_image,
                    positions,
                )

                # Deterministic approximately uniform sampling.
                ids = torch.linspace(
                    0,
                    positions - 1,
                    steps=count,
                    device=device,
                ).long()

                residual_vectors[aug].append(
                    residual_position[
                        ids
                    ].detach().cpu()
                )

                task_weights[aug].append(
                    impact[
                        ids
                    ].detach().cpu()
                )

        print(
            f"[calibration {step}/{len(indices)}] "
            f"{sample['sample_id']}",
            flush=True,
        )

    bases = {}
    basis_report = []

    for aug in augmentations:
        matrix = torch.cat(
            residual_vectors[aug],
            dim=0,
        )

        weights = torch.cat(
            task_weights[aug],
            dim=0,
        )

        ordinary = fit_basis(
            matrix,
            max_rank=max_rank,
            weights=None,
        )

        task = fit_basis(
            matrix,
            max_rank=max_rank,
            weights=weights,
        )

        bases[(aug, "pca")] = ordinary
        bases[(aug, "task")] = task

        basis_report.append(
            {
                "augmentation": aug,
                "vectors": int(
                    matrix.shape[0]
                ),
                "pca": {
                    k: v
                    for k, v in ordinary.items()
                    if k not in ("mean", "basis")
                },
                "task": {
                    k: v
                    for k, v in task.items()
                    if k not in ("mean", "basis")
                },
                "task_weight_mean": float(
                    weights.mean()
                ),
                "task_weight_max": float(
                    weights.max()
                ),
            }
        )

    sensitivity_report = []

    for aug in augmentations:
        for layer in layers:
            item = sensitivity[aug][layer]

            sensitivity_report.append(
                {
                    "augmentation": aug,
                    "layer": layer,
                    "mean_grad_norm": float(
                        np.mean(
                            item["grad_norm"]
                        )
                    ),
                    "mean_first_order_impact": float(
                        np.mean(
                            item["impact"]
                        )
                    ),
                    "mean_residual_norm": float(
                        np.mean(
                            item["residual_norm"]
                        )
                    ),
                }
            )

    return (
        bases,
        basis_report,
        sensitivity_report,
    )


def main():
    args = parse_args()

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(args.device)

    cache_root = args.cache.resolve()

    manifest = json.loads(
        (
            cache_root / "manifest.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    samples = manifest["samples"]

    layers = tuple(
        int(v)
        for v
        in manifest["identity"]["layers"]
    )

    if tuple(args.layers) != layers:
        raise ValueError(
            f"layers mismatch: "
            f"{args.layers} vs {layers}"
        )

    if 3 not in layers:
        raise ValueError(
            "layer 3 must exist in cache"
        )

    imgsz = int(
        manifest["identity"]["imgsz"]
    )

    dataset_root = Path(
        manifest["dataset_root"]
    )

    data = check_det_dataset(
        args.dataset,
        autodownload=False,
    )

    name_values = data["names"]

    names = (
        {
            int(k): str(v)
            for k, v in name_values.items()
        }
        if isinstance(name_values, dict)
        else
        {
            i: str(v)
            for i, v in enumerate(name_values)
        }
    )

    nc = len(names)

    train_indices, val_indices = split_sample_indices(
        sample_count=len(samples),
        train_count=400,
        val_count=100,
        split_seed=args.split_seed,
    )

    calibration_indices = train_indices[
        :args.calibration_count
    ]

    evaluation_indices = val_indices[
        :args.eval_count
    ]

    first = samples[0]

    in_channels = tuple(
        int(
            first["shapes"][
                f"layer_{layer}"
            ][0]
        )
        for layer in layers
    )

    model = load_model(
        args.checkpoint,
        in_channels,
        nc,
        imgsz,
        device,
    )

    criterion = E2EDetectLoss(model)

    extractor = DINOv3MultiLevelExtractor(
        args.model,
        revision=args.revision,
        layers=layers,
        device=args.device,
        dtype=args.dtype,
        local_files_only=True,
    )

    max_rank = max(args.ranks)

    print(
        "Building task-sensitive L3 subspaces...",
        flush=True,
    )

    (
        bases,
        basis_report,
        sensitivity_report,
    ) = calibration(
        samples=samples,
        indices=calibration_indices,
        cache_root=cache_root,
        dataset_root=dataset_root,
        layers=layers,
        augmentations=args.augmentations,
        extractor=extractor,
        model=model,
        criterion=criterion,
        nc=nc,
        imgsz=imgsz,
        device=device,
        positions_per_image=args.positions_per_image,
        max_rank=max_rank,
    )

    torch.save(
        bases,
        args.output / "bases.pt",
    )

    (
        args.output / "layer-sensitivity.json"
    ).write_text(
        json.dumps(
            sensitivity_report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    (
        args.output / "basis-summary.json"
    ).write_text(
        json.dumps(
            basis_report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    modes = [
        "true",
        "transport",
        "exact_l3",
    ]

    for method in (
        "pca",
        "task",
    ):
        for rank in args.ranks:
            modes.append(
                f"{method}_r{rank}"
            )

    accumulators = {
        (aug, mode): MetricAccumulator(names)
        for aug in args.augmentations
        for mode in modes
    }

    l3_level = layers.index(3)

    print(
        "Starting task-sensitive oracle evaluation...",
        flush=True,
    )

    with torch.inference_mode():
        for step, index in enumerate(
            evaluation_indices,
            start=1,
        ):
            sample = samples[index]

            base = load_cached_features(
                cache_root,
                sample,
                layers,
                device,
            )

            image_path = image_path_from_sample(
                dataset_root,
                sample,
            )

            image, _ = load_letterboxed_tensor(
                image_path,
                imgsz=imgsz,
            )

            image = image.unsqueeze(0).to(device)

            classes, boxes = read_yolo_labels(
                label_path(
                    dataset_root,
                    sample["image_path"],
                ),
                sample["letterbox"],
                imgsz,
                nc,
            )

            for aug in args.augmentations:
                augmented = augment_image(
                    image,
                    aug,
                )

                true_dict = extractor(
                    augmented
                )

                true_features = tuple(
                    true_dict[
                        f"layer_{layer}"
                    ].float()
                    for layer in layers
                )

                transported = tuple(
                    transport_feature(
                        base[level],
                        aug,
                    ).float()
                    for level, layer
                    in enumerate(layers)
                )

                aug_cls, aug_boxes = transform_boxes(
                    classes,
                    boxes,
                    aug,
                )

                conditions = {
                    "true": true_features,
                    "transport": transported,
                }

                exact_l3 = list(transported)
                exact_l3[l3_level] = (
                    true_features[l3_level]
                )

                conditions["exact_l3"] = tuple(
                    exact_l3
                )

                residual_l3 = (
                    true_features[l3_level]
                    -
                    transported[l3_level]
                )

                for method in (
                    "pca",
                    "task",
                ):
                    info = bases[
                        (aug, method)
                    ]

                    for rank in args.ranks:
                        approx = project_residual(
                            residual_l3,
                            info["mean"],
                            info["basis"],
                            rank,
                        )

                        corrected = list(
                            transported
                        )

                        corrected[l3_level] = (
                            transported[l3_level]
                            +
                            approx
                        )

                        conditions[
                            f"{method}_r{rank}"
                        ] = tuple(
                            corrected
                        )

                for mode, features in conditions.items():
                    accumulators[
                        (aug, mode)
                    ].update(
                        model=model,
                        criterion=criterion,
                        features=features,
                        classes=aug_cls,
                        boxes=aug_boxes,
                        image_name=(
                            f"{sample['image_path']}"
                            f"::{aug}::{mode}"
                        ),
                        conf=args.conf,
                        device=device,
                    )

            print(
                f"[eval {step}/{len(evaluation_indices)}] "
                f"{sample['sample_id']}",
                flush=True,
            )

    records = []

    for aug in args.augmentations:
        for mode in modes:
            result = accumulators[
                (aug, mode)
            ].finish()

            records.append(
                {
                    "augmentation": aug,
                    "mode": mode,
                    **result,
                }
            )

    lookup = defaultdict(dict)

    for row in records:
        lookup[
            row["augmentation"]
        ][
            row["mode"]
        ] = row

    for row in records:
        group = lookup[
            row["augmentation"]
        ]

        true_ap = group[
            "true"
        ].get(
            "metrics/mAP50-95(B)",
            0.0,
        )

        transport_ap = group[
            "transport"
        ].get(
            "metrics/mAP50-95(B)",
            0.0,
        )

        current_ap = row.get(
            "metrics/mAP50-95(B)",
            0.0,
        )

        denominator = (
            true_ap
            -
            transport_ap
        )

        if abs(denominator) > 1e-12:
            recovery = (
                current_ap
                -
                transport_ap
            ) / denominator
        else:
            recovery = float("nan")

        true_loss = group[
            "true"
        ]["val/total_loss"]

        transport_loss = group[
            "transport"
        ]["val/total_loss"]

        current_loss = row[
            "val/total_loss"
        ]

        loss_den = (
            transport_loss
            -
            true_loss
        )

        if abs(loss_den) > 1e-12:
            loss_recovery = (
                transport_loss
                -
                current_loss
            ) / loss_den
        else:
            loss_recovery = float("nan")

        row["ap_gap_recovery"] = recovery
        row["loss_gap_recovery"] = (
            loss_recovery
        )

    with (
        args.output / "results.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                records[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(records)

    summary = {
        "status": "PASS",
        "calibration_count": args.calibration_count,
        "eval_count": args.eval_count,
        "ranks": args.ranks,
        "augmentations": args.augmentations,
        "basis": basis_report,
        "layer_sensitivity": sensitivity_report,
        "results": records,
    }

    (
        args.output / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 110)

    print(
        f'{"aug":12s}'
        f'{"mode":16s}'
        f'{"mAP50-95":>12s}'
        f'{"AP恢复":>12s}'
        f'{"loss":>13s}'
        f'{"损失恢复":>12s}'
    )

    print("-" * 110)

    for row in records:
        print(
            f'{row["augmentation"]:12s}'
            f'{row["mode"]:16s}'
            f'{row.get("metrics/mAP50-95(B)", 0):12.6f}'
            f'{row["ap_gap_recovery"]:12.4f}'
            f'{row["val/total_loss"]:13.6f}'
            f'{row["loss_gap_recovery"]:12.4f}'
        )

    print("=" * 110)
    print(
        "Task-sensitive L3 oracle: PASS"
    )


if __name__ == "__main__":
    main()
