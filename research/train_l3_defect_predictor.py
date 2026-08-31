#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.d1_build_feature_cache import DINOv3MultiLevelExtractor
from scripts.d1_train_cached_detector import (
    label_path,
    read_yolo_labels,
    split_sample_indices,
)
from research.oracle_rank_detection import (
    MetricAccumulator,
    augment_image,
    image_path_from_sample,
    load_cached_features,
    transform_boxes,
    transport_feature,
)
from research.task_sensitive_l3_oracle import (
    calibration,
    load_model,
    project_residual,
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

    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--hidden", type=int, default=128)

    p.add_argument("--calibration-count", type=int, default=50)
    p.add_argument("--calibration-seed", type=int, default=0)
    p.add_argument("--eval-count", type=int, default=100)

    p.add_argument("--train-count", type=int, default=400)
    p.add_argument("--val-count", type=int, default=100)

    p.add_argument("--positions-per-image", type=int, default=128)

    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)

    p.add_argument("--coordinates", action="store_true")

    p.add_argument(
        "--context",
        choices=(
            "point",
            "local3",
            "local5",
            "multiscale",
            "global",
            "multiscale_global",
        ),
        default="point",
    )

    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="fp16")
    p.add_argument("--conf", type=float, default=0.001)

    p.add_argument("--output", type=Path, required=True)

    return p.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def coordinate_features(height, width, device):
    yy = torch.linspace(
        -1.0,
        1.0,
        height,
        device=device,
    )

    xx = torch.linspace(
        -1.0,
        1.0,
        width,
        device=device,
    )

    y, x = torch.meshgrid(
        yy,
        xx,
        indexing="ij",
    )

    return torch.stack(
        (x, y),
        dim=-1,
    ).reshape(-1, 2)


class CoefficientPredictor(nn.Module):
    def __init__(self, in_dim, rank, hidden):
        super().__init__()

        if hidden <= 0:
            self.net = nn.Linear(
                in_dim,
                rank,
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(
                    in_dim,
                    hidden,
                ),
                nn.GELU(),
                nn.Linear(
                    hidden,
                    rank,
                ),
            )

    def forward(self, x):
        return self.net(x)


def make_input(
    feature,
    coordinates,
    context,
):
    _, channels, height, width = feature.shape

    maps = [feature.float()]

    if context in (
        "local3",
        "multiscale",
        "multiscale_global",
    ):
        maps.append(
            F.avg_pool2d(
                feature.float(),
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )

    if context in (
        "local5",
        "multiscale",
        "multiscale_global",
    ):
        maps.append(
            F.avg_pool2d(
                feature.float(),
                kernel_size=5,
                stride=1,
                padding=2,
            )
        )

    if context in (
        "global",
        "multiscale_global",
    ):
        global_feature = F.adaptive_avg_pool2d(
            feature.float(),
            output_size=1,
        ).expand(
            -1,
            -1,
            height,
            width,
        )

        maps.append(
            global_feature
        )

    merged = torch.cat(
        maps,
        dim=1,
    )

    x = (
        merged[0]
        .permute(1, 2, 0)
        .reshape(
            -1,
            merged.shape[1],
        )
    )

    if coordinates:
        coords = coordinate_features(
            height,
            width,
            x.device,
        )

        x = torch.cat(
            (
                x,
                coords,
            ),
            dim=1,
        )

    return x


def residual_coefficients(
    residual,
    mean,
    basis,
):
    _, channels, _, _ = residual.shape

    matrix = (
        residual[0]
        .permute(1, 2, 0)
        .reshape(-1, channels)
        .float()
    )

    mean = mean.to(matrix.device)
    basis = basis.to(matrix.device)

    return (
        matrix
        -
        mean.unsqueeze(0)
    ) @ basis


def reconstruct_residual(
    coefficients,
    mean,
    basis,
    height,
    width,
):
    mean = mean.to(coefficients.device)
    basis = basis.to(coefficients.device)

    matrix = (
        mean.unsqueeze(0)
        +
        coefficients @ basis.T
    )

    channels = matrix.shape[1]

    return (
        matrix
        .reshape(
            height,
            width,
            channels,
        )
        .permute(2, 0, 1)
        .unsqueeze(0)
        .contiguous()
    )


def collect_pairs(
    *,
    samples,
    indices,
    cache_root,
    dataset_root,
    layers,
    augmentations,
    extractor,
    bases,
    imgsz,
    device,
    coordinates,
    context,
):
    level = layers.index(3)

    features = {
        aug: []
        for aug in augmentations
    }

    targets = {
        aug: []
        for aug in augmentations
    }

    for step, index in enumerate(
        indices,
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

        image = image.unsqueeze(0).to(
            device
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

            true_l3 = true_dict[
                "layer_3"
            ].float()

            transported_l3 = (
                transport_feature(
                    base[level],
                    aug,
                )
                .float()
            )

            residual = (
                true_l3
                -
                transported_l3
            )

            info = bases[
                (aug, "task")
            ]

            x = make_input(
                transported_l3,
                coordinates,
                context,
            )

            y = residual_coefficients(
                residual,
                info["mean"],
                info["basis"],
            )

            features[aug].append(
                x.detach().cpu()
            )

            targets[aug].append(
                y.detach().cpu()
            )

        print(
            f"[pairs {step}/{len(indices)}] "
            f"{sample['sample_id']}",
            flush=True,
        )

    result = {}

    for aug in augmentations:
        result[aug] = (
            torch.cat(
                features[aug],
                dim=0,
            ),
            torch.cat(
                targets[aug],
                dim=0,
            ),
        )

    return result


def train_predictor(
    x,
    y,
    *,
    rank,
    hidden,
    epochs,
    batch_size,
    lr,
    weight_decay,
    seed,
    device,
):
    generator = torch.Generator().manual_seed(
        seed
    )

    order = torch.randperm(
        len(x),
        generator=generator,
    )

    split = max(
        1,
        int(
            0.9
            *
            len(order)
        ),
    )

    train_ids = order[:split]
    val_ids = order[split:]

    x_mean = x[train_ids].mean(
        dim=0
    )

    x_std = x[train_ids].std(
        dim=0
    ).clamp_min(1e-5)

    y_mean = y[train_ids].mean(
        dim=0
    )

    y_std = y[train_ids].std(
        dim=0
    ).clamp_min(1e-5)

    x_norm = (
        x
        -
        x_mean
    ) / x_std

    y_norm = (
        y
        -
        y_mean
    ) / y_std

    dataset = TensorDataset(
        x_norm[train_ids],
        y_norm[train_ids],
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    predictor = CoefficientPredictor(
        in_dim=x.shape[1],
        rank=rank,
        hidden=hidden,
    ).to(device)

    optimizer = torch.optim.AdamW(
        predictor.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=lr * 0.01,
        )
    )

    best_loss = float("inf")
    best_state = None
    history = []

    for epoch in range(
        1,
        epochs + 1,
    ):
        predictor.train()

        train_sum = 0.0
        count = 0

        for bx, by in loader:
            bx = bx.to(
                device,
                non_blocking=True,
            )

            by = by.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            prediction = predictor(
                bx
            )

            loss = torch.nn.functional.mse_loss(
                prediction,
                by,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                predictor.parameters(),
                10.0,
            )

            optimizer.step()

            train_sum += (
                float(loss.detach())
                *
                len(bx)
            )

            count += len(bx)

        predictor.eval()

        with torch.inference_mode():
            if len(val_ids):
                vx = x_norm[
                    val_ids
                ].to(device)

                vy = y_norm[
                    val_ids
                ].to(device)

                val_loss = float(
                    torch.nn.functional.mse_loss(
                        predictor(vx),
                        vy,
                    )
                )
            else:
                val_loss = (
                    train_sum
                    /
                    max(count, 1)
                )

        train_loss = (
            train_sum
            /
            max(count, 1)
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

        if val_loss <= best_loss:
            best_loss = val_loss

            best_state = {
                key: value.detach().cpu().clone()
                for key, value
                in predictor.state_dict().items()
            }

        scheduler.step()

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == epochs
        ):
            print(
                f"epoch={epoch:03d} "
                f"train={train_loss:.6f} "
                f"val={val_loss:.6f}",
                flush=True,
            )

    predictor.load_state_dict(
        best_state
    )

    predictor.eval()

    stats = {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }

    return (
        predictor,
        stats,
        history,
    )


def predict_coefficients(
    predictor,
    stats,
    feature,
    coordinates,
    context,
):
    x = make_input(
        feature,
        coordinates,
        context,
    )

    x_mean = stats[
        "x_mean"
    ].to(x.device)

    x_std = stats[
        "x_std"
    ].to(x.device)

    y_mean = stats[
        "y_mean"
    ].to(x.device)

    y_std = stats[
        "y_std"
    ].to(x.device)

    normalized = (
        x
        -
        x_mean
    ) / x_std

    output = predictor(
        normalized
    )

    return (
        output
        *
        y_std
        +
        y_mean
    )


def main():
    args = parse_args()

    seed_everything(
        args.seed
    )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        args.device
    )

    cache_root = (
        args.cache.resolve()
    )

    manifest = json.loads(
        (
            cache_root
            /
            "manifest.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    samples = manifest["samples"]

    layers = tuple(
        int(v)
        for v
        in manifest[
            "identity"
        ]["layers"]
    )

    if tuple(args.layers) != layers:
        raise ValueError(
            "layer mismatch"
        )

    imgsz = int(
        manifest[
            "identity"
        ]["imgsz"]
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
            for k, v
            in name_values.items()
        }
        if isinstance(
            name_values,
            dict,
        )
        else
        {
            i: str(v)
            for i, v
            in enumerate(
                name_values
            )
        }
    )

    nc = len(names)

    train_indices, val_indices = (
        split_sample_indices(
            sample_count=len(samples),
            train_count=args.train_count,
            val_count=args.val_count,
            split_seed=args.split_seed,
        )
    )

    if args.calibration_count > len(train_indices):
        raise ValueError(
            "calibration-count exceeds training split"
        )

    if args.eval_count > len(val_indices):
        raise ValueError(
            "eval-count exceeds validation split"
        )

    generator = torch.Generator().manual_seed(
        args.calibration_seed
    )

    order = torch.randperm(
        len(train_indices),
        generator=generator,
    ).tolist()

    calibration_indices = [
        train_indices[i]
        for i in order[
            :args.calibration_count
        ]
    ]

    evaluation_indices = (
        val_indices[
            :args.eval_count
        ]
    )

    first = samples[0]

    in_channels = tuple(
        int(
            first["shapes"][
                f"layer_{layer}"
            ][0]
        )
        for layer in layers
    )

    detector = load_model(
        args.checkpoint,
        in_channels,
        nc,
        imgsz,
        device,
    )

    criterion = E2EDetectLoss(
        detector
    )

    extractor = DINOv3MultiLevelExtractor(
        args.model,
        revision=args.revision,
        layers=layers,
        device=args.device,
        dtype=args.dtype,
        local_files_only=True,
    )

    print(
        "Stage 1: fitting task-sensitive bases",
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
        model=detector,
        criterion=criterion,
        nc=nc,
        imgsz=imgsz,
        device=device,
        positions_per_image=args.positions_per_image,
        max_rank=args.rank,
    )

    print(
        "Stage 2: collecting coefficient targets",
        flush=True,
    )

    pairs = collect_pairs(
        samples=samples,
        indices=calibration_indices,
        cache_root=cache_root,
        dataset_root=dataset_root,
        layers=layers,
        augmentations=args.augmentations,
        extractor=extractor,
        bases=bases,
        imgsz=imgsz,
        device=device,
        coordinates=args.coordinates,
        context=args.context,
    )

    predictors = {}
    predictor_stats = {}
    histories = {}

    for aug in args.augmentations:
        print(
            f"Stage 3: training predictor for {aug}",
            flush=True,
        )

        x, y = pairs[aug]

        (
            predictor,
            stats,
            history,
        ) = train_predictor(
            x,
            y,
            rank=args.rank,
            hidden=args.hidden,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            device=device,
        )

        predictors[aug] = predictor
        predictor_stats[aug] = stats
        histories[aug] = history

        torch.save(
            {
                "state_dict": predictor.state_dict(),
                "stats": stats,
                "rank": args.rank,
                "hidden": args.hidden,
                "coordinates": args.coordinates,
                "context": args.context,
            },
            args.output
            /
            f"predictor-{aug}.pt",
        )

    del pairs

    modes = (
        "true",
        "transport",
        "mean_only",
        "oracle",
        "predicted",
    )

    accumulators = {
        (aug, mode): MetricAccumulator(
            names
        )
        for aug in args.augmentations
        for mode in modes
    }

    coefficient_errors = {
        aug: []
        for aug in args.augmentations
    }

    l3_level = layers.index(3)

    print(
        "Stage 4: held-out evaluation",
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

            image = (
                image
                .unsqueeze(0)
                .to(device)
            )

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

                true_l3 = (
                    true_features[
                        l3_level
                    ]
                )

                transported_l3 = (
                    transported[
                        l3_level
                    ]
                )

                residual = (
                    true_l3
                    -
                    transported_l3
                )

                info = bases[
                    (aug, "task")
                ]

                oracle_residual = project_residual(
                    residual,
                    info["mean"],
                    info["basis"],
                    args.rank,
                )

                _, _, height, width = (
                    transported_l3.shape
                )

                predicted_coeff = predict_coefficients(
                    predictors[aug],
                    predictor_stats[aug],
                    transported_l3,
                    args.coordinates,
                    args.context,
                )

                predicted_residual = reconstruct_residual(
                    predicted_coeff,
                    info["mean"],
                    info["basis"],
                    height,
                    width,
                )

                true_coeff = residual_coefficients(
                    residual,
                    info["mean"],
                    info["basis"],
                )

                coefficient_errors[
                    aug
                ].append(
                    float(
                        torch.nn.functional.mse_loss(
                            predicted_coeff,
                            true_coeff,
                        )
                    )
                )

                mean_residual = (
                    info["mean"]
                    .to(device)
                    .view(
                        1,
                        -1,
                        1,
                        1,
                    )
                    .expand_as(
                        transported_l3
                    )
                )

                exact = list(
                    transported
                )

                exact[l3_level] = (
                    true_l3
                )

                oracle = list(
                    transported
                )

                oracle[l3_level] = (
                    transported_l3
                    +
                    oracle_residual
                )

                predicted = list(
                    transported
                )

                predicted[l3_level] = (
                    transported_l3
                    +
                    predicted_residual
                )

                mean_only = list(
                    transported
                )

                mean_only[l3_level] = (
                    transported_l3
                    +
                    mean_residual
                )

                conditions = {
                    "true": true_features,
                    "transport": transported,
                    "mean_only": tuple(mean_only),
                    "oracle": tuple(oracle),
                    "predicted": tuple(predicted),
                }

                for mode, features in conditions.items():
                    accumulators[
                        (aug, mode)
                    ].update(
                        model=detector,
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

    lookup = {}

    for row in records:
        lookup[
            (
                row["augmentation"],
                row["mode"],
            )
        ] = row

    for row in records:
        aug = row["augmentation"]

        true = lookup[
            (aug, "true")
        ]

        transport = lookup[
            (aug, "transport")
        ]

        true_ap = float(
            true[
                "metrics/mAP50-95(B)"
            ]
        )

        transport_ap = float(
            transport[
                "metrics/mAP50-95(B)"
            ]
        )

        current_ap = float(
            row[
                "metrics/mAP50-95(B)"
            ]
        )

        denominator = (
            true_ap
            -
            transport_ap
        )

        row[
            "ap_gap_recovery"
        ] = (
            (
                current_ap
                -
                transport_ap
            )
            /
            denominator
            if abs(denominator) > 1e-12
            else float("nan")
        )

        true_loss = float(
            true["val/total_loss"]
        )

        transport_loss = float(
            transport[
                "val/total_loss"
            ]
        )

        current_loss = float(
            row[
                "val/total_loss"
            ]
        )

        loss_den = (
            transport_loss
            -
            true_loss
        )

        row[
            "loss_gap_recovery"
        ] = (
            (
                transport_loss
                -
                current_loss
            )
            /
            loss_den
            if abs(loss_den) > 1e-12
            else float("nan")
        )

    with (
        args.output
        /
        "results.csv"
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

    parameter_count = sum(
        sum(
            p.numel()
            for p in predictor.parameters()
        )
        for predictor in predictors.values()
    )

    summary = {
        "status": "PASS",
        "deployable_path_uses_dinov3": False,
        "dinov3_used_only_for_calibration_and_reference_evaluation": True,
        "rank": args.rank,
        "hidden": args.hidden,
        "coordinates": args.coordinates,
        "context": args.context,
        "train_count": args.train_count,
        "val_count": args.val_count,
        "calibration_count": args.calibration_count,
        "calibration_seed": args.calibration_seed,
        "training_seed": args.seed,
        "eval_count": args.eval_count,
        "predictor_parameters_total": parameter_count,
        "coefficient_mse": {
            aug: float(
                np.mean(
                    coefficient_errors[aug]
                )
            )
            for aug in args.augmentations
        },
        "basis": basis_report,
        "layer_sensitivity": sensitivity_report,
        "training_history": histories,
        "results": records,
    }

    (
        args.output
        /
        "summary.json"
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
    print("=" * 100)

    print(
        f'{"augmentation":14s}'
        f'{"mode":14s}'
        f'{"mAP50-95":>12s}'
        f'{"AP恢复":>12s}'
        f'{"loss":>13s}'
        f'{"损失恢复":>12s}'
    )

    print("-" * 100)

    for row in records:
        print(
            f'{row["augmentation"]:14s}'
            f'{row["mode"]:14s}'
            f'{float(row["metrics/mAP50-95(B)"]):12.6f}'
            f'{float(row["ap_gap_recovery"]):12.4f}'
            f'{float(row["val/total_loss"]):13.6f}'
            f'{float(row["loss_gap_recovery"]):12.4f}'
        )

    print("=" * 100)
    print(
        f"Predictor parameters: {parameter_count}"
    )
    print(
        "Deployable predictor experiment: PASS"
    )


if __name__ == "__main__":
    main()
