#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.d1_build_feature_cache import (
    DINOv3MultiLevelExtractor,
    collect_image_paths,
)
from ultralytics.data.foundation_cache import (
    load_letterboxed_tensor,
    select_images,
)
from ultralytics.data.utils import check_det_dataset


AUGMENTATIONS = (
    "hflip",
    "translate_x10",
    "zoom_in15",
    "brightness20",
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=100)

    parser.add_argument("--imgsz", type=int, default=640)

    parser.add_argument(
        "--model",
        default="Tooony133/dinov3-vits16-pretrain-lvd1689m",
    )
    parser.add_argument("--revision", required=True)

    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=[3, 7, 11],
    )

    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="fp16")

    parser.add_argument(
        "--augmentations",
        nargs="+",
        default=list(AUGMENTATIONS),
    )

    parser.add_argument(
        "--svd-positions",
        type=int,
        default=16,
    )

    parser.add_argument("--output", required=True)

    return parser.parse_args()


def affine_transform(
    tensor: torch.Tensor,
    *,
    sx: float = 1.0,
    sy: float = 1.0,
    tx: float = 0.0,
    ty: float = 0.0,
):
    batch = tensor.shape[0]

    theta = tensor.new_tensor(
        [
            [sx, 0.0, tx],
            [0.0, sy, ty],
        ]
    )

    theta = theta.unsqueeze(0).expand(batch, -1, -1)

    grid = F.affine_grid(
        theta,
        tensor.shape,
        align_corners=False,
    )

    return F.grid_sample(
        tensor,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )


def augment_image(
    image: torch.Tensor,
    name: str,
):
    if name == "hflip":
        return torch.flip(image, dims=(-1,))

    if name == "translate_x10":
        return affine_transform(
            image,
            tx=0.20,
        )

    if name == "zoom_in15":
        return affine_transform(
            image,
            sx=0.85,
            sy=0.85,
        )

    if name == "brightness20":
        return (image * 1.20).clamp(0.0, 1.0)

    raise ValueError(
        f"unsupported augmentation: {name}"
    )


def transport_feature(
    feature: torch.Tensor,
    name: str,
):
    if name == "hflip":
        return torch.flip(feature, dims=(-1,))

    if name == "translate_x10":
        return affine_transform(
            feature,
            tx=0.20,
        )

    if name == "zoom_in15":
        return affine_transform(
            feature,
            sx=0.85,
            sy=0.85,
        )

    if name == "brightness20":
        # 颜色变化没有解析的空间迁移，
        # 因此这里把原特征作为零阶近似。
        return feature

    raise ValueError(
        f"unsupported augmentation: {name}"
    )


def spatial_concentration_90(
    residual: torch.Tensor,
):
    # C x H x W
    energy = (
        residual.float()
        .pow(2)
        .sum(dim=0)
        .flatten()
    )

    total = energy.sum()

    if total <= 0:
        return 0.0

    values = torch.sort(
        energy,
        descending=True,
    ).values

    cumulative = torch.cumsum(values, dim=0)

    target = 0.90 * total

    count = int(
        torch.searchsorted(
            cumulative,
            target,
        ).item()
    ) + 1

    return count / energy.numel()


def mean(values):
    return (
        float(statistics.mean(values))
        if values
        else 0.0
    )


def main():
    args = parse_args()

    output = Path(args.output)
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(args.device)

    data = check_det_dataset(
        args.dataset,
        split=args.split,
    )

    image_paths = collect_image_paths(
        data[args.split]
    )

    selected = select_images(
        image_paths,
        root=data["path"],
        limit=args.limit,
    )

    extractor = DINOv3MultiLevelExtractor(
        args.model,
        revision=args.revision,
        layers=tuple(args.layers),
        device=args.device,
        dtype=args.dtype,
    )

    rows = []

    residual_samples = defaultdict(list)

    for index, item in enumerate(
        selected,
        start=1,
    ):
        image, _ = load_letterboxed_tensor(
            item.path,
            imgsz=args.imgsz,
        )

        image = (
            image
            .unsqueeze(0)
            .to(device)
        )

        base = extractor(image)

        for aug_name in args.augmentations:
            augmented_image = augment_image(
                image,
                aug_name,
            )

            true_augmented = extractor(
                augmented_image
            )

            for layer in args.layers:
                key = f"layer_{layer}"

                true_feature = (
                    true_augmented[key]
                    .float()
                )

                transported = (
                    transport_feature(
                        base[key],
                        aug_name,
                    )
                    .float()
                )

                residual = (
                    true_feature
                    -
                    transported
                )

                true_norm = (
                    true_feature
                    .norm()
                    .clamp_min(1e-12)
                )

                relative_l2 = (
                    residual.norm()
                    /
                    true_norm
                ).item()

                cosine = (
                    F.cosine_similarity(
                        true_feature.flatten(),
                        transported.flatten(),
                        dim=0,
                    )
                    .item()
                )

                residual_ratio = (
                    residual.pow(2).sum()
                    /
                    true_feature.pow(2)
                    .sum()
                    .clamp_min(1e-12)
                ).item()

                spatial90 = (
                    spatial_concentration_90(
                        residual[0]
                    )
                )

                rows.append(
                    {
                        "sample_index": index,
                        "sample_id": item.sample_id,
                        "augmentation": aug_name,
                        "layer": layer,
                        "relative_l2": relative_l2,
                        "residual_energy_ratio": residual_ratio,
                        "cosine_similarity": cosine,
                        "spatial_fraction_for_90pct_energy": spatial90,
                    }
                )

                r = (
                    residual[0]
                    .detach()
                    .float()
                    .reshape(
                        residual.shape[1],
                        -1,
                    )
                )

                positions = r.shape[1]
                count = min(
                    args.svd_positions,
                    positions,
                )

                ids = torch.linspace(
                    0,
                    positions - 1,
                    steps=count,
                    device=r.device,
                ).long()

                sampled = (
                    r[:, ids]
                    .T
                    .cpu()
                )

                residual_samples[
                    (aug_name, layer)
                ].append(sampled)

        print(
            f"[{index}/{len(selected)}] "
            f"{item.relative_path}",
            flush=True,
        )

    metrics_csv = output / "per-sample-metrics.csv"

    with metrics_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
        )

        writer.writeheader()
        writer.writerows(rows)

    grouped = defaultdict(list)

    for row in rows:
        grouped[
            (
                row["augmentation"],
                row["layer"],
            )
        ].append(row)

    summary = {
        "model": args.model,
        "revision": args.revision,
        "dataset": str(args.dataset),
        "sample_count": len(selected),
        "imgsz": args.imgsz,
        "layers": args.layers,
        "augmentations": args.augmentations,
        "results": [],
    }

    for (
        aug_name,
        layer,
    ), group in sorted(grouped.items()):

        matrix = torch.cat(
            residual_samples[
                (aug_name, layer)
            ],
            dim=0,
        )

        # 中心化后看通道残差的有效秩。
        matrix = (
            matrix
            -
            matrix.mean(
                dim=0,
                keepdim=True,
            )
        )

        singular = torch.linalg.svdvals(
            matrix
        )

        energy = singular.square()

        total_energy = (
            energy.sum()
            .clamp_min(1e-12)
        )

        def top_energy(k):
            k = min(
                k,
                energy.numel(),
            )

            return float(
                energy[:k].sum()
                /
                total_energy
            )

        record = {
            "augmentation": aug_name,
            "layer": layer,
            "mean_relative_l2": mean(
                [
                    row["relative_l2"]
                    for row in group
                ]
            ),
            "mean_residual_energy_ratio": mean(
                [
                    row[
                        "residual_energy_ratio"
                    ]
                    for row in group
                ]
            ),
            "mean_cosine_similarity": mean(
                [
                    row[
                        "cosine_similarity"
                    ]
                    for row in group
                ]
            ),
            "mean_spatial_fraction_for_90pct_energy": mean(
                [
                    row[
                        "spatial_fraction_for_90pct_energy"
                    ]
                    for row in group
                ]
            ),
            "rank8_energy": top_energy(8),
            "rank16_energy": top_energy(16),
            "rank32_energy": top_energy(32),
            "svd_vector_count": int(
                matrix.shape[0]
            ),
        }

        summary["results"].append(record)

    summary_path = output / "summary.json"

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
