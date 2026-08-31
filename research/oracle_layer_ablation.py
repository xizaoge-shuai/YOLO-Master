#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.d1_build_feature_cache import (
    DINOv3MultiLevelExtractor,
)

from scripts.d1_train_cached_detector import (
    CachedLatentDetector,
    label_path,
    match_predictions,
    read_yolo_labels,
    split_sample_indices,
)

from ultralytics.data.foundation_cache import (
    load_letterboxed_tensor,
)

from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.loss import E2EDetectLoss
from ultralytics.utils.metrics import DetMetrics
from ultralytics.utils.ops import xywh2xyxy


AUGMENTATIONS = (
    "brightness20",
    "hflip",
    "translate_x10",
    "zoom_in15",
)

LAYER_VARIANTS = (
    ("l3", (3,)),
    ("l7", (7,)),
    ("l11", (11,)),
    ("l3_l7", (3, 7)),
    ("l3_l11", (3, 11)),
    ("l7_l11", (7, 11)),
    ("all", (3, 7, 11)),
)

RANKS_BY_AUG = {
    "hflip": (32,),
    "zoom_in15": (16, 32),
}

MODES = ["true", "transport"]

for augmentation, ranks in RANKS_BY_AUG.items():
    for rank in ranks:
        for name, _ in LAYER_VARIANTS:
            MODES.append(
                f"{augmentation}_r{rank}_{name}"
            )

MODES = tuple(MODES)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)

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

    parser.add_argument(
        "--augmentations",
        nargs="+",
        default=list(AUGMENTATIONS),
    )

    parser.add_argument(
        "--calibration-count",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--eval-count",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--positions-per-image",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--max-rank",
        type=int,
        default=32,
    )

    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--output", type=Path, required=True)

    return parser.parse_args()


def affine_transform(
    tensor,
    *,
    sx=1.0,
    sy=1.0,
    tx=0.0,
    ty=0.0,
):
    batch = tensor.shape[0]

    theta = tensor.new_tensor(
        [
            [sx, 0.0, tx],
            [0.0, sy, ty],
        ]
    )

    theta = theta.unsqueeze(0).expand(
        batch,
        -1,
        -1,
    )

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


def augment_image(image, name):
    if name == "brightness20":
        return (image * 1.20).clamp(0.0, 1.0)

    if name == "hflip":
        return torch.flip(
            image,
            dims=(-1,),
        )

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

    raise ValueError(name)


def transport_feature(feature, name):
    if name == "brightness20":
        return feature

    if name == "hflip":
        return torch.flip(
            feature,
            dims=(-1,),
        )

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

    raise ValueError(name)


def transform_boxes(
    classes,
    boxes,
    name,
):
    if boxes.numel() == 0:
        return classes, boxes

    boxes = boxes.clone().float()
    classes = classes.clone()

    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2

    if name == "brightness20":
        pass

    elif name == "hflip":
        old_x1 = x1.clone()
        old_x2 = x2.clone()

        x1 = 1.0 - old_x2
        x2 = 1.0 - old_x1

    elif name == "translate_x10":
        # affine_grid tx=+0.20 causes image content
        # to move left by 10% in [0,1] coordinates.
        dx = -0.10

        x1 = x1 + dx
        x2 = x2 + dx

    elif name == "zoom_in15":
        scale = 1.0 / 0.85

        x1 = 0.5 + (x1 - 0.5) * scale
        x2 = 0.5 + (x2 - 0.5) * scale

        y1 = 0.5 + (y1 - 0.5) * scale
        y2 = 0.5 + (y2 - 0.5) * scale

    else:
        raise ValueError(name)

    x1 = x1.clamp(0.0, 1.0)
    y1 = y1.clamp(0.0, 1.0)
    x2 = x2.clamp(0.0, 1.0)
    y2 = y2.clamp(0.0, 1.0)

    new_w = x2 - x1
    new_h = y2 - y1

    keep = (
        (new_w > 1e-4)
        &
        (new_h > 1e-4)
    )

    classes = classes[keep]

    new_x = (x1 + x2) / 2
    new_y = (y1 + y2) / 2

    boxes = torch.stack(
        (
            new_x,
            new_y,
            new_w,
            new_h,
        ),
        dim=1,
    )[keep]

    return classes, boxes


def load_cached_features(
    cache_root,
    sample,
    layers,
    device,
):
    cache_file = (
        cache_root
        /
        sample["cache_path"]
    )

    tensors = load_file(
        str(cache_file),
        device="cpu",
    )

    return tuple(
        tensors[f"layer_{layer}"]
        .unsqueeze(0)
        .to(
            device=device,
            dtype=torch.float32,
        )
        for layer in layers
    )


def image_path_from_sample(
    dataset_root,
    sample,
):
    path = Path(sample["image_path"])

    if not path.is_absolute():
        path = dataset_root / path

    return path.resolve()


def collect_calibration_basis(
    *,
    samples,
    indices,
    cache_root,
    dataset_root,
    layers,
    augmentations,
    extractor,
    device,
    imgsz,
    positions_per_image,
    max_rank,
):
    vectors = defaultdict(list)

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

        image = (
            image
            .unsqueeze(0)
            .to(device)
        )

        for aug in augmentations:
            augmented = augment_image(
                image,
                aug,
            )

            true_dict = extractor(
                augmented
            )

            for level, layer in enumerate(layers):
                true_feature = (
                    true_dict[f"layer_{layer}"]
                    .float()
                )

                transported = (
                    transport_feature(
                        base[level],
                        aug,
                    )
                    .float()
                )

                residual = (
                    true_feature
                    -
                    transported
                )

                matrix = (
                    residual[0]
                    .permute(1, 2, 0)
                    .reshape(
                        -1,
                        residual.shape[1],
                    )
                )

                positions = matrix.shape[0]

                count = min(
                    positions_per_image,
                    positions,
                )

                ids = torch.linspace(
                    0,
                    positions - 1,
                    steps=count,
                    device=matrix.device,
                ).long()

                vectors[
                    (aug, layer)
                ].append(
                    matrix[ids]
                    .detach()
                    .cpu()
                )

        print(
            f"[calibration "
            f"{step}/{len(indices)}] "
            f"{sample['sample_id']}",
            flush=True,
        )

    basis = {}
    report = []

    for aug in augmentations:
        for layer in layers:
            matrix = torch.cat(
                vectors[(aug, layer)],
                dim=0,
            ).float()

            mean = matrix.mean(
                dim=0,
            )

            centered = (
                matrix
                -
                mean.unsqueeze(0)
            )

            _, singular, vh = torch.linalg.svd(
                centered,
                full_matrices=False,
            )

            rank = min(
                max_rank,
                vh.shape[0],
            )

            channels = (
                vh[:rank]
                .T
                .contiguous()
            )

            energy = singular.square()

            total = (
                energy.sum()
                .clamp_min(1e-12)
            )

            def energy_at(r):
                rr = min(
                    r,
                    energy.numel(),
                )

                return float(
                    energy[:rr].sum()
                    /
                    total
                )

            basis[
                (aug, layer)
            ] = {
                "mean": mean,
                "basis": channels,
            }

            report.append(
                {
                    "augmentation": aug,
                    "layer": layer,
                    "rank8_energy": energy_at(8),
                    "rank16_energy": energy_at(16),
                    "rank32_energy": energy_at(32),
                    "calibration_vectors": int(
                        matrix.shape[0]
                    ),
                }
            )

    return basis, report


def oracle_project(
    residual,
    mean,
    basis,
    rank,
):
    # residual: 1 x C x H x W

    _, channels, height, width = (
        residual.shape
    )

    matrix = (
        residual[0]
        .permute(1, 2, 0)
        .reshape(
            -1,
            channels,
        )
        .float()
    )

    mean = mean.to(
        matrix.device
    )

    basis = basis[
        :,
        :min(
            rank,
            basis.shape[1],
        )
    ].to(
        matrix.device
    )

    centered = (
        matrix
        -
        mean.unsqueeze(0)
    )

    coefficients = (
        centered
        @ basis
    )

    approximation = (
        mean.unsqueeze(0)
        +
        coefficients
        @ basis.T
    )

    approximation = (
        approximation
        .reshape(
            height,
            width,
            channels,
        )
        .permute(2, 0, 1)
        .unsqueeze(0)
        .contiguous()
    )

    return approximation


class MetricAccumulator:
    def __init__(self, names):
        self.metrics = DetMetrics(
            names=names
        )

        self.loss_sum = torch.zeros(
            3,
            dtype=torch.float64,
        )

        self.total_loss = 0.0
        self.samples = 0
        self.image_index = 0

    def update(
        self,
        *,
        model,
        criterion,
        features,
        classes,
        boxes,
        image_name,
        conf,
        device,
    ):
        targets = {
            "cls": classes.to(
                device
            ),
            "bboxes": boxes.to(
                device
            ),
            "batch_idx": torch.zeros(
                len(classes),
                dtype=torch.long,
                device=device,
            ),
        }

        raw = model.training_predictions(
            features
        )

        loss_vector, loss_items = criterion(
            raw,
            targets,
        )

        self.total_loss += float(
            loss_vector.sum()
        )

        self.loss_sum += (
            loss_items
            .detach()
            .double()
            .cpu()
        )

        output = model(features)

        if isinstance(
            output,
            (tuple, list),
        ):
            decoded = output[0]
        else:
            decoded = output

        predictions = decoded[0]

        predictions = predictions[
            predictions[:, 4] >= conf
        ]

        target_cls = (
            classes
            .view(-1)
            .to(device)
        )

        target_boxes = (
            boxes
            .to(device)
            *
            model.imgsz
        )

        target_boxes = xywh2xyxy(
            target_boxes
        )

        correct = match_predictions(
            predictions,
            target_cls,
            target_boxes,
        )

        count = len(target_cls)

        self.metrics.update_stats(
            {
                "tp": (
                    correct
                    .cpu()
                    .numpy()
                ),
                "conf": (
                    predictions[:, 4]
                    .float()
                    .cpu()
                    .numpy()
                ),
                "pred_cls": (
                    predictions[:, 5]
                    .float()
                    .cpu()
                    .numpy()
                ),
                "target_cls": (
                    target_cls
                    .float()
                    .cpu()
                    .numpy()
                ),
                "target_img": np.full(
                    count,
                    self.image_index,
                    dtype=np.int64,
                ),
                "im_name": image_name,
            }
        )

        self.samples += 1
        self.image_index += 1

    def finish(self):
        self.metrics.process(
            plot=False
        )

        results = {
            key: float(value)
            for key, value
            in self.metrics.results_dict.items()
            if key in self.metrics.keys
        }

        losses = (
            self.loss_sum
            /
            max(
                self.samples,
                1,
            )
        ).tolist()

        return {
            **results,
            "val/box_loss": losses[0],
            "val/cls_loss": losses[1],
            "val/dfl_loss": losses[2],
            "val/total_loss": (
                self.total_loss
                /
                max(
                    self.samples,
                    1,
                )
            ),
            "samples": self.samples,
        }


def main():
    args = parse_args()

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        args.device
    )

    cache_root = (
        args.cache
        .resolve()
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

    identity = manifest["identity"]

    layers = tuple(
        int(value)
        for value
        in identity["layers"]
    )

    if tuple(args.layers) != layers:
        raise ValueError(
            f"requested layers "
            f"{args.layers} != "
            f"cache layers {layers}"
        )

    imgsz = int(
        identity["imgsz"]
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
            int(key): str(value)
            for key, value
            in name_values.items()
        }
        if isinstance(
            name_values,
            dict,
        )
        else
        {
            index: str(value)
            for index, value
            in enumerate(name_values)
        }
    )

    nc = len(names)

    train_indices, val_indices = (
        split_sample_indices(
            sample_count=len(samples),
            train_count=400,
            val_count=100,
            split_seed=args.split_seed,
        )
    )

    if (
        args.calibration_count
        >
        len(train_indices)
    ):
        raise ValueError(
            "too many calibration samples"
        )

    if (
        args.eval_count
        >
        len(val_indices)
    ):
        raise ValueError(
            "too many evaluation samples"
        )

    calibration_indices = (
        train_indices[
            :args.calibration_count
        ]
    )

    evaluation_indices = (
        val_indices[
            :args.eval_count
        ]
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
        "Building calibration defect bases...",
        flush=True,
    )

    basis, basis_report = (
        collect_calibration_basis(
            samples=samples,
            indices=calibration_indices,
            cache_root=cache_root,
            dataset_root=dataset_root,
            layers=layers,
            augmentations=args.augmentations,
            extractor=extractor,
            device=device,
            imgsz=imgsz,
            positions_per_image=args.positions_per_image,
            max_rank=args.max_rank,
        )
    )

    torch.save(
        basis,
        args.output
        /
        "calibration-bases.pt",
    )

    (
        args.output
        /
        "calibration-summary.json"
    ).write_text(
        json.dumps(
            basis_report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
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

    model = CachedLatentDetector(
        in_channels=in_channels,
        nc=nc,
        imgsz=imgsz,
        epochs=100,
    ).to(device)

    payload = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    state = (
        payload.get("model")
        or
        payload.get("ema")
    )

    if state is None:
        raise RuntimeError(
            "checkpoint does not contain model"
        )

    if hasattr(
        state,
        "state_dict",
    ):
        state = state.state_dict()

    # The D1 training loop dynamically registers
    # _mixture_loss_ema_buf while collecting the LatentMixture
    # auxiliary loss. It is training bookkeeping rather than a
    # detector parameter and is not required for oracle inference.
    auxiliary_state_keys = [
        key
        for key in list(state.keys())
        if key == "_mixture_loss_ema_buf"
        or key.endswith("._mixture_loss_ema_buf")
    ]

    for key in auxiliary_state_keys:
        state.pop(key)

    missing_keys, unexpected_keys = model.load_state_dict(
        state,
        strict=False,
    )

    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "checkpoint mismatch after removing known "
            "training-only buffers: "
            f"missing={missing_keys}, "
            f"unexpected={unexpected_keys}"
        )

    print(
        "Checkpoint load: PASS; "
        f"dropped training-only keys={auxiliary_state_keys}",
        flush=True,
    )

    model.eval()

    criterion = E2EDetectLoss(
        model
    )

    accumulators = {}

    for aug in args.augmentations:
        valid_modes = [
            "true",
            "transport",
        ]

        if aug in RANKS_BY_AUG:
            for variant_name, _ in LAYER_VARIANTS:
                valid_modes.append(
                    f"{aug}_exact_{variant_name}"
                )

            for rank in RANKS_BY_AUG[aug]:
                for variant_name, _ in LAYER_VARIANTS:
                    valid_modes.append(
                        f"{aug}_r{rank}_{variant_name}"
                    )

        for mode in valid_modes:
            accumulators[
                (aug, mode)
            ] = MetricAccumulator(
                names
            )

    ranks = {
        "rank8": 8,
        "rank16": 16,
        "rank32": 32,
    }

    print(
        "Starting detection oracle evaluation...",
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

            image, _ = (
                load_letterboxed_tensor(
                    image_path,
                    imgsz=imgsz,
                )
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
                augmented_image = augment_image(
                    image,
                    aug,
                )

                true_dict = extractor(
                    augmented_image
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

                conditions = {
                    "true": true_features,
                    "transport": transported,
                }

                if aug in RANKS_BY_AUG:
                    for variant_name, exact_layers in LAYER_VARIANTS:
                        exact_features = tuple(
                            true_features[level]
                            if layer in exact_layers
                            else transported[level]
                            for level, layer in enumerate(layers)
                        )

                        conditions[
                            f"{aug}_exact_{variant_name}"
                        ] = exact_features

                if aug in RANKS_BY_AUG:
                    for rank in RANKS_BY_AUG[aug]:
                        for variant_name, corrected_layers in LAYER_VARIANTS:
                            corrected = []

                            for level, layer in enumerate(
                                layers
                            ):
                                if layer not in corrected_layers:
                                    corrected.append(
                                        transported[level]
                                    )
                                    continue

                                residual = (
                                    true_features[level]
                                    -
                                    transported[level]
                                )

                                info = basis[
                                    (aug, layer)
                                ]

                                approximation = oracle_project(
                                    residual,
                                    info["mean"],
                                    info["basis"],
                                    rank,
                                )

                                corrected.append(
                                    transported[level]
                                    +
                                    approximation
                                )

                            mode = (
                                f"{aug}_r{rank}_{variant_name}"
                            )

                            conditions[mode] = tuple(
                                corrected
                            )

                aug_cls, aug_boxes = transform_boxes(
                    classes,
                    boxes,
                    aug,
                )

                for mode, features in conditions.items():
                    image_name = (
                        f"{sample['image_path']}"
                        f"::{aug}"
                        f"::{mode}"
                    )

                    accumulators[
                        (aug, mode)
                    ].update(
                        model=model,
                        criterion=criterion,
                        features=features,
                        classes=aug_cls,
                        boxes=aug_boxes,
                        image_name=image_name,
                        conf=args.conf,
                        device=device,
                    )

            print(
                f"[eval "
                f"{step}/{len(evaluation_indices)}] "
                f"{sample['sample_id']}",
                flush=True,
            )

    records = []

    for aug in args.augmentations:
        valid_modes = [
            "true",
            "transport",
        ]

        if aug in RANKS_BY_AUG:
            for variant_name, _ in LAYER_VARIANTS:
                valid_modes.append(
                    f"{aug}_exact_{variant_name}"
                )

            for rank in RANKS_BY_AUG[aug]:
                for variant_name, _ in LAYER_VARIANTS:
                    valid_modes.append(
                        f"{aug}_r{rank}_{variant_name}"
                    )

        for mode in valid_modes:
            result = (
                accumulators[
                    (aug, mode)
                ].finish()
            )

            record = {
                "augmentation": aug,
                "mode": mode,
                **result,
            }

            records.append(record)

    true_lookup = {
        row["augmentation"]: row
        for row in records
        if row["mode"] == "true"
    }

    for row in records:
        true = true_lookup[
            row["augmentation"]
        ]

        row["map50_gap_to_true"] = (
            row.get(
                "metrics/mAP50(B)",
                0.0,
            )
            -
            true.get(
                "metrics/mAP50(B)",
                0.0,
            )
        )

        row["map50_95_gap_to_true"] = (
            row.get(
                "metrics/mAP50-95(B)",
                0.0,
            )
            -
            true.get(
                "metrics/mAP50-95(B)",
                0.0,
            )
        )

        row["loss_gap_to_true"] = (
            row["val/total_loss"]
            -
            true["val/total_loss"]
        )

    summary = {
        "status": "PASS",
        "scope": (
            "DINOv3 augmentation defect "
            "rank oracle detection evaluation"
        ),
        "cache": str(cache_root),
        "dataset": str(args.dataset),
        "checkpoint": str(
            args.checkpoint
        ),
        "model": args.model,
        "revision": args.revision,
        "layers": list(layers),
        "calibration_count": (
            args.calibration_count
        ),
        "eval_count": args.eval_count,
        "augmentations": (
            args.augmentations
        ),
        "basis": basis_report,
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

    csv_path = (
        args.output
        /
        "results.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(
                records[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            records
        )

    print()
    print("=" * 125)

    print(
        f'{"augmentation":16s}'
        f'{"mode":12s}'
        f'{"mAP50":>11s}'
        f'{"mAP50-95":>12s}'
        f'{"gap95":>11s}'
        f'{"loss":>12s}'
        f'{"loss_gap":>12s}'
    )

    print("-" * 125)

    for row in records:
        print(
            f'{row["augmentation"]:16s}'
            f'{row["mode"]:12s}'
            f'{row.get("metrics/mAP50(B)", 0.0):11.6f}'
            f'{row.get("metrics/mAP50-95(B)", 0.0):12.6f}'
            f'{row["map50_95_gap_to_true"]:11.6f}'
            f'{row["val/total_loss"]:12.6f}'
            f'{row["loss_gap_to_true"]:12.6f}'
        )

    print("=" * 125)
    print("Oracle rank detection evaluation: PASS")


if __name__ == "__main__":
    main()
