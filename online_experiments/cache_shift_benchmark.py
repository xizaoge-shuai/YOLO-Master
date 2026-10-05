#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from online_experiments.d1_online_compare import (
    geometry,
    training_feature,
)
from online_experiments.feature_baselines import (
    warp_features,
)
from online_experiments.full_visdrone_data import (
    NAMES,
    dump,
    read,
    sha,
    verify_inputs_and_cache,
)
from online_experiments.full_visdrone_train import (
    Batches,
)
from online_experiments.fusion_probe import (
    configure_fusion,
)
from scripts import d1_train_cached_detector as d
from ultralytics.data.foundation_cache import (
    load_letterboxed_tensor,
)


VIEWS = [
    ("identity", False, 1.00, "anchor"),
    ("flip",     True,  1.00, "anchor"),
    ("scale095", False, 0.95, "mild"),
    ("scale105", False, 1.05, "mild"),
    ("scale085", False, 0.85, "hard"),
    ("scale115", False, 1.15, "hard"),
]


def sync():
    torch.cuda.synchronize()


def load_head(checkpoint):
    state = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    if state["epoch"] != 100:
        raise RuntimeError(
            f"{checkpoint}: expected epoch100, got {state['epoch']}"
        )

    model = d.CachedLatentDetector(
        in_channels=(384,) * 3,
        nc=10,
        imgsz=640,
        epochs=100,
    ).cuda()

    configure_fusion(
        model,
        "mean_value",
    )

    model.detect.max_det = 500

    model.load_state_dict(
        state["model"],
        strict=True,
    )

    model.eval()
    model.requires_grad_(False)

    return model


class EvalState:
    def __init__(self, model):
        self.model = model
        self.metric = d.DetMetrics(
            names=dict(enumerate(NAMES))
        )
        self.criterion = d.E2EDetectLoss(
            model
        )
        self.loss_sum = 0.0
        self.count = 0

    @torch.no_grad()
    def update(
        self,
        features,
        items,
    ):
        # collate_cached() requires each sample to contain
        # a feature tuple. The benchmark keeps features
        # separate so that the same targets can be evaluated
        # with Exact and Transport routes. Attach the current
        # route's features only to temporary item copies.
        collate_items = []

        for j, item in enumerate(items):
            one = dict(item)
            one["features"] = tuple(
                level[j]
                for level in features
            )
            collate_items.append(one)

        collated = d.collate_cached(
            collate_items
        )

        _, targets = d.move_batch(
            collated,
            torch.device("cuda:0"),
        )

        _, components = self.criterion(
            self.model.training_predictions(
                features
            ),
            targets,
        )

        self.loss_sum += (
            float(
                components
                .sum()
                .detach()
            )
            * len(items)
        )

        decoded, _ = self.model(
            features
        )

        for j, item in enumerate(items):
            classes = (
                item["cls"]
                .reshape(-1)
                .cuda()
            )

            boxes = (
                item["bboxes"]
                .reshape(-1, 4)
                .cuda()
            )

            pred = decoded[j]

            pred = pred[
                pred[:, 4] >= 0.001
            ]

            correct = d.match_predictions(
                pred,
                classes,
                d.xywh2xyxy(
                    boxes * 640
                ),
            )

            self.metric.update_stats(
                {
                    "tp":
                        correct.cpu().numpy(),
                    "conf":
                        pred[:, 4]
                        .float()
                        .cpu()
                        .numpy(),
                    "pred_cls":
                        pred[:, 5]
                        .float()
                        .cpu()
                        .numpy(),
                    "target_cls":
                        classes
                        .float()
                        .cpu()
                        .numpy(),
                    "target_img":
                        np.full(
                            len(classes),
                            self.count,
                        ),
                    "im_name":
                        item["image_path"],
                }
            )

            self.count += 1

    def finish(self):
        self.metric.process(
            plot=False
        )

        r = {
            k: float(v)
            for k, v
            in self.metric.results_dict.items()
            if k in self.metric.keys
        }

        return {
            "AP50":
                100.0
                * r[
                    "metrics/mAP50(B)"
                ],
            "AP":
                100.0
                * r[
                    "metrics/mAP50-95(B)"
                ],
            "loss":
                self.loss_sum
                / self.count,
            "images":
                self.count,
        }


def rel_mse(
    approximate,
    exact,
):
    values = []

    for a, e in zip(
        approximate,
        exact,
    ):
        num = (
            (
                a.float()
                - e.float()
            )
            .square()
            .mean(
                dim=(1, 2, 3)
            )
        )

        den = (
            e.float()
            .square()
            .mean(
                dim=(1, 2, 3)
            )
            .clamp_min(1e-8)
        )

        values.extend(
            (num / den)
            .detach()
            .cpu()
            .tolist()
        )

    return values


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--source",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--pareto",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--batch",
        type=int,
        default=4,
    )

    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 means all 548 validation images",
    )

    a = p.parse_args()

    source = a.source.resolve()
    pareto = a.pareto.resolve()
    output = a.output.resolve()

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    # The launcher creates the fresh job directory with mktemp -d.
    # Refuse to overwrite an experiment that has already produced outputs.
    protected = (
        "status.txt",
        "provenance.json",
        "results.json",
        "summary.json",
        "comparison.txt",
    )

    existing = [
        name
        for name in protected
        if (output / name).exists()
    ]

    if existing:
        raise RuntimeError(
            f"Output already contains experiment artifacts: {existing}"
        )

    status_path = (
        output
        / "status.txt"
    )

    def status(x):
        status_path.write_text(
            x + "\n",
            encoding="utf-8",
        )
        print(
            x,
            flush=True,
        )

    try:
        status(
            "RUNNING preflight"
        )

        torch.set_num_threads(4)
        d.seed_everything(0)

        verify_inputs_and_cache(
            source
        )

        heads = {
            0:
                source
                / "train/cache2-s0/weights/last.pt",
            10:
                pareto
                / "train/mix10-s0/weights/last.pt",
            25:
                source
                / "train/mix25-s0/weights/last.pt",
            50:
                pareto
                / "train/mix50-s0/weights/last.pt",
            100:
                source
                / "train/online-geom-s0/weights/last.pt",
        }

        for q, path in heads.items():
            if not path.is_file():
                raise RuntimeError(
                    f"Missing q{q} checkpoint: {path}"
                )

            done = path.parents[1] / "DONE"

            if not done.is_file():
                raise RuntimeError(
                    f"q{q} training not DONE: {done}"
                )

        batches = Batches(
            source,
            "mix25",
            0,
        )

        batches.extractor.model.eval()
        batches.extractor.model.requires_grad_(
            False
        )

        total_images = len(
            batches.val
        )

        if total_images != 548:
            raise RuntimeError(
                f"Expected 548 val images, got {total_images}"
            )

        n = (
            total_images
            if a.limit <= 0
            else min(
                total_images,
                a.limit,
            )
        )

        models = {
            q: load_head(path)
            for q, path
            in heads.items()
        }

        provenance = {
            "source_job":
                str(source),
            "pareto_job":
                str(pareto),
            "images":
                n,
            "full_validation_images":
                total_images,
            "batch":
                a.batch,
            "views": [
                {
                    "name": name,
                    "flip": flip,
                    "scale": scale,
                    "difficulty": difficulty,
                }
                for (
                    name,
                    flip,
                    scale,
                    difficulty,
                )
                in VIEWS
            ],
            "heads": {
                str(q): {
                    "checkpoint":
                        str(path),
                    "sha256":
                        sha(path),
                }
                for q, path
                in heads.items()
            },
            "routes": [
                "exact",
                "transport",
            ],
            "metric_note":
                "Custom diagnostic detector AP; "
                "ignored VisDrone regions are not handled. "
                "This is not official VisDrone AP.",
            "test_dev_used":
                False,
            "training_updates":
                0,
        }

        dump(
            output
            / "provenance.json",
            provenance,
        )

        rows = []
        feature_rows = []

        started = time.perf_counter()

        for (
            view_name,
            flip,
            scale,
            difficulty,
        ) in VIEWS:

            status(
                f"RUNNING view={view_name}"
            )

            states = {
                (
                    q,
                    route,
                ):
                    EvalState(
                        model
                    )
                for q, model
                in models.items()
                for route
                in (
                    "exact",
                    "transport",
                )
            }

            view_feature_mse = []

            for start in range(
                0,
                n,
                a.batch,
            ):
                ids = list(
                    range(
                        start,
                        min(
                            start
                            + a.batch,
                            n,
                        ),
                    )
                )

                samples = [
                    batches.val[i]
                    for i in ids
                ]

                transformed_images = []
                transformed_items = []

                for sample in samples:
                    image, info = (
                        load_letterboxed_tensor(
                            sample[
                                "image_path"
                            ],
                            imgsz=640,
                        )
                    )

                    if (
                        info
                        != sample[
                            "letterbox"
                        ]
                    ):
                        raise RuntimeError(
                            "Letterbox metadata changed"
                        )

                    raw = batches.item(
                        sample
                    )

                    (
                        transformed_image,
                        cls,
                        boxes,
                    ) = geometry(
                        image,
                        raw["cls"],
                        raw["bboxes"],
                        flip,
                        scale,
                    )

                    transformed_images.append(
                        transformed_image
                    )

                    transformed_items.append(
                        {
                            **raw,
                            "cls": cls,
                            "bboxes": boxes,
                        }
                    )

                # ---------------------------------
                # Exact transformed DINO features
                # ---------------------------------
                exact_raw = batches.extractor(
                    torch.stack(
                        transformed_images
                    )
                )

                exact = tuple(
                    training_feature(
                        exact_raw[
                            f"layer_{layer}"
                        ]
                    )
                    for layer
                    in batches.meta[
                        "identity"
                    ][
                        "layers"
                    ]
                )

                # ---------------------------------
                # Transport from cached val anchor
                # ---------------------------------
                anchor_np = np.array(
                    batches.val_array[
                        ids
                    ],
                    copy=True,
                )

                anchor = torch.from_numpy(
                    anchor_np
                ).cuda().float()

                anchor_levels = tuple(
                    anchor[:, level]
                    for level
                    in range(3)
                )

                transport = tuple(
                    warp_features(
                        level,
                        [flip]
                        * len(ids),
                        [scale]
                        * len(ids),
                    )
                    for level
                    in anchor_levels
                )

                mse_values = rel_mse(
                    transport,
                    exact,
                )

                view_feature_mse.extend(
                    mse_values
                )

                # ---------------------------------
                # Evaluate every head on both routes
                # ---------------------------------
                for q in models:
                    states[
                        (q, "exact")
                    ].update(
                        exact,
                        transformed_items,
                    )

                    states[
                        (q, "transport")
                    ].update(
                        transport,
                        transformed_items,
                    )

                if (
                    start == 0
                    or start + len(ids) == n
                    or (
                        start
                        // a.batch
                    ) % 25 == 0
                ):
                    print(
                        f"view={view_name} "
                        f"images={start + len(ids)}/{n}",
                        flush=True,
                    )

            feature_summary = {
                "view":
                    view_name,
                "difficulty":
                    difficulty,
                "flip":
                    flip,
                "scale":
                    scale,
                "feature_rel_mse_mean":
                    float(
                        statistics.mean(
                            view_feature_mse
                        )
                    ),
                "feature_rel_mse_median":
                    float(
                        statistics.median(
                            view_feature_mse
                        )
                    ),
            }

            feature_rows.append(
                feature_summary
            )

            for q in sorted(models):
                exact_result = states[
                    (q, "exact")
                ].finish()

                transport_result = states[
                    (q, "transport")
                ].finish()

                for route, result in (
                    (
                        "exact",
                        exact_result,
                    ),
                    (
                        "transport",
                        transport_result,
                    ),
                ):
                    rows.append(
                        {
                            "query_pct":
                                q,
                            "view":
                                view_name,
                            "difficulty":
                                difficulty,
                            "flip":
                                flip,
                            "scale":
                                scale,
                            "route":
                                route,
                            **result,
                        }
                    )

                print(
                    f"q={q:3d} "
                    f"view={view_name:8s} "
                    f"exact_AP={exact_result['AP']:.4f} "
                    f"transport_AP={transport_result['AP']:.4f} "
                    f"gap="
                    f"{exact_result['AP'] - transport_result['AP']:+.4f}",
                    flush=True,
                )

        # -----------------------------------------
        # Save raw results
        # -----------------------------------------
        dump(
            output
            / "results.json",
            rows,
        )

        dump(
            output
            / "feature-results.json",
            feature_rows,
        )

        with (
            output
            / "results.csv"
        ).open(
            "w",
            newline="",
            encoding="utf-8",
        ) as stream:

            writer = csv.DictWriter(
                stream,
                fieldnames=list(
                    rows[0]
                ),
            )

            writer.writeheader()
            writer.writerows(
                rows
            )

        # -----------------------------------------
        # Derived benchmark summaries
        # -----------------------------------------
        def lookup(
            q,
            view,
            route,
        ):
            hits = [
                r
                for r in rows
                if r["query_pct"] == q
                and r["view"] == view
                and r["route"] == route
            ]

            if len(hits) != 1:
                raise RuntimeError(
                    "Non-unique result lookup"
                )

            return hits[0]

        summary_rows = []

        for q in sorted(models):
            identity = lookup(
                q,
                "identity",
                "exact",
            )["AP"]

            mild = statistics.mean(
                lookup(
                    q,
                    v,
                    "exact",
                )["AP"]
                for v in (
                    "scale095",
                    "scale105",
                )
            )

            hard = statistics.mean(
                lookup(
                    q,
                    v,
                    "exact",
                )["AP"]
                for v in (
                    "scale085",
                    "scale115",
                )
            )

            all_exact = statistics.mean(
                lookup(
                    q,
                    v[0],
                    "exact",
                )["AP"]
                for v in VIEWS
            )

            all_transport = statistics.mean(
                lookup(
                    q,
                    v[0],
                    "transport",
                )["AP"]
                for v in VIEWS
            )

            summary_rows.append(
                {
                    "query_pct":
                        q,
                    "identity_exact_AP":
                        identity,
                    "mild_exact_AP":
                        mild,
                    "hard_exact_AP":
                        hard,
                    "all_views_exact_AP":
                        all_exact,
                    "all_views_transport_AP":
                        all_transport,
                    "hard_drop_from_identity":
                        identity - hard,
                    "transport_gap_all_views":
                        all_exact
                        - all_transport,
                }
            )

        # Per-view q-recovery
        recovery_rows = []

        for (
            view_name,
            _,
            _,
            difficulty,
        ) in VIEWS:
            q0 = lookup(
                0,
                view_name,
                "exact",
            )["AP"]

            q100 = lookup(
                100,
                view_name,
                "exact",
            )["AP"]

            denom = q100 - q0

            for q in (
                10,
                25,
                50,
            ):
                value = lookup(
                    q,
                    view_name,
                    "exact",
                )["AP"]

                recovery = (
                    (
                        value - q0
                    )
                    / denom
                    if abs(denom)
                    > 1e-12
                    else None
                )

                recovery_rows.append(
                    {
                        "view":
                            view_name,
                        "difficulty":
                            difficulty,
                        "query_pct":
                            q,
                        "AP":
                            value,
                        "q0_AP":
                            q0,
                        "q100_AP":
                            q100,
                        "recovery":
                            recovery,
                    }
                )

        elapsed = (
            time.perf_counter()
            - started
        )

        summary = {
            "images":
                n,
            "views":
                len(VIEWS),
            "teacher_image_evaluations":
                n
                * len(VIEWS),
            "heads":
                sorted(models),
            "elapsed_seconds":
                elapsed,
            "feature_results":
                feature_rows,
            "head_summary":
                summary_rows,
            "recovery":
                recovery_rows,
            "limitations": [
                "Method-development validation benchmark.",
                "No test-dev access.",
                "No training updates.",
                "AP uses custom detector metric and does not handle VisDrone ignored regions.",
                "Recovery uses exact transformed features and is descriptive, not a statistical significance test.",
            ],
        }

        dump(
            output
            / "summary.json",
            summary,
        )

        lines = [
            "CACHE-SHIFT BENCHMARK V0",
            "",
            "Custom diagnostic AP (0-100); not official VisDrone AP.",
            f"images={n}",
            f"teacher_image_evaluations={n * len(VIEWS)}",
            f"elapsed_s={elapsed:.1f}",
            "",
            "FEATURE SHIFT",
            "view difficulty scale flip rel_mse_mean rel_mse_median",
        ]

        for r in feature_rows:
            lines.append(
                f"{r['view']} "
                f"{r['difficulty']} "
                f"{r['scale']:.2f} "
                f"{int(r['flip'])} "
                f"{r['feature_rel_mse_mean']:.6f} "
                f"{r['feature_rel_mse_median']:.6f}"
            )

        lines += [
            "",
            "HEAD ROBUSTNESS USING EXACT TRANSFORMED FEATURES",
            "q identity_AP mild_AP hard_AP all_AP hard_drop",
        ]

        for r in summary_rows:
            lines.append(
                f"{r['query_pct']:3d} "
                f"{r['identity_exact_AP']:.4f} "
                f"{r['mild_exact_AP']:.4f} "
                f"{r['hard_exact_AP']:.4f} "
                f"{r['all_views_exact_AP']:.4f} "
                f"{r['hard_drop_from_identity']:+.4f}"
            )

        lines += [
            "",
            "EXACT VS TRANSPORT ALL-VIEW MEAN",
            "q exact_AP transport_AP exact_minus_transport",
        ]

        for r in summary_rows:
            lines.append(
                f"{r['query_pct']:3d} "
                f"{r['all_views_exact_AP']:.4f} "
                f"{r['all_views_transport_AP']:.4f} "
                f"{r['transport_gap_all_views']:+.4f}"
            )

        lines += [
            "",
            "ONLINE-BENEFIT RECOVERY USING EXACT FEATURES",
            "view q AP q0_AP q100_AP recovery",
        ]

        for r in recovery_rows:
            recovery = (
                "NA"
                if r["recovery"]
                is None
                else
                f"{100.0 * r['recovery']:.2f}%"
            )

            lines.append(
                f"{r['view']} "
                f"{r['query_pct']:3d} "
                f"{r['AP']:.4f} "
                f"{r['q0_AP']:.4f} "
                f"{r['q100_AP']:.4f} "
                f"{recovery}"
            )

        text = (
            "\n".join(lines)
            + "\n"
        )

        (
            output
            / "comparison.txt"
        ).write_text(
            text,
            encoding="utf-8",
        )

        status(
            "SUCCEEDED"
        )

        print()
        print(text)

    except BaseException as exc:
        status(
            f"FAILED "
            f"{type(exc).__name__}: "
            f"{exc}"
        )
        raise


if __name__ == "__main__":
    main()
