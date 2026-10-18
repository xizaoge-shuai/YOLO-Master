"""Compare two frozen-head diagnostics using paired images and immutable raw results."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analyze_assignment_intervention import fingerprint, intervals, load_stage

METRICS = ["cls_cosine", "box_cosine", "cls_relative_l2", "box_relative_l2", "cls_norm_ratio", "loss_total"]
SEED = 20261018
COLORS = {"Online100": "#4878A8", "Mix25": "#C88443"}


def interval(series):
    return intervals(series, seed=SEED)


def save(fig, output, stem):
    for ext in ("pdf", "svg", "png"):
        path = output / "figures" / f"{stem}.{ext}"
        fig.savefig(path, dpi=600)
        if ext == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n",
                encoding="utf-8",
            )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online", type=Path, required=True)
    parser.add_argument("--mix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "figures").mkdir(exist_ok=True)
    frames, assignments, audits, provenance = {}, {}, {}, {}
    for head, source in (("Online100", args.online), ("Mix25", args.mix)):
        data, matching, audit = load_stage(source.parent, source.name)
        assert audit["images"] == 548 and audit["split"] == "val"
        assert audit["training_updates"] == 0 and audit["states_unchanged"]
        assert audit["native_loss_parity_max_absolute"] == 0
        assert audit["parity_max_relative_l2"] < 0.01
        assert np.isfinite(data[METRICS].to_numpy()).all()
        frames[head], assignments[head], audits[head] = data, matching, audit
        provenance[head] = json.loads((source / "provenance.json").read_text(encoding="utf-8"))
    old, new = provenance["Online100"], provenance["Mix25"]
    assert (old["head"], new["head"]) == ("online-geom", "mix25")
    same_fields = [
        "source_hashes",
        "views",
        "manifest_sha256",
        "cache_manifest",
        "cache_hash_verification",
        "reg_max",
        "batch",
        "seed",
        "split",
        "environment",
    ]
    for field in same_fields:
        assert old[field] == new[field], field
    assert old["initial_state_hashes"]["corrector"] == new["initial_state_hashes"]["corrector"]
    assert old["initial_state_hashes"]["head"] != new["initial_state_hashes"]["head"]
    assert (
        audits["Online100"]["outputs_sha256"]["selection.json"] == audits["Mix25"]["outputs_sha256"]["selection.json"]
    )
    keys = ["image_id", "flip", "scale", "method", "branch", "protocol", "cell"]
    first, second = (frames[h].set_index(keys).sort_index() for h in ("Online100", "Mix25"))
    assert first.index.equals(second.index)
    feature_difference = float(np.abs(first.feature_relative_mse - second.feature_relative_mse).max())
    assert feature_difference == 0
    group_keys = ["head", "method", "branch", "protocol", "cell"]
    all_data = pd.concat([frame.assign(head=head) for head, frame in frames.items()], ignore_index=True)
    nonidentity = all_data[all_data.scale != 1]
    per_image = nonidentity.groupby(group_keys + ["image_id"])[METRICS].mean().reset_index()
    cell_rows, contrasts = [], []
    for key, group in per_image.groupby(group_keys):
        for metric in METRICS:
            cell_rows.append(
                dict(zip(group_keys, key)) | {"metric": metric} | interval(group.set_index("image_id")[metric])
            )
    for key, group in per_image.groupby(group_keys[:-1]):
        for metric in METRICS:
            pivot = group.pivot(index="image_id", columns="cell", values=metric)
            for a, b in (("AE", "AA"), ("EA", "AA")):
                contrasts.append(
                    dict(zip(group_keys[:-1], key))
                    | {"metric": metric, "contrast": a + "-" + b}
                    | interval(pivot[a] - pivot[b])
                )
    head_rows = []
    for key, group in per_image.groupby(group_keys[1:]):
        for metric in METRICS:
            pivot = group.pivot(index="image_id", columns="head", values=metric)
            head_rows.append(
                dict(zip(group_keys[1:], key))
                | {"metric": metric, "contrast": "Mix25-Online100"}
                | interval(pivot.Mix25 - pivot.Online100)
            )
    scale_keys = ["head", "method", "branch", "protocol", "cell", "scale"]
    scale_data = all_data.groupby(scale_keys + ["image_id"])[METRICS].mean().reset_index()
    scale_rows = []
    for key, group in scale_data[scale_data.protocol == "hard_common"].groupby(scale_keys):
        for metric in ("cls_cosine", "box_cosine", "cls_norm_ratio"):
            scale_rows.append(
                dict(zip(scale_keys, key)) | {"metric": metric} | interval(group.set_index("image_id")[metric])
            )
    tables = {
        "cell-statistics": pd.DataFrame(cell_rows),
        "paired-interventions": pd.DataFrame(contrasts),
        "paired-head-differences": pd.DataFrame(head_rows),
        "scale-statistics": pd.DataFrame(scale_rows),
    }
    for name, table in tables.items():
        assert table.n.eq(548).all() and table.undefined_excluded_n.eq(0).all()
        table.to_csv(args.output / f"{name}.csv", index=False, lineterminator="\n")
    matching = pd.concat([data.assign(head=head) for head, data in assignments.items()], ignore_index=True)
    matching[matching.scale != 1].groupby(["head", "method", "branch"])[
        ["fg_jaccard", "gt_switch_fraction", "exact_coverage_small", "approx_coverage_small"]
    ].mean().to_csv(args.output / "matching-means.csv", lineterminator="\n")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    stats = tables["cell-statistics"]
    fig, axes = plt.subplots(2, 2, figsize=(7.09, 5.6), layout="constrained")
    cells = ["EE", "EA", "AE", "AA"]
    for row, branch in enumerate(("one2many", "one2one")):
        for col, metric in enumerate(("cls_cosine", "box_cosine")):
            ax = axes[row, col]
            for i, (head, method) in enumerate((h, m) for h in ("Online100", "Mix25") for m in ("transport", "rec")):
                values = (
                    stats[
                        (stats["head"] == head)
                        & (stats.method == method)
                        & (stats.branch == branch)
                        & (stats.protocol == "hard_common")
                        & (stats.metric == metric)
                    ]
                    .set_index("cell")
                    .loc[cells]
                )
                y = values["mean"].to_numpy()
                err = np.stack([y - values.prefix_boot_low, values.prefix_boot_high - y]).clip(0)
                ax.errorbar(
                    np.arange(4) + (i - 1.5) * 0.09,
                    y,
                    yerr=err,
                    color=COLORS[head],
                    fmt="o" if method == "transport" else "s",
                    fillstyle="none" if method == "transport" else "full",
                    capsize=2,
                    markersize=4,
                    linewidth=0.7,
                    label=f"{head} / {method.title()}",
                )
            ax.set(
                xticks=np.arange(4),
                xticklabels=cells,
                ylim=(-0.03, 1.05),
                xlabel="Feature / assignment source",
                ylabel="Output-gradient cosine vs EE",
                title=f"{'abcd'[row * 2 + col]}  {branch}: {'classification' if col == 0 else 'box'}",
            )
            ax.grid(axis="y", color="#e8e8e8", linewidth=0.5)
    axes[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Two frozen heads · same 548 images · hard matching", fontsize=10)
    save(fig, args.output, "figure-01-head-replication")
    fig, axes = plt.subplots(2, 2, figsize=(7.09, 5.6), layout="constrained")
    scales = tables["scale-statistics"]
    for i, (ax, (metric, cell, title, ylabel)) in enumerate(
        zip(
            axes.flat,
            [
                ("cls_cosine", "AA", "Classification: approximate matching", "Gradient cosine vs EE"),
                ("cls_cosine", "AE", "Classification: exact matching", "Gradient cosine vs EE"),
                ("box_cosine", "AA", "Box: approximate matching", "Gradient cosine vs EE"),
                ("cls_norm_ratio", "AA", "Classification gradient magnitude", "Gradient norm / EE norm"),
            ],
        )
    ):
        for head in ("Online100", "Mix25"):
            for method in ("transport", "rec"):
                values = scales[
                    (scales["head"] == head)
                    & (scales.method == method)
                    & (scales.branch == "one2many")
                    & (scales.metric == metric)
                    & (scales.cell == cell)
                ].sort_values("scale")
                y = values["mean"].to_numpy()
                err = np.stack([y - values.prefix_boot_low, values.prefix_boot_high - y]).clip(0)
                ax.errorbar(
                    values.scale,
                    y,
                    yerr=err,
                    color=COLORS[head],
                    fmt="o--" if method == "transport" else "s-",
                    fillstyle="none" if method == "transport" else "full",
                    capsize=2,
                    markersize=4,
                    linewidth=0.8,
                    label=f"{head} / {method.title()}",
                )
        ax.set(title=f"{'abcd'[i]}  {title}", xticks=[0.85, 1, 1.15], xlabel="Centered image scale", ylabel=ylabel)
        if metric != "cls_norm_ratio":
            ax.set_ylim(0, 1.05)
        else:
            ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#e8e8e8", linewidth=0.5)
    axes[0, 0].legend(fontsize=7, loc="lower right")
    fig.suptitle("Scale breakdown · one2many · same features, different heads", fontsize=10)
    save(fig, args.output, "figure-02-scale-breakdown")
    summary = {
        "source_audits": audits,
        "identical_provenance_fields": same_fields,
        "identical_corrector": True,
        "different_heads": True,
        "maximum_per_row_feature_mse_difference": feature_difference,
        "analysis_sha256": fingerprint(Path(__file__)),
        "helper_sha256": fingerprint(Path(__file__).with_name("analyze_assignment_intervention.py")),
        "images": 548,
        "seed_per_head_recipe": 1,
        "bootstrap_draws": 2000,
        "bootstrap_seed": SEED,
        "prefix_groups": int(stats.prefix_groups.iloc[0]),
        "inference": "conditional descriptive resampling only; no p-values or training-seed/population claims",
        "all_primary_metrics_finite": True,
    }
    (args.output / "analysis-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        tables["paired-interventions"]
        .query("protocol == 'hard_common' and contrast == 'AE-AA' and metric in ['cls_cosine', 'box_cosine']")[
            ["head", "method", "branch", "metric", "mean", "prefix_boot_low", "prefix_boot_high"]
        ]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
