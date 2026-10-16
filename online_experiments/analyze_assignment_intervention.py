"""Reproduce descriptive paired diagnostics and figures from immutable per-image results."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"transport": "#4878A8", "rec": "#C88443"}
LABELS = {"transport": "Transport", "rec": "Rec"}
METRICS = ["cls_cosine", "box_cosine", "cls_relative_l2", "box_relative_l2", "loss_total"]


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def intervals(series, seed=20261015):
    """Descriptive resampling sensitivity, conditional on this one trained checkpoint."""
    before = len(series)
    series = series.dropna()
    values = series.to_numpy(float)
    if not len(values):
        return {"n": 0}
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(values), size=(2000, len(values)))
    boot = values[draws].mean(1)
    prefixes = pd.Index(series.index).map(lambda x: str(x).split("/")[-1].split("_")[0])
    frame = pd.DataFrame({"value": values, "prefix": prefixes})
    clusters = frame.groupby("prefix").value.agg(["sum", "count"])
    draws = rng.integers(len(clusters), size=(2000, len(clusters)))
    cluster_boot = clusters["sum"].to_numpy()[draws].sum(1) / clusters["count"].to_numpy()[draws].sum(1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    clo, chi = np.quantile(cluster_boot, [0.025, 0.975])
    return {
        "n": len(values),
        "before_n": before,
        "undefined_excluded_n": before - len(values),
        "mean": values.mean(),
        "sd": values.std(ddof=1),
        "image_boot_low": lo,
        "image_boot_high": hi,
        "prefix_groups": len(clusters),
        "prefix_boot_low": clo,
        "prefix_boot_high": chi,
    }


def load_stage(raw, stage):
    source = raw / stage
    summary = json.loads((source / "summary.json").read_text())
    for name in ("per-image.csv", "assignments.csv", "selection.json"):
        if fingerprint(source / name) != summary["outputs_sha256"][name]:
            raise ValueError(f"Input checksum mismatch: {stage}/{name}")
    losses, matches = pd.read_csv(source / "per-image.csv"), pd.read_csv(source / "assignments.csv")
    if len(losses) != summary["images"] * 192 or len(matches) != summary["images"] * 24:
        raise ValueError("Incomplete per-image grid")
    keys = ["image_id", "flip", "scale", "method", "branch", "protocol", "cell"]
    if losses.duplicated(keys).any() or matches.duplicated(keys[:5]).any():
        raise ValueError("Duplicate diagnostic cells")
    if not np.allclose(losses.loss_box + losses.loss_cls + losses.loss_dfl, losses.loss_total, atol=1e-5):
        raise ValueError("Component loss mismatch")
    # The frozen native BboxLoss uses normalized ltrb L1 for reg_max=1,
    # returned in the historically named DFL slot. Retain this nonzero term.
    if not (np.isfinite(losses.loss_dfl) & (losses.loss_dfl >= 0)).all():
        raise ValueError("Invalid auxiliary bbox-distance L1 loss")
    return losses, matches, summary


def save_figure(figure, output, name):
    figure.savefig(output / "figures" / f"{name}.pdf")
    figure.savefig(output / "figures" / f"{name}.svg")
    figure.savefig(output / "figures" / f"{name}.png", dpi=600)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "figures").mkdir(exist_ok=True)
    tables, contrasts, frames, audits = [], [], {}, {}
    for stage in ("development", "confirmation"):
        losses, matches, audit = load_stage(args.raw, stage)
        frames[stage] = losses, matches
        audits[stage] = audit
        nonidentity = losses[losses.scale != 1]
        by_image = nonidentity.groupby(["method", "branch", "protocol", "cell", "image_id"])[METRICS].mean()
        for key, group in by_image.groupby(level=[0, 1, 2, 3]):
            local = group.droplevel([0, 1, 2, 3])
            for metric in METRICS:
                tables.append(
                    dict(zip(["method", "branch", "protocol", "cell"], key))
                    | {"stage": stage, "metric": metric}
                    | intervals(local[metric])
                )
        paired = by_image.reset_index()
        for (method, branch, protocol), group in paired.groupby(["method", "branch", "protocol"]):
            for metric in METRICS:
                pivot = group.pivot(index="image_id", columns="cell", values=metric)
                for first, second in (("AE", "AA"), ("EA", "AA")):
                    difference = pivot[first] - pivot[second]
                    contrasts.append(
                        {
                            "stage": stage,
                            "method": method,
                            "branch": branch,
                            "protocol": protocol,
                            "metric": metric,
                            "contrast": first + "-" + second,
                            "fraction_positive": float((difference.dropna() > 0).mean()),
                        }
                        | intervals(difference)
                    )
    stats, paired_stats = pd.DataFrame(tables), pd.DataFrame(contrasts)
    stats.to_csv(args.output / "cell-statistics.csv", index=False, lineterminator="\n")
    paired_stats.to_csv(args.output / "paired-contrasts.csv", index=False, lineterminator="\n")
    losses, matches = frames["confirmation"]
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
            "axes.linewidth": 0.7,
        }
    )
    # Figure contract: categorical four-cell intervention, no continuous-trajectory claim.
    fig, axes = plt.subplots(2, 2, figsize=(7.09, 5.15), layout="constrained")
    cells = ["EE", "EA", "AE", "AA"]
    for row, branch in enumerate(("one2many", "one2one")):
        for column, metric in enumerate(("cls_cosine", "box_cosine")):
            ax = axes[row, column]
            for offset, method in ((-0.08, "transport"), (0.08, "rec")):
                values = stats[
                    (stats.stage == "confirmation")
                    & (stats.method == method)
                    & (stats.branch == branch)
                    & (stats.protocol == "hard_common")
                    & (stats.metric == metric)
                ]
                values = values.set_index("cell").loc[cells]
                center = values["mean"].to_numpy()
                error = np.stack([center - values.image_boot_low, values.image_boot_high - center]).clip(0)
                ax.errorbar(
                    np.arange(4) + offset,
                    center,
                    yerr=error,
                    fmt="o" if method == "transport" else "s",
                    markersize=4,
                    capsize=2,
                    linewidth=0.8,
                    color=COLORS[method],
                    label=LABELS[method],
                )
            ax.set(
                xticks=np.arange(4),
                xticklabels=cells,
                ylim=(-0.03, 1.05),
                ylabel="Gradient cosine vs EE",
                xlabel="Feature / assignment source",
                title=f"{'abcd'[2 * row + column]}  {branch}: {'classification' if column == 0 else 'box regression'}",
            )
            ax.grid(axis="y", color="#e8e8e8", linewidth=0.5)
    axes[0, 0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Hard matching intervention · 548 validation images · one frozen head", fontsize=10)
    save_figure(fig, args.output, "figure-01-four-cell-gradients")

    feature = losses[(losses.branch == "one2many") & (losses.protocol == "native_soft") & (losses.cell == "AA")]
    feature = feature.groupby(["method", "scale", "image_id"]).feature_relative_mse.mean().reset_index()
    matching = matches.groupby(["method", "branch", "scale", "image_id"]).mean(numeric_only=True).reset_index()
    scale_rows = []
    for key, group in feature.groupby(["method", "scale"]):
        scale_rows.append(
            {"method": key[0], "scale": key[1], "metric": "feature_relative_mse", "branch": "all"}
            | intervals(group.set_index("image_id").feature_relative_mse)
        )
    for key, group in matching.groupby(["method", "branch", "scale"]):
        group = group.set_index("image_id")
        for metric in ("fg_jaccard", "gt_switch_fraction", "exact_coverage_small", "approx_coverage_small"):
            scale_rows.append(
                dict(zip(["method", "branch", "scale"], key)) | {"metric": metric} | intervals(group[metric])
            )
        scale_rows.append(
            dict(zip(["method", "branch", "scale"], key))
            | {"metric": "small_coverage_gap"}
            | intervals(group.approx_coverage_small - group.exact_coverage_small)
        )
    scale_stats = pd.DataFrame(scale_rows)
    scale_stats.to_csv(args.output / "scale-statistics.csv", index=False, lineterminator="\n")
    fig, axes = plt.subplots(2, 2, figsize=(7.09, 5.15), layout="constrained")
    definitions = [
        ("feature_relative_mse", "all", "a  Global feature error", "Relative MSE"),
        ("fg_jaccard", "one2many", "b  One2many matching", "Foreground Jaccard"),
        ("fg_jaccard", "one2one", "c  One2one matching", "Foreground Jaccard"),
        ("small_coverage_gap", "one2many", "d  Small-object coverage, one2many", "Approximate − exact coverage"),
    ]
    for ax, (metric, branch, title, ylabel) in zip(axes.flat, definitions):
        for method in ("transport", "rec"):
            data = scale_stats[
                (scale_stats.method == method) & (scale_stats.metric == metric) & (scale_stats.branch == branch)
            ].sort_values("scale")
            y = data["mean"].to_numpy()
            err = np.stack([y - data.image_boot_low, data.image_boot_high - y]).clip(0)
            ax.errorbar(
                data.scale,
                y,
                yerr=err,
                fmt="o-" if method == "transport" else "s-",
                markersize=4,
                capsize=2,
                linewidth=0.9,
                color=COLORS[method],
                label=LABELS[method],
            )
        ax.set(title=title, ylabel=ylabel, xlabel="Centered image scale", xticks=[0.85, 1.0, 1.15])
        ax.grid(axis="y", color="#e8e8e8", linewidth=0.5)
        if metric == "fg_jaccard":
            ax.set_ylim(0, 1.04)
        if metric == "feature_relative_mse":
            ax.set_ylim(bottom=0)
        if metric == "small_coverage_gap":
            ax.axhline(0, color="#999999", linewidth=0.6)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Reconstruction and matching · 548 validation images", fontsize=10)
    save_figure(fig, args.output, "figure-02-feature-matching")
    essentials = {}
    for method in ("transport", "rec"):
        subset = feature[(feature.method == method) & (feature.scale != 1)]
        essentials[method] = {
            "feature_relative_mse": float(subset.groupby("image_id").feature_relative_mse.mean().mean())
        }
    essentials["relative_mse_reduction"] = (
        1 - essentials["rec"]["feature_relative_mse"] / essentials["transport"]["feature_relative_mse"]
    )
    (args.output / "analysis-summary.json").write_text(
        json.dumps(
            {
                "source_audits": audits,
                "essentials": essentials,
                "analysis_code_sha256": fingerprint(Path(__file__)),
                "bootstrap_draws": 2000,
                "seed": 20261015,
                "uncertainty": "descriptive image/prefix resampling, not seed/population inference; no p-values",
                "loss_dfl_column_semantics": "reg_max=1: normalized ltrb L1 auxiliary loss, not distribution focal loss; initial protocol zero-term assumption corrected without changing raw results",
                "missing_validation_gradient_metrics": losses[METRICS].isna().sum().to_dict(),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(essentials, indent=2))
    primary = paired_stats[
        (paired_stats.stage == "confirmation")
        & (paired_stats.protocol == "hard_common")
        & (paired_stats.contrast == "AE-AA")
        & paired_stats.metric.isin(["cls_cosine", "box_cosine"])
    ]
    print(primary.to_string(index=False))


if __name__ == "__main__":
    main()
