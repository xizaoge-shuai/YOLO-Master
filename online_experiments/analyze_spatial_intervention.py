"""Analyze paired spatial diagnostics without treating views or heads as independent runs."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analyze_assignment_intervention import fingerprint, intervals

SEED = 20261018
ARMS = ["transport", "blank_fill", "padding_oracle", "rec", "content_oracle", "exact"]
LABELS = ["Transport", "Blank fill", "Pad oracle", "Rec", "Content oracle", "Exact"]
HEADS = ["mix25", "online-geom"]
COLORS = {"mix25": "#D55E00", "online-geom": "#0072B2"}
REGIONS = ["new_padding", "letterbox", "foreground", "background", "mixed"]


def stats(series):
    """Descriptive image and filename-prefix bootstrap; no training-seed inference."""
    result = intervals(series, seed=SEED)
    if result["n"] == 0:
        result.update(before_n=len(series), undefined_excluded_n=len(series), prefix_groups=0)
    return result


def save(fig, output, name):
    """Save inspectable vector figures plus a raster preview."""
    for suffix in ("pdf", "svg", "png"):
        path = output / "figures" / f"{name}.{suffix}"
        fig.savefig(path, dpi=400, bbox_inches="tight")
        if suffix == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n",
                encoding="utf-8",
            )
    plt.close(fig)


def load(raw):
    """Validate transferred raw bytes and the major measurement invariants."""
    summary = json.loads((raw / "summary.json").read_text(encoding="utf-8"))
    assert summary["images"] == 128 and summary["split"] == "train"
    assert summary["states_unchanged"] and summary["training_updates"] == 0
    assert summary["native_loss_parity_max_absolute"] == 0
    assert summary["teacher_images"] == 768 and summary["blank_teacher_images"] == 1
    checked = {}
    for name in (
        "provenance.json",
        "selection.json",
        "per-image.csv",
        "parameters.csv",
        "regions.csv",
        "assignments.csv",
        "comparison.txt",
    ):
        checked[name] = fingerprint(raw / name)
        assert checked[name] == summary["outputs_sha256"][name], name
    output, params, regions, matching = [
        pd.read_csv(raw / name) for name in ("per-image.csv", "parameters.csv", "regions.csv", "assignments.csv")
    ]
    common = ["image_id", "flip", "scale", "arm"]
    for frame, count, keys in (
        (output, 18432, common + ["head", "branch"]),
        (params, 258048, common + ["head", "branch", "task", "block"]),
        (regions, 69120, common + ["layer", "region"]),
        (matching, 18432, common + ["head", "branch"]),
    ):
        assert len(frame) == count and not frame.duplicated(keys).any()
        assert frame.image_id.nunique() == 128 and set(frame.arm) == set(ARMS)
        assert not np.isinf(frame.select_dtypes("number").to_numpy()).any()
    assert params.loc[params.reference_norm.eq(0), ["cosine", "relative_l2", "norm_ratio"]].isna().all().all()
    disconnected = params.branch.eq("one2one") & params.block.isin(["p3", "p4", "p5"])
    assert params.loc[disconnected, ["reference_norm", "value_norm"]].eq(0).all().all()
    assert params.loc[params.arm.eq("exact") & params.reference_norm.gt(0), "relative_l2"].eq(0).all()
    grouped = regions.groupby(common + ["layer"])
    assert grouped.tokens.sum().eq(1600).all()
    assert np.allclose(grouped.sse.sum(), grouped.global_sse.first(), rtol=1e-10, atol=1e-7)
    assert np.allclose(grouped.reference_energy.sum(), grouped.global_reference_energy.first(), rtol=1e-10, atol=1e-7)
    assert regions.loc[regions.arm.eq("exact"), "sse"].eq(0).all()
    return output, params, regions, matching, summary, checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "figures").mkdir(exist_ok=True)
    output, params, regions, matching, summary, checked = load(args.raw)
    units = ["head", "arm", "scale", "image_id"]
    chunks = []
    selected = output[output.branch.eq("one2many") & output.scale.ne(1)]
    for metric in ("cls_cosine", "box_cosine", "native_loss"):
        chunks.append(selected[units + [metric]].rename(columns={metric: "value"}).assign(metric="output_" + metric))
    specifications = [
        ("cls", "p3", "p3_cls"),
        ("box", "p3", "p3_box"),
        ("box", "one2many_box", "box_head"),
        ("box", "one2one_box", "one2one_box_head"),
    ]
    for task, block, label in specifications:
        branch = "one2one" if block == "one2one_box" else "one2many"
        selected = params[params.branch.eq(branch) & params.task.eq(task) & params.block.eq(block) & params.scale.ne(1)]
        for metric in ("cosine", "relative_l2", "norm_ratio"):
            chunks.append(
                selected[units + [metric]].rename(columns={metric: "value"}).assign(metric=label + "_" + metric)
            )
    long = pd.concat(chunks, ignore_index=True)
    group = units + ["metric"]
    counts = long.groupby(group).value.count()
    assert counts.eq(2).all(), "Primary metrics need both paired flip views per image"
    per_image = long.groupby(group).value.mean().reset_index()
    keys = ["head", "arm", "scale", "metric"]
    cells = pd.DataFrame(
        [dict(zip(keys, key)) | stats(frame.set_index("image_id").value) for key, frame in per_image.groupby(keys)]
    )
    contrast_pairs = [
        ("blank_fill", "transport"),
        ("padding_oracle", "transport"),
        ("rec", "transport"),
        ("content_oracle", "transport"),
        ("rec", "blank_fill"),
        ("rec", "padding_oracle"),
        ("content_oracle", "padding_oracle"),
    ]
    contrasts = []
    for key, frame in per_image.groupby(["head", "scale", "metric"]):
        wide = frame.pivot(index="image_id", columns="arm", values="value")
        for a, b in contrast_pairs:
            difference = wide[a] - wide[b]
            contrasts.append(
                dict(zip(["head", "scale", "metric"], key))
                | {
                    "contrast": a + "-" + b,
                    "positive_images": int(difference.gt(0).sum()),
                    "negative_images": int(difference.lt(0).sum()),
                }
                | stats(difference)
            )
    contrasts = pd.DataFrame(contrasts)
    assert cells.n.eq(128).all() and contrasts.n.eq(128).all()
    region_units = ["image_id", "arm", "scale", "layer", "region"]
    region_image = (
        regions[regions.scale.ne(1)]
        .groupby(region_units)[["sse_fraction", "regional_relative_mse", "tokens"]]
        .mean()
        .reset_index()
    )
    regional = []
    for key, frame in region_image.groupby(["arm", "scale", "layer", "region"]):
        for metric in ("sse_fraction", "regional_relative_mse", "tokens"):
            regional.append(
                dict(zip(["arm", "scale", "layer", "region"], key))
                | {"metric": metric}
                | stats(frame.set_index("image_id")[metric])
            )
    regional = pd.DataFrame(regional)
    # Global normalized MSE: sum layer SSE / sum layer reference energy within each view.
    global_view = (
        regions[regions.scale.ne(1)].groupby(["image_id", "arm", "scale", "flip"])[["sse", "reference_energy"]].sum()
    )
    global_view["relative_mse"] = global_view.sse / global_view.reference_energy
    feature_image = global_view.groupby(["image_id", "arm", "scale"]).relative_mse.mean().reset_index()
    features = pd.DataFrame(
        [
            dict(zip(["arm", "scale"], key)) | stats(frame.set_index("image_id").relative_mse)
            for key, frame in feature_image.groupby(["arm", "scale"])
        ]
    )
    matching_table = (
        matching[matching.scale.ne(1)]
        .groupby(["head", "arm", "branch", "scale"])[
            ["fg_jaccard", "gt_switch_fraction", "exact_coverage_small", "approx_coverage_small"]
        ]
        .mean()
        .reset_index()
    )
    # Preserve all blocks and both branches in an auxiliary table; NA counts remain explicit.
    param_image = (
        params[params.scale.ne(1)]
        .groupby(["head", "arm", "branch", "task", "block", "scale", "image_id"])[
            ["cosine", "relative_l2", "norm_ratio"]
        ]
        .mean()
    )
    all_blocks = param_image.groupby(level=["head", "arm", "branch", "task", "block", "scale"]).agg(
        ["mean", "std", "count"]
    )
    all_blocks.columns = ["_".join(x) for x in all_blocks.columns]
    tables = {
        "cell-statistics": cells,
        "paired-contrasts": contrasts,
        "regional-statistics": regional,
        "feature-statistics": features,
        "matching-means": matching_table,
        "all-parameter-blocks": all_blocks.reset_index(),
    }
    for name, frame in tables.items():
        frame.to_csv(args.output / f"{name}.csv", index=False, lineterminator="\n")
    lookup = cells.set_index(keys)

    def cell(head, arm, scale, metric):
        return lookup.loc[(head, arm, scale, metric)]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 6.9), layout="constrained")
    panel_metrics = [
        ("output_cls_cosine", "Classification output gradient"),
        ("p3_box_cosine", "P3 localization parameter gradient"),
        ("box_head_cosine", "Box-head parameter gradient"),
    ]
    for row, scale in enumerate((0.85, 1.15)):
        for col, (metric, title) in enumerate(panel_metrics):
            ax = axes[row, col]
            for head, offset, marker, label in [
                ("mix25", -0.11, "o", "Mix25"),
                ("online-geom", 0.11, "s", "Online100"),
            ]:
                points = [cell(head, arm, scale, metric) for arm in ARMS]
                means = np.array([p["mean"] for p in points])
                low = np.array([p.prefix_boot_low for p in points])
                high = np.array([p.prefix_boot_high for p in points])
                ax.errorbar(
                    np.arange(6) + offset,
                    means,
                    yerr=[means - low, high - means],
                    fmt=marker,
                    color=COLORS[head],
                    label=label,
                    capsize=2.5,
                    markersize=4.5,
                    lw=1.1,
                )
            ax.axhline(0, color="0.7", lw=0.6)
            ax.axhline(1, color="0.5", ls=":", lw=0.7)
            ax.set_ylim(-0.45, 1.06)
            ax.set_xticks(range(6), LABELS, rotation=35, ha="right")
            ax.set_title(f"{'abcdef'[row * 3 + col]}  {title}\nScale {scale:g}", fontsize=10)
            ax.set_ylabel("Cosine to exact (higher is better)")
            ax.grid(axis="y", alpha=0.18)
    axes[0, 0].legend(loc="lower left", frameon=False, fontsize=8)
    fig.suptitle("128 training images; paired flip means; 95% prefix-bootstrap intervals", fontsize=11)
    save(fig, args.output, "figure-01-spatial-gradient-interventions")
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), layout="constrained")
    palette = ["#0072B2", "#56B4E9", "#D55E00", "#E69F00", "#999999"]
    region_labels = ["New padding", "Letterbox", "Foreground", "Background", "Mixed boundary"]
    for ax, scale, letter in zip(axes[:2], (0.85, 1.15), ("a", "b")):
        base = np.zeros(3)
        for region, color, label, hatch in zip(REGIONS, palette, region_labels, ["", "//", "..", "\\\\", "xx"]):
            values = [
                regional[
                    (regional.arm == "transport")
                    & regional.scale.eq(scale)
                    & regional.layer.eq(layer)
                    & regional.region.eq(region)
                    & regional.metric.eq("sse_fraction")
                ]["mean"].iloc[0]
                * 100
                for layer in (3, 7, 11)
            ]
            ax.bar(
                range(3), values, bottom=base, color=color, label=label, hatch=hatch, edgecolor="white", linewidth=0.6
            )
            base += np.array(values)
        assert np.allclose(base, 100)
        ax.set_ylim(0, 100)
        ax.set_xticks(range(3), ["Layer 3", "Layer 7", "Layer 11"])
        ax.set_ylabel("Mean share of transport SSE (%)")
        ax.set_title(f"{letter}  Regional error, scale {scale:g}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, frameon=False, loc="lower left", bbox_to_anchor=(0.025, -0.06), ncol=3)
    ax = axes[2]
    for scale, offset, marker, color in [(0.85, -0.1, "o", "#0072B2"), (1.15, 0.1, "s", "#D55E00")]:
        points = [features[features.arm.eq(arm) & features.scale.eq(scale)].iloc[0] for arm in ARMS]
        means = np.array([p["mean"] for p in points])
        lo = np.array([p.prefix_boot_low for p in points])
        hi = np.array([p.prefix_boot_high for p in points])
        ax.errorbar(
            np.arange(6) + offset,
            means,
            yerr=[means - lo, hi - means],
            fmt=marker,
            color=color,
            label=f"Scale {scale:g}",
            capsize=2.5,
        )
    ax.set_ylim(bottom=0)
    ax.set_xticks(range(6), LABELS, rotation=35, ha="right")
    ax.set_ylabel("Global relative feature MSE (lower is better)")
    ax.set_title("c  Feature error after intervention")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.18)
    save(fig, args.output, "figure-02-regional-error-budget")

    table = [
        "| Scale | Arm | Class output cosine | P3 box param cosine | Box-head param cosine |",
        "|---|---|---:|---:|---:|",
    ]
    for scale in (0.85, 1.15):
        for arm, label in zip(ARMS, LABELS):
            values = [
                cell("mix25", arm, scale, m)["mean"] for m in ("output_cls_cosine", "p3_box_cosine", "box_head_cosine")
            ]
            table.append(f"| {scale} | {label} | " + " | ".join(f"{v:.6f}" for v in values) + " |")
    delta = contrasts.set_index(["head", "scale", "metric", "contrast"])
    selected_contrasts = []
    for head, scale, metric, contrast in [
        ("mix25", 0.85, "output_cls_cosine", "blank_fill-transport"),
        ("mix25", 0.85, "output_cls_cosine", "rec-blank_fill"),
        ("mix25", 0.85, "p3_box_cosine", "rec-blank_fill"),
        ("mix25", 0.85, "p3_box_cosine", "rec-padding_oracle"),
        ("online-geom", 0.85, "p3_box_cosine", "blank_fill-transport"),
        ("mix25", 1.15, "p3_box_cosine", "content_oracle-transport"),
    ]:
        r = delta.loc[(head, scale, metric, contrast)]
        selected_contrasts.append(
            f"- {head}, scale {scale}, {metric}, {contrast}: {r['mean']:+.6f}; descriptive 95% prefix-bootstrap interval [{r.prefix_boot_low:+.6f}, {r.prefix_boot_high:+.6f}]."
        )
    nclusters = int(cells.prefix_groups.iloc[0])
    report = (
        """# Spatial intervention analysis, 2026-10-18

This is a completed frozen-head diagnostic on 128 selected **training** images, with one seed-0 checkpoint per head recipe, six shared feature arms and six views. It is neither a new AP experiment nor independent seed/backbone/dataset replication. No parameters or buffers changed, no optimizer updates occurred, and native-loss parity error is zero. All downloaded raw hashes and row invariants passed.

## Questions and decisions

1. Does new padding explain shrink-view degradation? The paired padding oracle and cheap blank fill substantially repair the Mix25 classification and localization gradient failures at scale 0.85. This establishes sensitivity to the padding intervention in these frozen heads; it does not establish a general training cause or an AP gain.
2. Is the learned corrector doing more than filling blank space? Classification output alignment is nearly matched by blank fill, but Rec retains some parameter-gradient advantages. Output and parameter gradients must not be conflated. A cheap fill baseline is now mandatory in candidate training.
3. Where is the remaining localization opportunity? At scale 1.15, content replacement greatly improves localization parameter alignment; blank fill does essentially nothing because no new padding is created. At scale 0.85, content-only replacement leaves corrupted padding and can retain negative Mix25 P3 alignment. The next spatial study needs a padding-by-content factorial control before equal-token/equal-norm foreground/background interventions.

## Exact numerical summary

Mix25, one2many; mean of the two flip views within each image, then equal image means. Gradients use hard targets from the exact feature assignment. A cosine of one is the exact reference. Localization includes the reg_max=1 auxiliary normalized ltrb L1 term.

"""
        + "\n".join(table)
        + "\n\n## Paired effect summaries\n\n"
        + "\n".join(selected_contrasts)
        + """

## Interpretation and limits

- New padding occupies 444/1600 tokens (27.75%) at scale 0.85. Its mean shares of Transport SSE in layers 3, 7 and 11 are approximately 94.68%, 89.56% and 83.24%. A large global MSE reduction can therefore arise mainly outside valid image content; it is not evidence of recovered object geometry by itself.
- Feature errors are shared by the two heads, not two independent replications. Natural-area oracles are unequal in area/energy and require exact teacher features. They are diagnostic donors, not deployable methods or fair efficiency competitors.
- Cheap fill is not uniformly beneficial: at scale 0.85 it decreases Online100 P3 localization-parameter cosine by 0.068777 versus Transport (descriptive prefix-bootstrap interval [-0.125353, -0.011211]). Its Mix25 benefit must not be generalized to all heads or metrics.
- Blank fill uses one additional gray-image teacher query per stage and a 7,372,800-byte FP32 template. Its fill includes fractional interpolation support, whereas the padding oracle replaces only pure added-padding tokens; these masks are not identical.
- The global feature statistic here is summed-layer SSE divided by summed-layer reference energy; do not merge its values with earlier equal-layer normalized-MSE summaries or different image subsets.
- Views are not independent samples. One checkpoint per recipe, correlated image filenames and previously seen training images prevent population/seed significance claims. No p-values, statistical winner claims or equivalence claims are made. All intervals are descriptive, pointwise resampling sensitivity, not simultaneous confidence bands.
- Exact assignment is held fixed for gradient comparisons; native self-assigned losses and matching are separately recorded. These are frozen eval-state derivatives, not optimizer update trajectories. One2one shared gradients remain disconnected and undefined comparisons stay NA.
- Official AP, small-object AP, equal-time training, robustness and cross-domain/model claims remain untested by this campaign. Previous negative training results remain valid.
- Development took 1926.39 seconds (32.11 minutes); the initial smoke-based 3-4-hour estimate was too high. Timing changed substantially during execution; do not use this diagnostic runtime as a method efficiency benchmark.

## Claim candidates

- Claim: Shrink-view transport error is dominated by added padding in these cached features.
  - Source evidence: regional-statistics.csv, Transport scale 0.85, all three layers; figure 02a.
  - Allowed wording: New padding accounts for 83-95% of mean per-image Transport SSE at the tested shrink scale.
  - Forbidden stronger wording: Padding explains all detector degradation or all transforms.
  - Uncertainty: One backbone, training subset; no area/energy matching.
  - Next check: Padding-by-content factorial interventions, followed by matched residual controls.
  - Decision: keep, narrowly.
- Claim: A cheap blank template nearly matches Rec's classification output alignment here.
  - Source evidence: cell-statistics.csv and paired-contrasts.csv; figure 01a.
  - Allowed wording: The observed means are close, while Rec retains a small positive paired difference.
  - Forbidden stronger wording: The methods are statistically equivalent or achieve the same AP.
  - Uncertainty: No equivalence margin or training comparison; parameter gradients differ.
  - Next check: Include blank fill as a budget-matched training baseline after diagnostic controls.
  - Decision: revise the earlier broad representation-recovery story.
- Claim: Content restoration exposes localization headroom at enlargement.
  - Source evidence: P3 and box-head parameter comparisons at scale 1.15; figure 01e-f.
  - Allowed wording: The content oracle improves local gradient alignment under exact assignments.
  - Forbidden stronger wording: A learned content corrector already exists, is novel or will improve AP.
  - Uncertainty: Teacher oracle, unequal region energy, fixed eval heads; shrink-view interaction remains unresolved.
  - Next check: Factorial padding/content control, then equal-token/equal-norm residual interventions with padding fixed.
  - Decision: keep as a method-development lead.
"""
    )
    (args.output / "analysis-report.md").write_text(report, encoding="utf-8")
    appendix = f"""# Statistical appendix

- Units: 128 fixed training images; one trained seed (0) per recipe; two head recipes share all extracted features and the same corrector. {nclusters} filename-prefix groups; prefixes are a sensitivity proxy, not verified independent scenes.
- Pairing: average the two flip observations within each image and fixed scale before all main statistics. No flip, layer, branch, task or head is counted as an independent image or run. Both flips are required and checked for every primary cell.
- Descriptives: image mean, sample SD, defined/excluded image counts and paired raw differences. Full per-block means/SD/counts including zero-gradient NA rows are provided separately. Regional relative errors omit zero-energy regions with explicit counts; global normalized MSE sums SSE and energy across layers within a view before averaging views/images.
- Intervals: 2000 percentile bootstrap draws, seed {SEED}; image resampling and whole-filename-prefix resampling with ratio-of-sums means. The plots show prefix resampling; CSVs retain both. These are pointwise descriptive intervals conditional on selected images and trained checkpoints.
- Inference gate: no inferential p-values or seed-level significance tests. Image correlation, unverified grouping and n=1 seed per recipe block generalization claims. Normality tests cannot repair independence. No formal multiple-comparison claims are made; the full contrast family is exported, without selecting on interval exclusion.
- Contrasts: blank-fill/transport, padding-oracle/transport, Rec/transport, content-oracle/transport, Rec/blank-fill, Rec/padding-oracle and content-oracle/padding-oracle, separately for both heads, both nonidentity scales and every primary metric. Positive cosine differences favor the first arm; negative relative-L2 differences favor it. Native losses are self-assigned, so a loss reduction alone is not a controlled target comparison.
- No equivalence test, no AP inference and no gradient-to-AP conversion. Oracle region areas/energies differ. One2one shared blocks are disconnected by design.
- Validation: raw SHA256 checks, complete unique grids, finite metrics except defined NA, exact-reference zero parameter error, no one2one shared gradient, 1600-token disjoint partitions, additive regional SSE/energy.
"""
    (args.output / "stats-appendix.md").write_text(appendix, encoding="utf-8")
    catalog = """# Figure catalog

## Figure 01: spatial gradient interventions

- Files: figures/figure-01-spatial-gradient-interventions.{pdf,svg,png}.
- Purpose: separate padding-driven shrink degradation from residual localization mismatch; compare output derivatives with parameter derivatives.
- Six panels: rows are scales 0.85 and 1.15; columns are classification output, P3 localization parameters, and one2many box-head parameters. X: six intervention arms; Y: cosine to exact, with a shared range including negative values and the exact reference. Circles/orange: Mix25; squares/blue: Online100.
- Source: cell-statistics.csv; paired-contrasts.csv supports comparisons. Means use 128 equal image weights after paired flip averaging; whiskers are 95% filename-prefix bootstrap sensitivity intervals (46 groups, 2000 draws), not seed uncertainty. One checkpoint per recipe.
- Notice: cheap padding repairs Mix25 shrink failures; Rec remains imperfect for localization; the content oracle performs well at enlargement but does not remove shrink padding corruption.
- Decision: retain cheap fill as a baseline; resolve padding/content interactions before a new trainable candidate.
- Caption requirements: hard exact assignments, frozen eval-state derivatives, no optimizer/AP result, unequal oracle area/energy, one seed and NA policy.
- Checklist: exact=1; negative P3 cosine visible; no inference from overlapping intervals; no comparison of natural-area oracle magnitudes as causal importance.

## Figure 02: regional error budget

- Files: figures/figure-02-regional-error-budget.{pdf,svg,png}.
- Purpose: audit whether global feature MSE predominantly measures blank-space error.
- Panels a/b: x = DINO layers 3/7/11; y = mean per-image fraction of Transport SSE, stacked to 100%, at scales 0.85/1.15. Five distinct color/hatch regions. Panel c: x = six arms; y = summed-layer global relative MSE; scale indicated by color/shape, prefix-bootstrap intervals.
- Source: regional-statistics.csv and feature-statistics.csv. Stack heights are descriptive compositions; per-region uncertainty is available in CSV rather than overlapping stacked whiskers.
- Notice: padding dominates error at shrink scale; no new padding at enlargement. Feature-error gains alone do not establish task recovery.
- Decision: report region-normalized error and task-sensitive derivatives alongside global MSE.
- Caption requirements: pixel/feature support mask, mixed boundaries separate, regional sizes unequal, feature results shared by both heads, no AP claim.
- Checklist: stacks sum to 100; foreground is any visible GT-fragment patch overlap; zero-energy regional ratios remain undefined; template build cost is charged once.
"""
    (args.output / "figure-catalog.md").write_text(catalog, encoding="utf-8")
    audit = {
        "status": "PASS",
        "raw_hashes": checked,
        "summary_sha256": fingerprint(args.raw / "summary.json"),
        "analyzer_sha256": fingerprint(Path(__file__)),
        "images": 128,
        "prefix_groups": nclusters,
        "seed_per_head": 1,
        "bootstrap_draws": 2000,
        "bootstrap_seed": SEED,
        "native_loss_parity": 0,
        "states_unchanged": True,
        "training_updates": 0,
        "development_seconds": summary["elapsed_seconds"],
        "inference": "descriptive only; no AP or significance claim",
    }
    (args.output / "analysis-summary.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print("\n".join(table))
    print("\n".join(selected_contrasts))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
