#!/usr/bin/env python3
"""Analyze the checked-in, three-seed D1 feature-augmentation pilot."""

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def paired_test(differences):
    observed = abs(statistics.mean(differences))
    values = [
        abs(statistics.mean(s * d for s, d in zip(signs, differences)))
        for signs in itertools.product((-1, 1), repeat=len(differences))
    ]
    return sum(v >= observed - 1e-12 for v in values) / len(values)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive", type=Path)
    args = p.parse_args()
    root = args.archive.resolve()
    manifest = read(root / "archive-manifest.json")
    for rel, info in manifest["files"].items():
        if hashlib.sha256((root / rel).read_bytes()).hexdigest() != info["sha256"]:
            raise RuntimeError(f"Archived evidence changed: {rel}")
    plan = read(root / "original/plan.json")
    groups, curves = {}, {}
    for method, seed in plan["tasks"]:
        run = root / "original/train" / f"{method}-s{seed}"
        s = read(run / "summary.json")
        with (run / "metrics.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == plan["epochs"] == 100
        assert math.isclose(max(float(r["map50_95"]) for r in rows), s["best_map50_95"], abs_tol=1e-12)
        groups.setdefault(method, {})[seed] = dict(
            best=100 * s["best_map50_95"],
            last=100 * s["final_map50_95"],
            train_s=s["train_seconds"],
            total_s=s["build_plus_run_seconds"],
            vram=s["peak_vram_gib"],
            cache_gib=s["training_cache_bytes"] / 2**30,
        )
        curves.setdefault(method, {})[seed] = [100 * float(r["map50_95"]) for r in rows]
    methods = list(plan["cases"])
    assert set(groups) == set(methods)
    summary = []
    tcrit = 4.302652729911275  # Student-t 97.5th percentile, 2 degrees of freedom.
    for method in methods:
        g = groups[method]
        assert set(g) == {0, 1, 2}
        best = [g[s]["best"] for s in range(3)]
        mean, sd = statistics.mean(best), statistics.stdev(best)
        half = tcrit * sd / math.sqrt(3)
        summary.append(
            dict(
                method=method,
                n=3,
                best_mean=mean,
                best_sd=sd,
                best_t95_low=mean - half,
                best_t95_high=mean + half,
                last_mean=statistics.mean(g[s]["last"] for s in range(3)),
                train_mean=statistics.mean(g[s]["train_s"] for s in range(3)),
                total_mean=statistics.mean(g[s]["total_s"] for s in range(3)),
            )
        )
    comparisons = [
        ("cache2", "cache1"),
        ("frofa-c1-d01", "cache1"),
        ("frofa-c1-d03", "cache1"),
        ("frofa-c2-d01", "cache2"),
        ("frofa-c2-d03", "cache2"),
        ("transport-c1", "cache1"),
        ("loffta-c1-n005", "transport-c1"),
        ("transport-c2", "cache2"),
        ("loffta-c2-n005", "transport-c2"),
    ]
    contrasts = []
    for method, base in comparisons:
        diff = [groups[method][s]["best"] - groups[base][s]["best"] for s in range(3)]
        final = [groups[method][s]["last"] - groups[base][s]["last"] for s in range(3)]
        mean, sd = statistics.mean(diff), statistics.stdev(diff)
        half = tcrit * sd / math.sqrt(3)
        contrasts.append(
            dict(
                method=method,
                reference=base,
                paired_best_delta=diff,
                best_delta_mean=mean,
                best_delta_sd=sd,
                t95_low=mean - half,
                t95_high=mean + half,
                last_delta_mean=statistics.mean(final),
                sign_flip_p=paired_test(diff),
                positive_seed_count=sum(v > 0 for v in diff),
            )
        )
    running = 0.0
    for rank, idx in enumerate(sorted(range(len(contrasts)), key=lambda i: contrasts[i]["sign_flip_p"])):
        running = max(running, min(1.0, (len(contrasts) - rank) * contrasts[idx]["sign_flip_p"]))
        contrasts[idx]["holm_p"] = running
    payload = dict(
        summary=summary,
        paired_contrasts=contrasts,
        metric_scale="AP 0-100",
        unit="training seed, conditional on this fixed 400/100 split",
        evidence_scope=manifest["validation_scope"],
        warning="Post-hoc exploratory comparisons; no dataset-level generalization or significance claim.",
    )
    (root / "analysis.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (root / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    table = [
        "| Method | Best AP mean +/- SD | Last AP | Train s | Charged build + wall s |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in summary:
        table.append(
            f"| {r['method']} | {r['best_mean']:.4f} +/- {r['best_sd']:.4f} | "
            f"{r['last_mean']:.4f} | {r['train_mean']:.1f} | {r['total_mean']:.1f} |"
        )
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
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
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw={"width_ratios": [1.25, 1]})
    y = list(range(len(methods)))
    means = [r["best_mean"] for r in summary]
    sds = [r["best_sd"] for r in summary]
    axes[0].errorbar(means, y, xerr=sds, fmt="o", color="#0072B2", capsize=3, label="Best (mean +/- seed SD)")
    axes[0].scatter([r["last_mean"] for r in summary], y, marker="s", color="#D55E00", label="Final (mean)")
    axes[0].set_yticks(y, methods)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 4.3)
    axes[0].set_xlabel("Validation AP (0-100)")
    axes[0].legend(fontsize=8, loc="lower left")
    axes[0].set_title("Three training seeds; one 400/100 split")
    axes[1].scatter([r["total_mean"] for r in summary], means, color="#0072B2", s=32)
    selected = {
        "cache1": "C1",
        "cache2": "C2",
        "transport-c1": "Transport C1",
        "transport-c2": "Transport C2",
        "loffta-c2-n005": "LOFF-TA-style C2",
    }
    for r in summary:
        if r["method"] in selected:
            axes[1].annotate(
                selected[r["method"]],
                (r["total_mean"], r["best_mean"]),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=8,
            )
    axes[1].set_ylim(0, 4.3)
    axes[1].set_xlabel("Archived build charge + current wall time (s)")
    axes[1].set_ylabel("Best validation AP (0-100)")
    axes[1].set_title("Cost comparison within the new serial sweep")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"baseline-comparison.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    colors = ["#555555", "#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    selected_methods = ["cache1", "cache2", "transport-c1", "transport-c2", "loffta-c2-n005"]
    x = list(range(1, 101))
    for method, color in zip(selected_methods, colors):
        epoch_values = list(zip(*(curves[method][s] for s in range(3))))
        avg = [statistics.mean(v) for v in epoch_values]
        sd = [statistics.stdev(v) for v in epoch_values]
        ax.plot(x, avg, label=method, color=color)
        ax.fill_between(
            x, [m - s for m, s in zip(avg, sd)], [m + s for m, s in zip(avg, sd)], color=color, alpha=0.12, linewidth=0
        )
    ax.set(
        xlabel="Epoch",
        ylabel="Validation AP (0-100)",
        ylim=(0, 4.3),
        title="Training dynamics: seed mean and +/- one seed SD",
    )
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"training-dynamics.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    appendix = [
        "# Statistical appendix",
        "",
        "This is a descriptive, exploratory pilot, not a confirmatory benchmark.",
        "The unit is a training seed (n=3), conditional on one fixed 400/100-image split.",
        "Epochs and individual detections are not treated as independent replicates.",
        "Best AP uses validation checkpoint selection; two brightness strengths were also explored.",
        "All settings are reported, rather than publishing only the best strength.",
        "",
        "The tables include mean differences in AP points and paired differences.",
        "Paired two-sided exact sign-flip tests enumerate all 2^3 sign assignments.",
        "Their null requires sign exchangeability/symmetry of paired differences.",
        "With three seeds the smallest attainable two-sided p-value is 0.25.",
        "Holm adjustment is applied to the nine exploratory best-AP contrasts.",
        "Thus these tests cannot establish conventional significance in this experiment.",
        "",
        "Student-t 95% intervals use df=2 and multiplier 4.30265273.",
        "Normality cannot be meaningfully assessed with three seeds; these intervals",
        "are assumption-dependent descriptors, not evidence of generalization across datasets.",
        "Per-method intervals and seed SDs are available in analysis.json and summary.csv.",
        "",
        "| Contrast | Best AP delta | Paired seed deltas | Paired t 95% interval | Exact p | Holm p | Last AP delta |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for r in contrasts:
        deltas = ", ".join(f"{v:+.4f}" for v in r["paired_best_delta"])
        appendix.append(
            f"| {r['method']} - {r['reference']} | {r['best_delta_mean']:+.4f} | {deltas} | "
            f"[{r['t95_low']:+.4f}, {r['t95_high']:+.4f}] | {r['sign_flip_p']:.3f} | "
            f"{r['holm_p']:.3f} | {r['last_delta_mean']:+.4f} |"
        )
    (root / "stats-appendix.md").write_text("\n".join(appendix) + "\n", encoding="utf-8")
    report = [
        "# Feature augmentation pilot: analysis",
        "",
        "Question: with the same frozen DINOv3, mean-value head, split and training recipe,",
        "do direct cached-feature augmentations improve detection enough to form stronger baselines?",
        "Evidence: 30 completed runs, 100 epochs each, three seeds per method.",
        "Source hashes, actual run settings, complete curves and summary consistency were checked.",
        "AP has not been independently recomputed from model predictions.",
        "",
        "## Numeric results",
        "",
        *table,
        "",
        "## Findings and decision",
        "",
        "- C1 transport improves mean best AP from 2.5694 to 3.1775 (+0.6081).",
        "  The difference is positive in each of the three seeds.",
        "- The real two-view cache remains a strong control (3.6405 best AP).",
        "- Transport C2 gives 3.7666 best AP and 3.4723 final AP. Its best-AP gain",
        "  over C2 is not positive for every seed; the final-AP gain is positive for all three.",
        "- LOFF-TA-style C2 gives 3.7675 best AP: only about +0.0009 over Transport C2,",
        "  with lower mean final AP (3.4479 versus 3.4723). Mean training time is",
        "  232.2 versus 221.5 seconds. Prioritize Transport C2 as the simpler baseline.",
        "- FroFA delta 0.1 helps C1 on mean best AP; delta 0.3 does not.",
        "  Neither brightness strength improves C2 mean best AP in this sweep.",
        "  These findings are limited to these detection adaptations and two strengths.",
        "",
        "## Claim candidates",
        "",
        "1. Keep: direct geometric feature augmentation can be useful when training this detector.",
        "   Evidence: transport-c1 versus cache1, paired seeds in stats-appendix.md.",
        "   Do not claim it exactly reconstructs image-augmented DINO features or is universally effective.",
        "2. Revise: an earlier 'Transport is near random' observation cannot describe this protocol.",
        "   Check whether that experiment replaced features under a fixed head, used another head,",
        "   or used different preprocessing/geometry. The current runs retrain the detector.",
        "3. Weaken: the tiny mean advantage of adding noise to C2 is not evidence of superiority.",
        "   See paired seed differences; n=3 does not support a robust winner claim.",
        "4. Defer: there is no online-corrector result in this campaign. No claim of superior",
        "   correction, cross-domain generalization, official AP-small or CVPR readiness follows.",
        "",
        "## Next experiment",
        "",
        "Compare C2-Mix, C2-Transport and C2-Correct at 10% and 25% matched backbone-query",
        "budgets and seeds 0/1/2. Share query samples, transforms, cached anchors and head",
        "initialization. Count corrector time and all feature-extraction calls.",
        "Use the no-query Transport C2 result as the strong cheap reference.",
        "Then advance a short list to official splits/evaluation, small-object AP and equal GPU time.",
        "",
        "## Interpretation limits",
        "",
        "This remains 400/100 VisDrone images with a custom COCO-style evaluator.",
        "Best AP is selected on the same small validation set; report final AP too.",
        "FroFA/LOFF-TA are idea adaptations, not full reproductions of the classification papers.",
        "Geometric feature padding uses zeros; augmentation strength and boundary handling matter.",
        "Training caches are GPU resident; full-dataset I/O and memory will differ.",
        "The build charge is the archived initial training-cache cost, added once per run.",
        "Actual additional cache build cost for this sweep is zero. Validation-cache build is excluded.",
        "No raw-image inference acceleration or foundation-backbone removal is established.",
        "",
        "## Figures",
        "",
        "![Baseline comparison](figures/baseline-comparison.png)",
        "",
        "The first panel compares best/final AP and seed variability; the second includes",
        "the charged initial cache build. The tiny C2 noise gain has no clear advantage in cost.",
        "Error bars are seed SD, not confidence intervals.",
        "",
        "![Training dynamics](figures/training-dynamics.png)",
        "",
        "Inspect final-epoch retention as well as selected best AP. Cached spatial augmentation",
        "retains more of its best performance than the simple two-view control in this pilot.",
        "Bands indicate +/- one seed SD on the same fixed split.",
    ]
    (root / "analysis-report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    catalog = """# Figure catalog

Both figures are generated by online_experiments/analyze_feature_baselines.py
from hash-checked original/train/*/metrics.csv and summary.json.

## baseline-comparison.png / .pdf
- Purpose: compare best/final AP and measured training-campaign cost.
- Left: best AP mean +/- sample SD over 3 seeds, and final AP mean.
- Right: best AP versus archived training-cache build charge + new wall time.
- Notice: similar best AP for Transport C2 and LOFF-TA-style C2.
- Limits: selected validation best, one split, archived build accounting, no significance claims.

## training-dynamics.png / .pdf
- Purpose: expose checkpoint selection and final-epoch performance.
- Variables: epoch versus AP, mean curve and +/- one sample SD over 3 seeds.
- Notice: augmentation improves final performance retention in this pilot.
- Limits: epochs are correlated; the shaded area is not a confidence interval.
"""
    (root / "figure-catalog.md").write_text(catalog, encoding="utf-8")
    print(json.dumps({"runs": 30, "summary": summary, "contrasts": contrasts}, indent=2))


if __name__ == "__main__":
    main()
