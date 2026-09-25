#!/usr/bin/env python3
"""Analyze raw paired-budget pilot evidence without changing any training source."""

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

T95_DF2 = 4.302652729911275


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def interval(values):
    avg, sd = st.mean(values), st.stdev(values)
    half = T95_DF2 * sd / math.sqrt(len(values))
    return {"mean": avg, "sample_sd": sd, "t95_assumption_dependent": [avg - half, avg + half]}


def exact_sign_flip(values):
    observed = abs(st.mean(values))
    scores = [
        abs(st.mean(s * v for s, v in zip(signs, values))) for signs in itertools.product((-1, 1), repeat=len(values))
    ]
    return sum(s >= observed - 1e-12 for s in scores) / len(scores)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive", type=Path)
    a = p.parse_args()
    root = a.archive.resolve()
    manifest = read(root / "archive-manifest.json")
    for rel, meta in manifest["files"].items():
        if hashlib.sha256((root / rel).read_bytes()).hexdigest() != meta["sha256"]:
            raise RuntimeError(f"Archived raw evidence changed: {rel}")
    plan = read(root / "original/plan.json")
    provenance = read(root / "original/provenance.json")
    split = read(root / "original/reference_split.json")
    groups, curves, corr = {}, {}, {}
    for method, seed in plan["tasks"]:
        run = root / "original/train" / f"{method}-s{seed}"
        args, summary = read(run / "args.json"), read(run / "summary.json")
        with (run / "metrics.csv").open(newline="") as stream:
            rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
        assert (run / "DONE").exists()
        assert [r["epoch"] for r in rows] == list(range(1, 101))
        assert all(math.isfinite(v) for r in rows for v in r.values())
        assert summary["source_hashes"] == args["source_hashes"] == provenance["sources"]
        current = read(run / "split.json")
        assert all(current[k] == split[k] for k in split)
        assert len(current["train_sample_ids"]) == 400 and len(current["val_sample_ids"]) == 100
        assert not set(current["train_sample_ids"]) & set(current["val_sample_ids"])
        pct = int(method.rsplit("p", 1)[1])
        assert summary["training_teacher_images"] == 40000 * pct // 100
        assert summary["training_samples"] == 40000
        assert summary["setup_teacher_images"] == 4
        for field, expected in (
            ("best_map50_95", max(r["map50_95"] for r in rows)),
            ("final_map50_95", rows[-1]["map50_95"]),
            ("train_seconds", sum(r["train_seconds"] for r in rows)),
            ("validation_seconds", sum(r["val_seconds"] for r in rows)),
        ):
            assert math.isclose(summary[field], expected, rel_tol=1e-10, abs_tol=1e-10)
        groups.setdefault(method, {})[seed] = summary
        curves[(method, seed)] = rows
        if "-correct-" in method:
            entries = [json.loads(s) for s in (run / "correction.jsonl").read_text().splitlines()]
            assert len(entries) == 100
            assert entries[-1]["correction_steps"] == summary["training_teacher_calls"]
            corr[(method, seed)] = entries
    assert len(groups) == 6 and all(set(g) == {0, 1, 2} for g in groups.values())
    for pct in (10, 25):
        for seed in (0, 1, 2):
            for field in ("schedule_sha256", "query_sha256", "training_teacher_images"):
                assert len({groups[f"c2-{arm}-p{pct}"][seed][field] for arm in ("mix", "transport", "correct")}) == 1

    summary_rows, contrasts, diagnostics = [], [], []
    for method in plan["cases"]:
        g = groups[method]
        best = interval([100 * g[s]["best_map50_95"] for s in (0, 1, 2)])
        last = interval([100 * g[s]["final_map50_95"] for s in (0, 1, 2)])
        summary_rows.append(
            {
                "method": method,
                "n": 3,
                "best_AP": best,
                "final_AP": last,
                "train_mean": st.mean(x["train_seconds"] for x in g.values()),
                "train_median": st.median(x["train_seconds"] for x in g.values()),
                "val_mean": st.mean(x["validation_seconds"] for x in g.values()),
                "build_plus_wall_mean": st.mean(x["build_plus_run_seconds"] for x in g.values()),
            }
        )
    for pct in (10, 25):
        for ref in ("mix", "transport"):
            for metric, key in (("best", "best_map50_95"), ("final", "final_map50_95")):
                d = [
                    100 * (groups[f"c2-correct-p{pct}"][s][key] - groups[f"c2-{ref}-p{pct}"][s][key]) for s in (0, 1, 2)
                ]
                contrast = dict(
                    budget=pct,
                    reference=ref,
                    metric=metric,
                    seed_differences=d,
                    positive_seeds=sum(x > 0 for x in d),
                    **interval(d),
                    exact_sign_flip_p=exact_sign_flip(d),
                )
                contrast["paired_cohen_dz"] = st.mean(d) / st.stdev(d) if st.stdev(d) else None
                contrasts.append(contrast)
    running = 0.0
    for rank, i in enumerate(sorted(range(len(contrasts)), key=lambda i: contrasts[i]["exact_sign_flip_p"])):
        running = max(running, min(1.0, (len(contrasts) - rank) * contrasts[i]["exact_sign_flip_p"]))
        contrasts[i]["holm_p_eight_contrasts"] = running
    for (method, seed), entries in corr.items():
        item = {"method": method, "seed": seed}
        for label, block in (("first10", entries[:10]), ("last10", entries[-10:])):
            numerator = st.mean(x["preupdate_query_relative_mse"] for x in block)
            denominator = st.mean(x["transport_query_relative_mse"] for x in block)
            item[label] = {
                "corrected_mse": numerator,
                "transport_mse": denominator,
                "fractional_reduction": 1 - numerator / denominator,
                "residual_relative_mse": st.mean(x["residual_relative_mse"] for x in block),
            }
        diagnostics.append(item)
    anomaly = groups["c2-transport-p25"][1]
    anomaly_epochs = [int(r["epoch"]) for r in curves[("c2-transport-p25", 1)] if r["val_seconds"] > 2.0]
    # This is a transparent timing flag, not a rule to drop a seed's AP.
    timing = {
        "flagged_run": "c2-transport-p25-s1",
        "train_seconds": anomaly["train_seconds"],
        "validation_seconds": anomaly["validation_seconds"],
        "flagged_epochs_val_over_2s": anomaly_epochs,
        "all_seeds_retained": True,
        "unobserved_cause": "No contemporaneous GPU/CPU/I/O telemetry",
        "p25_correct_over_transport_median_run_train_ratio": st.median(
            x["train_seconds"] for x in groups["c2-correct-p25"].values()
        )
        / st.median(x["train_seconds"] for x in groups["c2-transport-p25"].values()),
        "p25_seed02_ratios": [
            groups["c2-correct-p25"][s]["train_seconds"] / groups["c2-transport-p25"][s]["train_seconds"]
            for s in (0, 2)
        ],
        "warning": "Median/seed02 are sensitivity diagnostics, not replacement benchmark results.",
    }
    payload = {
        "schema_version": 1,
        "raw_archive_sha256": hashlib.sha256((root / "archive-manifest.json").read_bytes()).hexdigest(),
        "evidence_files_verified": len(manifest["files"]),
        "runs": 18,
        "summary": summary_rows,
        "paired_contrasts": contrasts,
        "correction_diagnostics": diagnostics,
        "timing": timing,
        "unit": "training seed conditional on one fixed 400/100 split",
        "limitations": [
            "Best AP is selected on repeatedly inspected pilot validation; final AP also reported.",
            "Three seeds do not establish dataset-level uncertainty; no method superiority or equivalence claim.",
            "t95 and paired dz assume an adequate paired-difference distribution; n=3 cannot establish normality.",
            "Exact sign-flip assumes paired-difference symmetry/exchangeability; smallest two-sided p is .25.",
            "All eight metric/reference/budget contrasts form one exploratory Holm correction family.",
            "Reconstruction MSE is pre-update on queried TRAINING images, not held-out feature recovery.",
            "Raw predictions not reevaluated; existing custom D1 metric is not official full-dataset AP.",
            "Timing anomaly retained; no claim of corrector speedup at p25.",
        ],
    }
    (root / "analysis.json").write_text(json.dumps(payload, indent=2) + "\n")
    table = [
        "| Method | Best AP mean +/- SD | Final AP mean +/- SD | Train s mean | Train s median |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        table.append(
            f"| {r['method']} | {r['best_AP']['mean']:.4f} +/- {r['best_AP']['sample_sd']:.4f} | "
            f"{r['final_AP']['mean']:.4f} +/- {r['final_AP']['sample_sd']:.4f} | "
            f"{r['train_mean']:.1f} | {r['train_median']:.1f} |"
        )
    (root / "summary-table.md").write_text("\n".join(table) + "\n")

    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    colors = {"mix": "#666666", "transport": "#0072B2", "correct": "#D55E00"}
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6), sharey=True)
    for row, (metric, key) in enumerate((("Best validation", "best_map50_95"), ("Final validation", "final_map50_95"))):
        for col, pct in enumerate((10, 25)):
            ax = axes[row, col]
            for x, arm in enumerate(("mix", "transport", "correct")):
                vals = [100 * groups[f"c2-{arm}-p{pct}"][s][key] for s in (0, 1, 2)]
                ax.errorbar(x, st.mean(vals), yerr=st.stdev(vals), fmt="o", color=colors[arm], capsize=4, ms=7)
                ax.scatter([x - 0.12, x, x + 0.12], vals, s=16, color=colors[arm], alpha=0.5)
            ax.set_xticks([0, 1, 2], ["Mix", "Transport", "Correct"])
            ax.set_title(f"{metric} AP | {pct}% online")
            ax.set_ylabel("AP (0-100)")
            ax.set_ylim(2.9, 5.0)
            ax.grid(axis="y", alpha=0.2)
    fig.suptitle("400 train / 100 validation; 3 paired seeds; error bars = seed SD", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figures / f"budget-comparison.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.3))
    epochs = list(range(1, 101))
    for col, pct in enumerate((10, 25)):
        ax = axes[0, col]
        for key, label, color in (
            ("transport_query_relative_mse", "Transport", "#0072B2"),
            ("preupdate_query_relative_mse", "Correct (before update)", "#D55E00"),
        ):
            values = [st.mean(corr[(f"c2-correct-p{pct}", s)][i][key] for s in (0, 1, 2)) for i in range(100)]
            ax.plot(epochs, values, label=label, color=color, lw=1.4)
        ax.set_title(f"{pct}% budget: training-query feature error")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Relative MSE")
        ax.legend(fontsize=8)
    for col, field in enumerate(("train_seconds", "val_seconds")):
        ax = axes[1, col]
        for seed in (0, 1, 2):
            rows = curves[("c2-transport-p25", seed)]
            ax.plot(
                epochs,
                [r[field] for r in rows],
                label=f"Transport seed {seed}",
                color=["#999999", "#D55E00", "#0072B2"][seed],
                lw=1.3,
                alpha=0.9,
            )
        ax.set_title(f"25% Transport: {'training' if col == 0 else 'validation'} time")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Seconds per epoch")
        ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figures / f"reconstruction-and-timing.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)

    report = (
        """# Paired-budget pilot analysis

Question: does reconstruction-only online correction improve detection beyond
using the same fresh features with C2-Mix or C2-Transport?

"""
        + "\n".join(table)
        + """

The current corrector has not established an advantage over Transport. Best-AP
differences are negative for all three p10 seeds and two of three p25 seeds.
Final AP is negative on average at p10, almost tied on average at p25.
Mean improvements over Mix alone do not isolate a benefit beyond geometric transport.

Training-query reconstruction improves substantially: the final-ten-epoch relative
MSE reduction is about 52-54% at p10 and 62% at p25. This supports successful fitting
of the dense reconstruction objective on training queries, not held-out feature
generalization or task-relevant recovery. Objective mismatch, background dominance,
and a changing corrected-feature distribution are hypotheses to test, not findings.

Transport p25 seed1 has an anomalous training AND validation slowdown. Its 312.7 s
validation total contrasts with about 66 s elsewhere. The cause is unobserved.
Keep its AP and raw timing; do not report the aggregate 0.902 paired timing ratio
as an algorithmic speedup. A fresh interleaved timing control with resource telemetry
is required. Median timing is a sensitivity check only.

## Claim candidates

- Claim: reconstruction-only correction has no demonstrated AP advantage over matched Transport here.
  - Evidence: 18 raw 100-epoch CSVs, paired differences in analysis.json, budget-comparison figure.
  - Allowed: no consistent advantage on this fixed small pilot.
  - Forbidden: correction can never work, or the methods are statistically equivalent.
  - Uncertainty: three seeds and 100 development-validation images.
  - Next check: object-aware/task-aware diagnostics and a full VisDrone pilot.
  - Decision: keep.
- Claim: dense feature error falls while detector gains fail to follow.
  - Evidence: correction.jsonl and reconstruction-and-timing figure.
  - Allowed: a training-query feature/detection mismatch motivates a task-aware hypothesis.
  - Forbidden: proven foreground suppression or proven out-of-domain overfitting.
  - Uncertainty: no foreground/background or held-out feature decomposition yet.
  - Next check: matched unseen-image/augmentation feature probes, object-size and fixed-head analyses.
  - Decision: keep with qualifier.
- Claim: Correct is faster than Transport at p25.
  - Evidence: aggregate is dominated by a slowdown in one baseline run.
  - Allowed: timings need controlled remeasurement.
  - Forbidden: a 10% algorithmic speedup.
  - Uncertainty: no contemporaneous resource telemetry.
  - Next check: retain old data and repeat matched timing controls.
  - Decision: discard as a method claim.

## Decisions

Begin full-dataset loader/evaluator work and full VisDrone baseline profiling now.
Do not postpone those prerequisites until the corrector wins. Include this current
corrector as a diagnostic control in the first full VisDrone seed, then expand only
a short list. Limit additional small-pilot method searches to two specified changes:
object-balanced reconstruction, then the same loss plus task-aware supervision.
The raw archive is unchanged. No new training was launched by this analysis.
"""
    )
    (root / "analysis-report.md").write_text(report)
    stats = [
        "# Statistical appendix",
        "",
        *["- " + x for x in payload["limitations"]],
        "",
        "| Budget | Ref | Metric | Paired delta | Assumption-dependent t95 | Exact p | Holm p | Positive seeds |",
        "|---:|---|---|---:|---|---:|---:|---:|",
    ]
    for c in contrasts:
        lo, hi = c["t95_assumption_dependent"]
        stats.append(
            f"| {c['budget']} | {c['reference']} | {c['metric']} | {c['mean']:+.4f} | "
            f"[{lo:+.4f}, {hi:+.4f}] | {c['exact_sign_flip_p']:.3f} | {c['holm_p_eight_contrasts']:.3f} | {c['positive_seeds']}/3 |"
        )
    stats += [
        "",
        "Do not count epochs as independent repetitions or interpret p>.05 as equivalence.",
        "Absolute paired AP-point differences are the main effect size; paired Cohen dz is in analysis.json.",
        "No normality claim follows from a low-power n=3 normality test.",
        "Archived hashes and arithmetic consistency passed; this does not rerun prediction-based evaluation.",
    ]
    (root / "stats-appendix.md").write_text("\n".join(stats) + "\n")
    (root / "figure-catalog.md").write_text("""# Figure catalog

## budget-comparison.pdf / .png
- Purpose: compare matched methods at each online budget without hiding seed variability.
- Source: original/train/*/metrics.csv and summary.json; AP multiplied by 100 only.
- Variables: best/final validation AP; mean +/- sample SD; dots show three training seeds.
- Caption must state 400/100 internal split and selection of best on validation.
- Interpretation: compare Correct to Transport as well as Mix; p25 final averages nearly tie.
- Caveat: a small development pilot, not official benchmark or independent dataset uncertainty.

## reconstruction-and-timing.pdf / .png
- Purpose: separate successful dense reconstruction from detector benefit; expose timing anomaly.
- Source: correction.jsonl and epoch metrics.csv.
- Top: mean over three seeds of pre-update query relative MSE; queries are training images.
- Bottom: all three Transport p25 timing trajectories, including the anomalous seed.
- Caption must retain the anomaly and say that no hardware telemetry identifies its cause.
- Interpretation: lower reconstruction loss alone does not demonstrate task utility.
- Caveat: no held-out feature probe or foreground/object-size decomposition exists yet.
""")
    (root / ".gitignore").write_text("!figures/*.png\n")
    print(
        json.dumps(
            {
                "verified_files": len(manifest["files"]),
                "summary": summary_rows,
                "correction": diagnostics,
                "timing": timing,
                "contrasts": contrasts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
