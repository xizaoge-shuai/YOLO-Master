"""Audit the locked native-gradient screen and paired spatial interventions."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED = 20261019
ARMS = [
    "transport",
    "blank_fill",
    "padding_oracle",
    "rec",
    "content_oracle",
    "blank_content_oracle",
    "joint_oracle",
    "exact",
]
LABELS = [
    "Transport",
    "Blank\nfill",
    "Padding\noracle",
    "Rec",
    "Content\noracle",
    "Blank +\ncontent*",
    "Joint\noracle",
    "Exact",
]
HEADS = ["online-geom", "mix25"]


def sha(path):
    """Hash exact transferred bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prefix(index):
    """Use filename groups as sensitivity clusters, not verified independent videos."""
    return pd.Index(index).map(lambda x: str(x).split("/")[-1].split("_")[0])


def bootstrap_ratio(numerator, denominator):
    """Resample shared filename groups for a ratio of sums, retaining pairing."""
    assert numerator.index.equals(denominator.index)
    frame = pd.DataFrame({"n": numerator, "d": denominator, "group": prefix(numerator.index)})
    groups = frame.groupby("group")[["n", "d"]].sum()
    draws = np.random.default_rng(SEED).integers(len(groups), size=(2000, len(groups)))
    numerators = groups.n.to_numpy()[draws].sum(1)
    denominators = groups.d.to_numpy()[draws].sum(1)
    assert (denominators > 0).all()
    lo, hi = np.quantile(numerators / denominators, [0.025, 0.975])
    return {"prefix_groups": len(groups), "prefix_boot_low": lo, "prefix_boot_high": hi}


def describe(series):
    """Image-level mean and paired filename-bootstrap sensitivity interval."""
    assert len(series) == 128 and series.notna().all()
    return {
        "n": len(series),
        "mean": series.mean(),
        "sd": series.std(ddof=1),
        "median": series.median(),
        "q25": series.quantile(0.25),
        "q75": series.quantile(0.75),
        **bootstrap_ratio(series, pd.Series(1.0, index=series.index)),
    }


def save(fig, output, name):
    """Export vector versions and a local preview."""
    for ext in ("pdf", "svg", "png"):
        path = output / "figures" / (name + "." + ext)
        fig.savefig(path, dpi=240, bbox_inches="tight")
        if ext == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n",
                encoding="utf-8",
            )
    plt.close(fig)


def analyze(raw, output, previous=None):
    """Recompute gates from raw rows before making any interpretation."""
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    summary = json.loads((raw / "summary.json").read_text(encoding="utf-8"))
    assert summary["images"] == 128 and summary["states_unchanged"] and summary["training_updates"] == 0
    assert summary["teacher_images"] == 768 and summary["blank_teacher_images"] == 1
    assert summary["native_loss_parity_max_absolute"] == 0
    assert summary["expectation_max_relative_l2"] < 1e-6
    hashes = {}
    for path in raw.iterdir():
        if path.name in summary["outputs_sha256"]:
            hashes[path.name] = sha(path)
            assert hashes[path.name] == summary["outputs_sha256"][path.name], path.name
    native, params, outputs = [
        pd.read_csv(raw / name) for name in ("native-gradients.csv", "parameters.csv", "per-image.csv")
    ]
    keys = ["image_id", "head", "arm", "flip", "scale"]
    for frame, count, extra in [
        (native, 98304, ["block"]),
        (params, 344064, ["branch", "task", "block"]),
        (outputs, 24576, ["branch"]),
    ]:
        assert len(frame) == count and not frame.duplicated(keys + extra).any()
        assert frame.image_id.nunique() == 128 and set(frame.arm) == set(ARMS)
        assert not np.isinf(frame.select_dtypes("number").to_numpy()).any()
    energies = ["exact_squared_norm", "approximate_squared_norm", "residual_squared_norm"]
    total = native[native.block.eq("__all__")].set_index(keys).sort_index()
    blocks = native[native.block.ne("__all__")].groupby(keys)[energies].sum().sort_index()
    assert total.index.equals(blocks.index)
    np.testing.assert_allclose(total[energies], blocks[energies], rtol=1e-12, atol=1e-8)
    np.testing.assert_allclose(native.variance_trace, 3 * native.residual_squared_norm, rtol=1e-12)
    np.testing.assert_allclose(native.query_only_variance_trace, 3 * native.exact_squared_norm, rtol=1e-12)
    # Independently reconstruct selected outcome energy: ||4 ge - 3 ga||^2.
    dot = (total.exact_squared_norm + total.approximate_squared_norm - total.residual_squared_norm) / 2
    selected_energy = 16 * total.exact_squared_norm + 9 * total.approximate_squared_norm - 24 * dot
    np.testing.assert_allclose(
        selected_energy, total.selected_norm_ratio.pow(2) * total.exact_squared_norm, rtol=1e-10, atol=1e-7
    )
    assert total.loc[total.index.get_level_values("arm") == "exact", "residual_squared_norm"].eq(0).all()
    rows, block_rows, scalar_rows, influence = [], [], [], []
    selected_native = native[native.scale.ne(1)]
    for key, frame in selected_native.groupby(["head", "arm", "scale", "block"]):
        g = frame.groupby("image_id")[energies].sum()
        e, a, residual = g.exact_squared_norm.sum(), g.approximate_squared_norm.sum(), g.residual_squared_norm.sum()
        item = dict(zip(["head", "arm", "scale", "block"], key))
        item.update(
            images=len(g),
            exact_energy=e,
            approximate_energy=a,
            residual_energy=residual,
            ratio=residual / e if e else None,
        )
        block_rows.append(item)
        if key[-1] != "__all__":
            continue
        assert len(g) == 128 and (g.exact_squared_norm > 0).all()
        ratio = g.residual_squared_norm / g.exact_squared_norm
        item = item | {
            "image_ratio_median": ratio.median(),
            "image_ratio_q25": ratio.quantile(0.25),
            "image_ratio_q75": ratio.quantile(0.75),
            "image_ratio_q95": ratio.quantile(0.95),
            "images_ratio_below_one": int(ratio.lt(1).sum()),
            "top_image_residual_share": g.residual_squared_norm.max() / residual if residual else 0,
            **bootstrap_ratio(g.residual_squared_norm, g.exact_squared_norm),
        }
        rows.append(item)
        # Post-hoc, same-data oracle bound for ONE constant scalar per head/scale.
        # It is not a trained/held-out estimator and cannot override the locked gate.
        pooled_dot = (e + a - residual) / 2
        alpha = pooled_dot / a if a else 0
        bound = (e - pooled_dot**2 / a) / e if a and e else None
        scalar_rows.append(
            {
                **dict(zip(["head", "arm", "scale"], key[:3])),
                "oracle_alpha": alpha,
                "same_data_oracle_variance_ratio": bound,
                "status": "posthoc_same_data_constant_scalar_oracle_not_validation",
            }
        )
        ranked = g.sort_values("residual_squared_norm", ascending=False)
        for removed in (0, 1, 5):
            remaining = ranked.iloc[removed:]
            influence.append(
                {
                    **dict(zip(["head", "arm", "scale"], key[:3])),
                    "removed_top_images": removed,
                    "remaining_ratio": remaining.residual_squared_norm.sum() / remaining.exact_squared_norm.sum(),
                    "status": "sensitivity_only_locked_gate_uses_all_images",
                }
            )
    ratios = pd.DataFrame(rows)
    locked = json.loads((raw / "screening.json").read_text(encoding="utf-8"))
    for cell in locked["cells"]:
        found = ratios[
            (ratios["head"] == cell["head"]) & (ratios.arm == cell["arm"]) & (ratios.scale == cell["scale"])
        ].iloc[0]
        assert np.isclose(found.ratio, cell["variance_ratio"], rtol=1e-12)
        assert bool(found.ratio < 1) == cell["passes_cell"]
    eligible = [arm for arm in ("transport", "blank_fill", "rec") if ratios[ratios.arm.eq(arm)].ratio.lt(1).all()]
    assert eligible == locked["eligible_for_minibatch_probe"] == []
    chunks = []
    units = ["head", "arm", "scale", "image_id"]
    for task, block, branch, label in [
        ("box", "p3", "one2many", "p3_box_cosine"),
        ("box", "one2many_box", "one2many", "box_head_cosine"),
        ("box", "one2one_box", "one2one", "one2one_box_head_cosine"),
    ]:
        f = params[params.task.eq(task) & params.block.eq(block) & params.branch.eq(branch) & params.scale.ne(1)]
        chunks.append(f[units + ["cosine"]].rename(columns={"cosine": "value"}).assign(metric=label))
    for field in ("cls_cosine", "box_cosine"):
        f = outputs[outputs.branch.eq("one2many") & outputs.scale.ne(1)]
        chunks.append(f[units + [field]].rename(columns={field: "value"}).assign(metric="output_" + field))
    long = pd.concat(chunks, ignore_index=True)
    assert long.groupby(units + ["metric"]).value.count().eq(2).all()
    image = long.groupby(units + ["metric"]).value.mean().reset_index()
    cells = []
    for key, frame in image.groupby(["head", "arm", "scale", "metric"]):
        cells.append(dict(zip(["head", "arm", "scale", "metric"], key)) | describe(frame.set_index("image_id").value))
    contrasts = []
    for key, frame in image.groupby(["head", "scale", "metric"]):
        wide = frame.pivot(index="image_id", columns="arm", values="value")
        for first, second in [
            ("joint_oracle", "transport"),
            ("joint_oracle", "content_oracle"),
            ("joint_oracle", "padding_oracle"),
            ("blank_content_oracle", "content_oracle"),
            ("blank_content_oracle", "blank_fill"),
            ("joint_oracle", "blank_content_oracle"),
            ("rec", "blank_fill"),
        ]:
            diff = wide[first] - wide[second]
            contrasts.append(
                dict(zip(["head", "scale", "metric"], key))
                | {
                    "contrast": first + "-" + second,
                    "positive_images": int(diff.gt(0).sum()),
                    "negative_images": int(diff.lt(0).sum()),
                    **describe(diff),
                }
            )
    cells, contrasts = pd.DataFrame(cells), pd.DataFrame(contrasts)
    prior_audit = {"checked": False}
    if previous is not None:
        previous_summary = json.loads((previous / "summary.json").read_text(encoding="utf-8"))
        assert sha(previous / "per-image.csv") == previous_summary["outputs_sha256"]["per-image.csv"]
        old = pd.read_csv(previous / "per-image.csv")
        joined = old.merge(outputs, on=keys + ["branch"], suffixes=("_old", "_new"), validate="one_to_one")
        assert len(joined) == 18432
        prior_audit = {
            "checked": True,
            "shared_rows": len(joined),
            "old_input_sha256": sha(previous / "per-image.csv"),
            "max_absolute_deltas": {
                k: float((joined[k + "_old"] - joined[k + "_new"]).abs().max())
                for k in ("cls_cosine", "box_cosine", "native_loss")
            },
        }
        assert max(prior_audit["max_absolute_deltas"].values()) == 0
    for name, frame in [
        ("variance-screen.csv", ratios),
        ("native-block-energies.csv", pd.DataFrame(block_rows)),
        ("posthoc-scalar-oracle.csv", pd.DataFrame(scalar_rows)),
        ("influence-sensitivity.csv", pd.DataFrame(influence)),
        ("spatial-cells.csv", cells),
        ("paired-contrasts.csv", contrasts),
    ]:
        frame.to_csv(output / name, index=False)
    audit = {
        "input_hashes": hashes,
        "summary_sha256": sha(raw / "summary.json"),
        "source_runtime_seconds": summary["elapsed_seconds"],
        "images": 128,
        "heads": HEADS,
        "training_seeds_per_head": 1,
        "prefix_groups": 46,
        "bootstrap_draws": 2000,
        "bootstrap_seed": SEED,
        "eligible": eligible,
        "prior_six_arms": prior_audit,
        "source_job": "runs/paper/joint-gradient/job-20261018-pJHRhd15",
        "gate_unchanged": True,
        "inference": "Descriptive pointwise filename-bootstrap sensitivity only; no population or training-seed tests.",
    }
    (output / "verification.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    make_figures(ratios, cells, output)
    print(json.dumps(audit, indent=2))


def make_figures(ratios, cells, output):
    """Separate actual native-loss variance from fixed-assignment spatial diagnostics."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    plot_arms = ["transport", "blank_fill", "rec", "blank_content_oracle", "joint_oracle"]
    colors = ["#777777", "#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    labels = ["Transport", "Blank fill", "Rec", "Blank + true content (oracle)", "True padding + content (oracle)"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5))
    for panel, (ax, head) in enumerate(zip(axes, HEADS)):
        for j, (arm, color, label) in enumerate(zip(plot_arms, colors, labels)):
            f = ratios[ratios["head"].eq(head) & ratios.arm.eq(arm)].sort_values("scale")
            x = np.arange(2) + (j - 2) * 0.11
            err = np.vstack([f.ratio - f.prefix_boot_low, f.prefix_boot_high - f.ratio])
            ax.errorbar(
                x,
                f.ratio,
                yerr=err,
                fmt="o",
                markersize=6,
                color=color,
                capsize=3,
                mfc="white" if "oracle" in arm else color,
                label=label,
            )
        ax.axhline(1, color="black", linestyle="--", linewidth=1)
        ax.set_yscale("log")
        ax.set_xticks([0, 1], ["0.85 (shrink)", "1.15 (enlarge)"])
        ax.set_xlim(-0.4, 1.4)
        ax.set_xlabel("Image scale")
        ax.set_ylabel("Calibration variance / query-only variance")
        ax.set_title(f"{chr(97 + panel)}  {'Online100' if head == 'online-geom' else 'Mix25'} frozen head")
        ax.grid(axis="y", alpha=0.2)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=3, frameon=False, fontsize=9)
    fig.text(
        0.5,
        -0.03,
        "Native own assignments; p=0.25 enumeration. Dashed line: query-only reference. Panel y-ranges differ.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout()
    save(fig, output, "figure-01-native-variance-screen")
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.6), sharey=True)
    for i, head in enumerate(HEADS):
        for j, scale in enumerate((0.85, 1.15)):
            ax = axes[i, j]
            f = (
                cells[cells["head"].eq(head) & cells.scale.eq(scale) & cells.metric.eq("p3_box_cosine")]
                .set_index("arm")
                .loc[ARMS]
            )
            err = np.vstack([f["mean"] - f.prefix_boot_low, f.prefix_boot_high - f["mean"]])
            ax.errorbar(np.arange(8), f["mean"], yerr=err, fmt="o", color="#0072B2", capsize=3, markersize=5)
            ax.axhline(0, color="#888888", linewidth=0.7)
            ax.axhline(1, color="#888888", linestyle="--", linewidth=0.7)
            ax.set_xticks(np.arange(8), LABELS, rotation=35, ha="right", fontsize=9)
            ax.set_ylim(-0.45, 1.08)
            ax.set_title(f"{chr(97 + 2 * i + j)}  {'Online100' if head == 'online-geom' else 'Mix25'}, scale={scale}")
            ax.set_ylabel("P3 localization parameter-gradient cosine")
            ax.grid(axis="y", alpha=0.18)
    fig.suptitle(
        "Joint spatial restoration improves localization gradients; cheap padding is head dependent", fontsize=12
    )
    fig.text(
        0.5,
        -0.015,
        "Fixed exact assignments; two flips averaged per image. 128 images; one seed per head. *Content uses true features.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout()
    save(fig, output, "figure-02-joint-spatial-restoration")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    args = parser.parse_args()
    analyze(args.raw, args.output, args.previous)
