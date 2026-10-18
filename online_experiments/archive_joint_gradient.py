"""Verify and archive joint spatial/native-gradient diagnostics; optionally push the recorded feature branch."""

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from online_experiments.archive_assignment_intervention import atomic_json, commit_files, git, sha, write_preserving

HEADS = {"online-geom", "mix25"}
ARMS = {
    "exact",
    "transport",
    "rec",
    "padding_oracle",
    "content_oracle",
    "blank_fill",
    "joint_oracle",
    "blank_content_oracle",
}
BLOCKS = {"p3", "p4", "p5", "one2many_box", "one2many_cls", "one2one_box", "one2one_cls"}


def verify(stage, count):
    """Check measured row grids and source hashes before any result is committed."""
    if (stage / "status.txt").read_text().strip() != "SUCCEEDED":
        raise ValueError("Stage incomplete")
    summary = json.loads((stage / "summary.json").read_text())
    if (summary["images"], summary["teacher_images"], summary["blank_teacher_images"]) != (count, count * 6, 1):
        raise ValueError("Query/image count mismatch")
    if summary["training_updates"] != 0 or summary["states_unchanged"] is not True:
        raise ValueError("State changed")
    if set(summary["heads"]) != HEADS or set(summary["arms"]) != ARMS:
        raise ValueError("Wrong experiment arms")
    for field, limit in [("parity_max_relative_l2", 0.01), ("native_loss_parity_max_absolute", 2e-5)]:
        if not math.isfinite(summary[field]) or not 0 <= summary[field] <= limit:
            raise ValueError("Invalid parity")
    names = {
        "per-image.csv",
        "parameters.csv",
        "assignments.csv",
        "regions.csv",
        "native-gradients.csv",
        "native-summary.csv",
        "screening.json",
        "summary.csv",
        "parameter-summary.csv",
        "region-summary.csv",
        "assignment-summary.csv",
        "selection.json",
        "blank-template.json",
        "cache-slices.json",
        "comparison.txt",
        "provenance.json",
    }
    if set(summary["outputs_sha256"]) != names:
        raise ValueError("Missing required hashes")
    for name, expected in summary["outputs_sha256"].items():
        if sha(stage / name) != expected:
            raise ValueError(f"Changed output: {name}")
    provenance = json.loads((stage / "provenance.json").read_text())
    for relative, expected in provenance["source_hashes"].items():
        path = (stage / "source" / relative).resolve()
        if not path.is_relative_to((stage / "source").resolve()) or sha(path) != expected:
            raise ValueError("Changed source snapshot")
    protocol = (stage / "source" / provenance["protocol"]).resolve()
    if not protocol.is_relative_to((stage / "source").resolve()) or sha(protocol) != provenance["protocol_sha256"]:
        raise ValueError("Changed protocol snapshot")
    selected = json.loads((stage / "selection.json").read_text())
    ids = {sample["sample_id"] for sample in selected["samples"]}
    if selected["split"] != "train" or len(ids) != count or provenance["count"] != count:
        raise ValueError("Selection mismatch")
    definitions = [
        ("per-image.csv", 192, ["head", "arm", "branch"]),
        ("parameters.csv", 2688, ["head", "arm", "branch", "task", "block"]),
        ("assignments.csv", 192, ["head", "arm", "branch"]),
        ("regions.csv", 720, ["arm", "layer", "region"]),
        ("native-gradients.csv", 768, ["head", "arm", "block"]),
    ]
    row_fields = ("output_rows", "parameter_rows", "matching_rows", "region_rows", "native_rows")
    for (filename, per_image, extra), row_field in zip(definitions, row_fields):
        if summary[row_field] != count * per_image:
            raise ValueError("Declared row count mismatch")
        seen = set()
        with (stage / filename).open(newline="") as stream:
            for row in csv.DictReader(stream):
                if (
                    row["image_id"] not in ids
                    or row["flip"] not in ("True", "False")
                    or float(row["scale"]) not in (1.0, 0.85, 1.15)
                ):
                    raise ValueError("Unexpected view")
                if row["arm"] not in ARMS or ("head" in row and row["head"] not in HEADS):
                    raise ValueError("Unexpected arm/head")
                if "branch" in row and row["branch"] not in ("one2many", "one2one"):
                    raise ValueError("Unexpected branch")
                if filename == "parameters.csv" and (row["block"] not in BLOCKS or row["task"] not in ("cls", "box")):
                    raise ValueError("Unexpected parameter group")
                if "region" in row and (
                    row["region"] not in ("new_padding", "letterbox", "foreground", "background", "mixed")
                    or int(row["layer"]) not in (3, 7, 11)
                ):
                    raise ValueError("Unexpected region")
                key = tuple(row[k] for k in ["image_id", "flip", "scale", *extra])
                if key in seen:
                    raise ValueError("Duplicate row")
                seen.add(key)
                for key, value in row.items():
                    if value.lower() in ("nan", "inf", "-inf"):
                        raise ValueError("Nonfinite metric")
                if filename == "native-gradients.csv":
                    if row["block"] not in BLOCKS | {"__all__"}:
                        raise ValueError("Unknown native parameter block")
                    nr, residual = float(row["exact_squared_norm"]), float(row["residual_squared_norm"])
                    variance, reference = float(row["variance_trace"]), float(row["query_only_variance_trace"])
                    if min(nr, residual, variance, reference) < 0:
                        raise ValueError("Negative energy")
                    if not math.isclose(variance, 3 * residual, rel_tol=1e-7, abs_tol=1e-10):
                        raise ValueError("Incorrect calibration variance")
                    if not math.isclose(reference, 3 * nr, rel_tol=1e-7, abs_tol=1e-10):
                        raise ValueError("Incorrect query-only variance")
                    if nr > 0 and (not row["expectation_relative_l2"] or float(row["expectation_relative_l2"]) > 1e-6):
                        raise ValueError("Native gradient expectation identity failed")
                    if nr == 0 and row["expectation_relative_l2"]:
                        raise ValueError("Undefined expectation ratio must be empty")
                    if row["arm"] == "exact" and residual != 0:
                        raise ValueError("Exact native gradient mismatch")
                if filename == "parameters.csv":
                    nr, nv = float(row["reference_norm"]), float(row["value_norm"])
                    if min(nr, nv) < 0:
                        raise ValueError("Negative norm")
                    if nr == 0 and any(row[k] for k in ("cosine", "relative_l2", "norm_ratio")):
                        raise ValueError("Undefined ratio reported as finite")
                    if nr > 0 and (not row["relative_l2"] or not row["norm_ratio"]):
                        raise ValueError("Missing defined norm ratio")
                    if nr > 0 and nv > 0 and not row["cosine"]:
                        raise ValueError("Missing defined cosine")
                    if row["arm"] == "exact" and nr > 0 and float(row["relative_l2"]) != 0:
                        raise ValueError("Exact parameter-gradient reference differs")
                    if row["branch"] == "one2one" and row["block"] in ("p3", "p4", "p5") and (nr != 0 or nv != 0):
                        raise ValueError("One2one shared feature detachment changed")
        if len(seen) != count * per_image:
            raise ValueError(f"Incomplete grid: {filename}")
    from online_experiments.diagnose_joint_gradient import screening_decision

    with (stage / "native-gradients.csv").open(newline="") as stream:
        native = list(csv.DictReader(stream))
    for row in native:
        for key in ("scale", "exact_squared_norm", "residual_squared_norm"):
            row[key] = float(row[key])
    expected_screen = screening_decision(native, count)
    if json.loads((stage / "screening.json").read_text()) != expected_screen:
        raise ValueError("Screening decision differs from raw data")
    if not 0 <= summary["expectation_max_relative_l2"] <= 1e-6:
        raise ValueError("Declared expectation identity failed")
    if summary["query_probability_for_enumeration"] != 0.25:
        raise ValueError("Wrong enumeration probability")
    return summary


def archive(job, stage_name):
    stage = job / stage_name
    summary = verify(stage, 2 if stage_name == "smoke" else 128)
    target = ROOT / "research/experiments" / ("joint-gradient-" + job.name) / stage_name
    target.mkdir(parents=True, exist_ok=True)
    names = [
        "summary.json",
        "summary.csv",
        "parameter-summary.csv",
        "region-summary.csv",
        "native-summary.csv",
        "screening.json",
        "assignment-summary.csv",
        "comparison.txt",
        "provenance.json",
        "blank-template.json",
        "cache-slices.json",
    ]
    for name in names:
        if (target / name).exists() and sha(target / name) != sha(stage / name):
            raise ValueError("Archive differs")
        shutil.copyfile(stage / name, target / name)
    selection = json.loads((stage / "selection.json").read_text())
    write_preserving(
        target / "selection-ids.json",
        json.dumps(
            {
                "split": selection["split"],
                "seed": selection["seed"],
                "indices": selection["indices"],
                "ids": [x["sample_id"] for x in selection["samples"]],
                "full_selection_sha256": sha(stage / "selection.json"),
            },
            indent=2,
        )
        + "\n",
    )
    readme = f"# Joint spatial/native-gradient intervention: {stage_name}\n\nSource job `{job.name}`; {summary['images']} fixed training images, {summary['teacher_images']} paired teacher queries plus one blank-template query. Both seed-0 frozen heads share each exact feature. Eight arms and six views; no optimizer updates or AP evaluation. Natural-area oracles do not control residual energy. Hard exact-assignment output/parameter gradients plus separate actual native-loss gradients. The p=0.25 estimator is analytically enumerated; all views are queried for this diagnostic. This is a batch1 frozen-eval screen, not a training cost or AP result. Passing only nominates a later minibatch probe; full training is not launched. Regional errors and all failed screening cells are retained. Zero-reference parameter gradients are undefined, especially one2one shared blocks.\n\nFull row data and source snapshots stay under runs/paper/joint-gradient; measured hashes are retained here. No weights or feature cache in Git.\n"
    write_preserving(target / "README.md", readme)
    marker = f"<!-- joint-gradient:{job.name}:{stage_name} -->"
    link = str(target.relative_to(ROOT / "research")) + "/README.md"
    progress = f"\n{marker}\n## Joint spatial/native-gradient intervention {job.name}: {stage_name} complete\n\n- [Verified archive]({link}); {summary['images']} training images; {summary['teacher_images']}+1 teacher queries, two frozen heads, eight arms. States unchanged, zero updates; no AP claim.\n"
    files = [str((target / name).relative_to(ROOT)) for name in [*names, "selection-ids.json", "README.md"]]
    return commit_files(
        ROOT,
        files,
        progress,
        marker,
        str((target / "README.md").relative_to(ROOT)),
        f"Record verified joint gradient intervention {stage_name}: {job.name}",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--stage", choices=("smoke", "development"), required=True)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    job = args.job.resolve()
    state_path = job / "archive-status.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"commits": {}, "pushes": {}}
    try:
        commit = archive(job, args.stage)
        state["commits"][args.stage] = commit
        state["status"] = "ARCHIVED"
        atomic_json(state_path, state)
    except BaseException as error:
        state["status"] = "ARCHIVE_FAILED"
        state["error"] = repr(error)
        atomic_json(state_path, state)
        raise
    if args.push:
        try:
            branch = git(ROOT, "branch", "--show-current")
            if branch != "d1-final-offline-dinov3-latent":
                raise RuntimeError("Unexpected push branch")
            if git(ROOT, "remote", "get-url", "--push", "origin") != "git@github.com:xizaoge-shuai/YOLO-Master.git":
                raise RuntimeError("Unexpected origin")
            # Push the exact verified result commit; never force or include later concurrent commits.
            p = subprocess.run(
                ["git", "push", "origin", f"{commit}:refs/heads/{branch}"],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=120,
                check=False,
            )
            (job / f"push-{args.stage}.log").write_text(p.stdout + p.stderr)
            p.check_returncode()
            remote = git(ROOT, "ls-remote", "origin", f"refs/heads/{branch}").split()
            if not remote or remote[0] != commit:
                raise RuntimeError("Pushed commit is not the verified remote branch tip")
            state["pushes"][args.stage] = commit
        except (RuntimeError, OSError, subprocess.SubprocessError) as error:
            state["push_error"] = repr(error)
            print("Archive complete; push needs attention:", error, flush=True)
        atomic_json(state_path, state)
    print(json.dumps(state, indent=2), flush=True)


if __name__ == "__main__":
    main()
