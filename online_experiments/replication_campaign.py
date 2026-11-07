"""Run the fixed seed/baseline/ablation queue without changing historical experiments."""

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from online_experiments.detail_probe import dump, read, sha


def plan():
    cases = [(method, seed) for seed in (1, 2) for method in ("blank25", "detail25", "lowpass25")]
    cases += [("frofa-c2-d01", 0), ("loffta-c2-n005", 0), ("mix25", 0), ("online100", 0)]
    cases += [("blank0", 0), ("detail0", 0)]
    cases += [(method, seed) for seed in (1, 2) for method in ("transport25", "views25")]
    cases += [(method, seed) for seed in (1, 2) for method in ("frofa-c2-d01", "loffta-c2-n005")]
    return [{"method": method, "seed": seed, "name": f"{method}-s{seed}"} for method, seed in cases]


def write_status(job, message):
    (job / "status.txt").write_text(message + "\n", encoding="utf-8")
    print(message, flush=True)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def push():
    """Use existing GitHub CLI authentication without changing SSH host-key policy."""
    gh = Path(sys.executable).parent / "gh"
    if not gh.is_file():
        raise RuntimeError("Configured environment has no GitHub CLI")
    branch, head = git("branch", "--show-current"), git("rev-parse", "HEAD")
    if not branch:
        raise RuntimeError("Refuse publishing from detached HEAD")
    remote = git("config", "--get", "remote.origin.url")
    if remote.startswith("git@github.com:"):
        remote = "https://github.com/" + remote.split(":", 1)[1]
    if not remote.startswith("https://github.com/"):
        raise RuntimeError("Expected the configured GitHub repository")
    env = dict(os.environ)
    env.setdefault("GH_CONFIG_DIR", str(ROOT.parent / ".config/gh"))
    env["GIT_TERMINAL_PROMPT"] = "0"
    helper = "!" + shlex.quote(str(gh)) + " auth git-credential"
    subprocess.run(
        [
            "git",
            "-c",
            "credential.helper=",
            "-c",
            f"credential.helper={helper}",
            "push",
            remote,
            f"{head}:refs/heads/{branch}",
        ],
        cwd=ROOT,
        env=env,
        check=True,
        timeout=90,
    )
    remote_head = subprocess.check_output(
        ["git", "ls-remote", remote, f"refs/heads/{branch}"], cwd=ROOT, env=env, text=True, timeout=30
    ).split()[0]
    if remote_head != head:
        raise RuntimeError("Published branch tip differs from the committed result")
    return head


def sanitized(value):
    if isinstance(value, dict):
        return {sanitized(key): sanitized(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitized(item) for item in value]
    if isinstance(value, str):
        old_home = str(Path("/data/users") / ROOT.parent.name)
        return value.replace(str(ROOT), "repo:").replace(str(ROOT.parent), "user-data:").replace(old_home, "user-data:")
    return value


def publish_pending(job):
    head = push()
    for path in job.glob("archive-*.json"):
        record = read(path)
        subprocess.run(["git", "merge-base", "--is-ancestor", record["commit"], head], cwd=ROOT, check=True)
        record.update(pushed=True, pushed_head=head)
        record.pop("push_error", None)
        dump(path, record)
    if (job / "archive-complete.json").exists():
        write_status(job, "SUCCEEDED")


def archive(job, stage, paths, publish):
    from online_experiments.archive_assignment_intervention import commit_files

    destination = ROOT / "research/experiments/replication-suite" / job.name / stage
    destination.mkdir(parents=True, exist_ok=True)
    names = []
    for path in paths:
        target = destination / path.relative_to(job)
        target.parent.mkdir(parents=True, exist_ok=True)
        content = (
            (json.dumps(sanitized(read(path)), indent=2, allow_nan=False) + "\n")
            if path.suffix == ".json"
            else path.read_text()
        )
        if target.exists() and target.read_text() != content:
            raise RuntimeError(f"Refuse changed archive: {target.relative_to(ROOT)}")
        target.write_text(content, encoding="utf-8")
        names.append(target.relative_to(ROOT).as_posix())
    readme = destination / "README.md"
    readme.write_text(
        f"# Replication suite: {stage}\n\n"
        "Full VisDrone seed/baseline/ablation campaign. Seed repeats, adapted paper baselines, "
        "and zero-query ablations have explicit method contracts. Smoke metrics are not accuracy evidence. "
        "Paths in JSON are normalized for publication; raw records remain in the run directory.\n",
        encoding="utf-8",
    )
    names.append(readme.relative_to(ROOT).as_posix())
    marker = f"replication-suite:{job.name}:{stage}"
    block = f"\n\n<!-- {marker} -->\n## Replication suite {job.name}: {stage}\n\n"
    block += f"Archived [{stage}]({readme.relative_to(ROOT / 'research').as_posix()}). "
    block += "Fixed final official-reference AP for complete full runs; observed resource conditions retained.\n"
    commit = commit_files(
        ROOT, names, block, marker, readme.relative_to(ROOT).as_posix(), f"research: record replication suite {stage}"
    )
    result = {"stage": stage, "commit": commit, "pushed": False}
    if publish:
        try:
            result.update(pushed=True, pushed_head=push())
        except (RuntimeError, OSError, subprocess.SubprocessError) as error:
            result["push_error"] = str(error)
            print(f"PUSH PENDING {stage}: {error}", flush=True)
    dump(job / f"archive-{stage}.json", result)


def asset_check(job, assets):
    lineage = read(assets["lineage"])
    dump(
        job / "asset-check.json",
        {
            "old_metadata_unchanged": lineage["old_metadata_unchanged"],
            "cache_construction_costs_preserved": lineage["cache_construction_costs_preserved"],
            "verified_source_files": lineage["verified_source_files"],
            "verified_extra_banks": {
                name: {key: record[key] for key in ("expected_sha256", "actual_sha256", "matches_expected")}
                for name, record in lineage["verified_extra_banks"].items()
            },
            "dataset_manifest": lineage["dataset_manifest"],
            "source_cache_manifest": lineage["source_cache_manifest"],
            "seconds": lineage["seconds"],
            "lineage_sha256": sha(assets["lineage"]),
        },
    )


def asset_destination(job):
    """Retain interrupted metadata and retry in a separate directory before snapshot."""
    if (job / "provenance.json").exists():
        return Path(read(job / "provenance.json")["source_job"]).parent
    location = job / "asset-location.json"
    destination = Path(read(location)["destination"]) if location.exists() else job / "assets"
    if destination.exists() and any(destination.iterdir()) and not (destination / "complete.json").is_file():
        destination = job / f"assets-retry-{uuid.uuid4().hex[:8]}"
    dump(location, {"destination": str(destination)})
    return destination


def check_pairing(outputs):
    """Compare actual initial heads and requested schedules, including query masks."""
    by_seed, by_budget, compared = {}, {}, []
    for output in outputs:
        args, summary = read(output / "args.json"), read(output / "summary.json")
        head = read(output / "initialization.json")["head_state_sha256"]
        geometry = summary["geometry_schedule_sha256"]
        group = (args["seed"], args["train_images"], args["epochs"])
        expected = by_seed.setdefault(group, (head, geometry))
        if expected != (head, geometry):
            raise RuntimeError(f"Unpaired initial head or requested geometry: {output.name}")
        key = (*group, args["query_pct"])
        schedules = summary["schedule_sha256"]
        if by_budget.setdefault(key, schedules) != schedules:
            raise RuntimeError(f"Unpaired requested query schedule: {output.name}")
        compared.append(output.name)
    return {"compared": compared, "paired": True, "seed_groups": len(by_seed)}


def parity(source, cache, output):
    """Test fresh exact teacher features against the preserved original FP16 cache."""
    import numpy as np
    import torch

    from online_experiments.full_visdrone_data import make_extractor
    from ultralytics.data.foundation_cache import load_letterboxed_tensor

    meta, manifest = read(source / "dataset-manifest.json"), read(cache)
    teacher = make_extractor(meta["identity"])
    arrays = [np.load(manifest["parts"][f"train-v{view}"]["file"], mmap_mode="r") for view in (0, 1)]
    rows = []
    for view in (0, 1):
        images = []
        for sample in meta["splits"]["train"][:2]:
            image, info = load_letterboxed_tensor(sample["image_path"], imgsz=640)
            if info != sample["letterbox"]:
                raise RuntimeError("Fresh parity letterbox differs")
            images.append(image.flip(-1) if view else image)
        with torch.no_grad():
            exact = teacher(torch.stack(images))
        for level, layer in enumerate(meta["identity"]["layers"]):
            cached = torch.from_numpy(arrays[view][:2, level].copy()).float()
            target = exact[f"layer_{layer}"].detach().float().cpu()
            error = float((target - cached).norm() / target.norm().clamp_min(1e-8))
            if not np.isfinite(error) or error > 0.02:
                raise RuntimeError(f"Preserved cache and restored teacher mismatch: {error}")
            rows.append({"view": view, "layer": layer, "relative_l2": error})
    dump(output, {"teacher_images": 4, "rows": rows, "maximum_relative_l2": max(r["relative_l2"] for r in rows)})


def summarize(job):
    rows = []
    for case in plan():
        folder = job / "train" / case["name"]
        if not (folder / "DONE").exists():
            continue
        summary = read(folder / "summary.json")
        rows.append(
            {
                "method": case["method"],
                "seed": case["seed"],
                **{key: summary["official_final"][key] for key in ("AP", "AP50", "AP75", "AR500")},
                "query_images": summary["query_images"],
                "train_seconds": summary["train_seconds"],
                "total_charged_seconds": summary["total_charged_seconds"],
                "cache_GiB": summary["cache_bytes"] / 2**30,
            }
        )
    if not rows:
        return
    with (job / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    text = "Fixed final official-reference AP, 0-100. Each row is one completed full run.\n"
    text += "Seed0 mix25/online100 are cost repeats; do not count as new independent seeds.\n"
    text += "method seed AP AP50 AP75 AR500 queries train_s total_charged_s cache_GiB\n"
    for row in rows:
        text += " ".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in row.values()) + "\n"
    (job / "comparison.txt").write_text(text, encoding="utf-8")


def run(a):
    from online_experiments.relocate_feature_assets import relocate_assets

    job = a.job.resolve()
    job.mkdir(parents=True, exist_ok=True)
    queue = plan()
    if (job / "queue.json").exists() and read(job / "queue.json") != queue:
        raise RuntimeError("Existing queue differs from frozen plan")
    dump(job / "queue.json", queue)
    try:
        write_status(job, "RUNNING asset verification and relocation")
        assets = relocate_assets(
            ROOT / "runs/paper/full-visdrone/job-start-eDmH0odX",
            ROOT / "runs/paper/detail-probe/job-20261022-hjP4y3Lf/cache",
            asset_destination(job),
        )
        asset_check(job, assets)
        source, cache = Path(assets["source_job"]), Path(assets["cache_manifest"])
        common = [
            sys.executable,
            "-u",
            str(ROOT / "online_experiments/replication_suite.py"),
            "--job",
            str(job),
            "--source",
            str(source),
            "--cache",
            str(cache),
        ]
        subprocess.run(common[:3] + ["snapshot"] + common[3:], cwd=ROOT, check=True)
        if not (job / "parity.json").exists():
            write_status(job, "RUNNING restored teacher/cache parity")
            command = [
                sys.executable,
                "-u",
                __file__,
                "parity",
                "--job",
                str(job),
                "--source",
                str(source),
                "--cache",
                str(cache),
            ]
            with (job / "parity.log").open("a") as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        smoke = ("frofa-c2-d01", "loffta-c2-n005", "mix25", "online100", "blank0", "detail0", "detail25", "views25")
        for method in smoke:
            write_status(job, f"RUNNING smoke {method}")
            with (job / f"smoke-{method}.log").open("a") as log:
                subprocess.run(
                    common + ["--method", method, "--seed", "1", "--epochs", "1", "--smoke"],
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        dump(job / "smoke-pairing.json", check_pairing([job / "smoke" / f"{method}-s1" for method in smoke]))
        if not (job / "archive-smoke.json").exists():
            paths = [
                job / "queue.json",
                job / "asset-check.json",
                job / "parity.json",
                job / "provenance.json",
                job / "smoke-pairing.json",
            ]
            paths += [
                p
                for method in smoke
                for p in (job / "smoke" / f"{method}-s1").glob("*.json")
                if p.name in ("args.json", "summary.json", "initialization.json", "environment.json")
            ]
            archive(job, "smoke", paths, a.push)
        for number, case in enumerate(queue, 1):
            write_status(job, f"RUNNING full {number}/{len(queue)} {case['name']}")
            with (job / f"train-{case['name']}.log").open("a") as log:
                subprocess.run(
                    common + ["--method", case["method"], "--seed", str(case["seed"])],
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            summarize(job)
            completed = [
                job / "train" / item["name"] for item in queue if (job / "train" / item["name"] / "DONE").exists()
            ]
            dump(job / "full-pairing.json", check_pairing(completed))
            if not (job / f"archive-{case['name']}.json").exists():
                folder = job / "train" / case["name"]
                paths = [
                    folder / name
                    for name in (
                        "args.json",
                        "summary.json",
                        "initialization.json",
                        "environment.json",
                        "metrics.json",
                        "official-final.json",
                    )
                ]
                archive(job, case["name"], paths, a.push)
        archive(
            job,
            "complete",
            [job / "queue.json", job / "comparison.csv", job / "comparison.txt", job / "full-pairing.json"],
            a.push,
        )
        complete = read(job / "archive-complete.json")
        write_status(job, "SUCCEEDED" if complete["pushed"] or not a.push else "COMPLETED; PUSH PENDING")
    except Exception:
        write_status(job, "FAILED; inspect job.log and stage logs")
        (job / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        stage = f"failure-{time.time_ns()}"
        evidence = job / f"{stage}.json"
        dump(evidence, {"traceback": traceback.format_exc(), "stage": stage})
        try:
            archive(job, stage, [evidence, job / "queue.json"], a.push)
        except (RuntimeError, OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
            print(f"FAILURE ARCHIVE PENDING: {error}", flush=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("worker", "parity", "publish"))
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    if args.action == "parity":
        parity(args.source, args.cache, args.job / "parity.json")
    elif args.action == "publish":
        publish_pending(args.job)
    else:
        run(args)


if __name__ == "__main__":
    main()
