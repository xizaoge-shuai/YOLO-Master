"""Prioritize full paper baselines and retain the original unfinished experiments."""

import argparse
import csv
import fcntl
import os
import signal
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from online_experiments.detail_probe import dump, read, sha
from online_experiments.replication_campaign import archive, check_pairing, write_status


def plan():
    papers = ("frofa-c2-d01", "loffta-c2-n005", "cache-reuse-c1-int8", "vit-adapter-cache25")
    cases = [(method, 0) for method in papers]
    cases += [
        (method, 0)
        for method in ("mix25", "online100", "blank0", "detail0", "vit-adapter-cache0", "cache-reuse-c1-int4")
    ]
    cases += [(method, seed) for seed in (1, 2) for method in papers]
    cases += [
        (method, seed) for seed in (1, 2) for method in ("blank25", "detail25", "lowpass25", "transport25", "views25")
    ]
    return [
        {
            "method": method,
            "seed": seed,
            "name": f"{method}-s{seed}",
            "resume_original": method == "blank25" and seed == 1,
        }
        for method, seed in cases
    ]


def processes(script, job):
    found = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            args = (path / "cmdline").read_bytes().decode().split("\0")
            if path.stat().st_uid == os.getuid() and str(job) in args and any(Path(arg).name == script for arg in args):
                found.append(int(path.name))
        except (OSError, UnicodeError):
            continue
    return found


def finish_process(pid):
    try:
        os.kill(pid, signal.SIGTERM)
        os.kill(pid, signal.SIGCONT)
    except ProcessLookupError:
        return
    for _ in range(100):
        path = Path(f"/proc/{pid}/stat")
        try:
            if path.read_text().split(")", 1)[1].strip().startswith("Z"):
                return
        except FileNotFoundError:
            return
        time.sleep(0.1)
    raise RuntimeError(f"Own experiment process did not terminate: {pid}")


def switch_original(job, original):
    """Retain the next complete epoch, stop only identified own job processes."""
    parent = processes("replication_campaign.py", original)
    children = processes("replication_suite.py", original)
    shells = processes("run_replication_suite.sh", original)
    destination = job / "original-switch.json"
    if destination.exists() and read(destination).get("phase") == "checkpoint_saved":
        record = read(destination)
        if sha(original / "train/blank25-s1/weights/last.pt") != record["checkpoint_sha256"]:
            raise RuntimeError("Original checkpoint changed after interrupted priority switch")
        actual = set(parent + children + shells)
        if not actual.issubset(set(record["process_ids"])):
            raise RuntimeError("Unexpected original processes after interrupted switch")
        for pid in children + parent + shells:
            finish_process(pid)
        record["phase"] = "stopped_all"
        dump(destination, record)
        dump(original / "superseded-by-paper-baselines.json", {**record, "new_job": str(job)})
        (original / "status.txt").write_text("SUPERSEDED by full paper-baseline priority queue; checkpoint preserved\n")
        return record
    if not parent and not children and not shells:
        if destination.exists():
            return read(destination)
        raise RuntimeError("Original job is not active and no recorded priority switch exists")
    if len(parent) != 1 or len(children) != 1 or len(shells) != 1:
        raise RuntimeError("Ambiguous original process tree; refusing signal delivery")
    train = children[0]
    folder = original / "train/blank25-s1"
    previous = read(folder / "summary.json")["completed_epochs"]
    started, stopped = time.time(), False
    os.kill(parent[0], signal.SIGSTOP)
    try:
        write_status(job, f"WAITING next original epoch checkpoint after {previous}; baseline priority")
        while True:
            summary = read(folder / "summary.json")
            if summary["completed_epochs"] > previous and not (folder / "DONE").exists():
                rows = read(folder / "metrics.json")
                if rows and rows[-1]["epoch"] == summary["completed_epochs"]:
                    break
            if time.time() - started > 1200 or processes("replication_suite.py", original) != [train]:
                raise RuntimeError("No next complete checkpoint within 20 minutes, or original child changed")
            time.sleep(0.25)
        os.kill(train, signal.SIGSTOP)
        stopped = True
        for _ in range(200):
            status = Path(f"/proc/{train}/status").read_text()
            if any(line.startswith("State:") and "T" in line.split()[1] for line in status.splitlines()):
                break
            time.sleep(0.01)
        else:
            raise RuntimeError("Original training child did not acknowledge SIGSTOP")
        import torch

        checkpoint = folder / "weights/last.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        for key in ("model", "optimizer", "scheduler", "cpu_rng", "cuda_rng", "totals", "args", "rows", "summary"):
            if key not in state:
                raise RuntimeError(f"Original checkpoint lacks resume state: {key}")
        epoch = state["epoch"]
        if (
            (state["args"]["method"], state["args"]["seed"], state["args"]["epochs"], state["args"]["train_images"])
            != ("blank25", 1, 100, 6471)
            or len(state["rows"]) != epoch
            or state["summary"]["completed_epochs"] != epoch
        ):
            raise RuntimeError("Original full-data checkpoint identity or epoch coverage mismatch")
        if state["query_images"] != epoch * 6471 * 25 // 100:
            raise RuntimeError("Original cumulative query budget is inconsistent")
        record = {
            "phase": "checkpoint_saved",
            "process_ids": children + parent + shells,
            "original_job": str(original),
            "case": "blank25-s1",
            "preserved_epoch": epoch,
            "checkpoint_sha256": sha(checkpoint),
            "query_images": state["query_images"],
            "checkpoint_train_seconds": state["totals"]["train"],
            "time_unix": time.time(),
            "switch_wait_seconds": time.time() - started,
            "tail_elapsed_since_checkpoint_seconds": max(0.0, time.time() - checkpoint.stat().st_mtime),
            "tail_note": "Possible uncheckpointed tail work retained as elapsed overhead; no zero-cost claim",
            "reason": "User prioritizes four paper-idea baselines, direct full training, no smoke/pilot",
        }
        dump(destination, record)
        finish_process(train)
        finish_process(parent[0])
        finish_process(shells[0])
        record["phase"] = "stopped_all"
        dump(destination, record)
        dump(original / "superseded-by-paper-baselines.json", {**record, "new_job": str(job)})
        (original / "status.txt").write_text("SUPERSEDED by full paper-baseline priority queue; checkpoint preserved\n")
        return record
    except Exception:
        if stopped:
            try:
                os.kill(train, signal.SIGCONT)
            except ProcessLookupError:
                pass
        try:
            os.kill(parent[0], signal.SIGCONT)
        except ProcessLookupError:
            pass
        raise


def output_for(job, original, case):
    return (original if case["resume_original"] else job) / "train" / case["name"]


def table3_and_resources(job, original):
    """Publish observed complete endpoints only; retain explicit missing rows."""
    rows, resources = [], []
    for case in plan():
        output = output_for(job, original, case)
        done = (output / "DONE").exists()
        summary = read(output / "summary.json") if done else {}
        init = read(output / "initialization.json") if done else {}
        status = "complete" if done else "failed" if any(job.glob(f"failed-{case['name']}-*.json")) else "pending"
        rows.append(
            {
                "method": case["method"],
                "seed": case["seed"],
                "status": status,
                **{key: summary.get("official_final", {}).get(key, "") for key in ("AP", "AP50", "AP75", "AR500")},
                "query_pct": summary.get("query_pct", ""),
                "source": str(output),
            }
        )
        if done:
            resources.append(
                {
                    "method": case["method"],
                    "seed": case["seed"],
                    "cache_build_s": summary["cache_build_charged_seconds"],
                    "train_s": summary["train_seconds"],
                    "warm_wall_s": summary["wall_seconds"],
                    "cold_charged_s": summary["total_charged_seconds"],
                    "vram_GiB": summary["peak_vram_gib"],
                    "cache_GiB": summary["cache_bytes"] / 2**30,
                    "parameters": init["parameters"],
                    "extra_parameters": init["detail_parameters"],
                    "query_images": summary["query_images"],
                    "dequant_s": summary.get("dequant_seconds", 0.0),
                }
            )
    tables = job / "tables"
    tables.mkdir(exist_ok=True)
    for name, values in (
        ("all-runs.csv", rows),
        ("table5-resources.csv", resources),
        (
            "table3-related-methods.csv",
            [row for row in rows if row["method"].startswith(("frofa", "loffta", "cache-reuse", "vit-adapter"))],
        ),
        (
            "table4-spatial-ablations.csv",
            [
                row
                for row in rows
                if row["method"] in ("blank25", "detail25", "lowpass25", "transport25", "views25", "blank0", "detail0")
            ],
        ),
    ):
        if not values:
            continue
        with (tables / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    ledger = (ROOT / "research/RESULTS_LEDGER.md").read_text()

    def section_rows(number):
        section = ledger.split(f"## {number}.", 1)[1].split("\n## ", 1)[0]
        return [
            [item.strip() for item in line.strip().strip("|").split("|")]
            for line in section.splitlines()
            if line.startswith("| ") and not line.startswith("| ---")
        ]

    detectors = [
        {
            "method": fields[0],
            "seed": 0,
            "AP": fields[1],
            "AP50": fields[2],
            "AP75": fields[3],
            "AR500": "",
            "status": "historical saved reference",
            "source": "research/RESULTS_LEDGER.md section5",
            "note": fields[4],
        }
        for fields in section_rows(5)
        if fields[0].startswith("YOLO")
    ]
    detectors.append(
        {
            "method": "YOLO13",
            "seed": "",
            "AP": "",
            "AP50": "",
            "AP75": "",
            "AR500": "",
            "status": "not implemented/not run",
            "source": "",
            "note": "No proxy architecture substituted",
        }
    )
    historical_spatial = section_rows(4)
    for fields in historical_spatial:
        if fields[0] == "Detail25":
            detectors.append(
                {
                    "method": "Detail25 candidate",
                    "seed": 0,
                    "AP": fields[2],
                    "AP50": fields[3],
                    "AP75": fields[4],
                    "AR500": "",
                    "status": "historical saved reference",
                    "source": "research/RESULTS_LEDGER.md section4",
                    "note": "Candidate, not finalized method",
                }
            )
    for row in rows:
        if row["method"] == "detail25" and row["status"] == "complete":
            detectors.append(
                {key: row[key] for key in ("method", "seed", "AP", "AP50", "AP75", "AR500", "status", "source")}
                | {"note": "new full-data repeat"}
            )
    budget = []
    for fields in section_rows(3):
        if fields[0] in ("C1", "C2 / q0", "Random Mix10", "Random Mix25", "Random Mix50", "Online100"):
            for seed, value in enumerate(fields[1:4]):
                if value != "—":
                    budget.append(
                        {
                            "method": fields[0],
                            "seed": seed,
                            "AP": value,
                            "status": "historical saved reference",
                            "source": "research/RESULTS_LEDGER.md section3",
                            "note": "Do not pool historical seed0 with its new cost repeat as independent seeds",
                        }
                    )
    budget += [
        {
            "method": row["method"],
            "seed": row["seed"],
            "AP": row["AP"],
            "status": row["status"],
            "source": row["source"],
            "note": "new seed0 cost repeat",
        }
        for row in rows
        if row["method"] in ("mix25", "online100")
    ]
    for name, values in (("table1-detectors.csv", detectors), ("table2-cache-budget.csv", budget)):
        with (tables / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    # Preserve the existing five-arm seed0 evidence separately from new repeats.
    dump(
        tables / "historical-spatial.json",
        {
            "source": "research/RESULTS_LEDGER.md section4",
            "rows": [
                {
                    "method": fields[0],
                    "seed": 0,
                    "AP": fields[2],
                    "AP50": fields[3],
                    "AP75": fields[4],
                    "observed_train_seconds": fields[5],
                }
                for fields in historical_spatial
                if fields[0] in ("Transport25", "Blank25", "Views25", "Detail25", "Lowpass25")
            ],
        },
    )
    spatial = [
        {
            "method": fields[0].lower(),
            "seed": 0,
            "status": "historical saved reference",
            "AP": fields[2],
            "AP50": fields[3],
            "AP75": fields[4],
            "AR500": "",
            "query_pct": 25,
            "source": "research/RESULTS_LEDGER.md section4",
        }
        for fields in historical_spatial
        if fields[0] in ("Transport25", "Blank25", "Views25", "Detail25", "Lowpass25")
    ]
    spatial += [
        row
        for row in rows
        if row["method"] in ("blank25", "detail25", "lowpass25", "transport25", "views25", "blank0", "detail0")
    ]
    with (tables / "table4-spatial-ablations.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(spatial[0]))
        writer.writeheader()
        writer.writerows(spatial)
    if not resources:
        (tables / "table5-resources.csv").write_text(
            "method,seed,cache_build_s,train_s,warm_wall_s,cold_charged_s,vram_GiB,cache_GiB,parameters,extra_parameters,query_images,dequant_s\n"
        )
    message = "Fixed epoch100 reference AP, 0-100. Missing rows are not results. Shared GPU timings.\n"
    message += "method seed status AP AP50 AP75 AR500\n"
    for row in rows:
        if row["status"] == "complete":
            message += (
                " ".join(str(row[key]) for key in ("method", "seed", "status", "AP", "AP50", "AP75", "AR500")) + "\n"
            )
    (job / "comparison.txt").write_text(message)


def run(args):
    job, original = args.job.resolve(), args.original_job.resolve()
    job.mkdir(parents=True, exist_ok=True)
    queue = plan()
    if (job / "queue.json").exists() and read(job / "queue.json") != queue:
        raise RuntimeError("Frozen paper-baseline queue changed")
    dump(job / "queue.json", queue)
    source, cache = args.assets / "source", args.assets / "cache/manifest.json"
    common = [sys.executable, "-u", str(ROOT / "online_experiments/paper_baseline_suite.py")]
    shared = ["--job", str(job), "--source", str(source), "--cache", str(cache)]
    subprocess.run(common + ["snapshot"] + shared, cwd=ROOT, check=True)
    if not (job / "original-switch.json").exists() or read(job / "original-switch.json").get("phase") != "stopped_all":
        switch_original(job, original)
    # Hold the old launcher's lock so an accidental manual resume cannot start a
    # second supervisor. Resuming its preserved case directly does not need it.
    lock = (original.parent / "campaign.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not (job / "archive-priority-switch.json").exists():
        archive(
            job, "priority-switch", [job / "queue.json", job / "original-switch.json", job / "provenance.json"], True
        )
    table3_and_resources(job, original)
    failed = []
    for number, case in enumerate(queue, 1):
        write_status(job, f"RUNNING full {number}/{len(queue)} {case['name']}; no smoke/pilot")
        try:
            if case["resume_original"]:
                origin_root = args.origin_root.resolve()
                command = [
                    sys.executable,
                    "-u",
                    str(origin_root / "online_experiments/replication_suite.py"),
                    "--job",
                    str(original),
                    "--source",
                    str(source),
                    "--cache",
                    str(cache),
                    "--method",
                    case["method"],
                    "--seed",
                    str(case["seed"]),
                ]
                cwd = origin_root
            else:
                command = common + ["train"] + shared + ["--method", case["method"], "--seed", str(case["seed"])]
                cwd = ROOT
                if case["method"].startswith("cache-reuse-"):
                    from online_experiments.paper_quantized_cache import build_quantized_cache

                    bits = int(case["method"].rsplit("int", 1)[1])
                    location = job / f"quantized-int{bits}"
                    if location.exists() and any(location.iterdir()) and not (location / "complete.json").exists():
                        location = job / f"quantized-int{bits}-retry-{uuid.uuid4().hex[:8]}"
                    reference = job / f"quantized-int{bits}-location.json"
                    if reference.exists():
                        known = Path(read(reference)["directory"])
                        if (known / "complete.json").exists():
                            location = known
                    dump(reference, {"directory": str(location)})
                    quantized = build_quantized_cache(cache, location, bits=bits)
                    command += ["--quantized", str(quantized)]
            with (job / f"train-{case['name']}.log").open("a") as log:
                subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=True)
            output = output_for(job, original, case)
            table3_and_resources(job, original)
            current = [
                output_for(job, original, other)
                for other in queue
                if (output_for(job, original, other) / "DONE").exists() and not other["resume_original"]
            ]
            dump(job / "pairing.json", check_pairing(current))
            if not (job / f"archive-{case['name']}.json").exists():
                # Original resumed output lies outside new job; copy metadata only
                # to a source-labeled record, never its weights or predictions.
                if case["resume_original"]:
                    target = job / "resumed-original" / case["name"]
                    target.mkdir(parents=True, exist_ok=True)
                    for name in (
                        "args.json",
                        "summary.json",
                        "initialization.json",
                        "environment.json",
                        "metrics.json",
                        "official-final.json",
                    ):
                        (target / name).write_bytes((output / name).read_bytes())
                    output = target
                paths = [
                    output / name
                    for name in (
                        "args.json",
                        "summary.json",
                        "initialization.json",
                        "environment.json",
                        "metrics.json",
                        "official-final.json",
                    )
                ]
                archive(job, case["name"], paths, True)
        except (RuntimeError, OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
            failed.append(case["name"])
            record = job / f"failed-{case['name']}-{time.time_ns()}.json"
            dump(record, {"case": case, "traceback": traceback.format_exc(), "scope": "full-data attempt, not a pilot"})
            try:
                archive(job, record.stem, [record], True)
            except (RuntimeError, OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
                print(f"Failure archive pending: {error}", flush=True)
            print(traceback.format_exc(), flush=True)
    table3_and_resources(job, original)
    dump(job / "completion.json", {"failed": failed, "planned": len(queue), "training_complete": not failed})
    paths = [
        job / "queue.json",
        job / "completion.json",
        job / "comparison.txt",
        *sorted((job / "tables").glob("*.csv")),
    ]
    archive(job, "complete", paths, True)
    if failed:
        write_status(job, "FINISHED WITH FAILURES: " + ", ".join(failed))
    elif not read(job / "archive-complete.json")["pushed"]:
        write_status(job, "COMPLETED; PUSH PENDING")
    else:
        write_status(job, "SUCCEEDED")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--original-job", type=Path, required=True)
    parser.add_argument("--origin-root", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        write_status(args.job, "FAILED orchestration; inspect job.log")
        (args.job / "orchestration-failure.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
