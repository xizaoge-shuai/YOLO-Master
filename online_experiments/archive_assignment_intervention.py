"""Watch an existing diagnostic; verify and commit only its lightweight completed results."""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import time
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_stage(stage, count):
    """Fail closed on incomplete, inconsistent or changed results before committing."""
    if (stage / "status.txt").read_text().strip() != "SUCCEEDED":
        raise ValueError("Stage is not complete")
    summary = json.loads((stage / "summary.json").read_text())
    if (summary["images"], summary["teacher_images"], summary["rows"]) != (count, count * 6, count * 192):
        raise ValueError("Diagnostic count mismatch")
    if summary["training_updates"] != 0 or summary["states_unchanged"] is not True:
        raise ValueError("Frozen-state invariant failed")
    for field, limit in (("native_loss_parity_max_absolute", 2e-5), ("parity_max_relative_l2", 0.01)):
        value = summary[field]
        if not math.isfinite(value) or not 0 <= value <= limit:
            raise ValueError("Parity invariant failed")
    required = {"per-image.csv", "assignments.csv", "summary.csv", "assignment-summary.csv", "selection.json"}
    if not required.issubset(summary["outputs_sha256"]):
        raise ValueError("Missing required output hashes")
    for name, expected in summary["outputs_sha256"].items():
        path = (stage / name).resolve()
        if path.parent != stage.resolve() or sha(path) != expected:
            raise ValueError(f"Output hash mismatch: {name}")
    return summary


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def write_preserving(path, text):
    encoded = text.encode()
    if path.exists() and path.read_bytes() != encoded:
        raise ValueError(f"Refusing to overwrite changed generated archive: {path}")
    path.write_bytes(encoded)


def commit_files(root, artifact_paths, progress_block, marker, readme_path, message):
    """Commit one idempotent result, refusing unrelated staged or progress changes."""
    progress_path = "research/PROGRESS.md"
    paths = [*artifact_paths, progress_path]

    def blob(revision, path):
        return subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=root)

    head_progress = blob("HEAD", progress_path)
    if marker.encode() in head_progress:
        for path in artifact_paths:
            if blob("HEAD", path) != (root / path).read_bytes():
                raise RuntimeError("Previously committed archive differs; refusing duplicate result")
        return git(root, "log", "-1", "--format=%H", "--", readme_path)
    expected_progress = head_progress + progress_block.encode()
    current_progress = (root / progress_path).read_bytes()
    if current_progress not in (head_progress, expected_progress):
        raise RuntimeError("Unrelated progress changes exist; preserve them and defer this commit")
    staged = set(git(root, "diff", "--cached", "--name-only").splitlines())
    if not staged.issubset(paths):
        raise RuntimeError("Unrelated staged files exist; result commit deferred")
    for path in staged:
        expected = expected_progress if path == progress_path else (root / path).read_bytes()
        if blob("", path) != expected:
            raise RuntimeError("Unrelated staged content in result paths; commit deferred")
    (root / progress_path).write_bytes(expected_progress)
    # Enumerated result CSVs are intentionally versioned despite the global CSV ignore.
    subprocess.run(["git", "add", "-f", "--", *paths], cwd=root, check=True)
    subprocess.run(["git", "diff", "--cached", "--check"], cwd=root, check=True)
    subprocess.run(["git", "commit", "--only", "-m", message, "--", *paths], cwd=root, check=True)
    return git(root, "rev-parse", "HEAD")


def commit_stage(root, job, stage_name, count):
    stage = job / stage_name
    summary = verify_stage(stage, count)
    archive = root / "research/experiments" / ("assignment-intervention-" + job.name) / stage_name
    archive.mkdir(parents=True, exist_ok=True)
    names = [
        "summary.json",
        "summary.csv",
        "assignment-summary.csv",
        "comparison.txt",
        "provenance.json",
        "cache-slices.json",
    ]
    for name in names:
        target = archive / name
        if target.exists() and sha(target) != sha(stage / name):
            raise ValueError("Refusing to overwrite a different archived result")
        shutil.copyfile(stage / name, target)
    selection = json.loads((stage / "selection.json").read_text())
    compact = {
        "split": selection["split"],
        "seed": selection["seed"],
        "indices": selection["indices"],
        "sample_ids": [x["sample_id"] for x in selection["samples"]],
        "full_selection_sha256": sha(stage / "selection.json"),
    }
    write_preserving(archive / "selection-ids.json", json.dumps(compact, indent=2) + "\n")
    readme = (
        f"# Assignment intervention: {stage_name}\n\n"
        f"Source: `{stage}`. Images: {count}; teacher images: {summary['teacher_images']}; rows: {summary['rows']}.\n\n"
        "Batch-1 frozen epoch-100 seed-0 Online100 head; Transport/Rec; six views; one2many/one2one; "
        "native_soft/hard_common; EE/EA/AE/AA cells (feature donor, assignment donor). "
        "Full per-image CSVs stay in the run directory and their hashes are in summary.json. "
        "No weights, feature caches, optimizer updates or AP evaluation in this archive. "
        "Shared GPU elapsed time is not an efficiency comparison.\n\n"
        "Smoke is only an execution check. Development uses training images. Confirmation reuses the existing "
        "548-image validation split and is not an untouched test. Cross-view summaries average each image first. "
        "Gradient metrics refer to raw-output coordinates, not feature or parameter gradients. "
        "Exact-assignment interventions use oracle information; improvements do not establish AP recovery.\n"
    )
    write_preserving(archive / "README.md", readme)
    link = str(archive.relative_to(root / "research")) + "/README.md"
    marker = f"<!-- assignment-intervention:{job.name}:{stage_name} -->"
    block = (
        f"\n{marker}\n## Assignment intervention {job.name}: {stage_name} complete\n\n"
        f"- [Verified result archive]({link}); {count} images, {summary['teacher_images']} teacher images, {summary['rows']} cells.\n"
        f"- Native loss parity max absolute difference {summary['native_loss_parity_max_absolute']:.8g}; identity relative L2 {summary['parity_max_relative_l2']:.8g}; model states unchanged; zero updates. No AP claim.\n"
    )
    paths = [str((archive / name).relative_to(root)) for name in [*names, "selection-ids.json", "README.md"]]
    return commit_files(
        root,
        paths,
        block,
        marker,
        str((archive / "README.md").relative_to(root)),
        f"Record verified assignment intervention {stage_name}: {job.name}",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    job = args.job.resolve()
    base = (root / "runs/paper/assignment-intervention").resolve()
    if job.parent != base or not (job / "pid.txt").is_file():
        raise ValueError("Expected an existing assignment-intervention worker job")
    import fcntl

    with (base / "archive.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = job / "archive-status.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {"commits": {}}
        try:
            while True:
                for stage, count in (("smoke", 2), ("development", 128), ("confirmation", 548)):
                    status_path = job / stage / "status.txt"
                    if stage in state["commits"] or not status_path.exists():
                        continue
                    if status_path.read_text().strip() == "SUCCEEDED":
                        state["commits"][stage] = commit_stage(root, job, stage, count)
                        atomic_json(state_path, state)
                        print(stage, state["commits"][stage], flush=True)
                worker_status = (job / "status.txt").read_text().strip()
                if worker_status == "SUCCEEDED" or worker_status.startswith("FAILED") or args.once:
                    state["status"] = "ARCHIVED" if len(state["commits"]) == 3 else "PARTIAL: " + worker_status
                    break
                pid = (job / "pid.txt").read_text().strip()
                cmdline = Path("/proc") / pid / "cmdline"
                if not pid.isdecimal() or not cmdline.exists() or str(job).encode() not in cmdline.read_bytes():
                    state["status"] = "WORKER_NOT_RUNNING: inspect original job logs; completed stages preserved"
                    break
                time.sleep(30)
        except Exception as error:
            state["status"] = "ARCHIVE_FAILED: " + repr(error)
            raise
        finally:
            atomic_json(state_path, state)
            print(state, flush=True)


if __name__ == "__main__":
    main()
