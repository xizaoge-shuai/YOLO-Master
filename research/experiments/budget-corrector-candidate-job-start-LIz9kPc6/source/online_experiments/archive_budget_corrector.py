#!/usr/bin/env python3
"""Commit lightweight, byte-preserved evidence for one completed/failed campaign."""

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive(job):
    job = job.resolve()
    status = (job / "status.txt").read_text().strip()
    if status.startswith("RUNNING"):
        raise RuntimeError("Cannot archive a running job")
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT, check=False).returncode:
        raise RuntimeError("Git index already contains staged changes; leave them untouched")
    if git("status", "--porcelain", "--", "research/PROGRESS.md"):
        raise RuntimeError("Commit existing progress-file changes before archiving this job")
    name = "budget-corrector-" + job.name
    destination = ROOT / "research/experiments" / name
    relative = destination.relative_to(ROOT).as_posix()
    marker = destination / "archive-manifest.json"
    if marker.exists():
        committed = git("log", "-1", "--format=%H", "--", relative)
        if committed:
            print(f"Already archived: {committed} {relative}")
            return
        raise RuntimeError("Uncommitted archive exists; inspect it rather than overwrite evidence")
    files = []
    for filename in (
        "plan.json",
        "provenance.json",
        "reference_args.json",
        "reference_split.json",
        "comparison.json",
        "comparison.txt",
        "smoke-comparison.json",
        "smoke-comparison.txt",
        "status.txt",
    ):
        if (job / filename).is_file():
            files.append((job / filename, Path(filename)))
    for phase in ("train", "smoke"):
        for folder in sorted((job / phase).glob("*")):
            if not folder.is_dir():
                continue
            for filename in (
                "args.json",
                "split.json",
                "summary.json",
                "trainability.json",
                "parity.json",
                "results.csv",
                "correction.jsonl",
                "DONE",
            ):
                path = folder / filename
                if path.is_file():
                    # Root gitignore excludes results.csv.
                    destname = "metrics.csv" if filename == "results.csv" else filename
                    files.append((path, Path(phase) / folder.name / destname))
    if status.startswith("FAILED"):
        for path in sorted(job.glob("*.log")):
            if path.stat().st_size <= 2 * 1024**2 and path.name != "archive.log":
                files.append((path, Path("failure-logs") / (path.stem + ".txt")))
    manifest = {
        "source_job": str(job),
        "status": status,
        "archived_at": datetime.now(timezone.utc).isoformat(),
        "archival_git_parent": git("rev-parse", "HEAD"),
        "files": {},
        "note": "Raw metrics preserved; pilot only. No weights, datasets or cache tensors.",
    }
    destination.mkdir(parents=True)
    for src, rel in files:
        dst = destination / "original" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        if sha(src) != sha(dst):
            raise RuntimeError(f"Archive copy mismatch: {src}")
        manifest["files"][dst.relative_to(destination).as_posix()] = {
            "source": str(src),
            "sha256": sha(src),
            "bytes": src.stat().st_size,
        }
    # Full schedules stay in the run directory. Hash them to tie query equality to raw records.
    manifest["schedules"] = {}
    for path in sorted(job.glob("*/*/schedule.jsonl")):
        digest = sha(path)
        summary = path.parent / "summary.json"
        if (
            summary.exists()
            and (path.parent / "DONE").exists()
            and json.loads(summary.read_text())["schedule_sha256"] != digest
        ):
            raise RuntimeError(f"Saved schedule hash mismatch: {path}")
        manifest["schedules"][str(path)] = {"sha256": digest, "bytes": path.stat().st_size}
    marker.write_text(json.dumps(manifest, indent=2) + "\n")
    (destination / ".gitattributes").write_text("original/** -text -whitespace\n")
    (destination / "README.md").write_text(
        f"# Fixed-budget correction pilot\n\nStatus: {status}\n\n"
        "400 training / 100 validation images; three training seeds in the full plan.\n"
        "Use original/comparison.txt for paired differences and completed-run counts.\n"
        "C2-Mix, C2-Transport and C2-Correct share query identities and augmentation proposals.\n"
        "The corrector is supervised by queried training features only; validation is uncorrected.\n"
        "Schedule hashes are archived; full schedules remain at the source job path.\n"
        "Cache construction is charged once per run; prior validation-cache build is excluded.\n"
        "Three seeds on this pilot do not establish official benchmark performance.\n"
    )
    progress = ROOT / "research/PROGRESS.md"
    entry = (
        f"\n## Fixed-budget correction: {job.name}\n\n"
        f"- Status: {status}\n- Evidence: [archive](experiments/{name}/README.md)\n"
        f"- Source job: {job}\n"
        "- Report best/final AP, paired differences, actual query images and total charged time.\n"
        "- No full-dataset or independent-test claim; no conclusion inferred from smoke AP.\n"
    )
    with progress.open("a", encoding="utf-8") as stream:
        stream.write(entry)
    subprocess.run(["git", "add", "--", relative, "research/PROGRESS.md"], cwd=ROOT, check=True)
    staged = git("diff", "--cached", "--name-only").splitlines()
    if any(not (p.startswith(relative + "/") or p == "research/PROGRESS.md") for p in staged):
        raise RuntimeError("Unexpected staged path; inspect index")
    subprocess.run(["git", "diff", "--cached", "--check"], cwd=ROOT, check=True)
    subprocess.run(["git", "commit", "-m", f"docs(d1): archive {job.name} experiment progress"], cwd=ROOT, check=True)
    print(f"COMMIT={git('rev-parse', 'HEAD')} ARCHIVE={relative}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job", type=Path, required=True)
    a = p.parse_args()
    archive(a.job)


if __name__ == "__main__":
    main()
