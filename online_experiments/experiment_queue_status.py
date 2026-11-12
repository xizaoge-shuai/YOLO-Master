"""Read saved queue states and completed endpoints without starting any work."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def show_job(label, job, phase=None):
    folder = job / phase if phase else job
    state_path = folder / "pipeline-status.json"
    if not state_path.exists():
        state_path = folder / "status.json"
    state = read(state_path) if state_path.exists() else {}
    status_text = folder / "status.txt"
    text = status_text.read_text(encoding="utf-8").strip() if status_text.exists() else "NOT_STARTED"
    print(f"\n{label}: {state.get('state', text)}")
    print(f"  JOB={job}")
    if state.get("message"):
        print(f"  {state['message']}")
    if "completed" in state:
        print(f"  queue completed={state['completed']}/{state.get('planned', '?')}")
    done = sorted(
        path.parent.name
        for path in job.glob("train/*/DONE")
        if phase != "seed-repeats" or not path.parent.name.endswith("-s0")
    )
    if done:
        print(f"  DONE({len(done)}): {', '.join(done)}")
    method = state.get("method") or state.get("case")
    candidates = (
        [job / "train" / (method if method.endswith(("-s0", "-s1", "-s2")) else method + "-s0")] if method else []
    )
    for output in candidates:
        summary = output / "summary.json"
        if summary.exists():
            value = read(summary)
            epochs = value.get("epochs", [])
            count = value.get("completed_epochs", len(epochs) if isinstance(epochs, list) else None)
            print(f"  current={output.name}, completed_epochs={count}")


def main():
    print("Saved experiment states; DONE and validated endpoint are completion evidence.")
    cross = ROOT / "runs/paper/crossdataset"
    for pointer in sorted(cross.glob("latest-*-job.txt")):
        show_job(pointer.name, Path(pointer.read_text(encoding="utf-8").strip()))
    paper = ROOT / "runs/paper/paper-baselines/latest-job.txt"
    if paper.exists():
        job = Path(paper.read_text(encoding="utf-8").strip())
        show_job("VisDrone seed0", job, "dataset-first")
        show_job("VisDrone18 seed repeats", job, "seed-repeats")
    for group in ("expansion", "scalable-detection", "transformed-canvas-ap", "controlled-cost"):
        base = ROOT / "runs/paper" / group
        for job in sorted(base.glob("job-*")):
            if job.is_dir():
                show_job(group, job)
    assets = ROOT.parent / "datasets-extension-20261110"
    for name in ("second-backbone-assets", "coco2017"):
        path = assets / name / "status.json"
        print(f"\nASSETS {name}: {read(path) if path.exists() else 'NOT_STARTED'}")


if __name__ == "__main__":
    main()
