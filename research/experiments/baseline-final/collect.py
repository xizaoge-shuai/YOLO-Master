import csv
import json
import statistics
from pathlib import Path

ROOT = Path.cwd()
DEST = ROOT / "research/experiments/baseline-final"

def latest(name):
    p = ROOT / f"runs/paper/{name}/latest-job.txt"
    return Path(p.read_text().strip())

def load(path):
    return json.loads(path.read_text())

full = latest("full-visdrone")
pareto = latest("full-visdrone-pareto")
confirm = latest("full-visdrone-confirm")

old = load(
    ROOT /
    "research/experiments/"
    "visdrone-reference-job-full-20260927/"
    "original/results.json"
)

sources = [
    ("full", old, full),
    (
        "pareto",
        load(pareto / "reference-eval.json")["results"],
        pareto,
    ),
    (
        "confirm",
        load(confirm / "reference-eval.json")["results"],
        confirm,
    ),
]

fields = [
    "method", "seed", "checkpoint",
    "AP", "AP50", "AP75",
    "AR1", "AR10", "AR100", "AR500",
    "macro_AP_diagnostic",
    "query_images",
    "train_seconds", "val_seconds",
    "wall_seconds",
    "cache_build_charged_seconds",
    "build_plus_wall_seconds",
    "cache_bytes", "peak_vram_gib",
    "correction_seconds",
    "source",
]

all_rows = []
keys = set()

for tag, records, job in sources:
    for r in records:
        method = r["method"]
        seed = int(r.get("seed", 0))
        checkpoint = r["checkpoint"]

        key = (method, seed, checkpoint)

        if key in keys:
            raise RuntimeError(f"Duplicate result: {key}")
        keys.add(key)

        summary_path = (
            job / "train" /
            f"{method}-s{seed}" /
            "summary.json"
        )
        summary = (
            load(summary_path)
            if summary_path.exists()
            else {}
        )

        row = {
            "method": method,
            "seed": seed,
            "checkpoint": checkpoint,
            "source": tag,
        }

        for field in fields:
            if field in row:
                continue

            row[field] = r.get(field, summary.get(field))

        all_rows.append(row)

DEST.mkdir(parents=True, exist_ok=True)

with (DEST / "visdrone-baselines.csv").open(
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(all_rows)

print("===== PINNED VISDRONE REFERENCE =====")
print("AP/AP50/AP75/AR in 0-100 points")
print()
print(
    f"{'method':16s} {'seed':>4s} "
    f"{'AP':>8s} {'AP50':>8s} "
    f"{'AP75':>8s} {'AR500':>8s}"
)

for r in all_rows:
    if r["checkpoint"] != "last":
        continue

    print(
        f"{r['method']:16s} {r['seed']:4d} "
        f"{r['AP']:8.4f} "
        f"{r['AP50']:8.4f} "
        f"{r['AP75']:8.4f} "
        f"{r['AR500']:8.4f}"
    )

print()
print("===== 3-SEED CORE =====")

for method in ("cache2", "mix25", "online-geom"):
    vals = sorted(
        [
            r for r in all_rows
            if r["method"] == method
            and r["checkpoint"] == "last"
        ],
        key=lambda r: r["seed"],
    )

    if [r["seed"] for r in vals] != [0, 1, 2]:
        raise RuntimeError(f"Incomplete seeds: {method}")

    aps = [r["AP"] for r in vals]

    print(
        f"{method:12s} "
        f"{statistics.mean(aps):.4f} ± "
        f"{statistics.stdev(aps):.4f}"
    )

print()
print("RESULT FILE:", DEST / "visdrone-baselines.csv")
print("BASELINE COLLECTION: PASS")
