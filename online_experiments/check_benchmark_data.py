#!/usr/bin/env python3
"""Read-only inventory; image lists alone do not prove a dataset is installed."""

import argparse
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), "root": str(ROOT), "datasets": {}}
    vis = ROOT / "datasets/VisDrone"
    splits = {}
    for split, expected in (("train", 6471), ("val", 548), ("test", 1610)):
        images = {x.stem for x in (vis / "images" / split).glob("*.jpg")}
        labels = {x.stem for x in (vis / "labels" / split).glob("*.txt")}
        errors = []
        for path in (vis / "labels" / split).glob("*.txt"):
            for line_no, line in enumerate(path.read_text().splitlines(), 1):
                try:
                    fields = [float(x) for x in line.split()]
                    good = (
                        len(fields) == 5
                        and fields[0].is_integer()
                        and 0 <= fields[0] < 10
                        and all(0 <= x <= 1 for x in fields[1:])
                        and fields[3] > 0
                        and fields[4] > 0
                    )
                except ValueError:
                    good = False
                if not good:
                    errors.append(f"{path.name}:{line_no}")
        splits[split] = {
            "expected_images": expected,
            "images": len(images),
            "labels": len(labels),
            "missing_labels": sorted(images - labels),
            "orphan_labels": sorted(labels - images),
            "invalid_label_rows_count": len(errors),
            "invalid_label_rows_first20": errors[:20],
            "count_and_label_checks_pass": len(images) == expected and images == labels and not errors,
        }
    report["datasets"]["VisDrone"] = {"splits": splits, "original_annotation_archives": {}}
    for path in vis.glob("*.zip"):
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            count = sum("/annotations/" in x and x.endswith(".txt") for x in names)
        report["datasets"]["VisDrone"]["original_annotation_archives"][path.name] = count
    coco = ROOT / "datasets/coco"
    report["datasets"]["COCO"] = {
        "train2017_images": len(list((coco / "images/train2017").glob("*.jpg"))),
        "val2017_images": len(list((coco / "images/val2017").glob("*.jpg"))),
        "train_list_entries": len((coco / "train2017.txt").read_text().splitlines()),
        "val_list_entries": len((coco / "val2017.txt").read_text().splitlines()),
        "instances_train2017_json": (coco / "annotations/instances_train2017.json").exists(),
        "instances_val2017_json": (coco / "annotations/instances_val2017.json").exists(),
    }
    free = shutil.disk_usage(ROOT).free
    bytes_per_image = 3 * 384 * 40 * 40 * 2
    report["capacity"] = {
        "free_disk_GiB": free / 2**30,
        "feature_spec": "3 x [384,40,40] FP16; serialization overhead excluded",
        "visdrone_train_C2_GiB_estimate": 6471 * 2 * bytes_per_image / 2**30,
        "visdrone_val_C1_GiB_estimate": 548 * bytes_per_image / 2**30,
        "coco_train_C1_GiB_estimate": 118287 * bytes_per_image / 2**30,
        "full_training_requires_streaming_cache": True,
    }
    report["limits"] = [
        "File/count/label checks only; not a full image-decoding or official-evaluation audit.",
        "Current pilot split/evaluator remain unchanged and are not full-dataset results.",
        "Preserve raw VisDrone ignored-region annotations for official evaluation.",
        "COCO train list does not imply that train images or annotations are installed.",
    ]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
