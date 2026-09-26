"""Official split data and batch-streamed caches; historical pilot files stay immutable."""

import hashlib
import json
import math
import random
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def clean_labels(text):
    rows, removed = [], []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        row = [float(x) for x in line.split()]
        if (
            len(row) != 5
            or not all(math.isfinite(x) for x in row)
            or not row[0].is_integer()
            or not 0 <= row[0] < 10
            or not all(0 <= x <= 1 for x in row[1:])
        ):
            raise ValueError(f"Invalid YOLO label at line {number}: {line}")
        if row[3] == 0 or row[4] == 0:
            removed.append(number)
        else:
            rows.append(row)
    return rows, removed


def validate_splits(train, val):
    for key in ("sample_id", "image_sha256"):
        a, b = {x[key] for x in train}, {x[key] for x in val}
        if a & b:
            raise ValueError(f"Train/val overlap in {key}: {sorted(a & b)[:3]}")
    for split in (train, val):
        if len({x["sample_id"] for x in split}) != len(split):
            raise ValueError("Duplicate sample IDs")


def epoch_plan(n, pct, epoch, seed):
    """Exact cumulative image quota, including partial batches; method-independent RNG."""
    if n < 1 or not 0 <= pct <= 100 or epoch < 0:
        raise ValueError("Invalid epoch plan")
    order = list(range(n))
    random.Random(seed + 104729 + epoch * 1000003).shuffle(order)
    rng = random.Random(seed + 32452843 + epoch * 1000003)
    flips, scales = [], []
    for _ in order:
        flips.append(rng.random() < 0.5)
        scales.append(rng.uniform(0.85, 1.15))
    count = ((epoch + 1) * n * pct) // 100 - (epoch * n * pct) // 100
    selected = set(random.Random(seed + 15485863 + epoch * 1000003).sample(range(n), count))
    return order, flips, scales, [i in selected for i in range(n)]


def inverse_letterbox(predictions, info):
    """XYXY letterbox predictions -> original-pixel VisDrone eight-column rows."""
    p = np.asarray(predictions, dtype=np.float64).reshape(-1, 6).copy()
    height, width = info["original_shape"]
    rh, rw = info["resized_shape"]
    left, top = info["pad"]
    p[:, [0, 2]] = np.clip((p[:, [0, 2]] - left) * width / rw, 0, width)
    p[:, [1, 3]] = np.clip((p[:, [1, 3]] - top) * height / rh, 0, height)
    p[:, 2:4] -= p[:, :2]
    p[:, 5] += 1
    return np.column_stack((p, np.full((len(p), 2), -1.0)))


def read_feature_batch(arrays, ids, views):
    """Allocate only the requested FP16 batch; never load the complete bank into GPU RAM."""
    result = np.empty((len(ids), *arrays[0].shape[1:]), dtype=np.float16)
    for j, (i, view) in enumerate(zip(ids, views)):
        result[j] = arrays[view][i]
    return result


def source_snapshot():
    paths = subprocess.check_output(
        ["git", "ls-files", "*.py", "*.sh", "*.yaml", "*.toml"], cwd=ROOT, text=True
    ).splitlines()
    paths += [
        p.relative_to(ROOT).as_posix() for p in (ROOT / "online_experiments").glob("*full_visdrone*") if p.is_file()
    ]
    return {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True),
        "files": {p: sha(ROOT / p) for p in sorted(set(paths)) if (ROOT / p).is_file()},
    }


def verify_sources(snapshot):
    changed = [p for p, h in snapshot["files"].items() if not (ROOT / p).is_file() or sha(ROOT / p) != h]
    if changed:
        raise RuntimeError(f"Source changed since campaign launch: {changed[:10]}")


def prepare(job, smoke=False):
    """Create new derived labels and split manifest, never mutate the source dataset."""
    import torch
    from PIL import Image

    from scripts.d1_train_cached_detector import transform_letterbox_xywhn

    started = time.perf_counter()
    dataset = ROOT / "datasets/VisDrone"
    old = Path((ROOT / "runs/paper/flip-cache/latest-job.txt").read_text().strip())
    identity = read(old / "view-cache/manifest.json")["identity"]
    if identity["imgsz"] != 640 or identity["layers"] != [3, 7, 11]:
        raise ValueError("Expected the verified 640 / layers 3,7,11 teacher identity")
    derived = job / "derived"
    splits, removed = {}, []
    for split, expected, limit in [("train", 6471, 17), ("val", 548, 8)]:
        images = sorted((dataset / "images" / split).glob("*.jpg"))
        labels = {p.stem for p in (dataset / "labels" / split).glob("*.txt")}
        if len(images) != expected or labels != {p.stem for p in images}:
            raise ValueError(f"Incomplete official {split}: {len(images)} images / {len(labels)} labels")
        images = images[:limit] if smoke else images
        (derived / "images" / split).mkdir(parents=True)
        (derived / "labels" / split).mkdir(parents=True)
        entries = []
        for path in images:
            label = dataset / "labels" / split / f"{path.stem}.txt"
            rows, bad = clean_labels(label.read_text())
            if bad:
                removed.append({"path": str(label), "lines": bad, "reason": "zero-area box"})
            with Image.open(path) as im:
                width, height = im.size
            scale = min(640 / height, 640 / width)
            rw, rh = max(1, round(width * scale)), max(1, round(height * scale))
            info = {
                "original_shape": [height, width],
                "resized_shape": [rh, rw],
                "scale": scale,
                "pad": [(640 - rw) // 2, (640 - rh) // 2],
            }
            tensor = torch.tensor(rows, dtype=torch.float32).reshape(-1, 5)
            boxes = transform_letterbox_xywhn(tensor[:, 1:], info, 640)
            (derived / "labels" / split / label.name).write_text(
                "".join(" ".join(f"{x:.8f}" for x in r) + "\n" for r in rows)
            )
            (derived / "images" / split / path.name).symlink_to(path.resolve())
            entries.append(
                {
                    "sample_id": f"{split}/{path.stem}",
                    "image_path": str(path.resolve()),
                    "image_sha256": sha(path),
                    "label_sha256": sha(label),
                    "label_path": str(label),
                    "letterbox": info,
                    "cls": tensor[:, :1].tolist(),
                    "bboxes": boxes.tolist(),
                }
            )
        splits[split] = entries
    validate_splits(splits["train"], splits["val"])
    raw = job / "raw-val-annotations"
    raw.mkdir()
    archive_path = dataset / "VisDrone2019-DET-val.zip"
    with zipfile.ZipFile(archive_path) as archive:
        files = {Path(p).stem: p for p in archive.namelist() if "/annotations/" in p and p.endswith(".txt")}
        for item in splits["val"]:
            name = Path(item["image_path"]).stem
            (raw / f"{name}.txt").write_bytes(archive.read(files[name]))
    # YAML uses absolute paths. It has no download stanza and no test split.
    import yaml

    (job / "dataset.yaml").write_text(
        yaml.safe_dump(
            {"path": str(derived), "train": "images/train", "val": "images/val", "names": dict(enumerate(NAMES))}
        )
    )
    meta = {
        "identity": identity,
        "splits": splits,
        "smoke": smoke,
        "removed_labels": removed,
        "prepare_seconds": time.perf_counter() - started,
        "raw_val_annotation_sha256": {p.name: sha(p) for p in sorted(raw.glob("*.txt"))},
        "test_dev_used": False,
        "protocol": "full-visdrone-v1; diagnostic DetMetrics, not official VisDrone AP",
    }
    dump(job / "dataset-manifest.json", meta)
    print(
        f"DATA train={len(splits['train'])} val={len(splits['val'])} filtered={len(removed)} test_dev_used=false",
        flush=True,
    )
    return meta


def make_extractor(identity):
    from scripts.d1_build_feature_cache import DINOv3MultiLevelExtractor

    return DINOv3MultiLevelExtractor(
        identity["model_id"],
        revision=identity["revision"],
        layers=identity["layers"],
        device="cuda:0",
        dtype="fp32",
        local_files_only=True,
    )


def build(job, batch=8):
    import torch

    from ultralytics.data.foundation_cache import load_letterboxed_tensor

    torch.set_num_threads(4)
    meta = read(job / "dataset-manifest.json")
    ident = meta["identity"]
    root = job / "cache"
    root.mkdir(exist_ok=True)
    estimated = (2 * len(meta["splits"]["train"]) + len(meta["splits"]["val"])) * 3 * 384 * 40 * 40 * 2
    if shutil.disk_usage(root).free < estimated + 30 * 2**30:
        raise RuntimeError(f"Need at least {estimated / 2**30 + 30:.1f} GiB free for cache and checkpoints")
    started = time.perf_counter()
    extractor = make_extractor(ident)
    torch.cuda.synchronize()
    result = {
        "identity": ident,
        "teacher_dtype": "fp32",
        "cache_dtype": "fp16",
        "setup_seconds": time.perf_counter() - started,
        "parts": {},
        "parity": [],
        "dataset_manifest_sha256": sha(job / "dataset-manifest.json"),
    }
    for split, view in [("train", 0), ("train", 1), ("val", 0)]:
        key = f"{split}-v{view}"
        target = root / f"{key}.npy"
        if target.exists():
            raise RuntimeError(f"Cache already exists; refuse overwrite: {target}")
        samples = meta["splits"][split]
        start = time.perf_counter()
        bank = np.lib.format.open_memmap(
            target.with_suffix(".partial.npy"), mode="w+", dtype=np.float16, shape=(len(samples), 3, 384, 40, 40)
        )
        for offset in range(0, len(samples), batch):
            images = []
            for s in samples[offset : offset + batch]:
                if sha(s["image_path"]) != s["image_sha256"]:
                    raise RuntimeError(f"Image changed: {s['image_path']}")
                im, info = load_letterboxed_tensor(s["image_path"], imgsz=640)
                if info != s["letterbox"]:
                    raise RuntimeError("Letterbox mismatch")
                images.append(im.flip(-1) if view else im)
            levels = extractor(torch.stack(images))
            values = torch.stack([levels[f"layer_{l}"] for l in ident["layers"]], 1)
            if tuple(values.shape[1:]) != (3, 384, 40, 40) or not torch.isfinite(values).all():
                raise RuntimeError("Invalid extracted features")
            bank[offset : offset + len(images)] = values.cpu().half().numpy()
            if offset == 0:
                single = extractor(images[0][None])
                exact = torch.stack([single[f"layer_{l}"][0] for l in ident["layers"]]).cpu().float()
                saved = torch.from_numpy(bank[0].copy()).float()
                rel = float((saved - exact).norm() / exact.norm().clamp_min(1e-8))
                result["parity"].append({"part": key, "relative_L2": rel})
                if rel > 0.02:
                    raise RuntimeError(f"Cache/online mismatch: {rel}")
            if offset % (batch * 50) == 0:
                print(f"BUILD {key} {offset + len(images)}/{len(samples)}", flush=True)
        bank.flush()
        del bank
        target.with_suffix(".partial.npy").replace(target)
        digest = sha(target)
        torch.cuda.synchronize()
        result["parts"][key] = {
            "file": str(target),
            "bytes": target.stat().st_size,
            "sha256": digest,
            "seconds": time.perf_counter() - start,
            "images": len(samples),
            "extra_parity_images": 1,
        }
        dump(root / "build-progress.json", result)
    result["wall_seconds"] = time.perf_counter() - started
    dump(root / "manifest.json", result)
    print(
        f"BUILD DONE wall_seconds={result['wall_seconds']:.1f} parity_max={max(x['relative_L2'] for x in result['parity']):.6f}",
        flush=True,
    )


def validate_mode(meta, smoke):
    expected = (17, 8) if smoke else (6471, 548)
    counts = tuple(len(meta["splits"][k]) for k in ("train", "val"))
    if meta["smoke"] != smoke or counts != expected:
        raise ValueError(f"Dataset mode mismatch: smoke={smoke}, counts={counts}, expected={expected}")


def verify_cache_part(path, expected, shape):
    if path.stat().st_size != expected["bytes"] or sha(path) != expected["sha256"]:
        raise RuntimeError(f"Cache bytes/hash mismatch: {path}")
    data = np.load(path, mmap_mode="r")
    if data.dtype != np.float16 or data.shape != tuple(shape):
        raise RuntimeError(f"Cache dtype/shape mismatch: {path}")


def verify_inputs_and_cache(job):
    started = time.perf_counter()
    meta = read(job / "dataset-manifest.json")
    cache = read(job / "cache/manifest.json")
    if sha(job / "dataset-manifest.json") != cache["dataset_manifest_sha256"]:
        raise RuntimeError("Dataset manifest changed after cache construction")
    checked = {}
    for key, part in cache["parts"].items():
        split = key.split("-")[0]
        path = job / "cache" / f"{key}.npy"
        verify_cache_part(path, part, (len(meta["splits"][split]), 3, 384, 40, 40))
        checked[key] = part["sha256"]
    if set(checked) != {"train-v0", "train-v1", "val-v0"}:
        raise RuntimeError("Incomplete cache manifest")
    for samples in meta["splits"].values():
        for sample in samples:
            if sha(sample["image_path"]) != sample["image_sha256"]:
                raise RuntimeError(f"Raw image changed: {sample['image_path']}")
            if sha(sample["label_path"]) != sample["label_sha256"]:
                raise RuntimeError(f"Raw annotation changed: {sample['label_path']}")
    for name, digest in meta["raw_val_annotation_sha256"].items():
        if sha(job / "raw-val-annotations" / name) != digest:
            raise RuntimeError(f"Raw validation annotation changed: {name}")
    result = {
        "seconds": time.perf_counter() - started,
        "cache_sha256": checked,
        "raw_images": sum(len(v) for v in meta["splits"].values()),
    }
    dump(job / "integrity-check.json", result)
    print(f"INTEGRITY PASS images={result['raw_images']} seconds={result['seconds']:.1f}", flush=True)
