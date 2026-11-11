"""Prepare pinned second-backbone weights or full official COCO assets on CPU."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import stat
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "timm/vit_base_patch16_dinov3.lvd1689m"
REVISION = "c6a5fb7d12bbd3cf3b0079253141c3332aaed7da"
COCO_FILES = {
    "train2017.zip": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def status(root, state, **extra):
    write(root / "status.json", {"state": state, "time_unix": time.time(), **extra})
    (root / "status.txt").write_text(state + "\n", encoding="utf-8")


def model_assets(root):
    import certifi
    import requests
    from huggingface_hub import HfApi, snapshot_download

    os.environ["SSL_CERT_FILE"] = certifi.where()
    dependencies = ROOT / ".research-deps/expansion-20261110/site"
    pin = root / "timm-release.json"
    if pin.exists():
        release = read(pin)
    else:
        response = requests.get("https://pypi.org/pypi/timm/json", timeout=40)
        response.raise_for_status()
        release = {"version": response.json()["info"]["version"], "source": "https://pypi.org/pypi/timm/json"}
        write(pin, release)
    status(root, "INSTALLING_ISOLATED_TIMM", version=release["version"])
    dependencies.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(dependencies),
            f"timm=={release['version']}",
            "--report",
            str(root / "timm-install-report.json"),
        ],
        check=True,
    )
    sys.path.insert(0, str(dependencies))
    actual = importlib.metadata.version("timm")
    if actual != release["version"]:
        raise RuntimeError("Isolated timm version does not match its pinned release")
    status(root, "DOWNLOADING_PINNED_MODEL", model_id=MODEL_ID, revision=REVISION)
    info = HfApi().model_info(MODEL_ID, revision=REVISION, files_metadata=True)
    if info.sha != REVISION or info.gated:
        raise RuntimeError("Expected the pinned public timm model release")
    snapshot = Path(snapshot_download(MODEL_ID, revision=REVISION, allow_patterns=["config.json", "model.safetensors"]))
    weights, config = snapshot / "model.safetensors", snapshot / "config.json"
    digest = sha(weights)
    lfs = next(item.lfs for item in info.siblings if item.rfilename == "model.safetensors")
    expected = lfs.sha256 if hasattr(lfs, "sha256") else lfs["sha256"]
    if digest != expected:
        raise RuntimeError("Downloaded weights do not match the pinned Hub LFS SHA256")
    identity = {
        "backend": "timm-dinov3-eva",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "timm_model": "vit_base_patch16_dinov3",
        "layers": [3, 7, 11],
        "block_indices": [2, 6, 10],
        "channels": 768,
        "patch_size": 16,
        "imgsz": 640,
        "weights_file": str(weights),
        "weights_sha256": digest,
        "config_file": str(config),
        "config_sha256": sha(config),
        "timm_version": actual,
        "dependencies": str(dependencies),
        "layer_contract": "HF labels include stem; timm zero-based block indices2/6/10; normalized intermediates",
        "training_completed": False,
    }
    write(root / "model-identity.json", identity)
    write(root / "complete.json", {"identity_sha256": sha(root / "model-identity.json")})
    status(root, "PREPARED_MODEL", identity=str(root / "model-identity.json"))


def archive_assets(root):
    archives = root / "archives"
    archives.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, url in COCO_FILES.items():
        path = archives / name
        if not path.exists():
            partial = path.with_name(name + ".partial")
            status(root, "DOWNLOADING_COCO", file=name, source=url)
            subprocess.run(
                [
                    "curl",
                    "--fail",
                    "--location",
                    "--silent",
                    "--show-error",
                    "--continue-at",
                    "-",
                    "--retry",
                    "10",
                    "--retry-all-errors",
                    "--connect-timeout",
                    "30",
                    "--speed-limit",
                    "1024",
                    "--speed-time",
                    "120",
                    "--limit-rate",
                    "20M",
                    "--output",
                    str(partial),
                    url,
                ],
                check=True,
            )
            partial.replace(path)
        result[name] = {"source": url, "bytes": path.stat().st_size, "sha256": sha(path)}
    write(root / "archives.json", result)
    return result


def safe_member(name, mode):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or stat.S_ISLNK(mode):
        raise ValueError("Unsafe archive entry: " + name)


def contained_path(parent, name):
    safe_member(name, 0)
    target = parent / name
    if not target.resolve().is_relative_to(parent.resolve()):
        raise ValueError("Extraction or metadata path escapes its data root: " + name)
    return target


def coco_assets(root):
    from PIL import Image

    available = shutil.disk_usage(root).free
    if available < 100 * 1024**3:
        raise RuntimeError("COCO preparation requires at least100GiB free for archives and extraction")
    archives = archive_assets(root)
    raw = root / "raw"
    raw.mkdir(exist_ok=True)
    for name in COCO_FILES:
        status(root, "EXTRACTING_COCO", file=name)
        with zipfile.ZipFile(root / "archives" / name) as archive:
            members = archive.infolist()
            if sum(item.file_size for item in members) > 100 * 1024**3:
                raise ValueError("Archive exceeds extraction bound")
            for entry in members:
                safe_member(entry.filename, entry.external_attr >> 16)
            relevant = [
                item
                for item in members
                if not item.is_dir()
                and (
                    name != "annotations_trainval2017.zip"
                    or Path(item.filename).name in {"instances_train2017.json", "instances_val2017.json"}
                )
            ]
            expected = 118287 if name == "train2017.zip" else 5000 if name == "val2017.zip" else 2
            if len(relevant) != expected:
                raise RuntimeError("Official full archive member count mismatch")
            for index, entry in enumerate(relevant, 1):
                target = contained_path(raw, entry.filename)
                if target.exists():
                    # Read the existing and ZIP bytes, verifying ZIP CRC and avoiding silent overwrite.
                    with archive.open(entry) as stream:
                        expected_hash = hashlib.sha256(stream.read()).hexdigest()
                    if sha(target) != expected_hash:
                        raise RuntimeError("Existing extraction differs from official archive: " + str(target))
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_name(target.name + ".partial")
                    with archive.open(entry) as source, temporary.open("wb") as destination:
                        for block in iter(lambda: source.read(1024**2), b""):
                            destination.write(block)
                    temporary.replace(target)
                if index % 5000 == 0:
                    print(f"EXTRACT {name} {index}/{expected}", flush=True)
    counts, split_ids = {}, {}
    for split, expected in (("train", 118287), ("val", 5000)):
        status(root, "VERIFYING_COMPLETE_COCO", split=split)
        annotation = raw / "annotations" / f"instances_{split}2017.json"
        metadata = read(annotation)
        if len(metadata["images"]) != expected or len(metadata["categories"]) != 80:
            raise RuntimeError("Official COCO metadata counts mismatch")
        ids = set()
        for index, image in enumerate(metadata["images"], 1):
            if image["id"] in ids:
                raise RuntimeError("Duplicate COCO image ID")
            ids.add(image["id"])
            path = contained_path(raw / f"{split}2017", image["file_name"])
            with Image.open(path) as pixels:
                if pixels.size != (image["width"], image["height"]):
                    raise RuntimeError("COCO image metadata dimensions mismatch")
                pixels.convert("RGB").load()
            if index % 5000 == 0:
                print(f"DECODE {split} {index}/{expected}", flush=True)
        counts[split] = {
            "images": expected,
            "objects": len(metadata["annotations"]),
            "annotation_sha256": sha(annotation),
            "image_ids_sha256": hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest(),
        }
        split_ids[split] = ids
    if split_ids["train"] & split_ids["val"]:
        raise RuntimeError("COCO train and validation image IDs overlap")
    metadata = {
        "schema": "coco2017-assets-v1",
        "root": str(root),
        "raw": str(raw),
        "counts": counts,
        "archives": archives,
        "train_images": str(raw / "train2017"),
        "val_images": str(raw / "val2017"),
        "train_annotations": str(raw / "annotations/instances_train2017.json"),
        "val_annotations": str(raw / "annotations/instances_val2017.json"),
        "ground_truth_policy": "Original COCO category IDs, area, crowd and annotations remain unchanged",
        "training_started": False,
    }
    write(root / "assets.json", metadata)
    write(root / "complete.json", {"assets_sha256": sha(root / "assets.json")})
    status(root, "PREPARED_COCO", counts=counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("model", "coco"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.root.mkdir(parents=True, exist_ok=True)
    import fcntl

    with (args.root / "preparation.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit("Existing preparation owns this task") from error
        if (args.root / "complete.json").exists():
            seal = read(args.root / "complete.json")
            file = "model-identity.json" if args.kind == "model" else "assets.json"
            key = "identity_sha256" if args.kind == "model" else "assets_sha256"
            if seal[key] != sha(args.root / file):
                raise RuntimeError("Existing asset seal changed")
            print("SKIP sealed complete assets", flush=True)
            return
        try:
            (model_assets if args.kind == "model" else coco_assets)(args.root)
        except Exception as error:
            write(
                args.root / f"failure-{time.time_ns()}.json",
                {"type": type(error).__name__, "message": str(error), "preserve_all_partials": True},
            )
            status(args.root, "FAILED", error_type=type(error).__name__, message=str(error))
            raise


if __name__ == "__main__":
    main()
