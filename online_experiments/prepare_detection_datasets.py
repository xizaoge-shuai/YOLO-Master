"""Prepare full VOC and SKU official detection splits without running GPU jobs.

Only the Python standard library, Pillow and curl are required. Original JPEG,
XML and ImageSets files remain under raw/. Downloaded ZIPs remain in archives/.
YOLO labels omit difficult objects. The independent evaluation JSON retains
every object, difficult flag and original one-based inclusive bounding box.

Example: python online_experiments/prepare_autodl_voc.py --root /root/autodl-tmp/datasets/VOC
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import errno
import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
import time
import traceback
import xml.etree.ElementTree as ET
import zipfile
import zlib
from pathlib import Path, PurePosixPath

VERSION = "voc0712-fixed-splits-v1"
ASSET_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0"
ARCHIVES = (
    "VOCtrainval_06-Nov-2007.zip",
    "VOCtest_06-Nov-2007.zip",
    "VOCtrainval_11-May-2012.zip",
)
NAMES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)
EXPECTED = {("2007", "trainval"): 5011, ("2012", "trainval"): 11540, ("2007", "test"): 4952}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path, content, *, immutable=True):
    """Existing immutable files must agree byte-for-byte; interrupted writes cannot seal a run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"Refuse symlink output: {path}")
    if path.exists():
        if path.read_bytes() == content:
            return
        if immutable:
            raise ValueError(f"Refuse changed prepared file: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def safe_member(root, name):
    """Reject ZIP traversal, platform-specific path tricks and pre-existing symlink parents."""
    root = Path(root).resolve()
    if not name or "\\" in name or "\x00" in name or ":" in name:
        raise ValueError(f"Unsafe archive name: {name!r}")
    member = PurePosixPath(name)
    if member.is_absolute() or any(part in ("", ".", "..") for part in name.rstrip("/").split("/")):
        raise ValueError(f"Unsafe archive name: {name!r}")
    target = root.joinpath(*member.parts)
    for candidate in (target, *target.parents):
        if candidate == root:
            break
        if candidate.is_symlink():
            raise ValueError(f"Refuse symlink in archive destination: {candidate}")
    if not target.resolve().is_relative_to(root):
        raise ValueError(f"Archive path escapes root: {name!r}")
    return target


def validate_zip(path):
    """Check the central directory and every member's type/path, then stream CRC validation."""
    with zipfile.ZipFile(path) as archive:
        seen, total = set(), 0
        for member in archive.infolist():
            safe_member(Path(tempfile.gettempdir()) / "voc-zip-validation", member.filename)
            mode = member.external_attr >> 16
            if stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR):
                raise ValueError(f"Refuse non-regular archive member: {member.filename}")
            if stat.S_IFMT(mode) == stat.S_IFDIR and not member.is_dir():
                raise ValueError(f"Inconsistent archive directory metadata: {member.filename}")
            if member.flag_bits & 1:
                raise ValueError(f"Encrypted archive member: {member.filename}")
            if not member.is_dir() and member.filename in seen:
                raise ValueError(f"Duplicate archive member: {member.filename}")
            seen.add(member.filename)
            total += member.file_size
            if total > 20 * 2**30:
                raise ValueError("VOC archive exceeds the 20 GiB extraction safety limit")
        corrupt = archive.testzip()
        if corrupt is not None:
            raise ValueError(f"ZIP CRC validation failed: {corrupt}")


def download(root, filename, *, offline=False):
    folder = root / "archives"
    folder.mkdir(exist_ok=True)
    destination, partial = folder / filename, folder / f"{filename}.partial"
    if destination.is_symlink() or partial.is_symlink():
        raise ValueError(f"Refuse symlink archive: {destination}")
    if destination.exists():
        validate_zip(destination)
        return destination
    if offline:
        raise FileNotFoundError(f"Offline mode requires {destination}")
    if shutil.which("curl") is None:
        raise RuntimeError("curl is required for resumable downloads")
    print(f"DOWNLOAD {filename}; partial downloads are resumed", flush=True)
    subprocess.run(
        [
            "curl",
            "--silent",
            "--show-error",
            "--fail",
            "--location",
            "--continue-at",
            "-",
            "--retry",
            "10",
            "--limit-rate",
            "20M",
            "--retry-all-errors",
            "--connect-timeout",
            "30",
            "--speed-limit",
            "1024",
            "--speed-time",
            "120",
            "--output",
            str(partial),
            f"{ASSET_URL}/{filename}",
        ],
        check=True,
    )
    validate_zip(partial)
    partial.replace(destination)
    return destination


def crc32(path):
    result = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda stream=stream: stream.read(1024**2), b""):
            result = zlib.crc32(chunk, result)
    return result & 0xFFFFFFFF


def extract_checked(archive_path, destination):
    """Extract individually and atomically; repeated archives may share identical files."""
    if destination.is_symlink():
        raise ValueError(f"Refuse symlink extraction root: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = safe_member(destination, member.filename)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists():
                if not target.is_file() or target.stat().st_size != member.file_size or crc32(target) != member.CRC:
                    raise ValueError(f"Existing raw file differs from archive; preserved without overwrite: {target}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".partial", dir=target.parent)
            temporary = Path(temporary)
            try:
                with archive.open(member) as source, os.fdopen(descriptor, "wb") as output:
                    shutil.copyfileobj(source, output, length=1024**2)
                    output.flush()
                    os.fsync(output.fileno())
                if temporary.stat().st_size != member.file_size or crc32(temporary) != member.CRC:
                    raise ValueError(f"Extracted byte/CRC mismatch: {member.filename}")
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
            extracted += 1
            if extracted % 2000 == 0:
                print(f"EXTRACT {archive_path.name}: {extracted} new files", flush=True)


def year_directory(raw, year):
    candidates = [raw / "VOCdevkit" / f"VOC{year}", raw / f"VOC{year}"]
    present = [path for path in candidates if path.is_dir() and not path.is_symlink()]
    if len(present) != 1:
        raise ValueError(f"Expected one unambiguous VOC{year} directory; found {present}")
    return present[0]


def split_ids(directory, split, expected=None):
    path = directory / "ImageSets" / "Main" / f"{split}.txt"
    if path.is_symlink():
        raise ValueError(f"Refuse symlink split file: {path}")
    values = path.read_text(encoding="utf-8").split()
    if any(re.fullmatch(r"[A-Za-z0-9_-]+", identifier) is None for identifier in values):
        raise ValueError(f"Malformed image ID in {path}")
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate IDs in {path}")
    if expected is not None and len(values) != expected:
        raise ValueError(f"Expected {expected} IDs, found {len(values)}: {path}")
    return values


def voc_box_to_xyxy0(box, width, height):
    """VOC [xmin,ymin,xmax,ymax] inclusive -> zero-based half-open bounds.

    A VOC single-pixel box (1,1,1,1) becomes (0,0,1,1), not an empty box.
    This is a deliberate inclusive-coordinate conversion, not Ultralytics'
    historical xmin/xmax-difference convention. Evaluation keeps original VOC boxes.
    """
    if len(box) != 4 or any(not math.isfinite(value) or not float(value).is_integer() for value in box):
        raise ValueError(f"Non-integer/nonfinite VOC box: {box}")
    xmin, ymin, xmax, ymax = map(int, box)
    if not (1 <= xmin <= xmax <= width and 1 <= ymin <= ymax <= height):
        raise ValueError(f"VOC box outside {width}x{height}: {box}; no silent clipping")
    return [xmin - 1, ymin - 1, xmax, ymax]


def parse_annotation(xml_path, image_path):
    from PIL import Image

    if xml_path.is_symlink() or image_path.is_symlink():
        raise ValueError(f"Refuse symlink source: {xml_path}, {image_path}")
    payload = xml_path.read_bytes()
    if len(payload) > 2 * 1024**2 or b"<!DOCTYPE" in payload.upper() or b"<!ENTITY" in payload.upper():
        raise ValueError(f"Unexpected XML declarations or size: {xml_path}")
    annotation = ET.fromstring(payload)
    width, height = int(annotation.findtext("size/width")), int(annotation.findtext("size/height"))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {xml_path}")
    if annotation.findtext("filename") != image_path.name:
        raise ValueError(f"XML/image filename mismatch: {xml_path}")
    with Image.open(image_path) as image:
        if image.size != (width, height) or image.format != "JPEG":
            raise ValueError(f"XML/JPEG size or format mismatch: {image_path}")
        image.verify()  # Header/container validation only; full decode is deferred to the data loader.
    objects = []
    for item in annotation.findall("object"):
        name = item.findtext("name")
        difficult, truncated = int(item.findtext("difficult", "0")), int(item.findtext("truncated", "0"))
        if name not in NAMES or difficult not in (0, 1) or truncated not in (0, 1):
            raise ValueError(f"Invalid class/difficult/truncated metadata: {xml_path}")
        box = [float(item.findtext(f"bndbox/{key}")) for key in ("xmin", "ymin", "xmax", "ymax")]
        xyxy = voc_box_to_xyxy0(box, width, height)
        objects.append(
            {
                "class_id": NAMES.index(name),
                "name": name,
                "difficult": bool(difficult),
                "truncated": bool(truncated),
                "pose": item.findtext("pose", "Unspecified"),
                "bbox_voc_1based_inclusive": list(map(int, box)),
                "bbox_xyxy0_half_open": xyxy,
            }
        )
    return width, height, objects


def yolo_labels(objects, width, height):
    lines = []
    for item in objects:
        if item["difficult"]:
            continue
        x0, y0, x1, y1 = item["bbox_xyxy0_half_open"]
        coordinates = [(x0 + x1) / (2 * width), (y0 + y1) / (2 * height), (x1 - x0) / width, (y1 - y0) / height]
        lines.append(" ".join([str(item["class_id"]), *(f"{value:.12g}" for value in coordinates)]))
    return (("\n".join(lines) + "\n") if lines else "").encode()


def expose_image(source, destination, digest):
    """Hard-link within the dataset filesystem; copy only if hard links are unsupported."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ValueError(f"Refuse symlink prepared image: {destination}")
    if destination.exists():
        if not destination.is_file() or sha256(destination) != digest:
            raise ValueError(f"Prepared image differs from raw source: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError as error:
        if error.errno not in (errno.EXDEV, errno.EPERM, errno.ENOTSUP, errno.EACCES):
            raise
        atomic_write(destination, source.read_bytes())


def build_metadata(root, archives):
    raw = root / "raw"
    years = {year: year_directory(raw, year) for year in ("2007", "2012")}
    splits, source_splits = {}, {}
    for (year, split), expected in EXPECTED.items():
        identifiers = split_ids(years[year], split, expected)
        source_splits[f"{year}-{split}"] = {
            "path": str((years[year] / "ImageSets/Main" / f"{split}.txt").relative_to(root)),
            "sha256": sha256(years[year] / "ImageSets/Main" / f"{split}.txt"),
            "images": len(identifiers),
        }
        if split == "trainval":
            train, val = set(split_ids(years[year], "train")), set(split_ids(years[year], "val"))
            if train & val or train | val != set(identifiers):
                raise ValueError(f"VOC{year} trainval is not the exact disjoint train/val union")
        splits[(year, split)] = identifiers
    if set(splits[("2007", "trainval")]) & set(splits[("2007", "test")]):
        raise ValueError("VOC2007 official trainval/test overlap")
    entries = {"train": [], "test": []}
    for (year, split), identifiers in splits.items():
        target_split = "test" if split == "test" else "train"
        image_folder = f"{target_split}{year}"
        for index, identifier in enumerate(identifiers):
            image_path, xml_path = (
                years[year] / "JPEGImages" / f"{identifier}.jpg",
                years[year] / "Annotations" / f"{identifier}.xml",
            )
            width, height, objects = parse_annotation(xml_path, image_path)
            image_digest = sha256(image_path)
            exposed = safe_member(root, f"images/{image_folder}/{image_path.name}")
            expose_image(image_path, exposed, image_digest)
            label = safe_member(root, f"labels/{image_folder}/{identifier}.txt")
            atomic_write(label, yolo_labels(objects, width, height))
            entries[target_split].append(
                {
                    "sample_id": f"VOC{year}/{identifier}",
                    "year": year,
                    "official_split": split,
                    "image": str(exposed.relative_to(root)),
                    "raw_image": str(image_path.relative_to(root)),
                    "image_sha256": image_digest,
                    "image_bytes": image_path.stat().st_size,
                    "xml": str(xml_path.relative_to(root)),
                    "xml_sha256": sha256(xml_path),
                    "label": str(label.relative_to(root)),
                    "label_sha256": sha256(label),
                    "width": width,
                    "height": height,
                    "objects": objects,
                }
            )
            if (index + 1) % 1000 == 0:
                print(f"VERIFY VOC{year}/{split} {index + 1}/{len(identifiers)}", flush=True)
    if len(entries["train"]) != 16551 or len(entries["test"]) != 4952:
        raise ValueError("Official VOC07+12 / VOC2007 test count mismatch")
    for split, records in entries.items():
        atomic_write(root / f"{split}.txt", ("\n".join(f"./{record['image']}" for record in records) + "\n").encode())
    # This config supports final testing. Do not use VOC2007 test for tuning,
    # best-checkpoint selection, early stopping or per-epoch method selection.
    config = {
        "path": str(root),
        "train": "train.txt",
        "val": "test.txt",
        "test": "test.txt",
        "names": dict(enumerate(NAMES)),
    }
    atomic_write(root / "voc0712.yaml", json_bytes(config))  # JSON is valid YAML 1.2.
    evaluation = {
        "version": VERSION,
        "protocol": "VOC2007 AP50, 11-point interpolation; difficult ignored by evaluator",
        "box_convention": "Original VOC 1-based inclusive boxes are authoritative for VOC evaluation",
        "test_set_usage": "Fixed final evaluation only; no test-driven tuning or checkpoint selection",
        "names": list(NAMES),
        "images": entries["test"],
    }
    atomic_write(root / "evaluation" / "test2007.json", json_bytes(evaluation))
    manifest = {
        "version": VERSION,
        "names": list(NAMES),
        "training_images": 16551,
        "test_images": 4952,
        "archives": [
            {
                "filename": path.name,
                "url": f"{ASSET_URL}/{path.name}",
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in archives
        ],
        "source_splits": source_splits,
        "training_labels": "difficult objects omitted; original XML is unchanged",
        "training_coordinates": "VOC 1-based inclusive to XYXY0 half-open: xmin-1,ymin-1,xmax,ymax",
        "labels_test_usage": "Convenience labels omit difficult; use original XML or evaluation/test2007.json for primary AP",
        "evaluation_file": "evaluation/test2007.json",
        "evaluation_sha256": sha256(root / "evaluation/test2007.json"),
        "splits": entries,
    }
    atomic_write(root / "voc-source-manifest.json", json_bytes(manifest))
    seal = {
        "version": VERSION,
        "training_images": 16551,
        "test_images": 4952,
        "manifest_sha256": sha256(root / "voc-source-manifest.json"),
        "evaluation_sha256": manifest["evaluation_sha256"],
        "validation": "all listed JPEG headers checked; no full decode; XML, images, labels hashed",
    }
    atomic_write(root / "voc-source-complete.json", json_bytes(seal))
    return seal


@contextlib.contextmanager
def preparation_lock(root):
    if os.name != "posix":
        raise RuntimeError("Dataset preparation targets Linux/AutoDL (POSIX flock required)")
    import fcntl

    lock = root / ".prepare-dataset.lock"
    if lock.is_symlink():
        raise ValueError(f"Refuse symlink lock: {lock}")
    with lock.open("a") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another VOC preparation is running; inspect its existing log") from error
        yield


SCHEMA = "detection-dataset-v1"
SKU_URL = "https://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz"
SKU_BYTES = 12196152718
SKU_COUNTS = {"train": 8219, "val": 588, "test": 2936}
SOURCES = {
    "voc0712": {
        "dataset": "https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/",
        "splits_and_archives": "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/VOC.yaml",
        "rights": "Original image ownership and Flickr terms apply; see official VOC database rights.",
    },
    "sku110k": {
        "dataset": "https://github.com/eg4000/SKU110K_CVPR19",
        "download": SKU_URL,
        "splits": "https://docs.ultralytics.com/datasets/detect/sku-110k/",
        "rights": "Dataset authors limit use to academic and non-commercial purposes.",
    },
}


def state(root, status, **details):
    atomic_write(
        root / "status.json",
        json_bytes({"status": status, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **details}),
        immutable=False,
    )
    atomic_write(root / "status.txt", (status + "\n").encode(), immutable=False)
    print(status, flush=True)


def compact_json(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def checked_header(image_path):
    from PIL import Image

    if image_path.is_symlink() or not image_path.is_file():
        raise ValueError(f"Missing or symlink image: {image_path}")
    with Image.open(image_path) as image:
        size = image.size
        if image.format != "JPEG" or min(size) <= 0:
            raise ValueError(f"Not a positive-size JPEG: {image_path}")
        image.verify()
    return size


def record_identity(root, image_path, label_path, sample_id, width, height, objects, **extra):
    return {
        "sample_id": sample_id,
        "id": sample_id,
        "image": image_path.relative_to(root).as_posix(),
        "label": label_path.relative_to(root).as_posix(),
        "width": width,
        "height": height,
        "image_bytes": image_path.stat().st_size,
        "image_sha256": sha256(image_path),
        "label_sha256": sha256(label_path),
        "objects": objects,
        **extra,
    }


def seal_dataset(root, dataset, names, entries, archives, *, adjustments=None, source_splits=None):
    expected = {"train": 16551, "test": 4952} if dataset == "voc0712" else SKU_COUNTS
    identities = set()
    for split, records in entries.items():
        if len(records) != expected[split]:
            raise ValueError(f"{dataset}/{split}: expected {expected[split]}, got {len(records)}")
        for record in records:
            if record["sample_id"] in identities:
                raise ValueError(f"Duplicate or cross-split sample ID: {record['sample_id']}")
            identities.add(record["sample_id"])
        atomic_write(root / f"{split}.txt", ("\n".join(f"./{r['image']}" for r in records) + "\n").encode())
    evaluation = {
        "schema": SCHEMA,
        "dataset": dataset,
        "class_names": list(names),
        "names": list(names),
        "nc": len(names),
        "protocol": "VOC2007 11-point AP50; ignore difficult"
        if dataset == "voc0712"
        else "COCO AP50:95; report maxDets=100 and 500 separately",
        "test_set_usage": "Fixed final checkpoint evaluation only; no test-driven tuning or checkpoint selection",
        "images": entries["test"],
    }
    atomic_write(root / "evaluation/test.json", compact_json(evaluation))
    validation = {
        "missing": [],
        "corrupt": [],
        "duplicate_ids": [],
        "full_image_decode": False,
        "checks": "Archive streaming integrity, all listed JPEG headers and dimensions, all listed image/label SHA256",
        "decode_note": "JPEG header/container checks do not prove full image decode; data loader must surface decode failures.",
        "adjustments": adjustments or [],
        "silent_drops": 0,
    }
    manifest = {
        "schema": SCHEMA,
        "version": SCHEMA,
        "dataset": dataset,
        "class_names": list(names),
        "names": list(names),
        "nc": len(names),
        "expected_counts": expected,
        "box_convention": "zero-based half-open XYXY; original VOC inclusive boxes also preserved",
        "training_labels": "All valid objects, except VOC difficult objects omitted from training labels",
        "sources": SOURCES[dataset],
        "archives": archives,
        "source_splits": source_splits or {},
        "evaluation_file": "evaluation/test.json",
        "evaluation_sha256": sha256(root / "evaluation/test.json"),
        "splits": entries,
        "validation": validation,
    }
    atomic_write(root / "dataset-manifest.json", compact_json(manifest))
    config = {
        "path": str(root),
        "train": "train.txt",
        "val": "test.txt" if dataset == "voc0712" else "val.txt",
        "test": "test.txt",
        "names": dict(enumerate(names)),
    }
    atomic_write(root / f"{dataset}.yaml", json_bytes(config))
    complete = {
        "schema": SCHEMA,
        "dataset": dataset,
        "counts": {split: len(records) for split, records in entries.items()},
        "objects": {split: sum(len(r["objects"]) for r in records) for split, records in entries.items()},
        "manifest_sha256": sha256(root / "dataset-manifest.json"),
        "evaluation_sha256": manifest["evaluation_sha256"],
        "source_code_sha256": sha256(Path(__file__)),
        "validation": validation,
    }
    atomic_write(root / "complete.json", json_bytes(complete))
    return complete


def prepare_voc(root, offline):
    paths = []
    for filename in ARCHIVES:
        state(root, f"DOWNLOADING {filename}")
        archive = download(root, filename, offline=offline)
        paths.append(archive)
        state(root, f"EXTRACTING {filename}")
        extract_checked(archive, root / "raw")
    state(root, "INDEXING voc0712")
    build_metadata(root, paths)
    original = json.loads((root / "voc-source-manifest.json").read_text())
    entries = original["splits"]
    for records in entries.values():
        for record in records:
            record["id"] = record["sample_id"]
            for obj in record["objects"]:
                obj["category_id"] = obj["class_id"]
                obj["iscrowd"] = 0
    return seal_dataset(root, "voc0712", NAMES, entries, original["archives"], source_splits=original["source_splits"])


def download_sku(root, offline):
    folder = root / "archives"
    folder.mkdir(exist_ok=True)
    destination = folder / "SKU110K_fixed.tar.gz"
    partial = folder / "SKU110K_fixed.tar.gz.partial"
    if destination.is_symlink() or partial.is_symlink():
        raise ValueError("Refuse symlink SKU archive")
    if not destination.exists():
        if offline:
            raise FileNotFoundError(destination)
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
                "--limit-rate",
                "20M",
                "--connect-timeout",
                "30",
                "--speed-limit",
                "1024",
                "--speed-time",
                "120",
                "--output",
                str(partial),
                SKU_URL,
            ],
            check=True,
        )
        if partial.stat().st_size != SKU_BYTES:
            raise ValueError(f"SKU published archive byte count mismatch: {partial.stat().st_size} != {SKU_BYTES}")
        partial.replace(destination)
    if destination.stat().st_size != SKU_BYTES:
        raise ValueError("Existing SKU archive byte count mismatch; preserved")
    return destination


def extract_tar_checked(archive_path, destination):
    destination.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ValueError("Refuse symlink extraction root")
    total, count, seen = 0, 0, set()
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            name = member.name.rstrip("/")
            target = safe_member(destination, name)
            if name in seen or not (member.isfile() or member.isdir()):
                raise ValueError(f"Duplicate or nonregular tar member: {name}")
            seen.add(name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            total += member.size
            if total > 40 * 2**30:
                raise ValueError("SKU expanded archive exceeds 40GiB safety bound")
            target.parent.mkdir(parents=True, exist_ok=True)
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"Unreadable tar member: {name}")
            if target.exists():
                digest = hashlib.sha256()
                for chunk in iter(lambda stream=stream: stream.read(1024**2), b""):
                    digest.update(chunk)
                if not target.is_file() or target.stat().st_size != member.size or sha256(target) != digest.hexdigest():
                    raise ValueError(f"Existing raw file differs from official tar; preserved: {target}")
            else:
                descriptor, temporary_name = tempfile.mkstemp(
                    prefix=f".{target.name}.", suffix=".partial", dir=target.parent
                )
                temporary = Path(temporary_name)
                try:
                    with os.fdopen(descriptor, "wb") as output:
                        shutil.copyfileobj(stream, output, 1024**2)
                    if temporary.stat().st_size != member.size:
                        raise ValueError(f"Truncated tar member: {name}")
                    os.replace(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)
            count += 1
            if count % 1000 == 0:
                print(f"EXTRACT sku110k {count} files", flush=True)
    return {"files": count, "uncompressed_bytes": total}


def sku_rows(csv_path):
    records = {}
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        for number, row in enumerate(csv.reader(stream), 1):
            if len(row) != 8:
                raise ValueError(f"Expected eight SKU CSV columns at {csv_path}:{number}: {row}")
            filename, *rest = row
            if Path(filename).name != filename or not filename.lower().endswith(".jpg"):
                raise ValueError(f"Invalid SKU filename at {csv_path}:{number}: {filename}")
            x0, y0, x1, y1 = map(float, rest[:4])
            name, width, height = rest[4], int(rest[5]), int(rest[6])
            if name != "object" or width <= 0 or height <= 0 or any(not math.isfinite(v) for v in (x0, y0, x1, y1)):
                raise ValueError(f"Invalid SKU class/size/coordinates at {csv_path}:{number}")
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                raise ValueError(f"SKU box outside image at {csv_path}:{number}; preserved without silent clipping")
            record = records.setdefault(filename, {"width": width, "height": height, "objects": []})
            if (record["width"], record["height"]) != (width, height):
                raise ValueError(f"Inconsistent CSV dimensions for {filename}")
            record["objects"].append(
                {
                    "category_id": 0,
                    "class_id": 0,
                    "bbox_xyxy0_half_open": [x0, y0, x1, y1],
                    "difficult": False,
                    "iscrowd": 0,
                }
            )
    return records


def prepare_sku(root, offline):
    state(root, "DOWNLOADING SKU110K_fixed.tar.gz")
    archive = download_sku(root, offline)
    state(root, "EXTRACTING SKU110K_fixed.tar.gz")
    extraction = extract_tar_checked(archive, root / "raw")
    raw = root / "raw/SKU110K_fixed"
    if not (raw / "annotations").is_dir() or not (raw / "images").is_dir():
        raise ValueError("Official fixed SKU archive has unexpected layout")
    entries, source_splits = {}, {}
    for split, expected in SKU_COUNTS.items():
        state(root, f"INDEXING sku110k/{split}")
        csv_path = raw / "annotations" / f"annotations_{split}.csv"
        rows = sku_rows(csv_path)
        if len(rows) != expected:
            raise ValueError(f"SKU {split} CSV image count {len(rows)} != {expected}")
        source_splits[split] = {
            "path": csv_path.relative_to(root).as_posix(),
            "sha256": sha256(csv_path),
            "images": len(rows),
        }
        records = []
        for index, (filename, row) in enumerate(rows.items(), 1):
            image_path = raw / "images" / filename
            width, height = checked_header(image_path)
            if (width, height) != (row["width"], row["height"]):
                raise ValueError(f"SKU CSV/JPEG size mismatch: {filename}")
            exposed = root / "images" / split / filename
            expose_image(image_path, exposed, sha256(image_path))
            label = root / "labels" / split / f"{Path(filename).stem}.txt"
            atomic_write(label, yolo_labels(row["objects"], width, height))
            records.append(
                record_identity(
                    root,
                    exposed,
                    label,
                    f"SKU110K/{filename}",
                    width,
                    height,
                    row["objects"],
                    raw_image=image_path.relative_to(root).as_posix(),
                )
            )
            if index % 500 == 0:
                print(f"VERIFY sku110k/{split} {index}/{expected}", flush=True)
        entries[split] = records
    return seal_dataset(
        root,
        "sku110k",
        ("object",),
        entries,
        [
            {
                "filename": archive.name,
                "url": SKU_URL,
                "bytes": archive.stat().st_size,
                "sha256": sha256(archive),
                "extraction": extraction,
            }
        ],
        source_splits=source_splits,
    )


def main():
    parser = argparse.ArgumentParser(description="Prepare complete VOC/SKU data; no GPU or model imports")
    parser.add_argument("--dataset", required=True, choices=("voc0712", "sku110k"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.root.is_symlink():
        raise ValueError("Refuse symlink dataset root")
    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with preparation_lock(root):
        try:
            state(root, "STARTING", pid=os.getpid(), dataset=args.dataset, source_code_sha256=sha256(Path(__file__)))
            atomic_write(root / "pid.txt", f"{os.getpid()}\n".encode(), immutable=False)
            for relative in ("archives", "raw", "images", "labels", "evaluation"):
                safe_member(root, relative).mkdir(exist_ok=True)
            if (root / "complete.json").exists():
                seal = json.loads((root / "complete.json").read_text())
                if seal["manifest_sha256"] != sha256(root / "dataset-manifest.json"):
                    raise ValueError("Completed dataset manifest hash changed")
                if seal["evaluation_sha256"] != sha256(root / "evaluation/test.json"):
                    raise ValueError("Completed evaluation metadata hash changed")
            else:
                seal = prepare_voc(root, args.offline) if args.dataset == "voc0712" else prepare_sku(root, args.offline)
            state(root, "PREPARED", **seal, preparation_seconds=round(time.monotonic() - started, 2))
        except Exception as error:
            stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
            failure = {
                "dataset": args.dataset,
                "type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "source_code_sha256": sha256(Path(__file__)),
                "silent_drops": 0,
            }
            atomic_write(root / "failures" / f"{stamp}-{os.getpid()}.json", json_bytes(failure))
            state(root, "FAILED", **failure)
            raise


if __name__ == "__main__":
    main()
