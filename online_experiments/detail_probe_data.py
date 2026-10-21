"""Immutable extra-view caches and paired continuous-geometry detail batches."""

import io
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from online_experiments.d1_online_compare import geometry, training_feature
from online_experiments.feature_baselines import transform_targets, warp_features
from online_experiments.full_visdrone_data import dump, epoch_plan, make_extractor, read, read_feature_batch, sha
from scripts import d1_train_cached_detector as d
from ultralytics.data.foundation_cache import load_letterboxed_tensor

METHODS = ("transport25", "blank25", "views25", "detail25", "lowpass25")
ANCHORS = (0.85, 1.0, 1.15)
FEATURE_SHAPE = (3, 384, 40, 40)
SOURCE_PARTS = ("train-v0", "train-v1", "val-v0")
NEW_VIEWS = {"s085-v0": (0.85, False), "s085-v1": (0.85, True), "s115-v0": (1.15, False), "s115-v1": (1.15, True)}
VERSION = "detail-cache-v1"


def nearest_anchor(scale):
    """Select the nearest real view, resolving floating-point midpoint ties downward."""
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Scale must be finite and positive")
    for lower, upper in zip(ANCHORS, ANCHORS[1:]):
        midpoint = (lower + upper) / 2
        if scale <= midpoint or math.isclose(scale, midpoint, rel_tol=0, abs_tol=1e-12):
            return lower
    return ANCHORS[-1]


def transport_with_blank(values, scales, blank):
    """Center-warp features and fill missing bilinear support with true blank features."""
    if len(scales) != len(values) or not all(math.isfinite(s) and s > 0 for s in scales):
        raise ValueError("Invalid feature scales")
    values = values.float()
    flips = [False] * len(values)
    transported = warp_features(values, flips, scales)
    support = warp_features(torch.ones_like(values[:, :1]), flips, scales).clamp(0, 1)
    if blank is not None:
        blank = blank.to(device=values.device, dtype=values.dtype)
        transported = transported + (1 - support) * blank
    return transported, support


def cost_for_method(manifest, method):
    """Charge construction and physical storage for only the caches consumed by an arm."""
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    keys = list(SOURCE_PARTS)
    seconds = float(manifest["source_setup_seconds"])
    if method != "transport25":
        keys.append("blank")
        seconds += float(manifest["setup_seconds"])
    if method == "views25":
        keys += list(NEW_VIEWS)
    parts = [manifest["parts"][key] for key in keys]
    return {
        "cache_build_charged_seconds": seconds + sum(float(p["seconds"]) for p in parts),
        "cache_bytes": sum(int(p["bytes"]) for p in parts),
        "cache_build_query_images": sum(int(p["images"]) + int(p.get("extra_parity_images", 0)) for p in parts),
    }


def _stamp(path):
    stat = Path(path).stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns}


def _verify_part(part, shape, full_hash=True):
    path = Path(part["file"])
    if not path.is_file() or path.stat().st_size != part["bytes"]:
        raise RuntimeError(f"Cache bytes mismatch: {path}")
    if full_hash and sha(path) != part["sha256"]:
        raise RuntimeError(f"Cache hash mismatch: {path}")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.dtype != np.float16 or values.shape != tuple(shape):
        raise RuntimeError(f"Cache shape/dtype mismatch: {path}")
    del values


def _source_metadata(source):
    meta_path, cache_path = source / "dataset-manifest.json", source / "cache/manifest.json"
    meta, cache = read(meta_path), read(cache_path)
    if cache["dataset_manifest_sha256"] != sha(meta_path) or cache["identity"] != meta["identity"]:
        raise RuntimeError("Source dataset manifest/identity mismatch")
    if meta["identity"]["imgsz"] != 640 or meta["identity"]["layers"] != [3, 7, 11]:
        raise RuntimeError("Expected source image size 640 and layers 3/7/11")
    if set(cache["parts"]) != set(SOURCE_PARTS):
        raise RuntimeError("Incomplete source C2/validation cache")
    return meta, cache


def _verify_source(source, meta, cache, destination):
    """Hash the complete old source once; write evidence exclusively in the new cache."""
    started, files = time.perf_counter(), {}
    for key in SOURCE_PARTS:
        part = cache["parts"][key]
        count = len(meta["splits"][key.split("-")[0]])
        if part["images"] != count:
            raise RuntimeError(f"Source cache image count mismatch: {key}")
        _verify_part(part, (count, *FEATURE_SHAPE))
        path = str(Path(part["file"]).resolve())
        files[path] = {"sha256": part["sha256"], **_stamp(path)}
    for samples in meta["splits"].values():
        for sample in samples:
            for kind in ("image", "label"):
                path = str(Path(sample[f"{kind}_path"]).resolve())
                expected = sample[f"{kind}_sha256"]
                if path in files and files[path]["sha256"] == expected:
                    continue
                if sha(path) != expected:
                    raise RuntimeError(f"Source {kind} hash mismatch: {path}")
                files[path] = {"sha256": expected, **_stamp(path)}
    for name, expected in meta["raw_val_annotation_sha256"].items():
        path = str((source / "raw-val-annotations" / name).resolve())
        if sha(path) != expected:
            raise RuntimeError(f"Raw validation annotation mismatch: {path}")
        files[path] = {"sha256": expected, **_stamp(path)}
    result = {
        "source": str(source),
        "dataset_manifest_sha256": sha(source / "dataset-manifest.json"),
        "source_cache_manifest_sha256": sha(source / "cache/manifest.json"),
        "files": files,
        "seconds": time.perf_counter() - started,
        "raw_images": sum(len(samples) for samples in meta["splits"].values()),
        "verification": "full SHA256 initially; file stat identity checked on reuse",
    }
    dump(destination / "integrity.json", result)
    return result


def _verify_integrity(source, cache_dir, manifest):
    integrity_path = cache_dir / "integrity.json"
    if not integrity_path.is_file() or sha(integrity_path) != manifest["integrity_sha256"]:
        raise RuntimeError("Cache source integrity record hash mismatch")
    integrity = read(integrity_path)
    if (
        str(source) != integrity["source"]
        or sha(source / "dataset-manifest.json") != integrity["dataset_manifest_sha256"]
        or sha(source / "cache/manifest.json") != integrity["source_cache_manifest_sha256"]
    ):
        raise RuntimeError("Source manifests mismatch")
    for path, expected in integrity["files"].items():
        signature = {key: expected[key] for key in ("bytes", "mtime_ns", "ctime_ns")}
        if not Path(path).is_file() or _stamp(path) != signature:
            raise RuntimeError(f"Source file changed since integrity verification: {path}")


def _manifest_check(source, cache_dir, *, full_hash=False):
    path = cache_dir / "manifest.json"
    manifest = read(path)
    seal_path = cache_dir / "complete.json"
    if not seal_path.is_file() or read(seal_path)["manifest_sha256"] != sha(path):
        raise RuntimeError("Completed cache manifest hash mismatch")
    if manifest["version"] != VERSION or set(manifest["parts"]) != set(SOURCE_PARTS) | set(NEW_VIEWS) | {"blank"}:
        raise RuntimeError("Cache manifest version/parts mismatch")
    _verify_integrity(source, cache_dir, manifest)
    source_meta, source_cache = _source_metadata(source)
    if manifest["identity"] != source_meta["identity"]:
        raise RuntimeError("Cache teacher identity mismatch")
    for key in SOURCE_PARTS:
        if manifest["parts"][key] != source_cache["parts"][key]:
            raise RuntimeError(f"Source part record mismatch: {key}")
    for key in (*NEW_VIEWS, "blank"):
        part = manifest["parts"][key]
        shape = (1 if key == "blank" else manifest["train_images"], *FEATURE_SHAPE)
        if Path(part["file"]) != cache_dir / f"{key}.npy":
            raise RuntimeError(f"New cache file path mismatch: {key}")
        _verify_part(part, shape, full_hash=full_hash)
        if _stamp(part["file"]) != part["stat"]:
            raise RuntimeError(f"Cache bytes/hash identity changed: {key}")
    return manifest


def _extractor_device(extractor):
    device = getattr(extractor, "device", None)
    return torch.device(device if device is not None else next(extractor.model.parameters()).device)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _freeze(extractor):
    extractor.model.requires_grad_(False).eval()
    if any(p.requires_grad for p in extractor.model.parameters()):
        raise RuntimeError("Teacher must be frozen")


def _npy_bytes(shape):
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header, {"descr": np.dtype(np.float16).str, "fortran_order": False, "shape": tuple(shape)}
    )
    return header.tell() + math.prod(shape) * np.dtype(np.float16).itemsize


def _values(extractor, images, identity):
    with torch.no_grad():
        result = extractor(images)
        values = torch.stack([result[f"layer_{layer}"] for layer in identity["layers"]], dim=1)
    if tuple(values.shape[1:]) != FEATURE_SHAPE or not torch.isfinite(values).all():
        raise RuntimeError("Invalid extracted feature shape or nonfinite values")
    saved = values.detach().cpu().to(torch.float16)
    if not torch.isfinite(saved).all():
        raise RuntimeError("Nonfinite FP16 cache values")
    # Check storage parity against the very extraction being cached: no hidden teacher calls.
    exact = values.detach().float()
    relative = float((saved.to(exact.device).float() - exact).norm() / exact.norm().clamp_min(1e-8))
    if relative > 0.02:
        raise RuntimeError(f"FP16 cache/extraction parity mismatch: {relative}")
    return saved.numpy(), relative


def prepare_cache(source: Path, cache_dir: Path, batch_size=8, limit=None):
    """Stream four FP16 anchor banks and a true RGB-114 blank without modifying old caches."""
    source, cache_dir = Path(source).resolve(), Path(cache_dir).resolve()
    if batch_size < 1 or (limit is not None and limit < 1):
        raise ValueError("Batch size and optional training limit must be positive")
    if cache_dir == source or source in cache_dir.parents:
        raise RuntimeError("New cache must be outside the immutable source job")
    request = {"source": str(source), "limit": limit, "batch_size": batch_size}
    if (cache_dir / "manifest.json").exists():
        existing = read(cache_dir / "manifest.json")
        if existing.get("request") != request:
            raise RuntimeError("Completed cache request mismatch")
        return _manifest_check(source, cache_dir, full_hash=True)
    if cache_dir.exists() and any(cache_dir.iterdir()):
        raise RuntimeError("Refuse overwrite of nonempty/partial cache; use a new cache directory")
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta, source_cache = _source_metadata(source)
    samples = meta["splits"]["train"]
    if limit is not None and limit > len(samples):
        raise ValueError("Training limit exceeds source split")
    samples = samples[:limit] if limit is not None else samples
    if not samples:
        raise ValueError("Empty training split")
    required = 4 * _npy_bytes((len(samples), *FEATURE_SHAPE)) + _npy_bytes((1, *FEATURE_SHAPE))
    if shutil.disk_usage(cache_dir).free < required + 20 * 2**30:
        raise RuntimeError(f"Need {required / 2**30:.3f} GiB for new banks plus 20 GiB reserve")
    integrity = _verify_source(source, meta, source_cache, cache_dir)
    torch.set_num_threads(4)
    started = time.perf_counter()
    extractor = make_extractor(meta["identity"])
    _freeze(extractor)
    device = _extractor_device(extractor)
    _sync(device)
    manifest = {
        "version": VERSION,
        "request": request,
        "identity": meta["identity"],
        "dataset_manifest_sha256": source_cache["dataset_manifest_sha256"],
        "source_cache_manifest_sha256": integrity["source_cache_manifest_sha256"],
        "integrity_sha256": sha(cache_dir / "integrity.json"),
        "train_images": len(samples),
        "val_images": len(meta["splits"]["val"]),
        "source_setup_seconds": source_cache["setup_seconds"],
        "setup_seconds": time.perf_counter() - started,
        "teacher_dtype": "fp32",
        "cache_dtype": "fp16",
        "required_new_bytes": required,
        "reserve_bytes": 20 * 2**30,
        "parts": {key: source_cache["parts"][key] for key in SOURCE_PARTS},
        "new_query_images": 0,
        "new_query_calls": 0,
    }
    for key in (*NEW_VIEWS, "blank"):
        begin = time.perf_counter()
        target = cache_dir / f"{key}.npy"
        partial = cache_dir / f"{key}.partial.npy"
        count = 1 if key == "blank" else len(samples)
        bank = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float16, shape=(count, *FEATURE_SHAPE))
        parity, calls = 0.0, 0
        for offset in range(0, count, batch_size):
            if key == "blank":
                images = torch.full((1, 3, 640, 640), 114 / 255, dtype=torch.float32)
            else:
                scale, flip = NEW_VIEWS[key]
                images = []
                for sample in samples[offset : offset + batch_size]:
                    image, info = load_letterboxed_tensor(sample["image_path"], imgsz=640)
                    if info != sample["letterbox"]:
                        raise RuntimeError("Cache image letterbox mismatch")
                    image, _, _ = geometry(image, torch.empty((0, 1)), torch.empty((0, 4)), flip, scale)
                    images.append(image)
                images = torch.stack(images)
            values, relative = _values(extractor, images, meta["identity"])
            bank[offset : offset + len(images)] = values
            parity, calls = max(parity, relative), calls + 1
            manifest["new_query_images"] += len(images)
            manifest["new_query_calls"] += 1
            if offset % (batch_size * 50) == 0:
                print(f"BUILD {key} {offset + len(images)}/{count}", flush=True)
        bank.flush()
        del bank
        partial.replace(target)
        digest = sha(target)
        _sync(device)
        manifest["parts"][key] = {
            "file": str(target),
            "bytes": target.stat().st_size,
            "sha256": digest,
            "stat": _stamp(target),
            "images": count,
            "query_calls": calls,
            "extra_parity_images": 0,
            "parity_relative_L2": parity,
            "parity_protocol": "FP16 roundtrip of cached extraction; no extra query",
            "seconds": time.perf_counter() - begin,
        }
        dump(cache_dir / "build-progress.json", manifest)
    manifest["wall_seconds"] = time.perf_counter() - started
    manifest["source_verification_seconds"] = integrity["seconds"]
    dump(cache_dir / "manifest.json", manifest)
    dump(cache_dir / "complete.json", {"manifest_sha256": sha(cache_dir / "manifest.json")})
    return manifest


class DetailBatches:
    """Read only each selected feature batch; query and RGB geometry share one exact tensor."""

    def __init__(self, source, cache_manifest, method, seed, train_limit=None, val_limit=None):
        if method not in METHODS:
            raise ValueError(f"Unknown method: {method}")
        source, cache_manifest = Path(source).resolve(), Path(cache_manifest).resolve()
        self.cache = _manifest_check(source, cache_manifest.parent)
        if cache_manifest.name != "manifest.json":
            raise ValueError("Expected the completed cache manifest.json")
        self.meta = read(source / "dataset-manifest.json")
        self.train = self.meta["splits"]["train"]
        self.val = self.meta["splits"]["val"]
        for name, limit, count in (("train", train_limit, len(self.train)), ("val", val_limit, len(self.val))):
            if limit is not None and not 1 <= limit <= count:
                raise ValueError(f"Invalid {name} limit")
        self.train = self.train[:train_limit] if train_limit is not None else self.train
        self.val = self.val[:val_limit] if val_limit is not None else self.val
        if len(self.train) > self.cache["train_images"]:
            raise RuntimeError("Training split exceeds the built anchor cache")
        self.method, self.seed, self.pct = method, seed, 25
        self.arrays = [np.load(self.cache["parts"][f"train-v{v}"]["file"], mmap_mode="r") for v in (0, 1)]
        self.val_array = np.load(self.cache["parts"]["val-v0"]["file"], mmap_mode="r")
        self.view_arrays = (
            {
                (scale, int(flip)): np.load(self.cache["parts"][key]["file"], mmap_mode="r")
                for key, (scale, flip) in NEW_VIEWS.items()
            }
            if method == "views25"
            else {}
        )
        self.extractor = make_extractor(self.meta["identity"])
        _freeze(self.extractor)
        self.device = _extractor_device(self.extractor)
        self.blank = (
            None
            if method == "transport25"
            else torch.from_numpy(np.load(self.cache["parts"]["blank"]["file"])[0].copy()).to(self.device).float()
        )
        self.query_images = self.query_calls = self.correction_steps = 0
        self.correction_seconds = self.raw_image_seconds = 0.0
        self.raw_image_images = 0
        self.reconstruction = []
        self.corrector = self.optimizer = None

    def item(self, sample, flip=False, scale=1.0):
        classes = torch.tensor(sample["cls"], dtype=torch.float32).reshape(-1, 1)
        boxes = torch.tensor(sample["bboxes"], dtype=torch.float32).reshape(-1, 4)
        classes, boxes = transform_targets(classes, boxes, flip, scale, 640)
        return {"cls": classes, "bboxes": boxes, "sample_id": sample["sample_id"], "image_path": sample["image_path"]}

    def _image(self, sample, flip=False, scale=1.0):
        started = time.perf_counter()
        image, info = load_letterboxed_tensor(sample["image_path"], imgsz=640)
        self.raw_image_seconds = getattr(self, "raw_image_seconds", 0.0) + time.perf_counter() - started
        self.raw_image_images = getattr(self, "raw_image_images", 0) + 1
        if info != sample["letterbox"]:
            raise RuntimeError("Online/cache letterbox mismatch")
        raw, transformed = self.item(sample), self.item(sample, flip, scale)
        image, classes, boxes = geometry(image, raw["cls"], raw["bboxes"], flip, scale)
        if not torch.equal(classes, transformed["cls"]) or not torch.allclose(boxes, transformed["bboxes"], atol=2e-6):
            raise RuntimeError("Online/cache target mismatch")
        return image

    def training(self, epoch, batch_size, output):
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
        order, flips, scales, query = epoch_plan(len(self.train), self.pct, epoch, self.seed)
        schedule = {"epoch": epoch + 1, "order": order, "flip": flips, "scale": scales, "query": query}
        path = Path(output) / "schedules" / f"{epoch + 1:03d}.json"
        if path.exists():
            if read(path) != schedule:
                raise RuntimeError("Existing paired schedule mismatch")
        else:
            dump(path, schedule)
        has_rgb = self.method in ("detail25", "lowpass25")
        for start in range(0, len(order), batch_size):
            ids, fs, ss, qs = (values[start : start + batch_size] for values in (order, flips, scales, query))
            anchors = [nearest_anchor(s) for s in ss] if self.method == "views25" else [1.0] * len(ids)
            data = np.empty((len(ids), *self.arrays[0].shape[1:]), dtype=np.float16)
            for j, (index, flip, anchor) in enumerate(zip(ids, fs, anchors)):
                bank = self.arrays[int(flip)] if anchor == 1.0 else self.view_arrays[(anchor, int(flip))]
                data[j] = bank[index]
            values = torch.from_numpy(data).to(self.device).float()
            relative = [scale / anchor for scale, anchor in zip(ss, anchors)]
            levels = tuple(
                transport_with_blank(
                    values[:, level], relative, None if self.method == "transport25" else self.blank[level][None]
                )[0]
                for level in range(3)
            )
            items = [self.item(self.train[index], flip, scale) for index, flip, scale in zip(ids, fs, ss)]
            queried = [j for j, chosen in enumerate(qs) if chosen]
            needed = range(len(ids)) if has_rgb else queried
            images = {j: self._image(self.train[ids[j]], fs[j], ss[j]) for j in needed}
            if queried:
                with torch.no_grad():
                    result = self.extractor(torch.stack([images[j] for j in queried]))
                exact = tuple(
                    training_feature(result[f"layer_{layer}"]).to(self.device)
                    for layer in self.meta["identity"]["layers"]
                )
                if not all(torch.isfinite(value).all() for value in exact):
                    raise RuntimeError("Nonfinite queried teacher feature")
                for level, teacher in zip(levels, exact):
                    level[queried] = teacher
                self.query_images += len(queried)
                self.query_calls += 1
            for j, item in enumerate(items):
                item["features"] = tuple(level[j] for level in levels)
                if has_rgb:
                    item["features"] += (images[j],)
            yield d.collate_cached(items)

    def validation(self, batch_size):
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
        has_rgb = self.method in ("detail25", "lowpass25")
        for start in range(0, len(self.val), batch_size):
            ids = list(range(start, min(start + batch_size, len(self.val))))
            values = torch.from_numpy(read_feature_batch([self.val_array], ids, [0] * len(ids)))
            items = [self.item(self.val[index]) for index in ids]
            for j, item in enumerate(items):
                item["features"] = tuple(values[j, level] for level in range(3))
                if has_rgb:
                    item["features"] += (self._image(self.val[ids[j]]),)
            yield d.collate_cached(items)
