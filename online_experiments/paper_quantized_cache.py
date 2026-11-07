"""Persist the quantized C1-cache idea from Cache and Reuse (CVPRW 2024).

The paper's channelwise affine quantizer is applied independently to each
image/layer/channel. Its unspecified quantile parameter is fixed here at 0.001
as a detection-adaptation choice. This module does not run a backbone, augment
labels, train a detector, or change the common uncompressed validation cache.
"""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

VERSION = "paper-quantized-c1-cache-v1"
QUANTILE = 0.001


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stamp(path):
    stat = Path(path).stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns}


def _write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _record(path):
    path = Path(path).resolve(strict=True)
    return {"file": str(path), "sha256": _sha(path), "stat": _stamp(path)}


def _check_record(record, *, full_hash):
    path = Path(record["file"])
    if not path.is_file() or _stamp(path) != record["stat"]:
        raise RuntimeError(f"Verified file changed: {path}")
    if full_hash and _sha(path) != record["sha256"]:
        raise RuntimeError(f"Verified metadata hash mismatch: {path}")


def _source(cache_manifest):
    """Read sealed integrity evidence without rehashing the large source bank."""
    path = Path(cache_manifest).resolve(strict=True)
    manifest, seal = _read(path), _read(path.parent / "complete.json")
    if manifest.get("version") != "detail-cache-v1" or seal.get("manifest_sha256") != _sha(path):
        raise RuntimeError("Expected a sealed, verified detail-cache manifest")
    integrity_path = path.parent / "integrity.json"
    if manifest.get("integrity_sha256") != _sha(integrity_path):
        raise RuntimeError("Source cache integrity evidence hash mismatch")
    integrity = _read(integrity_path)
    source = Path(manifest["request"]["source"]).resolve(strict=True)
    dataset_path, source_cache_path = source / "dataset-manifest.json", source / "cache/manifest.json"
    if (
        integrity["source"] != str(source)
        or manifest["dataset_manifest_sha256"] != _sha(dataset_path)
        or integrity["dataset_manifest_sha256"] != manifest["dataset_manifest_sha256"]
        or manifest["source_cache_manifest_sha256"] != _sha(source_cache_path)
        or integrity["source_cache_manifest_sha256"] != manifest["source_cache_manifest_sha256"]
    ):
        raise RuntimeError("Verified source metadata mismatch")
    original_cache, dataset = _read(source_cache_path), _read(dataset_path)
    part = manifest["parts"]["train-v0"]
    if part != original_cache["parts"]["train-v0"] or manifest["identity"] != dataset["identity"]:
        raise RuntimeError("C1 source-bank record or teacher identity mismatch")
    bank = Path(part["file"]).resolve(strict=True)
    bank_stat = _stamp(bank)
    verified = integrity["files"].get(str(bank))
    if (
        verified is None
        or verified["sha256"] != part["sha256"]
        or {key: verified[key] for key in bank_stat} != bank_stat
        or part.get("stat") != bank_stat
        or int(part["bytes"]) != bank_stat["bytes"]
    ):
        raise RuntimeError("C1 bank lacks matching full-hash integrity evidence and current file identity")
    if verified.get("matches_expected") is False or verified.get("actual_sha256", part["sha256"]) != part["sha256"]:
        raise RuntimeError("Original C1 bank integrity verification did not pass")
    images = int(manifest["train_images"])
    if images != int(part["images"]) or images != len(dataset["splits"]["train"]):
        raise RuntimeError("C1 training-image count mismatch")
    arrays = np.load(bank, mmap_mode="r", allow_pickle=False)
    if arrays.dtype != np.float16 or arrays.ndim != 5 or arrays.shape[0] != images or min(arrays.shape) < 1:
        raise RuntimeError("Expected a nonempty N,L,C,H,W FP16 C1 feature bank")
    evidence = {
        "metadata": {
            "detail_cache_manifest": _record(path),
            "detail_completion": _record(path.parent / "complete.json"),
            "integrity": _record(integrity_path),
            "dataset_manifest": _record(dataset_path),
            "source_cache_manifest": _record(source_cache_path),
        },
        "bank": {"file": str(bank), "sha256": part["sha256"], "stat": bank_stat},
        "verification": "Existing full SHA256 verification plus current metadata hashes and source-bank file stat",
    }
    return manifest, evidence, arrays


def _check_source(evidence):
    for record in evidence["metadata"].values():
        _check_record(record, full_hash=True)
    _check_record(evidence["bank"], full_hash=False)


def _part(root, filename, shape, dtype):
    path = root / filename
    return {
        "file": filename,
        "shape": list(shape),
        "dtype": np.dtype(dtype).str,
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
        "stat": _stamp(path),
    }


def verify_quantized_cache(path, *, request=None):
    """Verify a finalized bank using its seal, immutable file stamps and metadata."""
    path = Path(path).resolve(strict=True)
    if path.is_dir():
        path = path / "manifest.json"
    manifest, seal = _read(path), _read(path.parent / "complete.json")
    if (
        manifest.get("version") != VERSION
        or seal.get("version") != VERSION
        or seal.get("manifest_sha256") != _sha(path)
    ):
        raise RuntimeError("Quantized cache completion seal mismatch")
    if request is not None and manifest["request"] != request:
        raise RuntimeError("Completed quantized cache request mismatch")
    bits, shape = int(manifest["bits"]), tuple(manifest["shape"])
    if bits not in (4, 8) or len(shape) != 5 or min(shape) < 1 or int(manifest["train_images"]) != shape[0]:
        raise RuntimeError("Invalid quantized cache dimensions or bit depth")
    _check_source(manifest["source"])
    if set(manifest["parts"]) != {"codes", "lo", "step"}:
        raise RuntimeError("Incomplete quantized cache parts")
    expected_shapes = {
        "codes": (*shape[:-1], (shape[-1] + 1) // 2) if bits == 4 else shape,
        "lo": shape[:3],
        "step": shape[:3],
    }
    for key, part in manifest["parts"].items():
        filename = {"codes": "codes.npy", "lo": "lo.npy", "step": "step.npy"}[key]
        if part["file"] != filename:
            raise RuntimeError("Unsafe or unexpected quantized cache filename")
        target = path.parent / filename
        if target.is_symlink() or _stamp(target) != part["stat"] or int(part["bytes"]) != part["stat"]["bytes"]:
            raise RuntimeError(f"Quantized cache file changed: {filename}")
        values = np.load(target, mmap_mode="r", allow_pickle=False)
        dtype = np.dtype(np.uint8 if key == "codes" else np.float32)
        if values.shape != expected_shapes[key] or values.dtype != dtype or part["shape"] != list(values.shape):
            raise RuntimeError(f"Quantized cache shape/dtype mismatch: {filename}")
        if part["dtype"] != dtype.str:
            raise RuntimeError(f"Quantized manifest dtype mismatch: {filename}")
        del values
    if int(manifest["cache_bytes"]) != sum(int(part["bytes"]) for part in manifest["parts"].values()):
        raise RuntimeError("Quantized cache byte accounting mismatch")
    if (
        manifest["total_bytes"] != manifest["cache_bytes"]
        or manifest["images"] != manifest["train_images"]
        or manifest["source_train_bank_sha256"] != manifest["source"]["bank"]["sha256"]
        or manifest["source_train_bank_bytes"] != manifest["source"]["bank"]["stat"]["bytes"]
        or manifest["build_seconds"] != manifest["cost"]["compression_build_seconds"]
    ):
        raise RuntimeError("Quantized cache summary fields disagree with source or cost evidence")
    return manifest


def build_quantized_cache(cache_manifest, destination, *, bits=8, quantile=QUANTILE):
    """Stream C1 images into actual UINT8 or packed INT4 files without teacher calls.

    Partial nonempty destinations are retained and refused. Completed banks are
    reused only if their exact source/quantizer request and file identities match.
    A caller can choose a fresh destination after an interrupted construction.
    """
    if isinstance(bits, bool) or not isinstance(bits, (int, np.integer)) or bits not in (4, 8):
        raise ValueError("bits must be 4 or 8")
    if not math.isfinite(quantile) or not 0 <= quantile < 0.5:
        raise ValueError("quantile must be finite and in [0,0.5)")
    if quantile != QUANTILE:
        raise ValueError("This baseline fixes quantile=0.001; changing it defines a separate adaptation")
    started = time.perf_counter()
    cache_manifest = Path(cache_manifest).resolve(strict=True)
    destination = Path(destination).resolve()
    if destination.is_relative_to(cache_manifest.parent) or cache_manifest.parent.is_relative_to(destination):
        raise ValueError("Quantized cache destination must be separate from the immutable source cache")
    request = {
        "source_cache_manifest": str(cache_manifest),
        "source_cache_manifest_sha256": _sha(cache_manifest),
        "bits": int(bits),
        "quantile": float(quantile),
    }
    if destination.exists() and any(destination.iterdir()):
        if not (destination / "complete.json").is_file():
            raise RuntimeError("Refuse overwrite of a partial/nonempty quantized cache; choose a new destination")
        verify_quantized_cache(destination, request=request)
        return destination / "manifest.json"
    original, evidence, source = _source(cache_manifest)
    shape, qmax = tuple(source.shape), (1 << bits) - 1
    destination.mkdir(parents=True, exist_ok=True)
    code_shape = (*shape[:-1], (shape[-1] + 1) // 2) if bits == 4 else shape
    codes = np.lib.format.open_memmap(destination / "codes.npy", mode="w+", dtype=np.uint8, shape=code_shape)
    offsets = np.lib.format.open_memmap(destination / "lo.npy", mode="w+", dtype=np.float32, shape=shape[:3])
    steps = np.lib.format.open_memmap(destination / "step.npy", mode="w+", dtype=np.float32, shape=shape[:3])
    clipped_low = clipped_high = constant_channels = 0
    for index in range(shape[0]):
        image = np.asarray(source[index], dtype=np.float32)
        if not np.isfinite(image).all():
            raise RuntimeError(f"Nonfinite source C1 feature at image {index}")
        bounds = np.quantile(image, [quantile, 1.0 - quantile], axis=(-2, -1), method="linear").astype(np.float32)
        lo, hi = bounds
        step = (hi - lo) / np.float32(qmax)
        if not np.isfinite(bounds).all() or not np.isfinite(step).all() or (step < 0).any():
            raise RuntimeError(f"Invalid quantile bounds at image {index}")
        constant_channels += int(np.count_nonzero(step == 0))
        clipped_low += int(np.count_nonzero(image < lo[..., None, None]))
        clipped_high += int(np.count_nonzero(image > hi[..., None, None]))
        normalized = np.zeros_like(image)
        np.divide(image - lo[..., None, None], step[..., None, None], out=normalized, where=step[..., None, None] > 0)
        np.clip(normalized, 0, qmax, out=normalized)
        quantized = np.rint(normalized).astype(np.uint8)
        if bits == 4:
            packed = quantized[..., ::2].copy()
            packed[..., : shape[-1] // 2] |= quantized[..., 1::2] << np.uint8(4)
            codes[index] = packed
        else:
            codes[index] = quantized
        offsets[index], steps[index] = lo, step
        if index == 0 or (index + 1) % 100 == 0 or index + 1 == shape[0]:
            print(
                json.dumps({"stage": "quantize-c1", "bits": bits, "images": index + 1, "total": shape[0]}), flush=True
            )
    for values in (codes, offsets, steps):
        values.flush()
    del codes, offsets, steps, source, values
    _check_source(evidence)
    parts = {
        "codes": _part(destination, "codes.npy", code_shape, np.uint8),
        "lo": _part(destination, "lo.npy", shape[:3], np.float32),
        "step": _part(destination, "step.npy", shape[:3], np.float32),
    }
    train, validation = original["parts"]["train-v0"], original["parts"]["val-v0"]
    total_bytes = sum(part["bytes"] for part in parts.values())
    build_seconds = time.perf_counter() - started
    manifest = {
        "version": VERSION,
        "method": f"cache-reuse-c1-int{bits}",
        "request": request,
        "source": evidence,
        "identity": original["identity"],
        "train_images": shape[0],
        "images": shape[0],
        "shape": list(shape),
        "bits": int(bits),
        "quantile": float(quantile),
        "quantile_method": "linear; per image/layer/channel over H,W",
        "quantization": "clip(round((x-lo)/step),0,2**bits-1); constant channels use zero codes and zero step",
        "dequantization": "float32(code)*float32(step)+float32(lo)",
        "packing": "low nibble first, high nibble second along W; odd-W final high nibble zero"
        if bits == 4
        else "uint8",
        "validation": "Original shared uncompressed FP16 validation cache; no validation compression",
        "paper_source": "Cache and Reuse, CVPRW 2024, Sections 4.2/4.3, Equations 1-3, Algorithm 2",
        "adaptation_assumptions": "The paper does not specify quantile k; k=0.001 is the fixed detection choice, not a recovered original setting",
        "teacher_query_images": 0,
        "raw_image_reads": 0,
        "parts": parts,
        "cache_bytes": total_bytes,
        "total_bytes": total_bytes,
        "build_seconds": build_seconds,
        "source_train_bank_sha256": train["sha256"],
        "source_train_bank_bytes": int(train["bytes"]),
        "statistics": {
            "clipped_low_values": clipped_low,
            "clipped_high_values": clipped_high,
            "zero_step_channels": constant_channels,
            "total_values": math.prod(shape),
        },
        "cost": {
            "compression_build_seconds": build_seconds,
            "source_setup_seconds": float(original["source_setup_seconds"]),
            "source_c1_build_seconds": float(original["source_setup_seconds"]) + float(train["seconds"]),
            "source_c1_build_query_images": int(train["images"]) + int(train.get("extra_parity_images", 0)),
            "source_train_cache_bytes": int(train["bytes"]),
            "validation_cache_bytes": int(validation["bytes"]),
            "validation_build_seconds": float(validation["seconds"]),
            "validation_build_query_images": int(validation["images"]) + int(validation.get("extra_parity_images", 0)),
            "note": "Measured historical C1 build charged for cold cost; warm reuse adds no teacher calls. Compression includes output-file hashing. Source FP16 remains preserved on disk but is not read during quantized training.",
        },
    }
    _write(destination / "manifest.json", manifest)
    _write(destination / "complete.json", {"version": VERSION, "manifest_sha256": _sha(destination / "manifest.json")})
    verify_quantized_cache(destination, request=request)
    return destination / "manifest.json"


class QuantizedBank:
    """Read actual compressed codes and count CPU reading/dequantization time.

    ``bank[int]`` returns L,C,H,W; lists/slices return N,L,C,H,W. Counters
    include the compressed read and conversion, are nested within training wall
    time, and must not be added to it a second time.
    """

    def __init__(self, manifest):
        path = Path(manifest).resolve(strict=True)
        self.path = path / "manifest.json" if path.is_dir() else path
        self.manifest = verify_quantized_cache(self.path)
        self.manifest_sha256 = _sha(self.path)
        self.shape = tuple(self.manifest["shape"])
        self.bits = int(self.manifest["bits"])
        self.codes = np.load(self.path.parent / "codes.npy", mmap_mode="r", allow_pickle=False)
        self.lo = np.load(self.path.parent / "lo.npy", mmap_mode="r", allow_pickle=False)
        self.step = np.load(self.path.parent / "step.npy", mmap_mode="r", allow_pickle=False)
        self.dequant_seconds, self.dequant_images = 0.0, 0

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, indices):
        if isinstance(indices, (bool, np.bool_)):
            raise TypeError("Boolean bank indexing is unsupported")
        if isinstance(indices, (list, np.ndarray)):
            indices = np.asarray(indices)
            if indices.size == 0:
                indices = indices.astype(np.int64)
            if indices.ndim != 1 or indices.dtype.kind not in "iu":
                raise TypeError("Bank index arrays must contain one-dimensional integers")
        elif not isinstance(indices, (int, np.integer, slice)):
            raise TypeError("Expected an integer, slice, or one-dimensional integer list/array")
        started = time.perf_counter()
        selected = self.codes[indices]
        if self.bits == 4:
            result = np.empty((*selected.shape[:-1], self.shape[-1]), dtype=np.float32)
            result[..., ::2] = selected & np.uint8(15)
            result[..., 1::2] = selected[..., : self.shape[-1] // 2] >> np.uint8(4)
        else:
            result = selected.astype(np.float32)
        np.multiply(result, self.step[indices][..., None, None], out=result)
        np.add(result, self.lo[indices][..., None, None], out=result)
        self.dequant_seconds += time.perf_counter() - started
        self.dequant_images += 1 if result.ndim == 4 else result.shape[0]
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--bits", type=int, choices=(4, 8), default=8)
    args = parser.parse_args()
    path = build_quantized_cache(args.cache_manifest, args.destination, bits=args.bits)
    print(json.dumps({"manifest": str(path), "teacher_query_images": 0}), flush=True)


if __name__ == "__main__":
    main()
