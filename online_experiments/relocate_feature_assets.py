"""Create verified shadow metadata after a mount move; never rebuild or modify old assets."""

import argparse
import copy
import getpass
import json
import time
from pathlib import Path

OLD_PREFIX = Path("/data/users") / getpass.getuser()
NEW_PREFIX = Path("/mnt/data/users") / getpass.getuser()
VERSION = "feature-assets-relocation-v1"


def map_path(value, old_prefix=OLD_PREFIX, new_prefix=NEW_PREFIX):
    """Map a path inside either named user root, rejecting traversal and symlink escapes."""
    path, old_root, new_root = Path(value), Path(old_prefix).absolute(), Path(new_prefix).resolve()
    if not path.is_absolute():
        raise ValueError(f"Expected absolute path within the configured prefix: {value}")
    try:
        relative = path.relative_to(old_root)
    except ValueError:
        try:
            relative = path.relative_to(new_root)
        except ValueError as exc:
            raise ValueError(f"Path outside configured prefixes: {value}") from exc
    mapped = (new_root / relative).resolve()
    if not mapped.is_relative_to(new_root):
        raise ValueError(f"Path escapes the configured user root: {value}")
    return mapped


def _rewrite(value, old_prefix, new_prefix, count):
    if isinstance(value, dict):
        return {key: _rewrite(item, old_prefix, new_prefix, count) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite(item, old_prefix, new_prefix, count) for item in value]
    if isinstance(value, str):
        path = Path(value)
        if path.is_absolute() and path.is_relative_to(Path(old_prefix).absolute()):
            count[0] += 1
            return str(map_path(value, old_prefix, new_prefix))
    return value


def _check_originals(records, data):
    for path, expected in records.items():
        if data.sha(path) != expected["sha256"] or data._stamp(path) != expected["stat"]:
            raise RuntimeError(f"Immutable old metadata changed: {path}")


def _check_links(links):
    for link, target in links.items():
        path = Path(link)
        if not path.is_symlink() or path.resolve() != Path(target):
            raise RuntimeError(f"Relocated asset symlink changed: {link}")


def relocate_assets(old_source, old_cache, destination, old_prefix=OLD_PREFIX, new_prefix=NEW_PREFIX):
    """Return source_job/cache_manifest paths for a fully verified, immutable relocation.

    The first call streams SHA256 over existing source banks, images, labels,
    raw annotations and extra banks. It writes only beneath destination.
    Completed reuse checks sealed metadata and current target file identities;
    an interrupted nonempty destination is retained and must not be overwritten.
    """
    from online_experiments import detail_probe_data as data

    started = time.perf_counter()
    old_prefix, new_prefix = Path(old_prefix).absolute(), Path(new_prefix).resolve()
    old_source = map_path(old_source, old_prefix, new_prefix)
    old_cache = map_path(old_cache, old_prefix, new_prefix)
    destination = map_path(destination, old_prefix, new_prefix)
    for original in (old_source, old_cache):
        if destination.is_relative_to(original) or original.is_relative_to(destination):
            raise ValueError("Destination must be separate from the immutable source and cache")
    request = {
        "old_source": str(old_source),
        "old_cache": str(old_cache),
        "old_prefix": str(old_prefix),
        "new_prefix": str(new_prefix),
    }
    seal_path = destination / "complete.json"
    if destination.exists() and any(destination.iterdir()):
        if not seal_path.is_file():
            raise RuntimeError("Refuse overwrite of nonempty/partial relocation; choose a new destination")
        seal = data.read(seal_path)
        lineage_path = destination / "relocation.json"
        if seal.get("version") != VERSION or seal.get("request") != request:
            raise RuntimeError("Completed relocation request mismatch")
        if data.sha(lineage_path) != seal["lineage_sha256"]:
            raise RuntimeError("Relocation lineage hash mismatch")
        lineage = data.read(lineage_path)
        _check_originals(lineage["old_metadata"], data)
        _check_links(lineage["symlinks"])
        result = seal["result"]
        data._manifest_check(Path(result["source_job"]), Path(result["cache_manifest"]).parent)
        if data.sha(result["cache_manifest"]) != seal["cache_manifest_sha256"]:
            raise RuntimeError("Relocated cache manifest hash mismatch")
        return result

    metadata_paths = [
        old_source / "dataset-manifest.json",
        old_source / "cache/manifest.json",
        old_cache / "manifest.json",
        old_cache / "integrity.json",
        old_cache / "complete.json",
    ]
    old_metadata = {str(path): {"sha256": data.sha(path), "stat": data._stamp(path)} for path in metadata_paths}
    meta, source_cache = data._source_metadata(old_source)
    previous = data.read(old_cache / "manifest.json")
    previous_integrity = data.read(old_cache / "integrity.json")
    expected_parts = set(data.SOURCE_PARTS) | set(data.NEW_VIEWS) | {"blank"}
    if previous.get("version") != data.VERSION or set(previous.get("parts", {})) != expected_parts:
        raise RuntimeError("Original detail cache version/parts mismatch")
    if data.read(old_cache / "complete.json")["manifest_sha256"] != data.sha(old_cache / "manifest.json"):
        raise RuntimeError("Original cache completion seal mismatch")
    if previous["integrity_sha256"] != data.sha(old_cache / "integrity.json"):
        raise RuntimeError("Original source integrity record hash mismatch")
    if map_path(previous["request"]["source"], old_prefix, new_prefix) != old_source:
        raise RuntimeError("Original detail request source mismatch")
    if previous["identity"] != meta["identity"]:
        raise RuntimeError("Original teacher identity mismatch")
    if previous["dataset_manifest_sha256"] != data.sha(old_source / "dataset-manifest.json"):
        raise RuntimeError("Original dataset manifest hash mismatch")
    if previous["source_cache_manifest_sha256"] != data.sha(old_source / "cache/manifest.json"):
        raise RuntimeError("Original source cache manifest hash mismatch")
    if previous_integrity["dataset_manifest_sha256"] != previous["dataset_manifest_sha256"]:
        raise RuntimeError("Original integrity dataset manifest mismatch")
    if previous_integrity["source_cache_manifest_sha256"] != previous["source_cache_manifest_sha256"]:
        raise RuntimeError("Original integrity source cache manifest mismatch")
    if map_path(previous_integrity["source"], old_prefix, new_prefix) != old_source:
        raise RuntimeError("Original integrity source mismatch")
    for key in data.SOURCE_PARTS:
        if previous["parts"][key] != source_cache["parts"][key]:
            raise RuntimeError(f"Original source part mismatch: {key}")
    for name in meta["raw_val_annotation_sha256"]:
        if Path(name).name != name or name in (".", ".."):
            raise ValueError(f"Unsafe raw annotation filename: {name}")
    for samples in meta["splits"].values():
        for sample in samples:
            for kind in ("image", "label"):
                map_path(sample[f"{kind}_path"], old_prefix, new_prefix)
    # Validate every old integrity filename before reading any large bank.
    mapped_integrity = {
        str(map_path(path, old_prefix, new_prefix)): record for path, record in previous_integrity["files"].items()
    }
    if len(mapped_integrity) != len(previous_integrity["files"]):
        raise RuntimeError("Relocated integrity paths collide")

    new_source, new_cache = destination / "source", destination / "cache"
    (new_source / "cache").mkdir(parents=True, exist_ok=False)
    new_cache.mkdir()
    mapped_count, symlinks = [0], {}

    def link(target, path, directory=False):
        target = map_path(target, old_prefix, new_prefix)
        if not (target.is_dir() if directory else target.is_file()):
            raise RuntimeError(f"Missing original asset: {target}")
        path.symlink_to(target, target_is_directory=directory)
        symlinks[str(path)] = str(target)

    new_meta = _rewrite(meta, old_prefix, new_prefix, mapped_count)
    data.dump(new_source / "dataset-manifest.json", new_meta)
    new_source_cache = _rewrite(source_cache, old_prefix, new_prefix, mapped_count)
    new_source_cache["dataset_manifest_sha256"] = data.sha(new_source / "dataset-manifest.json")
    for key in data.SOURCE_PARTS:
        part = new_source_cache["parts"][key]
        target = map_path(part["file"], old_prefix, new_prefix)
        if target != old_source / "cache" / f"{key}.npy":
            raise RuntimeError(f"Original source bank location mismatch: {key}")
        part["file"], part["stat"] = str(target), data._stamp(target)
        link(target, new_source / "cache" / f"{key}.npy")
    data.dump(new_source / "cache/manifest.json", new_source_cache)
    link(old_source / "raw-val-annotations", new_source / "raw-val-annotations", directory=True)
    for name in new_meta["raw_val_annotation_sha256"]:
        map_path(new_source / "raw-val-annotations" / name, old_prefix, new_prefix)

    # This verifier hashes the three source banks and all dataset inputs exactly once.
    integrity = data._verify_source(new_source, new_meta, new_source_cache, new_cache)
    for path, expected in mapped_integrity.items():
        actual = integrity["files"].get(path)
        if actual is None:
            digest = data.sha(path)
            actual = {"sha256": digest, **data._stamp(path)}
            integrity["files"][path] = actual
        if actual["sha256"] != expected["sha256"]:
            raise RuntimeError(f"Original integrity file hash mismatch: {path}")
    for record in integrity["files"].values():
        record["actual_sha256"], record["matches_expected"] = record["sha256"], True
    data.dump(new_cache / "integrity.json", integrity)

    manifest = _rewrite(copy.deepcopy(previous), old_prefix, new_prefix, mapped_count)
    manifest["request"]["source"] = str(new_source)
    manifest["dataset_manifest_sha256"] = data.sha(new_source / "dataset-manifest.json")
    manifest["source_cache_manifest_sha256"] = data.sha(new_source / "cache/manifest.json")
    manifest["integrity_sha256"] = data.sha(new_cache / "integrity.json")
    for key in data.SOURCE_PARTS:
        manifest["parts"][key] = copy.deepcopy(new_source_cache["parts"][key])
    banks = {}
    for key in (*data.NEW_VIEWS, "blank"):
        part = manifest["parts"][key]
        target = map_path(part["file"], old_prefix, new_prefix)
        if target != old_cache / f"{key}.npy":
            raise RuntimeError(f"Original extra bank location mismatch: {key}")
        path = new_cache / f"{key}.npy"
        link(target, path)
        part["file"], part["stat"] = str(path), data._stamp(path)
        shape = (1 if key == "blank" else manifest["train_images"], *data.FEATURE_SHAPE)
        data._verify_part(part, shape, full_hash=True)
        banks[key] = {
            "expected_sha256": part["sha256"],
            "actual_sha256": part["sha256"],
            "matches_expected": True,
            "target": str(target),
            "stat": part["stat"],
        }
    _check_originals(old_metadata, data)
    _check_links(symlinks)
    lineage = {
        "version": VERSION,
        "request": request,
        "old_metadata": old_metadata,
        "symlinks": symlinks,
        "mapped_path_values": mapped_count[0],
        "verified_source_files": len(integrity["files"]),
        "verified_extra_banks": banks,
        "seconds": time.perf_counter() - started,
        "dataset_manifest": {
            "old_sha256": old_metadata[str(old_source / "dataset-manifest.json")]["sha256"],
            "new_sha256": manifest["dataset_manifest_sha256"],
        },
        "source_cache_manifest": {
            "old_sha256": old_metadata[str(old_source / "cache/manifest.json")]["sha256"],
            "new_sha256": manifest["source_cache_manifest_sha256"],
        },
        "verification": "Full SHA256 of source banks, dataset inputs and five extra banks; no teacher queries",
        "old_metadata_unchanged": True,
        "cache_construction_costs_preserved": True,
    }
    data.dump(destination / "relocation.json", lineage)
    # Publish only after all original bytes, mappings and five bank hashes passed.
    data.dump(new_cache / "manifest.json", manifest)
    data.dump(new_cache / "complete.json", {"manifest_sha256": data.sha(new_cache / "manifest.json")})
    data._manifest_check(new_source, new_cache, full_hash=False)
    result = {
        "source_job": str(new_source),
        "cache_manifest": str(new_cache / "manifest.json"),
        "lineage": str(destination / "relocation.json"),
    }
    data.dump(
        seal_path,
        {
            "version": VERSION,
            "request": request,
            "result": result,
            "lineage_sha256": data.sha(destination / "relocation.json"),
            "cache_manifest_sha256": data.sha(new_cache / "manifest.json"),
        },
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-source", required=True, type=Path)
    parser.add_argument("--old-cache", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--old-prefix", default=OLD_PREFIX)
    parser.add_argument("--new-prefix", default=NEW_PREFIX)
    args = parser.parse_args()
    print(json.dumps(relocate_assets(**vars(args)), indent=2), flush=True)


if __name__ == "__main__":
    main()
