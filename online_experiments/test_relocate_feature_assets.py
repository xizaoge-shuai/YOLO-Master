"""CPU contracts for relocating metadata while retaining immutable cache bytes."""

import copy
import importlib
from pathlib import Path

import numpy as np
import pytest


def module():
    assert importlib.util.find_spec("online_experiments.relocate_feature_assets") is not None, "migration API missing"
    return importlib.import_module("online_experiments.relocate_feature_assets")


def fixture_assets(tmp_path, monkeypatch):
    from online_experiments import detail_probe_data as data

    monkeypatch.setattr(data, "FEATURE_SHAPE", (3, 2, 4, 4))
    old_prefix, new_prefix = tmp_path / "legacy", tmp_path / "mounted"
    source, cache = new_prefix / "runs/source", new_prefix / "runs/detail/cache"
    (source / "cache").mkdir(parents=True)
    (source / "raw-val-annotations").mkdir()
    cache.mkdir(parents=True)

    def historical(path):
        return str(old_prefix / path.relative_to(new_prefix))

    splits = {"train": [], "val": []}
    for split, count in (("train", 13), ("val", 2)):
        for i in range(count):
            image, label = source / f"{split}-{i}.jpg", source / f"{split}-{i}.txt"
            image.write_bytes(f"image {split} {i}".encode())
            label.write_bytes(b"0 .5 .5 .1 .1\n")
            splits[split].append(
                {
                    "sample_id": f"{split}-{i}",
                    "image_path": historical(image),
                    "label_path": historical(label),
                    "image_sha256": data.sha(image),
                    "label_sha256": data.sha(label),
                    "letterbox": {"original_shape": [400, 600], "pad": [0, 107]},
                    "cls": [[0]],
                    "bboxes": [[0.5, 0.5, 0.1, 0.1]],
                }
            )
    annotation = source / "raw-val-annotations/val-0.txt"
    annotation.write_bytes(b"raw ignored region\n")
    meta = {
        "identity": {"imgsz": 640, "layers": [3, 7, 11]},
        "splits": splits,
        "raw_val_annotation_sha256": {annotation.name: data.sha(annotation)},
        "prepare_seconds": 9.5,
        "notes": "preserve this text",
        "nested": [historical(source / "train-0.jpg"), 3],
    }
    data.dump(source / "dataset-manifest.json", meta)
    source_cache = {
        "identity": meta["identity"],
        "dataset_manifest_sha256": data.sha(source / "dataset-manifest.json"),
        "setup_seconds": 2.5,
        "parts": {},
    }
    for key in data.SOURCE_PARTS:
        count = len(splits[key.split("-")[0]])
        path = source / "cache" / f"{key}.npy"
        np.save(path, np.ones((count, *data.FEATURE_SHAPE), dtype=np.float16))
        source_cache["parts"][key] = {
            "file": historical(path),
            "bytes": path.stat().st_size,
            "sha256": data.sha(path),
            "images": count,
            "seconds": 4.5,
            "extra_parity_images": 1,
        }
    data.dump(source / "cache/manifest.json", source_cache)
    files = {}
    for path in source.rglob("*"):
        if path.is_file() and path.name not in ("manifest.json", "dataset-manifest.json"):
            files[historical(path)] = {"sha256": data.sha(path), **data._stamp(path)}
    integrity = {
        "source": historical(source),
        "dataset_manifest_sha256": data.sha(source / "dataset-manifest.json"),
        "source_cache_manifest_sha256": data.sha(source / "cache/manifest.json"),
        "files": files,
        "seconds": 1,
    }
    data.dump(cache / "integrity.json", integrity)
    detail = {
        "version": data.VERSION,
        "request": {"source": historical(source), "limit": None, "batch_size": 8},
        "identity": meta["identity"],
        "dataset_manifest_sha256": integrity["dataset_manifest_sha256"],
        "source_cache_manifest_sha256": integrity["source_cache_manifest_sha256"],
        "integrity_sha256": data.sha(cache / "integrity.json"),
        "train_images": 13,
        "val_images": 2,
        "source_setup_seconds": 2.5,
        "setup_seconds": 3.5,
        "parts": copy.deepcopy(source_cache["parts"]),
        "new_query_images": 53,
        "new_query_calls": 9,
        "wall_seconds": 40.5,
    }
    for key in (*data.NEW_VIEWS, "blank"):
        count = 1 if key == "blank" else 13
        path = cache / f"{key}.npy"
        np.save(path, np.full((count, *data.FEATURE_SHAPE), 2, dtype=np.float16))
        detail["parts"][key] = {
            "file": historical(path),
            "bytes": path.stat().st_size,
            "sha256": data.sha(path),
            "images": count,
            "seconds": 6.5,
            "stat": data._stamp(path),
        }
    data.dump(cache / "manifest.json", detail)
    data.dump(cache / "complete.json", {"manifest_sha256": data.sha(cache / "manifest.json")})
    monkeypatch.setattr(data, "make_extractor", lambda *_: pytest.fail("migration loaded teacher"))
    return data, source, cache, old_prefix, new_prefix


def test_relocation_preserves_semantics_costs_and_old_bytes_and_passes_real_checker(tmp_path, monkeypatch):
    data, source, cache, old_prefix, new_prefix = fixture_assets(tmp_path, monkeypatch)
    snapshots = {
        p: (p.read_bytes(), data._stamp(p)) for root in (source, cache) for p in root.rglob("*") if p.is_file()
    }
    destination = new_prefix / "runs/new/assets"
    result = module().relocate_assets(source, cache, destination, old_prefix=old_prefix, new_prefix=new_prefix)
    new_source, new_manifest = Path(result["source_job"]), Path(result["cache_manifest"])
    meta, original = data.read(new_source / "dataset-manifest.json"), data.read(source / "dataset-manifest.json")
    for split in original["splits"]:
        for old_sample, sample in zip(original["splits"][split], meta["splits"][split]):
            assert {k: v for k, v in sample.items() if not k.endswith("_path")} == {
                k: v for k, v in old_sample.items() if not k.endswith("_path")
            }
            assert Path(sample["image_path"]).is_file() and Path(sample["label_path"]).is_file()
    assert meta["nested"][0] == str(source / "train-0.jpg")
    assert meta["notes"] == original["notes"]
    manifest = data._manifest_check(new_source, new_manifest.parent, full_hash=True)
    previous = data.read(cache / "manifest.json")
    assert manifest["request"]["source"] == str(new_source)
    for method in data.METHODS:
        assert data.cost_for_method(manifest, method) == data.cost_for_method(previous, method)
    for key in data.SOURCE_PARTS:
        link = new_source / "cache" / f"{key}.npy"
        assert link.is_symlink() and np.load(link, mmap_mode="r").shape[0] == (2 if key == "val-v0" else 13)
    for key in (*data.NEW_VIEWS, "blank"):
        assert Path(manifest["parts"][key]["file"]).is_symlink()
        assert Path(manifest["parts"][key]["file"]).resolve() == cache / f"{key}.npy"
    assert (new_source / "raw-val-annotations").is_symlink()
    assert all(p.read_bytes() == saved and data._stamp(p) == stamp for p, (saved, stamp) in snapshots.items())
    assert result == module().relocate_assets(source, cache, destination, old_prefix=old_prefix, new_prefix=new_prefix)


def test_mapper_rejects_traversal_and_symlink_escape(tmp_path):
    m = module()
    old_prefix, new_prefix = tmp_path / "legacy", tmp_path / "mounted"
    new_prefix.mkdir()
    with pytest.raises(ValueError, match="outside|escape|prefix"):
        m.map_path(str(old_prefix / ".." / "escape"), old_prefix, new_prefix)
    with pytest.raises(ValueError, match="outside|escape|prefix"):
        m.map_path(str(tmp_path / "unrelated"), old_prefix, new_prefix)
    outside = tmp_path / "outside"
    outside.mkdir()
    (new_prefix / "alias").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="outside|escape|prefix"):
        m.map_path(str(old_prefix / "alias" / "file"), old_prefix, new_prefix)


def test_existing_partial_is_retained_and_changed_input_cannot_be_sealed(tmp_path, monkeypatch):
    _, source, cache, old_prefix, new_prefix = fixture_assets(tmp_path, monkeypatch)
    destination = new_prefix / "runs/partial/assets"
    destination.mkdir(parents=True)
    (destination / "retain.txt").write_bytes(b"retain")
    with pytest.raises(RuntimeError, match="partial|nonempty|overwrite"):
        module().relocate_assets(source, cache, destination, old_prefix=old_prefix, new_prefix=new_prefix)
    assert (destination / "retain.txt").read_bytes() == b"retain"
    (source / "train-0.txt").write_bytes(b"corrupt label")
    destination = new_prefix / "runs/corrupt/assets"
    with pytest.raises(RuntimeError, match="hash mismatch"):
        module().relocate_assets(source, cache, destination, old_prefix=old_prefix, new_prefix=new_prefix)
    assert not (destination / "complete.json").exists()


def test_reuse_rejects_changed_bank_or_lineage(tmp_path, monkeypatch):
    data, source, cache, old_prefix, new_prefix = fixture_assets(tmp_path, monkeypatch)
    destination = new_prefix / "runs/new/assets"
    m = module()
    m.relocate_assets(source, cache, destination, old_prefix=old_prefix, new_prefix=new_prefix)
    bank = np.load(cache / "s085-v0.npy", mmap_mode="r+")
    bank[0, 0, 0, 0, 0] = 7
    bank.flush()
    del bank
    with pytest.raises(RuntimeError, match="changed|identity|hash"):
        m.relocate_assets(source, cache, destination, old_prefix=old_prefix, new_prefix=new_prefix)
    assert data.read(cache / "manifest.json")["parts"]["s085-v0"]["sha256"] != data.sha(cache / "s085-v0.npy")
