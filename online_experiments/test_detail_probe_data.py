"""CPU contracts for the paired continuous-geometry detail experiment."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def module():
    assert importlib.util.find_spec("online_experiments.detail_probe_data") is not None, "detail data API is missing"
    return importlib.import_module("online_experiments.detail_probe_data")


def test_anchor_selection_has_explicit_lower_tie_and_uses_absolute_distance():
    m = module()
    assert [m.nearest_anchor(x) for x in (0.85, 0.9, 0.925, 1.0, 1.075, 1.1, 1.15)] == [
        0.85,
        0.85,
        0.85,
        1.0,
        1.0,
        1.15,
        1.15,
    ]
    for x in (0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            m.nearest_anchor(x)


def test_blank_support_preserves_constant_field_including_fractional_boundary():
    m = module()
    values = torch.full((2, 2, 40, 40), 7.0)
    blank = torch.full((1, 2, 40, 40), 7.0)
    filled, support = m.transport_with_blank(values, [0.85, 1.15], blank)
    torch.testing.assert_close(filled, values)
    assert support.shape == (2, 1, 40, 40)
    assert ((support > 0) & (support < 1)).any()
    assert support[0, 0, 0, 0] == 0
    assert support[1].min() > 0.999
    zeroed, _ = m.transport_with_blank(values, [0.85, 1.15], None)
    assert zeroed[0, 0, 0, 0] == 0


def synthetic_batches(monkeypatch, method):
    m = module()
    from online_experiments.full_visdrone_data import epoch_plan

    obj = m.DetailBatches.__new__(m.DetailBatches)
    obj.method, obj.seed, obj.pct, obj.device = method, 9, 25, torch.device("cpu")
    obj.meta = {"identity": {"layers": [3, 7, 11]}, "prepare_seconds": 0.0}
    obj.train = [
        {
            "sample_id": str(i),
            "image_path": str(i),
            "letterbox": {},
            "cls": [[i % 2]],
            "bboxes": [[0.2, 0.35, 0.1, 0.2]],
        }
        for i in range(17)
    ]
    obj.val = obj.train[:3]
    obj.arrays = [np.full((17, 3, 2, 4, 4), 10 + v, dtype=np.float16) for v in range(2)]
    obj.view_arrays = {(0.85, v): np.full((17, 3, 2, 4, 4), 20 + v, dtype=np.float16) for v in range(2)}
    obj.view_arrays.update({(1.15, v): np.full((17, 3, 2, 4, 4), 30 + v, dtype=np.float16) for v in range(2)})
    obj.val_array = obj.arrays[0][:3]
    obj.blank = torch.full((3, 2, 4, 4), 2.0)
    obj.query_images = obj.query_calls = obj.correction_steps = 0
    obj.correction_seconds = 0.0
    obj.reconstruction = []
    obj.corrector = obj.optimizer = None
    seen = []

    def image(path, imgsz):
        ramp = torch.arange(640, dtype=torch.float32)[None, None].expand(3, 640, -1) / 1280
        return ramp + int(path) / 40, {}

    class Teacher:
        model = torch.nn.Linear(1, 1).requires_grad_(False)

        def __call__(self, rgb):
            seen.append(rgb.clone())
            return {
                f"layer_{layer}": rgb.mean((1, 2, 3))[:, None, None, None].expand(-1, 2, 4, 4) for layer in [3, 7, 11]
            }

    obj.extractor = Teacher()
    monkeypatch.setattr(m, "load_letterboxed_tensor", image)
    return obj, seen, epoch_plan


def test_detail_queries_use_exact_same_rgb_and_keep_tail_and_labels(monkeypatch, tmp_path):
    obj, seen, plan = synthetic_batches(monkeypatch, "detail25")
    order, flips, scales, query = plan(17, 25, 0, 9)
    batches = list(obj.training(0, 8, tmp_path))
    assert [len(x["sample_ids"]) for x in batches] == [8, 8, 1]
    assert [i for b in batches for i in b["sample_ids"]] == [str(i) for i in order]
    rgb = torch.cat([b["features"][3] for b in batches])
    exact_rgb = torch.cat(seen)
    torch.testing.assert_close(exact_rgb, rgb[query])
    levels = torch.cat([b["features"][0] for b in batches])
    expected = exact_rgb.mean((1, 2, 3)).half().float()
    torch.testing.assert_close(levels[query, 0, 0, 0], expected)
    assert obj.query_images == 4 and obj.query_calls == len(seen)
    for position, (sample_id, flip, scale) in enumerate(zip(order, flips, scales)):
        target = obj.item(obj.train[sample_id], flip, scale)
        batch, row = divmod(position, 8)
        torch.testing.assert_close(batches[batch]["targets"][row][1], target["bboxes"])


def test_all_arms_have_identical_schedules_and_detail_blank_features(monkeypatch, tmp_path):
    outputs = {}
    schedules = []
    for method in module().METHODS:
        obj, _, _ = synthetic_batches(monkeypatch, method)
        output = tmp_path / method
        outputs[method] = list(obj.training(0, 8, output))
        schedules.append((output / "schedules/001.json").read_bytes())
    assert all(x == schedules[0] for x in schedules)
    for plain, detail, lowpass in zip(outputs["blank25"], outputs["detail25"], outputs["lowpass25"]):
        for k in range(3):
            torch.testing.assert_close(plain["features"][k], detail["features"][k])
            torch.testing.assert_close(detail["features"][k], lowpass["features"][k])
        torch.testing.assert_close(detail["features"][3], lowpass["features"][3])


def test_validation_keeps_original_rgb_and_four_feature_collation(monkeypatch):
    obj, seen, _ = synthetic_batches(monkeypatch, "lowpass25")
    batches = list(obj.validation(2))
    from scripts import d1_train_cached_detector as d

    moved, targets = d.move_batch(batches[0], torch.device("cpu"))
    assert [tuple(t.shape) for t in moved] == [(2, 2, 4, 4)] * 3 + [(2, 3, 640, 640)]
    assert targets["batch_idx"].tolist() == [0, 1]
    assert not seen and obj.query_images == 0
    assert moved[3][0, 0, 0, 0] == 0
    assert batches[-1]["sample_ids"] == ["2"]


def test_costs_charge_old_c2_val_to_every_arm_and_only_used_new_parts():
    m = module()
    manifest = {"source_setup_seconds": 5, "setup_seconds": 2, "parts": {}}
    for key in ("train-v0", "train-v1", "val-v0"):
        manifest["parts"][key] = {"seconds": 10, "bytes": 100, "images": 17, "extra_parity_images": 1}
    for key in ("s085-v0", "s085-v1", "s115-v0", "s115-v1"):
        manifest["parts"][key] = {"seconds": 3, "bytes": 200, "images": 17}
    manifest["parts"]["blank"] = {"seconds": 1, "bytes": 20, "images": 1}
    transport = m.cost_for_method(manifest, "transport25")
    blank = m.cost_for_method(manifest, "blank25")
    views = m.cost_for_method(manifest, "views25")
    assert transport == {"cache_build_charged_seconds": 35.0, "cache_bytes": 300, "cache_build_query_images": 54}
    assert blank == {"cache_build_charged_seconds": 38.0, "cache_bytes": 320, "cache_build_query_images": 55}
    assert views == {"cache_build_charged_seconds": 50.0, "cache_bytes": 1120, "cache_build_query_images": 123}
    assert m.cost_for_method(manifest, "detail25") == blank


def fixture_source(tmp_path, monkeypatch):
    m = module()
    monkeypatch.setattr(m, "FEATURE_SHAPE", (3, 2, 4, 4))
    source = tmp_path / "source"
    (source / "cache").mkdir(parents=True)
    sample = {
        "sample_id": "a",
        "image_path": str(source / "a.jpg"),
        "label_path": str(source / "a.txt"),
        "letterbox": {},
        "cls": [],
        "bboxes": [],
    }
    Path(sample["image_path"]).write_bytes(b"image")
    Path(sample["label_path"]).write_bytes(b"label")
    sample.update(image_sha256=m.sha(sample["image_path"]), label_sha256=m.sha(sample["label_path"]))
    meta = {
        "identity": {"imgsz": 640, "layers": [3, 7, 11]},
        "splits": {"train": [sample], "val": [sample]},
        "prepare_seconds": 3,
        "raw_val_annotation_sha256": {},
    }
    m.dump(source / "dataset-manifest.json", meta)
    cache = {
        "identity": meta["identity"],
        "dataset_manifest_sha256": m.sha(source / "dataset-manifest.json"),
        "setup_seconds": 1,
        "parts": {},
    }
    for key in ("train-v0", "train-v1", "val-v0"):
        path = source / "cache" / f"{key}.npy"
        np.save(path, np.ones((1, 3, 2, 4, 4), dtype=np.float16))
        cache["parts"][key] = {
            "file": str(path),
            "bytes": path.stat().st_size,
            "sha256": m.sha(path),
            "images": 1,
            "seconds": 2,
            "extra_parity_images": 1,
        }
    m.dump(source / "cache/manifest.json", cache)
    monkeypatch.setattr(m.shutil, "disk_usage", lambda _: SimpleNamespace(free=2**40))
    monkeypatch.setattr(m, "load_letterboxed_tensor", lambda *a, **k: (torch.full((3, 640, 640), 0.3), {}))
    calls = []

    class Teacher:
        model = torch.nn.Linear(1, 1).requires_grad_(False)

        def __call__(self, images):
            calls.append(images.clone())
            return {
                f"layer_{layer}": images.mean((1, 2, 3))[:, None, None, None].expand(-1, 2, 4, 4)
                for layer in [3, 7, 11]
            }

    monkeypatch.setattr(m, "make_extractor", lambda _: Teacher())
    return m, source, calls


def test_cache_builder_is_nonoverwriting_hash_verified_and_counts_blank(monkeypatch, tmp_path):
    m, source, calls = fixture_source(tmp_path, monkeypatch)
    dest = tmp_path / "new"
    old_hash = m.sha(source / "cache/manifest.json")
    result = m.prepare_cache(source, dest, batch_size=1, limit=1)
    assert len(calls) == 5 and result["new_query_images"] == 5
    assert set(result["parts"]) == {
        "train-v0",
        "train-v1",
        "val-v0",
        "s085-v0",
        "s085-v1",
        "s115-v0",
        "s115-v1",
        "blank",
    }
    assert m.sha(source / "cache/manifest.json") == old_hash
    torch.testing.assert_close(calls[-1], torch.full((1, 3, 640, 640), 114 / 255))
    assert (dest / "integrity.json").is_file()
    assert m.prepare_cache(source, dest, batch_size=1, limit=1) == result
    assert len(calls) == 5
    with pytest.raises(RuntimeError, match="mismatch"):
        m.prepare_cache(source, dest, batch_size=1, limit=None)
    bank = np.load(dest / "s085-v0.npy", mmap_mode="r+")
    bank[0, 0, 0, 0, 0] = 12
    bank.flush()
    del bank
    with pytest.raises(RuntimeError, match="hash|bytes"):
        m.prepare_cache(source, dest, batch_size=1, limit=1)


def test_partial_build_refused_and_source_corruption_detected(monkeypatch, tmp_path):
    m, source, _ = fixture_source(tmp_path, monkeypatch)
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "s085-v0.partial.npy").write_bytes(b"retain")
    with pytest.raises(RuntimeError, match="partial|nonempty"):
        m.prepare_cache(source, partial, limit=1)
    assert (partial / "s085-v0.partial.npy").read_bytes() == b"retain"
    (source / "a.txt").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="annotation|label"):
        m.prepare_cache(source, tmp_path / "corrupt", limit=1)


def test_queries_keep_exact_cumulative_quota_for_multiple_epochs_and_single_tail(monkeypatch, tmp_path):
    obj, seen, _ = synthetic_batches(monkeypatch, "transport25")
    for epoch in range(4):
        batches = list(obj.training(epoch, 8, tmp_path))
        assert [len(b["sample_ids"]) for b in batches] == [8, 8, 1]
        assert obj.query_images == ((epoch + 1) * 17 * 25) // 100
    assert sum(len(batch) for batch in seen) == 17
    assert obj.query_calls == len(seen)
    assert all(not tensor.requires_grad for batch in batches for tensor in batch["features"])


def test_views_use_real_flip_bank_and_residual_scale_without_changing_requested_labels(monkeypatch, tmp_path):
    m = module()
    obj, seen, _ = synthetic_batches(monkeypatch, "views25")
    scales = [0.9, 1.07, 1.13]
    monkeypatch.setattr(m, "epoch_plan", lambda *args: ([0, 1, 2], [False, True, False], scales, [False] * 3))
    batch = next(obj.training(0, 8, tmp_path))
    # .9 uses the .85 bank and expands it, while 1.07 uses the genuinely flipped 1.0 bank.
    torch.testing.assert_close(batch["features"][0][0], torch.full((2, 4, 4), 20.0))
    torch.testing.assert_close(batch["features"][0][1], torch.full((2, 4, 4), 11.0))
    # 1.13 uses the 1.15 bank; its corner mixes with true blank under the residual contraction.
    edge = 1 - 1.5 * (1.15 / 1.13 - 1)
    assert float(batch["features"][0][2, 0, 0, 0]) == pytest.approx(30 * edge**2 + 2 * (1 - edge**2), abs=2e-5)
    assert not seen
    for row, (scale, flip) in enumerate(zip(scales, [False, True, False])):
        expected = obj.item(obj.train[row], flip, scale)
        torch.testing.assert_close(batch["targets"][row][1], expected["bboxes"])


def test_completed_cache_constructor_and_source_mutations_are_checked(monkeypatch, tmp_path):
    m, source, calls = fixture_source(tmp_path, monkeypatch)
    dest = tmp_path / "new"
    m.prepare_cache(source, dest, batch_size=1, limit=1)
    for method in m.METHODS:
        obj = m.DetailBatches(source, dest / "manifest.json", method, seed=3, train_limit=1, val_limit=1)
        assert len(obj.train) == len(obj.val) == 1
        assert all(not parameter.requires_grad for parameter in obj.extractor.model.parameters())
        batch = next(obj.validation(1))
        assert len(batch["features"]) == (4 if method in ("detail25", "lowpass25") else 3)
    assert len(calls) == 5  # Constructors and validation never spend teacher queries.
    (source / "a.txt").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="changed|mismatch"):
        m.DetailBatches(source, dest / "manifest.json", "transport25", seed=3, train_limit=1)


def test_changed_manifest_is_rejected_and_insufficient_disk_keeps_teacher_unloaded(monkeypatch, tmp_path):
    m, source, calls = fixture_source(tmp_path, monkeypatch)
    dest = tmp_path / "new"
    manifest = m.prepare_cache(source, dest, batch_size=1, limit=1)
    manifest["setup_seconds"] += 1
    m.dump(dest / "manifest.json", manifest)
    with pytest.raises(RuntimeError, match="manifest hash mismatch"):
        m.prepare_cache(source, dest, batch_size=1, limit=1)
    monkeypatch.setattr(m.shutil, "disk_usage", lambda _: SimpleNamespace(free=20 * 2**30))
    with pytest.raises(RuntimeError, match="20 GiB reserve"):
        m.prepare_cache(source, tmp_path / "no-space", batch_size=1, limit=1)
    assert len(calls) == 5


def test_original_validation_cache_unchanged_and_raw_loading_only_for_needed_images(monkeypatch, tmp_path):
    obj, _, _ = synthetic_batches(monkeypatch, "blank25")
    list(obj.training(0, 8, tmp_path))
    assert obj.raw_image_images == 4
    validation = list(obj.validation(2))
    assert obj.raw_image_images == 4
    torch.testing.assert_close(validation[0]["features"][0], torch.full((2, 2, 4, 4), 10.0, dtype=torch.float16))
    branch, _, _ = synthetic_batches(monkeypatch, "detail25")
    list(branch.training(0, 8, tmp_path / "branch"))
    assert branch.raw_image_images == 17
    list(branch.validation(2))
    assert branch.raw_image_images == 20
