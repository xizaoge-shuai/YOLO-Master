"""Contracts for full-split replication, budget ablations, and feature baselines."""

import importlib
import json
from types import SimpleNamespace

import pytest


def module():
    assert importlib.util.find_spec("online_experiments.replication_suite") is not None, "suite runner is missing"
    return importlib.import_module("online_experiments.replication_suite")


@pytest.mark.parametrize(
    "method,pct,detail",
    [
        ("detail25", 25, "rgb"),
        ("lowpass25", 25, "lowpass"),
        ("detail0", 0, "rgb"),
        ("detail10", 10, "rgb"),
        ("online100", 100, "none"),
        ("mix25", 25, "none"),
        ("frofa-c2-d01", 0, "none"),
        ("loffta-c2-n005", 0, "none"),
    ],
)
def test_method_contract_and_cumulative_tail_budget(method, pct, detail):
    m = module()
    assert m.method_spec(method)["query_pct"] == pct
    assert m.method_spec(method)["detail_mode"] == detail
    for epoch in range(1, 5):
        m.check_counts(epoch, 17, 17, epoch * 17 * pct // 100, pct)
    m.check_counts(100, 6471, 6471, 6471 * pct, pct)
    with pytest.raises(RuntimeError):
        m.check_counts(4, 17, 17, 4 * 17 * pct // 100 + 1, pct)


def test_zero_query_and_online_costs_charge_only_consumed_cache_parts():
    m = module()
    parts = {
        name: {"seconds": 3, "bytes": 100, "images": 17}
        for name in ("train-v0", "train-v1", "val-v0", "blank", "s085-v0", "s085-v1", "s115-v0", "s115-v1")
    }
    manifest = {"source_setup_seconds": 5, "setup_seconds": 2, "parts": parts}
    assert m.cost_for_method(manifest, "online100")["cache_bytes"] == 100
    assert m.cost_for_method(manifest, "frofa-c2-d01")["cache_bytes"] == 300
    assert m.cost_for_method(manifest, "detail0")["cache_bytes"] == 400
    assert m.cost_for_method(manifest, "views25")["cache_bytes"] == 800


def synthetic(monkeypatch, method):
    pytest.importorskip("scripts.d1_train_cached_detector")
    import torch

    from online_experiments.test_detail_probe_data import synthetic_batches

    m = module()
    native, seen, plan = synthetic_batches(monkeypatch, "detail25" if method.startswith("detail") else "transport25")
    obj = m.batch_class().__new__(m.batch_class())
    obj.__dict__.update(native.__dict__)
    obj.suite_method = method
    obj.spec = m.method_spec(method)
    obj.method = obj.spec["native_method"] or method
    obj.pct = obj.spec["query_pct"]
    if not obj.pct:
        obj.extractor = None
    if method == "online100":
        obj.arrays = []
    obj.device = torch.device("cpu")
    return obj, seen, plan


def test_mix_pairs_requested_schedule_but_keeps_nonquery_targets_and_cache_unscaled(monkeypatch, tmp_path):
    import torch

    obj, seen, plan = synthetic(monkeypatch, "mix25")
    order, flips, scales, queries = plan(17, 25, 0, obj.seed)
    batches = list(obj.training(0, 8, tmp_path))
    assert [len(batch["sample_ids"]) for batch in batches] == [8, 8, 1]
    features = torch.cat([batch["features"][0] for batch in batches])
    assert obj.query_images == 4 and sum(len(x) for x in seen) == 4
    for pos, index in enumerate(order):
        batch, row = divmod(pos, 8)
        expected = obj.item(obj.train[index], flips[pos], scales[pos] if queries[pos] else 1.0)
        torch.testing.assert_close(batches[batch]["targets"][row][1], expected["bboxes"])
        if not queries[pos]:
            torch.testing.assert_close(features[pos], torch.full_like(features[pos], 10 + int(flips[pos])))


@pytest.mark.parametrize("method", ["online100", "detail0", "detail10"])
def test_query_ablations_keep_geometry_and_exact_cumulative_quota(monkeypatch, tmp_path, method):
    obj, seen, plan = synthetic(monkeypatch, method)
    for epoch in range(4):
        batches = list(obj.training(epoch, 8, tmp_path))
        assert sum(len(batch["sample_ids"]) for batch in batches) == 17
        assert obj.query_images == (epoch + 1) * 17 * obj.pct // 100
        schedule = json.loads((tmp_path / "schedules" / f"{epoch + 1:03d}.json").read_text())
        order, flip, scale, query = plan(17, obj.pct, epoch, obj.seed)
        assert schedule == {"epoch": epoch + 1, "order": order, "flip": flip, "scale": scale, "query": query}
    assert sum(len(x) for x in seen) == 4 * 17 * obj.pct // 100


@pytest.mark.parametrize(
    "method,operation,magnitude", [("frofa-c2-d01", "frofa", 0.1), ("loffta-c2-n005", "add_noise", 0.05)]
)
def test_feature_baselines_use_documented_operators_and_are_epoch_reproducible(
    monkeypatch, tmp_path, method, operation, magnitude
):
    import torch

    from online_experiments import feature_baselines as f

    obj, seen, plan = synthetic(monkeypatch, method)
    # A nonconstant feature makes both brightness and std-scaled noise observable.
    for bank in obj.arrays:
        bank[..., 0, 0] += 2
    first = list(obj.training(0, 8, tmp_path))
    second = list(obj.training(0, 8, tmp_path))
    assert not seen and obj.extractor is None and obj.query_images == 0
    for left, right in zip(first, second):
        torch.testing.assert_close(left["features"][0], right["features"][0], rtol=0, atol=0)
    order, flips, scales, _ = plan(17, 0, 0, obj.seed)
    values = torch.stack(
        [torch.from_numpy(obj.arrays[int(flip)][index]).float() for index, flip in zip(order[:8], flips[:8])]
    )
    generator = torch.Generator(device="cpu").manual_seed(obj.seed + 49979687)
    for level in range(3):
        reference = values[:, level]
        if method == "loffta-c2-n005":
            reference = f.warp_features(reference, [False] * 8, scales[:8])
        expected = getattr(f, operation)(reference, generator, magnitude)
        torch.testing.assert_close(first[0]["features"][level], expected)
    for pos, index in enumerate(order):
        batch, row = divmod(pos, 8)
        expected = obj.item(obj.train[index], flips[pos], scales[pos] if method.startswith("loffta") else 1.0)
        torch.testing.assert_close(first[batch]["targets"][row][1], expected["bboxes"])


def test_completion_rejects_wrong_seed_and_wrong_budget(tmp_path):
    m = module()
    schedules = tmp_path / "schedules"
    schedules.mkdir()
    path = schedules / "001.json"
    path.write_text("{}")
    args = {
        "method": "detail0",
        "seed": 1,
        "epochs": 1,
        "train_images": 17,
        "val_images": 8,
        "batch": 8,
        "query_pct": 0,
        "detail_mode": "rgb",
        "provenance_sha256": "snapshot",
        "dataset_manifest_sha256": "dataset",
        "cache_manifest_sha256": "cache",
    }
    summary = {**args, "completed_epochs": 1, "query_images": 0, "schedule_sha256": {path.name: m.sha(path)}}
    for name, value in (("args.json", args), ("summary.json", summary), ("metrics.json", [{"epoch": 1}])):
        (tmp_path / name).write_text(json.dumps(value))
    m.validate_completed(tmp_path, True, method="detail0", seed=1)
    with pytest.raises(RuntimeError, match="identity"):
        m.validate_completed(tmp_path, True, method="detail0", seed=2)
    summary["query_images"] = 4
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(RuntimeError, match="budget"):
        m.validate_completed(tmp_path, True)


def completed_run(tmp_path, *, smoke=True, batch=8):
    """Build complete metadata independently of training or GPU dependencies."""
    m = module()
    source, job, output = (tmp_path / name for name in ("source", "job", "output"))
    source.mkdir()
    job.mkdir()
    output.mkdir()
    cache, protocol = tmp_path / "cache.json", tmp_path / "protocol.md"
    m.dump(source / "dataset-manifest.json", {"dataset": "current"})
    m.dump(cache, {"cache": "current"})
    protocol.write_text("fixed protocol")
    m.dump(
        job / "provenance.json",
        {
            "schema": "replication-suite-v1",
            "methods": {name: m.method_spec(name) for name in m.METHODS},
            "source_sha256": {},
            "source_job": str(source.resolve()),
            "cache_manifest": str(cache.resolve()),
            "dataset_manifest_sha256": m.sha(source / "dataset-manifest.json"),
            "cache_manifest_sha256": m.sha(cache),
            "protocol_file": str(protocol.resolve()),
            "protocol_sha256": m.sha(protocol),
        },
    )
    epochs, n, nv = (1, 17, 8) if smoke else (100, 6471, 548)
    args = {
        "method": "detail0",
        "seed": 1,
        "epochs": epochs,
        "batch": batch,
        "train_images": n,
        "val_images": nv,
        "query_pct": 0,
        "detail_mode": "rgb",
        "provenance_sha256": m.sha(job / "provenance.json"),
        "dataset_manifest_sha256": m.sha(source / "dataset-manifest.json"),
        "cache_manifest_sha256": m.sha(cache),
    }
    schedules = {}
    for epoch in range(1, epochs + 1):
        path = output / "schedules" / f"{epoch:03d}.json"
        m.dump(path, {"epoch": epoch, "order": [], "flip": [], "scale": [], "query": []})
        schedules[path.name] = m.sha(path)
    summary = {**args, "completed_epochs": epochs, "query_images": 0, "schedule_sha256": schedules}
    if not smoke:
        weights, predictions = output / "weights/last.pt", output / "predictions/last"
        weights.parent.mkdir()
        weights.write_bytes(b"checkpoint fixture")
        predictions.mkdir(parents=True)
        prediction_hashes = {}
        for index in range(nv):
            path = predictions / f"image{index:03d}.txt"
            path.write_text("")
            prediction_hashes[path.stem] = m.sha(path)
        m.dump(
            output / "official-final.json",
            {
                "prediction_sha256": prediction_hashes,
                "checkpoint_sha256": m.sha(weights),
                "metrics": {},
            },
        )
        summary["official_final"] = {}
    for name, value in (
        ("args.json", args),
        ("summary.json", summary),
        ("metrics.json", [{"epoch": epoch} for epoch in range(1, epochs + 1)]),
    ):
        m.dump(output / name, value)
    (output / "DONE").write_text("SUCCEEDED\n")
    return m, SimpleNamespace(
        source=source, job=job, cache=cache, output=output, method="detail0", seed=1, batch=batch, smoke=smoke
    )


def test_full_completion_rejects_nonprotocol_batch(tmp_path):
    m, request = completed_run(tmp_path, smoke=False, batch=16)
    with pytest.raises(RuntimeError, match="batch"):
        m.validate_completed(request.output, False, request.method, request.seed)


def test_full_cli_rejects_nonprotocol_batch_before_snapshot(monkeypatch, tmp_path):
    m = module()
    monkeypatch.setattr(m, "snapshot", lambda _: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "replication_suite",
            "snapshot",
            "--job",
            str(tmp_path / "job"),
            "--source",
            str(tmp_path / "source"),
            "--cache",
            str(tmp_path / "cache"),
            "--batch",
            "16",
        ],
    )
    with pytest.raises(SystemExit) as error:
        m.main()
    assert error.value.code == 2


def test_smoke_cli_still_accepts_positive_alternative_batch(monkeypatch, tmp_path):
    m = module()
    observed = []
    monkeypatch.setattr(m, "snapshot", lambda args: observed.append(args.batch))
    monkeypatch.setattr(
        "sys.argv",
        [
            "replication_suite",
            "snapshot",
            "--job",
            str(tmp_path / "job"),
            "--source",
            str(tmp_path / "source"),
            "--cache",
            str(tmp_path / "cache"),
            "--smoke",
            "--epochs",
            "1",
            "--batch",
            "4",
        ],
    )
    m.main()
    assert observed == [4]


@pytest.mark.parametrize("field", ["provenance_sha256", "dataset_manifest_sha256", "cache_manifest_sha256"])
def test_validation_cli_rejects_completed_run_from_another_snapshot(monkeypatch, tmp_path, field):
    m, request = completed_run(tmp_path)
    argv = [
        "replication_suite",
        "validate",
        "--job",
        str(request.job),
        "--source",
        str(request.source),
        "--cache",
        str(request.cache),
        "--output",
        str(request.output),
        "--method",
        request.method,
        "--seed",
        str(request.seed),
        "--smoke",
        "--epochs",
        "1",
    ]
    monkeypatch.setattr("sys.argv", argv)
    m.main()  # Matching current evidence must be accepted before the identity changes.
    for name in ("args.json", "summary.json"):
        path = request.output / name
        evidence = m.read(path)
        evidence[field] = "another snapshot"
        m.dump(path, evidence)
    with pytest.raises(RuntimeError, match=field):
        m.main()


def test_completion_rejects_summary_identity_drift(tmp_path):
    m, request = completed_run(tmp_path)
    summary = m.read(request.output / "summary.json")
    summary["provenance_sha256"] = "another snapshot"
    m.dump(request.output / "summary.json", summary)
    with pytest.raises(RuntimeError, match="provenance_sha256"):
        m.validate_completed(request.output, True, request.method, request.seed)


def test_done_skip_rejects_completed_run_from_another_snapshot(monkeypatch, tmp_path):
    import builtins

    m, request = completed_run(tmp_path)
    # A DONE skip only uses metadata; isolate the unrelated training imports.
    imports = {
        "resource": SimpleNamespace(),
        "torch": SimpleNamespace(),
        "online_experiments.assignment_intervention": SimpleNamespace(state_hash=None),
        "online_experiments.detail_probe_model": SimpleNamespace(DetailDetector=None),
        "online_experiments.full_visdrone_train": SimpleNamespace(
            atomic_checkpoint=None, save_predictions=None, validate=None
        ),
        "online_experiments.fusion_probe": SimpleNamespace(configure_fusion=None),
        "scripts": SimpleNamespace(d1_train_cached_detector=None),
    }
    original_import = builtins.__import__

    def metadata_only_import(name, *args, **kwargs):
        return imports[name] if name in imports else original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", metadata_only_import)
    m.train(request)  # Matching DONE evidence must still skip without training.
    for name in ("args.json", "summary.json"):
        path = request.output / name
        evidence = m.read(path)
        evidence["provenance_sha256"] = "another snapshot"
        m.dump(path, evidence)
    with pytest.raises(RuntimeError, match="provenance_sha256"):
        m.train(request)
