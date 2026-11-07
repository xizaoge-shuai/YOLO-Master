"""Contracts for the fixed experiment queue and publishable result metadata."""

import json

import pytest

from online_experiments.replication_campaign import asset_destination, check_pairing, plan, sanitized


def test_queue_has_twenty_unique_runs_and_explicit_paper_baseline_repeats():
    queue = plan()
    assert len(queue) == len({case["name"] for case in queue}) == 20
    for method in ("frofa-c2-d01", "loffta-c2-n005"):
        assert {case["seed"] for case in queue if case["method"] == method} == {0, 1, 2}
    for method in ("blank25", "detail25", "lowpass25", "transport25", "views25"):
        assert {case["seed"] for case in queue if case["method"] == method} == {1, 2}
    assert {case["method"] for case in queue if case["method"].endswith("0")} == {"blank0", "detail0", "online100"}


def test_sanitization_preserves_measurements_and_relative_source_names():
    from online_experiments.replication_campaign import ROOT

    value = {str(ROOT / "runs/example.json"): {"AP": 12.3453, "source": "online_experiments/replication_suite.py"}}
    assert sanitized(value) == {
        "repo:/runs/example.json": {"AP": 12.3453, "source": "online_experiments/replication_suite.py"}
    }


def test_interrupted_asset_metadata_is_preserved_and_retried_elsewhere(tmp_path):
    partial = tmp_path / "assets"
    partial.mkdir()
    (partial / "evidence.json").write_text("old")
    result = asset_destination(tmp_path)
    assert result != partial and result.parent == tmp_path
    assert (partial / "evidence.json").read_text() == "old"
    assert asset_destination(tmp_path) == result
    (tmp_path / "provenance.json").write_text(json.dumps({"source_job": str(result / "source")}))
    assert asset_destination(tmp_path) == result


def test_pairing_rejects_changed_head_geometry_and_same_budget_masks(tmp_path):
    folders = []
    for name, pct, mask in (("blank", 25, "q25"), ("rgb", 25, "q25"), ("rgb0", 0, "q0")):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "args.json").write_text(json.dumps({"seed": 1, "train_images": 17, "epochs": 1, "query_pct": pct}))
        (folder / "summary.json").write_text(
            json.dumps({"geometry_schedule_sha256": {"001": "geom"}, "schedule_sha256": {"001": mask}})
        )
        (folder / "initialization.json").write_text(json.dumps({"head_state_sha256": "head"}))
        folders.append(folder)
    assert check_pairing(folders)["paired"]
    changed = folders[1] / "summary.json"
    record = json.loads(changed.read_text())
    record["schedule_sha256"]["001"] = "unpaired"
    changed.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="query schedule"):
        check_pairing(folders)
    record["schedule_sha256"]["001"] = "q25"
    record["geometry_schedule_sha256"]["001"] = "unpaired"
    changed.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="geometry"):
        check_pairing(folders)
    record["geometry_schedule_sha256"]["001"] = "geom"
    changed.write_text(json.dumps(record))
    (folders[1] / "initialization.json").write_text(json.dumps({"head_state_sha256": "changed"}))
    with pytest.raises(RuntimeError, match="initial head"):
        check_pairing(folders)
