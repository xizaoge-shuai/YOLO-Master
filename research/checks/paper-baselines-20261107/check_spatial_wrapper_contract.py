"""CPU-only mocked supervisor contracts; no experiment, GPU, subprocess or commit."""

import argparse
import contextlib
import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "online_experiments/run_spatial_evidence.py"
METHODS = ["blank25", "detail25", "lowpass25", "vit-adapter-cache25"]


def load_wrapper():
    specification = importlib.util.spec_from_file_location("spatial_wrapper_contract_target", SOURCE)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def completion(stage):
    record = {"status": "SUCCEEDED", "methods": METHODS.copy()}
    if stage == "geometry":
        record.update(images=548, teacher_images=3288, teacher_calls=3288,
                      teacher_blank_images_new=0, method_image_views=13152,
                      official_AP_computed=False, training_updates=0)
    elif stage == "features":
        record.update(images_each=548, feature_image_pairs=2192, teacher_forward_images=0)
    else:
        record.update(images_each=548, teacher_measured_images=2192,
                      teacher_warmup_images=80, feature_bank_reads=0)
    return record


def exercise(module, folder, mutation=None):
    commands, admitted, archived = [], [], []
    args = SimpleNamespace(job=folder / "job", campaign=folder / "campaign",
                           original_detail_job=folder / "old-detail", gpu=0,
                           min_free_mib=8192, publish=True, dataset="CPU-mock-contract")

    @contextlib.contextmanager
    def admission(arguments, message):
        arguments.gpu_lock_fd = 42
        admitted.append(message)
        try:
            yield 48000
        finally:
            del arguments.gpu_lock_fd

    class FakeChild:
        pid = 123456

        def __init__(self, command, **kwargs):
            commands.append(command)
            assert kwargs["pass_fds"] == (42,)
            assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0"
            script, action = Path(command[2]).name, command[3]
            stage = "geometry" if script == "geometric_evidence.py" else action
            assert stage in ("features", "latency", "geometry")
            assert action == ("run" if stage == "geometry" else stage)
            assert ("--warmup" in command) == (stage != "geometry")
            output = Path(command[command.index("--output") + 1])
            output.mkdir()
            record = completion(stage)
            if mutation is not None:
                mutation(stage, record)
            (output / "completion.json").write_text(json.dumps(record), encoding="utf-8")

        def wait(self):
            return 0

    def archive(arguments, stage, output):
        archived.append(stage)
        return {"commit": "CPU-mock-no-commit", "pushed": True}

    error = None
    with patch.object(module, "job_lock", lambda unused: contextlib.nullcontext()), \
            patch.object(module, "gpu_admission", admission), \
            patch.object(module.subprocess, "Popen", FakeChild), \
            patch.object(module, "archive", archive):
        try:
            module.run(args)
        except RuntimeError as problem:
            error = str(problem)
    return {"error": error, "commands": commands, "admissions": admitted, "archives": archived}


def archive_contract(module, folder):
    output = folder / "output"
    output.mkdir(parents=True)
    (output / "completion.json").write_text(json.dumps(completion("geometry")), encoding="utf-8")
    (output / "raw-dino-error.csv").write_text("image_id,relative_l2\nexample,0\n", encoding="utf-8")
    committed = []

    def fake_commit(root, names, block, marker, readme, message):
        committed.extend(names)
        assert "geometry" in marker and readme.endswith("README.md")
        return "CPU-mock-no-commit"

    modules = {
        "online_experiments.archive_assignment_intervention": SimpleNamespace(commit_files=fake_commit),
        "online_experiments.replication_campaign": SimpleNamespace(
            sanitized=lambda value: value, push=lambda: "CPU-mock-no-push"
        ),
    }
    args = SimpleNamespace(job=folder / "job", publish=True)
    args.job.mkdir()
    with patch.object(module, "ROOT", folder), patch.dict(sys.modules, modules):
        record = module.archive(args, "geometry", output)
    readme = folder / "research/experiments/spatial-evidence/job/geometry/README.md"
    text = readme.read_text(encoding="utf-8")
    assert "3288 new teacher images" in text and "four fixed models" in text
    assert "zero-error controls" in text and "not officialAP" in text
    assert "No weight updates" in text and "20 warmup" not in text
    assert len(committed) == 3 and record["pushed"] is True
    return {"geometry_counts": "548x6=3288 shared by4; correct", "files": committed,
            "commit_executed": False, "push_executed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("spatial-wrapper-contract-20261110.json"))
    args = parser.parse_args()
    initial = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    module = load_wrapper()
    checks = []
    with tempfile.TemporaryDirectory(prefix="spatial-wrapper-cpu-") as temporary:
        folder = Path(temporary)
        happy = exercise(module, folder / "happy")
        assert happy["error"] is None and happy["archives"] == ["features", "latency", "geometry"]
        checks.append({"contract": "three stages, correct CLI, inherited GPU lease FD, happy completion", "pass": True})
        mutations = (
            ("geometry teacher count", "geometry", {"teacher_images": 3287}),
            ("geometry cropped official AP prohibited", "geometry", {"official_AP_computed": True}),
            ("geometry full image coverage", "geometry", {"images": 547}),
            ("latency cached input prohibited", "latency", {"feature_bank_reads": 1}),
            ("features full pair coverage", "features", {"feature_image_pairs": 1}),
            ("features teacher-free workload", "features", {"teacher_forward_images": 1}),
            ("exact method identity no duplicates", "features", {"methods": ["blank25"] * 4}),
        )
        for index, (name, target_stage, altered) in enumerate(mutations):
            def mutate(stage, record, target=target_stage, values=altered):
                if stage == target:
                    record.update(copy.deepcopy(values))
            observed = exercise(module, folder / f"negative-{index}", mutate)
            checks.append({"contract": name, "pass": observed["error"] is not None,
                           "observed_rejection": observed["error"], "archives": observed["archives"]})
        archive = archive_contract(module, folder / "archive")
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == initial, "Supervisor changed during contract check"
    result = {"status": "PASS" if all(check["pass"] for check in checks) else "BLOCKED",
              "wrapper_sha256": initial, "contracts": checks, "archive_readme": archive,
              "GPU_execution": False, "GPU_model_execution": False,
              "real_subprocesses_launched": 0, "commits_or_pushes": 0,
              "scope": "CPU supervisor contracts with simulated child completions; no experiment results"}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "record": str(args.output),
                      "failed": [check["contract"] for check in checks if not check["pass"]]}))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
