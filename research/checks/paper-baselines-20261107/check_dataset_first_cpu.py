"""Check scheduling contracts on CPU without importing a model or starting training."""

import ast
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main():
    os.chdir(ROOT)
    from online_experiments import dataset_first_campaign as module

    job = ROOT / "runs/paper/paper-baselines/job-20261107-Imo9Yocz"
    source = ROOT / "online_experiments/dataset_first_campaign.py"
    checks = {"ast": "PASS"}
    tree = ast.parse(source.read_text())
    for arguments in (("check",), ("format", "--check")):
        command = [sys.executable, "-m", "ruff", *arguments, str(source), __file__]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        checks["ruff " + " ".join(arguments)] = result.returncode
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
    queue = module.read(job / "queue.json")
    first, deferred = module.partition(queue)
    assert (len(first), len(deferred)) == (10, 18)
    checks["partition_actual_28"] = "PASS"
    invalid = [
        queue[:-1],
        queue + [queue[0]],
        [dict(queue[0], seed=True), *queue[1:]],
        [dict(queue[0], name="bad"), *queue[1:]],
        [dict(queue[0], resume_original=True), *queue[1:]],
    ]
    for value in invalid:
        try:
            module.partition(value)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid queue accepted")
    checks["reject_invalid_queue"] = "PASS"
    observed = module.process(os.getpid())
    module.require_adopt_identity(observed, observed["argv"], observed["starttime"])
    for field, value in (
        ("uid", os.getuid() + 1),
        ("starttime", observed["starttime"] + 1),
        ("argv", ["other"]),
        ("cwd", "/other"),
    ):
        invalid = dict(observed)
        invalid[field] = value
        try:
            module.require_adopt_identity(invalid, observed["argv"], observed["starttime"])
        except RuntimeError:
            pass
        else:
            raise AssertionError("Invalid process identity accepted: " + field)
    checks["process_identity"] = "PASS"
    with tempfile.TemporaryDirectory(prefix="dataset-first-cpu-") as temporary:
        directory = Path(temporary)
        current, original = directory / "a/job", directory / "b/job"
        (current / module.PHASE).mkdir(parents=True)
        original.mkdir(parents=True)
        with module.locks(current, original):
            argv = [
                "coordinator",
                "run",
                "--job",
                str(current),
                "--origin-root",
                str(ROOT),
                "--original-job",
                str(original),
            ]
            with patch("sys.argv", argv):
                try:
                    with contextlib.redirect_stderr(io.StringIO()):
                        module.main()
                except SystemExit as error:
                    assert error.code == 2
                else:
                    raise AssertionError("Duplicate worker acquired locks")
            assert not (current / "status.txt").exists()
            assert not (current / module.PHASE / "status.json").exists()
        checks["lock_refusal_preserves_status"] = "PASS"
        case = first[0]
        try:
            module.validate(current, Path("/source"), Path("/cache"), case)
        except RuntimeError as error:
            assert "DONE" in str(error)
        else:
            raise AssertionError("Missing DONE marker accepted")
        output = current / "train" / case["name"]
        output.mkdir(parents=True)
        (output / "DONE").touch()
        with patch.object(module.subprocess, "run", side_effect=subprocess.CalledProcessError(1, ["validator"])):
            try:
                module.validate(current, Path("/source"), Path("/cache"), case)
            except subprocess.CalledProcessError:
                pass
            else:
                raise AssertionError("Failed full validator accepted")
        with patch.object(module.subprocess, "run") as runner:
            module.validate(current, Path("/source"), Path("/cache"), case)
            assert runner.call_count == 1
            assert runner.call_args.kwargs["check"] is True
        checks["DONE_plus_full_validator_required"] = "PASS"
    result = subprocess.run([sys.executable, str(source), "--help"], capture_output=True, text=True, check=False)
    assert result.returncode == 0
    checks["cli_help"] = "PASS"
    assert not any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"kill", "killpg"}
        for node in ast.walk(tree)
    )
    checks["coordinator_sends_no_signals"] = "PASS"
    provenance = module.read(job / "provenance.json")
    for name, expected in provenance["source_sha256"].items():
        assert module.digest(ROOT / name) == expected, name
    checks["frozen_source_unchanged"] = len(provenance["source_sha256"])
    checks["source_sha256"] = module.digest(source)
    checks["no_GPU_model_smoke_or_subset_run"] = True
    destination = Path(__file__).with_name("dataset-first-cpu-validation-20261109.json")
    destination.write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
