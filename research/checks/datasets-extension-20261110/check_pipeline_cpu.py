"""Exercise new-pipeline admission and duplicate-worker contracts without GPU work."""

import contextlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main():
    from online_experiments import crossdataset_campaign as module

    checks = {}
    assert len(module.METHODS) == len(set(module.METHODS)) == 12
    assert module.METHODS.index("detail25") > module.METHODS.index("vit-adapter-cache25")
    assert not any(name.endswith(("-s1", "-s2")) for name in module.METHODS)
    checks["baseline_first_twelve_seed_zero"] = "PASS"
    with patch.object(module.subprocess, "check_output", return_value="12000, P2\n"):
        assert module.gpu_memory(0) == 12000
    with patch.object(module.subprocess, "check_output", return_value="[GPU requires reset], [GPU requires reset]\n"):
        try:
            module.gpu_memory(1)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Broken device admitted")
    checks["healthy_device_only"] = "PASS"
    with tempfile.TemporaryDirectory(prefix="pipeline-cpu-") as temporary:
        job = Path(temporary) / "job"
        job.mkdir()
        with module.job_lock(job):
            try:
                with module.job_lock(job):
                    raise AssertionError("Duplicate worker admitted")
            except module.Busy:
                pass
        checks["duplicate_pipeline_lock_refused"] = "PASS"
        args = SimpleNamespace(job=job, gpu=0, min_free_mib=8192, dataset="voc0712")
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(module, "GPU_LOCK_ROOT", Path(temporary)))
            stack.enter_context(patch.object(module, "gpu_memory", return_value=12000))
            stack.enter_context(module.gpu_admission(args, "CPU contract"))
            assert hasattr(args, "gpu_lock_fd")
            os.fstat(args.gpu_lock_fd)
        assert not hasattr(args, "gpu_lock_fd")
        checks["GPU_lock_fd_available_for_inheritance"] = "PASS"
        second = SimpleNamespace(job=Path(temporary) / "job2", gpu=0, min_free_mib=8192, dataset="sku110k")
        third = SimpleNamespace(job=Path(temporary) / "job3", gpu=0, min_free_mib=8192, dataset="diagnostic")
        second.job.mkdir()
        third.job.mkdir()
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(module, "GPU_LOCK_ROOT", Path(temporary)))
            stack.enter_context(patch.object(module, "gpu_memory", return_value=32768))
            stack.enter_context(module.gpu_admission(args, "first"))
            stack.enter_context(module.gpu_admission(second, "second"))
            with patch.object(module.time, "sleep", side_effect=TimeoutError("third slot waits")):
                try:
                    with module.gpu_admission(third, "third"):
                        raise AssertionError("More than two new stages admitted")
                except TimeoutError:
                    pass
            assert module.read(third.job / "pipeline-status.json")["state"] == "WAITING_GPU"
            os.fstat(args.gpu_lock_fd)
            os.fstat(second.gpu_lock_fd)
        checks["two_stage_limit_and_third_waits"] = "PASS"
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(module, "GPU_LOCK_ROOT", Path(temporary)))
            stack.enter_context(patch.object(module, "gpu_memory", return_value=12000))
            stack.enter_context(module.gpu_admission(args, "first"))
            with patch.object(module.time, "sleep", side_effect=TimeoutError("reserve guard")):
                try:
                    with module.gpu_admission(second, "second"):
                        raise AssertionError("Second stage admitted without16GiB reserve")
                except TimeoutError:
                    pass
        checks["second_stage_memory_reserve"] = "PASS"
        with patch.object(module, "live_children", return_value=[{"pid": 123}]):
            try:
                module.run(args)
            except module.Busy:
                pass
            else:
                raise AssertionError("Live orphan child was ignored")
        assert not (job / "pipeline-status.json").exists()
        checks["live_child_refusal_preserves_status"] = "PASS"
    command = [sys.executable, str(ROOT / "online_experiments/crossdataset_campaign.py"), "--help"]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0
    checks["CLI_help_no_GPU"] = "PASS"
    checks["no_GPU_execution"] = True
    target = Path(__file__).with_name("pipeline-cpu-contracts.json")
    target.write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
