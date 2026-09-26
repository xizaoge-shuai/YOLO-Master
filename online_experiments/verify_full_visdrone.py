"""Integration verification: full data audit and interrupted-vs-continuous training."""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from online_experiments.full_visdrone_data import dump, prepare, read, source_snapshot


def main():
    import torch

    base = ROOT / "runs/paper/full-visdrone"
    source = Path((base / "latest-smoke.txt").read_text().strip())
    job = base / f"verification-{time.time_ns()}"
    job.mkdir()
    datajob = job / "full-data"
    datajob.mkdir()
    meta = prepare(datajob, False)
    assert [len(meta["splits"][x]) for x in ("train", "val")] == [6471, 548]
    assert len(meta["removed_labels"]) == 1, meta["removed_labels"]
    assert not meta["test_dev_used"]
    trainjob = job / "resume-check"
    trainjob.mkdir()
    shutil.copyfile(source / "dataset-manifest.json", trainjob / "dataset-manifest.json")
    (trainjob / "cache").symlink_to(source / "cache", target_is_directory=True)
    dump(trainjob / "provenance.json", source_snapshot())

    def command(output):
        return [
            sys.executable,
            "-u",
            "online_experiments/full_visdrone_train.py",
            "--job",
            str(trainjob),
            "--output",
            str(output),
            "--method",
            "correct25",
            "--epochs",
            "2",
        ]

    uninterrupted = trainjob / "uninterrupted"
    interrupted = trainjob / "interrupted"
    with (job / "uninterrupted.log").open("w") as stream:
        subprocess.run(command(uninterrupted), cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    proc = subprocess.Popen(
        command(interrupted), cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    stopped = False
    with (job / "interrupted.log").open("w") as stream:
        for line in proc.stdout:
            stream.write(line)
            if line.startswith('{"epoch": 1,'):
                proc.terminate()
                stopped = True
                break
    proc.wait(timeout=30)
    assert stopped, "First epoch was not reached"
    checkpoint = torch.load(interrupted / "weights/last.pt", map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == 1
    with (job / "resumed.log").open("w") as stream:
        subprocess.run(command(interrupted), cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    one = torch.load(uninterrupted / "weights/last.pt", map_location="cpu", weights_only=False)
    two = torch.load(interrupted / "weights/last.pt", map_location="cpu", weights_only=False)
    differences = {}
    for part in ("model", "corrector"):
        errors = []
        for k in one[part]:
            if torch.is_tensor(one[part][k]):
                errors.append(float((one[part][k].float() - two[part][k].float()).abs().max()))
            else:
                assert one[part][k] == two[part][k], k
        differences[part] = max(errors)
        assert differences[part] < 1e-6, differences
    for run in (uninterrupted, interrupted):
        assert read(run / "summary.json")["query_images"] == 8
        assert (run / "DONE").exists()

    # Crash exactly after committing the final epoch, before publishing best links/DONE.
    fault_run = trainjob / "final-commit-interruption"
    fault_code = """import os,sys
from online_experiments import full_visdrone_train as t
real_save=t.atomic_checkpoint
def crash(path,state):
    real_save(path,state)
    if path.name == 'last.pt':
        os._exit(99)
t.atomic_checkpoint=crash
t.main()
"""
    normal = command(fault_run)
    normal[-1] = "1"
    with (job / "final-commit-interruption.log").open("w") as stream:
        code = subprocess.call(
            [sys.executable, "-c", fault_code, *normal[3:]], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT
        )
    assert code == 99, code
    assert not (fault_run / "DONE").exists()
    with (job / "final-commit-resume.log").open("w") as stream:
        subprocess.run(normal, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    final = read(fault_run / "summary.json")
    assert final["completed_epochs"] == 1
    assert (fault_run / "DONE").exists()
    assert len(list((fault_run / "predictions/best").glob("*.txt"))) == 8
    best_state = torch.load(fault_run / "weights/best.pt", map_location="cpu", weights_only=False)
    assert best_state["epoch"] == final["best_epoch"]

    report = {
        "full_train_images": 6471,
        "full_val_images": 548,
        "test_dev_used": False,
        "filtered_labels": meta["removed_labels"],
        "resume_max_abs_differences": differences,
        "resume_query_images": 8,
        "final_commit_crash_recovery": "PASS",
        "smoke_source": str(source),
        "official_evaluation": False,
    }
    dump(job / "verification.json", report)
    print(json.dumps(report, indent=2))
    print("VERIFICATION_JOB=" + str(job), flush=True)


if __name__ == "__main__":
    main()
