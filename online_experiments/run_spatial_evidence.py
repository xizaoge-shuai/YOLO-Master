"""Queue full-split feature, fresh-image latency and geometric diagnostics."""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from online_experiments.crossdataset_campaign import gpu_admission, job_lock, read, sha, status, write


def archive(args, stage, output):
    from online_experiments.archive_assignment_intervention import commit_files
    from online_experiments.replication_campaign import push, sanitized

    destination = ROOT / "research/experiments/spatial-evidence" / args.job.name / stage
    destination.mkdir(parents=True, exist_ok=True)
    sources = [*sorted(output.glob("*.csv")), *sorted(output.glob("*.json"))]
    names = []
    for source in sources:
        target = destination / source.name
        content = (
            json.dumps(sanitized(read(source)), indent=2, allow_nan=False) + "\n"
            if source.suffix == ".json"
            else source.read_text(encoding="utf-8")
        )
        if target.exists() and target.read_text(encoding="utf-8") != content:
            raise RuntimeError("Refuse changing an archived diagnostic output")
        target.write_text(content, encoding="utf-8")
        names.append(target.relative_to(ROOT).as_posix())
    readme = destination / "README.md"
    scope = (
        "Six views of548 images,3288 new teacher images shared by four fixed models. "
        "Identity/flip are true C2 anchors and zero-error controls. Fixed-model feature/loss sensitivity "
        "is not officialAP; identical additiveRGB cancels in absolute P3 differences. "
        if stage == "geometry"
        else "Four final seed0 checkpoints;548 images each. Feature energy is descriptive. "
        "Fresh-image latency includes each method's independent frozen backbone and20 warmup images; "
        "sharedGPU latency is observational. "
    )
    readme.write_text(
        f"# Full VisDrone fixed-model diagnostic: {stage}\n\n"
        + scope
        + "No weight updates. Exact contracts and counts are in protocol.\n",
        encoding="utf-8",
    )
    names.append(readme.relative_to(ROOT).as_posix())
    marker = f"spatial-evidence:{args.job.name}:{stage}"
    block = f"\n\n<!-- {marker} -->\n## Full548 fixed-model {stage}\n\n"
    block += (
        f"Archived [{stage}]({readme.relative_to(ROOT / 'research').as_posix()}); no retraining or seed replication.\n"
    )
    commit = commit_files(
        ROOT, names, block, marker, readme.relative_to(ROOT).as_posix(), f"research: record full548 {stage} diagnostic"
    )
    record = {"commit": commit, "pushed": False}
    if args.publish:
        try:
            record.update(pushed=True, pushed_head=push())
        except (RuntimeError, OSError, subprocess.SubprocessError) as error:
            record["push_error"] = str(error)
    write(args.job / f"archive-{stage}.json", record)
    return record


def run(args):
    # A fresh directory prevents resuming over incomplete or differently sourced measurements.
    args.job.mkdir(parents=True, exist_ok=False)
    with job_lock(args.job):
        write(
            args.job / "plan.json",
            {
                "schema": "full548-spatial-evidence-v2",
                "stages": ["features", "latency", "geometry"],
                "gpu_index": args.gpu,
                "methods": ["blank25", "detail25", "lowpass25", "vit-adapter-cache25"],
                "images_each": 548,
                "latency_warmup_images_each": 20,
                "no_weight_updates": True,
                "script_sha256": sha(ROOT / "online_experiments/spatial_feature_analysis.py"),
                "geometry_script_sha256": sha(ROOT / "online_experiments/geometric_evidence.py"),
                "geometry_teacher_images": 3288,
                "geometry_official_AP_computed": False,
                "campaign": str(args.campaign),
                "original_detail_job": str(args.original_detail_job),
            },
        )
        write(args.job / "worker.json", {"pid": os.getpid(), "started_unix": time.time(), "argv": sys.argv})
        results = []
        for stage in ("features", "latency", "geometry"):
            output = args.job / stage
            with gpu_admission(args, f"Full548 fixed-model {stage}"):
                coverage = "4 models x548 images x6 views" if stage == "geometry" else "4 methods x548 images"
                status(args.job, "MEASURING", f"{stage}:{coverage}; no training", stage=stage)
                script = "geometric_evidence.py" if stage == "geometry" else "spatial_feature_analysis.py"
                command = [
                    sys.executable,
                    "-u",
                    str(ROOT / "online_experiments" / script),
                    "run" if stage == "geometry" else stage,
                    "--campaign",
                    str(args.campaign),
                    "--original-detail-job",
                    str(args.original_detail_job),
                    "--output",
                    str(output),
                ]
                if stage != "geometry":
                    command += ["--warmup", "20"]
                env = dict(os.environ)
                env.update(
                    CUDA_VISIBLE_DEVICES=str(args.gpu),
                    HF_HUB_OFFLINE="1",
                    TRANSFORMERS_OFFLINE="1",
                    HF_HOME=str(ROOT.parent / "cache/huggingface"),
                    GH_CONFIG_DIR=str(ROOT.parent / ".config/gh"),
                    OMP_NUM_THREADS="2",
                    OPENBLAS_NUM_THREADS="2",
                    MKL_NUM_THREADS="2",
                    PYTHONUNBUFFERED="1",
                )
                env["PYTHONPATH"] = str(ROOT / ".research-deps/evidence-20261110/site") + os.pathsep + str(ROOT)
                with (args.job / f"{stage}.log").open("a") as stream:
                    child = subprocess.Popen(
                        command,
                        cwd=ROOT,
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        pass_fds=(args.gpu_lock_fd,),
                    )
                    write(args.job / "active-child.json", {"pid": child.pid, "stage": stage, "argv": command})
                    code = child.wait()
                if code:
                    raise RuntimeError(f"{stage} exited{code}; its outputs are retained")
            result = read(output / "completion.json")
            image_count = result.get("images") if stage == "geometry" else result.get("images_each")
            methods = result.get("methods", [])
            expected_methods = {"blank25", "detail25", "lowpass25", "vit-adapter-cache25"}
            if (
                result.get("status") != "SUCCEEDED"
                or image_count != 548
                or len(methods) != 4
                or set(methods) != expected_methods
            ):
                raise RuntimeError("Measurement ended without full declared coverage")
            if stage == "features" and (result["feature_image_pairs"], result["teacher_forward_images"]) != (2192, 0):
                raise RuntimeError("Full cached feature diagnostic workload changed")
            if stage == "latency" and (
                result["teacher_measured_images"],
                result["teacher_warmup_images"],
                result["feature_bank_reads"],
            ) != (2192, 80, 0):
                raise RuntimeError("Fresh-image latency workload/teacher accounting changed")
            if stage == "geometry" and (
                result["teacher_images"],
                result["teacher_calls"],
                result["teacher_blank_images_new"],
                result["method_image_views"],
                result["official_AP_computed"],
                result["training_updates"],
            ) != (3288, 3288, 0, 13152, False, 0):
                raise RuntimeError("Full geometric feature/loss workload changed")
            results.append({"stage": stage, "completion": result, "archive": archive(args, stage, output)})
            write(args.job / "completed-stages.json", results)
        status(
            args.job,
            "SUCCEEDED",
            "Full548 features, fresh-image latency and geometric feature/loss diagnostics complete",
            completed_stages=len(results),
            all_pushed=all(item["archive"]["pushed"] for item in results),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--original-detail-job", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--min-free-mib", type=int, default=8192)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    if args.gpu < 0 or args.min_free_mib < 8192:
        parser.error("Use a healthy GPU and >=8192MiB admission reserve")
    args.job, args.campaign = args.job.resolve(), args.campaign.resolve()
    args.original_detail_job = args.original_detail_job.resolve()
    args.dataset = "VisDrone-existing-final-models"
    if args.job.exists():
        parser.error("Choose a fresh job directory; prior outputs are never overwritten")
    try:
        run(args)
    except Exception:
        write(args.job / "failure.json", {"traceback": traceback.format_exc(), "time_unix": time.time()})
        status(args.job, "FAILED", "Inspect stage log; no automatic retry, no subset fallback")
        raise


if __name__ == "__main__":
    main()
