"""Re-score saved full VisDrone predictions after parity with the pinned MATLAB toolkit."""

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
import time
import zipfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat
from visdrone_reference_eval import REFERENCE_COMMIT, evaluate_raw, prepare_arrays

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("cache1", "cache2", "mix25", "transport25", "correct25", "online-geom")
KEYS = ("AP", "AP50", "AP75", "AR1", "AR10", "AR100", "AR500")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def status(job, message):
    (job / "status.txt").write_text(message + "\n")
    print(message, flush=True)


def read_rows(data):
    if not data.strip():
        return np.empty((0, 8), dtype=np.float64)
    rows = np.loadtxt(io.BytesIO(data), delimiter=",", ndmin=2)
    if rows.shape[1] != 8 or not np.isfinite(rows).all():
        raise ValueError("Expected finite original-coordinate VisDrone rows with 8 columns")
    return rows


def validate_annotation_identity(samples, annotations, expected):
    ids = [Path(sample["image_path"]).stem for sample in samples]
    if len(ids) != 548 or len(set(ids)) != 548 or set(ids) != set(annotations):
        raise ValueError("Validation manifest must cover exactly the 548 unique annotation IDs")
    if not isinstance(expected, dict):
        raise TypeError("Original annotation hash manifest is required")
    hashes = {Path(name).stem: value for name, value in expected.items()}
    if len(expected) != 548 or len(hashes) != 548 or set(hashes) != set(ids):
        raise ValueError("Original annotation hash coverage is incomplete")
    return hashes


def verify_reference_files(toolkit):
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=toolkit, text=True).strip()
    if revision != REFERENCE_COMMIT:
        raise ValueError(f"Unexpected official toolkit revision: {revision}")
    names = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", REFERENCE_COMMIT, "--", "utils"], cwd=toolkit, text=True
    ).splitlines()
    if {p.name for p in (toolkit / "utils").iterdir()} != {Path(n).name for n in names}:
        raise ValueError("Unexpected reference helper files could shadow the pinned functions")
    for name in names:
        expected = subprocess.check_output(["git", "show", f"{REFERENCE_COMMIT}:{name}"], cwd=toolkit)
        if (toolkit / name).read_bytes() != expected:
            raise ValueError(f"Reference source differs from its pinned commit: {name}")
    return {Path(n).name: sha(toolkit / n) for n in names}


def load_ground_truth(source):
    manifest_path = source / "dataset-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["smoke"] or len(manifest["splits"]["val"]) != 548 or manifest["test_dev_used"]:
        raise ValueError("This campaign requires the full 548-image development validation split")
    archive = ROOT / "datasets/VisDrone/VisDrone2019-DET-val.zip"
    records, hashes = [], {}
    with zipfile.ZipFile(archive) as z:
        annotations = {Path(n).stem: n for n in z.namelist() if "/annotations/" in n and n.endswith(".txt")}
        samples = manifest["splits"]["val"]
        expected = validate_annotation_identity(samples, annotations, manifest.get("raw_val_annotation_sha256"))
        for sample in samples:
            image_id = Path(sample["image_path"]).stem
            raw = z.read(annotations[image_id])
            hashes[image_id] = hashlib.sha256(raw).hexdigest()
            if hashes[image_id] != expected[image_id]:
                raise ValueError(f"Raw annotation changed: {image_id}")
            records.append((image_id, read_rows(raw), tuple(sample["letterbox"]["original_shape"])))
    return records, hashes, sha(manifest_path)


def load_cases(directory, records):
    expected = {r[0] for r in records}
    files = {p.stem: p for p in directory.glob("*.txt")}
    if set(files) != expected:
        raise ValueError(f"Incomplete or extra prediction files: {directory}")
    cases, fingerprints = [], {}
    for image_id, gt, shape in records:
        raw = files[image_id].read_bytes()
        dt = read_rows(raw)
        if len(dt) > 500 or np.any(~np.isin(dt[:, 5], np.arange(1, 11))):
            raise ValueError(f"Invalid detection count/category: {image_id}")
        if len(dt) and np.any(np.diff(dt[:, 4]) > 0):
            raise ValueError(f"Unsorted detections: {image_id}")
        cases.append((gt, dt, shape))
        fingerprints[image_id] = hashlib.sha256(raw).hexdigest()
    return cases, fingerprints


def synthetic_cases():
    rng = np.random.default_rng(1709)
    cases = []
    for index in range(12):
        gt = np.zeros((20, 8), dtype=float)
        gt[:, :2] = rng.integers(0, 65, (20, 2))
        gt[:, 2:4] = rng.integers(1, 25, (20, 2))
        gt[:, 4] = rng.choice([0, 1], 20, p=[0.15, 0.85])
        gt[:, 5] = rng.integers(1, 12, 20)
        if index % 2:
            gt = np.vstack([gt, [[0, 0, 20, 15, 0, 0, 0, 0], [10, 2, 15, 18, 0, 0, 0, 0]]])
        dt = np.zeros((60, 8), dtype=float)
        dt[:, :2] = rng.uniform(-1, 75, (60, 2))
        dt[:, 2:4] = rng.uniform(0.5, 30, (60, 2))
        dt[:, 4] = rng.choice([0.1, 0.5, 0.9], 60)
        dt[:, 5] = rng.integers(1, 11, 60)
        exact = gt[(gt[:, 5] > 0) & (gt[:, 5] < 11)].copy()
        exact[:, 4] = 0.5
        dt = np.vstack([dt, exact, exact[:2]])
        dt = dt[np.argsort(-dt[:, 4], kind="stable")]
        cases.append((gt, dt, (80, 100)))
    # Exact IoU boundaries and an empty detection image.
    for step in range(10, 20):
        gt = np.array([[30, 20, 20, 20, 1, 1, 0, 0]], dtype=float)
        dt = np.array([[30, 20, step, 20, 0.8, 1, -1, -1]], dtype=float)
        cases.append((gt, dt, (80, 100)))
    cases.append((gt, np.empty((0, 8)), (80, 100)))
    # Global top-500 truncation; the only true positive is at position 501.
    dt = np.tile([70, 60, 5, 5, 0.9, 2, -1, -1], (501, 1)).astype(float)
    dt[-1] = [30, 20, 20, 20, 0.1, 1, -1, -1]
    cases.append((gt, dt, (80, 100)))
    return cases


def native_parity(job, toolkit, octave, cases, name):
    verify_reference_files(toolkit)
    work = job / "parity"
    work.mkdir(exist_ok=True)
    if any(
        p.suffix in (".m", ".oct", ".mex") and p.name not in {"mean2.m", "reference_runner.m"} for p in work.iterdir()
    ):
        raise ValueError("Unexpected helper could shadow official evaluation functions")
    gt_cells, dt_cells = np.empty((1, len(cases)), dtype=object), np.empty((1, len(cases)), dtype=object)
    for i, (gt, dt, _) in enumerate(cases):
        gt_cells[0, i], dt_cells[0, i] = gt, dt
    source = work / f"{name}-input.mat"
    target = work / f"{name}-reference.mat"
    savemat(source, {"raw_gt": gt_cells, "raw_dt": dt_cells, "shapes": np.asarray([c[2] for c in cases])})
    # mean2 is a MATLAB image-toolbox convenience; its exact scalar definition is sufficient.
    (work / "mean2.m").write_text("function y = mean2(x)\n  y = mean(x(:));\nend\n")
    runner = work / "reference_runner.m"
    runner.write_text(
        "args = argv();\naddpath(args{1});\naddpath(args{2});\ns = load(args{3});\n"
        "n = numel(s.raw_gt); allgt = cell(1,n); alldet = cell(1,n);\n"
        "for k = 1:n\n"
        "  [g,d] = dropObjectsInIgr(s.raw_gt{k},s.raw_dt{k},s.shapes(k,1),s.shapes(k,2));\n"
        "  old = g(:,5); g(old==0,5)=1; g(old==1,5)=0;\n"
        "  allgt{k}=g; alldet{k}=d;\nend\n"
        "[a,b,c,d,e,f,g] = calcAccuracy(n,allgt,alldet);\n"
        "values = [a,b,c,d,e,f,g];\n"
        "save('-mat7-binary',args{4},'values','allgt','alldet');\n"
    )
    env = os.environ.copy()
    env["OCTAVE_HOME"] = str(octave.resolve().parent.parent)
    with (work / f"{name}-octave.log").open("w") as log:
        subprocess.run(
            [
                str(octave),
                "--quiet",
                "--no-history",
                str(runner),
                str(toolkit / "utils"),
                str(work),
                str(source),
                str(target),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=1800,
            env=env,
        )
    native = loadmat(target)
    for i, case in enumerate(cases):
        pg, pd = prepare_arrays(*case)
        np.testing.assert_array_equal(pg, native["allgt"][0, i])
        np.testing.assert_array_equal(pd, native["alldet"][0, i])
    result = evaluate_raw(cases)
    delta = np.abs(native["values"].reshape(-1) - np.asarray([result[k] for k in KEYS]))
    if delta.max() > 1e-9:
        raise ValueError(f"Reference parity failure {name}: {dict(zip(KEYS, delta.tolist()))}")
    return {
        "name": name,
        "images": len(cases),
        "max_absolute_AP_point_error": float(delta.max()),
        "preprocessed_arrays_exact": True,
        "input_sha256": sha(source),
        "native_output_sha256": sha(target),
        "native_metrics": native["values"].reshape(-1).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--octave", type=Path, required=True)
    parser.add_argument("--toolkit", type=Path, required=True)
    args = parser.parse_args()
    job = args.output.resolve()
    job.mkdir(parents=True, exist_ok=True)
    source = args.source_job.resolve()
    try:
        if (source / "status.txt").read_text().strip() != "SUCCEEDED":
            raise ValueError("Source training campaign is incomplete")
        status(job, "RUNNING verify original validation annotations and prediction coverage")
        records, annotation_hashes, manifest_hash = load_ground_truth(source)
        sample, _ = load_cases(source / "train/cache2-s0/predictions/last", records)
        source_files = [Path(__file__), Path(__file__).with_name("visdrone_reference_eval.py")]
        sources = {p.name: sha(p) for p in source_files}
        reference_files = verify_reference_files(args.toolkit)
        provenance = {
            "source_job": str(source),
            "manifest_sha256": manifest_hash,
            "annotation_hashes": annotation_hashes,
            "evaluator_source_hashes": sources,
            "official_reference_commit": REFERENCE_COMMIT,
            "reference_file_hashes": reference_files,
            "checkpoint_selection": "last primary; best selected using original custom DetMetrics",
            "test_dev_used": False,
            "images": 548,
            "note": "Reference AP uses class-image occurrence weighting and VOC integral; macro AP separate.",
        }
        dump(job / "provenance.json", provenance)
        status(job, "RUNNING native MATLAB-code parity: synthetic boundary/ignore fixtures")
        parity = [native_parity(job, args.toolkit, args.octave, synthetic_cases(), "synthetic")]
        status(job, "RUNNING native MATLAB-code parity: two original full-size validation images")
        parity.append(native_parity(job, args.toolkit, args.octave, sample[:2], "real"))
        dump(job / "parity.json", parity)
        summary = []
        for method in METHODS:
            for checkpoint in ("last", "best"):
                for p in source_files:
                    if sha(p) != sources[p.name]:
                        raise ValueError("Evaluator implementation changed during this campaign")
                status(job, f"RUNNING evaluate/{method}/{checkpoint} images=548")
                directory = source / "train" / f"{method}-s0" / "predictions" / checkpoint
                cases, fingerprints = load_cases(directory, records)
                started = time.perf_counter()
                result = evaluate_raw(cases)
                result.update(
                    method=method,
                    seed=0,
                    checkpoint=checkpoint,
                    images=len(cases),
                    cpu_evaluation_seconds=time.perf_counter() - started,
                )
                dump(job / f"{method}-{checkpoint}.json", result)
                dump(job / f"{method}-{checkpoint}-input-hashes.json", fingerprints)
                summary.append(
                    {
                        key: result[key]
                        for key in (
                            "method",
                            "seed",
                            "checkpoint",
                            *KEYS,
                            "macro_AP_diagnostic",
                            "cpu_evaluation_seconds",
                        )
                    }
                )
                dump(job / "results.json", summary)
                lines = [
                    "VisDrone pinned-reference Python port; Octave parity passed.",
                    "AP 0-100; last is primary, best was selected by custom training AP.",
                    "method checkpoint AP AP50 AP75 AR500 macro_AP_diagnostic",
                ]
                lines += [
                    f"{r['method']} {r['checkpoint']} {r['AP']:.4f} {r['AP50']:.4f} "
                    f"{r['AP75']:.4f} {r['AR500']:.4f} {r['macro_AP_diagnostic']:.4f}"
                    for r in summary
                ]
                (job / "comparison.txt").write_text("\n".join(lines) + "\n")
                print(lines[-1], flush=True)
        with (job / "results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=summary[0])
            writer.writeheader()
            writer.writerows(summary)
        status(job, "SUCCEEDED")
    except BaseException as exc:
        status(job, f"FAILED {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
