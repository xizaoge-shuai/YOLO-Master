cd ~/YOLO-Master && cat > online_experiments/flip_cache_probe.py <<'PY'
#!/usr/bin/env python3
"""Build genuine original/flip features; compare teacher-free cached training."""
import argparse
import ast
import csv
import hashlib
import io
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    return json.loads(Path(path).read_text())

def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2))

def sha(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None

def verify(hashes, root=ROOT):
    for name, expected in hashes.items():
        if sha(root / name) != expected:
            raise RuntimeError(f"File changed: {name}")

def select_view(rng, views):
    return int(rng.random() < .5) if views == 2 else 0

def cache_cost(meta, views):
    return meta["setup_seconds"] + sum(meta["view_seconds"][:views]), sum(meta["view_bytes"][:views])

BATCHES = """
class Batches:
    def __init__(self, training):
        self.training = training
    def __iter__(self):
        ds = train if self.training else val
        indices = torch.randperm(len(ds), generator=generator).tolist() if self.training else list(range(len(ds)))
        for start in range(0, len(indices), a.batch):
            ids = indices[start:start+a.batch]
            if self.training:
                items = [train_bank[select_view(aug_rng, a.views)][i] for i in ids]
            else:
                items = [val_bank[i] for i in ids]
            yield d.collate_cached(items)
"""

def patched(source):
    tree = ast.parse(source)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    hits = [0, 0]
    for index, node in enumerate(main.body):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "train_bank"):
            expected = ast.parse("bank(train) if a.mode == 'offline' else None", mode="eval").body
            if ast.dump(node.value) != ast.dump(expected):
                raise RuntimeError("Pilot train_bank changed")
            node.value = ast.parse("cached_bank(train, device)", mode="eval").body
            hits[0] += 1
        if isinstance(node, ast.ClassDef) and node.name == "Batches":
            main.body[index] = ast.parse(BATCHES).body[0]
            hits[1] += 1
    if hits != [1, 1]:
        raise RuntimeError(f"Unsupported pilot layout: {hits}")
    return ast.fix_missing_locations(tree)

def build(a):
    import torch
    from scripts import d1_train_cached_detector as d
    from scripts.d1_build_feature_cache import DINOv3MultiLevelExtractor
    from ultralytics.data.foundation_cache import load_letterboxed_tensor
    from online_experiments.d1_online_compare import geometry

    cfg = read(a.job / "reference_args.json")
    torch.set_num_threads(a.threads)
    started = time.perf_counter()
    cache = Path(cfg["cache"])
    manifest = read(cache / "manifest.json")
    d.validate_manifest(manifest, cache_root=cache, verify_files=False)
    ident = manifest["identity"]
    if ident != cfg["cache_identity"]:
        raise RuntimeError("Cache identity changed")
    layers, imgsz = ident["layers"], ident["imgsz"]
    data = d.check_det_dataset(cfg["dataset"], autodownload=False)
    ids, vids = d.split_sample_indices(len(manifest["samples"]), cfg["train_count"],
                                      cfg["val_count"], cfg["split_seed"])
    train = d.CachedDetectionDataset(cache, manifest, ids, len(data["names"]))
    split = dict(train_sample_ids=[s["sample_id"] for s in train.samples],
                 val_sample_ids=[manifest["samples"][i]["sample_id"] for i in vids])
    previous = read(a.job / "reference_split.json")
    if any(split[k] != previous[k] for k in split):
        raise RuntimeError("Split differs from completed experiments")
    root = a.job / "view-cache"
    root.mkdir()
    estimated = 4 * len(train) * sum(math.prod(manifest["samples"][0]["shapes"][f"layer_{l}"]) for l in layers)
    if shutil.disk_usage(root).free < estimated + 2 * 2**30:
        raise RuntimeError(f"Need approximately {estimated/2**30+2:.1f} GiB free disk")
    source_names = set(read(a.job / "baseline_sources.json")) | {
        "online_experiments/flip_cache_probe.py",
        "ultralytics/data/foundation_cache.py",
        "ultralytics/nn/foundation/preprocessing.py",
        "ultralytics/utils/loss.py"}
    sources = {p: sha(ROOT / p) for p in source_names}
    if any(v is None for v in sources.values()):
        raise RuntimeError("Required source file missing")
    inputs = {str((cache / "manifest.json").resolve()): sha(cache / "manifest.json")}
    if Path(cfg["dataset"]).is_file():
        inputs[str(Path(cfg["dataset"]).resolve())] = sha(cfg["dataset"])
    for i in ids + vids:
        path = d.label_path(train.dataset_root, manifest["samples"][i]["image_path"])
        inputs[str(path.resolve())] = sha(path)
    d.seed_everything(0)
    extractor = DINOv3MultiLevelExtractor(
        ident["model_id"], revision=ident["revision"], layers=layers,
        device="cuda:0", dtype=cfg["teacher_dtype"], local_files_only=True)
    torch.cuda.synchronize()
    meta = dict(identity=ident, split=split, sources=sources, inputs=inputs,
                torch_version=torch.__version__, setup_seconds=time.perf_counter()-started,
                view_seconds=[], view_bytes=[], files={}, parity=[],
                teacher_calls=0, teacher_images=0)
    for view in (0, 1):
        folder = root / f"v{view}"
        folder.mkdir()
        start = time.perf_counter()
        size = 0
        for offset in range(0, len(train), cfg["batch"]):
            images, items = [], []
            indices = list(range(offset, min(offset+cfg["batch"], len(train))))
            for i in indices:
                sample = train.samples[i]
                path = train.dataset_root / sample["image_path"]
                if sha(path) != sample["image_sha256"]:
                    raise RuntimeError(f"Image changed: {path}")
                image, info = load_letterboxed_tensor(path, imgsz=imgsz)
                if info != sample["letterbox"]:
                    raise RuntimeError(f"Letterbox changed: {path}")
                cls, boxes = d.read_yolo_labels(
                    d.label_path(train.dataset_root, sample["image_path"]),
                    info, imgsz, len(data["names"]))
                image, cls, boxes = geometry(image, cls, boxes, bool(view), 1.0)
                images.append(image)
                items.append(dict(cls=cls, bboxes=boxes, sample_id=sample["sample_id"],
                                  image_path=sample["image_path"]))
            result = extractor(torch.stack(images))
            features = [result[f"layer_{l}"].detach().cpu().to(torch.float16) for l in layers]
            meta["teacher_calls"] += 1
            meta["teacher_images"] += len(indices)
            for j, i in enumerate(indices):
                item = items[j]
                item["features"] = tuple(x[j].clone() for x in features)
                if not all(torch.isfinite(x).all().item()
                           for x in (*item["features"], item["cls"], item["bboxes"])):
                    raise RuntimeError(f"Nonfinite cached values: view={view} sample={i}")
                if view == 0 and i < 4:
                    reference = train[i]["features"]
                    for l, actual, expected in zip(layers, item["features"], reference):
                        error = float((actual.float()-expected.float()).norm()
                                      / expected.float().norm().clamp_min(1e-8))
                        meta["parity"].append(error)
                        if not math.isfinite(error) or error > cfg["parity_tolerance"]:
                            raise RuntimeError(f"Original-view parity failed: layer={l} error={error}")
                path = folder / f"{i:06d}.pt"
                torch.save(item, path)
                meta["files"][path.relative_to(root).as_posix()] = sha(path)
                size += path.stat().st_size
            if offset % (10 * cfg["batch"]) == 0:
                print(f"BUILD view={view} images={offset+len(indices)}/{len(train)}", flush=True)
        torch.cuda.synchronize()
        meta["view_seconds"].append(time.perf_counter()-start)
        meta["view_bytes"].append(size)
    verify(inputs)
    verify(sources)
    meta["build_wall_seconds"] = time.perf_counter()-started
    meta["setup_seconds"] = meta["build_wall_seconds"] - sum(meta["view_seconds"])
    meta["build_peak_vram_gib"] = torch.cuda.max_memory_allocated()/2**30
    dump(root / "manifest.json", meta)
    print(f"BUILD DONE bytes={sum(meta['view_bytes'])} parity_max={max(meta['parity']):.6g}", flush=True)

def train_worker(a):
    import torch
    from scripts import d1_train_cached_detector as d
    from online_experiments import d1_online_compare as pilot
    from online_experiments.fusion_probe import configure_fusion

    torch.set_num_threads(a.threads)
    meta = read(a.job / "view-cache/manifest.json")
    verify(meta["sources"])
    verify(meta["inputs"])
    cfg = read(a.job / "reference_args.json")
    cfg.update(mode="offline", augment="none", parity_only=False, views=a.views,
               seed=a.seed, epochs=a.epochs, device="cuda:0",
               output=a.job / a.phase / f"cache{a.views}-s{a.seed}")
    cfg["cache"] = Path(cfg["cache"])
    if torch.__version__ != meta["torch_version"]:
        raise RuntimeError("PyTorch version changed since cache build")

    def cached_bank(dataset, device):
        if [s["sample_id"] for s in dataset.samples] != meta["split"]["train_sample_ids"]:
            raise RuntimeError("Training cache order changed")
        banks = []
        for view in range(a.views):
            items = []
            for i, sample in enumerate(dataset.samples):
                rel = f"v{view}/{i:06d}.pt"
                content = (a.job / "view-cache" / rel).read_bytes()
                if hashlib.sha256(content).hexdigest() != meta["files"][rel]:
                    raise RuntimeError(f"View-cache checksum mismatch: {rel}")
                item = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
                if item["sample_id"] != sample["sample_id"] or item["image_path"] != sample["image_path"]:
                    raise RuntimeError(f"View-cache sample mismatch: {rel}")
                item["features"] = tuple(x.to(device) for x in item["features"])
                items.append(item)
            banks.append(items)
        return banks

    base, original_write = d.CachedLatentDetector, d.atomic_write_json
    class MeanDetector(base):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            configure_fusion(self, "mean_value")

    def write(path, value):
        name = Path(path).name
        if name == "split.json" and any(value[k] != meta["split"][k] for k in meta["split"]):
            raise RuntimeError("Validation/train split changed")
        if name in ("args.json", "summary.json"):
            value = dict(value)
            cost, size = cache_cost(meta, a.views)
            value.update(method=f"cache{a.views}", detector_fusion="mean_value",
                         augment="hflip" if a.views == 2 else "none",
                         training_teacher_calls=0, training_cache_build_seconds=cost,
                         training_cache_bytes=size, existing_validation_cache_build_included=False)
            if name == "args.json":
                value.update(source_hashes=meta["sources"], cpu_threads=a.threads,
                             view_cache=str(a.job / "view-cache"), visible_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))
            else:
                value["build_plus_run_seconds"] = cost + value["wall_seconds"]
        return original_write(path, value)

    namespace = dict(vars(pilot))
    exec(compile(patched(Path(pilot.__file__).read_text()), pilot.__file__, "exec"), namespace)
    namespace.update(arguments=lambda: argparse.Namespace(**cfg),
                     cached_bank=cached_bank, select_view=select_view)
    d.CachedLatentDetector, d.atomic_write_json = MeanDetector, write
    try:
        namespace["main"]()
    finally:
        d.CachedLatentDetector, d.atomic_write_json = base, original_write

def report(job, old, epochs):
    rows = []
    candidates = [(job / "train" / f"cache{v}-s{s}", "new") for v in (1,2) for s in (0,1,2)]
    candidates += [(old / "train" / f"online-{m}-s{s}", "reference") for m in ("none","hflip") for s in (0,1,2)]
    for folder, origin in candidates:
        if not (folder / "DONE").exists():
            continue
        summary = read(folder / "summary.json")
        if summary["completed_epochs"] != epochs:
            raise RuntimeError(f"Epoch mismatch: {folder}")
        with (folder / "results.csv").open(newline="") as f:
            records = list(csv.DictReader(f))
        if [int(r["epoch"]) for r in records] != list(range(1, epochs+1)):
            raise RuntimeError(f"Incomplete CSV: {folder}")
        rows.append(dict(method=summary["method"], seed=summary["seed"], origin=origin,
            AP=100*summary["best_map50_95"], last_AP=100*summary["final_map50_95"],
            train_s=summary["train_seconds"], wall_s=summary["wall_seconds"],
            build_s=summary.get("training_cache_build_seconds", 0),
            cache_GiB=summary.get("training_cache_bytes", 0)/2**30,
            VRAM_GiB=summary["peak_vram_gib"]))
    text = "AP: 0-100. References are prior completed runs; new cases include fresh training-cache build.\n"
    text += "Existing validation-cache construction excluded for all; cache build counted once per configuration/run, not divided by seeds.\n"
    text += "method seed origin best_AP last_AP train_s build_s build+wall_s train_cache_GiB VRAM_GiB\n"
    for r in rows:
        text += (f"{r['method']} {r['seed']} {r['origin']} {r['AP']:.4f} {r['last_AP']:.4f} "
                 f"{r['train_s']:.1f} {r['build_s']:.1f} {r['build_s']+r['wall_s']:.1f} "
                 f"{r['cache_GiB']:.3f} {r['VRAM_GiB']:.3f}\n")
    text += "\nmethod n AP_mean AP_sample_sd last_AP_mean train_s_mean build+wall_s_mean\n"
    for method in ("cache1","cache2","online-none","online-hflip"):
        group = [r for r in rows if r["method"] == method]
        if group:
            mean = statistics.mean
            sd = f"{statistics.stdev(r['AP'] for r in group):.4f}" if len(group)>1 else "NA"
            text += (f"{method} {len(group)} {mean(r['AP'] for r in group):.4f} {sd} "
                     f"{mean(r['last_AP'] for r in group):.4f} {mean(r['train_s'] for r in group):.1f} "
                     f"{mean(r['build_s']+r['wall_s'] for r in group):.1f}\n")
    dump(job / "comparison.json", rows)
    (job / "comparison.txt").write_text(text)
    return text

def campaign(a):
    old = a.source.resolve()
    if (old / "status.txt").read_text().strip() != "SUCCEEDED":
        raise RuntimeError("Reference campaign must be completed")
    cfg = read(old / "reference_args.json")
    plan = read(old / "plan.json")
    a.epochs, a.threads = plan["epochs"], plan["threads"]
    required = [old / "train" / f"online-{m}-s{s}" for m in ("none","hflip") for s in (0,1,2)]
    ref = read(required[0] / "args.json")
    split = read(required[0] / "split.json")
    for folder in required:
        if not (folder / "DONE").exists():
            raise RuntimeError(f"Missing completed reference: {folder}")
        args = read(folder / "args.json")
        verify(args["source_hashes"])
        keys = ("cache_identity","epochs","batch","lr","weight_decay","aux_weight",
                "teacher_dtype","train_count","val_count","split_seed","detector_fusion","cpu_threads","source_hashes")
        if any(args[k] != ref[k] for k in keys) or read(folder / "split.json") != split:
            raise RuntimeError("Reference settings differ")
    if ref["detector_fusion"] != "mean_value":
        raise RuntimeError("References must use mean_value")
    if any(cfg[k] != ref[k] for k in ("cache_identity","batch","lr","weight_decay","aux_weight",
                                     "teacher_dtype","train_count","val_count","split_seed")):
        raise RuntimeError("Reference snapshot differs from completed runs")
    if ref["epochs"] != a.epochs or ref["cpu_threads"] != a.threads:
        raise RuntimeError("Plan differs from completed runs")
    if (a.job / "reference_args.json").exists():
        raise RuntimeError("Use a fresh job directory")
    dump(a.job / "reference_args.json", cfg)
    dump(a.job / "reference_split.json", split)
    dump(a.job / "baseline_sources.json", ref["source_hashes"])
    patched((ROOT / "online_experiments/d1_online_compare.py").read_text())
    uuid = subprocess.check_output(["nvidia-smi","-i","0","--query-gpu=uuid","--format=csv,noheader"], text=True).strip()
    if not uuid.startswith("GPU-") or len(uuid.splitlines()) != 1:
        raise RuntimeError("Cannot resolve GPU 0 UUID")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, HF_HUB_OFFLINE="1",
               PYTHONUNBUFFERED="1", CUBLAS_WORKSPACE_CONFIG=":4096:8",
               OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads))
    dump(a.job / "plan.json", dict(source_job=str(old), gpu_uuid=uuid, epochs=a.epochs,
                                  threads=a.threads, tasks=[(v,s) for s in (0,1,2) for v in (1,2)]))
    def run(stage, phase="train", views=1, seed=0):
        name = "build" if stage == "build" else f"{phase}-cache{views}-s{seed}"
        (a.job / "status.txt").write_text(f"RUNNING GPU=0 {name}\n")
        command = [sys.executable,"-u",str(Path(__file__).resolve()),"--job",str(a.job),
                   "--stage",stage,"--phase",phase,"--views",str(views),"--seed",str(seed),
                   "--epochs",str(1 if phase=="smoke" else a.epochs),"--threads",str(a.threads)]
        log = a.job / f"{name}.log"
        print(f"START {name}; log={log}", flush=True)
        with log.open("w") as stream:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            print("\n".join(log.read_text(errors="replace").splitlines()[-60:]), flush=True)
            raise RuntimeError(f"Stopped after failure: {log}")
        if stage == "train":
            folder = a.job / phase / f"cache{views}-s{seed}"
            summary = read(folder / "summary.json")
            if summary["completed_epochs"] != (1 if phase=="smoke" else a.epochs):
                raise RuntimeError(f"Incomplete run: {folder}")
            (folder / "DONE").write_text("exit=0\n")
        print(f"END {name}", flush=True)
    run("build")
    for views in (1,2):
        run("train","smoke",views)
    for seed in (0,1,2):
        for views in (1,2):
            run("train","train",views,seed)
            report(a.job, old, a.epochs)
    (a.job / "status.txt").write_text("SUCCEEDED\n")
    print(report(a.job, old, a.epochs), flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--job", type=Path, required=True)
    p.add_argument("--source", type=Path)
    p.add_argument("--stage", choices=["run","build","train"], default="run")
    p.add_argument("--phase", choices=["smoke","train"], default="train")
    p.add_argument("--views", type=int, choices=[1,2], default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    args.job = args.job.resolve()
    args.job.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    try:
        {"run":campaign, "build":build, "train":train_worker}[args.stage](args)
    except BaseException as exc:
        if args.stage == "run":
            (args.job / "status.txt").write_text(f"FAILED: {exc}\n")
        raise
PY

bash <<'BASH'
set -euo pipefail
cd ~/YOLO-Master
SOURCE="$(cat runs/paper/mean-retry/latest-job.txt)"
test -f "$SOURCE/plan.json"
test "$(tr -d '\r\n' < "$SOURCE/status.txt")" = "SUCCEEDED"
python -m py_compile online_experiments/flip_cache_probe.py
mkdir -p runs/paper/flip-cache
JOB="$(mktemp -d "$PWD/runs/paper/flip-cache/job-XXXXXXXX")"
printf '%s\n' "$JOB" > runs/paper/flip-cache/latest-job.txt
printf 'QUEUED\n' > "$JOB/status.txt"
nohup python -u online_experiments/flip_cache_probe.py \
    --source "$SOURCE" --job "$JOB" \
    > "$JOB/job.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$JOB/pid.txt"
printf 'JOB=%s\nPID=%s\n' "$JOB" "$(cat "$JOB/pid.txt")"
BASH
