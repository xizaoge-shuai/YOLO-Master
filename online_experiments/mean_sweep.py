#!/usr/bin/env python3
"""Mean-fusion cached/online/mixed pilot; one worker per selected GPU."""
import argparse
import ast
import csv
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "offline": (0., "none"),
    "online-none": (1., "none"),
    "online-hflip": (1., "hflip"),
    "online-geom": (1., "geom"),
    "mix10": (.10, "geom"),
    "mix25": (.25, "geom"),
    "mix50": (.50, "geom"),
}

class Route:
    def __init__(self, fraction, batches_per_epoch, seed, augment):
        self.p, self.n, self.seed, self.augment = fraction, batches_per_epoch, seed, augment
        self.batches = self.online_batches = self.online_samples = self.total_samples = 0

    def choose(self, size, aug_rng):
        epoch, step = divmod(self.batches, self.n)
        if step == 0:
            k = math.floor((epoch + 1) * self.n * self.p + 1e-9) - math.floor(epoch * self.n * self.p + 1e-9)
            order = list(range(self.n))
            random.Random(self.seed + 15485863 + epoch * 1000003).shuffle(order)
            self.selected = set(order[:k])
        use = step in self.selected
        self.batches += 1
        self.total_samples += size
        self.online_batches += int(use)
        self.online_samples += size * int(use)
        if not use:
            # Preserve augmentation draws for the same images across online budgets.
            for _ in range(size):
                if self.augment != "none":
                    aug_rng.random()
                if self.augment == "geom":
                    aug_rng.random()
        return use

def patched(source):
    tree = ast.parse(source)
    rules = {
        "train_bank": ("bank(train) if a.mode == 'offline' else None",
                       "bank(train) if a.online_fraction < 1 else None"),
        "source": ("train_bank if self.training else val_bank",
                   "(None if route.choose(len(ids), aug_rng) else train_bank) if self.training else val_bank"),
    }
    hits = dict.fromkeys(rules, 0)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in rules:
                expected, replacement = rules[name]
                if ast.dump(node.value) != ast.dump(ast.parse(expected, mode="eval").body):
                    raise RuntimeError(f"Pilot changed at {name}; inspect source before extending it")
                node.value = ast.parse(replacement, mode="eval").body
                hits[name] += 1
    expected_draws = {
        "flip": "a.augment != 'none' and aug_rng.random() < .5",
        "scale": "aug_rng.uniform(a.scale_min, a.scale_max) if a.augment == 'geom' else 1.0",
    }
    for name, expression in expected_draws.items():
        found = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                 and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                 and n.targets[0].id == name]
        if len(found) != 1 or ast.dump(found[0].value) != ast.dump(ast.parse(expression, mode="eval").body):
            raise RuntimeError(f"Augmentation changed at {name}; inspect pilot")
    if any(count != 1 for count in hits.values()):
        raise RuntimeError(f"Unsupported pilot layout: {hits}")
    return ast.fix_missing_locations(tree)

def worker(a):
    sys.path.insert(0, str(ROOT))
    import torch
    from scripts import d1_train_cached_detector as d
    from online_experiments import d1_online_compare as pilot
    from online_experiments.fusion_probe import configure_fusion

    torch.set_num_threads(a.threads)
    p, augment = CASES[a.worker]
    cfg = json.loads(a.reference.read_text())
    expected_identity = cfg["cache_identity"]
    if expected_identity["layers"] != [3, 7, 11]:
        raise RuntimeError("This campaign expects layers [3, 7, 11]")
    cfg.update(mode="offline" if p == 0 else "online", augment=augment,
               online_fraction=p, seed=a.seed, epochs=a.epochs, device="cuda:0",
               parity_only=False, output=a.job / a.phase / f"{a.worker}-s{a.seed}")
    cfg["cache"] = Path(cfg["cache"])
    route = Route(p, math.ceil(cfg["train_count"] / cfg["batch"]), a.seed, augment)
    base, original_write = d.CachedLatentDetector, d.atomic_write_json

    class MeanDetector(base):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            configure_fusion(self, "mean_value")

    def write(path, value):
        name = Path(path).name
        if name in ("args.json", "summary.json"):
            value = dict(value)
            value.update(method=a.worker, detector_fusion="mean_value",
                         online_fraction=p, mode="hybrid" if 0 < p < 1 else cfg["mode"])
            if name == "args.json":
                if value["cache_identity"] != expected_identity:
                    raise RuntimeError("Cache identity changed from reference")
                value.update(cpu_threads=a.threads, visible_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
                             reference_args=str(a.reference), parallel_sweep=True,
                             cache_policy="training cache resident for p<1; validation cache resident for all",
                             mixed_policy="online batches: exact image augmentation + frozen teacher; remaining batches: original cache + original labels")
                value["source_hashes"] = dict(value["source_hashes"])
                for rel in ("online_experiments/mean_sweep.py", "online_experiments/fusion_probe.py",
                            "online_experiments/d1_online_compare.py"):
                    value["source_hashes"][rel] = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
                props = torch.cuda.get_device_properties(0)
                value["gpu_name"] = props.name
                value["gpu_total_memory"] = props.total_memory
            else:
                expected = value["completed_epochs"] * cfg["train_count"]
                if route.total_samples != expected:
                    raise RuntimeError(f"Routing count mismatch: {route.total_samples} != {expected}")
                value.update(online_batches=route.online_batches, total_batches=route.batches,
                             online_samples=route.online_samples, total_samples=route.total_samples,
                             actual_online_fraction=route.online_samples / route.total_samples,
                             training_teacher_calls=route.online_batches,
                             parity_teacher_calls=min(4, cfg["train_count"]) if p else 0)
        return original_write(path, value)

    tree = patched(Path(pilot.__file__).read_text())
    namespace = dict(vars(pilot))
    exec(compile(tree, pilot.__file__, "exec"), namespace)
    namespace.update(arguments=lambda: argparse.Namespace(**cfg), route=route)
    d.CachedLatentDetector, d.atomic_write_json = MeanDetector, write
    try:
        namespace["main"]()
    finally:
        d.CachedLatentDetector, d.atomic_write_json = base, original_write

def collect(job, epochs):
    records, signature = [], None
    for folder in sorted((job / "train").glob("*")):
        if not (folder / "DONE").exists():
            continue
        s = json.loads((folder / "summary.json").read_text())
        c = json.loads((folder / "args.json").read_text())
        split = json.loads((folder / "split.json").read_text())
        same = {k: c[k] for k in ("cache_identity", "batch", "epochs", "lr", "weight_decay",
                "aux_weight", "teacher_dtype", "train_count", "val_count", "split_seed",
                "source_hashes", "detector_fusion", "cpu_threads")}
        same["split"] = [split["train_sample_ids"], split["val_sample_ids"]]
        if signature is not None and same != signature:
            raise RuntimeError("Run configurations/sources/splits differ; inspect results before comparing")
        signature = same
        with (folder / "results.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))
        if s["completed_epochs"] != epochs or len(rows) != epochs:
            raise RuntimeError(f"Incomplete run marked DONE: {folder}")
        records.append(dict(method=s["method"], seed=s["seed"],
            best_AP=100*s["best_map50_95"], last_AP=100*s["final_map50_95"],
            best_epoch=int(max(rows, key=lambda r: float(r["map50_95"]))["epoch"]),
            online_pct=100*s["actual_online_fraction"], train_s=s["train_seconds"],
            wall_s=s["wall_seconds"], VRAM_GiB=s["peak_vram_gib"]))
    text = "AP: 0-100. Parallel sweep timings; initial cache construction excluded.\n"
    text += "method seed best_AP last_AP best_epoch online_pct train_s wall_s VRAM_GiB\n"
    for r in records:
        text += (f"{r['method']} {r['seed']} {r['best_AP']:.4f} {r['last_AP']:.4f} "
                 f"{r['best_epoch']} {r['online_pct']:.2f} {r['train_s']:.1f} "
                 f"{r['wall_s']:.1f} {r['VRAM_GiB']:.3f}\n")
    text += "\nmethod n best_AP_mean best_AP_sample_sd last_AP_mean train_s_mean\n"
    for method in CASES:
        rs = [r for r in records if r["method"] == method]
        if rs:
            sd = f"{statistics.stdev(r['best_AP'] for r in rs):.4f}" if len(rs) > 1 else "NA"
            text += (f"{method} {len(rs)} {statistics.mean(r['best_AP'] for r in rs):.4f} {sd} "
                     f"{statistics.mean(r['last_AP'] for r in rs):.4f} "
                     f"{statistics.mean(r['train_s'] for r in rs):.1f}\n")
    for name, content in (("comparison.txt", text), ("comparison.json", json.dumps(records, indent=2))):
        tmp = job / (name + ".tmp")
        tmp.write_text(content)
        tmp.replace(job / name)

def pool(a, phase, tasks):
    pending, active, failed, done = list(tasks), {}, [], 0
    total = len(pending)
    try:
        while pending or active:
            for gpu in a.gpus:
                if gpu in active or not pending:
                    continue
                method, seed = pending.pop(0)
                folder = a.job / phase / f"{method}-s{seed}"
                folder.parent.mkdir(parents=True, exist_ok=True)
                log = folder.with_suffix(".log").open("w")
                command = [sys.executable, "-u", str(Path(__file__).resolve()),
                           "--reference", str(a.reference), "--job", str(a.job),
                           "--worker", method, "--seed", str(seed), "--phase", phase,
                           "--epochs", str(1 if phase == "smoke" else a.epochs),
                           "--threads", str(a.threads)]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, HF_HUB_OFFLINE="1",
                           PYTHONUNBUFFERED="1", CUBLAS_WORKSPACE_CONFIG=":4096:8",
                           OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads))
                proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
                active[gpu] = (proc, log, folder)
                print(f"START {phase} {folder.name} GPU={gpu} pid={proc.pid}", flush=True)
            for gpu, (proc, log, folder) in list(active.items()):
                code = proc.poll()
                if code is None:
                    continue
                log.close()
                done += 1
                del active[gpu]
                if code:
                    failed.append(folder.name)
                else:
                    (folder / "DONE").write_text("exit=0\n")
                print(f"END {phase} {folder.name} exit={code} ({done}/{total})", flush=True)
            (a.job / "status.txt").write_text(
                f"RUNNING {phase} finished={done}/{total} failed={len(failed)}\n" +
                "\n".join(f"GPU={g} pid={v[0].pid} {v[2].name}" for g, v in active.items()) + "\n")
            if phase == "train":
                collect(a.job, a.epochs)
            if pending or active:
                time.sleep(3)
    finally:
        for proc, log, folder in active.values():
            if proc.poll() is None:
                proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            log.close()
    if failed:
        raise RuntimeError(f"{phase} failed: {failed}; inspect corresponding .log files")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--worker", choices=CASES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--phase", choices=["smoke", "train"], default="train")
    a = parser.parse_args()
    a.reference, a.job = a.reference.resolve(), a.job.resolve()
    os.chdir(ROOT)
    if min(a.epochs, a.threads) < 1 or len(set(a.gpus)) != len(a.gpus) or len(set(a.seeds)) != len(a.seeds):
        parser.error("epochs/threads must be positive; GPU IDs and seeds must be unique")
    if a.worker:
        worker(a)
        return
    a.job.mkdir(parents=True, exist_ok=True)
    try:
        snapshot = a.job / "reference_args.json"
        if snapshot.exists():
            raise RuntimeError("Use a fresh job directory")
        patched((ROOT / "online_experiments/d1_online_compare.py").read_text())
        snapshot.write_text(a.reference.read_text())
        a.reference = snapshot
        gpu = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
        (a.job / "nvidia-smi-start.txt").write_text(gpu.stdout)
        ids = subprocess.check_output(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True).split()
        if not set(a.gpus) <= set(ids):
            raise RuntimeError(f"Requested GPUs {a.gpus}; available physical IDs {ids}")
        plan = [(method, seed) for seed in a.seeds for method in CASES]
        (a.job / "plan.json").write_text(json.dumps(dict(gpus=a.gpus, epochs=a.epochs,
            threads=a.threads, tasks=plan, mean_fusion=True), indent=2))
        pool(a, "smoke", [(m, a.seeds[0]) for m in CASES])
        pool(a, "train", plan)
        collect(a.job, a.epochs)
        (a.job / "status.txt").write_text("SUCCEEDED\n")
        print((a.job / "comparison.txt").read_text(), flush=True)
    except BaseException:
        (a.job / "status.txt").write_text("FAILED: inspect job.log and smoke/train case logs\n")
        raise

if __name__ == "__main__":
    main()
