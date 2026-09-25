#!/usr/bin/env python3
"""Paired C2 mix/transport/correction pilot; image-query budgets are explicit."""

import argparse
import ast
import hashlib
import io
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("mix", "transport", "correct")
CASES = [f"c2-{arm}-p{pct}" for pct in (10, 25) for arm in ARMS]


def query_batches(n, pct, epoch, seed):
    """Exact cumulative budget for equal-sized batches; independent routing RNG."""
    if n < 1 or pct not in (10, 25) or epoch < 0:
        raise ValueError("Invalid budget schedule")
    count = ((epoch + 1) * n * pct) // 100 - (epoch * n * pct) // 100
    return set(random.Random(seed + 15485863 + epoch * 1000003).sample(range(n), count))


class ConditionalCorrector(nn.Module):
    """Shared-layer local residual, conditioned on scale, flip, layer and position."""

    def __init__(self, channels, rank=64, seed=0):
        super().__init__()
        # Module creation must not perturb the detector or its training RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 32452843)
            self.net = nn.Sequential(
                nn.Conv2d(channels + 5, rank, 1),
                nn.SiLU(),
                nn.Conv2d(rank, rank, 3, padding=1, groups=rank),
                nn.SiLU(),
                nn.Conv2d(rank, channels, 1),
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, levels, scales, flips):
        outputs = []
        for level, x in enumerate(levels):
            x = x.detach().float()
            b, _, h, w = x.shape
            # No running statistics; calibration uses training queries only.
            energy = x.square().mean((1, 2, 3), keepdim=True).sqrt().clamp_min(1e-4)
            yy = (torch.arange(h, device=x.device, dtype=x.dtype) + 0.5) * (2 / h) - 1
            xx = (torch.arange(w, device=x.device, dtype=x.dtype) + 0.5) * (2 / w) - 1
            yy, xx = torch.meshgrid(yy, xx, indexing="ij")
            cond = torch.stack(
                (
                    x.new_tensor(scales).log(),
                    x.new_tensor(flips),
                    x.new_full((b,), level / max(len(levels) - 1, 1)),
                ),
                dim=1,
            )[:, :, None, None].expand(-1, -1, h, w)
            xy = torch.stack((xx, yy))[None].expand(b, -1, -1, -1)
            residual = self.net(torch.cat((x / energy, xy, cond), dim=1))
            outputs.append(x + residual * energy)
        return tuple(outputs)


def correction_loss(predictions, targets):
    """Per-image/layer relative squared error; detach all teacher values."""
    losses = []
    for predicted, target in zip(predictions, targets):
        target = target.detach().float()
        energy = target.square().mean((1, 2, 3)).clamp_min(1e-8)
        losses.append(((predicted - target).square().mean((1, 2, 3)) / energy).mean())
    return torch.stack(losses).mean()


BATCHES = """
class Batches:
    def __init__(self, training):
        self.training = training
    def __iter__(self):
        if self.training:
            indices = torch.randperm(len(train), generator=generator).tolist()
            begin_epoch(epoch - 1, len(indices), a.batch)
            for start in range(0, len(indices), a.batch):
                ids = indices[start:start+a.batch]
                yield make_training_batch(train_bank, ids, d.collate_cached, aug_rng, extractor, start//a.batch)
            finish_epoch(epoch)
        else:
            for start in range(0, len(val), a.batch):
                ids = list(range(start, min(start+a.batch, len(val))))
                yield d.collate_cached([val_bank[i] for i in ids])
"""


def patch_pilot(source):
    """Keep the verified optimizer/evaluator; patch only train data and checkpoint."""
    from online_experiments.flip_cache_probe import patched

    tree = patched(source)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    hits = [0, 0]
    for i, node in enumerate(main.body):
        if isinstance(node, ast.ClassDef) and node.name == "Batches":
            main.body[i] = ast.parse(BATCHES).body[0]
            hits[0] += 1
    for node in ast.walk(main):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "checkpoint" for t in node.targets):
            if not isinstance(node.value, ast.Dict):
                raise RuntimeError("Checkpoint layout changed")
            node.value.keys.append(ast.Constant("corrector"))
            node.value.values.append(ast.parse("correction_checkpoint()", mode="eval").body)
            hits[1] += 1
    if hits != [1, 1]:
        raise RuntimeError(f"Unsupported pilot layout: {hits}")
    return ast.fix_missing_locations(tree)


class Runtime:
    def __init__(self, a, cfg, meta, cache_root):
        self.a, self.cfg, self.meta, self.cache_root = a, cfg, meta, cache_root
        _, self.arm, pct = a.method.split("-")
        self.pct = int(pct[1:])
        self.corrector = self.optimizer = None
        self.query_images = self.query_calls = self.samples = self.correction_steps = 0
        self.digest = hashlib.sha256()
        self.query_digest = hashlib.sha256()
        self.corrector_seconds = 0.0
        self.epoch_losses = []
        self.epoch_baselines = []
        self.epoch_residuals = []
        self.dataset = None

    def cached_bank(self, dataset, device):
        from online_experiments.feature_baselines import sha

        self.dataset = dataset
        if [s["sample_id"] for s in dataset.samples] != self.meta["split"]["train_sample_ids"]:
            raise RuntimeError("Training split differs from shared view cache")
        # Verify all raw images, including images not drawn as queries in smoke.
        for sample in dataset.samples:
            if sha(dataset.dataset_root / sample["image_path"]) != sample["image_sha256"]:
                raise RuntimeError(f"Raw training image changed: {sample['image_path']}")
        banks = []
        for view in (0, 1):
            bank = []
            for i, sample in enumerate(dataset.samples):
                rel = f"v{view}/{i:06d}.pt"
                payload = (self.cache_root / rel).read_bytes()
                if hashlib.sha256(payload).hexdigest() != self.meta["files"][rel]:
                    raise RuntimeError(f"Corrupt view cache: {rel}")
                item = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
                if item["sample_id"] != sample["sample_id"] or item["image_path"] != sample["image_path"]:
                    raise RuntimeError(f"View cache identity mismatch: {rel}")
                item["features"] = tuple(x.to(device) for x in item["features"])
                bank.append(item)
            banks.append(bank)
        if self.arm == "correct":
            channels = {x.shape[0] for x in banks[0][0]["features"]}
            if len(channels) != 1:
                raise RuntimeError("This shared corrector requires equal layer widths")
            self.corrector = ConditionalCorrector(channels.pop(), self.a.rank, self.a.seed).to(device)
            self.optimizer = torch.optim.AdamW(self.corrector.parameters(), lr=self.a.corrector_lr, weight_decay=1e-4)
        return banks

    def begin_epoch(self, epoch, n, batch):
        # A tail batch would make a batch quota differ from an image quota.
        if n % batch:
            raise ValueError("Pilot requires equal-sized batches for an exact image budget")
        self.epoch = epoch
        self.queries = query_batches(n // batch, self.pct, epoch, self.a.seed)
        self.epoch_losses, self.epoch_baselines, self.epoch_residuals = [], [], []

    def make_batch(self, banks, ids, collate, rng, extractor, batch_index):
        from online_experiments.d1_online_compare import geometry, training_feature
        from online_experiments.feature_baselines import transform_targets, warp_features
        from ultralytics.data.foundation_cache import load_letterboxed_tensor

        flips = []
        scales = []
        for _ in ids:
            flips.append(rng.random() < 0.5)
            scales.append(rng.uniform(self.cfg["scale_min"], self.cfg["scale_max"]))
        queried = batch_index in self.queries
        record = [self.epoch, batch_index, [banks[0][i]["sample_id"] for i in ids], flips, scales, queried]
        payload = (json.dumps(record, separators=(",", ":")) + "\n").encode()
        self.digest.update(payload)
        if queried:
            self.query_digest.update(payload)
        with (self.cfg["output"] / "schedule.jsonl").open("ab") as stream:
            stream.write(payload)
        items = [dict(banks[int(flip)][i]) for i, flip in zip(ids, flips)]
        if queried or self.arm != "mix":
            for item, scale in zip(items, scales):
                item["cls"], item["bboxes"] = transform_targets(
                    item["cls"], item["bboxes"], False, scale, int(self.meta["identity"]["imgsz"])
                )
        batch = collate(items)
        transported = None
        if self.arm != "mix":
            transported = tuple(warp_features(x, [False] * len(ids), scales) for x in batch["features"])
        if queried:
            images = []
            for i, flip, scale, item in zip(ids, flips, scales, items):
                sample = self.dataset.samples[i]
                image, info = load_letterboxed_tensor(
                    self.dataset.dataset_root / sample["image_path"], imgsz=int(self.meta["identity"]["imgsz"])
                )
                if info != sample["letterbox"]:
                    raise RuntimeError("Raw-image letterbox differs from cache")
                raw = banks[0][i]
                image, cls, boxes = geometry(image, raw["cls"], raw["bboxes"], flip, scale)
                if not torch.equal(cls, item["cls"]) or not torch.allclose(boxes, item["bboxes"], atol=2e-6, rtol=1e-5):
                    raise RuntimeError("Teacher and cache labels disagree")
                images.append(image)
            result = extractor(torch.stack(images))
            exact = tuple(training_feature(result[f"layer_{layer}"]) for layer in self.meta["identity"]["layers"])
            if not all(torch.isfinite(x).all().item() for x in exact):
                raise RuntimeError("Nonfinite teacher feature")
            self.query_calls += 1
            self.query_images += len(ids)
            if self.corrector is not None:
                torch.cuda.synchronize()
                started = time.perf_counter()
                self.optimizer.zero_grad(set_to_none=True)
                predicted = self.corrector(transported, scales, flips)
                loss = correction_loss(predicted, exact)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite reconstruction loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.corrector.parameters(), 10.0)
                self.optimizer.step()
                with torch.no_grad():
                    baseline = correction_loss(transported, exact)
                    residual = correction_loss(predicted, transported)
                self.epoch_losses.append(float(loss.detach()))
                self.epoch_baselines.append(float(baseline))
                self.epoch_residuals.append(float(residual))
                self.correction_steps += 1
                torch.cuda.synchronize()
                self.corrector_seconds += time.perf_counter() - started
            batch["features"] = exact
        elif self.corrector is not None:
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.no_grad():
                batch["features"] = self.corrector(transported, scales, flips)
            torch.cuda.synchronize()
            self.corrector_seconds += time.perf_counter() - started
        elif transported is not None:
            batch["features"] = transported
        if any(x.requires_grad for x in batch["features"]):
            raise RuntimeError("Detection gradients must not reach corrector or teacher")
        self.samples += len(ids)
        return batch

    def finish_epoch(self, epoch):
        row = dict(epoch=epoch, **self.counters())
        for key, values in (
            ("preupdate_query_relative_mse", self.epoch_losses),
            ("transport_query_relative_mse", self.epoch_baselines),
            ("residual_relative_mse", self.epoch_residuals),
        ):
            row[key] = statistics.mean(values) if values else None
        with (self.cfg["output"] / "correction.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print("BUDGET " + json.dumps(row), flush=True)

    def counters(self):
        return {
            "training_samples": self.samples,
            "training_teacher_calls": self.query_calls,
            "training_teacher_images": self.query_images,
            "training_online_percent": 100 * self.query_images / max(self.samples, 1),
            "setup_teacher_calls": min(4, self.cfg["train_count"]),
            "setup_teacher_images": min(4, self.cfg["train_count"]),
            "total_teacher_images": self.query_images + min(4, self.cfg["train_count"]),
            "correction_steps": self.correction_steps,
            "corrector_seconds": self.corrector_seconds,
            "corrector_parameters": sum(p.numel() for p in self.corrector.parameters()) if self.corrector else 0,
            "schedule_sha256": self.digest.hexdigest(),
            "query_sha256": self.query_digest.hexdigest(),
        }

    def checkpoint(self):
        if self.corrector is None:
            return None
        return {
            "model": self.corrector.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps": self.correction_steps,
            "rank": self.a.rank,
            "lr": self.a.corrector_lr,
        }


def worker(a):
    from online_experiments import d1_online_compare as pilot
    from online_experiments import flip_cache_probe as f
    from online_experiments.feature_baselines import read
    from online_experiments.fusion_probe import configure_fusion
    from scripts import d1_train_cached_detector as d

    torch.set_num_threads(a.threads)
    provenance = read(a.job / "provenance.json")
    f.verify(provenance["sources"])
    cache_root = Path(provenance["source_job"]) / "view-cache"
    meta = read(cache_root / "manifest.json")
    if f.sha(cache_root / "manifest.json") != provenance["view_manifest_sha256"]:
        raise RuntimeError("View manifest changed")
    f.verify(meta["inputs"])
    if torch.__version__ != meta["torch_version"]:
        raise RuntimeError("PyTorch version differs from archived cache")
    cfg = read(a.job / "reference_args.json")
    cfg.update(
        mode="online",
        augment="geom",
        parity_only=False,
        seed=a.seed,
        epochs=a.epochs,
        device="cuda:0",
        output=a.job / a.phase / f"{a.method}-s{a.seed}",
    )
    cfg["cache"] = Path(cfg["cache"])
    runtime = Runtime(a, cfg, meta, cache_root)
    base, original_write = d.CachedLatentDetector, d.atomic_write_json

    class MeanDetector(base):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            configure_fusion(self, "mean_value")

    def write(path, value):
        name = Path(path).name
        if name == "split.json" and any(value[k] != meta["split"][k] for k in meta["split"]):
            raise RuntimeError("Train/validation split changed")
        if name in ("args.json", "summary.json"):
            value = dict(value)
            cost, size = f.cache_cost(meta, 2)
            value.update(
                method=a.method,
                detector_fusion="mean_value",
                online_budget_percent=runtime.pct,
                view_cache=str(cache_root),
                training_cache_bytes=size,
                actual_new_cache_build_seconds=0.0,
                charged_training_cache_build_seconds=cost,
                existing_validation_cache_build_included=False,
                corrector_rank=a.rank,
                corrector_lr=a.corrector_lr,
                corrector_supervision="queried training features only; no detection gradient",
                validation_policy="original cached validation; no correction or calibration",
                source_hashes=provenance["sources"],
                cpu_threads=a.threads,
                visible_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
                augmentation_policy="same flip/scale proposals and queries; mix uses anchor on nonqueries",
            )
            if name == "args.json":
                if value["cache_identity"] != meta["identity"]:
                    raise RuntimeError("Cache identity changed")
            else:
                expected = value["completed_epochs"] * cfg["train_count"]
                expected_queries = (
                    value["completed_epochs"] * (cfg["train_count"] // cfg["batch"]) * runtime.pct // 100
                ) * cfg["batch"]
                if runtime.samples != expected or runtime.query_images != expected_queries:
                    raise RuntimeError("Actual image budget differs from plan")
                value.update(runtime.counters(), build_plus_run_seconds=cost + value["wall_seconds"])
        return original_write(path, value)

    namespace = dict(vars(pilot))
    exec(compile(patch_pilot(Path(pilot.__file__).read_text()), pilot.__file__, "exec"), namespace)  # noqa: S102 - trusted, source-hashed local AST  # noqa: S102 - trusted, source-hashed local AST
    namespace.update(
        arguments=lambda: argparse.Namespace(**cfg),
        cached_bank=runtime.cached_bank,
        begin_epoch=runtime.begin_epoch,
        finish_epoch=runtime.finish_epoch,
        make_training_batch=runtime.make_batch,
        correction_checkpoint=runtime.checkpoint,
    )
    d.CachedLatentDetector, d.atomic_write_json = MeanDetector, write
    try:
        namespace["main"]()
    finally:
        d.CachedLatentDetector, d.atomic_write_json = base, original_write


def report(job, phase="train"):
    from online_experiments.feature_baselines import checked_result, dump, read

    plan = read(job / "plan.json")
    epochs = 1 if phase == "smoke" else plan["epochs"]
    records = []
    for method in CASES:
        for seed in (0, 1, 2):
            folder = job / phase / f"{method}-s{seed}"
            if not (folder / "DONE").exists():
                continue
            s, args, rows = checked_result(folder, epochs)
            if args["source_hashes"] != read(job / "provenance.json")["sources"]:
                raise RuntimeError(f"Source mismatch: {folder}")
            n, batch = args["train_count"], args["batch"]
            expected = (epochs * (n // batch) * s["online_budget_percent"] // 100) * batch
            if s["training_samples"] != n * epochs or s["training_teacher_images"] != expected:
                raise RuntimeError(f"Image quota mismatch: {folder}")
            records.append(
                {
                    "method": method,
                    "seed": seed,
                    "best_AP": 100 * s["best_map50_95"],
                    "last_AP": 100 * s["final_map50_95"],
                    "best_epoch": int(max(rows, key=lambda r: float(r["map50_95"]))["epoch"]),
                    "train_s": s["train_seconds"],
                    "wall_s": s["wall_seconds"],
                    "build_plus_wall_s": s["build_plus_run_seconds"],
                    "online_pct": s["training_online_percent"],
                    "teacher_images": s["training_teacher_images"],
                    "total_teacher_images": s["total_teacher_images"],
                    "correction_s": s["corrector_seconds"],
                    "corrector_parameters": s["corrector_parameters"],
                    "VRAM_GiB": s["peak_vram_gib"],
                    "schedule_sha256": s["schedule_sha256"],
                    "query_sha256": s["query_sha256"],
                }
            )
    for pct in (10, 25):
        for seed in (0, 1, 2):
            group = [r for r in records if r["method"].endswith(f"-p{pct}") and r["seed"] == seed]
            for key in ("schedule_sha256", "query_sha256", "teacher_images"):
                if len({r[key] for r in group}) > 1:
                    raise RuntimeError(f"Unpaired {key}: budget={pct}, seed={seed}")
    lines = [
        "AP: 0-100. 400/100 pilot; original cached validation; not official benchmark.",
        "Fresh query images/transformations paired across arms; reconstruction-only corrector.",
        "C2 training-cache build charged once/run; reused in practice; validation build excluded.",
        "method seed best_AP last_AP best_epoch online_pct query_images train_s correction_s build+wall_s VRAM_GiB",
    ]
    for r in records:
        lines.append(
            f"{r['method']} {r['seed']} {r['best_AP']:.4f} {r['last_AP']:.4f} {r['best_epoch']} "
            f"{r['online_pct']:.2f} {r['teacher_images']} {r['train_s']:.1f} {r['correction_s']:.1f} "
            f"{r['build_plus_wall_s']:.1f} {r['VRAM_GiB']:.3f}"
        )
    lines += ["", "method n best_AP_mean sample_sd last_AP_mean train_s_mean build+wall_s_mean"]
    for method in CASES:
        group = [r for r in records if r["method"] == method]
        if not group:
            continue
        mean = statistics.mean
        sd = f"{statistics.stdev(r['best_AP'] for r in group):.4f}" if len(group) > 1 else "NA"
        lines.append(
            f"{method} {len(group)} {mean(r['best_AP'] for r in group):.4f} {sd} "
            f"{mean(r['last_AP'] for r in group):.4f} {mean(r['train_s'] for r in group):.1f} "
            f"{mean(r['build_plus_wall_s'] for r in group):.1f}"
        )
    lines += ["", "Paired differences (correct minus baseline); positive AP means improvement."]
    for pct in (10, 25):
        for baseline in ("mix", "transport"):
            pairs = []
            for seed in (0, 1, 2):
                mapping = {r["method"]: r for r in records if r["seed"] == seed}
                c, b = mapping.get(f"c2-correct-p{pct}"), mapping.get(f"c2-{baseline}-p{pct}")
                if c and b:
                    pairs.append(
                        (c["best_AP"] - b["best_AP"], c["last_AP"] - b["last_AP"], c["train_s"] / b["train_s"])
                    )
            if pairs:
                lines.append(
                    f"p{pct} vs {baseline}: n={len(pairs)} "
                    f"best_AP_delta={statistics.mean(p[0] for p in pairs):+.4f} "
                    f"last_AP_delta={statistics.mean(p[1] for p in pairs):+.4f} "
                    f"train_ratio={statistics.mean(p[2] for p in pairs):.3f}; "
                    f"per_seed_best_deltas={[round(p[0], 4) for p in pairs]}"
                )
    prefix = "smoke-comparison" if phase == "smoke" else "comparison"
    dump(job / f"{prefix}.json", records)
    (job / f"{prefix}.txt").write_text("\n".join(lines) + "\n")
    return "\n".join(lines)


def campaign(a):
    import fcntl

    from online_experiments import flip_cache_probe as f
    from online_experiments.feature_baselines import dump, read

    lock_path = ROOT / "runs/paper/budget-corrector/.campaign.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    campaign_lock = lock_path.open("a")
    try:
        fcntl.flock(campaign_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Another budget-corrector campaign is running") from None
    source = a.source.resolve()
    if (source / "status.txt").read_text().strip() != "SUCCEEDED":
        raise RuntimeError("Shared flip-cache job must be SUCCEEDED")
    if (a.job / "plan.json").exists():
        raise RuntimeError("Use a fresh job; completed runs are never overwritten")
    meta = read(source / "view-cache/manifest.json")
    f.verify(meta["sources"])
    f.verify(meta["inputs"])
    cfg = read(source / "reference_args.json")
    ref = read(source / "train/cache2-s0/args.json")
    if cfg["train_count"] != 400 or cfg["val_count"] != 100 or cfg["train_count"] % cfg["batch"]:
        raise RuntimeError("This pilot is fixed to 400/100 with equal-sized batches")
    for key in (
        "cache_identity",
        "batch",
        "lr",
        "weight_decay",
        "aux_weight",
        "teacher_dtype",
        "train_count",
        "val_count",
        "split_seed",
    ):
        if cfg[key] != ref[key]:
            raise RuntimeError(f"Reference setting mismatch: {key}")
    for seed in (0, 1, 2):
        folder = source / "train" / f"cache2-s{seed}"
        if not (folder / "DONE").exists():
            raise RuntimeError(f"Incomplete C2 source: {folder}")
        c = read(folder / "args.json")
        f.verify(c["source_hashes"])
        split = read(folder / "split.json")
        if any(split[k] != meta["split"][k] for k in meta["split"]):
            raise RuntimeError("Source split mismatch")
    sources = dict(meta["sources"])
    for name in (
        "budget_corrector.py",
        "feature_baselines.py",
        "run_budget_corrector.sh",
        "archive_budget_corrector.py",
    ):
        sources[f"online_experiments/{name}"] = f.sha(ROOT / "online_experiments" / name)
        if sources[f"online_experiments/{name}"] is None:
            raise RuntimeError(f"Missing source: {name}")
    dump(
        a.job / "provenance.json",
        {
            "source_job": str(source),
            "sources": sources,
            "view_manifest_sha256": f.sha(source / "view-cache/manifest.json"),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        },
    )
    dump(a.job / "reference_args.json", cfg)
    dump(a.job / "reference_split.json", meta["split"])
    patch_pilot((ROOT / "online_experiments/d1_online_compare.py").read_text())
    uuid = subprocess.check_output(
        ["nvidia-smi", "-i", "0", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).strip()
    if not uuid.startswith("GPU-") or "\n" in uuid:
        raise RuntimeError("Cannot identify GPU0")
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=uuid,
        HF_HUB_OFFLINE="1",
        PYTHONUNBUFFERED="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        OMP_NUM_THREADS=str(a.threads),
        MKL_NUM_THREADS=str(a.threads),
    )
    tasks = []
    for seed in (0, 1, 2):
        cases = list(CASES)
        random.Random(49999 + seed).shuffle(cases)
        tasks.extend((method, seed) for method in cases)
    dump(
        a.job / "plan.json",
        {
            "epochs": a.epochs,
            "threads": a.threads,
            "corrector_rank": a.rank,
            "corrector_lr": a.corrector_lr,
            "gpu_uuid": uuid,
            "cases": CASES,
            "tasks": tasks,
            "smoke_only": a.smoke_only,
            "schema_version": 1,
            "note": "Reconstruction-only correction; no tuned hyperparameter selection; pilot not held-out test",
        },
    )

    def run(method, seed, phase, index, total):
        label = f"{phase}-{method}-s{seed}"
        (a.job / "status.txt").write_text(f"RUNNING GPU0 {label} ({index}/{total})\n")
        epochs = 1 if phase == "smoke" else a.epochs
        cmd = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--stage",
            "worker",
            "--job",
            str(a.job),
            "--method",
            method,
            "--seed",
            str(seed),
            "--phase",
            phase,
            "--epochs",
            str(epochs),
            "--threads",
            str(a.threads),
            "--rank",
            str(a.rank),
            "--corrector-lr",
            str(a.corrector_lr),
        ]
        log = a.job / f"{label}.log"
        print(f"START {label} ({index}/{total}) log={log}", flush=True)
        with log.open("w") as stream:
            result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            print("\n".join(log.read_text(errors="replace").splitlines()[-60:]), flush=True)
            raise RuntimeError(f"Stopped on failure: {log}")
        from online_experiments.feature_baselines import checked_result

        folder = a.job / phase / f"{method}-s{seed}"
        summary, _, _ = checked_result(folder, epochs)
        if summary["method"] != method or summary["seed"] != seed:
            raise RuntimeError(f"Worker identity mismatch: {folder}")
        (folder / "DONE").write_text("exit=0\n")
        report(a.job, phase)
        print(f"END {label} wall_s={summary['wall_seconds']:.1f}", flush=True)

    for i, method in enumerate(CASES, 1):
        run(method, 0, "smoke", i, len(CASES))
    if not a.smoke_only:
        for i, (method, seed) in enumerate(tasks, 1):
            run(method, seed, "train", i, len(tasks))
    f.verify(sources)
    f.verify(meta["inputs"])
    (a.job / "status.txt").write_text("SMOKE_SUCCEEDED\n" if a.smoke_only else "SUCCEEDED\n")
    print(report(a.job, "smoke" if a.smoke_only else "train"), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job", type=Path, required=True)
    p.add_argument("--source", type=Path)
    p.add_argument("--stage", choices=("run", "worker", "report"), default="run")
    p.add_argument("--method", choices=CASES, default=CASES[0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--phase", choices=("smoke", "train"), default="train")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--corrector-lr", type=float, default=0.001)
    p.add_argument("--smoke-only", action="store_true")
    a = p.parse_args()
    a.job = a.job.resolve()
    a.job.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    if min(a.epochs, a.threads, a.rank) <= 0 or not math.isfinite(a.corrector_lr) or a.corrector_lr <= 0:
        raise ValueError("Invalid run parameters")
    try:
        if a.stage == "run":
            if a.source is None:
                raise ValueError("--source required")
            campaign(a)
        elif a.stage == "worker":
            worker(a)
        else:
            print(report(a.job, a.phase))
    except BaseException as exc:
        if a.stage == "run":
            (a.job / "status.txt").write_text(f"FAILED: {exc}\n")
        raise


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
