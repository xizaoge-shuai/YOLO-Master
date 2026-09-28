"""Full-data gate/foreground ablation; historical training driver stays unchanged."""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from online_experiments import full_visdrone_train as base
from online_experiments.budget_corrector import ConditionalCorrector, correction_loss, foreground_correction_loss
from online_experiments.d1_online_compare import geometry, training_feature
from online_experiments.diagnose_full_corrector import mask_geometry
from online_experiments.feature_baselines import warp_features
from online_experiments.full_visdrone_data import dump, epoch_plan, read, read_feature_batch, sha, verify_sources
from ultralytics.data.foundation_cache import load_letterboxed_tensor

VARIANTS = {"rec25": (False, False), "gate25": (True, False), "fg25": (False, True), "gatefg25": (True, True)}


class Batches(base.Batches):
    """Keep original head batches and query plans; change only residual/loss."""

    def __init__(self, job, method, seed):
        super().__init__(job, "correct25", seed)
        self.variant = method
        gate, self.foreground = VARIANTS[method]
        if gate:
            self.corrector = ConditionalCorrector(384, 64, seed, use_identity_gate=True).cuda()
            self.optimizer = torch.optim.AdamW(self.corrector.parameters(), lr=0.001, weight_decay=1e-4)

    def training(self, epoch, batch_size, output):
        order, all_flips, all_scales, all_queries = epoch_plan(len(self.train), self.pct, epoch, self.seed)
        dump(
            output / "schedules" / f"{epoch + 1:03d}.json",
            {"epoch": epoch + 1, "order": order, "flip": all_flips, "scale": all_scales, "query": all_queries},
        )
        self.reconstruction = []
        for start in range(0, len(order), batch_size):
            ids = order[start : start + batch_size]
            flips, scales = all_flips[start : start + batch_size], all_scales[start : start + batch_size]
            queried = [i for i, selected in enumerate(all_queries[start : start + batch_size]) if selected]
            items = [self.item(self.train[i], f, s) for i, f, s in zip(ids, flips, scales)]
            values = torch.from_numpy(read_feature_batch(self.arrays, ids, [int(f) for f in flips])).cuda().float()
            transported = tuple(warp_features(values[:, l], [False] * len(ids), scales) for l in range(3))
            if queried:
                images, foreground_boxes = [], []
                for j in queried:
                    sample = self.train[ids[j]]
                    image, info = load_letterboxed_tensor(sample["image_path"], imgsz=640)
                    if info != sample["letterbox"]:
                        raise RuntimeError("Online/cache geometry mismatch")
                    raw = self.item(sample)
                    image, cls, boxes = geometry(image, raw["cls"], raw["bboxes"], flips[j], scales[j])
                    if not torch.equal(cls, items[j]["cls"]) or not torch.allclose(
                        boxes, items[j]["bboxes"], atol=2e-6
                    ):
                        raise RuntimeError("Online/cache target mismatch")
                    images.append(image)
                    if self.foreground:
                        foreground_boxes.append(mask_geometry(raw["bboxes"], flips[j], scales[j]))
                extracted = self.extractor(torch.stack(images))
                exact = tuple(training_feature(extracted[f"layer_{l}"]) for l in self.meta["identity"]["layers"])
                if not all(torch.isfinite(x).all() for x in exact):
                    raise RuntimeError("Nonfinite exact feature")
                self.query_calls += 1
                self.query_images += len(queried)
                base.sync()
                tick = time.perf_counter()
                self.optimizer.zero_grad(set_to_none=True)
                prediction = self.corrector(
                    tuple(x[queried] for x in transported), [scales[j] for j in queried], [flips[j] for j in queried]
                )
                loss = (
                    foreground_correction_loss(prediction, exact, foreground_boxes)[0]
                    if self.foreground
                    else correction_loss(prediction, exact)
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite correction loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.corrector.parameters(), 10.0)
                self.optimizer.step()
                self.reconstruction.append(float(loss.detach()))
                self.correction_steps += 1
                base.sync()
                self.correction_seconds += time.perf_counter() - tick
            with torch.no_grad():
                levels = tuple(x.detach() for x in self.corrector(transported, scales, flips))
            if queried:
                levels = tuple(x.clone() for x in levels)
                for level, teacher in zip(levels, exact):
                    level[queried] = teacher
            if any(x.requires_grad for x in levels):
                raise RuntimeError("Detector must not update the corrector")
            for j, item in enumerate(items):
                item["features"] = tuple(x[j] for x in levels)
            yield base.d.collate_cached(items)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--method", choices=VARIANTS, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--legacy", action="store_true", help="Smoke-only original Batches control")
    p.add_argument("--stop-after", type=int, default=0, help="Interrupt after an atomic epoch commit for resume test")
    a = p.parse_args()
    if min(a.epochs, a.batch) < 1:
        p.error("epochs and batch must be positive")
    if (a.legacy or a.stop_after) and not read(a.job / "dataset-manifest.json")["smoke"]:
        p.error("Verification switches are restricted to the smoke dataset")
    verify_sources(read(a.job / "provenance.json"))
    config = {
        "variant": a.method,
        "gate": VARIANTS[a.method][0],
        "foreground": VARIANTS[a.method][1],
        "gate_normalizer": "abs(log(1.15)); inherited candidate pilot, no tuning",
        "foreground_weight": 0.5,
        "foreground_boxes": "all positive-area visible transformed TRAINING boxes",
        "legacy_control": a.legacy,
        "extra_teacher_queries": 0,
        "task_gradient": False,
        "source_snapshot_sha256": sha(a.job / "provenance.json"),
    }
    target = a.output / "variant.json"
    if target.exists() and read(target) != config:
        raise RuntimeError("Variant/source changed on resume")
    dump(target, config)
    if a.legacy:
        a.method = "correct25"
    else:
        base.Batches = Batches
    if a.stop_after:
        original = base.atomic_checkpoint

        def interrupt(path, state):
            original(path, state)
            if path.name == "last.pt" and state["epoch"] == a.stop_after:
                raise SystemExit(77)

        base.atomic_checkpoint = interrupt
    base.train(a)


if __name__ == "__main__":
    main()
