"""Full-data idea adaptations of four supplied papers; no pilot or smoke mode."""

import argparse
import functools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from online_experiments import replication_suite as suite
from online_experiments.detail_probe import dump, read, sha

PAPER_METHODS = (
    "frofa-c2-d01",
    "loffta-c2-n005",
    "cache-reuse-c1-int8",
    "cache-reuse-c1-int4",
    "vit-adapter-cache25",
    "vit-adapter-cache0",
)
PROTOCOL = ROOT / "research/PAPER_BASELINE_PROTOCOL_20261107.md"


@functools.lru_cache(maxsize=1)
def batch_class():
    import torch

    from online_experiments.feature_baselines import warp_features
    from online_experiments.full_visdrone_data import epoch_plan
    from online_experiments.paper_quantized_cache import QuantizedBank
    from scripts import d1_train_cached_detector as d

    class PaperBatches(suite.batch_class()):
        def __init__(self, source, cache, method, seed, train_limit=None, val_limit=None, *, quantized=None):
            if train_limit is not None or val_limit is not None:
                raise ValueError("Paper baselines run the full official split; subset limits are disabled")
            compressed = method.startswith("cache-reuse-")
            super().__init__(source, cache, "cache1" if compressed else method, seed)
            if compressed:
                if quantized is None:
                    raise ValueError("Cache and Reuse requires a completed physical quantized cache")
                self.quantized_bank = QuantizedBank(quantized)
                expected_bits = int(method.rsplit("int", 1)[1])
                if self.quantized_bank.bits != expected_bits or len(self.quantized_bank) != len(self.train):
                    raise RuntimeError("Quantized-bank method or image coverage mismatch")
                original = self.cache["parts"]["train-v0"]
                if self.quantized_bank.manifest["source_train_bank_sha256"] != original["sha256"]:
                    raise RuntimeError("Quantized bank belongs to another original feature cache")
                self.cache = {**self.cache, "quantization": self.quantized_bank.manifest}
                self.arrays = []
                self.suite_method = self.method = method
                self.spec = suite.method_spec(method)
                self.auxiliary_inputs = {
                    "quantized_manifest_sha256": self.quantized_bank.manifest_sha256,
                    "quantized_source_sha256": original["sha256"],
                    "bits": expected_bits,
                    "quantile": self.quantized_bank.manifest["quantile"],
                }

        def training(self, epoch, batch_size, output):
            if not self.suite_method.startswith("cache-reuse-"):
                yield from super().training(epoch, batch_size, output)
                return
            order, flips, scales, query = epoch_plan(len(self.train), 0, epoch, self.seed)
            schedule = {"epoch": epoch + 1, "order": order, "flip": flips, "scale": scales, "query": query}
            path = Path(output) / "schedules" / f"{epoch + 1:03d}.json"
            if path.exists() and read(path) != schedule:
                raise RuntimeError("Existing quantized-cache geometry schedule changed")
            dump(path, schedule)
            for start in range(0, len(order), batch_size):
                ids, fs, ss = (values[start : start + batch_size] for values in (order, flips, scales))
                data = self.quantized_bank[ids]
                if not torch.isfinite(torch.from_numpy(data)).all():
                    raise RuntimeError("Nonfinite dequantized feature")
                values = torch.from_numpy(data).to(self.device)
                levels = tuple(warp_features(values[:, level], fs, ss) for level in range(3))
                items = [self.item(self.train[index], flip, scale) for index, flip, scale in zip(ids, fs, ss)]
                for index, item in enumerate(items):
                    item["features"] = tuple(level[index] for level in levels)
                yield d.collate_cached(items)

    return PaperBatches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("snapshot", "train", "validate"))
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--method", choices=suite.METHODS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quantized", type=Path)
    args = parser.parse_args()
    args.epochs, args.batch, args.smoke = 100, 8, False
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    if args.action == "snapshot":
        suite.snapshot(args)
        return
    if args.method is None:
        parser.error("--method is required")
    args.output = args.output or args.job / "train" / f"{args.method}-s{args.seed}"
    if args.method.startswith("cache-reuse-"):
        if args.quantized is None:
            parser.error("--quantized is required for Cache and Reuse")
        from online_experiments.paper_quantized_cache import verify_quantized_cache

        if args.quantized.is_dir():
            args.quantized = args.quantized / "manifest.json"
        quantized = verify_quantized_cache(args.quantized)
        if quantized["bits"] != int(args.method.rsplit("int", 1)[1]):
            raise RuntimeError("Requested method and physical quantized cache disagree")
        if (args.output / "DONE").exists():
            prior = read(args.output / "args.json")
            if prior["auxiliary_inputs"]["quantized_manifest_sha256"] != sha(args.quantized):
                raise RuntimeError("Completed quantized run belongs to another physical bank")
    if args.action == "validate":
        suite.verify_provenance(args)
        suite.validate_completed(args.output, False, args.method, args.seed, request=args)
    else:
        model = None
        if args.method.startswith("vit-adapter-"):
            from online_experiments.paper_adapter_model import PaperAdapterDetector

            model = PaperAdapterDetector
        suite.train(
            args, batches_factory=functools.partial(batch_class(), quantized=args.quantized), model_factory=model
        )


if __name__ == "__main__":
    main()
