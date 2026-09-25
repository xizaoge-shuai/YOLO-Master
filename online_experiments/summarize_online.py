#!/usr/bin/env python3
import json
import sys
from pathlib import Path


def main():
    root = Path(sys.argv[1])
    records = []
    reference = None
    configuration = None

    for path in sorted(root.glob("*/summary.json")):
        split = json.loads(path.with_name("split.json").read_text())
        ids = (split["train_sample_ids"], split["val_sample_ids"])
        args = json.loads(path.with_name("args.json").read_text())
        comparable = {
            k: args[k] for k in [
                "epochs", "batch", "lr", "weight_decay", "aux_weight",
                "cache_identity", "source_hashes", "teacher_dtype",
            ]
        }
        if reference is not None and (
            ids != reference or comparable != configuration
        ):
            raise RuntimeError("Runs differ in splits, source or configuration")

        reference, configuration = ids, comparable
        item = json.loads(path.read_text())
        if item["completed_epochs"] != args["epochs"]:
            raise RuntimeError(f"Incomplete run: {path.parent}")
        records.append(item)

    if not records:
        raise SystemExit("No completed summary.json found")

    offline = {r["seed"]: r for r in records if r["mode"] == "offline"}
    print("AP uses 0-100 scale. Original cache construction is NOT included.")
    print("All groups use the same unaugmented validation cache.")
    print(
        f'{"method":18} {"seed":>4} {"best_AP":>9} {"last_AP":>9} '
        f'{"train_s":>10} {"vs_cache":>9} {"val_s":>9} '
        f'{"wall_s":>10} {"VRAM_GiB":>9}'
    )

    for r in records:
        base = offline.get(r["seed"])
        ratio = (
            r["train_seconds"] / base["train_seconds"]
            if base else float("nan")
        )
        print(
            f'{r["mode"]+"-"+r["augment"]:18} {r["seed"]:4d} '
            f'{100*r["best_map50_95"]:9.4f} '
            f'{100*r["final_map50_95"]:9.4f} '
            f'{r["train_seconds"]:10.1f} {ratio:8.2f}x '
            f'{r["validation_seconds"]:9.1f} '
            f'{r["wall_seconds"]:10.1f} {r["peak_vram_gib"]:9.3f}'
        )


if __name__ == "__main__":
    main()
