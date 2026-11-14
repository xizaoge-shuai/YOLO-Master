"""Content-verified AutoDL resume of the two explicitly interrupted full-data runs.

The frozen trainers/config/checkpoints are not rewritten. A disclosed relocation
seal reconciles original stat metadata only for individually verified immutable
files. Actual hardware identities remain separate and costs span both machines.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCHEMA = "migrated-crossdataset-relocation-v1"
TRANSFER_SCHEMA = "migrated-crossdataset-transfer-v1"
ALLOWED = {"loffta-c2-n005": "voc0712", "detail25": "sku110k"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def actual_stat(path):
    value = Path(path).stat()
    return {"bytes": value.st_size, "mtime_ns": value.st_mtime_ns, "ctime_ns": value.st_ctime_ns}


def within(path, root, name):
    path, root = Path(path), Path(root)
    if not path.is_absolute() or ".." in path.parts or (path != root and root not in path.parents):
        raise RuntimeError("File escapes " + name + ": " + str(path))
    return path


def verify_file(record, source_root, target_root):
    source = within(record["path"], source_root, "source root")
    target = source.resolve(strict=True)
    within(target, Path(target_root).resolve(), "target root")
    if "source" in record and Path(record["source"]) != source:
        raise RuntimeError("Transfer source/path disagree")
    if "target" in record and Path(record["target"]).resolve(strict=True) != target:
        raise RuntimeError("Declared target differs from source alias")
    before = actual_stat(target)
    if before["bytes"] != record["bytes"] or record["origin_stat"]["bytes"] != record["bytes"]:
        raise RuntimeError("Transferred bytes differ: " + str(source))
    if sha(target) != record["sha256"]:
        raise RuntimeError("Transferred SHA256 differs: " + str(source))
    if actual_stat(target) != before:
        raise RuntimeError("Target changed while checking SHA256: " + str(source))
    return {
        "path": str(source),
        "realpath": str(target),
        "sha256": record["sha256"],
        "origin_stat": {key: record["origin_stat"][key] for key in ("bytes", "mtime_ns", "ctime_ns")},
        "target_stat": before,
    }


def verify_current(record):
    path = Path(record["path"])
    if str(path.resolve(strict=True)) != record["realpath"] or actual_stat(path) != record["target_stat"]:
        raise RuntimeError("Sealed target changed: " + str(path))


def verified_stamp(records):
    def stamp(path):
        key = str(Path(path).absolute())
        if key not in records:
            raise RuntimeError("Requested input is not sealed for relocation: " + key)
        record = records[key]
        verify_current(record)
        return dict(record["origin_stat"])

    return stamp


def is_runtime_input(path, output):
    path, output = Path(path), Path(output)
    return path == output or output in path.parents


def read_checkpoint(path):
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def check_checkpoint(path, origin, config, started):
    if not started and sha(path) != origin["checkpoint_sha256"]:
        raise RuntimeError("Transferred origin checkpoint hash changed")
    state = read_checkpoint(path)
    if state["args"] != config or not origin["epoch"] <= state["epoch"] <= 100 or len(state["rows"]) != state["epoch"]:
        raise RuntimeError("Checkpoint config, epoch or history changed outside resume contract")
    return state


def preserve_checkpoint(output, expected_hash):
    """Keep the byte-identical origin checkpoint after last.pt advances."""
    output = Path(output)
    backup = output / "weights/origin-migration.pt"
    if not backup.exists():
        source = output / "weights/last.pt"
        if sha(source) != expected_hash:
            raise RuntimeError("Cannot archive changed origin checkpoint")
        temporary = backup.with_suffix(".tmp")
        shutil.copyfile(source, temporary)
        if sha(temporary) != expected_hash:
            raise RuntimeError("Origin checkpoint archive copy failed")
        temporary.replace(backup)
    if sha(backup) != expected_hash:
        raise RuntimeError("Origin checkpoint archive conflicts with verified hash")
    return {
        "path": str(backup),
        "realpath": str(backup.resolve()),
        "sha256": expected_hash,
        "target_stat": actual_stat(backup),
    }


def require_coverage(origin, records):
    """Require every file the frozen integrity guards rely on, not just cache shards."""
    job, output = Path(origin["job"]), Path(origin["output"])
    repo = Path(origin.get("repo", str(job.parents[3])))
    if repo.resolve() != ROOT:
        raise RuntimeError("Origin repository alias does not resolve to this source tree")

    def covered(path, expected_hash=None, expected_stat=None):
        key = str(Path(path))
        if key not in records:
            raise RuntimeError("Required source/input missing from transfer list: " + key)
        row = records[key]
        if expected_hash and row["sha256"] != expected_hash:
            raise RuntimeError("Transfer hash does not match frozen evidence: " + key)
        if expected_stat and row["origin_stat"] != expected_stat:
            raise RuntimeError("Transfer origin stat does not match frozen evidence: " + key)
        return row

    for name in (
        "provenance.json",
        "input-integrity.json",
        "cache/manifest.json",
        "cache/complete.json",
        "cache/teacher-queries.jsonl",
    ):
        covered(job / name)
    provenance = read(job / "provenance.json")
    covered(provenance["manifest"], provenance["manifest_sha256"])
    covered(provenance["identity_file"], provenance["identity_file_sha256"])
    for relative, digest in provenance["source_sha256"].items():
        covered(repo / relative, digest)
    for path, row in read(job / "input-integrity.json").items():
        covered(path, row["sha256"], {key: row[key] for key in ("bytes", "mtime_ns", "ctime_ns")})
    cache = read(job / "cache/manifest.json")
    for row in cache["parts"].values():
        covered(row["file"], row["sha256"], row["stat"])
    for name in ("args.json", "environment.json", "initialization.json", "weights/last.pt"):
        covered(output / name)
    if covered(output / "weights/last.pt")["sha256"] != origin["checkpoint_sha256"]:
        raise RuntimeError("Origin checkpoint is not the declared transferred checkpoint")


def validate_origin(origin, config):
    method = origin.get("method")
    if method not in ALLOWED or config.get("dataset") != ALLOWED[method] or origin.get("seed") != 0:
        raise RuntimeError("Only VOC LOFF-TA and SKU Detail25 seed0 migrations are authorized")
    if (
        config.get("method") != method
        or config.get("seed") != 0
        or config.get("epochs") != 100
        or config.get("batch") != 8
    ):
        raise RuntimeError("Fixed method/seed100/batch8 resume protocol changed")
    if not isinstance(origin.get("epoch"), int) or not 0 < origin["epoch"] < 100:
        raise RuntimeError("Origin must have an interrupted complete epoch between1 and99")


def make_seal(args):
    transfer = read(args.transfer_list)
    if transfer.get("schema") != TRANSFER_SCHEMA:
        raise RuntimeError("Unexpected transfer schema")
    origin = transfer["origin"]
    source_root, target_root = args.source_root.absolute(), args.target_root.resolve(strict=True)
    for key in ("job", "output"):
        within(origin[key], source_root, "source root")
        within(Path(origin[key]).resolve(strict=True), target_root, "target root")
    if args.seal.exists():
        raise RuntimeError("Relocation seal already exists; inspect it rather than replacing evidence")
    records = {}
    for index, row in enumerate(transfer["files"], 1):
        record = verify_file(row, source_root, target_root)
        if record["path"] in records:
            raise RuntimeError("Duplicate transfer path: " + record["path"])
        records[record["path"]] = record
        if index % 500 == 0:
            print(json.dumps({"verified_files": index, "total": len(transfer["files"])}), flush=True)
    require_coverage(origin, records)
    output = Path(origin["output"])
    config = read(output / "args.json")
    validate_origin(origin, config)
    state = check_checkpoint(output / "weights/last.pt", origin, config, started=False)
    needed = {
        "model",
        "optimizer",
        "scheduler",
        "python_rng",
        "numpy_rng",
        "cpu_rng",
        "cuda_rng",
        "totals",
        "summary",
        "query_images",
        "query_calls",
    }
    if not needed <= state.keys():
        raise RuntimeError("Origin checkpoint lacks complete recovery state")
    checkpoint_origin = preserve_checkpoint(output, origin["checkpoint_sha256"])
    immutable = {key: row for key, row in records.items() if not is_runtime_input(key, output)}
    seal = {
        "schema": SCHEMA,
        "created_unix": time.time(),
        "origin": origin,
        "source_root": str(source_root),
        "target_root": str(target_root),
        "transfer_manifest_sha256": sha(args.transfer_list),
        "immutable": immutable,
        "origin_files": {key: row for key, row in records.items() if is_runtime_input(key, output)},
        "args_sha256": sha(output / "args.json"),
        "wrapper_sha256": sha(__file__),
        "source_tree_realpath": str(ROOT),
        "origin_environment_sha256": sha(output / "environment.json"),
        "origin_checkpoint_copy": checkpoint_origin,
        "policy": "Mixed-host continuation; no bitwise identity or controlled speed claim; all files SHA256 verified before stat reconciliation",
    }
    write(args.seal, seal)
    print(json.dumps({"sealed": str(args.seal), "files": len(records), "origin_epoch": origin["epoch"]}), flush=True)


def load_seal(path):
    seal = read(path)
    if (
        seal.get("schema") != SCHEMA
        or seal["wrapper_sha256"] != sha(__file__)
        or seal["source_tree_realpath"] != str(ROOT)
    ):
        raise RuntimeError("Relocation wrapper/source identity changed")
    target_root = Path(seal["target_root"]).resolve(strict=True)
    for row in seal["immutable"].values():
        within(Path(row["realpath"]), target_root, "target root")
        verify_current(row)
    output = Path(seal["origin"]["output"])
    within(output.resolve(strict=True), target_root, "target root")
    if sha(output / "args.json") != seal["args_sha256"]:
        raise RuntimeError("Origin config bytes changed")
    verify_current(seal["origin_checkpoint_copy"])
    return seal


def prepare_environment(output, expected_hash):
    output = Path(output)
    origin, current = output / "environment-origin.json", output / "environment.json"
    if origin.exists():
        if sha(origin) != expected_hash:
            raise RuntimeError("Preserved origin environment hash changed")
        if current.exists() and sha(current) == expected_hash:
            # Recover a crash after making the archive but before removing the
            # old identity. A real destination identity is left untouched.
            current.unlink()
        return
    if not current.is_file() or sha(current) != expected_hash:
        raise RuntimeError("Origin environment is missing or changed")
    shutil.copyfile(current, origin)
    if sha(origin) != expected_hash:
        raise RuntimeError("Origin environment archive copy failed")
    # The copy is verified before removing the old identity. The frozen trainer
    # writes the actual destination identity; no old hardware is impersonated.
    current.unlink()


def annotate_endpoint(output, seal, seal_hash):
    endpoint = read(Path(output) / "endpoint.json")
    endpoint["migration"] = {
        "schema": SCHEMA,
        "origin": seal["origin"],
        "relocation_seal_sha256": seal_hash,
        "environment_origin": "environment-origin.json",
        "environment_current": "environment.json",
        "origin_checkpoint": "weights/origin-migration.pt",
        "timing": "Cumulative checkpoint timings span origin and AutoDL; not a controlled hardware speed comparison",
    }
    write(Path(output) / "endpoint.json", endpoint)
    write(Path(output) / "migration-origin.json", endpoint["migration"])


def execute(args):
    from online_experiments.crossdataset_campaign import job_lock

    seal = load_seal(args.seal)
    origin, output = seal["origin"], Path(seal["origin"]["output"])
    with job_lock(output):
        seal = load_seal(args.seal)
        config = read(output / "args.json")
        validate_origin(origin, config)
        marker = output / "migration-resume.json"
        if marker.exists() and read(marker)["relocation_seal_sha256"] != sha(args.seal):
            raise RuntimeError("Output belongs to another migration seal")
        check_checkpoint(output / "weights/last.pt", origin, config, started=marker.exists())
        if origin["method"] == "detail25":
            from online_experiments.sku_official_loader_suite import install

            base = install()
        else:
            from online_experiments import crossdataset_suite as base
        base.stamp = verified_stamp(seal["immutable"])
        request = SimpleNamespace(job=Path(origin["job"]), output=output, method=origin["method"], seed=0)
        base.verify_provenance(request.job)
        base.cache_manifest(request.job)
        if args.action == "run":
            prepare_environment(output, seal["origin_environment_sha256"])
            progress = (
                read(marker)
                if marker.exists()
                else {
                    "schema": SCHEMA,
                    "relocation_seal_sha256": sha(args.seal),
                    "origin": origin,
                    "started_unix": time.time(),
                    "timing_policy": seal["policy"],
                    "attempts": [],
                }
            )
            progress["attempts"].append({"started_unix": time.time()})
            write(marker, progress)
            base.train(request)
        else:
            if not marker.exists():
                raise RuntimeError("Migration has not started; no final endpoint can be validated")
            base.validate_completed(request)
        annotate_endpoint(output, seal, sha(args.seal))
        base.validate_completed(request)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("seal", "run", "validate"))
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--transfer-list", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--target-root", type=Path)
    args = parser.parse_args()
    if args.action == "seal":
        if any(getattr(args, key) is None for key in ("transfer_list", "source_root", "target_root")):
            parser.error("seal requires --transfer-list, --source-root, and --target-root")
        make_seal(args)
    else:
        execute(args)


if __name__ == "__main__":
    main()
