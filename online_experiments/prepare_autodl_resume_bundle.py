"""Pack two paused, content-bound campus runs for one-file-per-dataset SFTP.

No trainer or source data is changed. Dataset/code members are regular files,
and feature-bank shards retain their original hashes for destination checking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import tarfile
import time
from pathlib import Path

SOURCE_ROOT = Path("/mnt/data/users/zhjia")
TARGET_ROOT = Path("/root/autodl-tmp/campus-mirror")
REPO_NAME = "YOLO-Master-paper-baselines-20261107"
GIB = 1024**3


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def stamp(path):
    value = Path(path).stat()
    return {"bytes": value.st_size, "mtime_ns": value.st_mtime_ns, "ctime_ns": value.st_ctime_ns}


def safe_source(path, root):
    path, root = Path(path), Path(root)
    if not path.is_absolute() or ".." in path.parts:
        raise RuntimeError("Source path must be canonical and absolute: " + str(path))
    if path == root or root not in path.parents:
        raise RuntimeError("Source escapes allowed root: " + str(path))
    resolved = path.resolve(strict=True)
    if root.resolve() not in resolved.parents or not stat.S_ISREG(resolved.stat().st_mode):
        raise RuntimeError("Source must resolve to a regular file within root: " + str(path))
    return path


def add_file(files, path, root, target_root, *, expected_sha=None, expected_stat=None, archive=True):
    path = safe_source(path, root)
    info = stamp(path)
    if expected_stat is not None and info != expected_stat:
        raise RuntimeError("Original source stat changed: " + str(path))
    key, relative = str(path), path.relative_to(root).as_posix()
    if key in files:
        old = files[key]
        if (expected_sha and old.get("sha256") and old["sha256"] != expected_sha) or old["origin_stat"] != info:
            raise RuntimeError("Conflicting source evidence: " + key)
        if expected_sha:
            old["sha256"] = expected_sha
        return old
    row = {
        "path": key,
        "source": key,
        "target": str(target_root / relative),
        "target_relative": relative,
        "sha256": expected_sha,
        "bytes": info["bytes"],
        "origin_stat": info,
        "archive": archive,
    }
    files[key] = row
    return row


class HashReader:
    def __init__(self, stream):
        self.stream, self.digest = stream, hashlib.sha256()

    def read(self, size):
        block = self.stream.read(size)
        self.digest.update(block)
        return block


def write_archive(path, files, root):
    """Read each member once, verifying pinned hashes while tar streams it."""
    with tarfile.open(path, "x", format=tarfile.PAX_FORMAT) as archive:
        for index, row in enumerate(files.values(), 1):
            if not row["archive"]:
                continue
            source = safe_source(row["source"], root)
            if stamp(source) != row["origin_stat"]:
                raise RuntimeError("Source stat changed before archive: " + str(source))
            member = tarfile.TarInfo(source.relative_to(root).as_posix())
            member.size, member.mode = row["bytes"], 0o644
            member.mtime = row["origin_stat"]["mtime_ns"] // 10**9
            with source.open("rb") as stream:
                reader = HashReader(stream)
                archive.addfile(member, reader)
                actual = reader.digest.hexdigest()
            if row["sha256"] and actual != row["sha256"]:
                raise RuntimeError("Original SHA256 differs while archiving: " + str(source))
            if stamp(source) != row["origin_stat"]:
                raise RuntimeError("Source changed during archive: " + str(source))
            row["sha256"] = actual
            if index % 1000 == 0:
                print(json.dumps({"archive": str(path), "files_processed": index}), flush=True)


def capacity(rows, archive_member_bytes, target_free_bytes, reserve_bytes=10 * GIB):
    transport = sum(row["bytes"] for row in rows)
    required = transport + archive_member_bytes + reserve_bytes
    if required > target_free_bytes:
        raise RuntimeError(f"Insufficient target space: need {required} bytes, free {target_free_bytes}")
    return {
        "transport_bytes": transport,
        "unpacked_member_bytes": archive_member_bytes,
        "reserve_bytes": reserve_bytes,
        "required_target_bytes": required,
        "reported_target_free_bytes": target_free_bytes,
    }


def check_paused(pid, start_ticks):
    folder = Path("/proc") / str(pid)
    fields = (folder / "stat").read_text().rsplit(")", 1)[1].split()
    if fields[0] not in {"T", "t"} or int(fields[19]) != start_ticks or folder.stat().st_uid != os.getuid():
        raise RuntimeError("Exact migration worker must remain paused: " + str(pid))


def collect(args, dataset, job, output):
    files, root = {}, args.source_root
    provenance = read(job / "provenance.json")
    if provenance["dataset"] != dataset:
        raise RuntimeError("Dataset provenance mismatch")
    config = read(output / "args.json")
    expected_method = {"sku110k": "detail25", "voc0712": "loffta-c2-n005"}[dataset]
    if (config.get("dataset"), config.get("method"), config.get("seed"), config.get("epochs"), config.get("batch")) != (
        dataset,
        expected_method,
        0,
        100,
        8,
    ):
        raise RuntimeError("Only the two authorized fixed-protocol paused runs may migrate")
    if config["provenance_sha256"] != sha(job / "provenance.json"):
        raise RuntimeError("Paused args/provenance binding changed")
    add = lambda path, **kw: add_file(files, path, root, args.target_root, **kw)
    add(job / "provenance.json", expected_sha=config["provenance_sha256"])
    add(job / "input-integrity.json", expected_sha=provenance["input_integrity_sha256"])
    add(provenance["manifest"], expected_sha=provenance["manifest_sha256"])
    add(provenance["identity_file"], expected_sha=provenance["identity_file_sha256"])
    for relative, digest in provenance["source_sha256"].items():
        add(args.repo / relative, expected_sha=digest)
    if "source_anchor" in provenance:
        add(provenance["source_anchor"]["file"], expected_sha=provenance["source_anchor"]["sha256"])
    for path, record in read(job / "input-integrity.json").items():
        add(
            path,
            expected_sha=record["sha256"],
            expected_stat={key: record[key] for key in ("bytes", "mtime_ns", "ctime_ns")},
        )
    manifest = Path(provenance["manifest"])
    seal = read(manifest.parent / "complete.json")
    if seal.get("manifest_sha256") != provenance["manifest_sha256"]:
        raise RuntimeError("Dataset manifest seal changed")
    for path in manifest.parent.glob("*.json"):
        add(path)
    metadata = read(manifest)
    if metadata.get("evaluation_file"):
        add(manifest.parent / metadata["evaluation_file"], expected_sha=seal.get("evaluation_sha256"))
    cache = read(job / "cache/manifest.json")
    seal = read(job / "cache/complete.json")
    if seal["manifest_sha256"] != config["cache_manifest_sha256"]:
        raise RuntimeError("Paused args/cache-manifest binding changed")
    add(job / "cache/manifest.json", expected_sha=config["cache_manifest_sha256"])
    add(job / "cache/complete.json")
    add(job / "cache/teacher-queries.jsonl", expected_sha=cache["teacher_query_journal_sha256"])
    if cache["provenance_sha256"] != sha(job / "provenance.json"):
        raise RuntimeError("Cache/provenance binding changed")
    if set(cache["parts"]) != {"train-v0", "train-v1", "test-v0", "blank"}:
        raise RuntimeError("Expected exactly four original cache shards")
    for part in cache["parts"].values():
        add(part["file"], expected_sha=part["sha256"], expected_stat=part["stat"], archive=False)
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "pipeline.lock" and path.suffix != ".tmp":
            add(path)
    for name in ("args.json", "environment.json", "initialization.json", "summary.json", "weights/last.pt"):
        if str(output / name) not in files:
            raise RuntimeError("Paused output lacks required recovery file: " + name)
    # Canonical logs may be symlinks into the new lane sidecar; store bytes at
    # the canonical alias so the destination watcher can discover them.
    log = output.parent.parent / f"train-{read(output / 'args.json')['method']}.log"
    if log.is_file():
        add(log)
    extras = {
        "online_experiments/prepare_autodl_resume_bundle.py",
        "online_experiments/transfer_autodl_resume.py",
        "online_experiments/resume_migrated_crossdataset.py",
        *args.extra_source_files,
    }
    for name in sorted(extras):
        add(args.repo / name)
    if dataset == "sku110k":
        identity = provenance["identity"]
        model = args.hf_root / "hub" / ("models--" + identity["model_id"].replace("/", "--"))
        snapshot = model / "snapshots" / identity["revision"]
        if not snapshot.is_dir():
            raise RuntimeError("Pinned offline teacher snapshot is missing: " + str(snapshot))
        for path in sorted(snapshot.rglob("*")):
            if path.is_file():
                blob = path.resolve().name
                add(path, expected_sha=blob if re.fullmatch(r"[0-9a-f]{64}", blob) else None)
    return files


def transport_row(dataset, path, target):
    return {
        "dataset": dataset,
        "source": str(path),
        "target": str(target),
        "bytes": path.stat().st_size,
        "sha256": sha(path),
        "origin_stat": stamp(path),
    }


def prepare(args):
    if args.source_root != SOURCE_ROOT or args.target_root != TARGET_ROOT:
        raise RuntimeError("Only the authorized campus root and AutoDL target are supported")
    args.bundle_root = args.bundle_root.absolute()
    if ".." in args.bundle_root.parts or args.source_root not in args.bundle_root.parents:
        raise RuntimeError("Bundle directory must be canonical and within campus source root")
    parent = args.bundle_root.parent.resolve()
    if parent != args.source_root and args.source_root not in parent.parents:
        raise RuntimeError("Bundle parent resolves outside campus source root")
    args.bundle_root.mkdir(parents=True, exist_ok=False)
    try:
        os.nice(10)
    except (AttributeError, OSError):
        pass
    rows, member_bytes = [], 0
    for dataset, job, output, pid, start in (
        ("sku110k", args.sku_job, args.sku_output, args.sku_pid, args.sku_start_ticks),
        ("voc0712", args.voc_job, args.voc_output, args.voc_pid, args.voc_start_ticks),
    ):
        check_paused(pid, start)
        files = collect(args, dataset, job, output)
        archived_bytes = sum(row["bytes"] for row in files.values() if row["archive"])
        estimated_tar = archived_bytes + len(files) * 4096 + 10240
        if shutil.disk_usage(args.bundle_root).free < estimated_tar + GIB:
            raise RuntimeError("Insufficient campus space for non-destructive tar archive")
        member_bytes += archived_bytes
        archive = args.bundle_root / f"{dataset}-inputs-and-code.tar"
        write_archive(archive, files, args.source_root)
        check_paused(pid, start)
        config, summary = read(output / "args.json"), read(output / "summary.json")
        epoch = summary["completed_epochs"]
        if not 0 < epoch < 100 or len(summary["epochs"]) != epoch:
            raise RuntimeError("Paused recovery summary lacks a complete interrupted epoch")
        origin = {
            "repo": str(args.repo),
            "job": str(job),
            "output": str(output),
            "method": config["method"],
            "seed": config["seed"],
            "epoch": epoch,
            "checkpoint_sha256": files[str(output / "weights/last.pt")]["sha256"],
        }
        plan = args.bundle_root / f"{dataset}-plan.json"
        write(
            plan,
            {
                "schema": "migrated-crossdataset-transfer-v1",
                "created_unix": time.time(),
                "origin": origin,
                "files": [{key: value for key, value in row.items() if key != "archive"} for row in files.values()],
                "archive": {"path": str(archive), "sha256": sha(archive), "bytes": archive.stat().st_size},
                "policy": "Frozen source/input SHA256 pinned; feature-bank source stat checked and original SHA retained; destination rechecks every SHA. No source deletion; paused process memory is not a checkpoint.",
            },
        )
        rows.append(transport_row(dataset, archive, args.target_root / ".migration" / archive.name))
        for row in files.values():
            if not row["archive"]:
                rows.append(
                    {
                        "dataset": dataset,
                        **{key: row[key] for key in ("source", "target", "bytes", "sha256", "origin_stat")},
                    }
                )
        rows.append(transport_row(dataset, plan, args.target_root / ".migration" / plan.name))
        print(
            json.dumps(
                {
                    "dataset": dataset,
                    "epoch": epoch,
                    "archive_bytes": archive.stat().st_size,
                    "listed_files": len(files),
                }
            ),
            flush=True,
        )
    disk = capacity(rows, member_bytes, args.target_free_gib * GIB)
    payload = {
        "schema": "autodl-resume-transfer-v1",
        "created_unix": time.time(),
        "files": rows,
        "capacity": disk,
        "source_root": str(args.source_root),
        "target_root": str(args.target_root),
        "extraction": "Only verified regular members, relative to campus source root; extract each dataset tar into target_root before sealing. Original source aliases must map into this target root.",
    }
    destination = args.bundle_root / "transfer-manifest.json"
    write(destination, payload)
    print(json.dumps({"manifest": str(destination), **disk}), flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--target-root", type=Path, default=TARGET_ROOT)
    parser.add_argument("--repo", type=Path, default=SOURCE_ROOT / REPO_NAME)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--hf-root", type=Path, default=SOURCE_ROOT / "cache/huggingface")
    parser.add_argument("--target-free-gib", type=int, default=449)
    parser.add_argument("--extra-source-files", nargs="*", default=[])
    priority = SOURCE_ROOT / REPO_NAME / "runs/paper/priority/job-20261112-single-seed"
    old = SOURCE_ROOT / REPO_NAME / "runs/paper/crossdataset"
    parser.add_argument("--sku-job", type=Path, default=old / "job-20261110-sku110k-official-loader-v2")
    parser.add_argument("--sku-output", type=Path, default=priority / "sku110k/train/detail25-s0")
    parser.add_argument("--voc-job", type=Path, default=old / "job-20261110-voc0712-s0-v2")
    parser.add_argument("--voc-output", type=Path, default=priority / "voc0712/train/loffta-c2-n005-s0")
    parser.add_argument("--sku-pid", type=int, default=2105170)
    parser.add_argument("--sku-start-ticks", type=int, default=45384841)
    parser.add_argument("--voc-pid", type=int, default=2300847)
    parser.add_argument("--voc-start-ticks", type=int, default=46203056)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
