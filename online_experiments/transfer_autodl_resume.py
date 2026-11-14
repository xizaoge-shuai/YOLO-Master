"""Single-worker, pinned-host SFTP migration; password is one stdin line only.

An explicit immutable manifest controls every source/destination. Original data
is never removed. SHA-bound partial files survive disconnects for a later run.
Paramiko is imported only for the real connection, not CPU contract checks.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import hmac
import json
import math
import os
import re
import shlex
import socket
import stat
import sys
import time
from pathlib import Path, PurePosixPath

SCHEMA = "autodl-resume-transfer-v1"
SOURCE_ROOT = PurePosixPath("/mnt/data/users/zhjia")
TARGET_ROOT = PurePosixPath("/root/autodl-tmp/campus-mirror")
DATASETS = ("sku110k", "voc0712")
CHUNK_BYTES = 8 * 1024**2


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def checked_path(value, root):
    if not isinstance(value, str) or any(char in value for char in ("\x00", "\n", "\r", "\\")):
        raise ValueError("Invalid manifest path")
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or path == root or not path.is_relative_to(root):
        raise ValueError("Manifest path escapes its allowed root")
    if str(path) != value:
        raise ValueError("Manifest paths must be canonical absolute POSIX paths")
    return path


def validate_manifest(payload):
    if isinstance(payload, dict):
        if payload.get("schema") != SCHEMA:
            raise ValueError("Unknown transfer manifest schema")
        payload = payload.get("files")
    if not isinstance(payload, list) or not payload:
        raise ValueError("Transfer manifest must contain files")
    rows, targets, previous = [], {}, -1
    for original in payload:
        if not isinstance(original, dict):
            raise ValueError("Invalid transfer file record")  # noqa: TRY004 - consistent manifest validation contract.
        row = {key: original[key] for key in ("dataset", "source", "target", "bytes", "sha256")}
        if row["dataset"] not in DATASETS or DATASETS.index(row["dataset"]) < previous:
            raise ValueError("Dataset groups must be contiguous and ordered SKU then VOC")
        previous = DATASETS.index(row["dataset"])
        checked_path(row["source"], SOURCE_ROOT)
        checked_path(row["target"], TARGET_ROOT)
        if type(row["bytes"]) is not int or row["bytes"] < 0:
            raise ValueError("File bytes must be a nonnegative integer")
        if not isinstance(row["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise ValueError("File sha256 must be 64 lowercase hexadecimal characters")
        if "origin_stat" in original:
            if not isinstance(original["origin_stat"], dict):
                raise ValueError("origin_stat must be a dictionary")
            row["origin_stat"] = dict(original["origin_stat"])
        if row["target"] in targets:
            prior = targets[row["target"]]
            if prior["dataset"] == row["dataset"] or any(
                prior[key] != row[key] for key in ("source", "bytes", "sha256")
            ):
                raise ValueError("Duplicate destination conflicts with its first immutable file record")
        else:
            targets[row["target"]] = row
        rows.append(row)
    return rows


def host_fingerprint(key):
    return "SHA256:" + base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip("=")


def verify_host_key(transport, expected):
    if not re.fullmatch(r"SHA256:[A-Za-z0-9+/]{43}", expected):
        raise ValueError("A pinned OpenSSH SHA256 host fingerprint is required")
    actual = host_fingerprint(transport.get_remote_server_key())
    if not hmac.compare_digest(actual, expected):
        raise RuntimeError("Remote host fingerprint differs; authentication refused")
    return actual


def read_password():
    line = sys.stdin.readline(4097)
    if not line.endswith("\n") or len(line) > 4096 or not line.rstrip("\r\n"):
        raise ValueError("Expected one nonempty password line on stdin")
    return line.rstrip("\r\n")


def connect(args, password):
    import paramiko

    transport = None
    connection = socket.create_connection((args.host, args.port), timeout=30)
    try:
        transport = paramiko.Transport(connection)
        transport.banner_timeout = transport.auth_timeout = 30
        transport.get_security_options().key_types = ("ssh-ed25519",)
        transport.start_client(timeout=30)
        verify_host_key(transport, args.host_key_sha256)
        transport.auth_password(args.username, password, fallback=False)
        if not transport.is_authenticated():
            raise RuntimeError("Remote authentication failed")
        handler = getattr(transport, "auth_handler", None)
        if handler is not None and hasattr(handler, "password"):
            handler.password = None
        transport.set_keepalive(30)
        return transport, paramiko.SFTPClient.from_transport(transport)
    except BaseException:
        if transport is not None:
            transport.close()
        connection.close()
        raise
    finally:
        if transport is not None:
            handler = getattr(transport, "auth_handler", None)
            if handler is not None and hasattr(handler, "password"):
                handler.password = None
        del password


def remote_hash(transport, name):
    checked_path(name, TARGET_ROOT)
    channel = transport.open_session(timeout=30)
    try:
        channel.settimeout(3600)
        channel.exec_command("sha256sum -- " + shlex.quote(name))
        with channel.makefile("rb") as stream:
            output = stream.read(4096)
        if channel.recv_exit_status() != 0:
            raise RuntimeError("Remote sha256sum failed")
        value = output.split(maxsplit=1)[0].decode("ascii")
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise RuntimeError("Remote sha256sum returned invalid output")
        return value
    finally:
        channel.close()


def remote_stat(sftp, name):
    try:
        return sftp.lstat(name)
    except OSError as error:
        if isinstance(error, FileNotFoundError) or getattr(error, "errno", None) == 2:
            return None
        raise


def remote_directories(sftp, parent):
    parent = PurePosixPath(parent)
    if parent != TARGET_ROOT and not parent.is_relative_to(TARGET_ROOT):
        raise RuntimeError("Remote parent escapes target root")
    current = PurePosixPath("/")
    for piece in parent.parts[1:]:
        current /= piece
        info = remote_stat(sftp, str(current))
        if info is None:
            sftp.mkdir(str(current))
            info = sftp.lstat(str(current))
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise RuntimeError("Remote parent is not a real directory")


def partial_path(entry):
    return entry["target"] + ".partial." + entry["sha256"][:16]


def source_stat(entry):
    source = Path(entry["source"])
    if not source.is_file() or source.is_symlink():
        raise RuntimeError("Transfer source must be a regular non-symlink file")
    info = source.stat()
    if info.st_size != entry["bytes"]:
        raise RuntimeError("Transfer source byte size changed")
    aliases = {
        "size": "st_size",
        "bytes": "st_size",
        "mtime_ns": "st_mtime_ns",
        "ctime_ns": "st_ctime_ns",
        "inode": "st_ino",
        "device": "st_dev",
    }
    for key, expected in entry.get("origin_stat", {}).items():
        attribute = aliases.get(key, key)
        if (
            attribute in {"st_size", "st_mtime_ns", "st_ctime_ns", "st_ino", "st_dev"}
            and getattr(info, attribute) != expected
        ):
            raise RuntimeError("Transfer source identity changed since manifest construction")
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def transfer_file(entry, sftp, hash_remote, throttle, progress, *, chunk_bytes=CHUNK_BYTES):
    """Validate/resume one immutable file; conflicting artifacts stay untouched."""
    initial = source_stat(entry)
    target, partial = entry["target"], partial_path(entry)
    remote_directories(sftp, PurePosixPath(target).parent)
    existing = remote_stat(sftp, target)
    if existing is not None:
        if (
            not stat.S_ISREG(existing.st_mode)
            or existing.st_size != entry["bytes"]
            or hash_remote(target) != entry["sha256"]
        ):
            raise RuntimeError("Existing final target conflicts; preserved without overwrite")
        return {"reused": True, "resumed_bytes": entry["bytes"], "uploaded_bytes": 0}
    partial_info = remote_stat(sftp, partial)
    offset = partial_info.st_size if partial_info else 0
    if partial_info and (not stat.S_ISREG(partial_info.st_mode) or offset > entry["bytes"]):
        raise RuntimeError("Partial target has invalid type or length")
    digest = hashlib.sha256()
    with Path(entry["source"]).open("rb") as source:
        remaining = offset
        while remaining:
            block = source.read(min(chunk_bytes, remaining))
            if not block:
                raise RuntimeError("Source ended during prefix validation")
            digest.update(block)
            remaining -= len(block)
            throttle(len(block))
        if offset and digest.hexdigest() != hash_remote(partial):
            raise RuntimeError("Partial prefix hash differs; preserved without overwrite")
        uploaded = 0
        with sftp.open(partial, "r+b" if partial_info else "wb") as destination:
            destination.set_pipelined(True)
            destination.seek(offset)
            while block := source.read(chunk_bytes):
                destination.write(block)
                digest.update(block)
                uploaded += len(block)
                throttle(len(block))
                progress(
                    {
                        "target": target,
                        "offset": offset + uploaded,
                        "uploaded_bytes": uploaded,
                        "resumed_bytes": offset,
                        "file_bytes": entry["bytes"],
                    }
                )
    if source_stat(entry) != initial or digest.hexdigest() != entry["sha256"]:
        raise RuntimeError("Source content changed; partial preserved without final rename")
    info = sftp.lstat(partial)
    if info.st_size != entry["bytes"] or hash_remote(partial) != entry["sha256"]:
        raise RuntimeError("Uploaded target hash or byte size differs")
    if remote_stat(sftp, target) is not None:
        raise RuntimeError("Final target appeared concurrently; refusing overwrite")
    sftp.rename(partial, target)
    return {"reused": False, "resumed_bytes": offset, "uploaded_bytes": uploaded}


def dataset_marker(dataset, rows):
    files = [{key: row[key] for key in ("target", "bytes", "sha256")} for row in rows if row["dataset"] == dataset]
    return {
        "schema": SCHEMA,
        "state": "DONE",
        "dataset": dataset,
        "files": len(files),
        "bytes": sum(row["bytes"] for row in files),
        "dataset_manifest_sha256": canonical_hash(files),
    }


def write_remote_json(sftp, target, value):
    checked_path(target, TARGET_ROOT)
    remote_directories(sftp, PurePosixPath(target).parent)
    existing = remote_stat(sftp, target)
    if existing is not None:
        if not stat.S_ISREG(existing.st_mode) or existing.st_size > 1024**2:
            raise RuntimeError("Existing completion marker has invalid type or size")
        with sftp.open(target, "rb") as stream:
            previous = json.loads(stream.read())
        if previous != value:
            raise RuntimeError("Existing completion marker belongs to another manifest")
        return
    partial = target + ".partial"
    info = remote_stat(sftp, partial)
    if info and not stat.S_ISREG(info.st_mode):
        raise RuntimeError("Completion marker partial is not a regular file")
    with sftp.open(partial, "wb") as stream:
        stream.write((json.dumps(value, sort_keys=True, indent=2) + "\n").encode())
    sftp.rename(partial, target)


class RateLimit:
    def __init__(self, mib):
        self.rate = mib * 1024**2
        self.next_read = time.monotonic()

    def __call__(self, amount):
        tick = time.monotonic()
        self.next_read = max(self.next_read, tick) + amount / self.rate
        delay = self.next_read - tick
        if delay > 0:
            time.sleep(delay)


class State:
    def __init__(self, path, rows, manifest_sha256):
        self.path, self.started = Path(path), time.monotonic()
        self.last_write = 0.0
        self.value = {
            "schema": SCHEMA,
            "state": "STARTING",
            "pid": os.getpid(),
            "manifest_sha256": manifest_sha256,
            "files_total": len(rows),
            "files_complete": 0,
            "bytes_total": sum(row["bytes"] for row in rows),
            "verified_bytes": 0,
            "uploaded_bytes": 0,
            "datasets_complete": [],
        }

    def write(self, force=True, **fields):
        self.value.update(fields)
        tick = time.monotonic()
        if not force and tick - self.last_write < 2:
            return
        self.value.update(updated_unix=time.time(), elapsed_seconds=tick - self.started)
        self.value["upload_mib_per_second"] = self.value["uploaded_bytes"] / 1024**2 / max(tick - self.started, 0.001)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(json.dumps(self.value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.path)
        self.last_write = tick


@contextlib.contextmanager
def owner_lock(path):
    import fcntl

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def run(args, rows, state):
    manifest_path = args.manifest.resolve(strict=True)
    if not manifest_path.is_relative_to(Path(str(SOURCE_ROOT))):
        raise RuntimeError("Transfer manifest itself must reside under the allowed source root")
    raw = manifest_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != state.value["manifest_sha256"]:
        raise RuntimeError("Transfer manifest changed after validation")
    password = read_password()
    try:
        transport, sftp = connect(args, password)
    finally:
        del password
    throttle = RateLimit(args.rate_mib)
    try:
        # Self-hashing manifest entries would be circular. Transfer the exact
        # original manifest as metadata before any dataset payload instead.
        meta = {
            "source": str(manifest_path),
            "target": str(PurePosixPath(args.marker_dir) / "manifest.json"),
            "bytes": len(raw),
            "sha256": state.value["manifest_sha256"],
        }
        state.write(state="TRANSFERRING_MANIFEST", target=meta["target"])
        result = transfer_file(meta, sftp, lambda path: remote_hash(transport, path), throttle, lambda _: None)
        state.write(manifest_target=meta["target"], metadata_uploaded_bytes=result["uploaded_bytes"])
        completed = []
        for index, entry in enumerate(rows):
            source = Path(entry["source"]).resolve(strict=True)
            if not source.is_relative_to(Path(str(SOURCE_ROOT))):
                raise RuntimeError("Resolved source escapes allowed root")
            prior_uploaded = state.value["uploaded_bytes"]
            state.write(
                state="TRANSFERRING",
                dataset=entry["dataset"],
                file_index=index + 1,
                source=entry["source"],
                target=entry["target"],
                file_bytes=entry["bytes"],
                file_offset=0,
            )

            def progress(update, prior_uploaded=prior_uploaded):
                state.write(
                    force=False,
                    uploaded_bytes=prior_uploaded + update["uploaded_bytes"],
                    file_offset=update["offset"],
                    resumed_file_bytes=update["resumed_bytes"],
                )

            result = transfer_file(entry, sftp, lambda path: remote_hash(transport, path), throttle, progress)
            state.write(
                files_complete=index + 1,
                verified_bytes=state.value["verified_bytes"] + entry["bytes"],
                uploaded_bytes=prior_uploaded + result["uploaded_bytes"],
                file_offset=entry["bytes"],
                last_file_reused=result["reused"],
            )
            if index + 1 == len(rows) or rows[index + 1]["dataset"] != entry["dataset"]:
                dataset = entry["dataset"]
                marker = str(PurePosixPath(args.marker_dir) / (dataset + ".DONE.json"))
                write_remote_json(sftp, marker, dataset_marker(dataset, rows))
                completed.append(dataset)
                state.write(datasets_complete=list(completed), last_dataset_marker=marker)
        write_remote_json(
            sftp,
            str(PurePosixPath(args.marker_dir) / "ALL.DONE.json"),
            {
                "schema": SCHEMA,
                "state": "DONE",
                "manifest_sha256": state.value["manifest_sha256"],
                "datasets": completed,
                "files": len(rows),
                "bytes": state.value["bytes_total"],
            },
        )
        state.write(state="SUCCEEDED")
    finally:
        sftp.close()
        transport.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--host", default="connect.weste.seetacloud.com")
    parser.add_argument("--port", type=int, default=37540)
    parser.add_argument("--username", default="root")
    parser.add_argument("--host-key-sha256", required=True)
    parser.add_argument("--marker-dir", default=str(TARGET_ROOT / ".transfer-markers"))
    parser.add_argument("--rate-mib", type=float, default=32.0)
    args = parser.parse_args()
    if not math.isfinite(args.rate_mib) or not 1 <= args.rate_mib <= 40 or not 1 <= args.port <= 65535:
        parser.error("rate-mib must be 1..40 and port must be 1..65535")
    checked_path(args.marker_dir, TARGET_ROOT)
    raw = args.manifest.read_bytes()
    rows = validate_manifest(json.loads(raw))
    state = State(args.state, rows, hashlib.sha256(raw).hexdigest())
    acquired = False
    try:
        with owner_lock(str(args.state) + ".lock"):
            acquired = True
            state.write()
            if hasattr(os, "nice"):
                os.nice(10)
            run(args, rows, state)
    except Exception as error:  # noqa: BLE001 - redact third-party failures before logging safe diagnostics.
        # Deliberately exclude exception text/tracebacks: third-party errors
        # must never serialize authentication material into logs or state.
        if acquired:
            state.write(state="FAILED", error_type=type(error).__name__)
        print(json.dumps({"state": "FAILED" if acquired else "BUSY", "error_type": type(error).__name__}), flush=True)
        return 1
    print(json.dumps({"state": "SUCCEEDED", "files": len(rows)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
