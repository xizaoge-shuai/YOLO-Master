"""Probe a pinned localhost SSH relay, or reuse the existing pair launcher."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import stat
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from online_experiments import launch_autodl_transfer_pair as launcher
from online_experiments import transfer_autodl_resume as transfer

PROBE_BYTES = 8 * 1024**2
HOST = "127.0.0.1"
PORT = 19822


def relay_commands(original):
    """Keep every launch argument, changing only the transport endpoint."""
    return [(name, [*argv, "--host", HOST, "--port", str(PORT)], log) for name, argv, log in original()]


def probe_payload(manifest):
    rows = transfer.validate_manifest(json.loads(Path(manifest).read_text(encoding="utf-8")))
    matches = [row for row in rows if row["dataset"] == "sku110k" and Path(row["source"]).name == "train-v0.npy"]
    if len(matches) != 1:
        raise RuntimeError("Expected exactly one SKU train-v0.npy manifest entry")
    row = matches[0]
    before = transfer.source_stat(row)
    with Path(row["source"]).open("rb") as stream:
        payload = stream.read(PROBE_BYTES)
    after = transfer.source_stat(row)
    if before != after or len(payload) != PROBE_BYTES:
        raise RuntimeError("Probe source changed or contains fewer than eight MiB")
    return payload


def write_probe(transport, sftp, payload):
    digest = hashlib.sha256(payload).hexdigest()
    target = launcher.REMOTE_MARKERS + "/relay-probe-" + digest + ".bin"
    transfer.remote_directories(sftp, Path(target).parent.as_posix())
    prior = transfer.remote_stat(sftp, target)
    if prior is not None:
        if not stat.S_ISREG(prior.st_mode) or prior.st_size != len(payload):
            raise RuntimeError("Existing probe destination differs; overwrite refused")
        if transfer.remote_hash(transport, target) != digest:
            raise RuntimeError("Existing probe destination hash differs; overwrite refused")
        return {
            "state": "VERIFIED_EXISTING",
            "bytes": len(payload),
            "sha256": digest,
            "remote": target,
            "seconds": None,
            "MiBs": None,
        }
    started = time.monotonic()
    with sftp.open(target, "wx") as stream:
        stream.set_pipelined(True)
        for offset in range(0, len(payload), 256 * 1024):
            stream.write(payload[offset : offset + 256 * 1024])
    seconds = time.monotonic() - started
    result = transfer.remote_stat(sftp, target)
    if result is None or result.st_size != len(payload) or transfer.remote_hash(transport, target) != digest:
        raise RuntimeError("Probe destination verification failed")
    return {
        "state": "VERIFIED",
        "bytes": len(payload),
        "sha256": digest,
        "remote": target,
        "seconds": seconds,
        "MiBs": len(payload) / 1024**2 / seconds,
    }


def probe():
    payload = probe_payload(launcher.JOB / "bundle-v1/transfer-manifest.json")
    args = argparse.Namespace(host=HOST, port=PORT, username="root", host_key_sha256=launcher.PIN)
    password = getpass.getpass("AutoDL password (connection only): ")
    if not password or any(char in password for char in ("\r", "\n")):
        del password
        raise ValueError("A nonempty single-line password is required")
    try:
        transport, sftp = transfer.connect(args, password)
    finally:
        del password
    try:
        result = write_probe(transport, sftp, payload)
    finally:
        sftp.close()
        transport.close()
    result.update(
        schema="autodl-relay-probe-v1", host=HOST, port=PORT, host_key_sha256=launcher.PIN, measured_unix=time.time()
    )
    path = launcher.JOB / "relay-probe.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return result


def launch():
    original = launcher.commands
    launcher.commands = lambda: relay_commands(original)
    try:
        return launcher.launch_pair()
    finally:
        launcher.commands = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("probe", "launch"))
    args = parser.parse_args()
    try:
        result = probe() if args.action == "probe" else launch()
    except Exception as error:  # noqa: BLE001 - never expose authentication exception text.
        print(json.dumps({"state": "FAILED", "error_type": type(error).__name__}), flush=True)
        return 1
    if args.action == "launch":
        result = {
            "state": result["state"],
            "children": [{"name": row["name"], "pid": row["pid"]} for row in result["children"]],
        }
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
