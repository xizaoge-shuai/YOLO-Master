"""Launch one detached campus upload/return pair with one terminal password.

The password reaches children only through their stdin pipes, never arguments,
environment, launch metadata or files. No training process is started here.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from online_experiments import transfer_autodl_resume as transfer

JOB = ROOT / "runs/paper/migration-autodl/job-20261114-two-lanes"
PIN = "SHA256:liZ36vNCsNcNdXeWs4f+g5ZIhPM/ZihP834vxs8Ulqc"
REMOTE_MARKERS = "/root/autodl-tmp/campus-mirror/.transfer-markers/job-20261114-two-lanes"


def commands():
    shared = ["--host-key-sha256", PIN]
    manifest = JOB / "bundle-v1/transfer-manifest.json"
    return [
        (
            "transfer",
            [
                sys.executable,
                "-u",
                str(ROOT / "online_experiments/run_autodl_upload.py"),
                "--manifest",
                str(manifest),
                "--state",
                str(JOB / "transfer-state.json"),
                "--marker-dir",
                REMOTE_MARKERS,
                "--rate-mib",
                "32",
                *shared,
            ],
            JOB / "transfer.log",
        ),
        (
            "return",
            [
                sys.executable,
                "-u",
                str(ROOT / "online_experiments/return_autodl_results.py"),
                "--job",
                str(JOB),
                "--bundle-root",
                str(JOB / "bundle-v1"),
                "--publish",
                *shared,
            ],
            JOB / "return.log",
        ),
    ]


def child_environment():
    allowed = (
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TERM",
        "SHELL",
        "TZ",
        "TMPDIR",
        "LD_LIBRARY_PATH",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
    )
    env = {name: os.environ[name] for name in allowed if name in os.environ}
    paths = [str(ROOT / ".research-deps/autodl-transfer/site"), str(ROOT)]
    if os.environ.get("PYTHONPATH"):
        paths.append(os.environ["PYTHONPATH"])
    env.update(
        PYTHONPATH=os.pathsep.join(paths),
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        GH_CONFIG_DIR=str(ROOT.parent / ".config/gh"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    # Git publication uses the existing credential-free reverse tunnel.
    # Paramiko connects through native sockets and ignores HTTP proxy variables.
    env.update(http_proxy="http://127.0.0.1:7908", https_proxy="http://127.0.0.1:7908")
    return env


def process_identity(pid):
    folder = Path("/proc") / str(pid)
    try:
        fields = (folder / "stat").read_text().rsplit(")", 1)[1].split()
        argv = (folder / "cmdline").read_bytes().decode(errors="surrogateescape").rstrip("\0").split("\0")
        return {
            "pid": pid,
            "uid": folder.stat().st_uid,
            "start_ticks": int(fields[19]),
            "state": fields[0],
            "argv": argv,
        }
    except FileNotFoundError:
        return None


def probe_lock(path):
    with transfer.owner_lock(path):
        pass


def save(value):
    path = JOB / "launch.json"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def prior_children_inactive():
    path = JOB / "launch.json"
    if not path.is_file():
        return
    prior = json.loads(path.read_text(encoding="utf-8"))
    for row in prior.get("children", []):
        current = process_identity(row["pid"])
        if (
            current
            and current.get("state") not in {"Z", "X"}
            and all(current.get(key) == row.get(key) for key in ("pid", "uid", "start_ticks", "argv"))
        ):
            raise RuntimeError("A previously launched transfer or return child is still active")
    for state_name in ("transfer-state.json", "return-status.json"):
        state = JOB / state_name
        if state.is_file() and json.loads(state.read_text(encoding="utf-8")).get("state") == "SUCCEEDED":
            raise RuntimeError("A completed child exists; do not relaunch the entire pair")


def launch_pair():
    JOB.mkdir(parents=True, exist_ok=True)
    with transfer.owner_lock(JOB / "launcher.lock"):
        prior_children_inactive()
        probe_lock(JOB / "transfer-state.json.lock")
        probe_lock(JOB / "return.lock")
        manifest = JOB / "bundle-v1/transfer-manifest.json"
        manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
        password = getpass.getpass("AutoDL password (stdin pipes only): ")
        if not password or any(char in password for char in ("\n", "\r")):
            del password
            raise ValueError("A nonempty single-line password is required")
        boot_path = Path("/proc/sys/kernel/random/boot_id")
        record = {
            "schema": "autodl-transfer-pair-launch-v1",
            "started_unix": time.time(),
            "manifest_sha256": manifest_sha256,
            "boot_id": boot_path.read_text().strip() if boot_path.exists() else None,
            "children": [],
            "state": "LAUNCHING",
        }
        try:
            save(record)
            for name, argv, log in commands():
                with log.open("ab", buffering=0) as stream:
                    child = subprocess.Popen(
                        argv,
                        cwd=ROOT,
                        env=child_environment(),
                        start_new_session=True,
                        stdin=subprocess.PIPE,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        close_fds=True,
                        text=True,
                    )
                identity = process_identity(child.pid)
                if identity is None or identity.get("state") in {"Z", "X"} or identity.get("argv") != argv:
                    raise RuntimeError("New child exited before its stdin credential handoff")
                identity.update(name=name, argv=argv, log=str(log), started_unix=time.time())
                record["children"].append(identity)
                save(record)
                try:
                    child.stdin.write(password + "\n")
                    child.stdin.flush()
                finally:
                    child.stdin.close()
            record["state"] = "LAUNCHED"
            save(record)
        except Exception as error:  # noqa: BLE001 - redact all credential-handoff failures.
            record.update(state="PARTIAL_LAUNCH_FAILURE", error_type=type(error).__name__)
            save(record)
            raise RuntimeError("Launch failed; inspect nonsecret child state and logs") from None
        finally:
            del password
        return record


def main():
    if sys.platform != "linux":
        print(json.dumps({"state": "FAILED", "error_type": "LinuxRequired"}), flush=True)
        return 1
    try:
        record = launch_pair()
    except Exception as error:  # noqa: BLE001 - the terminal prints error types only.
        print(json.dumps({"state": "FAILED", "error_type": type(error).__name__}), flush=True)
        return 1
    print(
        json.dumps(
            {
                "state": record["state"],
                "children": [{"name": row["name"], "pid": row["pid"]} for row in record["children"]],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
