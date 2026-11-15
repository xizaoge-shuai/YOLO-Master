"""Run the authorized loopback AutoDL and campus SSH relay outside tool sessions.

Use ``python online_experiments/windows_autodl_relay.py launch`` in PowerShell.
The password crosses a private stdin pipe once and exists only in worker memory.
``--status`` prints nonsecret runtime state; ``--help`` never imports Paramiko.
"""

from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = ROOT / "research/checks/relay-admission-20261115/windows-relay-status.json"
DEPENDENCY_PATH = ROOT / "ccfa-workfiles/paper-baselines-20261107/cache/windows-relay-deps"
PIN = "SHA256:liZ36vNCsNcNdXeWs4f+g5ZIhPM/ZihP834vxs8Ulqc"
AUTODL_HOST = "connect.weste.seetacloud.com"
AUTODL_PORT = 37540
LOCAL_ADDRESS = ("127.0.0.1", 19022)
PROXY_ADDRESS = ("127.0.0.1", 7897)
REVERSE_ADDRESS = ("127.0.0.1", 17897)
NO_WINDOW = 0x08000000
DETACHED = 0x00000008 | 0x00000200 | NO_WINDOW
MAX_CONNECTIONS = 16
WINDOWS = sys.platform == "win32"


class HostKeyMismatch(RuntimeError):
    """The peer did not present the previously verified host key."""


def authenticate(transport, password):
    raw = transport.get_remote_server_key().asbytes()
    fingerprint = "SHA256:" + base64.b64encode(hashlib.sha256(raw).digest()).decode().rstrip("=")
    if fingerprint != PIN:
        raise HostKeyMismatch("AutoDL host key mismatch; authentication refused")
    transport.auth_password("root", password, fallback=False)
    if not transport.is_authenticated():
        raise RuntimeError("AutoDL authentication failed")


def close_stream(stream):
    try:
        stream.shutdown(socket.SHUT_RDWR)
    except (AttributeError, OSError, EOFError):
        pass
    try:
        stream.close()
    except (OSError, EOFError):
        pass


def relay_streams(first, second):
    """Two bounded blocking pumps work with Windows sockets and Paramiko channels."""
    done = threading.Event()
    first.settimeout(30.0)
    second.settimeout(30.0)

    def pump(source, destination):
        try:
            while not done.is_set():
                try:
                    payload = source.recv(64 * 1024)
                except socket.timeout:
                    continue
                if not payload:
                    return
                destination.sendall(payload)
        except (OSError, EOFError):
            pass
        finally:
            done.set()
            close_stream(first)
            close_stream(second)

    backward = threading.Thread(target=pump, args=(second, first), daemon=True)
    backward.start()
    pump(first, second)
    backward.join(3)


class ConnectionPool:
    """Reject overload instead of accumulating connections or unlimited threads."""

    def __init__(self, limit=MAX_CONNECTIONS):
        self.slots = threading.BoundedSemaphore(limit)
        self.lock = threading.Lock()
        self.streams = set()
        self.threads = set()
        self.stopped = False

    def submit(self, first, second_factory):
        def run():
            second = None
            try:
                second = second_factory()
                with self.lock:
                    self.streams.add(second)
                    stopped = self.stopped
                if not stopped:
                    relay_streams(first, second)
            except Exception:  # noqa: BLE001, S110 - redact all third-party credential-bearing errors
                # No exception strings: third-party messages may include credentials.
                pass
            finally:
                close_stream(first)
                if second is not None:
                    close_stream(second)
                with self.lock:
                    self.streams.discard(first)
                    self.streams.discard(second)
                    self.threads.discard(threading.current_thread())
                self.slots.release()

        thread = threading.Thread(target=run, daemon=True)
        with self.lock:
            if self.stopped or not self.slots.acquire(blocking=False):
                close_stream(first)
                return False
            self.streams.add(first)
            self.threads.add(thread)
            # Starting under the same lock prevents close() joining an unstarted thread.
            try:
                thread.start()
            except Exception:
                self.threads.discard(thread)
                self.streams.discard(first)
                self.slots.release()
                close_stream(first)
                raise
        return True

    def close(self):
        with self.lock:
            self.stopped = True
            streams, threads = list(self.streams), list(self.threads)
        for stream in streams:
            close_stream(stream)
        deadline = time.monotonic() + 3
        for thread in threads:
            thread.join(max(0, deadline - time.monotonic()))


class Status:
    ALLOWED = frozenset(
        {
            "state",
            "error",
            "attempts",
            "connected_unix",
            "campus_state",
            "campus_error",
            "campus_exit",
            "campus_pid",
            "campus_attempts",
            "active_connections",
        }
    )

    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.values = {
            "schema": "windows-autodl-relay-v1",
            "pid": os.getpid(),
            "local_forward": "127.0.0.1:19022",
            "reverse_proxy": "127.0.0.1:17897",
            "campus_forward": "127.0.0.1:19822",
            "host_key_sha256": PIN,
            "campus_proxy": "127.0.0.1:7908",
            "depends_on": "Windows host, campus VPN and both SSH connections",
        }

    def update(self, **values):
        if set(values) - self.ALLOWED:
            raise ValueError("Unsupported status field")
        with self.lock:
            self.values.update(values)
            self.values["updated_unix"] = time.time()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(self.values, indent=2) + "\n", encoding="utf-8")
            temporary.replace(self.path)


def listener_socket():
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        listener.bind(LOCAL_ADDRESS)
        listener.listen(16)
        listener.settimeout(1)
        return listener
    except Exception:
        listener.close()
        raise


def listener_available():
    try:
        listener = listener_socket()
    except OSError:
        return False
    listener.close()
    return True


def campus_command():
    return [
        "ssh.exe",
        "-F",
        "NUL",
        "-NT",
        "-i",
        "C:/Users/jiazonghan/.ssh/id_ed25519_yang109_nopass",
        "-o",
        "BatchMode=yes",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
        "-R",
        "127.0.0.1:19822:127.0.0.1:19022",
        "-R",
        "127.0.0.1:7908:127.0.0.1:7897",
        "zhjia@10.102.109.3",
    ]


def campus_reconnect(stop, state):
    attempts = 0
    while not stop.is_set():
        process = None
        attempts += 1
        try:
            state.update(campus_state="CONNECTING", campus_attempts=attempts)
            process = subprocess.Popen(
                campus_command(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=NO_WINDOW,
            )
            state.update(campus_state="SSH_PROCESS_RUNNING", campus_pid=process.pid)
            while process.poll() is None and not stop.wait(1):
                pass
            if stop.is_set() and process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
            state.update(campus_state="RECONNECTING", campus_exit=process.returncode)
        except Exception as error:  # noqa: BLE001 - log error type only to avoid credential disclosure
            state.update(campus_state="RECONNECTING", campus_error=type(error).__name__)
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
        stop.wait(min(30, 2 * attempts))


def read_password(stream):
    payload = stream.read(4097)
    if len(payload) > 4096 or not payload.endswith(b"\n"):
        raise ValueError("A short, single-line credential is required")
    payload = payload[:-1]
    if not payload or b"\r" in payload or b"\n" in payload:
        raise ValueError("A nonempty single-line credential is required")
    return payload.decode("utf-8")


def worker(path, dependency_path=None):
    password = read_password(sys.stdin.buffer)
    sys.stdin.close()
    dependency_path = Path(dependency_path) if dependency_path else DEPENDENCY_PATH
    if dependency_path.is_dir():
        sys.path.insert(0, str(dependency_path.resolve()))
    import paramiko

    # Suppress library logging; no transport exception strings are persisted.
    paramiko_logger = logging.getLogger("paramiko")
    paramiko_logger.addHandler(logging.NullHandler())
    paramiko_logger.propagate = False
    listener = listener_socket()  # Exclusive port remains held across reconnections.
    state, stop = Status(path), threading.Event()
    state.update(state="LISTENING", attempts=0)
    current = {"transport": None, "pool": None}
    current_lock = threading.Lock()

    def accept_local():
        while not stop.is_set():
            try:
                client, address = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            with current_lock:
                transport, pool = current["transport"], current["pool"]
            if transport is None or pool is None or not transport.is_active():
                client.close()
                continue
            pool.submit(
                client,
                lambda transport=transport, address=address: transport.open_channel(
                    "direct-tcpip",
                    ("127.0.0.1", 22),
                    address,
                    timeout=15,
                ),
            )

    local_thread = threading.Thread(target=accept_local, daemon=True)
    campus_thread = threading.Thread(target=campus_reconnect, args=(stop, state), daemon=True)
    local_thread.start()
    campus_thread.start()
    attempts = 0
    try:
        while not stop.is_set():
            transport = network = pool = None
            attempts += 1
            state.update(state="CONNECTING", attempts=attempts, error=None)
            try:
                network = socket.create_connection((AUTODL_HOST, AUTODL_PORT), timeout=20)
                network.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                if hasattr(socket, "SIO_KEEPALIVE_VALS"):
                    network.ioctl(socket.SIO_KEEPALIVE_VALS, (1, 30000, 10000))
                transport = paramiko.Transport(network)
                transport.banner_timeout = 20
                transport.auth_timeout = 20
                transport.get_security_options().key_types = ("ssh-ed25519",)
                transport.start_client(timeout=25)
                authenticate(transport, password)
                transport.set_keepalive(30)
                actual_port = transport.request_port_forward(*REVERSE_ADDRESS)
                if actual_port != REVERSE_ADDRESS[1]:
                    raise RuntimeError("Unexpected reverse port")
                pool = ConnectionPool()
                with current_lock:
                    current.update(transport=transport, pool=pool)
                state.update(state="CONNECTED", connected_unix=time.time())
                while transport.is_active() and not stop.is_set():
                    channel = transport.accept(timeout=1)
                    if channel is not None:
                        pool.submit(channel, lambda: socket.create_connection(PROXY_ADDRESS, timeout=10))
                    with pool.lock:
                        active = len(pool.threads)
                    state.update(active_connections=active)
            except HostKeyMismatch:
                state.update(state="HOST_KEY_REFUSED", error="HostKeyMismatch")
                break
            except paramiko.AuthenticationException:
                state.update(state="AUTHENTICATION_REFUSED", error="AuthenticationException")
                break
            except Exception as error:  # noqa: BLE001 - reconnect without logging credential-bearing messages
                state.update(state="RECONNECTING", error=type(error).__name__)
            finally:
                with current_lock:
                    current.update(transport=None, pool=None)
                if transport is not None:
                    transport.close()
                if pool is not None:
                    pool.close()
                if network is not None:
                    network.close()
            stop.wait(5)
    finally:
        stop.set()
        listener.close()
        local_thread.join(3)
        campus_thread.join(12)
        del password


def wait_until_started(process, path):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Detached relay exited; inspect its nonsecret status")
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if data.get("pid") == process.pid:
                return data
        except (OSError, ValueError):
            pass
        time.sleep(0.2)
    raise RuntimeError("Relay launch not confirmed; inspect status before retrying")


def launch(path, dependency_path=None):
    if not WINDOWS:
        raise RuntimeError("This detached launcher is intended for Windows")
    if not listener_available():
        raise RuntimeError("Port 127.0.0.1:19022 is occupied; duplicate launch refused")
    password = getpass.getpass("AutoDL password (worker memory only): ")
    if not password or len(password.encode("utf-8")) > 4095 or any(char in password for char in "\r\n"):
        raise ValueError("A nonempty single-line credential is required")
    command = [sys.executable, str(Path(__file__).resolve()), "worker", "--status-file", str(Path(path).resolve())]
    if dependency_path:
        command.extend(["--dependency-path", str(Path(dependency_path).resolve())])
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=DETACHED,
            close_fds=True,
            cwd=str(ROOT),
        )
        try:
            process.stdin.write(password.encode("utf-8") + b"\n")
            process.stdin.flush()
        finally:
            process.stdin.close()
    finally:
        del password
    return wait_until_started(process, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("launch", "worker", "status"), default="status")
    parser.add_argument("--status", action="store_true", help="Print nonsecret runtime state")
    parser.add_argument("--status-file", type=Path, default=STATUS_PATH)
    parser.add_argument("--dependency-path", type=Path, help="Existing workspace-only Paramiko installation")
    args = parser.parse_args()
    try:
        if args.status or args.action == "status":
            print(args.status_file.read_text(encoding="utf-8"))
        elif args.action == "launch":
            print(json.dumps(launch(args.status_file, args.dependency_path), indent=2))
        else:
            worker(args.status_file, args.dependency_path)
    except Exception as error:  # noqa: BLE001 - CLI output must never expose credential-bearing exceptions
        print(f"Relay action failed: {type(error).__name__}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
