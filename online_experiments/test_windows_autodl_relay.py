"""CPU-only checks for the detached Windows relay; never connect to a server."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import socket
import tempfile
import threading
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

from online_experiments import windows_autodl_relay as relay


class WindowsRelayTests(unittest.TestCase):
    def test_fingerprint_check_refuses_before_password_authentication(self):
        transport = Mock()
        transport.get_remote_server_key.return_value.asbytes.return_value = b"wrong-key"
        with self.assertRaises(relay.HostKeyMismatch):
            relay.authenticate(transport, "secret")
        transport.auth_password.assert_not_called()
        raw = b"fixture-ed25519-key"
        pin = "SHA256:" + base64.b64encode(hashlib.sha256(raw).digest()).decode().rstrip("=")
        transport.get_remote_server_key.return_value.asbytes.return_value = raw
        with patch.object(relay, "PIN", pin):
            relay.authenticate(transport, "secret")
        transport.auth_password.assert_called_once_with("root", "secret", fallback=False)

    def test_bidirectional_bytes_and_peer_close_release_every_thread(self):
        first, left = socket.socketpair()
        right, last = socket.socketpair()
        worker = threading.Thread(target=relay.relay_streams, args=(left, right), daemon=True)
        worker.start()
        try:
            first.settimeout(3)
            last.settimeout(3)
            first.sendall(b"one\x00two")
            self.assertEqual(last.recv(100), b"one\x00two")
            last.sendall(b"return-bytes")
            self.assertEqual(first.recv(100), b"return-bytes")
            first.close()
            worker.join(3)
            self.assertFalse(worker.is_alive())
            self.assertEqual(last.recv(100), b"")
        finally:
            first.close()
            last.close()
            left.close()
            right.close()

    def test_forward_pool_rejects_excess_connections_and_closes_them(self):
        hold = threading.Event()
        pool = relay.ConnectionPool(limit=1)
        left, right = Mock(), Mock()
        self.assertTrue(pool.submit(left, lambda: (hold.wait(3), right)[1]))
        rejected = Mock()
        self.assertFalse(pool.submit(rejected, lambda: Mock()))
        rejected.close.assert_called_once()
        hold.set()
        pool.close()

    def test_status_contains_only_allowlisted_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            state = relay.Status(path)
            state.update(state="CONNECTING", error="OSError", attempts=2)
            data = json.loads(path.read_text())
            self.assertEqual(data["state"], "CONNECTING")
            self.assertEqual(data["local_forward"], "127.0.0.1:19022")
            self.assertNotIn("secret", json.dumps(data))
            with self.assertRaises(ValueError):
                state.update(password="secret")

    def test_launch_uses_pipe_only_for_password(self):
        process = Mock(pid=1234, stdin=Mock())
        with ExitStack() as stack:
            stack.enter_context(patch.object(relay, "WINDOWS", True))
            stack.enter_context(patch.object(relay, "listener_available", return_value=True))
            stack.enter_context(patch.object(relay.getpass, "getpass", return_value="unique-secret"))
            popen = stack.enter_context(patch.object(relay.subprocess, "Popen", return_value=process))
            stack.enter_context(
                patch.object(relay, "wait_until_started", return_value={"pid": 1234, "state": "LISTENING"})
            )
            result = relay.launch(Path("state.json"))
        args, kwargs = popen.call_args
        self.assertNotIn("unique-secret", repr(args))
        self.assertNotIn("env", kwargs)
        self.assertEqual(kwargs["stdin"], relay.subprocess.PIPE)
        self.assertTrue(kwargs["creationflags"] & 0x08000000)
        process.stdin.write.assert_called_once_with(b"unique-secret\n")
        process.stdin.close.assert_called_once()
        self.assertEqual(result["pid"], 1234)

    def test_exclusive_listener_fails_duplicate_without_prompt(self):
        with ExitStack() as stack:
            stack.enter_context(patch.object(relay, "WINDOWS", True))
            stack.enter_context(patch.object(relay, "listener_available", return_value=False))
            prompt = stack.enter_context(patch.object(relay.getpass, "getpass"))
            stack.enter_context(self.assertRaises(RuntimeError))
            relay.launch(Path("state.json"))
        prompt.assert_not_called()

    def test_password_pipe_rejects_empty_or_multiline(self):
        for payload in (b"", b"\n", b"password\nextra\n", b"a\rb\n"):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                relay.read_password(io.BytesIO(payload))
        self.assertEqual(relay.read_password(io.BytesIO(b"value\n")), "value")

    def test_campus_ssh_bind_and_reconnect_arguments(self):
        command = relay.campus_command()
        self.assertIn("127.0.0.1:19822:127.0.0.1:19022", command)
        self.assertIn("127.0.0.1:7908:127.0.0.1:7897", command)
        self.assertIn("ExitOnForwardFailure=yes", command)
        self.assertIn("ServerAliveInterval=30", command)
        self.assertIn("BatchMode=yes", command)
        self.assertEqual(command[-1], "zhjia@10.102.109.3")


if __name__ == "__main__":
    unittest.main()
