"""CPU-only contracts for the optional SSH transport endpoint wrapper."""

import hashlib
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from online_experiments import autodl_relay_client as relay


class RelayTests(unittest.TestCase):
    def test_probe_reads_exactly_eight_mib_without_mutating_source(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "train-v0.npy"
            content = b"z" * relay.PROBE_BYTES + b"must remain unread"
            source.write_bytes(content)
            manifest = Path(folder) / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            row = {"dataset": "sku110k", "source": str(source), "bytes": len(content)}
            with patch.object(relay.transfer, "validate_manifest", return_value=[row]):
                payload = relay.probe_payload(manifest)
            self.assertEqual(payload, content[: relay.PROBE_BYTES])
            self.assertEqual(source.read_bytes(), content)

    def test_commands_preserve_security_and_resume_arguments(self):
        original = [
            (
                "transfer",
                ["python", "upload.py", "--manifest", "sealed.json", "--host-key-sha256", relay.launcher.PIN],
                "log",
            )
        ]
        result = relay.relay_commands(lambda: original)
        self.assertEqual(result[0][1], original[0][1] + ["--host", "127.0.0.1", "--port", "19822"])
        self.assertNotIn("password", " ".join(result[0][1]).lower())
        self.assertEqual(original[0][1][-1], relay.launcher.PIN)

    def test_launch_retains_existing_guard_and_restores_commands(self):
        original = relay.launcher.commands
        with patch.object(relay.launcher, "launch_pair", side_effect=RuntimeError("guard")), self.assertRaisesRegex(
            RuntimeError, "guard"
        ):
            relay.launch()
        self.assertIs(relay.launcher.commands, original)

    def test_existing_probe_requires_matching_size_and_hash(self):
        payload = b"small CPU fixture"
        info = SimpleNamespace(st_mode=stat.S_IFREG, st_size=len(payload))
        with patch.object(relay.transfer, "remote_directories"), patch.object(
            relay.transfer, "remote_stat", return_value=info
        ), patch.object(relay.transfer, "remote_hash", return_value=hashlib.sha256(payload).hexdigest()):
            result = relay.write_probe(None, None, payload)
        self.assertEqual(result["state"], "VERIFIED_EXISTING")
        self.assertIsNone(result["MiBs"])

    def test_mismatching_probe_never_overwrites(self):
        info = SimpleNamespace(st_mode=stat.S_IFREG, st_size=17)
        with patch.object(relay.transfer, "remote_directories"), patch.object(
            relay.transfer, "remote_stat", return_value=info
        ), self.assertRaisesRegex(RuntimeError, "overwrite refused"):
            relay.write_probe(None, None, b"different")


if __name__ == "__main__":
    unittest.main()
