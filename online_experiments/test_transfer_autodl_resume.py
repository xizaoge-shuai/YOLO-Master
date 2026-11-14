"""Small CPU contracts; no network, credentials or large input files."""

import hashlib
import io
import json
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

try:
    from online_experiments import transfer_autodl_resume as transfer
except ImportError:
    transfer = None


def digest(value):
    return hashlib.sha256(value).hexdigest()


class RemoteFile(io.BytesIO):
    def __init__(self, owner, name, mode):
        self.owner, self.name, self.mode = owner, name, mode
        super().__init__(owner.files.get(name, b"") if "w" not in mode else b"")

    def set_pipelined(self, enabled):
        self.owner.pipelined = enabled

    def close(self):
        if not self.closed and ("w" in self.mode or "+" in self.mode):
            self.owner.files[self.name] = self.getvalue()
        super().close()


class SFTP:
    def __init__(self):
        self.files = {}
        self.dirs = {"/", "/root", "/root/autodl-tmp", "/root/autodl-tmp/campus-mirror"}
        self.links = set()
        self.pipelined = False

    def lstat(self, name):
        if name in self.links:
            return SimpleNamespace(st_mode=stat.S_IFLNK | 0o777, st_size=0)
        if name in self.files:
            return SimpleNamespace(st_mode=stat.S_IFREG | 0o600, st_size=len(self.files[name]))
        if name in self.dirs:
            return SimpleNamespace(st_mode=stat.S_IFDIR | 0o700, st_size=0)
        raise FileNotFoundError(name)

    def mkdir(self, name):
        self.dirs.add(name)

    def open(self, name, mode="r"):
        return RemoteFile(self, name, mode)

    def rename(self, source, target):
        if target in self.files:
            raise FileExistsError(target)
        self.files[target] = self.files.pop(source)


class Contracts(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(transfer, "transfer implementation is missing")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "data.bin"
        self.source.write_bytes(b"abcdefghijklmno")
        self.target = "/root/autodl-tmp/campus-mirror/sku/cache.bin"
        self.entry = {
            "dataset": "sku110k",
            "source": str(self.source),
            "target": self.target,
            "bytes": 15,
            "sha256": digest(self.source.read_bytes()),
        }
        self.sftp = SFTP()
        self.progress = []

    def remote_hash(self, name):
        return digest(self.sftp.files[name])

    def one(self):
        return transfer.transfer_file(
            self.entry, self.sftp, self.remote_hash, lambda _: None, self.progress.append, chunk_bytes=4
        )

    def test_manifest_rejects_escape_and_noncontiguous_dataset(self):
        row = dict(self.entry, source="/mnt/data/users/zhjia/cache/a")
        self.assertEqual(transfer.validate_manifest({"schema": transfer.SCHEMA, "files": [row]})[0], row)
        for field, value in (
            ("source", "/etc/shadow"),
            ("target", "/root/autodl-tmp/campus-mirror/../other"),
            ("target", "/root/autodl-tmp/campus-mirror-sibling/x"),
            ("sha256", "bad"),
        ):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                transfer.validate_manifest([dict(row, **{field: value})])
        with self.assertRaises(ValueError):
            transfer.validate_manifest([dict(row, dataset="voc0712"), row])

    def test_complete_upload_verifies_and_atomically_renames(self):
        result = self.one()
        self.assertEqual(self.sftp.files[self.target], self.source.read_bytes())
        self.assertEqual(result["uploaded_bytes"], 15)
        self.assertTrue(self.sftp.pipelined)
        self.assertNotIn(transfer.partial_path(self.entry), self.sftp.files)

    def test_matching_prefix_resumes_only_remaining_bytes(self):
        self.sftp.files[transfer.partial_path(self.entry)] = b"abcdef"
        result = self.one()
        self.assertEqual(result["resumed_bytes"], 6)
        self.assertEqual(result["uploaded_bytes"], 9)
        self.assertEqual(self.sftp.files[self.target], self.source.read_bytes())

    def test_wrong_prefix_preserves_partial_and_refuses_upload(self):
        partial = transfer.partial_path(self.entry)
        self.sftp.files[partial] = b"BAD"
        with self.assertRaises(RuntimeError):
            self.one()
        self.assertEqual(self.sftp.files[partial], b"BAD")
        self.assertNotIn(self.target, self.sftp.files)

    def test_verified_final_is_reused_without_any_write(self):
        self.sftp.files[self.target] = self.source.read_bytes()
        result = self.one()
        self.assertTrue(result["reused"])
        self.assertEqual(result["uploaded_bytes"], 0)

    def test_conflicting_final_is_never_overwritten(self):
        self.sftp.files[self.target] = b"x" * 15
        with self.assertRaises(RuntimeError):
            self.one()
        self.assertEqual(self.sftp.files[self.target], b"x" * 15)

    def test_remote_hash_failure_retains_partial_without_final(self):
        with self.assertRaises(RuntimeError):
            transfer.transfer_file(
                self.entry, self.sftp, lambda _: "0" * 64, lambda _: None, self.progress.append, chunk_bytes=4
            )
        self.assertNotIn(self.target, self.sftp.files)
        self.assertIn(transfer.partial_path(self.entry), self.sftp.files)

    def test_remote_symlink_parent_is_rejected(self):
        self.sftp.links.add("/root/autodl-tmp/campus-mirror/sku")
        with self.assertRaises(RuntimeError):
            self.one()

    def test_origin_stat_change_refuses_copy(self):
        self.entry["origin_stat"] = {"st_size": 15, "st_mtime_ns": 1}
        with self.assertRaises(RuntimeError):
            self.one()
        self.assertFalse(self.sftp.files)

    def test_origin_bytes_and_ctime_fields_are_enforced(self):
        self.entry["origin_stat"] = {"bytes": 15, "mtime_ns": self.source.stat().st_mtime_ns, "ctime_ns": 1}
        with self.assertRaises(RuntimeError):
            self.one()

    def test_identical_shared_code_can_repeat_in_later_dataset(self):
        row = dict(self.entry, source="/mnt/data/users/zhjia/repo/code.py")
        rows = transfer.validate_manifest([row, dict(row, dataset="voc0712")])
        self.assertEqual(len(rows), 2)
        with self.assertRaises(ValueError):
            transfer.validate_manifest([row, dict(row, dataset="voc0712", sha256="0" * 64)])

    def test_host_key_mismatch_never_sends_password(self):
        key = SimpleNamespace(asbytes=lambda: b"public-key")
        transport = SimpleNamespace(get_remote_server_key=lambda: key)
        with self.assertRaises(RuntimeError):
            transfer.verify_host_key(transport, "SHA256:" + "A" * 43)
        expected = transfer.host_fingerprint(key)
        self.assertEqual(transfer.verify_host_key(transport, expected), expected)

    def test_connection_pins_ed25519_before_authentication(self):
        key = SimpleNamespace(asbytes=lambda: b"trusted-key")
        security = SimpleNamespace(key_types=())
        connection, transport = Mock(), Mock()
        transport.get_security_options.return_value = security
        transport.get_remote_server_key.return_value = key
        transport.is_authenticated.return_value = True
        module = SimpleNamespace(
            Transport=Mock(return_value=transport),
            SFTPClient=SimpleNamespace(from_transport=Mock(return_value=self.sftp)),
        )
        args = SimpleNamespace(
            host="example.invalid", port=22, username="root", host_key_sha256=transfer.host_fingerprint(key)
        )
        with patch.dict(transfer.sys.modules, {"paramiko": module}), patch.object(
            transfer.socket, "create_connection", return_value=connection
        ):
            self.assertEqual(transfer.connect(args, "ephemeral-value"), (transport, self.sftp))
            self.assertEqual(security.key_types, ("ssh-ed25519",))
            transport.auth_password.assert_called_once_with("root", "ephemeral-value", fallback=False)
            transport.auth_password.reset_mock()
            args.host_key_sha256 = "SHA256:" + "A" * 43
            with self.assertRaises(RuntimeError):
                transfer.connect(args, "ephemeral-value")
            transport.auth_password.assert_not_called()

    def test_dataset_marker_is_bound_to_exact_files(self):
        marker = transfer.dataset_marker("sku110k", [self.entry])
        other = transfer.dataset_marker("sku110k", [dict(self.entry, sha256="0" * 64)])
        self.assertNotEqual(marker["dataset_manifest_sha256"], other["dataset_manifest_sha256"])
        transfer.write_remote_json(self.sftp, self.target + ".DONE.json", marker)
        self.assertEqual(json.loads(self.sftp.files[self.target + ".DONE.json"]), marker)
        with self.assertRaises(RuntimeError):
            transfer.write_remote_json(self.sftp, self.target + ".DONE.json", other)

    def test_password_is_read_from_one_line_only(self):
        source = io.StringIO("transient-secret\nnot-a-password\n")
        with patch.object(transfer.sys, "stdin", source):
            self.assertEqual(transfer.read_password(), "transient-secret")
        self.assertEqual(source.readline(), "not-a-password\n")


if __name__ == "__main__":
    unittest.main()
