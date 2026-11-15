"""CPU-only launch contracts; no network, real password, or real subprocess."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

try:
    from online_experiments import launch_autodl_transfer_pair as launch
except ImportError:
    launch = None


class Input(io.StringIO):
    def close(self):
        self.saved = self.getvalue()
        super().close()


class Contracts(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(launch, "launcher implementation is missing")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.job = self.root / "job"
        (self.job / "bundle-v1").mkdir(parents=True)
        (self.job / "bundle-v1/transfer-manifest.json").write_text('{"files":[]}')

    def test_commands_have_fixed_pin_and_no_credential_argument(self):
        with patch.object(launch, "ROOT", self.root), patch.object(launch, "JOB", self.job):
            rows = launch.commands()
        self.assertEqual(len(rows), 2)
        for _, argv, _ in rows:
            self.assertIn(launch.PIN, argv)
            self.assertNotIn("--password", argv)
        self.assertIn(str(self.job / "bundle-v1/transfer-manifest.json"), rows[0][1])
        self.assertEqual(Path(rows[0][1][2]).name, "run_autodl_upload.py")
        self.assertIn("--publish", rows[1][1])

    def test_environment_allows_only_fixed_loopback_proxy_and_omits_sensitive_tokens(self):
        with patch.dict(
            launch.os.environ,
            {
                "PATH": "bin",
                "HTTP_PROXY": "secret-proxy",
                "HTTPS_PROXY": "other",
                "HF_TOKEN": "token",
                "PYTHONPATH": "existing",
            },
            clear=True,
        ):
            env = launch.child_environment()
        self.assertNotIn("HTTP_PROXY", env)
        self.assertNotIn("HTTPS_PROXY", env)
        self.assertNotIn("HF_TOKEN", env)
        self.assertEqual(env["http_proxy"], "http://127.0.0.1:7908")
        self.assertEqual(env["https_proxy"], "http://127.0.0.1:7908")
        self.assertIn("autodl-transfer", env["PYTHONPATH"])
        self.assertTrue(env["PYTHONPATH"].endswith("existing"))

    def test_launch_inputs_password_once_and_records_only_nonsecret_metadata(self):
        children = [SimpleNamespace(pid=10, stdin=Input()), SimpleNamespace(pid=11, stdin=Input())]
        popen = Mock(side_effect=children)

        def identity(pid):
            return {"pid": pid, "uid": 1010, "start_ticks": pid + 100, "argv": popen.call_args.args[0], "state": "S"}

        with patch.object(launch, "ROOT", self.root), patch.object(launch, "JOB", self.job), patch.object(
            launch, "probe_lock"
        ), patch.object(launch.transfer, "owner_lock", lambda _: contextlib.nullcontext()), patch.object(
            launch, "process_identity", side_effect=identity
        ), patch.object(launch.getpass, "getpass", return_value="ephemeral-value") as prompt, patch.object(
            launch.subprocess, "Popen", popen
        ):
            launch.launch_pair()
        prompt.assert_called_once()
        self.assertEqual([child.stdin.saved for child in children], ["ephemeral-value\n"] * 2)
        saved = (self.job / "launch.json").read_text()
        self.assertNotIn("ephemeral-value", saved)
        self.assertEqual(len(json.loads(saved)["children"]), 2)
        for call in popen.call_args_list:
            self.assertTrue(call.kwargs["start_new_session"])
            self.assertTrue(call.kwargs["close_fds"])
            self.assertEqual(call.kwargs["stdin"], launch.subprocess.PIPE)

    def test_live_prior_child_blocks_before_password_prompt(self):
        child = {"pid": 10, "uid": 1010, "start_ticks": 100, "argv": ["python", "worker.py"], "state": "S"}
        (self.job / "launch.json").write_text(json.dumps({"children": [child]}))
        with patch.object(launch, "JOB", self.job), patch.object(
            launch.transfer, "owner_lock", lambda _: contextlib.nullcontext()
        ), patch.object(launch, "process_identity", return_value=child), patch.object(
            launch.getpass, "getpass"
        ) as prompt, self.assertRaises(RuntimeError):
            launch.launch_pair()
        prompt.assert_not_called()


if __name__ == "__main__":
    unittest.main()
