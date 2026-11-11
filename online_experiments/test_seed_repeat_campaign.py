"""CPU-only scheduling, completion, resume, and process-ownership contracts."""

from __future__ import annotations

import contextlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from online_experiments import seed_repeat_campaign as worker


def queue():
    seed0 = (*worker.BASELINES, "mix25", "online100", "blank0", "detail0", "vit-adapter-cache0", "cache-reuse-c1-int4")
    cases = [(method, 0) for method in seed0]
    cases += [(method, seed) for seed in (1, 2) for method in worker.BASELINES]
    cases += [(method, seed) for seed in (1, 2) for method in worker.SPATIAL]
    return [
        {
            "method": method,
            "seed": seed,
            "name": f"{method}-s{seed}",
            "resume_original": method == "blank25" and seed == 1,
        }
        for method, seed in cases
    ]


class Contracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        base = Path(self.temporary.name)
        self.args = SimpleNamespace(
            job=base / "new/job", original_job=base / "old/job", origin_root=base / "origin", poll_seconds=1
        )
        for path in (self.args.job / "dataset-first", self.args.original_job, self.args.origin_root):
            path.mkdir(parents=True)

    def test_exact_eighteen_baselines_before_spatial(self):
        first, selected = worker.repeats(queue())
        self.assertEqual((len(first), len(selected)), (10, 18))
        self.assertTrue(all(case["method"] in worker.BASELINES for case in selected[:8]))
        self.assertTrue(all(case["method"] in worker.SPATIAL for case in selected[8:]))
        self.assertEqual(len({case["name"] for case in selected}), 18)

    def test_invalid_duplicate_or_legacy_refused(self):
        cases = queue()
        cases[-1] = cases[-2]
        with self.assertRaises(ValueError):
            worker.repeats(cases)
        cases = queue()
        cases[18]["resume_original"] = False
        with self.assertRaises(ValueError):
            worker.repeats(cases)

    def test_epochs_configuration_is_not_completion(self):
        path = self.args.job / "dataset-first"
        worker.write(path / "completion.json", {"state": "PHASE_COMPLETE", "epochs": 100, "completed_seed_zero": 8})
        worker.write(path / "status.json", {"state": "PHASE_COMPLETE"})
        self.assertFalse(worker.ready(self.args))
        worker.write(
            path / "completion.json",
            {
                "state": "PHASE_COMPLETE",
                "completed_seed_zero": 10,
                "planned_seed_zero": 10,
                "deferred_seed_repeats": 18,
            },
        )
        self.assertTrue(worker.ready(self.args))
        worker.write(path / "status.json", {"state": "RUNNING"})
        self.assertFalse(worker.ready(self.args))

    def test_legacy_uses_original_output_and_original_trainer(self):
        case = next(case for case in queue() if case["resume_original"])
        worker.write(
            self.args.original_job / "provenance.json",
            {"source_job": "/frozen/source", "cache_manifest": "/frozen/cache.json"},
        )
        command, cwd = worker.command_for(self.args, case, "train")
        self.assertEqual(cwd, self.args.origin_root)
        self.assertEqual(Path(command[2]), self.args.origin_root / "online_experiments/replication_suite.py")
        self.assertEqual(command[3], "train")
        self.assertEqual(worker.output_for(self.args, case), self.args.original_job / "train/blank25-s1")
        self.assertNotIn("--epochs", command)

    def test_missing_legacy_checkpoint_cannot_restart(self):
        case = next(case for case in queue() if case["resume_original"])
        out = worker.output_for(self.args, case)
        out.mkdir(parents=True)
        worker.write(
            out / "args.json",
            {"method": "blank25", "seed": 1, "epochs": 100, "train_images": 6471, "val_images": 548, "batch": 8},
        )
        worker.write(out / "summary.json", {"completed_epochs": 10})
        with self.assertRaisesRegex(RuntimeError, "never restart"):
            worker.legacy_resume_contract(self.args, case)

    def test_completed_marker_is_required_before_validator(self):
        with self.assertRaisesRegex(RuntimeError, "Missing DONE"):
            worker.validate(self.args, queue()[10])

    def test_three_exclusive_locks_and_fds_held(self):
        import fcntl
        import os

        from online_experiments.dataset_first_campaign import CoordinationBusy

        with worker.campaign_locks(self.args) as descriptors:
            self.assertEqual(len(descriptors), 3)
            for descriptor in descriptors:
                os.fstat(descriptor)
            with self.assertRaises(CoordinationBusy), worker.campaign_locks(self.args):
                pass
        path = self.args.job.parent / "campaign.lock"
        with path.open("a") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_orphan_trainer_refused_after_completion_gate(self):
        @contextlib.contextmanager
        def locks(_):
            yield (10, 11, 12)

        with patch.object(worker, "ready", return_value=True), patch.object(worker, "campaign_locks", locks), patch(
            "online_experiments.dataset_first_campaign.live_trainers", return_value=[{"pid": 99}]
        ), self.assertRaisesRegex(RuntimeError, "refusing duplicate"), worker.wait_seed_zero(self.args):
            self.fail("Must not yield when a trainer exists")

    def test_locked_seed0_wait_does_not_start_any_worker(self):
        with patch.object(worker, "ready", return_value=False), patch.object(worker, "status") as emit, patch.object(
            worker.time, "sleep", side_effect=TimeoutError
        ):
            with self.assertRaises(TimeoutError), worker.wait_seed_zero(self.args):
                self.fail("Seed-zero not complete")
            self.assertEqual(emit.call_args.args[1], "WAITING_SEED0")


if __name__ == "__main__":
    unittest.main()
