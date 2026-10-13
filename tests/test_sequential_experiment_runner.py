import importlib.util
import json
from pathlib import Path
import tempfile
import subprocess
import unittest
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/sequential_experiment_runner.py"
SPEC = importlib.util.spec_from_file_location("sequential_experiment_runner", MODULE_PATH)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class SequentialRunnerTests(unittest.TestCase):
    def test_unknown_pbs_state_prevents_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = {"offload_backend": "windows_pull"}
            with patch.object(runner, "ROOT", Path(directory)), \
                    patch.object(runner.sys, "argv", ["runner", "submit-next"]), \
                    patch.object(runner, "read_plan", return_value=plan), \
                    patch.object(runner, "verify_inputs"), \
                    patch.object(runner, "next_stage", return_value=("phase_b/10demos/multi/train_seed1", "train")), \
                    patch.object(runner, "no_active_jobs", side_effect=runner.InfrastructureUnavailable(110, "offline")), \
                    patch.object(runner.subprocess, "run") as submit:
                with self.assertRaises(runner.InfrastructureUnavailable):
                    runner.main()
                submit.assert_not_called()
                self.assertFalse(runner.MUTATION_STARTED)

    def test_evaluation_cannot_start_without_home_headroom(self):
        with patch.object(runner.shutil, "disk_usage",
                          return_value=type("Usage", (), {"free": 560_000_000})()):
            for stage in ("select", "test"):
                with self.assertRaisesRegex(ValueError, "offload first"):
                    runner.admission(stage, None)
            runner.admission("offload", None)

    def test_pbs_outage_never_reports_idle(self):
        error = subprocess.CalledProcessError(1, ["qstat"], stderr="Connection refused")
        with patch.object(runner.subprocess, "run", side_effect=error) as query, \
                patch.object(runner.time, "sleep"):
            with self.assertRaises(runner.InfrastructureUnavailable):
                runner.no_active_jobs()
            self.assertEqual(query.call_count, 3)

    def test_pbs_recovers_and_array_jobs_are_active(self):
        error = subprocess.CalledProcessError(1, ["qstat"], stderr="cannot connect to server")
        result = subprocess.CompletedProcess(["qstat"], 0, stdout="113000[].ypbs0 job user Q")
        with patch.object(runner.subprocess, "run", side_effect=[error, result]), \
                patch.object(runner.time, "sleep"):
            self.assertFalse(runner.no_active_jobs())

    def test_verified_drive_receipt_allows_archived_selected_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            selection = run / "checkpoint_selection_deterministic"
            selection.mkdir()
            (selection / "validation_summary.json").write_text(json.dumps({
                "selection_dataset": "fixed_validation_manifest", "heldout_test_used_for_selection": False,
                "selected_checkpoint": "policy_epoch_2000_seed_0.ckpt", "selected_checkpoint_sha256": "a" * 64,
            }))
            (run / "selected_checkpoint_drive_receipt.json").write_text(json.dumps({
                "schema_version": 1, "drive_readback_verified": True,
                "selected_checkpoint": "policy_epoch_2000_seed_0.ckpt",
                "sha256": "a" * 64, "bytes": 10,
                "parts": [{"bytes": 10, "sha256": "b" * 64, "drive_file_id": "file1"}],
            }))
            with self.assertRaisesRegex(ValueError, "missing"):
                runner.selected_checkpoint(run)
            self.assertEqual(runner.selected_checkpoint(run, allow_archived=True).name,
                             "policy_epoch_2000_seed_0.ckpt")

    def test_failed_stage_is_preserved_for_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            name = "phase_b/10demos/multi/train_seed0"
            partial = root / "results/act" / name / "eval_test"
            partial.mkdir(parents=True)
            (partial / "test.log").write_text("interrupted")
            with patch.object(runner, "ROOT", root):
                archived = runner.quarantine_partial(name, "test")
                self.assertEqual((archived / "test.log").read_text(), "interrupted")
                self.assertFalse(partial.exists())

    def test_plan_rejects_duplicates_and_unsupported_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = Path(tmp) / "plan.json"
            for runs in (["phase_b/10demos/multi/train_seed1"] * 2,
                         ["../../other"], []):
                plan.write_text(json.dumps({"schema_version": 1, "inputs": {}, "runs": runs}))
                with self.assertRaises(ValueError):
                    runner.read_plan(plan)

    def test_offload_readback_then_prune_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            destination = Path(tmp) / "backup"
            destination.mkdir()
            run = root / "results/act/phase_b/10demos/multi/train_seed0"
            (run / "checkpoint_selection_deterministic").mkdir(parents=True)
            (run / "eval_test").mkdir()
            chosen = run / "chosen.ckpt"
            other = run / "other.ckpt"
            chosen.write_bytes(b"chosen")
            other.write_bytes(b"other")
            (run / "checkpoint_selection_deterministic/.complete").touch()
            (run / "checkpoint_selection_deterministic/validation_summary.json").write_text(json.dumps({
                "selection_dataset": "fixed_validation_manifest", "heldout_test_used_for_selection": False,
                "selected_checkpoint": chosen.name, "selected_checkpoint_sha256": runner.digest(chosen),
            }))
            (run / "checkpoint_manifest.json").write_text(json.dumps({"checkpoints": [
                {"name": path.name, "bytes": path.stat().st_size, "sha256": runner.digest(path)}
                for path in (chosen, other)
            ]}))
            (run / "eval_test/.complete").touch()
            with patch.object(runner, "ROOT", root), patch.object(runner, "verify_stage", return_value=True), \
                    patch.object(runner.shutil, "disk_usage", return_value=type("Usage", (), {"free": 10**11})()):
                target = runner.offload(run, destination, True)
                self.assertTrue(chosen.exists())
                self.assertFalse(other.exists())
                self.assertEqual((target / "other.ckpt").read_bytes(), b"other")
                self.assertEqual(runner.offload(run, destination, True), target)
                (target / "other.ckpt").write_bytes(b"broken")
                with self.assertRaisesRegex(ValueError, "readback failed"):
                    runner.offload(run, destination, True)


if __name__ == "__main__":
    unittest.main()
