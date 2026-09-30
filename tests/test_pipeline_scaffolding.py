import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PipelineScaffoldingTest(unittest.TestCase):
    def test_matrix_contains_all_phase_d_training_runs(self):
        planner = load_script("plan_experiment_pipeline.py")
        config = json.loads((ROOT / "configs/experiment_pipeline.json").read_text())
        jobs = planner.inventory(config)
        phase_d_train = [job for job in jobs if job["name"].startswith("d-train-")]
        self.assertEqual(12, len(phase_d_train))
        self.assertEqual(len({job["name"] for job in jobs}), len(jobs))

    def test_paired_statistics_detect_clean_degradation(self):
        summary = load_script("summarize_robustness.py")
        clean = [{"rollout": i, "success": i < 8, "episode_return": 1} for i in range(10)]
        corrupt = [{"rollout": i, "success": i < 5, "episode_return": 1} for i in range(10)]
        result = summary.paired_stats(clean, corrupt, bootstrap_samples=1000)
        self.assertAlmostEqual(-0.3, result["success_rate_delta"])
        self.assertEqual(3, result["clean_success_corrupt_failure"])

    def test_paired_statistics_reject_missing_or_duplicate_ids(self):
        summary = load_script("summarize_robustness.py")
        clean = [{"rollout": i, "success": True, "episode_return": 1} for i in range(3)]
        with self.assertRaisesRegex(ValueError, "identical"):
            summary.paired_stats(clean, clean[:-1], bootstrap_samples=10)
        duplicate = [clean[0], clean[0], clean[2]]
        with self.assertRaisesRegex(ValueError, "unique"):
            summary.paired_stats(clean, duplicate, bootstrap_samples=10)

    def test_robustness_job_verifies_selected_checkpoint_hash(self):
        script = (ROOT / "scripts/run_robustness_eval.pbs").read_text()
        self.assertIn("selected_checkpoint_sha256", script)
        self.assertIn("selected checkpoint SHA-256 mismatch", script)
        self.assertIn("fixed_validation_manifest", script)

    def test_hardware_template_fails_closed(self):
        audit_module = load_script("audit_hr4c_readiness.py")
        path = ROOT / "configs/hr4c_hardware.template.json"
        report = audit_module.audit(json.loads(path.read_text()), path.parent)
        self.assertFalse(report["ready"])
        self.assertFalse(report["hardware_contact_attempted"])

    def test_phase_b_aggregate_reports_seed_statistics(self):
        module = load_script("summarize_phase_b.py")
        rows = [
            {"demo_count": 5, "model": "multi", "training_seed": 0,
             "success_rate": 0.2, "average_return": 100},
            {"demo_count": 5, "model": "multi", "training_seed": 1,
             "success_rate": 0.4, "average_return": 200},
        ]
        result = module.aggregate(rows)
        self.assertEqual(2, result[0]["completed_seeds"])
        self.assertAlmostEqual(0.3, result[0]["seed_mean_success_rate"])

    def test_missing_candidate_checkpoints_are_detected(self):
        planner = load_script("plan_experiment_pipeline.py")
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "checkpoint_manifest.json").write_text(json.dumps({
                "checkpoints": [{"name": "policy_epoch_500_seed_0.ckpt"}]}))
            self.assertFalse(planner.candidate_checkpoints_available(run))
            (run / "policy_epoch_500_seed_0.ckpt").write_bytes(b"checkpoint")
            self.assertTrue(planner.candidate_checkpoints_available(run))

    def test_archived_run_is_not_scheduled_for_retraining(self):
        planner = load_script("plan_experiment_pipeline.py")
        config = json.loads((ROOT / "configs/experiment_pipeline.json").read_text())
        jobs = planner.inventory(config)
        job = next(item for item in jobs if item["name"] == "b-train-10-multi-s0")
        canonical = ROOT / "results/act/phase_b/10demos/multi/train_seed0"
        rebuild = ROOT / "results/act/phase_b_rebuild/10demos/multi/train_seed0"
        if ((canonical / ".training_complete").exists() and
                not planner.candidate_checkpoints_available(canonical) and
                not (rebuild / ".training_complete").exists()):
            self.assertEqual("blocked", job["status"])
            self.assertIn("restore", job["reason"])

    def test_audit_accepts_drive_verified_archived_checkpoint(self):
        audit_module = load_script("audit_phase_gate.py")
        item = {"name": "policy_epoch_500_seed_0.ckpt", "bytes": 10, "sha256": "a" * 64}
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "checkpoint_archive_receipt.json").write_text(json.dumps({
                "schema_version": 1,
                "drive_readback_verified": True,
                "archived_checkpoints": [item],
            }))
            self.assertTrue(audit_module.archived_checkpoint_matches(run, item))
            altered = dict(item, sha256="b" * 64)
            self.assertFalse(audit_module.archived_checkpoint_matches(run, altered))


if __name__ == "__main__":
    unittest.main()
