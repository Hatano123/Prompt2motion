import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "restore_phase_b_checkpoints.sh"


class RestorePhaseBCheckpointsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.repo = self.base / "repo"
        self.archive = self.base / "archive"
        self.source = self.base / "source"
        self.archive.mkdir()
        run = Path("results/act/phase_b/5demos/single_top/train_seed0")
        source_run = self.source / run
        target_run = self.repo / run
        source_run.mkdir(parents=True)
        target_run.mkdir(parents=True)

        checkpoints = []
        for epoch in (500, 1000, 1500, 2000):
            name = f"policy_epoch_{epoch}_seed_0.ckpt"
            payload = (f"checkpoint-{epoch}\n" * epoch).encode()
            (source_run / name).write_bytes(payload)
            checkpoints.append({
                "name": name,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            })
        (target_run / "checkpoint_manifest.json").write_text(
            json.dumps({"checkpoints": checkpoints}) + "\n"
        )

        tar_path = self.base / "backup.tar"
        with tarfile.open(tar_path, "w") as archive:
            archive.add(self.source / "results", arcname="results")
        part = self.archive / "fixture.tar.zst.part-000"
        subprocess.run(
            ["zstd", "-q", "-f", str(tar_path), "-o", str(part)], check=True
        )
        digest = hashlib.sha256(part.read_bytes()).hexdigest()
        (self.archive / "SHA256SUMS.txt").write_text(f"{digest}  {part.name}\n")
        self.run = target_run

    def tearDown(self):
        self.temporary.cleanup()

    def invoke(self, *extra, check=True):
        return subprocess.run(
            [
                str(SCRIPT),
                "--archive-dir", str(self.archive),
                "--repo-root", str(self.repo),
                "--demo-count", "5",
                "--view-config", "single_top",
                "--reserve-bytes", "0",
                *extra,
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=check,
        )

    def test_dry_run_writes_nothing(self):
        result = self.invoke("--dry-run")
        self.assertIn("dry-run completed", result.stdout)
        self.assertEqual(list(self.run.glob("*.ckpt")), [])

    def test_restore_is_verified_and_idempotent(self):
        self.invoke()
        restored = sorted(self.run.glob("*.ckpt"))
        self.assertEqual(len(restored), 4)
        result = self.invoke()
        self.assertIn("already restored and verified", result.stdout)

    def test_existing_mismatch_is_not_overwritten(self):
        checkpoint = self.run / "policy_epoch_500_seed_0.ckpt"
        checkpoint.write_bytes(b"do-not-overwrite")
        result = self.invoke(check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("refusing to overwrite", result.stderr)
        self.assertEqual(checkpoint.read_bytes(), b"do-not-overwrite")

    def test_rebuild_namespace_copies_training_metadata(self):
        (self.run / ".training_complete").write_text("")
        (self.run / "training_summary.json").write_text('{"optimizer_steps": 10000}\n')
        self.invoke("--output-namespace", "phase_b_rebuild")
        rebuild = self.repo / "results/act/phase_b_rebuild/5demos/single_top/train_seed0"
        self.assertTrue((rebuild / ".training_complete").exists())
        self.assertEqual(
            (rebuild / "training_summary.json").read_text(),
            '{"optimizer_steps": 10000}\n',
        )
        self.assertEqual(len(list(rebuild.glob("*.ckpt"))), 4)


if __name__ == "__main__":
    unittest.main()
