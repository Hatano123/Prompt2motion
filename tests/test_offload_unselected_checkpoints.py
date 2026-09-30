import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "offload_unselected_checkpoints.py"


class OffloadUnselectedCheckpointsTest(unittest.TestCase):
    def test_retains_selected_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            repo = base / "repo"
            staging = base / "staging"
            run = repo / "results/act/phase_b_rebuild/5demos/multi/train_seed0"
            selection = run / "checkpoint_selection_deterministic"
            selection.mkdir(parents=True)
            (selection / ".complete").write_text("")
            selected = "policy_epoch_1000_seed_0.ckpt"
            (selection / "validation_summary.json").write_text(json.dumps({
                "selection_dataset": "fixed_validation_manifest",
                "heldout_test_used_for_selection": False,
                "selected_checkpoint": selected,
            }))
            entries = []
            for epoch in (500, 1000, 1500, 2000):
                name = f"policy_epoch_{epoch}_seed_0.ckpt"
                data = f"checkpoint-{epoch}".encode()
                (run / name).write_bytes(data)
                entries.append({"name": name, "bytes": len(data),
                                "sha256": hashlib.sha256(data).hexdigest()})
            (run / "checkpoint_manifest.json").write_text(json.dumps({"checkpoints": entries}))
            command = [str(SCRIPT), str(run.relative_to(repo)), "--repo-root", str(repo),
                       "--staging-root", str(staging), "--drive-folder-id", "folder",
                       "--archive-name", "backup.tar.zst",
                       "--archive-parts-checksum-file-id", "checksum",
                       "--drive-readback-verified"]
            subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertTrue((run / selected).is_file())
            self.assertEqual(1, len(list(run.glob("*.ckpt"))))
            self.assertEqual(3, len(list(staging.rglob("*.ckpt"))))
            subprocess.run(command, check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
