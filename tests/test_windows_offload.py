import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import sequential_experiment_runner as runner
import windows_offload as server
import windows_experiment_controller as client


RUN = "phase_b/40demos/single_top/train_seed0"


def sha(data):
    return hashlib.sha256(data).hexdigest()


class WindowsOffloadTests(unittest.TestCase):
    def test_stream_verify_and_reuse_requires_hashes(self):
        files = {"chosen.ckpt": b"chosen", "other.ckpt": b"other", "eval_test/results.json": b"{}"}
        manifest = {"run": RUN, "files": [
            {"path": name, "bytes": len(data), "sha256": sha(data)} for name, data in files.items()
        ], "total_bytes": sum(map(len, files.values())), "manifest_sha256": "a" * 64}
        archive_bytes = io.BytesIO()
        with tarfile.open(fileobj=archive_bytes, mode="w") as archive:
            for name, data in files.items():
                entry = tarfile.TarInfo(RUN + "/" + name)
                entry.size = len(data)
                archive.addfile(entry, io.BytesIO(data))
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp)
            with patch.object(client.shutil, "disk_usage", return_value=type("Usage", (), {"free": 10**11})()):
                target = client.receive_tar(io.BytesIO(archive_bytes.getvalue()), manifest, destination)
                self.assertEqual((target / "other.ckpt").read_bytes(), b"other")
                self.assertEqual(client.receive_tar(None, manifest, destination), target)
                (target / "other.ckpt").write_bytes(b"tamper")
                with self.assertRaisesRegex(ValueError, "wrong size|SHA-256"):
                    client.receive_tar(None, manifest, destination)

    def test_bad_tar_never_installs_destination(self):
        manifest = {"run": RUN, "files": [{"path": "a", "bytes": 3, "sha256": sha(b"abc")}],
                    "total_bytes": 3, "manifest_sha256": "a" * 64}
        archive_bytes = io.BytesIO()
        with tarfile.open(fileobj=archive_bytes, mode="w") as archive:
            entry = tarfile.TarInfo(RUN + "/a")
            entry.size = 3
            archive.addfile(entry, io.BytesIO(b"bad"))
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp)
            with patch.object(client.shutil, "disk_usage", return_value=type("Usage", (), {"free": 10**11})()):
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    client.receive_tar(io.BytesIO(archive_bytes.getvalue()), manifest, destination)
            self.assertFalse((destination / RUN).exists())

    def test_commit_rejects_bad_receipt_then_prunes_only_unselected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "results/act" / RUN
            selection = run / "checkpoint_selection_deterministic"
            selection.mkdir(parents=True)
            chosen = run / "chosen.ckpt"
            other = run / "other.ckpt"
            chosen.write_bytes(b"chosen")
            other.write_bytes(b"other")
            (selection / "validation_summary.json").write_text(json.dumps({"selected_checkpoint": chosen.name}))
            checkpoints = [{"name": p.name, "bytes": p.stat().st_size, "sha256": runner.digest(p)}
                           for p in (chosen, other)]
            (run / "checkpoint_manifest.json").write_text(json.dumps({"checkpoints": checkpoints}))
            manifest = {"schema_version": 1, "run": RUN,
                        "files": [{"path": p.name, "bytes": p.stat().st_size, "sha256": runner.digest(p)}
                                  for p in (chosen, other)], "total_bytes": chosen.stat().st_size + other.stat().st_size}
            runner.atomic_json(run / server.MANIFEST, manifest)
            receipt = {"schema_version": 1, "run": RUN, "verified": True,
                       "manifest_sha256": runner.digest(run / server.MANIFEST),
                       "file_count": 2, "total_bytes": manifest["total_bytes"],
                       "destination": "C:\\Users\\chibi\\Research\\result\\" + RUN.replace("/", "\\")}
            with patch.object(runner, "ROOT", root), patch.object(runner, "no_active_jobs", return_value=True):
                bad = receipt | {"manifest_sha256": "0" * 64}
                with self.assertRaisesRegex(ValueError, "does not match"):
                    server.commit(RUN, bad)
                self.assertTrue(other.exists())
                server.commit(RUN, receipt)
                self.assertTrue(chosen.exists())
                self.assertFalse(other.exists())
                self.assertEqual(server.commit(RUN, receipt)["manifest_sha256"], receipt["manifest_sha256"])


if __name__ == "__main__":
    unittest.main()
