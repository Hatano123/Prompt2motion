import errno
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import experiment_io as io


class ExperimentIOTests(unittest.TestCase):
    def test_intermittent_read_recovers_and_persistent_failure_is_unknown(self):
        with patch.object(io.time, "sleep"), patch.object(Path, "read_text", side_effect=[
                PermissionError(errno.EPERM, "blocked"), "ok"]):
            self.assertEqual(io.read_text(Path("metadata.json")), "ok")
        with patch.object(io.time, "sleep"), patch.object(Path, "read_text",
                side_effect=PermissionError(errno.EPERM, "blocked")) as read:
            with self.assertRaises(io.InfrastructureUnavailable):
                io.read_text(Path("metadata.json"))
            self.assertEqual(read.call_count, 3)

    def test_missing_and_normal_permission_failure_are_not_retried(self):
        for code in (errno.ENOENT, errno.EACCES):
            with patch.object(Path, "read_text", side_effect=OSError(code, "fatal")) as read:
                with self.assertRaises(OSError):
                    io.read_text(Path("metadata.json"))
                self.assertEqual(read.call_count, 1)

    def test_copy_publishes_verified_bytes_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / "source.ckpt"
            dst = Path(directory) / "staged.ckpt"
            src.write_bytes(b"checkpoint")
            expected = hashlib.sha256(b"checkpoint").hexdigest()
            io.verified_copy(src, dst, expected, reserve_bytes=0)
            self.assertEqual(dst.read_bytes(), src.read_bytes())
            self.assertEqual(dst.stat().st_mode & 0o777, 0o600)
            self.assertEqual(io.verified_copy(src, dst, expected), dst)
            self.assertFalse(list(Path(directory).glob("*.partial.*")))

    def test_corruption_and_low_space_never_publish_or_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / "source.ckpt"
            dst = Path(directory) / "staged.ckpt"
            src.write_bytes(b"wrong")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                io.verified_copy(src, dst, "a" * 64, reserve_bytes=0)
            self.assertFalse(dst.exists())
            self.assertFalse(list(Path(directory).glob("*.partial.*")))
            dst.write_bytes(b"existing")
            with self.assertRaisesRegex(ValueError, "existing destination"):
                io.verified_copy(src, dst, "a" * 64)
            self.assertEqual(dst.read_bytes(), b"existing")
            dst.unlink()
            with patch.object(io.shutil, "disk_usage", return_value=type("Usage", (), {"free": 0})()):
                with self.assertRaisesRegex(ValueError, "reserve"):
                    io.verified_copy(src, dst, "a" * 64)
            self.assertFalse(dst.exists())

    def test_source_retry_cleans_partial_and_write_failure_is_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / "source.ckpt"
            dst = Path(directory) / "staged.ckpt"
            src.write_bytes(b"checkpoint")
            expected = io.digest(src)
            original_open = Path.open
            failures = [True]
            def intermittent(path, *args, **kwargs):
                if path == src and failures:
                    failures.pop()
                    raise OSError(errno.EPERM, "transient")
                return original_open(path, *args, **kwargs)
            with patch.object(Path, "open", intermittent), patch.object(io.time, "sleep"):
                io.verified_copy(src, dst, expected, reserve_bytes=0)
            self.assertFalse(list(Path(directory).glob("*.partial.*")))
            dst.unlink()
            with patch.object(io.os, "fsync", side_effect=OSError(errno.EIO, "write failure")) as sync:
                with self.assertRaises(OSError):
                    io.verified_copy(src, dst, expected, reserve_bytes=0)
                self.assertEqual(sync.call_count, 1)
            self.assertFalse(dst.exists())
            self.assertFalse(list(Path(directory).glob("*.partial.*")))


if __name__ == "__main__":
    unittest.main()
