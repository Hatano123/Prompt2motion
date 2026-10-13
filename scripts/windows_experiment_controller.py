#!/usr/bin/env python3
"""Run on Windows: pull verified HPC results, then submit the next PBS allocation.

Requires Python 3 and an SSH key that works with ``ssh -o BatchMode=yes``.
The destination defaults to the folder requested for this experiment.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import time

RUN_RE = re.compile(r"^(phase_b|phase_b_rebuild)/(5|10|25|40)demos/(single_top|single_side|multi)/train_seed([012])$")
RESERVE = 2 * 1024**3


class ConnectionLost(RuntimeError):
    """Transient SSH transport failure; safe to reconnect and inspect state."""


class RemoteInfrastructureUnavailable(ConnectionLost):
    """Remote read-only check could not establish trustworthy state."""


SSH_OPTIONS = ["-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
               "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3"]


@contextmanager
def controller_lock(destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / ".windows_experiment_controller.lock").open("a+b") as handle:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            if handle.read(1) == b"":
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


def remote_command(remote: str, root: str, args: list[str], input_data: bytes | None = None) -> str:
    command = "cd " + shlex.quote(root) + " && " + " ".join(shlex.quote(part) for part in args)
    result = subprocess.run(["ssh", *SSH_OPTIONS, remote, command],
                            input=input_data, capture_output=True)
    if result.returncode == 255:
        raise ConnectionLost(result.stderr.decode(errors="replace").strip())
    if result.returncode == 75:
        raise RemoteInfrastructureUnavailable(result.stderr.decode(errors="replace").strip())
    result.check_returncode()
    return result.stdout.decode().strip()


def validate_manifest(manifest: dict) -> dict[str, dict]:
    name = manifest.get("run")
    if not isinstance(name, str) or not RUN_RE.fullmatch(name):
        raise ValueError("unsupported run in remote manifest")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("empty remote file manifest")
    expected = {}
    for row in files:
        path = row.get("path")
        if not isinstance(path, str) or path in ("", ".") or "\\" in path or \
                PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts or \
                path in expected or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256"))):
            raise ValueError(f"unsafe manifest entry: {path}")
        if not isinstance(row.get("bytes"), int) or row["bytes"] < 0:
            raise ValueError(f"invalid manifest size: {path}")
        expected[path] = row
    if sum(row["bytes"] for row in files) != manifest.get("total_bytes") or \
            not re.fullmatch(r"[0-9a-f]{64}", str(manifest.get("manifest_sha256"))):
        raise ValueError("invalid remote manifest totals")
    return expected


def verify_local(target: Path, expected: dict[str, dict]) -> None:
    for name, row in expected.items():
        path = target.joinpath(*PurePosixPath(name).parts)
        if not path.is_file() or path.stat().st_size != row["bytes"]:
            raise ValueError(f"local file missing or wrong size: {path}")
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                h.update(block)
        if h.hexdigest() != row["sha256"]:
            raise ValueError(f"local SHA-256 mismatch: {path}")


def receive_tar(stream, manifest: dict, destination: Path) -> Path:
    expected = validate_manifest(manifest)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination.joinpath(*PurePosixPath(manifest["run"]).parts)
    receipt_path = target / "windows_offload_receipt.json"
    if target.exists():
        if not receipt_path.is_file():
            raise ValueError(f"existing destination is incomplete: {target}")
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("manifest_sha256") != manifest["manifest_sha256"]:
            raise ValueError("existing destination belongs to another manifest")
        verify_local(target, expected)
        return target
    if shutil.disk_usage(destination).free < manifest["total_bytes"] + RESERVE:
        raise ValueError("Windows destination lacks run bytes plus 2 GiB reserve")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".partial.{os.getpid()}")
    for stale in target.parent.glob(target.name + ".partial.*"):
        if stale != temporary:
            shutil.rmtree(stale)
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    try:
        observed = set()
        with tarfile.open(fileobj=stream, mode="r|") as archive:
            for member in archive:
                full = PurePosixPath(member.name)
                try:
                    relative = full.relative_to(PurePosixPath(manifest["run"]))
                except ValueError as exc:
                    raise ValueError(f"tar entry outside run: {member.name}") from exc
                if member.isdir():
                    continue
                name = str(relative)
                if name not in expected or not member.isfile() or name in observed:
                    raise ValueError(f"unexpected tar entry: {member.name}")
                row = expected[name]
                if member.size != row["bytes"]:
                    raise ValueError(f"tar size mismatch: {member.name}")
                output = temporary.joinpath(*relative.parts)
                output.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"unreadable tar entry: {member.name}")
                digest = hashlib.sha256()
                with output.open("xb") as sink:
                    for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                        sink.write(block)
                        digest.update(block)
                if digest.hexdigest() != row["sha256"]:
                    raise ValueError(f"SHA-256 mismatch: {member.name}")
                observed.add(name)
        if observed != set(expected):
            raise ValueError(f"tar omitted {len(set(expected) - observed)} files")
        receipt = {"schema_version": 1, "run": manifest["run"], "verified": True,
                   "manifest_sha256": manifest["manifest_sha256"], "file_count": len(expected),
                   "total_bytes": manifest["total_bytes"], "destination": str(target)}
        (temporary / "windows_offload_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        os.rename(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def pull(remote: str, root: str, destination: Path, name: str) -> None:
    prepared = remote_command(remote, root, ["python3", "scripts/windows_offload.py", "prepare", "--run", name])
    manifest = json.loads(prepared)
    validate_manifest(manifest)
    target = destination.joinpath(*PurePosixPath(name).parts)
    if not target.exists():
        tar_command = "tar -C " + shlex.quote(root + "/results/act") + \
            " --exclude=windows_offload_manifest.json --exclude=offload_windows_receipt.json -cf - -- " + shlex.quote(name)
        process = subprocess.Popen(["ssh", *SSH_OPTIONS, remote, tar_command],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert process.stdout is not None
        try:
            receive_tar(process.stdout, manifest, destination)
            process.stdout.close()
            stderr = process.communicate()[1]
        except (ValueError, tarfile.ReadError) as exc:
            try:
                returncode = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                returncode = process.wait()
            process.stdout.close()
            stderr = process.communicate()[1]
            if returncode == 255:
                raise ConnectionLost(stderr.decode(errors="replace").strip()) from exc
            raise
        except BaseException:
            process.kill()
            process.stdout.close()
            process.communicate()
            raise
        if process.returncode == 255:
            raise ConnectionLost(stderr.decode(errors="replace").strip())
        if process.returncode != 0:
            raise RuntimeError(f"remote tar failed: {stderr.decode(errors='replace')}")
    else:
        receive_tar(None, manifest, destination)
    receipt = json.loads((target / "windows_offload_receipt.json").read_text())
    remote_command(remote, root, ["python3", "scripts/windows_offload.py", "commit", "--run", name],
                   input_data=(json.dumps(receipt) + "\n").encode())
    print(f"[VERIFIED] {name} -> {target}")


def cycle(remote: str, root: str, plan: str, destination: Path) -> bool:
    raw = remote_command(remote, root, ["python3", "scripts/sequential_experiment_runner.py", "status", "--plan", plan])
    status = json.loads(raw.splitlines()[-1])
    if status.get("backend") != "windows_pull":
        raise ValueError("remote plan is not configured for Windows pull")
    pending = status.get("next")
    if status.get("active_jobs") or status.get("busy"):
        print("[WAIT] PBS job still active")
        return True
    if pending is None:
        print("[COMPLETE] all planned runs verified")
        return False
    name, stage = pending
    if stage == "offload":
        pull(remote, root, destination, name)
    else:
        destination.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(destination).free < 4 * 1024**3:
            raise ValueError("Windows destination has less than 4 GiB free")
        result = remote_command(remote, root, ["python3", "scripts/sequential_experiment_runner.py",
                                               "submit-next", "--plan", plan])
        print(result.splitlines()[-1])
    return True


def run_loop(args) -> None:
    while True:
        try:
            pending = cycle(args.remote, args.remote_root, args.plan, args.destination)
        except RemoteInfrastructureUnavailable as exc:
            if not args.loop:
                raise
            print(f"[WAIT_INFRA] {exc}", flush=True)
            pending = True
        except ConnectionLost as exc:
            if not args.loop:
                raise
            print(f"[RECONNECT] {exc}", flush=True)
            pending = True
        if not args.loop or not pending:
            return
        time.sleep(args.poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", required=True, help="SSH target from the Windows machine")
    parser.add_argument("--remote-root", default="/home/home0/y263357/Prompt2Motion")
    parser.add_argument("--plan", default="configs/sequential_phase_b_windows.json")
    parser.add_argument("--destination", type=Path, default=Path(r"C:\Users\chibi\Research\result"))
    parser.add_argument("--loop", action="store_true", help="keep polling and submitting while this PC is on")
    parser.add_argument("--log-file", type=Path, help="append UTF-8 controller output to this file")
    parser.add_argument("--poll-seconds", type=int, default=120)
    args = parser.parse_args()
    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        log = args.log_file.open("a", encoding="utf-8", buffering=1)
        sys.stdout = log
        sys.stderr = log
        print(f"[{datetime.now().astimezone().isoformat()}] Controller starting")
    if args.poll_seconds < 30:
        raise ValueError("poll interval must be at least 30 seconds")
    with controller_lock(args.destination):
        # Keep Windows awake while the controller is running; allow display sleep.
        if os.name == "nt":
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        try:
            run_loop(args)
        finally:
            if os.name == "nt":
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError, RuntimeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
