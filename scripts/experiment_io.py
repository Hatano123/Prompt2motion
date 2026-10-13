#!/usr/bin/env python3
"""Bounded retries for reads and hash-verified staging of experiment inputs."""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time


class InfrastructureUnavailable(OSError):
    """A bounded read retry failed; artifact validity remains unknown."""


class SourceReadFailure(OSError):
    """Distinguish source read failures from destination write failures."""


RETRY_ERRNOS = {errno.EPERM, errno.EIO, errno.ESTALE, errno.ETIMEDOUT,
                errno.ECONNRESET, errno.EAGAIN}


def retry_read(operation, label, attempts=3, retry_if=None):
    if attempts < 1:
        raise ValueError("attempts must be positive")
    for attempt in range(attempts):
        try:
            return operation()
        except OSError as exc:
            if exc.errno not in RETRY_ERRNOS or (retry_if is not None and not retry_if(exc)):
                raise
            if attempt == attempts - 1:
                raise InfrastructureUnavailable(
                    exc.errno, f"{label}: read failed after {attempts} attempts: {exc}") from exc
            print(f"[RETRY_READ] {label}: {exc}; retry {attempt + 1}/{attempts - 1}",
                  file=sys.stderr, flush=True)
            time.sleep(attempt + 1)


def read_text(path):
    return retry_read(lambda: Path(path).read_text(encoding="utf-8"), str(path))


def digest(path):
    def compute():
        result = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                result.update(block)
        return result.hexdigest()
    return retry_read(compute, str(path))


def verified_copy(source, destination, expected_sha256, reserve_bytes=1024**3):
    """Stage once per attempt, verify disk readback, then publish atomically.

    The original is never modified. Existing copies must match the pinned hash.
    This is staging, not a durable backup; choose backup storage separately.
    """
    source, destination = Path(source), Path(destination)
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("a pinned SHA-256 is required")
    if destination.exists():
        if digest(destination) != expected_sha256:
            raise ValueError(f"existing destination SHA-256 mismatch: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    size = retry_read(lambda: source.stat().st_size, str(source))
    if shutil.disk_usage(destination.parent).free < size + reserve_bytes:
        raise ValueError("destination lacks source bytes plus 1 GiB reserve")

    def copy():
        fd, name = tempfile.mkstemp(prefix=destination.name + ".partial.", dir=destination.parent)
        temp = Path(name)
        try:
            with os.fdopen(fd, "wb") as sink:
                # Retry only source reads, never an ambiguous destination write.
                try:
                    stream = source.open("rb")
                except OSError as exc:
                    raise SourceReadFailure(exc.errno, str(exc)) from exc
                with stream:
                    result = hashlib.sha256()
                    while True:
                        try:
                            block = stream.read(4 * 1024 * 1024)
                        except OSError as exc:
                            raise SourceReadFailure(exc.errno, str(exc)) from exc
                        if not block:
                            break
                        result.update(block)
                        sink.write(block)
                sink.flush()
                os.fsync(sink.fileno())
            if result.hexdigest() != expected_sha256 or digest(temp) != expected_sha256:
                raise ValueError(f"source or staged copy SHA-256 mismatch: {source}")
            # link publishes without overwriting a competing writer's copy.
            try:
                os.link(temp, destination)
            except FileExistsError:
                if digest(destination) != expected_sha256:
                    raise ValueError(f"competing destination SHA-256 mismatch: {destination}")
            return destination
        finally:
            temp.unlink(missing_ok=True)
    return retry_read(copy, f"stage {source}", retry_if=lambda exc: isinstance(exc, SourceReadFailure))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    pinned = parser.add_mutually_exclusive_group(required=True)
    pinned.add_argument("--sha256")
    pinned.add_argument("--manifest", type=Path, help="checkpoint_manifest.json with name/sha256 records")
    args = parser.parse_args()
    try:
        expected = args.sha256
        if args.manifest:
            rows = json.loads(read_text(args.manifest))["checkpoints"]
            matches = [row for row in rows if row["name"] == args.source.name]
            if len(matches) != 1:
                raise ValueError("checkpoint must occur exactly once in the pinned manifest")
            expected = matches[0]["sha256"]
        print(verified_copy(args.source, args.destination, expected))
    except InfrastructureUnavailable as exc:
        print(f"[WAIT_INFRA] {exc}", file=sys.stderr)
        sys.exit(75)
    except (OSError, ValueError, KeyError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
