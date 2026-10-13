#!/usr/bin/env python3
"""Resume one verified Phase B run at a time inside a PBS allocation.

The login-side ``submit-next`` command is suitable for a site-approved timer.
It never submits while any job owned by the user is present in qstat.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

try:
    from experiment_io import InfrastructureUnavailable, digest, read_text, retry_read
except ModuleNotFoundError:
    from scripts.experiment_io import InfrastructureUnavailable, digest, read_text, retry_read

MUTATION_STARTED = False
ROOT = Path(__file__).resolve().parents[1]
STAGES = ("train", "select", "test", "offload")
SCRIPTS = {
    "train": "scripts/run_phase_b_train.pbs",
    "select": "scripts/run_phase_b_checkpoint_selection.pbs",
    "test": "scripts/run_phase_b_test.pbs",
}
MIN_SECONDS = {"train": 7 * 3600 + 30 * 60, "select": 7 * 3600 + 30 * 60,
               "test": 4 * 3600 + 30 * 60, "offload": 1800}
RUN_RE = re.compile(r"^(phase_b|phase_b_rebuild)/(5|10|25|40)demos/(single_top|single_side|multi)/train_seed([012])$")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def read_plan(path: Path) -> dict:
    plan = json.loads(read_text(path))
    if plan.get("schema_version") != 1 or not isinstance(plan.get("runs"), list) or not plan["runs"]:
        raise ValueError("plan requires schema_version=1 and nonempty runs")
    if not isinstance(plan.get("inputs"), dict):
        raise ValueError("plan requires input SHA-256 mapping")
    if plan.get("offload_backend", "local") not in ("local", "windows_pull"):
        raise ValueError("offload_backend must be local or windows_pull")
    names = []
    for run in plan["runs"]:
        if not isinstance(run, str) or not RUN_RE.fullmatch(run):
            raise ValueError(f"unsupported run: {run}")
        names.append(run)
    if len(names) != len(set(names)):
        raise ValueError("duplicate run in plan")
    return plan


def verify_inputs(plan: dict) -> None:
    expected = {
        "archives/datasets/act_sim_insertion_scripted_top_side_seed0_50episodes.zip",
        "archives/environments/sim_insertion_validation_seed1100_50.json",
        "archives/environments/sim_insertion_test_seed1200_100.json",
    }
    if set(plan["inputs"]) != expected:
        raise ValueError("plan must pin dataset, validation, and exploratory test inputs")
    for name, sha in plan["inputs"].items():
        path = ROOT / name
        if not path.is_file() or digest(path) != sha:
            raise ValueError(f"input missing or SHA-256 mismatch: {name}")


def run_dir(name: str) -> Path:
    return ROOT / "results/act" / name


def selected_checkpoint(run: Path, allow_archived: bool = False) -> Path:
    summary = run / "checkpoint_selection_deterministic/validation_summary.json"
    data = json.loads(read_text(summary))
    if data.get("selection_dataset") != "fixed_validation_manifest" or data.get("heldout_test_used_for_selection") is not False:
        raise ValueError(f"invalid validation selection: {summary}")
    name = data["selected_checkpoint"]
    if Path(name).name != name or not name.endswith(".ckpt"):
        raise ValueError("unsafe checkpoint name")
    selected = run / name
    if selected.is_file():
        if digest(selected) != data["selected_checkpoint_sha256"]:
            raise ValueError(f"selected checkpoint corrupted: {selected}")
    elif allow_archived:
        receipt_path = run / "selected_checkpoint_drive_receipt.json"
        if not receipt_path.is_file():
            raise ValueError(f"selected checkpoint missing and no archive receipt: {selected}")
        receipt = json.loads(read_text(receipt_path))
        parts = receipt.get("parts", [])
        if (receipt.get("schema_version") != 1 or receipt.get("drive_readback_verified") is not True or
                receipt.get("selected_checkpoint") != name or
                receipt.get("sha256") != data["selected_checkpoint_sha256"] or
                not parts or sum(part["bytes"] for part in parts) != receipt.get("bytes") or
                not all(part.get("drive_file_id") and part.get("sha256") for part in parts)):
            raise ValueError(f"selected checkpoint archive receipt is invalid: {receipt_path}")
    else:
        raise ValueError(f"selected checkpoint missing: {selected}")
    return selected


def verify_stage(stage: str, run: Path) -> bool:
    if stage == "train":
        if not (run / ".training_complete").exists():
            return False
        validation = json.loads(read_text(run / "training_validation.json"))
        summary = json.loads(read_text(run / "training_summary.json"))
        if not validation.get("valid") or summary.get("optimizer_steps") != 10000:
            raise ValueError(f"invalid completed training: {run}")
        checkpoints = json.loads(read_text(run / "checkpoint_manifest.json")).get("checkpoints", [])
        if len(checkpoints) != 4:
            raise ValueError(f"expected four checkpoint records: {run}")
        for row in checkpoints:
            path = run / row["name"]
            # Earlier runs may have pruned nonselected checkpoints after a verified offload.
            if path.exists() and (path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]):
                raise ValueError(f"checkpoint hash mismatch: {path}")
        return True
    if stage == "select":
        if not (run / "checkpoint_selection_deterministic/.complete").exists():
            return False
        selected_checkpoint(run, allow_archived=True)
        return True
    if stage == "test":
        output = run / "eval_test"
        if not (output / ".complete").exists():
            return False
        selected_checkpoint(run, allow_archived=True)
        manifest_hash = output / "test_manifest_sha256.txt"
        if not manifest_hash.is_file() or read_text(manifest_hash).split()[0] != digest(
            ROOT / "archives/environments/sim_insertion_test_seed1200_100.json"
        ):
            raise ValueError(f"missing test manifest hash: {output}")
        rollouts = output / "rollouts_selected_policy.jsonl"
        rows = [json.loads(line) for line in read_text(rollouts).splitlines() if line.strip()]
        rollout_ids = [row.get("rollout_id", row.get("rollout")) for row in rows]
        if len(rows) != 100 or rollout_ids != list(range(100)):
            raise ValueError(f"expected 100 distinct test rollouts: {rollouts}")
        series_index = output / "timeseries/index.jsonl"
        index_rows = [json.loads(line) for line in read_text(series_index).splitlines() if line.strip()]
        if len(index_rows) != 100 or [row.get("rollout_id") for row in index_rows] != list(range(100)) or \
                [row.get("state_id") for row in index_rows] != list(range(100)) or \
                any(bool(index_rows[i]["success"]) != bool(rows[i]["success"]) for i in range(100)):
            raise ValueError(f"expected 100 timeseries index rows: {series_index}")
        checkpoint_hash = read_text(output / "selected_checkpoint_sha256.txt").split()[0]
        selection_hash = json.loads(read_text(run / "checkpoint_selection_deterministic/validation_summary.json"))[
            "selected_checkpoint_sha256"]
        if checkpoint_hash != selection_hash:
            raise ValueError(f"test checkpoint hash differs: {output}")
        return True
    raise ValueError(stage)


def inventory(run: Path) -> list[dict]:
    rows = []
    for path in sorted(run.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlink is not allowed in offload: {path}")
        if path.is_file():
            rows.append({"path": str(path.relative_to(run)), "bytes": path.stat().st_size, "sha256": digest(path)})
    return rows


def verify_copy(directory: Path, files: list[dict]) -> None:
    for row in files:
        path = directory / row["path"]
        if not path.is_file() or path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError(f"offload readback failed: {path}")


def offload(run: Path, destination: Path, prune: bool) -> Path:
    selected = selected_checkpoint(run, allow_archived=True)
    if not verify_stage("test", run):
        raise ValueError("test must finish before offload")
    destination = destination.resolve()
    if destination == ROOT or ROOT in destination.parents:
        raise ValueError("offload destination must be outside the repository")
    target = destination / run.relative_to(ROOT / "results/act")
    receipt = target / "offload_manifest.json"
    if target.exists():
        if not receipt.is_file():
            raise ValueError(f"unverified offload target exists: {target}")
        files = json.loads(read_text(receipt))["files"]
        verify_copy(target, files)
        for row in files:
            source = run / row["path"]
            if source.exists() and (source.stat().st_size != row["bytes"] or digest(source) != row["sha256"]):
                raise ValueError(f"local artifact differs from offload: {source}")
    else:
        files = inventory(run)
        needed = sum(row["bytes"] for row in files)
        target.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(target.parent).free < needed + 2 * 1024**3:
            raise ValueError("offload destination lacks bytes plus 2 GiB reserve")
        temp = target.with_name(target.name + f".partial.{os.getpid()}")
        if temp.exists():
            raise ValueError(f"partial offload exists: {temp}")
        temp.mkdir()
        try:
            for row in files:
                source = run / row["path"]
                output = temp / row["path"]
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, output)
            verify_copy(temp, files)
            verify_copy(run, files)
            atomic_json(temp / "offload_manifest.json", {"schema_version": 1, "source": str(run), "files": files})
            os.rename(temp, target)
            verify_copy(target, files)
        except Exception:
            # Keep the partial copy for inspection; never remove source artifacts.
            raise
    if prune:
        manifest = json.loads(read_text(run / "checkpoint_manifest.json"))
        archived = {row["path"]: row for row in files}
        for row in manifest["checkpoints"]:
            path = run / row["name"]
            if path == selected or not path.exists():
                continue
            saved = archived.get(path.name)
            if saved is None or digest(path) != saved["sha256"]:
                raise ValueError(f"cannot prune unverified checkpoint: {path}")
            path.unlink()
    return target


def stage_state_path(plan_path: Path) -> Path:
    return ROOT / "results/act/sequential" / (plan_path.stem + ".state.json")


def update_state(path: Path, run: str, stage: str, status: str, detail: str = "") -> None:
    state = json.loads(read_text(path)) if path.exists() else {"schema_version": 1, "runs": {}}
    state["runs"].setdefault(run, {})[stage] = {"status": status, "time": time.time(), "detail": detail}
    atomic_json(path, state)


def next_stage(plan: dict, destination: Path | None) -> tuple[str, str] | None:
    for name in plan["runs"]:
        run = run_dir(name)
        for stage in STAGES:
            if stage == "offload":
                if plan.get("offload_backend") == "windows_pull":
                    receipt = run / "offload_windows_receipt.json"
                    manifest = run / "windows_offload_manifest.json"
                    if receipt.exists():
                        if not manifest.is_file():
                            raise ValueError(f"Windows offload manifest missing: {manifest}")
                        data = json.loads(read_text(receipt))
                        if data.get("schema_version") != 1 or data.get("verified") is not True or \
                                data.get("run") != name or data.get("manifest_sha256") != digest(manifest):
                            raise ValueError(f"Windows offload receipt invalid: {receipt}")
                        continue
                else:
                    assert destination is not None
                    target = destination / name
                    if (target / "offload_manifest.json").exists():
                        files = json.loads(read_text(target / "offload_manifest.json"))["files"]
                        verify_copy(target, files)
                        continue
            elif verify_stage(stage, run):
                continue
            if stage != "train" and not verify_stage(STAGES[STAGES.index(stage) - 1], run):
                raise ValueError(f"prior stage incomplete: {name}/{stage}")
            if stage == "train" and name.endswith(("train_seed1", "train_seed2")):
                gate = ROOT / "results/act/phase_b/preflight/phase_b0_complete.json"
                if not gate.is_file():
                    raise ValueError("Phase B0 audit gate is required before seed 1/2 training")
            if stage == "select":
                checkpoints = json.loads(read_text(run / "checkpoint_manifest.json"))["checkpoints"]
                for row in checkpoints:
                    path = run / row["name"]
                    if not path.is_file() or digest(path) != row["sha256"]:
                        raise ValueError(f"checkpoint must be restored before selection: {path}")
            return name, stage
    return None


def quarantine_partial(name: str, stage: str) -> Path | None:
    run = run_dir(name)
    source = {"train": run, "select": run / "checkpoint_selection_deterministic",
              "test": run / "eval_test"}.get(stage)
    if source is None or not source.exists() or (source.is_dir() and not any(source.iterdir())):
        return None
    marker = ".training_complete" if stage == "train" else ".complete"
    if (source / marker).exists():
        raise ValueError(f"refusing to quarantine completed artifact: {source}")
    safe_name = name.replace("/", "_")
    base = ROOT / "results/act/sequential/failed_attempts" / safe_name / stage
    base.mkdir(parents=True, exist_ok=True)
    target = base / f"{int(time.time())}_{os.getpid()}"
    os.rename(source, target)
    return target


def variables(name: str) -> dict[str, str]:
    match = RUN_RE.fullmatch(name)
    assert match
    namespace, demos, view, seed = match.groups()
    return {"PBS_O_WORKDIR": str(ROOT), "EXPERIMENT_STAGE": "phase_b", "TRAINING_METHOD": "naive",
            "OUTPUT_NAMESPACE": namespace, "DEMO_COUNT": demos, "VIEW_CONFIG": view, "TRAIN_SEED": seed,
            "SELECTION_NAME": "checkpoint_selection_deterministic", "DETERMINISTIC_EVAL": "1"}


def admission(stage: str, destination: Path | None) -> None:
    if destination is not None and not destination.is_dir():
        raise ValueError("offload destination is unavailable")
    if stage == "train":
        if shutil.disk_usage(ROOT).free < 1024**3 + 1_500_000_000:
            raise ValueError("home lacks checkpoint estimate plus 1 GiB reserve")
        scratch = Path(os.environ.get("TMP_WORK_BASE", f"/tmp/{os.environ['USER']}_work"))
        scratch.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(scratch).free < 42 * 1024**3:
            raise ValueError("scratch lacks 42 GiB for expanded dataset and runtime")
    elif stage in ("select", "test") and shutil.disk_usage(ROOT).free < 1024**3:
        raise ValueError("home lacks 1 GiB reserve for evaluation artifacts; offload first")
    if destination is not None and shutil.disk_usage(destination).free < 2 * 1024**3 + 1_500_000_000:
        raise ValueError("offload destination lacks next run estimate plus 2 GiB reserve")


def no_active_jobs() -> bool:
    def query():
        try:
            result = subprocess.run(["qstat", "-u", os.environ["USER"]], check=True,
                                    text=True, capture_output=True, timeout=15)
        except subprocess.TimeoutExpired as exc:
            raise OSError(errno.ETIMEDOUT, "qstat timed out; job state is unknown") from exc
        except subprocess.CalledProcessError as exc:
            message = (exc.stderr or "").lower()
            if any(text in message for text in ("connection refused", "cannot connect", "timed out",
                                                "no server to connect", "connection reset")):
                raise OSError(errno.ETIMEDOUT, f"PBS unavailable; job state is unknown: {message.strip()}") from exc
            raise
        return not re.search(r"(?m)^\d+(?:\[.*?\])?\.", result.stdout)
    return retry_read(query, "PBS job status")


def main() -> int:
    global MUTATION_STARTED
    MUTATION_STARTED = False
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "run", "submit-next"))
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/sequential_phase_b.example.json")
    parser.add_argument("--destination", type=Path, help="external offload directory")
    parser.add_argument("--prune", action="store_true", help="delete nonselected local checkpoints after verified readback")
    parser.add_argument("--seconds-left", type=int, help="remaining PBS allocation seconds, for tests only")
    parser.add_argument("--retry-failed", action="store_true", help="explicitly retry a failed or incomplete PBS stage")
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    plan = read_plan(plan_path)
    backend = plan.get("offload_backend", "local")
    destination = args.destination or (Path(plan["offload_destination"]) if plan.get("offload_destination") else None)
    if backend == "local":
        if destination is None or not destination.is_absolute():
            raise SystemExit("set an absolute --destination or offload_destination in the plan")
        destination = destination.resolve()
    elif args.destination is not None:
        raise SystemExit("Windows pull mode does not use --destination on the HPC")
    lock_path = ROOT / "results/act/sequential" / (plan_path.stem + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if args.action == "status":
                print(json.dumps({"next": None, "backend": backend,
                                  "destination": str(destination) if destination else None,
                                  "active_jobs": True, "busy": True}))
                return 0
            if args.action == "run":
                fcntl.flock(lock, fcntl.LOCK_EX)
            else:
                raise ValueError("another runner operation is active")
        verify_inputs(plan)
        pending = next_stage(plan, destination)
        print(json.dumps({"next": pending, "backend": backend, "destination": str(destination) if destination else None,
                          "active_jobs": None if args.action == "run" else not no_active_jobs()}))
        if args.action == "status" or pending is None:
            return 0
        if args.action == "submit-next":
            if pending[1] == "offload" and backend == "windows_pull":
                print("[WAIT] pull and verify this run from Windows first")
                return 0
            if not no_active_jobs():
                print("[WAIT] existing PBS job found", file=sys.stderr)
                return 0
            state_path = stage_state_path(plan_path)
            if state_path.exists():
                state = json.loads(read_text(state_path))
                previous = state.get("runs", {}).get(pending[0], {}).get(pending[1], {})
                if previous.get("status") in ("submitted", "running", "failed") and not args.retry_failed:
                    raise ValueError(f"previous PBS stage did not complete: {pending}; inspect state and retry explicitly")
            admission(pending[1], destination)
            qsub_variables = f"SEQUENTIAL_PLAN={plan_path}"
            if destination is not None:
                qsub_variables += f",OFFLOAD_DESTINATION={destination}"
            # Never ask the controller to retry an uncertain submission.
            MUTATION_STARTED = True
            result = subprocess.run(["qsub", "-v", qsub_variables,
                                     "scripts/run_sequential_experiments.pbs"], cwd=ROOT, check=True,
                                    capture_output=True, text=True)
            update_state(state_path, pending[0], pending[1], "submitted", result.stdout.strip())
            print(f"[SUBMITTED] {result.stdout.strip()}")
            return 0
        if not os.environ.get("PBS_JOBID"):
            raise ValueError("run requires a PBS allocation")
        if pending[1] == "offload" and backend == "windows_pull":
            print("[WAIT] Windows controller must pull and verify this run")
            return 0
        admission(pending[1], destination)
        MUTATION_STARTED = True
        allocation_started = time.monotonic()
        limit = args.seconds_left if args.seconds_left is not None else int(os.environ.get("SEQUENTIAL_WALLTIME_SECONDS", "28800"))
        while pending:
            name, stage = pending
            remaining = limit - int(time.monotonic() - allocation_started)
            if remaining < MIN_SECONDS[stage] + 900:
                print(f"[WAIT] {remaining}s left; {stage} requires {MIN_SECONDS[stage]}s plus 900s margin")
                break
            run = run_dir(name)
            if stage == "offload" and backend == "windows_pull":
                print(f"[WAIT] Windows controller must pull and verify {name}")
                break
            state = stage_state_path(plan_path)
            update_state(state, name, stage, "running")
            try:
                admission(stage, destination)
                if stage == "offload":
                    detail = str(offload(run, destination, args.prune))
                else:
                    archived_attempt = quarantine_partial(name, stage)
                    if archived_attempt:
                        print(f"[RETRY] preserved incomplete artifacts at {archived_attempt}")
                    subprocess.run(["bash", str(ROOT / SCRIPTS[stage])], cwd=ROOT,
                                   env=os.environ | variables(name), check=True)
                    if not verify_stage(stage, run):
                        raise ValueError(f"stage exited without verified artifacts: {name}/{stage}")
                    detail = "artifact checks passed"
                update_state(state, name, stage, "verified", detail)
            except Exception as exc:
                update_state(state, name, stage, "failed", str(exc))
                raise
            pending = next_stage(plan, destination)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except InfrastructureUnavailable as exc:
        print(f"[WAIT_INFRA] {exc}", file=sys.stderr)
        sys.exit(1 if MUTATION_STARTED else 75)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
