#!/usr/bin/env python3
"""Offline Phase E readiness audit.  This script never connects to hardware."""

import argparse
import importlib.util
import json
from pathlib import Path


def audit(config, base):
    errors = []
    placeholders = []
    def placeholder(value): return not value or (isinstance(value, str) and "REPLACE" in value)
    for field in ("adapter_module", "robot_serial", "ros_version"):
        if placeholder(config.get(field)): placeholders.append(field)
    joints = config.get("joint_names", [])
    if len(joints) != 14 or len(set(joints)) != len(joints): errors.append("joint_names must contain 14 unique entries")
    for field in ("joint_min", "joint_max", "max_step"):
        if len(config.get(field, [])) != len(joints): errors.append(f"{field} length must match joint_names")
    for camera in ("top", "side"):
        item = config.get("cameras", {}).get(camera, {})
        for field in ("frame_id", "intrinsics_file", "extrinsics_file"):
            value = item.get(field)
            if placeholder(value): placeholders.append(f"cameras.{camera}.{field}")
            elif field.endswith("file") and not (base / value).is_file(): errors.append(f"missing {camera} {field}: {value}")
    for path in config.get("safety_documents", []):
        if not (base / path).is_file(): errors.append(f"missing safety document: {path}")
    gates = ["estop_verified", "protective_stop_verified", "synchronized_dual_arm_stop_verified",
             "workspace_limits_verified", "insertion_force_limit_verified"]
    for gate in gates:
        if config.get(gate) is not True: errors.append(f"safety gate is not verified: {gate}")
    module = config.get("adapter_module")
    if module and "REPLACE" not in module and importlib.util.find_spec(module) is None:
        errors.append(f"adapter module is not importable: {module}")
    if config.get("hardware_enabled") is not True: errors.append("hardware_enabled is false")
    return {"ready": not errors and not placeholders, "errors": errors, "placeholders": placeholders,
            "hardware_contact_attempted": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    report = audit(config, args.config.resolve().parent)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists(): raise SystemExit(f"refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(rendered)
    print(rendered, end="")
    raise SystemExit(0 if report["ready"] else 2)


if __name__ == "__main__": main()
