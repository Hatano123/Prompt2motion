#!/usr/bin/env python3
"""Run and record a safe end-to-end HR4C mock episode."""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hr4c.mock_adapter import MockHR4CAdapter
from hr4c.recording import write_episode, write_episode_npz
from hr4c.runtime import SafetyConfig, SafetyFilter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(ROOT, "configs/hr4c_mock.json"))
    parser.add_argument("--output", default=os.path.join(ROOT, "results/hr4c/mock/episode_0.hdf5"))
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as source:
        config = json.load(source)
    if config.get("mode") != "mock_only":
        raise SystemExit("This command refuses non-mock configurations")
    arrays = {name: np.asarray(config[name], dtype=float)
              for name in ("joint_min", "joint_max", "max_step")}
    safety = SafetyFilter(SafetyConfig(**arrays,
        observation_timeout_s=config["observation_timeout_s"],
        max_camera_skew_s=config["max_camera_skew_s"]))
    adapter = MockHR4CAdapter(len(config["joint_names"]), config["cameras"])
    observations, actions = [], []
    try:
        for step in range(args.steps):
            observation = adapter.read()
            # A bounded deterministic target stands in for ACT during interface validation.
            raw_target = np.full_like(observation.qpos, min(0.2, 0.002 * (step + 1)))
            action = safety.filter_action(raw_target, observation)
            adapter.command_positions(action)
            observations.append(observation)
            actions.append(action)
    finally:
        adapter.stop()
    output = args.output
    try:
        write_episode(output, observations, actions,
                      {"mode": "mock", "camera_names": json.dumps(config["cameras"])})
        output_format = "act_hdf5"
    except RuntimeError as error:
        if "h5py" not in str(error):
            raise
        output = os.path.splitext(output)[0] + ".npz"
        write_episode_npz(output, observations, actions)
        output_format = "diagnostic_npz"
    print(json.dumps({"output": output, "format": output_format, "steps": len(actions),
                      "cameras": config["cameras"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
