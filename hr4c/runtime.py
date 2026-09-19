"""Minimal, fail-closed runtime shared by mock and future HR4C adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import time

import numpy as np


class SafetyViolation(RuntimeError):
    """Raised when an action must not be sent to the robot."""


@dataclass(frozen=True)
class Observation:
    timestamp: float
    qpos: np.ndarray
    images: Mapping[str, np.ndarray]
    image_timestamps: Mapping[str, float]
    estop: bool = False


@dataclass(frozen=True)
class SafetyConfig:
    joint_min: np.ndarray
    joint_max: np.ndarray
    max_step: np.ndarray
    observation_timeout_s: float
    max_camera_skew_s: float

    def __post_init__(self) -> None:
        sizes = {np.asarray(x).size for x in (self.joint_min, self.joint_max, self.max_step)}
        if len(sizes) != 1 or next(iter(sizes)) == 0:
            raise ValueError("joint safety arrays must have the same non-zero length")
        if np.any(np.asarray(self.joint_min) >= np.asarray(self.joint_max)):
            raise ValueError("every joint_min must be smaller than joint_max")
        if np.any(np.asarray(self.max_step) <= 0):
            raise ValueError("max_step values must be positive")


class SafetyFilter:
    """Validate observations and clamp per-tick motion; never clips joint limits."""

    def __init__(self, config: SafetyConfig):
        self.config = config

    def validate_observation(self, observation: Observation, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        if observation.estop:
            raise SafetyViolation("emergency/protective stop is active")
        if now - observation.timestamp > self.config.observation_timeout_s:
            raise SafetyViolation("robot state is stale")
        if not observation.images:
            raise SafetyViolation("no camera frame available")
        missing = set(observation.images) - set(observation.image_timestamps)
        if missing:
            raise SafetyViolation(f"camera timestamps missing: {sorted(missing)}")
        timestamps = [observation.image_timestamps[name] for name in observation.images]
        if max(timestamps) - min(timestamps) > self.config.max_camera_skew_s:
            raise SafetyViolation("camera frames are not synchronized")
        if np.asarray(observation.qpos).shape != np.asarray(self.config.joint_min).shape:
            raise SafetyViolation("qpos dimension does not match configured joints")
        if not np.all(np.isfinite(observation.qpos)):
            raise SafetyViolation("qpos contains a non-finite value")

    def filter_action(self, target: np.ndarray, observation: Observation,
                      now: float | None = None) -> np.ndarray:
        self.validate_observation(observation, now)
        target = np.asarray(target, dtype=np.float64)
        qpos = np.asarray(observation.qpos, dtype=np.float64)
        if target.shape != qpos.shape or not np.all(np.isfinite(target)):
            raise SafetyViolation("action has invalid dimension or value")
        if np.any(target < self.config.joint_min) or np.any(target > self.config.joint_max):
            raise SafetyViolation("action exceeds a configured joint limit")
        delta = np.clip(target - qpos, -self.config.max_step, self.config.max_step)
        return qpos + delta
