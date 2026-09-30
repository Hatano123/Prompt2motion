"""Fail-closed bounded control loop shared by pilot and future hardware adapters."""

from __future__ import annotations

from collections.abc import Callable
import numpy as np

from .adapter import HR4CAdapter
from .runtime import Observation, SafetyFilter


def run_bounded_episode(adapter: HR4CAdapter, safety: SafetyFilter,
                        policy: Callable[[Observation], np.ndarray], max_steps: int,
                        continue_signal: Callable[[], bool]) -> list[tuple[Observation, np.ndarray]]:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    records = []
    try:
        for _ in range(max_steps):
            if not continue_signal():
                raise RuntimeError("operator continue/deadman signal released")
            observation = adapter.read()
            command = safety.filter_action(policy(observation), observation)
            adapter.command_positions(command)
            records.append((observation, command.copy()))
    finally:
        adapter.stop()
    return records
