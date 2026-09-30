"""Interfaces a vendor-specific HR4C driver must implement."""

from __future__ import annotations

from typing import Protocol
import numpy as np

from .runtime import Observation


class HR4CAdapter(Protocol):
    def read(self) -> Observation: ...
    def command_positions(self, target: np.ndarray) -> None: ...
    def stop(self) -> None: ...


def validate_adapter(adapter: HR4CAdapter, joint_count: int) -> Observation:
    observation = adapter.read()
    if np.asarray(observation.qpos).shape != (joint_count,):
        raise RuntimeError("adapter joint order/dimension does not match the audited configuration")
    return observation
