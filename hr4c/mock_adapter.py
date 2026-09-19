"""Deterministic adapter for exercising the complete runtime without hardware."""

from __future__ import annotations

import time
import numpy as np

from .runtime import Observation


class MockHR4CAdapter:
    def __init__(self, joint_count: int, cameras: list[str], image_size=(48, 64)):
        self.qpos = np.zeros(joint_count, dtype=np.float64)
        self.cameras = cameras
        self.image_size = image_size
        self.stopped = False

    def read(self) -> Observation:
        now = time.monotonic()
        images = {
            name: np.full((*self.image_size, 3), 30 + 40 * index, dtype=np.uint8)
            for index, name in enumerate(self.cameras)
        }
        return Observation(now, self.qpos.copy(), images,
                           {name: now for name in self.cameras}, self.stopped)

    def command_positions(self, target: np.ndarray) -> None:
        if self.stopped:
            raise RuntimeError("adapter is stopped")
        self.qpos = np.asarray(target, dtype=np.float64).copy()

    def stop(self) -> None:
        self.stopped = True
