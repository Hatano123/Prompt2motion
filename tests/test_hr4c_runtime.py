import os
import tempfile
import time
import unittest

import numpy as np

from hr4c.recording import write_episode, write_episode_npz
from hr4c.runtime import Observation, SafetyConfig, SafetyFilter, SafetyViolation


class HR4CRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.filter = SafetyFilter(SafetyConfig(
            joint_min=np.array([-1.0, -1.0]), joint_max=np.array([1.0, 1.0]),
            max_step=np.array([0.1, 0.2]), observation_timeout_s=0.2,
            max_camera_skew_s=0.03))

    def observation(self, **changes):
        now = time.monotonic()
        values = dict(timestamp=now, qpos=np.zeros(2),
                      images={"top": np.zeros((2, 3, 3), dtype=np.uint8),
                              "side": np.zeros((2, 3, 3), dtype=np.uint8)},
                      image_timestamps={"top": now, "side": now}, estop=False)
        values.update(changes)
        return Observation(**values)

    def test_action_is_rate_limited(self):
        actual = self.filter.filter_action(np.array([0.8, -0.8]), self.observation())
        np.testing.assert_allclose(actual, [0.1, -0.2])

    def test_limit_estop_stale_and_skew_fail_closed(self):
        with self.assertRaises(SafetyViolation):
            self.filter.filter_action(np.array([2.0, 0.0]), self.observation())
        with self.assertRaises(SafetyViolation):
            self.filter.filter_action(np.zeros(2), self.observation(estop=True))
        with self.assertRaises(SafetyViolation):
            self.filter.filter_action(np.zeros(2), self.observation(timestamp=time.monotonic() - 1))
        now = time.monotonic()
        with self.assertRaises(SafetyViolation):
            self.filter.filter_action(np.zeros(2), self.observation(
                image_timestamps={"top": now, "side": now - 0.1}))

    def test_episode_is_act_compatible(self):
        try:
            import h5py
        except ModuleNotFoundError:
            self.skipTest("h5py unavailable")
        observation = self.observation()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "episode_0.hdf5")
            write_episode(path, [observation, observation], [np.zeros(2), np.ones(2)])
            with h5py.File(path) as root:
                self.assertFalse(root.attrs["sim"])
                self.assertEqual((2, 2), root["action"].shape)
                self.assertEqual((2, 2, 3, 3), root["observations/images/top"].shape)

    def test_dependency_free_diagnostic_recording(self):
        observation = self.observation()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "episode_0.npz")
            write_episode_npz(path, [observation], [np.zeros(2)])
            with np.load(path) as data:
                self.assertEqual((1, 2), data["action"].shape)
                self.assertEqual((1, 2, 3, 3), data["image_top"].shape)


if __name__ == "__main__":
    unittest.main()
