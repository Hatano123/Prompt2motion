import json
import os
import sys
import tempfile
import unittest

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "repos", "act"))

from rollout_logging import (capture_contacts, make_step_record, milestone_timesteps,
                             save_episode_logs, validate_episode_logs)


class _Contact:
    def __init__(self, geom1, geom2):
        self.geom1 = geom1
        self.geom2 = geom2


class _Data:
    def __init__(self):
        self.contact = [_Contact(10, 20), _Contact(30, 40)]
        self.ncon = len(self.contact)


class _Model:
    names = {
        10: "socket-1",
        20: "red_peg",
        30: "vx300s_right/10_right_gripper_finger",
        40: "red_peg",
    }

    def id2name(self, identifier, kind):
        assert kind == "geom"
        return self.names[identifier]


class _Physics:
    def __init__(self):
        self.data = _Data()
        self.model = _Model()


class RolloutLoggingTest(unittest.TestCase):
    def test_contact_capture_is_order_independent_and_non_mutating(self):
        physics = _Physics()
        before = [(contact.geom1, contact.geom2) for contact in physics.data.contact]
        contacts = capture_contacts(physics)
        after = [(contact.geom1, contact.geom2) for contact in physics.data.contact]
        self.assertEqual(before, after)
        self.assertTrue(contacts["peg_socket"])
        self.assertTrue(contacts["right_gripper_peg"])

    def test_milestones_use_first_at_or_above_reward(self):
        self.assertEqual({
            "first_reward1_timestep": 1,
            "first_reward2_timestep": 2,
            "first_reward3_timestep": 4,
            "first_reward4_timestep": None,
        }, milestone_timesteps([0, 1, 2, 2, 3]))

    def test_round_trip_and_validator(self):
        physics = _Physics()
        records = []
        rewards = [0, 1, 3, 4]
        for timestep, reward in enumerate(rewards):
            env_state = np.array([
                0.1 + timestep, 0.2, 0.3, 1, 0, 0, 0,
                -0.1, 0.4, 0.5, 1, 0, 0, 0,
            ], dtype=np.float64)
            records.append(make_step_record(
                timestep, reward, np.full(14, timestep / 10), np.full(14, timestep),
                np.arange(14), np.arange(14) + 0.5, env_state, physics))

        initial_state = np.arange(14, dtype=float)
        with tempfile.TemporaryDirectory() as directory:
            summary_path, timeseries_path = save_episode_logs(
                directory, 3, records, {"initial_state": initial_state.tolist()})
            result = validate_episode_logs(summary_path, initial_state)
            self.assertTrue(result["valid"])
            self.assertGreater(timeseries_path.stat().st_size, 0)
            summary = json.loads(summary_path.read_text())
            self.assertEqual(4, summary["episode_length"])
            self.assertEqual(8.0, summary["episode_return"])
            self.assertEqual(4.0, summary["highest_reward"])
            self.assertEqual(2, summary["first_reward3_timestep"])
            self.assertEqual(3, summary["first_reward4_timestep"])
            self.assertEqual(1, summary["reward3_to_reward4_steps"])
            with np.load(timeseries_path, allow_pickle=False) as archive:
                self.assertEqual((4, 14), archive["action"].shape)
                self.assertTrue(np.array_equal(archive["action"], archive["target_qpos"]))
                self.assertEqual(np.float32, archive["peg_position"].dtype)
                self.assertEqual("socket-1", archive["contact_geom1_name"][0, 0])
                self.assertFalse(any(np.issubdtype(archive[key].dtype, np.number)
                                     and not np.isfinite(archive[key]).all()
                                     for key in archive.files))

    def test_collection_does_not_change_deterministic_behavior(self):
        def run(logging_enabled):
            physics = _Physics()
            state = np.zeros(14, dtype=np.float64)
            env_state = np.array([0, 0, 0, 1, 0, 0, 0,
                                  0, 0, 0, 1, 0, 0, 0], dtype=np.float64)
            actions, rewards = [], []
            for timestep in range(12):
                observed = state.copy()
                action = np.linspace(-0.1, 0.1, 14) + timestep * 0.001
                state = 0.8 * state + 0.2 * action
                env_state[:3] += state[:3] * 0.0001
                reward = float(timestep // 3)
                if logging_enabled:
                    make_step_record(timestep, reward, action, action, observed,
                                     state, env_state, physics)
                actions.append(action.copy())
                rewards.append(reward)
            return np.stack(actions), np.asarray(rewards), state.copy(), env_state.copy()

        disabled = run(False)
        enabled = run(True)
        for disabled_value, enabled_value in zip(disabled, enabled):
            self.assertTrue(np.array_equal(disabled_value, enabled_value))


if __name__ == "__main__":
    unittest.main()
