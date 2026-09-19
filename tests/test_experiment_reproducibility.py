import json
import os
import sys
import tempfile
import unittest

import numpy as np

REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACT_ROOT = os.path.join(REPOSITORY_ROOT, 'repos', 'act')
SCRIPTS_ROOT = os.path.join(REPOSITORY_ROOT, 'scripts')
sys.path.insert(0, ACT_ROOT)
sys.path.insert(0, SCRIPTS_ROOT)

from experiment_utils import sample_occlusion_rectangle
from generate_insertion_eval_manifest import generate_states


class ReproducibilityTest(unittest.TestCase):
    def test_initial_states_are_reproducible_and_distinct(self):
        first = generate_states(1000, 50)
        second = generate_states(1000, 50)
        self.assertEqual(first, second)
        self.assertEqual(50, len({tuple(state) for state in first}))

    def test_validation_and_test_states_are_disjoint(self):
        validation = generate_states(1100, 50)
        test = generate_states(1200, 100)
        self.assertFalse({tuple(state) for state in validation} &
                         {tuple(state) for state in test})

    def test_occlusion_area_and_seed(self):
        for fraction in (0.25, 0.5):
            rectangle = sample_occlusion_rectangle(480, 640, fraction, 2007)
            self.assertEqual(rectangle, sample_occlusion_rectangle(480, 640, fraction, 2007))
            y0, y1, x0, x1 = rectangle
            actual_fraction = (y1 - y0) * (x1 - x0) / (480 * 640)
            self.assertAlmostEqual(fraction, actual_fraction, delta=0.002)

    def test_one_dataset_supports_all_camera_selections(self):
        try:
            import h5py
            from utils import EpisodicDataset, get_norm_stats, load_data
        except ModuleNotFoundError as error:
            self.skipTest(f'ACT test dependency unavailable: {error.name}')
        with tempfile.TemporaryDirectory() as directory:
            for episode_id in range(5):
                path = os.path.join(directory, f'episode_{episode_id}.hdf5')
                with h5py.File(path, 'w') as root:
                    root.attrs['sim'] = True
                    root.create_dataset('action', data=np.full((2, 14), episode_id, dtype=float))
                    observations = root.create_group('observations')
                    observations.create_dataset('qpos', data=np.full((2, 14), episode_id, dtype=float))
                    observations.create_dataset('qvel', data=np.zeros((2, 14), dtype=float))
                    images = observations.create_group('images')
                    images.create_dataset('top', data=np.full((2, 4, 5, 3), 10, dtype=np.uint8))
                    images.create_dataset('side', data=np.full((2, 4, 5, 3), 20, dtype=np.uint8))

            stats = get_norm_stats(directory, 5)
            for cameras in (['top'], ['side'], ['top', 'side']):
                dataset = EpisodicDataset(np.array([0]), directory, cameras, stats)
                image, _, _, _ = dataset[0]
                self.assertEqual(len(cameras), image.shape[0])
                expected = [10 / 255, 20 / 255] if len(cameras) == 2 else [
                    (10 if cameras[0] == 'top' else 20) / 255]
                for camera_id, value in enumerate(expected):
                    self.assertTrue(np.allclose(image[camera_id].numpy(), value))

            manifest = os.path.join(directory, 'split.json')
            load_data(directory, 5, ['top'], 2, 1, split_seed=1,
                      train_episode_ids='0,1,2,3', val_episode_ids='4',
                      split_manifest_path=manifest)
            with open(manifest) as source:
                split = json.load(source)
            self.assertEqual([0, 1, 2, 3], split['train_episode_ids'])
            self.assertEqual([4], split['val_episode_ids'])

            train_only_stats = get_norm_stats(directory, episode_ids=[0, 1, 2, 3])
            self.assertTrue(np.allclose(train_only_stats['action_mean'], 1.5))
            self.assertFalse(np.allclose(train_only_stats['action_mean'],
                                         get_norm_stats(directory, episode_ids=[0, 1, 2, 3, 4])['action_mean']))


if __name__ == '__main__':
    unittest.main()
