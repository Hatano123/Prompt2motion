import json
import os
import unittest
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from summarize_clean50 import exact_mcnemar


class Clean50DesignTest(unittest.TestCase):
    def test_config_separates_validation_and_test(self):
        with open(os.path.join(ROOT, 'configs', 'clean50.json')) as source:
            config = json.load(source)
        self.assertNotEqual(config['validation']['seed'], config['test']['seed'])
        self.assertNotEqual(config['validation']['manifest'], config['test']['manifest'])
        self.assertEqual(50, config['validation']['rollouts'])
        self.assertEqual(100, config['test']['rollouts'])

    def test_training_script_never_reads_test_manifest(self):
        path = os.path.join(ROOT, 'scripts', 'run_clean50_train.pbs')
        with open(path) as source:
            text = source.read()
        self.assertIn('validation_seed1100_50', text)
        self.assertNotIn('test_seed1200_100', text)

    def test_test_script_only_uses_selected_policy(self):
        path = os.path.join(ROOT, 'scripts', 'run_clean50_test.pbs')
        with open(path) as source:
            text = source.read()
        self.assertIn('--eval_ckpt selected_policy.ckpt', text)
        self.assertNotIn('policy_epoch_', text)

    def test_exact_mcnemar_uses_paired_outcomes(self):
        first_only, second_only, p_value = exact_mcnemar(
            [True, True, False, False], [False, True, True, False])
        self.assertEqual((1, 1), (first_only, second_only))
        self.assertEqual(1.0, p_value)


if __name__ == '__main__':
    unittest.main()
