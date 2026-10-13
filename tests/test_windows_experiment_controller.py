import sys
from pathlib import Path
from types import SimpleNamespace
import subprocess
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import windows_experiment_controller as client


class ControllerTests(unittest.TestCase):
    def test_remote_infrastructure_failure_waits_and_never_submits(self):
        result = subprocess.CompletedProcess([], 75, b'', b'[WAIT_INFRA] PBS unavailable')
        with patch.object(client.subprocess, 'run', return_value=result):
            with self.assertRaises(client.RemoteInfrastructureUnavailable):
                client.remote_command('host', '/repo', ['status'])
        args = SimpleNamespace(remote='host', remote_root='/repo', plan='plan',
                               destination=Path('/unused'), loop=True, poll_seconds=120)
        with patch.object(client, 'cycle', side_effect=[
                client.RemoteInfrastructureUnavailable('read unavailable'), False]) as cycle, \
                patch.object(client.time, 'sleep') as sleep, patch.object(client, 'pull') as pull:
            client.run_loop(args)
            self.assertEqual(cycle.call_count, 2)
            sleep.assert_called_once_with(120)
            pull.assert_not_called()

    def test_busy_null_next_does_not_end_loop(self):
        with patch.object(client, 'remote_command', return_value='{"next":null,"busy":true,"active_jobs":true,"backend":"windows_pull"}'), patch.object(client, 'pull') as pull:
            self.assertTrue(client.cycle('host', '/repo', 'plan', Path('/unused')))
            pull.assert_not_called()

    def test_only_idle_complete_ends_loop(self):
        with patch.object(client, 'remote_command', return_value='{"next":null,"active_jobs":false,"backend":"windows_pull"}'):
            self.assertFalse(client.cycle('host', '/repo', 'plan', Path('/unused')))

    def test_connection_failure_retries_then_resumes(self):
        args = SimpleNamespace(remote='host', remote_root='/repo', plan='plan', destination=Path('/unused'), loop=True, poll_seconds=120)
        with patch.object(client, 'cycle', side_effect=[client.ConnectionLost('offline'), True, False]) as cycle, patch.object(client.time, 'sleep') as sleep:
            client.run_loop(args)
            self.assertEqual(cycle.call_count, 3)
            self.assertEqual(sleep.call_count, 2)

    def test_validation_failure_is_not_retried(self):
        args = SimpleNamespace(remote='host', remote_root='/repo', plan='plan', destination=Path('/unused'), loop=True, poll_seconds=120)
        with patch.object(client, 'cycle', side_effect=ValueError('hash mismatch')), patch.object(client.time, 'sleep') as sleep:
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                client.run_loop(args)
            sleep.assert_not_called()

    def test_remote_errors_distinguish_transport_from_experiment(self):
        for code, exception in [(255, client.ConnectionLost), (1, subprocess.CalledProcessError)]:
            result = subprocess.CompletedProcess([], code, b'', b'error')
            with patch.object(client.subprocess, 'run', return_value=result):
                with self.assertRaises(exception):
                    client.remote_command('host', '/repo', ['true'])


if __name__ == '__main__':
    unittest.main()
