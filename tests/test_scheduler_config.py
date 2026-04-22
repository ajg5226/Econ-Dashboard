import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scheduler import scheduler_config


class TestSchedulerConfig(unittest.TestCase):
    def test_load_runtime_config_logs_and_defaults_on_invalid_json(self):
        with TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'runtime_config.json'
            config_path.write_text('{invalid json', encoding='utf-8')

            with patch.object(scheduler_config, 'CONFIG_PATH', config_path):
                with self.assertLogs('scheduler.scheduler_config', level='WARNING') as logs:
                    loaded = scheduler_config.load_runtime_config()

        self.assertEqual(loaded, scheduler_config.DEFAULT_CONFIG)
        self.assertIn('not valid JSON', "\n".join(logs.output))

    def test_validate_config_ignores_invalid_train_end_date(self):
        with self.assertLogs('scheduler.scheduler_config', level='WARNING') as logs:
            validated = scheduler_config._validate_config({
                'train_end_date': '2026-02-30',
                'max_features': 75,
            })

        self.assertIsNone(validated['train_end_date'])
        self.assertEqual(validated['max_features'], 75)
        self.assertIn('Ignoring invalid train_end_date', "\n".join(logs.output))
