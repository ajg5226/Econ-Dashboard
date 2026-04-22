import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from scheduler.update_job import (
    _compare_bundles_for_promotion,
    _load_incumbent_snapshot,
    _resolve_manifest_model_config,
)


class TestUpdateJobHelpers(unittest.TestCase):
    def test_compare_bundles_blocks_underperforming_candidate(self):
        candidate_bundle = {
            'ensemble_metrics': {
                'AUC': 0.62,
                'PR_AUC': 0.15,
                'Brier': 0.058,
                'LogLoss': 0.41,
            }
        }
        incumbent_bundle = {
            'ensemble_metrics': {
                'AUC': 0.72,
                'PR_AUC': 0.26,
                'Brier': 0.056,
                'LogLoss': 0.21,
            }
        }

        result = _compare_bundles_for_promotion(candidate_bundle, incumbent_bundle)

        self.assertFalse(result['passed'])
        self.assertIn('AUC', result['checks'])
        self.assertFalse(result['checks']['AUC']['passed'])
        self.assertFalse(result['checks']['PR_AUC']['passed'])

    def test_compare_bundles_allows_candidate_that_matches_or_improves_all_metrics(self):
        candidate_bundle = {
            'ensemble_metrics': {
                'AUC': 0.73,
                'PR_AUC': 0.28,
                'Brier': 0.054,
                'LogLoss': 0.20,
            }
        }
        incumbent_bundle = {
            'ensemble_metrics': {
                'AUC': 0.72,
                'PR_AUC': 0.26,
                'Brier': 0.056,
                'LogLoss': 0.21,
            }
        }

        result = _compare_bundles_for_promotion(candidate_bundle, incumbent_bundle)

        self.assertTrue(result['passed'])
        self.assertTrue(result['checks']['AUC']['passed'])
        self.assertTrue(result['checks']['Brier']['passed'])

    def test_resolve_manifest_model_config_falls_back_to_default_search_candidate(self):
        manifest = {
            'model_selection': {
                'candidate_id': 'conservative_40',
            }
        }

        config = _resolve_manifest_model_config(manifest)

        self.assertEqual(config['probit']['C'], 0.08)
        self.assertEqual(config['random_forest']['max_depth'], 6)

    def test_load_incumbent_snapshot_coerces_invalid_manifest_fields(self):
        with TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir)
            pd.DataFrame(
                [
                    {
                        'Model': 'ensemble',
                        'AUC': 0.71,
                        'PR_AUC': 0.24,
                        'Brier': 0.056,
                        'LogLoss': 0.22,
                    }
                ]
            ).to_csv(models_dir / 'metrics.csv', index=False)
            (models_dir / 'run_manifest.json').write_text(
                (
                    '{'
                    '"max_features": "oops", '
                    '"n_cv_splits": "bad", '
                    '"model_config": ["not", "a", "dict"]'
                    '}'
                ),
                encoding='utf-8',
            )

            with self.assertLogs('scheduler.update_job', level='WARNING') as logs:
                snapshot = _load_incumbent_snapshot(models_dir)

        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot['max_features'], 50)
        self.assertEqual(snapshot['n_cv_splits'], 5)
        self.assertEqual(snapshot['model_config'], {})
        joined_logs = "\n".join(logs.output)
        self.assertIn('Invalid max_features', joined_logs)
        self.assertIn('Invalid n_cv_splits', joined_logs)
        self.assertIn('Invalid model_config', joined_logs)


if __name__ == '__main__':
    unittest.main()
