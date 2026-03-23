import unittest

from scheduler.update_job import (
    _compare_bundles_for_promotion,
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


if __name__ == '__main__':
    unittest.main()
