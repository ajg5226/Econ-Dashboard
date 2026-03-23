import unittest

import numpy as np
import pandas as pd

from recession_engine.backtester import RecessionBacktester


class DummyAcquisition:
    def __init__(self):
        self.fred = None

    def engineer_features(self, df):
        df_feat = df.copy()
        df_feat['signal_feature'] = (
            df_feat['RECESSION']
            .rolling(window=6, min_periods=1)
            .max()
            .shift(-6)
            .fillna(0)
        )
        return df_feat

    def create_forecast_target(self, df, horizon_months):
        df_target = df.copy()
        df_target[f'RECESSION_FORWARD_{horizon_months}M'] = (
            df_target['RECESSION']
            .rolling(window=horizon_months, min_periods=1)
            .max()
            .shift(-horizon_months)
        )
        return df_target


class RecordingModel:
    fit_end_dates = []

    def __init__(self, target_horizon=6, n_cv_splits=5, model_config=None):
        self.target_horizon = target_horizon
        self.target_col = f'RECESSION_FORWARD_{target_horizon}M'
        self.model_config = model_config or {}
        self.decision_threshold = 0.5
        self.ensemble_weights = {'dummy': 1.0}
        self.active_models = ['dummy']
        self.feature_cols = []

    def fit(self, train_df, max_features=50):
        self.feature_cols = [c for c in train_df.columns if c not in ['RECESSION', self.target_col]]
        RecordingModel.fit_end_dates.append(train_df.index.max())

    def predict(self, test_df):
        direction = float(self.model_config.get('direction', 1.0))
        signal = test_df['signal_feature'].astype(float).to_numpy()
        probs = np.where(signal > 0.5, 0.8, 0.2)
        if direction < 0:
            probs = 1.0 - probs
        return {'ensemble': probs}


class TestBacktester(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range('1990-01-31', periods=420, freq='ME')
        recession = np.zeros(len(dates))
        recession[120:132] = 1
        recession[240:252] = 1
        recession[336:342] = 1
        self.df_raw = pd.DataFrame(
            {
                'leading_T10Y3M': np.linspace(1.0, -1.0, len(dates)),
                'leading_ICSA': np.linspace(100, 250, len(dates)),
                'RECESSION': recession,
            },
            index=dates,
        )
        RecordingModel.fit_end_dates = []
        self.backtester = RecessionBacktester(DummyAcquisition(), RecordingModel, target_horizon=6)

    def test_strict_realtime_backtest_respects_label_cutoff(self):
        results = self.backtester.run_strict_realtime_backtest(
            self.df_raw,
            origin_dates=[('2010-06', 'origin')],
            min_train_months=120,
            candidate_id='test',
        )

        self.assertEqual(len(results), 1)
        row = results.iloc[0]
        self.assertEqual(row['Train_Label_Cutoff'], '2009-12')
        self.assertEqual(RecordingModel.fit_end_dates[-1].strftime('%Y-%m'), '2009-12')
        self.assertIn('Prob_Ensemble', results.columns)

    def test_model_config_search_ranks_better_candidate_first(self):
        candidate_configs = [
            {
                'id': 'good',
                'description': 'Aligned predictor',
                'max_features': 10,
                'n_cv_splits': 3,
                'model_config': {'direction': 1.0},
            },
            {
                'id': 'bad',
                'description': 'Inverted predictor',
                'max_features': 10,
                'n_cv_splits': 3,
                'model_config': {'direction': -1.0},
            },
        ]

        search_result = self.backtester.run_model_config_search(
            self.df_raw,
            candidate_configs=candidate_configs,
            origin_dates=[
                ('2000-06', 'early'),
                ('2008-06', 'mid'),
                ('2018-06', 'late'),
            ],
            min_train_months=120,
            alfred_top_k=0,
        )

        search_df = search_result['search_results']
        self.assertFalse(search_df.empty)
        self.assertEqual(search_df.iloc[0]['Candidate_ID'], 'good')
        self.assertEqual(search_result['best_candidate']['id'], 'good')


if __name__ == '__main__':
    unittest.main()
