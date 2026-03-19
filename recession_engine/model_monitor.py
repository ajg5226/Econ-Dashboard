"""
Model Monitoring & Drift Detection Module

Provides lightweight monitoring for production recession prediction models:
1. Feature drift detection (Population Stability Index)
2. Prediction stability tracking (rolling volatility of ensemble output)
3. Model disagreement alerting (base model spread)
4. Calibration drift (observed vs predicted recession rates)
"""

import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelMonitor:
    """
    Lightweight model monitoring for recession prediction ensemble.

    Designed for monthly-frequency economic models where:
    - Feature distributions shift slowly (business cycle dynamics)
    - Recessions are rare events (~12% base rate)
    - Model updates happen weekly/monthly, not real-time
    """

    def __init__(self, data_dir: Path = None):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "data" / "models"
        self.data_dir = data_dir
        self.alerts = []

    def run_all_checks(self, predictions_df: pd.DataFrame,
                       indicators_df: pd.DataFrame = None,
                       feature_cols: list = None) -> dict:
        """
        Run all monitoring checks and return a summary report.

        Args:
            predictions_df: DataFrame with Prob_Ensemble, Prob_Probit, etc. (Date index)
            indicators_df: Optional DataFrame with raw indicator values
            feature_cols: Optional list of feature column names used by the model

        Returns:
            dict with monitoring results and alerts
        """
        self.alerts = []
        report = {
            'timestamp': datetime.now().isoformat(),
            'checks': {},
        }

        # 1. Prediction stability
        report['checks']['prediction_stability'] = self._check_prediction_stability(
            predictions_df
        )

        # 2. Model disagreement
        report['checks']['model_disagreement'] = self._check_model_disagreement(
            predictions_df
        )

        # 3. Feature drift (if indicators provided)
        if indicators_df is not None and feature_cols:
            report['checks']['feature_drift'] = self._check_feature_drift(
                indicators_df, feature_cols
            )

        # 4. Calibration drift
        if 'Actual_Recession' in predictions_df.columns:
            report['checks']['calibration'] = self._check_calibration_drift(
                predictions_df
            )

        # 5. Data freshness
        report['checks']['data_freshness'] = self._check_data_freshness(
            predictions_df
        )

        report['alerts'] = self.alerts
        report['alert_count'] = len(self.alerts)
        report['status'] = 'WARNING' if self.alerts else 'OK'

        return report

    def _check_prediction_stability(self, df: pd.DataFrame,
                                     window: int = 6,
                                     vol_threshold: float = 0.15) -> dict:
        """
        Check if ensemble predictions are unusually volatile.

        A sudden spike in rolling std suggests the model is responding to
        a regime change or data anomaly.
        """
        result = {'status': 'OK', 'details': {}}

        if 'Prob_Ensemble' not in df.columns or len(df) < window:
            result['status'] = 'SKIP'
            result['details']['reason'] = 'Insufficient data'
            return result

        ensemble = df['Prob_Ensemble'].astype(float)

        # Rolling 6-month volatility of predictions
        rolling_vol = ensemble.rolling(window).std()
        current_vol = rolling_vol.iloc[-1] if not rolling_vol.empty else 0
        mean_vol = rolling_vol.mean()
        max_vol = rolling_vol.max()

        result['details'] = {
            'current_6m_vol': round(float(current_vol), 4) if pd.notna(current_vol) else None,
            'historical_mean_vol': round(float(mean_vol), 4) if pd.notna(mean_vol) else None,
            'historical_max_vol': round(float(max_vol), 4) if pd.notna(max_vol) else None,
        }

        # Month-over-month change
        if len(ensemble) >= 2:
            mom_change = abs(float(ensemble.iloc[-1]) - float(ensemble.iloc[-2]))
            result['details']['last_mom_change'] = round(mom_change, 4)

            if mom_change > 0.10:
                self.alerts.append({
                    'level': 'WARNING',
                    'check': 'prediction_stability',
                    'message': f'Large month-over-month probability change: {mom_change:.1%}'
                })

        if pd.notna(current_vol) and current_vol > vol_threshold:
            result['status'] = 'WARNING'
            self.alerts.append({
                'level': 'WARNING',
                'check': 'prediction_stability',
                'message': f'Elevated prediction volatility: {current_vol:.3f} (threshold: {vol_threshold})'
            })

        return result

    def _check_model_disagreement(self, df: pd.DataFrame,
                                   spread_threshold: float = 0.30) -> dict:
        """
        Check if base models disagree significantly.

        High model spread indicates epistemic uncertainty — the models
        are seeing different signals, which reduces ensemble reliability.
        """
        result = {'status': 'OK', 'details': {}}

        prob_cols = [c for c in df.columns if c.startswith('Prob_') and c != 'Prob_Ensemble']
        if len(prob_cols) < 2:
            result['status'] = 'SKIP'
            return result

        latest = df[prob_cols].iloc[-1].astype(float)
        spread = float(latest.max() - latest.min())
        std = float(latest.std())

        result['details'] = {
            'current_spread': round(spread, 4),
            'current_std': round(std, 4),
            'model_predictions': {col.replace('Prob_', ''): round(float(latest[col]), 4)
                                   for col in prob_cols},
        }

        # Also check Model_Spread column if available
        if 'Model_Spread' in df.columns:
            spread_series = df['Model_Spread'].astype(float)
            result['details']['mean_historical_spread'] = round(float(spread_series.mean()), 4)
            result['details']['p95_historical_spread'] = round(
                float(spread_series.quantile(0.95)), 4
            )

        if spread > spread_threshold:
            result['status'] = 'WARNING'
            self.alerts.append({
                'level': 'WARNING',
                'check': 'model_disagreement',
                'message': (f'High model disagreement: spread={spread:.1%}. '
                            f'Models: {", ".join(f"{c.replace("Prob_", "")}={float(latest[c]):.1%}" for c in prob_cols)}')
            })

        return result

    def _check_feature_drift(self, indicators_df: pd.DataFrame,
                              feature_cols: list,
                              lookback_months: int = 36,
                              psi_threshold: float = 0.20) -> dict:
        """
        Check for feature distribution drift using Population Stability Index (PSI).

        PSI < 0.10: No significant shift
        PSI 0.10-0.25: Moderate shift (investigate)
        PSI > 0.25: Significant shift (retrain recommended)

        We compare the most recent 12 months against the prior 24 months.
        """
        result = {'status': 'OK', 'details': {}}

        available_features = [f for f in feature_cols if f in indicators_df.columns]
        if not available_features or len(indicators_df) < lookback_months:
            result['status'] = 'SKIP'
            result['details']['reason'] = 'Insufficient data for drift check'
            return result

        # Split: recent 12 months vs prior period
        recent = indicators_df[available_features].iloc[-12:]
        reference = indicators_df[available_features].iloc[-lookback_months:-12]

        psi_scores = {}
        drifted_features = []

        for col in available_features:
            ref_vals = reference[col].dropna().values
            rec_vals = recent[col].dropna().values

            if len(ref_vals) < 10 or len(rec_vals) < 5:
                continue

            psi = self._compute_psi(ref_vals, rec_vals)
            psi_scores[col] = round(psi, 4)

            if psi > psi_threshold:
                drifted_features.append((col, psi))

        result['details'] = {
            'features_checked': len(psi_scores),
            'features_drifted': len(drifted_features),
            'mean_psi': round(np.mean(list(psi_scores.values())), 4) if psi_scores else 0,
            'max_psi': round(max(psi_scores.values()), 4) if psi_scores else 0,
        }

        if drifted_features:
            # Sort by PSI descending, show top 5
            drifted_features.sort(key=lambda x: x[1], reverse=True)
            top_drifted = drifted_features[:5]
            result['details']['top_drifted'] = {f: round(p, 3) for f, p in top_drifted}
            result['status'] = 'WARNING'
            self.alerts.append({
                'level': 'WARNING',
                'check': 'feature_drift',
                'message': (f'{len(drifted_features)} features show significant drift (PSI>{psi_threshold}). '
                            f'Top: {", ".join(f"{f}={p:.2f}" for f, p in top_drifted[:3])}')
            })

        return result

    @staticmethod
    def _compute_psi(reference: np.ndarray, recent: np.ndarray, bins: int = 10) -> float:
        """
        Compute Population Stability Index between two distributions.

        PSI = sum((recent_pct - reference_pct) * ln(recent_pct / reference_pct))
        """
        # Use reference distribution to define bins
        breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints)  # Remove duplicate edges

        if len(breakpoints) < 3:
            return 0.0

        ref_counts = np.histogram(reference, bins=breakpoints)[0]
        rec_counts = np.histogram(recent, bins=breakpoints)[0]

        # Convert to proportions (with smoothing to avoid division by zero)
        eps = 1e-6
        ref_pct = (ref_counts + eps) / (ref_counts.sum() + eps * len(ref_counts))
        rec_pct = (rec_counts + eps) / (rec_counts.sum() + eps * len(rec_counts))

        psi = np.sum((rec_pct - ref_pct) * np.log(rec_pct / ref_pct))
        return float(psi)

    def _check_calibration_drift(self, df: pd.DataFrame,
                                  window: int = 24) -> dict:
        """
        Check if predictions are well-calibrated over a rolling window.

        Compares mean predicted probability against observed recession rate
        over the last 24 months.
        """
        result = {'status': 'OK', 'details': {}}

        if len(df) < window:
            result['status'] = 'SKIP'
            return result

        recent = df.iloc[-window:]
        predicted_mean = float(recent['Prob_Ensemble'].astype(float).mean())
        observed_rate = float(recent['Actual_Recession'].astype(float).mean())

        cal_gap = abs(predicted_mean - observed_rate)

        result['details'] = {
            'window_months': window,
            'mean_predicted': round(predicted_mean, 4),
            'observed_rate': round(observed_rate, 4),
            'calibration_gap': round(cal_gap, 4),
        }

        # A gap > 15pp over 24 months is concerning
        if cal_gap > 0.15:
            result['status'] = 'WARNING'
            self.alerts.append({
                'level': 'INFO',
                'check': 'calibration',
                'message': (f'Calibration gap: predicted mean={predicted_mean:.1%}, '
                            f'observed rate={observed_rate:.1%} over last {window} months')
            })

        return result

    def _check_data_freshness(self, df: pd.DataFrame,
                               stale_days: int = 45) -> dict:
        """Check if the latest prediction date is reasonably current."""
        result = {'status': 'OK', 'details': {}}

        if df.index.dtype == 'object':
            try:
                latest_date = pd.to_datetime(df.index[-1])
            except Exception:
                result['status'] = 'SKIP'
                return result
        else:
            latest_date = df.index[-1]

        days_old = (pd.Timestamp.now() - pd.Timestamp(latest_date)).days
        result['details'] = {
            'latest_date': str(latest_date.date()) if hasattr(latest_date, 'date') else str(latest_date),
            'days_old': days_old,
        }

        if days_old > stale_days:
            result['status'] = 'WARNING'
            self.alerts.append({
                'level': 'WARNING',
                'check': 'data_freshness',
                'message': f'Data is {days_old} days old (threshold: {stale_days})'
            })

        return result

    def save_report(self, report: dict, filepath: Path = None):
        """Save monitoring report to JSON."""
        if filepath is None:
            filepath = self.data_dir / "monitor_report.json"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Saved monitoring report to {filepath}")

    def load_report(self, filepath: Path = None) -> dict:
        """Load most recent monitoring report."""
        if filepath is None:
            filepath = self.data_dir / "monitor_report.json"
        if filepath.exists():
            with open(filepath) as f:
                return json.load(f)
        return {}
