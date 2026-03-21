"""
Recession Prediction Engine — Historical Backtesting Module

Provides:
1. Pseudo out-of-sample backtesting: Train through date X, predict forward
2. ALFRED real-time vintage backtesting: Use only data available at prediction time
3. Multi-recession evaluation: Test detection of each historical recession

Key recessions tested:
- 1973-75 (Oil crisis)
- 1980 (Volcker tightening)
- 1981-82 (Double dip)
- 1990-91 (S&L crisis)
- 2001 (Dot-com bust)
- 2007-09 (Great Financial Crisis)
- 2020 (COVID pandemic)
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss
import logging
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


# NBER recession dates (peak, trough)
NBER_RECESSIONS = [
    ('1973-11', '1975-03', 'Oil Crisis'),
    ('1980-01', '1980-07', 'Volcker I'),
    ('1981-07', '1982-11', 'Volcker II'),
    ('1990-07', '1991-03', 'S&L Crisis'),
    ('2001-03', '2001-11', 'Dot-com'),
    ('2007-12', '2009-06', 'Great Financial Crisis'),
    ('2020-02', '2020-04', 'COVID'),
]

# Typical publication lags for key indicators (months)
PUBLICATION_LAGS = {
    'coincident_PAYEMS': 1,      # Employment situation ~5 weeks after month end
    'coincident_UNRATE': 1,
    'coincident_INDPRO': 2,      # Industrial production ~6 weeks lag
    'coincident_PI': 1,          # Personal income ~4 weeks lag
    'coincident_RSXFS': 2,       # Retail sales revised
    'coincident_CMRMTSPL': 2,    # Manufacturing sales ~8 weeks
    'lagging_CPIAUCSL': 1,       # CPI ~2-3 weeks lag
    'lagging_ISRATIO': 2,
    'leading_PERMIT': 2,
    'leading_HOUST': 2,
    'leading_NEWORDER': 2,
    'leading_DGORDER': 2,
    # These are available with minimal lag:
    'leading_T10Y2Y': 0,         # Market data — real-time
    'leading_T10Y3M': 0,
    'leading_GS2': 0,            # Market data — real-time
    'leading_TB3MS': 0,          # Market data — real-time
    'monetary_DFF': 0,
    'monetary_BAA10Y': 0,
    'monetary_TEDRATE': 0,
    'financial_NFCI': 0,         # Weekly release
    'financial_ANFCI': 0,
    'financial_BAMLH0A0HYM2': 0, # ICE BofA — daily
    'financial_BAMLC0A0CM': 0,   # ICE BofA — daily
    'leading_ICSA': 0,           # Weekly claims
    'leading_UMCSENT': 0,        # Monthly survey
    'coincident_UNRATE': 1,
    # Tier 1 new series (2024-2026 literature)
    'housing_CSUSHPINSA': 2,     # Case-Shiller: ~2 month lag
    'term_structure_THREEFYTP10': 0,  # Fed model — published with minimal lag
    'coincident_IURSA': 0,       # Insured unemployment — weekly administrative data
    'leading_PRFI': 2,           # BEA quarterly, interpolated monthly — ~2 month lag
    'leading_PCDG': 1,           # BEA personal consumption — ~1 month lag
    'leading_PNFI': 2,           # BEA quarterly, interpolated monthly — ~2 month lag
}


class RecessionBacktester:
    """
    Historical backtesting framework for recession prediction models.
    """

    def __init__(self, data_acquisition, model_class, target_horizon=6):
        """
        Args:
            data_acquisition: RecessionDataAcquisition instance (for feature engineering)
            model_class: RecessionEnsembleModel class (not instance)
            target_horizon: Prediction horizon in months
        """
        self.acq = data_acquisition
        self.model_class = model_class
        self.target_horizon = target_horizon

    def run_pseudo_oos_backtest(self, df_with_target, cutoff_dates=None):
        """
        Pseudo out-of-sample backtest: train through each cutoff, predict forward.

        For each cutoff date, trains the full model pipeline on data up to that
        date and evaluates on the subsequent period.

        Args:
            df_with_target: Full DataFrame with engineered features and target
            cutoff_dates: List of cutoff date strings. If None, uses pre-recession dates.

        Returns:
            DataFrame with backtest results per recession
        """
        target_col = f'RECESSION_FORWARD_{self.target_horizon}M'

        if cutoff_dates is None:
            # Default: train through 1 year before each major recession
            cutoff_dates = [
                ('1972-11', '1976-03', 'Oil Crisis (1973-75)'),
                ('1979-01', '1981-07', 'Volcker I (1980)'),
                ('1980-07', '1983-11', 'Volcker II (1981-82)'),
                ('1989-07', '1992-03', 'S&L Crisis (1990-91)'),
                ('2000-03', '2002-11', 'Dot-com (2001)'),
                ('2006-12', '2010-06', 'GFC (2007-09)'),
                ('2019-02', '2021-04', 'COVID (2020)'),
            ]

        results = []

        for train_end, test_end, label in cutoff_dates:
            logger.info(f"\n{'='*70}")
            logger.info(f"BACKTEST: {label}")
            logger.info(f"  Train: start–{train_end}, Test: {train_end}–{test_end}")
            logger.info(f"{'='*70}")

            try:
                model = self.model_class(target_horizon=self.target_horizon)
                train_df, test_df = model.prepare_data(
                    df_with_target, train_end_date=train_end
                )

                # Limit test to the relevant period
                test_df = test_df[test_df.index <= test_end]

                if len(test_df) == 0 or test_df[target_col].notna().sum() == 0:
                    logger.warning(f"  Skipping {label}: no valid test data")
                    continue

                model.fit(train_df)
                predictions = model.predict(test_df)

                y_true = test_df[target_col]
                y_ensemble = predictions['ensemble']

                # Metrics
                auc = roc_auc_score(y_true, y_ensemble) if len(set(y_true)) >= 2 else np.nan
                brier = brier_score_loss(y_true, y_ensemble)

                # Peak probability during test period
                peak_prob = np.max(y_ensemble)
                peak_date = test_df.index[np.argmax(y_ensemble)]

                # Did it cross threshold?
                crossed = np.any(y_ensemble >= model.decision_threshold)

                # Months of lead time before recession
                recession_months = test_df.index[y_true == 1]
                if len(recession_months) > 0:
                    first_recession = recession_months[0]
                    # Find first month probability exceeded threshold
                    above_thresh = test_df.index[y_ensemble >= model.decision_threshold]
                    if len(above_thresh) > 0:
                        first_signal = above_thresh[0]
                        lead_months = (first_recession - first_signal).days / 30.44
                    else:
                        lead_months = np.nan
                else:
                    lead_months = np.nan

                result = {
                    'Recession': label,
                    'Train_End': train_end,
                    'Test_Months': len(test_df),
                    'Recession_Months': int(y_true.sum()),
                    'AUC': auc,
                    'Brier': brier,
                    'Peak_Prob': peak_prob,
                    'Peak_Date': peak_date.strftime('%Y-%m'),
                    'Crossed_Threshold': crossed,
                    'Threshold': model.decision_threshold,
                    'Lead_Months': lead_months,
                    'Ensemble_Weights': model.ensemble_weights.copy(),
                }

                # Per-model peak probabilities
                for mname in ['probit', 'random_forest', 'xgboost', 'markov_switching']:
                    if mname in predictions:
                        result[f'Peak_{mname}'] = np.max(predictions[mname])

                results.append(result)

                logger.info(f"  AUC: {auc:.3f}" if not np.isnan(auc) else "  AUC: N/A")
                logger.info(f"  Brier: {brier:.4f}")
                logger.info(f"  Peak ensemble prob: {peak_prob:.1%} ({peak_date.strftime('%Y-%m')})")
                logger.info(f"  Crossed threshold ({model.decision_threshold:.2f}): {'YES' if crossed else 'NO'}")
                if not np.isnan(lead_months):
                    logger.info(f"  Lead time: {lead_months:.0f} months before recession")

            except Exception as e:
                logger.error(f"  FAILED: {e}")
                results.append({
                    'Recession': label,
                    'Train_End': train_end,
                    'Error': str(e),
                })

        return pd.DataFrame(results)

    def run_vintage_backtest(self, df_raw, key_dates=None):
        """
        Simulated real-time vintage backtest.

        Applies publication lags to simulate what data would have been
        available at each prediction date. This is a conservative approximation
        of true ALFRED vintage testing.

        Args:
            df_raw: Raw indicator DataFrame (before feature engineering)
            key_dates: List of (prediction_date, label) tuples to test

        Returns:
            DataFrame comparing revised-data vs vintage-simulated predictions
        """
        if key_dates is None:
            key_dates = [
                ('2007-06', 'Pre-GFC (6mo before peak)'),
                ('2007-12', 'GFC Peak Month'),
                ('2008-06', 'GFC Mid-recession'),
                ('2008-09', 'GFC Lehman Month'),
                ('2019-06', 'Pre-COVID (8mo before)'),
                ('2019-12', 'Pre-COVID (2mo before)'),
                ('2020-03', 'COVID Peak Month'),
            ]

        results = []

        for pred_date, label in key_dates:
            pred_ts = pd.Timestamp(pred_date)
            logger.info(f"\nVintage test: {label} ({pred_date})")

            try:
                # ── Revised data version (what we normally use) ──
                df_revised = df_raw[df_raw.index <= pred_ts].copy()
                df_feat_revised = self.acq.engineer_features(df_revised)
                df_target_revised = self.acq.create_forecast_target(
                    df_feat_revised, self.target_horizon
                )

                # Train on data up to 3 years before prediction date
                train_end = (pred_ts - pd.DateOffset(years=3)).strftime('%Y-%m-%d')
                model_revised = self.model_class(target_horizon=self.target_horizon)
                train_r, test_r = model_revised.prepare_data(
                    df_target_revised, train_end_date=train_end
                )
                model_revised.fit(train_r)
                preds_r = model_revised.predict(test_r)

                # Get prediction for the latest available month
                revised_prob = preds_r['ensemble'][-1] if len(preds_r['ensemble']) > 0 else np.nan

                # ── Vintage-simulated version ──
                df_vintage = df_raw[df_raw.index <= pred_ts].copy()

                # Apply publication lags: shift each indicator forward by its lag
                for col in df_vintage.columns:
                    lag = PUBLICATION_LAGS.get(col, 1)  # Default 1 month lag
                    if lag > 0 and col != 'RECESSION':
                        # Null out the most recent `lag` months (not available yet)
                        df_vintage.iloc[-lag:, df_vintage.columns.get_loc(col)] = np.nan

                df_feat_vintage = self.acq.engineer_features(df_vintage)
                df_target_vintage = self.acq.create_forecast_target(
                    df_feat_vintage, self.target_horizon
                )

                model_vintage = self.model_class(target_horizon=self.target_horizon)
                train_v, test_v = model_vintage.prepare_data(
                    df_target_vintage, train_end_date=train_end
                )
                model_vintage.fit(train_v)
                preds_v = model_vintage.predict(test_v)

                vintage_prob = preds_v['ensemble'][-1] if len(preds_v['ensemble']) > 0 else np.nan

                results.append({
                    'Date': pred_date,
                    'Label': label,
                    'Revised_Prob': revised_prob,
                    'Vintage_Prob': vintage_prob,
                    'Difference': revised_prob - vintage_prob if not (np.isnan(revised_prob) or np.isnan(vintage_prob)) else np.nan,
                    'Revised_Threshold': model_revised.decision_threshold,
                    'Vintage_Threshold': model_vintage.decision_threshold,
                })

                logger.info(f"  Revised data prob:  {revised_prob:.1%}" if not np.isnan(revised_prob) else "  Revised: N/A")
                logger.info(f"  Vintage sim prob:   {vintage_prob:.1%}" if not np.isnan(vintage_prob) else "  Vintage: N/A")

            except Exception as e:
                logger.error(f"  FAILED: {e}")
                results.append({
                    'Date': pred_date,
                    'Label': label,
                    'Error': str(e),
                })

        return pd.DataFrame(results)

    def _fetch_series_as_of(self, series_id: str, as_of_date: str, start_date: str = '1970-01-01'):
        """
        Fetch a single FRED/ALFRED series as known on a specific vintage date.
        Falls back gracefully if ALFRED methods are unavailable.
        """
        fred = getattr(self.acq, "fred", None)
        if fred is None:
            return None

        # Preferred ALFRED vintage method.
        if hasattr(fred, "get_series_as_of_date"):
            try:
                return fred.get_series_as_of_date(
                    series_id,
                    as_of_date=as_of_date,
                    observation_start=start_date,
                    observation_end=as_of_date
                )
            except Exception:
                return None
        return None

    def run_alfred_vintage_backtest(self, df_raw, key_dates=None, core_series=None):
        """
        ALFRED-backed vintage evaluation.

        For each key date:
        1) Replace a core subset of revised indicators with ALFRED as-of vintages.
        2) Re-run feature engineering + model fit/predict at that date.
        3) Compare revised-data probability vs vintage-data probability.

        Returns:
            DataFrame with per-date revised vs vintage probabilities and metadata.
        """
        if key_dates is None:
            key_dates = [
                ('2007-06', 'Pre-GFC'),
                ('2007-12', 'GFC peak month'),
                ('2019-12', 'Pre-COVID'),
                ('2020-03', 'COVID shock month'),
                ('2024-06', 'Recent cycle'),
            ]

        if core_series is None:
            core_series = [
                'leading_T10Y3M', 'leading_T10Y2Y', 'leading_ICSA', 'leading_USSLIND',
                'coincident_PAYEMS', 'coincident_UNRATE', 'coincident_INDPRO',
                'monetary_DFF', 'monetary_BAA10Y', 'financial_NFCI'
            ]

        # Verify ALFRED capability first.
        fred = getattr(self.acq, "fred", None)
        if fred is None or not hasattr(fred, "get_series_as_of_date"):
            return pd.DataFrame([{
                'Date': None,
                'Label': 'ALFRED unavailable',
                'Error': 'fredapi get_series_as_of_date is not available in this runtime'
            }])

        results = []
        for pred_date, label in key_dates:
            pred_ts = pd.Timestamp(pred_date)
            logger.info("ALFRED vintage test: %s (%s)", label, pred_date)
            try:
                # Revised baseline at date.
                df_revised = df_raw[df_raw.index <= pred_ts].copy()
                df_feat_revised = self.acq.engineer_features(df_revised)
                df_target_revised = self.acq.create_forecast_target(df_feat_revised, self.target_horizon)

                train_end = (pred_ts - pd.DateOffset(years=3)).strftime('%Y-%m-%d')
                model_revised = self.model_class(target_horizon=self.target_horizon)
                train_r, test_r = model_revised.prepare_data(df_target_revised, train_end_date=train_end)
                model_revised.fit(train_r)
                preds_r = model_revised.predict(test_r)
                revised_prob = preds_r['ensemble'][-1] if len(preds_r['ensemble']) > 0 else np.nan

                # Build vintage frame by substituting core series with ALFRED vintages.
                df_vintage = df_revised.copy()
                replaced_cols = 0
                for col in core_series:
                    if col not in df_vintage.columns:
                        continue
                    # Convert feature namespace column to FRED series_id, e.g. leading_T10Y3M -> T10Y3M
                    series_id = col.split('_', 1)[1] if '_' in col else col
                    vintage_series = self._fetch_series_as_of(series_id, pred_ts.strftime('%Y-%m-%d'))
                    if vintage_series is None or len(vintage_series) == 0:
                        continue
                    vintage_series.index = pd.to_datetime(vintage_series.index)
                    monthly = vintage_series.sort_index().resample('ME').last()
                    aligned = monthly.reindex(df_vintage.index)
                    if aligned.notna().any():
                        df_vintage[col] = aligned.values
                        replaced_cols += 1

                df_feat_vintage = self.acq.engineer_features(df_vintage)
                df_target_vintage = self.acq.create_forecast_target(df_feat_vintage, self.target_horizon)
                model_vintage = self.model_class(target_horizon=self.target_horizon)
                train_v, test_v = model_vintage.prepare_data(df_target_vintage, train_end_date=train_end)
                model_vintage.fit(train_v)
                preds_v = model_vintage.predict(test_v)
                vintage_prob = preds_v['ensemble'][-1] if len(preds_v['ensemble']) > 0 else np.nan

                results.append({
                    'Date': pred_date,
                    'Label': label,
                    'Revised_Prob': revised_prob,
                    'Vintage_Prob': vintage_prob,
                    'Difference': revised_prob - vintage_prob if not (np.isnan(revised_prob) or np.isnan(vintage_prob)) else np.nan,
                    'Columns_Replaced_With_ALFRED': replaced_cols,
                    'Revised_Threshold': model_revised.decision_threshold,
                    'Vintage_Threshold': model_vintage.decision_threshold,
                })
            except Exception as e:
                results.append({
                    'Date': pred_date,
                    'Label': label,
                    'Error': str(e),
                })
        return pd.DataFrame(results)

    def summarize_alfred_results(self, alfred_df: pd.DataFrame) -> str:
        """Summarize ALFRED vintage evaluation results for reporting/UI."""
        if alfred_df is None or alfred_df.empty:
            return "No ALFRED vintage results available."
        if 'Error' in alfred_df.columns and alfred_df['Error'].notna().all():
            return "ALFRED vintage evaluation unavailable in current runtime."

        valid = alfred_df.dropna(subset=['Revised_Prob', 'Vintage_Prob'], how='any')
        if valid.empty:
            return "ALFRED vintage evaluation ran, but no valid probability pairs were produced."

        mae = (valid['Revised_Prob'] - valid['Vintage_Prob']).abs().mean()
        bias = (valid['Revised_Prob'] - valid['Vintage_Prob']).mean()
        max_abs = (valid['Revised_Prob'] - valid['Vintage_Prob']).abs().max()

        return (
            f"ALFRED vintage checks: {len(valid)} date(s)\n"
            f"Mean absolute revised-vintage gap: {mae:.2%}\n"
            f"Mean signed gap (revised - vintage): {bias:.2%}\n"
            f"Max absolute gap: {max_abs:.2%}"
        )

    def summarize_results(self, backtest_df):
        """Generate summary statistics from backtest results."""
        summary = []

        if 'AUC' in backtest_df.columns:
            valid = backtest_df.dropna(subset=['AUC'])
            if len(valid) > 0:
                summary.append(f"Recessions tested: {len(valid)}")
                summary.append(f"Mean AUC: {valid['AUC'].mean():.3f} (range: {valid['AUC'].min():.3f}–{valid['AUC'].max():.3f})")
                summary.append(f"Mean Brier: {valid['Brier'].mean():.4f}")
                summary.append(f"Mean peak probability: {valid['Peak_Prob'].mean():.1%}")

                detected = valid[valid['Crossed_Threshold'] == True]
                summary.append(f"Recessions detected (crossed threshold): {len(detected)}/{len(valid)}")

                if 'Lead_Months' in valid.columns:
                    lead = valid['Lead_Months'].dropna()
                    if len(lead) > 0:
                        summary.append(f"Mean lead time: {lead.mean():.0f} months")

        return "\n".join(summary)
