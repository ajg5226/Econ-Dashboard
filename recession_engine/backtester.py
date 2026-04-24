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

Threshold-stability-gate support (C5, 2026-04-23):
    :meth:`run_pseudo_oos_backtest` emits two lead-time measurements per row:

    * ``Lead_Months`` — computed at the per-origin F1-optimized threshold,
      as before.
    * ``Lead_Months_Fixed`` — computed at the frozen baseline decision
      threshold loaded from
      ``data/models/baseline_efb307e/threshold.json`` (override via the
      ``fixed_threshold`` argument).

    Full per-origin probability trajectories are persisted in the
    ``Probability_Trajectory`` / ``Actual_Trajectory`` columns so that
    challenger runs can be re-scored at any future threshold via
    :mod:`recession_engine.threshold_stability`. Every new experiment is
    expected to verify that no in-scope recession's
    ``Lead_Months_Fixed`` regresses more than 2 months relative to the
    baseline; see :mod:`recession_engine.threshold_stability` for the
    gating helpers and ``data/reports/experiment_ledger.md`` "Notes for
    validators" for the policy.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
import json
import logging
from datetime import datetime
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen
import warnings

try:
    from .threshold_stability import load_baseline_threshold
except Exception:  # pragma: no cover — circular import guard
    load_baseline_threshold = None  # type: ignore[assignment]

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
    # B2 labor block additions
    # FIX 3 (AUDIT-1 LOW #7): JOLTS lag 2 -> 3 months. BLS JOLTS publishes
    # ~6-7 weeks after the reference month, so at or near month-end a 2-month
    # lag is mildly optimistic. 3 months reflects realistic real-time
    # availability for nowcasts and aligns the training / backtest / live
    # information sets more honestly.
    'coincident_JTSJOL': 3,      # JOLTS openings — ~6-7 week release cadence
    'coincident_JTSQUR': 3,      # JOLTS quits rate — ~6-7 week release cadence
    'coincident_CIVPART': 1,     # BLS Employment Situation
    'coincident_EMRATIO': 1,     # BLS Employment Situation
    'coincident_USGOOD': 1,      # BLS Current Employment Statistics
    'coincident_USSERV': 1,      # BLS Current Employment Statistics
    'coincident_UNEMPLOY': 1,    # BLS Employment Situation
    # B3 credit-supply block additions (SLOOS + non-financial credit)
    'financial_DRTSCILM': 1,     # SLOOS ~2 weeks after quarter end
    'financial_DRTSCIS': 1,      # SLOOS small firms — same release
    'financial_DRTSCLCC': 1,     # SLOOS consumer credit cards
    'financial_DRSDCILM': 1,     # SLOOS demand expectations (Large/Medium)
    'financial_TOTALSL': 1,      # Consumer credit outstanding (monthly, ~5 wk lag)
}


def apply_publication_lags(df_raw: pd.DataFrame,
                           lags: dict | None = None,
                           *,
                           default_lag: int = 1) -> pd.DataFrame:
    """Apply publication-lag NaN masking to raw indicator panels.

    FIX 5 (AUDIT-1 H2): training now uses this helper to align its information
    set with the vintage-replay backtest and live scoring. At time T, a feature
    with publication lag L is unknown for T-L+1 .. T and must be NaN there.
    This emulates the real-time information set available to a forecaster
    sitting at T — see AUDIT-1 finding H2 for the motivation and the perfect-
    foresight regression it prevents.

    Args:
        df_raw: DataFrame indexed by date (typically month-end). Columns are
            indicator names used to look up per-series lag in ``lags``.
        lags: Mapping from column name to publication lag in months. Columns
            not present in the mapping use ``default_lag``. Defaults to the
            module-level ``PUBLICATION_LAGS`` dict.
        default_lag: Months of lag to assume for columns missing from ``lags``.

    Returns:
        A copy of ``df_raw`` with the last ``lag`` rows of each column set to
        NaN (except the ``RECESSION`` column, which is an outcome label and
        not subject to publication-lag masking).
    """
    if lags is None:
        lags = PUBLICATION_LAGS
    df_vintage = df_raw.copy()
    for col in df_vintage.columns:
        if col == 'RECESSION':
            continue
        lag = lags.get(col, default_lag)
        if lag > 0:
            col_idx = df_vintage.columns.get_loc(col)
            df_vintage.iloc[-lag:, col_idx] = np.nan
    return df_vintage


DEFAULT_STRICT_SEARCH_ORIGINS = [
    ('1973-06', 'Oil Crisis (positive window)'),
    ('1979-08', 'Volcker I (positive window)'),
    ('1981-02', 'Volcker II (positive window)'),
    ('1990-02', 'S&L (positive window)'),
    ('2000-10', 'Dot-com (positive window)'),
    ('2007-07', 'GFC (positive window)'),
    ('2019-09', 'COVID (positive window)'),
    ('1978-06', 'Expansion check (late 1970s)'),
    ('1986-06', 'Expansion check (mid 1980s)'),
    ('1995-06', 'Expansion check (mid 1990s)'),
    ('1998-06', 'Expansion check (late 1990s)'),
    ('2004-06', 'Expansion check (mid 2000s)'),
    ('2014-06', 'Expansion check (mid 2010s)'),
    ('2017-06', 'Expansion check (late 2010s)'),
    ('2023-06', 'Expansion check (recent cycle)'),
]


DEFAULT_SEARCH_CANDIDATES = [
    {
        'id': 'baseline_50',
        'description': 'Current validated baseline',
        'max_features': 50,
        'n_cv_splits': 5,
        'model_config': {},
    },
    {
        'id': 'conservative_40',
        'description': 'Stronger regularization and leaner feature set',
        'max_features': 40,
        'n_cv_splits': 5,
        'model_config': {
            'probit': {'C': 0.08},
            'random_forest': {
                'n_estimators': 250,
                'max_depth': 6,
                'min_samples_split': 24,
                'min_samples_leaf': 12,
            },
            'xgboost': {
                'n_estimators': 300,
                'max_depth': 4,
                'learning_rate': 0.02,
                'min_child_weight': 12,
                'subsample': 0.8,
                'colsample_bytree': 0.7,
            },
        },
    },
    {
        'id': 'stability_first_50',
        'description': 'Shallower trees at the current feature budget',
        'max_features': 50,
        'n_cv_splits': 5,
        'model_config': {
            'probit': {'C': 0.08},
            'random_forest': {
                'n_estimators': 400,
                'max_depth': 6,
                'min_samples_split': 24,
                'min_samples_leaf': 12,
            },
            'xgboost': {
                'n_estimators': 300,
                'max_depth': 4,
                'learning_rate': 0.015,
                'min_child_weight': 14,
                'subsample': 0.9,
                'colsample_bytree': 0.7,
            },
        },
    },
    {
        'id': 'richer_60',
        'description': 'Moderately richer feature set and slightly more flexible trees',
        'max_features': 60,
        'n_cv_splits': 5,
        'model_config': {
            'probit': {'C': 0.12},
            'random_forest': {
                'n_estimators': 350,
                'max_depth': 8,
                'min_samples_split': 18,
                'min_samples_leaf': 8,
            },
            'xgboost': {
                'n_estimators': 500,
                'max_depth': 5,
                'learning_rate': 0.01,
                'min_child_weight': 8,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
            },
        },
    },
]


DEFAULT_ALFRED_CORE_SERIES = [
    'leading_PERMIT',
    'leading_HOUST',
    'leading_UMCSENT',
    'leading_NEWORDER',
    'leading_DGORDER',
    'leading_PRFI',
    'leading_PCDG',
    'leading_PNFI',
    'coincident_PAYEMS',
    'coincident_UNRATE',
    'coincident_INDPRO',
    'coincident_PI',
    'coincident_RSXFS',
    'coincident_CMRMTSPL',
    'lagging_CPIAUCSL',
    'lagging_ISRATIO',
    'lagging_UEMPMEAN',
    'housing_CSUSHPINSA',
]


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
        self._alfred_cache = {}

    @staticmethod
    def _to_month_end(value):
        """Normalize any date-like input to month-end Timestamp."""
        return pd.Timestamp(value).to_period('M').to_timestamp('M')

    def _instantiate_model(self, *, n_cv_splits=5, model_config=None):
        """Instantiate the configured model class with optional hyperparameters."""
        kwargs = {
            'target_horizon': self.target_horizon,
            'n_cv_splits': n_cv_splits,
        }
        if model_config is not None:
            kwargs['model_config'] = model_config
        try:
            return self.model_class(**kwargs)
        except TypeError:
            kwargs.pop('model_config', None)
            return self.model_class(**kwargs)

    @staticmethod
    def _coerce_probability(value):
        """Return the final scalar probability from list-like model output."""
        arr = np.asarray(value, dtype=float)
        return float(arr[-1]) if arr.size else np.nan

    def _apply_publication_lags(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Simulate the information set available at a forecast date.

        Kept as an instance method for backward compatibility with the vintage
        replay code path; delegates to the module-level helper which is now
        also used by the training pipeline (FIX 5).
        """
        return apply_publication_lags(df_raw, PUBLICATION_LAGS)

    def _build_realtime_feature_frame(self, df_raw: pd.DataFrame, as_of_date,
                                      use_alfred_core: bool = False,
                                      core_series=None) -> pd.DataFrame:
        """Build a real-time feature matrix as it would have looked at forecast time."""
        pred_ts = self._to_month_end(as_of_date)
        df_realtime = df_raw[df_raw.index <= pred_ts].copy()
        df_realtime = self._apply_publication_lags(df_realtime)

        if use_alfred_core:
            fred = getattr(self.acq, "fred", None)
            if fred is not None and hasattr(fred, "get_series_as_of_date"):
                for col in core_series or []:
                    if col not in df_realtime.columns:
                        continue
                    series_id = col.split('_', 1)[1] if '_' in col else col
                    vintage_series = self._fetch_series_as_of(
                        series_id,
                        pred_ts.strftime('%Y-%m-%d'),
                    )
                    if vintage_series is None or len(vintage_series) == 0:
                        continue
                    vintage_series.index = pd.to_datetime(vintage_series.index)
                    monthly = vintage_series.sort_index().resample('ME').last()
                    aligned = monthly.reindex(df_realtime.index)
                    if aligned.notna().any():
                        df_realtime[col] = aligned.values

        return self.acq.engineer_features(df_realtime)

    def _prepare_strict_origin_frames(self, df_raw: pd.DataFrame, df_target_full: pd.DataFrame,
                                      origin_date, *, min_train_months: int = 180,
                                      use_alfred_core: bool = False,
                                      core_series=None):
        """
        Create train/prediction frames using only labels knowable at forecast time.

        At forecast origin T, the label RECESSION_FORWARD_hM is only observable
        through T-h. Training rows beyond that point would leak future outcomes.
        """
        target_col = f'RECESSION_FORWARD_{self.target_horizon}M'
        origin_ts = self._to_month_end(origin_date)
        label_cutoff = self._to_month_end(origin_ts - pd.DateOffset(months=self.target_horizon))

        df_realtime_features = self._build_realtime_feature_frame(
            df_raw,
            origin_ts,
            use_alfred_core=use_alfred_core,
            core_series=core_series,
        )
        df_realtime = df_realtime_features.copy()
        df_realtime[target_col] = df_target_full[target_col].reindex(df_realtime.index)

        train_df = df_realtime[
            (df_realtime.index <= label_cutoff) & df_realtime[target_col].notna()
        ].copy()

        if len(train_df) < min_train_months:
            return None, None, {
                'Origin_Date': origin_ts.strftime('%Y-%m'),
                'Train_Label_Cutoff': label_cutoff.strftime('%Y-%m'),
                'Reason': f'Insufficient train history ({len(train_df)} rows)',
            }

        if train_df[target_col].nunique() < 2:
            return None, None, {
                'Origin_Date': origin_ts.strftime('%Y-%m'),
                'Train_Label_Cutoff': label_cutoff.strftime('%Y-%m'),
                'Reason': 'Training sample has only one class',
            }

        if origin_ts not in df_realtime.index:
            return None, None, {
                'Origin_Date': origin_ts.strftime('%Y-%m'),
                'Train_Label_Cutoff': label_cutoff.strftime('%Y-%m'),
                'Reason': 'Origin date missing from feature frame',
            }

        pred_df = df_realtime.loc[[origin_ts]].copy()
        if pred_df[target_col].isna().all():
            return None, None, {
                'Origin_Date': origin_ts.strftime('%Y-%m'),
                'Train_Label_Cutoff': label_cutoff.strftime('%Y-%m'),
                'Reason': 'Origin target unavailable for evaluation',
            }

        meta = {
            'Origin_Date': origin_ts.strftime('%Y-%m'),
            'Train_Label_Cutoff': label_cutoff.strftime('%Y-%m'),
            'Train_Rows': int(len(train_df)),
            'Train_Positive_Rows': int(train_df[target_col].sum()),
        }
        return train_df, pred_df, meta

    def _summarize_origin_results(self, origin_df: pd.DataFrame) -> dict:
        """Aggregate strict real-time origin results into comparable model metrics."""
        if origin_df is None or origin_df.empty:
            return {
                'Origins_Tested': 0,
                'Positive_Origins': 0,
                'AUC': np.nan,
                'PR_AUC': np.nan,
                'Brier': np.nan,
                'LogLoss': np.nan,
                'Precision': np.nan,
                'Recall': np.nan,
                'F1': np.nan,
                'Specificity': np.nan,
                'Signal_Rate': np.nan,
                'Mean_Threshold': np.nan,
                'Mean_Train_Rows': np.nan,
            }

        valid = origin_df.dropna(subset=['Actual_Recession', 'Prob_Ensemble']).copy()
        if valid.empty:
            return {
                'Origins_Tested': 0,
                'Positive_Origins': 0,
                'AUC': np.nan,
                'PR_AUC': np.nan,
                'Brier': np.nan,
                'LogLoss': np.nan,
                'Precision': np.nan,
                'Recall': np.nan,
                'F1': np.nan,
                'Specificity': np.nan,
                'Signal_Rate': np.nan,
                'Mean_Threshold': np.nan,
                'Mean_Train_Rows': np.nan,
            }

        y_true = valid['Actual_Recession'].astype(int).values
        y_proba = np.clip(valid['Prob_Ensemble'].astype(float).values, 1e-7, 1 - 1e-7)
        y_pred = valid['Signal'].astype(int).values
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'Origins_Tested': int(len(valid)),
            'Positive_Origins': int(y_true.sum()),
            'AUC': roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) >= 2 else np.nan,
            'PR_AUC': average_precision_score(y_true, y_proba) if y_true.sum() > 0 else np.nan,
            'Brier': brier_score_loss(y_true, y_proba),
            'LogLoss': log_loss(y_true, y_proba, labels=[0, 1]),
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'Specificity': specificity,
            'Signal_Rate': float(valid['Signal'].mean()),
            'Mean_Threshold': float(valid['Threshold'].mean()),
            'Mean_Train_Rows': float(valid['Train_Rows'].mean()),
        }

    def _rank_search_results(self, search_df: pd.DataFrame) -> pd.DataFrame:
        """Rank candidate configurations using rare-event and calibration priorities."""
        ranked = search_df.copy()
        if 'ALFRED_MAE' in ranked.columns:
            ranked['ALFRED_MAE'] = ranked['ALFRED_MAE'].fillna(np.inf)
            sort_cols = ['PR_AUC', 'Brier', 'F1', 'AUC', 'Precision', 'ALFRED_MAE']
            ascending = [False, True, False, False, False, True]
        else:
            sort_cols = ['PR_AUC', 'Brier', 'F1', 'AUC', 'Precision']
            ascending = [False, True, False, False, False]

        ranked = ranked.sort_values(sort_cols, ascending=ascending, na_position='last').reset_index(drop=True)
        ranked['Selection_Rank'] = np.arange(1, len(ranked) + 1)
        return ranked

    def run_strict_realtime_backtest(self, df_raw: pd.DataFrame, origin_dates=None,
                                     max_features: int = 50, model_config=None,
                                     n_cv_splits: int = 5, min_train_months: int = 180,
                                     use_alfred_core: bool = False, core_series=None,
                                     candidate_id: str = None) -> pd.DataFrame:
        """
        Evaluate forecast origins under a strict real-time information set.

        Each origin uses:
        1. Vintage-simulated features available as of the forecast month.
        2. Training labels only through origin minus forecast horizon.
        3. A single prediction for that origin month.
        """
        target_col = f'RECESSION_FORWARD_{self.target_horizon}M'
        df_full_features = self.acq.engineer_features(df_raw.copy())
        df_target_full = self.acq.create_forecast_target(df_full_features, self.target_horizon)
        latest_origin = df_target_full[df_target_full[target_col].notna()].index.max()

        if origin_dates is None:
            origin_dates = DEFAULT_STRICT_SEARCH_ORIGINS

        results = []
        for origin_date, label in origin_dates:
            origin_ts = self._to_month_end(origin_date)
            if latest_origin is not None and origin_ts > latest_origin:
                continue

            train_df, pred_df, meta = self._prepare_strict_origin_frames(
                df_raw,
                df_target_full,
                origin_ts,
                min_train_months=min_train_months,
                use_alfred_core=use_alfred_core,
                core_series=core_series,
            )
            if train_df is None or pred_df is None:
                results.append({
                    'Candidate_ID': candidate_id,
                    'Origin_Label': label,
                    **meta,
                    'Actual_Recession': np.nan,
                    'Prob_Ensemble': np.nan,
                    'Signal': np.nan,
                    'Threshold': np.nan,
                })
                continue

            try:
                model = self._instantiate_model(
                    n_cv_splits=n_cv_splits,
                    model_config=model_config,
                )
                model.fit(train_df, max_features=max_features)
                predictions = model.predict(pred_df)
                prob = self._coerce_probability(predictions['ensemble'])
                actual = int(pred_df[target_col].iloc[-1])
                threshold = float(model.decision_threshold)

                results.append({
                    'Candidate_ID': candidate_id,
                    'Origin_Label': label,
                    **meta,
                    'Actual_Recession': actual,
                    'Prob_Ensemble': prob,
                    'Signal': int(prob >= threshold),
                    'Threshold': threshold,
                    'Active_Models': ",".join(getattr(model, 'active_models', [])),
                    'Ensemble_Method': getattr(model, 'ensemble_method', 'unknown'),
                })
            except Exception as exc:
                results.append({
                    'Candidate_ID': candidate_id,
                    'Origin_Label': label,
                    **meta,
                    'Actual_Recession': np.nan,
                    'Prob_Ensemble': np.nan,
                    'Signal': np.nan,
                    'Threshold': np.nan,
                    'Error': str(exc),
                })

        return pd.DataFrame(results)

    def run_model_config_search(self, df_raw: pd.DataFrame, candidate_configs=None,
                                origin_dates=None, min_train_months: int = 180,
                                alfred_top_k: int = 2, core_series=None):
        """
        Rank curated model candidates using strict real-time backtests.

        Returns a dict containing candidate summary metrics, per-origin results,
        optional ALFRED audits on the top candidates, and the winning config.
        """
        candidate_configs = candidate_configs or DEFAULT_SEARCH_CANDIDATES
        search_rows = []
        origin_frames = []
        alfred_frames = []

        for candidate in candidate_configs:
            candidate_id = candidate.get('id', 'candidate')
            logger.info("Strict vintage search: evaluating %s", candidate_id)
            origin_df = self.run_strict_realtime_backtest(
                df_raw,
                origin_dates=origin_dates,
                max_features=int(candidate.get('max_features', 50)),
                model_config=candidate.get('model_config', {}),
                n_cv_splits=int(candidate.get('n_cv_splits', 5)),
                min_train_months=min_train_months,
                candidate_id=candidate_id,
            )
            summary = self._summarize_origin_results(origin_df)
            summary.update({
                'Candidate_ID': candidate_id,
                'Description': candidate.get('description', ''),
                'Max_Features': int(candidate.get('max_features', 50)),
                'CV_Splits': int(candidate.get('n_cv_splits', 5)),
            })
            search_rows.append(summary)
            origin_frames.append(origin_df)

        search_df = pd.DataFrame(search_rows)
        if search_df.empty:
            return {
                'search_results': pd.DataFrame(),
                'origin_results': pd.DataFrame(),
                'alfred_results': pd.DataFrame(),
                'best_candidate': None,
                'summary': 'No candidate results were produced.',
            }

        search_df = self._rank_search_results(search_df)
        candidate_map = {candidate['id']: candidate for candidate in candidate_configs}

        if alfred_top_k > 0:
            alfred_mae = {}
            alfred_bias = {}
            for candidate_id in search_df.head(alfred_top_k)['Candidate_ID']:
                candidate = candidate_map[candidate_id]
                alfred_df = self.run_alfred_vintage_backtest(
                    df_raw,
                    core_series=core_series,
                    model_config=candidate.get('model_config', {}),
                    max_features=int(candidate.get('max_features', 50)),
                    n_cv_splits=int(candidate.get('n_cv_splits', 5)),
                )
                if alfred_df is not None and not alfred_df.empty:
                    alfred_df = alfred_df.copy()
                    alfred_df['Candidate_ID'] = candidate_id
                    alfred_frames.append(alfred_df)
                    valid = alfred_df.dropna(subset=['Revised_Prob', 'Vintage_Prob'], how='any')
                    if not valid.empty:
                        diffs = valid['Revised_Prob'] - valid['Vintage_Prob']
                        alfred_mae[candidate_id] = float(diffs.abs().mean())
                        alfred_bias[candidate_id] = float(diffs.mean())

            if alfred_mae:
                search_df['ALFRED_MAE'] = search_df['Candidate_ID'].map(alfred_mae)
                search_df['ALFRED_Bias'] = search_df['Candidate_ID'].map(alfred_bias)
                search_df = self._rank_search_results(search_df)

        best_candidate_id = search_df.iloc[0]['Candidate_ID']
        best_candidate = candidate_map.get(best_candidate_id, {}).copy()
        best_candidate['selection_metrics'] = search_df.iloc[0].to_dict()

        return {
            'search_results': search_df,
            'origin_results': pd.concat(origin_frames, ignore_index=True) if origin_frames else pd.DataFrame(),
            'alfred_results': pd.concat(alfred_frames, ignore_index=True) if alfred_frames else pd.DataFrame(),
            'best_candidate': best_candidate,
            'summary': self.summarize_search_results(search_df),
        }

    def summarize_search_results(self, search_df: pd.DataFrame) -> str:
        """Generate a concise human-readable summary of candidate search results."""
        if search_df is None or search_df.empty:
            return "No strict vintage search results available."

        top = search_df.iloc[0]
        lines = [
            f"Top candidate: {top['Candidate_ID']}",
            f"Origins tested: {int(top.get('Origins_Tested', 0))}",
            (
                "Primary metrics | "
                f"PR-AUC: {top.get('PR_AUC', np.nan):.3f} | "
                f"Brier: {top.get('Brier', np.nan):.4f} | "
                f"F1: {top.get('F1', np.nan):.3f} | "
                f"AUC: {top.get('AUC', np.nan):.3f}"
            ),
        ]
        if 'ALFRED_MAE' in search_df.columns and np.isfinite(top.get('ALFRED_MAE', np.nan)):
            lines.append(f"ALFRED mean abs gap: {top.get('ALFRED_MAE', np.nan):.2%}")

        if len(search_df) > 1:
            lines.append("")
            lines.append("Top candidates:")
            for _, row in search_df.head(3).iterrows():
                lines.append(
                    f"- {row['Candidate_ID']}: PR-AUC {row.get('PR_AUC', np.nan):.3f}, "
                    f"Brier {row.get('Brier', np.nan):.4f}, F1 {row.get('F1', np.nan):.3f}"
                )
        return "\n".join(lines)

    def run_pseudo_oos_backtest(self, df_with_target, cutoff_dates=None,
                                model_config=None, max_features=50, n_cv_splits=5,
                                fixed_threshold=None, store_trajectory=True):
        """
        Pseudo out-of-sample backtest: train through each cutoff, predict forward.

        For each cutoff date, trains the full model pipeline on data up to that
        date and evaluates on the subsequent period.

        Args:
            df_with_target: Full DataFrame with engineered features and target
            cutoff_dates: List of cutoff date strings. If None, uses pre-recession dates.
            fixed_threshold: Reference threshold for the stability gate. Defaults to
                the value in ``data/models/baseline_efb307e/threshold.json``. Set to
                ``None`` to skip the ``Lead_Months_Fixed`` computation.
            store_trajectory: If True, preserve the full per-origin probability
                trajectory in the ``Probability_Trajectory`` column (and matched
                ``Actual_Trajectory``) so that downstream callers can re-score at
                any future threshold via
                :func:`recession_engine.threshold_stability
                .rescore_backtest_at_fixed_threshold`.

        Returns:
            DataFrame with backtest results per recession, including the
            ``Lead_Months_Fixed`` column whenever ``fixed_threshold`` is set.
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

        # Resolve fixed threshold reference; log once per call.
        if fixed_threshold is None and load_baseline_threshold is not None:
            try:
                fixed_threshold = load_baseline_threshold()
                logger.debug("Fixed stability threshold loaded: %s", fixed_threshold)
            except FileNotFoundError:
                logger.info(
                    "Baseline threshold reference not found — skipping "
                    "Lead_Months_Fixed column"
                )
                fixed_threshold = None
            except Exception as exc:
                logger.warning("Could not load baseline threshold: %s", exc)
                fixed_threshold = None

        results = []

        for train_end, test_end, label in cutoff_dates:
            logger.info(f"\n{'='*70}")
            logger.info(f"BACKTEST: {label}")
            logger.info(f"  Train: start–{train_end}, Test: {train_end}–{test_end}")
            logger.info(f"{'='*70}")

            try:
                model = self._instantiate_model(
                    n_cv_splits=n_cv_splits,
                    model_config=model_config,
                )
                train_df, test_df = model.prepare_data(
                    df_with_target, train_end_date=train_end
                )

                # Limit test to the relevant period
                test_df = test_df[test_df.index <= test_end]

                if len(test_df) == 0 or test_df[target_col].notna().sum() == 0:
                    logger.warning(f"  Skipping {label}: no valid test data")
                    continue

                model.fit(train_df, max_features=max_features)
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
                first_recession = recession_months[0] if len(recession_months) > 0 else None
                y_ensemble_arr = np.asarray(y_ensemble)
                if first_recession is not None:
                    # Find first month probability exceeded threshold
                    above_thresh = test_df.index[y_ensemble_arr >= model.decision_threshold]
                    if len(above_thresh) > 0:
                        first_signal = above_thresh[0]
                        lead_months = (first_recession - first_signal).days / 30.44
                    else:
                        lead_months = np.nan
                else:
                    lead_months = np.nan

                # Fixed-threshold lead time (threshold-stability gate; C5).
                lead_months_fixed = np.nan
                crossed_fixed = np.nan
                if fixed_threshold is not None:
                    crossed_fixed = bool(np.any(y_ensemble_arr >= fixed_threshold))
                    if first_recession is not None:
                        above_fixed = test_df.index[y_ensemble_arr >= fixed_threshold]
                        if len(above_fixed) > 0:
                            lead_months_fixed = (
                                first_recession - above_fixed[0]
                            ).days / 30.44

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
                    'Fixed_Threshold': fixed_threshold,
                    'Crossed_Threshold_Fixed': crossed_fixed,
                    'Lead_Months_Fixed': lead_months_fixed,
                    'Ensemble_Weights': model.ensemble_weights.copy(),
                }

                # Per-model peak probabilities
                for mname in ['probit', 'random_forest', 'xgboost', 'markov_switching']:
                    if mname in predictions:
                        result[f'Peak_{mname}'] = np.max(predictions[mname])

                # Store per-origin trajectories so future analyses can re-score
                # at any threshold without rerunning the model.
                if store_trajectory:
                    traj = list(zip(
                        [d.strftime('%Y-%m-%d') for d in test_df.index],
                        [float(v) for v in y_ensemble_arr],
                    ))
                    actuals = list(zip(
                        [d.strftime('%Y-%m-%d') for d in test_df.index],
                        [int(v) if not pd.isna(v) else None for v in y_true.values],
                    ))
                    result['Probability_Trajectory'] = traj
                    result['Actual_Trajectory'] = actuals

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
            pred_ts = self._to_month_end(pred_date)
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
        as_of_date = pd.Timestamp(as_of_date).strftime('%Y-%m-%d')
        cache_key = (series_id, as_of_date, start_date)
        if cache_key in self._alfred_cache:
            return self._alfred_cache[cache_key]

        api_key = getattr(self.acq, "fred_api_key", None)
        if not api_key:
            self._alfred_cache[cache_key] = None
            return None

        params = {
            'series_id': series_id,
            'api_key': api_key,
            'file_type': 'json',
            'realtime_start': as_of_date,
            'realtime_end': as_of_date,
            'observation_start': start_date,
            'observation_end': as_of_date,
        }
        url = "https://api.stlouisfed.org/fred/series/observations?" + urlencode(params)

        try:
            with urlopen(url) as response:
                payload = json.load(response)
        except HTTPError as exc:
            # Many market-derived series have no ALFRED vintages; treat as unavailable.
            try:
                body = exc.read().decode()
                logger.debug("ALFRED unavailable for %s at %s: %s", series_id, as_of_date, body)
            except Exception:
                logger.debug("ALFRED unavailable for %s at %s (HTTP %s)", series_id, as_of_date, exc.code)
            self._alfred_cache[cache_key] = None
            return None
        except (URLError, ValueError, json.JSONDecodeError) as exc:
            logger.debug("ALFRED fetch failed for %s at %s: %s", series_id, as_of_date, exc)
            self._alfred_cache[cache_key] = None
            return None

        observations = payload.get('observations', [])
        if not observations:
            self._alfred_cache[cache_key] = None
            return None

        values = pd.DataFrame(observations)[['date', 'value']].copy()
        values['date'] = pd.to_datetime(values['date'], errors='coerce')
        values['value'] = pd.to_numeric(values['value'], errors='coerce')
        values = values.dropna(subset=['date']).set_index('date').sort_index()
        if values.empty:
            self._alfred_cache[cache_key] = None
            return None

        series = values['value']
        self._alfred_cache[cache_key] = series
        return series

    def run_alfred_vintage_backtest(self, df_raw, key_dates=None, core_series=None,
                                    model_config=None, max_features=50, n_cv_splits=5):
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
            core_series = DEFAULT_ALFRED_CORE_SERIES

        # Verify ALFRED capability first.
        if not getattr(self.acq, "fred_api_key", None):
            return pd.DataFrame([{
                'Date': None,
                'Label': 'ALFRED unavailable',
                'Error': 'FRED/ALFRED API key is not available in this runtime'
            }])

        results = []
        for pred_date, label in key_dates:
            pred_ts = self._to_month_end(pred_date)
            logger.info("ALFRED vintage test: %s (%s)", label, pred_date)
            try:
                # Revised baseline at date.
                df_revised = df_raw[df_raw.index <= pred_ts].copy()
                df_feat_revised = self.acq.engineer_features(df_revised)
                df_target_revised = self.acq.create_forecast_target(df_feat_revised, self.target_horizon)

                train_end = (pred_ts - pd.DateOffset(years=3)).strftime('%Y-%m-%d')
                model_revised = self._instantiate_model(
                    n_cv_splits=n_cv_splits,
                    model_config=model_config,
                )
                train_r, test_r = model_revised.prepare_data(df_target_revised, train_end_date=train_end)
                model_revised.fit(train_r, max_features=max_features)
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
                model_vintage = self._instantiate_model(
                    n_cv_splits=n_cv_splits,
                    model_config=model_config,
                )
                train_v, test_v = model_vintage.prepare_data(df_target_vintage, train_end_date=train_end)
                model_vintage.fit(train_v, max_features=max_features)
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
        """Generate summary statistics from backtest results.

        FIX 4 (AUDIT-1 H1 extension): the fixed-threshold lead time is the
        primary headline metric for stability across baselines. The own-
        threshold ``Lead_Months`` stays in the summary but is explicitly
        labeled as diagnostic so it isn't confused with the comparable
        cross-baseline lead-time metric.
        """
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

                # Fixed-threshold lead time (primary metric — baseline-comparable)
                if 'Lead_Months_Fixed' in valid.columns:
                    lead_fixed = valid['Lead_Months_Fixed'].dropna()
                    if len(lead_fixed) > 0:
                        fixed_thr = None
                        if 'Fixed_Threshold' in valid.columns:
                            fixed_vals = valid['Fixed_Threshold'].dropna()
                            if len(fixed_vals) > 0:
                                fixed_thr = float(fixed_vals.iloc[0])
                        if fixed_thr is not None:
                            summary.append(
                                f"Mean lead time (fixed threshold {fixed_thr:.3f}): "
                                f"{lead_fixed.mean():.1f} months"
                            )
                        else:
                            summary.append(
                                f"Mean lead time (fixed threshold): {lead_fixed.mean():.1f} months"
                            )

                # Own-threshold lead time (secondary, diagnostic only — unstable
                # baseline-to-baseline because of threshold plateau flips).
                if 'Lead_Months' in valid.columns:
                    lead = valid['Lead_Months'].dropna()
                    if len(lead) > 0:
                        summary.append(
                            f"Mean lead time (own-threshold, diagnostic): "
                            f"{lead.mean():.1f} months"
                        )

        return "\n".join(summary)
