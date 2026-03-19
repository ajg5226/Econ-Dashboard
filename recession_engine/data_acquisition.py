"""
Recession Prediction Engine
Data Acquisition Module - FRED API Integration

Implements literature-backed indicator selection and feature engineering:
- Estrella & Mishkin (1998): Term spread as recession predictor
- Wright (2006): Federal funds rate augmentation
- Gilchrist & Zakrajsek (2012): Credit spreads / excess bond premium
- Sahm (2019): Sahm Rule unemployment trigger
- Billakanti & Shin (2025, Philly Fed): At-risk transformation
- Engstrom & Sharpe (2019): Near-term forward spread dynamics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from fredapi import Fred
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecessionDataAcquisition:
    """Acquire economic indicators from FRED for recession prediction"""

    def __init__(self, fred_api_key: str):
        """Initialize with FRED API key"""
        self.fred_api_key = fred_api_key
        self.fred = Fred(api_key=fred_api_key)
        self.indicators = self._define_indicators()

    def _define_indicators(self):
        """
        Define all FRED indicators to fetch.

        Organized by economic classification (leading / coincident / lagging)
        with additions from the academic literature:
        - DFF: Federal funds rate (Wright 2006)
        - BAA10Y: Baa-Treasury spread (Gilchrist-Zakrajsek proxy)
        - TEDRATE: TED spread (interbank stress)
        - T10Y3M: Primary yield curve measure (Estrella-Mishkin benchmark)
        """
        return {
            'leading': {
                'USSLIND': 'CB Leading Index',
                'T10Y2Y': 'Treasury 10Y-2Y Spread',
                'T10Y3M': 'Treasury 10Y-3M Spread',
                'PERMIT': 'Building Permits',
                'HOUST': 'Housing Starts',
                'ICSA': 'Initial Unemployment Claims',
                'UMCSENT': 'Consumer Sentiment',
                'NEWORDER': 'New Orders Consumer Goods',
                'DGORDER': 'New Orders Durable Goods',
            },
            'coincident': {
                'PAYEMS': 'Nonfarm Payrolls',
                'UNRATE': 'Unemployment Rate',
                'INDPRO': 'Industrial Production',
                'PI': 'Personal Income',
                'RSXFS': 'Retail Sales',
                'CMRMTSPL': 'Real Manufacturing Sales',
            },
            'lagging': {
                'UEMPMEAN': 'Avg Unemployment Duration',
                'CPIAUCSL': 'Consumer Price Index',
                'ISRATIO': 'Inventory Sales Ratio',
            },
            # New indicators from the literature (Wright 2006, Gilchrist-Zakrajsek 2012)
            'monetary': {
                'DFF': 'Federal Funds Rate',
                'BAA10Y': 'Baa Corporate Bond - 10Y Treasury Spread',
                'TEDRATE': 'TED Spread (3M LIBOR - 3M T-Bill)',
            },
            # Financial conditions (Chicago Fed)
            'financial': {
                'NFCI': 'Chicago Fed National Financial Conditions Index',
                'ANFCI': 'Chicago Fed Adjusted NFCI',
            },
            # Peer/reference models (for benchmarking, not used as features)
            'reference': {
                'RECPROUSM156N': 'Chauvet-Piger Smoothed Recession Probabilities (NY Fed)',
                'JHGDPBRINDX': 'GDP-Based Recession Indicator Index (Hamilton)',
            },
            'target': {
                'USREC': 'NBER Recession Indicator',
            }
        }

    def fetch_data(self, start_date='1970-01-01', end_date=None):
        """Fetch all indicator data from FRED and report basic data quality"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        all_data = {}

        for category, indicators in self.indicators.items():
            if category == 'target':
                continue

            logger.info(f"Fetching {category} indicators...")

            for series_id, description in indicators.items():
                try:
                    series = self.fred.get_series(
                        series_id,
                        observation_start=start_date,
                        observation_end=end_date
                    )
                    # Reference models use bare series_id (not prefixed) for clarity
                    if category == 'reference':
                        all_data[f"ref_{series_id}"] = series
                    else:
                        all_data[f"{category}_{series_id}"] = series
                    logger.info(f"  ✓ {description} ({len(series)} obs)")
                except Exception as e:
                    logger.warning(f"  ✗ Failed {series_id}: {str(e)}")

        # Fetch recession indicator
        try:
            recession = self.fred.get_series(
                'USREC',
                observation_start=start_date,
                observation_end=end_date
            )
            all_data['RECESSION'] = recession
        except Exception as e:
            logger.error(f"Failed recession indicator: {str(e)}")

        df = pd.DataFrame(all_data)
        logger.info(f"Fetched: {len(df)} observations, {len(df.columns)} series")

        # Align to monthly frequency and sort index (use last observation in each month)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.resample('ME').last()

        # Basic data quality summary
        missing_frac = df.isna().mean().sort_values(ascending=False)
        logger.info("Top 10 series by missing fraction:")
        for name, frac in missing_frac.head(10).items():
            logger.info(f"  {name:30s}: {frac:6.1%} missing")

        return df

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def engineer_features(self, df):
        """
        Engineer features from raw indicators.

        Three tiers of feature engineering:
        1. Standard transforms: pct_change, rolling mean, rolling std
        2. Term spread dynamics: inversion flags, momentum, duration (Engstrom-Sharpe 2019)
        3. At-risk transformation: percentile-based binary flags (Billakanti-Shin 2025)
        4. Sahm Rule: unemployment rate momentum trigger (Sahm 2019)
        """
        logger.info("Engineering features...")

        df_eng = df.copy()
        # Exclude RECESSION and reference model columns from feature engineering
        indicator_cols = [col for col in df.columns
                          if col != 'RECESSION' and not col.startswith('ref_')]

        # ── Tier 1: Standard transforms ──────────────────────────────
        for col in indicator_cols:
            # Percentage changes at multiple horizons
            df_eng[f'{col}_MoM'] = df[col].pct_change(1)
            df_eng[f'{col}_3M'] = df[col].pct_change(3)
            df_eng[f'{col}_6M'] = df[col].pct_change(6)
            df_eng[f'{col}_YoY'] = df[col].pct_change(12)

            # Moving averages (smoothing)
            df_eng[f'{col}_MA3'] = df[col].rolling(3).mean()
            df_eng[f'{col}_MA6'] = df[col].rolling(6).mean()

            # Volatility
            df_eng[f'{col}_Vol6M'] = df[col].rolling(6).std()

        # ── Tier 2: Term spread dynamics (Engstrom-Sharpe 2019) ──────
        for spread_col in ['leading_T10Y3M', 'leading_T10Y2Y']:
            if spread_col in df.columns:
                spread = df[spread_col]

                # Binary inversion flag
                df_eng[f'{spread_col}_inverted'] = (spread < 0).astype(float)

                # Inversion depth (how negative)
                df_eng[f'{spread_col}_inv_depth'] = spread.clip(upper=0)

                # Momentum of spread (3-month change)
                df_eng[f'{spread_col}_momentum'] = spread.diff(3)

                # Duration of current inversion (cumulative months inverted)
                inverted = (spread < 0).astype(int)
                # Reset counter when inversion ends
                groups = (inverted != inverted.shift()).cumsum()
                df_eng[f'{spread_col}_inv_duration'] = inverted.groupby(groups).cumsum()

                # Spread relative to its own 2-year rolling mean
                ma24 = spread.rolling(24, min_periods=12).mean()
                df_eng[f'{spread_col}_vs_ma24'] = spread - ma24

        # ── Tier 3: Sahm Rule (Sahm 2019) ───────────────────────────
        unrate_col = 'coincident_UNRATE'
        if unrate_col in df.columns:
            unrate = df[unrate_col]
            # 3-month moving average of unemployment rate
            unrate_ma3 = unrate.rolling(3).mean()
            # Trailing 12-month low of the 3-month MA
            unrate_ma3_min12 = unrate_ma3.rolling(12, min_periods=6).min()
            # Sahm indicator: rise above the trailing low
            df_eng['SAHM_INDICATOR'] = unrate_ma3 - unrate_ma3_min12
            # Binary trigger at 0.50pp threshold
            df_eng['SAHM_TRIGGER'] = (df_eng['SAHM_INDICATOR'] >= 0.50).astype(float)

        # ── Tier 4: At-risk transformation (Billakanti-Shin 2025) ────
        # Binarize indicators into "unusually weak" states using expanding
        # percentile thresholds (10th percentile for weakness)
        at_risk_cols = self._compute_at_risk_features(df, indicator_cols)
        df_eng = pd.concat([df_eng, at_risk_cols], axis=1)

        # ── Tier 5: Credit/financial stress composites ────────────────
        baa_col = 'monetary_BAA10Y'
        ted_col = 'monetary_TEDRATE'
        nfci_col = 'financial_NFCI'

        # Helper: expanding z-score with division-by-zero protection
        def _expanding_zscore(series, min_periods=24):
            mean = series.expanding(min_periods=min_periods).mean()
            std = series.expanding(min_periods=min_periods).std()
            # Replace zero/near-zero std with NaN to avoid inf
            std = std.where(std > 1e-8, np.nan)
            return (series - mean) / std

        # Core credit stress index (Baa spread + TED)
        # Compute z-scores once and reuse
        baa_z = None
        if baa_col in df.columns:
            baa_z = _expanding_zscore(df[baa_col])

        if baa_z is not None and ted_col in df.columns:
            ted_z = _expanding_zscore(df[ted_col])
            df_eng['CREDIT_STRESS_INDEX'] = (baa_z + ted_z) / 2

        # NFCI-augmented stress index (when available — NFCI starts ~1971)
        if nfci_col in df.columns:
            nfci_z = _expanding_zscore(df[nfci_col])
            df_eng['NFCI_Z'] = nfci_z
            if baa_z is not None:
                df_eng['FINANCIAL_STRESS_COMPOSITE'] = (nfci_z + baa_z) / 2

        # ── Tier 6: Monetary policy stance (Wright 2006) ─────────────
        ffr_col = 'monetary_DFF'
        spread_col = 'leading_T10Y3M'
        if ffr_col in df.columns and spread_col in df.columns:
            # Interaction: FFR level × term spread (Wright 2006 showed this matters)
            df_eng['FFR_x_SPREAD'] = df[ffr_col] * df[spread_col]
            # FFR relative to its 3-year mean (stance proxy)
            ffr_ma36 = df[ffr_col].rolling(36, min_periods=12).mean()
            df_eng['FFR_STANCE'] = df[ffr_col] - ffr_ma36

        # ── Clean up ─────────────────────────────────────────────────
        # Replace inf values from pct_change (division by zero when indicators cross zero)
        df_eng = df_eng.replace([np.inf, -np.inf], np.nan)

        logger.info(f"Feature engineering complete: {len(df_eng.columns)} total columns")
        logger.info(f"  Raw indicators: {len(indicator_cols)}")
        logger.info(f"  Engineered features: {len(df_eng.columns) - len(df.columns)}")

        return df_eng

    def _compute_at_risk_features(self, df, indicator_cols, percentile=10):
        """
        At-risk transformation (Billakanti & Shin, Philadelphia Fed, 2025).

        For each indicator, compute an expanding percentile threshold.
        Flag months where the indicator is below (or above, for countercyclical
        indicators like UNRATE and ICSA) this threshold.

        Uses expanding windows to avoid look-ahead bias.
        """
        # Countercyclical indicators: higher = worse economy
        countercyclical = {
            'coincident_UNRATE', 'lagging_UEMPMEAN', 'leading_ICSA',
            'lagging_ISRATIO', 'monetary_BAA10Y', 'monetary_TEDRATE',
            'financial_NFCI', 'financial_ANFCI',
        }

        at_risk = pd.DataFrame(index=df.index)

        for col in indicator_cols:
            if col not in df.columns:
                continue

            series = df[col]
            if series.notna().sum() < 60:  # Need at least 5 years of data
                continue

            # Expanding percentile (no look-ahead bias)
            if col in countercyclical:
                # For countercyclical: "at risk" when above the 90th percentile
                threshold = series.expanding(min_periods=60).quantile(
                    (100 - percentile) / 100
                )
                at_risk[f'{col}_AT_RISK'] = (series >= threshold).astype(float)
            else:
                # For procyclical: "at risk" when below the 10th percentile
                threshold = series.expanding(min_periods=60).quantile(
                    percentile / 100
                )
                at_risk[f'{col}_AT_RISK'] = (series <= threshold).astype(float)

        # Diffusion index: fraction of indicators in at-risk state
        if len(at_risk.columns) > 0:
            at_risk['AT_RISK_DIFFUSION'] = at_risk.mean(axis=1)
            # Weighted diffusion gives more weight to traditionally stronger predictors
            # (yield curve, credit spreads, initial claims)
            high_signal_cols = [c for c in at_risk.columns
                                if any(x in c for x in ['T10Y3M', 'T10Y2Y', 'BAA10Y',
                                                         'ICSA', 'DFF', 'TEDRATE'])]
            if high_signal_cols:
                at_risk['AT_RISK_DIFFUSION_WEIGHTED'] = (
                    at_risk[high_signal_cols].mean(axis=1) * 0.6 +
                    at_risk.drop(columns=high_signal_cols + ['AT_RISK_DIFFUSION'], errors='ignore').mean(axis=1) * 0.4
                )

        logger.info(f"  At-risk features: {len(at_risk.columns)} columns")
        return at_risk

    def create_forecast_target(self, df, horizon_months=6):
        """Create forward-looking recession target"""
        df_target = df.copy()

        recession_future = df['RECESSION'].rolling(
            window=horizon_months, min_periods=1
        ).max().shift(-horizon_months)

        df_target[f'RECESSION_FORWARD_{horizon_months}M'] = recession_future

        logger.info(f"Created {horizon_months}-month forward target")
        return df_target
