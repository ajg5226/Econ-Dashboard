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
- Ajello et al. (2022): Term-premium-adjusted yield spread
- Grigoli & Sandri (2024): House prices as yield-curve confirming indicator
- Scavette & O'Trakoun (2025): SOS recession indicator (insured unemployment)
- Leamer (2024): Residential investment and consumer durables as precursors
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
                'GS2': '2-Year Treasury Constant Maturity Rate',
                'TB3MS': '3-Month Treasury Bill Secondary Market Rate',
                'PERMIT': 'Building Permits',
                'HOUST': 'Housing Starts',
                'ICSA': 'Initial Unemployment Claims',
                'UMCSENT': 'Consumer Sentiment',
                'NEWORDER': 'New Orders Consumer Goods',
                'DGORDER': 'New Orders Durable Goods',
                # Leamer (2024): residential investment + consumer durables
                'PRFI': 'Private Residential Fixed Investment',
                'PCDG': 'Personal Consumption Expenditures: Durable Goods',
                # Sectoral divergence: nonresidential fixed investment
                'PNFI': 'Private Nonresidential Fixed Investment',
            },
            'coincident': {
                'PAYEMS': 'Nonfarm Payrolls',
                'UNRATE': 'Unemployment Rate',
                'INDPRO': 'Industrial Production',
                'PI': 'Personal Income',
                'RSXFS': 'Retail Sales',
                'CMRMTSPL': 'Real Manufacturing Sales',
                # Scavette & O'Trakoun (2025): insured unemployment rate
                'IURSA': 'Insured Unemployment Rate (SA)',
                # B2: Labor deterioration block (Richmond Fed SOS + Philly Fed
                # max-employment + sectoral divergence literature)
                'JTSJOL': 'JOLTS Job Openings',                   # 2000+, V/U gap
                'JTSQUR': 'JOLTS Quits Rate',                     # 2000+, quits signal
                'CIVPART': 'Civilian Labor Force Participation Rate',  # 1948+
                'EMRATIO': 'Employment-Population Ratio',          # 1948+
                'USGOOD': 'All Employees: Goods-Producing',        # 1939+, cyclical
                'USSERV': 'All Employees: Service-Providing',      # 1939+, less cyclical
                'UNEMPLOY': 'Unemployed Persons (thousands)',      # 1948+, V/U gap
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
            # Financial conditions (Chicago Fed) + credit spread proxies
            'financial': {
                'NFCI': 'Chicago Fed National Financial Conditions Index',
                'ANFCI': 'Chicago Fed Adjusted NFCI',
                'BAMLH0A0HYM2': 'ICE BofA US High Yield OAS',
                'BAMLC0A0CM': 'ICE BofA US Corporate Master OAS',
            },
            # Housing & term structure (Grigoli-Sandri 2024, Ajello et al. 2022)
            'housing': {
                'CSUSHPINSA': 'S&P/Case-Shiller US National Home Price Index (NSA)',
            },
            'term_structure': {
                'THREEFYTP10': 'Kim-Wright 10-Year Term Premium',
            },
            # GLR monitor: Growth / Liquidity / Risk Appetite composites.
            # Series here feed the standalone GLRRegimeEngine and land in
            # indicators.csv as GLR_* columns. They are isolated from the
            # recession ensemble's feature pool by scheduler/update_job.py.
            'glr': {
                'CUMFNS': 'Capacity Utilization: Manufacturing',
                'M2SL': 'M2 Money Stock',
                'WALCL': 'Fed Balance Sheet Total Assets',
                'WTREGEN': 'Treasury General Account',
                'RRPONTSYD': 'Overnight Reverse Repo',
                'WLRRAL': 'Reserves of Depository Institutions',
                'VIXCLS': 'VIX Close (30-day implied vol)',
                'VXVCLS': '3-Month VIX',
                'DTWEXBGS': 'Broad Trade-Weighted USD Index',
                'PCOPPUSDM': 'Global Price of Copper (USD/t, monthly)',
                'IQ12260': 'Export Price Index: Nonmonetary Gold (proxy for gold, monthly)',
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

        # Interpolate quarterly BEA series to monthly (linear interpolation)
        # These series (PRFI, PCDG, PNFI) are quarterly and would otherwise
        # have 2/3 NaN after monthly resampling, producing garbage pct_change.
        quarterly_series = ['leading_PRFI', 'leading_PCDG', 'leading_PNFI']
        for col in quarterly_series:
            if col in df.columns:
                non_null = df[col].notna().sum()
                if non_null > 0 and non_null < len(df) * 0.5:
                    df[col] = df[col].interpolate(method='linear')
                    logger.info(f"  Interpolated {col} from quarterly to monthly "
                                f"({non_null} → {df[col].notna().sum()} obs)")

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
        # Note: NEAR_TERM_FORWARD_SPREAD dynamics are computed separately in Tier 2b
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

        # ── Tier 2b: Near-term forward spread (Engstrom & Sharpe 2019) ──
        gs2_col = 'leading_GS2'
        tb3ms_col = 'leading_TB3MS'
        if gs2_col in df.columns and tb3ms_col in df.columns:
            ntfs = df[gs2_col] - df[tb3ms_col]
            df_eng['NEAR_TERM_FORWARD_SPREAD'] = ntfs

            # Inversion flag
            df_eng['NTFS_inverted'] = (ntfs < 0).astype(float)

            # Momentum (3-month change)
            df_eng['NTFS_momentum'] = ntfs.diff(3)

            # Inversion depth
            df_eng['NTFS_inv_depth'] = ntfs.clip(upper=0)

            # Duration of inversion
            ntfs_inv = (ntfs < 0).astype(int)
            ntfs_groups = (ntfs_inv != ntfs_inv.shift()).cumsum()
            df_eng['NTFS_inv_duration'] = ntfs_inv.groupby(ntfs_groups).cumsum()

            # Spread relative to 2-year rolling mean
            ntfs_ma24 = ntfs.rolling(24, min_periods=12).mean()
            df_eng['NTFS_vs_ma24'] = ntfs - ntfs_ma24

        # ── Tier 2c: Excess bond premium proxy (Gilchrist-Zakrajsek 2012) ──
        hy_col = 'financial_BAMLH0A0HYM2'
        ig_col = 'financial_BAMLC0A0CM'
        if hy_col in df.columns and ig_col in df.columns:
            ebp = df[hy_col] - df[ig_col]
            df_eng['EBP_PROXY'] = ebp

            # Expanding z-score of the EBP proxy
            ebp_mean = ebp.expanding(min_periods=24).mean()
            ebp_std = ebp.expanding(min_periods=24).std()
            ebp_std = ebp_std.where(ebp_std > 1e-8, np.nan)
            df_eng['EBP_PROXY_Z'] = (ebp - ebp_mean) / ebp_std

            # At-risk flag: EBP z-score above 1.5 signals elevated risk appetite deterioration
            df_eng['EBP_AT_RISK'] = (df_eng['EBP_PROXY_Z'] > 1.5).astype(float)

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

        # ── Tier 7: Term-premium-adjusted yield spread (Ajello et al. 2022) ──
        # The raw T10Y3M spread overstates recession risk when QE/QT distorts
        # term premia. Subtracting the Kim-Wright term premium isolates the
        # policy-expectations component that actually predicts recessions.
        tp_col = 'term_structure_THREEFYTP10'
        spread_col_t10 = 'leading_T10Y3M'
        if tp_col in df.columns and spread_col_t10 in df.columns:
            adj_spread = df[spread_col_t10] - df[tp_col]
            df_eng['TERM_PREMIUM_ADJ_SPREAD'] = adj_spread
            df_eng['TP_ADJ_SPREAD_inverted'] = (adj_spread < 0).astype(float)
            df_eng['TP_ADJ_SPREAD_momentum'] = adj_spread.diff(3)

            # Duration of inversion (term-premium-adjusted)
            tp_inv = (adj_spread < 0).astype(int)
            tp_groups = (tp_inv != tp_inv.shift()).cumsum()
            df_eng['TP_ADJ_SPREAD_inv_duration'] = tp_inv.groupby(tp_groups).cumsum()

            # vs 2-year rolling mean
            tp_ma24 = adj_spread.rolling(24, min_periods=12).mean()
            df_eng['TP_ADJ_SPREAD_vs_ma24'] = adj_spread - tp_ma24

            logger.info("  ✓ Term-premium-adjusted spread (Ajello et al. 2022)")

        # ── Tier 8: House prices as confirming indicator (Grigoli & Sandri 2024) ──
        # Yield curve inversion alone doesn't predict recessions. Recessions require
        # the combination of: (1) curve inversion, (2) falling house prices, AND
        # (3) rising excess bond premium. House price declines confirm recession risk.
        hp_col = 'housing_CSUSHPINSA'
        if hp_col in df.columns:
            hp = df[hp_col]
            # YoY house price growth
            df_eng['HOUSE_PRICE_YOY'] = hp.pct_change(12)
            # 3-month momentum
            df_eng['HOUSE_PRICE_MOM3'] = hp.pct_change(3)
            # Declining flag (negative YoY)
            df_eng['HOUSE_PRICE_DECLINING'] = (hp.pct_change(12) < 0).astype(float)
            # Grigoli-Sandri triple condition: curve inverted + house prices falling + EBP elevated
            if spread_col_t10 in df.columns:
                curve_inv = (df[spread_col_t10] < 0).astype(float)
                hp_declining = df_eng['HOUSE_PRICE_DECLINING']
                ebp_elevated = df_eng.get('EBP_AT_RISK', pd.Series(0, index=df.index))
                # All three must be true simultaneously
                df_eng['GRIGOLI_SANDRI_TRIPLE'] = (
                    curve_inv * hp_declining * ebp_elevated
                )
                # Softer version: 2 of 3 conditions met
                condition_sum = curve_inv + hp_declining + ebp_elevated
                df_eng['RECESSION_CONFIRM_2OF3'] = (condition_sum >= 2).astype(float)

            logger.info("  ✓ House price confirming indicator (Grigoli & Sandri 2024)")

        # ── Tier 9: SOS recession indicator (Scavette & O'Trakoun 2025) ──
        # Uses insured unemployment rate instead of survey-based UNRATE.
        # Zero false positives vs Sahm's 2 (including 2024 false trigger).
        # The threshold is 0.20pp rise in the moving average above its low.
        iur_col = 'coincident_IURSA'
        if iur_col in df.columns:
            iur = df[iur_col]
            # Monthly approximation of the SOS indicator:
            # 6-month moving average minus 12-month low
            iur_ma6 = iur.rolling(6, min_periods=3).mean()
            iur_low12 = iur_ma6.rolling(12, min_periods=6).min()
            df_eng['SOS_INDICATOR'] = iur_ma6 - iur_low12
            df_eng['SOS_TRIGGER'] = (df_eng['SOS_INDICATOR'] >= 0.20).astype(float)
            # Continuous signal strength
            df_eng['SOS_MOMENTUM'] = df_eng['SOS_INDICATOR'].diff(3)

            logger.info("  ✓ SOS recession indicator (Scavette & O'Trakoun 2025)")

        # ── Tier 10: Residential investment + consumer durables (Leamer 2024) ──
        # Among 20 variables, Leamer found these two GDP components are the
        # most reliable pre-recession signals after the yield curve.
        prfi_col = 'leading_PRFI'
        pcdg_col = 'leading_PCDG'
        if prfi_col in df.columns:
            prfi = df[prfi_col]
            df_eng['RESIDENTIAL_INV_YOY'] = prfi.pct_change(12)
            df_eng['RESIDENTIAL_INV_MOM3'] = prfi.pct_change(3)
            # At-risk: residential investment declining YoY
            df_eng['RESIDENTIAL_INV_DECLINING'] = (prfi.pct_change(12) < 0).astype(float)
            # Expanding z-score of YoY growth
            ri_yoy = prfi.pct_change(12)
            ri_mean = ri_yoy.expanding(min_periods=24).mean()
            ri_std = ri_yoy.expanding(min_periods=24).std().where(
                lambda x: x > 1e-8, np.nan
            )
            df_eng['RESIDENTIAL_INV_Z'] = (ri_yoy - ri_mean) / ri_std

            logger.info("  ✓ Residential investment features (Leamer 2024)")

        if pcdg_col in df.columns:
            pcdg = df[pcdg_col]
            df_eng['DURABLES_YOY'] = pcdg.pct_change(12)
            df_eng['DURABLES_MOM3'] = pcdg.pct_change(3)
            # At-risk: durables declining YoY
            df_eng['DURABLES_DECLINING'] = (pcdg.pct_change(12) < 0).astype(float)
            # Expanding z-score
            dur_yoy = pcdg.pct_change(12)
            dur_mean = dur_yoy.expanding(min_periods=24).mean()
            dur_std = dur_yoy.expanding(min_periods=24).std().where(
                lambda x: x > 1e-8, np.nan
            )
            df_eng['DURABLES_Z'] = (dur_yoy - dur_mean) / dur_std

            logger.info("  ✓ Consumer durables features (Leamer 2024)")

        # ── Tier 11: Sectoral divergence — capex vs employment ──────────
        # When nonresidential fixed investment (AI/datacenter capex) is booming
        # but employment is stalling, a capex bust creates acute recession risk.
        pnfi_col = 'leading_PNFI'
        payems_col = 'coincident_PAYEMS'
        if pnfi_col in df.columns and payems_col in df.columns:
            pnfi_yoy = df[pnfi_col].pct_change(12)
            payems_yoy = df[payems_col].pct_change(12)
            # Sectoral divergence: capex growth minus employment growth
            df_eng['SECTORAL_DIVERGENCE'] = pnfi_yoy - payems_yoy
            # Expanding z-score of the divergence
            div = df_eng['SECTORAL_DIVERGENCE']
            div_mean = div.expanding(min_periods=24).mean()
            div_std = div.expanding(min_periods=24).std().where(
                lambda x: x > 1e-8, np.nan
            )
            df_eng['SECTORAL_DIVERGENCE_Z'] = (div - div_mean) / div_std
            # Flag: capex booming (>5% YoY) but employment weak (<1% YoY)
            df_eng['K_SHAPE_FLAG'] = (
                (pnfi_yoy > 0.05) & (payems_yoy < 0.01)
            ).astype(float)

            # Also compute Leamer's residential vs nonresidential ratio
            if prfi_col in df.columns:
                prfi_yoy = df[prfi_col].pct_change(12)
                # When residential is falling but nonresidential is rising,
                # rate-sensitive sectors are weakening first
                df_eng['RES_VS_NONRES_SPREAD'] = prfi_yoy - pnfi_yoy

            logger.info("  ✓ Sectoral divergence features (capex vs employment)")

        # ── Tier 12: Labor deterioration block (B2) ──────────────────
        # Adds: vacancy/unemployment gap (JOLTS), quits-rate signal,
        # participation gap, employment-to-population gap, and cyclical
        # vs acyclical (goods vs services) mix.
        # Rationale — Richmond Fed SOS + Phil Fed max-employment +
        # sectoral-divergence literature shows Sahm + SOS alone miss
        # signal available in JOLTS, CIVPART, EMRATIO, and the
        # goods/services mix.
        # NOTE: JOLTS features (JTSJOL, JTSQUR) are NaN pre-Dec-2000;
        # the pipeline's expanding transforms and feature selector
        # handle NaN-dense columns gracefully.
        jolts_openings_col = 'coincident_JTSJOL'
        unemploy_col = 'coincident_UNEMPLOY'
        quits_col = 'coincident_JTSQUR'
        civpart_col = 'coincident_CIVPART'
        emratio_col = 'coincident_EMRATIO'
        usgood_col = 'coincident_USGOOD'
        usserv_col = 'coincident_USSERV'

        # (a) Vacancy-Unemployment gap (JOLTS + UNEMPLOY)
        if jolts_openings_col in df.columns and unemploy_col in df.columns:
            openings = df[jolts_openings_col]
            unemployed = df[unemploy_col]
            # UNEMPLOY is persons in thousands; JTSJOL is also in thousands.
            # Ratio gives labor-market tightness (higher = tighter).
            vu_ratio = openings / unemployed.replace(0, np.nan)
            df_eng['VU_RATIO'] = vu_ratio
            df_eng['VU_RATIO_YoY'] = vu_ratio.pct_change(12)

            vu_mean = vu_ratio.expanding(min_periods=24).mean()
            vu_std = vu_ratio.expanding(min_periods=24).std().where(
                lambda x: x > 1e-8, np.nan
            )
            df_eng['VU_RATIO_Z'] = (vu_ratio - vu_mean) / vu_std
            # At-risk: V/U ratio deteriorating >15% YoY
            df_eng['VU_DETERIORATION'] = (vu_ratio.pct_change(12) < -0.15).astype(float)

            logger.info("  ✓ V/U gap features (JOLTS + UNEMPLOY)")

        # (b) Quits rate signal (JOLTS)
        if quits_col in df.columns:
            quits = df[quits_col]
            quits_mean = quits.expanding(min_periods=24).mean()
            quits_std = quits.expanding(min_periods=24).std().where(
                lambda x: x > 1e-8, np.nan
            )
            df_eng['QUITS_Z'] = (quits - quits_mean) / quits_std
            df_eng['QUITS_DECLINE_YOY'] = quits.pct_change(12)
            df_eng['QUITS_AT_RISK'] = (df_eng['QUITS_Z'] < -1.0).astype(float)

            logger.info("  ✓ Quits rate features (JOLTS)")

        # (c) Labor force participation gap (CIVPART)
        if civpart_col in df.columns:
            civpart = df[civpart_col]
            cp_mean = civpart.expanding(min_periods=24).mean()
            cp_std = civpart.expanding(min_periods=24).std().where(
                lambda x: x > 1e-8, np.nan
            )
            df_eng['CIVPART_Z'] = (civpart - cp_mean) / cp_std
            df_eng['CIVPART_GAP_60M'] = civpart - civpart.rolling(60, min_periods=24).mean()
            df_eng['CIVPART_DROP_6M'] = civpart.diff(6)

            logger.info("  ✓ Civilian participation gap features (CIVPART)")

        # (d) Employment-to-population ratio gap (EMRATIO)
        if emratio_col in df.columns:
            emratio = df[emratio_col]
            em_mean = emratio.expanding(min_periods=24).mean()
            em_std = emratio.expanding(min_periods=24).std().where(
                lambda x: x > 1e-8, np.nan
            )
            df_eng['EMRATIO_Z'] = (emratio - em_mean) / em_std
            df_eng['EMRATIO_GAP_60M'] = emratio - emratio.rolling(60, min_periods=24).mean()
            df_eng['EMRATIO_DROP_6M'] = emratio.diff(6)

            logger.info("  ✓ Employment-to-population ratio features (EMRATIO)")

        # (e) Cyclical vs acyclical mix (USGOOD + USSERV + PAYEMS)
        if usgood_col in df.columns and payems_col in df.columns:
            goods = df[usgood_col]
            payems = df[payems_col]
            goods_share = goods / payems.replace(0, np.nan)
            df_eng['GOODS_SHARE'] = goods_share
            df_eng['GOODS_SHARE_YoY'] = goods_share.pct_change(12)

            goods_yoy = goods.pct_change(12)
            df_eng['GOODS_DECLINE_FLAG'] = (goods_yoy < 0).astype(float)

            if usserv_col in df.columns:
                serv = df[usserv_col]
                serv_yoy = serv.pct_change(12)
                df_eng['GOODS_YoY_MINUS_SERV_YoY'] = goods_yoy - serv_yoy

            logger.info("  ✓ Cyclical vs acyclical employment mix (goods/services)")

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
            'financial_BAMLH0A0HYM2', 'financial_BAMLC0A0CM',
            'coincident_IURSA',  # Insured unemployment rate (SOS indicator)
            'coincident_UNEMPLOY',  # Level of unemployed persons (B2)
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
