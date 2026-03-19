"""
Goldman Sachs Recession Prediction Engine
Data Acquisition Module - FRED API Integration
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
        """Define all FRED indicators to fetch"""
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
                    series = self.fred.get_series(series_id, 
                                                  observation_start=start_date,
                                                  observation_end=end_date)
                    all_data[f"{category}_{series_id}"] = series
                    logger.info(f"  ✓ {description}")
                except Exception as e:
                    logger.warning(f"  ✗ Failed {series_id}: {str(e)}")
        
        # Fetch recession indicator
        try:
            recession = self.fred.get_series('USREC', 
                                            observation_start=start_date,
                                            observation_end=end_date)
            all_data['RECESSION'] = recession
        except Exception as e:
            logger.error(f"Failed recession indicator: {str(e)}")
        
        df = pd.DataFrame(all_data)
        logger.info(f"Fetched: {len(df)} observations, {len(df.columns)} series")

        # Align to monthly frequency and sort index (use last observation in each month)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.resample('M').last()

        # Basic data quality summary
        missing_frac = df.isna().mean().sort_values(ascending=False)
        logger.info("Top 10 series by missing fraction:")
        for name, frac in missing_frac.head(10).items():
            logger.info(f"  {name:25s}: {frac:6.1%} missing")

        return df
    
    def engineer_features(self, df):
        """Engineer features from raw indicators"""
        logger.info("Engineering features...")
        
        df_eng = df.copy()
        indicator_cols = [col for col in df.columns if col != 'RECESSION']
        
        for col in indicator_cols:
            # Changes
            df_eng[f'{col}_MoM'] = df[col].pct_change(1)
            df_eng[f'{col}_3M'] = df[col].pct_change(3)
            df_eng[f'{col}_6M'] = df[col].pct_change(6)
            df_eng[f'{col}_YoY'] = df[col].pct_change(12)
            
            # Moving averages
            df_eng[f'{col}_MA3'] = df[col].rolling(3).mean()
            df_eng[f'{col}_MA6'] = df[col].rolling(6).mean()
            
            # Volatility
            df_eng[f'{col}_Vol6M'] = df[col].rolling(6).std()
        
        # Replace inf values from pct_change (division by zero when indicators cross zero)
        df_eng = df_eng.replace([np.inf, -np.inf], np.nan)

        logger.info(f"Feature engineering complete: {len(df_eng.columns)} features")
        return df_eng
    
    def create_forecast_target(self, df, horizon_months=6):
        """Create forward-looking recession target"""
        df_target = df.copy()
        
        recession_future = df['RECESSION'].rolling(
            window=horizon_months, min_periods=1
        ).max().shift(-horizon_months)
        
        df_target[f'RECESSION_FORWARD_{horizon_months}M'] = recession_future
        
        logger.info(f"Created {horizon_months}-month forward target")
        return df_target