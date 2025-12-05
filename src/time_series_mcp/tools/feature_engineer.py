"""
Feature Engineer for Time Series MCP
====================================
Creates time-based features for modeling.
"""

import pandas as pd
import numpy as np
from typing import Any, Optional


class TimeSeriesFeatureEngineer:
    """Create features from time series data."""
    
    def __init__(self):
        pass
    
    def create_features(
        self,
        data: pd.DataFrame,
        metadata: dict[str, Any],
        include_lags: bool = True,
        include_rolling: bool = True,
        include_datetime: bool = True
    ) -> pd.DataFrame:
        """
        Create time series features.
        
        Args:
            data: Time series DataFrame (with datetime index)
            metadata: Data metadata
            include_lags: Include lag features
            include_rolling: Include rolling statistics
            include_datetime: Include datetime components
        
        Returns:
            DataFrame with additional features
        """
        df = data.copy()
        target_col = metadata["target_column"]
        
        if include_datetime:
            df = self._add_datetime_features(df)
        
        if include_lags:
            df = self._add_lag_features(df, target_col)
        
        if include_rolling:
            df = self._add_rolling_features(df, target_col)
        
        return df
    
    def _add_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add datetime component features."""
        idx = df.index
        
        # Basic datetime components
        df['year'] = idx.year
        df['month'] = idx.month
        df['day'] = idx.day
        df['dayofweek'] = idx.dayofweek
        df['dayofyear'] = idx.dayofyear
        df['weekofyear'] = idx.isocalendar().week.values
        df['quarter'] = idx.quarter
        
        # Cyclical encoding for periodic features
        df['month_sin'] = np.sin(2 * np.pi * idx.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * idx.month / 12)
        df['day_sin'] = np.sin(2 * np.pi * idx.day / 31)
        df['day_cos'] = np.cos(2 * np.pi * idx.day / 31)
        df['dayofweek_sin'] = np.sin(2 * np.pi * idx.dayofweek / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * idx.dayofweek / 7)
        
        # Binary features
        df['is_weekend'] = (idx.dayofweek >= 5).astype(int)
        df['is_month_start'] = idx.is_month_start.astype(int)
        df['is_month_end'] = idx.is_month_end.astype(int)
        df['is_quarter_start'] = idx.is_quarter_start.astype(int)
        df['is_quarter_end'] = idx.is_quarter_end.astype(int)
        
        return df
    
    def _add_lag_features(
        self,
        df: pd.DataFrame,
        target_col: str,
        lags: Optional[list[int]] = None
    ) -> pd.DataFrame:
        """Add lag features."""
        if lags is None:
            # Default lags based on common patterns
            lags = [1, 2, 3, 7, 14, 21, 28]
        
        for lag in lags:
            if lag < len(df):
                df[f'lag_{lag}'] = df[target_col].shift(lag)
        
        return df
    
    def _add_rolling_features(
        self,
        df: pd.DataFrame,
        target_col: str,
        windows: Optional[list[int]] = None
    ) -> pd.DataFrame:
        """Add rolling statistics features."""
        if windows is None:
            windows = [7, 14, 30]
        
        for window in windows:
            if window < len(df):
                # Rolling mean
                df[f'rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
                
                # Rolling std
                df[f'rolling_std_{window}'] = df[target_col].rolling(window=window).std()
                
                # Rolling min/max
                df[f'rolling_min_{window}'] = df[target_col].rolling(window=window).min()
                df[f'rolling_max_{window}'] = df[target_col].rolling(window=window).max()
                
                # Expanding mean (cumulative)
                if window == windows[0]:
                    df['expanding_mean'] = df[target_col].expanding().mean()
        
        return df
    
    def create_holiday_features(
        self,
        df: pd.DataFrame,
        country: str = 'US'
    ) -> pd.DataFrame:
        """Add holiday indicators."""
        try:
            import holidays
            
            country_holidays = holidays.country_holidays(country)
            
            df['is_holiday'] = df.index.map(lambda x: 1 if x in country_holidays else 0)
            
            # Days until next holiday / since last holiday
            holiday_dates = pd.to_datetime([d for d in country_holidays.keys() 
                                           if df.index.min() <= pd.Timestamp(d) <= df.index.max()])
            
            if len(holiday_dates) > 0:
                def days_to_nearest_holiday(date):
                    diffs = abs((holiday_dates - date).days)
                    return diffs.min() if len(diffs) > 0 else 999
                
                df['days_to_holiday'] = df.index.map(days_to_nearest_holiday)
            
        except ImportError:
            # holidays package not installed
            df['is_holiday'] = 0
        
        return df