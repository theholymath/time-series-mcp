"""
Data Cleaner for Time Series MCP
================================
Handles data cleaning, preprocessing, and analysis.
"""

import pandas as pd
import numpy as np
from typing import Any, Optional
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats


class TimeSeriesDataCleaner:
    """Clean and preprocess time series data."""
    
    def __init__(self):
        pass
    
    def analyze(self, data: pd.DataFrame, metadata: dict[str, Any]) -> dict[str, Any]:
        """
        Perform comprehensive time series analysis.
        
        Args:
            data: Time series DataFrame
            metadata: Data metadata
        
        Returns:
            Analysis results dictionary
        """
        target_col = metadata["target_column"]
        series = data[target_col].dropna()
        
        # Basic statistics
        statistics = series.describe()
        
        # Stationarity tests
        adf_result = adfuller(series, autolag='AIC')
        try:
            kpss_result = kpss(series, regression='c', nlags='auto')
        except:
            kpss_result = (np.nan, np.nan, {})
        
        # Seasonality detection
        seasonality_result = self._detect_seasonality(series, metadata["frequency"])
        
        # Trend analysis
        trend_result = self._analyze_trend(series)
        
        # Autocorrelation
        acf_values = acf(series, nlags=min(40, len(series) // 2))
        pacf_values = pacf(series, nlags=min(40, len(series) // 2))
        
        # Find significant lags
        confidence_interval = 1.96 / np.sqrt(len(series))
        significant_acf = [i for i, v in enumerate(acf_values) if abs(v) > confidence_interval]
        significant_pacf = [i for i, v in enumerate(pacf_values) if abs(v) > confidence_interval]
        
        # Model recommendations
        recommendations = self._recommend_models(
            is_stationary=adf_result[1] < 0.05,
            has_seasonality=seasonality_result["has_seasonality"],
            seasonal_period=seasonality_result["period"],
            trend_strength=trend_result["strength"],
            data_length=len(series)
        )
        
        return {
            "statistics": statistics,
            "adf_stat": adf_result[0],
            "adf_pvalue": adf_result[1],
            "kpss_stat": kpss_result[0],
            "kpss_pvalue": kpss_result[1],
            "has_seasonality": seasonality_result["has_seasonality"],
            "seasonal_period": seasonality_result["period"],
            "seasonality_strength": seasonality_result["strength"],
            "trend_direction": trend_result["direction"],
            "trend_strength": trend_result["strength"],
            "significant_acf_lags": significant_acf,
            "significant_pacf_lags": significant_pacf,
            "recommended_models": recommendations,
            "acf_values": acf_values.tolist(),
            "pacf_values": pacf_values.tolist()
        }
    
    def clean(
        self,
        data: pd.DataFrame,
        metadata: dict[str, Any],
        missing_strategy: str = "interpolate",
        outlier_strategy: str = "clip",
        outlier_threshold: float = 1.5
    ) -> dict[str, Any]:
        """
        Clean time series data.
        
        Args:
            data: Time series DataFrame
            metadata: Data metadata
            missing_strategy: Strategy for missing values
            outlier_strategy: Strategy for outliers
            outlier_threshold: IQR multiplier for outlier detection
        
        Returns:
            Cleaned data and cleaning summary
        """
        df = data.copy()
        target_col = metadata["target_column"]
        
        # Track changes
        original_missing = df[target_col].isna().sum()
        
        # Handle missing values
        df, missing_filled = self._handle_missing(df, target_col, missing_strategy)
        
        # Handle outliers
        df, outliers_handled = self._handle_outliers(
            df, target_col, outlier_strategy, outlier_threshold
        )
        
        # Remove duplicates
        duplicate_count = df.index.duplicated().sum()
        if duplicate_count > 0:
            df = df[~df.index.duplicated(keep='first')]
        
        # Check if resampling needed
        resampled = False
        # Could add resampling logic here if needed
        
        # Check readiness
        remaining_missing = df[target_col].isna().sum()
        ready = remaining_missing == 0
        blocking_issues = "" if ready else f"{remaining_missing} missing values remain"
        
        return {
            "data": df,
            "missing_filled": missing_filled,
            "outliers_handled": outliers_handled,
            "duplicates_removed": duplicate_count,
            "resampled": resampled,
            "remaining_missing": remaining_missing,
            "ready_for_modeling": ready,
            "blocking_issues": blocking_issues
        }
    
    def _detect_seasonality(self, series: pd.Series, frequency: str) -> dict[str, Any]:
        """Detect seasonality in the series."""
        # Determine period based on frequency
        period_map = {'H': 24, 'D': 7, 'W': 52, 'M': 12}
        period = period_map.get(frequency, 7)
        
        if len(series) < 2 * period:
            return {"has_seasonality": False, "period": None, "strength": 0.0}
        
        try:
            decomposition = seasonal_decompose(series, model='additive', period=period)
            
            # Calculate seasonality strength
            var_seasonal = np.var(decomposition.seasonal.dropna())
            var_residual = np.var(decomposition.resid.dropna())
            
            if var_seasonal + var_residual > 0:
                strength = var_seasonal / (var_seasonal + var_residual)
            else:
                strength = 0.0
            
            has_seasonality = strength > 0.1
            
            return {
                "has_seasonality": has_seasonality,
                "period": period if has_seasonality else None,
                "strength": float(strength)
            }
        except:
            return {"has_seasonality": False, "period": None, "strength": 0.0}
    
    def _analyze_trend(self, series: pd.Series) -> dict[str, Any]:
        """Analyze trend in the series."""
        x = np.arange(len(series))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, series.values)
        
        # Determine direction
        if p_value < 0.05:
            direction = "upward" if slope > 0 else "downward"
        else:
            direction = "no significant trend"
        
        # Trend strength (R-squared)
        strength = r_value ** 2
        
        return {
            "direction": direction,
            "strength": float(strength),
            "slope": float(slope),
            "p_value": float(p_value)
        }
    
    def _recommend_models(
        self,
        is_stationary: bool,
        has_seasonality: bool,
        seasonal_period: Optional[int],
        trend_strength: float,
        data_length: int
    ) -> list[dict[str, str]]:
        """Recommend appropriate models based on data characteristics."""
        recommendations = []
        
        # ARIMA/SARIMA recommendation
        if is_stationary or data_length > 50:
            if has_seasonality:
                recommendations.append({
                    "name": "SARIMA",
                    "reason": "Data shows seasonality; SARIMA can capture seasonal patterns"
                })
            else:
                recommendations.append({
                    "name": "ARIMA",
                    "reason": "Classic approach for stationary/near-stationary data"
                })
        
        # Prophet recommendation
        if data_length > 100:
            recommendations.append({
                "name": "Prophet",
                "reason": "Good for data with strong seasonal effects and multiple seasonalities"
            })
        
        # Exponential Smoothing
        if trend_strength > 0.1 or has_seasonality:
            recommendations.append({
                "name": "Exponential Smoothing (ETS)",
                "reason": "Effective for data with trend and/or seasonality"
            })
        
        # LSTM for longer series
        if data_length > 500:
            recommendations.append({
                "name": "LSTM",
                "reason": "Deep learning approach suitable for large datasets with complex patterns"
            })
        
        # Ensemble
        if len(recommendations) >= 2:
            recommendations.append({
                "name": "Ensemble",
                "reason": "Combine multiple models for improved accuracy and robustness"
            })
        
        return recommendations
    
    def _handle_missing(
        self,
        df: pd.DataFrame,
        target_col: str,
        strategy: str
    ) -> tuple[pd.DataFrame, int]:
        """Handle missing values."""
        missing_before = df[target_col].isna().sum()
        
        if strategy == "interpolate":
            df[target_col] = df[target_col].interpolate(method='time')
        elif strategy == "ffill":
            df[target_col] = df[target_col].ffill()
        elif strategy == "bfill":
            df[target_col] = df[target_col].bfill()
        elif strategy == "mean":
            df[target_col] = df[target_col].fillna(df[target_col].mean())
        elif strategy == "median":
            df[target_col] = df[target_col].fillna(df[target_col].median())
        elif strategy == "drop":
            df = df.dropna(subset=[target_col])
        
        # Handle any remaining edge cases
        df[target_col] = df[target_col].ffill().bfill()
        
        missing_filled = missing_before - df[target_col].isna().sum()
        return df, int(missing_filled)
    
    def _handle_outliers(
        self,
        df: pd.DataFrame,
        target_col: str,
        strategy: str,
        threshold: float
    ) -> tuple[pd.DataFrame, int]:
        """Handle outliers."""
        if strategy == "none":
            return df, 0
        
        series = df[target_col]
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outlier_mask = (series < lower_bound) | (series > upper_bound)
        outlier_count = outlier_mask.sum()
        
        if strategy == "clip":
            df[target_col] = series.clip(lower=lower_bound, upper=upper_bound)
        elif strategy == "remove":
            df = df[~outlier_mask]
        
        return df, int(outlier_count)