"""
Prophet Model Wrapper
=====================
"""

import pandas as pd
import numpy as np
from typing import Any, Optional, Tuple

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


class ProphetModel:
    """Facebook Prophet model wrapper."""
    
    def __init__(
        self,
        yearly_seasonality: str = 'auto',
        weekly_seasonality: str = 'auto',
        daily_seasonality: str = 'auto',
        interval_width: float = 0.95
    ):
        """
        Initialize Prophet model.
        
        Args:
            yearly_seasonality: Yearly seasonality setting
            weekly_seasonality: Weekly seasonality setting
            daily_seasonality: Daily seasonality setting
            interval_width: Width of uncertainty intervals
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is not installed. Install with: pip install prophet")
        
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.interval_width = interval_width
        self.model = None
        self.fitted = False
    
    def fit(self, series: pd.Series, **kwargs) -> 'ProphetModel':
        """
        Fit the Prophet model.
        
        Args:
            series: Time series to fit (with datetime index)
            **kwargs: Additional arguments for Prophet
        
        Returns:
            Self
        """
        # Prepare data for Prophet
        df = pd.DataFrame({
            'ds': series.index,
            'y': series.values
        })
        
        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            interval_width=self.interval_width,
            **kwargs
        )
        
        self.model.fit(df)
        self.fitted = True
        return self
    
    def predict(
        self,
        periods: int,
        frequency: str = 'D',
        return_conf_int: bool = True
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Generate predictions.
        
        Args:
            periods: Number of periods to forecast
            frequency: Frequency of predictions
            return_conf_int: Whether to return confidence intervals
        
        Returns:
            Tuple of (predictions DataFrame, None)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods, freq=frequency)
        forecast = self.model.predict(future)
        
        # Return only the forecast portion
        forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
        
        return forecast_df, None
    
    def get_params(self) -> dict[str, Any]:
        """Get model parameters."""
        if not self.fitted:
            return {}
        
        return {
            "yearly_seasonality": self.yearly_seasonality,
            "weekly_seasonality": self.weekly_seasonality,
            "daily_seasonality": self.daily_seasonality,
            "interval_width": self.interval_width
        }
    
    def plot_components(self, forecast: pd.DataFrame):
        """Plot forecast components."""
        if not self.fitted:
            raise ValueError("Model must be fitted before plotting")
        
        return self.model.plot_components(forecast)