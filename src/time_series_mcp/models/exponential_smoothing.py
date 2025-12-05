"""
Exponential Smoothing (ETS) Model Wrapper
=========================================
"""

import pandas as pd
import numpy as np
from typing import Any, Optional, Tuple
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class ETSModel:
    """Exponential Smoothing (ETS) model wrapper."""
    
    def __init__(
        self,
        trend: Optional[str] = 'add',
        seasonal: Optional[str] = None,
        seasonal_periods: Optional[int] = None,
        damped_trend: bool = True
    ):
        """
        Initialize ETS model.
        
        Args:
            trend: Type of trend ('add', 'mul', or None)
            seasonal: Type of seasonality ('add', 'mul', or None)
            seasonal_periods: Number of periods in a season
            damped_trend: Whether to damp the trend
        """
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.damped_trend = damped_trend
        self.model = None
        self.fitted_model = None
        self.fitted = False
    
    def fit(self, series: pd.Series, **kwargs) -> 'ETSModel':
        """
        Fit the ETS model.
        
        Args:
            series: Time series to fit
            **kwargs: Additional arguments for ExponentialSmoothing
        
        Returns:
            Self
        """
        # Determine if we can use seasonal component
        use_seasonal = (
            self.seasonal is not None and 
            self.seasonal_periods is not None and 
            len(series) >= 2 * self.seasonal_periods
        )
        
        try:
            if use_seasonal:
                self.model = ExponentialSmoothing(
                    series,
                    trend=self.trend,
                    seasonal=self.seasonal,
                    seasonal_periods=self.seasonal_periods,
                    damped_trend=self.damped_trend,
                    **kwargs
                )
            else:
                self.model = ExponentialSmoothing(
                    series,
                    trend=self.trend,
                    damped_trend=self.damped_trend,
                    **kwargs
                )
            
            self.fitted_model = self.model.fit(optimized=True)
            self.fitted = True
            
        except Exception as e:
            # Fallback to simple exponential smoothing
            self.model = ExponentialSmoothing(series, trend='add')
            self.fitted_model = self.model.fit()
            self.fitted = True
        
        return self
    
    def predict(
        self,
        periods: int,
        return_conf_int: bool = True,
        alpha: float = 0.05
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Generate predictions.
        
        Args:
            periods: Number of periods to forecast
            return_conf_int: Whether to return confidence intervals
            alpha: Significance level for confidence intervals
        
        Returns:
            Tuple of (predictions, confidence_intervals)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        forecast = self.fitted_model.forecast(periods)
        
        if return_conf_int:
            # ETS doesn't provide built-in confidence intervals
            # Estimate using residual standard deviation
            residuals = self.fitted_model.resid
            std_resid = np.std(residuals)
            z_score = 1.96 if (1 - alpha) == 0.95 else 1.645
            
            # Widen intervals over forecast horizon
            interval_widths = std_resid * z_score * np.sqrt(np.arange(1, periods + 1))
            
            conf_int = np.column_stack([
                forecast.values - interval_widths,
                forecast.values + interval_widths
            ])
            
            return forecast.values, conf_int
        
        return forecast.values, None
    
    def get_params(self) -> dict[str, Any]:
        """Get model parameters."""
        if not self.fitted:
            return {}
        
        return {
            "trend": self.trend,
            "seasonal": self.seasonal,
            "seasonal_periods": self.seasonal_periods,
            "damped_trend": self.damped_trend,
            "aic": self.fitted_model.aic,
            "bic": self.fitted_model.bic,
            "sse": self.fitted_model.sse
        }
    
    def summary(self) -> str:
        """Get model summary."""
        if not self.fitted:
            return "Model not fitted"
        return str(self.fitted_model.summary())