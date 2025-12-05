"""
ARIMA Model Wrapper
===================
"""

import pandas as pd
import numpy as np
from typing import Any, Optional, Tuple
import pmdarima as pm


class ARIMAModel:
    """ARIMA/SARIMA model wrapper."""
    
    def __init__(self, seasonal: bool = False, m: int = 1):
        """
        Initialize ARIMA model.
        
        Args:
            seasonal: Whether to use seasonal ARIMA
            m: Seasonal period
        """
        self.seasonal = seasonal
        self.m = m
        self.model = None
        self.fitted = False
    
    def fit(self, series: pd.Series, **kwargs) -> 'ARIMAModel':
        """
        Fit the ARIMA model.
        
        Args:
            series: Time series to fit
            **kwargs: Additional arguments for auto_arima
        
        Returns:
            Self
        """
        self.model = pm.auto_arima(
            series,
            seasonal=self.seasonal,
            m=self.m if self.seasonal else 1,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            max_p=kwargs.get('max_p', 5),
            max_q=kwargs.get('max_q', 5),
            max_d=kwargs.get('max_d', 2),
            max_P=kwargs.get('max_P', 2) if self.seasonal else 0,
            max_Q=kwargs.get('max_Q', 2) if self.seasonal else 0,
            max_D=kwargs.get('max_D', 1) if self.seasonal else 0,
            trace=kwargs.get('trace', False)
        )
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
        
        if return_conf_int:
            forecast, conf_int = self.model.predict(
                n_periods=periods,
                return_conf_int=True,
                alpha=alpha
            )
            return forecast, conf_int
        else:
            forecast = self.model.predict(n_periods=periods)
            return forecast, None
    
    def get_params(self) -> dict[str, Any]:
        """Get model parameters."""
        if not self.fitted:
            return {}
        
        return {
            "order": self.model.order,
            "seasonal_order": self.model.seasonal_order if self.seasonal else None,
            "aic": self.model.aic(),
            "bic": self.model.bic()
        }
    
    def summary(self) -> str:
        """Get model summary."""
        if not self.fitted:
            return "Model not fitted"
        return str(self.model.summary())