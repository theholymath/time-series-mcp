"""
Time Series Metrics
===================
Evaluation metrics for time series forecasting.
"""

import numpy as np
import pandas as pd
from typing import Union, Dict


class TimeSeriesMetrics:
    """Calculate forecasting evaluation metrics."""
    
    @staticmethod
    def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
        """Mean Absolute Error."""
        return float(np.mean(np.abs(actual - predicted)))
    
    @staticmethod
    def mse(actual: np.ndarray, predicted: np.ndarray) -> float:
        """Mean Squared Error."""
        return float(np.mean((actual - predicted) ** 2))
    
    @staticmethod
    def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
        """Root Mean Squared Error."""
        return float(np.sqrt(np.mean((actual - predicted) ** 2)))
    
    @staticmethod
    def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
        """Mean Absolute Percentage Error."""
        mask = actual != 0
        if mask.sum() == 0:
            return float('inf')
        return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])))
    
    @staticmethod
    def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
        """Symmetric Mean Absolute Percentage Error."""
        denominator = (np.abs(actual) + np.abs(predicted)) / 2
        mask = denominator != 0
        if mask.sum() == 0:
            return float('inf')
        return float(np.mean(np.abs(actual[mask] - predicted[mask]) / denominator[mask]))
    
    @staticmethod
    def mase(actual: np.ndarray, predicted: np.ndarray, seasonal_period: int = 1) -> float:
        """Mean Absolute Scaled Error."""
        n = len(actual)
        
        # Calculate in-sample MAE of naive forecast
        naive_errors = np.abs(actual[seasonal_period:] - actual[:-seasonal_period])
        scale = np.mean(naive_errors)
        
        if scale == 0:
            return float('inf')
        
        forecast_errors = np.abs(actual - predicted)
        return float(np.mean(forecast_errors) / scale)
    
    @staticmethod
    def r2(actual: np.ndarray, predicted: np.ndarray) -> float:
        """R-squared (coefficient of determination)."""
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        return float(1 - (ss_res / ss_tot))
    
    @classmethod
    def calculate_all(
        cls,
        actual: Union[np.ndarray, pd.Series],
        predicted: Union[np.ndarray, pd.Series],
        seasonal_period: int = 1
    ) -> Dict[str, float]:
        """
        Calculate all metrics.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            seasonal_period: Seasonal period for MASE
        
        Returns:
            Dictionary of metric names to values
        """
        # Convert to numpy arrays
        if isinstance(actual, pd.Series):
            actual = actual.values
        if isinstance(predicted, pd.Series):
            predicted = predicted.values
        
        # Ensure same length
        min_len = min(len(actual), len(predicted))
        actual = actual[:min_len]
        predicted = predicted[:min_len]
        
        return {
            "mae": cls.mae(actual, predicted),
            "mse": cls.mse(actual, predicted),
            "rmse": cls.rmse(actual, predicted),
            "mape": cls.mape(actual, predicted),
            "smape": cls.smape(actual, predicted),
            "mase": cls.mase(actual, predicted, seasonal_period),
            "r2": cls.r2(actual, predicted)
        }