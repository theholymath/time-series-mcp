"""
Model Selector for Time Series MCP
==================================
Handles automatic model selection and comparison.
"""

import pandas as pd
import numpy as np
from typing import Any
import time
from sklearn.model_selection import TimeSeriesSplit


class ModelSelector:
    """Select and compare forecasting models."""
    
    def __init__(self):
        pass
    
    def select_models(
        self,
        data: pd.DataFrame,
        metadata: dict[str, Any]
    ) -> list[str]:
        """
        Automatically select appropriate models based on data characteristics.
        
        Args:
            data: Time series DataFrame
            metadata: Data metadata
        
        Returns:
            List of recommended model names
        """
        target_col = metadata["target_column"]
        series = data[target_col].dropna()
        n = len(series)
        
        models = []
        
        # ARIMA is generally always applicable
        if n >= 30:
            models.append("arima")
        
        # SARIMA for seasonal data
        if n >= 100:
            models.append("sarima")
        
        # Prophet for data with strong seasonality and holidays
        if n >= 100:
            models.append("prophet")
        
        # Exponential Smoothing
        if n >= 20:
            models.append("ets")
        
        # Ensemble if multiple models available
        if len(models) >= 2:
            models.append("ensemble")
        
        # Default to ARIMA if nothing else
        if not models:
            models = ["arima"]
        
        return models
    
    def compare_models(
        self,
        data: pd.DataFrame,
        metadata: dict[str, Any],
        models: list[str],
        cv_folds: int = 5,
        test_size: float = 0.2
    ) -> dict[str, Any]:
        """
        Compare multiple models using cross-validation.
        
        Args:
            data: Time series DataFrame
            metadata: Data metadata
            models: List of model names to compare
            cv_folds: Number of CV folds
            test_size: Proportion for hold-out test
        
        Returns:
            Comparison results dictionary
        """
        from .forecaster import TimeSeriesForecaster
        
        target_col = metadata["target_column"]
        series = data[target_col].dropna()
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        cv_results = {}
        test_results = {}
        characteristics = {}
        
        for model_name in models:
            if model_name == "ensemble":
                continue
            
            cv_metrics = {"mae": [], "rmse": [], "mape": []}
            train_times = []
            
            # Cross-validation
            for train_idx, val_idx in tscv.split(series):
                train_data = series.iloc[train_idx]
                val_data = series.iloc[val_idx]
                
                start_time = time.time()
                
                try:
                    forecaster = TimeSeriesForecaster()
                    result = forecaster.forecast(
                        data=pd.DataFrame({target_col: train_data}),
                        metadata={**metadata, "target_column": target_col},
                        periods=len(val_data),
                        frequency=metadata["frequency"],
                        models=[model_name],
                        include_backtesting=False
                    )
                    
                    train_time = time.time() - start_time
                    train_times.append(train_time)
                    
                    if model_name in result["forecasts"]:
                        forecast = result["forecasts"][model_name]["yhat"].values
                        actual = val_data.values
                        
                        # Calculate metrics
                        mae = np.mean(np.abs(actual - forecast[:len(actual)]))
                        rmse = np.sqrt(np.mean((actual - forecast[:len(actual)])**2))
                        mask = actual != 0
                        mape = np.mean(np.abs((actual[mask] - forecast[:len(actual)][mask]) / actual[mask]))
                        
                        cv_metrics["mae"].append(mae)
                        cv_metrics["rmse"].append(rmse)
                        cv_metrics["mape"].append(mape)
                except Exception as e:
                    print(f"Error in CV for {model_name}: {str(e)}")
                    continue
            
            if cv_metrics["mae"]:
                cv_results[model_name] = {
                    "mae": np.mean(cv_metrics["mae"]),
                    "mae_std": np.std(cv_metrics["mae"]),
                    "rmse": np.mean(cv_metrics["rmse"]),
                    "rmse_std": np.std(cv_metrics["rmse"]),
                    "mape": np.mean(cv_metrics["mape"]),
                    "train_time": np.mean(train_times)
                }
            
            # Model characteristics
            characteristics[model_name] = self._get_model_characteristics(model_name)
        
        # Hold-out test
        test_idx = int(len(series) * (1 - test_size))
        train_series = series.iloc[:test_idx]
        test_series = series.iloc[test_idx:]
        
        for model_name in models:
            if model_name == "ensemble":
                continue
            
            try:
                forecaster = TimeSeriesForecaster()
                result = forecaster.forecast(
                    data=pd.DataFrame({target_col: train_series}),
                    metadata={**metadata, "target_column": target_col},
                    periods=len(test_series),
                    frequency=metadata["frequency"],
                    models=[model_name],
                    include_backtesting=False
                )
                
                if model_name in result["forecasts"]:
                    forecast = result["forecasts"][model_name]["yhat"].values
                    actual = test_series.values
                    
                    mae = np.mean(np.abs(actual - forecast[:len(actual)]))
                    rmse = np.sqrt(np.mean((actual - forecast[:len(actual)])**2))
                    mask = actual != 0
                    mape = np.mean(np.abs((actual[mask] - forecast[:len(actual)][mask]) / actual[mask]))
                    
                    test_results[model_name] = {
                        "mae": mae,
                        "rmse": rmse,
                        "mape": mape
                    }
            except Exception as e:
                print(f"Error in test for {model_name}: {str(e)}")
                continue
        
        # Determine best model
        if cv_results:
            best_model = min(cv_results.items(), key=lambda x: x[1]["mape"])[0]
            recommendation = f"Based on cross-validation, {best_model} shows the lowest MAPE and provides reliable forecasts."
        else:
            best_model = models[0] if models else "arima"
            recommendation = "Unable to complete full comparison; defaulting to first available model."
        
        return {
            "cv_results": cv_results,
            "test_results": test_results,
            "best_model": best_model,
            "recommendation": recommendation,
            "characteristics": characteristics
        }
    
    def _get_model_characteristics(self, model_name: str) -> list[str]:
        """Get characteristics for a model."""
        characteristics = {
            "arima": [
                "Best for stationary data or data that can be made stationary",
                "Captures autocorrelation patterns",
                "Fast training time",
                "Interpretable parameters"
            ],
            "sarima": [
                "Extends ARIMA with seasonal components",
                "Good for data with clear seasonal patterns",
                "More parameters to tune",
                "Can be slow for complex seasonality"
            ],
            "prophet": [
                "Handles multiple seasonalities automatically",
                "Robust to missing data and outliers",
                "Easy to incorporate holidays",
                "Good for business time series"
            ],
            "ets": [
                "Exponential smoothing approach",
                "Good for trend and seasonal data",
                "Fast and simple",
                "Works well with shorter series"
            ],
            "lstm": [
                "Deep learning approach",
                "Can capture complex nonlinear patterns",
                "Requires more data",
                "Longer training time"
            ],
            "ensemble": [
                "Combines multiple models",
                "Generally more robust",
                "Reduces model selection risk",
                "May be slower due to multiple models"
            ]
        }
        
        return characteristics.get(model_name.lower(), ["No specific characteristics available"])