"""
Ensemble Model
==============
Combines multiple forecasting models.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple


class EnsembleModel:
    """Ensemble of multiple forecasting models."""
    
    def __init__(
        self,
        models: Optional[Dict[str, Any]] = None,
        weights: Optional[Dict[str, float]] = None,
        method: str = 'mean'
    ):
        """
        Initialize ensemble model.
        
        Args:
            models: Dictionary of model name -> fitted model
            weights: Dictionary of model name -> weight (for weighted average)
            method: Ensemble method ('mean', 'weighted', 'median')
        """
        self.models = models or {}
        self.weights = weights
        self.method = method
        self.forecasts = {}
    
    def add_model(self, name: str, model: Any, weight: float = 1.0):
        """Add a model to the ensemble."""
        self.models[name] = model
        if self.weights is None:
            self.weights = {}
        self.weights[name] = weight
    
    def predict(
        self,
        periods: int,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate ensemble predictions.
        
        Args:
            periods: Number of periods to forecast
            **kwargs: Arguments passed to individual model predict methods
        
        Returns:
            Tuple of (ensemble forecast, confidence intervals)
        """
        all_forecasts = []
        all_lower = []
        all_upper = []
        
        for name, model in self.models.items():
            try:
                forecast, conf_int = model.predict(periods=periods, **kwargs)
                
                # Handle different return formats
                if isinstance(forecast, pd.DataFrame):
                    values = forecast['yhat'].values if 'yhat' in forecast.columns else forecast.iloc[:, 0].values
                    if conf_int is not None:
                        lower = conf_int[:, 0] if isinstance(conf_int, np.ndarray) else forecast['yhat_lower'].values
                        upper = conf_int[:, 1] if isinstance(conf_int, np.ndarray) else forecast['yhat_upper'].values
                    else:
                        lower = forecast['yhat_lower'].values if 'yhat_lower' in forecast.columns else values
                        upper = forecast['yhat_upper'].values if 'yhat_upper' in forecast.columns else values
                else:
                    values = forecast if isinstance(forecast, np.ndarray) else np.array(forecast)
                    if conf_int is not None:
                        lower = conf_int[:, 0]
                        upper = conf_int[:, 1]
                    else:
                        lower = values
                        upper = values
                
                all_forecasts.append(values[:periods])
                all_lower.append(lower[:periods])
                all_upper.append(upper[:periods])
                
                self.forecasts[name] = {
                    'values': values[:periods],
                    'lower': lower[:periods],
                    'upper': upper[:periods]
                }
                
            except Exception as e:
                print(f"Error getting forecast from {name}: {str(e)}")
                continue
        
        if not all_forecasts:
            raise ValueError("No valid forecasts from ensemble models")
        
        # Combine forecasts
        forecasts_array = np.array(all_forecasts)
        lower_array = np.array(all_lower)
        upper_array = np.array(all_upper)
        
        if self.method == 'mean':
            ensemble_forecast = np.mean(forecasts_array, axis=0)
            ensemble_lower = np.mean(lower_array, axis=0)
            ensemble_upper = np.mean(upper_array, axis=0)
            
        elif self.method == 'weighted':
            if self.weights is None:
                self.weights = {name: 1.0 / len(self.models) for name in self.models}
            
            weight_array = np.array([self.weights.get(name, 1.0) for name in self.forecasts.keys()])
            weight_array = weight_array / weight_array.sum()  # Normalize
            
            ensemble_forecast = np.average(forecasts_array, axis=0, weights=weight_array)
            ensemble_lower = np.average(lower_array, axis=0, weights=weight_array)
            ensemble_upper = np.average(upper_array, axis=0, weights=weight_array)
            
        elif self.method == 'median':
            ensemble_forecast = np.median(forecasts_array, axis=0)
            ensemble_lower = np.median(lower_array, axis=0)
            ensemble_upper = np.median(upper_array, axis=0)
            
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")
        
        conf_int = np.column_stack([ensemble_lower, ensemble_upper])
        
        return ensemble_forecast, conf_int
    
    def get_model_contributions(self) -> Dict[str, np.ndarray]:
        """Get individual model forecasts."""
        return self.forecasts