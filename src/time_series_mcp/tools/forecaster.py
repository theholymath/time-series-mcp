"""
Forecaster for Time Series MCP
==============================
Handles model training and forecast generation.
"""

import pandas as pd
import numpy as np
from typing import Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Statistical models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pmdarima as pm

# Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error


class TimeSeriesForecaster:
    """Generate time series forecasts using multiple models."""
    
    def __init__(self):
        self.models = {}
        self.forecasts = {}
    
    def forecast(
        self,
        data: pd.DataFrame,
        metadata: dict[str, Any],
        periods: int,
        frequency: str,
        models: list[str],
        confidence_level: float = 0.95,
        include_backtesting: bool = True
    ) -> dict[str, Any]:
        """
        Generate forecasts using specified models.
        
        Args:
            data: Time series DataFrame
            metadata: Data metadata
            periods: Number of periods to forecast
            frequency: Forecast frequency
            models: List of model names to use
            confidence_level: Confidence level for intervals
            include_backtesting: Whether to perform backtesting
        
        Returns:
            Dictionary with forecasts, metrics, and insights
        """
        target_col = metadata["target_column"]
        series = data[target_col].dropna()
        
        # Split for backtesting if requested
        if include_backtesting:
            test_size = min(int(len(series) * 0.2), periods)
            train_series = series[:-test_size]
            test_series = series[-test_size:]
        else:
            train_series = series
            test_series = None
        
        results = {
            "models": {},
            "forecasts": {},
            "metrics": {},
            "best_model": None,
            "selection_reason": "",
            "insights": []
        }
        
        # Train each model
        for model_name in models:
            try:
                if model_name.lower() == "arima":
                    model_result = self._fit_arima(train_series, periods, frequency, confidence_level)
                elif model_name.lower() == "sarima":
                    model_result = self._fit_sarima(train_series, periods, frequency, confidence_level, metadata)
                elif model_name.lower() == "prophet":
                    model_result = self._fit_prophet(train_series, periods, frequency, confidence_level)
                elif model_name.lower() == "ets":
                    model_result = self._fit_ets(train_series, periods, frequency, confidence_level, metadata)
                elif model_name.lower() == "ensemble":
                    # Ensemble of available models
                    continue  # Handle after individual models
                else:
                    continue
                
                results["models"][model_name] = model_result["model"]
                results["forecasts"][model_name] = model_result["forecast"]
                
                # Calculate metrics if we have test data
                if test_series is not None:
                    backtest_forecast = model_result.get("backtest_forecast")
                    if backtest_forecast is not None:
                        metrics = self._calculate_metrics(test_series, backtest_forecast)
                        results["metrics"][model_name] = metrics
                
            except Exception as e:
                print(f"Error fitting {model_name}: {str(e)}")
                continue
        
        # Create ensemble if multiple models available
        if "ensemble" in [m.lower() for m in models] and len(results["forecasts"]) >= 2:
            ensemble_forecast = self._create_ensemble(results["forecasts"], periods)
            results["forecasts"]["ensemble"] = ensemble_forecast
            
            if test_series is not None:
                # Calculate ensemble backtest metrics
                ensemble_backtest = self._create_ensemble(
                    {k: v.iloc[:len(test_series)] for k, v in results["forecasts"].items() if k != "ensemble"},
                    len(test_series)
                )
                results["metrics"]["ensemble"] = self._calculate_metrics(test_series, ensemble_backtest['yhat'])
        
        # Select best model
        if results["metrics"]:
            best_model = min(results["metrics"].items(), key=lambda x: x[1]["mape"])
            results["best_model"] = best_model[0]
            results["selection_reason"] = f"Lowest MAPE ({best_model[1]['mape']:.2%}) on backtest"
        elif results["forecasts"]:
            results["best_model"] = list(results["forecasts"].keys())[0]
            results["selection_reason"] = "Only model available"
        
        # Generate insights
        results["insights"] = self._generate_insights(
            series=series,
            forecasts=results["forecasts"],
            best_model=results["best_model"],
            periods=periods
        )
        
        return results
    
    def _fit_arima(
        self,
        series: pd.Series,
        periods: int,
        frequency: str,
        confidence_level: float
    ) -> dict[str, Any]:
        """Fit Auto-ARIMA model."""
        # Use auto_arima for automatic order selection
        model = pm.auto_arima(
            series,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            max_p=5,
            max_q=5,
            max_d=2
        )
        
        # Generate forecast
        forecast_values, conf_int = model.predict(n_periods=periods, return_conf_int=True, alpha=1-confidence_level)
        
        # Create forecast DataFrame
        last_date = series.index[-1]
        forecast_index = pd.date_range(start=last_date, periods=periods + 1, freq=frequency)[1:]
        
        forecast_df = pd.DataFrame({
            'ds': forecast_index,
            'yhat': forecast_values,
            'yhat_lower': conf_int[:, 0],
            'yhat_upper': conf_int[:, 1]
        })
        forecast_df = forecast_df.set_index('ds')
        
        return {
            "model": model,
            "forecast": forecast_df,
            "backtest_forecast": model.predict_in_sample()[-int(len(series) * 0.2):]
        }
    
    def _fit_sarima(
        self,
        series: pd.Series,
        periods: int,
        frequency: str,
        confidence_level: float,
        metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Fit Seasonal ARIMA model."""
        # Determine seasonal period
        period_map = {'H': 24, 'D': 7, 'W': 52, 'M': 12}
        m = period_map.get(frequency, 7)
        
        model = pm.auto_arima(
            series,
            seasonal=True,
            m=m,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            max_p=3,
            max_q=3,
            max_P=2,
            max_Q=2,
            max_d=2,
            max_D=1
        )
        
        forecast_values, conf_int = model.predict(n_periods=periods, return_conf_int=True, alpha=1-confidence_level)
        
        last_date = series.index[-1]
        forecast_index = pd.date_range(start=last_date, periods=periods + 1, freq=frequency)[1:]
        
        forecast_df = pd.DataFrame({
            'ds': forecast_index,
            'yhat': forecast_values,
            'yhat_lower': conf_int[:, 0],
            'yhat_upper': conf_int[:, 1]
        })
        forecast_df = forecast_df.set_index('ds')
        
        return {
            "model": model,
            "forecast": forecast_df,
            "backtest_forecast": model.predict_in_sample()[-int(len(series) * 0.2):]
        }
    
    def _fit_prophet(
        self,
        series: pd.Series,
        periods: int,
        frequency: str,
        confidence_level: float
    ) -> dict[str, Any]:
        """Fit Facebook Prophet model."""
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is not installed")
        
        # Prepare data for Prophet
        df = pd.DataFrame({
            'ds': series.index,
            'y': series.values
        })
        
        model = Prophet(
            interval_width=confidence_level,
            daily_seasonality='auto',
            weekly_seasonality='auto',
            yearly_seasonality='auto'
        )
        model.fit(df)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=periods, freq=frequency)
        forecast = model.predict(future)
        
        # Extract forecast portion
        forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
        forecast_df = forecast_df.set_index('ds')
        
        # Backtest forecast
        backtest = forecast['yhat'].iloc[-periods-int(len(series)*0.2):-periods].values
        
        return {
            "model": model,
            "forecast": forecast_df,
            "backtest_forecast": backtest
        }
    
    def _fit_ets(
        self,
        series: pd.Series,
        periods: int,
        frequency: str,
        confidence_level: float,
        metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Fit Exponential Smoothing (ETS) model."""
        # Determine seasonal period
        period_map = {'H': 24, 'D': 7, 'W': 52, 'M': 12}
        seasonal_periods = period_map.get(frequency, None)
        
        # Fit model
        try:
            if seasonal_periods and len(series) >= 2 * seasonal_periods:
                model = ExponentialSmoothing(
                    series,
                    seasonal_periods=seasonal_periods,
                    trend='add',
                    seasonal='add',
                    damped_trend=True
                ).fit()
            else:
                model = ExponentialSmoothing(
                    series,
                    trend='add',
                    damped_trend=True
                ).fit()
        except:
            model = ExponentialSmoothing(series, trend='add').fit()
        
        # Generate forecast
        forecast_values = model.forecast(periods)
        
        # Estimate confidence intervals (ETS doesn't provide them directly)
        residuals = model.resid
        std_resid = np.std(residuals)
        z_score = 1.96 if confidence_level == 0.95 else 1.645
        
        last_date = series.index[-1]
        forecast_index = pd.date_range(start=last_date, periods=periods + 1, freq=frequency)[1:]
        
        # Widen intervals over forecast horizon
        interval_widths = std_resid * z_score * np.sqrt(np.arange(1, periods + 1))
        
        forecast_df = pd.DataFrame({
            'ds': forecast_index,
            'yhat': forecast_values.values,
            'yhat_lower': forecast_values.values - interval_widths,
            'yhat_upper': forecast_values.values + interval_widths
        })
        forecast_df = forecast_df.set_index('ds')
        
        # Backtest
        backtest = model.fittedvalues[-int(len(series) * 0.2):]
        
        return {
            "model": model,
            "forecast": forecast_df,
            "backtest_forecast": backtest.values
        }
    
    def _create_ensemble(
        self,
        forecasts: dict[str, pd.DataFrame],
        periods: int
    ) -> pd.DataFrame:
        """Create ensemble forecast from multiple models."""
        # Simple average ensemble
        forecast_values = []
        lower_values = []
        upper_values = []
        
        for name, fc in forecasts.items():
            if name != "ensemble":
                forecast_values.append(fc['yhat'].values[:periods])
                lower_values.append(fc['yhat_lower'].values[:periods])
                upper_values.append(fc['yhat_upper'].values[:periods])
        
        ensemble_yhat = np.mean(forecast_values, axis=0)
        ensemble_lower = np.mean(lower_values, axis=0)
        ensemble_upper = np.mean(upper_values, axis=0)
        
        # Use index from first forecast
        first_forecast = list(forecasts.values())[0]
        
        return pd.DataFrame({
            'yhat': ensemble_yhat,
            'yhat_lower': ensemble_lower,
            'yhat_upper': ensemble_upper
        }, index=first_forecast.index[:periods])
    
    def _calculate_metrics(
        self,
        actual: pd.Series,
        predicted: np.ndarray
    ) -> dict[str, float]:
        """Calculate forecast accuracy metrics."""
        # Align lengths
        min_len = min(len(actual), len(predicted))
        actual = actual.iloc[:min_len].values
        predicted = predicted[:min_len]
        
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        
        # MAPE (handle zeros)
        mask = actual != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask]))
        else:
            mape = np.nan
        
        return {
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape) if not np.isnan(mape) else 0.0
        }
    
    def _generate_insights(
        self,
        series: pd.Series,
        forecasts: dict[str, pd.DataFrame],
        best_model: str,
        periods: int
    ) -> list[str]:
        """Generate insights from the forecast."""
        insights = []
        
        if best_model and best_model in forecasts:
            forecast = forecasts[best_model]
            
            # Trend insight
            current_value = series.iloc[-1]
            forecast_end = forecast['yhat'].iloc[-1]
            change_pct = (forecast_end - current_value) / current_value * 100
            
            if change_pct > 5:
                insights.append(f"📈 Forecast shows {change_pct:.1f}% increase over the forecast period")
            elif change_pct < -5:
                insights.append(f"📉 Forecast shows {abs(change_pct):.1f}% decrease over the forecast period")
            else:
                insights.append(f"➡️ Forecast shows relatively stable values (±{abs(change_pct):.1f}%)")
            
            # Uncertainty insight
            avg_interval = (forecast['yhat_upper'] - forecast['yhat_lower']).mean()
            avg_value = forecast['yhat'].mean()
            uncertainty_pct = (avg_interval / avg_value) * 100 if avg_value != 0 else 0
            
            if uncertainty_pct > 30:
                insights.append(f"⚠️ High forecast uncertainty ({uncertainty_pct:.0f}% average interval width)")
            elif uncertainty_pct > 15:
                insights.append(f"📊 Moderate forecast uncertainty ({uncertainty_pct:.0f}% average interval width)")
            else:
                insights.append(f"✅ Low forecast uncertainty ({uncertainty_pct:.0f}% average interval width)")
            
            # Peak/trough insight
            peak_date = forecast['yhat'].idxmax()
            trough_date = forecast['yhat'].idxmin()
            insights.append(f"🔝 Forecast peak expected around {peak_date.strftime('%Y-%m-%d')}")
            insights.append(f"🔻 Forecast low expected around {trough_date.strftime('%Y-%m-%d')}")
        
        return insights