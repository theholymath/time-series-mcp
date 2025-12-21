from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import acf, pacf
from scipy.stats import boxcox

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter('ignore', ConvergenceWarning)

from typing import Optional, Any
from mcp.server.fastmcp import FastMCP
import pandas as pd
import numpy as np
from io import StringIO
import json

from data_models import NumericalDistributionAnalysis, CategoricalDistributionAnalysis


mcp = FastMCP("TimeSeriesMCP")

# key: dataset_name, value: pandas DF
DATA_STORE = {}

def _is_distribution_analysis(value: Any) -> bool:
    """Check if dict is a serialized distribution analysis."""
    return isinstance(value, dict) and "distribution_type" in value


@mcp.tool()
async def load_analysis_from_disk(filepath: str) -> str:
    """
    Load previously saved analysis data back into memory.

    Args:
        filepath: Path to the analysis JSON file created by write_data_analysis_to_disk

    Returns:
        Success message with loaded dataset names
    """
    try:
        # Load JSON file
        with open(filepath, 'r') as f:
            saved_data = json.load(f)

        # Reconstruct DATA_STORE
        loaded_datasets = []
        for dataset_name, dataset_data in saved_data.items():
            DATA_STORE[dataset_name] = {}

            for key, value in dataset_data.items():
                if key == "df":
                    # Reconstruct DataFrame from JSON string
                    df = pd.read_json(StringIO(value))
                    DATA_STORE[dataset_name]["df"] = df

                elif _is_distribution_analysis(value):
                    # Reconstruct Pydantic model based on distribution_type
                    if value["distribution_type"] == "numerical":
                        model = NumericalDistributionAnalysis(**value)
                    elif value["distribution_type"] == "categorical":
                        model = CategoricalDistributionAnalysis(**value)
                    else:
                        # Unknown type, skip
                        continue

                    DATA_STORE[dataset_name][key] = model
                else:
                    # Other data, store as-is
                    DATA_STORE[dataset_name][key] = value

            loaded_datasets.append(dataset_name)

        return f"Successfully loaded analysis for {len(loaded_datasets)} dataset(s): {', '.join(loaded_datasets)}"

    except FileNotFoundError:
        return f"Error: File '{filepath}' not found"
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON format - {str(e)}"
    except Exception as e:
        return f"Error loading analysis: {str(e)}"
    
@mcp.tool()
def load_dataset(filepath:str) -> str:
    """
    Load a CSV or JSON file into server's memory

    Args:
        filepath: the absolute path of the csv or json
    """
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.json'):
            df = pd.read_json(filepath)
        else:
            return "Error: only csva and json"
        
        name = filepath.split('/')[-1]
        DATA_STORE[name] = {"df":df}

        return  f"Successfully loaded '{name}' with {len(df)} rows."
    except Exception as e:
        return f"Error loading file: {str(e)}"


@mcp.tool()
def check_stationarity(dataset_name: str, target_column: str) -> str:
    """
    Checks if a time series is stationary using the Augmented Dickey-Fuller test.
    Essential for deciding if differencing (d > 0) is needed for ARIMA.
    """
    if dataset_name not in DATA_STORE:
        return f"Error: Dataset '{dataset_name}' not found."
    
    df = DATA_STORE[dataset_name]["df"]
    if target_column not in df.columns:
        return f"Error: Column '{target_column}' not found."

    # Drop NAs just for the test
    series = df[target_column].dropna()
    
    try:
        result = adfuller(series)
        p_value = result[1]
        
        output = []
        output.append(f"=== Stationarity Test (ADF) for '{target_column}' ===")
        output.append(f"ADF Statistic: {result[0]:.4f}")
        output.append(f"p-value: {result[1]:.4f}")
        
        if p_value < 0.05:
            output.append("Verdict: STATIONARY (Reject Null Hypothesis)")
            output.append("Suggestion: You likely do NOT need differencing (d=0).")
        else:
            output.append("Verdict: NON-STATIONARY (Fail to Reject Null Hypothesis)")
            output.append("Suggestion: You likely NEED differencing (d=1 or more).")
            
        return "\n".join(output)
    except Exception as e:
        return f"Error running ADF test: {str(e)}"

@mcp.tool()
def decompose_series(dataset_name: str, target_column: str, period: int = 1) -> str:
    """
    Decomposes the series into Trend, Seasonality, and Residuals.
    
    Args:
        dataset_name: Name of the loaded dataset.
        target_column: The column to analyze.
        period: The frequency of the data (e.g., 7 for daily, 12 for monthly, 4 for quarterly).
                If unknown, try 1, but results will be limited.
    """
    if dataset_name not in DATA_STORE:
        return f"Error: Dataset '{dataset_name}' not found."
    
    df = DATA_STORE[dataset_name]["df"]
    series = df[target_column].dropna()
    
    try:
        # We use an additive model by default for simplicity
        decomposition = seasonal_decompose(series, model='additive', period=period)
        
        trend = decomposition.trend.dropna()
        seasonal = decomposition.seasonal.dropna()
        resid = decomposition.resid.dropna()
        
        output = []
        output.append(f"=== Decomposition Analysis (Period={period}) ===")
        
        # Analyze Trend
        trend_direction = "Upward" if trend.iloc[-1] > trend.iloc[0] else "Downward"
        output.append(f"Trend: Appears to be {trend_direction}")
        
        # Analyze Seasonality Strength (Variance comparison)
        # If seasonal variance is high relative to residual variance, seasonality is strong
        seasonal_var = seasonal.var()
        resid_var = resid.var()
        
        output.append(f"Seasonal Variance: {seasonal_var:.4f}")
        output.append(f"Residual Variance: {resid_var:.4f}")
        
        if seasonal_var > resid_var:
            output.append("Verdict: Strong Seasonality detected.")
        else:
            output.append("Verdict: Weak or No Seasonality detected (or wrong period used).")
            
        return "\n".join(output)
        
    except Exception as e:
        return f"Error running decomposition: {str(e)}. Did you specify the correct 'period'?"

@mcp.tool()
def apply_transformation(dataset_name: str, target_column: str, transformation_type: str) -> str:
    """
    Apply a transformation to a column to stabilize variance or mean.
    Creates a NEW column in the dataset.
    
    Args:
        dataset_name: Name of the dataset.
        target_column: The column to transform.
        transformation_type: 'log', 'diff' (first difference), 'boxcox', or 'sqrt'.
    """
    if dataset_name not in DATA_STORE:
        return f"Error: Dataset '{dataset_name}' not found."
    
    df = DATA_STORE[dataset_name]["df"]
    series = df[target_column].dropna()
    
    new_col_name = f"{target_column}_{transformation_type}"
    
    try:
        if transformation_type == 'log':
            # Handle zeros/negatives by adding a small constant if needed
            if (series <= 0).any():
                return "Error: Cannot log-transform data with zero or negative values."
            df[new_col_name] = np.log(df[target_column])
            
        elif transformation_type == 'diff':
            df[new_col_name] = df[target_column].diff()
            
        elif transformation_type == 'sqrt':
            if (series < 0).any():
                return "Error: Cannot sqrt-transform negative values."
            df[new_col_name] = np.sqrt(df[target_column])
            
        elif transformation_type == 'boxcox':
            if (series <= 0).any():
                return "Error: Box-Cox requires strictly positive data."
            transformed_data, lambda_param = boxcox(series)
            df[new_col_name] = pd.Series(transformed_data, index=series.index)
            return f"Applied Box-Cox (lambda={lambda_param:.4f}). Created column '{new_col_name}'."
            
        else:
            return "Error: Unknown transformation. Use 'log', 'diff', 'sqrt', or 'boxcox'."
        
        DATA_STORE[dataset_name]["df"] = df
            
        return f"Success. Created new column '{new_col_name}'. You can now analyze this column."
        
    except Exception as e:
        return f"Error applying transformation: {str(e)}"




@mcp.tool()
def calculate_acf_pacf(dataset_name: str, target_column: str, lags: int = 20) -> str:
    """
    Calculate Autocorrelation (ACF) and Partial Autocorrelation (PACF) 
    to identify ARIMA parameters (p and q).
    """
    if dataset_name not in DATA_STORE:
        return f"Error: Dataset '{dataset_name}' not found."
    
    df = DATA_STORE[dataset_name]["df"]
    # Drop NAs (crucial for diffed data)
    series = df[target_column].dropna()
    
    try:
        # Calculate ACF and PACF
        # nlags is the number of lags to return
        acf_values, acf_confint = acf(series, nlags=lags, alpha=0.05)
        pacf_values, pacf_confint = pacf(series, nlags=lags, alpha=0.05)
        
        output = []
        output.append(f"=== ACF & PACF Analysis for '{target_column}' ===")
        output.append("Values > 1.96/sqrt(N) are roughly significant (marked with *)")
        
        # Simple threshold for significance (approximate)
        threshold = 1.96 / np.sqrt(len(series))
        
        # Create a readable table
        rows = []
        for i in range(1, lags + 1): # Skip lag 0
            acf_val = acf_values[i]
            pacf_val = pacf_values[i]
            
            acf_sig = "*" if abs(acf_val) > threshold else " "
            pacf_sig = "*" if abs(pacf_val) > threshold else " "
            
            rows.append({
                "Lag": i,
                "ACF": f"{acf_val:.3f} {acf_sig}",
                "PACF": f"{pacf_val:.3f} {pacf_sig}"
            })
            
        output.append(pd.DataFrame(rows).to_markdown(index=False))
        
        # Heuristic Hints
        output.append("\n--- Rules of Thumb ---")
        output.append("AR(p): ACF tails off, PACF cuts off after lag p.")
        output.append("MA(q): ACF cuts off after lag q, PACF tails off.")
        
        return "\n".join(output)
        
    except Exception as e:
        return f"Error calculating ACF/PACF: {str(e)}"


@mcp.tool()
def run_arima_forecast(dataset_name: str, target_column: str, p: int, d: int, q: int, steps: int = 5) -> str:
    """
    Fit an ARIMA(p,d,q) model and forecast future values.
    
    Args:
        dataset_name: Name of the dataset.
        target_column: The value column to forecast.
        p: AR order (lag observations).
        d: I order (differencing).
        q: MA order (moving average).
        steps: Number of future steps to forecast.
    """
    if dataset_name not in DATA_STORE:
        return f"Error: Dataset '{dataset_name}' not found."
    
    df = DATA_STORE[dataset_name]["df"]
    series = df[target_column].dropna()
    
    try:
        # Fit Model
        model = ARIMA(series, order=(p, d, q))
        model_fit = model.fit()
        
        # Forecast
        forecast = model_fit.forecast(steps=steps)
        
        output = []
        output.append(f"=== ARIMA({p},{d},{q}) Forecast Results ===")
        output.append(f"AIC Score: {model_fit.aic:.2f} (Lower is better)")
        output.append("\nForecasted Values:")
        
        # Create a simple table for the forecast
        forecast_df = pd.DataFrame({'Step': range(1, steps + 1), 'Forecast': forecast.values})
        output.append(forecast_df.to_markdown(index=False))
        
        return "\n".join(output)
        
    except Exception as e:
        return f"Error fitting ARIMA model: {str(e)}"

@mcp.tool()
def run_exponential_smoothing_forecast(
    dataset_name: str, 
    target_column: str, 
    trend: Optional[str] = 'add',           # Explicitly allow None
    seasonal: Optional[str] = None,         # Explicitly allow None
    seasonal_periods: Optional[int] = None, # Explicitly allow None
    steps: int = 5
) -> str:
    """
    Fit a Holt-Winters Exponential Smoothing model.
    
    Args:
        dataset_name: Name of the dataset.
        target_column: The value column.
        trend: 'add', 'mul', or None.
        seasonal: 'add', 'mul', or None.
        seasonal_periods: The number of periods in a season (e.g., 12 for monthly). Required if seasonal is set.
        steps: Number of future steps to forecast.
    """
    if dataset_name not in DATA_STORE:
        return f"Error: Dataset '{dataset_name}' not found."
    
    df = DATA_STORE[dataset_name]["df"]
    series = df[target_column].dropna()
    
    # Clean inputs: Ensure string "None" or empty strings become Python None
    if isinstance(trend, str) and trend.lower() in ['none', 'null', '']:
        trend = None
    
    if isinstance(seasonal, str) and seasonal.lower() in ['none', 'null', '']:
        seasonal = None
        
    # Validation: If seasonal is used, periods must be provided
    if seasonal is not None and seasonal_periods is None:
        return "Error: You must provide 'seasonal_periods' (e.g., 12) when using seasonality."

    try:
        # Fit Model
        model = ExponentialSmoothing(
            series, 
            trend=trend, 
            seasonal=seasonal, 
            seasonal_periods=seasonal_periods
        )
        model_fit = model.fit()
        
        # Forecast
        forecast = model_fit.forecast(steps=steps)
        
        output = []
        output.append("=== Exponential Smoothing Forecast ===")
        output.append(f"Params: Trend={trend}, Seasonal={seasonal}, Periods={seasonal_periods}")
        output.append("\nForecasted Values:")
        
        forecast_df = pd.DataFrame({'Step': range(1, steps + 1), 'Forecast': forecast.values})
        output.append(forecast_df.to_markdown(index=False))
        
        return "\n".join(output)
        
    except Exception as e:
        return f"Error fitting Exponential Smoothing: {str(e)}"
    
if __name__ == "__main__":
    mcp.run()
