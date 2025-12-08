from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter('ignore', ConvergenceWarning)

from typing import Optional
from mcp.server.fastmcp import FastMCP
import pandas as pd
import io

mcp = FastMCP("TimeSeriesMCP")

# key: dataset_name, value: pandas DF
DATA_STORE = {}

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
        DATA_STORE[name] = df

        return  f"Successfully loaded '{name}' with {len(df)} rows."
    except Exception as e:
        return f"Error loading file: {str(e)}"

@mcp.tool()
def get_dataset_metadata(dataset_name: str) -> str:
    """
    Get the 'Nutrition Label' of the dataset. Returns columns, types, and samples.
    NEVER read the whole file to the user.
    """
    if dataset_name not in DATA_STORE:
        return f"Error: Dataset '{dataset_name}' not found. Please load it first."

    df = DATA_STORE[dataset_name]

    output = []
    output.append(f"=== Metadata for {dataset_name} ===")
    output.append(f"Shape: {df.shape}")
    output.append("\nColumns & Types:")
    output.append(str(df.dtypes))
    
    output.append("\n=== First 5 Rows ===")
    output.append(df.head().to_markdown(index=False))
    
    output.append("\n=== Last 5 Rows ===")
    output.append(df.tail().to_markdown(index=False))
    
    output.append("\n=== Summary Statistics ===")
    output.append(df.describe().to_markdown())

    # Simple heuristic: look for 'date', 'time', 'year' in column names
    # this is lazy and needs to be figured out later. 
    time_cols = [c for c in df.columns if any(x in c.lower() for x in ['date', 'time', 'year', 'day'])]
    if time_cols:
        output.append(f"\nPotential Time Columns: {time_cols}")
        
    return "\n".join(output)

@mcp.tool()
def check_stationarity(dataset_name: str, target_column: str) -> str:
    """
    Checks if a time series is stationary using the Augmented Dickey-Fuller test.
    Essential for deciding if differencing (d > 0) is needed for ARIMA.
    """
    if dataset_name not in DATA_STORE:
        return f"Error: Dataset '{dataset_name}' not found."
    
    df = DATA_STORE[dataset_name]
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
    
    df = DATA_STORE[dataset_name]
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
    
    df = DATA_STORE[dataset_name]
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
    
    df = DATA_STORE[dataset_name]
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
        output.append(f"=== Exponential Smoothing Forecast ===")
        output.append(f"Params: Trend={trend}, Seasonal={seasonal}, Periods={seasonal_periods}")
        output.append("\nForecasted Values:")
        
        forecast_df = pd.DataFrame({'Step': range(1, steps + 1), 'Forecast': forecast.values})
        output.append(forecast_df.to_markdown(index=False))
        
        return "\n".join(output)
        
    except Exception as e:
        return f"Error fitting Exponential Smoothing: {str(e)}"
    
if __name__ == "__main__":
    mcp.run()
