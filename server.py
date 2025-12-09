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
import matplotlib.pyplot as plt
import io
import os

mcp = FastMCP("TimeSeriesMCP")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "forecast_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
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
def run_sarima_forecast(dataset_name: str, target_column: str, p: int, d: int, q: int, 
                        P:int = 0, D: int = 0, Q: int = 0, s: int = 0, steps: int = 5) -> str:
    """
    Fit a SARIMA(p,d,q)(P,D,Q)[s] model and forecast.
    
    Args:
        dataset_name: Name of the dataset.
        target_column: The value column.
        p, d, q: Non-seasonal order.
        P, D, Q: Seasonal order (default 0).
        s: Seasonal periodicity (e.g., 12 for monthly). Default 0 (no seasonality).
        steps: Future steps to forecast.
    """
    if dataset_name not in DATA_STORE:
        return f"Error: Dataset '{dataset_name}' not found."
    
    df = DATA_STORE[dataset_name]
    series = df[target_column].dropna()
    
    try:
        # Construct seasonal order tuple
        seasonal_order = (P, D, Q, s) if s > 0 else None
        
        # Fit Model
        model = ARIMA(series, order=(p, d, q), seasonal_order=seasonal_order)
        model_fit = model.fit()
        
        # Forecast
        forecast = model_fit.forecast(steps=steps)
        
        # --- NEW: Generate Plot ---
        # Create a proper index for the forecast for plotting
        last_idx = series.index[-1]
        if isinstance(last_idx, (int, float)):
            forecast_idx = range(int(last_idx) + 1, int(last_idx) + 1 + steps)
        else:
            # Fallback for non-numeric index: just use steps 1..N
            forecast_idx = range(len(series), len(series) + steps)
            
        forecast_series = pd.Series(forecast.values, index=forecast_idx)
        
        plot_path = _save_forecast_plot(
            dataset_name, 
            series, 
            forecast_series, 
            f"SARIMA({p},{d},{q})x({P},{D},{Q},{s})"
        )
        
        # Output Construction
        output = []
        output.append(f"=== SARIMA Forecast Results ===")
        output.append(f"AIC: {model_fit.aic:.2f}")
        output.append(f"Plot Saved To: {plot_path}")  # <--- Tell the user where it is
        output.append("\nForecasted Values:")
        
        forecast_df = pd.DataFrame({'Step': range(1, steps + 1), 'Forecast': forecast.values})
        output.append(forecast_df.to_markdown(index=False))
        
        return "\n".join(output)
        
    except Exception as e:
        return f"Error fitting SARIMA model: {str(e)}"
@mcp.tool()
def run_exponential_smoothing_forecast(
    dataset_name: str, 
    target_column: str, 
    trend: Optional[str] = 'add', 
    seasonal: Optional[str] = None, 
    seasonal_periods: Optional[int] = None, 
    steps: int = 5
) -> str:
    """
    Fit a Holt-Winters Exponential Smoothing model.
    """
    if dataset_name not in DATA_STORE:
        return f"Error: Dataset '{dataset_name}' not found."
    
    df = DATA_STORE[dataset_name]
    series = df[target_column].dropna()
    
    # Input cleaning (same as before)
    if isinstance(trend, str) and trend.lower() in ['none', 'null', '']: trend = None
    if isinstance(seasonal, str) and seasonal.lower() in ['none', 'null', '']: seasonal = None
    if seasonal is not None and seasonal_periods is None:
        return "Error: You must provide 'seasonal_periods' when using seasonality."

    try:
        model = ExponentialSmoothing(
            series, 
            trend=trend, 
            seasonal=seasonal, 
            seasonal_periods=seasonal_periods
        )
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)
        
        # --- NEW: Generate Plot ---
        # Simple index handling
        last_idx = len(series)
        forecast_idx = range(last_idx, last_idx + steps)
        forecast_series = pd.Series(forecast.values, index=forecast_idx)
        
        plot_path = _save_forecast_plot(
            dataset_name, 
            series, 
            forecast_series, 
            f"ETS(trend={trend}, seasonal={seasonal})"
        )
        
        output = []
        output.append(f"=== Exponential Smoothing Forecast ===")
        output.append(f"Plot Saved To: {plot_path}") # <--- Tell the user
        output.append("\nForecasted Values:")
        
        forecast_df = pd.DataFrame({'Step': range(1, steps + 1), 'Forecast': forecast.values})
        output.append(forecast_df.to_markdown(index=False))
        
        return "\n".join(output)
        
    except Exception as e:
        return f"Error fitting Exponential Smoothing: {str(e)}"

def _save_forecast_plot(
    dataset_name: str, 
    history: pd.Series, 
    forecast: pd.Series, 
    title: str
) -> str:
    """
    Internal helper to plot history + forecast and save to disk.
    Returns the absolute path of the saved file.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot last 50 points of history context (to keep plot readable)
    history_tail = history.tail(50)
    plt.plot(history_tail.index, history_tail.values, label='History (Last 50)', color='blue')
    
    # Plot Forecast
    # Create a continuous index for plotting if possible, else just append
    plt.plot(forecast.index, forecast.values, label='Forecast', color='red', linestyle='--')
    
    plt.title(f"{title} - {dataset_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save file
    filename = f"{dataset_name}_{title.replace(' ', '_')}.png"
    filepath = os.path.abspath(os.path.join(OUTPUT_DIR, filename))
    plt.savefig(filepath)
    plt.close()
    
    return filepath

if __name__ == "__main__":
    mcp.run()
