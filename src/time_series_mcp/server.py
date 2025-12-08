"""
Time Series Forecasting MCP Server
===================================
An MCP server that provides intelligent time series analysis and forecasting.
"""

import logging
from typing import Optional, List, Dict, Any

from mcp.server import FastMCP

from .models.schemas import DatasetManager
from .tools import dataset_tools
from .tools.data_cleaner import TimeSeriesDataCleaner
from .tools.feature_engineer import TimeSeriesFeatureEngineer
from .tools.model_selector import ModelSelector
from .tools.forecaster import TimeSeriesForecaster
from .tools.visualizer import TimeSeriesVisualizer
from .utils.output_handler import OutputHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the FastMCP server instance
mcp = FastMCP(
    name="time-series-mcp",
    dependencies=[
        "mcp>=1.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "statsmodels>=0.14.0",
        "prophet>=1.1.4",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.15.0",
    ]
)

# Initialize components
data_cleaner = TimeSeriesDataCleaner()
feature_engineer = TimeSeriesFeatureEngineer()
model_selector = ModelSelector()
forecaster = TimeSeriesForecaster()
visualizer = TimeSeriesVisualizer()
output_handler = OutputHandler()


# ============================================================================
# DATASET MANAGEMENT TOOLS
# ============================================================================

@mcp.tool()
async def load_dataset(
    file_path: str,
    dataset_name: str,
    date_column: Optional[str] = None,
    target_column: Optional[str] = None,
    frequency: str = "auto",
    sample_size: Optional[int] = None
) -> dict:
    """
    Load a time series dataset from a file path.
    Supports CSV, Excel, JSON, and Parquet formats.
    Automatically detects date and target columns.

    Args:
        file_path: Path to the data file
        dataset_name: Name to store the dataset under (for later reference)
        date_column: Name of the datetime column (auto-detected if not provided)
        target_column: Name of the target variable to forecast (auto-detected if not provided)
        frequency: Data frequency: 'D' (daily), 'W' (weekly), 'M' (monthly), 'H' (hourly), 'auto'
        sample_size: If provided, sample the dataset to this size

    Returns:
        dict: Result with dataset info and preview
    """
    return await dataset_tools.load_dataset(
        file_path=file_path,
        dataset_name=dataset_name,
        date_column=date_column,
        target_column=target_column,
        frequency=frequency,
        sample_size=sample_size
    )


@mcp.tool()
async def load_dataset_from_content(
    content: str,
    dataset_name: str,
    content_type: str = "csv",
    date_column: Optional[str] = None,
    target_column: Optional[str] = None,
    frequency: str = "auto"
) -> dict:
    """
    Load a time series dataset from raw CSV or JSON content.
    Use this when the user uploads a file directly in the chat.

    Args:
        content: Raw CSV or JSON content as a string
        dataset_name: Name to store the dataset under
        content_type: Type of content ('csv' or 'json')
        date_column: Name of the datetime column (auto-detected if not provided)
        target_column: Name of the target variable (auto-detected if not provided)
        frequency: Data frequency

    Returns:
        dict: Result with dataset info and preview
    """
    return await dataset_tools.load_dataset_from_content(
        content=content,
        dataset_name=dataset_name,
        content_type=content_type,
        date_column=date_column,
        target_column=target_column,
        frequency=frequency
    )


@mcp.tool()
async def list_datasets() -> dict:
    """List all currently loaded datasets with their basic info."""
    return await dataset_tools.list_datasets()


@mcp.tool()
async def get_dataset_info(dataset_name: str) -> dict:
    """
    Get detailed information about a loaded dataset including statistics,
    missing values, outliers, and a preview.

    Args:
        dataset_name: Name of the dataset

    Returns:
        dict: Detailed dataset information
    """
    return await dataset_tools.get_dataset_info(dataset_name)


@mcp.tool()
async def delete_dataset(dataset_name: str) -> dict:
    """
    Delete a loaded dataset from memory.

    Args:
        dataset_name: Name of the dataset to delete

    Returns:
        dict: Confirmation message
    """
    return await dataset_tools.delete_dataset(dataset_name)


@mcp.tool()
async def query_dataset(
    dataset_name: str,
    query: Optional[str] = None,
    columns: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> dict:
    """
    Query a loaded dataset with optional filtering by date range or conditions.

    Args:
        dataset_name: Name of the dataset
        query: Pandas query string (e.g., 'sales > 100')
        columns: Columns to return
        start_date: Filter data after this date (YYYY-MM-DD)
        end_date: Filter data before this date (YYYY-MM-DD)

    Returns:
        dict: Query results
    """
    return await dataset_tools.query_dataset(
        dataset_name=dataset_name,
        query=query,
        columns=columns,
        start_date=start_date,
        end_date=end_date
    )


# ============================================================================
# ANALYSIS TOOLS
# ============================================================================

@mcp.tool()
async def analyze_time_series(dataset_name: str, include_plots: bool = True) -> dict:
    """
    Perform comprehensive analysis on a loaded dataset.
    Includes stationarity tests, seasonality detection, trend analysis,
    autocorrelation analysis, and model recommendations.

    Args:
        dataset_name: Name of the dataset to analyze
        include_plots: Generate analysis plots

    Returns:
        dict: Analysis results
    """
    df = DatasetManager.get_dataset(dataset_name)
    schema = DatasetManager.get_schema(dataset_name)

    if df is None or schema is None:
        return {"status": "error", "message": f"Dataset '{dataset_name}' not found."}

    metadata = {
        "date_column": schema.date_column,
        "target_column": schema.target_column,
        "frequency": schema.frequency
    }

    analysis = data_cleaner.analyze(data=df, metadata=metadata)
    return {"status": "success", "analysis": analysis}


@mcp.tool()
async def clean_dataset(
    dataset_name: str,
    missing_strategy: str = "interpolate",
    outlier_strategy: str = "clip",
    outlier_threshold: float = 1.5
) -> dict:
    """
    Clean and preprocess a time series dataset.
    Handles missing values, outliers, duplicates, and resampling.

    Args:
        dataset_name: Name of the dataset to clean
        missing_strategy: Strategy for handling missing values (interpolate, ffill, bfill, mean, median, drop)
        outlier_strategy: Strategy for handling outliers (clip, remove, none)
        outlier_threshold: IQR multiplier for outlier detection

    Returns:
        dict: Cleaning results
    """
    df = DatasetManager.get_dataset(dataset_name)
    schema = DatasetManager.get_schema(dataset_name)

    if df is None or schema is None:
        return {"status": "error", "message": f"Dataset '{dataset_name}' not found."}

    metadata = {
        "date_column": schema.date_column,
        "target_column": schema.target_column,
        "frequency": schema.frequency
    }

    result = data_cleaner.clean(
        data=df,
        metadata=metadata,
        missing_strategy=missing_strategy,
        outlier_strategy=outlier_strategy,
        outlier_threshold=outlier_threshold
    )

    # Update the stored dataset
    from .models.schemas import loaded_datasets
    loaded_datasets[dataset_name] = result["data"]
    schema.row_count = len(result["data"])
    schema.missing_count = result["remaining_missing"]
    schema.missing_pct = (result["remaining_missing"] / len(result["data"])) * 100

    return {"status": "success", **result}


# ============================================================================
# FORECASTING TOOLS
# ============================================================================

@mcp.tool()
async def create_forecast(
    dataset_name: str,
    periods: int,
    frequency: Optional[str] = None,
    models: List[str] = None,
    confidence_level: float = 0.95
) -> dict:
    """
    Create a time series forecast for a loaded dataset.
    Automatically selects best model(s) based on data characteristics.
    Returns forecasts with confidence intervals.

    Args:
        dataset_name: Name of the dataset to forecast
        periods: Number of periods to forecast
        frequency: Forecast frequency (D, W, M, H). Uses data frequency if not specified
        models: Models to use: ['arima', 'sarima', 'prophet', 'ets', 'ensemble', 'auto']
        confidence_level: Confidence level for prediction intervals (0-1)

    Returns:
        dict: Forecast results with predictions and metrics
    """
    if models is None:
        models = ["auto"]

    df = DatasetManager.get_dataset(dataset_name)
    schema = DatasetManager.get_schema(dataset_name)

    if df is None or schema is None:
        return {"status": "error", "message": f"Dataset '{dataset_name}' not found."}

    metadata = {
        "date_column": schema.date_column,
        "target_column": schema.target_column,
        "frequency": schema.frequency
    }

    freq = frequency or schema.frequency

    # Auto-select models if needed
    if "auto" in models:
        models = model_selector.select_models(data=df, metadata=metadata)

    # Generate forecasts
    result = forecaster.forecast(
        data=df,
        metadata=metadata,
        periods=periods,
        frequency=freq,
        models=models,
        confidence_level=confidence_level,
        include_backtesting=True
    )

    # Store results
    for model_name, forecast_df in result["forecasts"].items():
        DatasetManager.store_forecast(
            dataset_name=dataset_name,
            model_name=model_name,
            forecast_df=forecast_df,
            metrics=result["metrics"].get(model_name, {}),
            periods=periods,
            frequency=freq
        )

    return {"status": "success", **result}


@mcp.tool()
async def compare_models(
    dataset_name: str,
    models: Optional[List[str]] = None,
    cv_folds: int = 5
) -> dict:
    """
    Compare multiple forecasting models on a dataset.
    Performs cross-validation and returns metrics for each model.

    Args:
        dataset_name: Name of the dataset
        models: Models to compare: ['arima', 'sarima', 'prophet', 'ets']
        cv_folds: Number of cross-validation folds

    Returns:
        dict: Comparison results with metrics
    """
    if models is None:
        models = ["arima", "prophet", "ets"]

    df = DatasetManager.get_dataset(dataset_name)
    schema = DatasetManager.get_schema(dataset_name)

    if df is None or schema is None:
        return {"status": "error", "message": f"Dataset '{dataset_name}' not found."}

    metadata = {
        "date_column": schema.date_column,
        "target_column": schema.target_column,
        "frequency": schema.frequency
    }

    comparison = model_selector.compare_models(
        data=df,
        metadata=metadata,
        models=models,
        cv_folds=cv_folds
    )

    return {"status": "success", **comparison}


@mcp.tool()
async def quick_forecast(
    forecast_description: str,
    file_path: Optional[str] = None,
    content: Optional[str] = None,
    date_column: Optional[str] = None,
    target_column: Optional[str] = None,
    output_dir: str = "./forecast_output"
) -> dict:
    """
    One-shot forecasting: load data, clean, train models, and forecast in one call.
    Perfect for quick analysis like "Create a 2-month sales forecast from this data".

    Can accept either a file path OR raw content from an uploaded file.

    Args:
        forecast_description: Natural language description like '2 month forecast' or '30 day prediction'
        file_path: Path to the data file (use this OR content, not both)
        content: Raw CSV/JSON content (use this OR file_path, not both)
        date_column: Name of date column (auto-detected if not provided)
        target_column: Name of target column (auto-detected if not provided)
        output_dir: Directory to save outputs

    Returns:
        dict: Forecast results with predictions
    """
    if not file_path and not content:
        return {"status": "error", "message": "Please provide either 'file_path' or 'content'."}

    dataset_name = "quick_forecast_data"

    # Step 1: Load data
    if file_path:
        load_result = await load_dataset(
            file_path=file_path,
            dataset_name=dataset_name,
            date_column=date_column,
            target_column=target_column
        )
    else:
        load_result = await load_dataset_from_content(
            content=content,
            dataset_name=dataset_name,
            date_column=date_column,
            target_column=target_column
        )

    if load_result.get("status") == "error":
        return load_result

    # Step 2: Clean data
    clean_result = await clean_dataset(
        dataset_name=dataset_name,
        missing_strategy="interpolate",
        outlier_strategy="clip"
    )

    # Step 3: Parse forecast description
    periods, frequency = _parse_forecast_description(forecast_description)

    # Step 4: Create forecast
    forecast_result = await create_forecast(
        dataset_name=dataset_name,
        periods=periods,
        frequency=frequency,
        models=["auto"]
    )

    return forecast_result


@mcp.tool()
async def export_forecast(
    dataset_name: str,
    output_path: str,
    formats: List[str] = None
) -> dict:
    """
    Export forecast results to files (CSV, PNG, HTML, JSON).

    Args:
        dataset_name: Name of the dataset with forecasts
        output_path: Path for output files (without extension)
        formats: Output formats: ['csv', 'png', 'html', 'json']

    Returns:
        dict: Paths to created files
    """
    if formats is None:
        formats = ["csv", "png"]

    df = DatasetManager.get_dataset(dataset_name)
    schema = DatasetManager.get_schema(dataset_name)
    forecasts_dict = DatasetManager.get_forecast(dataset_name)

    if df is None or schema is None:
        return {"status": "error", "message": f"Dataset '{dataset_name}' not found."}

    if not forecasts_dict:
        return {"status": "error", "message": f"No forecasts found for '{dataset_name}'. Run create_forecast first."}

    # Convert ForecastResult objects to DataFrames
    forecasts = {name: fr.forecast_df for name, fr in forecasts_dict.items()}

    # Find best model (lowest MAPE)
    best_model = min(forecasts_dict.items(), key=lambda x: x[1].metrics.get('mape', float('inf')))[0]

    metadata = {
        "date_column": schema.date_column,
        "target_column": schema.target_column,
        "frequency": schema.frequency
    }

    files_created = output_handler.export(
        forecasts=forecasts,
        best_model=best_model,
        data=df,
        metadata=metadata,
        output_path=output_path,
        formats=formats
    )

    return {"status": "success", "files": files_created}


# ============================================================================
# RESOURCES - Dynamic data context
# ============================================================================

@mcp.resource("datasets://loaded")
async def get_loaded_datasets_resource() -> str:
    """List of all currently loaded datasets with basic info."""
    result = await list_datasets()
    import json
    return json.dumps(result, indent=2)


@mcp.resource("datasets://{dataset_name}/info")
async def get_dataset_info_resource(dataset_name: str) -> str:
    """Detailed information about a specific dataset."""
    result = await get_dataset_info(dataset_name)
    import json
    return json.dumps(result, indent=2)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _parse_forecast_description(description: str) -> tuple[int, str]:
    """Parse natural language forecast description to periods and frequency."""
    import re

    description = description.lower()

    numbers = re.findall(r'\d+', description)
    num = int(numbers[0]) if numbers else 30

    if 'day' in description:
        return num, 'D'
    elif 'week' in description:
        return num * 7, 'D'
    elif 'month' in description:
        return num * 30, 'D'
    elif 'year' in description:
        return num * 365, 'D'
    elif 'hour' in description:
        return num, 'H'
    else:
        return num, 'D'


def get_server():
    """Get the MCP server instance."""
    return mcp


def main():
    """Main entry point for the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
