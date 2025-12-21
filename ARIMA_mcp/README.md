# ARIMA Time Series MCP Server

Time series analysis and forecasting using ARIMA and Exponential Smoothing models.

## Features

- Load and analyze time series data (CSV/JSON)
- Stationarity testing (Augmented Dickey-Fuller)
- Seasonal decomposition
- ARIMA forecasting
- Exponential Smoothing (Holt-Winters)

## Running the Server

```bash
# From project root
uv run ARIMA-mcp/server.py

# Or from this directory
cd ARIMA-mcp
uv run server.py
```

## Testing with MCP Inspector

```bash
npx @modelcontextprotocol/inspector uv run ARIMA-mcp/server.py
```

## Available Tools

- `load_dataset(filepath)` - Load CSV/JSON into memory
- `load_analysis_from_disk(filepath)` - Load previously saved analysis data
- `check_stationarity(dataset_name, target_column)` - Run ADF test
- `decompose_series(dataset_name, target_column, period)` - Seasonal decomposition
- `apply_transformation(dataset_name, target_column, transformation_type)` - Apply transformations (log, diff, sqrt, boxcox)
- `calculate_acf_pacf(dataset_name, target_column, lags)` - Calculate ACF/PACF for parameter identification
- `run_arima_forecast(dataset_name, target_column, p, d, q, steps)` - ARIMA forecasting
- `run_exponential_smoothing_forecast(...)` - Holt-Winters forecasting

## Example Workflow

1. Load your time series: `load_dataset("/path/to/data.csv")`
2. Inspect the data: `get_dataset_metadata("data.csv")`
3. Check stationarity: `check_stationarity("data.csv", "value_column")`
4. Run forecast: `run_arima_forecast("data.csv", "value_column", 1, 1, 1, steps=10)`
