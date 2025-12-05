## UNDER DEVELOPMENT ... NOT FOR USE YET 


# 🕐 Time Series Forecasting MCP

A Model Context Protocol (MCP) server for intelligent time series analysis and forecasting.

## Features

- **Automatic Data Loading**: Load CSV, Excel, JSON, or Parquet files with auto-detection of date and target columns
- **Intelligent Data Cleaning**: Handle missing values, outliers, and inconsistent frequencies
- **Comprehensive Analysis**: Stationarity tests, seasonality detection, trend analysis, autocorrelation
- **Multiple Forecasting Models**:
  - ARIMA/SARIMA (Statistical)
  - Prophet (Facebook)
  - Exponential Smoothing (ETS)
  - LSTM (Deep Learning)
  - Ensemble methods
- **Automatic Model Selection**: AI-driven selection of best model based on data characteristics
- **Rich Visualizations**: Time series plots, forecasts, decomposition, model comparison
- **Export Options**: CSV, PNG, HTML, JSON outputs

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/time-series-mcp.git
cd time-series-mcp

# Install with pip
pip install -e .

# Or with uv
uv pip install -e .
```

## Quick Start

### Configure with Claude Desktop

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "time-series-mcp": {
      "command": "python",
      "args": ["-m", "time_series_mcp.server"]
    }
  }
}
```

### Example Usage

Simply ask Claude:

> "Load my sales data from sales_2024.csv and create a 2-month forecast"

The MCP will:
1. Load and validate your data
2. Clean and preprocess it
3. Analyze patterns (trend, seasonality)
4. Select and train appropriate models
5. Generate forecasts with confidence intervals
6. Export plots and CSV files

## Available Tools

| Tool | Description |
|------|-------------|
| `load_time_series` | Load time series data from file |
| `analyze_time_series` | Comprehensive time series analysis |
| `clean_time_series` | Clean and preprocess data |
| `create_forecast` | Generate forecasts with selected models |
| `compare_models` | Compare multiple forecasting models |
| `generate_forecast_report` | Create comprehensive report |
| `export_forecast` | Export results to files |
| `quick_forecast` | One-shot forecasting (load → clean → forecast → export) |

## Example Prompts

- "Create a 30-day sales forecast based on my data"
- "Analyze this time series for seasonality and trends"
- "Compare ARIMA, Prophet, and ETS models on my dataset"
- "Generate a forecast report with visualizations"

## License

MIT License

---

## Summary

This Time Series Forecasting MCP provides:

1. **7 Powerful Tools** for complete forecasting workflow
2. **Multiple Model Support**: ARIMA, SARIMA, Prophet, ETS, LSTM, Ensemble
3. **Intelligent Auto-Selection**: AI picks the best model based on data
4. **Data Quality Handling**: Missing values, outliers, frequency detection
5. **Rich Outputs**: CSV forecasts, PNG plots, HTML reports
6. **One-Shot Forecasting**: The `quick_forecast` tool handles everything in one call

The architecture mirrors the quick-data-mcp pattern with:
- Clean separation of concerns (loader, cleaner, forecaster, visualizer)
- Session state management
- Comprehensive error handling
- Rich markdown output for Claude

Would you like me to add any additional features, such as:
- More advanced models (XGBoost, Transformer-based)?
- Anomaly detection capabilities?
- Multi-variate forecasting support?
- Custom seasonality handling?
