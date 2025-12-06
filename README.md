# 🕐 Time Series Forecasting MCP

A Model Context Protocol (MCP) server for intelligent time series analysis and forecasting.

## Features

- **Automatic Data Loading**: CSV, Excel, JSON, or Parquet files
- **Intelligent Data Cleaning**: Handle missing values, outliers, and inconsistent frequencies
- **Multiple Forecasting Models**: ARIMA/SARIMA, Prophet, ETS, LSTM, Ensemble
- **Automatic Model Selection**: AI-driven selection based on data characteristics
- **Rich Visualizations**: Time series plots, forecasts, decomposition
- **Export Options**: CSV, PNG, HTML, JSON outputs

## Installation

### Option 1: From PyPI (when published)

```bash
pip install time-series-mcp
```

### Option 2: From Source (for development)

```bash
git clone https://github.com/yourusername/time-series-mcp.git
cd time-series-mcp
pip install -e .
```

## Configuration

Add to your Claude Desktop config file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

### If Installed from PyPI

```json
{
  "mcpServers": {
    "time-series-mcp": {
      "command": "uvx",
      "args": ["time-series-mcp"]
    }
  }
}
```

### If Installed from Source/Development

```json
{
  "mcpServers": {
    "time-series-mcp": {
      "command": "python",
      "args": ["-m", "time_series_mcp"]
    }
  }
}
```

**Restart Claude Desktop** after updating the config.

## Quick Start

Ask Claude:

```
"Load my sales data from sales_2024.csv and create a 2-month forecast"
```

Claude will:
1. Load and validate your data
2. Clean and preprocess it
3. Analyze patterns (trend, seasonality)
4. Select and train appropriate models
5. Generate forecasts with confidence intervals
6. Export plots and CSV files

## Available Tools

| Tool | Description |
|------|-------------|
| `load_dataset` | Load time series data from file |
| `analyze_time_series` | Comprehensive time series analysis |
| `clean_dataset` | Clean and preprocess data |
| `create_forecast` | Generate forecasts with selected models |
| `compare_models` | Compare multiple forecasting models |
| `export_forecast` | Export results to files |
| `quick_forecast` | One-shot forecasting (load → clean → forecast → export) |

## Example Prompts

- "Create a 30-day sales forecast based on my data"
- "Analyze this time series for seasonality and trends"
- "Compare ARIMA, Prophet, and ETS models on my dataset"
- "Generate a forecast report with visualizations"

## Development

```bash
# Clone the repository
git clone https://github.com/yourusername/time-series-mcp.git
cd time-series-mcp

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run the server directly
python -m time_series_mcp
```

## Publishing to PyPI

To make this available via `uvx time-series-mcp`:

```bash
# Build the package
python -m build

# Upload to PyPI
python -m twine upload dist/*
```

Then users can simply:
```bash
uvx time-series-mcp
```

## Troubleshooting

### Server not showing in Claude Desktop

1. Check the config file path is correct
2. Ensure JSON is valid (no trailing commas)
3. Restart Claude Desktop completely
4. Check Claude logs: `~/Library/Logs/Claude/` (macOS)

### Python module not found

- Ensure package is installed: `pip list | grep time-series-mcp`
- Try reinstalling: `pip install -e .`

### Dependencies failing

This package has large dependencies (PyTorch, etc). Ensure you have:
- Python 3.11+
- Sufficient disk space (~2GB for all dependencies)
- Stable internet connection

## License

MIT
