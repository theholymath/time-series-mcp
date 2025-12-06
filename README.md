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

### Prerequisites

- Python 3.11 or higher
- `uv` package manager (recommended) or `pip`

### Step 1: Install uv (Recommended)

If you don't have `uv` installed:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

### Step 2: Clone and Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/time-series-mcp.git
cd time-series-mcp

# Install dependencies with uv (recommended)
uv sync

# OR install with pip
pip install -e .
```

## Configuration

### Method 1: Using uv (Recommended)

This is the most reliable method as it uses `uv` to manage the Python environment automatically.

1. **Find your absolute path to the project:**
   ```bash
   pwd
   ```
   This will output something like `/Users/yourname/projects/time-series-mcp`

2. **Copy the sample config:**
   ```bash
   cp .mcp.json.sample .mcp.json
   ```

3. **Edit `.mcp.json`** and replace `/ABSOLUTE/PATH/TO/time-series-mcp` with your actual project path:
   ```json
   {
     "mcpServers": {
       "time-series-mcp": {
         "command": "uv",
         "args": [
           "--directory",
           "/Users/yourname/projects/time-series-mcp",
           "run",
           "time-series-mcp"
         ],
         "env": {
           "LOG_LEVEL": "INFO"
         }
       }
     }
   }
   ```

4. **Add to Claude Desktop config** (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS or `%APPDATA%\Claude\claude_desktop_config.json` on Windows):

   Copy the contents of your edited `.mcp.json` file into your Claude Desktop config.

### Method 2: Using pip install with system Python

If you installed with `pip install -e .` and the package is globally available:

1. **Copy the pip sample config:**
   ```bash
   cp .mcp.json.pip.sample .mcp.json
   ```

2. **Add to Claude Desktop config:**
   ```json
   {
     "mcpServers": {
       "time-series-mcp": {
         "command": "python",
         "args": ["-m", "time_series_mcp.server"],
         "env": {}
       }
     }
   }
   ```

   **Note:** This requires `time-series-mcp` to be installed in your system Python or active virtual environment.

### Method 3: Using a specific virtual environment

If you prefer to use a specific virtual environment:

1. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -e .
   ```

2. **Find the absolute path to your venv Python:**
   ```bash
   which python  # On macOS/Linux
   where python  # On Windows
   ```
   This will output something like `/Users/yourname/projects/time-series-mcp/.venv/bin/python`

3. **Copy the venv sample config:**
   ```bash
   cp .mcp.json.venv.sample .mcp.json
   ```

4. **Edit `.mcp.json`** and replace the path:
   ```json
   {
     "mcpServers": {
       "time-series-mcp": {
         "command": "/Users/yourname/projects/time-series-mcp/.venv/bin/python",
         "args": ["-m", "time_series_mcp.server"],
         "env": {}
       }
     }
   }
   ```

5. **Add to Claude Desktop config** using the contents of your edited `.mcp.json`.

## Verification

After configuring, restart Claude Desktop and verify the MCP server is working:

1. Open Claude Desktop
2. Look for a small tool/hammer icon in the input area (indicates MCP tools are available)
3. Try asking Claude: "List the available time series datasets"

If you see MCP tools available, you're all set!

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
