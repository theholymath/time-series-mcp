# Smoke Tests for Time Series MCP

This directory contains smoke tests for the time-series-mcp server using the mcp-use framework.

## Prerequisites

1. **Install dependencies:**
   ```bash
   cd /home/user/time-series-mcp
   uv sync --extra smoke
   ```

2. **Set up environment variables:**
   Create a `.env` file in the project root with your OpenAI API key:
   ```bash
   OPENAI_API_KEY=your_api_key_here
   ```

3. **Prepare test data (for applicable tests):**
   - **test_forecast_longley.py**: No external data needed (uses statsmodels built-in dataset)
   - **test_forecast_uber_lyft.py**: Requires Uber & Lyft cab ride dataset
     - Download from: https://www.kaggle.com/datasets/ravi72munde/uber-lyft-cab-prices
     - Place `cab_rides.csv` at: `data/cab_weather/cab_rides.csv`
   - **test_mcp_use.py**: No data needed (tests MCP server connectivity)

## Running the Tests

### From the smoke test directory:

```bash
cd tests/smoke

# Test 1: Basic MCP server connectivity
python test_mcp_use.py

# Test 2: Longley dataset forecast (built-in data)
python test_forecast_longley.py

# Test 3: Uber & Lyft cab ride forecast (requires external data)
python test_forecast_uber_lyft.py
```

### From the project root:

```bash
cd /home/user/time-series-mcp

# Run smoke tests
python tests/smoke/test_mcp_use.py
python tests/smoke/test_forecast_longley.py
python tests/smoke/test_forecast_uber_lyft.py
```

## Test Descriptions

### test_mcp_use.py
Tests basic MCP server connectivity and tool listing. This verifies that:
- The MCP server can be started
- The AI agent can connect to the server
- Available tools and resources can be listed

### test_forecast_longley.py
Tests time series forecasting using the statsmodels Longley dataset (US macroeconomic data 1947-1962). This verifies:
- Dataset loading and preparation
- Time series forecasting with employment data
- Forecast generation with confidence intervals

### test_forecast_uber_lyft.py
Tests time series forecasting using the Uber & Lyft cab ride dataset. This verifies:
- Loading external CSV data
- Forecasting cab ride demand/prices
- Handling high-frequency time series data (5-minute intervals)

## Configuration

The tests use the MCP server configuration from `config/time_series_mcp_config.json`, which specifies how to run the time-series-mcp server using `uv`.

## Troubleshooting

**Issue: "OPENAI_API_KEY not found"**
- Solution: Create a `.env` file in the project root with your API key

**Issue: "Dataset not found" for Uber & Lyft test**
- Solution: Download the dataset from Kaggle and place it in `data/cab_weather/cab_rides.csv`

**Issue: "ModuleNotFoundError: No module named 'mcp_use'"**
- Solution: Run `uv sync --extra smoke` to install smoke test dependencies

**Issue: MCP server connection errors**
- Solution: Ensure the MCP server can run with `uv run time-series-mcp`
- Check that all dependencies are installed with `uv sync`
