# Time Series MCP Collection

A workspace of independent MCP servers for time series analysis and data science.

## Structure

This is a UV workspace containing multiple independent MCP servers:

- **ARIMA_mcp/** - ARIMA and Exponential Smoothing forecasting
- **data_analysis_mcp/** - General data analysis and distribution tools
- **hello_world_mcp/** - Simple example server
- **data_models/** - Shared Pydantic models and utilities

Each MCP server has its own dependencies and can be run independently.

## Running MCP Servers

```bash
# ARIMA time series server
uv run ARIMA_mcp/server.py

# Test with MCP Inspector
npx @modelcontextprotocol/inspector uv run ARIMA_mcp/server.py

# add to Claude code locally from root directory
claude mcp add --transport stdio data_analysis_mcp -- uv run python data_analysis_mcp/server.py
 claude mcp add --transport stdio arima_mcp -- uv run python ARIMA_mcp/server.py
```

## Shared Modules

The `data_models` package provides shared Pydantic models and utilities used across MCP servers. This prevents code duplication and ensures consistency.

**Why src layout?**
- Prevents accidental imports from working directory
- Changes are immediately available (editable install)
- Follows Python packaging best practices

**Using shared modules in an MCP:**

1. Add dependency in `{mcp_name}/pyproject.toml`:
```toml
dependencies = ["data_models"]

[tool.uv.sources]
data_models = { workspace = true }
```

2. Import in your server:
```python
from data_models import NumericalDistributionAnalysis, CategoricalDistributionAnalysis
```

3. Run `uv sync` to update dependencies

## Adding New MCP Servers

1. Create a new directory ending with `_mcp`
2. Add it to `members` in root `pyproject.toml`
3. Create `{name}_mcp/pyproject.toml` with dependencies
4. Add your `server.py` and code

## Workspace Benefits

- **Independent dependencies** - Each MCP can use different package versions
- **Clean separation** - No dependency conflicts between MCPs
- **Easy testing** - Run and test each MCP in isolation
- **Scalable** - Add new MCPs without affecting existing ones