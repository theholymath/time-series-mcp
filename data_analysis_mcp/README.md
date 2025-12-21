# Data Analysis MCP Server

Data analysis and distribution analysis for datasets.

## Features

- Load and analyze datasets (CSV/JSON)
- View dataset metadata and statistics
- Distribution analysis (numerical and categorical)
- Save and load analysis results

## Running the Server

```bash
# From project root
uv run data_analysis_mcp/server.py

# Or from this directory
cd data_analysis_mcp
uv run server.py
```

## Testing with MCP Inspector

```bash
npx @modelcontextprotocol/inspector uv run data_analysis_mcp/server.py
```

## Available Tools

- `load_dataset(filepath)` - Load CSV/JSON into memory
- `get_dataset_metadata(dataset_name)` - View data summary and statistics
- `column_distributions_analysis(dataset_name, column_name)` - Analyze numerical/categorical distributions
- `write_data_analysis_to_disk(dataset_names)` - Save analysis results to JSON
- `load_analysis_from_disk(filepath)` - Load previously saved analysis

## Example Workflow

1. Load your dataset: `load_dataset("/path/to/data.csv")`
2. Inspect the data: `get_dataset_metadata("data.csv")`
3. Analyze distributions: `column_distributions_analysis("data.csv", "column_name")`
4. Save results: `write_data_analysis_to_disk(["data.csv"])`
