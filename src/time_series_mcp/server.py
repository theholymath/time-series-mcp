"""
Time Series Forecasting MCP Server
===================================
An MCP server that provides intelligent time series analysis and forecasting.
"""



from .models.resources import resource_manager, ResourceType
from .tools.plot_tools import (
    create_forecast_plot,
    create_analysis_plot,
    modify_plot,
    list_plot_resources,
    export_plot,
    get_plot_for_display,
    delete_plot,
)


import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    Resource,
    ResourceTemplate,
)

from .models.schemas import DatasetManager, loaded_datasets, dataset_schemas, forecast_results
from .tools.dataset_tools import (
    load_dataset,
    load_dataset_from_content,
    list_datasets,
    get_dataset_info,
    delete_dataset,
    get_dataset_sample,
    query_dataset,
)
from .tools.data_cleaner import TimeSeriesDataCleaner
from .tools.feature_engineer import TimeSeriesFeatureEngineer
from .tools.model_selector import ModelSelector
from .tools.forecaster import TimeSeriesForecaster
from .tools.visualizer import TimeSeriesVisualizer
from .utils.output_handler import OutputHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the MCP server
server = Server("time-series-mcp")

# Initialize components
data_cleaner = TimeSeriesDataCleaner()
feature_engineer = TimeSeriesFeatureEngineer()
model_selector = ModelSelector()
forecaster = TimeSeriesForecaster()
visualizer = TimeSeriesVisualizer()
output_handler = OutputHandler()

@server.list_resources()
async def list_resources() -> list[Resource]:
    """List all available resources."""
    resources = []
    
    for managed_resource in resource_manager.list_resources():
        resources.append(Resource(
            uri=managed_resource.uri,
            name=managed_resource.name,
            description=managed_resource.description,
            mimeType=managed_resource.mime_type
        ))
    
    return resources


@server.read_resource()
async def read_resource(uri: str) -> str:
    """Read a resource by URI."""
    resource = resource_manager.get_resource(uri)
    
    if not resource:
        raise ValueError(f"Resource not found: {uri}")
    
    if resource.base64_data:
        return resource.base64_data
    
    if resource.file_path:
        with open(resource.file_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    raise ValueError(f"Resource has no data: {uri}")


@server.list_resource_templates()
async def list_resource_templates() -> list[ResourceTemplate]:
    """List resource templates."""
    return [
        ResourceTemplate(
            uriTemplate="ts://plots/{plot_name}",
            name="Time Series Plot",
            description="A generated time series visualization",
            mimeType="image/png"
        ),
        ResourceTemplate(
            uriTemplate="ts://data/{data_name}",
            name="Time Series Data",
            description="Exported time series data (CSV/JSON)",
            mimeType="text/csv"
        ),
        ResourceTemplate(
            uriTemplate="ts://reports/{report_name}",
            name="Forecast Report",
            description="Generated forecast report",
            mimeType="text/html"
        )
    ]

@server.list_tools()
async def list_tools() -> list[Tool]:
    """List all available time series tools."""
    return [
        # ============ Dataset Management Tools ============
        Tool(
            name="load_dataset",
            description="""Load a time series dataset from a file path.
            Supports CSV, Excel, JSON, and Parquet formats.
            Automatically detects date and target columns.
            
            Example: Load sales data from /path/to/sales.csv
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the data file"
                    },
                    "dataset_name": {
                        "type": "string",
                        "description": "Name to store the dataset under (for later reference)"
                    },
                    "date_column": {
                        "type": "string",
                        "description": "Name of the datetime column (auto-detected if not provided)"
                    },
                    "target_column": {
                        "type": "string",
                        "description": "Name of the target variable to forecast (auto-detected if not provided)"
                    },
                    "frequency": {
                        "type": "string",
                        "description": "Data frequency: 'D' (daily), 'W' (weekly), 'M' (monthly), 'H' (hourly), 'auto'",
                        "default": "auto"
                    },
                    "sample_size": {
                        "type": "integer",
                        "description": "If provided, sample the dataset to this size"
                    }
                },
                "required": ["file_path", "dataset_name"]
            }
        ),
        Tool(
            name="load_dataset_from_content",
            description="""Load a time series dataset from raw CSV or JSON content.
            Use this when the user uploads a file directly in the chat.
            
            Example: User uploads a CSV file, pass the content here.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Raw CSV or JSON content as a string"
                    },
                    "dataset_name": {
                        "type": "string",
                        "description": "Name to store the dataset under"
                    },
                    "content_type": {
                        "type": "string",
                        "enum": ["csv", "json"],
                        "description": "Type of content",
                        "default": "csv"
                    },
                    "date_column": {
                        "type": "string",
                        "description": "Name of the datetime column (auto-detected if not provided)"
                    },
                    "target_column": {
                        "type": "string",
                        "description": "Name of the target variable (auto-detected if not provided)"
                    },
                    "frequency": {
                        "type": "string",
                        "description": "Data frequency",
                        "default": "auto"
                    }
                },
                "required": ["content", "dataset_name"]
            }
        ),
        Tool(
            name="list_datasets",
            description="""List all currently loaded datasets with their basic info.""",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_dataset_info",
            description="""Get detailed information about a loaded dataset including statistics, 
            missing values, outliers, and a preview.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_name": {
                        "type": "string",
                        "description": "Name of the dataset"
                    }
                },
                "required": ["dataset_name"]
            }
        ),
        Tool(
            name="delete_dataset",
            description="""Delete a loaded dataset from memory.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_name": {
                        "type": "string",
                        "description": "Name of the dataset to delete"
                    }
                },
                "required": ["dataset_name"]
            }
        ),
        Tool(
            name="query_dataset",
            description="""Query a loaded dataset with optional filtering by date range or conditions.
            
            Example: Get sales > 100 between 2024-01-01 and 2024-06-01
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_name": {
                        "type": "string",
                        "description": "Name of the dataset"
                    },
                    "query": {
                        "type": "string",
                        "description": "Pandas query string (e.g., 'sales > 100')"
                    },
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Columns to return"
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Filter data after this date (YYYY-MM-DD)"
                    },
                    "end_date": {
                        "type": "string",
                        "description": "Filter data before this date (YYYY-MM-DD)"
                    }
                },
                "required": ["dataset_name"]
            }
        ),
        
        # ============ Analysis Tools ============
        Tool(
            name="analyze_time_series",
            description="""Perform comprehensive analysis on a loaded dataset.
            Includes stationarity tests, seasonality detection, trend analysis, 
            autocorrelation analysis, and model recommendations.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_name": {
                        "type": "string",
                        "description": "Name of the dataset to analyze"
                    },
                    "include_plots": {
                        "type": "boolean",
                        "description": "Generate analysis plots",
                        "default": True
                    }
                },
                "required": ["dataset_name"]
            }
        ),
        Tool(
            name="clean_dataset",
            description="""Clean and preprocess a time series dataset.
            Handles missing values, outliers, duplicates, and resampling.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_name": {
                        "type": "string",
                        "description": "Name of the dataset to clean"
                    },
                    "missing_strategy": {
                        "type": "string",
                        "enum": ["interpolate", "ffill", "bfill", "mean", "median", "drop"],
                        "description": "Strategy for handling missing values",
                        "default": "interpolate"
                    },
                    "outlier_strategy": {
                        "type": "string",
                        "enum": ["clip", "remove", "none"],
                        "description": "Strategy for handling outliers",
                        "default": "clip"
                    },
                    "outlier_threshold": {
                        "type": "number",
                        "description": "IQR multiplier for outlier detection",
                        "default": 1.5
                    }
                },
                "required": ["dataset_name"]
            }
        ),
        
        # ============ Forecasting Tools ============
        Tool(
            name="create_forecast",
            description="""Create a time series forecast for a loaded dataset.
            Automatically selects best model(s) based on data characteristics.
            Returns forecasts with confidence intervals.
            
            Example: "Create a 60-day forecast for my sales data"
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_name": {
                        "type": "string",
                        "description": "Name of the dataset to forecast"
                    },
                    "periods": {
                        "type": "integer",
                        "description": "Number of periods to forecast"
                    },
                    "frequency": {
                        "type": "string",
                        "description": "Forecast frequency (D, W, M, H). Uses data frequency if not specified."
                    },
                    "models": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Models to use: ['arima', 'sarima', 'prophet', 'ets', 'ensemble', 'auto']",
                        "default": ["auto"]
                    },
                    "confidence_level": {
                        "type": "number",
                        "description": "Confidence level for prediction intervals (0-1)",
                        "default": 0.95
                    }
                },
                "required": ["dataset_name", "periods"]
            }
        ),
        Tool(
            name="compare_models",
            description="""Compare multiple forecasting models on a dataset.
            Performs cross-validation and returns metrics for each model.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_name": {
                        "type": "string",
                        "description": "Name of the dataset"
                    },
                    "models": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Models to compare: ['arima', 'sarima', 'prophet', 'ets']"
                    },
                    "cv_folds": {
                        "type": "integer",
                        "description": "Number of cross-validation folds",
                        "default": 5
                    }
                },
                "required": ["dataset_name"]
            }
        ),
        Tool(
            name="quick_forecast",
            description="""One-shot forecasting: load data, clean, train models, and forecast in one call.
            Perfect for quick analysis like "Create a 2-month sales forecast from this data".
            
            Can accept either a file path OR raw content from an uploaded file.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the data file (use this OR content, not both)"
                    },
                    "content": {
                        "type": "string",
                        "description": "Raw CSV/JSON content (use this OR file_path, not both)"
                    },
                    "forecast_description": {
                        "type": "string",
                        "description": "Natural language description like '2 month forecast' or '30 day prediction'"
                    },
                    "date_column": {
                        "type": "string",
                        "description": "Name of date column (auto-detected if not provided)"
                    },
                    "target_column": {
                        "type": "string",
                        "description": "Name of target column (auto-detected if not provided)"
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Directory to save outputs",
                        "default": "./forecast_output"
                    }
                },
                "required": ["forecast_description"]
            }
        ),
        
        # ============ Export Tools ============
        Tool(
            name="export_forecast",
            description="""Export forecast results to files (CSV, PNG, HTML, JSON).""",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_name": {
                        "type": "string",
                        "description": "Name of the dataset with forecasts"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path for output files (without extension)"
                    },
                    "formats": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Output formats: ['csv', 'png', 'html', 'json']",
                        "default": ["csv", "png"]
                    }
                },
                "required": ["dataset_name", "output_path"]
            }
        ),
        Tool(
            name="create_plot",
            description="""Create a visualization plot and save it as a resource.
            The plot is automatically saved to disk and can be exported.
            
            Plot types:
            - 'forecast': Forecast plot with confidence intervals
            - 'overview': Time series with rolling mean
            - 'distribution': Histogram and box plot
            - 'acf': Autocorrelation and partial autocorrelation
            - 'decomposition': Seasonal decomposition
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_name": {
                        "type": "string",
                        "description": "Name of the dataset"
                    },
                    "plot_type": {
                        "type": "string",
                        "enum": ["forecast", "overview", "distribution", "acf", "decomposition"],
                        "description": "Type of plot to create",
                        "default": "forecast"
                    },
                    "plot_name": {
                        "type": "string",
                        "description": "Name for the plot resource (auto-generated if not provided)"
                    },
                    "title": {
                        "type": "string",
                        "description": "Custom plot title"
                    },
                    "style": {
                        "type": "string",
                        "enum": ["default", "seaborn", "dark", "minimal"],
                        "description": "Plot style",
                        "default": "default"
                    },
                    "save_format": {
                        "type": "string",
                        "enum": ["png", "svg", "pdf"],
                        "description": "Output format",
                        "default": "png"
                    },
                    "dpi": {
                        "type": "integer",
                        "description": "Resolution (dots per inch)",
                        "default": 150
                    }
                },
                "required": ["dataset_name"]
            }
        ),
        Tool(
            name="list_plots",
            description="""List all generated plot resources.
            Returns URIs that can be used to export or modify plots.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_name": {
                        "type": "string",
                        "description": "Filter by dataset name"
                    }
                }
            }
        ),
        Tool(
            name="export_plot",
            description="""Export a plot to a specific file path.
            Use this to save plots for PowerPoint, reports, etc.
            
            Example: Export forecast plot to ~/Desktop/sales_forecast.png
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "resource_uri": {
                        "type": "string",
                        "description": "URI of the plot resource (e.g., ts://plots/sales_forecast)"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Destination path (directory or full file path)"
                    },
                    "format": {
                        "type": "string",
                        "enum": ["png", "svg", "pdf", "jpg"],
                        "description": "Output format (uses original format if not specified)"
                    }
                },
                "required": ["resource_uri", "output_path"]
            }
        ),
        Tool(
            name="get_plot",
            description="""Get a plot resource for display.
            Returns the image data that can be shown in the conversation.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "resource_uri": {
                        "type": "string",
                        "description": "URI of the plot resource"
                    }
                },
                "required": ["resource_uri"]
            }
        ),
        Tool(
            name="customize_plot",
            description="""Customize an existing plot with new settings.
            Allows changing title, colors, style, etc.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "resource_uri": {
                        "type": "string",
                        "description": "URI of the plot to modify"
                    },
                    "title": {
                        "type": "string",
                        "description": "New plot title"
                    },
                    "style": {
                        "type": "string",
                        "enum": ["default", "seaborn", "dark", "minimal"],
                        "description": "New style"
                    },
                    "colors": {
                        "type": "object",
                        "description": "Color overrides: {historical: '#...', forecast: '#...', ci: '#...'}"
                    }
                },
                "required": ["resource_uri"]
            }
        ),
        Tool(
            name="delete_plot",
            description="""Delete a plot resource and optionally its file.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "resource_uri": {
                        "type": "string",
                        "description": "URI of the plot to delete"
                    },
                    "delete_file": {
                        "type": "boolean",
                        "description": "Also delete the file on disk",
                        "default": True
                    }
                },
                "required": ["resource_uri"]
            }
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent | ImageContent | EmbeddedResource]:
    """Handle tool calls."""
    
    try:
        if name == "load_dataset":
            return await handle_load_dataset(arguments)
        elif name == "load_dataset_from_content":
            return await handle_load_dataset_from_content(arguments)
        elif name == "list_datasets":
            return await handle_list_datasets(arguments)
        elif name == "get_dataset_info":
            return await handle_get_dataset_info(arguments)
        elif name == "delete_dataset":
            return await handle_delete_dataset(arguments)
        elif name == "query_dataset":
            return await handle_query_dataset(arguments)
        elif name == "analyze_time_series":
            return await handle_analyze_time_series(arguments)
        elif name == "clean_dataset":
            return await handle_clean_dataset(arguments)
        elif name == "create_forecast":
            return await handle_create_forecast(arguments)
        elif name == "compare_models":
            return await handle_compare_models(arguments)
        elif name == "quick_forecast":
            return await handle_quick_forecast(arguments)
        elif name == "export_forecast":
            return await handle_export_forecast(arguments)
        elif name == "create_plot":
            return await handle_create_plot(arguments)
        elif name == "list_plots":
            return await handle_list_plots(arguments)
        elif name == "export_plot":
            return await handle_export_plot(arguments)
        elif name == "get_plot":
            return await handle_get_plot(arguments)
        elif name == "customize_plot":
            return await handle_customize_plot(arguments)
        elif name == "delete_plot":
            return await handle_delete_plot(arguments)
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    except Exception as e:
        logger.error(f"Error in tool {name}: {str(e)}")
        import traceback
        return [TextContent(type="text", text=f"Error: {str(e)}\n\n{traceback.format_exc()}")]


# ============ Handler Functions ============
async def handle_create_plot(arguments: dict) -> list[TextContent | ImageContent]:
    """Handle create_plot tool call."""
    dataset_name = arguments["dataset_name"]
    plot_type = arguments.get("plot_type", "forecast")
    
    if plot_type == "forecast":
        result = await create_forecast_plot(
            dataset_name=dataset_name,
            plot_name=arguments.get("plot_name"),
            title=arguments.get("title"),
            style=arguments.get("style", "default"),
            save_format=arguments.get("save_format", "png"),
            dpi=arguments.get("dpi", 150)
        )
    else:
        result = await create_analysis_plot(
            dataset_name=dataset_name,
            plot_type=plot_type,
            plot_name=arguments.get("plot_name"),
            save_format=arguments.get("save_format", "png"),
            dpi=arguments.get("dpi", 150)
        )
    
    if result["status"] == "error":
        return [TextContent(type="text", text=f"❌ {result['message']}")]
    
    # Get the resource to display
    resource = resource_manager.get_resource(result["resource_uri"])
    
    parts = []
    
    summary = f"""
## 📊 Plot Created

| Property | Value |
|----------|-------|
| **Resource URI** | `{result['resource_uri']}` |
| **File Path** | `{result['file_path']}` |
| **Type** | {plot_type} |

Use `export_plot` to save to a different location, or `customize_plot` to modify.
"""
    parts.append(TextContent(type="text", text=summary))
    
    # Include the image
    if resource and resource.base64_data:
        parts.append(ImageContent(
            type="image",
            data=resource.base64_data,
            mimeType=resource.mime_type
        ))
    
    return parts


async def handle_list_plots(arguments: dict) -> list[TextContent]:
    """Handle list_plots tool call."""
    result = await list_plot_resources(
        dataset_name=arguments.get("dataset_name")
    )
    
    if result["count"] == 0:
        return [TextContent(type="text", text="📭 No plots generated yet. Use `create_plot` to generate visualizations.")]
    
    summary = f"## 📊 Available Plots ({result['count']})\n\n"
    summary += "| URI | Name | Type | File | Saved |\n"
    summary += "|-----|------|------|------|-------|\n"
    
    for r in result["resources"]:
        saved = "✅" if r["is_saved"] else "❌"
        file_path = r.get("file_path", "N/A")
        if file_path and len(file_path) > 40:
            file_path = "..." + file_path[-37:]
        summary += f"| `{r['uri']}` | {r['name']} | {r['metadata'].get('plot_type', 'N/A')} | `{file_path}` | {saved} |\n"
    
    return [TextContent(type="text", text=summary)]


async def handle_export_plot(arguments: dict) -> list[TextContent]:
    """Handle export_plot tool call."""
    result = await export_plot(
        resource_uri=arguments["resource_uri"],
        output_path=arguments["output_path"],
        format=arguments.get("format")
    )
    
    if result["status"] == "error":
        return [TextContent(type="text", text=f"❌ {result['message']}")]
    
    summary = f"""
## ✅ Plot Exported

**File saved to:** `{result['file_path']}`

You can now use this file in PowerPoint, reports, or any other application.
"""
    
    return [TextContent(type="text", text=summary)]


async def handle_get_plot(arguments: dict) -> list[TextContent | ImageContent]:
    """Handle get_plot tool call."""
    result = await get_plot_for_display(arguments["resource_uri"])
    
    if result["status"] == "error":
        return [TextContent(type="text", text=f"❌ {result['message']}")]
    
    parts = [
        TextContent(type="text", text=f"**Plot:** `{result['name']}` (`{result['uri']}`)\n**File:** `{result['file_path']}`")
    ]
    
    if result.get("base64_data"):
        parts.append(ImageContent(
            type="image",
            data=result["base64_data"],
            mimeType=result["mime_type"]
        ))
    
    return parts


async def handle_customize_plot(arguments: dict) -> list[TextContent | ImageContent]:
    """Handle customize_plot tool call."""
    modifications = {
        k: v for k, v in arguments.items() 
        if k != "resource_uri" and v is not None
    }
    
    result = await modify_plot(
        resource_uri=arguments["resource_uri"],
        modifications=modifications
    )
    
    if result["status"] == "error":
        return [TextContent(type="text", text=f"❌ {result['message']}")]
    
    # Get updated resource
    resource = resource_manager.get_resource(result["resource_uri"])
    
    parts = [TextContent(type="text", text=f"✅ Plot updated: `{result['resource_uri']}`")]
    
    if resource and resource.base64_data:
        parts.append(ImageContent(
            type="image",
            data=resource.base64_data,
            mimeType=resource.mime_type
        ))
    
    return parts


async def handle_delete_plot(arguments: dict) -> list[TextContent]:
    """Handle delete_plot tool call."""
    result = await delete_plot(
        resource_uri=arguments["resource_uri"],
        delete_file=arguments.get("delete_file", True)
    )
    
    if result["status"] == "error":
        return [TextContent(type="text", text=f"❌ {result['message']}")]
    
    return [TextContent(type="text", text=f"✅ {result['message']}")]



async def handle_load_dataset(arguments: dict) -> list[TextContent]:
    """Handle load_dataset tool call."""
    result = await load_dataset(
        file_path=arguments["file_path"],
        dataset_name=arguments["dataset_name"],
        date_column=arguments.get("date_column"),
        target_column=arguments.get("target_column"),
        frequency=arguments.get("frequency", "auto"),
        sample_size=arguments.get("sample_size")
    )

    if result["status"] == "error":
        return [TextContent(type="text", text=f"❌ {result['message']}")]

    # --- CREATE DATA RESOURCE ---
    # Register the loaded dataset as a resource so it can be accessed via ts://data/{name}
    dataset_name = result['dataset_name']
    df = DatasetManager.get_dataset(dataset_name)

    if df is not None:
        resource = resource_manager.create_data_resource(
            name=dataset_name,
            data=df,
            description=f"Loaded time series dataset: {dataset_name}",
            data_type="csv",
            dataset_name=dataset_name,
            auto_save=True
        )
        resource_uri = resource.uri
    else:
        resource_uri = None

    summary = f"""
## ✅ Dataset Loaded: `{result['dataset_name']}`

| Property | Value |
|----------|-------|
| **Rows** | {result['rows']} |
| **Date Column** | `{result['date_column']}` |
| **Target Column** | `{result['target_column']}` |
| **Frequency** | {result['frequency']} |
| **Date Range** | {result['date_range']} |
| **Missing Values** | {result['missing_values']} ({result['missing_pct']:.1f}%) |
| **Outliers** | {result['outliers']} |
"""

    if resource_uri:
        summary += f"| **Resource URI** | `{resource_uri}` |\n"

    summary += f"""
### Preview
{result['preview']}


"""

    if result.get("sampled"):
        summary += f"\n⚠️ **Sampled** from {result['original_rows']} to {result['rows']} rows"

    if resource_uri:
        summary += f"\n\n💡 **Tip:** You can now access this dataset via the resource `{resource_uri}` or use it in forecasting tools."

    return [TextContent(type="text", text=summary)]


async def handle_load_dataset_from_content(arguments: dict) -> list[TextContent]:
    """Handle load_dataset_from_content tool call."""
    result = await load_dataset_from_content(
        content=arguments["content"],
        dataset_name=arguments["dataset_name"],
        content_type=arguments.get("content_type", "csv"),
        date_column=arguments.get("date_column"),
        target_column=arguments.get("target_column"),
        frequency=arguments.get("frequency", "auto"),
        sample_size=arguments.get("sample_size")
    )

    if result["status"] == "error":
        return [TextContent(type="text", text=f"❌ {result['message']}")]

    # --- CREATE DATA RESOURCE ---
    # Register the loaded dataset as a resource so it can be accessed via ts://data/{name}
    dataset_name = result['dataset_name']
    df = DatasetManager.get_dataset(dataset_name)

    if df is not None:
        resource = resource_manager.create_data_resource(
            name=dataset_name,
            data=df,
            description=f"Loaded time series dataset from content: {dataset_name}",
            data_type="csv",
            dataset_name=dataset_name,
            auto_save=True
        )
        resource_uri = resource.uri
    else:
        resource_uri = None

    summary = f"""
## ✅ Dataset Loaded from Content: `{result['dataset_name']}`

| Property | Value |
|----------|-------|
| **Rows** | {result['rows']} |
| **Date Column** | `{result['date_column']}` |
| **Target Column** | `{result['target_column']}` |
| **Frequency** | {result['frequency']} |
| **Date Range** | {result['date_range']} |
| **Missing Values** | {result['missing_values']} ({result['missing_pct']:.1f}%) |
"""

    if resource_uri:
        summary += f"| **Resource URI** | `{resource_uri}` |\n"

    summary += f"""
### Preview
{result['preview']}


"""

    if resource_uri:
        summary += f"\n💡 **Tip:** You can now access this dataset via the resource `{resource_uri}`.\n"

    summary += "\n**Ready for analysis!** Use `analyze_time_series` or `create_forecast`."

    return [TextContent(type="text", text=summary)]


async def handle_list_datasets(arguments: dict) -> list[TextContent]:
    """Handle list_datasets tool call."""
    result = await list_datasets()
    
    if result["count"] == 0:
        return [TextContent(type="text", text="📭 No datasets loaded. Use `load_dataset` or `load_dataset_from_content` to load data.")]
    
    summary = f"## 📊 Loaded Datasets ({result['count']})\n\n"
    summary += "| Name | Rows | Date Range | Target | Frequency |\n"
    summary += "|------|------|------------|--------|----------|\n"
    
    for ds in result["datasets"]:
        summary += f"| `{ds['name']}` | {ds['rows']} | {ds['date_range']} | `{ds['target']}` | {ds['frequency']} |\n"
    
    return [TextContent(type="text", text=summary)]


async def handle_get_dataset_info(arguments: dict) -> list[TextContent]:
    """Handle get_dataset_info tool call."""
    result = await get_dataset_info(arguments["dataset_name"])
    
    if result["status"] == "error":
        return [TextContent(type="text", text=f"❌ {result['message']}")]
    
    stats = result["statistics"]
    summary = f"""
## 📊 Dataset: `{result['dataset_name']}`

### Overview
| Property | Value |
|----------|-------|
| **Rows** | {result['rows']} |
| **Columns** | {', '.join(f'`{c}`' for c in result['columns'])} |
| **Date Column** | `{result['date_column']}` |
| **Target Column** | `{result['target_column']}` |
| **Frequency** | {result['frequency']} |
| **Date Range** | {result['date_range']} |

### Data Quality
| Metric | Value |
|--------|-------|
| **Missing Values** | {result['missing_values']} ({result['missing_pct']}) |
| **Outliers** | {result['outliers']} |
| **Duplicates** | {result['duplicates']} |

### Statistics
| Metric | Value |
|--------|-------|
| **Mean** | {stats['mean']:.2f} |
| **Std Dev** | {stats['std']:.2f} |
| **Min** | {stats['min']:.2f} |
| **Max** | {stats['max']:.2f} |
| **Median** | {stats['median']:.2f} |

### Preview
{result['preview']}


"""
    
    return [TextContent(type="text", text=summary)]


async def handle_delete_dataset(arguments: dict) -> list[TextContent]:
    """Handle delete_dataset tool call."""
    result = await delete_dataset(arguments["dataset_name"])
    
    if result["status"] == "error":
        return [TextContent(type="text", text=f"❌ {result['message']}")]
    
    return [TextContent(type="text", text=f"✅ {result['message']}")]


async def handle_query_dataset(arguments: dict) -> list[TextContent]:
    """Handle query_dataset tool call."""
    result = await query_dataset(
        dataset_name=arguments["dataset_name"],
        query=arguments.get("query"),
        columns=arguments.get("columns"),
        start_date=arguments.get("start_date"),
        end_date=arguments.get("end_date")
    )
    
    if result["status"] == "error":
        return [TextContent(type="text", text=f"❌ {result['message']}")]
    
    import pandas as pd
    df = pd.DataFrame(result["data"])
    
    summary = f"""
## 🔍 Query Results: `{result['dataset_name']}`

**Rows returned:** {result['rows_returned']}
**Columns:** {', '.join(f'`{c}`' for c in result['columns'])}
{df.to_string()}


"""
    
    if result.get("truncated"):
        summary += "\n⚠️ Results truncated to 100 rows"
    
    return [TextContent(type="text", text=summary)]


async def handle_analyze_time_series(arguments: dict) -> list[TextContent | ImageContent]:
    """Handle analyze_time_series tool call."""
    dataset_name = arguments["dataset_name"]
    include_plots = arguments.get("include_plots", True)
    
    df = DatasetManager.get_dataset(dataset_name)
    schema = DatasetManager.get_schema(dataset_name)
    
    if df is None or schema is None:
        return [TextContent(type="text", text=f"❌ Dataset '{dataset_name}' not found.")]
    
    metadata = {
        "date_column": schema.date_column,
        "target_column": schema.target_column,
        "frequency": schema.frequency
    }
    
    analysis = data_cleaner.analyze(data=df, metadata=metadata)
    
    result_parts = []
    
    summary = f"""
## 📊 Time Series Analysis: `{dataset_name}`

### 1. Stationarity Tests
| Test | Statistic | p-value | Stationary? |
|------|-----------|---------|-------------|
| ADF  | {analysis['adf_stat']:.4f} | {analysis['adf_pvalue']:.4f} | {'✅ Yes' if analysis['adf_pvalue'] < 0.05 else '❌ No'} |
| KPSS | {analysis['kpss_stat']:.4f} | {analysis['kpss_pvalue']:.4f} | {'✅ Yes' if analysis['kpss_pvalue'] > 0.05 else '❌ No'} |

### 2. Seasonality
- **Detected:** {'✅ Yes' if analysis['has_seasonality'] else '❌ No'}
- **Period:** {analysis['seasonal_period'] if analysis['has_seasonality'] else 'N/A'}
- **Strength:** {analysis['seasonality_strength']:.2%}

### 3. Trend
- **Direction:** {analysis['trend_direction']}
- **Strength:** {analysis['trend_strength']:.2%}

### 4. Recommended Models
"""
    
    for i, model in enumerate(analysis['recommended_models'], 1):
        summary += f"\n{i}. **{model['name']}** - {model['reason']}"
    
    result_parts.append(TextContent(type="text", text=summary))
    
    if include_plots:
        plots = visualizer.create_analysis_plots(data=df, metadata=metadata, analysis=analysis)
        for plot_name, plot_data in plots.items():
            result_parts.append(ImageContent(type="image", data=plot_data, mimeType="image/png"))
    
    return result_parts


async def handle_clean_dataset(arguments: dict) -> list[TextContent]:
    """Handle clean_dataset tool call."""
    dataset_name = arguments["dataset_name"]
    
    df = DatasetManager.get_dataset(dataset_name)
    schema = DatasetManager.get_schema(dataset_name)
    
    if df is None or schema is None:
        return [TextContent(type="text", text=f"❌ Dataset '{dataset_name}' not found.")]
    
    metadata = {
        "date_column": schema.date_column,
        "target_column": schema.target_column,
        "frequency": schema.frequency
    }
    
    result = data_cleaner.clean(
        data=df,
        metadata=metadata,
        missing_strategy=arguments.get("missing_strategy", "interpolate"),
        outlier_strategy=arguments.get("outlier_strategy", "clip"),
        outlier_threshold=arguments.get("outlier_threshold", 1.5)
    )
    
    # Update the stored dataset
    loaded_datasets[dataset_name] = result["data"]
    schema.row_count = len(result["data"])
    schema.missing_count = result["remaining_missing"]
    schema.missing_pct = (result["remaining_missing"] / len(result["data"])) * 100
    
    summary = f"""
## 🧹 Dataset Cleaned: `{dataset_name}`

### Actions Performed
| Action | Count |
|--------|-------|
| **Missing Values Filled** | {result['missing_filled']} |
| **Outliers Handled** | {result['outliers_handled']} |
| **Duplicates Removed** | {result['duplicates_removed']} |

### Result
- **Total Rows:** {len(result['data'])}
- **Remaining Missing:** {result['remaining_missing']}
- **Ready for Modeling:** {'✅ Yes' if result['ready_for_modeling'] else '❌ No'}

**Dataset updated in memory.** Ready for forecasting!
"""
    
    return [TextContent(type="text", text=summary)]


async def handle_create_forecast(arguments: dict) -> list[TextContent | ImageContent]:
    """Handle create_forecast tool call."""
    dataset_name = arguments["dataset_name"]
    periods = arguments["periods"]
    
    df = DatasetManager.get_dataset(dataset_name)
    schema = DatasetManager.get_schema(dataset_name)
    
    if df is None or schema is None:
        return [TextContent(type="text", text=f"❌ Dataset '{dataset_name}' not found.")]
    
    metadata = {
        "date_column": schema.date_column,
        "target_column": schema.target_column,
        "frequency": schema.frequency
    }
    
    frequency = arguments.get("frequency", schema.frequency)
    models = arguments.get("models", ["auto"])
    confidence_level = arguments.get("confidence_level", 0.95)
    
    # Auto-select models if needed
    if "auto" in models:
        models = model_selector.select_models(data=df, metadata=metadata)
    
    # Generate forecasts
    result = forecaster.forecast(
        data=df,
        metadata=metadata,
        periods=periods,
        frequency=frequency,
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
            frequency=frequency
        )

    # --- CREATE FORECAST DATA RESOURCE ---
    # Register the best forecast as a resource so it can be accessed via ts://data/{name}_forecast
    best_model = result["best_model"]
    best_forecast_df = result["forecasts"][best_model]
    forecast_resource_name = f"{dataset_name}_forecast_{best_model}"

    forecast_resource = resource_manager.create_data_resource(
        name=forecast_resource_name,
        data=best_forecast_df,
        description=f"Forecast results for {dataset_name} using {best_model} model",
        data_type="csv",
        dataset_name=dataset_name,
        auto_save=True
    )
    forecast_resource_uri = forecast_resource.uri

    result_parts = []

    summary = f"""
## 🔮 Forecast Generated: `{dataset_name}`

### Model Performance
| Model | MAE | RMSE | MAPE | Status |
|-------|-----|------|------|--------|
"""
    
    for model_name, metrics in result["metrics"].items():
        status = "🏆 Best" if model_name == result["best_model"] else "✓"
        summary += f"| {model_name} | {metrics['mae']:.2f} | {metrics['rmse']:.2f} | {metrics['mape']:.2%} | {status} |\n"
    
    summary += f"""
### Best Model: **{result['best_model']}**

**Forecast Resource:** `{forecast_resource_uri}`

### Forecast Preview ({periods} periods)
{result['forecasts'][result['best_model']].head(15).to_string()}



### Insights
"""

    for insight in result["insights"]:
        summary += f"- {insight}\n"

    summary += f"\n\n💡 **Tip:** Access the full forecast data via resource `{forecast_resource_uri}`"

    result_parts.append(TextContent(type="text", text=summary))
    
    # Add forecast plot
    forecast_plot = visualizer.create_forecast_plot(
        historical_data=df,
        forecasts=result["forecasts"],
        best_model=result["best_model"],
        metadata=metadata
    )
    result_parts.append(ImageContent(type="image", data=forecast_plot, mimeType="image/png"))
    
    return result_parts


async def handle_compare_models(arguments: dict) -> list[TextContent | ImageContent]:
    """Handle compare_models tool call."""
    dataset_name = arguments["dataset_name"]
    
    df = DatasetManager.get_dataset(dataset_name)
    schema = DatasetManager.get_schema(dataset_name)
    
    if df is None or schema is None:
        return [TextContent(type="text", text=f"❌ Dataset '{dataset_name}' not found.")]
    
    metadata = {
        "date_column": schema.date_column,
        "target_column": schema.target_column,
        "frequency": schema.frequency
    }
    
    models = arguments.get("models", ["arima", "prophet", "ets"])
    cv_folds = arguments.get("cv_folds", 5)
    
    comparison = model_selector.compare_models(
        data=df,
        metadata=metadata,
        models=models,
        cv_folds=cv_folds
    )
    
    result_parts = []
    
    summary = f"""
## 📈 Model Comparison: `{dataset_name}`

### Cross-Validation Results ({cv_folds} folds)
| Model | MAE | RMSE | MAPE | Time |
|-------|-----|------|------|------|
"""
    
    for model_name, metrics in comparison["cv_results"].items():
        summary += f"| {model_name} | {metrics['mae']:.2f} ± {metrics['mae_std']:.2f} | {metrics['rmse']:.2f} ± {metrics['rmse_std']:.2f} | {metrics['mape']:.2%} | {metrics['train_time']:.1f}s |\n"
    
    summary += f"""
### Recommendation
**Best Model:** {comparison['best_model']}

{comparison['recommendation']}
"""
    
    result_parts.append(TextContent(type="text", text=summary))
    
    # Add comparison plot
    comparison_plot = visualizer.create_comparison_plot(comparison)
    result_parts.append(ImageContent(type="image", data=comparison_plot, mimeType="image/png"))
    
    return result_parts


async def handle_quick_forecast(arguments: dict) -> list[TextContent | ImageContent]:
    """Handle quick_forecast tool call - one-shot forecasting."""
    file_path = arguments.get("file_path")
    content = arguments.get("content")
    forecast_description = arguments["forecast_description"]
    date_column = arguments.get("date_column")
    target_column = arguments.get("target_column")
    output_dir = arguments.get("output_dir", "./forecast_output")
    
    if not file_path and not content:
        return [TextContent(type="text", text="❌ Please provide either 'file_path' or 'content'.")]
    
    result_parts = []
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
    
    if load_result["status"] == "error":
        return [TextContent(type="text", text=f"❌ Failed to load data: {load_result['message']}")]
    
    result_parts.append(TextContent(type="text", text=f"✅ Data loaded: {load_result['rows']} rows"))
    
    # Step 2: Clean data
    clean_result = await handle_clean_dataset({
        "dataset_name": dataset_name,
        "missing_strategy": "interpolate",
        "outlier_strategy": "clip"
    })
    
    # Step 3: Parse forecast description
    periods, frequency = parse_forecast_description(forecast_description)
    
    # Step 4: Create forecast
    forecast_result = await handle_create_forecast({
        "dataset_name": dataset_name,
        "periods": periods,
        "frequency": frequency,
        "models": ["auto"]
    })
    result_parts.extend(forecast_result)
    
    # Step 5: Export
    export_result = await handle_export_forecast({
        "dataset_name": dataset_name,
        "output_path": f"{output_dir}/forecast",
        "formats": ["csv", "png"]
    })
    result_parts.extend(export_result)
    
    return result_parts


async def handle_export_forecast(arguments: dict) -> list[TextContent]:
    """Handle export_forecast tool call."""
    dataset_name = arguments["dataset_name"]
    output_path = arguments["output_path"]
    formats = arguments.get("formats", ["csv", "png"])
    
    df = DatasetManager.get_dataset(dataset_name)
    schema = DatasetManager.get_schema(dataset_name)
    forecasts_dict = DatasetManager.get_forecast(dataset_name)
    
    if df is None or schema is None:
        return [TextContent(type="text", text=f"❌ Dataset '{dataset_name}' not found.")]
    
    if not forecasts_dict:
        return [TextContent(type="text", text=f"❌ No forecasts found for '{dataset_name}'. Run `create_forecast` first.")]
    
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
    
    summary = f"""
## 📤 Forecast Exported: `{dataset_name}`

### Files Created
"""
    
    for file_path, description in files_created.items():
        summary += f"- `{file_path}` - {description}\n"
    
    return [TextContent(type="text", text=summary)]


def parse_forecast_description(description: str) -> tuple[int, str]:
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


def main():
    """Main entry point for the MCP server."""
    async def run():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options()
            )
    
    asyncio.run(run())


if __name__ == "__main__":
    main()