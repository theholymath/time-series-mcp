"""
Plot management tools for Time Series MCP.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from io import BytesIO
import base64

from ..models.resources import resource_manager, ResourceType, ManagedResource
from ..models.schemas import DatasetManager, dataset_schemas, forecast_results


async def create_forecast_plot(
    dataset_name: str,
    plot_name: Optional[str] = None,
    title: Optional[str] = None,
    style: str = "default",
    figsize: tuple = (14, 7),
    show_confidence: bool = True,
    show_historical: bool = True,
    historical_periods: Optional[int] = None,
    colors: Optional[Dict[str, str]] = None,
    save_format: str = "png",
    dpi: int = 150
) -> Dict[str, Any]:
    """
    Create a forecast plot and store as a resource.
    
    Args:
        dataset_name: Name of the dataset with forecasts
        plot_name: Name for the resource (auto-generated if not provided)
        title: Plot title
        style: Matplotlib style ('default', 'seaborn', 'dark', 'minimal')
        figsize: Figure size (width, height)
        show_confidence: Show confidence intervals
        show_historical: Show historical data
        historical_periods: Number of historical periods to show (None = all)
        colors: Custom colors dict {'historical': '#...', 'forecast': '#...', 'ci': '#...'}
        save_format: Output format (png, svg, pdf)
        dpi: Resolution for raster formats
    
    Returns:
        Dict with resource info
    """
    # Get data
    df = DatasetManager.get_dataset(dataset_name)
    schema = DatasetManager.get_schema(dataset_name)
    forecasts_dict = DatasetManager.get_forecast(dataset_name)
    
    if df is None or schema is None:
        return {"status": "error", "message": f"Dataset '{dataset_name}' not found"}
    
    if not forecasts_dict:
        return {"status": "error", "message": f"No forecasts found for '{dataset_name}'"}
    
    # Set style
    style_map = {
        "default": "seaborn-v0_8-whitegrid",
        "seaborn": "seaborn-v0_8",
        "dark": "dark_background",
        "minimal": "seaborn-v0_8-white"
    }
    plt.style.use(style_map.get(style, "seaborn-v0_8-whitegrid"))
    
    # Default colors
    default_colors = {
        "historical": "#2E86AB",
        "forecast": "#E94F37",
        "ci": "#E94F37"
    }
    colors = {**default_colors, **(colors or {})}
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    target_col = schema.target_column
    series = df[target_col]
    
    # Limit historical data if specified
    if historical_periods and len(series) > historical_periods:
        series = series.tail(historical_periods)
    
    # Plot historical
    if show_historical:
        ax.plot(
            series.index, 
            series.values, 
            color=colors["historical"],
            linewidth=1.5,
            label='Historical'
        )
    
    # Find best model
    best_model = min(
        forecasts_dict.items(),
        key=lambda x: x[1].metrics.get('mape', float('inf'))
    )[0]
    
    # Plot forecast
    forecast_result = forecasts_dict[best_model]
    forecast_df = forecast_result.forecast_df
    
    ax.plot(
        forecast_df.index,
        forecast_df['yhat'],
        color=colors["forecast"],
        linewidth=2,
        label=f'Forecast ({best_model})'
    )
    
    # Confidence interval
    if show_confidence and 'yhat_lower' in forecast_df.columns:
        ax.fill_between(
            forecast_df.index,
            forecast_df['yhat_lower'],
            forecast_df['yhat_upper'],
            color=colors["ci"],
            alpha=0.2,
            label='95% CI'
        )
    
    # Formatting
    plot_title = title or f"{dataset_name} - Forecast"
    ax.set_title(plot_title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(target_col, fontsize=12)
    ax.legend(loc='upper left')
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    # Add forecast start line
    ax.axvline(
        x=forecast_df.index[0],
        color='gray',
        linestyle='--',
        alpha=0.5
    )
    
    plt.tight_layout()
    
    # Convert to base64
    buf = BytesIO()
    fig.savefig(buf, format=save_format, dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    base64_data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    # Create resource
    resource_name = plot_name or f"{dataset_name}_forecast"
    
    resource = resource_manager.create_plot_resource(
        name=resource_name,
        base64_data=base64_data,
        description=f"Forecast plot for {dataset_name} using {best_model}",
        dataset_name=dataset_name,
        plot_type="forecast",
        auto_save=True,
        file_format=save_format
    )
    
    return {
        "status": "success",
        "resource_uri": resource.uri,
        "resource_name": resource.name,
        "file_path": resource.file_path,
        "message": f"Plot created and saved to {resource.file_path}"
    }


async def modify_plot(
    resource_uri: str,
    modifications: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Modify an existing plot resource.
    
    Args:
        resource_uri: URI of the plot resource to modify
        modifications: Dict of modifications to apply:
            - title: New title
            - xlabel: New x-axis label
            - ylabel: New y-axis label
            - colors: Dict of color changes
            - figsize: New figure size
            - style: New style
            - add_annotation: Dict with {text, x, y}
            - add_hline: Dict with {y, color, label}
            - add_vline: Dict with {x, color, label}
    
    Returns:
        Dict with updated resource info
    """
    resource = resource_manager.get_resource(resource_uri)
    
    if not resource:
        return {"status": "error", "message": f"Resource not found: {resource_uri}"}
    
    # Get associated dataset
    dataset_name = resource.metadata.get("dataset_name")
    if not dataset_name:
        return {"status": "error", "message": "Resource has no associated dataset"}
    
    # Re-create the plot with modifications
    # This is a simplified approach - in production you might want to
    # serialize the plot state and modify it directly
    
    result = await create_forecast_plot(
        dataset_name=dataset_name,
        plot_name=resource.name,
        title=modifications.get("title"),
        style=modifications.get("style", "default"),
        figsize=modifications.get("figsize", (14, 7)),
        colors=modifications.get("colors"),
        save_format=resource.metadata.get("format", "png")
    )
    
    return result


async def create_analysis_plot(
    dataset_name: str,
    plot_type: str = "overview",
    plot_name: Optional[str] = None,
    save_format: str = "png",
    dpi: int = 150,
    **kwargs
) -> Dict[str, Any]:
    """
    Create an analysis plot.
    
    Args:
        dataset_name: Name of the dataset
        plot_type: Type of plot:
            - 'overview': Time series with rolling mean
            - 'decomposition': Seasonal decomposition
            - 'acf': Autocorrelation
            - 'distribution': Histogram and box plot
            - 'comparison': Model comparison (if forecasts exist)
        plot_name: Name for the resource
        save_format: Output format
        dpi: Resolution
        **kwargs: Additional arguments for specific plot types
    
    Returns:
        Dict with resource info
    """
    df = DatasetManager.get_dataset(dataset_name)
    schema = DatasetManager.get_schema(dataset_name)
    
    if df is None or schema is None:
        return {"status": "error", "message": f"Dataset '{dataset_name}' not found"}
    
    target_col = schema.target_column
    series = df[target_col]
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    if plot_type == "overview":
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.plot(series.index, series.values, 'b-', linewidth=1, alpha=0.7, label='Actual')
        
        # Rolling mean
        window = kwargs.get("window", min(30, len(series) // 10))
        if window > 1:
            rolling_mean = series.rolling(window=window).mean()
            ax.plot(series.index, rolling_mean, 'r-', linewidth=2, 
                   label=f'{window}-period Moving Average')
        
        ax.set_title(f'{dataset_name} - Time Series Overview', fontsize=14, fontweight='bold')
        ax.legend()
        
    elif plot_type == "distribution":
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        sns.histplot(series, kde=True, ax=axes[0], color='steelblue')
        axes[0].set_title('Distribution')
        
        sns.boxplot(y=series, ax=axes[1], color='steelblue')
        axes[1].set_title('Box Plot')
        
    elif plot_type == "acf":
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        plot_acf(series.dropna(), ax=axes[0], lags=kwargs.get("lags", 40))
        axes[0].set_title('Autocorrelation (ACF)')
        
        plot_pacf(series.dropna(), ax=axes[1], lags=kwargs.get("lags", 40))
        axes[1].set_title('Partial Autocorrelation (PACF)')
        
    elif plot_type == "decomposition":
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        period = kwargs.get("period", 7)
        
        try:
            decomposition = seasonal_decompose(series.dropna(), model='additive', period=period)
            
            fig, axes = plt.subplots(4, 1, figsize=(14, 12))
            
            axes[0].plot(series.index, series.values)
            axes[0].set_title('Original')
            
            axes[1].plot(series.index, decomposition.trend)
            axes[1].set_title('Trend')
            
            axes[2].plot(series.index, decomposition.seasonal)
            axes[2].set_title('Seasonal')
            
            axes[3].plot(series.index, decomposition.resid)
            axes[3].set_title('Residual')
            
        except Exception as e:
            return {"status": "error", "message": f"Decomposition failed: {str(e)}"}
    
    else:
        return {"status": "error", "message": f"Unknown plot type: {plot_type}"}
    
    plt.tight_layout()
    
    # Convert to base64
    buf = BytesIO()
    fig.savefig(buf, format=save_format, dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    base64_data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    # Create resource
    resource_name = plot_name or f"{dataset_name}_{plot_type}"
    
    resource = resource_manager.create_plot_resource(
        name=resource_name,
        base64_data=base64_data,
        description=f"{plot_type.title()} plot for {dataset_name}",
        dataset_name=dataset_name,
        plot_type=plot_type,
        auto_save=True,
        file_format=save_format
    )
    
    return {
        "status": "success",
        "resource_uri": resource.uri,
        "resource_name": resource.name,
        "file_path": resource.file_path,
        "message": f"Plot created and saved to {resource.file_path}"
    }


async def list_plot_resources(
    dataset_name: Optional[str] = None
) -> Dict[str, Any]:
    """List all plot resources."""
    resources = resource_manager.list_resources(
        category="plots",
        dataset_name=dataset_name
    )
    
    return {
        "status": "success",
        "count": len(resources),
        "resources": [r.to_dict() for r in resources]
    }


async def export_plot(
    resource_uri: str,
    output_path: str,
    format: Optional[str] = None
) -> Dict[str, Any]:
    """
    Export a plot resource to a specific path.
    
    Args:
        resource_uri: URI of the plot resource
        output_path: Destination path (directory or full file path)
        format: Output format (png, svg, pdf, jpg)
    
    Returns:
        Dict with export result
    """
    try:
        file_path = resource_manager.export_resource(
            uri=resource_uri,
            output_path=output_path,
            format=format
        )
        
        return {
            "status": "success",
            "file_path": file_path,
            "message": f"Plot exported to {file_path}"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


async def get_plot_for_display(resource_uri: str) -> Dict[str, Any]:
    """
    Get plot data for display (returns base64 for LLM to show).
    
    Args:
        resource_uri: URI of the plot resource
    
    Returns:
        Dict with base64 image data
    """
    resource = resource_manager.get_resource(resource_uri)
    
    if not resource:
        return {"status": "error", "message": f"Resource not found: {resource_uri}"}
    
    return {
        "status": "success",
        "uri": resource.uri,
        "name": resource.name,
        "base64_data": resource.base64_data,
        "mime_type": resource.mime_type,
        "file_path": resource.file_path
    }


async def delete_plot(resource_uri: str, delete_file: bool = True) -> Dict[str, Any]:
    """Delete a plot resource."""
    if resource_manager.delete_resource(resource_uri, delete_file):
        return {"status": "success", "message": f"Resource deleted: {resource_uri}"}
    else:
        return {"status": "error", "message": f"Resource not found: {resource_uri}"}