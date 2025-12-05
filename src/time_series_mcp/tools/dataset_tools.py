"""
Dataset loading and management tools for Time Series MCP.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from ..models.schemas import DatasetManager, loaded_datasets, dataset_schemas, forecast_results


async def load_dataset(
    file_path: str,
    dataset_name: str,
    date_column: Optional[str] = None,
    target_column: Optional[str] = None,
    frequency: str = "auto",
    sample_size: Optional[int] = None
) -> Dict[str, Any]:
    """
    Load a time series dataset from a file path.
    
    Args:
        file_path: Path to the data file (CSV, Excel, JSON, Parquet)
        dataset_name: Name to store the dataset under
        date_column: Name of datetime column (auto-detected if None)
        target_column: Name of target variable (auto-detected if None)
        frequency: Data frequency ('D', 'W', 'M', 'H', 'auto')
        sample_size: If provided, sample the dataset to this size
    
    Returns:
        Dictionary with loading results and metadata
    """
    try:
        result = DatasetManager.load_dataset(
            source=file_path,
            dataset_name=dataset_name,
            date_column=date_column,
            target_column=target_column,
            frequency=frequency,
            is_file_path=True
        )
        
        if result["status"] != "success":
            return result
        
        # Apply sampling if requested
        if sample_size and sample_size < result["rows"]:
            df = DatasetManager.get_dataset(dataset_name)
            original_rows = len(df)
            
            # Sample while preserving time order (take evenly spaced samples)
            indices = np.linspace(0, len(df) - 1, sample_size, dtype=int)
            sampled_df = df.iloc[indices]
            
            loaded_datasets[dataset_name] = sampled_df
            
            # Update schema
            schema = dataset_schemas[dataset_name]
            schema.row_count = len(sampled_df)
            schema.start_date = str(sampled_df.index.min())
            schema.end_date = str(sampled_df.index.max())
            
            result["rows"] = len(sampled_df)
            result["sampled"] = True
            result["original_rows"] = original_rows
            result["preview"] = sampled_df.head(5).to_string()
        
        return result
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to load dataset: {str(e)}"
        }


async def load_dataset_from_content(
    content: str,
    dataset_name: str,
    content_type: str = "csv",
    date_column: Optional[str] = None,
    target_column: Optional[str] = None,
    frequency: str = "auto",
    sample_size: Optional[int] = None
) -> Dict[str, Any]:
    """
    Load a time series dataset from raw content (e.g., uploaded file content).
    
    Args:
        content: Raw CSV or JSON content as string
        dataset_name: Name to store the dataset under
        content_type: Type of content ('csv' or 'json')
        date_column: Name of datetime column (auto-detected if None)
        target_column: Name of target variable (auto-detected if None)
        frequency: Data frequency ('D', 'W', 'M', 'H', 'auto')
        sample_size: If provided, sample the dataset to this size
    
    Returns:
        Dictionary with loading results and metadata
    """
    try:
        result = DatasetManager.load_from_content(
            content=content,
            dataset_name=dataset_name,
            content_type=content_type,
            date_column=date_column,
            target_column=target_column,
            frequency=frequency
        )
        
        if result["status"] != "success":
            return result
        
        # Apply sampling if requested
        if sample_size and sample_size < result["rows"]:
            df = DatasetManager.get_dataset(dataset_name)
            original_rows = len(df)
            
            indices = np.linspace(0, len(df) - 1, sample_size, dtype=int)
            sampled_df = df.iloc[indices]
            
            loaded_datasets[dataset_name] = sampled_df
            
            schema = dataset_schemas[dataset_name]
            schema.row_count = len(sampled_df)
            schema.start_date = str(sampled_df.index.min())
            schema.end_date = str(sampled_df.index.max())
            
            result["rows"] = len(sampled_df)
            result["sampled"] = True
            result["original_rows"] = original_rows
            result["preview"] = sampled_df.head(5).to_string()
        
        return result
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to load dataset from content: {str(e)}"
        }


async def list_datasets() -> Dict[str, Any]:
    """
    List all loaded datasets.
    
    Returns:
        Dictionary with list of datasets and their metadata
    """
    datasets = DatasetManager.list_datasets()
    
    return {
        "status": "success",
        "count": len(datasets),
        "datasets": datasets
    }


async def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """
    Get detailed information about a loaded dataset.
    
    Args:
        dataset_name: Name of the dataset
    
    Returns:
        Dictionary with dataset information
    """
    schema = DatasetManager.get_schema(dataset_name)
    
    if schema is None:
        return {
            "status": "error",
            "message": f"Dataset '{dataset_name}' not found. Use list_datasets to see available datasets."
        }
    
    df = DatasetManager.get_dataset(dataset_name)
    
    return {
        "status": "success",
        "dataset_name": dataset_name,
        "rows": schema.row_count,
        "columns": schema.columns,
        "numeric_columns": schema.numeric_columns,
        "date_column": schema.date_column,
        "target_column": schema.target_column,
        "frequency": schema.frequency,
        "date_range": f"{schema.start_date} to {schema.end_date}",
        "missing_values": schema.missing_count,
        "missing_pct": f"{schema.missing_pct:.2f}%",
        "outliers": schema.outlier_count,
        "duplicates": schema.duplicate_count,
        "statistics": schema.statistics,
        "preview": df.head(10).to_string() if df is not None else None
    }


async def delete_dataset(dataset_name: str) -> Dict[str, Any]:
    """
    Delete a loaded dataset from memory.
    
    Args:
        dataset_name: Name of the dataset to delete
    
    Returns:
        Dictionary with deletion result
    """
    if DatasetManager.delete_dataset(dataset_name):
        return {
            "status": "success",
            "message": f"Dataset '{dataset_name}' deleted successfully."
        }
    else:
        return {
            "status": "error",
            "message": f"Dataset '{dataset_name}' not found."
        }


async def get_dataset_sample(
    dataset_name: str,
    n: int = 10,
    head: bool = True
) -> Dict[str, Any]:
    """
    Get a sample of rows from a loaded dataset.
    
    Args:
        dataset_name: Name of the dataset
        n: Number of rows to return
        head: If True, return first n rows; if False, return last n rows
    
    Returns:
        Dictionary with sample data
    """
    df = DatasetManager.get_dataset(dataset_name)
    
    if df is None:
        return {
            "status": "error",
            "message": f"Dataset '{dataset_name}' not found."
        }
    
    sample = df.head(n) if head else df.tail(n)
    
    return {
        "status": "success",
        "dataset_name": dataset_name,
        "sample_type": "head" if head else "tail",
        "rows_returned": len(sample),
        "data": sample.reset_index().to_dict(orient='records')
    }


async def query_dataset(
    dataset_name: str,
    query: Optional[str] = None,
    columns: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Query a loaded dataset with optional filtering.
    
    Args:
        dataset_name: Name of the dataset
        query: Pandas query string (e.g., "sales > 100")
        columns: List of columns to return
        start_date: Filter data after this date
        end_date: Filter data before this date
    
    Returns:
        Dictionary with query results
    """
    df = DatasetManager.get_dataset(dataset_name)
    
    if df is None:
        return {
            "status": "error",
            "message": f"Dataset '{dataset_name}' not found."
        }
    
    result_df = df.copy()
    
    # Apply date filters
    if start_date:
        result_df = result_df[result_df.index >= pd.to_datetime(start_date)]
    if end_date:
        result_df = result_df[result_df.index <= pd.to_datetime(end_date)]
    
    # Apply query
    if query:
        try:
            result_df = result_df.query(query)
        except Exception as e:
            return {
                "status": "error",
                "message": f"Invalid query: {str(e)}"
            }
    
    # Select columns
    if columns:
        valid_cols = [c for c in columns if c in result_df.columns]
        if valid_cols:
            result_df = result_df[valid_cols]
    
    return {
        "status": "success",
        "dataset_name": dataset_name,
        "rows_returned": len(result_df),
        "columns": result_df.columns.tolist(),
        "data": result_df.head(100).reset_index().to_dict(orient='records'),
        "truncated": len(result_df) > 100
    }