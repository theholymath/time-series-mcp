import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter('ignore', ConvergenceWarning)

from typing import Dict, Any, Union, Literal
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, field_validator
from io import StringIO
import pandas as pd
import numpy as np
import json

from data_models import NumericalDistributionAnalysis, CategoricalDistributionAnalysis

mcp = FastMCP("TimeSeriesMCP")

# Pydantic Models for Distribution Analysis


# key: dataset_name, value: {"DF":pandas DF,"COLUMN_NAME_DISTRIBUTION_ANALYSIS":DICT}
DATA_STORE = {}

@mcp.tool()
def load_dataset(filepath:str) -> str:
    """
    Load a CSV or JSON file into server's memory

    Args:
        filepath: the absolute path of the csv or json
    """
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.json'):
            df = pd.read_json(filepath)
        else:
            return "Error: only csv and json"
        
        name = filepath.split('/')[-1]
        DATA_STORE[name] = {"df":df}

        return  f"Successfully loaded '{name}' with {len(df)} rows."
    except Exception as e:
        return f"Error loading file: {str(e)}"

@mcp.tool()
def get_dataset_metadata(dataset_name: str) -> str:
    """
    Get the 'Nutrition Label' of the dataset. Returns columns, types, and samples.
    NEVER read the whole file to the user.
    """
    if dataset_name not in DATA_STORE:
        return f"Error: Dataset '{dataset_name}' not found. Please load it first."

    df = DATA_STORE[dataset_name]["df"]

    output = []
    output.append(f"=== Metadata for {dataset_name} ===")
    output.append(f"Shape: {df.shape}")
    output.append("\nColumns & Types:")
    output.append(str(df.dtypes))
    
    output.append("\n=== First 5 Rows ===")
    output.append(df.head().to_markdown(index=False))
    
    output.append("\n=== Last 5 Rows ===")
    output.append(df.tail().to_markdown(index=False))
    
    output.append("\n=== Summary Statistics ===")
    output.append(df.describe().to_markdown())

    # Simple heuristic: look for 'date', 'time', 'year' in column names
    # this is lazy and needs to be figured out later. 
    time_cols = [c for c in df.columns if any(x in c.lower() for x in ['date', 'time', 'year', 'day'])]
    if time_cols:
        output.append(f"\nPotential Time Columns: {time_cols}")
        
    return "\n".join(output)

"""Distribution analysis tool implementation."""

@mcp.tool()
async def column_distributions_analysis(dataset_name: str, column_name: str) -> dict:
    """Analyze distribution of any column. Includes summary statistics. """
    try:
        if dataset_name not in DATA_STORE:
          return {}

        df = DATA_STORE[dataset_name]["df"]

        if column_name not in df.columns:
            return {"error": f"Column '{column_name}' not found in dataset"}
        
        series = df[column_name]
        
        result = {
            "dataset": dataset_name,
            "column": column_name,
            "dtype": str(series.dtype),
            "total_values": len(series),
            "unique_values": series.nunique(),
            "null_values": series.isnull().sum(),
            "null_percentage": round(series.isnull().mean() * 100, 2)
        }
        
        if pd.api.types.is_numeric_dtype(series):
            # Numerical distribution
            result.update({
                "distribution_type": "numerical",
                "mean": round(series.mean(), 3),
                "median": round(series.median(), 3),
                "std": round(series.std(), 3),
                "min": series.min(),
                "max": series.max(),
                "quartiles": {
                    "q25": round(series.quantile(0.25), 3),
                    "q50": round(series.quantile(0.50), 3),
                    "q75": round(series.quantile(0.75), 3)
                },
                "skewness": round(series.skew(), 3),
                "kurtosis": round(series.kurtosis(), 3)
            })
            # Create Pydantic model instance
            model = NumericalDistributionAnalysis(**result)
        else:
            # Categorical distribution
            value_counts = series.value_counts().head(10)
            result.update({
                "distribution_type": "categorical",
                "most_frequent": value_counts.index[0] if len(value_counts) > 0 else None,
                "frequency_of_most_common": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                "top_10_values": value_counts.to_dict()
            })
            # Create Pydantic model instance
            model = CategoricalDistributionAnalysis(**result)

        # Store Pydantic model in DATA_STORE
        DATA_STORE[dataset_name][f"{column_name}_DISTRIBUTION_ANALYSIS"] = model

        # Return dict for MCP compatibility
        return model.model_dump()
        
    except Exception as e:
        return {"error": f"Distribution analysis failed: {str(e)}"}
    
@mcp.tool()
async def write_data_analysis_to_disk(dataset_names: list) -> None:
    """Write global object to disk. Can be all or subset of names """
    DATA_STORE_SUBSET = {key: DATA_STORE[key] for key in dataset_names if key in DATA_STORE}

    for dataset_name in dataset_names:
        dataset_data = DATA_STORE_SUBSET[dataset_name]

        # Convert DataFrame to JSON
        df_temp = dataset_data["df"]
        dataset_data["df"] = df_temp.to_json()

        # Convert Pydantic models to dicts
        for key, value in list(dataset_data.items()):
            if hasattr(value, 'model_dump'):
                dataset_data[key] = value.model_dump()

    with open('data_analysis_HASH_NAME_LATER.json', 'w') as json_file:
        json.dump(DATA_STORE_SUBSET, json_file, indent=4)

def _is_distribution_analysis(value: Any) -> bool:
    """Check if dict is a serialized distribution analysis."""
    return isinstance(value, dict) and "distribution_type" in value

@mcp.tool()
async def load_analysis_from_disk(filepath: str) -> str:
    """
    Load previously saved analysis data back into memory.

    Args:
        filepath: Path to the analysis JSON file created by write_data_analysis_to_disk

    Returns:
        Success message with loaded dataset names
    """
    try:
        # Load JSON file
        with open(filepath, 'r') as f:
            saved_data = json.load(f)

        # Reconstruct DATA_STORE
        loaded_datasets = []
        for dataset_name, dataset_data in saved_data.items():
            DATA_STORE[dataset_name] = {}

            for key, value in dataset_data.items():
                if key == "df":
                    # Reconstruct DataFrame from JSON string
                    df = pd.read_json(StringIO(value))
                    DATA_STORE[dataset_name]["df"] = df

                elif _is_distribution_analysis(value):
                    # Reconstruct Pydantic model based on distribution_type
                    if value["distribution_type"] == "numerical":
                        model = NumericalDistributionAnalysis(**value)
                    elif value["distribution_type"] == "categorical":
                        model = CategoricalDistributionAnalysis(**value)
                    else:
                        # Unknown type, skip
                        continue

                    DATA_STORE[dataset_name][key] = model
                else:
                    # Other data, store as-is
                    DATA_STORE[dataset_name][key] = value

            loaded_datasets.append(dataset_name)

        return f"Successfully loaded analysis for {len(loaded_datasets)} dataset(s): {', '.join(loaded_datasets)}"

    except FileNotFoundError:
        return f"Error: File '{filepath}' not found"
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON format - {str(e)}"
    except Exception as e:
        return f"Error loading analysis: {str(e)}"

if __name__ == "__main__":
    mcp.run()
