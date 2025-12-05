"""
Dataset schemas and manager for Time Series MCP.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import json
import io


@dataclass
class TimeSeriesSchema:
    """Schema for a loaded time series dataset."""
    name: str
    date_column: str
    target_column: str
    frequency: str
    row_count: int
    start_date: str
    end_date: str
    columns: List[str]
    numeric_columns: List[str]
    missing_count: int
    missing_pct: float
    outlier_count: int
    duplicate_count: int
    statistics: Dict[str, Any] = field(default_factory=dict)
    

@dataclass 
class ForecastResult:
    """Schema for forecast results."""
    model_name: str
    periods: int
    frequency: str
    forecast_df: pd.DataFrame
    metrics: Dict[str, float]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


# Global storage for loaded datasets and their schemas
loaded_datasets: Dict[str, pd.DataFrame] = {}
dataset_schemas: Dict[str, TimeSeriesSchema] = {}
forecast_results: Dict[str, Dict[str, ForecastResult]] = {}  # dataset_name -> model_name -> result


class DatasetManager:
    """Manages loading, storing, and accessing time series datasets."""
    
    SUPPORTED_FORMATS = ['.csv', '.xlsx', '.xls', '.json', '.parquet']
    
    @classmethod
    def load_dataset(
        cls,
        source: str,
        dataset_name: str,
        date_column: Optional[str] = None,
        target_column: Optional[str] = None,
        frequency: str = "auto",
        is_file_path: bool = True
    ) -> Dict[str, Any]:
        """
        Load a dataset from file path or raw content.
        
        Args:
            source: File path OR raw content string (CSV/JSON)
            dataset_name: Name to store the dataset under
            date_column: Name of datetime column (auto-detected if None)
            target_column: Name of target variable (auto-detected if None)
            frequency: Data frequency ('D', 'W', 'M', 'H', 'auto')
            is_file_path: If True, source is a file path; if False, source is raw content
        
        Returns:
            Dictionary with loading results and metadata
        """
        try:
            # Load the data
            if is_file_path:
                df = cls._load_from_file(source)
            else:
                df = cls._load_from_content(source)
            
            # Detect date column
            date_col = date_column or cls._detect_date_column(df)
            if date_col is None:
                return {
                    "status": "error",
                    "message": "Could not detect date column. Please specify 'date_column' parameter.",
                    "available_columns": df.columns.tolist()
                }
            
            # Parse and set datetime index
            df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True)
            df = df.sort_values(date_col).reset_index(drop=True)
            df = df.set_index(date_col)
            
            # Detect target column
            target_col = target_column or cls._detect_target_column(df)
            if target_col is None:
                return {
                    "status": "error", 
                    "message": "Could not detect target column. Please specify 'target_column' parameter.",
                    "available_columns": df.columns.tolist()
                }
            
            # Detect frequency
            detected_freq = frequency if frequency != "auto" else cls._detect_frequency(df)
            
            # Build schema
            schema = cls._build_schema(df, dataset_name, date_col, target_col, detected_freq)
            
            # Store dataset and schema
            loaded_datasets[dataset_name] = df
            dataset_schemas[dataset_name] = schema
            
            return {
                "status": "success",
                "dataset_name": dataset_name,
                "rows": schema.row_count,
                "columns": schema.columns,
                "date_column": schema.date_column,
                "target_column": schema.target_column,
                "frequency": schema.frequency,
                "date_range": f"{schema.start_date} to {schema.end_date}",
                "missing_values": schema.missing_count,
                "missing_pct": schema.missing_pct,
                "outliers": schema.outlier_count,
                "preview": df.head(5).to_string()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to load dataset: {str(e)}"
            }
    
    @classmethod
    def load_from_content(
        cls,
        content: str,
        dataset_name: str,
        content_type: str = "csv",
        date_column: Optional[str] = None,
        target_column: Optional[str] = None,
        frequency: str = "auto"
    ) -> Dict[str, Any]:
        """
        Load dataset from raw content string (for file uploads).
        
        Args:
            content: Raw CSV or JSON content as string
            dataset_name: Name to store the dataset under
            content_type: Type of content ('csv' or 'json')
            date_column: Name of datetime column
            target_column: Name of target variable
            frequency: Data frequency
        
        Returns:
            Dictionary with loading results
        """
        return cls.load_dataset(
            source=content,
            dataset_name=dataset_name,
            date_column=date_column,
            target_column=target_column,
            frequency=frequency,
            is_file_path=False
        )
    
    @classmethod
    def get_dataset(cls, dataset_name: str) -> Optional[pd.DataFrame]:
        """Get a loaded dataset by name."""
        return loaded_datasets.get(dataset_name)
    
    @classmethod
    def get_schema(cls, dataset_name: str) -> Optional[TimeSeriesSchema]:
        """Get schema for a loaded dataset."""
        return dataset_schemas.get(dataset_name)
    
    @classmethod
    def list_datasets(cls) -> List[Dict[str, Any]]:
        """List all loaded datasets."""
        return [
            {
                "name": name,
                "rows": schema.row_count,
                "date_range": f"{schema.start_date} to {schema.end_date}",
                "target": schema.target_column,
                "frequency": schema.frequency
            }
            for name, schema in dataset_schemas.items()
        ]
    
    @classmethod
    def delete_dataset(cls, dataset_name: str) -> bool:
        """Delete a loaded dataset."""
        if dataset_name in loaded_datasets:
            del loaded_datasets[dataset_name]
            del dataset_schemas[dataset_name]
            if dataset_name in forecast_results:
                del forecast_results[dataset_name]
            return True
        return False
    
    @classmethod
    def store_forecast(
        cls,
        dataset_name: str,
        model_name: str,
        forecast_df: pd.DataFrame,
        metrics: Dict[str, float],
        periods: int,
        frequency: str
    ) -> None:
        """Store forecast results for a dataset."""
        if dataset_name not in forecast_results:
            forecast_results[dataset_name] = {}
        
        forecast_results[dataset_name][model_name] = ForecastResult(
            model_name=model_name,
            periods=periods,
            frequency=frequency,
            forecast_df=forecast_df,
            metrics=metrics
        )
    
    @classmethod
    def get_forecast(cls, dataset_name: str, model_name: Optional[str] = None) -> Optional[Union[ForecastResult, Dict[str, ForecastResult]]]:
        """Get forecast results for a dataset."""
        if dataset_name not in forecast_results:
            return None
        
        if model_name:
            return forecast_results[dataset_name].get(model_name)
        
        return forecast_results[dataset_name]
    
    # ============ Private Helper Methods ============
    
    @classmethod
    def _load_from_file(cls, file_path: str) -> pd.DataFrame:
        """Load DataFrame from file path."""
        from pathlib import Path
        
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        suffix = path.suffix.lower()
        
        if suffix == '.csv':
            return pd.read_csv(path)
        elif suffix in ['.xlsx', '.xls']:
            return pd.read_excel(path)
        elif suffix == '.json':
            return pd.read_json(path)
        elif suffix == '.parquet':
            return pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    @classmethod
    def _load_from_content(cls, content: str) -> pd.DataFrame:
        """Load DataFrame from raw content string."""
        content = content.strip()
        
        # Try to detect format
        if content.startswith('{') or content.startswith('['):
            # Likely JSON
            return pd.read_json(io.StringIO(content))
        else:
            # Assume CSV
            return pd.read_csv(io.StringIO(content))
    
    @classmethod
    def _detect_date_column(cls, df: pd.DataFrame) -> Optional[str]:
        """Auto-detect the datetime column."""
        date_names = ['date', 'datetime', 'timestamp', 'time', 'ds', 'period', 'day', 'month', 'year']
        
        # Check column names
        for col in df.columns:
            if col.lower() in date_names:
                return col
        
        # Check dtypes
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col
        
        # Try parsing object columns
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col].head(100))
                    return col
                except:
                    continue
        
        return None
    
    @classmethod
    def _detect_target_column(cls, df: pd.DataFrame) -> Optional[str]:
        """Auto-detect the target variable column."""
        target_names = ['y', 'value', 'values', 'sales', 'revenue', 'price', 'count', 
                       'amount', 'quantity', 'target', 'total', 'sum']
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Check for common target names
        for col in numeric_cols:
            if col.lower() in target_names:
                return col
        
        # Return first numeric column
        if numeric_cols:
            return numeric_cols[0]
        
        return None
    
    @classmethod
    def _detect_frequency(cls, df: pd.DataFrame) -> str:
        """Detect the time series frequency."""
        if len(df) < 2:
            return 'D'
        
        time_diffs = pd.Series(df.index).diff().dropna()
        median_diff = time_diffs.median()
        
        if median_diff <= pd.Timedelta(hours=1):
            return 'H'
        elif median_diff <= pd.Timedelta(days=1):
            return 'D'
        elif median_diff <= pd.Timedelta(weeks=1):
            return 'W'
        elif median_diff <= pd.Timedelta(days=31):
            return 'M'
        else:
            return 'D'
    
    @classmethod
    def _build_schema(
        cls,
        df: pd.DataFrame,
        name: str,
        date_col: str,
        target_col: str,
        frequency: str
    ) -> TimeSeriesSchema:
        """Build schema for a dataset."""
        target_series = df[target_col]
        
        # Missing values
        missing_count = int(target_series.isna().sum())
        missing_pct = (missing_count / len(target_series)) * 100
        
        # Duplicates
        duplicate_count = int(df.index.duplicated().sum())
        
        # Outliers (IQR method)
        Q1 = target_series.quantile(0.25)
        Q3 = target_series.quantile(0.75)
        IQR = Q3 - Q1
        outlier_mask = (target_series < (Q1 - 1.5 * IQR)) | (target_series > (Q3 + 1.5 * IQR))
        outlier_count = int(outlier_mask.sum())
        
        # Statistics
        statistics = {
            "mean": float(target_series.mean()),
            "std": float(target_series.std()),
            "min": float(target_series.min()),
            "max": float(target_series.max()),
            "median": float(target_series.median())
        }
        
        return TimeSeriesSchema(
            name=name,
            date_column=date_col,
            target_column=target_col,
            frequency=frequency,
            row_count=len(df),
            start_date=str(df.index.min()),
            end_date=str(df.index.max()),
            columns=df.columns.tolist(),
            numeric_columns=df.select_dtypes(include=[np.number]).columns.tolist(),
            missing_count=missing_count,
            missing_pct=float(missing_pct),
            outlier_count=outlier_count,
            duplicate_count=duplicate_count,
            statistics=statistics
        )