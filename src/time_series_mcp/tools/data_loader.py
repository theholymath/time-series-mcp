"""
Data Loader for Time Series MCP
===============================
Handles loading and initial validation of time series data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Optional
from datetime import datetime
import json


class TimeSeriesDataLoader:
    """Load and validate time series data from various sources."""
    
    SUPPORTED_FORMATS = ['.csv', '.xlsx', '.xls', '.json', '.parquet']
    
    def __init__(self):
        self.data = None
        self.metadata = {}
    
    def load(
        self,
        file_path: str,
        date_column: Optional[str] = None,
        target_column: Optional[str] = None,
        frequency: str = "auto"
    ) -> dict[str, Any]:
        """
        Load time series data from file.
        
        Args:
            file_path: Path to the data file
            date_column: Name of datetime column (auto-detected if None)
            target_column: Name of target variable (auto-detected if None)
            frequency: Data frequency ('D', 'W', 'M', 'H', 'auto')
        
        Returns:
            Dictionary with 'data' and 'metadata'
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        suffix = path.suffix.lower()
        
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {suffix}. Supported: {self.SUPPORTED_FORMATS}")
        
        # Load data based on format
        df = self._load_file(path, suffix)
        
        # Detect or validate date column
        date_col = date_column or self._detect_date_column(df)
        if date_col is None:
            raise ValueError("Could not detect date column. Please specify 'date_column' parameter.")
        
        # Parse dates
        df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True)
        df = df.sort_values(date_col).reset_index(drop=True)
        df = df.set_index(date_col)
        
        # Detect or validate target column
        target_col = target_column or self._detect_target_column(df)
        if target_col is None:
            raise ValueError("Could not detect target column. Please specify 'target_column' parameter.")
        
        # Detect frequency
        detected_freq = frequency if frequency != "auto" else self._detect_frequency(df)
        
        # Build metadata
        self.metadata = self._build_metadata(df, date_col, target_col, detected_freq)
        self.data = df
        
        return {
            "data": df,
            "metadata": self.metadata
        }
    
    def _load_file(self, path: Path, suffix: str) -> pd.DataFrame:
        """Load file based on format."""
        if suffix == '.csv':
            return pd.read_csv(path)
        elif suffix in ['.xlsx', '.xls']:
            return pd.read_excel(path)
        elif suffix == '.json':
            return pd.read_json(path)
        elif suffix == '.parquet':
            return pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported format: {suffix}")
    
    def _detect_date_column(self, df: pd.DataFrame) -> Optional[str]:
        """Auto-detect the datetime column."""
        # Check for common date column names
        date_names = ['date', 'datetime', 'timestamp', 'time', 'ds', 'period', 'day', 'month']
        
        for col in df.columns:
            if col.lower() in date_names:
                return col
        
        # Try to parse each column as datetime
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col].head(100))
                    return col
                except:
                    continue
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                return col
        
        return None
    
    def _detect_target_column(self, df: pd.DataFrame) -> Optional[str]:
        """Auto-detect the target variable column."""
        # Common target column names
        target_names = ['y', 'value', 'sales', 'revenue', 'price', 'count', 'amount', 'quantity', 'target']
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols:
            if col.lower() in target_names:
                return col
        
        # Return first numeric column if no match
        if numeric_cols:
            return numeric_cols[0]
        
        return None
    
    def _detect_frequency(self, df: pd.DataFrame) -> str:
        """Detect the time series frequency."""
        if len(df) < 2:
            return 'D'
        
        # Calculate median time difference
        time_diffs = pd.Series(df.index).diff().dropna()
        median_diff = time_diffs.median()
        
        # Map to pandas frequency string
        if median_diff <= pd.Timedelta(hours=1):
            return 'H'
        elif median_diff <= pd.Timedelta(days=1):
            return 'D'
        elif median_diff <= pd.Timedelta(weeks=1):
            return 'W'
        elif median_diff <= pd.Timedelta(days=31):
            return 'M'
        else:
            return 'D'  # Default to daily
    
    def _build_metadata(
        self,
        df: pd.DataFrame,
        date_col: str,
        target_col: str,
        frequency: str
    ) -> dict[str, Any]:
        """Build metadata dictionary."""
        target_series = df[target_col]
        
        # Detect missing values
        missing_count = target_series.isna().sum()
        missing_pct = (missing_count / len(target_series)) * 100
        
        # Detect duplicates
        duplicate_count = df.index.duplicated().sum()
        
        # Detect outliers using IQR
        Q1 = target_series.quantile(0.25)
        Q3 = target_series.quantile(0.75)
        IQR = Q3 - Q1
        outlier_mask = (target_series < (Q1 - 1.5 * IQR)) | (target_series > (Q3 + 1.5 * IQR))
        outlier_count = outlier_mask.sum()
        
        return {
            "date_column": date_col,
            "target_column": target_col,
            "frequency": frequency,
            "start_date": str(df.index.min()),
            "end_date": str(df.index.max()),
            "total_rows": len(df),
            "missing_count": int(missing_count),
            "missing_pct": float(missing_pct),
            "duplicate_count": int(duplicate_count),
            "outlier_count": int(outlier_count),
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "all_columns": df.columns.tolist()
        }