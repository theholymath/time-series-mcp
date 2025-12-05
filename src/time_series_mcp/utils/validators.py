"""
Data Validators
===============
Validation utilities for time series data.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple


class DataValidator:
    """Validate time series data."""
    
    @staticmethod
    def validate_time_series(
        data: pd.DataFrame,
        date_column: Optional[str] = None,
        target_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate time series data and return validation report.
        
        Args:
            data: DataFrame to validate
            date_column: Expected date column name
            target_column: Expected target column name
        
        Returns:
            Validation report dictionary
        """
        report = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "info": {}
        }
        
        # Check if DataFrame is empty
        if data.empty:
            report["is_valid"] = False
            report["errors"].append("DataFrame is empty")
            return report
        
        # Check for date column
        if date_column and date_column not in data.columns:
            report["is_valid"] = False
            report["errors"].append(f"Date column '{date_column}' not found in data")
        
        # Check for target column
        if target_column and target_column not in data.columns:
            report["is_valid"] = False
            report["errors"].append(f"Target column '{target_column}' not found in data")
        
        # Check for numeric target
        if target_column and target_column in data.columns:
            if not np.issubdtype(data[target_column].dtype, np.number):
                report["is_valid"] = False
                report["errors"].append(f"Target column '{target_column}' is not numeric")
        
        # Check minimum data points
        min_points = 10
        if len(data) < min_points:
            report["warnings"].append(f"Only {len(data)} data points. Recommend at least {min_points} for reliable forecasting")
        
        # Check for missing values
        if target_column and target_column in data.columns:
            missing_count = data[target_column].isna().sum()
            missing_pct = (missing_count / len(data)) * 100
            
            if missing_pct > 50:
                report["is_valid"] = False
                report["errors"].append(f"Too many missing values ({missing_pct:.1f}%)")
            elif missing_pct > 10:
                report["warnings"].append(f"Significant missing values ({missing_pct:.1f}%)")
            
            report["info"]["missing_count"] = int(missing_count)
            report["info"]["missing_pct"] = float(missing_pct)
        
        # Check for constant values
        if target_column and target_column in data.columns:
            unique_values = data[target_column].nunique()
            if unique_values == 1:
                report["warnings"].append("Target column has only one unique value")
            
            report["info"]["unique_values"] = int(unique_values)
        
        # Check for negative values (might be valid, but worth noting)
        if target_column and target_column in data.columns:
            negative_count = (data[target_column] < 0).sum()
            if negative_count > 0:
                report["info"]["negative_values"] = int(negative_count)
        
        # Record data info
        report["info"]["total_rows"] = len(data)
        report["info"]["total_columns"] = len(data.columns)
        
        return report
    
    @staticmethod
    def check_frequency_consistency(
        data: pd.DataFrame,
        date_column: str
    ) -> Dict[str, Any]:
        """
        Check if the time series has consistent frequency.
        
        Args:
            data: DataFrame with datetime index or column
            date_column: Name of the date column
        
        Returns:
            Frequency consistency report
        """
        report = {
            "is_consistent": True,
            "detected_frequency": None,
            "gaps": [],
            "duplicates": []
        }
        
        # Get datetime series
        if date_column in data.columns:
            dates = pd.to_datetime(data[date_column])
        elif isinstance(data.index, pd.DatetimeIndex):
            dates = data.index.to_series()
        else:
            report["is_consistent"] = False
            return report
        
        dates = dates.sort_values()
        
        # Calculate time differences
        diffs = dates.diff().dropna()
        
        if len(diffs) == 0:
            report["is_consistent"] = True
            return report
        
        # Find most common difference (mode)
        mode_diff = diffs.mode().iloc[0] if len(diffs.mode()) > 0 else diffs.median()
        
        # Check for inconsistencies
        tolerance = pd.Timedelta(hours=1)  # Allow small variations
        inconsistent = diffs[abs(diffs - mode_diff) > tolerance]
        
        if len(inconsistent) > 0:
            report["is_consistent"] = False
            report["gaps"] = [
                {
                    "index": int(idx),
                    "expected": str(mode_diff),
                    "actual": str(diff)
                }
                for idx, diff in inconsistent.items()
            ][:10]  # Limit to first 10
        
        # Check for duplicates
        duplicate_mask = dates.duplicated()
        if duplicate_mask.any():
            report["duplicates"] = dates[duplicate_mask].tolist()[:10]
        
        # Determine frequency string
        freq_map = {
            'H': pd.Timedelta(hours=1),
            'D': pd.Timedelta(days=1),
            'W': pd.Timedelta(weeks=1),
            'M': pd.Timedelta(days=30),
        }
        
        for freq_str, freq_delta in freq_map.items():
            if abs(mode_diff - freq_delta) < pd.Timedelta(hours=12):
                report["detected_frequency"] = freq_str
                break
        
        if report["detected_frequency"] is None:
            report["detected_frequency"] = str(mode_diff)
        
        return report
    
    @staticmethod
    def validate_forecast_request(
        periods: int,
        frequency: str,
        data_length: int
    ) -> Dict[str, Any]:
        """
        Validate a forecast request.
        
        Args:
            periods: Number of periods to forecast
            frequency: Forecast frequency
            data_length: Length of historical data
        
        Returns:
            Validation report
        """
        report = {
            "is_valid": True,
            "warnings": []
        }
        
        # Check periods
        if periods <= 0:
            report["is_valid"] = False
            report["warnings"].append("Periods must be positive")
        
        # Check if forecast horizon is too long relative to data
        max_horizon_ratio = 0.5  # Don't forecast more than 50% of data length
        if periods > data_length * max_horizon_ratio:
            report["warnings"].append(
                f"Forecast horizon ({periods}) is long relative to data length ({data_length}). "
                f"Results may be unreliable."
            )
        
        # Validate frequency
        valid_frequencies = ['H', 'D', 'W', 'M', 'Q', 'Y', 'B', 'h', 'd', 'w', 'm']
        if frequency and frequency not in valid_frequencies:
            report["warnings"].append(f"Unusual frequency '{frequency}'. Expected one of: {valid_frequencies}")
        
        return report