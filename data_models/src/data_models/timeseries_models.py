from typing import Dict, Any, Union, Literal
from pydantic import BaseModel, field_validator
import numpy as np

class Quartiles(BaseModel):
    """Quartile statistics for numerical distributions."""
    q25: float
    q50: float
    q75: float

    @field_validator('q25', 'q50', 'q75', mode='before')
    @classmethod
    def convert_numpy_float(cls, v):
        if isinstance(v, np.floating):
            return float(v)
        return v


class BaseDistributionAnalysis(BaseModel):
    """Base model for common distribution analysis fields."""
    dataset: str
    column: str
    dtype: str
    total_values: int
    unique_values: int
    null_values: int
    null_percentage: float
    distribution_type: Literal["numerical", "categorical"]

    @field_validator('total_values', 'unique_values', 'null_values', mode='before')
    @classmethod
    def convert_numpy_int(cls, v):
        if isinstance(v, np.integer):
            return int(v)
        return v

    @field_validator('null_percentage', mode='before')
    @classmethod
    def convert_numpy_float(cls, v):
        if isinstance(v, np.floating):
            return float(v)
        return v


class NumericalDistributionAnalysis(BaseDistributionAnalysis):
    """Distribution analysis for numerical columns."""
    distribution_type: Literal["numerical"] = "numerical"
    mean: float
    median: float
    std: float
    min: Union[float, int]
    max: Union[float, int]
    quartiles: Quartiles
    skewness: float
    kurtosis: float

    @field_validator('mean', 'median', 'std', 'min', 'max', 'skewness', 'kurtosis', mode='before')
    @classmethod
    def convert_numpy_float(cls, v):
        if isinstance(v, np.floating):
            return float(v)
        if isinstance(v, np.integer):
            return int(v)
        return v


class CategoricalDistributionAnalysis(BaseDistributionAnalysis):
    """Distribution analysis for categorical columns."""
    distribution_type: Literal["categorical"] = "categorical"
    most_frequent: Any
    frequency_of_most_common: int
    top_10_values: Dict[str, int]

    @field_validator('frequency_of_most_common', mode='before')
    @classmethod
    def convert_numpy_int(cls, v):
        if isinstance(v, np.integer):
            return int(v)
        return v

    @field_validator('top_10_values', mode='before')
    @classmethod
    def convert_dict_values(cls, v):
        if isinstance(v, dict):
            return {str(k): int(val) if isinstance(val, np.integer) else val for k, val in v.items()}
        return v
