"""
Visualizer for Time Series MCP
==============================
Creates visualizations for time series analysis and forecasts.
"""

import pandas as pd
import numpy as np
from typing import Any, Optional
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from io import BytesIO
import base64

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class TimeSeriesVisualizer:
    """Create time series visualizations."""
    
    def __init__(self):
        self.fig_size = (12, 6)
        self.dpi = 100
    
    def create_analysis_plots(
        self,
        data: pd.DataFrame,
        metadata: dict[str, Any],
        analysis: dict[str, Any]
    ) -> dict[str, str]:
        """
        Create analysis plots.
        
        Returns:
            Dictionary of plot names to base64-encoded images
        """
        plots = {}
        target_col = metadata["target_column"]
        series = data[target_col]
        
        # 1. Time series plot with trend
        plots["time_series"] = self._create_time_series_plot(series, metadata)
        
        # 2. ACF/PACF plots
        plots["autocorrelation"] = self._create_acf_pacf_plot(analysis)
        
        # 3. Distribution plot
        plots["distribution"] = self._create_distribution_plot(series)
        
        # 4. Seasonal decomposition (if applicable)
        if analysis.get("has_seasonality"):
            plots["decomposition"] = self._create_decomposition_plot(series, analysis)
        
        return plots
    
    def create_forecast_plot(
        self,
        historical_data: pd.DataFrame,
        forecasts: dict[str, pd.DataFrame],
        best_model: str,
        metadata: dict[str, Any]
    ) -> str:
        """Create forecast visualization."""
        fig, ax = plt.subplots(figsize=(14, 7))
        
        target_col = metadata["target_column"]
        series = historical_data[target_col]
        
        # Plot historical data
        ax.plot(series.index, series.values, 'b-', label='Historical', linewidth=1.5)
        
        # Plot forecasts
        colors = plt.cm.Set2(np.linspace(0, 1, len(forecasts)))
        
        for (name, forecast), color in zip(forecasts.items(), colors):
            if name == best_model:
                # Highlight best model
                ax.plot(forecast.index, forecast['yhat'], 
                       color='red', linewidth=2, label=f'{name} (Best)', linestyle='-')
                ax.fill_between(
                    forecast.index,
                    forecast['yhat_lower'],
                    forecast['yhat_upper'],
                    color='red',
                    alpha=0.2,
                    label=f'{name} 95% CI'
                )
            else:
                ax.plot(forecast.index, forecast['yhat'],
                       color=color, linewidth=1, label=name, linestyle='--', alpha=0.7)
        
        # Formatting
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(target_col, fontsize=12)
        ax.set_title(f'Time Series Forecast - Best Model: {best_model}', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        # Add vertical line at forecast start
        if len(forecasts) > 0:
            first_forecast = list(forecasts.values())[0]
            ax.axvline(x=first_forecast.index[0], color='gray', linestyle='--', alpha=0.5)
            ax.text(first_forecast.index[0], ax.get_ylim()[1], ' Forecast Start',
                   fontsize=9, color='gray')
        
        plt.tight_layout()
        
        return self._fig_to_base64(fig)
    
    def create_comparison_plot(self, comparison: dict[str, Any]) -> str:
        """Create model comparison visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        models = list(comparison["cv_results"].keys())
        
        # 1. MAE comparison
        ax = axes[0, 0]
        mae_values = [comparison["cv_results"][m]["mae"] for m in models]
        mae_stds = [comparison["cv_results"][m]["mae_std"] for m in models]
        bars = ax.bar(models, mae_values, yerr=mae_stds, capsize=5, color=sns.color_palette("husl", len(models)))
        ax.set_title('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
        ax.set_ylabel('MAE')
        
        # 2. RMSE comparison
        ax = axes[0, 1]
        rmse_values = [comparison["cv_results"][m]["rmse"] for m in models]
        rmse_stds = [comparison["cv_results"][m]["rmse_std"] for m in models]
        ax.bar(models, rmse_values, yerr=rmse_stds, capsize=5, color=sns.color_palette("husl", len(models)))
        ax.set_title('Root Mean Square Error (RMSE)', fontsize=12, fontweight='bold')
        ax.set_ylabel('RMSE')
        
        # 3. MAPE comparison
        ax = axes[1, 0]
        mape_values = [comparison["cv_results"][m]["mape"] * 100 for m in models]
        ax.bar(models, mape_values, color=sns.color_palette("husl", len(models)))
        ax.set_title('Mean Absolute Percentage Error (MAPE)', fontsize=12, fontweight='bold')
        ax.set_ylabel('MAPE (%)')
        
        # 4. Training time
        ax = axes[1, 1]
        times = [comparison["cv_results"][m]["train_time"] for m in models]
        ax.bar(models, times, color=sns.color_palette("husl", len(models)))
        ax.set_title('Training Time', fontsize=12, fontweight='bold')
        ax.set_ylabel('Time (seconds)')
        
        plt.tight_layout()
        
        return self._fig_to_base64(fig)
    
    def _create_time_series_plot(self, series: pd.Series, metadata: dict[str, Any]) -> str:
        """Create basic time series plot."""
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        ax.plot(series.index, series.values, 'b-', linewidth=1)
        
        # Add rolling mean
        window = min(30, len(series) // 10)
        if window > 1:
            rolling_mean = series.rolling(window=window).mean()
            ax.plot(series.index, rolling_mean, 'r-', linewidth=2, label=f'{window}-period Moving Average')
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(metadata["target_column"], fontsize=12)
        ax.set_title('Time Series Overview', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        
        plt.tight_layout()
        
        return self._fig_to_base64(fig)
    
    def _create_acf_pacf_plot(self, analysis: dict[str, Any]) -> str:
        """Create ACF/PACF plots."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        acf_values = analysis.get("acf_values", [])
        pacf_values = analysis.get("pacf_values", [])
        
        # ACF plot
        ax = axes[0]
        ax.bar(range(len(acf_values)), acf_values, color='steelblue', width=0.3)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.axhline(y=1.96/np.sqrt(50), color='red', linestyle='--', alpha=0.7)
        ax.axhline(y=-1.96/np.sqrt(50), color='red', linestyle='--', alpha=0.7)
        ax.set_title('Autocorrelation Function (ACF)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Lag')
        ax.set_ylabel('ACF')
        
        # PACF plot
        ax = axes[1]
        ax.bar(range(len(pacf_values)), pacf_values, color='steelblue', width=0.3)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.axhline(y=1.96/np.sqrt(50), color='red', linestyle='--', alpha=0.7)
        ax.axhline(y=-1.96/np.sqrt(50), color='red', linestyle='--', alpha=0.7)
        ax.set_title('Partial Autocorrelation Function (PACF)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Lag')
        ax.set_ylabel('PACF')
        
        plt.tight_layout()
        
        return self._fig_to_base64(fig)
    
    def _create_distribution_plot(self, series: pd.Series) -> str:
        """Create distribution plot."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram with KDE
        ax = axes[0]
        sns.histplot(series, kde=True, ax=ax, color='steelblue')
        ax.set_title('Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Value')
        
        # Box plot
        ax = axes[1]
        sns.boxplot(y=series, ax=ax, color='steelblue')
        ax.set_title('Box Plot', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        return self._fig_to_base64(fig)
    
    def _create_decomposition_plot(self, series: pd.Series, analysis: dict[str, Any]) -> str:
        """Create seasonal decomposition plot."""
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        period = analysis.get("seasonal_period", 7)
        
        try:
            decomposition = seasonal_decompose(series, model='additive', period=period)
            
            fig, axes = plt.subplots(4, 1, figsize=(14, 12))
            
            axes[0].plot(series.index, series.values, 'b-')
            axes[0].set_title('Original', fontsize=12, fontweight='bold')
            
            axes[1].plot(series.index, decomposition.trend, 'g-')
            axes[1].set_title('Trend', fontsize=12, fontweight='bold')
            
            axes[2].plot(series.index, decomposition.seasonal, 'r-')
            axes[2].set_title('Seasonal', fontsize=12, fontweight='bold')
            
            axes[3].plot(series.index, decomposition.resid, 'purple')
            axes[3].set_title('Residual', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            
            return self._fig_to_base64(fig)
        except Exception as e:
            # Return empty plot if decomposition fails
            fig, ax = plt.subplots(figsize=self.fig_size)
            ax.text(0.5, 0.5, f'Could not create decomposition: {str(e)}',
                   ha='center', va='center', transform=ax.transAxes)
            return self._fig_to_base64(fig)
    
    def _fig_to_base64(self, fig: plt.Figure) -> str:
        """Convert matplotlib figure to base64 string."""
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64