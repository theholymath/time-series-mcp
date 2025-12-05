"""
Output Handler for Time Series MCP
==================================
Handles exporting forecasts and generating reports.
"""

import pandas as pd
import numpy as np
from typing import Any
from pathlib import Path
import json
from datetime import datetime


class OutputHandler:
    """Handle output generation and file exports."""
    
    def __init__(self):
        pass
    
    def export(
        self,
        forecasts: dict[str, pd.DataFrame],
        best_model: str,
        data: pd.DataFrame,
        metadata: dict[str, Any],
        output_path: str,
        formats: list[str]
    ) -> dict[str, str]:
        """
        Export forecast results to files.
        
        Args:
            forecasts: Dictionary of model forecasts
            best_model: Name of the best model
            data: Historical data
            metadata: Data metadata
            output_path: Base path for output files
            formats: List of output formats
        
        Returns:
            Dictionary of created file paths and descriptions
        """
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        base_name = Path(output_path).stem
        
        files_created = {}
        
        # Export CSV
        if 'csv' in formats:
            csv_path = output_dir / f"{base_name}_forecast.csv"
            
            # Get best model forecast
            if best_model and best_model in forecasts:
                forecast_df = forecasts[best_model].copy()
                forecast_df = forecast_df.reset_index()
                forecast_df.columns = ['date', 'forecast', 'lower_bound', 'upper_bound']
                forecast_df['model'] = best_model
                
                forecast_df.to_csv(csv_path, index=False)
                files_created[str(csv_path)] = "Forecast values with confidence intervals"
            
            # Export all models comparison
            all_forecasts_path = output_dir / f"{base_name}_all_models.csv"
            all_dfs = []
            for name, fc in forecasts.items():
                fc_copy = fc.copy()
                fc_copy['model'] = name
                fc_copy = fc_copy.reset_index()
                all_dfs.append(fc_copy)
            
            if all_dfs:
                all_forecasts_df = pd.concat(all_dfs, ignore_index=True)
                all_forecasts_df.to_csv(all_forecasts_path, index=False)
                files_created[str(all_forecasts_path)] = "All model forecasts comparison"
        
        # Export PNG plot
        if 'png' in formats:
            from .visualizer import TimeSeriesVisualizer
            visualizer = TimeSeriesVisualizer()
            
            plot_data = visualizer.create_forecast_plot(
                historical_data=data,
                forecasts=forecasts,
                best_model=best_model,
                metadata=metadata
            )
            
            import base64
            png_path = output_dir / f"{base_name}_plot.png"
            with open(png_path, 'wb') as f:
                f.write(base64.b64decode(plot_data))
            
            files_created[str(png_path)] = "Forecast visualization plot"
        
        # Export JSON
        if 'json' in formats:
            json_path = output_dir / f"{base_name}_forecast.json"
            
            if best_model and best_model in forecasts:
                forecast_df = forecasts[best_model]
                json_data = {
                    "model": best_model,
                    "generated_at": datetime.now().isoformat(),
                    "metadata": metadata,
                    "forecast": forecast_df.reset_index().to_dict(orient='records')
                }
                
                with open(json_path, 'w') as f:
                    json.dump(json_data, f, indent=2, default=str)
                
                files_created[str(json_path)] = "Forecast data in JSON format"
        
        # Export HTML report
        if 'html' in formats:
            html_path = output_dir / f"{base_name}_report.html"
            html_content = self._generate_html_report(
                forecasts=forecasts,
                best_model=best_model,
                data=data,
                metadata=metadata
            )
            
            with open(html_path, 'w') as f:
                f.write(html_content)
            
            files_created[str(html_path)] = "Interactive HTML report"
        
        return files_created
    
    def generate_report(
        self,
        data: pd.DataFrame,
        metadata: dict[str, Any],
        models: dict[str, Any],
        forecasts: dict[str, pd.DataFrame],
        best_model: str,
        output_dir: str,
        include_raw_data: bool = True
    ) -> dict[str, Any]:
        """Generate comprehensive forecast report."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export all formats
        self.export(
            forecasts=forecasts,
            best_model=best_model,
            data=data,
            metadata=metadata,
            output_path=str(output_path / "forecast"),
            formats=['csv', 'png', 'html', 'json']
        )
        
        # Export historical data if requested
        if include_raw_data:
            data_path = output_path / "historical_data.csv"
            data.reset_index().to_csv(data_path, index=False)
        
        # Generate summary
        summary = self._generate_summary(
            data=data,
            metadata=metadata,
            forecasts=forecasts,
            best_model=best_model
        )
        
        return {
            "summary": summary,
            "output_dir": str(output_path)
        }
    
    def _generate_summary(
        self,
        data: pd.DataFrame,
        metadata: dict[str, Any],
        forecasts: dict[str, pd.DataFrame],
        best_model: str
    ) -> str:
        """Generate text summary of the forecast."""
        target_col = metadata["target_column"]
        series = data[target_col]
        
        if best_model and best_model in forecasts:
            forecast = forecasts[best_model]
            
            current_value = series.iloc[-1]
            forecast_end = forecast['yhat'].iloc[-1]
            change_pct = (forecast_end - current_value) / current_value * 100
            
            summary = f"""
**Forecast Summary**

- **Best Model:** {best_model}
- **Forecast Period:** {forecast.index[0].strftime('%Y-%m-%d')} to {forecast.index[-1].strftime('%Y-%m-%d')}
- **Current Value:** {current_value:,.2f}
- **Forecasted End Value:** {forecast_end:,.2f}
- **Expected Change:** {change_pct:+.1f}%
- **Forecast Mean:** {forecast['yhat'].mean():,.2f}
- **95% CI Range:** [{forecast['yhat_lower'].mean():,.2f}, {forecast['yhat_upper'].mean():,.2f}]
"""
            return summary
        
        return "No forecast available to summarize."
    
    def _generate_html_report(
        self,
        forecasts: dict[str, pd.DataFrame],
        best_model: str,
        data: pd.DataFrame,
        metadata: dict[str, Any]
    ) -> str:
        """Generate interactive HTML report."""
        target_col = metadata["target_column"]
        series = data[target_col]
        
        # Build forecast table
        if best_model and best_model in forecasts:
            forecast = forecasts[best_model]
            forecast_rows = ""
            for idx, row in forecast.head(30).iterrows():
                forecast_rows += f"""
                <tr>
                    <td>{idx.strftime('%Y-%m-%d')}</td>
                    <td>{row['yhat']:,.2f}</td>
                    <td>{row['yhat_lower']:,.2f}</td>
                    <td>{row['yhat_upper']:,.2f}</td>
                </tr>
                """
        else:
            forecast_rows = "<tr><td colspan='4'>No forecast available</td></tr>"
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Time Series Forecast Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .metric {{
            display: inline-block;
            margin: 10px 20px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 32px;
            font-weight: bold;
            color: #667eea;
        }}
        .metric-label {{
            font-size: 14px;
            color: #666;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #667eea;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .best-model {{
            background-color: #e8f5e9;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            color: #2e7d32;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📈 Time Series Forecast Report</h1>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="card">
        <h2>Overview</h2>
        <div class="metric">
            <div class="metric-value">{len(series):,}</div>
            <div class="metric-label">Historical Data Points</div>
        </div>
        <div class="metric">
            <div class="metric-value">{len(forecasts.get(best_model, [])):,}</div>
            <div class="metric-label">Forecast Periods</div>
        </div>
        <div class="metric">
            <div class="metric-value"><span class="best-model">{best_model}</span></div>
            <div class="metric-label">Best Model</div>
        </div>
    </div>
    
    <div class="card">
        <h2>Data Summary</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Date Range</td><td>{metadata['start_date']} to {metadata['end_date']}</td></tr>
            <tr><td>Frequency</td><td>{metadata['frequency']}</td></tr>
            <tr><td>Target Column</td><td>{target_col}</td></tr>
            <tr><td>Mean</td><td>{series.mean():,.2f}</td></tr>
            <tr><td>Std Dev</td><td>{series.std():,.2f}</td></tr>
            <tr><td>Min</td><td>{series.min():,.2f}</td></tr>
            <tr><td>Max</td><td>{series.max():,.2f}</td></tr>
        </table>
    </div>
    
    <div class="card">
        <h2>Forecast Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Date</th>
                    <th>Forecast</th>
                    <th>Lower Bound (95%)</th>
                    <th>Upper Bound (95%)</th>
                </tr>
            </thead>
            <tbody>
                {forecast_rows}
            </tbody>
        </table>
    </div>
</body>
</html>
"""
        return html