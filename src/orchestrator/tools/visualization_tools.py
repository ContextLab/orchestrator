"""Visualization tools for creating charts, plots, and dashboards."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import io
import base64
from datetime import datetime

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from .base import Tool


class VisualizationTool(Tool):
    """Tool for creating data visualizations and dashboards."""

    def __init__(self):
        super().__init__(
            name="visualization",
            description="Create charts, plots, and interactive dashboards from data",
        )
        self.add_parameter(
            "action",
            "string",
            "Action: 'create_charts', 'create_dashboard', 'export_figures'",
        )
        self.add_parameter("data", "object", "Input data (dict, list, or file path)", required=False)
        self.add_parameter(
            "chart_types",
            "array",
            "List of chart types to create",
            required=False,
            default=["auto"],
        )
        self.add_parameter(
            "output_dir",
            "string",
            "Output directory for charts",
            required=False,
            default="examples/outputs/modular_analysis/charts",
        )
        self.add_parameter(
            "title", "string", "Title for charts or dashboard", required=False
        )
        self.add_parameter(
            "layout",
            "string",
            "Dashboard layout: 'grid', 'vertical', 'horizontal'",
            required=False,
            default="grid",
        )
        self.add_parameter(
            "charts",
            "array",
            "List of chart files for dashboard creation",
            required=False,
        )
        self.add_parameter(
            "format",
            "string",
            "Output format: 'png', 'svg', 'pdf', 'html'",
            required=False,
            default="png",
        )
        self.add_parameter(
            "width", "integer", "Chart width in pixels", required=False, default=800
        )
        self.add_parameter(
            "height", "integer", "Chart height in pixels", required=False, default=600
        )
        self.add_parameter(
            "theme",
            "string",
            "Visual theme: 'default', 'dark', 'seaborn'",
            required=False,
            default="seaborn",
        )

        self.logger = logging.getLogger(__name__)

    def _load_data(self, data: Any) -> pd.DataFrame:
        """Load and convert data to pandas DataFrame."""
        if isinstance(data, pd.DataFrame):
            return data
        
        if isinstance(data, str):
            # Check if it looks like CSV content (has newlines and commas)
            if '\n' in data and ',' in data:
                # Treat as CSV string
                return pd.read_csv(io.StringIO(data))
            
            # Check if it's a file path (short string without newlines)
            if len(data) < 500 and '\n' not in data:
                try:
                    path = Path(data)
                    if path.exists():
                        if path.suffix == '.csv':
                            return pd.read_csv(path)
                        elif path.suffix == '.json':
                            with open(path, 'r') as f:
                                json_data = json.load(f)
                                return pd.DataFrame(json_data)
                except:
                    pass
            
            # Try to parse as JSON string
            try:
                json_data = json.loads(data)
                return pd.DataFrame(json_data)
            except:
                # Final fallback - treat as CSV
                return pd.read_csv(io.StringIO(data))
        
        if isinstance(data, dict):
            # Check if it's nested data that needs flattening
            if all(isinstance(v, dict) for v in data.values()):
                return pd.DataFrame.from_dict(data, orient='index')
            return pd.DataFrame([data])
        
        if isinstance(data, list):
            return pd.DataFrame(data)
        
        raise ValueError(f"Unsupported data type: {type(data)}")

    def _detect_chart_types(self, df: pd.DataFrame) -> List[str]:
        """Automatically detect appropriate chart types based on data."""
        chart_types = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Get categorical columns but exclude timestamp/date columns
        all_categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        categorical_cols = [col for col in all_categorical_cols 
                           if 'date' not in col.lower() and 'time' not in col.lower()]
        
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Time series chart if datetime column exists
        if datetime_cols and numeric_cols:
            chart_types.append('line')
        
        # Bar chart for categorical vs numeric
        if categorical_cols and numeric_cols:
            chart_types.append('bar')
        
        # Scatter plot for multiple numeric columns
        if len(numeric_cols) >= 2:
            chart_types.append('scatter')
        
        # Histogram for single numeric column
        if len(numeric_cols) >= 1:
            chart_types.append('histogram')
        
        # Pie chart if there's a categorical column with reasonable cardinality
        if categorical_cols and len(df[categorical_cols[0]].unique()) <= 10:
            chart_types.append('pie')
        
        # Heatmap for correlation if multiple numeric columns
        if len(numeric_cols) >= 3:
            chart_types.append('heatmap')
        
        return chart_types if chart_types else ['bar']  # Default to bar chart

    def _apply_theme(self, theme: str):
        """Apply visual theme to matplotlib."""
        if theme == 'seaborn':
            sns.set_theme()
        elif theme == 'dark':
            plt.style.use('dark_background')
        else:
            plt.style.use('default')

    async def _create_line_chart(
        self, df: pd.DataFrame, title: str, output_path: Path, **kwargs
    ) -> str:
        """Create a line chart."""
        # Use proper figure size (convert pixels to inches at 100 DPI)
        width_inches = kwargs.get('width', 1000) / 100
        height_inches = kwargs.get('height', 600) / 100
        
        # Plot numeric columns - select appropriate ones for time series
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Group columns by scale to avoid massive differences
        # Remove redundant columns (sales and units_sold are identical)
        small_scale_cols = ['conversion_rate', 'return_rate', 'discount_percentage']  # 0-1 scale
        medium_scale_cols = ['customer_satisfaction', 'sales', 'average_price']  # Different scales
        large_scale_cols = ['revenue', 'inventory_level', 'marketing_spend']
        
        # Find which columns exist in the data
        small_cols = [col for col in small_scale_cols if col in numeric_cols]
        medium_cols = [col for col in medium_scale_cols if col in numeric_cols]
        large_cols = [col for col in large_scale_cols if col in numeric_cols]
        
        # Choose columns and determine if we need normalization or dual axes
        if small_cols and 'customer_satisfaction' in numeric_cols:
            # We have both small scale (0-0.1) and customer satisfaction (1-5)
            # Normalize all to 0-1 scale for comparability
            cols_to_plot = small_cols[:2] + ['customer_satisfaction']
            chart_title = 'Performance Metrics Over Time (Normalized)'
            normalize = True
        elif small_cols and len(small_cols) >= 2:
            cols_to_plot = small_cols[:3]  # Rate metrics on same scale
            chart_title = 'Performance Rates Over Time'
            normalize = False
        elif medium_cols:
            if 'sales' in medium_cols:
                cols_to_plot = ['sales']  # Just sales
                chart_title = 'Sales Volume Over Time'
            else:
                cols_to_plot = medium_cols[:2]
                chart_title = 'Sales Metrics Over Time'
            normalize = False
        elif large_cols:
            cols_to_plot = large_cols[:3]  # Financial metrics
            chart_title = 'Financial Metrics Over Time'
            normalize = False
        else:
            cols_to_plot = list(numeric_cols[:3])
            chart_title = 'Metrics Over Time'
            normalize = False
        
        fig, ax = plt.subplots(figsize=(width_inches, height_inches))
        
        # Plot each column
        for i, col in enumerate(cols_to_plot):
            if normalize:
                # Normalize to 0-1 scale
                col_data = df[col].values
                min_val = col_data.min()
                max_val = col_data.max()
                if max_val > min_val:
                    normalized = (col_data - min_val) / (max_val - min_val)
                else:
                    normalized = col_data
                
                # Add original scale range to label
                if col == 'customer_satisfaction':
                    label = f'{col.replace("_", " ").title()} (1-5 scale)'
                elif col in ['conversion_rate', 'return_rate', 'discount_percentage']:
                    label = f'{col.replace("_", " ").title()} ({df[col].min():.1%}-{df[col].max():.1%})'
                else:
                    label = col.replace('_', ' ').title()
                
                if len(df) > 100:
                    ax.plot(df.index, normalized, label=label, linewidth=1.5, alpha=0.8)
                else:
                    ax.plot(df.index, normalized, label=label, marker='o', markersize=4, linewidth=1.5, alpha=0.8)
            else:
                # Plot without normalization
                if len(df) > 100:
                    ax.plot(df.index, df[col], label=col.replace('_', ' ').title(), linewidth=1.5, alpha=0.8)
                else:
                    ax.plot(df.index, df[col], label=col.replace('_', ' ').title(), marker='o', markersize=4, linewidth=1.5, alpha=0.8)
        
        # Check if index looks like dates/time
        x_label = 'Date' if 'date' in str(df.index.name).lower() or 'time' in str(df.index.name).lower() else 'Day'
        
        ax.set_title(chart_title, fontsize=14, fontweight='bold')
        ax.set_xlabel(x_label, fontsize=12)
        
        # Y-axis label depends on normalization
        if normalize:
            ax.set_ylabel('Normalized Value (0-1)', fontsize=12)
            ax.set_ylim(-0.05, 1.05)  # Add some padding
        else:
            ax.set_ylabel('Value', fontsize=12)
        
        # Position legend outside plot area but not overlapping
        ax.legend(bbox_to_anchor=(0.02, 0.98), loc='upper left', fontsize=9, framealpha=0.9)
        
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return str(output_path)

    async def _create_bar_chart(
        self, df: pd.DataFrame, title: str, output_path: Path, **kwargs
    ) -> str:
        """Create a bar chart."""
        width_inches = kwargs.get('width', 800) / 100
        height_inches = kwargs.get('height', 600) / 100
        fig, ax = plt.subplots(figsize=(width_inches, height_inches))
        
        # Get categorical columns but exclude timestamp/date columns
        all_categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        categorical_cols = [col for col in all_categorical_cols 
                           if 'date' not in col.lower() and 'time' not in col.lower()]
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Prioritize meaningful categorical columns
        priority_cat_cols = ['product_category', 'region', 'product_type', 'category']
        cat_col = None
        for col in priority_cat_cols:
            if col in categorical_cols:
                cat_col = col
                break
        if not cat_col and len(categorical_cols) > 0:
            cat_col = categorical_cols[0]
        
        # Prioritize meaningful numeric columns
        priority_num_cols = ['revenue', 'sales', 'units_sold', 'amount', 'value']
        num_col = None
        for col in priority_num_cols:
            if col in numeric_cols:
                num_col = col
                break
        if not num_col and len(numeric_cols) > 0:
            num_col = numeric_cols[0]
        
        if cat_col and num_col:
            # Group by categorical and sum numeric
            grouped = df.groupby(cat_col)[num_col].sum()
            grouped = grouped.nlargest(20)  # Limit to top 20 for readability
            grouped.plot(kind='bar', ax=ax, color='steelblue')
            ax.set_xlabel(cat_col.replace('_', ' ').title(), fontsize=12)
            ax.set_ylabel(f'Total {num_col.replace("_", " ").title()}', fontsize=12)
            
            # Format y-axis for large numbers
            import matplotlib.ticker as ticker
            if grouped.max() > 10000:
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x/1000:.0f}K' if num_col == 'revenue' else f'{x/1000:.0f}K'))
            elif grouped.max() > 1000:
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x:.0f}' if num_col == 'revenue' else f'{x:.0f}'))
        elif len(numeric_cols) > 0:
            # Use first numeric column
            df[numeric_cols[0]].plot(kind='bar', ax=ax, color='steelblue')
            ax.set_xlabel('Index', fontsize=12)
            ax.set_ylabel(numeric_cols[0], fontsize=12)
        else:
            # Fallback to first column
            df.iloc[:, 0].value_counts().plot(kind='bar', ax=ax, color='steelblue')
            ax.set_xlabel('Values', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
        
        # Create descriptive title based on what's being plotted
        if cat_col and num_col:
            chart_title = f'Total {num_col.replace("_", " ").title()} by {cat_col.replace("_", " ").title()}'
        else:
            chart_title = 'Category Distribution'
        
        ax.set_title(chart_title, fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return str(output_path)

    async def _create_scatter_chart(
        self, df: pd.DataFrame, title: str, output_path: Path, **kwargs
    ) -> str:
        """Create a scatter plot."""
        width_inches = kwargs.get('width', 800) / 100
        height_inches = kwargs.get('height', 600) / 100
        fig, ax = plt.subplots(figsize=(width_inches, height_inches))
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            ax.scatter(df[numeric_cols[0]], df[numeric_cols[1]], alpha=0.6)
            ax.set_xlabel(numeric_cols[0])
            ax.set_ylabel(numeric_cols[1])
        else:
            ax.scatter(df.index, df[numeric_cols[0] if numeric_cols else df.columns[0]], alpha=0.6)
            ax.set_xlabel('Index')
            ax.set_ylabel(numeric_cols[0] if numeric_cols else df.columns[0])
        
        # Create descriptive title
        if len(numeric_cols) >= 2:
            chart_title = f'{numeric_cols[1].replace("_", " ").title()} vs {numeric_cols[0].replace("_", " ").title()}'
        else:
            chart_title = f'{(numeric_cols[0] if numeric_cols else df.columns[0]).replace("_", " ").title()} Distribution'
        
        ax.set_title(chart_title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return str(output_path)

    async def _create_histogram(
        self, df: pd.DataFrame, title: str, output_path: Path, **kwargs
    ) -> str:
        """Create a histogram."""
        width_inches = kwargs.get('width', 800) / 100
        height_inches = kwargs.get('height', 600) / 100
        fig, ax = plt.subplots(figsize=(width_inches, height_inches))
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if numeric_cols.any():
            df[numeric_cols[0]].hist(ax=ax, bins=30, edgecolor='black', alpha=0.7)
            ax.set_xlabel(numeric_cols[0])
        else:
            df[df.columns[0]].value_counts().plot(kind='bar', ax=ax)
            ax.set_xlabel(df.columns[0])
        
        # Create descriptive title
        if numeric_cols.any():
            chart_title = f'Distribution of {numeric_cols[0].replace("_", " ").title()}'
        else:
            chart_title = f'Distribution of {df.columns[0].replace("_", " ").title()}'
        
        ax.set_title(chart_title, fontsize=14, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return str(output_path)

    async def _create_pie_chart(
        self, df: pd.DataFrame, title: str, output_path: Path, **kwargs
    ) -> str:
        """Create a pie chart."""
        width_inches = kwargs.get('width', 800) / 100
        height_inches = kwargs.get('height', 800) / 100
        fig, ax = plt.subplots(figsize=(width_inches, height_inches))
        
        # Get categorical columns but exclude timestamp/date columns
        all_categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        categorical_cols = [col for col in all_categorical_cols 
                           if 'date' not in col.lower() and 'time' not in col.lower()]
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Find the best categorical column for pie chart
        priority_cat_cols = ['product_category', 'region', 'category', 'type', 'status']
        cat_col_to_use = None
        
        for col in priority_cat_cols:
            if col in categorical_cols:
                cat_col_to_use = col
                break
        
        # If no priority column found, use first categorical column
        if cat_col_to_use is None and len(categorical_cols) > 0:
            # Skip date-like columns for pie charts
            for col in categorical_cols:
                if 'date' not in col.lower() and 'time' not in col.lower():
                    cat_col_to_use = col
                    break
        
        # Find best numeric column for values
        priority_num_cols = ['revenue', 'sales', 'amount', 'value', 'count']
        num_col_to_use = None
        
        for col in priority_num_cols:
            if col in numeric_cols:
                num_col_to_use = col
                break
        
        if num_col_to_use is None and len(numeric_cols) > 0:
            num_col_to_use = numeric_cols[0]
        
        if cat_col_to_use and num_col_to_use:
            # Group by categorical column and sum numeric values
            data = df.groupby(cat_col_to_use)[num_col_to_use].sum()
            data = data.nlargest(10)  # Limit to top 10 categories
        elif cat_col_to_use:
            # Use categorical column value counts
            data = df[cat_col_to_use].value_counts().head(10)
        else:
            # Fallback - don't use timestamp or date columns
            non_date_cols = [col for col in df.columns if 'date' not in col.lower() and 'time' not in col.lower()]
            if non_date_cols:
                data = df[non_date_cols[0]].value_counts().head(10)
            else:
                data = df[df.columns[0]].value_counts().head(10)
        
        # For pie charts, handle text overlap
        # Combine small slices into 'Other' category
        threshold = 0.02  # 2% threshold
        total = data.sum()
        small_slices = data[data / total < threshold]
        
        if len(small_slices) > 0:
            # Combine small slices into 'Other'
            other_value = small_slices.sum()
            data = data[data / total >= threshold]
            if other_value > 0:
                data['Other'] = other_value
        
        # Create pie chart with better text placement
        wedges, texts, autotexts = ax.pie(data.values, labels=None, autopct='%1.1f%%', 
                                           startangle=90, pctdistance=0.85)
        
        # Add legend to avoid label overlap
        ax.legend(wedges, data.index, 
                 title=cat_col_to_use.replace('_', ' ').title() if cat_col_to_use else 'Categories',
                 loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
                 fontsize=10)
        
        # Make percentage text smaller and only show for larger slices
        for i, autotext in enumerate(autotexts):
            if data.values[i] / total < 0.03:  # Hide percentage for slices < 3%
                autotext.set_text('')
            else:
                autotext.set_fontsize(9)
                autotext.set_color('white' if data.values[i] / total > 0.15 else 'black')
        
        # Create descriptive title
        if cat_col_to_use and num_col_to_use:
            chart_title = f'{num_col_to_use.replace("_", " ").title()} by {cat_col_to_use.replace("_", " ").title()}'
        elif cat_col_to_use:
            chart_title = f'Distribution of {cat_col_to_use.replace("_", " ").title()}'
        else:
            chart_title = 'Category Distribution'
        
        ax.set_title(chart_title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return str(output_path)

    async def _create_heatmap(
        self, df: pd.DataFrame, title: str, output_path: Path, **kwargs
    ) -> str:
        """Create a correlation heatmap."""
        width_inches = kwargs.get('width', 1000) / 100
        height_inches = kwargs.get('height', 800) / 100
        fig, ax = plt.subplots(figsize=(width_inches, height_inches))
        
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) >= 2:
            corr_matrix = numeric_df.corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, ax=ax, cbar_kws={'label': 'Correlation'})
            ax.set_title('Correlation Matrix of Numeric Variables', fontsize=14, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Insufficient numeric data for heatmap', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title or 'Heatmap', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return str(output_path)

    async def _create_dashboard_html(
        self, charts: List[str], title: str, layout: str, output_path: Path
    ) -> str:
        """Create an HTML dashboard with embedded charts."""
        # Always use simple HTML with embedded images - plotly interactive mode is broken
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title or 'Analysis Dashboard'}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                h1 {{ color: #333; text-align: center; }}
                .chart-container {{ 
                    display: {'grid' if layout == 'grid' else 'flex'};
                    {'grid-template-columns: repeat(2, 1fr);' if layout == 'grid' else ''}
                    {'flex-direction: column;' if layout == 'vertical' else ''}
                    gap: 20px;
                    margin-top: 20px;
                    max-width: 1400px;
                    margin: 0 auto;
                }}
                .chart {{ 
                    border: 1px solid #ddd;
                    padding: 15px;
                    background: white;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    border-radius: 8px;
                }}
                .chart h3 {{
                    margin-top: 0;
                    color: #555;
                    font-size: 1.1em;
                }}
                img {{ max-width: 100%; height: auto; display: block; }}
            </style>
        </head>
        <body>
            <h1>{title or 'Analysis Dashboard'}</h1>
            <div class="chart-container">
        """
        
        for chart_path in charts:
            if Path(chart_path).exists():
                # Embed image as base64
                with open(chart_path, 'rb') as f:
                    img_data = base64.b64encode(f.read()).decode()
                # Don't add redundant h3 title - the chart already has its title in the image
                html_content += f"""
                <div class="chart">
                    <img src="data:image/png;base64,{img_data}" alt="Chart">
                </div>
                """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Old plotly code removed - it was creating broken dashboards
        if False and PLOTLY_AVAILABLE:
            # Use plotly for interactive dashboard
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go
            
            # Create subplots based on layout
            if layout == 'grid':
                rows = (len(charts) + 1) // 2
                cols = 2
            elif layout == 'vertical':
                rows = len(charts)
                cols = 1
            else:  # horizontal
                rows = 1
                cols = len(charts)
            
            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=[Path(c).stem.replace('_', ' ').title() for c in charts]
            )
            
            # Add a simple placeholder chart for each position
            for idx, chart_path in enumerate(charts):
                row = (idx // cols) + 1 if layout == 'grid' else idx + 1 if layout == 'vertical' else 1
                col = (idx % cols) + 1 if layout == 'grid' else 1 if layout == 'vertical' else idx + 1
                
                # Add a placeholder trace
                fig.add_trace(
                    go.Scatter(x=[1, 2, 3], y=[1, 2, 3], 
                              mode='lines+markers',
                              name=Path(chart_path).stem),
                    row=row, col=col
                )
            
            fig.update_layout(
                title_text=title or 'Analysis Dashboard',
                showlegend=True,
                height=600 * rows,
                width=1200
            )
            
            html_content = fig.to_html(include_plotlyjs='cdn')
        
        # Write HTML file
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        return str(output_path)

    async def _execute_impl(self, **kwargs) -> Dict[str, Any]:
        """Execute visualization operation."""
        action = kwargs.get("action", "")
        
        if action == "create_charts":
            return await self._create_charts(**kwargs)
        elif action == "create_dashboard":
            return await self._create_dashboard(**kwargs)
        elif action == "export_figures":
            return await self._export_figures(**kwargs)
        else:
            raise ValueError(f"Unknown visualization action: {action}")

    async def _create_charts(self, **kwargs) -> Dict[str, Any]:
        """Create multiple charts from data."""
        data = kwargs.get("data")
        chart_types = kwargs.get("chart_types", ["auto"])
        output_dir = Path(kwargs.get("output_dir", "examples/outputs/modular_analysis/charts"))
        title = kwargs.get("title", "")
        theme = kwargs.get("theme", "seaborn")
        
        # Apply theme
        self._apply_theme(theme)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        df = self._load_data(data)
        
        # Detect chart types if auto
        if chart_types == ["auto"] or "auto" in chart_types:
            chart_types = self._detect_chart_types(df)
            self.logger.info(f"Auto-detected chart types: {chart_types}")
        
        # Create charts
        created_files = []
        chart_creators = {
            'line': self._create_line_chart,
            'bar': self._create_bar_chart,
            'scatter': self._create_scatter_chart,
            'histogram': self._create_histogram,
            'pie': self._create_pie_chart,
            'heatmap': self._create_heatmap,
        }
        
        for chart_type in chart_types:
            if chart_type in chart_creators:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = output_dir / f"{chart_type}_chart_{timestamp}.png"
                
                try:
                    # Remove title from kwargs if present to avoid duplicate
                    chart_kwargs = {k: v for k, v in kwargs.items() if k != 'title'}
                    file_path = await chart_creators[chart_type](
                        df, None,  # Let each chart creator determine its own descriptive title
                        output_path, **chart_kwargs
                    )
                    created_files.append(file_path)
                    self.logger.info(f"Created {chart_type} chart: {file_path}")
                except Exception as e:
                    self.logger.error(f"Failed to create {chart_type} chart: {e}")
        
        return {
            "action": "create_charts",
            "charts": created_files,
            "files": created_files,  # Alias for compatibility
            "count": len(created_files),
            "output_dir": str(output_dir),
            "success": len(created_files) > 0,
            "chart_types": chart_types,
        }

    async def _create_dashboard(self, **kwargs) -> Dict[str, Any]:
        """Create an interactive dashboard from charts."""
        charts = kwargs.get("charts", [])
        title = kwargs.get("title", "Analysis Dashboard")
        layout = kwargs.get("layout", "grid")
        output_dir = Path(kwargs.get("output_dir", "examples/outputs/modular_analysis"))
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure charts is a list
        if isinstance(charts, str):
            # If it's a string representation of a list, try to parse it
            if charts.startswith('[') and charts.endswith(']'):
                import ast
                try:
                    charts = ast.literal_eval(charts)
                except:
                    charts = []
            else:
                charts = [charts]  # Single chart path
        
        # If no charts provided, look for existing chart files
        if not charts:
            chart_dir = output_dir / "charts"
            if chart_dir.exists():
                charts = [str(p) for p in chart_dir.glob("*.png")]
        
        if not charts:
            return {
                "action": "create_dashboard",
                "success": False,
                "error": "No charts provided or found",
            }
        
        # Create dashboard HTML
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dashboard_path = output_dir / f"dashboard_{timestamp}.html"
        
        try:
            url = await self._create_dashboard_html(charts, title, layout, dashboard_path)
            
            return {
                "action": "create_dashboard",
                "url": url,
                "dashboard_path": url,
                "charts_included": len(charts),
                "layout": layout,
                "success": True,
            }
        except Exception as e:
            self.logger.error(f"Failed to create dashboard: {e}")
            return {
                "action": "create_dashboard",
                "success": False,
                "error": str(e),
            }

    async def _export_figures(self, **kwargs) -> Dict[str, Any]:
        """Export figures in multiple formats."""
        charts = kwargs.get("charts", [])
        formats = kwargs.get("format", "png")
        if isinstance(formats, str):
            formats = [formats]
        
        output_dir = Path(kwargs.get("output_dir", "examples/outputs/modular_analysis/exports"))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exported_files = []
        
        for chart_path in charts:
            if not Path(chart_path).exists():
                continue
            
            # For now, just copy the files to export directory
            # In a full implementation, we'd convert between formats
            for fmt in formats:
                export_path = output_dir / f"{Path(chart_path).stem}.{fmt}"
                if fmt == 'png' and chart_path.endswith('.png'):
                    import shutil
                    shutil.copy2(chart_path, export_path)
                    exported_files.append(str(export_path))
        
        return {
            "action": "export_figures",
            "exported_files": exported_files,
            "formats": formats,
            "count": len(exported_files),
            "success": len(exported_files) > 0,
        }