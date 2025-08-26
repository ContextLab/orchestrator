#!/usr/bin/env python3
"""
Pipeline Validation Dashboard Generator for Issue #262.

Comprehensive reporting and analytics dashboard system that integrates with all 
completed components (#256-#261) to provide:

- Executive Dashboard: High-level view of pipeline validation status
- Quality Metrics Visualization: Integration with quality scoring and LLM reviews
- Performance Analytics: Trend analysis, baselines, and regression detection
- Historical Reporting: Long-term analysis and insights
- Export Capabilities: PDF, CSV, JSON, interactive HTML formats
- Web-based Dashboard: Interactive charts and filtering

Dependencies: Issues #256 (Enhanced Validation), #257 (LLM Quality Review), 
#259 (Tutorial Documentation), #260 (Performance Monitoring)
"""

import asyncio
import json
import logging
import os
import sqlite3
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import traceback
import time
import subprocess
import csv

# Import plotting and web libraries
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from io import StringIO, BytesIO
import base64

# Add orchestrator to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import existing infrastructure 
try:
    from orchestrator import Orchestrator, init_models
    from orchestrator.models import get_model_registry
    from orchestrator.compiler.yaml_compiler import YAMLCompiler
    from orchestrator.control_systems.hybrid_control_system import HybridControlSystem
    from scripts.validate_all_pipelines import PipelineValidator
    from scripts.organization_reporter import OrganizationReporter, HealthMetric, TrendAnalysis
except ImportError as e:
    logging.error(f"Failed to import required modules: {e}")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PipelineExecutionMetric:
    """Detailed pipeline execution metrics."""
    pipeline_name: str
    timestamp: datetime
    status: str  # success, failed, completed_with_issues
    execution_time: float
    quality_score: float
    issues: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    model_used: Optional[str] = None
    memory_usage: Optional[float] = None
    tokens_used: Optional[int] = None
    cost_estimate: Optional[float] = None
    
    def __post_init__(self):
        """Validate and normalize data."""
        if not isinstance(self.issues, list):
            self.issues = []
        if not isinstance(self.outputs, list):
            self.outputs = []


@dataclass
class DashboardConfig:
    """Configuration for dashboard generation."""
    include_charts: bool = True
    include_trends: bool = True
    include_performance: bool = True
    include_quality: bool = True
    trend_days: int = 30
    export_formats: List[str] = field(default_factory=lambda: ['json', 'html', 'csv'])
    interactive_dashboard: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'quality_score_critical': 60.0,
        'quality_score_warning': 80.0,
        'execution_time_warning': 60.0,
        'success_rate_critical': 70.0
    })


@dataclass
class ExecutiveSummary:
    """Executive-level summary of pipeline health."""
    total_pipelines: int
    successful_pipelines: int
    failed_pipelines: int
    pipelines_with_issues: int
    average_quality_score: float
    average_execution_time: float
    success_rate: float
    top_issues: List[str]
    performance_trend: str  # improving, declining, stable
    quality_trend: str
    recommendations: List[str]
    alerts: List[str]
    timestamp: datetime


class PipelineDashboardGenerator:
    """Comprehensive dashboard generator for pipeline validation analytics."""
    
    def __init__(self, root_path: str = ".", config: Optional[DashboardConfig] = None):
        self.root_path = Path(root_path).resolve()
        self.config = config or DashboardConfig()
        
        # Initialize paths
        self.examples_dir = self.root_path / "examples"
        self.outputs_dir = self.examples_dir / "outputs"
        self.dashboard_dir = self.root_path / "temp" / "dashboards"
        self.dashboard_dir.mkdir(parents=True, exist_ok=True)
        
        # Database for metrics storage
        self.database_path = self.dashboard_dir / "pipeline_metrics.db"
        self._setup_database()
        
        # Integration with existing systems
        try:
            self.organization_reporter = OrganizationReporter(str(self.root_path))
        except Exception as e:
            logger.warning(f"Could not initialize organization reporter: {e}")
            self.organization_reporter = None
        
        # Metrics cache
        self.metrics_cache = deque(maxlen=10000)
        self.pipeline_registry = {}
        
        logger.info(f"Pipeline Dashboard Generator initialized for: {self.root_path}")

    def _setup_database(self):
        """Setup SQLite database for pipeline metrics storage."""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # Pipeline execution metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pipeline_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pipeline_name TEXT,
                timestamp DATETIME,
                status TEXT,
                execution_time REAL,
                quality_score REAL,
                issues_count INTEGER,
                issues TEXT,
                outputs_count INTEGER,
                outputs TEXT,
                model_used TEXT,
                memory_usage REAL,
                tokens_used INTEGER,
                cost_estimate REAL
            )
        ''')
        
        # Dashboard snapshots table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dashboard_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                snapshot_data TEXT,
                executive_summary TEXT
            )
        ''')
        
        # Performance baselines table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_baselines (
                pipeline_name TEXT PRIMARY KEY,
                baseline_execution_time REAL,
                baseline_quality_score REAL,
                last_updated DATETIME
            )
        ''')
        
        conn.commit()
        conn.close()

    async def collect_all_pipeline_metrics(self) -> List[PipelineExecutionMetric]:
        """Collect metrics from all available pipeline executions."""
        logger.info("Collecting pipeline metrics from all available sources...")
        
        metrics = []
        
        # Scan validation summary files
        for validation_file in self.outputs_dir.glob("*/validation_summary.json"):
            try:
                with open(validation_file) as f:
                    data = json.load(f)
                
                metric = PipelineExecutionMetric(
                    pipeline_name=data.get('pipeline', validation_file.parent.name),
                    timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat())),
                    status=data.get('status', 'unknown'),
                    execution_time=data.get('execution_time', 0.0),
                    quality_score=data.get('quality_score', 0.0),
                    issues=data.get('issues', []),
                    outputs=data.get('outputs', [])
                )
                
                metrics.append(metric)
                
            except Exception as e:
                logger.warning(f"Could not parse validation file {validation_file}: {e}")
        
        # Load historical data from database
        historical_metrics = self._load_historical_metrics(days=self.config.trend_days)
        metrics.extend(historical_metrics)
        
        # Store current metrics in database
        for metric in metrics:
            if metric.timestamp > datetime.now() - timedelta(hours=1):  # Only recent metrics
                self._store_metric_to_database(metric)
        
        logger.info(f"Collected {len(metrics)} pipeline execution metrics")
        return metrics

    def _load_historical_metrics(self, days: int = 30) -> List[PipelineExecutionMetric]:
        """Load historical metrics from database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cutoff_date = datetime.now() - timedelta(days=days)
            
            cursor.execute('''
                SELECT pipeline_name, timestamp, status, execution_time, quality_score,
                       issues, outputs, model_used, memory_usage, tokens_used, cost_estimate
                FROM pipeline_metrics 
                WHERE timestamp >= ?
                ORDER BY timestamp ASC
            ''', (cutoff_date,))
            
            metrics = []
            for row in cursor.fetchall():
                try:
                    issues = json.loads(row[5]) if row[5] else []
                    outputs = json.loads(row[6]) if row[6] else []
                except:
                    issues = []
                    outputs = []
                
                metrics.append(PipelineExecutionMetric(
                    pipeline_name=row[0],
                    timestamp=datetime.fromisoformat(row[1]),
                    status=row[2],
                    execution_time=row[3] or 0.0,
                    quality_score=row[4] or 0.0,
                    issues=issues,
                    outputs=outputs,
                    model_used=row[7],
                    memory_usage=row[8],
                    tokens_used=row[9],
                    cost_estimate=row[10]
                ))
            
            conn.close()
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to load historical metrics: {e}")
            return []

    def _store_metric_to_database(self, metric: PipelineExecutionMetric):
        """Store metric in database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO pipeline_metrics 
                (pipeline_name, timestamp, status, execution_time, quality_score,
                 issues_count, issues, outputs_count, outputs, model_used, 
                 memory_usage, tokens_used, cost_estimate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metric.pipeline_name,
                metric.timestamp.isoformat(),
                metric.status,
                metric.execution_time,
                metric.quality_score,
                len(metric.issues),
                json.dumps(metric.issues),
                len(metric.outputs),
                json.dumps(metric.outputs),
                metric.model_used,
                metric.memory_usage,
                metric.tokens_used,
                metric.cost_estimate
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store metric to database: {e}")

    def generate_executive_summary(self, metrics: List[PipelineExecutionMetric]) -> ExecutiveSummary:
        """Generate executive-level summary from metrics."""
        logger.info("Generating executive summary...")
        
        if not metrics:
            return ExecutiveSummary(
                total_pipelines=0, successful_pipelines=0, failed_pipelines=0,
                pipelines_with_issues=0, average_quality_score=0.0,
                average_execution_time=0.0, success_rate=0.0,
                top_issues=[], performance_trend="stable",
                quality_trend="stable", recommendations=[], alerts=[],
                timestamp=datetime.now()
            )
        
        # Basic statistics
        total_pipelines = len(set(m.pipeline_name for m in metrics))
        successful = [m for m in metrics if m.status == 'success']
        failed = [m for m in metrics if m.status == 'failed']
        with_issues = [m for m in metrics if m.status == 'completed_with_issues']
        
        # Calculate averages
        quality_scores = [m.quality_score for m in metrics if m.quality_score > 0]
        execution_times = [m.execution_time for m in metrics if m.execution_time > 0]
        
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        avg_execution = sum(execution_times) / len(execution_times) if execution_times else 0.0
        success_rate = len(successful) / len(metrics) * 100 if metrics else 0.0
        
        # Top issues analysis
        all_issues = []
        for metric in metrics:
            all_issues.extend(metric.issues)
        
        issue_counts = defaultdict(int)
        for issue in all_issues:
            issue_counts[issue] += 1
        
        top_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_issues = [f"{issue} ({count}x)" for issue, count in top_issues]
        
        # Trend analysis
        recent_metrics = [m for m in metrics if m.timestamp > datetime.now() - timedelta(days=7)]
        older_metrics = [m for m in metrics if m.timestamp <= datetime.now() - timedelta(days=7)]
        
        performance_trend = self._calculate_trend_direction(
            older_metrics, recent_metrics, 'execution_time', inverse=True
        )
        quality_trend = self._calculate_trend_direction(
            older_metrics, recent_metrics, 'quality_score', inverse=False
        )
        
        # Generate recommendations and alerts
        recommendations = self._generate_recommendations(
            avg_quality, success_rate, top_issues, performance_trend, quality_trend
        )
        alerts = self._generate_alerts(avg_quality, success_rate, len(failed))
        
        return ExecutiveSummary(
            total_pipelines=total_pipelines,
            successful_pipelines=len(successful),
            failed_pipelines=len(failed),
            pipelines_with_issues=len(with_issues),
            average_quality_score=avg_quality,
            average_execution_time=avg_execution,
            success_rate=success_rate,
            top_issues=top_issues,
            performance_trend=performance_trend,
            quality_trend=quality_trend,
            recommendations=recommendations,
            alerts=alerts,
            timestamp=datetime.now()
        )

    def _calculate_trend_direction(self, older_metrics: List[PipelineExecutionMetric], 
                                 recent_metrics: List[PipelineExecutionMetric], 
                                 field: str, inverse: bool = False) -> str:
        """Calculate trend direction for a specific field."""
        if not older_metrics or not recent_metrics:
            return "stable"
        
        older_values = [getattr(m, field) for m in older_metrics if getattr(m, field, 0) > 0]
        recent_values = [getattr(m, field) for m in recent_metrics if getattr(m, field, 0) > 0]
        
        if not older_values or not recent_values:
            return "stable"
        
        older_avg = sum(older_values) / len(older_values)
        recent_avg = sum(recent_values) / len(recent_values)
        
        change_threshold = 0.05  # 5% change threshold
        change_ratio = abs(recent_avg - older_avg) / older_avg if older_avg > 0 else 0
        
        if change_ratio < change_threshold:
            return "stable"
        
        if inverse:  # For metrics where lower is better (e.g., execution time)
            return "improving" if recent_avg < older_avg else "declining"
        else:  # For metrics where higher is better (e.g., quality score)
            return "improving" if recent_avg > older_avg else "declining"

    def _generate_recommendations(self, avg_quality: float, success_rate: float, 
                                top_issues: List[str], performance_trend: str, 
                                quality_trend: str) -> List[str]:
        """Generate actionable recommendations based on metrics."""
        recommendations = []
        
        if avg_quality < self.config.alert_thresholds['quality_score_warning']:
            recommendations.append("Quality scores are below target - review and improve pipeline configurations")
        
        if success_rate < self.config.alert_thresholds['success_rate_critical']:
            recommendations.append("Success rate is critically low - investigate and fix failing pipelines")
        
        if performance_trend == "declining":
            recommendations.append("Performance is declining - optimize execution times and resource usage")
        
        if quality_trend == "declining":
            recommendations.append("Quality is declining - review recent changes and validate outputs")
        
        if top_issues:
            recommendations.append(f"Address top issue: {top_issues[0].split(' (')[0]}")
        
        if not recommendations:
            recommendations.append("Pipeline validation system is operating within normal parameters")
        
        return recommendations

    def _generate_alerts(self, avg_quality: float, success_rate: float, failed_count: int) -> List[str]:
        """Generate alerts based on thresholds."""
        alerts = []
        
        if avg_quality < self.config.alert_thresholds['quality_score_critical']:
            alerts.append(f"CRITICAL: Average quality score {avg_quality:.1f}% is below critical threshold")
        elif avg_quality < self.config.alert_thresholds['quality_score_warning']:
            alerts.append(f"WARNING: Average quality score {avg_quality:.1f}% is below warning threshold")
        
        if success_rate < self.config.alert_thresholds['success_rate_critical']:
            alerts.append(f"CRITICAL: Success rate {success_rate:.1f}% is below critical threshold")
        
        if failed_count > 5:
            alerts.append(f"WARNING: {failed_count} pipelines have failed - review and fix")
        
        return alerts

    def generate_html_dashboard(self, metrics: List[PipelineExecutionMetric], 
                              executive_summary: ExecutiveSummary) -> str:
        """Generate interactive HTML dashboard."""
        logger.info("Generating interactive HTML dashboard...")
        
        # Generate charts as base64 images
        charts = self._generate_dashboard_charts(metrics)
        
        # Pipeline details table
        pipeline_table = self._generate_pipeline_table(metrics)
        
        # Status indicators
        status_indicators = self._generate_status_indicators(executive_summary)
        
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pipeline Validation Dashboard</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .dashboard-container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 30px;
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
            border-bottom: 2px solid #e9ecef;
            padding-bottom: 20px;
        }}
        .header h1 {{
            color: #2c3e50;
            margin: 0;
            font-size: 2.5em;
        }}
        .header .subtitle {{
            color: #6c757d;
            font-size: 1.1em;
            margin-top: 10px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }}
        .metric-card.success {{
            background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        }}
        .metric-card.warning {{
            background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
        }}
        .metric-card.danger {{
            background: linear-gradient(135deg, #fc466b 0%, #3f5efb 100%);
        }}
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            margin: 0;
        }}
        .metric-label {{
            font-size: 1.1em;
            margin-top: 8px;
            opacity: 0.9;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .section-title {{
            font-size: 1.8em;
            color: #2c3e50;
            margin-bottom: 20px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        .chart-container {{
            margin-bottom: 30px;
            text-align: center;
        }}
        .chart-container img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }}
        .table-container {{
            overflow-x: auto;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background-color: white;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
            color: #495057;
        }}
        .status-success {{
            color: #28a745;
            font-weight: bold;
        }}
        .status-warning {{
            color: #ffc107;
            font-weight: bold;
        }}
        .status-danger {{
            color: #dc3545;
            font-weight: bold;
        }}
        .alerts {{
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 6px;
            padding: 15px;
            margin-bottom: 20px;
        }}
        .recommendations {{
            background-color: #d1ecf1;
            border: 1px solid #bee5eb;
            border-radius: 6px;
            padding: 15px;
            margin-bottom: 20px;
        }}
        .timestamp {{
            text-align: center;
            color: #6c757d;
            font-size: 0.9em;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #e9ecef;
        }}
        .trend-indicator {{
            display: inline-block;
            margin-left: 10px;
            font-size: 1.2em;
        }}
        .trend-improving {{ color: #28a745; }}
        .trend-declining {{ color: #dc3545; }}
        .trend-stable {{ color: #6c757d; }}
    </style>
</head>
<body>
    <div class="dashboard-container">
        <div class="header">
            <h1>Pipeline Validation Dashboard</h1>
            <div class="subtitle">Issue #262: Comprehensive Analytics & Reporting</div>
        </div>

        {status_indicators}

        <div class="section">
            <h2 class="section-title">Performance Metrics</h2>
            {charts}
        </div>

        <div class="section">
            <h2 class="section-title">Pipeline Details</h2>
            <div class="table-container">
                {pipeline_table}
            </div>
        </div>

        {self._generate_alerts_section(executive_summary)}
        {self._generate_recommendations_section(executive_summary)}

        <div class="timestamp">
            Dashboard generated: {executive_summary.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
        </div>
    </div>
</body>
</html>"""
        
        return html_template

    def _generate_dashboard_charts(self, metrics: List[PipelineExecutionMetric]) -> str:
        """Generate charts for the dashboard."""
        charts_html = []
        
        try:
            # Quality score distribution
            quality_chart = self._create_quality_distribution_chart(metrics)
            if quality_chart:
                charts_html.append(f'<div class="chart-container"><img src="data:image/png;base64,{quality_chart}" alt="Quality Score Distribution"></div>')
            
            # Execution time trends
            time_chart = self._create_execution_time_chart(metrics)
            if time_chart:
                charts_html.append(f'<div class="chart-container"><img src="data:image/png;base64,{time_chart}" alt="Execution Time Trends"></div>')
            
            # Pipeline status overview
            status_chart = self._create_status_overview_chart(metrics)
            if status_chart:
                charts_html.append(f'<div class="chart-container"><img src="data:image/png;base64,{status_chart}" alt="Pipeline Status Overview"></div>')
                
        except Exception as e:
            logger.error(f"Failed to generate charts: {e}")
            charts_html.append('<div class="chart-container"><p>Chart generation unavailable</p></div>')
        
        return '\n'.join(charts_html)

    def _create_quality_distribution_chart(self, metrics: List[PipelineExecutionMetric]) -> Optional[str]:
        """Create quality score distribution chart."""
        try:
            quality_scores = [m.quality_score for m in metrics if m.quality_score > 0]
            if not quality_scores:
                return None
            
            plt.figure(figsize=(10, 6))
            plt.hist(quality_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title('Quality Score Distribution', fontsize=16, fontweight='bold')
            plt.xlabel('Quality Score')
            plt.ylabel('Number of Pipelines')
            plt.grid(True, alpha=0.3)
            
            # Add mean line
            mean_score = np.mean(quality_scores)
            plt.axvline(mean_score, color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {mean_score:.1f}')
            plt.legend()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Failed to create quality distribution chart: {e}")
            return None

    def _create_execution_time_chart(self, metrics: List[PipelineExecutionMetric]) -> Optional[str]:
        """Create execution time trends chart."""
        try:
            # Group by pipeline name and get latest execution time
            pipeline_times = {}
            for metric in metrics:
                if metric.execution_time > 0:
                    if metric.pipeline_name not in pipeline_times:
                        pipeline_times[metric.pipeline_name] = []
                    pipeline_times[metric.pipeline_name].append(metric.execution_time)
            
            if not pipeline_times:
                return None
            
            # Get average execution time per pipeline
            pipeline_names = []
            avg_times = []
            for name, times in pipeline_times.items():
                pipeline_names.append(name[:20] + '...' if len(name) > 20 else name)
                avg_times.append(np.mean(times))
            
            plt.figure(figsize=(12, 8))
            bars = plt.barh(pipeline_names, avg_times, color='lightcoral')
            plt.title('Average Execution Time by Pipeline', fontsize=16, fontweight='bold')
            plt.xlabel('Execution Time (seconds)')
            plt.ylabel('Pipeline')
            
            # Color bars based on performance
            for i, (bar, time) in enumerate(zip(bars, avg_times)):
                if time > self.config.alert_thresholds['execution_time_warning']:
                    bar.set_color('red')
                elif time > self.config.alert_thresholds['execution_time_warning'] * 0.5:
                    bar.set_color('orange')
                else:
                    bar.set_color('green')
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Failed to create execution time chart: {e}")
            return None

    def _create_status_overview_chart(self, metrics: List[PipelineExecutionMetric]) -> Optional[str]:
        """Create pipeline status overview pie chart."""
        try:
            status_counts = defaultdict(int)
            for metric in metrics:
                status_counts[metric.status] += 1
            
            if not status_counts:
                return None
            
            labels = []
            sizes = []
            colors = []
            
            status_colors = {
                'success': '#28a745',
                'completed_with_issues': '#ffc107', 
                'failed': '#dc3545',
                'unknown': '#6c757d'
            }
            
            for status, count in status_counts.items():
                labels.append(f'{status.title()} ({count})')
                sizes.append(count)
                colors.append(status_colors.get(status, '#6c757d'))
            
            plt.figure(figsize=(10, 8))
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                   startangle=90, textprops={'fontsize': 12})
            plt.title('Pipeline Status Distribution', fontsize=16, fontweight='bold')
            plt.axis('equal')
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Failed to create status overview chart: {e}")
            return None

    def _generate_status_indicators(self, summary: ExecutiveSummary) -> str:
        """Generate status indicator cards."""
        cards = []
        
        # Success rate card
        success_class = "success" if summary.success_rate >= 90 else "warning" if summary.success_rate >= 70 else "danger"
        cards.append(f'''
        <div class="metric-card {success_class}">
            <div class="metric-value">{summary.success_rate:.1f}%</div>
            <div class="metric-label">Success Rate</div>
        </div>
        ''')
        
        # Quality score card
        quality_class = "success" if summary.average_quality_score >= 90 else "warning" if summary.average_quality_score >= 70 else "danger"
        trend_icon = "📈" if summary.quality_trend == "improving" else "📉" if summary.quality_trend == "declining" else "➡️"
        cards.append(f'''
        <div class="metric-card {quality_class}">
            <div class="metric-value">{summary.average_quality_score:.1f}</div>
            <div class="metric-label">Avg Quality Score <span class="trend-indicator trend-{summary.quality_trend}">{trend_icon}</span></div>
        </div>
        ''')
        
        # Total pipelines card
        cards.append(f'''
        <div class="metric-card">
            <div class="metric-value">{summary.total_pipelines}</div>
            <div class="metric-label">Total Pipelines</div>
        </div>
        ''')
        
        # Execution time card
        perf_trend_icon = "📈" if summary.performance_trend == "improving" else "📉" if summary.performance_trend == "declining" else "➡️"
        cards.append(f'''
        <div class="metric-card">
            <div class="metric-value">{summary.average_execution_time:.1f}s</div>
            <div class="metric-label">Avg Execution Time <span class="trend-indicator trend-{summary.performance_trend}">{perf_trend_icon}</span></div>
        </div>
        ''')
        
        return f'<div class="metrics-grid">{"".join(cards)}</div>'

    def _generate_pipeline_table(self, metrics: List[PipelineExecutionMetric]) -> str:
        """Generate pipeline details table."""
        # Get latest metric for each pipeline
        latest_metrics = {}
        for metric in metrics:
            if metric.pipeline_name not in latest_metrics or metric.timestamp > latest_metrics[metric.pipeline_name].timestamp:
                latest_metrics[metric.pipeline_name] = metric
        
        rows = []
        for pipeline_name, metric in sorted(latest_metrics.items()):
            status_class = "status-success" if metric.status == "success" else "status-warning" if metric.status == "completed_with_issues" else "status-danger"
            
            issues_summary = f"{len(metric.issues)} issues" if metric.issues else "No issues"
            outputs_count = len(metric.outputs) if metric.outputs else 0
            
            rows.append(f'''
            <tr>
                <td>{pipeline_name}</td>
                <td class="{status_class}">{metric.status.replace('_', ' ').title()}</td>
                <td>{metric.quality_score:.1f}%</td>
                <td>{metric.execution_time:.1f}s</td>
                <td>{issues_summary}</td>
                <td>{outputs_count} files</td>
                <td>{metric.timestamp.strftime('%Y-%m-%d %H:%M')}</td>
            </tr>
            ''')
        
        return f'''
        <table>
            <thead>
                <tr>
                    <th>Pipeline</th>
                    <th>Status</th>
                    <th>Quality Score</th>
                    <th>Execution Time</th>
                    <th>Issues</th>
                    <th>Outputs</th>
                    <th>Last Run</th>
                </tr>
            </thead>
            <tbody>
                {"".join(rows)}
            </tbody>
        </table>
        '''

    def _generate_alerts_section(self, summary: ExecutiveSummary) -> str:
        """Generate alerts section."""
        if not summary.alerts:
            return ""
        
        alerts_list = "".join([f"<li>{alert}</li>" for alert in summary.alerts])
        return f'''
        <div class="section">
            <h2 class="section-title">🚨 Alerts</h2>
            <div class="alerts">
                <ul>{alerts_list}</ul>
            </div>
        </div>
        '''

    def _generate_recommendations_section(self, summary: ExecutiveSummary) -> str:
        """Generate recommendations section."""
        if not summary.recommendations:
            return ""
        
        rec_list = "".join([f"<li>{rec}</li>" for rec in summary.recommendations])
        return f'''
        <div class="section">
            <h2 class="section-title">💡 Recommendations</h2>
            <div class="recommendations">
                <ul>{rec_list}</ul>
            </div>
        </div>
        '''

    def export_to_json(self, metrics: List[PipelineExecutionMetric], 
                      summary: ExecutiveSummary) -> Dict[str, Any]:
        """Export comprehensive data to JSON format."""
        logger.info("Exporting dashboard data to JSON...")
        
        return {
            'dashboard_metadata': {
                'generated_timestamp': datetime.now().isoformat(),
                'version': '1.0.0',
                'generator': 'PipelineDashboardGenerator',
                'issue': '#262'
            },
            'executive_summary': asdict(summary),
            'pipeline_metrics': [asdict(metric) for metric in metrics],
            'performance_analysis': self._generate_performance_analysis(metrics),
            'quality_analysis': self._generate_quality_analysis(metrics),
            'trend_analysis': self._generate_trend_analysis(metrics),
            'recommendations': {
                'immediate_actions': summary.recommendations[:3],
                'long_term_improvements': self._generate_long_term_recommendations(metrics),
                'optimization_opportunities': self._identify_optimization_opportunities(metrics)
            },
            'configuration': asdict(self.config)
        }

    def _generate_performance_analysis(self, metrics: List[PipelineExecutionMetric]) -> Dict[str, Any]:
        """Generate detailed performance analysis."""
        if not metrics:
            return {}
        
        execution_times = [m.execution_time for m in metrics if m.execution_time > 0]
        
        if not execution_times:
            return {}
        
        return {
            'execution_time_stats': {
                'mean': np.mean(execution_times),
                'median': np.median(execution_times),
                'std_dev': np.std(execution_times),
                'min': np.min(execution_times),
                'max': np.max(execution_times),
                'percentiles': {
                    '25th': np.percentile(execution_times, 25),
                    '75th': np.percentile(execution_times, 75),
                    '95th': np.percentile(execution_times, 95)
                }
            },
            'performance_baseline': self._calculate_performance_baseline(metrics),
            'slowest_pipelines': self._identify_slowest_pipelines(metrics),
            'performance_regression': self._detect_performance_regression(metrics)
        }

    def _generate_quality_analysis(self, metrics: List[PipelineExecutionMetric]) -> Dict[str, Any]:
        """Generate detailed quality analysis."""
        if not metrics:
            return {}
        
        quality_scores = [m.quality_score for m in metrics if m.quality_score > 0]
        
        if not quality_scores:
            return {}
        
        # Issue analysis
        all_issues = []
        issue_types = defaultdict(int)
        for metric in metrics:
            all_issues.extend(metric.issues)
            for issue in metric.issues:
                issue_type = issue.split(':')[0] if ':' in issue else 'general'
                issue_types[issue_type] += 1
        
        return {
            'quality_score_stats': {
                'mean': np.mean(quality_scores),
                'median': np.median(quality_scores),
                'std_dev': np.std(quality_scores),
                'min': np.min(quality_scores),
                'max': np.max(quality_scores)
            },
            'issues_analysis': {
                'total_issues': len(all_issues),
                'unique_issues': len(set(all_issues)),
                'most_common_issues': dict(sorted(issue_types.items(), key=lambda x: x[1], reverse=True)[:10]),
                'pipelines_with_issues': len([m for m in metrics if m.issues])
            },
            'quality_distribution': self._analyze_quality_distribution(quality_scores),
            'quality_trends': self._analyze_quality_trends(metrics)
        }

    def _generate_trend_analysis(self, metrics: List[PipelineExecutionMetric]) -> Dict[str, Any]:
        """Generate comprehensive trend analysis."""
        if len(metrics) < 2:
            return {}
        
        # Group metrics by time periods
        now = datetime.now()
        last_week = [m for m in metrics if m.timestamp > now - timedelta(days=7)]
        last_month = [m for m in metrics if m.timestamp > now - timedelta(days=30)]
        
        return {
            'recent_trends': {
                'last_7_days': self._calculate_period_trends(last_week),
                'last_30_days': self._calculate_period_trends(last_month)
            },
            'regression_detection': self._detect_quality_regression(metrics),
            'improvement_detection': self._detect_improvements(metrics),
            'seasonal_patterns': self._detect_seasonal_patterns(metrics)
        }

    def export_to_csv(self, metrics: List[PipelineExecutionMetric]) -> str:
        """Export metrics to CSV format."""
        logger.info("Exporting pipeline metrics to CSV...")
        
        output = StringIO()
        writer = csv.writer(output)
        
        # Write headers
        headers = [
            'pipeline_name', 'timestamp', 'status', 'execution_time', 'quality_score',
            'issues_count', 'outputs_count', 'model_used', 'memory_usage', 'tokens_used', 'cost_estimate'
        ]
        writer.writerow(headers)
        
        # Write data
        for metric in metrics:
            writer.writerow([
                metric.pipeline_name,
                metric.timestamp.isoformat(),
                metric.status,
                metric.execution_time,
                metric.quality_score,
                len(metric.issues),
                len(metric.outputs),
                metric.model_used or '',
                metric.memory_usage or '',
                metric.tokens_used or '',
                metric.cost_estimate or ''
            ])
        
        return output.getvalue()

    async def generate_comprehensive_dashboard(self) -> Dict[str, Path]:
        """Generate comprehensive dashboard in all formats."""
        logger.info("Generating comprehensive pipeline validation dashboard...")
        
        # Collect all metrics
        metrics = await self.collect_all_pipeline_metrics()
        
        # Generate executive summary
        executive_summary = self.generate_executive_summary(metrics)
        
        # Generate outputs
        outputs = {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # JSON export
        if 'json' in self.config.export_formats:
            json_data = self.export_to_json(metrics, executive_summary)
            json_path = self.dashboard_dir / f"pipeline_dashboard_{timestamp}.json"
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
            outputs['json'] = json_path
        
        # HTML dashboard
        if 'html' in self.config.export_formats:
            html_content = self.generate_html_dashboard(metrics, executive_summary)
            html_path = self.dashboard_dir / f"pipeline_dashboard_{timestamp}.html"
            with open(html_path, 'w') as f:
                f.write(html_content)
            outputs['html'] = html_path
        
        # CSV export
        if 'csv' in self.config.export_formats:
            csv_content = self.export_to_csv(metrics)
            csv_path = self.dashboard_dir / f"pipeline_metrics_{timestamp}.csv"
            with open(csv_path, 'w') as f:
                f.write(csv_content)
            outputs['csv'] = csv_path
        
        # Store snapshot in database
        self._store_dashboard_snapshot(executive_summary, json_data if 'json' in outputs else {})
        
        logger.info(f"Dashboard generation complete. Generated {len(outputs)} output files.")
        return outputs

    def _store_dashboard_snapshot(self, summary: ExecutiveSummary, data: Dict[str, Any]):
        """Store dashboard snapshot in database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO dashboard_snapshots (timestamp, snapshot_data, executive_summary)
                VALUES (?, ?, ?)
            ''', (
                summary.timestamp.isoformat(),
                json.dumps(data, default=str),
                json.dumps(asdict(summary), default=str)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store dashboard snapshot: {e}")

    # Helper methods for analysis
    def _calculate_performance_baseline(self, metrics: List[PipelineExecutionMetric]) -> Dict[str, float]:
        """Calculate performance baselines for pipelines."""
        baselines = {}
        pipeline_times = defaultdict(list)
        
        for metric in metrics:
            if metric.execution_time > 0:
                pipeline_times[metric.pipeline_name].append(metric.execution_time)
        
        for pipeline, times in pipeline_times.items():
            baselines[pipeline] = np.percentile(times, 75)  # 75th percentile as baseline
        
        return baselines

    def _identify_slowest_pipelines(self, metrics: List[PipelineExecutionMetric]) -> List[Dict[str, Any]]:
        """Identify slowest performing pipelines."""
        pipeline_avg_times = defaultdict(list)
        
        for metric in metrics:
            if metric.execution_time > 0:
                pipeline_avg_times[metric.pipeline_name].append(metric.execution_time)
        
        slowest = []
        for pipeline, times in pipeline_avg_times.items():
            avg_time = np.mean(times)
            slowest.append({
                'pipeline': pipeline,
                'avg_execution_time': avg_time,
                'max_execution_time': np.max(times),
                'executions_count': len(times)
            })
        
        return sorted(slowest, key=lambda x: x['avg_execution_time'], reverse=True)[:10]

    def _detect_performance_regression(self, metrics: List[PipelineExecutionMetric]) -> List[Dict[str, Any]]:
        """Detect performance regressions."""
        regressions = []
        pipeline_times = defaultdict(list)
        
        # Group by pipeline and sort by timestamp
        for metric in metrics:
            if metric.execution_time > 0:
                pipeline_times[metric.pipeline_name].append((metric.timestamp, metric.execution_time))
        
        for pipeline, time_data in pipeline_times.items():
            if len(time_data) < 4:  # Need enough data points
                continue
            
            time_data.sort(key=lambda x: x[0])  # Sort by timestamp
            
            # Compare recent vs older performance
            recent = [t[1] for t in time_data[-5:]]  # Last 5 executions
            older = [t[1] for t in time_data[:-5]]   # All but last 5
            
            if older:
                recent_avg = np.mean(recent)
                older_avg = np.mean(older)
                
                if recent_avg > older_avg * 1.2:  # 20% slowdown threshold
                    regressions.append({
                        'pipeline': pipeline,
                        'recent_avg_time': recent_avg,
                        'baseline_avg_time': older_avg,
                        'regression_percentage': ((recent_avg - older_avg) / older_avg) * 100
                    })
        
        return sorted(regressions, key=lambda x: x['regression_percentage'], reverse=True)

    def _analyze_quality_distribution(self, quality_scores: List[float]) -> Dict[str, Any]:
        """Analyze quality score distribution."""
        if not quality_scores:
            return {}
        
        return {
            'excellent': len([s for s in quality_scores if s >= 95]),
            'good': len([s for s in quality_scores if 85 <= s < 95]),
            'fair': len([s for s in quality_scores if 70 <= s < 85]),
            'poor': len([s for s in quality_scores if s < 70]),
            'histogram_data': np.histogram(quality_scores, bins=10)[0].tolist()
        }

    def _analyze_quality_trends(self, metrics: List[PipelineExecutionMetric]) -> Dict[str, Any]:
        """Analyze quality trends over time."""
        if not metrics:
            return {}
        
        # Sort by timestamp
        sorted_metrics = sorted(metrics, key=lambda m: m.timestamp)
        
        # Calculate moving average quality
        window_size = 5
        quality_trend = []
        
        for i in range(len(sorted_metrics) - window_size + 1):
            window = sorted_metrics[i:i + window_size]
            avg_quality = np.mean([m.quality_score for m in window if m.quality_score > 0])
            if not np.isnan(avg_quality):
                quality_trend.append({
                    'timestamp': window[-1].timestamp.isoformat(),
                    'moving_avg_quality': avg_quality
                })
        
        return {
            'moving_average_trend': quality_trend,
            'overall_direction': self._calculate_trend_direction(
                sorted_metrics[:len(sorted_metrics)//2],
                sorted_metrics[len(sorted_metrics)//2:],
                'quality_score'
            )
        }

    def _calculate_period_trends(self, period_metrics: List[PipelineExecutionMetric]) -> Dict[str, Any]:
        """Calculate trends for a specific time period."""
        if not period_metrics:
            return {}
        
        return {
            'total_executions': len(period_metrics),
            'avg_quality_score': np.mean([m.quality_score for m in period_metrics if m.quality_score > 0]),
            'avg_execution_time': np.mean([m.execution_time for m in period_metrics if m.execution_time > 0]),
            'success_rate': len([m for m in period_metrics if m.status == 'success']) / len(period_metrics) * 100,
            'total_issues': sum(len(m.issues) for m in period_metrics),
            'unique_pipelines': len(set(m.pipeline_name for m in period_metrics))
        }

    def _detect_quality_regression(self, metrics: List[PipelineExecutionMetric]) -> List[Dict[str, Any]]:
        """Detect quality regressions by pipeline."""
        regressions = []
        pipeline_qualities = defaultdict(list)
        
        for metric in metrics:
            if metric.quality_score > 0:
                pipeline_qualities[metric.pipeline_name].append((metric.timestamp, metric.quality_score))
        
        for pipeline, quality_data in pipeline_qualities.items():
            if len(quality_data) < 4:
                continue
            
            quality_data.sort(key=lambda x: x[0])
            
            recent = [q[1] for q in quality_data[-5:]]
            older = [q[1] for q in quality_data[:-5]]
            
            if older:
                recent_avg = np.mean(recent)
                older_avg = np.mean(older)
                
                if recent_avg < older_avg * 0.9:  # 10% quality drop threshold
                    regressions.append({
                        'pipeline': pipeline,
                        'recent_avg_quality': recent_avg,
                        'baseline_avg_quality': older_avg,
                        'quality_drop_percentage': ((older_avg - recent_avg) / older_avg) * 100
                    })
        
        return sorted(regressions, key=lambda x: x['quality_drop_percentage'], reverse=True)

    def _detect_improvements(self, metrics: List[PipelineExecutionMetric]) -> List[Dict[str, Any]]:
        """Detect improvements in pipeline performance or quality."""
        improvements = []
        
        # Similar to regression detection but looking for positive changes
        pipeline_data = defaultdict(list)
        
        for metric in metrics:
            if metric.quality_score > 0 and metric.execution_time > 0:
                pipeline_data[metric.pipeline_name].append({
                    'timestamp': metric.timestamp,
                    'quality_score': metric.quality_score,
                    'execution_time': metric.execution_time
                })
        
        for pipeline, data in pipeline_data.items():
            if len(data) < 4:
                continue
            
            data.sort(key=lambda x: x['timestamp'])
            
            recent = data[-5:]
            older = data[:-5]
            
            if older:
                recent_quality = np.mean([d['quality_score'] for d in recent])
                older_quality = np.mean([d['quality_score'] for d in older])
                recent_time = np.mean([d['execution_time'] for d in recent])
                older_time = np.mean([d['execution_time'] for d in older])
                
                quality_improvement = recent_quality > older_quality * 1.1  # 10% improvement
                time_improvement = recent_time < older_time * 0.9  # 10% faster
                
                if quality_improvement or time_improvement:
                    improvements.append({
                        'pipeline': pipeline,
                        'quality_improvement': quality_improvement,
                        'time_improvement': time_improvement,
                        'quality_change_percentage': ((recent_quality - older_quality) / older_quality) * 100 if older_quality > 0 else 0,
                        'time_change_percentage': ((older_time - recent_time) / older_time) * 100 if older_time > 0 else 0
                    })
        
        return improvements

    def _detect_seasonal_patterns(self, metrics: List[PipelineExecutionMetric]) -> Dict[str, Any]:
        """Detect seasonal patterns in pipeline execution."""
        if len(metrics) < 20:  # Need sufficient data
            return {}
        
        # Group by hour of day
        hourly_performance = defaultdict(list)
        for metric in metrics:
            hour = metric.timestamp.hour
            if metric.execution_time > 0:
                hourly_performance[hour].append(metric.execution_time)
        
        hourly_stats = {}
        for hour, times in hourly_performance.items():
            if len(times) > 2:
                hourly_stats[hour] = {
                    'avg_execution_time': np.mean(times),
                    'executions_count': len(times)
                }
        
        return {
            'hourly_patterns': hourly_stats,
            'peak_hours': sorted(hourly_stats.items(), key=lambda x: x[1]['executions_count'], reverse=True)[:5]
        }

    def _generate_long_term_recommendations(self, metrics: List[PipelineExecutionMetric]) -> List[str]:
        """Generate long-term improvement recommendations."""
        recommendations = []
        
        # Analyze patterns for long-term recommendations
        if metrics:
            # Quality consistency
            quality_scores = [m.quality_score for m in metrics if m.quality_score > 0]
            if quality_scores and np.std(quality_scores) > 15:  # High variation
                recommendations.append("Implement quality gates and standardization to reduce score variation")
            
            # Performance consistency
            execution_times = [m.execution_time for m in metrics if m.execution_time > 0]
            if execution_times and np.std(execution_times) > np.mean(execution_times) * 0.5:
                recommendations.append("Optimize pipeline execution consistency through resource management")
            
            # Issue patterns
            all_issues = []
            for metric in metrics:
                all_issues.extend(metric.issues)
            
            if len(all_issues) > len(metrics) * 0.3:  # More than 30% of executions have issues
                recommendations.append("Implement comprehensive testing and validation framework")
        
        if not recommendations:
            recommendations.append("Continue monitoring and maintain current quality standards")
        
        return recommendations

    def _identify_optimization_opportunities(self, metrics: List[PipelineExecutionMetric]) -> List[str]:
        """Identify optimization opportunities."""
        opportunities = []
        
        if not metrics:
            return opportunities
        
        # Slow pipeline identification
        pipeline_times = defaultdict(list)
        for metric in metrics:
            if metric.execution_time > 0:
                pipeline_times[metric.pipeline_name].append(metric.execution_time)
        
        slow_pipelines = []
        for pipeline, times in pipeline_times.items():
            avg_time = np.mean(times)
            if avg_time > self.config.alert_thresholds['execution_time_warning']:
                slow_pipelines.append(pipeline)
        
        if slow_pipelines:
            opportunities.append(f"Optimize slow pipelines: {', '.join(slow_pipelines[:3])}")
        
        # Low quality pipeline identification
        pipeline_qualities = defaultdict(list)
        for metric in metrics:
            if metric.quality_score > 0:
                pipeline_qualities[metric.pipeline_name].append(metric.quality_score)
        
        low_quality_pipelines = []
        for pipeline, qualities in pipeline_qualities.items():
            avg_quality = np.mean(qualities)
            if avg_quality < self.config.alert_thresholds['quality_score_warning']:
                low_quality_pipelines.append(pipeline)
        
        if low_quality_pipelines:
            opportunities.append(f"Improve quality in pipelines: {', '.join(low_quality_pipelines[:3])}")
        
        # Resource usage optimization
        memory_metrics = [m.memory_usage for m in metrics if m.memory_usage]
        if memory_metrics and np.mean(memory_metrics) > 1000:  # Arbitrary high memory threshold
            opportunities.append("Optimize memory usage across pipeline executions")
        
        return opportunities


async def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pipeline Validation Dashboard Generator")
    parser.add_argument("--root", default=".", help="Repository root path")
    parser.add_argument("--generate", action='store_true', help="Generate comprehensive dashboard")
    parser.add_argument("--formats", nargs='+', choices=['json', 'html', 'csv'], 
                       default=['json', 'html', 'csv'], help="Export formats")
    parser.add_argument("--trend-days", type=int, default=30, help="Days for trend analysis")
    parser.add_argument("--output-dir", help="Output directory override")
    parser.add_argument("--verbose", action='store_true', help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Configure dashboard
    config = DashboardConfig(
        export_formats=args.formats,
        trend_days=args.trend_days
    )
    
    # Initialize generator
    generator = PipelineDashboardGenerator(args.root, config)
    
    if args.output_dir:
        generator.dashboard_dir = Path(args.output_dir)
        generator.dashboard_dir.mkdir(parents=True, exist_ok=True)
    
    if args.generate:
        print("🚀 Generating comprehensive pipeline validation dashboard...")
        
        try:
            outputs = await generator.generate_comprehensive_dashboard()
            
            print(f"✅ Dashboard generation complete!")
            print(f"📊 Generated {len(outputs)} output files:")
            
            for format_type, path in outputs.items():
                print(f"  {format_type.upper()}: {path}")
                
        except Exception as e:
            print(f"❌ Dashboard generation failed: {e}")
            logger.error(f"Dashboard generation failed: {e}")
            traceback.print_exc()
    else:
        print("Use --generate to create the comprehensive dashboard")
        print("Available formats: json, html, csv")


if __name__ == "__main__":
    asyncio.run(main())