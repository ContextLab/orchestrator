"""Advanced performance reporting and dashboard generation system."""

import html
import json
import logging
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

from .performance_monitor import PerformanceMonitor
from .performance_tracker import PerformanceTracker, PipelinePerformanceProfile
from .regression_detector import RegressionAlert, RegressionSeverity

logger = logging.getLogger(__name__)


class PerformanceReporter:
    """
    Comprehensive performance reporting system.
    
    Features:
    - HTML dashboard generation
    - Executive summary reports
    - Trend analysis reports
    - Regression alert summaries
    - Performance comparison reports
    - Historical performance analysis
    """
    
    def __init__(self, 
                 performance_tracker: PerformanceTracker,
                 performance_monitor: PerformanceMonitor):
        """
        Initialize performance reporter.
        
        Args:
            performance_tracker: PerformanceTracker instance
            performance_monitor: PerformanceMonitor instance
        """
        self.performance_tracker = performance_tracker
        self.performance_monitor = performance_monitor
        
        logger.info("Initialized PerformanceReporter")
    
    def generate_executive_dashboard(self, 
                                   output_path: Path,
                                   analysis_period_days: int = 30,
                                   pipeline_filter: Optional[List[str]] = None) -> Path:
        """
        Generate executive dashboard with high-level performance overview.
        
        Args:
            output_path: Path to save dashboard
            analysis_period_days: Period for analysis
            pipeline_filter: Optional list of pipelines to include
            
        Returns:
            Path: Path to generated dashboard HTML file
        """
        logger.info(f"Generating executive dashboard for {analysis_period_days} days")
        
        # Get performance summary
        summary = self.performance_tracker.get_performance_summary(
            pipeline_names=pipeline_filter,
            analysis_period_days=analysis_period_days
        )
        
        # Generate HTML dashboard
        dashboard_html = self._generate_executive_dashboard_html(summary, analysis_period_days)
        
        # Save dashboard
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        dashboard_file = output_path / f"executive_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(dashboard_file, 'w', encoding='utf-8') as f:
            f.write(dashboard_html)
        
        # Also save data as JSON
        data_file = output_path / f"dashboard_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(data_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Executive dashboard saved to {dashboard_file}")
        return dashboard_file
    
    def generate_pipeline_performance_report(self, 
                                           pipeline_name: str,
                                           output_path: Path,
                                           analysis_period_days: int = 30,
                                           include_detailed_analysis: bool = True) -> Dict[str, Path]:
        """
        Generate detailed performance report for a specific pipeline.
        
        Args:
            pipeline_name: Name of the pipeline
            output_path: Path to save reports
            analysis_period_days: Period for analysis
            include_detailed_analysis: Include detailed trend analysis
            
        Returns:
            Dict[str, Path]: Dictionary of generated report files
        """
        logger.info(f"Generating performance report for {pipeline_name}")
        
        # Get comprehensive performance profile
        profile = self.performance_tracker.track_pipeline_performance(
            pipeline_name, analysis_period_days
        )
        
        # Generate reports
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        reports = {}
        
        # 1. HTML Report
        html_report = self._generate_pipeline_html_report(profile, analysis_period_days)
        html_file = output_path / f"{pipeline_name}_performance_report.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_report)
        reports['html'] = html_file
        
        # 2. JSON Data Report
        json_data = {
            "pipeline_name": pipeline_name,
            "profile": asdict(profile),
            "generated_at": datetime.now().isoformat(),
            "analysis_period_days": analysis_period_days
        }
        json_file = output_path / f"{pipeline_name}_performance_data.json"
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        reports['json'] = json_file
        
        # 3. Markdown Summary
        markdown_report = self._generate_pipeline_markdown_summary(profile)
        md_file = output_path / f"{pipeline_name}_performance_summary.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(markdown_report)
        reports['markdown'] = md_file
        
        if include_detailed_analysis:
            # 4. Detailed Trend Analysis
            trend_analysis = self._generate_trend_analysis_report(profile)
            trend_file = output_path / f"{pipeline_name}_trend_analysis.json"
            with open(trend_file, 'w') as f:
                json.dump(trend_analysis, f, indent=2, default=str)
            reports['trend_analysis'] = trend_file
        
        logger.info(f"Performance report generated for {pipeline_name}: {len(reports)} files")
        return reports
    
    def generate_regression_alert_report(self, 
                                       output_path: Path,
                                       days_back: int = 7,
                                       severity_filter: Optional[List[RegressionSeverity]] = None) -> Path:
        """
        Generate comprehensive regression alert report.
        
        Args:
            output_path: Path to save report
            days_back: Days to look back for alerts
            severity_filter: Optional severity filter
            
        Returns:
            Path: Path to generated report
        """
        logger.info(f"Generating regression alert report for last {days_back} days")
        
        # Collect regression alerts from all pipelines
        all_alerts = []
        
        # Get all pipelines with recent activity
        executions = self.performance_monitor.get_execution_history(days_back=days_back)
        pipeline_names = list(set(e.pipeline_name for e in executions))
        
        for pipeline_name in pipeline_names:
            try:
                profile = self.performance_tracker.track_pipeline_performance(
                    pipeline_name, days_back
                )
                if profile.active_regressions:
                    all_alerts.extend(profile.active_regressions)
            except Exception as e:
                logger.warning(f"Failed to get alerts for {pipeline_name}: {e}")
        
        # Filter by severity if requested
        if severity_filter:
            all_alerts = [alert for alert in all_alerts if alert.severity in severity_filter]
        
        # Generate report
        report_html = self._generate_regression_alert_html(all_alerts, days_back)
        
        # Save report
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report_file = output_path / f"regression_alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_html)
        
        # Also save alerts data as JSON
        alerts_data = {
            "alerts": [asdict(alert) for alert in all_alerts],
            "summary": {
                "total_alerts": len(all_alerts),
                "critical_alerts": len([a for a in all_alerts if a.severity == RegressionSeverity.CRITICAL]),
                "high_alerts": len([a for a in all_alerts if a.severity == RegressionSeverity.HIGH]),
                "medium_alerts": len([a for a in all_alerts if a.severity == RegressionSeverity.MEDIUM]),
                "low_alerts": len([a for a in all_alerts if a.severity == RegressionSeverity.LOW]),
                "actionable_alerts": len([a for a in all_alerts if a.is_actionable])
            },
            "generated_at": datetime.now().isoformat(),
            "analysis_period_days": days_back
        }
        
        json_file = output_path / f"regression_alerts_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_file, 'w') as f:
            json.dump(alerts_data, f, indent=2, default=str)
        
        logger.info(f"Regression alert report saved to {report_file} ({len(all_alerts)} alerts)")
        return report_file
    
    def generate_performance_comparison_report(self, 
                                             pipeline_names: List[str],
                                             output_path: Path,
                                             analysis_period_days: int = 30) -> Path:
        """
        Generate performance comparison report across multiple pipelines.
        
        Args:
            pipeline_names: List of pipeline names to compare
            output_path: Path to save report
            analysis_period_days: Period for analysis
            
        Returns:
            Path: Path to generated report
        """
        logger.info(f"Generating performance comparison for {len(pipeline_names)} pipelines")
        
        # Get performance profiles for all pipelines
        profiles = {}
        for pipeline_name in pipeline_names:
            try:
                profiles[pipeline_name] = self.performance_tracker.track_pipeline_performance(
                    pipeline_name, analysis_period_days
                )
            except Exception as e:
                logger.warning(f"Failed to get profile for {pipeline_name}: {e}")
        
        # Generate comparison report
        report_html = self._generate_comparison_html(profiles, analysis_period_days)
        
        # Save report
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report_file = output_path / f"pipeline_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_html)
        
        logger.info(f"Performance comparison report saved to {report_file}")
        return report_file
    
    def _generate_executive_dashboard_html(self, summary: Dict[str, Any], period_days: int) -> str:
        """Generate executive dashboard HTML."""
        summary_stats = summary.get('summary', {})
        pipeline_profiles = summary.get('pipeline_profiles', {})
        
        # Calculate additional metrics
        total_cost = summary_stats.get('total_cost', 0.0)
        avg_health_score = summary_stats.get('average_health_score', 0.0)
        total_executions = summary_stats.get('total_executions', 0)
        
        # Get top performing and concerning pipelines
        healthy_pipelines = []
        concerning_pipelines = []
        
        for name, profile in pipeline_profiles.items():
            if isinstance(profile, dict) and 'health_score' in profile:
                if profile['health_score'] >= 80:
                    healthy_pipelines.append((name, profile['health_score']))
                elif profile['health_score'] < 50:
                    concerning_pipelines.append((name, profile['health_score']))
        
        healthy_pipelines.sort(key=lambda x: x[1], reverse=True)
        concerning_pipelines.sort(key=lambda x: x[1])
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pipeline Performance Executive Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .header h1 {{ color: #333; margin-bottom: 10px; }}
        .header .subtitle {{ color: #666; font-size: 16px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .metric-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; border-left: 4px solid #007bff; }}
        .metric-card.success {{ border-left-color: #28a745; }}
        .metric-card.warning {{ border-left-color: #ffc107; }}
        .metric-card.danger {{ border-left-color: #dc3545; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #333; margin-bottom: 5px; }}
        .metric-label {{ color: #666; font-size: 14px; }}
        .section {{ margin-bottom: 30px; }}
        .section h2 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
        .pipeline-list {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; }}
        .pipeline-card {{ background: white; border: 1px solid #ddd; border-radius: 8px; padding: 15px; }}
        .pipeline-card.healthy {{ border-left: 4px solid #28a745; }}
        .pipeline-card.concerning {{ border-left: 4px solid #dc3545; }}
        .pipeline-name {{ font-weight: bold; margin-bottom: 10px; }}
        .pipeline-stats {{ display: flex; justify-content: space-between; align-items: center; }}
        .health-score {{ font-size: 18px; font-weight: bold; }}
        .health-score.excellent {{ color: #28a745; }}
        .health-score.good {{ color: #6cb04a; }}
        .health-score.fair {{ color: #ffc107; }}
        .health-score.poor {{ color: #fd7e14; }}
        .health-score.critical {{ color: #dc3545; }}
        .timestamp {{ text-align: center; margin-top: 30px; color: #666; font-size: 12px; }}
        .alert {{ background: #fff3cd; border: 1px solid #ffecb5; border-radius: 4px; padding: 10px; margin-bottom: 10px; }}
        .alert.danger {{ background: #f8d7da; border-color: #f5c6cb; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Pipeline Performance Executive Dashboard</h1>
            <div class="subtitle">Performance overview for the last {period_days} days</div>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card success">
                <div class="metric-value">{summary_stats.get('healthy_pipelines', 0)}</div>
                <div class="metric-label">Healthy Pipelines</div>
            </div>
            <div class="metric-card warning">
                <div class="metric-value">{summary_stats.get('concerning_pipelines', 0)}</div>
                <div class="metric-label">Concerning Pipelines</div>
            </div>
            <div class="metric-card danger">
                <div class="metric-value">{summary_stats.get('critical_pipelines', 0)}</div>
                <div class="metric-label">Critical Pipelines</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{avg_health_score:.1f}</div>
                <div class="metric-label">Average Health Score</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{total_executions:,}</div>
                <div class="metric-label">Total Executions</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${total_cost:.2f}</div>
                <div class="metric-label">Total Cost</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary_stats.get('average_success_rate', 0):.1%}</div>
                <div class="metric-label">Average Success Rate</div>
            </div>
            <div class="metric-card warning">
                <div class="metric-value">{summary_stats.get('pipelines_with_regressions', 0)}</div>
                <div class="metric-label">Pipelines with Regressions</div>
            </div>
        </div>
        
        {"<div class='alert danger'><strong>Critical Issues Detected:</strong> " + str(summary_stats.get('critical_pipelines', 0)) + " pipelines require immediate attention.</div>" if summary_stats.get('critical_pipelines', 0) > 0 else ""}
        
        {"<div class='alert'><strong>Regressions Alert:</strong> " + str(summary_stats.get('pipelines_with_regressions', 0)) + " pipelines showing performance regressions.</div>" if summary_stats.get('pipelines_with_regressions', 0) > 0 else ""}
        
        <div class="section">
            <h2>Top Performing Pipelines</h2>
            <div class="pipeline-list">
"""
        
        # Add healthy pipelines
        for name, score in healthy_pipelines[:6]:  # Top 6
            profile = pipeline_profiles.get(name, {})
            html += f"""
                <div class="pipeline-card healthy">
                    <div class="pipeline-name">{html.escape(name)}</div>
                    <div class="pipeline-stats">
                        <span>Success Rate: {profile.get('success_rate', 0):.1%}</span>
                        <span class="health-score excellent">{score:.1f}</span>
                    </div>
                </div>
"""
        
        html += """
            </div>
        </div>
        
        <div class="section">
            <h2>Pipelines Requiring Attention</h2>
            <div class="pipeline-list">
"""
        
        # Add concerning pipelines
        for name, score in concerning_pipelines[:6]:  # Bottom 6
            profile = pipeline_profiles.get(name, {})
            status_class = "critical" if score < 30 else "poor"
            html += f"""
                <div class="pipeline-card concerning">
                    <div class="pipeline-name">{html.escape(name)}</div>
                    <div class="pipeline-stats">
                        <span>Success Rate: {profile.get('success_rate', 0):.1%}</span>
                        <span class="health-score {status_class}">{score:.1f}</span>
                    </div>
                    <div style="margin-top: 10px; font-size: 12px; color: #666;">
                        Executions: {profile.get('total_executions', 0)} | Regressions: {profile.get('active_regressions', 0)}
                    </div>
                </div>
"""
        
        html += f"""
            </div>
        </div>
        
        <div class="timestamp">
            Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>
"""
        
        return html
    
    def _generate_pipeline_html_report(self, profile: PipelinePerformanceProfile, period_days: int) -> str:
        """Generate detailed pipeline HTML report."""
        status_class = profile.performance_status
        health_score = profile.overall_health_score
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{profile.pipeline_name} - Performance Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .header h1 {{ color: #333; margin-bottom: 10px; }}
        .header .pipeline-name {{ color: #007bff; font-size: 24px; font-weight: bold; }}
        .health-indicator {{ display: inline-block; padding: 8px 16px; border-radius: 20px; font-weight: bold; margin: 10px 0; }}
        .health-excellent {{ background-color: #d4edda; color: #155724; }}
        .health-good {{ background-color: #d1ecf1; color: #0c5460; }}
        .health-fair {{ background-color: #fff3cd; color: #856404; }}
        .health-poor {{ background-color: #f8d7da; color: #721c24; }}
        .health-critical {{ background-color: #dc3545; color: white; }}
        .metrics-section {{ margin-bottom: 30px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
        .metric-card {{ background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; }}
        .metric-value {{ font-size: 1.5em; font-weight: bold; color: #333; }}
        .metric-label {{ color: #666; font-size: 12px; margin-top: 5px; }}
        .section {{ margin-bottom: 30px; }}
        .section h2 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
        .trend-indicator {{ display: inline-block; padding: 4px 8px; border-radius: 12px; font-size: 12px; font-weight: bold; }}
        .trend-improving {{ background-color: #d4edda; color: #155724; }}
        .trend-stable {{ background-color: #e2e3e5; color: #383d41; }}
        .trend-degrading {{ background-color: #f8d7da; color: #721c24; }}
        .regression-alert {{ background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 4px; padding: 15px; margin-bottom: 10px; }}
        .regression-critical {{ border-color: #dc3545; }}
        .regression-high {{ border-color: #fd7e14; }}
        .recommendation {{ background: #d1ecf1; border: 1px solid #bee5eb; border-radius: 4px; padding: 15px; margin-bottom: 10px; }}
        .timestamp {{ text-align: center; margin-top: 30px; color: #666; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Pipeline Performance Report</h1>
            <div class="pipeline-name">{html.escape(profile.pipeline_name)}</div>
            <div class="health-indicator health-{status_class}">
                Health Score: {health_score:.1f} ({status_class.upper()})
            </div>
        </div>
        
        <div class="metrics-section">
            <h2>Performance Metrics ({period_days} days)</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{profile.success_rate:.1%}</div>
                    <div class="metric-label">Success Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{profile.total_executions}</div>
                    <div class="metric-label">Total Executions</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{profile.avg_execution_time:.1f}s</div>
                    <div class="metric-label">Avg Execution Time</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${profile.total_cost:.2f}</div>
                    <div class="metric-label">Total Cost</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{profile.avg_memory_usage:.0f}MB</div>
                    <div class="metric-label">Avg Memory Usage</div>
                </div>
                {"<div class='metric-card'><div class='metric-value'>" + f"{profile.avg_quality_score:.1f}" + "</div><div class='metric-label'>Avg Quality Score</div></div>" if profile.avg_quality_score else ""}
            </div>
        </div>
        
        <div class="section">
            <h2>Performance Trends</h2>
"""
        
        # Add trend information
        trends = [
            ("Execution Time", profile.execution_time_trend),
            ("Cost", profile.cost_trend),
            ("Memory Usage", profile.memory_trend),
            ("Quality", profile.quality_trend),
            ("Throughput", profile.throughput_trend)
        ]
        
        for trend_name, trend in trends:
            if trend:
                trend_class = f"trend-{trend.trend_direction}"
                html += f"""
            <div style="margin-bottom: 15px;">
                <strong>{trend_name}:</strong>
                <span class="trend-indicator {trend_class}">{trend.trend_direction.upper()}</span>
                <span style="margin-left: 10px; color: #666;">{trend.change_percent:+.1f}% over {trend.trend_period_days} days</span>
            </div>
"""
        
        html += """
        </div>
        
        <div class="section">
            <h2>Active Performance Alerts</h2>
"""
        
        if profile.active_regressions:
            for regression in profile.active_regressions:
                severity_class = f"regression-{regression.severity.value}"
                html += f"""
            <div class="regression-alert {severity_class}">
                <strong>{regression.severity.value.upper()}: {regression.regression_type.value.replace('_', ' ').title()}</strong><br>
                {html.escape(regression.description)}<br>
                <small><strong>Recommendation:</strong> {html.escape(regression.recommendation)}</small>
            </div>
"""
        else:
            html += "<p>No active performance regressions detected.</p>"
        
        html += f"""
        </div>
        
        <div class="section">
            <h2>Baseline Information</h2>
            <p><strong>Baseline Available:</strong> {'Yes' if profile.has_baseline else 'No'}</p>
            {"<p><strong>Baseline Age:</strong> " + str(profile.baseline_age_days) + " days</p>" if profile.has_baseline else ""}
            {"<p><strong>Baseline Confidence:</strong> " + f"{profile.baseline_confidence:.2f}" + "</p>" if profile.has_baseline else ""}
        </div>
        
        <div class="timestamp">
            Report generated on {profile.profile_date.strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>
"""
        
        return html
    
    def _generate_pipeline_markdown_summary(self, profile: PipelinePerformanceProfile) -> str:
        """Generate pipeline performance markdown summary."""
        md = f"""# Pipeline Performance Summary: {profile.pipeline_name}

Generated: {profile.profile_date.strftime('%Y-%m-%d %H:%M:%S')}
Analysis Period: {profile.analysis_period_days} days

## Overall Health
- **Health Score:** {profile.overall_health_score:.1f}/100
- **Status:** {profile.performance_status.upper()}

## Key Metrics
- **Success Rate:** {profile.success_rate:.1%}
- **Total Executions:** {profile.total_executions:,}
- **Average Execution Time:** {profile.avg_execution_time:.1f}s
- **Total Cost:** ${profile.total_cost:.2f}
- **Average Memory Usage:** {profile.avg_memory_usage:.0f}MB

## Performance Trends
"""
        
        trends = [
            ("Execution Time", profile.execution_time_trend),
            ("Cost", profile.cost_trend),
            ("Memory Usage", profile.memory_trend),
            ("Quality", profile.quality_trend),
            ("Throughput", profile.throughput_trend)
        ]
        
        for trend_name, trend in trends:
            if trend:
                direction_emoji = {"improving": "📈", "degrading": "📉", "stable": "➡️"}.get(trend.trend_direction, "❓")
                md += f"- **{trend_name}:** {direction_emoji} {trend.trend_direction.upper()} ({trend.change_percent:+.1f}%)\n"
        
        if profile.active_regressions:
            md += f"\n## Active Alerts ({len(profile.active_regressions)})\n"
            for regression in profile.active_regressions:
                severity_emoji = {"critical": "🚨", "high": "⚠️", "medium": "⚡", "low": "ℹ️"}.get(regression.severity.value, "❓")
                md += f"- {severity_emoji} **{regression.severity.value.upper()}:** {regression.description}\n"
        else:
            md += "\n## Active Alerts\n✅ No active performance regressions detected.\n"
        
        return md
    
    def _generate_trend_analysis_report(self, profile: PipelinePerformanceProfile) -> Dict[str, Any]:
        """Generate detailed trend analysis data."""
        return {
            "pipeline_name": profile.pipeline_name,
            "analysis_date": profile.profile_date.isoformat(),
            "trends": {
                "execution_time": asdict(profile.execution_time_trend) if profile.execution_time_trend else None,
                "cost": asdict(profile.cost_trend) if profile.cost_trend else None,
                "memory": asdict(profile.memory_trend) if profile.memory_trend else None,
                "quality": asdict(profile.quality_trend) if profile.quality_trend else None,
                "throughput": asdict(profile.throughput_trend) if profile.throughput_trend else None
            },
            "stability_metrics": {
                "execution_time_stability": profile.execution_time_stability,
                "cost_stability": profile.cost_stability
            },
            "summary": {
                "trends_improving": len([t for t in [profile.execution_time_trend, profile.cost_trend, 
                                                   profile.memory_trend, profile.quality_trend, 
                                                   profile.throughput_trend] 
                                       if t and t.trend_direction == "improving"]),
                "trends_degrading": len([t for t in [profile.execution_time_trend, profile.cost_trend, 
                                                   profile.memory_trend, profile.quality_trend, 
                                                   profile.throughput_trend] 
                                       if t and t.trend_direction == "degrading"]),
                "trends_stable": len([t for t in [profile.execution_time_trend, profile.cost_trend, 
                                                profile.memory_trend, profile.quality_trend, 
                                                profile.throughput_trend] 
                                    if t and t.trend_direction == "stable"])
            }
        }
    
    def _generate_regression_alert_html(self, alerts: List[RegressionAlert], days_back: int) -> str:
        """Generate regression alert report HTML."""
        # Group alerts by severity
        alerts_by_severity = defaultdict(list)
        for alert in alerts:
            alerts_by_severity[alert.severity].append(alert)
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Performance Regression Alert Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .alert-summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 30px; }}
        .summary-card {{ background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; }}
        .summary-card.critical {{ border-left: 4px solid #dc3545; }}
        .summary-card.high {{ border-left: 4px solid #fd7e14; }}
        .summary-card.medium {{ border-left: 4px solid #ffc107; }}
        .summary-card.low {{ border-left: 4px solid #6c757d; }}
        .alert-card {{ border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin-bottom: 15px; }}
        .alert-card.critical {{ border-left: 4px solid #dc3545; background-color: #fff5f5; }}
        .alert-card.high {{ border-left: 4px solid #fd7e14; background-color: #fff8f0; }}
        .alert-card.medium {{ border-left: 4px solid #ffc107; background-color: #fffcf0; }}
        .alert-card.low {{ border-left: 4px solid #6c757d; background-color: #f8f9fa; }}
        .alert-header {{ display: flex; justify-content: between; align-items: center; margin-bottom: 10px; }}
        .severity-badge {{ padding: 4px 8px; border-radius: 12px; font-size: 12px; font-weight: bold; }}
        .severity-critical {{ background-color: #dc3545; color: white; }}
        .severity-high {{ background-color: #fd7e14; color: white; }}
        .severity-medium {{ background-color: #ffc107; color: #212529; }}
        .severity-low {{ background-color: #6c757d; color: white; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Performance Regression Alert Report</h1>
            <p>Analysis period: Last {days_back} days | Total alerts: {len(alerts)}</p>
        </div>
        
        <div class="alert-summary">
            <div class="summary-card critical">
                <h3>{len(alerts_by_severity[RegressionSeverity.CRITICAL])}</h3>
                <p>Critical Alerts</p>
            </div>
            <div class="summary-card high">
                <h3>{len(alerts_by_severity[RegressionSeverity.HIGH])}</h3>
                <p>High Alerts</p>
            </div>
            <div class="summary-card medium">
                <h3>{len(alerts_by_severity[RegressionSeverity.MEDIUM])}</h3>
                <p>Medium Alerts</p>
            </div>
            <div class="summary-card low">
                <h3>{len(alerts_by_severity[RegressionSeverity.LOW])}</h3>
                <p>Low Alerts</p>
            </div>
        </div>
"""
        
        # Add alerts by severity (highest first)
        for severity in [RegressionSeverity.CRITICAL, RegressionSeverity.HIGH, 
                        RegressionSeverity.MEDIUM, RegressionSeverity.LOW]:
            severity_alerts = alerts_by_severity[severity]
            if not severity_alerts:
                continue
                
            html += f"""
        <div class="section">
            <h2>{severity.value.upper()} Severity Alerts ({len(severity_alerts)})</h2>
"""
            
            for alert in severity_alerts:
                html += f"""
            <div class="alert-card {severity.value}">
                <div class="alert-header">
                    <h3>{html.escape(alert.pipeline_name)}</h3>
                    <span class="severity-badge severity-{severity.value}">{severity.value.upper()}</span>
                </div>
                <p><strong>{alert.regression_type.value.replace('_', ' ').title()}:</strong> {html.escape(alert.description)}</p>
                <p><strong>Change:</strong> {alert.change_percent:+.1f}% ({alert.baseline_value:.2f} → {alert.current_value:.2f})</p>
                <p><strong>Recommendation:</strong> {html.escape(alert.recommendation)}</p>
                <p><small><strong>Confidence:</strong> {alert.confidence:.2f} | <strong>Sample Size:</strong> {alert.sample_size} | <strong>Detected:</strong> {alert.detected_at.strftime('%Y-%m-%d %H:%M:%S')}</small></p>
            </div>
"""
            
            html += "</div>"
        
        if not alerts:
            html += "<p style='text-align: center; color: #28a745; font-size: 18px;'>🎉 No performance regressions detected!</p>"
        
        html += f"""
        <div style="text-align: center; margin-top: 30px; color: #666; font-size: 12px;">
            Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>
"""
        
        return html
    
    def _generate_comparison_html(self, profiles: Dict[str, PipelinePerformanceProfile], period_days: int) -> str:
        """Generate pipeline comparison HTML report."""
        # Sort pipelines by health score
        sorted_profiles = sorted(profiles.items(), key=lambda x: x[1].overall_health_score, reverse=True)
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pipeline Performance Comparison</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .comparison-table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        .comparison-table th, .comparison-table td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        .comparison-table th {{ background-color: #f8f9fa; font-weight: bold; }}
        .comparison-table tr:nth-child(even) {{ background-color: #f8f9fa; }}
        .health-score {{ font-weight: bold; padding: 4px 8px; border-radius: 4px; }}
        .health-excellent {{ background-color: #d4edda; color: #155724; }}
        .health-good {{ background-color: #d1ecf1; color: #0c5460; }}
        .health-fair {{ background-color: #fff3cd; color: #856404; }}
        .health-poor {{ background-color: #f8d7da; color: #721c24; }}
        .health-critical {{ background-color: #dc3545; color: white; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Pipeline Performance Comparison</h1>
            <p>Comparing {len(profiles)} pipelines over {period_days} days</p>
        </div>
        
        <table class="comparison-table">
            <thead>
                <tr>
                    <th>Pipeline Name</th>
                    <th>Health Score</th>
                    <th>Status</th>
                    <th>Success Rate</th>
                    <th>Executions</th>
                    <th>Avg Time</th>
                    <th>Total Cost</th>
                    <th>Active Alerts</th>
                </tr>
            </thead>
            <tbody>
"""
        
        for pipeline_name, profile in sorted_profiles:
            status_class = f"health-{profile.performance_status}"
            html += f"""
                <tr>
                    <td><strong>{html.escape(pipeline_name)}</strong></td>
                    <td><span class="health-score {status_class}">{profile.overall_health_score:.1f}</span></td>
                    <td>{profile.performance_status.title()}</td>
                    <td>{profile.success_rate:.1%}</td>
                    <td>{profile.total_executions:,}</td>
                    <td>{profile.avg_execution_time:.1f}s</td>
                    <td>${profile.total_cost:.2f}</td>
                    <td>{len(profile.active_regressions)}</td>
                </tr>
"""
        
        html += f"""
            </tbody>
        </table>
        
        <div style="text-align: center; margin-top: 30px; color: #666; font-size: 12px;">
            Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>
"""
        
        return html