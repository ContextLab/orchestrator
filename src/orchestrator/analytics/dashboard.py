"""Performance Dashboard - Issue #206 Task 3.2

Real-time performance dashboard that provides visual analytics, monitoring displays,
and interactive performance insights for system administrators and developers.
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import statistics
from collections import defaultdict, deque
import threading

from .performance_monitor import PerformanceMonitor, MetricType, AlertSeverity

logger = logging.getLogger(__name__)


@dataclass
class DashboardWidget:
    """Dashboard widget configuration."""
    widget_id: str
    widget_type: str
    title: str
    data_source: str
    refresh_interval: float = 5.0
    config: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'widget_id': self.widget_id,
            'widget_type': self.widget_type,
            'title': self.title,
            'data_source': self.data_source,
            'refresh_interval': self.refresh_interval,
            'config': self.config or {}
        }


class DashboardDataProvider:
    """Provides data for dashboard widgets."""
    
    def __init__(self, performance_monitor: PerformanceMonitor):
        self.performance_monitor = performance_monitor
        self.cached_data: Dict[str, Any] = {}
        self.cache_timestamps: Dict[str, float] = {}
        self.cache_ttl = 2.0  # 2 second cache TTL
    
    def get_data(self, data_source: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get data for a specific data source."""
        # Check cache first
        cache_key = f"{data_source}_{hash(str(params))}"
        now = time.time()
        
        if (cache_key in self.cached_data and 
            now - self.cache_timestamps.get(cache_key, 0) < self.cache_ttl):
            return self.cached_data[cache_key]
        
        # Generate fresh data
        data = self._fetch_data(data_source, params or {})
        
        # Cache the result
        self.cached_data[cache_key] = data
        self.cache_timestamps[cache_key] = now
        
        return data
    
    def _fetch_data(self, data_source: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch data from the specified source."""
        try:
            if data_source == "system_overview":
                return self._get_system_overview()
            elif data_source == "execution_metrics":
                return self._get_execution_metrics(params)
            elif data_source == "component_performance":
                return self._get_component_performance(params)
            elif data_source == "active_alerts":
                return self._get_active_alerts()
            elif data_source == "system_resources":
                return self._get_system_resources()
            elif data_source == "performance_trends":
                return self._get_performance_trends(params)
            elif data_source == "error_analysis":
                return self._get_error_analysis()
            elif data_source == "throughput_metrics":
                return self._get_throughput_metrics()
            else:
                return {"error": f"Unknown data source: {data_source}"}
        except Exception as e:
            logger.error(f"Error fetching data for {data_source}: {e}")
            return {"error": str(e)}
    
    def _get_system_overview(self) -> Dict[str, Any]:
        """Get system overview data."""
        summary = self.performance_monitor.get_performance_summary()
        
        return {
            "timestamp": time.time(),
            "system_health": summary.get("system_health", 0),
            "total_components": summary.get("component_profiles", 0),
            "active_alerts": summary.get("active_alerts", 0),
            "metrics_processed": summary.get("metrics_processed", 0),
            "uptime": summary.get("monitor_uptime", 0),
            "recent_averages": summary.get("recent_averages", {}),
            "performance_score": min(100, max(0, summary.get("system_health", 0)))
        }
    
    def _get_execution_metrics(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get execution performance metrics."""
        component = params.get("component", "all")
        time_range = params.get("time_range", 3600)  # 1 hour default
        
        if component != "all":
            perf_data = self.performance_monitor.get_component_performance(component)
            if not perf_data:
                return {"error": f"Component {component} not found"}
            
            return {
                "component": component,
                "data": perf_data,
                "timestamp": time.time()
            }
        else:
            # Get all components
            summary = self.performance_monitor.get_performance_summary()
            components_data = []
            
            for profile in self.performance_monitor.analyzer.profiles.values():
                components_data.append(profile.to_dict())
            
            return {
                "components": components_data,
                "summary": summary,
                "timestamp": time.time()
            }
    
    def _get_component_performance(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed component performance data."""
        component_name = params.get("component")
        if not component_name:
            return {"error": "Component name required"}
        
        profile = self.performance_monitor.analyzer.get_component_profile(component_name)
        if not profile:
            return {"error": f"Component {component_name} not found"}
        
        # Get recent execution times for trending
        recent_times = list(profile.execution_times)[-50:]  # Last 50 executions
        
        return {
            "component": component_name,
            "performance_data": profile.to_dict(),
            "execution_trend": recent_times,
            "timestamp": time.time()
        }
    
    def _get_active_alerts(self) -> Dict[str, Any]:
        """Get active performance alerts."""
        alerts = self.performance_monitor.get_active_alerts()
        
        # Group alerts by severity
        alerts_by_severity = defaultdict(list)
        for alert in alerts:
            alerts_by_severity[alert['severity']].append(alert)
        
        return {
            "total_alerts": len(alerts),
            "alerts_by_severity": dict(alerts_by_severity),
            "all_alerts": alerts,
            "timestamp": time.time()
        }
    
    def _get_system_resources(self) -> Dict[str, Any]:
        """Get system resource utilization data."""
        # Get recent system metrics from the analyzer
        recent_metrics = list(self.performance_monitor.analyzer.metrics_history)[-20:]
        
        cpu_metrics = [m for m in recent_metrics if m.metric_type == MetricType.CPU_USAGE]
        memory_metrics = [m for m in recent_metrics if m.metric_type == MetricType.MEMORY_USAGE]
        
        cpu_values = [m.value for m in cpu_metrics]
        memory_values = [m.value for m in memory_metrics]
        
        return {
            "cpu_usage": {
                "current": cpu_values[-1] if cpu_values else 0,
                "average": statistics.mean(cpu_values) if cpu_values else 0,
                "trend": cpu_values[-10:] if len(cpu_values) >= 10 else cpu_values
            },
            "memory_usage": {
                "current": memory_values[-1] if memory_values else 0,
                "average": statistics.mean(memory_values) if memory_values else 0,
                "trend": memory_values[-10:] if len(memory_values) >= 10 else memory_values
            },
            "timestamp": time.time()
        }
    
    def _get_performance_trends(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get performance trends over time."""
        metric_type = params.get("metric_type", "execution_time")
        time_range = params.get("time_range", 3600)  # 1 hour
        
        now = time.time()
        cutoff_time = now - time_range
        
        # Filter metrics by time range and type
        recent_metrics = [
            m for m in self.performance_monitor.analyzer.metrics_history
            if m.timestamp >= cutoff_time and m.metric_type.value == metric_type
        ]
        
        # Group by time buckets (e.g., 5-minute intervals)
        bucket_size = min(300, time_range / 20)  # 20 buckets max
        time_buckets = defaultdict(list)
        
        for metric in recent_metrics:
            bucket = int((metric.timestamp - cutoff_time) / bucket_size)
            time_buckets[bucket].append(metric.value)
        
        # Calculate averages for each bucket
        trend_data = []
        for bucket in sorted(time_buckets.keys()):
            bucket_time = cutoff_time + (bucket * bucket_size)
            bucket_avg = statistics.mean(time_buckets[bucket])
            trend_data.append({
                "timestamp": bucket_time,
                "value": bucket_avg,
                "count": len(time_buckets[bucket])
            })
        
        return {
            "metric_type": metric_type,
            "time_range": time_range,
            "trend_data": trend_data,
            "total_data_points": len(recent_metrics),
            "timestamp": time.time()
        }
    
    def _get_error_analysis(self) -> Dict[str, Any]:
        """Get error analysis data."""
        error_data = defaultdict(list)
        error_counts = defaultdict(int)
        
        for profile in self.performance_monitor.analyzer.profiles.values():
            if profile.failed_executions > 0:
                error_rate = (profile.failed_executions / profile.total_executions) * 100
                error_data[profile.name].append({
                    "error_rate": error_rate,
                    "failed_executions": profile.failed_executions,
                    "total_executions": profile.total_executions,
                    "recent_errors": list(profile.error_messages)[-5:]  # Last 5 errors
                })
                error_counts[profile.name] = profile.failed_executions
        
        # Sort by error count
        top_error_components = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "error_summary": dict(error_data),
            "top_error_components": top_error_components,
            "total_components_with_errors": len(error_data),
            "timestamp": time.time()
        }
    
    def _get_throughput_metrics(self) -> Dict[str, Any]:
        """Get throughput and performance metrics."""
        recent_metrics = list(self.performance_monitor.analyzer.metrics_history)[-100:]
        
        throughput_metrics = [m for m in recent_metrics if m.metric_type == MetricType.THROUGHPUT]
        execution_metrics = [m for m in recent_metrics if m.metric_type == MetricType.EXECUTION_TIME]
        
        # Calculate overall system throughput
        if execution_metrics:
            execution_times = [m.value for m in execution_metrics]
            avg_execution_time = statistics.mean(execution_times)
            estimated_throughput = 1.0 / avg_execution_time if avg_execution_time > 0 else 0
        else:
            estimated_throughput = 0
        
        return {
            "estimated_throughput": estimated_throughput,
            "recent_execution_times": [m.value for m in execution_metrics[-20:]],
            "throughput_metrics": [m.value for m in throughput_metrics],
            "timestamp": time.time()
        }


class PerformanceDashboard:
    """
    Real-time performance dashboard that provides visual analytics and monitoring.
    """
    
    def __init__(self, performance_monitor: PerformanceMonitor):
        self.performance_monitor = performance_monitor
        self.data_provider = DashboardDataProvider(performance_monitor)
        self.widgets: List[DashboardWidget] = []
        self.dashboard_config = {
            "title": "Orchestrator Performance Dashboard",
            "refresh_interval": 5.0,
            "theme": "dark",
            "auto_refresh": True
        }
        
        # Initialize default widgets
        self._create_default_widgets()
        
        logger.info("PerformanceDashboard initialized")
    
    def _create_default_widgets(self):
        """Create default dashboard widgets."""
        default_widgets = [
            DashboardWidget(
                widget_id="system_overview",
                widget_type="metric_cards",
                title="System Overview",
                data_source="system_overview",
                refresh_interval=5.0,
                config={
                    "metrics": ["system_health", "active_alerts", "total_components", "uptime"],
                    "format": "percentage_and_count"
                }
            ),
            DashboardWidget(
                widget_id="resource_usage",
                widget_type="line_chart",
                title="System Resources",
                data_source="system_resources",
                refresh_interval=5.0,
                config={
                    "metrics": ["cpu_usage", "memory_usage"],
                    "time_range": 300,  # 5 minutes
                    "y_axis_max": 100
                }
            ),
            DashboardWidget(
                widget_id="execution_performance",
                widget_type="bar_chart",
                title="Component Performance",
                data_source="execution_metrics",
                refresh_interval=10.0,
                config={
                    "metric": "average_execution_time",
                    "top_n": 10,
                    "sort_order": "desc"
                }
            ),
            DashboardWidget(
                widget_id="active_alerts",
                widget_type="alert_list",
                title="Active Alerts",
                data_source="active_alerts",
                refresh_interval=5.0,
                config={
                    "max_alerts": 10,
                    "group_by_severity": True
                }
            ),
            DashboardWidget(
                widget_id="performance_trends",
                widget_type="trend_chart",
                title="Performance Trends",
                data_source="performance_trends",
                refresh_interval=15.0,
                config={
                    "metric_type": "execution_time",
                    "time_range": 3600,  # 1 hour
                    "show_average": True
                }
            ),
            DashboardWidget(
                widget_id="error_analysis",
                widget_type="error_dashboard",
                title="Error Analysis",
                data_source="error_analysis",
                refresh_interval=10.0,
                config={
                    "show_recent_errors": True,
                    "max_components": 5
                }
            )
        ]
        
        self.widgets.extend(default_widgets)
    
    def add_widget(self, widget: DashboardWidget):
        """Add a widget to the dashboard."""
        # Check for duplicate IDs
        existing_ids = [w.widget_id for w in self.widgets]
        if widget.widget_id in existing_ids:
            raise ValueError(f"Widget ID {widget.widget_id} already exists")
        
        self.widgets.append(widget)
        logger.info(f"Added widget: {widget.widget_id}")
    
    def remove_widget(self, widget_id: str) -> bool:
        """Remove a widget from the dashboard."""
        original_count = len(self.widgets)
        self.widgets = [w for w in self.widgets if w.widget_id != widget_id]
        
        if len(self.widgets) < original_count:
            logger.info(f"Removed widget: {widget_id}")
            return True
        return False
    
    def get_widget_data(self, widget_id: str) -> Dict[str, Any]:
        """Get data for a specific widget."""
        widget = next((w for w in self.widgets if w.widget_id == widget_id), None)
        if not widget:
            return {"error": f"Widget {widget_id} not found"}
        
        # Get data from the data provider
        data = self.data_provider.get_data(widget.data_source, widget.config)
        
        return {
            "widget_id": widget_id,
            "widget_type": widget.widget_type,
            "title": widget.title,
            "data": data,
            "timestamp": time.time(),
            "config": widget.config
        }
    
    def get_all_widget_data(self) -> Dict[str, Any]:
        """Get data for all dashboard widgets."""
        widget_data = {}
        
        for widget in self.widgets:
            widget_data[widget.widget_id] = self.get_widget_data(widget.widget_id)
        
        return {
            "dashboard_config": self.dashboard_config,
            "widgets": widget_data,
            "timestamp": time.time()
        }
    
    def get_dashboard_config(self) -> Dict[str, Any]:
        """Get dashboard configuration."""
        return {
            **self.dashboard_config,
            "widgets": [w.to_dict() for w in self.widgets],
            "total_widgets": len(self.widgets)
        }
    
    def update_dashboard_config(self, config: Dict[str, Any]):
        """Update dashboard configuration."""
        allowed_keys = ["title", "refresh_interval", "theme", "auto_refresh"]
        
        for key, value in config.items():
            if key in allowed_keys:
                self.dashboard_config[key] = value
        
        logger.info("Dashboard configuration updated")
    
    def export_dashboard_config(self) -> str:
        """Export dashboard configuration as JSON."""
        config = self.get_dashboard_config()
        return json.dumps(config, indent=2, default=str)
    
    def import_dashboard_config(self, config_json: str) -> bool:
        """Import dashboard configuration from JSON."""
        try:
            config = json.loads(config_json)
            
            # Update dashboard settings
            if "dashboard_config" in config:
                self.update_dashboard_config(config["dashboard_config"])
            
            # Update widgets
            if "widgets" in config:
                self.widgets = []
                for widget_data in config["widgets"]:
                    widget = DashboardWidget(**widget_data)
                    self.widgets.append(widget)
            
            logger.info("Dashboard configuration imported successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import dashboard configuration: {e}")
            return False
    
    def generate_dashboard_html(self) -> str:
        """Generate HTML for the dashboard (simplified version)."""
        html_template = """<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f0f0f0; }}
        .dashboard-header {{ background: #333; color: white; padding: 20px; border-radius: 5px; }}
        .widget-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin-top: 20px; }}
        .widget {{ background: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .widget-title {{ font-weight: bold; margin-bottom: 15px; color: #333; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2196F3; }}
        .metric-label {{ color: #666; margin-bottom: 10px; }}
        .alert-item {{ padding: 10px; margin: 5px 0; border-radius: 3px; }}
        .alert-warning {{ background: #fff3cd; border-left: 4px solid #ffc107; }}
        .alert-critical {{ background: #f8d7da; border-left: 4px solid #dc3545; }}
        .status-healthy {{ color: #28a745; }}
        .status-warning {{ color: #ffc107; }}
        .status-critical {{ color: #dc3545; }}
    </style>
</head>
<body>
    <div class="dashboard-header">
        <h1>{title}</h1>
        <p>Real-time Performance Monitoring | Last Updated: {timestamp}</p>
    </div>
    
    <div class="widget-grid">
        {widgets_html}
    </div>
    
    <script>
        // Auto-refresh functionality
        setTimeout(function() {{
            location.reload();
        }}, {refresh_interval} * 1000);
    </script>
</body>
</html>
"""
        
        # Generate widgets HTML
        widgets_html = ""
        all_data = self.get_all_widget_data()
        
        for widget_id, widget_data in all_data["widgets"].items():
            widget_html = self._generate_widget_html(widget_data)
            widgets_html += widget_html
        
        # Format the complete HTML
        formatted_html = html_template.format(
            title=self.dashboard_config["title"],
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            widgets_html=widgets_html,
            refresh_interval=self.dashboard_config["refresh_interval"]
        )
        
        return formatted_html
    
    def _generate_widget_html(self, widget_data: Dict[str, Any]) -> str:
        """Generate HTML for a single widget."""
        widget_type = widget_data.get("widget_type", "unknown")
        title = widget_data.get("title", "Unknown Widget")
        data = widget_data.get("data", {})
        
        if widget_type == "metric_cards":
            return self._generate_metric_cards_html(title, data)
        elif widget_type == "alert_list":
            return self._generate_alert_list_html(title, data)
        elif widget_type == "line_chart" or widget_type == "bar_chart":
            return self._generate_chart_html(title, data, widget_type)
        elif widget_type == "error_dashboard":
            return self._generate_error_dashboard_html(title, data)
        else:
            return f"""
            <div class="widget">
                <div class="widget-title">{title}</div>
                <pre>{json.dumps(data, indent=2)}</pre>
            </div>
            """
    
    def _generate_metric_cards_html(self, title: str, data: Dict[str, Any]) -> str:
        """Generate HTML for metric cards widget."""
        html = f'<div class="widget"><div class="widget-title">{title}</div>'
        
        metrics = {
            "System Health": f"{data.get('system_health', 0):.1f}%",
            "Active Alerts": str(data.get('active_alerts', 0)),
            "Components": str(data.get('total_components', 0)),
            "Uptime": f"{data.get('uptime', 0) / 3600:.1f}h"
        }
        
        for label, value in metrics.items():
            html += f"""
            <div style="margin-bottom: 15px;">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """
        
        html += '</div>'
        return html
    
    def _generate_alert_list_html(self, title: str, data: Dict[str, Any]) -> str:
        """Generate HTML for alerts widget."""
        html = f'<div class="widget"><div class="widget-title">{title}</div>'
        
        alerts = data.get('all_alerts', [])
        if not alerts:
            html += '<p>No active alerts</p>'
        else:
            for alert in alerts[:5]:  # Show top 5 alerts
                severity = alert.get('severity', 'info')
                message = alert.get('message', 'Unknown alert')
                css_class = f"alert-{severity}"
                
                html += f'<div class="alert-item {css_class}">{message}</div>'
        
        html += '</div>'
        return html
    
    def _generate_chart_html(self, title: str, data: Dict[str, Any], chart_type: str) -> str:
        """Generate HTML for chart widgets (simplified representation)."""
        html = f'<div class="widget"><div class="widget-title">{title}</div>'
        
        # For now, show data as text (in a real implementation, you'd use Chart.js or similar)
        if "cpu_usage" in data:
            cpu_data = data["cpu_usage"]
            html += f"""
            <div>CPU Usage: <span class="metric-value">{cpu_data.get('current', 0):.1f}%</span></div>
            <div>Memory Usage: <span class="metric-value">{data.get('memory_usage', {}).get('current', 0):.1f}%</span></div>
            """
        else:
            html += f'<pre>{json.dumps(data, indent=2)[:500]}...</pre>'
        
        html += '</div>'
        return html
    
    def _generate_error_dashboard_html(self, title: str, data: Dict[str, Any]) -> str:
        """Generate HTML for error dashboard widget."""
        html = f'<div class="widget"><div class="widget-title">{title}</div>'
        
        top_components = data.get('top_error_components', [])
        if not top_components:
            html += '<p>No errors detected</p>'
        else:
            html += '<div><strong>Top Error Components:</strong></div>'
            for component, error_count in top_components[:5]:
                html += f'<div>{component}: {error_count} errors</div>'
        
        html += '</div>'
        return html


# Export classes
__all__ = [
    'PerformanceDashboard',
    'DashboardWidget',
    'DashboardDataProvider'
]