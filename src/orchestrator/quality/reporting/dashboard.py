"""
Quality Dashboard Interface for Monitoring and Insights

This module provides a comprehensive web-based dashboard interface for quality
control monitoring, metrics visualization, and actionable insights. It integrates
with the metrics collection, analytics, and alerting systems to provide a unified
quality monitoring experience.

Key Features:
- Real-time quality metrics visualization and monitoring
- Interactive charts and graphs for trend analysis
- Quality insights and recommendations display
- Alert management and acknowledgment interface
- Configurable dashboard layouts and widgets
- Export capabilities for reports and data
- REST API endpoints for external integrations
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import socketserver

from .metrics import QualityMetricsCollector, TimeSeriesMetric, MetricsSnapshot, QualityMetric
from .analytics import QualityAnalytics, AnalyticsResult, QualityInsight, TrendAnalysis
from .alerts import QualityAlertSystem, AlertNotification, AlertSeverity
from ..logging.logger import StructuredLogger, get_logger, LogCategory


class WidgetType(Enum):
    """Types of dashboard widgets."""
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"  
    GAUGE = "gauge"
    COUNTER = "counter"
    TABLE = "table"
    ALERT_LIST = "alert_list"
    INSIGHT_PANEL = "insight_panel"
    TREND_INDICATOR = "trend_indicator"
    HEATMAP = "heatmap"
    STATUS_GRID = "status_grid"


@dataclass
class DashboardWidget:
    """Individual dashboard widget configuration."""
    widget_id: str
    title: str
    widget_type: WidgetType
    metric_patterns: List[str] = field(default_factory=list)
    time_window_hours: int = 24
    refresh_interval_seconds: int = 30
    position: Tuple[int, int] = (0, 0)  # (row, column)
    size: Tuple[int, int] = (1, 1)      # (height, width)
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class DashboardConfig:
    """Dashboard configuration and layout."""
    dashboard_id: str
    title: str
    description: str
    widgets: List[DashboardWidget] = field(default_factory=list)
    layout_columns: int = 12
    auto_refresh: bool = True
    refresh_interval_seconds: int = 30
    theme: str = "light"
    metadata: Dict[str, Any] = field(default_factory=dict)


class QualityDashboard:
    """
    Comprehensive quality dashboard with web interface and API endpoints.
    
    Provides real-time quality monitoring, visualization, and management
    capabilities through a web-based interface.
    """
    
    def __init__(
        self,
        metrics_collector: QualityMetricsCollector,
        analytics: QualityAnalytics,
        alert_system: QualityAlertSystem,
        logger: Optional[StructuredLogger] = None,
        port: int = 8080,
        host: str = "localhost"
    ):
        """
        Initialize quality dashboard.
        
        Args:
            metrics_collector: Metrics collector instance
            analytics: Analytics engine instance  
            alert_system: Alert system instance
            logger: Optional structured logger
            port: Web server port
            host: Web server host
        """
        self.metrics_collector = metrics_collector
        self.analytics = analytics
        self.alert_system = alert_system
        self.logger = logger or get_logger("quality_dashboard")
        self.port = port
        self.host = host
        
        # Dashboard state
        self._dashboards: Dict[str, DashboardConfig] = {}
        self._dashboard_lock = threading.RLock()
        
        # Web server
        self._server: Optional[HTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._is_running = False
        
        # Data cache for performance
        self._data_cache: Dict[str, Tuple[float, Any]] = {}  # key -> (timestamp, data)
        self._cache_ttl = 30  # seconds
        self._cache_lock = threading.RLock()
        
        # Default dashboard
        self._create_default_dashboard()
        
        self.logger.info(f"Initialized QualityDashboard on {host}:{port}", category=LogCategory.MONITORING)
    
    def _create_default_dashboard(self) -> None:
        """Create default dashboard configuration."""
        default_widgets = [
            DashboardWidget(
                widget_id="quality_score_gauge",
                title="Overall Quality Score",
                widget_type=WidgetType.GAUGE,
                metric_patterns=["quality.*.score"],
                position=(0, 0),
                size=(1, 3),
                config={
                    "min_value": 0,
                    "max_value": 100,
                    "thresholds": [{"value": 60, "color": "red"}, {"value": 75, "color": "yellow"}, {"value": 90, "color": "green"}]
                }
            ),
            DashboardWidget(
                widget_id="violations_trend",
                title="Violations Trend",
                widget_type=WidgetType.LINE_CHART,
                metric_patterns=["quality.validation.total_violations"],
                position=(0, 3),
                size=(1, 6),
                config={"show_points": True, "smooth": True}
            ),
            DashboardWidget(
                widget_id="success_rate",
                title="Validation Success Rate",
                widget_type=WidgetType.LINE_CHART,
                metric_patterns=["quality.validation.success_rate"],
                position=(0, 9),
                size=(1, 3),
                config={"format": "percentage"}
            ),
            DashboardWidget(
                widget_id="active_alerts",
                title="Active Alerts",
                widget_type=WidgetType.ALERT_LIST,
                position=(1, 0),
                size=(1, 6),
                config={"max_alerts": 10}
            ),
            DashboardWidget(
                widget_id="quality_insights",
                title="Quality Insights",
                widget_type=WidgetType.INSIGHT_PANEL,
                position=(1, 6),
                size=(1, 6),
                config={"max_insights": 5}
            ),
            DashboardWidget(
                widget_id="rule_performance",
                title="Validation Rule Performance",
                widget_type=WidgetType.BAR_CHART,
                metric_patterns=["quality.validation.rule_duration_ms"],
                position=(2, 0),
                size=(1, 12),
                config={"orientation": "horizontal", "limit": 10}
            )
        ]
        
        default_dashboard = DashboardConfig(
            dashboard_id="default",
            title="Quality Control Dashboard",
            description="Main quality monitoring dashboard",
            widgets=default_widgets,
            layout_columns=12,
            auto_refresh=True,
            refresh_interval_seconds=30
        )
        
        self.add_dashboard(default_dashboard)
    
    def add_dashboard(self, dashboard_config: DashboardConfig) -> None:
        """Add or update a dashboard configuration."""
        with self._dashboard_lock:
            self._dashboards[dashboard_config.dashboard_id] = dashboard_config
        
        self.logger.info(f"Added dashboard: {dashboard_config.title} ({dashboard_config.dashboard_id})")
    
    def remove_dashboard(self, dashboard_id: str) -> bool:
        """Remove a dashboard configuration."""
        with self._dashboard_lock:
            removed = self._dashboards.pop(dashboard_id, None)
        
        if removed:
            self.logger.info(f"Removed dashboard: {dashboard_id}")
            return True
        return False
    
    def get_dashboard_config(self, dashboard_id: str) -> Optional[DashboardConfig]:
        """Get dashboard configuration."""
        with self._dashboard_lock:
            return self._dashboards.get(dashboard_id)
    
    def list_dashboards(self) -> List[Dict[str, Any]]:
        """List available dashboards."""
        with self._dashboard_lock:
            return [
                {
                    'dashboard_id': config.dashboard_id,
                    'title': config.title,
                    'description': config.description,
                    'widget_count': len(config.widgets),
                    'auto_refresh': config.auto_refresh
                }
                for config in self._dashboards.values()
            ]
    
    def get_widget_data(self, dashboard_id: str, widget_id: str) -> Optional[Dict[str, Any]]:
        """Get data for a specific widget."""
        cache_key = f"{dashboard_id}_{widget_id}"
        
        # Check cache first
        with self._cache_lock:
            cached_data = self._data_cache.get(cache_key)
            if cached_data and time.time() - cached_data[0] < self._cache_ttl:
                return cached_data[1]
        
        # Get dashboard and widget config
        dashboard = self.get_dashboard_config(dashboard_id)
        if not dashboard:
            return None
        
        widget = next((w for w in dashboard.widgets if w.widget_id == widget_id), None)
        if not widget or not widget.enabled:
            return None
        
        # Generate widget data based on type
        try:
            widget_data = self._generate_widget_data(widget)
            
            # Cache the result
            with self._cache_lock:
                self._data_cache[cache_key] = (time.time(), widget_data)
            
            return widget_data
            
        except Exception as e:
            self.logger.error(f"Failed to generate widget data for {widget_id}: {e}", exception=e)
            return None
    
    def _generate_widget_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Generate data for a widget based on its type."""
        if widget.widget_type == WidgetType.GAUGE:
            return self._generate_gauge_data(widget)
        elif widget.widget_type == WidgetType.LINE_CHART:
            return self._generate_line_chart_data(widget)
        elif widget.widget_type == WidgetType.BAR_CHART:
            return self._generate_bar_chart_data(widget)
        elif widget.widget_type == WidgetType.COUNTER:
            return self._generate_counter_data(widget)
        elif widget.widget_type == WidgetType.ALERT_LIST:
            return self._generate_alert_list_data(widget)
        elif widget.widget_type == WidgetType.INSIGHT_PANEL:
            return self._generate_insight_panel_data(widget)
        elif widget.widget_type == WidgetType.TREND_INDICATOR:
            return self._generate_trend_indicator_data(widget)
        elif widget.widget_type == WidgetType.TABLE:
            return self._generate_table_data(widget)
        else:
            return {"error": f"Unsupported widget type: {widget.widget_type.value}"}
    
    def _generate_gauge_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Generate data for gauge widget."""
        # Get latest values for metrics matching patterns
        values = []
        for pattern in widget.metric_patterns:
            matching_metrics = self.metrics_collector.get_metrics_by_pattern(pattern)
            for time_series in matching_metrics.values():
                latest_value = time_series.get_latest_value()
                if latest_value is not None:
                    values.append(latest_value)
        
        if not values:
            current_value = 0
        else:
            current_value = sum(values) / len(values)  # Average of matching metrics
        
        return {
            "type": "gauge",
            "value": current_value,
            "min_value": widget.config.get("min_value", 0),
            "max_value": widget.config.get("max_value", 100),
            "thresholds": widget.config.get("thresholds", []),
            "timestamp": time.time()
        }
    
    def _generate_line_chart_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Generate data for line chart widget."""
        datasets = []
        
        for pattern in widget.metric_patterns:
            matching_metrics = self.metrics_collector.get_metrics_by_pattern(pattern)
            
            for key, time_series in matching_metrics.items():
                # Get values in time window
                cutoff_time = time.time() - (widget.time_window_hours * 3600)
                window_values = time_series.get_values_in_range(cutoff_time, time.time())
                
                if window_values:
                    datasets.append({
                        "label": time_series.name,
                        "data": [{"x": ts * 1000, "y": val} for ts, val in window_values],  # Convert to milliseconds for JS
                        "borderColor": self._get_metric_color(time_series.name),
                        "backgroundColor": self._get_metric_color(time_series.name, alpha=0.2)
                    })
        
        return {
            "type": "line",
            "datasets": datasets,
            "options": {
                "responsive": True,
                "scales": {
                    "x": {"type": "time"},
                    "y": {"beginAtZero": True}
                },
                **widget.config
            },
            "timestamp": time.time()
        }
    
    def _generate_bar_chart_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Generate data for bar chart widget."""
        labels = []
        values = []
        
        for pattern in widget.metric_patterns:
            matching_metrics = self.metrics_collector.get_metrics_by_pattern(pattern)
            
            metric_values = {}
            for key, time_series in matching_metrics.items():
                latest_value = time_series.get_latest_value()
                if latest_value is not None:
                    # Group by rule or category if available
                    group_key = time_series.labels.get('rule_id', time_series.name)
                    if group_key not in metric_values:
                        metric_values[group_key] = []
                    metric_values[group_key].append(latest_value)
            
            # Calculate averages for each group
            for group_key, group_values in metric_values.items():
                labels.append(group_key)
                values.append(sum(group_values) / len(group_values))
        
        # Apply limit if configured
        limit = widget.config.get('limit', len(labels))
        if limit < len(labels):
            # Sort by value and take top N
            sorted_data = sorted(zip(labels, values), key=lambda x: x[1], reverse=True)
            labels = [item[0] for item in sorted_data[:limit]]
            values = [item[1] for item in sorted_data[:limit]]
        
        return {
            "type": "bar",
            "data": {
                "labels": labels,
                "datasets": [{
                    "label": widget.title,
                    "data": values,
                    "backgroundColor": [self._get_metric_color(label) for label in labels]
                }]
            },
            "options": {
                "responsive": True,
                "indexAxis": widget.config.get("orientation") == "horizontal" and "y" or "x",
                **{k: v for k, v in widget.config.items() if k not in ["orientation", "limit"]}
            },
            "timestamp": time.time()
        }
    
    def _generate_counter_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Generate data for counter widget."""
        total_value = 0
        
        for pattern in widget.metric_patterns:
            matching_metrics = self.metrics_collector.get_metrics_by_pattern(pattern)
            for time_series in matching_metrics.values():
                latest_value = time_series.get_latest_value()
                if latest_value is not None:
                    total_value += latest_value
        
        return {
            "type": "counter",
            "value": total_value,
            "format": widget.config.get("format", "number"),
            "timestamp": time.time()
        }
    
    def _generate_alert_list_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Generate data for alert list widget."""
        max_alerts = widget.config.get("max_alerts", 10)
        active_alerts = self.alert_system.get_active_alerts()[:max_alerts]
        
        alert_data = []
        for alert in active_alerts:
            alert_data.append({
                "id": alert.alert_id,
                "title": alert.title,
                "severity": alert.severity.value,
                "timestamp": alert.timestamp * 1000,  # Convert to milliseconds
                "metric_name": alert.metric_name,
                "current_value": alert.current_value,
                "acknowledged": alert.acknowledged,
                "age_minutes": int((time.time() - alert.timestamp) / 60)
            })
        
        return {
            "type": "alert_list",
            "alerts": alert_data,
            "total_active": len(self.alert_system.get_active_alerts()),
            "timestamp": time.time()
        }
    
    def _generate_insight_panel_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Generate data for insight panel widget."""
        max_insights = widget.config.get("max_insights", 5)
        
        # Get recent analytics results
        analytics_result = self.analytics.analyze_quality_trends(time_window_hours=widget.time_window_hours)
        
        insights_data = []
        for insight in analytics_result.quality_insights[:max_insights]:
            insights_data.append({
                "type": insight.insight_type.value,
                "severity": insight.severity,
                "title": insight.title,
                "description": insight.description,
                "recommendations": insight.recommendations[:3],  # Limit recommendations
                "confidence": insight.confidence,
                "timestamp": insight.timestamp * 1000
            })
        
        return {
            "type": "insight_panel",
            "insights": insights_data,
            "quality_score": analytics_result.quality_score,
            "recommendations": analytics_result.recommendations[:3],
            "timestamp": time.time()
        }
    
    def _generate_trend_indicator_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Generate data for trend indicator widget."""
        trend_data = []
        
        # Get trend analysis for matching metrics
        analytics_result = self.analytics.analyze_quality_trends(time_window_hours=widget.time_window_hours)
        
        for trend in analytics_result.trend_analyses:
            if any(self._pattern_matches(trend.metric_name, pattern) for pattern in widget.metric_patterns):
                trend_data.append({
                    "metric_name": trend.metric_name,
                    "direction": trend.direction.value,
                    "change_percent": trend.percentage_change,
                    "confidence": trend.confidence,
                    "description": trend.change_description
                })
        
        return {
            "type": "trend_indicator", 
            "trends": trend_data,
            "timestamp": time.time()
        }
    
    def _generate_table_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Generate data for table widget."""
        table_rows = []
        
        for pattern in widget.metric_patterns:
            matching_metrics = self.metrics_collector.get_metrics_by_pattern(pattern)
            
            for key, time_series in matching_metrics.items():
                latest_value = time_series.get_latest_value()
                if latest_value is not None:
                    stats = time_series.calculate_statistics(widget.time_window_hours * 3600)
                    
                    table_rows.append({
                        "metric_name": time_series.name,
                        "current_value": latest_value,
                        "mean": stats.get("mean", 0),
                        "min": stats.get("min", 0),
                        "max": stats.get("max", 0),
                        "change": stats.get("change", 0),
                        "labels": time_series.labels
                    })
        
        return {
            "type": "table",
            "columns": ["metric_name", "current_value", "mean", "min", "max", "change"],
            "rows": table_rows,
            "timestamp": time.time()
        }
    
    def _get_metric_color(self, metric_name: str, alpha: float = 1.0) -> str:
        """Get consistent color for a metric."""
        # Simple hash-based color assignment
        import hashlib
        hash_val = int(hashlib.md5(metric_name.encode()).hexdigest()[:6], 16)
        
        r = (hash_val >> 16) & 0xFF
        g = (hash_val >> 8) & 0xFF  
        b = hash_val & 0xFF
        
        if alpha == 1.0:
            return f"rgb({r}, {g}, {b})"
        else:
            return f"rgba({r}, {g}, {b}, {alpha})"
    
    def _pattern_matches(self, text: str, pattern: str) -> bool:
        """Check if text matches glob pattern."""
        import fnmatch
        return fnmatch.fnmatch(text, pattern)
    
    def get_dashboard_data(self, dashboard_id: str) -> Optional[Dict[str, Any]]:
        """Get complete dashboard data."""
        dashboard = self.get_dashboard_config(dashboard_id)
        if not dashboard:
            return None
        
        widget_data = {}
        for widget in dashboard.widgets:
            if widget.enabled:
                data = self.get_widget_data(dashboard_id, widget.widget_id)
                if data:
                    widget_data[widget.widget_id] = data
        
        return {
            "dashboard_id": dashboard.dashboard_id,
            "title": dashboard.title,
            "description": dashboard.description,
            "layout_columns": dashboard.layout_columns,
            "auto_refresh": dashboard.auto_refresh,
            "refresh_interval_seconds": dashboard.refresh_interval_seconds,
            "theme": dashboard.theme,
            "widgets": [asdict(widget) for widget in dashboard.widgets],
            "widget_data": widget_data,
            "timestamp": time.time()
        }
    
    def start_server(self) -> None:
        """Start the web server."""
        if self._is_running:
            self.logger.warning("Dashboard server is already running")
            return
        
        try:
            # Create HTTP server
            handler = self._create_request_handler()
            self._server = HTTPServer((self.host, self.port), handler)
            
            # Start server in separate thread
            self._server_thread = threading.Thread(target=self._run_server, daemon=True)
            self._is_running = True
            self._server_thread.start()
            
            self.logger.info(f"Dashboard server started on http://{self.host}:{self.port}")
            
        except Exception as e:
            self.logger.error(f"Failed to start dashboard server: {e}", exception=e)
            self._is_running = False
    
    def stop_server(self) -> None:
        """Stop the web server."""
        if not self._is_running:
            return
        
        self._is_running = False
        
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        
        if self._server_thread and self._server_thread.is_alive():
            self._server_thread.join(timeout=5.0)
        
        self.logger.info("Dashboard server stopped")
    
    def _run_server(self) -> None:
        """Run the web server."""
        try:
            self._server.serve_forever()
        except Exception as e:
            if self._is_running:  # Only log if unexpected shutdown
                self.logger.error(f"Dashboard server error: {e}", exception=e)
    
    def _create_request_handler(self):
        """Create HTTP request handler class."""
        dashboard_instance = self
        
        class DashboardRequestHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                try:
                    url_parts = urlparse(self.path)
                    path = url_parts.path
                    query_params = parse_qs(url_parts.query)
                    
                    if path == "/":
                        self._serve_dashboard_html()
                    elif path.startswith("/api/dashboards"):
                        self._serve_dashboard_api(path, query_params)
                    elif path.startswith("/api/widgets"):
                        self._serve_widget_api(path, query_params)
                    elif path.startswith("/api/alerts"):
                        self._serve_alerts_api(path, query_params)
                    elif path.startswith("/static/"):
                        self._serve_static_file(path)
                    else:
                        self._send_404()
                        
                except Exception as e:
                    dashboard_instance.logger.error(f"Request handler error: {e}", exception=e)
                    self._send_500(str(e))
            
            def do_POST(self):
                try:
                    url_parts = urlparse(self.path)
                    path = url_parts.path
                    
                    if path.startswith("/api/alerts/") and path.endswith("/acknowledge"):
                        self._handle_alert_acknowledge(path)
                    elif path.startswith("/api/alerts/") and path.endswith("/resolve"):
                        self._handle_alert_resolve(path)
                    else:
                        self._send_404()
                        
                except Exception as e:
                    dashboard_instance.logger.error(f"POST handler error: {e}", exception=e)
                    self._send_500(str(e))
            
            def _serve_dashboard_html(self):
                html_content = self._generate_dashboard_html()
                self._send_response(200, html_content, "text/html")
            
            def _serve_dashboard_api(self, path, query_params):
                if path == "/api/dashboards":
                    # List all dashboards
                    dashboards = dashboard_instance.list_dashboards()
                    self._send_json_response(200, dashboards)
                elif path.startswith("/api/dashboards/"):
                    # Get specific dashboard
                    dashboard_id = path.split("/")[-1]
                    dashboard_data = dashboard_instance.get_dashboard_data(dashboard_id)
                    if dashboard_data:
                        self._send_json_response(200, dashboard_data)
                    else:
                        self._send_404()
            
            def _serve_widget_api(self, path, query_params):
                parts = path.split("/")
                if len(parts) >= 5:  # /api/widgets/{dashboard_id}/{widget_id}
                    dashboard_id = parts[3]
                    widget_id = parts[4] 
                    widget_data = dashboard_instance.get_widget_data(dashboard_id, widget_id)
                    if widget_data:
                        self._send_json_response(200, widget_data)
                    else:
                        self._send_404()
                else:
                    self._send_404()
            
            def _serve_alerts_api(self, path, query_params):
                if path == "/api/alerts":
                    # Get active alerts
                    alerts = dashboard_instance.alert_system.get_active_alerts()
                    alert_data = [
                        {
                            "id": alert.alert_id,
                            "title": alert.title,
                            "severity": alert.severity.value,
                            "timestamp": alert.timestamp * 1000,
                            "acknowledged": alert.acknowledged,
                            "resolved": alert.resolved
                        }
                        for alert in alerts
                    ]
                    self._send_json_response(200, alert_data)
                elif path == "/api/alerts/stats":
                    # Get alert statistics
                    stats = dashboard_instance.alert_system.get_alert_statistics()
                    self._send_json_response(200, stats)
                else:
                    self._send_404()
            
            def _handle_alert_acknowledge(self, path):
                alert_id = path.split("/")[-2]
                success = dashboard_instance.alert_system.acknowledge_alert(alert_id, "dashboard")
                self._send_json_response(200 if success else 404, {"success": success})
            
            def _handle_alert_resolve(self, path):
                alert_id = path.split("/")[-2]
                success = dashboard_instance.alert_system.resolve_alert(alert_id, "dashboard")
                self._send_json_response(200 if success else 404, {"success": success})
            
            def _serve_static_file(self, path):
                # Serve static files (CSS, JS, images)
                self._send_404()  # Placeholder - would serve actual static files
            
            def _send_response(self, status_code, content, content_type):
                self.send_response(status_code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(content.encode())))
                self.end_headers()
                self.wfile.write(content.encode())
            
            def _send_json_response(self, status_code, data):
                content = json.dumps(data, default=str)
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(content.encode())))
                self.end_headers()
                self.wfile.write(content.encode())
            
            def _send_404(self):
                self._send_response(404, "Not Found", "text/plain")
            
            def _send_500(self, error_message):
                self._send_response(500, f"Internal Server Error: {error_message}", "text/plain")
            
            def _generate_dashboard_html(self):
                """Generate basic dashboard HTML."""
                return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quality Control Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .header { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .dashboard-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .widget { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .widget h3 { margin-top: 0; color: #333; }
        .loading { text-align: center; color: #666; padding: 20px; }
        .error { color: #d32f2f; background: #ffebee; padding: 10px; border-radius: 4px; }
        .gauge { text-align: center; font-size: 2em; font-weight: bold; }
        .alerts { max-height: 400px; overflow-y: auto; }
        .alert-item { padding: 10px; margin: 5px 0; border-left: 4px solid; border-radius: 4px; }
        .alert-critical { background: #ffebee; border-color: #f44336; }
        .alert-warning { background: #fff3e0; border-color: #ff9800; }
        .alert-info { background: #e3f2fd; border-color: #2196f3; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Quality Control Dashboard</h1>
        <p>Real-time quality monitoring and insights</p>
    </div>
    
    <div class="dashboard-grid">
        <div class="widget">
            <h3>Overall Quality Score</h3>
            <div id="quality-score" class="gauge loading">Loading...</div>
        </div>
        
        <div class="widget">
            <h3>Active Alerts</h3>
            <div id="active-alerts" class="alerts loading">Loading...</div>
        </div>
        
        <div class="widget">
            <h3>Quality Trends</h3>
            <canvas id="trends-chart"></canvas>
        </div>
        
        <div class="widget">
            <h3>Quality Insights</h3>
            <div id="quality-insights" class="loading">Loading...</div>
        </div>
    </div>
    
    <script>
        // Basic dashboard JavaScript
        async function loadDashboard() {
            try {
                const response = await fetch('/api/dashboards/default');
                const dashboard = await response.json();
                
                updateQualityScore(dashboard);
                updateAlerts();
                updateInsights(dashboard);
                
            } catch (error) {
                console.error('Failed to load dashboard:', error);
            }
        }
        
        async function updateQualityScore(dashboard) {
            const scoreElement = document.getElementById('quality-score');
            try {
                const widgetData = dashboard.widget_data['quality_score_gauge'];
                if (widgetData) {
                    const score = Math.round(widgetData.value);
                    const color = score >= 90 ? 'green' : score >= 75 ? 'orange' : 'red';
                    scoreElement.innerHTML = `<div style="color: ${color}">${score}</div>`;
                    scoreElement.classList.remove('loading');
                }
            } catch (error) {
                scoreElement.innerHTML = '<div class="error">Error loading quality score</div>';
            }
        }
        
        async function updateAlerts() {
            const alertsElement = document.getElementById('active-alerts');
            try {
                const response = await fetch('/api/alerts');
                const alerts = await response.json();
                
                if (alerts.length === 0) {
                    alertsElement.innerHTML = '<p>No active alerts</p>';
                } else {
                    alertsElement.innerHTML = alerts.map(alert => `
                        <div class="alert-item alert-${alert.severity}">
                            <strong>${alert.title}</strong><br>
                            <small>${new Date(alert.timestamp).toLocaleString()}</small>
                        </div>
                    `).join('');
                }
                alertsElement.classList.remove('loading');
            } catch (error) {
                alertsElement.innerHTML = '<div class="error">Error loading alerts</div>';
            }
        }
        
        function updateInsights(dashboard) {
            const insightsElement = document.getElementById('quality-insights');
            try {
                const widgetData = dashboard.widget_data['quality_insights'];
                if (widgetData && widgetData.insights) {
                    insightsElement.innerHTML = widgetData.insights.map(insight => `
                        <div style="margin-bottom: 15px; padding: 10px; background: #f8f9fa; border-radius: 4px;">
                            <strong>${insight.title}</strong><br>
                            <small>${insight.description}</small>
                        </div>
                    `).join('');
                } else {
                    insightsElement.innerHTML = '<p>No insights available</p>';
                }
                insightsElement.classList.remove('loading');
            } catch (error) {
                insightsElement.innerHTML = '<div class="error">Error loading insights</div>';
            }
        }
        
        // Load dashboard on page load
        loadDashboard();
        
        // Auto-refresh every 30 seconds
        setInterval(loadDashboard, 30000);
    </script>
</body>
</html>
                """.strip()
            
            def log_message(self, format, *args):
                # Suppress default log messages
                pass
        
        return DashboardRequestHandler
    
    def export_dashboard_config(self, dashboard_id: str, output_path: Path) -> bool:
        """Export dashboard configuration to file."""
        dashboard = self.get_dashboard_config(dashboard_id)
        if not dashboard:
            return False
        
        try:
            config_dict = asdict(dashboard)
            with open(output_path, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
            
            self.logger.info(f"Exported dashboard config to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export dashboard config: {e}", exception=e)
            return False
    
    def import_dashboard_config(self, config_path: Path) -> bool:
        """Import dashboard configuration from file."""
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            # Convert back to objects
            widgets = []
            for widget_dict in config_dict.get('widgets', []):
                widget = DashboardWidget(
                    widget_id=widget_dict['widget_id'],
                    title=widget_dict['title'],
                    widget_type=WidgetType(widget_dict['widget_type']),
                    metric_patterns=widget_dict.get('metric_patterns', []),
                    time_window_hours=widget_dict.get('time_window_hours', 24),
                    refresh_interval_seconds=widget_dict.get('refresh_interval_seconds', 30),
                    position=tuple(widget_dict.get('position', [0, 0])),
                    size=tuple(widget_dict.get('size', [1, 1])),
                    config=widget_dict.get('config', {}),
                    enabled=widget_dict.get('enabled', True)
                )
                widgets.append(widget)
            
            dashboard = DashboardConfig(
                dashboard_id=config_dict['dashboard_id'],
                title=config_dict['title'],
                description=config_dict['description'],
                widgets=widgets,
                layout_columns=config_dict.get('layout_columns', 12),
                auto_refresh=config_dict.get('auto_refresh', True),
                refresh_interval_seconds=config_dict.get('refresh_interval_seconds', 30),
                theme=config_dict.get('theme', 'light'),
                metadata=config_dict.get('metadata', {})
            )
            
            self.add_dashboard(dashboard)
            self.logger.info(f"Imported dashboard config from {config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to import dashboard config: {e}", exception=e)
            return False