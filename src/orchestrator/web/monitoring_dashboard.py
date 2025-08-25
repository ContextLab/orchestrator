"""
Web-based monitoring dashboard for wrapper performance tracking.

This module provides a comprehensive web interface for monitoring wrapper
performance, cost analytics, and system health in real-time.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
import asyncio

# Flask and web dependencies - with fallbacks for optional imports
try:
    from flask import Flask, render_template, jsonify, request, Response
    from flask_socketio import SocketIO, emit
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False
    Flask = None
    SocketIO = None
    logger.warning("Flask not available - web dashboard disabled")

try:
    import plotly.graph_objects as go
    import plotly.utils
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    logger.warning("Plotly not available - chart generation disabled")

from collections import defaultdict
from threading import Lock, Thread
import time

from ..core.wrapper_monitoring import WrapperMonitoring, AlertSeverity, Alert
from ..integrations.cost_monitoring import CostMonitoringIntegration

logger = logging.getLogger(__name__)


@dataclass
class DashboardConfig:
    """Configuration for monitoring dashboard."""
    
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    refresh_interval_seconds: int = 30
    max_data_points: int = 1000
    enable_real_time: bool = True
    enable_alerts: bool = True
    enable_cost_tracking: bool = True
    themes: List[str] = None
    default_theme: str = "light"
    auth_required: bool = False
    secret_key: str = "dev-key-change-in-production"
    
    def __post_init__(self):
        if self.themes is None:
            self.themes = ["light", "dark", "high-contrast"]


@dataclass 
class ChartConfig:
    """Configuration for chart generation."""
    
    chart_type: str = "line"
    title: str = ""
    x_axis_title: str = "Time"
    y_axis_title: str = "Value"
    color_scheme: str = "viridis"
    show_legend: bool = True
    height: int = 400
    width: Optional[int] = None


class MonitoringDashboard:
    """Web-based monitoring dashboard for wrapper performance."""
    
    def __init__(
        self,
        monitoring: WrapperMonitoring,
        cost_monitoring: Optional[CostMonitoringIntegration] = None,
        config: Optional[DashboardConfig] = None
    ):
        """
        Initialize monitoring dashboard.
        
        Args:
            monitoring: WrapperMonitoring instance
            cost_monitoring: Optional cost monitoring integration
            config: Dashboard configuration
        """
        if not HAS_FLASK:
            raise RuntimeError("Flask is required for web dashboard")
        
        self.monitoring = monitoring
        self.cost_monitoring = cost_monitoring
        self.config = config or DashboardConfig()
        
        # Flask app setup
        self.app = Flask(__name__, 
                        template_folder='templates',
                        static_folder='static')
        self.app.secret_key = self.config.secret_key
        
        # SocketIO for real-time updates
        if self.config.enable_real_time:
            try:
                self.socketio = SocketIO(self.app, cors_allowed_origins="*")
                self._setup_socketio_handlers()
            except Exception as e:
                logger.warning(f"Failed to setup SocketIO: {e}")
                self.socketio = None
        else:
            self.socketio = None
        
        # Data cache
        self._cache: Dict[str, Any] = {}
        self._cache_lock = Lock()
        self._cache_timestamps: Dict[str, datetime] = {}
        
        # Real-time data
        self._real_time_thread: Optional[Thread] = None
        self._stop_real_time = False
        
        # Setup routes
        self._setup_routes()
        
        logger.info(f"Initialized monitoring dashboard on {self.config.host}:{self.config.port}")
    
    def start(self, threaded: bool = True) -> None:
        """Start the dashboard server."""
        
        # Start real-time data thread
        if self.config.enable_real_time and not self._real_time_thread:
            self._real_time_thread = Thread(target=self._real_time_worker, daemon=True)
            self._real_time_thread.start()
        
        # Start Flask app
        if self.socketio:
            self.socketio.run(
                self.app,
                host=self.config.host,
                port=self.config.port,
                debug=self.config.debug
            )
        else:
            self.app.run(
                host=self.config.host,
                port=self.config.port,
                debug=self.config.debug,
                threaded=threaded
            )
    
    def stop(self) -> None:
        """Stop the dashboard server."""
        self._stop_real_time = True
        if self._real_time_thread:
            self._real_time_thread.join(timeout=5)
    
    def _setup_routes(self) -> None:
        """Setup Flask routes for dashboard."""
        
        @self.app.route('/')
        def dashboard_home():
            """Main dashboard page."""
            return render_template('dashboard.html', config=asdict(self.config))
        
        @self.app.route('/api/system-health')
        def get_system_health():
            """Get overall system health metrics."""
            try:
                health_data = self.monitoring.get_system_health()
                
                # Add cost information if available
                if self.cost_monitoring:
                    cost_analysis = self.cost_monitoring.get_cost_analysis(time_range_hours=24)
                    health_data['cost_summary'] = {
                        'total_cost_24h': cost_analysis.get('costs', {}).get('total', 0.0),
                        'total_operations_24h': cost_analysis.get('operations', {}).get('total', 0)
                    }
                
                return jsonify(health_data)
                
            except Exception as e:
                logger.error(f"Failed to get system health: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/wrapper-stats/<wrapper_name>')
        def get_wrapper_stats(wrapper_name: str):
            """Get detailed statistics for a specific wrapper."""
            try:
                cache_key = f"wrapper_stats_{wrapper_name}"
                cached_data = self._get_cached_data(cache_key, max_age_seconds=30)
                
                if cached_data is not None:
                    return jsonify(cached_data)
                
                # Get stats from monitoring
                stats = self.monitoring.get_wrapper_stats(wrapper_name)
                health = self.monitoring.get_wrapper_health(wrapper_name)
                
                response_data = {
                    "stats": stats,
                    "health": asdict(health),
                    "charts": self._generate_wrapper_charts(wrapper_name)
                }
                
                # Add cost information if available
                if self.cost_monitoring:
                    cost_analysis = self.cost_monitoring.get_cost_analysis(
                        wrapper_name=wrapper_name, 
                        time_range_hours=24
                    )
                    response_data['cost_analysis'] = cost_analysis
                    
                    budget_status = self.cost_monitoring.get_budget_status(wrapper_name)
                    if 'error' not in budget_status:
                        response_data['budget_status'] = budget_status
                
                # Cache the response
                self._set_cached_data(cache_key, response_data)
                
                return jsonify(response_data)
                
            except Exception as e:
                logger.error(f"Failed to get wrapper stats for {wrapper_name}: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/alerts')
        def get_active_alerts():
            """Get current active alerts."""
            try:
                severity_filter = request.args.get('severity')
                wrapper_filter = request.args.get('wrapper')
                limit = request.args.get('limit', type=int)
                
                alerts = self.monitoring.get_alerts(
                    wrapper_name=wrapper_filter,
                    severity=AlertSeverity(severity_filter) if severity_filter else None,
                    include_resolved=False
                )
                
                if limit:
                    alerts = alerts[:limit]
                
                return jsonify([asdict(alert) for alert in alerts])
                
            except Exception as e:
                logger.error(f"Failed to get alerts: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/performance-metrics')
        def get_performance_metrics():
            """Get performance metrics for charting."""
            try:
                wrapper_name = request.args.get('wrapper')
                time_range = request.args.get('range', '24h')
                
                cache_key = f"performance_metrics_{wrapper_name}_{time_range}"
                cached_data = self._get_cached_data(cache_key, max_age_seconds=60)
                
                if cached_data is not None:
                    return jsonify(cached_data)
                
                metrics = self._get_performance_metrics(wrapper_name, time_range)
                
                self._set_cached_data(cache_key, metrics)
                return jsonify(metrics)
                
            except Exception as e:
                logger.error(f"Failed to get performance metrics: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/cost-analysis')
        def get_cost_analysis():
            """Get cost analysis data."""
            if not self.cost_monitoring:
                return jsonify({"error": "Cost monitoring not available"}), 404
            
            try:
                wrapper_name = request.args.get('wrapper')
                time_range = request.args.get('range', '7d')
                
                # Convert time range to hours
                time_range_hours = self._parse_time_range(time_range)
                
                cost_data = self.cost_monitoring.get_cost_analysis(
                    wrapper_name=wrapper_name,
                    time_range_hours=time_range_hours
                )
                
                # Add budget information
                if wrapper_name:
                    budget_status = self.cost_monitoring.get_budget_status(wrapper_name)
                    if 'error' not in budget_status:
                        cost_data['budget_status'] = budget_status
                    
                    # Add optimization recommendations
                    recommendations = self.cost_monitoring.get_cost_optimization_recommendations(wrapper_name)
                    cost_data['recommendations'] = recommendations
                
                return jsonify(cost_data)
                
            except Exception as e:
                logger.error(f"Failed to get cost analysis: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/wrappers')
        def list_wrappers():
            """List all available wrappers."""
            try:
                system_health = self.monitoring.get_system_health()
                
                # Get wrapper names from health data
                wrapper_names = []
                for wrapper_name in self.monitoring._wrapper_health.keys():
                    health = self.monitoring.get_wrapper_health(wrapper_name)
                    wrapper_info = {
                        "name": wrapper_name,
                        "health_score": health.health_score,
                        "is_healthy": health.is_healthy,
                        "total_operations": health.total_operations,
                        "success_rate": health.success_rate,
                        "last_check": health.last_check.isoformat()
                    }
                    wrapper_names.append(wrapper_info)
                
                # Sort by health score (descending)
                wrapper_names.sort(key=lambda x: x['health_score'], reverse=True)
                
                return jsonify({
                    "wrappers": wrapper_names,
                    "system_summary": system_health
                })
                
            except Exception as e:
                logger.error(f"Failed to list wrappers: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/export/<wrapper_name>')
        def export_wrapper_data(wrapper_name: str):
            """Export wrapper data."""
            try:
                format_type = request.args.get('format', 'json')
                days = request.args.get('days', 7, type=int)
                
                start_time = datetime.utcnow() - timedelta(days=days)
                
                if self.cost_monitoring:
                    data = self.cost_monitoring.export_cost_data(
                        wrapper_name=wrapper_name,
                        start_time=start_time,
                        format=format_type
                    )
                else:
                    # Export monitoring data
                    data = self.monitoring.export_metrics(
                        wrapper_name=wrapper_name,
                        start_time=start_time
                    )
                
                if format_type == 'csv':
                    return Response(
                        data,
                        mimetype='text/csv',
                        headers={'Content-Disposition': f'attachment; filename={wrapper_name}_data.csv'}
                    )
                else:
                    return jsonify(data)
                    
            except Exception as e:
                logger.error(f"Failed to export data for {wrapper_name}: {e}")
                return jsonify({"error": str(e)}), 500
    
    def _setup_socketio_handlers(self) -> None:
        """Setup SocketIO handlers for real-time updates."""
        
        if not self.socketio:
            return
        
        @self.socketio.on('connect')
        def handle_connect():
            logger.debug("Client connected to real-time updates")
            emit('status', {'msg': 'Connected to monitoring dashboard'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            logger.debug("Client disconnected from real-time updates")
        
        @self.socketio.on('subscribe_wrapper')
        def handle_wrapper_subscription(data):
            wrapper_name = data.get('wrapper_name')
            if wrapper_name:
                logger.debug(f"Client subscribed to wrapper: {wrapper_name}")
                # Join room for wrapper-specific updates
                # socketio.join_room(f"wrapper_{wrapper_name}")
    
    def _real_time_worker(self) -> None:
        """Worker thread for real-time data updates."""
        
        logger.info("Started real-time monitoring worker")
        
        while not self._stop_real_time:
            try:
                if self.socketio:
                    # Get current system health
                    health_data = self.monitoring.get_system_health()
                    
                    # Emit system health update
                    self.socketio.emit('system_health_update', health_data)
                    
                    # Get recent alerts
                    alerts = self.monitoring.get_alerts(include_resolved=False)
                    recent_alerts = [
                        asdict(alert) for alert in alerts[-5:]  # Last 5 alerts
                        if alert.timestamp >= datetime.utcnow() - timedelta(minutes=5)
                    ]
                    
                    if recent_alerts:
                        self.socketio.emit('new_alerts', recent_alerts)
                    
                    # Emit wrapper-specific updates
                    for wrapper_name in self.monitoring._wrapper_health.keys():
                        wrapper_health = self.monitoring.get_wrapper_health(wrapper_name)
                        self.socketio.emit('wrapper_health_update', {
                            'wrapper_name': wrapper_name,
                            'health_data': asdict(wrapper_health)
                        })
                
                # Sleep for refresh interval
                time.sleep(self.config.refresh_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in real-time worker: {e}")
                time.sleep(5)  # Brief pause on error
        
        logger.info("Real-time monitoring worker stopped")
    
    def _generate_wrapper_charts(self, wrapper_name: str) -> Dict[str, Any]:
        """Generate chart data for a specific wrapper."""
        
        if not HAS_PLOTLY:
            return {"error": "Plotly not available for chart generation"}
        
        try:
            # Get recent metrics
            metrics = self.monitoring.export_metrics(
                wrapper_name=wrapper_name,
                start_time=datetime.utcnow() - timedelta(hours=24)
            )
            
            if not metrics:
                return {"message": "No data available for charting"}
            
            charts = {}
            
            # Success rate over time
            charts['success_rate'] = self._create_success_rate_chart(metrics)
            
            # Response time distribution
            charts['response_time'] = self._create_response_time_chart(metrics)
            
            # Error frequency chart
            charts['error_frequency'] = self._create_error_frequency_chart(metrics)
            
            # Cost analysis chart if available
            if self.cost_monitoring:
                charts['cost_analysis'] = self._create_cost_chart(wrapper_name)
            
            return charts
            
        except Exception as e:
            logger.error(f"Failed to generate charts for {wrapper_name}: {e}")
            return {"error": str(e)}
    
    def _create_success_rate_chart(self, metrics: List[Dict[str, Any]]) -> str:
        """Create success rate chart data."""
        
        if not HAS_PLOTLY:
            return ""
        
        try:
            # Group metrics by hour
            hourly_data = defaultdict(list)
            
            for metric in metrics:
                start_time = datetime.fromisoformat(metric['start_time'])
                hour = start_time.replace(minute=0, second=0, microsecond=0)
                hourly_data[hour].append(metric['success'])
            
            if not hourly_data:
                return ""
            
            timestamps = sorted(hourly_data.keys())
            success_rates = [
                sum(hourly_data[ts]) / len(hourly_data[ts]) * 100
                for ts in timestamps
            ]
            
            fig = go.Figure(data=go.Scatter(
                x=timestamps,
                y=success_rates,
                mode='lines+markers',
                name='Success Rate %',
                line=dict(color='green', width=2),
                marker=dict(size=6)
            ))
            
            fig.update_layout(
                title="Success Rate Over Time",
                xaxis_title="Time",
                yaxis_title="Success Rate (%)",
                yaxis=dict(range=[0, 100]),
                height=400,
                template="plotly_white"
            )
            
            return json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))
            
        except Exception as e:
            logger.error(f"Failed to create success rate chart: {e}")
            return ""
    
    def _create_response_time_chart(self, metrics: List[Dict[str, Any]]) -> str:
        """Create response time distribution chart."""
        
        if not HAS_PLOTLY:
            return ""
        
        try:
            response_times = [
                metric.get('duration_ms', 0) for metric in metrics 
                if metric.get('duration_ms', 0) > 0
            ]
            
            if not response_times:
                return ""
            
            fig = go.Figure(data=go.Histogram(
                x=response_times,
                nbinsx=30,
                name='Response Time Distribution',
                marker=dict(color='blue', opacity=0.7)
            ))
            
            fig.update_layout(
                title="Response Time Distribution",
                xaxis_title="Response Time (ms)",
                yaxis_title="Frequency",
                height=400,
                template="plotly_white"
            )
            
            return json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))
            
        except Exception as e:
            logger.error(f"Failed to create response time chart: {e}")
            return ""
    
    def _create_error_frequency_chart(self, metrics: List[Dict[str, Any]]) -> str:
        """Create error frequency chart."""
        
        if not HAS_PLOTLY:
            return ""
        
        try:
            # Group errors by hour
            hourly_errors = defaultdict(int)
            hourly_total = defaultdict(int)
            
            for metric in metrics:
                start_time = datetime.fromisoformat(metric['start_time'])
                hour = start_time.replace(minute=0, second=0, microsecond=0)
                
                hourly_total[hour] += 1
                if not metric.get('success', True):
                    hourly_errors[hour] += 1
            
            if not hourly_total:
                return ""
            
            timestamps = sorted(hourly_total.keys())
            error_rates = [
                (hourly_errors[ts] / hourly_total[ts]) * 100
                for ts in timestamps
            ]
            
            fig = go.Figure(data=go.Scatter(
                x=timestamps,
                y=error_rates,
                mode='lines+markers',
                name='Error Rate %',
                line=dict(color='red', width=2),
                marker=dict(size=6)
            ))
            
            fig.update_layout(
                title="Error Rate Over Time",
                xaxis_title="Time",
                yaxis_title="Error Rate (%)",
                yaxis=dict(range=[0, max(error_rates) * 1.1] if error_rates else [0, 1]),
                height=400,
                template="plotly_white"
            )
            
            return json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))
            
        except Exception as e:
            logger.error(f"Failed to create error frequency chart: {e}")
            return ""
    
    def _create_cost_chart(self, wrapper_name: str) -> str:
        """Create cost analysis chart."""
        
        if not HAS_PLOTLY or not self.cost_monitoring:
            return ""
        
        try:
            # Get cost data for the last 24 hours
            cost_analysis = self.cost_monitoring.get_cost_analysis(
                wrapper_name=wrapper_name,
                time_range_hours=24
            )
            
            if not cost_analysis or cost_analysis.get('operations', {}).get('total', 0) == 0:
                return ""
            
            # Create cost breakdown pie chart
            breakdown = cost_analysis.get('costs', {}).get('breakdown', {})
            
            if not any(breakdown.values()):
                return ""
            
            labels = list(breakdown.keys())
            values = list(breakdown.values())
            
            fig = go.Figure(data=go.Pie(
                labels=labels,
                values=values,
                hole=0.3
            ))
            
            fig.update_layout(
                title=f"Cost Breakdown - {wrapper_name}",
                height=400,
                template="plotly_white"
            )
            
            return json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))
            
        except Exception as e:
            logger.error(f"Failed to create cost chart for {wrapper_name}: {e}")
            return ""
    
    def _get_performance_metrics(self, wrapper_name: Optional[str], time_range: str) -> Dict[str, Any]:
        """Get performance metrics for the specified time range."""
        
        time_range_hours = self._parse_time_range(time_range)
        start_time = datetime.utcnow() - timedelta(hours=time_range_hours)
        
        metrics = self.monitoring.export_metrics(
            wrapper_name=wrapper_name,
            start_time=start_time
        )
        
        if not metrics:
            return {"message": "No metrics available"}
        
        # Calculate aggregated metrics
        total_operations = len(metrics)
        successful_operations = sum(1 for m in metrics if m.get('success', True))
        failed_operations = total_operations - successful_operations
        
        response_times = [m.get('duration_ms', 0) for m in metrics if m.get('duration_ms', 0) > 0]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        fallback_operations = sum(1 for m in metrics if m.get('fallback_used', False))
        
        return {
            "time_range": time_range,
            "period_start": start_time.isoformat(),
            "period_end": datetime.utcnow().isoformat(),
            "metrics": {
                "total_operations": total_operations,
                "successful_operations": successful_operations,
                "failed_operations": failed_operations,
                "fallback_operations": fallback_operations,
                "success_rate": successful_operations / max(1, total_operations),
                "error_rate": failed_operations / max(1, total_operations),
                "fallback_rate": fallback_operations / max(1, total_operations),
                "average_response_time_ms": avg_response_time,
                "min_response_time_ms": min(response_times) if response_times else 0,
                "max_response_time_ms": max(response_times) if response_times else 0
            },
            "raw_data_points": len(metrics)
        }
    
    def _parse_time_range(self, time_range: str) -> int:
        """Parse time range string to hours."""
        
        time_range = time_range.lower()
        
        if time_range.endswith('h'):
            return int(time_range[:-1])
        elif time_range.endswith('d'):
            return int(time_range[:-1]) * 24
        elif time_range.endswith('w'):
            return int(time_range[:-1]) * 24 * 7
        else:
            # Default to hours
            try:
                return int(time_range)
            except ValueError:
                return 24  # Default to 24 hours
    
    def _get_cached_data(self, key: str, max_age_seconds: int = 300) -> Optional[Any]:
        """Get cached data if still valid."""
        
        with self._cache_lock:
            if key in self._cache and key in self._cache_timestamps:
                age = (datetime.utcnow() - self._cache_timestamps[key]).total_seconds()
                if age < max_age_seconds:
                    return self._cache[key]
                else:
                    # Remove expired cache entry
                    del self._cache[key]
                    del self._cache_timestamps[key]
        
        return None
    
    def _set_cached_data(self, key: str, data: Any) -> None:
        """Set cached data with timestamp."""
        
        with self._cache_lock:
            self._cache[key] = data
            self._cache_timestamps[key] = datetime.utcnow()
            
            # Simple cache cleanup - remove oldest entries if cache is too large
            if len(self._cache) > 1000:
                oldest_key = min(self._cache_timestamps, key=self._cache_timestamps.get)
                del self._cache[oldest_key]
                del self._cache_timestamps[oldest_key]


# Factory function for easy instantiation
def create_monitoring_dashboard(
    monitoring: WrapperMonitoring,
    cost_monitoring: Optional[CostMonitoringIntegration] = None,
    config: Optional[DashboardConfig] = None
) -> MonitoringDashboard:
    """Create a monitoring dashboard."""
    return MonitoringDashboard(monitoring, cost_monitoring, config)