"""
Real-time monitoring dashboard for wrapper performance and system health.

This module provides a Flask-based web interface for monitoring:
- Wrapper operation metrics and health status
- Real-time performance analytics
- Cost tracking and budget monitoring
- System-wide health dashboards
- Interactive charts and visualizations

Integration with Issue #251: Configuration & Monitoring infrastructure
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, render_template, request, Response
from flask_socketio import SocketIO, emit
import plotly
import plotly.graph_objs as go
from plotly.utils import PlotlyJSONEncoder

from ..core.wrapper_monitoring import WrapperMonitoring
from ..analytics.performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)


class MonitoringDashboard:
    """
    Real-time monitoring dashboard for wrapper system.
    
    Provides web-based interface for monitoring wrapper performance,
    health status, and operational metrics with real-time updates.
    """
    
    def __init__(self, monitoring: WrapperMonitoring, performance_monitor: PerformanceMonitor):
        """
        Initialize monitoring dashboard.
        
        Args:
            monitoring: Wrapper monitoring system instance
            performance_monitor: Performance monitoring system instance
        """
        self.monitoring = monitoring
        self.performance_monitor = performance_monitor
        
        # Initialize Flask app
        self.app = Flask(__name__, 
                        template_folder='templates',
                        static_folder='static')
        self.app.config['SECRET_KEY'] = 'monitoring-dashboard-secret'
        
        # Initialize SocketIO for real-time updates
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Setup routes
        self._setup_routes()
        self._setup_socketio_events()
        
        # Dashboard state
        self._last_metrics_update = datetime.utcnow()
        self._update_interval = 5  # seconds
        
        logger.info("Monitoring dashboard initialized")
    
    def _setup_routes(self) -> None:
        """Setup Flask routes for dashboard endpoints."""
        
        @self.app.route('/')
        def dashboard_home():
            """Main dashboard page."""
            return render_template('dashboard/index.html')
        
        @self.app.route('/health')
        def health_check():
            """Health check endpoint."""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat(),
                'version': '1.0.0'
            })
        
        @self.app.route('/api/system/health')
        def system_health():
            """Get overall system health status."""
            try:
                health_data = self.monitoring.get_system_health()
                return jsonify({
                    'success': True,
                    'data': health_data,
                    'timestamp': datetime.utcnow().isoformat()
                })
            except Exception as e:
                logger.error(f"Error getting system health: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                }), 500
        
        @self.app.route('/api/wrappers/health')
        def wrappers_health():
            """Get health status for all wrappers."""
            try:
                # Get all wrapper names from monitoring system
                wrapper_names = self.monitoring.get_active_wrappers()
                health_data = {}
                
                for wrapper_name in wrapper_names:
                    health_status = self.monitoring.get_wrapper_health(wrapper_name)
                    health_data[wrapper_name] = {
                        'health_score': health_status.health_score,
                        'status': health_status.status,
                        'last_activity': health_status.last_activity.isoformat() if health_status.last_activity else None,
                        'success_rate': health_status.success_rate,
                        'error_rate': health_status.error_rate,
                        'fallback_rate': health_status.fallback_rate,
                        'total_operations': health_status.total_operations
                    }
                
                return jsonify({
                    'success': True,
                    'data': health_data,
                    'timestamp': datetime.utcnow().isoformat()
                })
            except Exception as e:
                logger.error(f"Error getting wrapper health: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                }), 500
        
        @self.app.route('/api/wrappers/<wrapper_name>/metrics')
        def wrapper_metrics(wrapper_name: str):
            """Get detailed metrics for specific wrapper."""
            try:
                # Get time range from query parameters
                hours_str = request.args.get('hours', '24')
                try:
                    hours = int(hours_str)
                except (ValueError, TypeError):
                    hours = 24
                since = datetime.utcnow() - timedelta(hours=hours)
                
                # Get metrics data
                metrics = self.monitoring.get_wrapper_metrics(wrapper_name, since)
                
                return jsonify({
                    'success': True,
                    'data': metrics,
                    'wrapper_name': wrapper_name,
                    'time_range_hours': hours,
                    'timestamp': datetime.utcnow().isoformat()
                })
            except Exception as e:
                logger.error(f"Error getting wrapper metrics for {wrapper_name}: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                }), 500
        
        @self.app.route('/api/metrics/export')
        def export_metrics():
            """Export all metrics data."""
            try:
                format_type = request.args.get('format', 'json')
                
                if format_type == 'json':
                    metrics = self.monitoring.export_metrics()
                    return jsonify({
                        'success': True,
                        'data': metrics,
                        'export_format': 'json',
                        'timestamp': datetime.utcnow().isoformat()
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': f'Unsupported export format: {format_type}',
                        'timestamp': datetime.utcnow().isoformat()
                    }), 400
            except Exception as e:
                logger.error(f"Error exporting metrics: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                }), 500
        
        @self.app.route('/api/charts/performance')
        def performance_chart():
            """Generate performance chart data."""
            try:
                wrapper_name = request.args.get('wrapper', 'all')
                hours_str = request.args.get('hours', '24')
                try:
                    hours = int(hours_str)
                except (ValueError, TypeError):
                    hours = 24
                
                chart_data = self._generate_performance_chart(wrapper_name, hours)
                
                return Response(
                    json.dumps(chart_data, cls=PlotlyJSONEncoder),
                    mimetype='application/json'
                )
            except Exception as e:
                logger.error(f"Error generating performance chart: {e}")
                return jsonify({
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                }), 500
        
        @self.app.route('/api/charts/health')
        def health_chart():
            """Generate health status chart data."""
            try:
                chart_data = self._generate_health_chart()
                
                return Response(
                    json.dumps(chart_data, cls=PlotlyJSONEncoder),
                    mimetype='application/json'
                )
            except Exception as e:
                logger.error(f"Error generating health chart: {e}")
                return jsonify({
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                }), 500
    
    def _setup_socketio_events(self) -> None:
        """Setup SocketIO events for real-time updates."""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection."""
            logger.info(f"Client connected: {request.sid}")
            emit('status', {'msg': 'Connected to monitoring dashboard'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection."""
            logger.info(f"Client disconnected: {request.sid}")
        
        @self.socketio.on('subscribe_metrics')
        def handle_subscribe_metrics(data):
            """Handle metrics subscription."""
            wrapper_name = data.get('wrapper', 'all')
            logger.info(f"Client {request.sid} subscribed to metrics for {wrapper_name}")
            
            # Send initial data
            try:
                if wrapper_name == 'all':
                    health_data = self.monitoring.get_system_health()
                else:
                    health_data = self.monitoring.get_wrapper_health(wrapper_name)
                
                emit('metrics_update', {
                    'wrapper': wrapper_name,
                    'data': health_data,
                    'timestamp': datetime.utcnow().isoformat()
                })
            except Exception as e:
                emit('error', {'message': str(e)})
    
    def _generate_performance_chart(self, wrapper_name: str, hours: int) -> Dict[str, Any]:
        """
        Generate performance chart data.
        
        Args:
            wrapper_name: Name of wrapper or 'all' for system-wide
            hours: Number of hours of data to include
            
        Returns:
            Plotly chart configuration dictionary
        """
        since = datetime.utcnow() - timedelta(hours=hours)
        
        if wrapper_name == 'all':
            # System-wide performance chart
            wrappers = self.monitoring.get_active_wrappers()
            fig = go.Figure()
            
            for wrapper in wrappers:
                metrics = self.monitoring.get_wrapper_metrics(wrapper, since)
                if metrics:
                    times = [m['timestamp'] for m in metrics]
                    success_rates = [m.get('success_rate', 0) * 100 for m in metrics]
                    
                    fig.add_trace(go.Scatter(
                        x=times,
                        y=success_rates,
                        mode='lines+markers',
                        name=f'{wrapper} Success Rate',
                        line=dict(width=2)
                    ))
        else:
            # Single wrapper performance chart
            metrics = self.monitoring.get_wrapper_metrics(wrapper_name, since)
            fig = go.Figure()
            
            if metrics:
                times = [m['timestamp'] for m in metrics]
                success_rates = [m.get('success_rate', 0) * 100 for m in metrics]
                response_times = [m.get('avg_response_time', 0) for m in metrics]
                
                # Success rate trace
                fig.add_trace(go.Scatter(
                    x=times,
                    y=success_rates,
                    mode='lines+markers',
                    name='Success Rate (%)',
                    yaxis='y',
                    line=dict(color='green', width=2)
                ))
                
                # Response time trace (secondary y-axis)
                fig.add_trace(go.Scatter(
                    x=times,
                    y=response_times,
                    mode='lines+markers',
                    name='Avg Response Time (ms)',
                    yaxis='y2',
                    line=dict(color='blue', width=2)
                ))
        
        # Update layout
        fig.update_layout(
            title={'text': f'Performance Metrics - {wrapper_name}'},
            xaxis_title='Time',
            yaxis=dict(
                title='Success Rate (%)',
                side='left',
                range=[0, 100]
            ),
            yaxis2=dict(
                title='Response Time (ms)',
                side='right',
                overlaying='y'
            ),
            legend=dict(x=0, y=1),
            hovermode='x unified'
        )
        
        return fig.to_dict()
    
    def _generate_health_chart(self) -> Dict[str, Any]:
        """
        Generate health status chart data.
        
        Returns:
            Plotly chart configuration dictionary
        """
        wrappers = self.monitoring.get_active_wrappers()
        health_scores = []
        wrapper_names = []
        colors = []
        
        for wrapper in wrappers:
            health_status = self.monitoring.get_wrapper_health(wrapper)
            health_scores.append(health_status.health_score * 100)
            wrapper_names.append(wrapper)
            
            # Color based on health score
            score = health_status.health_score
            if score >= 0.8:
                colors.append('green')
            elif score >= 0.6:
                colors.append('yellow')
            else:
                colors.append('red')
        
        fig = go.Figure([go.Bar(
            x=wrapper_names,
            y=health_scores,
            marker_color=colors,
            text=[f'{score:.1f}%' for score in health_scores],
            textposition='auto'
        )])
        
        fig.update_layout(
            title={'text': 'Wrapper Health Scores'},
            xaxis_title='Wrappers',
            yaxis_title='Health Score (%)',
            yaxis=dict(range=[0, 100]),
            showlegend=False
        )
        
        return fig.to_dict()
    
    def broadcast_metrics_update(self) -> None:
        """Broadcast metrics update to all connected clients."""
        try:
            system_health = self.monitoring.get_system_health()
            
            self.socketio.emit('metrics_broadcast', {
                'system_health': system_health,
                'timestamp': datetime.utcnow().isoformat()
            })
            
            self._last_metrics_update = datetime.utcnow()
            logger.debug("Metrics update broadcasted to all clients")
            
        except Exception as e:
            logger.error(f"Error broadcasting metrics update: {e}")
    
    def run(self, host: str = '0.0.0.0', port: int = 5000, debug: bool = False) -> None:
        """
        Run the monitoring dashboard server.
        
        Args:
            host: Host address to bind to
            port: Port number to bind to
            debug: Enable debug mode
        """
        logger.info(f"Starting monitoring dashboard on {host}:{port}")
        self.socketio.run(self.app, host=host, port=port, debug=debug)


def create_monitoring_dashboard(monitoring: WrapperMonitoring, 
                              performance_monitor: PerformanceMonitor) -> MonitoringDashboard:
    """
    Create and configure monitoring dashboard instance.
    
    Args:
        monitoring: Wrapper monitoring system instance
        performance_monitor: Performance monitoring system instance
        
    Returns:
        Configured MonitoringDashboard instance
    """
    dashboard = MonitoringDashboard(monitoring, performance_monitor)
    return dashboard