"""
Advanced error handler monitoring and analytics system.
Provides comprehensive monitoring, metrics, and analysis for error handling performance.
"""

import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from ..core.error_handling import ErrorContext, ErrorHandler, ErrorHandlerResult
from ..core.error_handler_registry import ErrorHandlerRegistry


logger = logging.getLogger(__name__)


@dataclass
class ErrorPatternAnalysis:
    """Analysis of error patterns over time."""
    error_type: str
    frequency: int
    first_occurrence: datetime
    last_occurrence: datetime
    affected_tasks: List[str]
    common_patterns: List[str]
    success_rate: float
    avg_resolution_time: float


@dataclass
class HandlerPerformanceMetrics:
    """Performance metrics for error handlers."""
    handler_id: str
    total_executions: int
    successful_executions: int
    failed_executions: int
    success_rate: float
    avg_execution_time: float
    min_execution_time: float
    max_execution_time: float
    error_types_handled: List[str]
    last_execution: Optional[datetime]
    trend_direction: str  # "improving", "declining", "stable"


@dataclass
class SystemHealthMetrics:
    """Overall system health metrics for error handling."""
    total_errors: int
    handled_errors: int
    unhandled_errors: int
    overall_handling_rate: float
    avg_recovery_time: float
    most_common_errors: List[Tuple[str, int]]
    problematic_tasks: List[str]
    handler_efficiency: float
    timestamp: datetime


class ErrorHandlerMonitor:
    """Advanced monitoring and analytics for error handling system."""
    
    def __init__(self, registry: ErrorHandlerRegistry, history_limit: int = 10000):
        self.registry = registry
        self.history_limit = history_limit
        
        # Monitoring data structures
        self.execution_history: deque = deque(maxlen=history_limit)
        self.error_timeline: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.handler_metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.task_error_patterns: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Performance tracking
        self.performance_snapshots: List[SystemHealthMetrics] = []
        self.alert_thresholds = {
            "error_rate_threshold": 0.10,  # Alert if >10% of tasks fail
            "handler_failure_threshold": 0.30,  # Alert if handler fails >30% of time
            "avg_recovery_time_threshold": 30.0,  # Alert if recovery takes >30s
            "unhandled_error_threshold": 5  # Alert if >5 unhandled errors per hour
        }
        
        # Metrics collection intervals
        self.last_snapshot_time = datetime.now()
        self.snapshot_interval = timedelta(minutes=5)
    
    def record_error_occurrence(
        self, 
        task_id: str, 
        error: Exception, 
        context: ErrorContext,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Record an error occurrence for monitoring."""
        timestamp = timestamp or datetime.now()
        error_type = type(error).__name__
        
        # Record in timeline
        error_record = {
            "timestamp": timestamp,
            "task_id": task_id,
            "error_type": error_type,
            "error_message": str(error),
            "context": context.to_dict() if context else {}
        }
        
        self.error_timeline[error_type].append(error_record)
        
        # Update task error patterns
        if task_id not in self.task_error_patterns:
            self.task_error_patterns[task_id] = {
                "error_count": 0,
                "error_types": set(),
                "first_error": timestamp,
                "last_error": timestamp
            }
        
        task_pattern = self.task_error_patterns[task_id]
        task_pattern["error_count"] += 1
        task_pattern["error_types"].add(error_type)
        task_pattern["last_error"] = timestamp
        
        logger.debug(f"Recorded error occurrence: {task_id} - {error_type}")
    
    def record_handler_execution(
        self,
        handler_id: str,
        handler_result: ErrorHandlerResult,
        execution_time: float,
        error_context: ErrorContext,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Record handler execution for performance analysis."""
        timestamp = timestamp or datetime.now()
        
        # Record in execution history
        execution_record = {
            "timestamp": timestamp,
            "handler_id": handler_id,
            "success": handler_result.success,
            "execution_time": execution_time,
            "error_type": error_context.error_type,
            "task_id": error_context.failed_task_id,
            "handler_attempts": handler_result.handler_attempts,
            "result": handler_result.to_dict()
        }
        
        self.execution_history.append(execution_record)
        
        # Update handler metrics
        if handler_id not in self.handler_metrics:
            self.handler_metrics[handler_id] = {
                "executions": 0,
                "successes": 0,
                "failures": 0,
                "execution_times": [],
                "error_types": set(),
                "first_execution": timestamp,
                "last_execution": timestamp
            }
        
        metrics = self.handler_metrics[handler_id]
        metrics["executions"] += 1
        metrics["execution_times"].append(execution_time)
        metrics["error_types"].add(error_context.error_type)
        metrics["last_execution"] = timestamp
        
        if handler_result.success:
            metrics["successes"] += 1
        else:
            metrics["failures"] += 1
        
        # Limit execution times history
        if len(metrics["execution_times"]) > 1000:
            metrics["execution_times"] = metrics["execution_times"][-500:]
        
        logger.debug(f"Recorded handler execution: {handler_id} - {'success' if handler_result.success else 'failure'}")
    
    def analyze_error_patterns(self, time_window: timedelta = timedelta(hours=24)) -> List[ErrorPatternAnalysis]:
        """Analyze error patterns over a time window."""
        cutoff_time = datetime.now() - time_window
        patterns = []
        
        for error_type, timeline in self.error_timeline.items():
            # Filter to time window
            recent_errors = [
                record for record in timeline 
                if record["timestamp"] >= cutoff_time
            ]
            
            if not recent_errors:
                continue
            
            # Analyze patterns
            affected_tasks = list(set(record["task_id"] for record in recent_errors))
            error_messages = [record["error_message"] for record in recent_errors]
            
            # Find common patterns in error messages
            common_patterns = self._find_common_patterns(error_messages)
            
            # Calculate success rate (approximate - based on handler registry data)
            total_occurrences = len(recent_errors)
            handled_count = sum(
                1 for record in recent_errors 
                if record.get("handled", False)
            )
            success_rate = handled_count / total_occurrences if total_occurrences > 0 else 0.0
            
            # Calculate average resolution time
            resolution_times = []
            for record in recent_errors:
                if "resolution_time" in record:
                    resolution_times.append(record["resolution_time"])
            
            avg_resolution_time = (
                sum(resolution_times) / len(resolution_times) 
                if resolution_times else 0.0
            )
            
            pattern = ErrorPatternAnalysis(
                error_type=error_type,
                frequency=len(recent_errors),
                first_occurrence=min(record["timestamp"] for record in recent_errors),
                last_occurrence=max(record["timestamp"] for record in recent_errors),
                affected_tasks=affected_tasks,
                common_patterns=common_patterns,
                success_rate=success_rate,
                avg_resolution_time=avg_resolution_time
            )
            
            patterns.append(pattern)
        
        # Sort by frequency (most common first)
        patterns.sort(key=lambda p: p.frequency, reverse=True)
        return patterns
    
    def get_handler_performance_metrics(self, handler_id: Optional[str] = None) -> List[HandlerPerformanceMetrics]:
        """Get performance metrics for handlers."""
        metrics_list = []
        
        handler_ids = [handler_id] if handler_id else self.handler_metrics.keys()
        
        for h_id in handler_ids:
            if h_id not in self.handler_metrics:
                continue
            
            metrics = self.handler_metrics[h_id]
            execution_times = metrics["execution_times"]
            
            # Calculate performance metrics
            success_rate = (
                metrics["successes"] / metrics["executions"] 
                if metrics["executions"] > 0 else 0.0
            )
            
            avg_execution_time = (
                sum(execution_times) / len(execution_times) 
                if execution_times else 0.0
            )
            
            min_execution_time = min(execution_times) if execution_times else 0.0
            max_execution_time = max(execution_times) if execution_times else 0.0
            
            # Determine trend direction
            trend_direction = self._calculate_trend_direction(h_id)
            
            performance = HandlerPerformanceMetrics(
                handler_id=h_id,
                total_executions=metrics["executions"],
                successful_executions=metrics["successes"],
                failed_executions=metrics["failures"],
                success_rate=success_rate,
                avg_execution_time=avg_execution_time,
                min_execution_time=min_execution_time,
                max_execution_time=max_execution_time,
                error_types_handled=list(metrics["error_types"]),
                last_execution=metrics["last_execution"],
                trend_direction=trend_direction
            )
            
            metrics_list.append(performance)
        
        # Sort by success rate (best performers first)
        metrics_list.sort(key=lambda m: m.success_rate, reverse=True)
        return metrics_list
    
    def generate_system_health_snapshot(self) -> SystemHealthMetrics:
        """Generate current system health metrics."""
        now = datetime.now()
        
        # Calculate overall statistics
        total_errors = sum(
            len(timeline) for timeline in self.error_timeline.values()
        )
        
        # Get handled vs unhandled errors from registry
        registry_stats = self.registry.get_error_statistics()
        handled_errors = registry_stats.get("handled_errors", 0)
        unhandled_errors = registry_stats.get("unhandled_errors", 0)
        
        overall_handling_rate = (
            handled_errors / (handled_errors + unhandled_errors)
            if (handled_errors + unhandled_errors) > 0 else 0.0
        )
        
        # Calculate average recovery time from execution history
        recent_executions = [
            record for record in self.execution_history
            if record["timestamp"] >= now - timedelta(hours=1)
        ]
        
        recovery_times = [
            record["execution_time"] for record in recent_executions
            if record["success"]
        ]
        
        avg_recovery_time = (
            sum(recovery_times) / len(recovery_times)
            if recovery_times else 0.0
        )
        
        # Get most common errors
        error_frequencies = {}
        for error_type, timeline in self.error_timeline.items():
            # Count errors in last 24 hours
            recent_count = sum(
                1 for record in timeline
                if record["timestamp"] >= now - timedelta(hours=24)
            )
            if recent_count > 0:
                error_frequencies[error_type] = recent_count
        
        most_common_errors = sorted(
            error_frequencies.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        # Identify problematic tasks (high error rates)
        problematic_tasks = []
        for task_id, pattern in self.task_error_patterns.items():
            if pattern["error_count"] >= 5:  # Tasks with 5+ errors
                problematic_tasks.append(task_id)
        
        # Calculate handler efficiency (weighted by execution count)
        total_handler_executions = sum(
            metrics["executions"] for metrics in self.handler_metrics.values()
        )
        
        if total_handler_executions > 0:
            weighted_success_rate = sum(
                (metrics["successes"] / metrics["executions"]) * metrics["executions"]
                for metrics in self.handler_metrics.values()
                if metrics["executions"] > 0
            ) / total_handler_executions
        else:
            weighted_success_rate = 0.0
        
        snapshot = SystemHealthMetrics(
            total_errors=total_errors,
            handled_errors=handled_errors,
            unhandled_errors=unhandled_errors,
            overall_handling_rate=overall_handling_rate,
            avg_recovery_time=avg_recovery_time,
            most_common_errors=most_common_errors,
            problematic_tasks=problematic_tasks,
            handler_efficiency=weighted_success_rate,
            timestamp=now
        )
        
        # Store snapshot
        self.performance_snapshots.append(snapshot)
        
        # Limit snapshot history
        if len(self.performance_snapshots) > 1000:
            self.performance_snapshots = self.performance_snapshots[-500:]
        
        self.last_snapshot_time = now
        return snapshot
    
    def check_alerts(self, current_metrics: Optional[SystemHealthMetrics] = None) -> List[Dict[str, Any]]:
        """Check for alert conditions and return active alerts."""
        if not current_metrics:
            current_metrics = self.generate_system_health_snapshot()
        
        alerts = []
        
        # Check error rate threshold
        if current_metrics.overall_handling_rate < (1.0 - self.alert_thresholds["error_rate_threshold"]):
            alerts.append({
                "type": "high_error_rate",
                "severity": "warning",
                "message": f"Error handling rate is {current_metrics.overall_handling_rate:.2%}, below threshold",
                "threshold": self.alert_thresholds["error_rate_threshold"],
                "current_value": 1.0 - current_metrics.overall_handling_rate
            })
        
        # Check handler failure rates
        for metrics in self.get_handler_performance_metrics():
            if metrics.success_rate < (1.0 - self.alert_thresholds["handler_failure_threshold"]):
                alerts.append({
                    "type": "handler_high_failure_rate",
                    "severity": "warning",
                    "message": f"Handler '{metrics.handler_id}' has {metrics.success_rate:.2%} success rate",
                    "handler_id": metrics.handler_id,
                    "threshold": self.alert_thresholds["handler_failure_threshold"],
                    "current_value": 1.0 - metrics.success_rate
                })
        
        # Check recovery time threshold
        if current_metrics.avg_recovery_time > self.alert_thresholds["avg_recovery_time_threshold"]:
            alerts.append({
                "type": "slow_recovery_time",
                "severity": "warning",
                "message": f"Average recovery time is {current_metrics.avg_recovery_time:.1f}s",
                "threshold": self.alert_thresholds["avg_recovery_time_threshold"],
                "current_value": current_metrics.avg_recovery_time
            })
        
        # Check unhandled error threshold
        recent_unhandled = self._count_recent_unhandled_errors(timedelta(hours=1))
        if recent_unhandled > self.alert_thresholds["unhandled_error_threshold"]:
            alerts.append({
                "type": "high_unhandled_errors",
                "severity": "critical",
                "message": f"{recent_unhandled} unhandled errors in the last hour",
                "threshold": self.alert_thresholds["unhandled_error_threshold"],
                "current_value": recent_unhandled
            })
        
        return alerts
    
    def export_metrics(self, format: str = "json") -> str:
        """Export all metrics in specified format."""
        data = {
            "system_health": asdict(self.generate_system_health_snapshot()),
            "error_patterns": [asdict(pattern) for pattern in self.analyze_error_patterns()],
            "handler_performance": [asdict(metrics) for metrics in self.get_handler_performance_metrics()],
            "alerts": self.check_alerts(),
            "export_timestamp": datetime.now().isoformat()
        }
        
        if format.lower() == "json":
            return json.dumps(data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        current_health = self.generate_system_health_snapshot()
        error_patterns = self.analyze_error_patterns(timedelta(hours=24))
        handler_metrics = self.get_handler_performance_metrics()
        alerts = self.check_alerts(current_health)
        
        # Historical trends
        recent_snapshots = [
            snapshot for snapshot in self.performance_snapshots
            if snapshot.timestamp >= datetime.now() - timedelta(hours=24)
        ]
        
        return {
            "current_health": asdict(current_health),
            "error_patterns": [asdict(pattern) for pattern in error_patterns[:10]],
            "top_handlers": [asdict(metrics) for metrics in handler_metrics[:10]],
            "active_alerts": alerts,
            "trends": {
                "error_rate_trend": [
                    {
                        "timestamp": snapshot.timestamp.isoformat(),
                        "handling_rate": snapshot.overall_handling_rate,
                        "total_errors": snapshot.total_errors
                    }
                    for snapshot in recent_snapshots
                ],
                "recovery_time_trend": [
                    {
                        "timestamp": snapshot.timestamp.isoformat(),
                        "avg_recovery_time": snapshot.avg_recovery_time
                    }
                    for snapshot in recent_snapshots
                ]
            },
            "summary": {
                "total_handlers": len(self.handler_metrics),
                "monitoring_since": min(
                    metrics["first_execution"] 
                    for metrics in self.handler_metrics.values()
                ) if self.handler_metrics else datetime.now(),
                "data_points": len(self.execution_history)
            }
        }
    
    def _find_common_patterns(self, error_messages: List[str]) -> List[str]:
        """Find common patterns in error messages."""
        if not error_messages:
            return []
        
        # Simple pattern detection - look for common substrings
        patterns = []
        
        # Check for common prefixes
        if len(error_messages) > 1:
            common_words = set()
            for message in error_messages:
                words = message.lower().split()
                if len(words) > 0:
                    common_words.update(words[:3])  # First 3 words
            
            # Find words that appear in multiple messages
            for word in common_words:
                count = sum(1 for msg in error_messages if word in msg.lower())
                if count > len(error_messages) * 0.5:  # Appears in >50% of messages
                    patterns.append(f"Common word: '{word}'")
        
        # Check for HTTP status codes
        http_codes = []
        for message in error_messages:
            import re
            codes = re.findall(r'HTTP\s+(\d{3})', message)
            http_codes.extend(codes)
        
        if http_codes:
            most_common_code = max(set(http_codes), key=http_codes.count)
            patterns.append(f"HTTP {most_common_code} errors")
        
        return patterns[:5]  # Limit to top 5 patterns
    
    def _calculate_trend_direction(self, handler_id: str) -> str:
        """Calculate trend direction for handler performance."""
        # Get recent executions for this handler
        recent_executions = [
            record for record in self.execution_history
            if record["handler_id"] == handler_id and
            record["timestamp"] >= datetime.now() - timedelta(hours=24)
        ]
        
        if len(recent_executions) < 10:
            return "stable"  # Not enough data
        
        # Split into two halves and compare success rates
        mid_point = len(recent_executions) // 2
        first_half = recent_executions[:mid_point]
        second_half = recent_executions[mid_point:]
        
        first_success_rate = sum(1 for r in first_half if r["success"]) / len(first_half)
        second_success_rate = sum(1 for r in second_half if r["success"]) / len(second_half)
        
        if second_success_rate > first_success_rate + 0.1:
            return "improving"
        elif second_success_rate < first_success_rate - 0.1:
            return "declining"
        else:
            return "stable"
    
    def _count_recent_unhandled_errors(self, time_window: timedelta) -> int:
        """Count unhandled errors in the time window."""
        cutoff_time = datetime.now() - time_window
        count = 0
        
        for timeline in self.error_timeline.values():
            for record in timeline:
                if (record["timestamp"] >= cutoff_time and 
                    not record.get("handled", False)):
                    count += 1
        
        return count


class ErrorHandlerDashboard:
    """Web dashboard for error handler monitoring."""
    
    def __init__(self, monitor: ErrorHandlerMonitor):
        self.monitor = monitor
    
    def generate_html_dashboard(self) -> str:
        """Generate HTML dashboard for error handler monitoring."""
        data = self.monitor.get_monitoring_dashboard_data()
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Error Handler Monitoring Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric-card {{ border: 1px solid #ccc; padding: 15px; margin: 10px; border-radius: 5px; }}
                .alert {{ background-color: #ffebee; border-left: 5px solid #f44336; padding: 10px; margin: 10px 0; }}
                .success {{ color: #4caf50; }}
                .warning {{ color: #ff9800; }}
                .error {{ color: #f44336; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Error Handler Monitoring Dashboard</h1>
            <p>Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="metric-card">
                <h2>System Health Overview</h2>
                <p><strong>Total Errors:</strong> {data['current_health']['total_errors']}</p>
                <p><strong>Handling Rate:</strong> <span class="{'success' if data['current_health']['overall_handling_rate'] > 0.9 else 'warning'}">{data['current_health']['overall_handling_rate']:.2%}</span></p>
                <p><strong>Average Recovery Time:</strong> {data['current_health']['avg_recovery_time']:.2f}s</p>
                <p><strong>Handler Efficiency:</strong> {data['current_health']['handler_efficiency']:.2%}</p>
            </div>
            
            <div class="metric-card">
                <h2>Active Alerts</h2>
                {self._format_alerts_html(data['active_alerts'])}
            </div>
            
            <div class="metric-card">
                <h2>Top Error Patterns (24h)</h2>
                {self._format_error_patterns_html(data['error_patterns'])}
            </div>
            
            <div class="metric-card">
                <h2>Handler Performance</h2>
                {self._format_handler_metrics_html(data['top_handlers'])}
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _format_alerts_html(self, alerts: List[Dict[str, Any]]) -> str:
        """Format alerts as HTML."""
        if not alerts:
            return "<p class='success'>No active alerts</p>"
        
        html = ""
        for alert in alerts:
            severity_class = "error" if alert["severity"] == "critical" else "warning"
            html += f"<div class='alert {severity_class}'><strong>{alert['type']}:</strong> {alert['message']}</div>"
        
        return html
    
    def _format_error_patterns_html(self, patterns: List[Dict[str, Any]]) -> str:
        """Format error patterns as HTML table."""
        if not patterns:
            return "<p>No error patterns detected</p>"
        
        html = "<table><tr><th>Error Type</th><th>Frequency</th><th>Success Rate</th><th>Avg Resolution Time</th></tr>"
        
        for pattern in patterns[:10]:
            html += f"""
            <tr>
                <td>{pattern['error_type']}</td>
                <td>{pattern['frequency']}</td>
                <td>{pattern['success_rate']:.2%}</td>
                <td>{pattern['avg_resolution_time']:.2f}s</td>
            </tr>
            """
        
        html += "</table>"
        return html
    
    def _format_handler_metrics_html(self, handlers: List[Dict[str, Any]]) -> str:
        """Format handler metrics as HTML table."""
        if not handlers:
            return "<p>No handler metrics available</p>"
        
        html = "<table><tr><th>Handler ID</th><th>Success Rate</th><th>Executions</th><th>Avg Time</th><th>Trend</th></tr>"
        
        for handler in handlers[:10]:
            trend_class = {
                "improving": "success",
                "declining": "error",
                "stable": ""
            }.get(handler['trend_direction'], "")
            
            html += f"""
            <tr>
                <td>{handler['handler_id']}</td>
                <td class="{'success' if handler['success_rate'] > 0.9 else 'warning' if handler['success_rate'] > 0.7 else 'error'}">{handler['success_rate']:.2%}</td>
                <td>{handler['total_executions']}</td>
                <td>{handler['avg_execution_time']:.3f}s</td>
                <td class="{trend_class}">{handler['trend_direction']}</td>
            </tr>
            """
        
        html += "</table>"
        return html