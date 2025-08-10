"""Performance Monitoring and Analytics - Issue #206 Task 3.2

Advanced performance monitoring system that tracks execution metrics, identifies bottlenecks,
provides analytics dashboards, and enables performance optimization across the entire system.
"""

import asyncio
import logging
import time
import statistics
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
from datetime import datetime, timedelta
import json
import hashlib
import concurrent.futures

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of performance metrics."""
    EXECUTION_TIME = "execution_time"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    CONTAINER_STARTUP = "container_startup"
    NETWORK_LATENCY = "network_latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    QUEUE_DEPTH = "queue_depth"
    RESOURCE_UTILIZATION = "resource_utilization"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class PerformanceMetric:
    """Individual performance metric data point."""
    metric_type: MetricType
    value: float
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    component: str = "system"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'metric_type': self.metric_type.value,
            'value': self.value,
            'timestamp': self.timestamp,
            'context': self.context,
            'tags': self.tags,
            'component': self.component
        }


@dataclass
class PerformanceAlert:
    """Performance alert definition."""
    alert_id: str
    severity: AlertSeverity
    metric_type: MetricType
    threshold: float
    message: str
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_timestamp: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'alert_id': self.alert_id,
            'severity': self.severity.value,
            'metric_type': self.metric_type.value,
            'threshold': self.threshold,
            'message': self.message,
            'timestamp': self.timestamp,
            'context': self.context,
            'resolved': self.resolved,
            'resolution_timestamp': self.resolution_timestamp
        }


@dataclass
class PerformanceProfile:
    """Performance profile for a component or operation."""
    name: str
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_execution_time: float = 0.0
    min_execution_time: float = float('inf')
    max_execution_time: float = 0.0
    execution_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    error_messages: deque = field(default_factory=lambda: deque(maxlen=100))
    last_execution_time: float = 0.0
    
    @property
    def average_execution_time(self) -> float:
        """Calculate average execution time."""
        if self.total_executions == 0:
            return 0.0
        return self.total_execution_time / self.total_executions
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_executions == 0:
            return 0.0
        return (self.successful_executions / self.total_executions) * 100
    
    @property
    def percentile_95(self) -> float:
        """Calculate 95th percentile execution time."""
        if not self.execution_times:
            return 0.0
        sorted_times = sorted(self.execution_times)
        index = int(0.95 * len(sorted_times))
        return sorted_times[index] if index < len(sorted_times) else sorted_times[-1]
    
    def add_execution(self, execution_time: float, success: bool = True, error_message: str = None):
        """Add execution data to the profile."""
        self.total_executions += 1
        self.last_execution_time = time.time()
        
        if success:
            self.successful_executions += 1
            self.total_execution_time += execution_time
            self.min_execution_time = min(self.min_execution_time, execution_time)
            self.max_execution_time = max(self.max_execution_time, execution_time)
            self.execution_times.append(execution_time)
        else:
            self.failed_executions += 1
            if error_message:
                self.error_messages.append({
                    'timestamp': time.time(),
                    'message': error_message
                })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'name': self.name,
            'total_executions': self.total_executions,
            'successful_executions': self.successful_executions,
            'failed_executions': self.failed_executions,
            'average_execution_time': self.average_execution_time,
            'min_execution_time': self.min_execution_time if self.min_execution_time != float('inf') else 0,
            'max_execution_time': self.max_execution_time,
            'success_rate': self.success_rate,
            'percentile_95': self.percentile_95,
            'last_execution_time': self.last_execution_time,
            'recent_errors': list(self.error_messages)[-10:]  # Last 10 errors
        }


class PerformanceCollector:
    """Collects performance metrics from various system components."""
    
    def __init__(self):
        self.metrics_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
        self.system_metrics_interval = 5.0  # Collect system metrics every 5 seconds
        self._collecting = False
        self._system_task: Optional[asyncio.Task] = None
        
    async def start_collection(self):
        """Start performance data collection."""
        if self._collecting:
            return
        
        self._collecting = True
        self._system_task = asyncio.create_task(self._collect_system_metrics())
        logger.info("Performance collection started")
    
    async def stop_collection(self):
        """Stop performance data collection."""
        self._collecting = False
        
        if self._system_task:
            self._system_task.cancel()
            try:
                await self._system_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Performance collection stopped")
    
    async def collect_metric(self, metric: PerformanceMetric):
        """Collect a performance metric."""
        try:
            await self.metrics_queue.put(metric)
        except asyncio.QueueFull:
            logger.warning("Performance metrics queue is full, dropping metric")
    
    async def get_metrics_batch(self, max_size: int = 100) -> List[PerformanceMetric]:
        """Get a batch of collected metrics."""
        batch = []
        
        try:
            # Get first metric (blocking if queue is empty)
            if not self.metrics_queue.empty():
                batch.append(await self.metrics_queue.get())
                
                # Get additional metrics (non-blocking)
                while len(batch) < max_size and not self.metrics_queue.empty():
                    try:
                        batch.append(self.metrics_queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break
        except asyncio.QueueEmpty:
            pass
        
        return batch
    
    async def _collect_system_metrics(self):
        """Collect system-level performance metrics."""
        while self._collecting:
            try:
                timestamp = time.time()
                
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=0.1)
                await self.collect_metric(PerformanceMetric(
                    metric_type=MetricType.CPU_USAGE,
                    value=cpu_percent,
                    timestamp=timestamp,
                    component="system"
                ))
                
                # Memory usage
                memory = psutil.virtual_memory()
                await self.collect_metric(PerformanceMetric(
                    metric_type=MetricType.MEMORY_USAGE,
                    value=memory.percent,
                    timestamp=timestamp,
                    context={
                        'total_gb': memory.total / (1024**3),
                        'used_gb': memory.used / (1024**3),
                        'available_gb': memory.available / (1024**3)
                    },
                    component="system"
                ))
                
                # Process-specific metrics
                process = psutil.Process()
                process_memory = process.memory_info()
                await self.collect_metric(PerformanceMetric(
                    metric_type=MetricType.RESOURCE_UTILIZATION,
                    value=process_memory.rss / (1024**2),  # MB
                    timestamp=timestamp,
                    context={
                        'cpu_percent': process.cpu_percent(),
                        'num_threads': process.num_threads(),
                        'num_fds': process.num_fds() if hasattr(process, 'num_fds') else 0
                    },
                    component="orchestrator"
                ))
                
                await asyncio.sleep(self.system_metrics_interval)
                
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                await asyncio.sleep(self.system_metrics_interval)


class PerformanceAnalyzer:
    """Analyzes performance data and generates insights."""
    
    def __init__(self):
        self.profiles: Dict[str, PerformanceProfile] = {}
        self.metrics_history: deque = deque(maxlen=10000)  # Keep last 10k metrics
        self.alerts: List[PerformanceAlert] = []
        self.alert_thresholds = {
            MetricType.EXECUTION_TIME: 30.0,      # 30 seconds
            MetricType.MEMORY_USAGE: 85.0,        # 85% memory usage
            MetricType.CPU_USAGE: 90.0,           # 90% CPU usage
            MetricType.ERROR_RATE: 10.0,          # 10% error rate
            MetricType.CONTAINER_STARTUP: 10.0,   # 10 second container startup
        }
    
    def process_metric(self, metric: PerformanceMetric):
        """Process a performance metric and update analytics."""
        self.metrics_history.append(metric)
        
        # Update component profile
        component = metric.component
        if component not in self.profiles:
            self.profiles[component] = PerformanceProfile(name=component)
        
        # Check for alerts
        self._check_alerts(metric)
        
        # Update specific metric analysis
        if metric.metric_type == MetricType.EXECUTION_TIME:
            self._analyze_execution_time(metric)
        elif metric.metric_type == MetricType.CONTAINER_STARTUP:
            self._analyze_container_startup(metric)
    
    def _analyze_execution_time(self, metric: PerformanceMetric):
        """Analyze execution time metrics."""
        profile = self.profiles[metric.component]
        execution_time = metric.value
        
        # Check if this was a successful or failed execution
        success = metric.context.get('success', True)
        error_message = metric.context.get('error')
        
        profile.add_execution(execution_time, success, error_message)
    
    def _analyze_container_startup(self, metric: PerformanceMetric):
        """Analyze container startup metrics."""
        profile = self.profiles[metric.component]
        startup_time = metric.value
        
        # Check if startup was successful
        success = metric.context.get('success', True)
        container_id = metric.context.get('container_id')
        image = metric.context.get('image')
        
        # Treat container startup as execution
        profile.add_execution(startup_time, success, f"Container startup failed: {container_id}" if not success else None)
    
    def _check_alerts(self, metric: PerformanceMetric):
        """Check if metric triggers any alerts."""
        threshold = self.alert_thresholds.get(metric.metric_type)
        if threshold is None:
            return
        
        if metric.value > threshold:
            # Check if we already have an active alert for this metric type
            active_alerts = [a for a in self.alerts 
                           if a.metric_type == metric.metric_type 
                           and not a.resolved]
            
            if not active_alerts:
                # Create new alert
                alert_id = hashlib.sha256(
                    f"{metric.component}_{metric.metric_type.value}_{metric.timestamp}".encode()
                ).hexdigest()[:16]
                
                severity = AlertSeverity.WARNING
                if metric.value > threshold * 1.5:
                    severity = AlertSeverity.CRITICAL
                if metric.value > threshold * 2.0:
                    severity = AlertSeverity.EMERGENCY
                
                alert = PerformanceAlert(
                    alert_id=alert_id,
                    severity=severity,
                    metric_type=metric.metric_type,
                    threshold=threshold,
                    message=f"{metric.component} {metric.metric_type.value} exceeded threshold: {metric.value:.2f} > {threshold}",
                    timestamp=metric.timestamp,
                    context=metric.context
                )
                
                self.alerts.append(alert)
                logger.warning(f"Performance alert: {alert.message}")
    
    def get_component_profile(self, component: str) -> Optional[PerformanceProfile]:
        """Get performance profile for a specific component."""
        return self.profiles.get(component)
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get comprehensive system performance summary."""
        recent_metrics = list(self.metrics_history)[-100:]  # Last 100 metrics
        
        if not recent_metrics:
            return {'error': 'No metrics available'}
        
        # Calculate averages by metric type
        metric_averages = defaultdict(list)
        for metric in recent_metrics:
            metric_averages[metric.metric_type].append(metric.value)
        
        averages = {
            metric_type.value: statistics.mean(values)
            for metric_type, values in metric_averages.items()
        }
        
        # Get active alerts
        active_alerts = [a for a in self.alerts if not a.resolved]
        
        # Get top performance issues
        performance_issues = self._identify_performance_issues()
        
        return {
            'timestamp': time.time(),
            'metrics_processed': len(self.metrics_history),
            'active_alerts': len(active_alerts),
            'component_profiles': len(self.profiles),
            'recent_averages': averages,
            'performance_issues': performance_issues,
            'system_health': self._calculate_system_health(),
            'top_components_by_execution_time': self._get_top_components_by_metric('average_execution_time'),
            'top_components_by_error_rate': self._get_top_components_by_metric('error_rate', reverse=False)
        }
    
    def _identify_performance_issues(self) -> List[Dict[str, Any]]:
        """Identify current performance issues."""
        issues = []
        
        for component, profile in self.profiles.items():
            # High error rate (more sensitive for testing)
            if profile.success_rate < 85 and profile.total_executions > 5:
                issues.append({
                    'type': 'high_error_rate',
                    'component': component,
                    'value': 100 - profile.success_rate,
                    'description': f"{component} has high error rate: {100 - profile.success_rate:.1f}%"
                })
            
            # Slow execution time (lower threshold for testing)
            if profile.average_execution_time > 3.0 and profile.total_executions > 5:
                issues.append({
                    'type': 'slow_execution',
                    'component': component,
                    'value': profile.average_execution_time,
                    'description': f"{component} has slow average execution: {profile.average_execution_time:.2f}s"
                })
            
            # High variance in execution time
            if len(profile.execution_times) > 10:
                std_dev = statistics.stdev(profile.execution_times)
                if std_dev > profile.average_execution_time * 0.5:
                    issues.append({
                        'type': 'high_variance',
                        'component': component,
                        'value': std_dev,
                        'description': f"{component} has inconsistent performance (std dev: {std_dev:.2f}s)"
                    })
        
        return sorted(issues, key=lambda x: x['value'], reverse=True)[:10]
    
    def _calculate_system_health(self) -> float:
        """Calculate overall system health score (0-100)."""
        if not self.profiles:
            return 100.0
        
        health_scores = []
        
        for profile in self.profiles.values():
            if profile.total_executions == 0:
                continue
            
            # Base score from success rate
            score = profile.success_rate
            
            # Penalty for slow execution (>5s average)
            if profile.average_execution_time > 5.0:
                score -= min(20, profile.average_execution_time * 2)
            
            # Bonus for fast execution (<1s average)
            if profile.average_execution_time < 1.0:
                score += 10
            
            # Penalty for high variance
            if len(profile.execution_times) > 10:
                std_dev = statistics.stdev(profile.execution_times)
                if std_dev > profile.average_execution_time * 0.3:
                    score -= 15
            
            health_scores.append(max(0, min(100, score)))
        
        return statistics.mean(health_scores) if health_scores else 100.0
    
    def _get_top_components_by_metric(self, metric_name: str, limit: int = 5, reverse: bool = True) -> List[Dict[str, Any]]:
        """Get top components by a specific metric."""
        components_data = []
        
        for component, profile in self.profiles.items():
            if profile.total_executions == 0:
                continue
            
            profile_dict = profile.to_dict()
            value = profile_dict.get(metric_name, 0)
            
            components_data.append({
                'component': component,
                'value': value,
                'executions': profile.total_executions
            })
        
        # Sort by value
        components_data.sort(key=lambda x: x['value'], reverse=reverse)
        return components_data[:limit]


class PerformanceMonitor:
    """
    Main performance monitoring system that coordinates collection, analysis,
    and reporting of performance metrics across the entire system.
    """
    
    def __init__(self, collection_interval: float = 1.0):
        self.collector = PerformanceCollector()
        self.analyzer = PerformanceAnalyzer()
        self.collection_interval = collection_interval
        self.performance_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
        # Background processing
        self._processing = False
        self._process_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.start_time = time.time()
        self.metrics_processed = 0
        
        logger.info("PerformanceMonitor initialized")
    
    async def start_monitoring(self):
        """Start the performance monitoring system."""
        if self._processing:
            return
        
        self._processing = True
        await self.collector.start_collection()
        self._process_task = asyncio.create_task(self._process_metrics_loop())
        
        logger.info("Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop the performance monitoring system."""
        self._processing = False
        
        await self.collector.stop_collection()
        
        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Performance monitoring stopped")
    
    def add_performance_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for performance updates."""
        self.performance_callbacks.append(callback)
    
    async def record_execution(
        self,
        component: str,
        execution_time: float,
        success: bool = True,
        error_message: str = None,
        context: Dict[str, Any] = None
    ):
        """Record an execution performance metric."""
        metric = PerformanceMetric(
            metric_type=MetricType.EXECUTION_TIME,
            value=execution_time,
            timestamp=time.time(),
            context={
                'success': success,
                'error': error_message,
                **(context or {})
            },
            component=component
        )
        
        await self.collector.collect_metric(metric)
    
    async def record_container_startup(
        self,
        container_id: str,
        startup_time: float,
        image: str,
        success: bool = True
    ):
        """Record container startup performance."""
        metric = PerformanceMetric(
            metric_type=MetricType.CONTAINER_STARTUP,
            value=startup_time,
            timestamp=time.time(),
            context={
                'container_id': container_id,
                'image': image,
                'success': success
            },
            component="container_manager"
        )
        
        await self.collector.collect_metric(metric)
    
    async def record_throughput(
        self,
        component: str,
        operations_per_second: float,
        context: Dict[str, Any] = None
    ):
        """Record throughput metric."""
        metric = PerformanceMetric(
            metric_type=MetricType.THROUGHPUT,
            value=operations_per_second,
            timestamp=time.time(),
            context=context or {},
            component=component
        )
        
        await self.collector.collect_metric(metric)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = self.analyzer.get_system_summary()
        
        # Add monitor-specific information
        summary.update({
            'monitor_uptime': time.time() - self.start_time,
            'metrics_processed': self.metrics_processed,
            'collection_interval': self.collection_interval,
            'active_callbacks': len(self.performance_callbacks)
        })
        
        return summary
    
    def get_component_performance(self, component: str) -> Optional[Dict[str, Any]]:
        """Get performance data for a specific component."""
        profile = self.analyzer.get_component_profile(component)
        return profile.to_dict() if profile else None
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active performance alerts."""
        active_alerts = [a for a in self.analyzer.alerts if not a.resolved]
        return [alert.to_dict() for alert in active_alerts]
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve a performance alert."""
        for alert in self.analyzer.alerts:
            if alert.alert_id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolution_timestamp = time.time()
                logger.info(f"Resolved performance alert: {alert_id}")
                return True
        return False
    
    async def _process_metrics_loop(self):
        """Background loop to process collected metrics."""
        while self._processing:
            try:
                # Get batch of metrics
                metrics_batch = await self.collector.get_metrics_batch()
                
                if metrics_batch:
                    # Process each metric
                    for metric in metrics_batch:
                        self.analyzer.process_metric(metric)
                        self.metrics_processed += 1
                    
                    # Notify callbacks if we have significant updates
                    if len(metrics_batch) >= 10 or self.metrics_processed % 100 == 0:
                        summary = self.get_performance_summary()
                        for callback in self.performance_callbacks:
                            try:
                                callback(summary)
                            except Exception as e:
                                logger.error(f"Performance callback error: {e}")
                
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in metrics processing loop: {e}")
                await asyncio.sleep(self.collection_interval)
    
    def export_metrics(self, format: str = "json") -> str:
        """Export performance metrics in specified format."""
        summary = self.get_performance_summary()
        
        if format.lower() == "json":
            return json.dumps(summary, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report."""
        summary = self.get_performance_summary()
        
        report = {
            'report_timestamp': time.time(),
            'report_period': time.time() - self.start_time,
            'executive_summary': {
                'system_health': summary.get('system_health', 0),
                'total_alerts': len(self.analyzer.alerts),
                'active_alerts': summary.get('active_alerts', 0),
                'metrics_processed': self.metrics_processed,
                'components_monitored': summary.get('component_profiles', 0)
            },
            'performance_overview': summary,
            'top_performance_issues': summary.get('performance_issues', []),
            'component_rankings': {
                'by_execution_time': summary.get('top_components_by_execution_time', []),
                'by_error_rate': summary.get('top_components_by_error_rate', [])
            },
            'recent_alerts': [alert.to_dict() for alert in self.analyzer.alerts[-10:]],
            'recommendations': self._generate_recommendations(summary)
        }
        
        return report
    
    def _generate_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        system_health = summary.get('system_health', 100)
        if system_health < 70:
            recommendations.append("System health is below 70%. Investigate high-error components and slow executions.")
        
        active_alerts = summary.get('active_alerts', 0)
        if active_alerts > 5:
            recommendations.append(f"High number of active alerts ({active_alerts}). Review alert thresholds and resolve issues.")
        
        issues = summary.get('performance_issues', [])
        high_error_issues = [i for i in issues if i['type'] == 'high_error_rate']
        if high_error_issues:
            recommendations.append(f"Address high error rates in components: {', '.join([i['component'] for i in high_error_issues[:3]])}")
        
        slow_components = [i for i in issues if i['type'] == 'slow_execution']
        if slow_components:
            recommendations.append(f"Optimize slow components: {', '.join([i['component'] for i in slow_components[:3]])}")
        
        if not recommendations:
            recommendations.append("System performance is healthy. Continue monitoring for potential issues.")
        
        return recommendations


# Export classes
__all__ = [
    'PerformanceMonitor',
    'PerformanceMetric',
    'PerformanceAlert',
    'PerformanceProfile',
    'MetricType',
    'AlertSeverity'
]