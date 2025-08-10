"""Resource Monitoring and Enforcement - Issue #206 Task 1.4

Real-time resource monitoring, enforcement, and analytics for secure container execution
with proactive threat detection and performance optimization.
"""

import asyncio
import psutil
import docker
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import statistics
from collections import deque, defaultdict

from .docker_manager import SecureContainer, ResourceLimits

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of system resources."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    PROCESSES = "processes"
    FILE_HANDLES = "file_handles"


class ViolationType(Enum):
    """Types of resource violations."""
    LIMIT_EXCEEDED = "limit_exceeded"
    RAPID_INCREASE = "rapid_increase"
    SUSTAINED_HIGH_USAGE = "sustained_high_usage"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class ResourceUsage:
    """Snapshot of resource usage at a point in time."""
    timestamp: float
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    memory_percent: float = 0.0
    disk_read_mb: float = 0.0
    disk_write_mb: float = 0.0
    network_rx_mb: float = 0.0
    network_tx_mb: float = 0.0
    process_count: int = 0
    file_handles: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'cpu_percent': self.cpu_percent,
            'memory_mb': self.memory_mb,
            'memory_percent': self.memory_percent,
            'disk_read_mb': self.disk_read_mb,
            'disk_write_mb': self.disk_write_mb,
            'network_rx_mb': self.network_rx_mb,
            'network_tx_mb': self.network_tx_mb,
            'process_count': self.process_count,
            'file_handles': self.file_handles
        }


@dataclass
class ResourceViolation:
    """Represents a resource usage violation."""
    container_id: str
    violation_type: ViolationType
    resource_type: ResourceType
    severity: AlertSeverity
    description: str
    current_value: float
    limit_value: float
    timestamp: float
    duration_seconds: float = 0.0
    remediation_action: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'container_id': self.container_id,
            'violation_type': self.violation_type.value,
            'resource_type': self.resource_type.value,
            'severity': self.severity.value,
            'description': self.description,
            'current_value': self.current_value,
            'limit_value': self.limit_value,
            'timestamp': self.timestamp,
            'duration_seconds': self.duration_seconds,
            'remediation_action': self.remediation_action
        }


@dataclass
class ResourceAlert:
    """Alert generated from resource violations."""
    alert_id: str
    container_id: str
    alert_type: str
    severity: AlertSeverity
    message: str
    timestamp: float
    violations: List[ResourceViolation] = field(default_factory=list)
    resolved: bool = False
    resolution_timestamp: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'alert_id': self.alert_id,
            'container_id': self.container_id,
            'alert_type': self.alert_type,
            'severity': self.severity.value,
            'message': self.message,
            'timestamp': self.timestamp,
            'violations': [v.to_dict() for v in self.violations],
            'resolved': self.resolved,
            'resolution_timestamp': self.resolution_timestamp
        }


@dataclass
class ResourceTrend:
    """Resource usage trend analysis."""
    resource_type: ResourceType
    avg_usage: float
    min_usage: float
    max_usage: float
    trend_direction: str  # "increasing", "decreasing", "stable"
    change_rate: float  # rate of change per second
    prediction_60s: float  # predicted usage in 60 seconds
    anomaly_score: float = 0.0  # 0-1, higher means more anomalous
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'resource_type': self.resource_type.value,
            'avg_usage': self.avg_usage,
            'min_usage': self.min_usage,
            'max_usage': self.max_usage,
            'trend_direction': self.trend_direction,
            'change_rate': self.change_rate,
            'prediction_60s': self.prediction_60s,
            'anomaly_score': self.anomaly_score
        }


class ResourceAnalyzer:
    """Analyzes resource usage patterns and detects anomalies."""
    
    def __init__(self, history_size: int = 300):  # 5 minutes at 1-second intervals
        self.history_size = history_size
        self.usage_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))
        self.baseline_stats: Dict[str, Dict[str, float]] = {}
        
    def add_usage_sample(self, container_id: str, usage: ResourceUsage):
        """Add resource usage sample to history."""
        history = self.usage_history[container_id]
        history.append(usage)
        
        # Update baseline statistics if we have enough samples
        if len(history) >= 30:  # Need at least 30 samples for baseline
            self._update_baseline_stats(container_id)
    
    def _update_baseline_stats(self, container_id: str):
        """Update baseline statistics for container."""
        history = list(self.usage_history[container_id])
        if len(history) < 30:
            return
        
        # Take last 30 samples for baseline (30 seconds)
        recent_samples = history[-30:]
        
        self.baseline_stats[container_id] = {
            'cpu_mean': statistics.mean(s.cpu_percent for s in recent_samples),
            'cpu_stdev': statistics.stdev(s.cpu_percent for s in recent_samples) if len(recent_samples) > 1 else 0,
            'memory_mean': statistics.mean(s.memory_mb for s in recent_samples),
            'memory_stdev': statistics.stdev(s.memory_mb for s in recent_samples) if len(recent_samples) > 1 else 0,
            'network_rx_mean': statistics.mean(s.network_rx_mb for s in recent_samples),
            'network_tx_mean': statistics.mean(s.network_tx_mb for s in recent_samples),
        }
    
    def analyze_trends(self, container_id: str) -> List[ResourceTrend]:
        """Analyze resource usage trends."""
        history = list(self.usage_history.get(container_id, []))
        if len(history) < 10:
            return []
        
        trends = []
        
        # Analyze CPU trend
        cpu_values = [s.cpu_percent for s in history[-60:]]  # Last 60 seconds
        if len(cpu_values) >= 10:
            trends.append(self._analyze_resource_trend(ResourceType.CPU, cpu_values, history[-1].timestamp))
        
        # Analyze Memory trend
        memory_values = [s.memory_mb for s in history[-60:]]
        if len(memory_values) >= 10:
            trends.append(self._analyze_resource_trend(ResourceType.MEMORY, memory_values, history[-1].timestamp))
        
        # Analyze Network trend
        network_values = [s.network_rx_mb + s.network_tx_mb for s in history[-60:]]
        if len(network_values) >= 10:
            trends.append(self._analyze_resource_trend(ResourceType.NETWORK, network_values, history[-1].timestamp))
        
        return trends
    
    def _analyze_resource_trend(self, resource_type: ResourceType, values: List[float], timestamp: float) -> ResourceTrend:
        """Analyze trend for specific resource."""
        if len(values) < 2:
            return ResourceTrend(
                resource_type=resource_type,
                avg_usage=values[0] if values else 0,
                min_usage=values[0] if values else 0,
                max_usage=values[0] if values else 0,
                trend_direction="stable",
                change_rate=0.0,
                prediction_60s=values[0] if values else 0
            )
        
        avg_usage = statistics.mean(values)
        min_usage = min(values)
        max_usage = max(values)
        
        # Calculate trend using linear regression
        n = len(values)
        x = list(range(n))
        
        # Simple linear regression
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(values)
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            slope = 0
        else:
            slope = numerator / denominator
        
        # Determine trend direction
        if abs(slope) < 0.01:
            trend_direction = "stable"
        elif slope > 0:
            trend_direction = "increasing"
        else:
            trend_direction = "decreasing"
        
        # Predict usage in 60 seconds
        prediction_60s = values[-1] + (slope * 60)
        prediction_60s = max(0, prediction_60s)  # Can't be negative
        
        # Calculate anomaly score
        anomaly_score = self._calculate_anomaly_score(values)
        
        return ResourceTrend(
            resource_type=resource_type,
            avg_usage=avg_usage,
            min_usage=min_usage,
            max_usage=max_usage,
            trend_direction=trend_direction,
            change_rate=slope,
            prediction_60s=prediction_60s,
            anomaly_score=anomaly_score
        )
    
    def _calculate_anomaly_score(self, values: List[float]) -> float:
        """Calculate anomaly score based on statistical deviation."""
        if len(values) < 5:
            return 0.0
        
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0
        
        if stdev == 0:
            return 0.0
        
        # Calculate z-score for recent values
        recent_values = values[-5:]  # Last 5 values
        z_scores = [abs(v - mean) / stdev for v in recent_values]
        max_z_score = max(z_scores)
        
        # Convert z-score to anomaly score (0-1)
        anomaly_score = min(max_z_score / 3.0, 1.0)  # 3-sigma rule
        
        return anomaly_score
    
    def detect_violations(self, container_id: str, usage: ResourceUsage, limits: ResourceLimits) -> List[ResourceViolation]:
        """Detect resource violations based on current usage and limits."""
        violations = []
        timestamp = usage.timestamp
        
        # CPU violations
        cpu_limit = limits.cpu_cores * 100  # Convert to percentage
        if usage.cpu_percent > cpu_limit * 0.9:  # 90% of limit
            violations.append(ResourceViolation(
                container_id=container_id,
                violation_type=ViolationType.LIMIT_EXCEEDED,
                resource_type=ResourceType.CPU,
                severity=AlertSeverity.WARNING if usage.cpu_percent < cpu_limit else AlertSeverity.CRITICAL,
                description=f"CPU usage {usage.cpu_percent:.1f}% exceeds {cpu_limit * 0.9:.1f}% threshold",
                current_value=usage.cpu_percent,
                limit_value=cpu_limit,
                timestamp=timestamp
            ))
        
        # Memory violations
        memory_limit_mb = limits.memory_mb
        if usage.memory_mb > memory_limit_mb * 0.9:  # 90% of limit
            violations.append(ResourceViolation(
                container_id=container_id,
                violation_type=ViolationType.LIMIT_EXCEEDED,
                resource_type=ResourceType.MEMORY,
                severity=AlertSeverity.WARNING if usage.memory_mb < memory_limit_mb else AlertSeverity.CRITICAL,
                description=f"Memory usage {usage.memory_mb:.1f}MB exceeds {memory_limit_mb * 0.9:.1f}MB threshold",
                current_value=usage.memory_mb,
                limit_value=memory_limit_mb,
                timestamp=timestamp
            ))
        
        # Process count violations
        if hasattr(limits, 'pids_limit') and usage.process_count > limits.pids_limit * 0.8:
            violations.append(ResourceViolation(
                container_id=container_id,
                violation_type=ViolationType.LIMIT_EXCEEDED,
                resource_type=ResourceType.PROCESSES,
                severity=AlertSeverity.WARNING,
                description=f"Process count {usage.process_count} exceeds {limits.pids_limit * 0.8:.0f} threshold",
                current_value=usage.process_count,
                limit_value=limits.pids_limit,
                timestamp=timestamp
            ))
        
        # Anomaly-based violations
        baseline = self.baseline_stats.get(container_id, {})
        if baseline:
            # CPU anomaly
            cpu_mean = baseline.get('cpu_mean', 0)
            cpu_stdev = baseline.get('cpu_stdev', 0)
            if cpu_stdev > 0 and abs(usage.cpu_percent - cpu_mean) > 3 * cpu_stdev:
                violations.append(ResourceViolation(
                    container_id=container_id,
                    violation_type=ViolationType.ANOMALOUS_BEHAVIOR,
                    resource_type=ResourceType.CPU,
                    severity=AlertSeverity.WARNING,
                    description=f"CPU usage {usage.cpu_percent:.1f}% is anomalous (baseline: {cpu_mean:.1f}±{cpu_stdev:.1f}%)",
                    current_value=usage.cpu_percent,
                    limit_value=cpu_mean + 3 * cpu_stdev,
                    timestamp=timestamp
                ))
            
            # Memory anomaly
            memory_mean = baseline.get('memory_mean', 0)
            memory_stdev = baseline.get('memory_stdev', 0)
            if memory_stdev > 0 and abs(usage.memory_mb - memory_mean) > 3 * memory_stdev:
                violations.append(ResourceViolation(
                    container_id=container_id,
                    violation_type=ViolationType.ANOMALOUS_BEHAVIOR,
                    resource_type=ResourceType.MEMORY,
                    severity=AlertSeverity.WARNING,
                    description=f"Memory usage {usage.memory_mb:.1f}MB is anomalous (baseline: {memory_mean:.1f}±{memory_stdev:.1f}MB)",
                    current_value=usage.memory_mb,
                    limit_value=memory_mean + 3 * memory_stdev,
                    timestamp=timestamp
                ))
        
        return violations
    
    def get_container_statistics(self, container_id: str) -> Dict[str, Any]:
        """Get comprehensive statistics for container."""
        history = list(self.usage_history.get(container_id, []))
        if not history:
            return {'error': 'No usage history available'}
        
        # Calculate basic statistics
        recent_samples = history[-30:]  # Last 30 seconds
        
        stats = {
            'sample_count': len(history),
            'time_span_minutes': (history[-1].timestamp - history[0].timestamp) / 60 if len(history) > 1 else 0,
            'cpu': {
                'current': recent_samples[-1].cpu_percent if recent_samples else 0,
                'average': statistics.mean(s.cpu_percent for s in recent_samples),
                'max': max(s.cpu_percent for s in recent_samples),
                'min': min(s.cpu_percent for s in recent_samples)
            },
            'memory': {
                'current_mb': recent_samples[-1].memory_mb if recent_samples else 0,
                'average_mb': statistics.mean(s.memory_mb for s in recent_samples),
                'max_mb': max(s.memory_mb for s in recent_samples),
                'min_mb': min(s.memory_mb for s in recent_samples)
            },
            'network': {
                'total_rx_mb': recent_samples[-1].network_rx_mb if recent_samples else 0,
                'total_tx_mb': recent_samples[-1].network_tx_mb if recent_samples else 0
            },
            'processes': {
                'current': recent_samples[-1].process_count if recent_samples else 0,
                'max': max(s.process_count for s in recent_samples),
            }
        }
        
        # Add baseline statistics if available
        if container_id in self.baseline_stats:
            stats['baseline'] = self.baseline_stats[container_id]
        
        return stats


class ResourceEnforcer:
    """Enforces resource limits and takes remediation actions."""
    
    def __init__(self, docker_client: docker.DockerClient):
        self.docker_client = docker_client
        self.enforcement_stats = {
            'violations_detected': 0,
            'actions_taken': 0,
            'containers_terminated': 0,
            'warnings_issued': 0
        }
    
    async def enforce_limits(self, container: SecureContainer, violations: List[ResourceViolation]) -> List[str]:
        """Enforce resource limits and take remediation actions."""
        actions_taken = []
        
        if not violations:
            return actions_taken
        
        self.enforcement_stats['violations_detected'] += len(violations)
        
        # Group violations by severity
        critical_violations = [v for v in violations if v.severity == AlertSeverity.CRITICAL]
        warning_violations = [v for v in violations if v.severity == AlertSeverity.WARNING]
        
        try:
            # Handle critical violations
            if critical_violations:
                for violation in critical_violations:
                    action = await self._handle_critical_violation(container, violation)
                    if action:
                        actions_taken.append(action)
                        self.enforcement_stats['actions_taken'] += 1
            
            # Handle warning violations
            if warning_violations:
                for violation in warning_violations:
                    action = await self._handle_warning_violation(container, violation)
                    if action:
                        actions_taken.append(action)
                        self.enforcement_stats['warnings_issued'] += 1
            
        except Exception as e:
            logger.error(f"Error enforcing limits for container {container.name}: {e}")
            actions_taken.append(f"Enforcement error: {e}")
        
        return actions_taken
    
    async def _handle_critical_violation(self, container: SecureContainer, violation: ResourceViolation) -> Optional[str]:
        """Handle critical resource violations."""
        
        if violation.resource_type == ResourceType.MEMORY:
            # Memory exhaustion - terminate container
            try:
                if container.docker_container:
                    container.docker_container.kill()
                    self.enforcement_stats['containers_terminated'] += 1
                    return f"Terminated container due to memory exhaustion ({violation.current_value:.1f}MB > {violation.limit_value:.1f}MB)"
            except Exception as e:
                return f"Failed to terminate container: {e}"
        
        elif violation.resource_type == ResourceType.CPU:
            # CPU exhaustion - throttle container
            try:
                if container.docker_container:
                    # Reduce CPU quota to half
                    new_quota = container.resource_limits.cpu_quota // 2
                    container.docker_container.update(cpu_quota=new_quota)
                    return f"Reduced CPU quota to {new_quota} due to excessive usage ({violation.current_value:.1f}%)"
            except Exception as e:
                return f"Failed to throttle CPU: {e}"
        
        elif violation.resource_type == ResourceType.PROCESSES:
            # Too many processes - warn and monitor
            return f"Process count limit approaching: {violation.current_value}/{violation.limit_value}"
        
        return None
    
    async def _handle_warning_violation(self, container: SecureContainer, violation: ResourceViolation) -> Optional[str]:
        """Handle warning-level resource violations."""
        
        # Log warning and set remediation action
        violation.remediation_action = f"Monitor {violation.resource_type.value} usage closely"
        
        return f"Warning: {violation.description} - monitoring closely"
    
    def get_enforcement_statistics(self) -> Dict[str, Any]:
        """Get enforcement statistics."""
        return self.enforcement_stats.copy()


class AlertManager:
    """Manages resource alerts and notifications."""
    
    def __init__(self, max_alerts: int = 1000):
        self.max_alerts = max_alerts
        self.active_alerts: Dict[str, ResourceAlert] = {}
        self.alert_history: deque = deque(maxlen=max_alerts)
        self.alert_callbacks: List[Callable[[ResourceAlert], None]] = []
        self._alert_counter = 0
    
    def create_alert(self, container_id: str, violations: List[ResourceViolation]) -> ResourceAlert:
        """Create alert from resource violations."""
        if not violations:
            return None
        
        # Determine alert severity (highest violation severity)
        max_severity = max(v.severity for v in violations)
        
        # Generate alert ID
        self._alert_counter += 1
        alert_id = f"alert_{self._alert_counter}_{int(time.time())}"
        
        # Create alert message
        violation_descriptions = [v.description for v in violations[:3]]  # Limit to first 3
        if len(violations) > 3:
            violation_descriptions.append(f"... and {len(violations) - 3} more")
        
        message = f"Resource violations detected: {'; '.join(violation_descriptions)}"
        
        alert = ResourceAlert(
            alert_id=alert_id,
            container_id=container_id,
            alert_type="resource_violation",
            severity=max_severity,
            message=message,
            timestamp=time.time(),
            violations=violations
        )
        
        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Trigger callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.warning(f"Alert callback failed: {e}")
        
        logger.warning(f"Resource alert created: {alert.message}")
        
        return alert
    
    def resolve_alert(self, alert_id: str, resolution_message: Optional[str] = None):
        """Mark alert as resolved."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolution_timestamp = time.time()
            
            if resolution_message:
                alert.message += f" [Resolved: {resolution_message}]"
            
            del self.active_alerts[alert_id]
            logger.info(f"Resource alert resolved: {alert_id}")
    
    def add_alert_callback(self, callback: Callable[[ResourceAlert], None]):
        """Add callback function for alert notifications."""
        self.alert_callbacks.append(callback)
    
    def get_active_alerts(self, container_id: Optional[str] = None) -> List[ResourceAlert]:
        """Get active alerts, optionally filtered by container."""
        alerts = list(self.active_alerts.values())
        
        if container_id:
            alerts = [a for a in alerts if a.container_id == container_id]
        
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_alert_history(self, hours: int = 24) -> List[ResourceAlert]:
        """Get alert history for specified hours."""
        cutoff_time = time.time() - (hours * 3600)
        return [alert for alert in self.alert_history if alert.timestamp > cutoff_time]
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        total_alerts = len(self.alert_history)
        active_count = len(self.active_alerts)
        
        # Count by severity
        severity_counts = defaultdict(int)
        for alert in self.alert_history:
            severity_counts[alert.severity.value] += 1
        
        # Recent alerts (last hour)
        recent_alerts = self.get_alert_history(1)
        
        return {
            'total_alerts': total_alerts,
            'active_alerts': active_count,
            'resolved_alerts': total_alerts - active_count,
            'recent_alerts_1h': len(recent_alerts),
            'severity_distribution': dict(severity_counts),
            'callbacks_registered': len(self.alert_callbacks)
        }


class ResourceMonitor:
    """Main resource monitoring system orchestrating all monitoring components."""
    
    def __init__(self, docker_client: docker.DockerClient, monitoring_interval: float = 1.0):
        self.docker_client = docker_client
        self.monitoring_interval = monitoring_interval
        
        # Initialize components
        self.analyzer = ResourceAnalyzer()
        self.enforcer = ResourceEnforcer(docker_client)
        self.alert_manager = AlertManager()
        
        # Monitoring state
        self.monitored_containers: Dict[str, SecureContainer] = {}
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.monitoring_stats = {
            'samples_collected': 0,
            'containers_monitored': 0,
            'uptime_seconds': 0.0,
            'start_time': 0.0
        }
        
        logger.info("ResourceMonitor initialized")
    
    def add_container(self, container: SecureContainer):
        """Add container to monitoring."""
        self.monitored_containers[container.container_id] = container
        logger.info(f"Added container {container.name} to resource monitoring")
    
    def remove_container(self, container_id: str):
        """Remove container from monitoring."""
        if container_id in self.monitored_containers:
            container = self.monitored_containers[container_id]
            del self.monitored_containers[container_id]
            logger.info(f"Removed container {container.name} from resource monitoring")
    
    async def start_monitoring(self):
        """Start resource monitoring."""
        if self.monitoring_active:
            logger.warning("Resource monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_stats['start_time'] = time.time()
        
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Resource monitoring started")
    
    async def stop_monitoring(self):
        """Stop resource monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None
        
        # Update uptime
        if self.monitoring_stats['start_time'] > 0:
            self.monitoring_stats['uptime_seconds'] = time.time() - self.monitoring_stats['start_time']
        
        logger.info("Resource monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        logger.info("Resource monitoring loop started")
        
        while self.monitoring_active:
            try:
                await asyncio.sleep(self.monitoring_interval)
                await self._monitor_containers()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)
        
        logger.info("Resource monitoring loop stopped")
    
    async def _monitor_containers(self):
        """Monitor all registered containers."""
        if not self.monitored_containers:
            return
        
        self.monitoring_stats['containers_monitored'] = len(self.monitored_containers)
        
        # Monitor each container
        for container_id, container in list(self.monitored_containers.items()):
            try:
                await self._monitor_single_container(container)
                self.monitoring_stats['samples_collected'] += 1
                
            except Exception as e:
                logger.warning(f"Error monitoring container {container.name}: {e}")
    
    async def _monitor_single_container(self, container: SecureContainer):
        """Monitor single container."""
        if not container.docker_container:
            return
        
        try:
            # Get current resource usage
            usage = await self._collect_container_usage(container)
            
            # Add to analyzer
            self.analyzer.add_usage_sample(container.container_id, usage)
            
            # Detect violations
            violations = self.analyzer.detect_violations(
                container.container_id, usage, container.resource_limits
            )
            
            # Update container metrics
            container.metrics.cpu_usage_percent = usage.cpu_percent
            container.metrics.memory_usage_mb = usage.memory_mb
            container.metrics.network_rx_bytes = int(usage.network_rx_mb * 1024 * 1024)
            container.metrics.network_tx_bytes = int(usage.network_tx_mb * 1024 * 1024)
            container.metrics.pids_current = usage.process_count
            
            # Handle violations
            if violations:
                # Create alert
                alert = self.alert_manager.create_alert(container.container_id, violations)
                
                # Enforce limits
                actions = await self.enforcer.enforce_limits(container, violations)
                
                # Log actions
                if actions:
                    logger.warning(f"Enforcement actions for {container.name}: {actions}")
            
        except Exception as e:
            logger.error(f"Error monitoring container {container.name}: {e}")
    
    async def _collect_container_usage(self, container: SecureContainer) -> ResourceUsage:
        """Collect current resource usage for container."""
        usage = ResourceUsage(timestamp=time.time())
        
        try:
            # Get Docker container stats
            stats = container.docker_container.stats(stream=False)
            
            # Parse CPU usage
            cpu_stats = stats.get('cpu_stats', {})
            precpu_stats = stats.get('precpu_stats', {})
            
            if cpu_stats and precpu_stats:
                cpu_usage_delta = cpu_stats.get('cpu_usage', {}).get('total_usage', 0) - precpu_stats.get('cpu_usage', {}).get('total_usage', 0)
                system_cpu_delta = cpu_stats.get('system_cpu_usage', 0) - precpu_stats.get('system_cpu_usage', 0)
                
                if system_cpu_delta > 0 and cpu_usage_delta >= 0:
                    num_cpus = len(cpu_stats.get('cpu_usage', {}).get('percpu_usage', [1]))
                    usage.cpu_percent = (cpu_usage_delta / system_cpu_delta) * num_cpus * 100.0
            
            # Parse memory usage
            memory_stats = stats.get('memory', {})
            if memory_stats:
                usage.memory_mb = memory_stats.get('usage', 0) / (1024 * 1024)
                memory_limit = memory_stats.get('limit', 0)
                if memory_limit > 0:
                    usage.memory_percent = (memory_stats.get('usage', 0) / memory_limit) * 100
            
            # Parse network usage
            networks = stats.get('networks', {})
            if networks and isinstance(networks, dict):
                total_rx_bytes = sum(net.get('rx_bytes', 0) for net in networks.values() if isinstance(net, dict))
                total_tx_bytes = sum(net.get('tx_bytes', 0) for net in networks.values() if isinstance(net, dict))
                usage.network_rx_mb = total_rx_bytes / (1024 * 1024)
                usage.network_tx_mb = total_tx_bytes / (1024 * 1024)
            
            # Parse PIDs
            pids_stats = stats.get('pids', {})
            if pids_stats:
                usage.process_count = pids_stats.get('current', 0)
            
            # Parse block I/O
            blkio_stats = stats.get('blkio_stats', {})
            if blkio_stats and isinstance(blkio_stats, dict):
                io_service_bytes = blkio_stats.get('io_service_bytes_recursive', [])
                if isinstance(io_service_bytes, list):
                    for entry in io_service_bytes:
                        if isinstance(entry, dict):
                            if entry.get('op') == 'Read':
                                usage.disk_read_mb = entry.get('value', 0) / (1024 * 1024)
                            elif entry.get('op') == 'Write':
                                usage.disk_write_mb = entry.get('value', 0) / (1024 * 1024)
        
        except Exception as e:
            logger.warning(f"Error collecting usage stats for {container.name}: {e}")
        
        return usage
    
    def get_container_trends(self, container_id: str) -> List[ResourceTrend]:
        """Get resource trends for container."""
        return self.analyzer.analyze_trends(container_id)
    
    def get_container_statistics(self, container_id: str) -> Dict[str, Any]:
        """Get comprehensive container statistics."""
        stats = self.analyzer.get_container_statistics(container_id)
        
        # Add trend analysis
        trends = self.get_container_trends(container_id)
        stats['trends'] = [t.to_dict() for t in trends]
        
        # Add alert information
        active_alerts = self.alert_manager.get_active_alerts(container_id)
        stats['active_alerts'] = len(active_alerts)
        
        return stats
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive monitoring system statistics."""
        # Update uptime if monitoring is active
        if self.monitoring_active and self.monitoring_stats['start_time'] > 0:
            self.monitoring_stats['uptime_seconds'] = time.time() - self.monitoring_stats['start_time']
        
        return {
            'monitoring_stats': self.monitoring_stats.copy(),
            'analyzer_stats': {
                'containers_tracked': len(self.analyzer.usage_history),
                'baseline_stats_available': len(self.analyzer.baseline_stats)
            },
            'enforcement_stats': self.enforcer.get_enforcement_statistics(),
            'alert_stats': self.alert_manager.get_alert_statistics(),
            'system_resources': self._get_host_system_stats()
        }
    
    def _get_host_system_stats(self) -> Dict[str, Any]:
        """Get host system resource statistics."""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage_percent': psutil.disk_usage('/').percent,
                'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0],
                'boot_time': psutil.boot_time(),
                'process_count': len(psutil.pids())
            }
        except Exception as e:
            logger.warning(f"Error getting host system stats: {e}")
            return {'error': str(e)}
    
    async def cleanup(self):
        """Clean up monitoring resources."""
        await self.stop_monitoring()
        self.monitored_containers.clear()
        logger.info("Resource monitor cleaned up")


# Utility functions
async def monitor_container_resources(
    container: SecureContainer,
    docker_client: docker.DockerClient,
    duration_seconds: int = 60
) -> Dict[str, Any]:
    """Monitor container resources for specified duration."""
    monitor = ResourceMonitor(docker_client, monitoring_interval=1.0)
    monitor.add_container(container)
    
    await monitor.start_monitoring()
    await asyncio.sleep(duration_seconds)
    await monitor.stop_monitoring()
    
    return monitor.get_container_statistics(container.container_id)


def create_resource_alert_callback(log_level: str = "WARNING") -> Callable[[ResourceAlert], None]:
    """Create callback function for resource alerts."""
    def alert_callback(alert: ResourceAlert):
        message = f"Resource Alert [{alert.severity.value.upper()}]: {alert.message}"
        
        if log_level.upper() == "CRITICAL":
            logger.critical(message)
        elif log_level.upper() == "ERROR":
            logger.error(message)
        elif log_level.upper() == "WARNING":
            logger.warning(message)
        else:
            logger.info(message)
    
    return alert_callback


# Export main classes
__all__ = [
    'ResourceMonitor',
    'ResourceAnalyzer',
    'ResourceEnforcer',
    'AlertManager',
    'ResourceUsage',
    'ResourceViolation',
    'ResourceAlert',
    'ResourceTrend',
    'ResourceType',
    'ViolationType',
    'AlertSeverity',
    'monitor_container_resources',
    'create_resource_alert_callback'
]