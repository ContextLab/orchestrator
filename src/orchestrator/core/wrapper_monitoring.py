"""
Centralized monitoring and logging infrastructure for wrapper architecture.

This module provides comprehensive monitoring capabilities including:
- Operation tracking and metrics collection
- Performance monitoring and alerting
- Health checking and status reporting
- Audit trails and logging
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Callable, Deque
from threading import Lock
import statistics

logger = logging.getLogger(__name__)


class OperationStatus(Enum):
    """Status of wrapper operations."""
    
    STARTED = "started"
    SUCCESS = "success"
    ERROR = "error"
    FALLBACK = "fallback"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class AlertSeverity(Enum):
    """Alert severity levels."""
    
    INFO = "info"
    WARNING = "warning"
    ERROR = "error" 
    CRITICAL = "critical"


@dataclass
class OperationMetrics:
    """Comprehensive metrics for a wrapper operation."""
    
    # Operation identification
    operation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    wrapper_name: str = ""
    operation_type: str = "default"
    
    # Timing information
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    
    # Operation outcome
    status: OperationStatus = OperationStatus.STARTED
    success: bool = True
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    
    # Fallback information
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    fallback_duration_ms: Optional[float] = None
    
    # Performance metrics
    cpu_usage_percent: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    network_calls: int = 0
    network_latency_ms: Optional[float] = None
    
    # Business metrics
    cost_estimate: Optional[float] = None
    quality_score: Optional[float] = None
    user_satisfaction: Optional[float] = None
    
    # Context information
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    
    # Custom metrics (wrapper-specific)
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def finalize(self, status: OperationStatus = OperationStatus.SUCCESS) -> None:
        """Finalize operation with end time and duration calculation."""
        if self.end_time is None:
            self.end_time = datetime.utcnow()
        
        if self.duration_ms is None:
            self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        
        self.status = status
        self.success = status in [OperationStatus.SUCCESS, OperationStatus.FALLBACK]
    
    def add_custom_metric(self, key: str, value: Any) -> None:
        """Add a custom metric."""
        self.custom_metrics[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        
        # Convert datetime objects to ISO strings
        if data['start_time']:
            data['start_time'] = data['start_time'].isoformat()
        if data['end_time']:
            data['end_time'] = data['end_time'].isoformat()
        
        # Convert enum to value
        data['status'] = data['status'].value if isinstance(data['status'], OperationStatus) else data['status']
        
        return data


@dataclass
class WrapperHealthStatus:
    """Health status for a wrapper."""
    
    wrapper_name: str
    is_healthy: bool = True
    last_check: datetime = field(default_factory=datetime.utcnow)
    
    # Operation statistics
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    fallback_operations: int = 0
    
    # Performance metrics
    average_duration_ms: float = 0.0
    success_rate: float = 1.0
    fallback_rate: float = 0.0
    
    # Error tracking
    recent_errors: List[str] = field(default_factory=list)
    last_error: Optional[datetime] = None
    error_rate: float = 0.0
    
    # Health indicators
    health_score: float = 1.0  # 0.0 to 1.0 scale
    status_message: str = "Healthy"
    
    def update_from_metrics(self, metrics: List[OperationMetrics]) -> None:
        """Update health status from operation metrics."""
        if not metrics:
            return
        
        self.total_operations = len(metrics)
        self.successful_operations = len([m for m in metrics if m.success and not m.fallback_used])
        self.failed_operations = len([m for m in metrics if not m.success])
        self.fallback_operations = len([m for m in metrics if m.fallback_used])
        
        # Calculate rates
        if self.total_operations > 0:
            self.success_rate = self.successful_operations / self.total_operations
            self.fallback_rate = self.fallback_operations / self.total_operations
            self.error_rate = self.failed_operations / self.total_operations
        
        # Calculate average duration
        durations = [m.duration_ms for m in metrics if m.duration_ms is not None]
        if durations:
            self.average_duration_ms = statistics.mean(durations)
        
        # Track recent errors
        self.recent_errors = [
            m.error_message for m in metrics[-10:]  # Last 10 operations
            if m.error_message is not None
        ]
        
        # Find last error time
        error_metrics = [m for m in metrics if not m.success]
        if error_metrics:
            self.last_error = max(m.end_time for m in error_metrics if m.end_time)
        
        # Calculate health score
        self._calculate_health_score()
        
        self.last_check = datetime.utcnow()
    
    def _calculate_health_score(self) -> None:
        """Calculate overall health score based on various factors."""
        # Base score from success rate
        score = self.success_rate
        
        # Penalty for high fallback rate
        if self.fallback_rate > 0.5:
            score *= 0.8
        elif self.fallback_rate > 0.2:
            score *= 0.9
        
        # Penalty for recent errors
        if self.last_error and self.last_error > datetime.utcnow() - timedelta(minutes=10):
            score *= 0.7
        
        # Penalty for high error rate
        if self.error_rate > 0.1:
            score *= 0.6
        elif self.error_rate > 0.05:
            score *= 0.8
        
        self.health_score = max(0.0, min(1.0, score))
        
        # Update status
        if self.health_score >= 0.9:
            self.is_healthy = True
            self.status_message = "Healthy"
        elif self.health_score >= 0.7:
            self.is_healthy = True
            self.status_message = "Degraded performance"
        elif self.health_score >= 0.5:
            self.is_healthy = False
            self.status_message = "Unhealthy - high error rate"
        else:
            self.is_healthy = False
            self.status_message = "Critical - system failing"


@dataclass
class AlertRule:
    """Rule for generating alerts based on metrics."""
    
    name: str
    description: str
    severity: AlertSeverity
    condition: Callable[[WrapperHealthStatus], bool]
    cooldown_minutes: int = 5
    last_triggered: Optional[datetime] = None
    
    def should_trigger(self, health_status: WrapperHealthStatus) -> bool:
        """Check if alert should be triggered."""
        # Check cooldown
        if self.last_triggered:
            cooldown_period = timedelta(minutes=self.cooldown_minutes)
            if datetime.utcnow() - self.last_triggered < cooldown_period:
                return False
        
        # Check condition
        return self.condition(health_status)
    
    def trigger(self) -> None:
        """Mark alert as triggered."""
        self.last_triggered = datetime.utcnow()


@dataclass
class Alert:
    """Alert generated by monitoring system."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    wrapper_name: str = ""
    rule_name: str = ""
    severity: AlertSeverity = AlertSeverity.INFO
    message: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    
    def resolve(self) -> None:
        """Mark alert as resolved."""
        self.resolved = True
        self.resolved_at = datetime.utcnow()


class WrapperMonitoring:
    """
    Centralized monitoring system for all wrappers.
    
    Provides comprehensive monitoring including:
    - Operation tracking and metrics collection  
    - Performance monitoring and health checks
    - Alerting based on configurable rules
    - Audit trails and reporting
    """
    
    def __init__(
        self, 
        retention_days: int = 30,
        max_operations_in_memory: int = 10000,
        health_check_interval_minutes: int = 5
    ):
        """
        Initialize monitoring system.
        
        Args:
            retention_days: How long to keep operation metrics
            max_operations_in_memory: Maximum operations to keep in memory
            health_check_interval_minutes: How often to run health checks
        """
        self.retention_days = retention_days
        self.max_operations_in_memory = max_operations_in_memory
        self.health_check_interval_minutes = health_check_interval_minutes
        
        # Data storage
        self._active_operations: Dict[str, OperationMetrics] = {}
        self._completed_operations: Deque[OperationMetrics] = deque(maxlen=max_operations_in_memory)
        self._wrapper_health: Dict[str, WrapperHealthStatus] = {}
        self._alerts: List[Alert] = []
        self._alert_rules: Dict[str, AlertRule] = {}
        
        # Thread safety
        self._lock = Lock()
        
        # Performance tracking
        self._last_cleanup = datetime.utcnow()
        self._last_health_check = datetime.utcnow()
        
        # Initialize default alert rules
        self._setup_default_alert_rules()
    
    def start_operation(
        self, 
        operation_id: str, 
        wrapper_name: str, 
        operation_type: str = "default"
    ) -> str:
        """
        Start tracking a wrapper operation.
        
        Args:
            operation_id: Unique operation identifier
            wrapper_name: Name of the wrapper
            operation_type: Type of operation
            
        Returns:
            Operation ID for tracking
        """
        with self._lock:
            metrics = OperationMetrics(
                operation_id=operation_id,
                wrapper_name=wrapper_name,
                operation_type=operation_type
            )
            
            self._active_operations[operation_id] = metrics
            
            logger.debug(f"Started tracking operation {operation_id} for {wrapper_name}")
            return operation_id
    
    def record_success(
        self, 
        operation_id: str, 
        result: Any = None,
        custom_metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record successful completion of an operation.
        
        Args:
            operation_id: Operation identifier
            result: Operation result (for custom metrics extraction)
            custom_metrics: Additional custom metrics
        """
        with self._lock:
            if operation_id not in self._active_operations:
                logger.warning(f"Cannot record success for unknown operation: {operation_id}")
                return
            
            metrics = self._active_operations[operation_id]
            metrics.finalize(OperationStatus.SUCCESS)
            
            if custom_metrics:
                metrics.custom_metrics.update(custom_metrics)
            
            # Extract metrics from result if possible
            self._extract_result_metrics(metrics, result)
            
            logger.debug(f"Recorded success for operation {operation_id}")
    
    def record_error(
        self, 
        operation_id: str, 
        error_message: str,
        error_code: Optional[str] = None,
        custom_metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record an error in wrapper operation.
        
        Args:
            operation_id: Operation identifier
            error_message: Error description
            error_code: Optional error code
            custom_metrics: Additional custom metrics
        """
        with self._lock:
            if operation_id not in self._active_operations:
                logger.warning(f"Cannot record error for unknown operation: {operation_id}")
                return
            
            metrics = self._active_operations[operation_id]
            metrics.finalize(OperationStatus.ERROR)
            metrics.success = False
            metrics.error_message = error_message
            metrics.error_code = error_code
            
            if custom_metrics:
                metrics.custom_metrics.update(custom_metrics)
            
            logger.debug(f"Recorded error for operation {operation_id}: {error_message}")
    
    def record_fallback(
        self, 
        operation_id: str, 
        fallback_reason: str,
        custom_metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record that fallback was used for an operation.
        
        Args:
            operation_id: Operation identifier
            fallback_reason: Reason for using fallback
            custom_metrics: Additional custom metrics
        """
        with self._lock:
            if operation_id not in self._active_operations:
                logger.warning(f"Cannot record fallback for unknown operation: {operation_id}")
                return
            
            metrics = self._active_operations[operation_id]
            metrics.fallback_used = True
            metrics.fallback_reason = fallback_reason
            
            if custom_metrics:
                metrics.custom_metrics.update(custom_metrics)
            
            logger.debug(f"Recorded fallback for operation {operation_id}: {fallback_reason}")
    
    def record_fatal_error(
        self, 
        operation_id: str, 
        error_message: str,
        custom_metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a fatal error (both wrapper and fallback failed).
        
        Args:
            operation_id: Operation identifier
            error_message: Error description
            custom_metrics: Additional custom metrics
        """
        with self._lock:
            if operation_id not in self._active_operations:
                logger.warning(f"Cannot record fatal error for unknown operation: {operation_id}")
                return
            
            metrics = self._active_operations[operation_id]
            metrics.finalize(OperationStatus.ERROR)
            metrics.success = False
            metrics.error_message = error_message
            metrics.error_code = "FATAL_ERROR"
            
            if custom_metrics:
                metrics.custom_metrics.update(custom_metrics)
            
            logger.error(f"Recorded fatal error for operation {operation_id}: {error_message}")
    
    def end_operation(self, operation_id: str) -> None:
        """
        End tracking of an operation.
        
        Args:
            operation_id: Operation identifier
        """
        with self._lock:
            if operation_id not in self._active_operations:
                logger.warning(f"Cannot end unknown operation: {operation_id}")
                return
            
            metrics = self._active_operations.pop(operation_id)
            
            # Ensure operation is finalized
            if metrics.end_time is None:
                metrics.finalize()
            
            # Move to completed operations
            self._completed_operations.append(metrics)
            
            # Update wrapper health
            self._update_wrapper_health(metrics.wrapper_name)
            
            # Periodic maintenance
            self._periodic_maintenance()
            
            logger.debug(f"Ended tracking for operation {operation_id}")
    
    def get_wrapper_stats(self, wrapper_name: str) -> Dict[str, Any]:
        """
        Get comprehensive statistics for a wrapper.
        
        Args:
            wrapper_name: Name of the wrapper
            
        Returns:
            Dictionary containing wrapper statistics
        """
        with self._lock:
            # Get recent operations for this wrapper
            recent_ops = [
                op for op in self._completed_operations 
                if op.wrapper_name == wrapper_name
                and op.end_time and op.end_time >= datetime.utcnow() - timedelta(days=1)
            ]
            
            if not recent_ops:
                return {
                    "wrapper_name": wrapper_name,
                    "total_operations": 0,
                    "success_rate": 1.0,
                    "average_duration_ms": 0.0,
                    "last_operation": None
                }
            
            # Calculate statistics
            total_ops = len(recent_ops)
            successful_ops = len([op for op in recent_ops if op.success and not op.fallback_used])
            fallback_ops = len([op for op in recent_ops if op.fallback_used])
            failed_ops = len([op for op in recent_ops if not op.success])
            
            durations = [op.duration_ms for op in recent_ops if op.duration_ms is not None]
            avg_duration = statistics.mean(durations) if durations else 0.0
            
            last_operation = max(recent_ops, key=lambda x: x.end_time or x.start_time)
            
            return {
                "wrapper_name": wrapper_name,
                "total_operations": total_ops,
                "successful_operations": successful_ops,
                "failed_operations": failed_ops,
                "fallback_operations": fallback_ops,
                "success_rate": successful_ops / total_ops if total_ops > 0 else 1.0,
                "fallback_rate": fallback_ops / total_ops if total_ops > 0 else 0.0,
                "error_rate": failed_ops / total_ops if total_ops > 0 else 0.0,
                "average_duration_ms": avg_duration,
                "last_operation": last_operation.end_time.isoformat() if last_operation.end_time else None
            }
    
    def get_wrapper_health(self, wrapper_name: str) -> WrapperHealthStatus:
        """
        Get current health status for a wrapper.
        
        Args:
            wrapper_name: Name of the wrapper
            
        Returns:
            Health status for the wrapper
        """
        with self._lock:
            if wrapper_name not in self._wrapper_health:
                # Create initial health status
                self._wrapper_health[wrapper_name] = WrapperHealthStatus(wrapper_name=wrapper_name)
                self._update_wrapper_health(wrapper_name)
            
            return self._wrapper_health[wrapper_name]
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get overall system health metrics.
        
        Returns:
            Dictionary containing system health information
        """
        with self._lock:
            total_wrappers = len(self._wrapper_health)
            healthy_wrappers = len([h for h in self._wrapper_health.values() if h.is_healthy])
            active_operations = len(self._active_operations)
            total_operations = len(self._completed_operations)
            
            # Calculate aggregate success rate
            if self._completed_operations:
                successful_ops = len([op for op in self._completed_operations if op.success])
                overall_success_rate = successful_ops / len(self._completed_operations)
            else:
                overall_success_rate = 1.0
            
            # Get recent alerts
            recent_alerts = [
                a for a in self._alerts 
                if a.timestamp >= datetime.utcnow() - timedelta(hours=24) and not a.resolved
            ]
            
            return {
                "total_wrappers": total_wrappers,
                "healthy_wrappers": healthy_wrappers,
                "unhealthy_wrappers": total_wrappers - healthy_wrappers,
                "health_percentage": (healthy_wrappers / total_wrappers * 100) if total_wrappers > 0 else 100,
                "active_operations": active_operations,
                "completed_operations": total_operations,
                "overall_success_rate": overall_success_rate,
                "active_alerts": len(recent_alerts),
                "last_health_check": self._last_health_check.isoformat()
            }
    
    def get_alerts(
        self, 
        wrapper_name: Optional[str] = None,
        severity: Optional[AlertSeverity] = None,
        include_resolved: bool = False
    ) -> List[Alert]:
        """
        Get alerts based on filters.
        
        Args:
            wrapper_name: Filter by wrapper name
            severity: Filter by severity
            include_resolved: Whether to include resolved alerts
            
        Returns:
            List of matching alerts
        """
        with self._lock:
            alerts = self._alerts.copy()
            
            if wrapper_name:
                alerts = [a for a in alerts if a.wrapper_name == wrapper_name]
            
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
            
            if not include_resolved:
                alerts = [a for a in alerts if not a.resolved]
            
            # Sort by timestamp (newest first)
            alerts.sort(key=lambda x: x.timestamp, reverse=True)
            
            return alerts
    
    def add_alert_rule(self, alert_rule: AlertRule) -> None:
        """
        Add a custom alert rule.
        
        Args:
            alert_rule: Alert rule to add
        """
        with self._lock:
            self._alert_rules[alert_rule.name] = alert_rule
            logger.info(f"Added alert rule: {alert_rule.name}")
    
    def remove_alert_rule(self, rule_name: str) -> bool:
        """
        Remove an alert rule.
        
        Args:
            rule_name: Name of rule to remove
            
        Returns:
            True if rule was found and removed
        """
        with self._lock:
            if rule_name in self._alert_rules:
                del self._alert_rules[rule_name]
                logger.info(f"Removed alert rule: {rule_name}")
                return True
            return False
    
    def export_metrics(
        self, 
        wrapper_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Export operation metrics as list of dictionaries.
        
        Args:
            wrapper_name: Filter by wrapper name
            start_time: Filter by start time
            end_time: Filter by end time
            
        Returns:
            List of operation metrics as dictionaries
        """
        with self._lock:
            operations = list(self._completed_operations)
            
            # Apply filters
            if wrapper_name:
                operations = [op for op in operations if op.wrapper_name == wrapper_name]
            
            if start_time:
                operations = [op for op in operations if op.start_time >= start_time]
            
            if end_time:
                operations = [op for op in operations if op.start_time <= end_time]
            
            return [op.to_dict() for op in operations]
    
    def _update_wrapper_health(self, wrapper_name: str) -> None:
        """Update health status for a wrapper."""
        # Get recent operations for this wrapper
        recent_ops = [
            op for op in self._completed_operations 
            if op.wrapper_name == wrapper_name
            and op.end_time and op.end_time >= datetime.utcnow() - timedelta(hours=1)
        ]
        
        # Update or create health status
        if wrapper_name not in self._wrapper_health:
            self._wrapper_health[wrapper_name] = WrapperHealthStatus(wrapper_name=wrapper_name)
        
        health_status = self._wrapper_health[wrapper_name]
        health_status.update_from_metrics(recent_ops)
        
        # Check alert rules
        self._check_alert_rules(health_status)
    
    def _check_alert_rules(self, health_status: WrapperHealthStatus) -> None:
        """Check alert rules for a wrapper health status."""
        for rule in self._alert_rules.values():
            if rule.should_trigger(health_status):
                alert = Alert(
                    wrapper_name=health_status.wrapper_name,
                    rule_name=rule.name,
                    severity=rule.severity,
                    message=f"{rule.description} - Health score: {health_status.health_score:.2f}"
                )
                
                self._alerts.append(alert)
                rule.trigger()
                
                logger.warning(f"Alert triggered: {rule.name} for {health_status.wrapper_name}")
    
    def _setup_default_alert_rules(self) -> None:
        """Setup default alert rules."""
        # High error rate alert
        self.add_alert_rule(AlertRule(
            name="high_error_rate",
            description="High error rate detected",
            severity=AlertSeverity.ERROR,
            condition=lambda h: h.error_rate > 0.1,
            cooldown_minutes=10
        ))
        
        # Low health score alert
        self.add_alert_rule(AlertRule(
            name="low_health_score",
            description="Low health score detected",
            severity=AlertSeverity.WARNING,
            condition=lambda h: h.health_score < 0.7,
            cooldown_minutes=15
        ))
        
        # High fallback rate alert
        self.add_alert_rule(AlertRule(
            name="high_fallback_rate",
            description="High fallback rate detected", 
            severity=AlertSeverity.WARNING,
            condition=lambda h: h.fallback_rate > 0.5,
            cooldown_minutes=20
        ))
        
        # Critical health alert
        self.add_alert_rule(AlertRule(
            name="critical_health",
            description="Wrapper in critical health state",
            severity=AlertSeverity.CRITICAL,
            condition=lambda h: h.health_score < 0.3,
            cooldown_minutes=5
        ))
    
    def _extract_result_metrics(self, metrics: OperationMetrics, result: Any) -> None:
        """Extract metrics from operation result if possible."""
        if hasattr(result, 'execution_time_ms'):
            metrics.add_custom_metric('result_execution_time_ms', result.execution_time_ms)
        
        if hasattr(result, 'cost_estimate'):
            metrics.cost_estimate = result.cost_estimate
        
        if hasattr(result, 'quality_score'):
            metrics.quality_score = result.quality_score
    
    def _periodic_maintenance(self) -> None:
        """Perform periodic maintenance tasks."""
        now = datetime.utcnow()
        
        # Cleanup old data every hour
        if now - self._last_cleanup > timedelta(hours=1):
            self._cleanup_old_data()
            self._last_cleanup = now
        
        # Health checks every few minutes
        if now - self._last_health_check > timedelta(minutes=self.health_check_interval_minutes):
            self._run_health_checks()
            self._last_health_check = now
    
    def _cleanup_old_data(self) -> None:
        """Cleanup old data based on retention policy."""
        cutoff_time = datetime.utcnow() - timedelta(days=self.retention_days)
        
        # Clean up old alerts
        original_alert_count = len(self._alerts)
        self._alerts = [
            alert for alert in self._alerts 
            if alert.timestamp >= cutoff_time
        ]
        
        cleaned_alerts = original_alert_count - len(self._alerts)
        if cleaned_alerts > 0:
            logger.debug(f"Cleaned up {cleaned_alerts} old alerts")
    
    def _run_health_checks(self) -> None:
        """Run health checks for all wrappers."""
        for wrapper_name in list(self._wrapper_health.keys()):
            self._update_wrapper_health(wrapper_name)