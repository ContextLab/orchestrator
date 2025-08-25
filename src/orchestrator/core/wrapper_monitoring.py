"""
Wrapper-specific monitoring and alerting system for Issue #251.

This module extends the existing performance monitoring system with
wrapper-specific metrics, cost tracking, and enhanced alerting capabilities.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Deque
from threading import Lock
import statistics

from ..analytics.performance_monitor import PerformanceMonitor, MetricType, AlertSeverity

logger = logging.getLogger(__name__)


class WrapperOperationStatus(Enum):
    """Status of wrapper operations."""
    
    STARTED = "started"
    SUCCESS = "success"
    ERROR = "error"
    FALLBACK = "fallback"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class WrapperOperationMetrics:
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
    status: WrapperOperationStatus = WrapperOperationStatus.STARTED
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
    
    def finalize(self, status: WrapperOperationStatus = WrapperOperationStatus.SUCCESS) -> None:
        """Finalize operation with end time and duration calculation."""
        if self.end_time is None:
            self.end_time = datetime.utcnow()
        
        if self.duration_ms is None:
            self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        
        self.status = status
        self.success = status in [WrapperOperationStatus.SUCCESS, WrapperOperationStatus.FALLBACK]
    
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
        data['status'] = data['status'].value if isinstance(data['status'], WrapperOperationStatus) else data['status']
        
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
    
    def update_from_metrics(self, metrics: List[WrapperOperationMetrics]) -> None:
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


class WrapperMonitoring:
    """
    Enhanced wrapper monitoring system that extends the existing performance monitoring.
    
    Integrates with the existing PerformanceMonitor while providing wrapper-specific
    functionality including cost tracking, health monitoring, and alerting.
    """
    
    def __init__(
        self, 
        performance_monitor: Optional[PerformanceMonitor] = None,
        retention_days: int = 30,
        max_operations_in_memory: int = 10000,
        health_check_interval_minutes: int = 5
    ):
        """
        Initialize wrapper monitoring system.
        
        Args:
            performance_monitor: Optional existing performance monitor to extend
            retention_days: How long to keep operation metrics
            max_operations_in_memory: Maximum operations to keep in memory
            health_check_interval_minutes: How often to run health checks
        """
        self.performance_monitor = performance_monitor
        self.retention_days = retention_days
        self.max_operations_in_memory = max_operations_in_memory
        self.health_check_interval_minutes = health_check_interval_minutes
        
        # Data storage
        self._active_operations: Dict[str, WrapperOperationMetrics] = {}
        self._completed_operations: Deque[WrapperOperationMetrics] = deque(maxlen=max_operations_in_memory)
        self._wrapper_health: Dict[str, WrapperHealthStatus] = {}
        self._alerts: List[Dict[str, Any]] = []
        
        # Thread safety
        self._lock = Lock()
        
        # Performance tracking
        self._last_cleanup = datetime.utcnow()
        self._last_health_check = datetime.utcnow()
        
        logger.info("Initialized wrapper monitoring system")
    
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
            metrics = WrapperOperationMetrics(
                operation_id=operation_id,
                wrapper_name=wrapper_name,
                operation_type=operation_type
            )
            
            self._active_operations[operation_id] = metrics
            
            # Also track in performance monitor if available
            if self.performance_monitor:
                self.performance_monitor.start_timer(f"wrapper_{wrapper_name}_{operation_type}")
            
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
            metrics.finalize(WrapperOperationStatus.SUCCESS)
            
            if custom_metrics:
                metrics.custom_metrics.update(custom_metrics)
            
            # Extract metrics from result if possible
            self._extract_result_metrics(metrics, result)
            
            # Track in performance monitor
            if self.performance_monitor:
                self.performance_monitor.record_metric(
                    MetricType.EXECUTION_TIME,
                    metrics.duration_ms,
                    {"wrapper": metrics.wrapper_name, "operation": metrics.operation_type}
                )
            
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
            metrics.finalize(WrapperOperationStatus.ERROR)
            metrics.success = False
            metrics.error_message = error_message
            metrics.error_code = error_code
            
            if custom_metrics:
                metrics.custom_metrics.update(custom_metrics)
            
            logger.debug(f"Recorded error for operation {operation_id}: {error_message}")
    
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
            
            return {
                "total_wrappers": total_wrappers,
                "healthy_wrappers": healthy_wrappers,
                "unhealthy_wrappers": total_wrappers - healthy_wrappers,
                "health_percentage": (healthy_wrappers / total_wrappers * 100) if total_wrappers > 0 else 100,
                "active_operations": active_operations,
                "completed_operations": total_operations,
                "overall_success_rate": overall_success_rate,
                "last_health_check": self._last_health_check.isoformat()
            }
    
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
    
    def _extract_result_metrics(self, metrics: WrapperOperationMetrics, result: Any) -> None:
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
            if datetime.fromisoformat(alert.get('timestamp', '1970-01-01')) >= cutoff_time
        ]
        
        cleaned_alerts = original_alert_count - len(self._alerts)
        if cleaned_alerts > 0:
            logger.debug(f"Cleaned up {cleaned_alerts} old alerts")
    
    def _run_health_checks(self) -> None:
        """Run health checks for all wrappers."""
        for wrapper_name in list(self._wrapper_health.keys()):
            self._update_wrapper_health(wrapper_name)