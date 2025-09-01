"""
Quality Metrics Collection and Aggregation System

This module provides comprehensive metrics collection and aggregation capabilities
for quality control reporting. It integrates with the validation engine and
structured logging system to capture, process, and store quality metrics.

Key Features:
- Real-time quality metric collection from validation sessions
- Time-series data storage and retrieval for trend analysis
- Metric aggregation and rollup calculations
- Performance-optimized data structures for high-volume metrics
- Integration with external monitoring systems
"""

from __future__ import annotations

import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, Iterator, Tuple
from pathlib import Path
import threading
import json

from ..validation.engine import ValidationSession, RuleExecutionResult
from ..validation.rules import RuleViolation, RuleSeverity, RuleCategory
from ..logging.logger import StructuredLogger, get_logger, LogCategory
from ...execution.state import ExecutionContext


class MetricType(Enum):
    """Types of quality metrics for categorization and analysis."""
    COUNTER = "counter"           # Monotonically increasing values
    GAUGE = "gauge"              # Point-in-time values  
    HISTOGRAM = "histogram"      # Distribution of values
    TIMER = "timer"              # Duration measurements
    RATE = "rate"                # Events per time period
    RATIO = "ratio"              # Proportional measurements


@dataclass
class QualityMetric:
    """Individual quality metric data point."""
    name: str
    value: Union[float, int]
    metric_type: MetricType
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def datetime(self) -> datetime:
        """Get timestamp as datetime object."""
        return datetime.fromtimestamp(self.timestamp, tz=timezone.utc)


@dataclass  
class TimeSeriesMetric:
    """Time series of quality metric values."""
    name: str
    metric_type: MetricType
    values: List[Tuple[float, Union[float, int]]] = field(default_factory=list)  # (timestamp, value)
    labels: Dict[str, str] = field(default_factory=dict)
    
    def add_value(self, value: Union[float, int], timestamp: Optional[float] = None) -> None:
        """Add a value to the time series."""
        ts = timestamp or time.time()
        self.values.append((ts, value))
        
        # Keep only recent values for performance (configurable window)
        max_points = 10000  # Keep last 10k points
        if len(self.values) > max_points:
            self.values = self.values[-max_points:]
    
    def get_values_in_range(self, start_time: float, end_time: float) -> List[Tuple[float, Union[float, int]]]:
        """Get values within time range."""
        return [(ts, val) for ts, val in self.values if start_time <= ts <= end_time]
    
    def get_latest_value(self) -> Optional[Union[float, int]]:
        """Get the most recent value."""
        return self.values[-1][1] if self.values else None
    
    def calculate_statistics(self, time_window_seconds: Optional[float] = None) -> Dict[str, float]:
        """Calculate statistical metrics for the time series."""
        values_to_analyze = self.values
        
        if time_window_seconds:
            cutoff_time = time.time() - time_window_seconds
            values_to_analyze = [(ts, val) for ts, val in self.values if ts >= cutoff_time]
        
        if not values_to_analyze:
            return {}
        
        numeric_values = [val for _, val in values_to_analyze]
        
        return {
            'count': len(numeric_values),
            'min': min(numeric_values),
            'max': max(numeric_values),
            'mean': statistics.mean(numeric_values),
            'median': statistics.median(numeric_values),
            'stdev': statistics.stdev(numeric_values) if len(numeric_values) > 1 else 0.0,
            'latest': numeric_values[-1],
            'first': numeric_values[0],
            'change': numeric_values[-1] - numeric_values[0] if len(numeric_values) > 1 else 0.0
        }


@dataclass
class MetricsSnapshot:
    """Point-in-time snapshot of all quality metrics."""
    timestamp: float
    metrics: Dict[str, QualityMetric] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    
    @property 
    def datetime(self) -> datetime:
        """Get snapshot timestamp as datetime."""
        return datetime.fromtimestamp(self.timestamp, tz=timezone.utc)


class QualityMetricsCollector:
    """
    Comprehensive quality metrics collection and aggregation system.
    
    Collects quality metrics from validation sessions, execution contexts,
    and logging events to provide a complete picture of system quality.
    """
    
    def __init__(
        self,
        logger: Optional[StructuredLogger] = None,
        retention_days: int = 30,
        aggregation_interval_seconds: int = 300,  # 5 minutes
        enable_real_time_collection: bool = True
    ):
        """
        Initialize quality metrics collector.
        
        Args:
            logger: Optional structured logger instance
            retention_days: How long to retain detailed metrics
            aggregation_interval_seconds: Interval for metric aggregation
            enable_real_time_collection: Enable real-time metric collection
        """
        self.logger = logger or get_logger("metrics_collector")
        self.retention_days = retention_days
        self.aggregation_interval_seconds = aggregation_interval_seconds
        self.enable_real_time_collection = enable_real_time_collection
        
        # Metric storage
        self._metrics: Dict[str, TimeSeriesMetric] = {}
        self._snapshots: deque[MetricsSnapshot] = deque(maxlen=int(retention_days * 24 * 60 * 60 / aggregation_interval_seconds))
        self._lock = threading.RLock()
        
        # Collection state
        self._collection_hooks: Dict[str, List[Callable]] = {
            'before_collection': [],
            'after_collection': [],
            'on_metric_collected': []
        }
        
        # Performance tracking
        self._collection_stats = {
            'total_metrics_collected': 0,
            'collection_errors': 0,
            'last_collection_time': 0.0,
            'average_collection_duration_ms': 0.0
        }
        
        self.logger.info("Initialized QualityMetricsCollector", category=LogCategory.MONITORING)
    
    def add_collection_hook(self, hook_type: str, callback: Callable) -> None:
        """Add callback hook for metric collection events."""
        if hook_type in self._collection_hooks:
            self._collection_hooks[hook_type].append(callback)
        else:
            raise ValueError(f"Unknown hook type: {hook_type}")
    
    def collect_from_validation_session(self, session: ValidationSession, execution_context: Optional[ExecutionContext] = None) -> None:
        """
        Collect metrics from a validation session.
        
        Args:
            session: Completed validation session
            execution_context: Optional execution context for additional labels
        """
        if not self.enable_real_time_collection:
            return
            
        start_time = time.perf_counter()
        
        try:
            self._execute_hooks('before_collection', session=session)
            
            labels = self._create_base_labels(execution_context)
            labels.update({
                'session_id': session.session_id,
                'data_source': 'validation_session'
            })
            
            # Core session metrics
            self._collect_metric('quality.validation.sessions.total', 1, MetricType.COUNTER, labels)
            self._collect_metric('quality.validation.duration_ms', session.duration_ms, MetricType.TIMER, labels)
            self._collect_metric('quality.validation.rules_executed', session.rules_executed, MetricType.GAUGE, labels)
            self._collect_metric('quality.validation.rules_failed', session.rules_failed, MetricType.GAUGE, labels)
            self._collect_metric('quality.validation.success_rate', session.success_rate, MetricType.GAUGE, labels)
            self._collect_metric('quality.validation.total_violations', session.total_violations, MetricType.GAUGE, labels)
            
            # Violations by severity
            for severity, count in session.violations_by_severity.items():
                severity_labels = labels.copy()
                severity_labels['severity'] = severity.value
                self._collect_metric('quality.validation.violations_by_severity', count, MetricType.GAUGE, severity_labels)
            
            # Rule performance metrics
            for result in session.rule_results:
                rule_labels = labels.copy()
                rule_labels.update({
                    'rule_id': result.rule_id,
                    'rule_name': result.rule_name
                })
                
                self._collect_metric('quality.validation.rule_duration_ms', result.execution_time_ms, MetricType.TIMER, rule_labels)
                self._collect_metric('quality.validation.rule_success', 1 if result.success else 0, MetricType.GAUGE, rule_labels)
                self._collect_metric('quality.validation.rule_violations', len(result.violations), MetricType.GAUGE, rule_labels)
            
            # Violation category analysis
            violations_by_category = self._analyze_violations_by_category(session.rule_results)
            for category, count in violations_by_category.items():
                category_labels = labels.copy()
                category_labels['category'] = category
                self._collect_metric('quality.validation.violations_by_category', count, MetricType.GAUGE, category_labels)
            
            self._collection_stats['total_metrics_collected'] += 1
            self.logger.debug(f"Collected metrics from validation session {session.session_id}", category=LogCategory.MONITORING)
            
            self._execute_hooks('after_collection', session=session)
            
        except Exception as e:
            self._collection_stats['collection_errors'] += 1
            self.logger.error(f"Failed to collect metrics from validation session: {e}", category=LogCategory.MONITORING, exception=e)
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self._update_collection_performance(duration_ms)
    
    def collect_custom_metric(
        self,
        name: str,
        value: Union[float, int],
        metric_type: MetricType,
        labels: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Collect a custom quality metric.
        
        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
            labels: Optional labels for categorization
            metadata: Optional additional metadata
        """
        self._collect_metric(name, value, metric_type, labels or {}, metadata)
    
    def _collect_metric(
        self,
        name: str,
        value: Union[float, int],
        metric_type: MetricType,
        labels: Dict[str, str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Internal metric collection implementation."""
        timestamp = time.time()
        
        # Create metric object
        metric = QualityMetric(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=timestamp,
            labels=labels.copy(),
            metadata=metadata
        )
        
        with self._lock:
            # Get or create time series for this metric
            metric_key = self._get_metric_key(name, labels)
            
            if metric_key not in self._metrics:
                self._metrics[metric_key] = TimeSeriesMetric(
                    name=name,
                    metric_type=metric_type,
                    labels=labels.copy()
                )
            
            # Add value to time series
            self._metrics[metric_key].add_value(value, timestamp)
        
        # Execute metric collection hooks
        self._execute_hooks('on_metric_collected', metric=metric)
    
    def _get_metric_key(self, name: str, labels: Dict[str, str]) -> str:
        """Generate unique key for metric with labels."""
        if not labels:
            return name
            
        # Sort labels for consistent key generation
        label_parts = [f"{k}={v}" for k, v in sorted(labels.items())]
        return f"{name}{{{','.join(label_parts)}}}"
    
    def _create_base_labels(self, execution_context: Optional[ExecutionContext]) -> Dict[str, str]:
        """Create base labels from execution context."""
        labels = {}
        
        if execution_context:
            labels.update({
                'pipeline_id': execution_context.pipeline_id,
                'execution_id': execution_context.execution_id,
                'environment': execution_context.environment
            })
            
            if execution_context.current_step_id:
                labels['step_id'] = execution_context.current_step_id
        
        return labels
    
    def _analyze_violations_by_category(self, rule_results: List[RuleExecutionResult]) -> Dict[str, int]:
        """Analyze violations grouped by category."""
        violations_by_category = defaultdict(int)
        
        for result in rule_results:
            for violation in result.violations:
                violations_by_category[violation.category.value] += 1
        
        return dict(violations_by_category)
    
    def _execute_hooks(self, hook_type: str, **kwargs) -> None:
        """Execute registered hooks."""
        for callback in self._collection_hooks.get(hook_type, []):
            try:
                callback(**kwargs)
            except Exception as e:
                self.logger.error(f"Hook {hook_type} failed: {e}", category=LogCategory.MONITORING)
    
    def _update_collection_performance(self, duration_ms: float) -> None:
        """Update collection performance statistics."""
        self._collection_stats['last_collection_time'] = time.time()
        
        # Update running average
        current_avg = self._collection_stats['average_collection_duration_ms']
        total_collections = self._collection_stats['total_metrics_collected']
        
        if total_collections == 1:
            self._collection_stats['average_collection_duration_ms'] = duration_ms
        else:
            # Exponential moving average with alpha = 0.1
            alpha = 0.1
            self._collection_stats['average_collection_duration_ms'] = (1 - alpha) * current_avg + alpha * duration_ms
    
    def get_metric_time_series(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[TimeSeriesMetric]:
        """
        Get time series for a specific metric.
        
        Args:
            name: Metric name
            labels: Optional labels to filter by
            
        Returns:
            TimeSeriesMetric if found, None otherwise
        """
        metric_key = self._get_metric_key(name, labels or {})
        
        with self._lock:
            return self._metrics.get(metric_key)
    
    def get_all_metrics(self) -> Dict[str, TimeSeriesMetric]:
        """Get all collected metrics."""
        with self._lock:
            return self._metrics.copy()
    
    def get_metrics_by_pattern(self, name_pattern: str) -> Dict[str, TimeSeriesMetric]:
        """Get metrics matching name pattern."""
        import fnmatch
        
        matching_metrics = {}
        with self._lock:
            for key, metric in self._metrics.items():
                if fnmatch.fnmatch(metric.name, name_pattern):
                    matching_metrics[key] = metric
        
        return matching_metrics
    
    def create_snapshot(self, include_patterns: Optional[List[str]] = None) -> MetricsSnapshot:
        """
        Create a point-in-time snapshot of metrics.
        
        Args:
            include_patterns: Optional patterns to filter metrics
            
        Returns:
            MetricsSnapshot with current metric values
        """
        timestamp = time.time()
        snapshot_metrics = {}
        
        with self._lock:
            metrics_to_include = self._metrics
            
            # Filter by patterns if provided
            if include_patterns:
                import fnmatch
                filtered_metrics = {}
                for pattern in include_patterns:
                    for key, metric in self._metrics.items():
                        if fnmatch.fnmatch(metric.name, pattern):
                            filtered_metrics[key] = metric
                metrics_to_include = filtered_metrics
            
            # Create snapshot metrics
            for key, time_series in metrics_to_include.items():
                latest_value = time_series.get_latest_value()
                if latest_value is not None:
                    snapshot_metrics[key] = QualityMetric(
                        name=time_series.name,
                        value=latest_value,
                        metric_type=time_series.metric_type,
                        timestamp=timestamp,
                        labels=time_series.labels.copy()
                    )
        
        # Calculate summary statistics
        summary = self._calculate_snapshot_summary(snapshot_metrics)
        
        snapshot = MetricsSnapshot(
            timestamp=timestamp,
            metrics=snapshot_metrics,
            summary=summary
        )
        
        # Store snapshot
        with self._lock:
            self._snapshots.append(snapshot)
        
        return snapshot
    
    def _calculate_snapshot_summary(self, metrics: Dict[str, QualityMetric]) -> Dict[str, Any]:
        """Calculate summary statistics for a snapshot."""
        if not metrics:
            return {}
        
        # Group metrics by type and calculate summaries
        metrics_by_type = defaultdict(list)
        for metric in metrics.values():
            metrics_by_type[metric.metric_type].append(metric.value)
        
        summary = {
            'total_metrics': len(metrics),
            'metric_types': {
                metric_type.value: len(values) 
                for metric_type, values in metrics_by_type.items()
            }
        }
        
        # Calculate statistics for numeric metrics
        quality_scores = []
        violation_counts = []
        
        for metric in metrics.values():
            if 'quality' in metric.name.lower() and 'score' in metric.name.lower():
                quality_scores.append(metric.value)
            elif 'violation' in metric.name.lower():
                violation_counts.append(metric.value)
        
        if quality_scores:
            summary['quality_score_stats'] = {
                'count': len(quality_scores),
                'avg': sum(quality_scores) / len(quality_scores),
                'min': min(quality_scores),
                'max': max(quality_scores)
            }
        
        if violation_counts:
            summary['violation_stats'] = {
                'total': sum(violation_counts),
                'avg': sum(violation_counts) / len(violation_counts),
                'max': max(violation_counts)
            }
        
        return summary
    
    def get_snapshots_in_range(self, start_time: float, end_time: float) -> List[MetricsSnapshot]:
        """Get snapshots within time range."""
        with self._lock:
            return [s for s in self._snapshots if start_time <= s.timestamp <= end_time]
    
    def export_metrics(self, output_path: Path, format: str = "json", time_range_hours: Optional[int] = None) -> None:
        """
        Export metrics to file.
        
        Args:
            output_path: Path to export file
            format: Export format (json, csv)
            time_range_hours: Optional time range to export (hours from now)
        """
        metrics_data = {}
        
        with self._lock:
            # Determine which metrics to export
            if time_range_hours:
                cutoff_time = time.time() - (time_range_hours * 3600)
                for key, time_series in self._metrics.items():
                    filtered_values = [(ts, val) for ts, val in time_series.values if ts >= cutoff_time]
                    if filtered_values:
                        metrics_data[key] = {
                            'name': time_series.name,
                            'type': time_series.metric_type.value,
                            'labels': time_series.labels,
                            'values': filtered_values
                        }
            else:
                for key, time_series in self._metrics.items():
                    metrics_data[key] = {
                        'name': time_series.name,
                        'type': time_series.metric_type.value,
                        'labels': time_series.labels,
                        'values': time_series.values
                    }
        
        export_data = {
            'export_timestamp': time.time(),
            'time_range_hours': time_range_hours,
            'collection_stats': self._collection_stats.copy(),
            'metrics': metrics_data
        }
        
        if format.lower() == "json":
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        self.logger.info(f"Exported {len(metrics_data)} metrics to {output_path}", category=LogCategory.MONITORING)
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get metrics collection statistics."""
        return self._collection_stats.copy()
    
    def cleanup_old_data(self) -> None:
        """Clean up old metric data based on retention policy."""
        cutoff_time = time.time() - (self.retention_days * 24 * 3600)
        
        cleaned_count = 0
        with self._lock:
            # Clean time series data
            for time_series in self._metrics.values():
                original_count = len(time_series.values)
                time_series.values = [(ts, val) for ts, val in time_series.values if ts >= cutoff_time]
                cleaned_count += original_count - len(time_series.values)
        
        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} old metric data points", category=LogCategory.MONITORING)