"""
Comprehensive tests for quality metrics collection system.

Tests all aspects of the QualityMetricsCollector including:
- Metric collection from validation sessions
- Time series data management
- Metric aggregation and statistics
- Data export and import functionality
- Performance and error handling
"""

import pytest
import time
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch
from dataclasses import dataclass, field
from typing import List, Dict, Any
from datetime import datetime, timezone

# Import the system under test
from src.orchestrator.quality.reporting.metrics import (
    QualityMetricsCollector,
    MetricType,
    QualityMetric,
    TimeSeriesMetric,
    MetricsSnapshot
)
from src.orchestrator.quality.validation.engine import (
    ValidationSession,
    RuleExecutionResult
)
from src.orchestrator.quality.validation.rules import (
    RuleViolation,
    RuleSeverity,
    RuleCategory
)
from src.orchestrator.execution.state import ExecutionContext


@pytest.fixture
def mock_logger():
    """Create mock logger for testing."""
    return Mock()


@pytest.fixture
def metrics_collector(mock_logger):
    """Create QualityMetricsCollector instance for testing."""
    return QualityMetricsCollector(
        logger=mock_logger,
        retention_days=7,
        aggregation_interval_seconds=60,
        enable_real_time_collection=True
    )


@pytest.fixture
def sample_execution_context():
    """Create sample execution context."""
    return ExecutionContext(
        execution_id="test_exec_001",
        pipeline_id="test_pipeline",
        environment="test",
        current_step_id="step_1"
    )


@pytest.fixture
def sample_validation_session():
    """Create sample validation session with realistic data."""
    session = ValidationSession(
        session_id="validation_test_001",
        start_time=time.time() - 10,
        total_rules=5,
        rules_executed=5,
        rules_failed=1,
        total_violations=8
    )
    
    # Add rule execution results
    session.rule_results = [
        RuleExecutionResult(
            rule_id="rule_1",
            rule_name="Content Quality Check",
            success=True,
            execution_time_ms=150.5,
            violations=[
                RuleViolation(
                    rule_id="rule_1",
                    message="Minor content issue",
                    severity=RuleSeverity.WARNING,
                    category=RuleCategory.CONTENT,
                    file_path="/test/file1.txt",
                    line_number=10
                ),
                RuleViolation(
                    rule_id="rule_1", 
                    message="Another content issue",
                    severity=RuleSeverity.INFO,
                    category=RuleCategory.CONTENT,
                    file_path="/test/file1.txt",
                    line_number=25
                )
            ]
        ),
        RuleExecutionResult(
            rule_id="rule_2",
            rule_name="Format Validation",
            success=True,
            execution_time_ms=75.2,
            violations=[
                RuleViolation(
                    rule_id="rule_2",
                    message="Format issue detected",
                    severity=RuleSeverity.ERROR,
                    category=RuleCategory.FORMAT,
                    file_path="/test/file2.json",
                    line_number=5
                )
            ]
        ),
        RuleExecutionResult(
            rule_id="rule_3",
            rule_name="Security Check",
            success=False,
            execution_time_ms=250.0,
            error_message="Rule execution timeout",
            violations=[]
        ),
        RuleExecutionResult(
            rule_id="rule_4",
            rule_name="Performance Check",
            success=True,
            execution_time_ms=88.7,
            violations=[
                RuleViolation(
                    rule_id="rule_4",
                    message="Performance threshold exceeded",
                    severity=RuleSeverity.CRITICAL,
                    category=RuleCategory.PERFORMANCE,
                    file_path="/test/file3.py",
                    line_number=1,
                    metadata={"execution_time": 5.2, "memory_usage": 1024}
                )
            ]
        ),
        RuleExecutionResult(
            rule_id="rule_5",
            rule_name="Structure Validation",
            success=True,
            execution_time_ms=45.1,
            violations=[
                RuleViolation(
                    rule_id="rule_5",
                    message="Structure issue 1",
                    severity=RuleSeverity.WARNING,
                    category=RuleCategory.STRUCTURE,
                    file_path="/test/file4.md",
                    line_number=3
                ),
                RuleViolation(
                    rule_id="rule_5",
                    message="Structure issue 2", 
                    severity=RuleSeverity.INFO,
                    category=RuleCategory.STRUCTURE,
                    file_path="/test/file4.md",
                    line_number=15
                ),
                RuleViolation(
                    rule_id="rule_5",
                    message="Structure issue 3",
                    severity=RuleSeverity.WARNING,
                    category=RuleCategory.STRUCTURE,
                    file_path="/test/file4.md",
                    line_number=22
                )
            ]
        )
    ]
    
    # Update violation counts by severity
    for result in session.rule_results:
        for violation in result.violations:
            session.violations_by_severity[violation.severity] += 1
    
    session.finalize()
    return session


class TestQualityMetricsCollector:
    """Test suite for QualityMetricsCollector."""
    
    def test_initialization(self, mock_logger):
        """Test proper initialization of metrics collector."""
        collector = QualityMetricsCollector(
            logger=mock_logger,
            retention_days=30,
            aggregation_interval_seconds=300,
            enable_real_time_collection=True
        )
        
        assert collector.logger == mock_logger
        assert collector.retention_days == 30
        assert collector.aggregation_interval_seconds == 300
        assert collector.enable_real_time_collection is True
        assert collector._metrics == {}
        assert len(collector._snapshots) == 0
        assert collector._collection_stats['total_metrics_collected'] == 0
    
    def test_collect_from_validation_session(self, metrics_collector, sample_validation_session, sample_execution_context):
        """Test collecting metrics from validation session."""
        # Collect metrics
        metrics_collector.collect_from_validation_session(sample_validation_session, sample_execution_context)
        
        # Verify metrics were collected
        all_metrics = metrics_collector.get_all_metrics()
        assert len(all_metrics) > 0
        
        # Verify core session metrics exist
        session_metrics = [m for m in all_metrics.values() if 'validation.sessions.total' in m.name]
        assert len(session_metrics) > 0
        
        duration_metrics = [m for m in all_metrics.values() if 'validation.duration_ms' in m.name]
        assert len(duration_metrics) > 0
        
        # Verify rule-specific metrics
        rule_metrics = [m for m in all_metrics.values() if 'rule_duration_ms' in m.name]
        assert len(rule_metrics) > 0
        
        # Verify violation metrics by severity
        violation_metrics = [m for m in all_metrics.values() if 'violations_by_severity' in m.name]
        assert len(violation_metrics) > 0
        
        # Verify collection stats updated
        assert metrics_collector._collection_stats['total_metrics_collected'] == 1
        assert metrics_collector._collection_stats['collection_errors'] == 0
    
    def test_collect_custom_metric(self, metrics_collector):
        """Test collecting custom metrics."""
        # Collect various types of custom metrics
        metrics_collector.collect_custom_metric(
            name="custom.test.counter",
            value=42,
            metric_type=MetricType.COUNTER,
            labels={"component": "test", "environment": "unittest"},
            metadata={"test_run": True}
        )
        
        metrics_collector.collect_custom_metric(
            name="custom.test.gauge",
            value=78.5,
            metric_type=MetricType.GAUGE,
            labels={"component": "test"}
        )
        
        # Verify metrics were stored
        all_metrics = metrics_collector.get_all_metrics()
        assert len(all_metrics) == 2
        
        # Verify counter metric
        counter_key = 'custom.test.counter{component=test,environment=unittest}'
        assert counter_key in all_metrics
        counter_metric = all_metrics[counter_key]
        assert counter_metric.name == "custom.test.counter"
        assert counter_metric.metric_type == MetricType.COUNTER
        assert counter_metric.get_latest_value() == 42
        assert counter_metric.labels["component"] == "test"
        assert counter_metric.labels["environment"] == "unittest"
        
        # Verify gauge metric  
        gauge_key = 'custom.test.gauge{component=test}'
        assert gauge_key in all_metrics
        gauge_metric = all_metrics[gauge_key]
        assert gauge_metric.name == "custom.test.gauge"
        assert gauge_metric.metric_type == MetricType.GAUGE
        assert gauge_metric.get_latest_value() == 78.5
    
    def test_time_series_functionality(self, metrics_collector):
        """Test time series data management."""
        # Add multiple data points over time
        base_time = time.time()
        for i in range(10):
            timestamp = base_time + i * 60  # Every minute
            metric = QualityMetric(
                name="test.time_series",
                value=i * 10,
                metric_type=MetricType.GAUGE,
                timestamp=timestamp,
                labels={"test": "series"}
            )
            metrics_collector._collect_metric(
                metric.name, metric.value, metric.metric_type, metric.labels
            )
        
        # Get time series
        time_series = metrics_collector.get_metric_time_series("test.time_series", {"test": "series"})
        assert time_series is not None
        assert len(time_series.values) == 10
        assert time_series.get_latest_value() == 90  # Last value (9 * 10)
        
        # Test statistics calculation
        stats = time_series.calculate_statistics()
        assert stats['count'] == 10
        assert stats['min'] == 0
        assert stats['max'] == 90
        assert stats['mean'] == 45  # Average of 0, 10, 20, ..., 90
        assert stats['latest'] == 90
        assert stats['first'] == 0
        assert stats['change'] == 90  # 90 - 0
        
        # Test time window filtering
        recent_window = 5 * 60  # 5 minutes
        recent_values = time_series.get_values_in_range(
            base_time + 5 * 60,  # Start from 5 minutes in
            base_time + 10 * 60  # End at 10 minutes
        )
        assert len(recent_values) == 5  # Values 5, 6, 7, 8, 9
    
    def test_metrics_snapshot_creation(self, metrics_collector, sample_validation_session):
        """Test metrics snapshot functionality."""
        # Collect some metrics first
        metrics_collector.collect_from_validation_session(sample_validation_session)
        metrics_collector.collect_custom_metric("test.snapshot", 123, MetricType.GAUGE)
        
        # Create snapshot
        snapshot = metrics_collector.create_snapshot(include_patterns=["quality.validation.*"])
        
        assert isinstance(snapshot, MetricsSnapshot)
        assert len(snapshot.metrics) > 0
        assert snapshot.summary is not None
        assert 'total_metrics' in snapshot.summary
        
        # Verify snapshot filtering works
        filtered_snapshot = metrics_collector.create_snapshot(include_patterns=["test.*"])
        assert len(filtered_snapshot.metrics) >= 1  # Should include test.snapshot metric
        
        # Verify snapshot is stored
        assert len(metrics_collector._snapshots) >= 1
    
    def test_metrics_pattern_matching(self, metrics_collector):
        """Test metric pattern matching functionality."""
        # Create various metrics
        metrics_collector.collect_custom_metric("quality.validation.rules.total", 5, MetricType.GAUGE)
        metrics_collector.collect_custom_metric("quality.validation.violations", 3, MetricType.GAUGE)  
        metrics_collector.collect_custom_metric("quality.performance.duration", 150, MetricType.TIMER)
        metrics_collector.collect_custom_metric("system.memory.usage", 1024, MetricType.GAUGE)
        
        # Test pattern matching
        quality_metrics = metrics_collector.get_metrics_by_pattern("quality.*")
        assert len(quality_metrics) == 3
        
        validation_metrics = metrics_collector.get_metrics_by_pattern("quality.validation.*")
        assert len(validation_metrics) == 2
        
        all_metrics = metrics_collector.get_metrics_by_pattern("*")
        assert len(all_metrics) == 4
    
    def test_collection_hooks(self, metrics_collector):
        """Test collection hook functionality."""
        hook_calls = []
        
        def before_hook(**kwargs):
            hook_calls.append(("before", kwargs))
        
        def after_hook(**kwargs):
            hook_calls.append(("after", kwargs))
        
        def metric_hook(**kwargs):
            hook_calls.append(("metric", kwargs))
        
        # Register hooks
        metrics_collector.add_collection_hook('before_collection', before_hook)
        metrics_collector.add_collection_hook('after_collection', after_hook)
        metrics_collector.add_collection_hook('on_metric_collected', metric_hook)
        
        # Collect metrics to trigger hooks
        metrics_collector.collect_custom_metric("test.hooks", 42, MetricType.COUNTER)
        
        # Verify hooks were called
        assert len(hook_calls) >= 1  # At least the metric hook should be called
        
        # Find the metric hook call
        metric_calls = [call for call in hook_calls if call[0] == "metric"]
        assert len(metric_calls) >= 1
        assert "metric" in metric_calls[0][1]
    
    def test_data_export_and_import(self, metrics_collector, sample_validation_session):
        """Test metrics export functionality."""
        # Collect metrics
        metrics_collector.collect_from_validation_session(sample_validation_session)
        metrics_collector.collect_custom_metric("export.test", 999, MetricType.GAUGE)
        
        # Test export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        
        try:
            metrics_collector.export_metrics(temp_path, format="json", time_range_hours=24)
            
            # Verify export file exists and has content
            assert temp_path.exists()
            assert temp_path.stat().st_size > 0
            
            # Verify export content
            with open(temp_path, 'r') as f:
                exported_data = json.load(f)
            
            assert 'export_timestamp' in exported_data
            assert 'time_range_hours' in exported_data
            assert 'collection_stats' in exported_data
            assert 'metrics' in exported_data
            assert len(exported_data['metrics']) > 0
            
            # Verify specific metrics in export
            assert any('export.test' in key for key in exported_data['metrics'].keys())
            
        finally:
            # Clean up
            if temp_path.exists():
                temp_path.unlink()
    
    def test_data_cleanup(self, metrics_collector):
        """Test old data cleanup functionality."""
        # Add old metrics (simulate old data)
        old_timestamp = time.time() - (10 * 24 * 3600)  # 10 days ago
        
        time_series = TimeSeriesMetric(
            name="old.test.metric",
            metric_type=MetricType.GAUGE
        )
        
        # Add old and new values
        time_series.add_value(100, old_timestamp)
        time_series.add_value(200, time.time())
        
        metrics_collector._metrics["old.test.metric"] = time_series
        
        # Verify both values exist before cleanup
        assert len(time_series.values) == 2
        
        # Run cleanup (retention is 7 days in test fixture)
        metrics_collector.cleanup_old_data()
        
        # Verify old data was removed
        assert len(time_series.values) == 1
        assert time_series.get_latest_value() == 200
    
    def test_error_handling(self, metrics_collector, mock_logger):
        """Test error handling in metrics collection."""
        # Test invalid hook type
        with pytest.raises(ValueError):
            metrics_collector.add_collection_hook('invalid_hook', lambda: None)
        
        # Test collection with disabled real-time collection
        metrics_collector.enable_real_time_collection = False
        
        # This should not collect metrics
        initial_count = len(metrics_collector.get_all_metrics())
        metrics_collector.collect_custom_metric("disabled.test", 42, MetricType.GAUGE)
        assert len(metrics_collector.get_all_metrics()) == initial_count
        
        # Re-enable collection
        metrics_collector.enable_real_time_collection = True
        
        # Test export with invalid format
        with tempfile.NamedTemporaryFile(suffix='.txt') as temp_file:
            with pytest.raises(ValueError):
                metrics_collector.export_metrics(Path(temp_file.name), format="invalid")
    
    def test_performance_and_threading(self, metrics_collector):
        """Test performance and thread safety."""
        import threading
        import concurrent.futures
        
        def collect_metrics(thread_id):
            """Collect metrics from multiple threads."""
            for i in range(10):
                metrics_collector.collect_custom_metric(
                    f"thread.{thread_id}.metric",
                    i,
                    MetricType.COUNTER,
                    labels={"thread": str(thread_id)}
                )
        
        # Run metrics collection from multiple threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for thread_id in range(5):
                future = executor.submit(collect_metrics, thread_id)
                futures.append(future)
            
            # Wait for all threads to complete
            concurrent.futures.wait(futures)
        
        # Verify all metrics were collected
        all_metrics = metrics_collector.get_all_metrics()
        thread_metrics = [m for m in all_metrics.values() if m.name.startswith("thread.")]
        assert len(thread_metrics) == 50  # 5 threads * 10 metrics each
        
        # Verify collection stats
        stats = metrics_collector.get_collection_stats()
        assert stats['total_metrics_collected'] >= 50
        assert stats['collection_errors'] == 0
    
    def test_metric_key_generation(self, metrics_collector):
        """Test metric key generation with labels."""
        # Test key generation without labels
        key1 = metrics_collector._get_metric_key("test.metric", {})
        assert key1 == "test.metric"
        
        # Test key generation with labels
        key2 = metrics_collector._get_metric_key("test.metric", {"env": "test", "app": "myapp"})
        assert key2 == "test.metric{app=myapp,env=test}"  # Should be sorted
        
        # Test consistent key generation
        key3 = metrics_collector._get_metric_key("test.metric", {"app": "myapp", "env": "test"})
        assert key2 == key3  # Should be identical regardless of input order


class TestTimeSeriesMetric:
    """Test suite for TimeSeriesMetric class."""
    
    def test_time_series_initialization(self):
        """Test TimeSeriesMetric initialization."""
        ts = TimeSeriesMetric(
            name="test.metric",
            metric_type=MetricType.GAUGE,
            labels={"test": "value"}
        )
        
        assert ts.name == "test.metric"
        assert ts.metric_type == MetricType.GAUGE
        assert ts.labels == {"test": "value"}
        assert len(ts.values) == 0
    
    def test_add_value(self):
        """Test adding values to time series."""
        ts = TimeSeriesMetric("test.metric", MetricType.GAUGE)
        
        # Add values with timestamps
        ts.add_value(10, 1000.0)
        ts.add_value(20, 2000.0)
        ts.add_value(30)  # Should use current time
        
        assert len(ts.values) == 3
        assert ts.values[0] == (1000.0, 10)
        assert ts.values[1] == (2000.0, 20)
        assert ts.values[2][1] == 30  # Value should be 30
    
    def test_value_retrieval(self):
        """Test value retrieval methods."""
        ts = TimeSeriesMetric("test.metric", MetricType.GAUGE)
        
        # Empty time series
        assert ts.get_latest_value() is None
        
        # Add values
        base_time = 1000.0
        for i in range(5):
            ts.add_value(i * 10, base_time + i * 100)
        
        # Test latest value
        assert ts.get_latest_value() == 40
        
        # Test range retrieval
        range_values = ts.get_values_in_range(1100.0, 1300.0)
        assert len(range_values) == 3  # Values at 1100, 1200, 1300
        assert range_values[0] == (1100.0, 10)
        assert range_values[1] == (1200.0, 20)
        assert range_values[2] == (1300.0, 30)
    
    def test_statistics_calculation(self):
        """Test statistics calculation."""
        ts = TimeSeriesMetric("test.metric", MetricType.GAUGE)
        
        # Empty time series
        stats = ts.calculate_statistics()
        assert stats == {}
        
        # Add values: 10, 20, 30, 40, 50
        base_time = time.time()
        for i in range(5):
            ts.add_value((i + 1) * 10, base_time + i)
        
        stats = ts.calculate_statistics()
        assert stats['count'] == 5
        assert stats['min'] == 10
        assert stats['max'] == 50
        assert stats['mean'] == 30
        assert stats['median'] == 30
        assert stats['latest'] == 50
        assert stats['first'] == 10
        assert stats['change'] == 40  # 50 - 10
        
        # Test with time window
        window_stats = ts.calculate_statistics(time_window_seconds=2)
        assert window_stats['count'] <= 5  # Should be filtered by time
    
    def test_max_points_limit(self):
        """Test maximum points limit enforcement."""
        ts = TimeSeriesMetric("test.metric", MetricType.COUNTER)
        
        # Add more than the maximum points (10000 default)
        for i in range(10050):
            ts.add_value(i, time.time() + i)
        
        # Should be limited to 10000 points
        assert len(ts.values) == 10000
        # Should keep the most recent values
        assert ts.values[-1][1] == 10049  # Last value should be the most recent


class TestQualityMetric:
    """Test suite for QualityMetric class."""
    
    def test_quality_metric_creation(self):
        """Test QualityMetric creation and properties."""
        timestamp = time.time()
        metric = QualityMetric(
            name="test.metric",
            value=42.5,
            metric_type=MetricType.GAUGE,
            timestamp=timestamp,
            labels={"env": "test"},
            metadata={"source": "unittest"}
        )
        
        assert metric.name == "test.metric"
        assert metric.value == 42.5
        assert metric.metric_type == MetricType.GAUGE
        assert metric.timestamp == timestamp
        assert metric.labels == {"env": "test"}
        assert metric.metadata == {"source": "unittest"}
        
        # Test datetime property
        dt = metric.datetime
        assert isinstance(dt, datetime)
        assert dt.timestamp() == timestamp
        assert dt.tzinfo == timezone.utc


class TestMetricsSnapshot:
    """Test suite for MetricsSnapshot class."""
    
    def test_snapshot_creation(self):
        """Test MetricsSnapshot creation and properties."""
        timestamp = time.time()
        metrics = {
            "test1": QualityMetric("test1", 10, MetricType.GAUGE, timestamp),
            "test2": QualityMetric("test2", 20, MetricType.COUNTER, timestamp)
        }
        summary = {"total_metrics": 2, "timestamp": timestamp}
        
        snapshot = MetricsSnapshot(
            timestamp=timestamp,
            metrics=metrics,
            summary=summary
        )
        
        assert snapshot.timestamp == timestamp
        assert len(snapshot.metrics) == 2
        assert snapshot.summary["total_metrics"] == 2
        
        # Test datetime property
        dt = snapshot.datetime
        assert isinstance(dt, datetime)
        assert dt.timestamp() == timestamp


if __name__ == "__main__":
    pytest.main([__file__])