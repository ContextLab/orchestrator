"""
Test suite for wrapper monitoring and configuration integration (Issue #251).

This test validates the comprehensive configuration management and performance 
monitoring systems that integrate with the unified wrapper architecture.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal

from src.orchestrator.core.wrapper_config import ExternalToolConfig, ConfigValidationError
from src.orchestrator.core.wrapper_monitoring import WrapperMonitoring, WrapperOperationStatus


class TestWrapperConfigurationIntegration:
    """Test wrapper configuration management."""
    
    def test_external_tool_config_creation(self):
        """Test creating external tool configuration."""
        config = ExternalToolConfig(
            enabled=True,
            api_endpoint="https://api.example.com",
            api_key="test-key-123",
            daily_budget=100.0,
            monthly_budget=3000.0
        )
        
        assert config.enabled is True
        assert config.api_endpoint == "https://api.example.com"
        assert config.daily_budget == 100.0
        assert config.cost_tracking_enabled is True
        assert config.is_valid is True
    
    def test_external_tool_config_validation(self):
        """Test configuration validation."""
        # Test valid configuration
        valid_config = ExternalToolConfig(
            enabled=True,
            api_endpoint="https://api.example.com",
            timeout_seconds=30.0,
            max_retry_attempts=3
        )
        valid_config.validate()  # Should not raise
        
        # Test invalid configuration - empty endpoint
        with pytest.raises(ConfigValidationError):
            invalid_config = ExternalToolConfig(
                enabled=True,
                api_endpoint="",  # Invalid: empty endpoint when enabled
                timeout_seconds=30.0
            )
    
    def test_config_field_validation(self):
        """Test individual field validation."""
        config = ExternalToolConfig()
        config_fields = config.get_config_fields()
        
        # Test API endpoint validation
        api_endpoint_field = config_fields["api_endpoint"]
        assert api_endpoint_field.validate("https://api.example.com") is True
        assert api_endpoint_field.validate("http://api.example.com") is True
        assert api_endpoint_field.validate("invalid-url") is False
        assert api_endpoint_field.validate("") is False
    
    def test_sensitive_field_masking(self):
        """Test that sensitive fields are masked in output."""
        config = ExternalToolConfig(
            api_key="secret-key-123",
            api_endpoint="https://api.example.com"
        )
        
        config_dict = config.to_dict(mask_sensitive=True)
        assert config_dict["api_key"] == "********"  # Masked
        assert config_dict["api_endpoint"] == "https://api.example.com"  # Not masked


class TestWrapperMonitoringIntegration:
    """Test wrapper monitoring system."""
    
    def setup_method(self):
        """Set up test monitoring system."""
        self.monitoring = WrapperMonitoring(
            retention_days=7,
            max_operations_in_memory=1000
        )
    
    def test_operation_lifecycle_tracking(self):
        """Test complete operation lifecycle tracking."""
        # Start operation
        operation_id = self.monitoring.start_operation(
            operation_id="test-op-1",
            wrapper_name="test-wrapper",
            operation_type="test-operation"
        )
        
        assert operation_id == "test-op-1"
        assert operation_id in self.monitoring._active_operations
        
        # Record success
        self.monitoring.record_success(
            operation_id,
            custom_metrics={
                "tokens_used": 150,
                "cost_estimate": 0.02
            }
        )
        
        # End operation
        self.monitoring.end_operation(operation_id)
        
        # Verify operation is completed
        assert operation_id not in self.monitoring._active_operations
        assert len(self.monitoring._completed_operations) == 1
        
        # Check metrics
        completed_op = self.monitoring._completed_operations[0]
        assert completed_op.status == WrapperOperationStatus.SUCCESS
        assert completed_op.success is True
        assert completed_op.custom_metrics["tokens_used"] == 150
        assert completed_op.custom_metrics["cost_estimate"] == 0.02
    
    def test_error_tracking(self):
        """Test error tracking and health impact."""
        # Start and fail an operation
        operation_id = self.monitoring.start_operation(
            "error-op-1",
            "test-wrapper",
            "error-test"
        )
        
        self.monitoring.record_error(
            operation_id,
            "Test error message",
            error_code="TEST_ERROR"
        )
        
        self.monitoring.end_operation(operation_id)
        
        # Check error was recorded
        completed_op = self.monitoring._completed_operations[0]
        assert completed_op.status == WrapperOperationStatus.ERROR
        assert completed_op.success is False
        assert completed_op.error_message == "Test error message"
        assert completed_op.error_code == "TEST_ERROR"
        
        # Check health status
        health = self.monitoring.get_wrapper_health("test-wrapper")
        assert health.failed_operations >= 1
        assert health.error_rate > 0
    
    def test_fallback_tracking(self):
        """Test fallback operation tracking."""
        # Start operation that uses fallback
        operation_id = self.monitoring.start_operation(
            "fallback-op-1",
            "test-wrapper",
            "fallback-test"
        )
        
        # Record that we're using fallback (but still successful)
        self.monitoring.record_success(
            operation_id,
            custom_metrics={"fallback_used": True, "fallback_reason": "primary_failed"}
        )
        
        # Simulate fallback in the metrics
        with self.monitoring._lock:
            metrics = self.monitoring._active_operations[operation_id]
            metrics.fallback_used = True
            metrics.fallback_reason = "primary_failed"
        
        self.monitoring.end_operation(operation_id)
        
        # Check fallback was recorded
        completed_op = self.monitoring._completed_operations[0]
        assert completed_op.fallback_used is True
        assert completed_op.fallback_reason == "primary_failed"
        assert completed_op.success is True  # Still successful with fallback
        
        # Check health status includes fallback
        health = self.monitoring.get_wrapper_health("test-wrapper")
        assert health.fallback_operations >= 1
        assert health.fallback_rate > 0
    
    def test_wrapper_health_calculation(self):
        """Test wrapper health score calculation."""
        wrapper_name = "health-test-wrapper"
        
        # Simulate successful operations
        for i in range(8):
            op_id = f"success-op-{i}"
            self.monitoring.start_operation(op_id, wrapper_name, "test")
            self.monitoring.record_success(op_id)
            self.monitoring.end_operation(op_id)
        
        # Simulate failed operations
        for i in range(2):
            op_id = f"error-op-{i}"
            self.monitoring.start_operation(op_id, wrapper_name, "test")
            self.monitoring.record_error(op_id, "Test error")
            self.monitoring.end_operation(op_id)
        
        # Check health calculation
        health = self.monitoring.get_wrapper_health(wrapper_name)
        assert health.total_operations == 10
        assert health.successful_operations == 8
        assert health.failed_operations == 2
        assert health.success_rate == 0.8
        assert health.error_rate == 0.2
        assert 0.6 <= health.health_score <= 0.9  # Should be reduced due to errors
    
    def test_system_health_summary(self):
        """Test system-wide health summary."""
        # Create operations for multiple wrappers
        for wrapper_num in range(3):
            wrapper_name = f"wrapper-{wrapper_num}"
            for op_num in range(5):
                op_id = f"op-{wrapper_num}-{op_num}"
                self.monitoring.start_operation(op_id, wrapper_name, "test")
                self.monitoring.record_success(op_id)
                self.monitoring.end_operation(op_id)
        
        # Get system health
        system_health = self.monitoring.get_system_health()
        
        assert system_health["total_wrappers"] == 3
        assert system_health["healthy_wrappers"] >= 0
        assert system_health["completed_operations"] == 15
        assert system_health["overall_success_rate"] == 1.0
        assert system_health["health_percentage"] >= 0
    
    def test_metrics_export(self):
        """Test metrics export functionality."""
        wrapper_name = "export-test-wrapper"
        
        # Create some test operations
        for i in range(5):
            op_id = f"export-op-{i}"
            self.monitoring.start_operation(op_id, wrapper_name, "export-test")
            self.monitoring.record_success(
                op_id,
                custom_metrics={
                    "test_metric": i * 10,
                    "operation_index": i
                }
            )
            self.monitoring.end_operation(op_id)
        
        # Export all metrics
        all_metrics = self.monitoring.export_metrics()
        assert len(all_metrics) == 5
        
        # Export wrapper-specific metrics
        wrapper_metrics = self.monitoring.export_metrics(wrapper_name=wrapper_name)
        assert len(wrapper_metrics) == 5
        assert all(m["wrapper_name"] == wrapper_name for m in wrapper_metrics)
        
        # Export time-filtered metrics
        now = datetime.utcnow()
        recent_metrics = self.monitoring.export_metrics(
            start_time=now - timedelta(minutes=1)
        )
        assert len(recent_metrics) == 5  # All should be recent
        
        # Check metric structure
        metric = wrapper_metrics[0]
        assert "operation_id" in metric
        assert "wrapper_name" in metric
        assert "start_time" in metric
        assert "status" in metric
        assert "custom_metrics" in metric
    
    def test_wrapper_stats_calculation(self):
        """Test detailed wrapper statistics calculation."""
        wrapper_name = "stats-test-wrapper"
        
        # Create mixed operation results
        for i in range(10):
            op_id = f"stats-op-{i}"
            self.monitoring.start_operation(op_id, wrapper_name, "stats-test")
            
            if i < 7:  # 7 successes
                self.monitoring.record_success(op_id)
            elif i < 9:  # 2 errors
                self.monitoring.record_error(op_id, f"Error {i}")
            else:  # 1 fallback
                self.monitoring.record_success(op_id)
                # Manually set fallback flag
                with self.monitoring._lock:
                    self.monitoring._active_operations[op_id].fallback_used = True
            
            self.monitoring.end_operation(op_id)
        
        # Get wrapper stats
        stats = self.monitoring.get_wrapper_stats(wrapper_name)
        
        assert stats["wrapper_name"] == wrapper_name
        assert stats["total_operations"] == 10
        assert stats["successful_operations"] == 7  # Not including fallback
        assert stats["failed_operations"] == 2
        assert stats["fallback_operations"] == 1
        assert stats["success_rate"] == 0.7
        assert stats["error_rate"] == 0.2
        assert stats["fallback_rate"] == 0.1


class TestIntegratedMonitoringAndConfiguration:
    """Test integration between monitoring and configuration systems."""
    
    def test_config_driven_monitoring(self):
        """Test that configuration drives monitoring behavior."""
        # Create config with monitoring enabled
        config = ExternalToolConfig(
            enabled=True,
            monitoring_enabled=True,
            metrics_retention_days=7,
            debug_logging=True
        )
        
        # Create monitoring with config-driven settings
        monitoring = WrapperMonitoring(
            retention_days=config.metrics_retention_days,
            max_operations_in_memory=1000
        )
        
        assert monitoring.retention_days == config.metrics_retention_days
        
        # Test operation with config context
        operation_id = monitoring.start_operation(
            "config-test-op",
            "configured-wrapper",
            "config-test"
        )
        
        # Add config-related metrics
        monitoring.record_success(
            operation_id,
            custom_metrics={
                "config_version": str(config._last_updated),
                "monitoring_enabled": config.monitoring_enabled,
                "debug_mode": config.debug_logging
            }
        )
        
        monitoring.end_operation(operation_id)
        
        # Verify config data is in metrics
        metrics = monitoring.export_metrics(wrapper_name="configured-wrapper")
        assert len(metrics) == 1
        assert metrics[0]["custom_metrics"]["monitoring_enabled"] is True
        assert metrics[0]["custom_metrics"]["debug_mode"] is True
    
    def test_cost_tracking_integration_points(self):
        """Test integration points for cost tracking (without full cost monitoring)."""
        monitoring = WrapperMonitoring()
        
        operation_id = monitoring.start_operation(
            "cost-test-op",
            "cost-wrapper",
            "cost-test"
        )
        
        # Record success with cost-related metrics
        monitoring.record_success(
            operation_id,
            custom_metrics={
                "api_cost": 0.05,
                "tokens_used": 200,
                "model_used": "gpt-4",
                "cost_optimization_applied": True
            }
        )
        
        monitoring.end_operation(operation_id)
        
        # Verify cost metrics are captured
        metrics = monitoring.export_metrics(wrapper_name="cost-wrapper")
        cost_metrics = metrics[0]["custom_metrics"]
        
        assert cost_metrics["api_cost"] == 0.05
        assert cost_metrics["tokens_used"] == 200
        assert cost_metrics["model_used"] == "gpt-4"
        assert cost_metrics["cost_optimization_applied"] is True
        
        # Verify cost estimate is set at operation level
        assert "cost_estimate" in metrics[0]