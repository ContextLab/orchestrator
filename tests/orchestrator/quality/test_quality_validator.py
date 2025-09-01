"""
Tests for the main OutputQualityValidator and integration functionality.

These tests verify the primary validator interface, real-time integration,
and quality control management components.
"""

import pytest
import tempfile
import time
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, call

from src.orchestrator.quality.validation.validator import (
    OutputQualityValidator, ValidationResult, ValidationSeverity
)
from src.orchestrator.quality.validation.integration import (
    ExecutionQualityMonitor, QualityControlManager, ValidationTrigger
)
from src.orchestrator.quality.validation.rules import (
    RuleRegistry, RuleSeverity, RuleCategory
)
from src.orchestrator.execution.state import ExecutionContext, ExecutionStatus, ExecutionMetrics


class TestOutputQualityValidator:
    """Test suite for OutputQualityValidator functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        # Create mock execution context
        self.execution_context = Mock(spec=ExecutionContext)
        self.execution_context.execution_id = "test_execution_001"
        self.execution_context.pipeline_id = "test_pipeline"
        self.execution_context.current_step_id = "test_step"
        self.execution_context.status = ExecutionStatus.RUNNING
        
        # Create test metrics
        self.metrics = Mock(spec=ExecutionMetrics)
        self.metrics.duration = None
        self.metrics.memory_peak_mb = 100.0
        self.metrics.steps_completed = 5
        self.metrics.steps_failed = 0
        self.metrics.cpu_time_seconds = 15.5
        self.execution_context.metrics = self.metrics
        
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create test file
        self.test_file = self.temp_path / "test_output.txt"
        self.test_file.write_text("Test output content for validation")
        
        # Initialize validator
        self.validator = OutputQualityValidator(quality_threshold=75.0)
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_validator_initialization(self):
        """Test validator initialization with different configurations."""
        # Test default initialization
        validator = OutputQualityValidator()
        assert validator.rule_registry is not None
        assert validator.validation_engine is not None
        assert validator.quality_threshold == 70.0  # Default
        assert validator.enable_real_time is True
        
        # Test custom initialization
        custom_registry = RuleRegistry()
        validator = OutputQualityValidator(
            rule_registry=custom_registry,
            enable_real_time=False,
            quality_threshold=85.0
        )
        assert validator.rule_registry == custom_registry
        assert validator.quality_threshold == 85.0
        assert validator.enable_real_time is False
    
    def test_validator_with_config_file(self):
        """Test validator initialization with configuration file."""
        # Create test configuration file
        config_data = {
            'validation_config': {
                'quality_threshold': 80.0,
                'enable_real_time': False
            },
            'rules': [
                {
                    'rule_id': 'file_size_limit',
                    'enabled': True,
                    'config': {
                        'max_size_mb': 200
                    }
                }
            ]
        }
        
        config_file = self.temp_path / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Initialize validator with config
        validator = OutputQualityValidator(config_path=config_file)
        
        # Verify file size rule was configured
        file_size_rule = validator.rule_registry.get_rule('file_size_limit')
        assert file_size_rule is not None
        assert file_size_rule.max_size_mb == 200
    
    def test_register_execution_context(self):
        """Test registering execution context for monitoring."""
        # Mock execution context methods
        self.execution_context.add_step_handler = Mock()
        self.execution_context.add_status_handler = Mock()
        
        # Register context
        self.validator.register_execution_context(self.execution_context)
        
        # Verify context was registered
        assert self.execution_context.execution_id in self.validator.execution_contexts
        assert self.validator.execution_contexts[self.execution_context.execution_id] == self.execution_context
        
        # Verify handlers were added (if real-time enabled)
        if self.validator.enable_real_time:
            self.execution_context.add_step_handler.assert_called_once()
            self.execution_context.add_status_handler.assert_called_once()
    
    def test_validate_output_basic(self):
        """Test basic output validation functionality."""
        result = self.validator.validate_output(
            execution_context=self.execution_context,
            output_path=self.test_file
        )
        
        # Verify result structure
        assert isinstance(result, ValidationResult)
        assert result.session_id is not None
        assert result.pipeline_id == "test_pipeline"
        assert result.execution_id == "test_execution_001"
        assert result.step_id == "test_step"
        assert result.output_path == str(self.test_file)
        
        # Should have some quality score
        assert 0 <= result.quality_score <= 100
        
        # Should have execution metrics
        assert result.execution_metrics is not None
        assert result.execution_metrics["steps_completed"] == 5
        assert result.execution_metrics["memory_peak_mb"] == 100.0
    
    def test_validate_output_with_violations(self):
        """Test validation that detects quality issues."""
        # Create file with prohibited content
        problem_file = self.temp_path / "problem.py"
        problem_content = '''
        def authenticate():
            password = "hardcoded_secret_123"  # TODO: Fix this
            api_key = "sk-test-key-12345"      # FIXME: Move to environment
            return True
        '''
        problem_file.write_text(problem_content)
        
        # Configure content quality rule to detect issues
        content_rule = self.validator.rule_registry.get_rule('content_quality')
        if content_rule:
            content_rule.configure(
                prohibited_patterns=[
                    r"TODO:",
                    r"FIXME:",
                    r"password\s*=\s*['\"][^'\"]*['\"]",
                    r"api_key\s*=\s*['\"][^'\"]*['\"]"
                ]
            )
        
        result = self.validator.validate_output(
            execution_context=self.execution_context,
            output_path=problem_file
        )
        
        # Should detect quality issues
        assert result.total_violations > 0
        assert result.quality_score < 100.0
        
        # Should have violation details for Stream C
        assert len(result.violation_details) > 0
        
        # Verify violation structure
        violation = result.violation_details[0]
        assert "rule_id" in violation
        assert "message" in violation
        assert "severity" in violation
        assert "category" in violation
    
    def test_validate_output_error_handling(self):
        """Test error handling during validation."""
        # Try to validate non-existent file
        non_existent_file = self.temp_path / "does_not_exist.txt"
        
        result = self.validator.validate_output(
            execution_context=self.execution_context,
            output_path=non_existent_file
        )
        
        # Should handle error gracefully
        assert result.severity == ValidationSeverity.CRITICAL
        assert result.passed is False
        assert result.quality_score == 0.0
        assert len(result.recommendations) > 0
        assert any("error" in rec.lower() for rec in result.recommendations)
    
    def test_validate_pipeline_outputs_batch(self):
        """Test batch validation of multiple outputs."""
        # Create multiple test files
        test_files = []
        for i in range(3):
            test_file = self.temp_path / f"output_{i}.txt"
            test_file.write_text(f"Test content {i}")
            test_files.append(test_file)
        
        # Run batch validation
        results = self.validator.validate_pipeline_outputs(
            execution_context=self.execution_context,
            output_paths=test_files,
            batch_size=2
        )
        
        # Verify results
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.output_path == str(test_files[i])
            assert result.execution_id == "test_execution_001"
    
    def test_get_quality_metrics(self):
        """Test retrieving quality metrics for an execution."""
        # First validate some outputs to generate metrics
        result1 = self.validator.validate_output(
            execution_context=self.execution_context,
            output_path=self.test_file
        )
        
        # Create another output and validate
        test_file2 = self.temp_path / "output2.txt"
        test_file2.write_text("Another test file")
        result2 = self.validator.validate_output(
            execution_context=self.execution_context,
            output_path=test_file2
        )
        
        # Get aggregated metrics
        metrics = self.validator.get_quality_metrics("test_execution_001")
        
        # Verify metrics structure
        assert metrics["execution_id"] == "test_execution_001"
        assert metrics["total_validations"] >= 2
        assert 0 <= metrics["average_quality_score"] <= 100
        assert metrics["total_violations"] >= 0
        assert "timestamp" in metrics
    
    def test_real_time_handlers(self):
        """Test real-time validation handlers."""
        # Track handler calls
        handler_calls = []
        alert_calls = []
        
        def quality_handler(result):
            handler_calls.append(result)
        
        def alert_handler(result):
            alert_calls.append(result)
        
        # Register handlers
        self.validator.add_real_time_handler(quality_handler)
        self.validator.add_quality_alert_handler(alert_handler)
        
        # Create low-quality output to trigger alerts
        low_quality_file = self.temp_path / "low_quality.py"
        low_quality_content = '''
        # TODO: Complete implementation
        # FIXME: Handle edge cases
        # XXX: This is broken
        password = "secret123"
        api_key = "test_key_456"
        '''
        low_quality_file.write_text(low_quality_content)
        
        # Set low quality threshold to trigger alert
        self.validator.set_quality_threshold(90.0)
        
        # Validate (should trigger handlers)
        result = self.validator.validate_output(
            execution_context=self.execution_context,
            output_path=low_quality_file,
            real_time=True
        )
        
        # Verify handlers were called
        assert len(handler_calls) == 1
        assert handler_calls[0] == result
        
        # Alert handler should be called if quality is low
        if result.quality_score < 90.0:
            assert len(alert_calls) == 1
            assert alert_calls[0] == result
    
    def test_quality_threshold_management(self):
        """Test quality threshold setting and validation."""
        # Test default threshold
        assert self.validator.quality_threshold == 75.0
        
        # Test setting valid thresholds
        self.validator.set_quality_threshold(80.0)
        assert self.validator.quality_threshold == 80.0
        
        # Test boundary values
        self.validator.set_quality_threshold(-10.0)  # Should clamp to 0
        assert self.validator.quality_threshold == 0.0
        
        self.validator.set_quality_threshold(150.0)  # Should clamp to 100
        assert self.validator.quality_threshold == 100.0
    
    def test_validation_statistics(self):
        """Test validation statistics collection."""
        # Get initial statistics
        initial_stats = self.validator.get_validation_statistics()
        assert initial_stats["total_sessions"] == 0
        assert initial_stats["average_violations"] == 0.0
        
        # Perform some validations
        result1 = self.validator.validate_output(
            execution_context=self.execution_context,
            output_path=self.test_file
        )
        
        # Get updated statistics
        updated_stats = self.validator.get_validation_statistics()
        assert updated_stats["total_sessions"] >= 1
        assert updated_stats["total_violations"] >= 0
        assert "rule_usage" in updated_stats
        assert "active_execution_contexts" in updated_stats
    
    def test_session_cleanup(self):
        """Test validation session cleanup."""
        # Perform validation to create session
        result = self.validator.validate_output(
            execution_context=self.execution_context,
            output_path=self.test_file
        )
        
        session_id = result.session_id
        
        # Verify session exists
        assert session_id in self.validator.validation_sessions
        
        # Clean up session
        success = self.validator.cleanup_session(session_id)
        assert success is True
        assert session_id not in self.validator.validation_sessions
        
        # Test cleaning up non-existent session
        success = self.validator.cleanup_session("non_existent_session")
        assert success is False


class TestExecutionQualityMonitor:
    """Test suite for ExecutionQualityMonitor functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.validator = Mock()
        self.monitor = ExecutionQualityMonitor(self.validator)
        
        # Create mock execution context
        self.execution_context = Mock(spec=ExecutionContext)
        self.execution_context.execution_id = "monitor_test_001"
        self.execution_context.pipeline_id = "test_pipeline"
        self.execution_context.current_step_id = "test_step"
        self.execution_context.add_step_handler = Mock()
        self.execution_context.add_status_handler = Mock()
    
    def test_start_stop_monitoring(self):
        """Test starting and stopping execution monitoring."""
        # Start monitoring
        self.monitor.start_monitoring(self.execution_context)
        
        # Verify monitoring was started
        assert "monitor_test_001" in self.monitor.monitored_executions
        assert self.monitor.monitored_executions["monitor_test_001"] == self.execution_context
        assert "monitor_test_001" in self.monitor.validation_results
        
        # Verify handlers were added
        self.execution_context.add_step_handler.assert_called_once()
        self.execution_context.add_status_handler.assert_called_once()
        
        # Verify validator registration
        self.validator.register_execution_context.assert_called_once_with(self.execution_context)
        
        # Stop monitoring
        self.monitor.stop_monitoring("monitor_test_001")
        
        # Verify monitoring was stopped
        assert "monitor_test_001" not in self.monitor.monitored_executions
        # Results should be kept for reporting
        assert "monitor_test_001" in self.monitor.validation_results
    
    def test_monitoring_with_custom_trigger_config(self):
        """Test monitoring with custom trigger configuration."""
        # Create custom trigger config
        trigger_config = ValidationTrigger(
            on_step_completion=False,  # Disable step monitoring
            on_execution_completion=True,
            step_filter={"specific_step"}
        )
        
        monitor = ExecutionQualityMonitor(self.validator, trigger_config)
        
        # Verify configuration
        assert monitor.trigger_config.on_step_completion is False
        assert monitor.trigger_config.on_execution_completion is True
        assert monitor.trigger_config.step_filter == {"specific_step"}
    
    def test_quality_handlers(self):
        """Test quality result and alert handlers."""
        # Track handler calls
        quality_calls = []
        alert_calls = []
        
        def quality_handler(execution_id, result):
            quality_calls.append((execution_id, result))
        
        def alert_handler(result):
            alert_calls.append(result)
        
        # Add handlers
        self.monitor.add_quality_handler(quality_handler)
        self.monitor.add_alert_handler(alert_handler)
        
        # Verify handlers were added
        assert len(self.monitor.quality_handlers) == 1
        assert len(self.monitor.alert_handlers) == 1
    
    def test_get_execution_quality_summary(self):
        """Test execution quality summary generation."""
        execution_id = "summary_test_001"
        
        # Test with no results
        summary = self.monitor.get_execution_quality_summary(execution_id)
        assert summary["execution_id"] == execution_id
        assert summary["total_validations"] == 0
        assert summary["average_quality_score"] == 0.0
        assert summary["quality_trend"] == "no_data"
        
        # Add mock validation results
        mock_results = []
        for i in range(3):
            result = Mock()
            result.quality_score = 80.0 + i * 5  # Improving scores
            result.severity = ValidationSeverity.PASS if i > 0 else ValidationSeverity.WARNING
            result.has_critical_issues.return_value = False
            result.get_summary_for_stream_c.return_value = {"test": f"data_{i}"}
            mock_results.append(result)
        
        # Store results in monitor
        self.monitor.validation_results[execution_id] = mock_results
        
        # Get updated summary
        summary = self.monitor.get_execution_quality_summary(execution_id)
        assert summary["total_validations"] == 3
        assert summary["average_quality_score"] == 85.0  # Average of 80, 85, 90
        assert summary["quality_trend"] in ["stable", "improving"]
        assert "severity_distribution" in summary
        assert summary["alerts_count"] == 0  # No critical issues
    
    def test_monitoring_statistics(self):
        """Test monitoring statistics collection."""
        # Get initial statistics
        stats = self.monitor.get_monitoring_statistics()
        assert stats["monitored_executions"] == 0
        assert stats["total_executions_processed"] == 0
        assert stats["total_validations"] == 0
        assert stats["average_quality_score"] == 0.0
        
        # Start monitoring an execution
        self.monitor.start_monitoring(self.execution_context)
        
        # Add mock results
        execution_id = self.execution_context.execution_id
        mock_result = Mock()
        mock_result.quality_score = 75.0
        mock_result.severity = ValidationSeverity.WARNING
        self.monitor.validation_results[execution_id] = [mock_result]
        
        # Get updated statistics
        stats = self.monitor.get_monitoring_statistics()
        assert stats["monitored_executions"] == 1
        assert stats["total_executions_processed"] == 1
        assert stats["total_validations"] == 1
        assert stats["average_quality_score"] == 75.0


class TestQualityControlManager:
    """Test suite for QualityControlManager functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.manager = QualityControlManager(enable_monitoring=True)
        
        # Create mock execution context
        self.execution_context = Mock(spec=ExecutionContext)
        self.execution_context.execution_id = "manager_test_001"
        self.execution_context.pipeline_id = "test_pipeline"
        self.execution_context.add_step_handler = Mock()
        self.execution_context.add_status_handler = Mock()
        
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = Path(self.temp_dir) / "test_output.txt"
        self.test_file.write_text("Test content")
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_manager_initialization(self):
        """Test quality control manager initialization."""
        # Test with monitoring enabled
        manager = QualityControlManager(enable_monitoring=True)
        assert manager.validator is not None
        assert manager.monitor is not None
        
        # Test with monitoring disabled
        manager = QualityControlManager(enable_monitoring=False)
        assert manager.validator is not None
        assert manager.monitor is None
    
    def test_register_execution_context(self):
        """Test registering execution context with manager."""
        # Register context
        self.manager.register_execution_context(self.execution_context)
        
        # Verify registration
        assert "manager_test_001" in self.manager.registered_executions
        
        # Try to register same context again (should warn but not fail)
        self.manager.register_execution_context(self.execution_context)
        assert "manager_test_001" in self.manager.registered_executions
    
    def test_validate_output_through_manager(self):
        """Test validating output through the manager."""
        # Register execution context first
        self.manager.register_execution_context(self.execution_context)
        
        # Validate output
        result = self.manager.validate_output(
            execution_context=self.execution_context,
            output_path=self.test_file
        )
        
        # Verify result
        assert isinstance(result, ValidationResult)
        assert result.execution_id == "manager_test_001"
        assert result.output_path == str(self.test_file)
    
    def test_get_comprehensive_quality_metrics(self):
        """Test getting comprehensive quality metrics."""
        execution_id = "manager_test_001"
        
        # Register and perform validation
        self.manager.register_execution_context(self.execution_context)
        result = self.manager.validate_output(
            execution_context=self.execution_context,
            output_path=self.test_file
        )
        
        # Get comprehensive metrics
        metrics = self.manager.get_quality_metrics(execution_id)
        
        # Verify metrics structure
        assert metrics["execution_id"] == execution_id
        assert "validator_metrics" in metrics
        assert "monitor_metrics" in metrics
        assert "overall_quality_score" in metrics
        assert "quality_status" in metrics
        assert "timestamp" in metrics
    
    def test_quality_status_determination(self):
        """Test quality status determination logic."""
        # Test with no monitor metrics (unknown status)
        status = self.manager._determine_quality_status({})
        assert status == "unknown"
        
        # Test with critical issues
        critical_metrics = {
            "severity_distribution": {"critical": 1, "fail": 0, "warning": 0}
        }
        status = self.manager._determine_quality_status(critical_metrics)
        assert status == "critical"
        
        # Test with failures
        fail_metrics = {
            "severity_distribution": {"critical": 0, "fail": 1, "warning": 0}
        }
        status = self.manager._determine_quality_status(fail_metrics)
        assert status == "failed"
        
        # Test with warnings
        warning_metrics = {
            "severity_distribution": {"critical": 0, "fail": 0, "warning": 1}
        }
        status = self.manager._determine_quality_status(warning_metrics)
        assert status == "warning"
        
        # Test with clean results
        clean_metrics = {
            "severity_distribution": {"critical": 0, "fail": 0, "warning": 0}
        }
        status = self.manager._determine_quality_status(clean_metrics)
        assert status == "passed"
    
    def test_system_statistics(self):
        """Test system-wide statistics collection."""
        # Get initial statistics
        stats = self.manager.get_system_statistics()
        assert stats["registered_executions"] == 0
        assert "validator_statistics" in stats
        assert "monitor_statistics" in stats
        
        # Register execution
        self.manager.register_execution_context(self.execution_context)
        
        # Get updated statistics
        stats = self.manager.get_system_statistics()
        assert stats["registered_executions"] == 1
    
    def test_execution_cleanup(self):
        """Test execution resource cleanup."""
        execution_id = "manager_test_001"
        
        # Register execution
        self.manager.register_execution_context(self.execution_context)
        assert execution_id in self.manager.registered_executions
        
        # Clean up execution
        self.manager.cleanup_execution(execution_id)
        
        # Verify cleanup
        assert execution_id not in self.manager.registered_executions


class TestValidationResult:
    """Test suite for ValidationResult functionality."""
    
    def test_validation_result_creation(self):
        """Test creating validation results."""
        result = ValidationResult(
            session_id="test_session_001",
            pipeline_id="test_pipeline",
            execution_id="test_execution_001",
            step_id="test_step",
            output_path="/path/to/output.txt",
            severity=ValidationSeverity.WARNING,
            passed=False,
            quality_score=65.0
        )
        
        # Verify basic attributes
        assert result.session_id == "test_session_001"
        assert result.pipeline_id == "test_pipeline"
        assert result.execution_id == "test_execution_001"
        assert result.step_id == "test_step"
        assert result.output_path == "/path/to/output.txt"
        assert result.severity == ValidationSeverity.WARNING
        assert result.passed is False
        assert result.quality_score == 65.0
        
        # Verify default collections are initialized
        assert isinstance(result.violations_by_severity, dict)
        assert isinstance(result.violations_by_category, dict)
        assert isinstance(result.violation_details, list)
        assert isinstance(result.rule_results, list)
        assert isinstance(result.recommendations, list)
    
    def test_validation_result_issue_checks(self):
        """Test validation result issue checking methods."""
        # Test critical result
        critical_result = ValidationResult(
            session_id="critical_test",
            pipeline_id="test",
            execution_id="test", 
            step_id="test",
            output_path="test",
            severity=ValidationSeverity.CRITICAL,
            passed=False,
            quality_score=0.0
        )
        assert critical_result.has_critical_issues() is True
        assert critical_result.has_warnings() is False
        
        # Test warning result
        warning_result = ValidationResult(
            session_id="warning_test",
            pipeline_id="test",
            execution_id="test",
            step_id="test", 
            output_path="test",
            severity=ValidationSeverity.WARNING,
            passed=True,
            quality_score=75.0,
            total_violations=3
        )
        assert warning_result.has_critical_issues() is False
        assert warning_result.has_warnings() is True
        
        # Test clean result
        clean_result = ValidationResult(
            session_id="clean_test",
            pipeline_id="test",
            execution_id="test",
            step_id="test",
            output_path="test", 
            severity=ValidationSeverity.PASS,
            passed=True,
            quality_score=100.0
        )
        assert clean_result.has_critical_issues() is False
        assert clean_result.has_warnings() is False
    
    def test_stream_c_summary(self):
        """Test Stream C summary generation."""
        result = ValidationResult(
            session_id="stream_c_test",
            pipeline_id="test_pipeline",
            execution_id="test_execution", 
            step_id="test_step",
            output_path="/path/to/output",
            severity=ValidationSeverity.WARNING,
            passed=False,
            quality_score=72.5,
            total_violations=5,
            violations_by_severity={"warning": 3, "error": 2},
            violations_by_category={"content": 4, "format": 1},
            validation_duration_ms=250.0,
            rules_executed=10,
            rules_failed=1,
            recommendations=["Fix content issues", "Check format"]
        )
        
        # Get Stream C summary
        summary = result.get_summary_for_stream_c()
        
        # Verify summary structure for Stream C analytics
        assert summary["validation_id"] == "stream_c_test"
        assert summary["pipeline_id"] == "test_pipeline"
        assert summary["execution_id"] == "test_execution"
        assert summary["step_id"] == "test_step"
        
        # Verify quality metrics
        quality_metrics = summary["quality_metrics"]
        assert quality_metrics["quality_score"] == 72.5
        assert quality_metrics["severity"] == "warning"
        assert quality_metrics["passed"] is False
        assert quality_metrics["total_violations"] == 5
        assert quality_metrics["violation_breakdown"] == {"warning": 3, "error": 2}
        assert quality_metrics["category_breakdown"] == {"content": 4, "format": 1}
        
        # Verify performance metrics
        perf_metrics = summary["performance_metrics"]
        assert perf_metrics["validation_duration_ms"] == 250.0
        assert perf_metrics["rules_executed"] == 10
        assert perf_metrics["rules_failed"] == 1
        assert perf_metrics["success_rate"] == 90.0  # (10-1)/10 * 100
        
        # Verify other fields
        assert summary["recommendations"] == ["Fix content issues", "Check format"]
        assert "timestamp" in summary


def test_create_quality_control_manager():
    """Test the factory function for creating quality control manager."""
    from src.orchestrator.quality.validation.integration import create_quality_control_manager
    
    # Test default creation
    manager = create_quality_control_manager()
    assert isinstance(manager, QualityControlManager)
    assert manager.monitor is not None  # Should have monitoring enabled by default
    
    # Test with monitoring disabled
    manager = create_quality_control_manager(enable_monitoring=False)
    assert isinstance(manager, QualityControlManager)
    assert manager.monitor is None


if __name__ == "__main__":
    pytest.main([__file__])