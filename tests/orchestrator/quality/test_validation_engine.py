"""
Tests for validation engine functionality.

These tests verify the core validation engine that orchestrates rule execution,
manages validation sessions, and provides comprehensive quality assessment.
"""

import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

from src.orchestrator.quality.validation.engine import (
    ValidationEngine, ValidationSession, RuleExecutionResult, RuleExecutionContext
)
from src.orchestrator.quality.validation.rules import (
    ValidationRule, RuleRegistry, ValidationContext, RuleViolation,
    RuleSeverity, RuleCategory, FileSizeRule, ContentQualityRule
)
from src.orchestrator.execution.state import ExecutionContext, ExecutionMetrics


class MockRule(ValidationRule):
    """Mock validation rule for testing."""
    
    def __init__(
        self,
        rule_id: str = "mock_rule",
        violation_count: int = 0,
        should_fail: bool = False,
        execution_delay: float = 0.0
    ):
        super().__init__(
            rule_id=rule_id,
            name="Mock Rule",
            description="Mock rule for testing",
            severity=RuleSeverity.WARNING,
            category=RuleCategory.CONTENT
        )
        self.violation_count = violation_count
        self.should_fail = should_fail
        self.execution_delay = execution_delay
        self.validation_calls = 0
    
    def validate(self, context: ValidationContext):
        """Mock validation that can be configured to return specific results."""
        self.validation_calls += 1
        
        if self.execution_delay > 0:
            time.sleep(self.execution_delay)
        
        if self.should_fail:
            raise RuntimeError("Mock rule failure")
        
        # Generate mock violations
        violations = []
        for i in range(self.violation_count):
            violations.append(RuleViolation(
                rule_id=self.rule_id,
                message=f"Mock violation {i + 1}",
                severity=self.severity,
                category=self.category,
                file_path=context.output_path,
                metadata={"mock": True, "violation_index": i}
            ))
        
        return violations


class TestValidationEngine:
    """Test suite for validation engine functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        # Create mock execution context
        self.execution_context = Mock(spec=ExecutionContext)
        self.execution_context.execution_id = "test_execution_001"
        self.execution_context.pipeline_id = "test_pipeline"
        self.execution_context.current_step_id = "test_step"
        
        # Create test metrics
        self.metrics = Mock(spec=ExecutionMetrics)
        self.metrics.duration = None
        self.metrics.memory_peak_mb = 100.0
        self.execution_context.metrics = self.metrics
        
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create test file
        self.test_file = self.temp_path / "test_output.txt"
        self.test_file.write_text("Test output content")
        
        # Create rule registry and engine
        self.rule_registry = RuleRegistry()
        self.validation_engine = ValidationEngine(rule_registry=self.rule_registry)
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_validation_engine_initialization(self):
        """Test validation engine initialization."""
        # Test with default registry
        engine = ValidationEngine()
        assert engine.rule_registry is not None
        assert len(engine.rule_registry.rules) > 0  # Should have built-in rules
        assert engine.max_workers == 4
        assert engine.timeout_seconds == 60.0
        
        # Test with custom parameters
        custom_registry = RuleRegistry()
        engine = ValidationEngine(
            rule_registry=custom_registry,
            max_workers=8,
            timeout_seconds=120.0
        )
        assert engine.rule_registry == custom_registry
        assert engine.max_workers == 8
        assert engine.timeout_seconds == 120.0
    
    def test_validate_output_basic(self):
        """Test basic output validation."""
        session = self.validation_engine.validate_output(
            execution_context=self.execution_context,
            output_path=str(self.test_file),
            parallel=False  # Use sequential for predictable testing
        )
        
        # Verify session structure
        assert session.session_id.startswith("validation_test_execution_001")
        assert session.total_rules > 0
        assert session.rules_executed == session.total_rules
        assert session.end_time is not None
        assert session.duration_ms > 0
        
        # Should have some built-in rules executed
        assert len(session.rule_results) > 0
        
        # All rules should have completed successfully (with our simple test file)
        for result in session.rule_results:
            assert result.rule_id is not None
            assert result.rule_name is not None
            assert result.execution_time_ms >= 0
    
    def test_validate_output_with_violations(self):
        """Test validation with rules that generate violations."""
        # Add a mock rule that generates violations
        mock_rule = MockRule(rule_id="violation_rule", violation_count=3)
        self.rule_registry.register_rule(mock_rule)
        
        session = self.validation_engine.validate_output(
            execution_context=self.execution_context,
            output_path=str(self.test_file),
            parallel=False
        )
        
        # Find the mock rule result
        mock_result = next(
            (r for r in session.rule_results if r.rule_id == "violation_rule"),
            None
        )
        assert mock_result is not None
        assert len(mock_result.violations) == 3
        assert mock_result.success is True  # Rule executed successfully
        
        # Verify session violation counts
        assert session.total_violations >= 3
        assert session.violations_by_severity[RuleSeverity.WARNING] >= 3
    
    def test_validate_output_with_rule_failure(self):
        """Test validation with a rule that fails during execution."""
        # Add a mock rule that fails
        failing_rule = MockRule(rule_id="failing_rule", should_fail=True)
        self.rule_registry.register_rule(failing_rule)
        
        session = self.validation_engine.validate_output(
            execution_context=self.execution_context,
            output_path=str(self.test_file),
            parallel=False
        )
        
        # Find the failing rule result
        failing_result = next(
            (r for r in session.rule_results if r.rule_id == "failing_rule"),
            None
        )
        assert failing_result is not None
        assert failing_result.success is False
        assert failing_result.error_message is not None
        assert "Mock rule failure" in failing_result.error_message
        
        # Verify session failure count
        assert session.rules_failed >= 1
    
    def test_validate_output_with_rule_filters(self):
        """Test validation with specific rule filters."""
        # Add mock rules
        rule1 = MockRule(rule_id="rule_1", violation_count=1)
        rule2 = MockRule(rule_id="rule_2", violation_count=2)
        self.rule_registry.register_rule(rule1)
        self.rule_registry.register_rule(rule2)
        
        # Validate with rule filter
        session = self.validation_engine.validate_output(
            execution_context=self.execution_context,
            output_path=str(self.test_file),
            rule_filters=["rule_1", "file_size_limit"],  # Only specific rules
            parallel=False
        )
        
        # Verify only filtered rules were executed
        executed_rule_ids = [r.rule_id for r in session.rule_results]
        assert "rule_1" in executed_rule_ids
        assert "file_size_limit" in executed_rule_ids
        assert "rule_2" not in executed_rule_ids  # Should be filtered out
        
        # Verify rule_1 was called
        assert rule1.validation_calls == 1
        assert rule2.validation_calls == 0  # Should not have been called
    
    def test_validate_output_with_category_filters(self):
        """Test validation with category filters."""
        # Add mock rules in different categories
        content_rule = MockRule(rule_id="content_rule", violation_count=1)
        content_rule.category = RuleCategory.CONTENT
        
        format_rule = MockRule(rule_id="format_rule", violation_count=1)
        format_rule.category = RuleCategory.FORMAT
        
        self.rule_registry.register_rule(content_rule)
        self.rule_registry.register_rule(format_rule)
        
        # Validate with category filter
        session = self.validation_engine.validate_output(
            execution_context=self.execution_context,
            output_path=str(self.test_file),
            category_filters=[RuleCategory.CONTENT],
            parallel=False
        )
        
        # Verify only content rules were executed
        executed_rule_ids = [r.rule_id for r in session.rule_results]
        content_rules_executed = [r for r in session.rule_results 
                                 if r.rule_id in ["content_rule", "content_quality"]]
        assert len(content_rules_executed) > 0
        
        # Format rules should not be executed (except built-ins)
        format_rules_executed = [r for r in session.rule_results if r.rule_id == "format_rule"]
        assert len(format_rules_executed) == 0
        
        # Verify calls
        assert content_rule.validation_calls == 1
        assert format_rule.validation_calls == 0
    
    def test_parallel_execution(self):
        """Test parallel rule execution."""
        # Add multiple mock rules with execution delays
        for i in range(5):
            rule = MockRule(
                rule_id=f"delayed_rule_{i}",
                violation_count=1,
                execution_delay=0.1  # 100ms delay
            )
            self.rule_registry.register_rule(rule)
        
        # Measure execution time for parallel execution
        start_time = time.time()
        session = self.validation_engine.validate_output(
            execution_context=self.execution_context,
            output_path=str(self.test_file),
            parallel=True
        )
        parallel_time = time.time() - start_time
        
        # Verify rules were executed
        assert session.rules_executed > 0
        
        # Parallel execution should be faster than sequential
        # (though we can't test this reliably in unit tests due to threading overhead)
        assert parallel_time < 2.0  # Should complete within 2 seconds
    
    def test_validation_hooks(self):
        """Test validation execution hooks."""
        hook_calls = {
            'before_validation': 0,
            'after_validation': 0,
            'before_rule': 0,
            'after_rule': 0,
            'on_violation': 0
        }
        
        def track_hook(hook_type):
            def hook_callback(context, **kwargs):
                hook_calls[hook_type] += 1
            return hook_callback
        
        # Register hooks
        for hook_type in hook_calls:
            self.validation_engine.add_hook(hook_type, track_hook(hook_type))
        
        # Add a rule with violations
        violation_rule = MockRule(rule_id="hook_test_rule", violation_count=2)
        self.rule_registry.register_rule(violation_rule)
        
        # Run validation
        session = self.validation_engine.validate_output(
            execution_context=self.execution_context,
            output_path=str(self.test_file),
            parallel=False
        )
        
        # Verify hooks were called
        assert hook_calls['before_validation'] == 1
        assert hook_calls['after_validation'] == 1
        assert hook_calls['before_rule'] > 0  # Called for each rule
        assert hook_calls['after_rule'] > 0   # Called for each rule
        assert hook_calls['on_violation'] >= 2  # Called for each violation
    
    def test_get_validation_summary(self):
        """Test validation summary generation."""
        # Add mock rules with different violation types
        error_rule = MockRule(rule_id="error_rule", violation_count=2)
        error_rule.severity = RuleSeverity.ERROR
        
        warning_rule = MockRule(rule_id="warning_rule", violation_count=3)
        warning_rule.severity = RuleSeverity.WARNING
        
        self.rule_registry.register_rule(error_rule)
        self.rule_registry.register_rule(warning_rule)
        
        # Run validation
        session = self.validation_engine.validate_output(
            execution_context=self.execution_context,
            output_path=str(self.test_file),
            parallel=False
        )
        
        # Get summary
        summary = self.validation_engine.get_validation_summary(session)
        
        # Verify summary structure
        assert summary["session_id"] == session.session_id
        assert summary["duration_ms"] == session.duration_ms
        assert summary["rules_executed"] == session.rules_executed
        assert summary["rules_failed"] == session.rules_failed
        assert summary["total_violations"] >= 5  # At least from our mock rules
        
        # Verify severity breakdown
        severity_dist = summary["violations_by_severity"]
        assert severity_dist["error"] >= 2
        assert severity_dist["warning"] >= 3
        
        # Verify quality score
        assert 0 <= summary["quality_score"] <= 100
        
        # Verify recommendations
        assert isinstance(summary["recommendations"], list)
    
    def test_quality_score_calculation(self):
        """Test quality score calculation logic."""
        # Test with no violations (perfect score)
        clean_session = ValidationSession(
            session_id="clean_test",
            start_time=time.time()
        )
        clean_session.finalize()
        clean_score = self.validation_engine._calculate_quality_score(clean_session)
        assert clean_score == 100.0
        
        # Test with violations
        violation_session = ValidationSession(
            session_id="violation_test",
            start_time=time.time()
        )
        
        # Simulate violations by severity
        violation_session.violations_by_severity[RuleSeverity.CRITICAL] = 1
        violation_session.violations_by_severity[RuleSeverity.ERROR] = 2
        violation_session.violations_by_severity[RuleSeverity.WARNING] = 3
        violation_session.violations_by_severity[RuleSeverity.INFO] = 4
        violation_session.total_violations = 10
        
        violation_session.finalize()
        violation_score = self.validation_engine._calculate_quality_score(violation_session)
        
        # Should be significantly lower than perfect score
        assert violation_score < 100.0
        assert violation_score >= 0.0
        
        # Critical violations should impact score more than warnings
        assert violation_score < 80.0  # Should be substantially reduced
    
    def test_export_results(self):
        """Test exporting validation results to file."""
        # Add a mock rule for predictable results
        mock_rule = MockRule(rule_id="export_test_rule", violation_count=1)
        self.rule_registry.register_rule(mock_rule)
        
        # Run validation
        session = self.validation_engine.validate_output(
            execution_context=self.execution_context,
            output_path=str(self.test_file),
            parallel=False
        )
        
        # Export results
        export_file = self.temp_path / "validation_results.json"
        self.validation_engine.export_results(session, export_file, format="json")
        
        # Verify file was created
        assert export_file.exists()
        
        # Verify file content
        import json
        with open(export_file, 'r') as f:
            exported_data = json.load(f)
        
        assert "summary" in exported_data
        assert "rule_results" in exported_data
        
        # Verify summary data
        summary = exported_data["summary"]
        assert summary["session_id"] == session.session_id
        assert summary["total_violations"] >= 1
        
        # Verify rule results data
        rule_results = exported_data["rule_results"]
        assert len(rule_results) > 0
        
        # Find our mock rule result
        mock_result = next(
            (r for r in rule_results if r["rule_id"] == "export_test_rule"),
            None
        )
        assert mock_result is not None
        assert len(mock_result["violations"]) == 1
    
    def test_unsupported_export_format(self):
        """Test error handling for unsupported export format."""
        # Create minimal session
        session = ValidationSession(
            session_id="test_session",
            start_time=time.time()
        )
        session.finalize()
        
        # Attempt to export with unsupported format
        export_file = self.temp_path / "results.xml"
        with pytest.raises(ValueError, match="Unsupported export format"):
            self.validation_engine.export_results(session, export_file, format="xml")


class TestValidationSession:
    """Test suite for validation session functionality."""
    
    def test_validation_session_creation(self):
        """Test creating a validation session."""
        start_time = time.time()
        session = ValidationSession(
            session_id="test_session_001",
            start_time=start_time,
            total_rules=5
        )
        
        assert session.session_id == "test_session_001"
        assert session.start_time == start_time
        assert session.total_rules == 5
        assert session.rules_executed == 0
        assert session.rules_failed == 0
        assert session.total_violations == 0
        assert session.end_time is None
    
    def test_validation_session_add_result(self):
        """Test adding results to a validation session."""
        session = ValidationSession(
            session_id="test_session",
            start_time=time.time()
        )
        
        # Create mock violations
        violations = [
            RuleViolation(
                rule_id="test_rule",
                message="Test violation",
                severity=RuleSeverity.ERROR,
                category=RuleCategory.CONTENT
            )
        ]
        
        # Create and add result
        result = RuleExecutionResult(
            rule_id="test_rule",
            rule_name="Test Rule",
            success=True,
            violations=violations,
            execution_time_ms=100.0
        )
        
        session.add_result(result)
        
        # Verify session was updated
        assert session.rules_executed == 1
        assert session.rules_failed == 0
        assert session.total_violations == 1
        assert session.violations_by_severity[RuleSeverity.ERROR] == 1
        assert len(session.rule_results) == 1
        assert session.rule_results[0] == result
    
    def test_validation_session_failed_result(self):
        """Test adding a failed result to validation session."""
        session = ValidationSession(
            session_id="test_session",
            start_time=time.time()
        )
        
        # Create failed result
        result = RuleExecutionResult(
            rule_id="failing_rule",
            rule_name="Failing Rule",
            success=False,
            error_message="Rule execution failed"
        )
        
        session.add_result(result)
        
        # Verify session counts failure
        assert session.rules_executed == 1
        assert session.rules_failed == 1
        assert session.total_violations == 0  # Failed rules don't generate violations
    
    def test_validation_session_success_rate(self):
        """Test success rate calculation."""
        session = ValidationSession(
            session_id="test_session",
            start_time=time.time()
        )
        
        # Initially no executions
        assert session.success_rate() == 0.0
        
        # Add successful result
        successful_result = RuleExecutionResult(
            rule_id="success_rule",
            rule_name="Successful Rule",
            success=True
        )
        session.add_result(successful_result)
        assert session.success_rate() == 100.0
        
        # Add failed result
        failed_result = RuleExecutionResult(
            rule_id="failed_rule", 
            rule_name="Failed Rule",
            success=False
        )
        session.add_result(failed_result)
        assert session.success_rate() == 50.0
    
    def test_validation_session_duration(self):
        """Test session duration calculation."""
        start_time = time.time()
        session = ValidationSession(
            session_id="test_session",
            start_time=start_time
        )
        
        # Before finalization, duration should be current
        duration_before = session.duration_ms
        assert duration_before > 0
        
        # Simulate some processing time
        time.sleep(0.01)
        
        # Finalize session
        session.finalize()
        
        # After finalization, duration should be fixed
        duration_after = session.duration_ms
        assert duration_after > duration_before
        assert session.end_time is not None
        
        # Duration shouldn't change after finalization
        time.sleep(0.01)
        assert session.duration_ms == duration_after


class TestRuleExecutionContext:
    """Test suite for rule execution context."""
    
    def setup_method(self):
        """Set up test environment."""
        self.execution_context = Mock(spec=ExecutionContext)
        self.validation_context = Mock(spec=ValidationContext)
        self.session = ValidationSession("test_session", time.time())
        self.rule_registry = RuleRegistry()
        
        self.rule_context = RuleExecutionContext(
            validation_context=self.validation_context,
            session=self.session,
            rule_registry=self.rule_registry
        )
    
    def test_shared_state_management(self):
        """Test shared state functionality."""
        # Initially empty
        assert self.rule_context.get_shared_state("test_key") is None
        assert self.rule_context.get_shared_state("test_key", "default") == "default"
        
        # Set and retrieve values
        self.rule_context.set_shared_state("test_key", "test_value")
        assert self.rule_context.get_shared_state("test_key") == "test_value"
        
        # Set complex object
        complex_data = {"numbers": [1, 2, 3], "nested": {"key": "value"}}
        self.rule_context.set_shared_state("complex", complex_data)
        retrieved_data = self.rule_context.get_shared_state("complex")
        assert retrieved_data == complex_data
    
    def test_execution_metadata_logging(self):
        """Test execution metadata logging."""
        # Initially empty
        assert len(self.rule_context.execution_metadata) == 0
        
        # Log metadata for a rule
        metadata = {"execution_time": 100.5, "memory_usage": 50.2}
        self.rule_context.log_execution_metadata("test_rule", metadata)
        
        # Verify metadata was stored
        assert "test_rule" in self.rule_context.execution_metadata
        assert self.rule_context.execution_metadata["test_rule"] == metadata
        
        # Log metadata for another rule
        metadata2 = {"cache_hits": 15, "cache_misses": 3}
        self.rule_context.log_execution_metadata("cache_rule", metadata2)
        
        # Verify both sets of metadata
        assert len(self.rule_context.execution_metadata) == 2
        assert self.rule_context.execution_metadata["cache_rule"] == metadata2


if __name__ == "__main__":
    pytest.main([__file__])