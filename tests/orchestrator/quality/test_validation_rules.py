"""
Tests for validation rules system.

These tests verify the functionality of individual validation rules and the
rule registry system using real validation scenarios.
"""

import pytest
import tempfile
import json
import yaml
from pathlib import Path
from unittest.mock import Mock, patch

from src.orchestrator.quality.validation.rules import (
    ValidationRule, QualityRule, RuleViolation, ValidationContext,
    RuleSeverity, RuleCategory, RuleRegistry,
    FileSizeRule, ContentFormatRule, ContentQualityRule, PerformanceRule
)
from src.orchestrator.execution.state import ExecutionContext, ExecutionMetrics


class TestValidationRules:
    """Test suite for individual validation rules."""
    
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
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_file(self, filename: str, content: str, size_mb: float = None) -> str:
        """Create a test file with specified content."""
        file_path = self.temp_path / filename
        
        if size_mb:
            # Create file of specific size
            with open(file_path, 'wb') as f:
                f.write(b'x' * int(size_mb * 1024 * 1024))
        else:
            # Create file with content
            with open(file_path, 'w') as f:
                f.write(content)
        
        return str(file_path)
    
    def create_validation_context(self, output_path: str) -> ValidationContext:
        """Create validation context for testing."""
        return ValidationContext(
            execution_context=self.execution_context,
            output_path=output_path,
            output_content=None,
            output_metadata={},
            pipeline_config={}
        )
    
    def test_file_size_rule_pass(self):
        """Test file size rule with file under limit."""
        # Create small test file
        test_file = self.create_test_file("small.txt", "Hello, world!", size_mb=0.001)  # 1KB
        
        # Create and configure rule
        rule = FileSizeRule()
        rule.configure(max_size_mb=1.0)  # 1MB limit
        
        # Create validation context
        context = self.create_validation_context(test_file)
        
        # Run validation
        violations = rule.validate(context)
        
        # Assert no violations
        assert len(violations) == 0
    
    def test_file_size_rule_fail(self):
        """Test file size rule with file over limit."""
        # Create large test file
        test_file = self.create_test_file("large.txt", "", size_mb=2.0)  # 2MB
        
        # Create and configure rule
        rule = FileSizeRule()
        rule.configure(max_size_mb=1.0)  # 1MB limit
        
        # Create validation context
        context = self.create_validation_context(test_file)
        
        # Run validation
        violations = rule.validate(context)
        
        # Assert violation detected
        assert len(violations) == 1
        assert violations[0].rule_id == "file_size_limit"
        assert violations[0].severity == RuleSeverity.WARNING
        assert "exceeds limit" in violations[0].message
        assert violations[0].file_path == test_file
    
    def test_content_format_rule_json_valid(self):
        """Test content format rule with valid JSON."""
        # Create valid JSON file
        test_data = {"name": "test", "value": 123, "active": True}
        test_file = self.create_test_file("test.json", json.dumps(test_data, indent=2))
        
        # Create and configure rule
        rule = ContentFormatRule()
        rule.configure(expected_format="json", required_fields=["name", "value"])
        
        # Create validation context
        context = self.create_validation_context(test_file)
        
        # Run validation
        violations = rule.validate(context)
        
        # Assert no violations
        assert len(violations) == 0
    
    def test_content_format_rule_json_invalid(self):
        """Test content format rule with invalid JSON."""
        # Create invalid JSON file
        test_file = self.create_test_file("invalid.json", '{"name": "test", "value":}')  # Missing value
        
        # Create and configure rule
        rule = ContentFormatRule()
        rule.configure(expected_format="json")
        
        # Create validation context
        context = self.create_validation_context(test_file)
        
        # Run validation
        violations = rule.validate(context)
        
        # Assert violation detected
        assert len(violations) == 1
        assert violations[0].rule_id == "content_format"
        assert violations[0].severity == RuleSeverity.ERROR
        assert "Invalid JSON format" in violations[0].message
    
    def test_content_format_rule_json_missing_fields(self):
        """Test content format rule with JSON missing required fields."""
        # Create JSON file missing required fields
        test_data = {"name": "test"}  # Missing "value" field
        test_file = self.create_test_file("incomplete.json", json.dumps(test_data))
        
        # Create and configure rule
        rule = ContentFormatRule()
        rule.configure(expected_format="json", required_fields=["name", "value", "active"])
        
        # Create validation context
        context = self.create_validation_context(test_file)
        
        # Run validation
        violations = rule.validate(context)
        
        # Assert violations for missing fields
        assert len(violations) == 2  # Missing "value" and "active"
        violation_messages = [v.message for v in violations]
        assert any("Required field 'value' missing" in msg for msg in violation_messages)
        assert any("Required field 'active' missing" in msg for msg in violation_messages)
    
    def test_content_format_rule_yaml_valid(self):
        """Test content format rule with valid YAML."""
        # Create valid YAML file
        test_data = {"config": {"debug": True, "timeout": 30}}
        test_file = self.create_test_file("config.yaml", yaml.dump(test_data))
        
        # Create and configure rule
        rule = ContentFormatRule()
        rule.configure(expected_format="yaml", required_fields=["config"])
        
        # Create validation context
        context = self.create_validation_context(test_file)
        
        # Run validation
        violations = rule.validate(context)
        
        # Assert no violations
        assert len(violations) == 0
    
    def test_content_quality_rule_prohibited_patterns(self):
        """Test content quality rule with prohibited patterns."""
        # Create content with prohibited patterns
        content = '''
        def process_data():
            password = "secret123"  # TODO: Move to config
            api_key = "sk-1234567890"  # FIXME: Use environment variable
            print("Processing...")  # XXX: Remove debug output
            return True
        '''
        test_file = self.create_test_file("code.py", content)
        
        # Create and configure rule
        rule = ContentQualityRule()
        rule.configure(
            prohibited_patterns=[
                r"TODO:",
                r"FIXME:",
                r"XXX:",
                r"password\s*=\s*['\"][^'\"]*['\"]",
                r"api_key\s*=\s*['\"][^'\"]*['\"]"
            ]
        )
        
        # Create validation context
        context = self.create_validation_context(test_file)
        
        # Run validation
        violations = rule.validate(context)
        
        # Assert violations detected
        assert len(violations) == 5  # All 5 prohibited patterns found
        violation_messages = [v.message for v in violations]
        assert any("TODO:" in msg for msg in violation_messages)
        assert any("FIXME:" in msg for msg in violation_messages)
        assert any("XXX:" in msg for msg in violation_messages)
        assert any("password" in msg for msg in violation_messages)
        assert any("api_key" in msg for msg in violation_messages)
    
    def test_content_quality_rule_length_checks(self):
        """Test content quality rule with length constraints."""
        # Test minimum length violation
        short_content = "Hi"  # Too short
        short_file = self.create_test_file("short.txt", short_content)
        
        # Test maximum length violation  
        long_content = "x" * 1000  # Too long
        long_file = self.create_test_file("long.txt", long_content)
        
        # Create and configure rule
        rule = ContentQualityRule()
        rule.configure(min_length=10, max_length=500)
        
        # Test short file
        short_context = self.create_validation_context(short_file)
        short_violations = rule.validate(short_context)
        assert len(short_violations) == 1
        assert "below minimum" in short_violations[0].message
        
        # Test long file
        long_context = self.create_validation_context(long_file)
        long_violations = rule.validate(long_context)
        assert len(long_violations) == 1
        assert "exceeds maximum" in long_violations[0].message
    
    def test_performance_rule_execution_time(self):
        """Test performance rule with execution time limits."""
        from datetime import timedelta
        
        # Create test file
        test_file = self.create_test_file("result.txt", "Processing complete")
        
        # Configure metrics with long execution time
        self.metrics.duration = timedelta(seconds=600)  # 10 minutes
        
        # Create and configure rule
        rule = PerformanceRule()
        rule.configure(max_execution_time_seconds=300)  # 5 minute limit
        
        # Create validation context
        context = self.create_validation_context(test_file)
        
        # Run validation
        violations = rule.validate(context)
        
        # Assert violation detected
        assert len(violations) == 1
        assert violations[0].rule_id == "performance_metrics"
        assert "execution time" in violations[0].message.lower()
        assert "exceeds limit" in violations[0].message
    
    def test_performance_rule_memory_usage(self):
        """Test performance rule with memory usage limits."""
        # Create test file
        test_file = self.create_test_file("result.txt", "Processing complete")
        
        # Configure metrics with high memory usage
        self.metrics.memory_peak_mb = 2048.0  # 2GB
        
        # Create and configure rule
        rule = PerformanceRule()
        rule.configure(max_memory_mb=1024)  # 1GB limit
        
        # Create validation context
        context = self.create_validation_context(test_file)
        
        # Run validation
        violations = rule.validate(context)
        
        # Assert violation detected
        assert len(violations) == 1
        assert violations[0].rule_id == "performance_metrics"
        assert "memory usage" in violations[0].message.lower()
        assert "exceeds limit" in violations[0].message


class TestRuleRegistry:
    """Test suite for rule registry functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.registry = RuleRegistry()
    
    def test_builtin_rules_registered(self):
        """Test that built-in rules are automatically registered."""
        rules = self.registry.list_rules()
        
        # Check that built-in rules are present
        expected_rules = ["file_size_limit", "content_format", "content_quality", "performance_metrics"]
        for rule_id in expected_rules:
            assert rule_id in rules
            assert rules[rule_id]["enabled"] is True
    
    def test_register_custom_rule(self):
        """Test registering a custom validation rule."""
        class CustomRule(ValidationRule):
            def __init__(self):
                super().__init__(
                    rule_id="custom_test_rule",
                    name="Custom Test Rule",
                    description="Test rule for validation",
                    severity=RuleSeverity.INFO,
                    category=RuleCategory.CONTENT
                )
            
            def validate(self, context):
                return []  # No violations
        
        # Register custom rule
        custom_rule = CustomRule()
        self.registry.register_rule(custom_rule)
        
        # Verify rule is registered
        assert "custom_test_rule" in self.registry.rules
        assert self.registry.get_rule("custom_test_rule") == custom_rule
        
        # Verify rule appears in category
        content_rules = self.registry.get_rules_by_category(RuleCategory.CONTENT)
        assert any(rule.rule_id == "custom_test_rule" for rule in content_rules)
    
    def test_unregister_rule(self):
        """Test unregistering a rule."""
        # Unregister built-in rule
        success = self.registry.unregister_rule("file_size_limit")
        assert success is True
        
        # Verify rule is removed
        assert "file_size_limit" not in self.registry.rules
        assert self.registry.get_rule("file_size_limit") is None
        
        # Test unregistering non-existent rule
        success = self.registry.unregister_rule("non_existent_rule")
        assert success is False
    
    def test_enable_disable_rule(self):
        """Test enabling and disabling rules."""
        rule_id = "content_format"
        
        # Test disable
        success = self.registry.disable_rule(rule_id)
        assert success is True
        assert self.registry.get_rule(rule_id).enabled is False
        
        # Test enable
        success = self.registry.enable_rule(rule_id)
        assert success is True
        assert self.registry.get_rule(rule_id).enabled is True
        
        # Test non-existent rule
        assert self.registry.enable_rule("non_existent") is False
        assert self.registry.disable_rule("non_existent") is False
    
    def test_configure_rule(self):
        """Test configuring rule parameters."""
        rule_id = "file_size_limit"
        
        # Configure rule
        success = self.registry.configure_rule(rule_id, max_size_mb=500)
        assert success is True
        
        # Verify configuration
        rule = self.registry.get_rule(rule_id)
        assert rule.max_size_mb == 500
        
        # Test non-existent rule
        success = self.registry.configure_rule("non_existent", param=123)
        assert success is False
    
    def test_get_enabled_rules(self):
        """Test getting only enabled rules."""
        # All rules should be enabled by default
        enabled_rules = self.registry.get_enabled_rules()
        all_rules = list(self.registry.rules.values())
        assert len(enabled_rules) == len(all_rules)
        
        # Disable one rule
        self.registry.disable_rule("file_size_limit")
        enabled_rules = self.registry.get_enabled_rules()
        assert len(enabled_rules) == len(all_rules) - 1
        
        # Verify disabled rule is not included
        enabled_rule_ids = [rule.rule_id for rule in enabled_rules]
        assert "file_size_limit" not in enabled_rule_ids
    
    def test_get_rules_by_category(self):
        """Test getting rules by category."""
        # Get format rules
        format_rules = self.registry.get_rules_by_category(RuleCategory.FORMAT)
        format_rule_ids = [rule.rule_id for rule in format_rules]
        assert "content_format" in format_rule_ids
        
        # Get performance rules
        performance_rules = self.registry.get_rules_by_category(RuleCategory.PERFORMANCE)
        performance_rule_ids = [rule.rule_id for rule in performance_rules]
        assert "performance_metrics" in performance_rule_ids
        
        # Test category with no rules
        security_rules = self.registry.get_rules_by_category(RuleCategory.SECURITY)
        assert len(security_rules) == 0
    
    def test_load_rules_from_config(self):
        """Test loading rule configuration from YAML file."""
        # Create test configuration file
        config_data = {
            'rules': [
                {
                    'rule_id': 'file_size_limit',
                    'enabled': False,
                    'config': {
                        'max_size_mb': 200
                    }
                },
                {
                    'rule_id': 'content_quality',
                    'enabled': True,
                    'config': {
                        'min_length': 50,
                        'max_length': 10000
                    }
                }
            ]
        }
        
        # Write config to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name
        
        try:
            # Load configuration
            self.registry.load_rules_from_config(config_file)
            
            # Verify file_size_limit rule was configured and disabled
            file_size_rule = self.registry.get_rule('file_size_limit')
            assert file_size_rule.enabled is False
            assert file_size_rule.max_size_mb == 200
            
            # Verify content_quality rule was configured
            content_rule = self.registry.get_rule('content_quality')
            assert content_rule.enabled is True
            assert content_rule.min_length == 50
            assert content_rule.max_length == 10000
            
        finally:
            # Clean up temporary file
            Path(config_file).unlink()
    
    def test_load_invalid_config(self):
        """Test error handling when loading invalid configuration."""
        # Create invalid configuration file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_file = f.name
        
        try:
            # Attempt to load invalid configuration
            with pytest.raises(RuntimeError, match="Failed to load rules configuration"):
                self.registry.load_rules_from_config(config_file)
                
        finally:
            # Clean up temporary file
            Path(config_file).unlink()


class TestRuleViolation:
    """Test suite for rule violation functionality."""
    
    def test_rule_violation_creation(self):
        """Test creating rule violations."""
        violation = RuleViolation(
            rule_id="test_rule",
            message="Test violation message",
            severity=RuleSeverity.ERROR,
            category=RuleCategory.CONTENT,
            file_path="/path/to/file.txt",
            line_number=42,
            column_number=15,
            metadata={"extra_info": "test"}
        )
        
        assert violation.rule_id == "test_rule"
        assert violation.message == "Test violation message"
        assert violation.severity == RuleSeverity.ERROR
        assert violation.category == RuleCategory.CONTENT
        assert violation.file_path == "/path/to/file.txt"
        assert violation.line_number == 42
        assert violation.column_number == 15
        assert violation.metadata["extra_info"] == "test"
    
    def test_rule_violation_minimal(self):
        """Test creating rule violation with minimal information."""
        violation = RuleViolation(
            rule_id="minimal_rule",
            message="Minimal violation",
            severity=RuleSeverity.WARNING,
            category=RuleCategory.STRUCTURE
        )
        
        assert violation.rule_id == "minimal_rule"
        assert violation.message == "Minimal violation"
        assert violation.severity == RuleSeverity.WARNING
        assert violation.category == RuleCategory.STRUCTURE
        assert violation.file_path is None
        assert violation.line_number is None
        assert violation.column_number is None
        assert violation.metadata == {}


if __name__ == "__main__":
    pytest.main([__file__])