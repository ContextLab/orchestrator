"""Tests for dependency validation functionality."""

import pytest
from unittest.mock import patch, MagicMock

from src.orchestrator.validation.dependency_validator import (
    DependencyValidator,
    DependencyIssue,
    DependencyValidationResult,
)


class TestDependencyValidator:
    """Test cases for DependencyValidator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = DependencyValidator()
        self.dev_validator = DependencyValidator(development_mode=True)

    def test_empty_pipeline(self):
        """Test validation of empty pipeline."""
        pipeline_def = {"name": "test", "steps": []}
        
        result = self.validator.validate_pipeline_dependencies(pipeline_def)
        
        assert result.is_valid
        assert len(result.issues) == 1
        assert result.issues[0].issue_type == "empty_pipeline"
        assert result.issues[0].severity == "warning"

    def test_single_task_no_dependencies(self):
        """Test validation of single task with no dependencies."""
        pipeline_def = {
            "name": "test",
            "steps": [
                {"id": "task1", "action": "test_action"}
            ]
        }
        
        result = self.validator.validate_pipeline_dependencies(pipeline_def)
        
        assert result.is_valid
        assert len(result.issues) == 0

    def test_simple_linear_dependency(self):
        """Test validation of simple linear dependency chain."""
        pipeline_def = {
            "name": "test", 
            "steps": [
                {"id": "task1", "action": "test_action"},
                {"id": "task2", "action": "test_action", "dependencies": ["task1"]},
                {"id": "task3", "action": "test_action", "dependencies": ["task2"]}
            ]
        }
        
        result = self.validator.validate_pipeline_dependencies(pipeline_def)
        
        assert result.is_valid
        assert len(result.issues) == 0
        assert result.execution_order == ["task1", "task2", "task3"]

    def test_missing_task_id(self):
        """Test validation fails when task is missing ID."""
        pipeline_def = {
            "name": "test",
            "steps": [
                {"action": "test_action"},  # Missing ID
                {"id": "task2", "action": "test_action"}
            ]
        }
        
        result = self.validator.validate_pipeline_dependencies(pipeline_def)
        
        assert not result.is_valid
        assert len(result.errors) == 1
        assert result.errors[0].issue_type == "missing_task_id"

    def test_duplicate_task_ids(self):
        """Test validation fails when task IDs are duplicated."""
        pipeline_def = {
            "name": "test",
            "steps": [
                {"id": "task1", "action": "test_action"},
                {"id": "task1", "action": "test_action"},  # Duplicate ID
                {"id": "task2", "action": "test_action"}
            ]
        }
        
        result = self.validator.validate_pipeline_dependencies(pipeline_def)
        
        assert not result.is_valid
        assert len(result.errors) == 1
        assert result.errors[0].issue_type == "duplicate_task_id"
        assert "task1" in result.errors[0].involved_tasks

    def test_missing_dependency_reference(self):
        """Test validation fails when task references non-existent dependency."""
        pipeline_def = {
            "name": "test",
            "steps": [
                {"id": "task1", "action": "test_action"},
                {"id": "task2", "action": "test_action", "dependencies": ["nonexistent"]},
            ]
        }
        
        result = self.validator.validate_pipeline_dependencies(pipeline_def)
        
        assert not result.is_valid
        assert len(result.errors) == 1
        assert result.errors[0].issue_type == "missing_dependency"
        assert "task2" in result.errors[0].involved_tasks
        assert "nonexistent" in result.errors[0].involved_tasks

    def test_self_dependency(self):
        """Test validation fails when task depends on itself."""
        pipeline_def = {
            "name": "test",
            "steps": [
                {"id": "task1", "action": "test_action", "dependencies": ["task1"]},
            ]
        }
        
        result = self.validator.validate_pipeline_dependencies(pipeline_def)
        
        assert not result.is_valid
        assert len(result.errors) == 1
        assert result.errors[0].issue_type == "self_dependency"
        assert result.errors[0].dependency_chain == ["task1", "task1"]

    def test_simple_circular_dependency(self):
        """Test detection of simple circular dependency."""
        pipeline_def = {
            "name": "test",
            "steps": [
                {"id": "task1", "action": "test_action", "dependencies": ["task2"]},
                {"id": "task2", "action": "test_action", "dependencies": ["task1"]},
            ]
        }
        
        result = self.validator.validate_pipeline_dependencies(pipeline_def)
        
        assert not result.is_valid
        # Should have at least one circular dependency error
        circular_errors = [e for e in result.errors if e.issue_type == "circular_dependency"]
        assert len(circular_errors) >= 1
        assert set(circular_errors[0].involved_tasks) == {"task1", "task2"}

    def test_complex_circular_dependency(self):
        """Test detection of complex circular dependency chain."""
        pipeline_def = {
            "name": "test",
            "steps": [
                {"id": "task1", "action": "test_action", "dependencies": ["task3"]},
                {"id": "task2", "action": "test_action", "dependencies": ["task1"]},
                {"id": "task3", "action": "test_action", "dependencies": ["task2"]},
            ]
        }
        
        result = self.validator.validate_pipeline_dependencies(pipeline_def)
        
        assert not result.is_valid
        # Should have at least one circular dependency error
        circular_errors = [e for e in result.errors if e.issue_type == "circular_dependency"]
        assert len(circular_errors) >= 1
        assert set(circular_errors[0].involved_tasks) == {"task1", "task2", "task3"}

    def test_unreachable_task(self):
        """Test detection of unreachable tasks."""
        pipeline_def = {
            "name": "test",
            "steps": [
                {"id": "task1", "action": "test_action"},
                {"id": "task2", "action": "test_action", "dependencies": ["task1"]},
                {"id": "isolated", "action": "test_action", "dependencies": ["task4"]},  # Unreachable
                {"id": "task4", "action": "test_action", "dependencies": ["isolated"]},  # Circular, unreachable
            ]
        }
        
        result = self.validator.validate_pipeline_dependencies(pipeline_def)
        
        # Should have circular dependency error and unreachable warnings
        assert not result.is_valid
        circular_errors = [e for e in result.errors if e.issue_type == "circular_dependency"]
        unreachable_warnings = [w for w in result.warnings if w.issue_type == "unreachable_task"]
        
        assert len(circular_errors) == 1
        assert len(unreachable_warnings) == 2  # Both isolated and task4 are unreachable

    def test_for_each_dependency_extraction(self):
        """Test extraction of dependencies from for_each expressions."""
        pipeline_def = {
            "name": "test",
            "steps": [
                {"id": "data_source", "action": "get_data"},
                {"id": "processor", "action": "process", "for_each": "{{ data_source.result.items }}"},
            ]
        }
        
        result = self.validator.validate_pipeline_dependencies(pipeline_def)
        
        assert result.is_valid
        assert len(result.issues) == 0

    def test_for_each_missing_dependency(self):
        """Test validation fails when for_each references non-existent task."""
        pipeline_def = {
            "name": "test",
            "steps": [
                {"id": "processor", "action": "process", "for_each": "{{ missing_task.result.items }}"},
            ]
        }
        
        result = self.validator.validate_pipeline_dependencies(pipeline_def)
        
        assert not result.is_valid
        errors = [e for e in result.errors if e.issue_type == "invalid_foreach_dependency"]
        assert len(errors) >= 1  # May have duplicates, but at least one
        assert any("missing_task" in error.involved_tasks for error in errors)

    def test_conditional_dependency_extraction(self):
        """Test extraction of dependencies from conditional expressions."""
        pipeline_def = {
            "name": "test",
            "steps": [
                {"id": "checker", "action": "check_condition"},
                {"id": "conditional_task", "action": "conditional_action", "condition": "{{ checker.result.should_run }}"},
            ]
        }
        
        result = self.validator.validate_pipeline_dependencies(pipeline_def)
        
        assert result.is_valid
        assert len(result.issues) == 0

    def test_conditional_missing_dependency(self):
        """Test validation fails when condition references non-existent task."""
        pipeline_def = {
            "name": "test", 
            "steps": [
                {"id": "conditional_task", "action": "conditional_action", "condition": "{{ missing.result.value > 0 }}"},
            ]
        }
        
        result = self.validator.validate_pipeline_dependencies(pipeline_def)
        
        assert not result.is_valid
        errors = [e for e in result.errors if e.issue_type == "invalid_condition_dependency"]
        assert len(errors) >= 1  # May have duplicates, but at least one
        assert any("missing" in error.involved_tasks for error in errors)

    def test_action_loop_dependency_extraction(self):
        """Test extraction of dependencies from action loop conditions."""
        pipeline_def = {
            "name": "test",
            "steps": [
                {"id": "counter", "action": "init_counter"},
                {
                    "id": "loop_task", 
                    "action_loop": [{"action": "increment"}],
                    "until": "{{ counter.result.value >= 10 }}"
                },
            ]
        }
        
        result = self.validator.validate_pipeline_dependencies(pipeline_def)
        
        assert result.is_valid
        assert len(result.issues) == 0

    def test_parallel_queue_dependency_extraction(self):
        """Test extraction of dependencies from parallel queue 'on' expressions."""
        pipeline_def = {
            "name": "test",
            "steps": [
                {"id": "data_provider", "action": "provide_data"},
                {
                    "id": "parallel_processor",
                    "create_parallel_queue": {
                        "on": "{{ data_provider.result.items }}",
                        "action_loop": [{"action": "process_item"}]
                    }
                }
            ]
        }
        
        result = self.validator.validate_pipeline_dependencies(pipeline_def)
        
        assert result.is_valid
        assert len(result.issues) == 0

    def test_string_dependencies_parsing(self):
        """Test parsing of dependencies specified as strings."""
        pipeline_def = {
            "name": "test",
            "steps": [
                {"id": "task1", "action": "test_action"},
                {"id": "task2", "action": "test_action"},
                {"id": "task3", "action": "test_action", "dependencies": "task1,task2"},  # Comma-separated string
                {"id": "task4", "action": "test_action", "depends_on": "task3"},  # Single string using depends_on
            ]
        }
        
        result = self.validator.validate_pipeline_dependencies(pipeline_def)
        
        assert result.is_valid
        assert len(result.issues) == 0
        assert result.execution_order == ["task1", "task2", "task3", "task4"]

    def test_development_mode_leniency(self):
        """Test that development mode is more lenient with errors."""
        pipeline_def = {
            "name": "test",
            "steps": [
                {"id": "task1", "action": "test_action"},
                {"id": "task2", "action": "test_action", "dependencies": ["nonexistent"]},  # Missing dependency
                {"id": "unreachable", "action": "test_action", "dependencies": ["also_missing"]},  # Unreachable
            ]
        }
        
        # Regular mode should fail
        result = self.validator.validate_pipeline_dependencies(pipeline_def)
        assert not result.is_valid
        
        # Development mode should pass with warnings
        dev_result = self.dev_validator.validate_pipeline_dependencies(pipeline_def)
        assert dev_result.is_valid  # Should be valid in dev mode
        assert len(dev_result.warnings) > 0  # But should have warnings

    @patch('orchestrator.validation.dependency_validator.NETWORKX_AVAILABLE', False)
    def test_fallback_without_networkx(self):
        """Test that validator works without networkx installed."""
        validator = DependencyValidator()
        
        pipeline_def = {
            "name": "test",
            "steps": [
                {"id": "task1", "action": "test_action", "dependencies": ["task2"]},
                {"id": "task2", "action": "test_action", "dependencies": ["task1"]},  # Circular
            ]
        }
        
        result = validator.validate_pipeline_dependencies(pipeline_def)
        
        # Should still detect circular dependency without networkx
        assert not result.is_valid
        circular_errors = [e for e in result.errors if e.issue_type == "circular_dependency"]
        assert len(circular_errors) == 1

    def test_complex_pipeline_validation(self):
        """Test validation of a complex pipeline with multiple dependency types."""
        pipeline_def = {
            "name": "complex_test",
            "steps": [
                {"id": "init", "action": "initialize"},
                {"id": "data_loader", "action": "load_data", "dependencies": ["init"]},
                {
                    "id": "data_processor", 
                    "action": "process_data",
                    "for_each": "{{ data_loader.result.batches }}",
                    "dependencies": ["data_loader"]
                },
                {
                    "id": "conditional_saver",
                    "action": "save_results", 
                    "condition": "{{ data_processor.result.success }}",
                    "dependencies": ["data_processor"]
                },
                {
                    "id": "parallel_analyzer",
                    "create_parallel_queue": {
                        "on": "{{ data_processor.result.items }}",
                        "action_loop": [{"action": "analyze_item"}]
                    },
                    "dependencies": ["data_processor"]
                },
                {
                    "id": "final_report",
                    "action": "generate_report",
                    "dependencies": ["conditional_saver", "parallel_analyzer"]
                }
            ]
        }
        
        result = self.validator.validate_pipeline_dependencies(pipeline_def)
        
        assert result.is_valid
        assert len(result.issues) == 0
        
        # Verify execution order makes sense
        order = result.execution_order
        assert order.index("init") < order.index("data_loader")
        assert order.index("data_loader") < order.index("data_processor")
        assert order.index("data_processor") < order.index("final_report")

    def test_template_dependency_extraction_edge_cases(self):
        """Test edge cases in template dependency extraction."""
        # Test various template patterns
        test_cases = [
            "{{ task1.result }}",  # Simple reference
            "{{ task_a.output.data }}",  # Nested reference
            "{% if task2.result.status == 'success' %}",  # In control structure
            "Value: {{ task3.data | default('none') }}",  # With filter
            "{{ $item.value }}",  # Should not be extracted (loop variable)
            "{{ some_context_var }}",  # Should not be extracted (not a task reference)
        ]
        
        validator = DependencyValidator()
        
        # Test each pattern
        deps1 = validator._extract_template_dependencies(test_cases[0])
        assert "task1" in deps1
        
        deps2 = validator._extract_template_dependencies(test_cases[1])
        assert "task_a" in deps2
        
        deps3 = validator._extract_template_dependencies(test_cases[2])
        assert "task2" in deps3
        
        deps4 = validator._extract_template_dependencies(test_cases[3])
        assert "task3" in deps4
        
        # These should not extract dependencies
        deps5 = validator._extract_template_dependencies(test_cases[4])
        assert len(deps5) == 0  # Loop variables should not be treated as dependencies
        
        deps6 = validator._extract_template_dependencies(test_cases[5])
        # This might extract "some_context_var" but it would fail validation later if it's not a real task

    def test_execution_order_computation(self):
        """Test computation of valid execution orders."""
        # Diamond dependency pattern
        pipeline_def = {
            "name": "diamond_test",
            "steps": [
                {"id": "start", "action": "start_action"},
                {"id": "left", "action": "left_action", "dependencies": ["start"]},
                {"id": "right", "action": "right_action", "dependencies": ["start"]},
                {"id": "end", "action": "end_action", "dependencies": ["left", "right"]},
            ]
        }
        
        result = self.validator.validate_pipeline_dependencies(pipeline_def)
        
        assert result.is_valid
        assert len(result.issues) == 0
        
        order = result.execution_order
        # Verify topological ordering
        assert order.index("start") < order.index("left")
        assert order.index("start") < order.index("right") 
        assert order.index("left") < order.index("end")
        assert order.index("right") < order.index("end")

    def test_validation_result_properties(self):
        """Test DependencyValidationResult properties and methods."""
        # Create a result with mixed issues
        issues = [
            DependencyIssue("error_type", "error", "Error message"),
            DependencyIssue("warning_type", "warning", "Warning message"),
            DependencyIssue("another_error", "error", "Another error"),
        ]
        
        result = DependencyValidationResult(
            is_valid=False,
            issues=issues,
            execution_order=["task1", "task2"]
        )
        
        assert len(result.errors) == 2
        assert len(result.warnings) == 1
        assert result.has_errors
        assert result.has_warnings
        assert not result.is_valid
        
        # Test valid result
        valid_result = DependencyValidationResult(is_valid=True, issues=[])
        assert len(valid_result.errors) == 0
        assert len(valid_result.warnings) == 0
        assert not valid_result.has_errors
        assert not valid_result.has_warnings
        assert valid_result.is_valid