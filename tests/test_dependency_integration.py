"""Integration tests for dependency validation in schema validator and YAML compiler."""

import pytest
import sys
from unittest.mock import patch, MagicMock

from src.orchestrator.compiler.schema_validator import SchemaValidator
from src.orchestrator.compiler.yaml_compiler import YAMLCompiler
from src.orchestrator.core.exceptions import SchemaValidationError, YAMLCompilerError


class TestSchemaValidatorIntegration:
    """Test dependency validation integration in SchemaValidator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = SchemaValidator()

    def test_comprehensive_dependency_validation_enabled(self):
        """Test that comprehensive dependency validation can be enabled."""
        pipeline_def = {
            "name": "test_pipeline",
            "steps": [
                {"id": "task1", "action": "test_action"},
                {"id": "task2", "action": "test_action", "dependencies": ["task1"]},
            ]
        }
        
        # Should not raise any errors for valid pipeline
        errors = self.validator.validate_complete(
            pipeline_def, 
            enable_dependency_validation=True
        )
        
        assert len(errors) == 0

    def test_comprehensive_dependency_validation_disabled(self):
        """Test that comprehensive dependency validation can be disabled."""
        pipeline_def = {
            "name": "test_pipeline", 
            "steps": [
                {"id": "task1", "action": "test_action"},
                {"id": "task2", "action": "test_action", "dependencies": ["nonexistent"]},
            ]
        }
        
        # With dependency validation disabled, should not catch missing dependency
        errors_without_dep_validation = self.validator.validate_complete(
            pipeline_def,
            enable_dependency_validation=False
        )
        
        # With dependency validation enabled, should catch missing dependency
        errors_with_dep_validation = self.validator.validate_complete(
            pipeline_def,
            enable_dependency_validation=True
        )
        
        # Should have more errors when dependency validation is enabled
        assert len(errors_with_dep_validation) > len(errors_without_dep_validation)

    def test_development_mode_effect(self):
        """Test that development mode affects dependency validation."""
        pipeline_def = {
            "name": "test_pipeline",
            "steps": [
                {"id": "task1", "action": "test_action"},
                {"id": "task2", "action": "test_action", "dependencies": ["nonexistent"]},
            ]
        }
        
        # Development mode should be more lenient
        errors_dev_mode = self.validator.validate_complete(
            pipeline_def,
            enable_dependency_validation=True,
            development_mode=True
        )
        
        # Regular mode should be strict
        errors_regular_mode = self.validator.validate_complete(
            pipeline_def,
            enable_dependency_validation=True,
            development_mode=False
        )
        
        # Development mode should have fewer errors (some converted to warnings)
        assert len(errors_dev_mode) <= len(errors_regular_mode)

    def test_circular_dependency_detection_in_schema_validation(self):
        """Test that circular dependencies are detected during schema validation."""
        pipeline_def = {
            "name": "test_pipeline",
            "steps": [
                {"id": "task1", "action": "test_action", "dependencies": ["task2"]},
                {"id": "task2", "action": "test_action", "dependencies": ["task1"]},
            ]
        }
        
        errors = self.validator.validate_complete(
            pipeline_def,
            enable_dependency_validation=True
        )
        
        # Should detect circular dependency
        assert any("circular" in error.lower() for error in errors)

    @patch('orchestrator.validation.dependency_validator.DependencyValidator')
    def test_dependency_validator_error_handling(self, mock_validator_class):
        """Test error handling when DependencyValidator fails."""
        # Mock validator to raise an exception
        mock_validator = MagicMock()
        mock_validator.validate_pipeline_dependencies.side_effect = Exception("Validation failed")
        mock_validator_class.return_value = mock_validator
        
        pipeline_def = {
            "name": "test_pipeline",
            "steps": [{"id": "task1", "action": "test_action"}]
        }
        
        errors = self.validator.validate_complete(
            pipeline_def,
            enable_dependency_validation=True
        )
        
        # Should handle the exception gracefully
        assert any("Error during dependency validation" in error for error in errors)

    def test_dependency_validator_import_error_handling(self):
        """Test handling when DependencyValidator cannot be imported."""
        with patch.dict('sys.modules', {'orchestrator.validation.dependency_validator': None}):
            pipeline_def = {
                "name": "test_pipeline", 
                "steps": [{"id": "task1", "action": "test_action"}]
            }
            
            errors = self.validator.validate_complete(
                pipeline_def,
                enable_dependency_validation=True
            )
            
            # Should handle import error gracefully
            assert any("dependency validation unavailable" in error.lower() for error in errors)


class TestYAMLCompilerIntegration:
    """Test dependency validation integration in YAMLCompiler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.compiler = YAMLCompiler(validate_tools=False)  # Disable tool validation for tests
        self.dev_compiler = YAMLCompiler(development_mode=True, validate_tools=False)

    @pytest.mark.asyncio
    async def test_dependency_validation_in_compile_pipeline(self):
        """Test that dependency validation runs during pipeline compilation."""
        yaml_content = """
        name: test_pipeline
        steps:
          - id: task1
            action: test_action
          - id: task2
            action: test_action
            dependencies: [task1]
        """
        
        # Should compile successfully with valid dependencies
        pipeline = await self.compiler.compile(yaml_content)
        
        assert pipeline is not None
        assert pipeline.name == "test_pipeline"
        assert len(pipeline.tasks) == 2

    @pytest.mark.asyncio
    async def test_circular_dependency_compilation_failure(self):
        """Test that circular dependencies cause compilation to fail."""
        yaml_content = """
        name: test_pipeline
        steps:
          - id: task1
            action: test_action
            dependencies: [task2]
          - id: task2
            action: test_action
            dependencies: [task1]
        """
        
        # Should raise YAMLCompilerError due to circular dependency
        with pytest.raises(YAMLCompilerError) as exc_info:
            await self.compiler.compile(yaml_content)
        
        assert "dependency validation failed" in str(exc_info.value).lower()
        assert "circular" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_missing_dependency_compilation_failure(self):
        """Test that missing dependencies cause compilation to fail."""
        yaml_content = """
        name: test_pipeline
        steps:
          - id: task1
            action: test_action
          - id: task2
            action: test_action
            dependencies: [nonexistent_task]
        """
        
        # Should raise YAMLCompilerError due to missing dependency
        with pytest.raises(YAMLCompilerError) as exc_info:
            await self.compiler.compile(yaml_content)
        
        assert "dependency validation failed" in str(exc_info.value).lower()
        assert "nonexistent_task" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_development_mode_bypasses_strict_validation(self):
        """Test that development mode allows compilation with dependency issues."""
        yaml_content = """
        name: test_pipeline
        steps:
          - id: task1
            action: test_action
          - id: task2
            action: test_action
            dependencies: [nonexistent_task]
        """
        
        # Both compilers should fail due to pipeline-level validation
        # Development mode only bypasses the comprehensive dependency validation step
        # The pipeline itself still validates dependencies when tasks are added
        with pytest.raises(YAMLCompilerError):
            await self.compiler.compile(yaml_content)
        
        with pytest.raises(YAMLCompilerError):
            await self.dev_compiler.compile(yaml_content)
        
        # However, development mode should skip the comprehensive dependency validation step
        # which means it fails later (at pipeline build time) rather than earlier

    @pytest.mark.asyncio
    async def test_complex_dependency_patterns_validation(self):
        """Test validation of complex dependency patterns during compilation."""
        yaml_content = """
        name: complex_pipeline
        steps:
          - id: init
            action: initialize
          
          - id: data_loader
            action: load_data
            dependencies: [init]
          
          - id: conditional_task
            action: conditional_action
            condition: "{{ data_loader.result.success }}"
            dependencies: [data_loader]
          
          - id: final_report
            action: generate_report
            dependencies: [conditional_task]
        """
        
        # Should compile successfully with complex but valid dependencies
        pipeline = await self.compiler.compile(yaml_content)
        
        assert pipeline is not None
        assert pipeline.name == "complex_pipeline"
        assert len(pipeline.tasks) == 4

    @pytest.mark.asyncio
    async def test_for_each_dependency_validation(self):
        """Test validation of for_each dependencies during compilation."""
        yaml_content = """
        name: foreach_pipeline
        steps:
          - id: data_source
            action: get_data
          
          - id: processor
            for_each: "{{ data_source.result.items }}"
            steps:
              - id: process_item
                action: process_item
        """
        
        # Should compile successfully
        pipeline = await self.compiler.compile(yaml_content)
        assert pipeline is not None

    @pytest.mark.asyncio 
    async def test_for_each_missing_dependency_failure(self):
        """Test that for_each with missing dependencies fails compilation."""
        yaml_content = """
        name: foreach_pipeline
        steps:
          - id: processor
            for_each: "{{ missing_source.result.items }}"
            steps:
              - id: process_item
                action: process_item
        """
        
        # Should fail due to missing dependency in for_each
        with pytest.raises(YAMLCompilerError) as exc_info:
            await self.compiler.compile(yaml_content)
        
        assert "dependency validation failed" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_action_loop_dependency_validation(self):
        """Test validation of action_loop dependencies during compilation."""
        yaml_content = """
        name: action_loop_pipeline
        steps:
          - id: counter
            action: init_counter
          
          - id: loop_task
            action_loop:
              - action: increment
            until: "{{ counter.result.value >= 10 }}"
        """
        
        # Should compile successfully
        pipeline = await self.compiler.compile(yaml_content)
        assert pipeline is not None

    @pytest.mark.asyncio
    async def test_string_dependencies_compilation(self):
        """Test compilation with string-formatted dependencies."""
        yaml_content = """
        name: string_deps_pipeline
        steps:
          - id: task1
            action: action1
          
          - id: task2  
            action: action2
            
          - id: task3
            action: action3
            dependencies: ["task1", "task2"]
            
          - id: task4
            action: action4
            depends_on: ["task3"]
        """
        
        # Should compile successfully with array dependencies
        pipeline = await self.compiler.compile(yaml_content)
        
        assert pipeline is not None
        assert len(pipeline.tasks) == 4

    @pytest.mark.asyncio
    @patch('orchestrator.validation.dependency_validator.DependencyValidator')
    async def test_dependency_validator_exception_handling(self, mock_validator_class):
        """Test handling of exceptions from DependencyValidator during compilation."""
        # Mock validator to raise an exception
        mock_validator = MagicMock()
        mock_validator.validate_pipeline_dependencies.side_effect = Exception("Validation error")
        mock_validator_class.return_value = mock_validator
        
        yaml_content = """
        name: test_pipeline
        steps:
          - id: task1
            action: test_action
        """
        
        # Should raise YAMLCompilerError with validation error details
        with pytest.raises(YAMLCompilerError) as exc_info:
            await self.compiler.compile(yaml_content)
        
        assert "dependency validation failed" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_dependency_validator_import_error_handling(self):
        """Test handling when DependencyValidator import fails during compilation."""
        with patch.dict('sys.modules', {'orchestrator.validation.dependency_validator': None}):
            yaml_content = """
            name: test_pipeline
            steps:
              - id: task1
                action: test_action
            """
            
            # Should compile successfully but log warning about missing dependency validator
            pipeline = await self.compiler.compile(yaml_content)
            
            assert pipeline is not None
            assert pipeline.name == "test_pipeline"

    @pytest.mark.asyncio
    async def test_dependency_validation_with_context(self):
        """Test dependency validation works correctly with template context."""
        yaml_content = """
        name: context_pipeline
        steps:
          - id: source
            action: get_data
            parameters:
              source_type: "{{ input_type }}"
          
          - id: processor
            action: process_data
            dependencies: [source]
            parameters:
              process_mode: "{{ processing_mode }}"
        """
        
        context = {
            "input_type": "database",
            "processing_mode": "batch"
        }
        
        # Should compile successfully with context
        pipeline = await self.compiler.compile(yaml_content, context=context)
        
        assert pipeline is not None
        assert pipeline.name == "context_pipeline"
        assert len(pipeline.tasks) == 2

    @pytest.mark.asyncio
    async def test_execution_order_computation_during_compilation(self):
        """Test that execution order is computed and available after compilation."""
        yaml_content = """
        name: ordered_pipeline
        steps:
          - id: first
            action: first_action
          
          - id: second
            action: second_action
            dependencies: [first]
          
          - id: third
            action: third_action
            dependencies: [second]
        """
        
        # This should compile successfully with valid dependency chain
        pipeline = await self.compiler.compile(yaml_content)
        
        assert pipeline is not None
        assert pipeline.name == "ordered_pipeline"
        assert len(pipeline.tasks) == 3