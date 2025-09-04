"""Integration tests for pipeline integration infrastructure.

This module provides comprehensive tests for the new PipelineTestModel/PipelineTestProvider
patterns, demonstrating systematic pipeline validation using proven infrastructure.

Tests cover:
- PipelineTestModel functionality and validation capabilities
- PipelineTestProvider integration and model management
- PipelineIntegrationValidator systematic validation
- Integration with existing orchestrator framework
- End-to-end pipeline execution validation
"""

import asyncio
import pytest
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock, patch

from src.orchestrator.testing.pipeline_integration_infrastructure import (
    PipelineTestModel,
    PipelineTestProvider,
    PipelineIntegrationValidator,
    PipelineIntegrationResult,
    create_pipeline_test_orchestrator,
    create_pipeline_integration_validator
)
from src.orchestrator.core.model import ModelCapabilities, ModelRequirements, ModelMetrics, ModelCost


class TestPipelineTestModel:
    """Test suite for PipelineTestModel extending proven TestModel patterns."""
    
    def test_pipeline_test_model_initialization(self):
        """Test that PipelineTestModel initializes correctly with enhanced capabilities."""
        
        model = PipelineTestModel()
        
        # Verify basic model properties
        assert model.name == "pipeline-test-model"
        assert model.provider == "pipeline-test-provider"
        
        # Verify enhanced capabilities for pipeline testing
        assert model.capabilities.context_window == 32768
        assert "pipeline-execution" in model.capabilities.supported_tasks
        assert "template-resolution" in model.capabilities.supported_tasks
        assert "quality-assessment" in model.capabilities.supported_tasks
        
        # Verify pipeline-specific configuration
        assert model.pipeline_validation_enabled is True
        assert model.execution_count == 0
        assert isinstance(model.validation_history, list)
    
    def test_pipeline_test_model_custom_initialization(self):
        """Test PipelineTestModel with custom configuration."""
        
        custom_capabilities = ModelCapabilities(
            supported_tasks=["custom-task"],
            context_window=16384,
            supports_function_calling=False,
            supports_structured_output=True
        )
        
        mock_responses = {
            "test_pipeline": "Custom response for test pipeline"
        }
        
        model = PipelineTestModel(
            name="custom-pipeline-model",
            capabilities=custom_capabilities,
            mock_responses=mock_responses,
            pipeline_validation_enabled=False
        )
        
        assert model.name == "custom-pipeline-model"
        assert model.capabilities.context_window == 16384
        assert model.capabilities.supports_function_calling is False
        assert model.mock_responses == mock_responses
        assert model.pipeline_validation_enabled is False
    
    @pytest.mark.asyncio
    async def test_pipeline_test_model_text_generation(self):
        """Test text generation with pipeline context."""
        
        model = PipelineTestModel()
        
        # Test basic generation
        response = await model.generate("Test prompt")
        assert isinstance(response, str)
        assert len(response) > 0
        assert model.execution_count == 1
        
        # Test generation with pipeline context
        pipeline_context = {
            'pipeline_name': 'test_pipeline'
        }
        
        response = await model.generate(
            "Validate this pipeline",
            pipeline_context=pipeline_context
        )
        
        assert "test_pipeline" in response
        assert model.execution_count == 2
        assert len(model.validation_history) == 2
        
        # Verify validation history tracking
        latest_record = model.validation_history[-1]
        assert latest_record['pipeline_name'] == 'test_pipeline'
        assert latest_record['prompt_type'] == 'validation'
    
    @pytest.mark.asyncio
    async def test_pipeline_test_model_mock_responses(self):
        """Test mock response functionality."""
        
        mock_responses = {
            "specific_pipeline": "Specific mock response for this pipeline"
        }
        
        model = PipelineTestModel(mock_responses=mock_responses)
        
        # Test mock response usage
        response = await model.generate(
            "Any prompt",
            pipeline_context={'pipeline_name': 'specific_pipeline'}
        )
        
        assert response == "Specific mock response for this pipeline"
        
        # Test fallback for unknown pipeline
        response = await model.generate(
            "Validation request",
            pipeline_context={'pipeline_name': 'unknown_pipeline'}
        )
        
        assert "unknown_pipeline" in response
        assert "validation" in response.lower()
    
    @pytest.mark.asyncio
    async def test_pipeline_test_model_structured_output(self):
        """Test structured output generation for pipeline validation."""
        
        model = PipelineTestModel()
        
        schema = {
            "type": "object",
            "properties": {
                "validation_status": {"type": "string"},
                "quality_score": {"type": "number"},
                "issues": {"type": "array"},
                "recommendations": {"type": "array"}
            }
        }
        
        response = await model.generate_structured(
            "Generate validation result",
            schema,
            pipeline_context={'pipeline_name': 'test_pipeline'}
        )
        
        # Verify structured response
        assert isinstance(response, dict)
        assert 'validation_status' in response
        assert 'quality_score' in response
        assert 'issues' in response
        assert 'recommendations' in response
        
        # Verify response content
        assert response['validation_status'] == 'passed'
        assert isinstance(response['quality_score'], (int, float))
        assert isinstance(response['issues'], list)
        assert isinstance(response['recommendations'], list)
        assert len(response['recommendations']) > 0
    
    @pytest.mark.asyncio
    async def test_pipeline_test_model_health_check(self):
        """Test enhanced health check functionality."""
        
        # Test with validation enabled
        model = PipelineTestModel(pipeline_validation_enabled=True)
        health = await model.health_check()
        assert health is True
        
        # Test with validation disabled
        model_no_validation = PipelineTestModel(pipeline_validation_enabled=False)
        health = await model_no_validation.health_check()
        assert health is True
    
    @pytest.mark.asyncio
    async def test_pipeline_test_model_cost_estimation(self):
        """Test cost estimation with usage tracking."""
        
        model = PipelineTestModel()
        
        cost = await model.estimate_cost(100, 50)
        assert cost == 0.0  # Always free for testing
        
        # Verify usage tracking
        cost_records = [record for record in model.validation_history if record.get('type') == 'cost_estimation']
        assert len(cost_records) == 1
        
        usage = cost_records[0]['usage']
        assert usage['input_tokens'] == 100
        assert usage['output_tokens'] == 50
        assert usage['estimated_cost'] == 0.0
    
    def test_pipeline_test_model_validation_summary(self):
        """Test validation summary functionality."""
        
        model = PipelineTestModel()
        
        # Initially empty
        summary = model.get_validation_summary()
        assert summary['total_executions'] == 0
        assert summary['validation_attempts'] == 0
        
        # Add some validation history manually for testing
        model.execution_count = 5
        model.validation_history = [
            {'pipeline_name': 'pipeline1', 'prompt_type': 'validation'},
            {'pipeline_name': 'pipeline1', 'prompt_type': 'analysis'},
            {'pipeline_name': 'pipeline2', 'prompt_type': 'validation'},
            {'type': 'cost_estimation'},  # Should not count as validation attempt
        ]
        
        summary = model.get_validation_summary()
        assert summary['total_executions'] == 5
        assert summary['validation_attempts'] == 3  # Excludes cost_estimation
        assert summary['unique_pipelines'] == 2
        assert summary['prompt_types']['validation'] == 2
        assert summary['prompt_types']['analysis'] == 1


class TestPipelineTestProvider:
    """Test suite for PipelineTestProvider extending proven MockTestProvider patterns."""
    
    def test_pipeline_test_provider_initialization(self):
        """Test provider initialization with enhanced capabilities."""
        
        provider = PipelineTestProvider()
        
        assert provider.name == "pipeline-test-provider"
        assert provider.is_initialized is True
        assert provider.validation_enabled is True
        assert provider.integration_tracking is True
        
        # Verify enhanced model registry
        models = provider.available_models
        assert len(models) >= 7  # At least 7 models including aliases
        assert "pipeline-test-model" in models
        assert "pipeline-validation-model" in models
        assert "pipeline-quality-model" in models
        assert "openai/gpt-4" in models
        assert "anthropic/claude-sonnet-4-20250514" in models
    
    def test_pipeline_test_provider_model_support(self):
        """Test model support checking."""
        
        provider = PipelineTestProvider()
        
        # Test supported models
        assert provider.supports_model("pipeline-test-model") is True
        assert provider.supports_model("openai/gpt-4") is True
        assert provider.supports_model("nonexistent-model") is False
    
    def test_pipeline_test_provider_model_capabilities(self):
        """Test model capabilities retrieval."""
        
        provider = PipelineTestProvider()
        
        # Test pipeline-specific model capabilities
        capabilities = provider.get_model_capabilities("pipeline-test-model")
        assert isinstance(capabilities, ModelCapabilities)
        assert capabilities.context_window == 32768
        assert "pipeline-execution" in capabilities.supported_tasks
        
        # Test unsupported model
        with pytest.raises(ValueError, match="Model 'unknown' not supported"):
            provider.get_model_capabilities("unknown")
    
    def test_pipeline_test_provider_model_requirements(self):
        """Test model requirements retrieval."""
        
        provider = PipelineTestProvider()
        
        requirements = provider.get_model_requirements("pipeline-test-model")
        assert isinstance(requirements, ModelRequirements)
        assert requirements.memory_gb == 0.2
        assert requirements.cpu_cores == 1
    
    def test_pipeline_test_provider_model_cost(self):
        """Test model cost information."""
        
        provider = PipelineTestProvider()
        
        cost = provider.get_model_cost("pipeline-test-model")
        assert isinstance(cost, ModelCost)
        assert cost.is_free is True
    
    def test_pipeline_test_provider_info(self):
        """Test provider information with pipeline capabilities."""
        
        provider = PipelineTestProvider()
        
        info = provider.get_provider_info()
        assert info["name"] == "pipeline-test-provider"
        assert info["type"] == "pipeline-test"
        assert info["validation_enabled"] is True
        assert info["integration_tracking"] is True
        assert "supported_pipeline_features" in info
        
        features = info["supported_pipeline_features"]
        assert "template-resolution" in features
        assert "quality-assessment" in features
        assert "execution-validation" in features
    
    @pytest.mark.asyncio
    async def test_pipeline_test_provider_get_model(self):
        """Test model retrieval with usage tracking."""
        
        provider = PipelineTestProvider()
        
        # Get model and verify it's the right type
        model = await provider.get_model("pipeline-test-model")
        assert isinstance(model, PipelineTestModel)
        
        # Verify usage tracking
        stats = provider.get_usage_statistics()
        assert stats['total_requests'] == 1
        assert "pipeline-test-model" in stats['model_usage']
        assert stats['model_usage']["pipeline-test-model"]['requests'] == 1
    
    @pytest.mark.asyncio
    async def test_pipeline_test_provider_initialization(self):
        """Test provider initialization process."""
        
        provider = PipelineTestProvider()
        await provider.initialize()
        
        assert provider.is_initialized is True
    
    def test_pipeline_test_provider_usage_statistics(self):
        """Test usage statistics functionality."""
        
        provider = PipelineTestProvider()
        
        # Initial state
        stats = provider.get_usage_statistics()
        assert stats['total_requests'] == 0
        assert stats['total_models'] >= 7
        
        # Reset statistics
        provider.reset_usage_statistics()
        stats = provider.get_usage_statistics()
        assert stats['total_requests'] == 0
        assert len(stats['model_usage']) == 0


class TestPipelineIntegrationValidator:
    """Test suite for PipelineIntegrationValidator systematic validation."""
    
    def test_pipeline_integration_validator_initialization(self):
        """Test validator initialization with test infrastructure."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            examples_dir = Path(temp_dir)
            
            validator = PipelineIntegrationValidator(examples_dir=examples_dir)
            
            assert validator.examples_dir == examples_dir
            assert isinstance(validator.test_provider, PipelineTestProvider)
            assert validator.model_registry is not None
            assert validator.orchestrator is not None
            assert validator.pipeline_validator is not None
            assert validator.pipeline_test_suite is not None
    
    @pytest.mark.asyncio
    async def test_pipeline_integration_validator_model_integration_test(self):
        """Test model integration validation."""
        
        validator = PipelineIntegrationValidator()
        
        result = await validator._test_model_integration("test_pipeline")
        
        # Verify all model integration tests
        assert result['model_available'] is True
        assert result['health_check_passed'] is True
        assert result['generation_test_passed'] is True
        assert result['structured_output_test_passed'] is True
        assert result['cost_estimation_working'] is True
        assert 'validation_summary' in result
    
    @pytest.mark.asyncio
    async def test_pipeline_integration_validator_provider_integration_test(self):
        """Test provider integration validation."""
        
        validator = PipelineIntegrationValidator()
        
        result = await validator._test_provider_integration("test_pipeline")
        
        # Verify provider integration tests
        assert result['provider_initialized'] is True
        assert result['models_available'] >= 7
        assert result['all_models_healthy'] is True
        assert result['healthy_models_count'] > 0
        assert result['usage_tracking_working'] is True
    
    @pytest.mark.asyncio
    async def test_pipeline_integration_validator_single_pipeline_validation(self):
        """Test validation of a single pipeline."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            examples_dir = Path(temp_dir)
            
            # Create a test pipeline file
            test_pipeline = examples_dir / "test_pipeline.yaml"
            test_pipeline.write_text("""
# Test Pipeline
name: Test Pipeline
description: Simple test pipeline for validation

steps:
  - id: test_step
    tool: filesystem
    action: read
    parameters:
      path: "test.txt"
""")
            
            validator = PipelineIntegrationValidator(examples_dir=examples_dir)
            
            result = await validator.validate_pipeline_integration("test_pipeline", test_pipeline)
            
            # Verify result structure
            assert isinstance(result, PipelineIntegrationResult)
            assert result.pipeline_name == "test_pipeline"
            assert isinstance(result.integration_score, float)
            assert result.integration_score >= 0.0
            assert result.integration_score <= 100.0
            assert isinstance(result.test_model_performance, dict)
            assert isinstance(result.provider_integration_status, dict)
            assert isinstance(result.orchestrator_compatibility, dict)
            assert isinstance(result.recommendations, list)
    
    def test_pipeline_integration_validator_integration_score_calculation(self):
        """Test integration score calculation logic."""
        
        validator = PipelineIntegrationValidator()
        
        # Create test result with perfect scores
        perfect_result = PipelineIntegrationResult(
            pipeline_name="perfect",
            validation_passed=True,
            execution_successful=True,
            integration_score=0.0,
            test_model_performance={
                'model_available': True,
                'health_check_passed': True,
                'generation_test_passed': True,
                'structured_output_test_passed': True,
                'cost_estimation_working': True
            },
            provider_integration_status={
                'provider_initialized': True,
                'models_available': 10,
                'all_models_healthy': True,
                'usage_tracking_working': True
            },
            orchestrator_compatibility={
                'orchestrator_available': True,
                'pipeline_loadable': True,
                'execution_successful': True
            }
        )
        
        score = validator._calculate_integration_score(perfect_result)
        assert score == 100.0
        
        # Test with issues
        result_with_issues = perfect_result
        result_with_issues.issues = ["Issue 1", "Issue 2", "Issue 3"]
        
        score_with_penalty = validator._calculate_integration_score(result_with_issues)
        assert score_with_penalty < 100.0
        assert score_with_penalty >= 90.0  # Max 10 point penalty
    
    def test_pipeline_integration_validator_recommendations(self):
        """Test recommendation generation."""
        
        validator = PipelineIntegrationValidator()
        
        # Create result with various issues
        problematic_result = PipelineIntegrationResult(
            pipeline_name="problematic",
            validation_passed=False,
            execution_successful=False,
            integration_score=50.0,
            test_model_performance={'model_available': False},
            provider_integration_status={'models_available': 0},
            orchestrator_compatibility={'execution_successful': False, 'error': 'Test error'}
        )
        
        recommendations = validator._generate_recommendations(problematic_result)
        
        assert len(recommendations) > 0
        assert any("validation issues" in rec.lower() for rec in recommendations)
        assert any("model" in rec.lower() for rec in recommendations)
        assert any("test error" in rec.lower() for rec in recommendations)
    
    def test_pipeline_integration_validator_summary_generation(self):
        """Test integration summary generation."""
        
        validator = PipelineIntegrationValidator()
        
        # Add mock results
        validator.integration_results = [
            PipelineIntegrationResult(
                pipeline_name="pipeline1",
                validation_passed=True,
                execution_successful=True,
                integration_score=95.0,
                execution_time=1.5
            ),
            PipelineIntegrationResult(
                pipeline_name="pipeline2", 
                validation_passed=False,
                execution_successful=False,
                integration_score=40.0,
                execution_time=0.8,
                issues=["Validation failed", "Execution error"]
            )
        ]
        
        summary = validator.get_integration_summary()
        
        assert summary['total_validations'] == 2
        assert summary['successful_validations'] == 1
        assert summary['successful_executions'] == 1
        assert summary['validation_success_rate'] == 50.0
        assert summary['execution_success_rate'] == 50.0
        assert summary['average_integration_score'] == 67.5
        assert summary['quality_distribution']['high_quality'] == 1
        assert summary['quality_distribution']['needs_work'] == 1


class TestUtilityFunctions:
    """Test suite for utility functions."""
    
    def test_create_pipeline_test_orchestrator(self):
        """Test orchestrator creation utility."""
        
        orchestrator = create_pipeline_test_orchestrator()
        
        assert orchestrator is not None
        assert orchestrator.model_registry is not None
        assert orchestrator.control_system is not None
        
        # Verify test provider is registered
        providers = orchestrator.model_registry.get_registered_providers()
        assert any(provider.name == "pipeline-test-provider" for provider in providers)
    
    def test_create_pipeline_integration_validator(self):
        """Test validator creation utility."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            examples_dir = Path(temp_dir)
            
            validator = create_pipeline_integration_validator(examples_dir)
            
            assert isinstance(validator, PipelineIntegrationValidator)
            assert validator.examples_dir == examples_dir


class TestIntegrationScenarios:
    """Integration test scenarios demonstrating end-to-end functionality."""
    
    @pytest.mark.asyncio
    async def test_complete_pipeline_integration_workflow(self):
        """Test complete integration workflow from model creation to validation."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            examples_dir = Path(temp_dir)
            
            # Create test pipeline
            pipeline_content = """
name: Integration Test Pipeline  
description: Test pipeline for integration validation
version: "1.0.0"

parameters:
  input_text:
    type: string
    default: "Hello, world!"
    description: Input text for processing

steps:
  - id: process_text
    tool: text-processing
    action: analyze
    parameters:
      text: "{{ input_text }}"
      analysis_type: "sentiment"
    
  - id: save_result
    tool: filesystem
    action: write
    parameters:
      path: "outputs/result.txt"
      content: "Analysis: {{ process_text.result }}"
    dependencies:
      - process_text
"""
            
            test_pipeline = examples_dir / "integration_test.yaml"
            test_pipeline.write_text(pipeline_content)
            
            # Create validator
            validator = PipelineIntegrationValidator(examples_dir=examples_dir)
            
            # Run complete validation
            result = await validator.validate_pipeline_integration("integration_test", test_pipeline)
            
            # Verify comprehensive validation
            assert result.pipeline_name == "integration_test"
            assert isinstance(result.integration_score, float)
            assert result.execution_time > 0.0
            
            # Verify all validation components ran
            assert len(result.test_model_performance) > 0
            assert len(result.provider_integration_status) > 0
            assert len(result.orchestrator_compatibility) > 0
            
            # Verify recommendations generated
            assert isinstance(result.recommendations, list)
            
            # Get integration summary
            summary = validator.get_integration_summary()
            assert summary['total_validations'] == 1
    
    @pytest.mark.asyncio
    async def test_multiple_pipeline_validation(self):
        """Test validation of multiple pipelines systematically."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            examples_dir = Path(temp_dir)
            
            # Create multiple test pipelines
            pipelines = {
                "simple.yaml": """
name: Simple Pipeline
steps:
  - id: step1
    tool: test
    action: run
""",
                "complex.yaml": """
name: Complex Pipeline
description: More complex pipeline with dependencies

parameters:
  threshold:
    type: number
    default: 0.5

steps:
  - id: analyze
    tool: analysis
    action: compute
    parameters:
      threshold: "{{ threshold }}"
      
  - id: process
    tool: processing
    action: transform
    parameters:
      data: "{{ analyze.result }}"
    dependencies:
      - analyze
      
  - id: output
    tool: filesystem
    action: write
    parameters:
      path: "output.json"
      content: "{{ process.transformed }}"
    dependencies:
      - process
"""
            }
            
            for filename, content in pipelines.items():
                (examples_dir / filename).write_text(content)
            
            # Run validation on all pipelines
            validator = PipelineIntegrationValidator(examples_dir=examples_dir)
            results = await validator.validate_all_examples()
            
            # Verify results
            assert len(results) == 2
            assert "simple" in results
            assert "complex" in results
            
            # Verify each result
            for pipeline_name, result in results.items():
                assert isinstance(result, PipelineIntegrationResult)
                assert result.pipeline_name == pipeline_name
                assert result.integration_score >= 0.0
            
            # Get comprehensive summary
            summary = validator.get_integration_summary()
            assert summary['total_validations'] == 2
            assert summary['average_integration_score'] >= 0.0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])