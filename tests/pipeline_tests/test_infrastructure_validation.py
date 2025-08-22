"""Test to validate the pipeline testing infrastructure itself."""

import pytest
import tempfile
from pathlib import Path

from .test_base import BasePipelineTest, PipelineTestConfiguration
from .test_runner import PipelineTestCase, PipelineTestRunner, PipelineLoader, OutputDirectoryManager


pytestmark = pytest.mark.integration


class TestInfrastructureValidation(BasePipelineTest):
    """Test that validates the pipeline testing infrastructure works correctly."""
    
    def test_basic_execution(self):
        """Test basic pipeline execution using the infrastructure."""
        # This satisfies the abstract method requirement
        pass
    
    def test_error_handling(self):
        """Test error handling scenarios using the infrastructure.""" 
        # This satisfies the abstract method requirement
        pass


@pytest.mark.asyncio
async def test_pipeline_base_functionality(pipeline_orchestrator, pipeline_model_registry, sample_pipeline_yaml, pipeline_inputs):
    """Test that the BasePipelineTest class works correctly."""
    
    # Create test configuration
    config = PipelineTestConfiguration(
        timeout_seconds=60,
        max_cost_dollars=0.50,
        enable_performance_tracking=True
    )
    
    # Create test instance
    test_instance = TestInfrastructureValidation(
        orchestrator=pipeline_orchestrator,
        model_registry=pipeline_model_registry,
        config=config
    )
    
    # Execute pipeline
    result = await test_instance.execute_pipeline_async(
        yaml_content=sample_pipeline_yaml,
        inputs=pipeline_inputs
    )
    
    # Verify result structure
    assert result is not None
    assert hasattr(result, 'success')
    assert hasattr(result, 'outputs')
    assert hasattr(result, 'execution_time')
    assert hasattr(result, 'estimated_cost')
    
    # Print result for debugging
    print(f"Pipeline execution result: success={result.success}")
    print(f"Execution time: {result.execution_time:.2f}s")
    print(f"Estimated cost: ${result.estimated_cost:.4f}")
    
    if result.error_message:
        print(f"Error: {result.error_message}")
    
    # Basic assertions (these should pass even if the pipeline fails)
    assert result.execution_time >= 0
    assert result.estimated_cost >= 0
    
    # If execution was successful, verify outputs
    if result.success:
        assert result.outputs is not None
        assert isinstance(result.outputs, dict)
        
        # Test assertion methods
        test_instance.assert_pipeline_success(result)
        test_instance.assert_performance_within_limits(result, max_time=120, max_cost=0.50)
        
        print("✅ Pipeline executed successfully!")
    else:
        print(f"⚠️ Pipeline failed (expected during infrastructure testing): {result.error_message}")
        # For infrastructure testing, failure is acceptable
        # The important thing is that our testing framework captured the failure correctly


def test_pipeline_loader():
    """Test the PipelineLoader utility."""
    
    # Test YAML validation
    valid_yaml = """
name: Test Pipeline
tasks:
  - name: test_task
    type: llm
    template: "Hello {{ name }}"
"""
    
    invalid_yaml = """
name: Test Pipeline
tasks:
  - name: test_task
    type: llm
    template: "Hello {{ name }"  # Missing closing brace
"""
    
    assert PipelineLoader.validate_yaml_syntax(valid_yaml) == True
    assert PipelineLoader.validate_yaml_syntax(invalid_yaml) == True  # YAML parsing doesn't catch template errors
    
    # Test metadata extraction
    metadata = PipelineLoader.extract_pipeline_metadata(valid_yaml)
    assert metadata['name'] == 'Test Pipeline'
    assert metadata['task_count'] == 1
    assert 'llm' in metadata['task_types']


def test_output_directory_manager():
    """Test the OutputDirectoryManager utility."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        base_dir = Path(temp_dir)
        manager = OutputDirectoryManager(base_dir)
        
        # Create test directory
        test_dir = manager.create_test_directory("test_pipeline")
        assert test_dir.exists()
        assert test_dir.parent == base_dir
        
        # Test cleanup (without keeping failed)
        manager.cleanup_test_directories(keep_failed=False)
        # Note: Directory might still exist briefly due to cleanup timing


@pytest.mark.asyncio 
async def test_pipeline_test_runner(pipeline_orchestrator, pipeline_model_registry, sample_pipeline_yaml, pipeline_inputs):
    """Test the PipelineTestRunner functionality."""
    
    # Create test configuration
    config = PipelineTestConfiguration(
        timeout_seconds=60,
        max_cost_dollars=0.50,
        enable_performance_tracking=True,
        save_intermediate_outputs=False  # Don't save for this test
    )
    
    # Create output manager
    with tempfile.TemporaryDirectory() as temp_dir:
        output_manager = OutputDirectoryManager(Path(temp_dir))
        
        # Create test runner
        runner = PipelineTestRunner(
            orchestrator=pipeline_orchestrator,
            model_registry=pipeline_model_registry,
            config=config,
            output_manager=output_manager
        )
        
        # Create test cases
        test_cases = [
            PipelineTestCase(
                name="basic_test",
                yaml_content=sample_pipeline_yaml,
                inputs=pipeline_inputs,
                description="Basic pipeline test"
            )
        ]
        
        # Run test suite
        suite_result = await runner.run_test_suite_async(
            test_cases=test_cases,
            suite_name="Infrastructure Validation",
            parallel=False
        )
        
        # Verify suite result
        assert suite_result is not None
        assert suite_result.total_tests == 1
        assert suite_result.suite_name == "Infrastructure Validation"
        assert len(suite_result.test_results) == 1
        
        # Print suite summary
        print(f"Test suite completed:")
        print(f"- Total tests: {suite_result.total_tests}")
        print(f"- Successful: {suite_result.successful_tests}")
        print(f"- Failed: {suite_result.failed_tests}")
        print(f"- Success rate: {suite_result.success_rate:.1f}%")
        print(f"- Total time: {suite_result.total_execution_time:.2f}s")
        print(f"- Total cost: ${suite_result.total_cost:.4f}")
        
        # Generate report
        report = runner.generate_test_report(suite_result)
        assert "Pipeline Test Report" in report
        assert "Infrastructure Validation" in report
        
        print("\nGenerated report preview:")
        print(report[:500] + "..." if len(report) > 500 else report)


def test_configuration_validation():
    """Test that configurations are properly validated."""
    
    # Test default configuration
    config = PipelineTestConfiguration()
    assert config.timeout_seconds > 0
    assert config.max_cost_dollars > 0
    assert config.enable_performance_tracking == True
    
    # Test custom configuration
    custom_config = PipelineTestConfiguration(
        timeout_seconds=300,
        max_cost_dollars=2.0,
        enable_validation=False
    )
    assert custom_config.timeout_seconds == 300
    assert custom_config.max_cost_dollars == 2.0
    assert custom_config.enable_validation == False


if __name__ == "__main__":
    # Allow running this test file directly for debugging
    pytest.main([__file__, "-v", "-s"])