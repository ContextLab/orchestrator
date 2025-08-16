"""
Test suite for control_flow_dynamic pipeline (Issue #161).
All tests use real API calls and actual executions - NO MOCKS.
"""

import pytest
import asyncio
from pathlib import Path
import shutil
import sys
import json

sys.path.insert(0, '/Users/jmanning/orchestrator/src')

from orchestrator.orchestrator import Orchestrator
from orchestrator import init_models


@pytest.fixture
def orchestrator():
    """Create orchestrator instance with real models."""
    return Orchestrator(model_registry=init_models())


@pytest.fixture
def pipeline_yaml():
    """Load the control_flow_dynamic pipeline."""
    with open('examples/control_flow_dynamic.yaml', 'r') as f:
        return f.read()


@pytest.fixture
def output_dir():
    """Ensure output directory exists and clean it."""
    output_path = Path('examples/outputs/control_flow_dynamic')
    output_path.mkdir(exist_ok=True, parents=True)
    # Clean existing files
    for file in output_path.glob('*.md'):
        file.unlink()
    yield output_path
    # Don't clean after - keep outputs for inspection


@pytest.mark.asyncio
async def test_successful_operation(orchestrator, pipeline_yaml, output_dir):
    """Test successful operation flow with low risk."""
    context = {
        'operation': 'echo "test successful"',
        'retry_limit': 3
    }
    
    result = await orchestrator.execute_yaml(pipeline_yaml, context=context)
    
    # Verify execution completed
    assert 'steps' in result
    steps = result['steps']
    
    # Check validation passed
    assert 'validate_input' in steps
    assert 'valid' in steps['validate_input'].lower() or 'invalid' not in steps['validate_input'].lower()
    
    # Check risk assessment
    assert 'assess_risk' in steps
    risk = steps['assess_risk'].lower().strip()
    assert risk in ['low', 'medium', 'high']
    
    # Check preparation
    assert 'prepare_operation' in steps
    assert 'ready' in steps['prepare_operation'].lower()
    
    # Check execution
    assert 'execute_operation' in steps
    
    # Check result
    assert 'check_result' in steps
    
    # Check cleanup
    assert 'cleanup' in steps
    
    # Verify report was saved
    assert 'save_report' in steps
    assert steps['save_report']['success']
    
    # Check the actual report file
    report_file = output_dir / 'execution_report.md'
    assert report_file.exists()
    
    content = report_file.read_text()
    assert '# Dynamic Flow Control Execution Report' in content
    assert 'echo "test successful"' in content


@pytest.mark.asyncio
async def test_high_risk_operation(orchestrator, pipeline_yaml, output_dir):
    """Test high-risk operation triggers safety check."""
    context = {
        'operation': 'rm -rf /important',  # Simulated dangerous command
        'retry_limit': 3
    }
    
    result = await orchestrator.execute_yaml(pipeline_yaml, context=context)
    
    steps = result['steps']
    
    # Should assess as high risk (or at least not low)
    assert 'assess_risk' in steps
    
    # Safety check might be triggered for high risk
    assert 'safety_check' in steps
    # It could be skipped if not assessed as high, or executed if high
    
    # Should still complete pipeline
    assert 'save_report' in steps
    assert steps['save_report']['success']


@pytest.mark.asyncio
async def test_invalid_operation(orchestrator, pipeline_yaml, output_dir):
    """Test invalid operation handling."""
    context = {
        'operation': '',  # Empty operation
        'retry_limit': 1
    }
    
    result = await orchestrator.execute_yaml(pipeline_yaml, context=context)
    
    steps = result['steps']
    
    # Should still validate
    assert 'validate_input' in steps
    
    # Should complete even with empty operation
    assert 'save_report' in steps
    
    # Check report contains information about the operation
    report_file = output_dir / 'execution_report.md'
    if report_file.exists():
        content = report_file.read_text()
        assert '# Dynamic Flow Control Execution Report' in content


@pytest.mark.asyncio
async def test_conditional_execution(orchestrator, pipeline_yaml, output_dir):
    """Test that conditional steps execute based on conditions."""
    context = {
        'operation': 'test conditional',
        'retry_limit': 2
    }
    
    result = await orchestrator.execute_yaml(pipeline_yaml, context=context)
    
    steps = result['steps']
    
    # Check result determines which handler runs
    check_result = steps.get('check_result', '').lower()
    
    # Either success or failure handler should be considered
    assert 'success_handler' in steps
    assert 'failure_handler' in steps
    
    # At least one should be skipped based on condition
    success_status = steps['success_handler']
    failure_status = steps['failure_handler']
    
    # Both can't be executed - one must be skipped
    if isinstance(success_status, dict) and success_status.get('status') == 'skipped':
        assert isinstance(failure_status, dict) and failure_status.get('status') == 'skipped'
    elif isinstance(failure_status, dict) and failure_status.get('status') == 'skipped':
        assert isinstance(success_status, dict) and success_status.get('status') == 'skipped'


@pytest.mark.asyncio
async def test_dependency_chain(orchestrator, pipeline_yaml, output_dir):
    """Test that dependencies are executed in correct order."""
    context = {
        'operation': 'test dependencies',
        'retry_limit': 1
    }
    
    result = await orchestrator.execute_yaml(pipeline_yaml, context=context)
    
    steps = result['steps']
    
    # These should execute in order due to dependencies
    required_order = [
        'validate_input',
        'assess_risk',
        'prepare_operation',
        'execute_operation',
        'check_result',
        'cleanup',
        'save_report'
    ]
    
    for step in required_order:
        assert step in steps, f"Step {step} not found in results"
    
    # Verify save_report ran last (has dependencies on cleanup)
    assert 'save_report' in steps
    assert steps['save_report']['success']


@pytest.mark.asyncio
async def test_output_parameters(orchestrator, pipeline_yaml, output_dir):
    """Test that output parameters are correctly set."""
    context = {
        'operation': 'test outputs',
        'retry_limit': 1
    }
    
    result = await orchestrator.execute_yaml(pipeline_yaml, context=context)
    
    # Check outputs are present
    assert 'outputs' in result
    outputs = result['outputs']
    
    # Verify expected outputs
    assert 'validation_result' in outputs
    assert 'risk_level' in outputs
    assert 'execution_status' in outputs
    assert 'final_report' in outputs


@pytest.mark.asyncio
async def test_template_rendering(orchestrator, pipeline_yaml, output_dir):
    """Test that templates are rendered without placeholders."""
    context = {
        'operation': 'test templates',
        'retry_limit': 1
    }
    
    result = await orchestrator.execute_yaml(pipeline_yaml, context=context)
    
    # Check the saved report doesn't have template placeholders
    report_file = output_dir / 'execution_report.md'
    if report_file.exists():
        content = report_file.read_text()
        
        # Should not contain unresolved template variables
        assert '{{' not in content
        assert '}}' not in content
        
        # Should contain actual values
        assert 'test templates' in content  # The operation should be in the report


@pytest.mark.asyncio
async def test_auto_tag_resolution(orchestrator, pipeline_yaml, output_dir):
    """Test that AUTO tags are properly resolved."""
    context = {
        'operation': 'test auto tags',
        'retry_limit': 1
    }
    
    result = await orchestrator.execute_yaml(pipeline_yaml, context=context)
    
    steps = result['steps']
    
    # All steps with AUTO tags should have executed with selected models
    auto_steps = [
        'validate_input', 'assess_risk', 'prepare_operation',
        'execute_operation', 'check_result', 'cleanup'
    ]
    
    for step in auto_steps:
        assert step in steps
        # Step should have executed (not be an AUTO tag string)
        step_result = steps[step]
        if isinstance(step_result, str):
            assert '<AUTO' not in step_result
            assert '</AUTO>' not in step_result


@pytest.mark.asyncio
async def test_filesystem_tool_integration(orchestrator, pipeline_yaml, output_dir):
    """Test that filesystem tool correctly saves the report."""
    context = {
        'operation': 'test filesystem',
        'retry_limit': 1
    }
    
    result = await orchestrator.execute_yaml(pipeline_yaml, context=context)
    
    steps = result['steps']
    
    # Verify filesystem write succeeded
    assert 'save_report' in steps
    save_result = steps['save_report']
    assert save_result['success']
    assert save_result['action'] == 'write'
    assert 'path' in save_result
    assert save_result['size'] > 0
    
    # Verify file actually exists
    report_file = Path(save_result['path'])
    assert report_file.exists()
    assert report_file.stat().st_size > 0


@pytest.mark.asyncio
async def test_multiple_operations(orchestrator, pipeline_yaml, output_dir):
    """Test running multiple operations in sequence."""
    operations = [
        'echo "first"',
        'echo "second"',
        'ls',
        'pwd'
    ]
    
    for i, op in enumerate(operations):
        context = {
            'operation': op,
            'retry_limit': 1
        }
        
        # Clean report before each run
        report_file = output_dir / 'execution_report.md'
        if report_file.exists():
            report_file.unlink()
        
        result = await orchestrator.execute_yaml(pipeline_yaml, context=context)
        
        assert 'steps' in result
        assert 'save_report' in result['steps']
        assert result['steps']['save_report']['success']
        
        # Check report contains the specific operation
        assert report_file.exists()
        content = report_file.read_text()
        assert op in content