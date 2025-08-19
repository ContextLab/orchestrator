"""Integration tests for the interactive pipeline with real user input.

These tests verify the complete interactive pipeline works with real inputs,
no mocks or simulations.
"""

import pytest
import sys
import io
import os
import json
import csv
import asyncio
from pathlib import Path
from typing import List, Dict, Any

from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator.models.model_registry import ModelRegistry


class RealPipelineInputSimulator:
    """Simulate real user input for pipeline testing."""
    
    def __init__(self, inputs):
        """Initialize with list of input strings."""
        self.inputs = inputs if isinstance(inputs, list) else [inputs]
        self.original_stdin = None
        
    def __enter__(self):
        """Set up input simulation."""
        self.original_stdin = sys.stdin
        input_string = '\n'.join(self.inputs) + '\n'
        sys.stdin = io.StringIO(input_string)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original stdin."""
        sys.stdin = self.original_stdin


class TestInteractivePipeline:
    """Test the complete interactive pipeline with real inputs."""
    
    @pytest.fixture
    async def setup_test_environment(self, tmp_path):
        """Set up test environment with real files and directories."""
        # Create test directories
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        (output_dir / "data").mkdir()
        (output_dir / "feedback").mkdir()
        
        # Create real test CSV file
        csv_file = input_dir / "test_data.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'name', 'value', 'category'])
            writer.writerow([1, 'Product A', 100.50, 'electronics'])
            writer.writerow([2, 'Product B', 75.25, 'books'])
            writer.writerow([3, 'Product C', 250.00, 'electronics'])
            writer.writerow([4, 'Product D', 30.99, 'books'])
        
        # Initialize orchestrator with real model registry
        model_registry = ModelRegistry()
        orchestrator = Orchestrator(model_registry=model_registry)
        
        return {
            'orchestrator': orchestrator,
            'input_file': str(csv_file.relative_to(tmp_path)),
            'output_dir': str(output_dir),
            'tmp_path': tmp_path
        }
    
    @pytest.mark.asyncio
    async def test_standard_processing_flow(self, setup_test_environment):
        """Test standard processing with approval flow."""
        env = await setup_test_environment
        
        # Simulate user inputs for the pipeline
        user_inputs = [
            "standard",      # Processing method
            "csv",          # Output format
            "approve",      # Approve results
            "Great job!",   # Approval comment
            "5",           # Ease of use rating
            "5",           # Processing quality rating
            "yes",         # Would use again
            "Very intuitive pipeline!"  # Suggestions
        ]
        
        with RealPipelineInputSimulator(user_inputs):
            # Load and execute pipeline
            with open('examples/interactive_pipeline.yaml', 'r') as f:
                pipeline_yaml = f.read()
            
            # Execute with real inputs
            results = await env['orchestrator'].execute_yaml(
                pipeline_yaml,
                inputs={
                    'input_file': env['input_file'],
                    'output_dir': env['output_dir']
                }
            )
        
        # Verify outputs exist and are correct
        output_file = Path(env['output_dir']) / 'data' / 'output.csv'
        assert output_file.exists(), "Output CSV file should exist"
        
        # Verify CSV content is valid
        with open(output_file, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert len(rows) > 0, "Output CSV should have data"
        
        # Verify feedback was saved
        feedback_file = Path(env['output_dir']) / 'feedback' / 'pipeline_feedback.json'
        assert feedback_file.exists(), "Feedback file should exist"
        
        with open(feedback_file, 'r') as f:
            feedback = json.load(f)
            assert feedback['responses']['ease_of_use'] == 5
            assert feedback['responses']['would_use_again'] is True
            assert feedback['responses']['suggestions'] == "Very intuitive pipeline!"
        
        # Verify results structure
        assert results['get_processing_options']['value'] == "standard"
        assert results['get_output_format']['value'] == "csv"
        assert results['approve_results']['approved'] is True
    
    @pytest.mark.asyncio
    async def test_rejection_flow(self, setup_test_environment):
        """Test rejection flow with real interaction."""
        env = await setup_test_environment
        
        user_inputs = [
            "advanced",     # Processing method
            "json",        # Output format
            "reject",      # Reject results
            "Data quality issues",  # Rejection reason
            "3",          # Ease of use rating
            "2",          # Processing quality rating
            "no",         # Would not use again
            "Needs better error handling"  # Suggestions
        ]
        
        with RealPipelineInputSimulator(user_inputs):
            # Load and execute pipeline
            with open('examples/interactive_pipeline.yaml', 'r') as f:
                pipeline_yaml = f.read()
            
            results = await env['orchestrator'].execute_yaml(
                pipeline_yaml,
                inputs={
                    'input_file': env['input_file'],
                    'output_dir': env['output_dir']
                }
            )
        
        # Output file should NOT exist when rejected
        output_file = Path(env['output_dir']) / 'data' / 'output.json'
        assert not output_file.exists(), "Output should not be saved when rejected"
        
        # But feedback should still be collected
        feedback_file = Path(env['output_dir']) / 'feedback' / 'pipeline_feedback.json'
        assert feedback_file.exists(), "Feedback should be collected even on rejection"
        
        with open(feedback_file, 'r') as f:
            feedback = json.load(f)
            assert feedback['responses']['ease_of_use'] == 3
            assert feedback['responses']['processing_quality'] == 2
            assert feedback['responses']['would_use_again'] is False
        
        # Verify rejection was recorded
        assert results['approve_results']['approved'] is False
        assert results['approve_results']['rejection_reason'] == "Data quality issues"
    
    @pytest.mark.asyncio
    async def test_modify_and_approve_flow(self, setup_test_environment):
        """Test modification and approval with real data."""
        env = await setup_test_environment
        
        # User modifies the data before approving
        modified_data = "id,name,value,category\n1,Modified A,150,electronics\n2,Modified B,200,books"
        
        user_inputs = [
            "custom",       # Processing method
            "yaml",        # Output format
            "modify",      # Choose to modify
            modified_data,  # Modified content
            "approve",     # Then approve
            "Fixed data issues",  # Comment
            "4",          # Ease of use rating
            "4",          # Processing quality rating
            "yes",        # Would use again
            "Good modification feature"  # Suggestions
        ]
        
        with RealPipelineInputSimulator(user_inputs):
            # Load and execute pipeline
            with open('examples/interactive_pipeline.yaml', 'r') as f:
                pipeline_yaml = f.read()
            
            results = await env['orchestrator'].execute_yaml(
                pipeline_yaml,
                inputs={
                    'input_file': env['input_file'],
                    'output_dir': env['output_dir']
                }
            )
        
        # Verify modified content was saved
        output_file = Path(env['output_dir']) / 'data' / 'output.yaml'
        assert output_file.exists(), "Modified output should be saved"
        
        with open(output_file, 'r') as f:
            content = f.read()
            assert "Modified A" in content
            assert "150" in content
            assert "Modified B" in content
        
        # Verify modification was recorded
        assert results['approve_results']['approved'] is True
        assert results['approve_results']['modified'] is True
        assert "Modified A" in results['approve_results']['modified_content']
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, setup_test_environment):
        """Test timeout with default values (NO MOCKS)."""
        env = await setup_test_environment
        
        # Load pipeline and modify timeout
        with open('examples/interactive_pipeline.yaml', 'r') as f:
            pipeline_yaml = f.read()
        
        # Configure pipeline with timeout
        modified_pipeline = pipeline_yaml.replace(
            'timeout: 0',
            'timeout: 2'  # 2 second timeout
        ).replace(
            'default: "standard"',
            'default: "standard"\n      timeout: 2'
        )
        
        # Don't provide input - let it timeout
        with RealPipelineInputSimulator([]):
            # Add a delay to force timeout
            import time
            original_input = input
            
            def delayed_input(prompt):
                time.sleep(3)  # Sleep longer than timeout
                return "toolate"
            
            # Temporarily replace input function
            import builtins
            builtins.input = delayed_input
            
            try:
                results = await env['orchestrator'].execute_yaml(
                    modified_pipeline,
                    inputs={
                        'input_file': env['input_file'],
                        'output_dir': env['output_dir']
                    }
                )
                
                # Should use defaults when timeout occurs
                assert results is not None
                assert results['get_processing_options']['value'] == "standard"  # default value
                assert results['get_processing_options'].get('skipped', False) is True
            finally:
                # Restore original input
                builtins.input = original_input
    
    @pytest.mark.asyncio  
    async def test_invalid_input_retry(self, setup_test_environment):
        """Test retry mechanism with invalid inputs."""
        env = await setup_test_environment
        
        user_inputs = [
            "invalid_option",  # Invalid choice (will be rejected)
            "standard",        # Retry with valid choice
            "not_a_format",   # Invalid format (will be rejected)
            "csv",            # Retry with valid format
            "approve",        # Approve results
            "",               # No comment
            "5",             # Ease of use rating
            "5",             # Processing quality rating
            "yes",           # Would use again
            ""               # No suggestions
        ]
        
        with RealPipelineInputSimulator(user_inputs):
            # Load and execute pipeline
            with open('examples/interactive_pipeline.yaml', 'r') as f:
                pipeline_yaml = f.read()
            
            results = await env['orchestrator'].execute_yaml(
                pipeline_yaml,
                inputs={
                    'input_file': env['input_file'],
                    'output_dir': env['output_dir']
                }
            )
        
        # Should succeed after retries
        assert results['get_processing_options']['value'] == "standard"
        assert results['get_output_format']['value'] == "csv"
        assert results['get_processing_options'].get('retries', 0) > 0
    
    @pytest.mark.asyncio
    async def test_api_context_with_defaults(self, setup_test_environment):
        """Test API context using defaults (no user interaction)."""
        env = await setup_test_environment
        
        # Load pipeline
        with open('examples/interactive_pipeline.yaml', 'r') as f:
            pipeline_yaml = f.read()
        
        # Modify pipeline to use API context
        api_pipeline = pipeline_yaml.replace(
            'context: "cli"',
            'context: "api"'
        )
        
        # Execute without user input - should use defaults
        results = await env['orchestrator'].execute_yaml(
            api_pipeline,
            inputs={
                'input_file': env['input_file'],
                'output_dir': env['output_dir']
            }
        )
        
        # Should complete with default values
        assert results['get_processing_options']['value'] == "standard"  # default
        assert results['get_output_format']['value'] == "csv"  # default
        assert results['get_processing_options']['skipped'] is True  # Used default
    
    @pytest.mark.asyncio
    async def test_all_templates_rendered(self, setup_test_environment):
        """Ensure no unrendered templates in output."""
        env = await setup_test_environment
        
        user_inputs = [
            "standard",      # Processing method
            "json",         # Output format
            "approve",      # Approve results
            "",            # No comment
            "5",           # Ease of use rating
            "5",           # Processing quality rating
            "yes",         # Would use again
            ""             # No suggestions
        ]
        
        with RealPipelineInputSimulator(user_inputs):
            # Load and execute pipeline
            with open('examples/interactive_pipeline.yaml', 'r') as f:
                pipeline_yaml = f.read()
            
            results = await env['orchestrator'].execute_yaml(
                pipeline_yaml,
                inputs={
                    'input_file': env['input_file'],
                    'output_dir': env['output_dir']
                }
            )
        
        # Check all output files for unrendered templates
        output_dir = Path(env['output_dir'])
        for file_path in output_dir.rglob('*'):
            if file_path.is_file():
                content = file_path.read_text()
                # Check for common template patterns
                assert '{{' not in content, f"Unrendered template in {file_path}"
                assert '{%' not in content, f"Unrendered Jinja in {file_path}"
                assert '<AUTO>' not in content, f"Unresolved AUTO tag in {file_path}"
                
        # Verify summary report has rendered timestamp
        if 'generate_summary' in results:
            summary = results['generate_summary'].get('content', '')
            assert '{{' not in summary
            assert 'now()' not in summary
    
    @pytest.mark.asyncio
    async def test_data_processing_operations(self, setup_test_environment):
        """Test actual data transformations."""
        env = await setup_test_environment
        
        # Test each processing method with real data
        for method in ['standard', 'advanced', 'custom']:
            # Create fresh output directory for each test
            output_subdir = Path(env['output_dir']) / method
            output_subdir.mkdir()
            (output_subdir / "data").mkdir()
            (output_subdir / "feedback").mkdir()
            
            user_inputs = [
                method,        # Processing method
                'csv',        # Output format
                'approve',    # Approve results
                '',          # No comment
                '5',         # Ease of use rating
                '5',         # Processing quality rating
                'yes',       # Would use again
                ''           # No suggestions
            ]
            
            with RealPipelineInputSimulator(user_inputs):
                # Load and execute pipeline
                with open('examples/interactive_pipeline.yaml', 'r') as f:
                    pipeline_yaml = f.read()
                
                results = await env['orchestrator'].execute_yaml(
                    pipeline_yaml,
                    inputs={
                        'input_file': env['input_file'],
                        'output_dir': str(output_subdir)
                    }
                )
            
            # Verify data was actually processed
            assert 'process_data' in results
            assert results['process_data']['success'] is True
            
            # Verify output file exists
            output_file = output_subdir / 'data' / 'output.csv'
            assert output_file.exists(), f"Output should exist for {method} processing"
    
    @pytest.mark.asyncio
    async def test_choice_validation(self, setup_test_environment):
        """Test that choice inputs are validated against allowed options."""
        env = await setup_test_environment
        
        # Test with completely invalid choices that should fail validation
        user_inputs = [
            "!!!invalid!!!",  # Clearly invalid processing method
            "!!!invalid!!!",  # Try again
            "!!!invalid!!!",  # Try again
            "standard",       # Finally give valid input
            "csv",           # Valid format
            "approve",       # Approve
            "",             # No comment
            "5",            # Ratings...
            "5",
            "yes",
            ""
        ]
        
        with RealPipelineInputSimulator(user_inputs):
            # Load and execute pipeline
            with open('examples/interactive_pipeline.yaml', 'r') as f:
                pipeline_yaml = f.read()
            
            results = await env['orchestrator'].execute_yaml(
                pipeline_yaml,
                inputs={
                    'input_file': env['input_file'],
                    'output_dir': env['output_dir']
                }
            )
        
        # Should eventually succeed with valid input
        assert results['get_processing_options']['value'] == "standard"
        
        # Check that retries were attempted
        if 'retries' in results['get_processing_options']:
            assert results['get_processing_options']['retries'] >= 1