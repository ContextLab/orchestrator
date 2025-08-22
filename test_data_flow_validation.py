#!/usr/bin/env python3
"""Test the DataFlowValidator implementation for Issue #241 Stream 4."""

import asyncio
import sys
import yaml
from typing import Dict, Any

# Add the src directory to the path
sys.path.insert(0, '/Users/jmanning/orchestrator/src')

from orchestrator.validation.data_flow_validator import DataFlowValidator
from orchestrator.compiler.yaml_compiler import YAMLCompiler


def test_basic_data_flow_validation():
    """Test basic data flow validation functionality."""
    print("Testing basic data flow validation...")
    
    # Create a simple pipeline with data flow
    pipeline_yaml = """
id: test_data_flow
name: Test Data Flow Pipeline
inputs:
  topic: "machine learning"
  
steps:
  - id: fetch_data
    action: web_search
    parameters:
      query: "{{ inputs.topic }} research papers"
      max_results: 10
      
  - id: analyze_data
    action: llm_tools
    depends_on: fetch_data
    parameters:
      prompt: "Analyze these search results: {{ fetch_data.results }}"
      model: "gpt-4"
      
  - id: invalid_reference
    action: filesystem
    parameters:
      action: write
      path: "/tmp/output.txt"
      content: "{{ nonexistent_task.result }}"  # This should cause an error
"""
    
    # Parse the YAML
    pipeline_def = yaml.safe_load(pipeline_yaml)
    
    # Create validator
    validator = DataFlowValidator(development_mode=False)
    
    # Validate data flow
    result = validator.validate_pipeline_data_flow(pipeline_def)
    
    print(f"Validation result: {result.summary()}")
    print(f"Valid: {result.valid}")
    print(f"Errors: {len(result.errors)}")
    print(f"Warnings: {len(result.warnings)}")
    
    # Print errors
    if result.errors:
        print("\nErrors found:")
        for error in result.errors:
            print(f"  - {error}")
    
    # Print warnings
    if result.warnings:
        print("\nWarnings found:")
        for warning in result.warnings:
            print(f"  - {warning}")
    
    # Print data flow graph
    print("\nData flow graph:")
    for task_id, deps in result.data_flow_graph.items():
        if deps:
            print(f"  {task_id} depends on: {', '.join(sorted(deps))}")
        else:
            print(f"  {task_id} has no dependencies")
    
    return result


def test_development_mode():
    """Test data flow validation in development mode."""
    print("\n" + "="*50)
    print("Testing data flow validation in development mode...")
    
    # Same pipeline but with development mode enabled
    pipeline_yaml = """
id: test_dev_mode
name: Test Development Mode
inputs:
  topic: "machine learning"
  
steps:
  - id: fetch_data
    action: web_search
    parameters:
      query: "{{ inputs.topic }} research papers"
      max_results: 10
      
  - id: analyze_data
    action: llm_tools
    depends_on: fetch_data
    parameters:
      prompt: "Analyze these search results: {{ fetch_data.results }}"
      model: "gpt-4"
      
  - id: reference_unknown_task
    action: filesystem
    parameters:
      action: write
      path: "/tmp/output.txt"
      content: "{{ unknown_task.result }}"  # This should be a warning in dev mode
"""
    
    # Parse the YAML
    pipeline_def = yaml.safe_load(pipeline_yaml)
    
    # Create validator in development mode
    validator = DataFlowValidator(development_mode=True)
    
    # Validate data flow
    result = validator.validate_pipeline_data_flow(pipeline_def)
    
    print(f"Validation result: {result.summary()}")
    print(f"Valid: {result.valid}")
    print(f"Errors: {len(result.errors)}")
    print(f"Warnings: {len(result.warnings)}")
    
    # Print errors and warnings
    if result.errors:
        print("\nErrors found:")
        for error in result.errors:
            print(f"  - {error}")
    
    if result.warnings:
        print("\nWarnings found:")
        for warning in result.warnings:
            print(f"  - {warning}")
    
    return result


def test_complex_data_flow():
    """Test complex data flow validation with multiple dependencies."""
    print("\n" + "="*50)
    print("Testing complex data flow validation...")
    
    pipeline_yaml = """
id: complex_pipeline
name: Complex Pipeline with Multiple Dependencies
inputs:
  source_url: "https://example.com"
  output_format: "json"
  
steps:
  - id: fetch_source
    action: web_tools
    parameters:
      url: "{{ inputs.source_url }}"
      method: "GET"
      
  - id: extract_data
    action: llm_tools
    depends_on: fetch_source
    parameters:
      prompt: "Extract key information from: {{ fetch_source.content }}"
      model: "gpt-3.5-turbo"
      
  - id: validate_data
    action: data_tools
    depends_on: extract_data
    parameters:
      data: "{{ extract_data.result }}"
      schema_check: true
      
  - id: format_output
    action: data_tools
    depends_on: [extract_data, validate_data]
    parameters:
      data: "{{ extract_data.result }}"
      format: "{{ inputs.output_format }}"
      validation_results: "{{ validate_data.is_valid }}"
      
  - id: save_results
    action: filesystem
    depends_on: format_output
    parameters:
      action: write
      path: "/tmp/results.{{ inputs.output_format }}"
      content: "{{ format_output.formatted_data }}"
      
  - id: circular_reference  # This should cause a circular dependency if it tries to reference save_results
    action: llm_tools
    depends_on: save_results
    parameters:
      prompt: "Summarize the results: {{ save_results.status }}"
      context: "{{ fetch_source.metadata }}"  # Valid reference
"""
    
    # Parse the YAML
    pipeline_def = yaml.safe_load(pipeline_yaml)
    
    # Create validator
    validator = DataFlowValidator(development_mode=False)
    
    # Validate data flow
    result = validator.validate_pipeline_data_flow(pipeline_def)
    
    print(f"Validation result: {result.summary()}")
    print(f"Valid: {result.valid}")
    print(f"Errors: {len(result.errors)}")
    print(f"Warnings: {len(result.warnings)}")
    
    # Print data flow graph
    print("\nData flow graph:")
    for task_id, deps in result.data_flow_graph.items():
        if deps:
            print(f"  {task_id} depends on: {', '.join(sorted(deps))}")
        else:
            print(f"  {task_id} has no dependencies")
    
    # Print any issues
    if result.errors:
        print("\nErrors found:")
        for error in result.errors:
            print(f"  - {error}")
    
    if result.warnings:
        print("\nWarnings found:")
        for warning in result.warnings:
            print(f"  - {warning}")
    
    return result


async def test_yaml_compiler_integration():
    """Test the integration with YAMLCompiler."""
    print("\n" + "="*50)
    print("Testing YAMLCompiler integration...")
    
    pipeline_yaml = """
id: integration_test
name: YAMLCompiler Integration Test
inputs:
  search_term: "artificial intelligence"
  
steps:
  - id: search_step
    action: web_search
    parameters:
      query: "{{ inputs.search_term }} latest developments"
      max_results: 5
      
  - id: analysis_step
    action: llm_tools
    depends_on: search_step
    parameters:
      prompt: "Analyze these search results: {{ search_step.results }}"
      model: "gpt-4"
      
  - id: broken_step
    action: filesystem
    parameters:
      action: write
      path: "/tmp/analysis.txt"
      content: "{{ invalid_task.output }}"  # This should fail validation
"""
    
    try:
        # Create compiler with data flow validation enabled
        compiler = YAMLCompiler(
            development_mode=False,  # Strict mode
            validate_data_flow=True,
            enable_validation_report=True
        )
        
        # This should fail due to data flow validation errors
        pipeline = await compiler.compile(
            pipeline_yaml,
            context={"search_term": "AI research"}
        )
        
        print("ERROR: Compilation should have failed but didn't!")
        return False
        
    except Exception as e:
        print(f"Compilation failed as expected: {e}")
        
        # Check validation report
        if compiler.validation_report:
            print("\nValidation report summary:")
            print(f"  Total issues: {compiler.validation_report.stats.total_issues}")
            print(f"  Errors: {compiler.validation_report.stats.errors}")
            print(f"  Warnings: {compiler.validation_report.stats.warnings}")
            
            # Print data flow issues
            data_flow_issues = [issue for issue in compiler.validation_report.issues if issue.category == "data_flow"]
            if data_flow_issues:
                print(f"\nData flow issues ({len(data_flow_issues)}):")
                for issue in data_flow_issues:
                    print(f"  - {issue.severity.value.upper()}: {issue.message}")
        
        return True


async def test_development_mode_integration():
    """Test YAMLCompiler integration in development mode."""
    print("\n" + "="*50)
    print("Testing YAMLCompiler integration in development mode...")
    
    pipeline_yaml = """
id: dev_mode_test
name: Development Mode Test
inputs:
  topic: "machine learning"
  
steps:
  - id: research_step
    action: web_search
    parameters:
      query: "{{ inputs.topic }} research"
      max_results: 10
      
  - id: analysis_step
    action: llm_tools
    depends_on: research_step
    parameters:
      prompt: "Analyze: {{ research_step.results }}"
      model: "gpt-4"
      
  - id: questionable_step
    action: filesystem
    parameters:
      action: write
      path: "/tmp/output.txt"
      content: "{{ maybe_missing_task.result }}"  # This should be a warning in dev mode
"""
    
    try:
        # Create compiler with development mode enabled
        compiler = YAMLCompiler(
            development_mode=True,  # Development mode
            validate_data_flow=True,
            enable_validation_report=True
        )
        
        # This should succeed with warnings in development mode
        pipeline = await compiler.compile(
            pipeline_yaml,
            context={"topic": "AI research"}
        )
        
        print(f"Compilation succeeded in development mode!")
        print(f"Pipeline ID: {pipeline.id}")
        print(f"Tasks: {len(pipeline.tasks)}")
        
        # Check validation report
        if compiler.validation_report:
            print("\nValidation report summary:")
            print(f"  Total issues: {compiler.validation_report.stats.total_issues}")
            print(f"  Errors: {compiler.validation_report.stats.errors}")
            print(f"  Warnings: {compiler.validation_report.stats.warnings}")
            
            # Print data flow issues
            data_flow_issues = [issue for issue in compiler.validation_report.issues if issue.category == "data_flow"]
            if data_flow_issues:
                print(f"\nData flow issues ({len(data_flow_issues)}):")
                for issue in data_flow_issues:
                    print(f"  - {issue.severity.value.upper()}: {issue.message}")
        
        return True
        
    except Exception as e:
        print(f"Unexpected compilation failure: {e}")
        return False


def main():
    """Run all tests."""
    print("Running DataFlowValidator tests for Issue #241 Stream 4")
    print("="*60)
    
    # Test basic validation
    result1 = test_basic_data_flow_validation()
    
    # Test development mode
    result2 = test_development_mode()
    
    # Test complex data flow
    result3 = test_complex_data_flow()
    
    # Test integration with YAMLCompiler
    success1 = asyncio.run(test_yaml_compiler_integration())
    success2 = asyncio.run(test_development_mode_integration())
    
    print("\n" + "="*60)
    print("Test Summary:")
    print(f"  Basic validation: {'PASS' if not result1.valid else 'UNEXPECTED PASS'}")
    print(f"  Development mode: {'PASS' if result2.valid else 'FAIL'}")
    print(f"  Complex data flow: {'PASS' if not result3.has_errors else 'FAIL'}")
    print(f"  Compiler integration (strict): {'PASS' if success1 else 'FAIL'}")
    print(f"  Compiler integration (dev): {'PASS' if success2 else 'FAIL'}")
    
    print("\nDataFlowValidator implementation test completed!")


if __name__ == "__main__":
    main()