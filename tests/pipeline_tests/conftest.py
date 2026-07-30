"""Pytest configuration and fixtures for pipeline testing."""

import asyncio
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Optional

import pytest
import yaml

from orchestrator import Orchestrator, init_models
from orchestrator.models.model_registry import ModelRegistry
from orchestrator.utils.api_keys_flexible import load_api_keys_optional


@pytest.fixture(scope="session")
def pipeline_model_registry() -> ModelRegistry:
    """
    Provide a cost-optimized model registry for pipeline tests.
    
    Prioritizes faster, cheaper models for testing while ensuring
    real API functionality.
    
    Returns:
        ModelRegistry: Registry with cost-optimized model configuration
    """
    print("\n>> Pipeline tests: Initializing cost-optimized models...")
    registry = init_models()
    
    available_models = registry.list_models()
    print(f">> Pipeline tests: Found {len(available_models)} models: {available_models}")
    
    # Prioritize cost-effective models for testing
    cost_effective_models = []
    for model in available_models:
        model_lower = model.lower()
        # Prefer smaller, faster models for testing
        if any(pattern in model_lower for pattern in [
            'gpt-4o-mini', 'gpt-3.5', 'claude-3-haiku', 'gemini-1.5-flash'
        ]):
            cost_effective_models.append(model)
    
    if cost_effective_models:
        print(f">> Pipeline tests: Using cost-effective models: {cost_effective_models}")
    else:
        print(">> Pipeline tests: No cost-effective models found, using all available")
    
    return registry


@pytest.fixture(scope="session") 
def pipeline_orchestrator(pipeline_model_registry) -> Orchestrator:
    """
    Provide an orchestrator instance configured for pipeline testing.
    
    Args:
        pipeline_model_registry: Cost-optimized model registry
        
    Returns:
        Orchestrator: Configured orchestrator instance
    """
    print(">> Pipeline tests: Creating orchestrator instance...")
    return Orchestrator(model_registry=pipeline_model_registry)


@pytest.fixture
def temp_output_dir():
    """
    Provide a temporary directory for pipeline outputs.
    
    Yields:
        Path: Temporary directory path that gets cleaned up after test
    """
    with tempfile.TemporaryDirectory(prefix="pipeline_test_") as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_pipeline_yaml():
    """
    Provide a sample YAML pipeline for testing.
    
    Returns:
        str: YAML content for a simple test pipeline
    """
    return """
name: Test Pipeline
description: Simple test pipeline for validation

tasks:
  - name: simple_task
    type: llm
    model: "{{ model | default('anthropic:claude-sonnet-4-20250514') }}"
    template: |
      Generate a single sentence about {{ topic }}.
      Keep it brief and factual.
    inputs:
      topic: "{{ topic | default('artificial intelligence') }}"
    
  - name: verification_task
    type: llm 
    model: "{{ model | default('anthropic:claude-sonnet-4-20250514') }}"
    template: |
      Review this text and respond with "VALID" if it's a proper sentence, or "INVALID" if not:
      
      {{ simple_task.content }}
    dependencies:
      - simple_task

outputs:
  main_content: "{{ simple_task.content }}"
  validation_result: "{{ verification_task.content }}"
"""


@pytest.fixture 
def complex_pipeline_yaml():
    """
    Provide a more complex YAML pipeline for advanced testing.
    
    Returns:
        str: YAML content for a complex test pipeline
    """
    return """
name: Complex Test Pipeline
description: Multi-step pipeline with conditions and loops

tasks:
  - name: data_generator
    type: llm
    model: "{{ model }}"
    template: |
      Generate exactly {{ count }} items in this format:
      Item 1: [description]
      Item 2: [description]
      ...
      
      Each item should be about {{ theme }}.
    inputs:
      count: "{{ item_count | default(3) }}"
      theme: "{{ theme | default('technology') }}"
      model: "{{ model | default('gpt-4o-mini') }}"
  
  - name: item_processor
    type: for_each
    items: "{{ data_generator.content | extract_lines }}"
    task:
      name: process_item
      type: llm
      model: "{{ model }}"
      template: |
        Analyze this item and provide a one-word category:
        {{ item }}
        
        Response format: CATEGORY: [word]
      inputs:
        model: "{{ model | default('gpt-4o-mini') }}"
    dependencies:
      - data_generator
  
  - name: summary_generator
    type: llm
    model: "{{ model }}"
    template: |
      Summarize the analysis results in exactly one sentence:
      
      Original data: {{ data_generator.content }}
      Categories: {{ item_processor.results | join(', ') }}
    inputs:
      model: "{{ model | default('gpt-4o-mini') }}"
    dependencies:
      - item_processor
    condition: "{{ item_processor.results | length > 0 }}"

outputs:
  generated_data: "{{ data_generator.content }}"
  processed_items: "{{ item_processor.results }}"
  final_summary: "{{ summary_generator.content }}"
  task_count: "{{ tasks | length }}"
"""


@pytest.fixture
def pipeline_inputs():
    """
    Provide default inputs for pipeline testing.
    
    Returns:
        Dict[str, Any]: Default pipeline inputs
    """
    return {
        "topic": "machine learning",
        "theme": "artificial intelligence", 
        "item_count": 2,
        "model": "anthropic:claude-sonnet-4-20250514"  # Available model
    }


@pytest.fixture
async def async_temp_file():
    """
    Provide a temporary file for async operations.
    
    Yields:
        Path: Temporary file path
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        temp_path = Path(f.name)
    
    try:
        yield temp_path
    finally:
        if temp_path.exists():
            temp_path.unlink()


@pytest.fixture
def performance_baseline():
    """
    Provide performance baseline expectations for testing.
    
    Returns:
        Dict[str, Any]: Performance thresholds and expectations
    """
    return {
        "max_execution_time": 300,  # 5 minutes max
        "max_cost_per_task": 0.10,  # $0.10 per task max
        "min_success_rate": 0.95,   # 95% success rate minimum
        "max_memory_mb": 500,       # 500MB memory max
        "api_timeout": 60,          # 60 second API timeout
    }


@pytest.fixture(scope="session")
def test_api_keys():
    """
    Load API keys for testing, skipping tests if not available.
    
    Returns:
        Dict[str, str]: Available API keys
    """
    try:
        api_keys = load_api_keys_optional()
        return api_keys
    except Exception as e:
        pytest.skip(f"API keys not available: {e}")


@pytest.fixture
def mock_execution_metadata():
    """
    Provide mock execution metadata for testing.
    
    Returns:
        Dict[str, Any]: Mock execution metadata
    """
    return {
        "start_time": time.time(),
        "total_time": 45.2,
        "task_count": 3,
        "success_count": 3,
        "failure_count": 0,
        "total_cost": 0.05,
        "model_calls": 3,
        "tokens_used": 1500,
        "memory_peak_mb": 85.3
    }


@pytest.fixture
def pipeline_test_config():
    """
    Provide configuration for pipeline testing.
    
    Returns:
        Dict[str, Any]: Test configuration settings
    """
    return {
        "enable_performance_tracking": True,
        "enable_cost_tracking": True,
        "parallel_execution": False,  # Safer for tests
        "timeout_seconds": 180,
        "retry_attempts": 2,
        "save_outputs": True,
        "validate_templates": True,
        "check_dependencies": True
    }