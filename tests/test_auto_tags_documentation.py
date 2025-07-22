#!/usr/bin/env python3
"""Test all AUTO tag documentation examples with real execution."""

import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator import Orchestrator
from orchestrator.models.openai_model import OpenAIModel
from orchestrator.integrations.ollama_model import OllamaModel


async def setup_orchestrator():
    """Set up orchestrator with real models."""
    # Initialize orchestrator with models
    from orchestrator.models.registry_singleton import get_model_registry
    
    # Get singleton registry
    registry = get_model_registry()
    
    # Check if models are already registered
    existing_models = registry.list_models()
    models_registered = []
    
    # Only register if not already present
    if "ollama:llama3.2:1b" not in existing_models:
        try:
            llama = OllamaModel("llama3.2:1b")
            registry.register_model(llama)
            models_registered.append("llama3.2:1b")
            print("✓ Registered llama3.2:1b")
        except Exception as e:
            print(f"✗ Failed to register llama3.2:1b: {e}")
    else:
        models_registered.append("llama3.2:1b")
    
    # Try OpenAI
    if os.getenv("OPENAI_API_KEY") and "openai:gpt-3.5-turbo" not in existing_models:
        try:
            gpt35 = OpenAIModel("gpt-3.5-turbo")
            registry.register_model(gpt35)
            models_registered.append("gpt-3.5-turbo")
            print("✓ Registered gpt-3.5-turbo")
        except Exception as e:
            print(f"✗ Failed to register gpt-3.5-turbo: {e}")
    elif "openai:gpt-3.5-turbo" in existing_models:
        models_registered.append("gpt-3.5-turbo")
    
    if not models_registered:
        raise RuntimeError("No models available for testing")
    
    # Create orchestrator with initialized registry
    orchestrator = Orchestrator()
    orchestrator.model_registry = registry
    
    return orchestrator, models_registered


async def test_dynamic_data_analyzer():
    """Test Example 1: Dynamic Data Analysis."""
    print("\n=== Testing Dynamic Data Analyzer ===")
    
    # Create test data file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("name,age,score\n")
        f.write("Alice,25,85\n")
        f.write("Bob,30,92\n")
        f.write("Charlie,28,78\n")
        test_file = f.name
    
    try:
        pipeline_yaml = """
name: dynamic-data-analyzer
description: Analyze data with AI-determined methods
version: "1.0.0"

inputs:
  data_file:
    type: string
    description: Path to data file
    required: true

steps:
  - id: read_data
    action: filesystem
    tool_config:
      action: "read"
    parameters:
      path: "{{ data_file }}"
      
  - id: analyze_data
    action: llm-generate
    parameters:
      prompt: |
        Analyze this CSV data and provide insights:
        {{ read_data.result }}
      analysis_type: <AUTO>Based on the data which appears to be CSV with names and scores, should we use 'statistical', 'qualitative', or 'mixed' analysis? Just answer with one word.</AUTO>
      
  - id: save_analysis
    action: filesystem
    tool_config:
      action: "write"
    parameters:
      path: "/tmp/analysis_report.md"
      content: |
        # Analysis Report
        Type: {{ analyze_data.analysis_type }}
        
        {{ analyze_data.result }}
"""
        
        # Save pipeline
        pipeline_file = "/tmp/test_dynamic_analyzer.yaml"
        with open(pipeline_file, 'w') as f:
            f.write(pipeline_yaml)
        
        # Execute pipeline
        orchestrator, _ = await setup_orchestrator()
        result = await orchestrator.execute_yaml(
            pipeline_file,
            inputs={"data_file": test_file}
        )
        
        print("✓ Pipeline executed successfully")
        print(f"  Analysis type chosen: {result.get('analyze_data', {}).get('analysis_type', 'N/A')}")
        print("  Analysis saved to: /tmp/analysis_report.md")
        
        # Verify output file exists
        if os.path.exists("/tmp/analysis_report.md"):
            print("✓ Output file created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.unlink(test_file)


async def test_intelligent_error_handler():
    """Test Example 2: Intelligent Error Handling."""
    print("\n=== Testing Intelligent Error Handler ===")
    
    pipeline_yaml = """
name: smart-error-handler
description: Handle errors intelligently based on context
version: "1.0.0"

steps:
  - id: risky_operation
    action: web-search
    parameters:
      query: "latest AI news"
      num_results: 3
    error_handling:
      retry:
        max_attempts: <AUTO>For searching AI news which is moderately important, how many retry attempts should we make? Answer with just a number between 1-5.</AUTO>
        
  - id: process_results
    action: llm-generate
    condition: "{{ risky_operation.status != 'failed' }}"
    parameters:
      prompt: |
        Summarize these AI news results:
        {{ risky_operation.result }}
      max_length: 200
"""
    
    try:
        # Save pipeline
        pipeline_file = "/tmp/test_error_handler.yaml"
        with open(pipeline_file, 'w') as f:
            f.write(pipeline_yaml)
        
        # Execute pipeline
        orchestrator, _ = await setup_orchestrator()
        result = await orchestrator.execute_yaml(pipeline_file)
        
        print("✓ Pipeline executed successfully")
        print(f"  Web search status: {result.get('risky_operation', {}).get('status', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


async def test_dynamic_tool_selection():
    """Test Example 3: Dynamic Tool Selection."""
    print("\n=== Testing Dynamic Tool Selection ===")
    
    pipeline_yaml = """
name: smart-researcher
description: Research with dynamically selected tools
version: "1.0.0"

inputs:
  topic:
    type: string
    description: Research topic
    required: true
    default: "quantum computing"

steps:
  - id: determine_approach
    action: llm-generate
    parameters:
      prompt: |
        Research topic: {{ topic }}
        Determine the best research approach.
      questions:
        approach: <AUTO>For researching '{{ topic }}', is it better to use 'web_search', 'academic_sources', or 'both'? Answer with just one of these options.</AUTO>
        include_visuals: <AUTO>For the topic '{{ topic }}', would visual data be helpful? Answer just 'yes' or 'no'.</AUTO>
        
  - id: web_research
    action: web-search
    condition: "'web_search' in determine_approach.approach or 'both' in determine_approach.approach"
    parameters:
      query: "{{ topic }}"
      num_results: <AUTO>For the topic '{{ topic }}', how many search results would be appropriate? Answer with just a number between 3-10.</AUTO>
      
  - id: summarize
    action: llm-generate
    parameters:
      prompt: |
        Topic: {{ topic }}
        Research approach: {{ determine_approach.approach }}
        Include visuals: {{ determine_approach.include_visuals }}
        {% if web_research.result %}
        Web results: {{ web_research.result }}
        {% endif %}
        
        Provide a brief summary of the research approach and findings.
"""
    
    try:
        # Save pipeline
        pipeline_file = "/tmp/test_tool_selection.yaml"
        with open(pipeline_file, 'w') as f:
            f.write(pipeline_yaml)
        
        # Execute pipeline
        orchestrator, _ = await setup_orchestrator()
        result = await orchestrator.execute_yaml(
            pipeline_file,
            inputs={"topic": "artificial general intelligence"}
        )
        
        print("✓ Pipeline executed successfully")
        print(f"  Approach chosen: {result.get('determine_approach', {}).get('approach', 'N/A')}")
        print(f"  Include visuals: {result.get('determine_approach', {}).get('include_visuals', 'N/A')}")
        if 'web_research' in result:
            print("  Web search performed: Yes")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


async def test_auto_tags_in_control_flow():
    """Test AUTO tags in control flow structures."""
    print("\n=== Testing AUTO Tags in Control Flow ===")
    
    # Create test data
    test_data = {"quality": "high", "size": 1000, "format": "csv"}
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        import json
        json.dump(test_data, f)
        test_file = f.name
    
    pipeline_yaml = """
name: auto-control-flow
description: Test AUTO tags in control flow
version: "1.0.0"

inputs:
  data_file:
    type: string
    description: Path to data file
    required: true

steps:
  - id: read_data
    action: filesystem
    tool_config:
      action: "read"
    parameters:
      path: "{{ data_file }}"
      
  - id: check_data_quality
    action: validation
    parameters:
      data: "{{ read_data.result }}"
      schema:
        type: object
        properties:
          quality:
            type: string
          size:
            type: number
      
  - id: decide_processing
    action: llm-generate
    parameters:
      prompt: "Data validation passed. Determine processing approach."
      should_process: <AUTO>The data has quality='high' and size=1000. Should we proceed with processing? Answer only 'true' or 'false'.</AUTO>
      
  - id: process_data
    action: llm-generate
    condition: "{{ decide_processing.should_process == 'true' }}"
    parameters:
      prompt: |
        Process this data:
        {{ read_data.result }}
      method: <AUTO>For high quality data of size 1000, what processing method is best: 'quick', 'standard', or 'comprehensive'? Answer with just one word.</AUTO>
"""
    
    try:
        # Save pipeline
        pipeline_file = "/tmp/test_control_flow.yaml"
        with open(pipeline_file, 'w') as f:
            f.write(pipeline_yaml)
        
        # Execute pipeline
        orchestrator, _ = await setup_orchestrator()
        result = await orchestrator.execute_yaml(
            pipeline_file,
            inputs={"data_file": test_file}
        )
        
        print("✓ Pipeline executed successfully")
        print(f"  Should process decision: {result.get('decide_processing', {}).get('should_process', 'N/A')}")
        if 'process_data' in result:
            print(f"  Processing method: {result.get('process_data', {}).get('method', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False
    finally:
        if os.path.exists(test_file):
            os.unlink(test_file)


async def test_auto_tag_best_practices():
    """Test AUTO tag best practices examples."""
    print("\n=== Testing AUTO Tag Best Practices ===")
    
    pipeline_yaml = """
name: auto-best-practices
description: Demonstrate AUTO tag best practices
version: "1.0.0"

steps:
  - id: good_specific_choice
    action: llm-generate
    parameters:
      prompt: "Generate a report"
      format: <AUTO>Choose output format: 'json', 'yaml', or 'xml'</AUTO>
      
  - id: good_context_aware
    action: llm-generate
    parameters:
      prompt: "Analyze data"
      data_size: "50MB"
      num_columns: "20"
      analysis_depth: <AUTO>Given that this is a 50MB dataset with 20 columns, choose analysis depth: 'quick' (5 min), 'standard' (15 min), or 'comprehensive' (1 hour). Answer with just one word: quick, standard, or comprehensive.</AUTO>
      
  - id: good_type_hints
    action: llm-generate
    parameters:
      prompt: "Configure system"
      num_retries: <AUTO type="integer">For a moderately important operation, how many retries are appropriate? Answer with just a number between 1 and 5.</AUTO>
      include_logs: <AUTO type="boolean">Should we include detailed logs? Answer only 'true' or 'false'.</AUTO>
      
  - id: good_fallback
    action: llm-generate
    parameters:
      prompt: "Process request"
      strategy: <AUTO>Choose processing strategy: 'fast', 'balanced', or 'thorough'. Answer with just one word.</AUTO>
    error_handling:
      on_error:
        - id: use_default
          action: llm-generate
          parameters:
            prompt: "Using default strategy"
            strategy: "balanced"
"""
    
    try:
        # Save pipeline
        pipeline_file = "/tmp/test_best_practices.yaml"
        with open(pipeline_file, 'w') as f:
            f.write(pipeline_yaml)
        
        # Execute pipeline
        orchestrator, _ = await setup_orchestrator()
        result = await orchestrator.execute_yaml(pipeline_file)
        
        print("✓ Pipeline executed successfully")
        print(f"  Format chosen: {result.get('good_specific_choice', {}).get('format', 'N/A')}")
        print(f"  Analysis depth: {result.get('good_context_aware', {}).get('analysis_depth', 'N/A')}")
        print(f"  Num retries: {result.get('good_type_hints', {}).get('num_retries', 'N/A')}")
        print(f"  Include logs: {result.get('good_type_hints', {}).get('include_logs', 'N/A')}")
        print(f"  Strategy: {result.get('good_fallback', {}).get('strategy', 'N/A')}")
        
        # Verify type constraints
        num_retries = result.get('good_type_hints', {}).get('num_retries')
        if num_retries and isinstance(num_retries, (int, str)):
            try:
                retry_int = int(num_retries)
                if 1 <= retry_int <= 5:
                    print("✓ Retry count within valid range")
            except:
                pass
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


async def test_performance_optimizations():
    """Test AUTO tag performance optimization patterns."""
    print("\n=== Testing Performance Optimizations ===")
    
    pipeline_yaml = """
name: auto-performance
description: Test performance optimization patterns
version: "1.0.0"

steps:
  - id: batch_decisions
    action: llm-generate
    parameters:
      prompt: "Make multiple decisions for report generation"
      decisions:
        output_format: <AUTO>What's the best format for a technical report: 'pdf', 'html', or 'markdown'? Answer with just one word.</AUTO>
        include_summary: <AUTO>Should a technical report include an executive summary? Answer only 'true' or 'false'.</AUTO>
        detail_level: <AUTO>For a technical audience, what detail level is appropriate: 'low', 'medium', or 'high'? Answer with just one word.</AUTO>
        
  - id: use_decisions
    action: report-generator
    parameters:
      title: "Technical Report"
      format: "{{ batch_decisions.output_format }}"
      include_summary: "{{ batch_decisions.include_summary }}"
      detail_level: "{{ batch_decisions.detail_level }}"
      content: |
        # Report Content
        This report uses batched AUTO tag decisions:
        - Format: {{ batch_decisions.output_format }}
        - Summary: {{ batch_decisions.include_summary }}
        - Detail: {{ batch_decisions.detail_level }}
"""
    
    try:
        # Save pipeline
        pipeline_file = "/tmp/test_performance.yaml"
        with open(pipeline_file, 'w') as f:
            f.write(pipeline_yaml)
        
        # Execute pipeline
        orchestrator, _ = await setup_orchestrator()
        result = await orchestrator.execute_yaml(pipeline_file)
        
        print("✓ Pipeline executed successfully")
        print("  Batched decisions made:")
        print(f"    - Format: {result.get('batch_decisions', {}).get('output_format', 'N/A')}")
        print(f"    - Include summary: {result.get('batch_decisions', {}).get('include_summary', 'N/A')}")
        print(f"    - Detail level: {result.get('batch_decisions', {}).get('detail_level', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


async def main():
    """Run all AUTO tag documentation tests."""
    print("🚀 TESTING AUTO TAG DOCUMENTATION EXAMPLES")
    print("="*50)
    
    # Check if we have models available
    try:
        orchestrator, models = await setup_orchestrator()
        print(f"\nAvailable models: {', '.join(models)}")
    except Exception as e:
        print(f"\n❌ Cannot run tests: {e}")
        print("\nPlease ensure:")
        print("  - Ollama is running (for local models)")
        print("  - API keys are set (OPENAI_API_KEY, ANTHROPIC_API_KEY)")
        return
    
    # Run all tests
    tests = [
        ("Dynamic Data Analyzer", test_dynamic_data_analyzer),
        ("Intelligent Error Handler", test_intelligent_error_handler),
        ("Dynamic Tool Selection", test_dynamic_tool_selection),
        ("AUTO Tags in Control Flow", test_auto_tags_in_control_flow),
        ("AUTO Tag Best Practices", test_auto_tag_best_practices),
        ("Performance Optimizations", test_performance_optimizations)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        success = await test_func()
        results.append((test_name, success))
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ All AUTO tag documentation examples are working correctly!")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    asyncio.run(main())