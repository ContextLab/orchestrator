# Test Failure Summary

## Overview
Total test files analyzed: 52

## Test Status Breakdown

### ✓ PASSED (32 tests)
- tests/integration/test_auto_resolution_quality.py
- tests/integration/test_edge_cases.py
- tests/integration/test_full_integration.py
- tests/integration/test_models_comprehensive.py
- tests/integration/test_pipeline_real_auto.py
- tests/integration/test_quick_real_models.py
- tests/integration/test_real_models.py
- tests/integration/test_real_world_pipelines.py
- tests/integration/test_research_assistant_with_report.py
- tests/integration/test_user_interaction_tools.py
- tests/integration/test_yaml_compilation.py
- tests/local/test_simple_ollama.py
- tests/test_adaptive_checkpoint.py
- tests/test_auto_tag_yaml_parser.py
- tests/test_auto_tags_documentation.py
- tests/test_control_flow_real.py
- tests/test_control_flow_simple.py
- tests/test_core_pipeline_coverage.py
- tests/test_declarative_framework_old/test_phase1_core_engine.py
- tests/test_error_handling.py
- tests/test_mcp_tools_examples.py
- tests/test_mcp_tools.py
- tests/test_pipeline_extended.py
- tests/test_pipeline_recursion_simple.py
- tests/test_pipeline.py
- tests/test_report_tools.py
- tests/test_resource_allocator.py
- tests/test_task_comprehensive.py
- tests/test_task.py
- tests/test_validation_tool.py

### ✗ FAILED (10 tests)

1. **tests/local/test_ollama_local.py**
   - Failure in: `test_auto_model_detection`
   - Error: `ValueError: No AI model available for ambiguity resolution. A real model must be provided.`
   - Location: `src/orchestrator/compiler/ambiguity_resolver.py:64`

2. **tests/test_ambiguity_resolver.py**
   - Failure in: `test_resolver_with_model_registry`
   - Error: Assertion failed - `resolver.model is None` when it should have selected a model

3. **tests/test_domain_routing.py**
   - Generic test errors detected

4. **tests/test_failing_components.py**
   - Generic test errors detected

5. **tests/test_intelligent_routing_comprehensive.py**
   - Generic test errors detected

6. **tests/test_intelligent_routing.py**
   - Generic test errors detected

7. **tests/test_load_balancer.py**
   - Generic test errors detected

8. **tests/test_multimodal_tools.py**
   - Failure in: `test_image_analysis_describe`
   - Error: `OpenAI API error: openai.resources.chat.completions.completions.Completions.create() got multiple values for keyword argument 'messages'`
   - Location: `src.orchestrator.tools.multimodal_tools:multimodal_tools.py:220`

9. **tests/test_research_assistant_example.py**
   - Generic test errors detected

### ⏱️ TIMEOUT (12 tests - exceeded 30s)
1. **tests/integration/test_basic_yaml_execution.py**
2. **tests/integration/test_minimal_yaml_integration.py**
3. **tests/integration/test_real_web_tools.py**
4. **tests/integration/test_simple_pipeline_integration.py**
5. **tests/integration/test_simplified_yaml_pipelines.py**
6. **tests/integration/test_tools_real_world.py**
7. **tests/test_adapters.py**
8. **tests/test_control_flow.py**
9. **tests/test_model_routing_documentation.py**
10. **tests/test_pipeline_recursion_tools.py**
11. **tests/test_process_pool_execution.py**
12. **tests/test_tool_catalog_documentation.py**

## Common Failure Patterns

1. **Model Selection Issues**: Multiple tests failing due to model selection/initialization problems
   - AmbiguityResolver not finding available models
   - Model registry not properly selecting models

2. **API Parameter Conflicts**: OpenAI API calls failing due to duplicate 'messages' parameter

3. **Long-Running Tests**: Many integration tests timing out at 30 seconds, likely due to:
   - Real API calls taking too long
   - Recursive operations not terminating
   - Process pool execution hanging

## Next Steps
To get detailed failure information for each failed test, run them individually with verbose output:
```bash
python -m pytest <test_file> -vvs --tb=long
```