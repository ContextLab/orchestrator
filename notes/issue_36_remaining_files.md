# Remaining Files to Review for Issue #36

Based on the original checklist and progress updates, here are the files that still need to be reviewed:

## Source Files (src/orchestrator/) - Mostly done, just need to verify

### Remaining to check:
- [ ] ./src/orchestrator/__init__.py
- [ ] ./src/orchestrator/__main__.py
- [ ] ./src/orchestrator/adapters/__init__.py
- [ ] ./src/orchestrator/compiler/__init__.py
- [ ] ./src/orchestrator/compiler/schema_validator.py
- [ ] ./src/orchestrator/compiler/template_renderer.py
- [ ] ./src/orchestrator/compiler/yaml_compiler.py
- [ ] ./src/orchestrator/compiler/ambiguity_resolver.py
- [ ] ./src/orchestrator/compiler/auto_tag_yaml_parser.py
- [ ] ./src/orchestrator/control_systems/__init__.py
- [ ] ./src/orchestrator/core/__init__.py
- [ ] ./src/orchestrator/engine/__init__.py
- [ ] ./src/orchestrator/engine/advanced_executor.py
- [ ] ./src/orchestrator/engine/auto_resolver.py
- [ ] ./src/orchestrator/engine/declarative_engine.py
- [ ] ./src/orchestrator/engine/enhanced_executor.py
- [ ] ./src/orchestrator/engine/pipeline_spec.py
- [ ] ./src/orchestrator/engine/runtime_auto_resolver.py
- [ ] ./src/orchestrator/engine/task_executor.py
- [ ] ./src/orchestrator/executor/__init__.py
- [ ] ./src/orchestrator/integrations/__init__.py
- [ ] ./src/orchestrator/integrations/anthropic_model.py
- [ ] ./src/orchestrator/integrations/google_model.py
- [ ] ./src/orchestrator/integrations/huggingface_model.py
- [ ] ./src/orchestrator/integrations/lazy_huggingface_model.py
- [ ] ./src/orchestrator/integrations/lazy_ollama_model.py
- [ ] ./src/orchestrator/integrations/ollama_model.py
- [ ] ./src/orchestrator/integrations/openai_model.py
- [ ] ./src/orchestrator/models/__init__.py
- [ ] ./src/orchestrator/models/anthropic_model.py
- [ ] ./src/orchestrator/models/auto_register.py
- [ ] ./src/orchestrator/models/openai_model.py
- [ ] ./src/orchestrator/state/__init__.py
- [ ] ./src/orchestrator/state/adaptive_checkpoint.py
- [ ] ./src/orchestrator/state/backends.py
- [ ] ./src/orchestrator/state/simple_state_manager.py
- [ ] ./src/orchestrator/state/state_manager.py
- [ ] ./src/orchestrator/tools/__init__.py
- [ ] ./src/orchestrator/tools/base.py
- [ ] ./src/orchestrator/tools/data_tools.py
- [ ] ./src/orchestrator/tools/discovery.py
- [ ] ./src/orchestrator/tools/structured_output_handler.py
- [ ] ./src/orchestrator/tools/system_tools.py
- [ ] ./src/orchestrator/tools/update_models.py
- [ ] ./src/orchestrator/tools/web_tools.py
- [ ] ./src/orchestrator/utils/__init__.py
- [ ] ./src/orchestrator/utils/api_keys.py
- [ ] ./src/orchestrator/utils/model_config_loader.py
- [ ] ./src/orchestrator/utils/model_utils.py

## Test Files (tests/) - Many still need review

### Critical test files not yet reviewed:
- [ ] ./tests/__init__.py
- [ ] ./tests/conftest.py
- [ ] ./tests/examples/__init__.py (Part of Issue #93)
- [ ] ./tests/integration/test_auto_resolution_quality.py
- [ ] ./tests/integration/test_code_optimization.py
- [ ] ./tests/integration/test_input_agnostic.py
- [ ] ./tests/integration/test_research_assistant_with_report.py
- [ ] ./tests/integration/test_research_assistant.py
- [ ] ./tests/integration/test_successful_processing.py
- [ ] ./tests/integration/test_yaml_compilation.py
- [ ] ./tests/local/test_ollama_local.py
- [ ] ./tests/local/test_simple_ollama.py
- [ ] ./tests/test_declarative_framework_old/__init__.py
- [ ] ./tests/test_declarative_framework_old/test_phase1_core_engine.py
- [ ] ./tests/test_declarative_framework_old/test_phase2_tool_discovery.py
- [ ] ./tests/test_declarative_framework_old/test_phase3_advanced_features.py
- [ ] ./tests/test_model_registry_debug.py
- [ ] ./tests/test_model_registry_final.py

### Snippet tests (need to check for MockModel references):
- [ ] ./tests/snippet_tests/__init__.py
- [ ] Most snippet test files (test_snippets_batch_*.py) - Already noted in Issue #71

## Priority Actions:
1. Complete review of remaining source files (should be mostly clean)
2. Review remaining integration tests
3. Check local test files
4. Verify old declarative framework tests
5. Update final checklist on issue #36