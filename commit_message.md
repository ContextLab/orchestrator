# Commit Message for Issue #123

feat: Complete ambiguity resolver implementation with structured output and retry logic

## Summary
Implemented a comprehensive ambiguity resolver system for AUTO tags in YAML pipelines with structured output support, intelligent type inference, and robust error handling.

## Changes

### Phase 1: Core Implementation
- Removed MockAmbiguityResolver and integrated real resolver into YAMLCompiler
- Fixed template resolution to happen before AUTO tag processing
- Added nested input support in _merge_defaults_with_context

### Phase 2: Structured Output Support
- Created StructuredAmbiguityResolver with Pydantic models for typed responses
- Implemented intelligent type inference for boolean, number, list, choice, and string types
- Fixed mixed template processing with _process_mixed_templates method
- Added comprehensive test suite (9 tests) for structured resolution

### Phase 3: Error Handling & Retry Logic
- Created utils.py with async_retry decorator and transient error detection
- Added exponential backoff retry logic to all model API calls
- Improved error messages with context preservation
- Added comprehensive error handling tests (10 tests)

## Files Changed
- src/orchestrator/compiler/structured_ambiguity_resolver.py (new)
- src/orchestrator/compiler/utils.py (new)
- src/orchestrator/compiler/yaml_compiler.py (modified)
- src/orchestrator/compiler/ambiguity_resolver.py (modified)
- tests/test_structured_ambiguity_resolver.py (new)
- tests/test_template_resolution.py (modified)
- tests/test_ambiguity_resolver_errors.py (new)

## Test Results
All 25 tests passing:
- Structured ambiguity resolver: 9 tests
- Template resolution: 6 tests
- Error handling: 10 tests

Fixes #123