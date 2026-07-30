# Backward Compatibility Guide

## Overview

The orchestrator refactor maintains **100% backward compatibility** for existing user code and pipeline definitions. All existing YAML pipelines, Python scripts, and API usage patterns continue to work without modification.

## Compatibility Guarantees

### ✅ Maintained APIs

All user-facing APIs remain unchanged:

```python
# All of these continue to work exactly as before
import orchestrator

# Initialize models
orchestrator.init_models()

# Compile pipelines (synchronous)
pipeline = orchestrator.compile("my_pipeline.yaml")
result = pipeline.run(topic="AI research")

# Compile pipelines (asynchronous)
pipeline = await orchestrator.compile_async("my_pipeline.yaml")
result = await pipeline.run_async(topic="AI research")
```

### ✅ Import Compatibility

All existing imports continue to work:

```python
# Core imports
from orchestrator import Orchestrator, Pipeline, Task, TaskStatus
from orchestrator import ModelRegistry, YAMLCompiler
from orchestrator import compile, compile_async, init_models

# Model integrations
from orchestrator import HuggingFaceModel, OllamaModel

# Error handling
from orchestrator import (
    OrchestratorError, PipelineError, TaskError, ModelError,
    ValidationError, ResourceError, StateError, ToolError
)
```

### ✅ YAML Pipeline Compatibility

All existing YAML pipeline definitions work without changes:

```yaml
# Classic pipeline structure (still supported)
id: my-pipeline
name: My Pipeline
description: Example pipeline

steps:
  - id: step1
    action: generate_text
    parameters:
      prompt: "Hello world"
      model: <AUTO>

  - id: step2  
    tool: filesystem
    action: write
    parameters:
      path: "output.txt"
      content: "{{ step1.result }}"
    dependencies:
      - step1

outputs:
  result: "{{ step2.path }}"
```

### ✅ Configuration Compatibility

All configuration files (`models.yaml`, API keys, environment variables) work unchanged.

## New Architecture Benefits

While maintaining backward compatibility, the new architecture provides:

### Enhanced Error Handling
- More descriptive error messages
- Better error recovery and retry mechanisms  
- Structured error reporting with context

### Improved Performance
- Parallel execution optimizations
- Better resource management
- Reduced memory footprint

### Extended Functionality
- Advanced control flow (conditional, loops, dynamic flows)
- Enhanced model routing and selection
- Improved tool integration
- Better state management and checkpointing

## Migration Benefits (Optional)

While not required, users can optionally take advantage of new features:

### 1. Enhanced Input/Output Definitions

```yaml
# Optional: Enhanced input definitions (backward compatible)
inputs:
  topic:
    type: string
    required: true
    description: "Research topic"
  depth:
    type: string
    default: "standard"
    choices: ["quick", "standard", "comprehensive"]

# Your existing parameter format still works too:
parameters:
  topic: "AI research"
```

### 2. Advanced Control Flow

```yaml
# New: Conditional execution
- id: conditional_step
  action: analyze_text
  condition: "{{ depth == 'comprehensive' }}"
  parameters:
    text: "{{ input_text }}"

# New: Loop constructs  
- id: process_items
  action: generate_text
  foreach: "{{ item_list }}"
  parameters:
    prompt: "Process {{ item }}"
```

### 3. Enhanced Model Selection

```yaml
# Existing AUTO tags still work
model: <AUTO>

# New: More specific AUTO targeting
model: <AUTO task="analysis">Best model for text analysis</AUTO>
model: <AUTO domain="creative">Best model for creative tasks</AUTO>
```

## Testing Your Existing Code

Use our backward compatibility test suite to verify your existing pipelines:

```bash
# Test basic compatibility
python examples/simple_compatibility_test.py

# Test your specific pipeline
python examples/compatibility_test.py your_pipeline.yaml
```

## Zero Breaking Changes Policy

The refactor follows a **zero breaking changes** policy:

1. **API Compatibility**: All public APIs maintain identical signatures
2. **YAML Compatibility**: All existing YAML pipeline definitions work unchanged  
3. **Configuration Compatibility**: All existing configuration files work unchanged
4. **Dependency Compatibility**: No new required dependencies for existing functionality
5. **Behavior Compatibility**: Existing behavior is preserved unless it was clearly buggy

## Troubleshooting

### If You Experience Issues

1. **Import Errors**: Verify that your Python path includes the orchestrator package
2. **Model Issues**: Run `orchestrator.init_models()` before using pipelines
3. **API Key Issues**: Ensure your API keys are set in environment variables or config files
4. **YAML Parsing**: Validate your YAML syntax with any standard YAML parser

### Getting Help

- Check existing examples in `examples/` directory
- Review error messages - they now include more context and suggestions
- File issues with "backward compatibility" label if you find any breaking changes

## Future Compatibility

The new architecture is designed for long-term stability:

- New features are additive, not replacement
- Deprecation warnings will be provided well in advance of any removals  
- Migration tools will be provided for any future changes
- Semantic versioning ensures breaking changes only occur in major versions

## Summary

**You don't need to change anything.** Your existing orchestrator code will continue to work exactly as before, but now with improved performance, better error handling, and access to new optional features.

The refactor was specifically designed to ensure a seamless transition for all existing users.