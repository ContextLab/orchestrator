# Model Registry Unification Guide

## Overview

As part of Issue #108, the model registry has been unified to use a singleton pattern, ensuring that all components share the same registry instance. This prevents issues with models being registered in different registries across the codebase.

## Changes Made

### 1. New Singleton Module

Created `src/orchestrator/models/registry_singleton.py` with three functions:
- `get_model_registry()` - Returns the global registry instance
- `set_model_registry(registry)` - Sets a custom registry (for testing)
- `reset_model_registry()` - Resets the global registry (for testing)

### 2. Updated Components

The following components now use `get_model_registry()` instead of creating new instances:

- `src/orchestrator/__init__.py` - Main initialization
- `src/orchestrator/orchestrator.py` - Orchestrator class
- `src/orchestrator/adapters/langgraph_adapter.py` - LangGraph adapter
- `src/orchestrator/adapters/mcp_adapter.py` - MCP adapter

## Migration Instructions

### For Library Users

No changes required. The library will automatically use the unified registry.

### For Library Developers

1. **Importing the Registry**
   ```python
   # Old way - DON'T DO THIS
   from orchestrator.models import ModelRegistry
   registry = ModelRegistry()
   
   # New way - DO THIS
   from orchestrator.models import get_model_registry
   registry = get_model_registry()
   ```

2. **In Class Constructors**
   ```python
   # Old way
   def __init__(self, model_registry=None):
       self.model_registry = model_registry or ModelRegistry()
   
   # New way
   def __init__(self, model_registry=None):
       from orchestrator.models import get_model_registry
       self.model_registry = model_registry or get_model_registry()
   ```

3. **For Testing**
   ```python
   from orchestrator.models import reset_model_registry, set_model_registry
   
   def setUp(self):
       # Reset to ensure clean state
       reset_model_registry()
       
       # Or set a custom registry for testing
       test_registry = ModelRegistry()
       set_model_registry(test_registry)
   ```

## Benefits

1. **Consistency**: All components share the same model registry
2. **Simplicity**: No need to pass registry instances between components
3. **Reliability**: Models registered in one place are available everywhere
4. **Testing**: Easy to reset or mock the registry for tests

## Remaining Work

The following components may still need updating:
- Task executors in `src/orchestrator/engine/`
- Compiler components in `src/orchestrator/compiler/`
- Control systems in `src/orchestrator/control_systems/`

These will be addressed in subsequent commits.