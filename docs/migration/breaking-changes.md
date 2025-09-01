# Breaking Changes in Orchestrator Refactor

This document details all breaking changes introduced in the Orchestrator refactor. These changes require user action to maintain functionality when upgrading from the legacy v1 architecture.

## Critical Breaking Changes

### 1. Main Entry Point Changed ⚠️

**What Changed:** The primary way to use Orchestrator has completely changed.

**Old (v1):**
```python
from orchestrator import Orchestrator

orchestrator = Orchestrator()
result = orchestrator.execute_pipeline_from_file("pipeline.yaml")
```

**New (Refactored):**
```python
from orchestrator.api import PipelineAPI
import asyncio

async def main():
    api = PipelineAPI()
    pipeline = await api.compile_pipeline("pipeline.yaml")
    result = await api.execute_pipeline(pipeline)
    return result

result = asyncio.run(main())
```

**Action Required:** 
- Replace all `from orchestrator import Orchestrator` imports
- Replace all `Orchestrator()` instantiations with `PipelineAPI()`  
- Replace direct execution with compile + execute pattern
- Add async/await handling

---

### 2. Synchronous to Asynchronous API ⚠️

**What Changed:** All API methods are now async-first.

**Old (v1):**
```python
result = orchestrator.run_pipeline(pipeline_path)
task_result = task.execute(context)
```

**New (Refactored):**
```python
result = await api.execute_pipeline(pipeline)
task_result = await task.execute(context)
```

**Action Required:**
- Add `async`/`await` keywords to all pipeline operations
- Wrap existing synchronous code in `asyncio.run()` or async functions
- Update all custom task implementations to be async

---

### 3. Import Paths Reorganized ⚠️

**What Changed:** Most import paths have changed due to architectural reorganization.

**Import Migration Table:**

| Old Import (v1) | New Import (Refactored) |
|-----------------|-------------------------|
| `from orchestrator import Orchestrator` | `from orchestrator.api import PipelineAPI` |
| `from orchestrator.core.task import Task` | `from orchestrator.core.foundation.task import Task` |
| `from orchestrator.models.model_registry import ModelRegistry` | `from orchestrator.core.foundation.model_registry import ModelRegistry` |
| `from orchestrator.core.exceptions import ExecutionError` | `from orchestrator.core.foundation.exceptions import PipelineExecutionError` |
| `from orchestrator.utils import *` | `from orchestrator.core.foundation.utils import *` |
| `from orchestrator.state.state_manager import StateManager` | `from orchestrator.core.foundation.state import StateManager` |

**Action Required:**
- Update all import statements according to the migration table
- Use find/replace in your IDE for bulk updates

---

### 4. Constructor Parameters Changed ⚠️

**What Changed:** Orchestrator constructor parameters have been redesigned.

**Old (v1):**
```python
orchestrator = Orchestrator(
    debug=True,
    enable_parallel_execution=True,
    max_retries=3,
    model_fallback=True,
    cache_enabled=True,
    state_manager=state_manager,
    resource_allocator=resource_allocator
)
```

**New (Refactored):**
```python
api = PipelineAPI(
    model_registry=model_registry,      # Optional
    development_mode=True,              # Replaces debug=True
    validation_level="permissive"       # New parameter
)

# Other options moved to execution config
config = {
    "parallel_execution": True,          # Was enable_parallel_execution
    "max_retries": 3,                   # Same name
    "enable_fallback": True,            # Was model_fallback
    "cache_enabled": True,              # Same name
}

result = await api.execute_pipeline(pipeline, config=config)
```

**Parameter Migration:**

| Old Parameter (v1) | New Location | Notes |
|--------------------|--------------|-------|
| `debug` | `development_mode` | Constructor parameter |
| `enable_parallel_execution` | `config.parallel_execution` | Execution config |
| `max_retries` | `config.max_retries` | Execution config |
| `model_fallback` | `config.enable_fallback` | Execution config |
| `cache_enabled` | `config.cache_enabled` | Execution config |
| `state_manager` | Removed | Now handled internally |
| `resource_allocator` | Removed | Now handled internally |

**Action Required:**
- Update constructor calls
- Move execution options to config parameter
- Remove deprecated parameters

---

### 5. Custom Task Implementation Changed ⚠️

**What Changed:** Custom task creation and registration has a new API.

**Old (v1):**
```python
from orchestrator.core.task import Task

class MyCustomTask(Task):
    def execute(self, context):
        # Synchronous implementation
        result = do_work()
        return result

# Registration
orchestrator.register_custom_task('my_task', MyCustomTask)
```

**New (Refactored):**
```python
from orchestrator.tools.base_tool import BaseTool

class MyCustomTool(BaseTool):
    async def execute(self, context):
        # Asynchronous implementation
        result = await do_work()
        return result

# Registration
api = PipelineAPI()
api.register_tool('my_task', MyCustomTool)
```

**Action Required:**
- Inherit from `BaseTool` instead of `Task`
- Make `execute` method async
- Update registration calls
- Rename task classes to tool classes (recommended)

---

### 6. Error Handling and Exceptions ⚠️

**What Changed:** Exception hierarchy and error handling patterns updated.

**Old (v1):**
```python
from orchestrator.core.exceptions import ExecutionError

try:
    result = orchestrator.run_pipeline("pipeline.yaml")
except ExecutionError as e:
    handle_error(e)
except Exception as e:
    handle_generic_error(e)
```

**New (Refactored):**
```python
from orchestrator.core.foundation.exceptions import (
    PipelineCompilationError,
    PipelineExecutionError,
    TaskExecutionError
)

try:
    pipeline = await api.compile_pipeline("pipeline.yaml")
    result = await api.execute_pipeline(pipeline)
except PipelineCompilationError as e:
    handle_compilation_error(e)  # New: separate compilation errors
except PipelineExecutionError as e:
    handle_execution_error(e)    # Replaces ExecutionError
except TaskExecutionError as e:
    handle_task_error(e)         # New: specific task errors
```

**Exception Migration:**

| Old Exception (v1) | New Exception (Refactored) | Notes |
|--------------------|----------------------------|--------|
| `ExecutionError` | `PipelineExecutionError` | Renamed for clarity |
| `ValidationError` | `PipelineCompilationError` | Compilation vs runtime separation |
| Generic exceptions | `TaskExecutionError` | New: task-specific errors |

**Action Required:**
- Update exception imports
- Update exception handling to use new hierarchy
- Handle compilation and execution errors separately

---

### 7. Configuration and State Management ⚠️

**What Changed:** Configuration and state management is now internal.

**Old (v1):**
```python
from orchestrator.state.state_manager import StateManager
from orchestrator.core.resource_allocator import ResourceAllocator

state_manager = StateManager()
resource_allocator = ResourceAllocator()

orchestrator = Orchestrator(
    state_manager=state_manager,
    resource_allocator=resource_allocator
)

# Direct state access
state_manager.set_variable('key', 'value')
value = state_manager.get_variable('key')
```

**New (Refactored):**
```python
# Configuration is handled internally
api = PipelineAPI()

# State access through context during execution
class MyTool(BaseTool):
    async def execute(self, context):
        # State access through context
        context.set_variable('key', 'value')
        value = context.get_variable('key')
        return {"result": value}
```

**Action Required:**
- Remove manual state manager and resource allocator creation
- Access state through execution context instead of direct state manager
- Update any code that directly manipulates state

---

### 8. Pipeline File Processing ⚠️

**What Changed:** Pipeline compilation is now separate from execution.

**Old (v1):**
```python
# Direct file execution
result = orchestrator.execute_pipeline_from_file("pipeline.yaml", inputs)

# Or with loading
pipeline = orchestrator.load_pipeline("pipeline.yaml")
result = orchestrator.run_pipeline(pipeline)
```

**New (Refactored):**
```python
# Two-step process: compile then execute
pipeline = await api.compile_pipeline("pipeline.yaml")  # Compilation step
result = await api.execute_pipeline(pipeline, inputs=inputs)  # Execution step

# Or with string content
yaml_content = Path("pipeline.yaml").read_text()
pipeline = await api.compile_pipeline(yaml_content)
result = await api.execute_pipeline(pipeline, inputs=inputs)
```

**Action Required:**
- Replace single-step execution with compile + execute pattern
- Handle compilation errors separately from execution errors

---

### 9. Model Registry and Tool Registration ⚠️

**What Changed:** Model registry and tool registration have new APIs.

**Old (v1):**
```python
from orchestrator.models.model_registry import ModelRegistry
from orchestrator.tools import register_tool

model_registry = ModelRegistry()
model_registry.register_model('gpt-4', GPT4Config)

register_tool('web_search', WebSearchTool)
```

**New (Refactored):**
```python
from orchestrator.core.foundation.model_registry import ModelRegistry
from orchestrator.api import PipelineAPI

# Model registry setup
model_registry = ModelRegistry()
model_registry.register_model('gpt-4', GPT4Config)

# API with registry
api = PipelineAPI(model_registry=model_registry)

# Tool registration
api.register_tool('web_search', WebSearchTool)
```

**Action Required:**
- Update model registry imports  
- Pass model registry to PipelineAPI constructor
- Use API instance methods for tool registration

---

### 10. Logging and Debugging ⚠️

**What Changed:** Debugging and logging configuration has changed.

**Old (v1):**
```python
orchestrator = Orchestrator(debug=True)
orchestrator.set_log_level("DEBUG")
```

**New (Refactored):**
```python
import logging

# Standard Python logging
logging.basicConfig(level=logging.DEBUG)

api = PipelineAPI(development_mode=True)
```

**Action Required:**
- Use standard Python logging instead of orchestrator-specific logging
- Replace `debug=True` with `development_mode=True`
- Remove calls to `set_log_level`

---

## Non-Breaking Changes (Still Work)

These features continue to work without modification:

### YAML Pipeline Syntax ✅
- All existing YAML pipeline files work unchanged
- Task types and configurations remain the same
- Template syntax (`{{ variable }}`) unchanged

### Built-in Tools ✅
- All built-in tools (`llm_task`, `python_task`, etc.) work unchanged
- Tool parameters and configurations are compatible
- Output formats remain the same

### Template System ✅  
- Jinja2 templating syntax works unchanged
- Variable substitution works the same
- Template functions and filters available

---

## Deprecation Warnings

These features still work but will be removed in future versions:

### Synchronous Wrappers ⚠️
```python
# Deprecated but still works
from orchestrator.compat import SyncOrchestrator
sync_orch = SyncOrchestrator()  # Will be removed in v3.0
```

### Legacy Import Aliases ⚠️
```python
# Deprecated but still works
from orchestrator import Orchestrator  # Will be removed in v3.0
```

---

## Migration Priority

### High Priority (Must Fix) 🔴
1. Main entry point changes (`Orchestrator` → `PipelineAPI`)
2. Async/await additions
3. Import path updates
4. Custom task implementations

### Medium Priority (Should Fix) 🟡  
5. Constructor parameter updates
6. Error handling improvements
7. Configuration changes

### Low Priority (Nice to Have) 🟢
8. State management cleanup
9. Logging improvements
10. Model registry updates

---

## Automated Migration Tools

### Migration Script
```bash
# Run automated migration helper
curl -O https://raw.githubusercontent.com/.../migrate.py
python migrate.py --check  # Check for issues
python migrate.py --fix    # Apply automatic fixes
```

### Find/Replace Patterns

For bulk updates in your IDE:

```regex
# Find orchestrator imports
from orchestrator import Orchestrator
# Replace with
from orchestrator.api import PipelineAPI

# Find constructor calls  
Orchestrator\(
# Replace with
PipelineAPI(

# Find synchronous execution
\.execute_pipeline_from_file\(
# Replace with
await api.execute_pipeline(await api.compile_pipeline(
```

---

## Testing Your Migration

### Compatibility Test
```python
# test_compatibility.py
import asyncio
from orchestrator.api import PipelineAPI

async def test_migration():
    """Test that migration works correctly."""
    api = PipelineAPI(development_mode=True)
    
    try:
        # Test compilation
        pipeline = await api.compile_pipeline("test_pipeline.yaml")
        print("✅ Pipeline compilation works")
        
        # Test execution  
        result = await api.execute_pipeline(pipeline)
        print("✅ Pipeline execution works")
        
        return True
    except Exception as e:
        print(f"❌ Migration test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_migration())
    print("Migration test:", "PASSED" if success else "FAILED")
```

---

## Getting Help

- **Migration Guide**: See [from-v1.md](from-v1.md) for detailed migration steps
- **API Reference**: Check [core.md](../api/core.md) for new API documentation
- **Examples**: Updated examples in `/docs/tutorials/`
- **Support**: File issues on GitHub or join the community chat

This breaking changes document should be reviewed before starting any migration to understand the scope of changes required.