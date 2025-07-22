# Orchestrator API Reference

Complete API documentation for the Orchestrator framework.

## Core Classes

### Orchestrator

The main orchestration engine class.

```python
from orchestrator import Orchestrator

# Initialize with defaults
orchestrator = Orchestrator()

# Initialize with specific components
orchestrator = Orchestrator(
    model_registry=my_registry,
    control_system=my_control_system,
    state_manager=my_state_manager
)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_registry` | `ModelRegistry` | `None` | Model registry for AI model management |
| `control_system` | `ControlSystem` | `None` | Control system for task execution |
| `state_manager` | `StateManager` | `None` | State manager for checkpointing |
| `yaml_compiler` | `YAMLCompiler` | `None` | YAML compiler for pipeline parsing |
| `error_handler` | `ErrorHandler` | `None` | Error handler for fault tolerance |
| `resource_allocator` | `ResourceAllocator` | `None` | Resource allocator for task scheduling |
| `parallel_executor` | `ParallelExecutor` | `None` | Parallel executor for concurrent execution |
| `max_concurrent_tasks` | `int` | `10` | Maximum concurrent tasks |

#### Methods

##### `execute_yaml(yaml_content, context=None, **kwargs)`

Execute a pipeline from YAML content.

**Parameters:**
- `yaml_content` (str): YAML pipeline definition
- `context` (dict, optional): Template context variables
- `**kwargs`: Additional execution parameters

**Returns:** `Dict[str, Any]` - Execution results

**Example:**
```python
yaml_content = """
name: example
steps:
  - id: hello
    tool: report-generator
    action: generate
    parameters:
      content: "Hello {{ name }}!"
"""

result = await orchestrator.execute_yaml(yaml_content, {"name": "World"})
```

##### `execute_yaml_file(yaml_file, context=None, **kwargs)`

Execute a pipeline from YAML file.

**Parameters:**
- `yaml_file` (str): Path to YAML file
- `context` (dict, optional): Template context variables
- `**kwargs`: Additional execution parameters

**Returns:** `Dict[str, Any]` - Execution results

**Example:**
```python
result = await orchestrator.execute_yaml_file(
    "pipeline.yaml", 
    {"input_data": data}
)
```

##### `execute_pipeline(pipeline, checkpoint_enabled=True, max_retries=3)`

Execute a Pipeline object.

**Parameters:**
- `pipeline` (Pipeline): Pipeline to execute
- `checkpoint_enabled` (bool): Whether to enable checkpointing
- `max_retries` (int): Maximum number of retries for failed tasks

**Returns:** `Dict[str, Any]` - Execution results

##### `recover_pipeline(execution_id, from_checkpoint=None)`

Recover a failed pipeline from checkpoint.

**Parameters:**
- `execution_id` (str): Execution ID to recover
- `from_checkpoint` (str, optional): Specific checkpoint to recover from

**Returns:** `Dict[str, Any]` - Recovery results

##### `get_execution_status(execution_id)`

Get execution status for a running or completed pipeline.

**Parameters:**
- `execution_id` (str): Execution ID

**Returns:** `Dict[str, Any]` - Status information

##### `health_check()`

Perform health check on all components.

**Returns:** `Dict[str, Any]` - Health status

### Model Registry

Registry for managing AI models and routing.

```python
from orchestrator.models import ModelRegistry

registry = ModelRegistry()
```

#### Methods

##### `register_model(model_id, model_config)`

Register a new model.

**Parameters:**
- `model_id` (str): Unique model identifier
- `model_config` (dict): Model configuration

**Example:**
```python
registry.register_model("gpt-4", {
    "provider": "openai",
    "model": "gpt-4-turbo",
    "capabilities": ["reasoning", "coding"],
    "context_window": 128000,
    "cost_per_1k_tokens": 0.01
})
```

##### `get_model(model_id)`

Get a registered model by ID.

**Parameters:**
- `model_id` (str): Model identifier

**Returns:** `Model` - Model object or None

##### `select_model(requirements)`

Select optimal model based on requirements.

**Parameters:**
- `requirements` (dict): Model requirements

**Returns:** `Model` - Selected model

**Example:**
```python
model = await registry.select_model({
    "tasks": ["reasoning"],
    "context_window": 50000,
    "max_cost_per_1k": 0.005
})
```

##### `list_models(filter_by=None)`

List all registered models.

**Parameters:**
- `filter_by` (dict, optional): Filter criteria

**Returns:** `List[Model]` - List of models

### Pipeline

Represents a pipeline with tasks and dependencies.

```python
from orchestrator.core import Pipeline

pipeline = Pipeline(
    id="my-pipeline",
    name="My Pipeline",
    description="Example pipeline",
    tasks=task_list
)
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `id` | `str` | Pipeline identifier |
| `name` | `str` | Pipeline name |
| `description` | `str` | Pipeline description |
| `tasks` | `Dict[str, Task]` | Pipeline tasks |
| `context` | `dict` | Pipeline context variables |
| `metadata` | `dict` | Pipeline metadata |

#### Methods

##### `add_task(task)`

Add a task to the pipeline.

**Parameters:**
- `task` (Task): Task to add

##### `get_task(task_id)`

Get a task by ID.

**Parameters:**
- `task_id` (str): Task identifier

**Returns:** `Task` - Task object or None

##### `get_execution_levels()`

Get tasks grouped by execution levels for parallel execution.

**Returns:** `List[List[str]]` - Lists of task IDs that can run in parallel

##### `validate()`

Validate the pipeline for circular dependencies and other issues.

**Returns:** `bool` - True if valid

### Task

Represents a single task within a pipeline.

```python
from orchestrator.core import Task

task = Task(
    id="my-task",
    action="generate",
    parameters={"prompt": "Hello"},
    tool="llm-generate"
)
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `id` | `str` | Task identifier |
| `action` | `str` | Action to perform |
| `parameters` | `dict` | Task parameters |
| `tool` | `str` | Tool to use |
| `dependencies` | `List[str]` | Task dependencies |
| `status` | `TaskStatus` | Current task status |
| `timeout` | `int` | Task timeout in seconds |
| `metadata` | `dict` | Task metadata |

#### Methods

##### `start()`

Mark task as started.

##### `complete(result)`

Mark task as completed with result.

**Parameters:**
- `result`: Task result

##### `fail(error)`

Mark task as failed with error.

**Parameters:**
- `error`: Error that caused failure

##### `can_retry()`

Check if task can be retried.

**Returns:** `bool` - True if retryable

## Tool Classes

### Base Tool

All tools inherit from the base Tool class.

```python
from orchestrator.tools import Tool

class MyTool(Tool):
    def __init__(self):
        super().__init__(
            id="my-tool",
            name="My Tool",
            description="Custom tool"
        )
    
    async def execute(self, action, parameters):
        # Implementation
        return result
```

### FileSystemTool

File system operations.

```python
from orchestrator.tools import FileSystemTool

tool = FileSystemTool()
result = await tool.execute("read", {"path": "/path/to/file"})
```

**Actions:**
- `read` - Read file content
- `write` - Write content to file
- `delete` - Delete file or directory
- `copy` - Copy file or directory
- `move` - Move/rename file
- `list` - List directory contents
- `exists` - Check if path exists
- `mkdir` - Create directory

### LLMGenerateTool

Language model text generation.

```python
from orchestrator.tools import LLMGenerateTool

tool = LLMGenerateTool()
result = await tool.execute("generate", {
    "prompt": "Write a summary",
    "temperature": 0.7,
    "max_tokens": 200
})
```

**Actions:**
- `generate` - Generate text
- `complete` - Complete text
- `chat` - Chat conversation

### WebSearchTool

Web search functionality.

```python
from orchestrator.tools import WebSearchTool

tool = WebSearchTool()
result = await tool.execute("search", {
    "query": "python tutorial",
    "max_results": 10
})
```

**Actions:**
- `search` - Perform web search

## Utility Functions

### `init_models()`

Initialize the model registry with default models.

```python
from orchestrator import init_models

registry = init_models()
```

**Returns:** `ModelRegistry` - Initialized model registry

### Configuration

#### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | None |
| `ANTHROPIC_API_KEY` | Anthropic API key | None |
| `GOOGLE_API_KEY` | Google API key | None |
| `ORCHESTRATOR_LOG_LEVEL` | Logging level | `INFO` |
| `ORCHESTRATOR_MAX_CONCURRENT` | Max concurrent tasks | `10` |

#### Configuration Files

Orchestrator looks for configuration in:
- `orchestrator.yaml` - Main configuration
- `models.yaml` - Model definitions
- `.orchestrator/` - User configuration directory

**Example orchestrator.yaml:**
```yaml
orchestrator:
  max_concurrent_tasks: 20
  enable_checkpointing: true
  log_level: DEBUG
  
models:
  default_provider: openai
  routing_strategy: ucb
  cost_optimization: true
  
tools:
  enable_web_search: true
  enable_filesystem: true
  sandbox_mode: false
```

## Error Handling

### Exception Classes

#### `ExecutionError`

Raised when pipeline execution fails.

```python
from orchestrator import ExecutionError

try:
    result = await orchestrator.execute_yaml(yaml_content)
except ExecutionError as e:
    print(f"Pipeline failed: {e}")
```

#### `CompilationError`

Raised when YAML compilation fails.

```python
from orchestrator.compiler import CompilationError

try:
    pipeline = await compiler.compile(yaml_content)
except CompilationError as e:
    print(f"Compilation failed: {e}")
```

#### `ValidationError`

Raised when validation fails.

```python
from orchestrator.tools import ValidationError

try:
    result = await validation_tool.execute("validate", params)
except ValidationError as e:
    print(f"Validation failed: {e}")
```

### Error Recovery

```python
# Automatic recovery
result = await orchestrator.recover_pipeline(execution_id)

# Manual error handling
try:
    result = await orchestrator.execute_yaml(yaml_content)
except ExecutionError as e:
    # Custom recovery logic
    result = await orchestrator.execute_yaml(fallback_yaml)
```

## Advanced Usage

### Custom Control Systems

```python
from orchestrator.core import ControlSystem

class MyControlSystem(ControlSystem):
    async def execute_task(self, task, context):
        # Custom execution logic
        return result
    
    def get_capabilities(self):
        return ["custom_capability"]

orchestrator = Orchestrator(control_system=MyControlSystem())
```

### Custom State Management

```python
from orchestrator.state import StateManager

class MyStateManager(StateManager):
    async def save_checkpoint(self, execution_id, state, context):
        # Custom checkpoint storage
        pass
    
    async def restore_checkpoint(self, execution_id, checkpoint_id):
        # Custom checkpoint restoration
        return state

orchestrator = Orchestrator(state_manager=MyStateManager())
```

### Hooks and Plugins

```python
# Pre-execution hook
@orchestrator.hook('before_execution')
async def log_execution_start(pipeline, context):
    print(f"Starting pipeline: {pipeline.name}")

# Post-execution hook
@orchestrator.hook('after_execution')
async def log_execution_end(pipeline, context, result):
    print(f"Completed pipeline: {pipeline.name}")

# Tool plugin
@orchestrator.register_tool('custom-tool')
class CustomTool(Tool):
    async def execute(self, action, parameters):
        return {"message": "Custom tool executed"}
```

## Performance and Monitoring

### Metrics Collection

```python
# Get performance metrics
metrics = await orchestrator.get_performance_metrics()
print(f"Total executions: {metrics['total_executions']}")
print(f"Success rate: {metrics['success_rate']}")

# Model usage metrics
model_metrics = await orchestrator.model_registry.get_usage_metrics()
```

### Resource Monitoring

```python
# Check resource utilization
utilization = await orchestrator.resource_allocator.get_utilization()
print(f"CPU usage: {utilization['cpu_usage']}%")
print(f"Memory usage: {utilization['memory_usage']}MB")
```

### Logging

```python
import logging

# Configure logging
logging.getLogger('orchestrator').setLevel(logging.DEBUG)

# Custom logger
logger = logging.getLogger('my_app')
orchestrator = Orchestrator(logger=logger)
```

## Testing

### Unit Testing

```python
import pytest
from orchestrator import Orchestrator
from orchestrator.testing import MockTool

@pytest.fixture
async def orchestrator():
    registry = init_models()
    return Orchestrator(model_registry=registry)

async def test_simple_pipeline(orchestrator):
    yaml_content = """
    name: test
    steps:
      - id: hello
        tool: report-generator
        action: generate
        parameters:
          content: "Hello Test"
    """
    
    result = await orchestrator.execute_yaml(yaml_content)
    assert result['hello']['content'] == "Hello Test"
```

### Integration Testing

```python
async def test_full_workflow():
    orchestrator = Orchestrator()
    
    # Test with real file operations
    yaml_content = """
    name: integration-test
    steps:
      - id: create_file
        tool: filesystem
        action: write
        parameters:
          path: "/tmp/test.txt"
          content: "Test content"
          
      - id: read_file
        tool: filesystem
        action: read
        parameters:
          path: "/tmp/test.txt"
    """
    
    result = await orchestrator.execute_yaml(yaml_content)
    assert result['read_file']['content'] == "Test content"
```

## Migration and Compatibility

### Version Compatibility

- v1.0.0: Initial release
- v1.1.0: Added AUTO tags
- v1.2.0: Enhanced model routing
- Current: v1.3.0

### Migration Guide

#### From v1.0 to v1.1

```python
# Old way
pipeline = Pipeline(tasks=[...])

# New way - AUTO tags supported
pipeline = await YAMLCompiler().compile(yaml_with_auto_tags)
```

#### From v1.1 to v1.2

```python
# Old way
model = registry.get_model("gpt-4")

# New way - intelligent selection
model = await registry.select_model(requirements)
```

## Best Practices

### Pipeline Design

1. **Use descriptive task IDs**
2. **Minimize dependencies**
3. **Include error handling**
4. **Document complex logic**
5. **Test with real data**

### Model Usage

1. **Let router choose when possible**
2. **Specify requirements, not models**
3. **Use cost-effective models for simple tasks**
4. **Monitor usage and costs**

### Error Handling

1. **Handle expected failures gracefully**
2. **Use retries for transient issues**
3. **Log sufficient detail for debugging**
4. **Implement circuit breakers**

### Performance

1. **Use parallel execution where possible**
2. **Cache expensive operations**
3. **Monitor resource usage**
4. **Profile critical paths**