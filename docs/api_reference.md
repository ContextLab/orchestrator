# API Reference

## Core Classes

### Orchestrator

The main orchestration engine that coordinates pipeline execution.

```python
from orchestrator import Orchestrator

orchestrator = Orchestrator(
    model_registry=model_registry,
    control_system=control_system,
    state_manager=state_manager,  # Optional
    yaml_compiler=yaml_compiler,   # Optional
    max_concurrent_tasks=10        # Optional
)
```

#### Methods

##### execute_yaml(yaml_content, inputs, **kwargs)
Execute a pipeline from YAML content.

```python
result = await orchestrator.execute_yaml(
    yaml_content=yaml_string,
    inputs={"query": "AI applications", "max_results": 10}
)
```

**Parameters:**
- `yaml_content` (str): YAML pipeline definition
- `inputs` (Dict[str, Any]): Input parameters for the pipeline
- `**kwargs`: Additional execution parameters

**Returns:**
- Dict[str, Any]: Execution results with step outputs

##### execute_yaml_file(yaml_file, inputs, **kwargs)
Execute a pipeline from a YAML file.

```python
result = await orchestrator.execute_yaml_file(
    yaml_file="pipelines/research.yaml",
    inputs={"topic": "quantum computing"}
)
```

### Pipeline

Represents a collection of tasks with dependencies.

```python
from orchestrator.core.pipeline import Pipeline

pipeline = Pipeline(
    id="research-pipeline",
    name="Research Assistant",
    description="Automated research pipeline"
)
```

#### Methods

##### add_task(task)
Add a task to the pipeline.

```python
pipeline.add_task(task)
```

##### get_task(task_id)
Get a task by ID.

```python
task = pipeline.get_task("analyze_data")
```

##### get_execution_levels()
Get tasks grouped by execution level (dependency order).

```python
levels = pipeline.get_execution_levels()
# Returns: [[task1, task2], [task3], [task4, task5]]
```

### Task

Represents a single unit of work in a pipeline.

```python
from orchestrator.core.task import Task, TaskStatus

task = Task(
    id="process_data",
    name="Process Data",
    action="Transform and analyze input data",
    parameters={"format": "json"},
    dependencies=["fetch_data"],
    timeout=30.0
)
```

#### Properties

- `id` (str): Unique task identifier
- `name` (str): Human-readable task name
- `action` (str): Task action/prompt
- `parameters` (Dict[str, Any]): Task parameters
- `dependencies` (List[str]): Task IDs this task depends on
- `status` (TaskStatus): Current task status
- `result` (Any): Task execution result
- `error` (Exception): Error if task failed
- `timeout` (int): Timeout in seconds

#### Methods

##### start()
Mark task as running.

```python
task.start()
```

##### complete(result)
Mark task as completed with result.

```python
task.complete({"data": processed_data})
```

##### fail(error)
Mark task as failed with error.

```python
task.fail(Exception("Processing failed"))
```

## Model Integration

### ModelRegistry

Manages available AI models.

```python
from orchestrator.models.model_registry import ModelRegistry

registry = ModelRegistry()
```

#### Methods

##### register_model(model)
Register a model with the registry.

```python
from orchestrator.integrations.openai_model import OpenAIModel

model = OpenAIModel(model_name="gpt-4")
registry.register_model(model)
```

##### select_model(requirements)
Select best model based on requirements.

```python
model = await registry.select_model({
    "tasks": ["generate", "analyze"],
    "context_window": 8000,
    "expertise": ["reasoning", "creative"]
})
```

### Model Implementations

#### OpenAIModel

```python
import os
from orchestrator.integrations.openai_model import OpenAIModel

model = OpenAIModel(
    model_name="gpt-4o-mini",
    api_key=os.environ.get("OPENAI_API_KEY"),
    temperature=0.7,
    max_tokens=1000
)
```

#### AnthropicModel

```python
import os
from orchestrator.integrations.anthropic_model import AnthropicModel

model = AnthropicModel(
    model_name="claude-sonnet-4-20250514",
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)
```

#### GoogleModel

```python
import os
from orchestrator.integrations.google_model import GoogleModel

model = GoogleModel(
    model_name="gemini-1.5-flash",
    api_key=os.environ.get("GOOGLE_API_KEY")
)
```

## Control Systems

### ModelBasedControlSystem

Control system that uses AI models for task execution.

```python
from orchestrator.control_systems.model_based_control_system import ModelBasedControlSystem

control_system = ModelBasedControlSystem(
    model_registry=registry,
    name="model-based-control",
    config={
        "default_temperature": 0.7,
        "max_retries": 3
    }
)
```

#### Methods

##### execute_task(task, context)
Execute a single task.

```python
result = await control_system.execute_task(task, context)
```

##### execute_pipeline(pipeline)
Execute an entire pipeline.

```python
results = await control_system.execute_pipeline(pipeline)
```

## YAML Compilation

### YAMLCompiler

Compiles YAML definitions to executable pipelines.

```python
from orchestrator.compiler.yaml_compiler import YAMLCompiler

compiler = YAMLCompiler(
    auto_tag_parser=auto_tag_parser,  # Optional
    schema_validator=validator         # Optional
)
```

#### Methods

##### compile(yaml_content, context)
Compile YAML to pipeline.

```python
pipeline = await compiler.compile(yaml_content, {
    "user": "john_doe",
    "project": "research"
})
```

### AutoTagYAMLParser

Handles AUTO tags in YAML.

```python
from orchestrator.compiler.auto_tag_yaml_parser import parse_yaml_with_auto_tags

parsed = parse_yaml_with_auto_tags(yaml_content)
```

## State Management

### StateManager

Manages pipeline state and checkpoints.

```python
from orchestrator.state.state_manager import StateManager

state_manager = StateManager(
    backend="postgresql",
    connection_string="postgresql://..."
)
```

#### Methods

##### save_checkpoint(execution_id, state, metadata)
Save execution checkpoint.

```python
await state_manager.save_checkpoint(
    execution_id="exec_123",
    state={"completed_tasks": ["task1", "task2"]},
    metadata={"timestamp": datetime.now()}
)
```

##### load_checkpoint(execution_id)
Load execution checkpoint.

```python
checkpoint = await state_manager.load_checkpoint("exec_123")
```

## Error Handling

### ErrorHandler

Handles errors during pipeline execution.

```python
from orchestrator.core.error_handler import ErrorHandler

error_handler = ErrorHandler(
    retry_policy={
        "max_retries": 3,
        "backoff_factor": 2.0
    }
)
```

#### Error Types

```python
from orchestrator.core.exceptions import (
    PipelineError,
    TaskError,
    CompilationError,
    ValidationError,
    TimeoutError
)
```

## Utilities

### Template Resolution

```python
from orchestrator.utils.template_utils import resolve_template

resolved = resolve_template(
    "Hello {{name}}, your score is {{score}}",
    {"name": "Alice", "score": 95}
)
# Returns: "Hello Alice, your score is 95"
```

### Schema Validation

```python
from orchestrator.utils.schema_validator import validate_yaml_schema

is_valid = validate_yaml_schema(yaml_content)
```

## Complete Example

```python
import asyncio
import os
from orchestrator import Orchestrator
from orchestrator.models.model_registry import ModelRegistry
from orchestrator.integrations.openai_model import OpenAIModel
from orchestrator.control_systems.model_based_control_system import ModelBasedControlSystem

async def main():
    # Setup
    model_registry = ModelRegistry()
    model = OpenAIModel(model_name="gpt-4o-mini")
    model_registry.register_model(model)
    
    control_system = ModelBasedControlSystem(model_registry)
    orchestrator = Orchestrator(
        model_registry=model_registry,
        control_system=control_system
    )
    
    # Define pipeline
    yaml_content = """
    name: "Analysis Pipeline"
    description: "Analyze and summarize data"
    
    inputs:
      data:
        type: string
        required: true
    
    steps:
      - id: analyze
        action: |
          Analyze the following data:
          {{data}}
          
          Provide key insights and patterns.
          
      - id: summarize
        action: |
          Create an executive summary of:
          {{analyze.result}}
          
          Keep it concise and actionable.
        depends_on: [analyze]
    
    outputs:
      analysis: "{{analyze.result}}"
      summary: "{{summarize.result}}"
    """
    
    # Execute
    result = await orchestrator.execute_yaml(
        yaml_content,
        {"data": "Q1 sales increased by 15%..."}
    )
    
    print(result["outputs"]["summary"])

if __name__ == "__main__":
    asyncio.run(main())
```

## Configuration

### Environment Variables

```bash
# Model API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...

# Database (for state management)
DATABASE_URL=postgresql://user:pass@host/db

# Redis (for caching)
REDIS_URL=redis://localhost:6379

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

### Configuration Files

```yaml
# config.yaml
orchestrator:
  max_concurrent_tasks: 10
  default_timeout: 300
  checkpoint_interval: 60

models:
  default_model: "gpt-4o-mini"
  fallback_models:
    - "claude-sonnet-4-20250514"
    - "gemini-1.5-flash"

control_system:
  retry_policy:
    max_retries: 3
    backoff_factor: 2.0
  
state_manager:
  backend: "postgresql"
  checkpoint_retention_days: 30
```

## Async Patterns

All orchestrator operations are async-first:

```python
# Concurrent pipeline execution
results = await asyncio.gather(
    orchestrator.execute_yaml(yaml1, inputs1),
    orchestrator.execute_yaml(yaml2, inputs2),
    orchestrator.execute_yaml(yaml3, inputs3)
)

# With timeout
try:
    result = await asyncio.wait_for(
        orchestrator.execute_yaml(yaml_content, inputs),
        timeout=300.0
    )
except asyncio.TimeoutError:
    print("Pipeline execution timed out")
```

## Testing

### Unit Testing

```python
import pytest
from orchestrator.core.task import Task

@pytest.mark.asyncio
async def test_task_execution():
    task = Task(
        id="test_task",
        name="Test Task",
        action="Process test data"
    )
    
    task.start()
    assert task.status == TaskStatus.RUNNING
    
    task.complete("result")
    assert task.status == TaskStatus.COMPLETED
    assert task.result == "result"
```

### Integration Testing

```python
@pytest.mark.asyncio
async def test_pipeline_execution(orchestrator):
    yaml_content = """
    name: "Test Pipeline"
    steps:
      - id: step1
        action: "Return test data"
    outputs:
      result: "{{step1.result}}"
    """
    
    result = await orchestrator.execute_yaml(yaml_content, {})
    assert "result" in result["outputs"]
```

## Performance Optimization

### Caching

```python
# Enable result caching
orchestrator = Orchestrator(
    cache_backend="redis",
    cache_ttl=3600
)

# Cache specific steps in YAML
steps:
  - id: expensive_analysis
    action: "Perform complex analysis"
    cache_results: true
    cache_ttl: 7200
```

### Resource Management

```python
# Limit concurrent tasks
orchestrator = Orchestrator(
    max_concurrent_tasks=5,
    resource_allocator=ResourceAllocator(
        max_memory_mb=4096,
        max_cpu_percent=80
    )
)
```

## Monitoring and Logging

### Structured Logging

```python
import logging
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)
```

### Metrics

```python
from prometheus_client import Counter, Histogram

# Define metrics
pipeline_executions = Counter(
    'orchestrator_pipeline_executions_total',
    'Total pipeline executions',
    ['pipeline_name', 'status']
)

task_duration = Histogram(
    'orchestrator_task_duration_seconds',
    'Task execution duration',
    ['task_id', 'pipeline_name']
)
```

## Troubleshooting

### Common Issues

1. **Template Resolution Errors**
```python
# Check for undefined variables
from orchestrator.utils.template_utils import find_undefined_variables

undefined = find_undefined_variables(yaml_content, inputs)
if undefined:
    print(f"Missing inputs: {undefined}")
```

2. **Dependency Cycles**
```python
# Detect circular dependencies
from orchestrator.utils.dependency_utils import detect_cycles

cycles = detect_cycles(pipeline)
if cycles:
    raise ValueError(f"Circular dependencies: {cycles}")
```

3. **Memory Issues**
```python
# Stream large results
orchestrator = Orchestrator(
    streaming_threshold_mb=10,
    enable_result_streaming=True
)
```

## Migration Guide

### From v1.x to v2.x

```python
# Old (v1.x)
orchestrator = Orchestrator()
result = orchestrator.run_pipeline(pipeline_dict)

# New (v2.x)
orchestrator = Orchestrator(
    model_registry=model_registry,
    control_system=control_system
)
result = await orchestrator.execute_yaml(yaml_content, inputs)
```

## Support

- Documentation: https://orchestrator.readthedocs.io
- GitHub: https://github.com/your-org/orchestrator
- Issues: https://github.com/your-org/orchestrator/issues
- Discord: https://discord.gg/orchestrator