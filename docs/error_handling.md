# Error Handling in Orchestrator

The Orchestrator framework provides a comprehensive error hierarchy for consistent error handling across all components. All errors inherit from a base `OrchestratorError` class, making it easy to catch framework-specific errors while maintaining granular error types for specific scenarios.

## Error Hierarchy Overview

```
OrchestratorError (Base)
├── PipelineError
│   ├── PipelineCompilationError
│   ├── PipelineExecutionError
│   ├── CircularDependencyError
│   └── InvalidDependencyError
├── TaskError
│   ├── TaskExecutionError
│   ├── TaskValidationError
│   └── TaskTimeoutError
├── ModelError
│   ├── ModelNotFoundError
│   ├── NoEligibleModelsError
│   ├── ModelExecutionError
│   └── ModelConfigurationError
├── ValidationError
│   ├── SchemaValidationError
│   ├── YAMLValidationError
│   └── ParameterValidationError
├── ResourceError
│   ├── ResourceAllocationError
│   └── ResourceLimitError
├── StateError
│   ├── StateManagerError
│   └── StateCorruptionError
├── ToolError
│   ├── ToolNotFoundError
│   └── ToolExecutionError
├── ControlSystemError
│   ├── CircuitBreakerOpenError
│   └── SystemUnavailableError
├── CompilationError
│   ├── YAMLCompilerError
│   └── AmbiguityResolutionError
├── AdapterError
│   ├── AdapterConfigurationError
│   └── AdapterConnectionError
├── ConfigurationError
│   ├── MissingConfigurationError
│   └── InvalidConfigurationError
├── NetworkError
│   └── APIError
│       ├── RateLimitError
│       └── AuthenticationError
└── TimeoutError
```

## Base Error Class

All Orchestrator errors inherit from `OrchestratorError`:

```python
from orchestrator.core.exceptions import OrchestratorError

try:
    # Some orchestrator operation
    pass
except OrchestratorError as e:
    print(f"Orchestrator error: {e}")
    print(f"Error details: {e.details}")
    print(f"Error code: {e.error_code}")
```

### Error Attributes

- `message`: Human-readable error message
- `details`: Dictionary containing additional error context
- `error_code`: Optional error code for programmatic handling

### Error Methods

- `to_dict()`: Convert error to dictionary for serialization

## Common Error Types

### Pipeline Errors

```python
from orchestrator.core.exceptions import (
    PipelineError,
    CircularDependencyError,
    InvalidDependencyError
)

# Circular dependency detected
try:
    pipeline.validate()
except CircularDependencyError as e:
    print(f"Cycle detected: {e.details['cycle']}")

# Invalid task dependency
try:
    pipeline.add_task(task)
except InvalidDependencyError as e:
    print(f"Task {e.details['task_id']} has invalid dependency")
```

### Task Errors

```python
from orchestrator.core.exceptions import (
    TaskExecutionError,
    TaskTimeoutError
)

# Task execution failure
try:
    await task.execute()
except TaskExecutionError as e:
    print(f"Task {e.details['task_id']} failed: {e.details['reason']}")

# Task timeout
try:
    await task.execute(timeout=30)
except TaskTimeoutError as e:
    print(f"Task timed out after {e.details['timeout']} seconds")
```

### Model Errors

```python
from orchestrator.core.exceptions import (
    ModelNotFoundError,
    NoEligibleModelsError
)

# Model not found
try:
    model = registry.get_model("gpt-5")
except ModelNotFoundError as e:
    print(f"Model {e.details['model_id']} not found")

# No eligible models
try:
    model = registry.select_model(requirements)
except NoEligibleModelsError as e:
    print(f"No models meet requirements: {e.details['requirements']}")
```

### Validation Errors

```python
from orchestrator.core.exceptions import (
    SchemaValidationError,
    ParameterValidationError
)

# Schema validation
try:
    validator.validate(data, schema)
except SchemaValidationError as e:
    for error in e.details['validation_errors']:
        print(f"Validation error: {error}")

# Parameter validation
try:
    validate_params(params)
except ParameterValidationError as e:
    print(f"Invalid {e.details['parameter']}: {e.details['reason']}")
```

### Resource Errors

```python
from orchestrator.core.exceptions import ResourceAllocationError

# Resource allocation failure
try:
    resources = allocator.allocate(request)
except ResourceAllocationError as e:
    print(f"Cannot allocate {e.details['requested']} {e.details['resource_type']}")
    print(f"Only {e.details['available']} available")
```

### API and Network Errors

```python
from orchestrator.core.exceptions import (
    RateLimitError,
    AuthenticationError
)

# Rate limiting
try:
    response = await api_client.call()
except RateLimitError as e:
    retry_after = e.details.get('retry_after', 60)
    print(f"Rate limited. Retry after {retry_after} seconds")

# Authentication failure
try:
    client = APIClient(api_key)
except AuthenticationError as e:
    print(f"Auth failed for {e.details['service']}")
```

## Error Handling Best Practices

### 1. Catch Specific Errors When Possible

```python
try:
    result = await pipeline.execute()
except TaskTimeoutError as e:
    # Handle timeout specifically
    logger.warning(f"Task {e.details['task_id']} timed out")
    result = await retry_with_longer_timeout(e.details['task_id'])
except TaskExecutionError as e:
    # Handle execution failure
    logger.error(f"Task failed: {e.details['reason']}")
    raise
except OrchestratorError as e:
    # Catch any other orchestrator error
    logger.error(f"Unexpected error: {e}")
    raise
```

### 2. Preserve Error Context

```python
try:
    data = load_data(file_path)
except FileNotFoundError as e:
    # Wrap in orchestrator error with context
    raise TaskExecutionError(
        task.id,
        f"Input file not found: {file_path}"
    ) from e
```

### 3. Use Error Details for Debugging

```python
try:
    model = registry.select_model(requirements)
except NoEligibleModelsError as e:
    # Log detailed requirements for debugging
    logger.error(
        "Model selection failed",
        extra={
            "requirements": e.details['requirements'],
            "available_models": registry.list_models()
        }
    )
```

### 4. Handle Errors at Appropriate Levels

```python
class Pipeline:
    async def execute(self):
        results = {}
        for task in self.tasks:
            try:
                results[task.id] = await task.execute()
            except TaskError as e:
                if self.on_failure == "continue":
                    results[task.id] = {"error": e.to_dict()}
                elif self.on_failure == "fail":
                    raise PipelineExecutionError(
                        f"Pipeline failed at task {task.id}",
                        details={"failed_task": task.id}
                    ) from e
```

### 5. Use Error Codes for Programmatic Handling

```python
try:
    result = await operation()
except OrchestratorError as e:
    if e.error_code == "E001":
        # Handle specific error code
        return handle_e001_error()
    elif e.error_code and e.error_code.startswith("RETRY"):
        # Retry errors with RETRY prefix
        return await retry_operation()
    else:
        raise
```

## Creating Custom Errors

If you need to create custom errors for your application:

```python
from orchestrator.core.exceptions import OrchestratorError

class MyCustomError(OrchestratorError):
    """Custom error for my specific use case."""
    
    def __init__(self, custom_field: str, **kwargs):
        message = f"Custom error with field: {custom_field}"
        super().__init__(
            message,
            details={"custom_field": custom_field},
            **kwargs
        )
```

## Error Serialization

Errors can be serialized for logging or API responses:

```python
try:
    result = await pipeline.execute()
except OrchestratorError as e:
    error_response = {
        "status": "error",
        "error": e.to_dict()
    }
    return json.dumps(error_response)
```

Example serialized error:
```json
{
    "status": "error",
    "error": {
        "error_type": "TaskExecutionError",
        "message": "Task 'process_data' failed: Invalid input format",
        "details": {
            "task_id": "process_data",
            "reason": "Invalid input format"
        },
        "error_code": "TASK001"
    }
}
```

## Circuit Breaker Pattern

The framework includes circuit breaker support:

```python
from orchestrator.core.exceptions import (
    CircuitBreakerOpenError,
    SystemUnavailableError
)

async def call_with_circuit_breaker(service_name: str):
    if circuit_breaker.is_open(service_name):
        raise CircuitBreakerOpenError(service_name)
    
    try:
        result = await call_service(service_name)
        circuit_breaker.record_success(service_name)
        return result
    except Exception as e:
        circuit_breaker.record_failure(service_name)
        if circuit_breaker.is_open(service_name):
            raise CircuitBreakerOpenError(service_name) from e
        raise SystemUnavailableError(service_name, str(e)) from e
```

## Debugging with Error Hierarchy

To get a complete view of the error hierarchy:

```python
from orchestrator.core.exceptions import get_error_hierarchy

# Print the entire error hierarchy
hierarchy = get_error_hierarchy()
for base_class, subclasses in hierarchy.items():
    print(f"{base_class}:")
    for subclass in subclasses:
        print(f"  - {subclass}")
```

This is useful for understanding available error types and their relationships.