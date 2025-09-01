# Core API Reference

The Core API provides the main entry point for pipeline compilation and execution in the Orchestrator framework. This API integrates all foundation components into a clean, intuitive interface that enables users to build, compile, and execute sophisticated data processing pipelines.

## PipelineAPI Class

The `PipelineAPI` class is the primary interface for all pipeline operations.

### Constructor

```python
from orchestrator.api import PipelineAPI

api = PipelineAPI(
    model_registry=None,          # Optional ModelRegistry instance
    development_mode=False,       # Enable development mode features
    validation_level="strict"     # Validation strictness level
)
```

**Parameters:**
- `model_registry` (Optional[ModelRegistry]): Custom model registry for AUTO tag resolution
- `development_mode` (bool): Enable development mode with relaxed validation
- `validation_level` (str): Validation strictness ("strict", "permissive", "development")

**Example:**
```python
# Basic usage
api = PipelineAPI()

# Development mode with permissive validation
dev_api = PipelineAPI(
    development_mode=True,
    validation_level="permissive"
)
```

### Pipeline Compilation

#### compile_pipeline()

Compile a YAML pipeline specification into an executable Pipeline object.

```python
async def compile_pipeline(
    yaml_content: Union[str, Path],
    context: Optional[Dict[str, Any]] = None,
    resolve_ambiguities: bool = True,
    validate: bool = True
) -> Pipeline
```

**Parameters:**
- `yaml_content`: YAML content as string or path to YAML file
- `context`: Template context variables for compilation
- `resolve_ambiguities`: Whether to resolve AUTO tags during compilation
- `validate`: Whether to perform comprehensive validation

**Returns:**
- `Pipeline`: Compiled pipeline object ready for execution

**Raises:**
- `CompilationError`: If compilation fails
- `FileNotFoundError`: If YAML file path does not exist

**Example:**
```python
# From string content
yaml_content = """
name: my_pipeline
steps:
  - name: process_data
    action: data_transform
    parameters:
      input_file: ${input_file}
      operation: normalize
"""

context = {"input_file": "data.csv"}
pipeline = await api.compile_pipeline(yaml_content, context)

# From file
pipeline = await api.compile_pipeline("pipelines/data_processing.yaml")

# With custom validation
pipeline = await api.compile_pipeline(
    yaml_content,
    context=context,
    resolve_ambiguities=True,
    validate=True
)
```

### Pipeline Execution

#### execute_pipeline()

Execute a compiled pipeline with comprehensive monitoring and control.

```python
async def execute_pipeline(
    pipeline: Union[Pipeline, str, Path],
    context: Optional[Dict[str, Any]] = None,
    execution_id: Optional[str] = None
) -> ComprehensiveExecutionManager
```

**Parameters:**
- `pipeline`: Pipeline object, YAML content, or path to YAML file
- `context`: Additional execution context variables
- `execution_id`: Optional custom execution ID

**Returns:**
- `ComprehensiveExecutionManager`: Execution manager for monitoring and control

**Raises:**
- `ExecutionError`: If execution initialization fails
- `CompilationError`: If pipeline compilation fails (for non-Pipeline inputs)

**Example:**
```python
# Execute compiled pipeline
execution_manager = await api.execute_pipeline(pipeline)

# Execute from YAML with context
execution_manager = await api.execute_pipeline(
    "pipelines/data_processing.yaml",
    context={"input_file": "data.csv", "output_dir": "results/"}
)

# With custom execution ID
execution_manager = await api.execute_pipeline(
    pipeline,
    execution_id="custom_execution_001"
)

# Monitor execution
status = execution_manager.get_execution_status()
print(f"Progress: {status['progress']['percentage']}%")
```

### Execution Monitoring

#### get_execution_status()

Get comprehensive status information for a running or completed execution.

```python
def get_execution_status(execution_id: str) -> Dict[str, Any]
```

**Parameters:**
- `execution_id`: Unique execution identifier

**Returns:**
- Dictionary containing execution status, progress, metrics, and recovery info

**Raises:**
- `ExecutionError`: If execution not found

**Example:**
```python
status = api.get_execution_status("my_execution_id")

print(f"Status: {status['status']}")
print(f"Progress: {status['progress']['percentage']}%")
print(f"Completed Steps: {status['progress']['completed_steps']}")
print(f"Total Steps: {status['progress']['total_steps']}")
print(f"Duration: {status['metrics']['duration']} seconds")
```

#### list_active_executions()

Get a list of all currently active execution IDs.

```python
def list_active_executions() -> List[str]
```

**Returns:**
- List of active execution identifiers

**Example:**
```python
active_executions = api.list_active_executions()
for execution_id in active_executions:
    status = api.get_execution_status(execution_id)
    print(f"{execution_id}: {status['progress']['percentage']}% complete")
```

### Execution Control

#### stop_execution()

Stop a running pipeline execution.

```python
def stop_execution(execution_id: str, graceful: bool = True) -> bool
```

**Parameters:**
- `execution_id`: Unique execution identifier
- `graceful`: Whether to wait for current step to complete

**Returns:**
- `True` if execution was stopped successfully

**Raises:**
- `ExecutionError`: If execution not found

**Example:**
```python
# Graceful stop (wait for current step)
success = api.stop_execution("my_execution_id", graceful=True)

# Immediate stop
success = api.stop_execution("my_execution_id", graceful=False)
```

#### cleanup_execution()

Clean up resources for a completed or failed execution.

```python
def cleanup_execution(execution_id: str) -> bool
```

**Parameters:**
- `execution_id`: Unique execution identifier

**Returns:**
- `True` if cleanup was successful

**Example:**
```python
# Clean up after execution completes
success = api.cleanup_execution("my_execution_id")
```

### Validation and Introspection

#### validate_yaml()

Validate YAML pipeline specification without full compilation.

```python
def validate_yaml(yaml_content: Union[str, Path]) -> bool
```

**Parameters:**
- `yaml_content`: YAML content as string or path to YAML file

**Returns:**
- `True` if YAML is valid, `False` otherwise

**Example:**
```python
# Validate pipeline YAML
is_valid = api.validate_yaml("pipelines/data_processing.yaml")
if not is_valid:
    print("Pipeline YAML has validation errors")
```

#### get_template_variables()

Extract template variables from YAML pipeline specification.

```python
def get_template_variables(yaml_content: Union[str, Path]) -> List[str]
```

**Parameters:**
- `yaml_content`: YAML content as string or path to YAML file

**Returns:**
- List of template variable names found in the YAML

**Example:**
```python
variables = api.get_template_variables("pipelines/data_processing.yaml")
print("Required variables:", variables)
# Output: ['input_file', 'output_dir', 'processing_mode']
```

#### get_compilation_report()

Get detailed validation report from the last compilation.

```python
def get_compilation_report() -> Optional[Dict[str, Any]]
```

**Returns:**
- Dictionary containing validation results, or `None` if no report available

**Example:**
```python
pipeline = await api.compile_pipeline(yaml_content)
report = api.get_compilation_report()

if report:
    print(f"Total issues: {report['stats']['total_issues']}")
    print(f"Errors: {report['stats']['errors']}")
    print(f"Warnings: {report['stats']['warnings']}")
    
    if report['has_errors']:
        print("Compilation errors found:")
        print(report['details'])
```

### Context Management

The PipelineAPI supports context manager usage for automatic resource cleanup:

```python
async with PipelineAPI() as api:
    pipeline = await api.compile_pipeline(yaml_content)
    execution_manager = await api.execute_pipeline(pipeline)
    
    # Wait for completion
    while True:
        status = api.get_execution_status(execution_manager.execution_id)
        if status['status'] in ['completed', 'failed']:
            break
        await asyncio.sleep(1)
    
    # Resources automatically cleaned up on exit
```

### Complete Example

Here's a comprehensive example showing typical API usage:

```python
import asyncio
from orchestrator.api import PipelineAPI

async def main():
    # Initialize API
    api = PipelineAPI(development_mode=False, validation_level="strict")
    
    try:
        # Define pipeline YAML
        pipeline_yaml = """
        name: data_processing_pipeline
        description: Process customer data with validation
        
        variables:
          batch_size: 1000
          validation_threshold: 0.95
          
        steps:
          - name: load_data
            action: data_load
            parameters:
              source: ${input_file}
              format: csv
              
          - name: validate_data  
            action: data_validate
            parameters:
              data: ${load_data.output}
              threshold: ${validation_threshold}
              
          - name: process_data
            action: data_transform
            parameters:
              data: ${validate_data.output}
              batch_size: ${batch_size}
              operations:
                - normalize
                - clean_nulls
                
          - name: save_results
            action: data_save
            parameters:
              data: ${process_data.output}
              destination: ${output_file}
              format: parquet
        """
        
        # Define execution context
        context = {
            "input_file": "customer_data.csv",
            "output_file": "processed_data.parquet"
        }
        
        # Validate YAML first
        if not api.validate_yaml(pipeline_yaml):
            print("Pipeline YAML validation failed")
            return
            
        # Get required variables
        variables = api.get_template_variables(pipeline_yaml)
        print(f"Required variables: {variables}")
        
        # Compile pipeline
        print("Compiling pipeline...")
        pipeline = await api.compile_pipeline(pipeline_yaml, context)
        
        # Check compilation report
        report = api.get_compilation_report()
        if report and report['has_errors']:
            print("Compilation errors found:", report['summary'])
            return
            
        # Execute pipeline
        print("Starting execution...")
        execution_manager = await api.execute_pipeline(
            pipeline, 
            execution_id="customer_data_processing_001"
        )
        
        # Monitor execution
        while True:
            status = api.get_execution_status(execution_manager.execution_id)
            progress = status['progress']
            
            print(f"Progress: {progress['percentage']:.1f}% "
                  f"({progress['completed_steps']}/{progress['total_steps']} steps)")
            
            if status['status'] == 'completed':
                print("Pipeline execution completed successfully!")
                break
            elif status['status'] == 'failed':
                print("Pipeline execution failed!")
                print(f"Recovery status: {status['recovery']}")
                break
                
            await asyncio.sleep(2)
        
        # Clean up
        api.cleanup_execution(execution_manager.execution_id)
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Shutdown API
        api.shutdown()

# Run the example
if __name__ == "__main__":
    asyncio.run(main())
```

## Convenience Functions

For simpler use cases, convenience functions are available:

### create_pipeline_api()

Create a PipelineAPI instance with specified configuration.

```python
from orchestrator.api import create_pipeline_api

api = create_pipeline_api(
    development_mode=True,
    validation_level="permissive"
)
```

## Error Handling

The Core API defines specific exception types for different error scenarios:

### PipelineAPIError

Base exception for all Pipeline API errors.

### CompilationError

Raised when pipeline compilation fails due to:
- Invalid YAML syntax
- Missing required variables
- Validation errors
- Template resolution failures

### ExecutionError

Raised when pipeline execution fails due to:
- Runtime errors
- Resource allocation failures
- Step execution failures
- System errors

**Example Error Handling:**
```python
from orchestrator.api import PipelineAPI, CompilationError, ExecutionError

api = PipelineAPI()

try:
    pipeline = await api.compile_pipeline(yaml_content)
    execution_manager = await api.execute_pipeline(pipeline)
    
except CompilationError as e:
    print(f"Pipeline compilation failed: {e}")
    # Handle compilation issues
    
except ExecutionError as e:
    print(f"Pipeline execution failed: {e}")
    # Handle execution issues
    
except Exception as e:
    print(f"Unexpected error: {e}")
    # Handle unexpected issues
```

## Best Practices

1. **Use Context Managers**: Always use the PipelineAPI as a context manager or explicitly call `shutdown()` to ensure proper resource cleanup.

2. **Validate Early**: Use `validate_yaml()` before compilation to catch syntax errors early.

3. **Monitor Progress**: For long-running pipelines, regularly check execution status to track progress and detect issues.

4. **Handle Errors Gracefully**: Implement proper error handling for both compilation and execution phases.

5. **Clean Up Resources**: Always clean up completed executions to free system resources.

6. **Use Meaningful IDs**: Provide descriptive execution IDs to help with monitoring and debugging.

7. **Leverage Validation Reports**: Use compilation reports to understand and fix validation issues.