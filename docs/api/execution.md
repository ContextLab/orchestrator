# Execution Engine API Reference

The Execution Engine provides comprehensive execution management, variable handling, progress tracking, and recovery mechanisms for pipeline operations. The primary interface is the `ComprehensiveExecutionManager` which integrates all execution subsystems.

## ComprehensiveExecutionManager

The `ComprehensiveExecutionManager` is the main interface for managing pipeline execution with integrated variable management, progress tracking, and error recovery.

### Constructor

```python
from orchestrator.execution import ComprehensiveExecutionManager, create_comprehensive_execution_manager

# Recommended way - use factory function
execution_manager = create_comprehensive_execution_manager(
    execution_id="my_execution_001",
    pipeline_id="data_processing_pipeline"
)

# Direct construction (advanced usage)
execution_manager = ComprehensiveExecutionManager(
    execution_id="my_execution_001",
    pipeline_id="data_processing_pipeline",
    execution_context=None,    # Optional - will create if None
    variable_manager=None,     # Optional - will create if None
    progress_tracker=None,     # Optional - will create if None
    recovery_manager=None      # Optional - will create if None
)
```

**Parameters:**
- `execution_id` (str): Unique identifier for this execution
- `pipeline_id` (str): Identifier of the pipeline being executed
- `execution_context` (Optional[ExecutionContext]): Custom execution context
- `variable_manager` (Optional[VariableManager]): Custom variable manager
- `progress_tracker` (Optional[ProgressTracker]): Custom progress tracker
- `recovery_manager` (Optional[RecoveryManager]): Custom recovery manager

### Execution Lifecycle Management

#### start_execution()

Initialize and start the execution with all subsystems coordinated.

```python
def start_execution(total_steps: int = 1) -> None
```

**Parameters:**
- `total_steps` (int): Total number of steps in the pipeline

**Example:**
```python
execution_manager = create_comprehensive_execution_manager(
    execution_id="data_pipeline_001",
    pipeline_id="customer_analytics"
)

# Start execution with 5 steps
execution_manager.start_execution(total_steps=5)
```

#### complete_execution()

Complete the execution and perform cleanup operations.

```python
def complete_execution(success: bool = True) -> None
```

**Parameters:**
- `success` (bool): Whether the execution completed successfully

**Example:**
```python
try:
    # Execute pipeline steps...
    execution_manager.complete_execution(success=True)
except Exception as e:
    execution_manager.complete_execution(success=False)
```

### Step Management

#### start_step()

Start execution of a pipeline step with coordinated tracking.

```python
def start_step(step_id: str, step_name: str) -> None
```

**Parameters:**
- `step_id` (str): Unique identifier for the step
- `step_name` (str): Human-readable name for the step

**Example:**
```python
execution_manager.start_step("load_data", "Load Customer Data")
```

#### complete_step()

Complete a step with success/failure status and optional error information.

```python
def complete_step(
    step_id: str,
    success: bool = True,
    error: Optional[Exception] = None,
    progress_percentage: float = 100.0
) -> None
```

**Parameters:**
- `step_id` (str): Unique identifier for the step
- `success` (bool): Whether the step completed successfully
- `error` (Optional[Exception]): Exception that caused failure (if any)
- `progress_percentage` (float): Final progress percentage for the step

**Example:**
```python
try:
    # Execute step logic...
    result = perform_data_loading()
    execution_manager.complete_step("load_data", success=True)
except Exception as e:
    execution_manager.complete_step("load_data", success=False, error=e)
```

#### update_step_progress()

Update progress for a currently executing step.

```python
def update_step_progress(
    step_id: str,
    progress_percentage: float,
    message: Optional[str] = None
) -> None
```

**Parameters:**
- `step_id` (str): Unique identifier for the step
- `progress_percentage` (float): Current progress percentage (0-100)
- `message` (Optional[str]): Optional progress message

**Example:**
```python
execution_manager.start_step("process_data", "Processing Customer Records")

for i, batch in enumerate(data_batches):
    # Process batch...
    progress = (i + 1) / len(data_batches) * 100
    execution_manager.update_step_progress(
        "process_data", 
        progress,
        f"Processed batch {i + 1}/{len(data_batches)}"
    )

execution_manager.complete_step("process_data", success=True)
```

### Step Execution with Recovery

#### execute_step_with_recovery()

Execute a step with integrated error handling and automatic recovery.

```python
async def execute_step_with_recovery(
    step_id: str,
    step_name: str,
    step_executor: Callable[[], Any]
) -> bool
```

**Parameters:**
- `step_id` (str): Unique identifier for the step
- `step_name` (str): Human-readable name for the step
- `step_executor` (Callable): Async function that executes the step

**Returns:**
- `bool`: True if step completed successfully (including after recovery)

**Example:**
```python
async def load_customer_data():
    # Step implementation that might fail
    response = await api_client.get("/customers")
    if response.status_code != 200:
        raise Exception("Failed to load customer data")
    return response.json()

# Execute with automatic recovery
success = await execution_manager.execute_step_with_recovery(
    "load_data",
    "Load Customer Data", 
    load_customer_data
)

if success:
    print("Data loaded successfully")
else:
    print("Data loading failed after recovery attempts")
```

### Error Handling and Recovery

#### handle_step_error()

Handle step errors and generate recovery plans.

```python
def handle_step_error(
    step_id: str,
    step_name: str,
    error: Exception,
    context: Optional[Dict[str, Any]] = None
) -> RecoveryPlan
```

**Parameters:**
- `step_id` (str): Unique identifier for the step
- `step_name` (str): Human-readable name for the step
- `error` (Exception): The error that occurred
- `context` (Optional[Dict[str, Any]]): Additional context for recovery

**Returns:**
- `RecoveryPlan`: Recovery plan with suggested actions

**Example:**
```python
try:
    # Step execution...
    pass
except Exception as error:
    recovery_plan = execution_manager.handle_step_error(
        "api_call",
        "Call External API",
        error,
        context={"retry_count": 2, "timeout": 30}
    )
    
    if recovery_plan.is_automated():
        print(f"Automated recovery available: {recovery_plan.description}")
    else:
        print(f"Manual intervention required: {recovery_plan.description}")
```

### Checkpoint Management

#### create_checkpoint()

Create a checkpoint for execution state persistence.

```python
def create_checkpoint(description: str = None) -> Checkpoint
```

**Parameters:**
- `description` (Optional[str]): Description of the checkpoint

**Returns:**
- `Checkpoint`: Created checkpoint with unique ID

**Example:**
```python
# Create checkpoint before critical step
checkpoint = execution_manager.create_checkpoint("Before data transformation")
print(f"Created checkpoint: {checkpoint.id}")

try:
    # Execute critical step...
    pass
except Exception:
    # Restore from checkpoint if needed
    execution_manager.restore_checkpoint(checkpoint.id)
```

#### restore_checkpoint()

Restore execution state from a previous checkpoint.

```python
def restore_checkpoint(checkpoint_id: str) -> bool
```

**Parameters:**
- `checkpoint_id` (str): ID of the checkpoint to restore

**Returns:**
- `bool`: True if restoration was successful

**Example:**
```python
# Create checkpoint
checkpoint = execution_manager.create_checkpoint("Before risky operation")

try:
    # Perform risky operation...
    risky_operation()
except Exception as e:
    print(f"Operation failed: {e}")
    print("Restoring from checkpoint...")
    
    success = execution_manager.restore_checkpoint(checkpoint.id)
    if success:
        print("Successfully restored from checkpoint")
        # Retry with different parameters
    else:
        print("Failed to restore checkpoint")
```

### Status and Monitoring

#### get_execution_status()

Get comprehensive status information about the execution.

```python
def get_execution_status() -> Dict[str, Any]
```

**Returns:**
- Dictionary containing detailed execution status information

**Example:**
```python
status = execution_manager.get_execution_status()

print(f"Execution ID: {status['execution_id']}")
print(f"Pipeline ID: {status['pipeline_id']}")  
print(f"Status: {status['status']}")

progress = status['progress']
print(f"Progress: {progress['percentage']:.1f}%")
print(f"Steps: {progress['completed_steps']}/{progress['total_steps']}")
print(f"Running: {progress['running_steps']}")
print(f"Failed: {progress['failed_steps']}")

metrics = status['metrics']
print(f"Start Time: {metrics['start_time']}")
print(f"Duration: {metrics['duration']} seconds")

recovery = status['recovery']
print(f"Recovery Status: {recovery}")
```

### Variable Access

The execution manager provides direct access to the integrated variable manager:

```python
# Set variables
execution_manager.variable_manager.set_variable(
    "customer_count",
    1500,
    var_type=VariableType.OUTPUT,
    source_step="load_data"
)

# Get variables  
customer_count = execution_manager.variable_manager.get_variable("customer_count")

# List all variables
variables = execution_manager.variable_manager.list_variables()
```

### Resource Management

#### cleanup()

Clean up execution resources without shutting down the manager.

```python
def cleanup() -> None
```

**Example:**
```python
# Clean up after execution completes
execution_manager.cleanup()
```

#### shutdown()

Shutdown all subsystems and release resources.

```python
def shutdown() -> None
```

**Example:**
```python
# Shutdown when done with execution manager
execution_manager.shutdown()
```

## ExecutionContext

The `ExecutionContext` manages execution state, metrics, and checkpoints.

### Constructor

```python
from orchestrator.execution import ExecutionContext, create_execution_context

# Create execution context
execution_context = create_execution_context(
    execution_id="my_execution", 
    pipeline_id="my_pipeline"
)

# Or use constructor directly
execution_context = ExecutionContext(
    execution_id="my_execution",
    pipeline_id="my_pipeline"
)
```

### Key Methods

#### start() / complete()

```python
# Start execution
execution_context.start()

# Complete execution
execution_context.complete(success=True)
```

#### start_step() / complete_step()

```python
# Start step
execution_context.start_step("step_1")

# Complete step  
execution_context.complete_step("step_1", success=True)
```

#### export_state() / import_state()

```python
# Export state for persistence
state = execution_context.export_state()

# Import state from persistence
execution_context.import_state(state)
```

## ExecutionStateBridge

Bridge between new execution system and legacy runtime state.

### Constructor

```python
from orchestrator.execution import ExecutionStateBridge

bridge = ExecutionStateBridge(
    execution_context=execution_context,
    pipeline_execution_state=legacy_pipeline_state
)
```

### Key Methods

#### register_variable() / register_step_result()

```python
# Register variable in both systems
bridge.register_variable(
    "result",
    {"status": "success", "count": 1500},
    var_type=VariableType.OUTPUT,
    source_step="process_data"
)

# Register step result
bridge.register_step_result("load_data", customer_data)
```

#### sync_to_pipeline_state() / sync_from_pipeline_state()

```python
# Sync variables to legacy system
bridge.sync_to_pipeline_state()

# Sync variables from legacy system  
bridge.sync_from_pipeline_state()
```

## Complete Usage Example

Here's a comprehensive example showing typical execution engine usage:

```python
import asyncio
from orchestrator.execution import create_comprehensive_execution_manager
from orchestrator.execution import VariableType, VariableScope

async def main():
    # Create execution manager
    execution_manager = create_comprehensive_execution_manager(
        execution_id="customer_analytics_001",
        pipeline_id="customer_analytics"
    )
    
    try:
        # Start execution
        execution_manager.start_execution(total_steps=4)
        
        # Step 1: Load data
        async def load_data():
            await asyncio.sleep(1)  # Simulate work
            return {"customers": 1500, "records": 45000}
        
        success = await execution_manager.execute_step_with_recovery(
            "load_data", "Load Customer Data", load_data
        )
        
        if not success:
            raise Exception("Failed to load data")
            
        # Store result
        data_info = await load_data()
        execution_manager.variable_manager.set_variable(
            "data_info", data_info,
            var_type=VariableType.OUTPUT,
            source_step="load_data"
        )
        
        # Step 2: Process data with progress tracking
        execution_manager.start_step("process_data", "Process Customer Records")
        
        records = data_info["records"]
        batch_size = 1000
        batches = (records + batch_size - 1) // batch_size
        
        processed_count = 0
        for i in range(batches):
            # Process batch
            await asyncio.sleep(0.5)  # Simulate processing
            processed_count += min(batch_size, records - i * batch_size)
            
            # Update progress
            progress = (i + 1) / batches * 100
            execution_manager.update_step_progress(
                "process_data",
                progress, 
                f"Processed {processed_count}/{records} records"
            )
        
        execution_manager.complete_step("process_data", success=True)
        
        # Step 3: Create checkpoint and perform risky operation
        checkpoint = execution_manager.create_checkpoint("Before aggregation")
        
        execution_manager.start_step("aggregate_data", "Aggregate Results")
        
        try:
            # Simulate risky operation that might fail
            await asyncio.sleep(1)
            if processed_count < records:
                raise Exception("Incomplete data processing")
                
            aggregated_results = {
                "total_customers": data_info["customers"],
                "processed_records": processed_count,
                "completion_rate": processed_count / records * 100
            }
            
            execution_manager.complete_step("aggregate_data", success=True)
            
        except Exception as e:
            print(f"Aggregation failed: {e}")
            
            # Restore from checkpoint
            if execution_manager.restore_checkpoint(checkpoint.id):
                print("Restored from checkpoint, retrying...")
                # Retry logic here
                execution_manager.complete_step("aggregate_data", success=False)
            
        # Step 4: Final step
        execution_manager.start_step("save_results", "Save Final Results")
        await asyncio.sleep(0.5)
        execution_manager.complete_step("save_results", success=True)
        
        # Complete execution
        execution_manager.complete_execution(success=True)
        
        # Get final status
        final_status = execution_manager.get_execution_status()
        print(f"Execution completed: {final_status['status']}")
        print(f"Total duration: {final_status['metrics']['duration']} seconds")
        print(f"Checkpoints created: {final_status['checkpoints']}")
        
    except Exception as e:
        print(f"Execution failed: {e}")
        execution_manager.complete_execution(success=False)
        
    finally:
        # Clean up resources
        execution_manager.cleanup()
        execution_manager.shutdown()

# Run the example
if __name__ == "__main__":
    asyncio.run(main())
```

## Best Practices

1. **Use Factory Functions**: Use `create_comprehensive_execution_manager()` instead of direct construction for proper integration.

2. **Coordinate Lifecycle**: Always call `start_execution()` before steps and `complete_execution()` when done.

3. **Track Progress**: Use `update_step_progress()` for long-running steps to provide user feedback.

4. **Handle Errors**: Use `execute_step_with_recovery()` for automatic error handling, or `handle_step_error()` for custom recovery.

5. **Create Strategic Checkpoints**: Create checkpoints before risky operations or at key milestones.

6. **Monitor Status**: Regularly check `get_execution_status()` for progress tracking and debugging.

7. **Clean Up**: Always call `cleanup()` and `shutdown()` to release resources properly.

8. **Use Variable Management**: Store step results and intermediate values using the integrated variable manager.