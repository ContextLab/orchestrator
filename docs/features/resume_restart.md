# Pipeline Resume and Restart

## Overview

The Pipeline Resume Manager provides comprehensive support for resuming failed or interrupted pipeline executions. It integrates with the state manager to create checkpoints and enables seamless recovery from failures.

## Key Features

### 1. Automatic Checkpointing

- **Task-level checkpoints**: Save state after each task completion
- **Periodic checkpoints**: Regular snapshots during long-running tasks
- **Error checkpoints**: Automatic saves on failures
- **Configurable intervals**: Customize checkpoint frequency

### 2. Resume Strategies

```python
from orchestrator.core.pipeline_resume_manager import ResumeStrategy

# Configure resume behavior
strategy = ResumeStrategy(
    retry_failed_tasks=True,        # Retry tasks that failed
    reset_running_tasks=True,       # Reset tasks that were running
    preserve_completed_tasks=True,  # Keep completed task results
    max_retry_attempts=3,          # Maximum retries per task
    checkpoint_interval_seconds=60  # Checkpoint every minute
)
```

### 3. Resume States

- **Completed tasks**: Preserved with their results
- **Failed tasks**: Tracked with retry counts
- **Running tasks**: Reset or marked as failed
- **Task results**: Maintained across resumes
- **Execution context**: Restored completely

## Usage Examples

### Basic Resume

```python
import orchestrator as orc

# Initialize and compile pipeline
orc.init_models()
pipeline = orc.compile("pipeline.yaml")

# Execute pipeline
try:
    result = await pipeline.run()
except Exception as e:
    print(f"Pipeline failed: {e}")
    
    # Resume from last checkpoint
    orchestrator = orc.Orchestrator()
    result = await orchestrator.resume_pipeline(execution_id)
```

### Custom Resume Strategy

```python
from orchestrator import Orchestrator, ResumeStrategy

# Create custom strategy
strategy = ResumeStrategy(
    retry_failed_tasks=True,
    max_retry_attempts=5,
    checkpoint_interval_seconds=30
)

orchestrator = Orchestrator()

# Resume with custom strategy
result = await orchestrator.resume_pipeline(
    execution_id="exec_123",
    resume_strategy=strategy
)
```

### Checking Resume Capability

```python
# Check if execution can be resumed
can_resume = await orchestrator.resume_manager.can_resume("exec_123")

if can_resume:
    # Get resume state details
    pipeline, resume_state = await orchestrator.resume_manager.get_resume_state("exec_123")
    
    print(f"Completed tasks: {len(resume_state.completed_tasks)}")
    print(f"Failed tasks: {len(resume_state.failed_tasks)}")
```

### Manual Checkpointing

```python
# Create checkpoint during execution
checkpoint_id = await orchestrator.resume_manager.create_resume_checkpoint(
    execution_id="exec_123",
    pipeline=pipeline,
    completed_tasks={"task1", "task2"},
    task_results={
        "task1": {"output": "data1"},
        "task2": {"output": "data2"}
    },
    context={"user": "john_doe"},
    failed_tasks={"task3": 1}  # task3 failed once
)
```

### Resume History

```python
# Get resume history for a pipeline
history = await orchestrator.resume_manager.get_resume_history(
    pipeline_id="my_pipeline",
    limit=10
)

for entry in history:
    print(f"Checkpoint: {entry['checkpoint_id']}")
    print(f"Timestamp: {entry['timestamp']}")
    print(f"Progress: {entry['completed_count']}/{entry['total_count']}")
```

## Integration with Pipeline Definition

### YAML Configuration

```yaml
id: resumable_pipeline
name: Pipeline with Resume Support
version: "1.0.0"

# Resume configuration
resume:
  enabled: true
  checkpoint_interval: 30  # seconds
  max_retries: 3
  retry_failed: true

steps:
  - id: data_fetch
    tool: filesystem
    action: read
    parameters:
      path: "data/input.csv"
    # Task-specific resume config
    resume:
      critical: true  # Always checkpoint after this task
      
  - id: process_data
    tool: data-processing
    # ...
```

## Best Practices

### 1. Checkpoint Strategy

- **Critical tasks**: Always checkpoint after data fetches or irreversible operations
- **Long tasks**: Enable periodic checkpointing for tasks > 1 minute
- **Resource-intensive**: Checkpoint before and after heavy computations
- **External calls**: Save state before API calls or network operations

### 2. Error Handling

```python
try:
    result = await pipeline.run()
except Exception as e:
    # Log error for debugging
    logger.error(f"Pipeline failed: {e}")
    
    # Check if resumable
    if await orchestrator.resume_manager.can_resume(execution_id):
        # Wait before retry (exponential backoff)
        await asyncio.sleep(retry_delay)
        
        # Resume with adjusted strategy
        strategy = ResumeStrategy(max_retry_attempts=5)
        result = await orchestrator.resume_pipeline(execution_id, strategy)
    else:
        # Start fresh if no checkpoint
        result = await pipeline.run()
```

### 3. State Management

- **Minimize state size**: Only checkpoint essential data
- **Compress large states**: Enable compression for big pipelines
- **Clean old checkpoints**: Regular cleanup of obsolete checkpoints
- **Version compatibility**: Ensure checkpoint format versioning

## Advanced Features

### Periodic Checkpointing

```python
# Start automatic checkpointing
await orchestrator.resume_manager.start_periodic_checkpointing(
    execution_id="exec_123",
    pipeline=pipeline,
    get_state_func=lambda: get_current_state(),
    interval=30  # seconds
)

# Stop when done
await orchestrator.resume_manager.stop_periodic_checkpointing("exec_123")
```

### State Validation

```python
# Validate checkpoint before resume
if strategy.validate_checkpoint_integrity:
    pipeline, resume_state = await orchestrator.resume_manager.get_resume_state(execution_id)
    
    # Verify state consistency
    for task_id in resume_state.completed_tasks:
        if task_id not in pipeline.tasks:
            raise ValueError(f"Invalid checkpoint: unknown task {task_id}")
```

## Performance Considerations

1. **Checkpoint overhead**: Balance frequency vs performance
2. **State size**: Use compression for large states
3. **Storage backend**: Choose appropriate backend (file, Redis, PostgreSQL)
4. **Cleanup policy**: Remove old checkpoints to save space

## Troubleshooting

### Common Issues

1. **"No resume checkpoint found"**
   - Ensure checkpointing was enabled during execution
   - Check if checkpoints were cleaned up
   - Verify execution ID is correct

2. **"Failed to load resume state"**
   - Check state manager health
   - Verify checkpoint file permissions
   - Ensure checkpoint format compatibility

3. **Resume loops (task keeps failing)**
   - Increase max_retry_attempts
   - Check task error logs
   - Consider skipping problematic tasks

### Debug Mode

```python
# Enable debug logging
import logging
logging.getLogger("orchestrator.resume_manager").setLevel(logging.DEBUG)

# Inspect checkpoint details
checkpoints = await orchestrator.state_manager.list_checkpoints(execution_id)
for cp in checkpoints:
    print(f"Checkpoint: {cp['checkpoint_id']}")
    print(f"Metadata: {cp['metadata']}")
```