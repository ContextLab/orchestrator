# Pipeline Status Tracking

## Overview

The Pipeline Status Tracker provides comprehensive monitoring and tracking of pipeline executions. It maintains detailed metrics, handles status transitions, and provides event notifications for pipeline and task status changes.

## Key Features

### 1. Execution Tracking

- **Unique Execution IDs**: Each pipeline run gets a unique execution ID
- **Status Management**: Tracks pipeline status (pending, running, paused, completed, failed, cancelled)
- **Task-Level Tracking**: Monitors individual task statuses within each execution
- **Metrics Collection**: Captures timing, errors, warnings, and custom metrics

### 2. Status States

```python
class PipelineStatus(Enum):
    PENDING = "pending"      # Not yet started
    RUNNING = "running"      # Currently executing
    PAUSED = "paused"       # Temporarily halted
    COMPLETED = "completed"  # Successfully finished
    FAILED = "failed"       # Encountered fatal error
    CANCELLED = "cancelled"  # Manually stopped
```

### 3. Execution Metrics

- **Duration Tracking**: Start time, end time, and total duration
- **Task Metrics**: Per-task execution times and custom metrics
- **Error/Warning Collection**: Structured logging of issues
- **Progress Calculation**: Real-time completion percentage

## Usage

### Basic Usage

```python
from orchestrator.core.pipeline_status_tracker import PipelineStatusTracker

# Create tracker
tracker = PipelineStatusTracker(max_history=1000)

# Start tracking an execution
execution = await tracker.start_execution(
    execution_id="exec_123",
    pipeline=pipeline,
    context={"user": "john_doe"}
)

# Update task status
await tracker.update_task_status(
    execution_id="exec_123",
    task_id="task_1",
    status=TaskStatus.COMPLETED,
    metrics={"duration": 1.5, "records_processed": 1000}
)

# Update pipeline status
await tracker.update_status("exec_123", PipelineStatus.COMPLETED)
```

### Event Handlers

```python
# Register status change handler
async def on_status_change(execution_id, new_status, old_status):
    print(f"Pipeline {execution_id}: {old_status} -> {new_status}")

tracker.register_status_handler(on_status_change)

# Register task status handler
async def on_task_status(execution_id, task_id, status):
    print(f"Task {task_id} in {execution_id}: {status}")

tracker.register_task_handler(on_task_status)
```

### Querying Status

```python
# Get execution details
execution = tracker.get_execution("exec_123")
print(f"Progress: {execution.progress}%")
print(f"Duration: {execution.metrics.duration}s")

# Get all running executions
running = tracker.get_running_executions()
for exec in running:
    print(f"{exec.execution_id}: {exec.pipeline.name}")

# Get execution summary
summary = tracker.get_execution_summary("exec_123")
print(summary)
```

### Statistics

```python
# Get overall statistics
stats = tracker.get_statistics()
print(f"Total executions: {stats['total_executions']}")
print(f"Success rate: {stats['success_rate']}%")
print(f"Currently running: {stats['running_executions']}")
```

## Integration with Orchestrator

The status tracker is integrated into the Orchestrator class:

```python
from orchestrator import Orchestrator

orchestrator = Orchestrator()

# Access status tracker
tracker = orchestrator.status_tracker

# Execute pipeline - automatically tracked
result = await orchestrator.execute_pipeline(pipeline)

# Check status
status = tracker.get_execution_summary(result['execution_id'])
```

## Benefits

1. **Visibility**: Real-time monitoring of pipeline executions
2. **Debugging**: Detailed metrics and error tracking
3. **Performance Analysis**: Historical data for optimization
4. **Event-Driven**: React to status changes programmatically
5. **Resource Management**: Track running executions for capacity planning

## Future Enhancements

- Persistence layer for long-term storage
- Web dashboard for visualization
- Alerting and notification integrations
- Advanced analytics and reporting
- Distributed execution tracking