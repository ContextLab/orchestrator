# Timeout Configuration

The Orchestrator framework supports configurable timeouts for pipeline tasks to prevent long-running operations from blocking execution indefinitely.

## Overview

Timeouts can be configured at multiple levels:
1. **Task-level timeout**: Specific timeout for individual tasks
2. **Metadata timeout**: Timeout specified in task metadata
3. **Pipeline default timeout**: Default timeout for all tasks in a pipeline

## Configuration

### Task-Level Timeout

Set a timeout directly on a task (in seconds):

```yaml
steps:
  - id: my_task
    action: "Process data"
    timeout: 30  # 30 second timeout
```

### Pipeline Default Timeout

Set a default timeout for all tasks in the pipeline config:

```yaml
name: my_pipeline
config:
  default_timeout: 60  # 60 seconds default for all tasks

steps:
  - id: task1
    action: "Quick operation"
    # Uses default_timeout of 60 seconds
    
  - id: task2
    action: "Slower operation"
    timeout: 120  # Override with task-specific timeout
```

### Metadata Timeout

Timeouts can also be specified in task metadata:

```yaml
steps:
  - id: complex_task
    action: "Complex operation"
    metadata:
      timeout: 90
      description: "Task with metadata-based timeout"
```

## Timeout Behavior

When a task exceeds its timeout:
1. The task is marked as failed with a `TimeoutError`
2. The error message indicates which task timed out and the timeout duration
3. Standard error handling applies (retries, on_error behavior)

## Example: Handling Timeouts

```yaml
name: timeout_handling
description: Example of timeout handling with retries and error continuation

steps:
  - id: unreliable_api_call
    tool: web-fetch
    parameters:
      url: "https://slow-api.example.com/data"
    timeout: 10  # 10 second timeout
    max_retries: 3  # Retry up to 3 times on timeout
    on_error: continue  # Continue pipeline even if all retries fail
    
  - id: fallback_task
    action: "Use cached data"
    dependencies: [unreliable_api_call]
    condition: "{{ steps.unreliable_api_call.error != None }}"
```

## Best Practices

1. **Set reasonable timeouts**: Balance between allowing enough time for operations and preventing indefinite hangs
2. **Use pipeline defaults**: Set a sensible default timeout at the pipeline level
3. **Override for specific tasks**: Long-running tasks should have explicit longer timeouts
4. **Consider retries**: Combine timeouts with retry logic for transient failures
5. **Handle timeout errors**: Use conditional logic to handle timeout scenarios gracefully

## Timeout Priority

Timeouts are resolved in the following order (first match wins):
1. Task-level `timeout` field
2. Task metadata `timeout` field  
3. Pipeline config `default_timeout`
4. No timeout (task runs indefinitely)

## Example: Research Pipeline with Timeouts

```yaml
name: research_pipeline
config:
  default_timeout: 30  # 30 seconds default

inputs:
  topic:
    type: string
    required: true

steps:
  - id: web_search
    tool: web-search
    parameters:
      query: "{{ topic }} latest research"
    timeout: 20  # Quick timeout for search
    
  - id: extract_content
    tool: web-extract
    parameters:
      url: "{{ steps.web_search.results[0].url }}"
    timeout: 60  # Longer timeout for content extraction
    
  - id: summarize
    action: "Summarize the extracted content"
    parameters:
      content: "{{ steps.extract_content.text }}"
    timeout: 120  # Even longer for AI summarization
    metadata:
      model_requirements:
        min_context_window: 8000
```

## Troubleshooting

### Common Issues

1. **Tasks timing out too quickly**: Increase the timeout value
2. **Pipeline hanging**: Add timeouts to prevent indefinite waits
3. **Timeout not applying**: Check timeout priority order
4. **Async operations**: Timeouts work with both sync and async operations

### Debugging Timeouts

Enable debug logging to see timeout information:

```python
import logging
logging.getLogger("orchestrator").setLevel(logging.DEBUG)
```

This will show when timeouts are applied and when tasks exceed their limits.