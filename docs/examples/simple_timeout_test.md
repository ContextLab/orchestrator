# Simple Timeout Test Pipeline

**Pipeline**: `examples/simple_timeout_test.yaml`  
**Category**: Quality & Testing  
**Complexity**: Beginner  
**Key Features**: Timeout configuration, Execution limits, Python code execution, Test scenarios

## Overview

The Simple Timeout Test Pipeline demonstrates basic timeout functionality by executing a Python sleep operation that exceeds the configured timeout limit. It serves as a simple test case for validating timeout mechanisms and understanding execution time constraints.

## Key Features Demonstrated

### 1. Step-Level Timeout Configuration
```yaml
- id: test_timeout
  action: "Sleep for 5 seconds to test timeout"
  timeout: 2  # This should timeout after 2 seconds
```

### 2. Python Code Execution
```yaml
tool: python-executor
parameters:
  code: |
    import time
    print("Starting sleep...")
    time.sleep(5)
    print("Finished sleeping")
    return {"status": "completed"}
```

### 3. Intentional Timeout Scenario
```yaml
# Sleep duration: 5 seconds
# Timeout limit: 2 seconds  
# Expected result: TimeoutError after 2 seconds
```

## Pipeline Architecture

### Input Parameters
None (uses hardcoded timeout values for testing)

### Processing Flow

1. **Execute Timeout Test** - Run Python sleep operation with timeout constraint

### Timeout Configuration

#### Step Timeout
```yaml
timeout: 2  # 2-second timeout limit
```

#### Execution Details
- **Sleep Duration**: 5 seconds (hardcoded in Python)
- **Timeout Limit**: 2 seconds (configured timeout)
- **Expected Behavior**: Operation terminated after 2 seconds
- **Error Type**: TimeoutError or similar timeout exception

## Usage Examples

### Basic Timeout Test
```bash
python scripts/run_pipeline.py examples/simple_timeout_test.yaml
# Expected: Timeout error after ~2 seconds
```

### Timeout Behavior Analysis
```bash
# Run with timing to observe timeout behavior
time python scripts/run_pipeline.py examples/simple_timeout_test.yaml
# Should complete in ~2 seconds instead of 5 seconds
```

## Expected Execution Behavior

### Timeline
```
t=0s:    Pipeline starts
t=0s:    Python sleep operation begins
t=0s:    "Starting sleep..." printed to output
t=2s:    Timeout limit reached
t=2s:    Operation forcibly terminated
t=2s:    TimeoutError raised
t=2s:    Pipeline execution fails
```

### Output Sequence
1. **Start Message**: "Starting sleep..." appears in logs
2. **Timeout Trigger**: After 2 seconds, timeout mechanism activates
3. **Termination**: Python process terminated before completion
4. **Missing Output**: "Finished sleeping" never appears
5. **Error Result**: Pipeline fails with timeout error

## Timeout Mechanism Details

### Timeout Types
- **Step Timeout**: Applied to individual pipeline steps
- **Global Timeout**: Applied to entire pipeline (not shown in this example)
- **Tool Timeout**: Applied to specific tool operations

### Timeout Units
```yaml
timeout: 2          # 2 seconds (integer)
timeout: 2.5        # 2.5 seconds (float)
timeout: 120        # 2 minutes as seconds
```

### Timeout Scope
```yaml
# Step-level timeout (applies to this step only)
- id: test_timeout
  timeout: 2
  
# Could also be configured at pipeline level
timeout: 30         # Global pipeline timeout
```

## Python Executor Integration

### Code Execution Environment
```yaml
tool: python-executor
parameters:
  code: |
    # Python code executed in isolated environment
    import time
    time.sleep(5)
    return {"status": "completed"}
```

### Return Value Handling
```yaml
# Successful return (if no timeout)
return {"status": "completed"}

# Timeout prevents return value
# Results in timeout error instead
```

## Testing Timeout Scenarios

### Scenario 1: Timeout Occurs (Current Configuration)
```yaml
sleep_duration: 5 seconds
timeout_limit: 2 seconds
result: TimeoutError after 2 seconds
```

### Scenario 2: No Timeout (Modified Configuration)
```yaml
sleep_duration: 5 seconds  
timeout_limit: 10 seconds
result: Successful completion after 5 seconds
```

### Scenario 3: Immediate Timeout
```yaml
sleep_duration: 5 seconds
timeout_limit: 0.1 seconds  
result: Immediate timeout, no output
```

## Timeout Error Handling

### Default Behavior
- **Timeout Reached**: Operation terminated immediately
- **Error Type**: TimeoutError or similar timeout exception
- **Pipeline Result**: Step failure, pipeline failure
- **Cleanup**: Process resources cleaned up automatically

### Enhanced Error Handling
```yaml
- id: test_timeout
  timeout: 2
  on_error:
    handler_action: "Log timeout and continue with fallback"
    error_types: ["TimeoutError"]
    fallback_value: {"status": "timeout_occurred"}
```

## Practical Applications

### Performance Testing
- Validate timeout mechanisms work correctly
- Test operation termination behavior
- Verify resource cleanup after timeout

### Integration Testing  
- Test timeout behavior in CI/CD pipelines
- Validate timeout configurations for production
- Ensure graceful handling of long-running operations

### Development and Debugging
- Test timeout settings during development
- Debug hanging operations and processes
- Validate execution time constraints

## Common Timeout Patterns

### API Call Timeouts
```yaml
- id: api_request
  tool: http-client
  parameters:
    url: "https://slow-api.example.com/data"
  timeout: 30  # 30-second timeout for API calls
```

### File Processing Timeouts
```yaml
- id: process_large_file
  tool: data-processor
  parameters:
    input_file: "large_dataset.csv"
  timeout: 300  # 5-minute timeout for large files
```

### Model Inference Timeouts
```yaml
- id: ai_analysis
  action: generate_text
  parameters:
    model: "claude-sonnet-4-20250514"
    prompt: "Complex analysis task..."
  timeout: 60  # 1-minute timeout for AI operations
```

## Best Practices

### 1. Appropriate Timeout Values
- **API Calls**: 10-30 seconds typically
- **File Processing**: Based on file size and complexity
- **AI Operations**: 30-120 seconds for complex tasks
- **Database Operations**: 5-30 seconds usually sufficient

### 2. Error Handling Integration
```yaml
on_error:
  - error_types: ["TimeoutError"]
    handler_action: "Handle timeout gracefully"
    fallback_value: "timeout_result"
```

### 3. Progressive Timeouts
```yaml
# Start with shorter timeout, increase if needed
timeout: 10      # Initial attempt
# If timeout occurs, retry with longer timeout
```

### 4. Monitoring and Alerting
- Log timeout occurrences for analysis
- Monitor timeout rates across operations
- Alert on excessive timeout patterns

## Troubleshooting

### Timeouts Not Working
- Verify timeout values are correctly specified
- Check if tool supports timeout configuration
- Ensure timeout mechanism is properly implemented

### Operations Timing Out Unexpectedly
- Increase timeout values for complex operations
- Monitor actual execution times
- Consider operation optimization

### Inconsistent Timeout Behavior
- Check system load and resource availability
- Verify timeout implementation consistency
- Test with different operation types

## Related Examples
- [simple_error_handling.md](simple_error_handling.md) - Error handling including timeout errors
- [error_handling_examples.md](error_handling_examples.md) - Advanced timeout and error scenarios
- [terminal_automation.md](terminal_automation.md) - Command execution with timeouts

## Technical Requirements

- **Python Executor**: Python environment for code execution
- **Timeout Support**: Framework-level timeout implementation
- **Process Management**: Ability to terminate long-running processes
- **Error Handling**: Timeout exception handling capabilities

This pipeline provides a fundamental understanding of timeout mechanisms and serves as a building block for implementing robust execution time constraints in production workflows.