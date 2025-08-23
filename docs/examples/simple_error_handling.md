# Simple Error Handling Pipeline

**Pipeline**: `examples/simple_error_handling.yaml`  
**Category**: Quality & Testing  
**Complexity**: Beginner  
**Key Features**: Basic error handling, Fallback strategies, Retry logic, Multiple error types

## Overview

The Simple Error Handling Pipeline demonstrates fundamental error handling patterns for common scenarios. It showcases basic fallback strategies, retry mechanisms, file handling recovery, and multi-tier error handling approaches that form the foundation for robust pipeline design.

## Key Features Demonstrated

### 1. Simple Fallback Error Handling
```yaml
- id: api_call_with_fallback
  action: "Fetch data from API"
  parameters:
    url: "{{api_url}}/posts/1"
  
  on_error: "Use cached data when API fails"
```

### 2. Retry with Exponential Backoff
```yaml
on_error:
  handler_action: "Retry API call with exponential backoff"
  error_types: ["ConnectionError", "TimeoutError"]
  retry_with_handler: true
  max_handler_retries: 3
```

### 3. File Recovery Handling
```yaml
on_error:
  handler_action: "Create default data file"
  error_types: ["FileNotFoundError"]
  retry_with_handler: true
```

### 4. Priority-Based Error Handling
```yaml
on_error:
  - handler_action: "Retry on network error"
    error_types: ["ConnectionError", "TimeoutError"]
    priority: 1
  - handler_action: "Use default data on data error"
    error_types: ["ValueError", "KeyError"]
    priority: 2
  - handler_action: "Log unexpected error"
    error_types: ["*"]
    priority: 10
```

## Pipeline Architecture

### Input Parameters
- **api_url** (optional): API endpoint for testing (default: "https://jsonplaceholder.typicode.com")

### Processing Flow

1. **API Call with Fallback** - Demonstrate simple fallback error handling
2. **API Call with Retry** - Show retry logic for connection errors
3. **File Processing** - Handle file system errors with recovery
4. **Robust Operation** - Multi-tier error handling for complex scenarios

### Error Handling Patterns

#### Pattern 1: Simple String Fallback
```yaml
on_error: "Use cached data when API fails"
# Simplest form - just a string description
```

#### Pattern 2: Structured Retry
```yaml
on_error:
  handler_action: "Retry API call with exponential backoff"
  error_types: ["ConnectionError", "TimeoutError"]
  retry_with_handler: true
  max_handler_retries: 3
```

#### Pattern 3: File System Recovery
```yaml
on_error:
  handler_action: "Create default data file"
  error_types: ["FileNotFoundError"]
  retry_with_handler: true
```

#### Pattern 4: Multi-Tier Handling
```yaml
on_error:
  - priority: 1    # Handle network errors first
  - priority: 2    # Then handle data errors
  - priority: 10   # Finally catch all others
```

## Usage Examples

### Basic Error Handling Test
```bash
python scripts/run_pipeline.py examples/simple_error_handling.yaml
```

### Custom API Endpoint
```bash
python scripts/run_pipeline.py examples/simple_error_handling.yaml \
  -i api_url="https://api.example.com"
```

### Test with Unreachable Endpoint
```bash
python scripts/run_pipeline.py examples/simple_error_handling.yaml \
  -i api_url="https://unreachable-endpoint.invalid"
```

## Error Handling Scenarios

### Scenario 1: API Service Unavailable
```yaml
# Error: ConnectionError
# Handler: Simple fallback
# Action: "Use cached data when API fails"
# Result: Pipeline continues with fallback data
```

### Scenario 2: Network Timeout
```yaml
# Error: TimeoutError
# Handler: Retry with backoff
# Action: Retry up to 3 times with increasing delays
# Result: Success after retry or fallback after max attempts
```

### Scenario 3: Missing Data File
```yaml
# Error: FileNotFoundError
# Handler: Create default file
# Action: Generate default data file and retry
# Result: Pipeline continues with default data
```

### Scenario 4: Data Validation Error
```yaml
# Error: ValueError or KeyError
# Handler: Fallback value
# Action: Use empty data structure
# Result: Pipeline continues with safe fallback
```

## Error Types and Handlers

### Network-Related Errors
- **ConnectionError**: Network connectivity issues
- **TimeoutError**: Request timeout scenarios
- **Handler Strategy**: Retry with exponential backoff
- **Max Retries**: 2-3 attempts recommended

### File System Errors
- **FileNotFoundError**: Missing files or directories
- **PermissionError**: Access permission issues
- **Handler Strategy**: Create defaults or fix permissions
- **Recovery Action**: Generate required resources

### Data Processing Errors
- **ValueError**: Invalid data format or values
- **KeyError**: Missing required data fields
- **Handler Strategy**: Use fallback values or default structures
- **Safety Measure**: Prevent pipeline cascade failures

### Catch-All Handling
- **Error Pattern**: `["*"]` matches any error
- **Purpose**: Log unexpected errors for analysis
- **Strategy**: Continue pipeline execution when possible
- **Priority**: Lowest (10) to catch unhandled errors

## Retry Logic Implementation

### Basic Retry Configuration
```yaml
retry_with_handler: true          # Enable retry mechanism
max_handler_retries: 3            # Maximum retry attempts
# Exponential backoff automatically applied
```

### Retry Sequence
1. **First Attempt**: Immediate retry (0 seconds)
2. **Second Attempt**: Wait 1 second
3. **Third Attempt**: Wait 2 seconds  
4. **Fourth Attempt**: Wait 4 seconds
5. **Failure**: Execute next priority handler or fail

### Retry Best Practices
- **Network Operations**: 2-3 retries recommended
- **File Operations**: 1-2 retries usually sufficient
- **API Calls**: Consider rate limiting in retry strategy
- **Resource Creation**: Single retry after creation attempt

## Fallback Strategies

### Static Fallback Values
```yaml
fallback_value: {"posts": []}           # Empty data structure
fallback_value: "default_content"       # Default string
fallback_value: 0                       # Default number
```

### Dynamic Fallback Actions
```yaml
handler_action: "Use cached data when API fails"
handler_action: "Create default data file" 
handler_action: "Switch to backup service"
```

### Continuation Strategies
```yaml
continue_on_handler_failure: true       # Continue even if handler fails
continue_on_handler_failure: false      # Fail pipeline if handler fails
```

## Common Error Handling Patterns

### Pattern 1: API Client with Fallback
```yaml
on_error:
  - error_types: ["ConnectionError", "TimeoutError"]
    retry_with_handler: true
    max_handler_retries: 3
  - error_types: ["*"]
    fallback_value: "cached_data"
```

### Pattern 2: File Processor with Recovery
```yaml
on_error:
  - error_types: ["FileNotFoundError"]
    handler_action: "Create default file"
    retry_with_handler: true
  - error_types: ["PermissionError"]
    handler_action: "Fix file permissions"
    retry_with_handler: true
```

### Pattern 3: Data Validator with Defaults
```yaml
on_error:
  - error_types: ["ValueError", "KeyError"]
    fallback_value: {"status": "default"}
  - error_types: ["*"]
    continue_on_handler_failure: true
```

## Technical Implementation

### Handler Priority System
```yaml
priority: 1     # Highest priority (executed first)
priority: 5     # Medium priority  
priority: 10    # Lowest priority (catch-all)
```

### Error Type Matching
```yaml
error_types: ["ConnectionError"]           # Specific error type
error_types: ["ConnectionError", "TimeoutError"]  # Multiple types
error_types: ["*"]                        # Match any error
```

### Handler Actions
```yaml
# Simple string description
on_error: "Log error and continue"

# Structured handler configuration
on_error:
  handler_action: "Detailed recovery action"
  error_types: ["SpecificError"]
  retry_with_handler: true
```

## Best Practices Demonstrated

1. **Layered Error Handling**: Multiple handlers for different error types
2. **Appropriate Retry Logic**: Reasonable retry counts for different operations
3. **Graceful Degradation**: Continue with reduced functionality when possible
4. **Error Type Specificity**: Handle specific errors with appropriate strategies
5. **Fallback Values**: Provide safe defaults for critical data
6. **Logging Integration**: Log unexpected errors for debugging
7. **Priority Organization**: Handle most likely/critical errors first

## Common Use Cases

- **API Integration**: Handle service outages and network issues
- **File Processing**: Manage missing files and permission errors
- **Data Validation**: Provide defaults for malformed input
- **Service Communication**: Handle inter-service communication failures
- **Resource Management**: Create missing resources automatically
- **Workflow Resilience**: Continue processing despite individual step failures

## Troubleshooting

### Handler Not Triggering
- Verify error type names match actual exceptions
- Check handler priority ordering
- Ensure error types are correctly specified

### Retry Loop Issues
- Set appropriate `max_handler_retries` limits
- Verify retry conditions are achievable
- Monitor for infinite retry scenarios

### Fallback Value Problems
- Ensure fallback values match expected data types
- Test fallback values with downstream processing
- Validate fallback data structure compatibility

## Related Examples
- [error_handling_examples.md](error_handling_examples.md) - Advanced error handling patterns
- [simple_timeout_test.md](simple_timeout_test.md) - Timeout-specific scenarios
- [validation_pipeline.md](validation_pipeline.md) - Data validation with error handling

## Technical Requirements

- **Error Handling System**: Framework support for error handlers
- **Retry Mechanism**: Automatic retry with backoff capabilities
- **Logging**: Error logging and monitoring integration
- **Network Access**: Internet connectivity for API testing
- **File System**: Read/write access for file operations

This pipeline provides the foundation for understanding error handling patterns and serves as a template for building resilient, production-ready workflows.