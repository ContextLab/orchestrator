# Advanced Error Handling Examples Pipeline

**Pipeline**: `examples/error_handling_examples.yaml`  
**Category**: Quality & Testing  
**Complexity**: Expert  
**Key Features**: Advanced error handling, Multiple handlers, Priority-based recovery, Circuit breaker pattern

## Overview

The Advanced Error Handling Examples Pipeline showcases the comprehensive error handling capabilities of the orchestrator framework. It demonstrates 10 real-world scenarios with sophisticated recovery strategies, from simple fallbacks to complex circuit breaker patterns and context-aware recovery mechanisms.

## Key Features Demonstrated

### 1. Multiple Error Handlers with Priority
```yaml
on_error:
  - handler_action: "Switch to backup endpoint: {{backup_endpoint}}/critical"
    error_types: ["ConnectionError", "HTTPError"]
    priority: 1
  - handler_action: "Retry with exponential backoff"
    priority: 5
  - handler_action: "Send alert and use cached data"
    priority: 10
```

### 2. Advanced Pattern Matching
```yaml
on_error:
  - handler_action: "Refresh authentication token"
    error_patterns: ["token.*expired", "unauthorized.*access", "invalid.*credentials"]
    error_codes: [401, 403]
```

### 3. Circuit Breaker Implementation
```yaml
on_error:
  - handler_action: "Implement circuit breaker logic"
    retry_with_handler: false
    continue_on_handler_failure: true
    fallback_value: "Service temporarily unavailable - circuit breaker open"
```

## Pipeline Architecture

### Input Parameters
- **api_endpoint** (optional): Primary API endpoint (default: "https://api.example.com")
- **backup_endpoint** (optional): Backup API endpoint for failover
- **data_file** (optional): Input data file path for filesystem operations

### Processing Flow

1. **Simple Error Example** - Basic fallback demonstration
2. **Multi-Handler Example** - Priority-based error recovery
3. **Pattern Matching Example** - Advanced error classification
4. **Filesystem Error Example** - File operation error handling
5. **Network Recovery Example** - Sequential mirror site fallback
6. **Model API Example** - AI service error handling
7. **Task Chain Recovery** - Inter-task error propagation
8. **Circuit Breaker Example** - Service reliability patterns
9. **Context-Aware Example** - Dynamic handler selection
10. **Monitored Operation** - Critical business operation protection

### Error Handling Patterns

#### 1. Simple Fallback
```yaml
on_error:
  handler_action: "Log error and use default data"
  error_types: ["ConnectionError", "TimeoutError"]
  fallback_value: {"data": "default_fallback_data"}
```

#### 2. Priority-Based Recovery
```yaml
on_error:
  - priority: 1   # High priority: Try backup
  - priority: 5   # Medium priority: Retry
  - priority: 10  # Low priority: Use cache
```

#### 3. Pattern-Based Classification
```yaml
error_patterns: ["token.*expired", "unauthorized.*access"]
error_codes: [401, 403]
```

## Usage Examples

### Basic Error Handling Test
```bash
python scripts/run_pipeline.py examples/error_handling_examples.yaml
```

### With Custom Endpoints
```bash
python scripts/run_pipeline.py examples/error_handling_examples.yaml \
  -i api_endpoint="https://my-api.com" \
  -i backup_endpoint="https://backup-api.com"
```

### With Custom Data File
```bash
python scripts/run_pipeline.py examples/error_handling_examples.yaml \
  -i data_file="./custom_data/input.json"
```

## Advanced Error Handling Features

### 1. Handler Chaining
Multiple handlers execute in priority order:
```yaml
on_error:
  - priority: 1    # Try first
  - priority: 5    # Then try this
  - priority: 10   # Finally try this
```

### 2. Retry Logic
```yaml
retry_with_handler: true
max_handler_retries: 3
continue_on_handler_failure: true
```

### 3. Error Context Capture
```yaml
capture_error_context: true
log_level: "critical"
```

### 4. Conditional Handlers
```yaml
enabled: "{{request.priority == 'high'}}"
```

## Error Handler Types

### Connection Errors
- **Primary Strategy**: Switch to backup endpoints
- **Secondary Strategy**: Exponential backoff retry
- **Fallback**: Use cached data

### Authentication Errors
- **Pattern Matching**: Token expiration detection
- **Recovery Action**: Automatic token refresh
- **Escalation**: Alternative auth service

### File System Errors
- **FileNotFoundError**: Create default file
- **PermissionError**: Fix permissions and retry
- **Corruption**: Use backup file

### API Rate Limiting
- **Detection**: Rate limit patterns and HTTP 429
- **Strategy**: Wait with exponential backoff
- **Alternative**: Switch to backup service

## Circuit Breaker Pattern

### Implementation
```yaml
parameters:
  max_failures: 5
  failure_window: 300  # 5 minutes

on_error:
  - handler_action: "Implement circuit breaker logic"
    retry_with_handler: false
    fallback_value: "Service temporarily unavailable"
```

### States
1. **Closed**: Normal operation
2. **Open**: Service unavailable, return fallback
3. **Half-Open**: Test service recovery

## Context-Aware Recovery

### Priority-Based Handling
```yaml
# High priority requests get aggressive recovery
- enabled: "{{request.priority == 'high'}}"
  max_handler_retries: 5

# Normal priority requests get standard recovery
- enabled: "{{request.priority != 'high'}}"
  max_handler_retries: 2
```

## Global Error Handlers

### System-Wide Fallback
```yaml
global_error_handlers:
  - handler_action: "Log all unhandled errors to central logging system"
    error_types: ["*"]
    priority: 1000
    capture_error_context: true
```

## Monitoring Integration

### Critical Operations
```yaml
on_error:
  - handler_action: "Send critical alert to operations team"
    priority: 1
    log_level: "critical"
  - handler_action: "Attempt automated transaction recovery"
    priority: 2
  - handler_action: "Escalate to manual intervention queue"
    priority: 10
```

## Technical Implementation

### Error Type Classification
- **ConnectionError**: Network-related issues
- **TimeoutError**: Request timeout scenarios
- **PermissionError**: Authentication/authorization
- **FileNotFoundError**: Missing resources
- **ValueError**: Data validation failures
- **AuthenticationError**: API key issues

### Handler Selection Logic
1. Match error type against `error_types`
2. Check error code against `error_codes`
3. Test error message against `error_patterns`
4. Evaluate `enabled` conditions
5. Execute handlers in priority order

### Recovery Strategies
- **Immediate Retry**: Simple retry with same parameters
- **Modified Retry**: Retry with different endpoints/parameters
- **Fallback Value**: Return predetermined safe value
- **Task Delegation**: Hand off to specialized recovery task
- **Circuit Breaking**: Temporarily disable failing service

## Best Practices Demonstrated

1. **Layered Recovery**: Multiple fallback strategies
2. **Priority Ordering**: Critical operations first
3. **Context Awareness**: Dynamic handler selection
4. **Monitoring Integration**: Alert and logging systems
5. **Graceful Degradation**: Service continues with reduced functionality
6. **Circuit Breaker**: Prevent cascade failures
7. **Error Context**: Capture detailed error information
8. **Conditional Logic**: Smart handler activation

## Common Use Cases

- **API Integration**: Handle service outages and rate limits
- **File Processing**: Manage missing or corrupted files
- **Authentication**: Handle token expiration and refresh
- **Data Validation**: Process malformed or incomplete data
- **Network Operations**: Manage connectivity issues
- **Business Logic**: Protect critical operations
- **Monitoring Systems**: Integrate with alerting platforms

## Troubleshooting

### Handler Not Triggering
- Verify `error_types` matches actual error
- Check `error_patterns` regex syntax
- Ensure `enabled` conditions evaluate to true
- Confirm priority ordering

### Infinite Retry Loops
- Set appropriate `max_handler_retries`
- Use `continue_on_handler_failure: false` when needed
- Implement circuit breaker for persistent failures

### Performance Issues
- Reduce retry attempts for non-critical operations
- Implement exponential backoff
- Use circuit breaker for failing services
- Cache successful responses

## Related Examples
- [simple_error_handling.md](simple_error_handling.md) - Basic error handling patterns
- [simple_timeout_test.md](simple_timeout_test.md) - Timeout-specific handling
- [validation_pipeline.md](validation_pipeline.md) - Data validation with error recovery

## Technical Requirements

- **Framework**: Advanced error handling system
- **Logging**: Structured logging capability
- **Monitoring**: Integration with alerting systems
- **Network**: Multiple endpoints for failover testing
- **Storage**: File system access for backup strategies

This pipeline demonstrates enterprise-grade error handling suitable for production systems requiring high availability and graceful failure recovery.