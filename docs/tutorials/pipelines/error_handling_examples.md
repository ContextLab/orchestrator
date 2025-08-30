# Pipeline Tutorial: error_handling_examples

## Overview

**Complexity Level**: Advanced  
**Difficulty Score**: 75/100  
**Estimated Runtime**: 15+ minutes  

### Purpose
This pipeline demonstrates data_flow, error_handling, interactive_workflows and provides a practical example of orchestrator's capabilities for advanced-level workflows.

### Use Cases
- AI-powered content generation

### Prerequisites
- Basic understanding of YAML syntax
- Experience with intermediate pipeline patterns
- Understanding of error handling and system integration
- Familiarity with external APIs and tools

### Key Concepts
- Data flow between pipeline steps
- Error handling and recovery
- Large language model integration
- Template variable substitution

## Pipeline Breakdown

### Configuration Analysis
- **input_section**: Defines the data inputs and parameters for the pipeline
- **steps_section**: Contains the sequence of operations to be executed
- **output_section**: Specifies how results are formatted and stored
- **template_usage**: Uses 8 template patterns for dynamic content
- **feature_highlights**: Demonstrates 7 key orchestrator features

### Data Flow
This pipeline processes input parameters through 7 steps to generate the specified outputs.

### Control Flow
Follows linear execution flow from first step to last step.

### Pipeline Configuration
```yaml
# Advanced Error Handling Examples for Issue 192
# This file demonstrates the comprehensive error handling capabilities implemented in the orchestrator framework.

name: error_handling_showcase
version: 1.0.0
description: |
  Comprehensive examples of advanced error handling with real-world scenarios.
  Demonstrates the new ErrorHandler system with various recovery strategies.

inputs:
  api_endpoint:
    type: string
    default: "https://api.example.com"
    description: "Primary API endpoint"
  
  backup_endpoint:
    type: string 
    default: "https://backup-api.example.com"
    description: "Backup API endpoint for failover"
  
  data_file:
    type: string
    default: "./data/input.json"
    description: "Input data file path"

steps:
  # Example 1: Simple Error Handler with Fallback
  - id: simple_error_example
    action: "Process data from API"
    parameters:
      url: "{{api_endpoint}}/data"
      method: "GET"
    
    # Simple error handling - single handler
    on_error:
      handler_action: "Log error and use default data"
      error_types: ["ConnectionError", "TimeoutError"]
      fallback_value: {"data": "default_fallback_data"}
      log_level: "warning"

  # Example 2: Multiple Error Handlers with Priority
  - id: multi_handler_example
    action: "Fetch critical business data"
    parameters:
      primary_url: "{{api_endpoint}}/critical"
      timeout: 10
    
    # Multiple handlers in priority order
    on_error:
      # High priority: Try backup endpoint
      - handler_action: "Switch to backup endpoint: {{backup_endpoint}}/critical"
        error_types: ["ConnectionError", "HTTPError"]
        error_codes: [500, 502, 503, 504]
        priority: 1
        retry_with_handler: true
        max_handler_retries: 2
        timeout: 15
      
      # Medium priority: Retry with exponential backoff
      - handler_action: "Retry with exponential backoff"
        error_types: ["TimeoutError", "ConnectionError"]
        priority: 5
        retry_with_handler: true
        max_handler_retries: 3
      
      # Low priority: Alert and use cached data
      - handler_action: "Send alert and use cached data"
        error_types: ["*"]  # Catch all remaining errors
        priority: 10
        continue_on_handler_failure: true
        fallback_value: "{{cached_data}}"

  # Example 3: Advanced Error Pattern Matching
  - id: pattern_matching_example
    action: "Process user authentication"
    parameters:
      auth_endpoint: "{{api_endpoint}}/auth"
      credentials: "{{user_credentials}}"
    
    on_error:
      # Handle specific authentication errors
      - handler_action: "Refresh authentication token"
        error_types: ["PermissionError"]
        error_patterns: ["token.*expired", "unauthorized.*access", "invalid.*credentials"]
        error_codes: [401, 403]
        priority: 1
        retry_with_handler: true
      
      # Handle rate limiting
      - handler_action: "Wait and retry with backoff"
        error_types: ["ConnectionError"]
        error_patterns: ["rate.*limit.*exceeded", "too.*many.*requests"]
        error_codes: [429]
        priority: 2
        retry_with_handler: true
        max_handler_retries: 5
      
      # Handle server errors
      - handler_action: "Switch to alternative auth service"
        error_types: ["ConnectionError"]
        error_codes: [500, 502, 503]
        priority: 3
        retry_with_handler: true

  # Example 4: File System Error Handling
  - id: filesystem_error_example
    action: "Process data file"
    parameters:
      file_path: "{{data_file}}"
      backup_path: "./data/backup.json"
    
    on_error:
      # File not found - create default
      - handler_action: "Create default data file"
        error_types: ["FileNotFoundError"]
        priority: 1
        retry_with_handler: true
      
      # Permission denied - fix permissions
      - handler_action: "Fix file permissions and retry"
        error_types: ["PermissionError"]
        priority: 2
        retry_with_handler: true
        max_handler_retries: 1
      
      # Corrupted file - use backup
      - handler_action: "Use backup file: {{backup_path}}"
        error_types: ["ValueError", "UnicodeDecodeError"]
        priority: 3
        retry_with_handler: true

  # Example 5: Network Error Recovery Chain
  - id: network_recovery_example
    action: "Download external resource"
    parameters:
      primary_url: "{{api_endpoint}}/download"
      mirror_urls: 
        - "https://mirror1.example.com/resource"
        - "https://mirror2.example.com/resource"
        - "https://mirror3.example.com/resource"
    
    on_error:
      # Try mirror sites in sequence
      - handler_action: "Try first mirror: {{mirror_urls[0]}}"
        error_types: ["ConnectionError", "TimeoutError"]
        priority: 1
        retry_with_handler: true
        continue_on_handler_failure: true
      
      - handler_action: "Try second mirror: {{mirror_urls[1]}}"
        error_types: ["*"]
        priority: 2
        retry_with_handler: true
        continue_on_handler_failure: true
      
      - handler_action: "Try third mirror: {{mirror_urls[2]}}"
        error_types: ["*"]
        priority: 3
        retry_with_handler: true
        continue_on_handler_failure: true
      
      # Final fallback
      - handler_action: "Use local cached version"
        error_types: ["*"]
        priority: 10
        fallback_value: "local_cache://resource"

  # Example 6: Model API Error Handling
  - id: model_api_example
    action: "Generate AI response"
    parameters:
      model: "claude-sonnet-4-20250514"
      prompt: "Analyze the following data: {{processed_data}}"
      max_tokens: 1000
    
    on_error:
      # Authentication/API key issues
      - handler_action: "Refresh API credentials"
        error_types: ["PermissionError", "AuthenticationError"]
        error_patterns: ["api.*key.*invalid", "authentication.*failed"]
        error_codes: [401, 403]
        priority: 1
        retry_with_handler: true
      
      # Rate limiting
      - handler_action: "Wait for rate limit reset"
        error_types: ["ConnectionError"]
        error_patterns: ["rate.*limit", "quota.*exceeded"]
        error_codes: [429]
        priority: 2
        retry_with_handler: true
        max_handler_retries: 3
      
      # Model not available - use fallback model
      - handler_action: "Use fallback model: gpt-4"
        error_types: ["FileNotFoundError", "ValueError"]
        error_patterns: ["model.*not.*found", "model.*unavailable"]
        priority: 3
        retry_with_handler: true
      
      # Service temporarily down
      - handler_action: "Use local model as backup"
        error_types: ["ConnectionError", "TimeoutError"]
        priority: 5
        fallback_value: "Using local model due to service unavailability"

  # Example 7: Task Chain Error Recovery
  - id: data_processing_chain
    action: "Start data processing pipeline"
    parameters:
      input_source: "{{api_endpoint}}/raw-data"
    depends_on: [simple_error_example]
    
    on_error:
      - handler_task_id: "error_recovery_task"  # Reference to another task
        error_types: ["*"]
        priority: 1

  # Error recovery task (referenced above)
  - id: error_recovery_task
    action: "Execute comprehensive error recovery"
    parameters:
      failed_task_id: "{{error.failed_task_id}}"
      error_type: "{{error.error_type}}"
      error_message: "{{error.error_message}}"
      recovery_strategy: "comprehensive"

  # Example 8: Circuit Breaker Pattern
  - id: circuit_breaker_example
    action: "Call unreliable service"
    parameters:
      service_url: "{{api_endpoint}}/unreliable"
      max_failures: 5
      failure_window: 300  # 5 minutes
    
    on_error:
      - handler_action: "Implement circuit breaker logic"
        error_types: ["ConnectionError", "TimeoutError"]
        priority: 1
        # Circuit breaker is implemented in the handler logic
        retry_with_handler: false  # Don't retry immediately
        continue_on_handler_failure: true
        fallback_value: "Service temporarily unavailable - circuit breaker open"

  # Example 9: Advanced Context-Aware Error Handling
  - id: context_aware_example
    action: "Process user request with context"
    parameters:
      user_id: "{{user.id}}"
      request_type: "{{request.type}}"
      priority: "{{request.priority}}"
    
    on_error:
      # High priority requests get more aggressive recovery
      - handler_action: "High priority recovery for user {{user_id}}"
        error_types: ["*"]
        priority: 1
        # Only trigger for high priority requests
        enabled: "{{request.priority == 'high'}}"
        retry_with_handler: true
        max_handler_retries: 5
      
      # Normal priority requests
      - handler_action: "Standard recovery procedure"
        error_types: ["*"]
        priority: 5
        enabled: "{{request.priority != 'high'}}"
        retry_with_handler: true
        max_handler_retries: 2

  # Example 10: Monitoring and Alerting Integration
  - id: monitored_operation
    action: "Critical business operation with monitoring"
    parameters:
      operation_type: "financial_transaction"
      transaction_id: "{{transaction.id}}"
    
    on_error:
      # Immediate alerting for critical errors
      - handler_action: "Send critical alert to operations team"
        error_types: ["*"]
        priority: 1
        continue_on_handler_failure: true
        # Don't retry - just alert and continue to next handler
        retry_with_handler: false
        capture_error_context: true
        log_level: "critical"
      
      # Attempt automated recovery
      - handler_action: "Attempt automated transaction recovery"
        error_types: ["ConnectionError", "TimeoutError", "ValueError"]
        priority: 2
        retry_with_handler: true
        max_handler_retries: 3
      
      # Manual intervention required
      - handler_action: "Escalate to manual intervention queue"
        error_types: ["*"]
        priority: 10
        fallback_value: "Transaction queued for manual review"

# Global error handlers (applied to all tasks if no specific handlers match)
global_error_handlers:
  - handler_action: "Log all unhandled errors to central logging system"
    error_types: ["*"]
    priority: 1000  # Very low priority - only if no other handlers match
    continue_on_handler_failure: true
    capture_error_context: true
    log_level: "error"

outputs:
  processing_results: "{{data_processing_chain.result}}"
  error_summary: "{{error_recovery_task.result}}"
  monitoring_data: "{{monitored_operation.result}}"
```

## Customization Guide

### Input Modifications
- Modify input parameters to match your specific data sources
- Adjust file paths and data formats as needed for your environment

### Parameter Tuning
- Adjust model parameters (temperature, max_tokens) for different output styles
- Modify prompts to change the tone and focus of generated content
- Fine-tune performance parameters for your specific use case

### Step Modifications
- Add new steps by following the same pattern as existing ones
- Remove steps that aren't needed for your specific use case
- Reorder steps if your workflow requires different sequencing
- Replace tool actions with alternatives that provide similar functionality

### Output Customization
- Change output file paths and formats to match your requirements
- Modify output templates to customize the structure and content
- This pipeline produces JSON data - adjust output configuration accordingly

## Remixing Instructions

### Compatible Patterns
- fact_checker.yaml - for content verification
- research workflows - for information gathering

### Extension Ideas
- Build modular components for reusability
- Add performance monitoring and optimization
- Implement advanced security and access controls

### Combination Examples
- Can be combined with most other pipeline patterns

### Advanced Variations
- Scale to handle larger datasets and more complex processing
- Add real-time processing capabilities for streaming data
- Implement distributed processing across multiple systems
- Use multiple AI models for comparison and validation

## Hands-On Exercise

### Execution Instructions
- 1. Navigate to your orchestrator project directory
- 1.5. Ensure you have access to required services: Anthropic API, OpenAI API
- 2. Run: python scripts/run_pipeline.py examples/error_handling_examples.yaml
- 3. Monitor the output for progress and any error messages
- 4. Check the output directory for generated results

### Expected Outputs
- Generated JSON data in the specified output directory
- Execution logs showing step-by-step progress
- Completion message with runtime statistics
- No error messages or warnings (successful execution)

### Troubleshooting
- **API Authentication Errors**: Ensure all required API keys are properly configured in your environment
- **Template Resolution Errors**: Check that all input parameters are provided and template syntax is correct
- **Complex Logic Errors**: Review the pipeline configuration and ensure all advanced features are properly configured
- **General Execution Errors**: Check the logs for specific error messages and verify your orchestrator installation

### Verification Steps
- Check that the pipeline completed without errors
- Verify all expected output files were created
- Review the output content for quality and accuracy
- Review generated content for relevance and quality

---

*Tutorial generated on 2025-08-27T23:40:24.396099*
