# Advanced Patterns in Orchestrator

This guide covers sophisticated workflow patterns that enable you to build complex, production-ready pipelines. These patterns leverage Orchestrator's advanced features for control flow, error handling, optimization, and system integration.

## Table of Contents

1. [Control Flow Patterns](#control-flow-patterns)
2. [Dynamic Pipeline Generation](#dynamic-pipeline-generation)
3. [Advanced Error Handling](#advanced-error-handling)
4. [Parallel Execution](#parallel-execution)
5. [State Management](#state-management)
6. [System Integration](#system-integration)
7. [Performance Optimization](#performance-optimization)
8. [Security Patterns](#security-patterns)

## Control Flow Patterns

### Conditional Execution

Execute tasks based on runtime conditions:

```yaml
name: "Conditional Processing Pipeline"

input_variables:
  data_type:
    type: string
    options: ["text", "image", "audio"]

tasks:
  - name: "analyze_input"
    type: "python_task"
    script: |
      data_type = "{{ data_type }}"
      return {
          "is_text": data_type == "text",
          "is_image": data_type == "image", 
          "is_audio": data_type == "audio",
          "processing_strategy": f"{data_type}_processing"
      }

  # Conditional text processing
  - name: "process_text"
    type: "llm_task"
    model: "gpt-4"
    prompt: "Analyze this text: {{ input_content }}"
    condition: "{{ analyze_input.is_text }}"
    
  # Conditional image processing  
  - name: "process_image"
    type: "python_task"
    script: |
      from PIL import Image
      import numpy as np
      
      # Image processing logic here
      return {"dimensions": [1024, 768], "format": "RGB"}
    condition: "{{ analyze_input.is_image }}"
    
  # Conditional audio processing
  - name: "process_audio"
    type: "python_task" 
    script: |
      import librosa
      
      # Audio processing logic here
      return {"sample_rate": 44100, "duration": 180}
    condition: "{{ analyze_input.is_audio }}"
```

### For Loops with Dynamic Data

Process collections of items:

```yaml
name: "Batch Processing Pipeline"

input_variables:
  items:
    type: array
    description: "List of items to process"

tasks:
  - name: "process_batch"
    type: "for_loop"
    iterate_over: "{{ items }}"
    item_variable: "current_item"
    tasks:
      - name: "validate_item"
        type: "python_task"
        script: |
          item = {{ current_item }}
          if not item.get('id'):
              raise ValueError(f"Item missing ID: {item}")
          return {"valid": True, "item_id": item['id']}
          
      - name: "process_item"
        type: "llm_task"
        model: "gpt-3.5-turbo"
        prompt: |
          Process this item:
          ID: {{ current_item.id }}
          Content: {{ current_item.content }}
          
          Provide a structured analysis.
        depends_on: ["validate_item"]
        
      - name: "store_result"
        type: "python_task"
        script: |
          result = context.get_task_output('process_item')
          item_id = context.get_task_output('validate_item')['item_id']
          
          # Store in database or file
          storage[item_id] = result
          return {"stored": True, "item_id": item_id}
```

### While Loops with Conditions

Implement iterative improvement patterns:

```yaml
name: "Iterative Improvement Pipeline"

input_variables:
  initial_content: 
    type: string
  quality_threshold:
    type: float
    default: 0.8

tasks:
  - name: "improve_content"
    type: "while_loop" 
    condition: "{{ not loop_state.quality_met }}"
    max_iterations: 5
    tasks:
      - name: "analyze_quality"
        type: "llm_task"
        model: "gpt-4"
        prompt: |
          Rate the quality of this content on a scale of 0-1:
          
          {{ loop_state.current_content | default(initial_content) }}
          
          Consider: clarity, completeness, accuracy, engagement.
          Respond with just the number.
        output_type: "number"
        
      - name: "check_threshold"
        type: "python_task"
        script: |
          quality_score = float(context.get_task_output('analyze_quality'))
          quality_threshold = float("{{ quality_threshold }}")
          
          return {
              "quality_score": quality_score,
              "quality_met": quality_score >= quality_threshold,
              "needs_improvement": quality_score < quality_threshold
          }
          
      - name: "improve_if_needed"
        type: "llm_task"
        model: "gpt-4"
        prompt: |
          The current content has a quality score of {{ check_threshold.quality_score }}.
          Please improve it to be more clear, complete, and engaging:
          
          {{ loop_state.current_content | default(initial_content) }}
        condition: "{{ check_threshold.needs_improvement }}"
        
      - name: "update_state"
        type: "python_task"
        script: |
          check_result = context.get_task_output('check_threshold')
          improved_content = context.get_task_output('improve_if_needed')
          current_content = context.get_loop_variable('current_content', "{{ initial_content }}")
          
          return {
              "current_content": improved_content if improved_content else current_content,
              "quality_met": check_result['quality_met'],
              "iteration": context.get_loop_iteration() + 1
          }
```

## Dynamic Pipeline Generation

Create pipelines that modify themselves at runtime:

```yaml
name: "Dynamic Task Generation"

input_variables:
  workflow_type:
    type: string
    options: ["research", "creative", "analysis"]

tasks:
  - name: "generate_workflow"
    type: "python_task"
    script: |
      workflow_type = "{{ workflow_type }}"
      
      if workflow_type == "research":
          return {
              "task_list": [
                  {"type": "web_search_task", "query": "{{ research_topic }}"},
                  {"type": "llm_task", "prompt": "Summarize research findings"},
                  {"type": "python_task", "script": "generate_report()"}
              ]
          }
      elif workflow_type == "creative":
          return {
              "task_list": [
                  {"type": "llm_task", "prompt": "Generate creative ideas"},
                  {"type": "llm_task", "prompt": "Refine and expand ideas"},
                  {"type": "python_task", "script": "format_creative_output()"}
              ]
          }
      # Add more workflow types as needed
      
  - name: "execute_dynamic_tasks"
    type: "dynamic_task_group"
    tasks: "{{ generate_workflow.task_list }}"
    parallel: false
```

## Advanced Error Handling

### Hierarchical Error Handling

```yaml
name: "Robust Processing Pipeline"

# Global error handling
error_handling:
  default_retry_count: 3
  retry_delay: 5
  on_failure: "continue_with_degraded_service"

tasks:
  - name: "critical_task"
    type: "llm_task"
    model: "gpt-4"
    prompt: "{{ critical_prompt }}"
    error_handling:
      retry_count: 5
      retry_backoff: "exponential"
      fallback:
        type: "llm_task"
        model: "gpt-3.5-turbo"
        prompt: "{{ simplified_prompt }}"
      on_failure: "abort_pipeline"
      
  - name: "optional_enhancement"
    type: "llm_task"
    model: "claude-3"
    prompt: "Enhance: {{ critical_task }}"
    error_handling:
      on_failure: "skip"
      fallback_value: "{{ critical_task }}"
      
  - name: "recovery_task"
    type: "python_task"
    script: |
      # Check if previous tasks succeeded
      critical_result = context.get_task_output('critical_task', None)
      enhanced_result = context.get_task_output('optional_enhancement', None)
      
      if not critical_result:
          # Implement emergency fallback
          return {"status": "emergency_mode", "result": "minimal_output"}
          
      return {
          "status": "success",
          "result": enhanced_result or critical_result
      }
```

### Circuit Breaker Pattern

```yaml
name: "Circuit Breaker Example"

tasks:
  - name: "external_service_call"
    type: "python_task"
    script: |
      import time
      import random
      from datetime import datetime, timedelta
      
      # Circuit breaker state management
      circuit_state = context.get_global_state('circuit_breaker', {
          'failures': 0,
          'last_failure': None,
          'state': 'closed'  # closed, open, half_open
      })
      
      # Check circuit state
      if circuit_state['state'] == 'open':
          if circuit_state['last_failure']:
              last_failure = datetime.fromisoformat(circuit_state['last_failure'])
              if datetime.now() - last_failure < timedelta(minutes=5):
                  raise Exception("Circuit breaker is open")
              else:
                  circuit_state['state'] = 'half_open'
      
      try:
          # Simulate external service call
          if random.random() < 0.3:  # 30% failure rate
              raise Exception("External service error")
          
          # Success - reset circuit breaker
          circuit_state = {'failures': 0, 'last_failure': None, 'state': 'closed'}
          context.set_global_state('circuit_breaker', circuit_state)
          
          return {"result": "success", "circuit_state": "closed"}
          
      except Exception as e:
          # Handle failure
          circuit_state['failures'] += 1
          circuit_state['last_failure'] = datetime.now().isoformat()
          
          if circuit_state['failures'] >= 3:
              circuit_state['state'] = 'open'
              
          context.set_global_state('circuit_breaker', circuit_state)
          raise e
```

## Parallel Execution

### Task-Level Parallelism

```yaml
name: "Parallel Processing Pipeline"

tasks:
  - name: "data_preprocessing"
    type: "python_task"
    script: |
      # Prepare data for parallel processing
      data_chunks = chunk_data("{{ input_data }}", chunk_size=100)
      return {"chunks": data_chunks, "total_chunks": len(data_chunks)}
      
  # Process chunks in parallel
  - name: "parallel_processing"
    type: "parallel_group"
    depends_on: ["data_preprocessing"]
    max_concurrency: 4
    tasks:
      - name: "process_chunk_1"
        type: "llm_task"
        model: "gpt-3.5-turbo"
        prompt: "Process this data chunk: {{ data_preprocessing.chunks.0 }}"
        
      - name: "process_chunk_2" 
        type: "llm_task"
        model: "gpt-3.5-turbo"
        prompt: "Process this data chunk: {{ data_preprocessing.chunks.1 }}"
        
      - name: "process_chunk_3"
        type: "llm_task" 
        model: "gpt-3.5-turbo"
        prompt: "Process this data chunk: {{ data_preprocessing.chunks.2 }}"
        
      - name: "process_chunk_4"
        type: "llm_task"
        model: "gpt-3.5-turbo"
        prompt: "Process this data chunk: {{ data_preprocessing.chunks.3 }}"
        
  - name: "aggregate_results"
    type: "python_task"
    depends_on: ["parallel_processing"]
    script: |
      results = []
      for i in range(1, 5):
          task_name = f"process_chunk_{i}"
          result = context.get_task_output(task_name)
          results.append(result)
      
      # Combine results
      combined_result = combine_results(results)
      return {"final_result": combined_result, "processed_chunks": len(results)}
```

### Pipeline-Level Parallelism

```yaml
name: "Multi-Pipeline Orchestration"

tasks:
  - name: "launch_parallel_pipelines"
    type: "pipeline_group"
    mode: "parallel"
    pipelines:
      - pipeline: "data_analysis_pipeline.yaml"
        inputs: {"dataset": "{{ dataset_a }}"}
      - pipeline: "model_training_pipeline.yaml"
        inputs: {"training_data": "{{ training_set }}"}
      - pipeline: "validation_pipeline.yaml"
        inputs: {"validation_data": "{{ validation_set }}"}
        
  - name: "consolidate_pipeline_results"
    type: "python_task"
    depends_on: ["launch_parallel_pipelines"]
    script: |
      analysis_result = context.get_pipeline_output('data_analysis_pipeline.yaml')
      training_result = context.get_pipeline_output('model_training_pipeline.yaml')  
      validation_result = context.get_pipeline_output('validation_pipeline.yaml')
      
      return {
          "analysis": analysis_result,
          "model": training_result,
          "validation": validation_result,
          "combined_score": calculate_combined_score(analysis_result, validation_result)
      }
```

## State Management

### Persistent State Across Runs

```yaml
name: "Stateful Processing Pipeline"

state_management:
  persistence: true
  storage_backend: "redis"  # or "filesystem", "database"
  ttl: 86400  # 24 hours

tasks:
  - name: "load_previous_state"
    type: "python_task"
    script: |
      previous_runs = context.get_persistent_state('previous_runs', [])
      last_result = context.get_persistent_state('last_result', None)
      
      return {
          "run_count": len(previous_runs),
          "has_previous_result": last_result is not None,
          "last_result": last_result
      }
      
  - name: "incremental_processing"
    type: "llm_task"
    model: "gpt-4"
    prompt: |
      {% if load_previous_state.has_previous_result %}
      Continue from previous result:
      {{ load_previous_state.last_result }}
      
      New input to process: {{ new_input }}
      {% else %}
      Process this input from scratch: {{ new_input }}
      {% endif %}
      
  - name: "save_state"
    type: "python_task"
    script: |
      from datetime import datetime
      
      result = context.get_task_output('incremental_processing')
      previous_runs = context.get_persistent_state('previous_runs', [])
      
      # Update state
      run_info = {
          "timestamp": datetime.now().isoformat(),
          "result_preview": result[:100] + "..." if len(result) > 100 else result
      }
      
      previous_runs.append(run_info)
      context.set_persistent_state('previous_runs', previous_runs[-10:])  # Keep last 10
      context.set_persistent_state('last_result', result)
      
      return {"state_saved": True, "total_runs": len(previous_runs)}
```

### Cross-Pipeline State Sharing

```yaml
name: "Shared State Pipeline"

tasks:
  - name: "update_global_metrics"
    type: "python_task"
    script: |
      # Access shared metrics across all pipeline instances
      global_metrics = context.get_shared_state('global_metrics', {
          'total_processed': 0,
          'success_rate': 1.0,
          'last_updated': None
      })
      
      # Update metrics
      global_metrics['total_processed'] += 1
      global_metrics['last_updated'] = datetime.now().isoformat()
      
      context.set_shared_state('global_metrics', global_metrics)
      
      return global_metrics
```

## System Integration

### Database Integration Pattern

```yaml
name: "Database Integration Pipeline"

dependencies:
  - "psycopg2-binary"  # PostgreSQL adapter
  - "sqlalchemy"       # ORM

tasks:
  - name: "database_operation"
    type: "python_task"
    script: |
      import os
      from sqlalchemy import create_engine, text
      from datetime import datetime
      
      # Database connection
      db_url = os.getenv('DATABASE_URL', 'postgresql://user:pass@localhost/dbname')
      engine = create_engine(db_url)
      
      try:
          with engine.connect() as conn:
              # Execute query
              result = conn.execute(text("SELECT * FROM data_table WHERE status = :status"), 
                                  {"status": "{{ status_filter }}"})
              rows = result.fetchall()
              
              # Process results
              processed_data = []
              for row in rows:
                  processed_data.append({
                      "id": row.id,
                      "data": row.data,
                      "processed_at": datetime.now().isoformat()
                  })
                  
              # Update status
              conn.execute(text("UPDATE data_table SET status = 'processed' WHERE status = :old_status"),
                         {"old_status": "{{ status_filter }}"})
              conn.commit()
              
              return {"processed_count": len(processed_data), "data": processed_data}
              
      except Exception as e:
          return {"error": str(e), "processed_count": 0}
```

### API Integration with Authentication

```yaml
name: "API Integration Pipeline"

tasks:
  - name: "authenticated_api_call"
    type: "python_task"
    script: |
      import requests
      import os
      from datetime import datetime, timedelta
      
      # Token management
      cached_token = context.get_task_cache('api_token')
      token_expiry = context.get_task_cache('token_expiry')
      
      # Check if token is still valid
      if not cached_token or (token_expiry and datetime.now() > datetime.fromisoformat(token_expiry)):
          # Refresh token
          auth_response = requests.post('https://api.example.com/auth', {
              'client_id': os.getenv('API_CLIENT_ID'),
              'client_secret': os.getenv('API_CLIENT_SECRET'),
              'grant_type': 'client_credentials'
          })
          auth_response.raise_for_status()
          
          token_data = auth_response.json()
          cached_token = token_data['access_token']
          token_expiry = (datetime.now() + timedelta(seconds=token_data['expires_in'] - 60)).isoformat()
          
          context.set_task_cache('api_token', cached_token)
          context.set_task_cache('token_expiry', token_expiry)
      
      # Make authenticated request
      headers = {'Authorization': f'Bearer {cached_token}'}
      response = requests.get('https://api.example.com/data', headers=headers)
      response.raise_for_status()
      
      return response.json()
```

## Performance Optimization

### Caching Strategy

```yaml
name: "Optimized Pipeline with Caching"

cache_config:
  enabled: true
  ttl: 3600  # 1 hour
  strategy: "content_hash"  # or "input_hash", "custom"

tasks:
  - name: "expensive_computation"
    type: "python_task"
    cache_key: "computation_{{ input_hash }}"
    cache_ttl: 7200  # 2 hours
    script: |
      import time
      import hashlib
      
      # Check if result is cached
      input_data = "{{ input_data }}"
      cache_key = f"expensive_result_{hashlib.md5(input_data.encode()).hexdigest()}"
      
      cached_result = context.get_cache(cache_key)
      if cached_result:
          return cached_result
      
      # Expensive computation
      time.sleep(10)  # Simulate expensive operation
      result = {"computed_value": len(input_data) * 42, "timestamp": time.time()}
      
      # Cache the result
      context.set_cache(cache_key, result, ttl=7200)
      
      return result
      
  - name: "cached_llm_call"
    type: "llm_task"
    model: "gpt-4"
    prompt: "{{ expensive_computation }}"
    cache_strategy: "prompt_hash"
    cache_ttl: 86400  # 24 hours
```

### Resource Management

```yaml
name: "Resource-Aware Pipeline"

resource_limits:
  memory_limit: "2GB"
  cpu_limit: 2
  timeout: 1800  # 30 minutes

tasks:
  - name: "memory_intensive_task"
    type: "python_task"
    resource_limits:
      memory_limit: "1GB"
      cpu_limit: 1
    script: |
      import gc
      import psutil
      import os
      
      def monitor_memory():
          process = psutil.Process(os.getpid())
          return process.memory_info().rss / 1024 / 1024  # MB
          
      # Monitor memory usage
      initial_memory = monitor_memory()
      
      # Your memory-intensive operation here
      large_data = process_large_dataset("{{ input_file }}")
      
      # Clean up
      gc.collect()
      final_memory = monitor_memory()
      
      return {
          "result": large_data.summary(),
          "memory_used": final_memory - initial_memory,
          "peak_memory": final_memory
      }
```

## Security Patterns

### Secure Credential Management

```yaml
name: "Secure Pipeline"

security:
  encrypt_variables: true
  sensitive_variables: ["api_key", "database_password"]

tasks:
  - name: "secure_operation"
    type: "python_task"
    script: |
      import os
      from cryptography.fernet import Fernet
      
      # Use environment variables for secrets
      api_key = os.getenv('SECURE_API_KEY')
      if not api_key:
          raise ValueError("SECURE_API_KEY environment variable not set")
      
      # Use the orchestrator's built-in secret management
      encrypted_value = context.get_secret('database_password')
      
      # Perform secure operation
      result = secure_api_call(api_key, encrypted_value)
      
      # Never log sensitive data
      return {"status": "success", "result_hash": hash(str(result))}
      
  - name: "audit_log"
    type: "python_task"
    script: |
      from datetime import datetime
      import hashlib
      
      # Log pipeline execution for audit purposes
      audit_entry = {
          "pipeline_id": context.get_pipeline_id(),
          "execution_id": context.get_execution_id(),
          "timestamp": datetime.now().isoformat(),
          "user_id": context.get_user_id(),
          "input_hash": hashlib.sha256(str("{{ input_data }}").encode()).hexdigest()
      }
      
      context.log_audit_event(audit_entry)
      return {"audit_logged": True}
```

### Input Validation and Sanitization

```yaml
name: "Input Validation Pipeline"

input_validation:
  strict_mode: true
  sanitization: true

tasks:
  - name: "validate_inputs"
    type: "python_task"
    script: |
      import re
      from html import escape
      
      user_input = "{{ user_input }}"
      
      # Validate input format
      if not re.match(r'^[a-zA-Z0-9\s\-_\.]+$', user_input):
          raise ValueError("Input contains invalid characters")
          
      if len(user_input) > 1000:
          raise ValueError("Input exceeds maximum length")
      
      # Sanitize for safe processing
      sanitized_input = escape(user_input.strip())
      
      return {"sanitized_input": sanitized_input, "validation_passed": True}
      
  - name: "secure_processing"
    type: "llm_task"
    model: "gpt-4"
    prompt: |
      Process this sanitized input safely:
      {{ validate_inputs.sanitized_input }}
      
      Important: Treat this as user-generated content.
```

## Best Practices Summary

### Pattern Selection Guidelines

1. **Control Flow**: Use conditionals for branching logic, loops for batch processing
2. **Error Handling**: Implement circuit breakers for external services, fallbacks for critical paths
3. **State Management**: Use persistent state for long-running processes, shared state for coordination
4. **Performance**: Cache expensive operations, limit resource usage, use parallel execution for independent tasks
5. **Security**: Validate all inputs, encrypt sensitive data, audit critical operations

### Common Anti-Patterns to Avoid

- **Excessive nesting**: Keep pipeline structure flat and readable
- **Tight coupling**: Make tasks independent and reusable
- **Resource leaks**: Always clean up connections and temporary resources
- **Poor error handling**: Don't let errors propagate without context
- **Security by obscurity**: Use proper encryption and validation

These advanced patterns provide the foundation for building robust, scalable, and maintainable Orchestrator pipelines. Combine them thoughtfully based on your specific requirements and constraints.