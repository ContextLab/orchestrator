# Best Practices for Orchestrator

This guide provides comprehensive recommendations for building maintainable, performant, and secure Orchestrator pipelines. Follow these practices to create production-ready systems that scale reliably.

## Table of Contents

1. [Pipeline Design Principles](#pipeline-design-principles)
2. [Performance Optimization](#performance-optimization)
3. [Security & Privacy](#security--privacy)
4. [Error Handling & Reliability](#error-handling--reliability)
5. [Maintainability & Documentation](#maintainability--documentation)
6. [Testing & Validation](#testing--validation)
7. [Deployment & Operations](#deployment--operations)
8. [Cost Optimization](#cost-optimization)

## Pipeline Design Principles

### Keep Pipelines Focused and Modular

**✅ Good Practice:**
```yaml
# focused_data_processor.yaml
name: "Customer Data Processor"
description: "Processes customer data records with validation and enrichment"

tasks:
  - name: "validate_data"
    type: "python_task"
    script: "validate_customer_data({{ input_data }})"
    
  - name: "enrich_data"
    type: "llm_task"
    prompt: "Enrich customer profile: {{ validate_data }}"
    
  - name: "store_result"
    type: "python_task" 
    script: "store_processed_data({{ enrich_data }})"
```

**❌ Anti-pattern:**
```yaml
# monolithic_processor.yaml - Too many responsibilities
name: "Everything Processor"
description: "Processes data, sends emails, generates reports, updates inventory..."

tasks:
  - name: "do_everything"
    type: "python_task"
    script: |
      # 500+ lines of code doing multiple unrelated things
      data = process_customer_data()
      send_marketing_emails()
      update_inventory_counts()
      generate_monthly_reports()
      backup_database()
      # ... and more
```

### Use Meaningful Names and Clear Structure

**✅ Good Practice:**
```yaml
name: "Research Article Analysis Pipeline"
version: "2.1.0"
description: "Analyzes research articles for key insights and generates summaries"

input_variables:
  article_url:
    type: string
    description: "URL of the research article to analyze"
    validation: "^https?://.*\\.(pdf|html)$"
  
  analysis_depth:
    type: string
    options: ["shallow", "detailed", "comprehensive"]
    default: "detailed"
    description: "Level of analysis detail required"

tasks:
  - name: "fetch_article_content"
    description: "Download and extract text from the research article"
    type: "web_scraping_task"
    
  - name: "analyze_methodology"
    description: "Identify and analyze the research methodology used"
    type: "llm_task"
    depends_on: ["fetch_article_content"]
    
  - name: "extract_key_findings"
    description: "Extract the main findings and conclusions"
    type: "llm_task"
    depends_on: ["fetch_article_content"]
    
  - name: "generate_executive_summary"
    description: "Create a concise executive summary of the analysis"
    type: "llm_task"
    depends_on: ["analyze_methodology", "extract_key_findings"]
```

### Design for Reusability

**✅ Create reusable components:**
```yaml
# components/data_validation.yaml
name: "Data Validation Component"
type: "component"

parameters:
  data_schema:
    type: object
    required: true
  strict_mode:
    type: boolean
    default: false

tasks:
  - name: "validate_schema"
    type: "python_task"
    script: |
      from jsonschema import validate, ValidationError
      
      try:
          validate(instance={{ input_data }}, schema={{ data_schema }})
          return {"valid": True, "errors": []}
      except ValidationError as e:
          if {{ strict_mode }}:
              raise e
          return {"valid": False, "errors": [str(e)]}
```

**✅ Use the component:**
```yaml
# main_pipeline.yaml
tasks:
  - name: "validate_customer_data"
    type: "component"
    component: "components/data_validation.yaml"
    parameters:
      data_schema: "{{ customer_data_schema }}"
      strict_mode: true
    inputs:
      input_data: "{{ raw_customer_data }}"
```

## Performance Optimization

### Implement Intelligent Caching

**✅ Cache expensive operations:**
```yaml
name: "Optimized Analysis Pipeline"

cache_strategy:
  default_ttl: 3600  # 1 hour
  compression: true
  storage: "redis"   # or "memory", "disk"

tasks:
  - name: "expensive_data_processing"
    type: "python_task"
    cache_config:
      enabled: true
      key_template: "data_processing_{{ input_hash }}"
      ttl: 7200  # 2 hours - longer for expensive operations
      invalidation_tags: ["data_version", "schema_version"]
    script: |
      # Expensive computation that benefits from caching
      result = perform_heavy_data_analysis("{{ input_data }}")
      return result
      
  - name: "llm_analysis"
    type: "llm_task"
    model: "gpt-4"
    cache_config:
      enabled: true
      key_template: "llm_analysis_{{ prompt_hash }}"
      ttl: 86400  # 24 hours - LLM responses are stable
    prompt: "Analyze this data: {{ expensive_data_processing }}"
```

### Use Parallel Processing Wisely

**✅ Effective parallelization:**
```yaml
name: "Parallel Processing Pipeline"

tasks:
  - name: "prepare_data_chunks"
    type: "python_task"
    script: |
      # Split data into optimal chunks
      chunk_size = min(100, len(input_data) // 4)  # 4-way parallelism
      chunks = [input_data[i:i+chunk_size] for i in range(0, len(input_data), chunk_size)]
      return {"chunks": chunks, "total_items": len(input_data)}
      
  - name: "process_in_parallel"
    type: "parallel_group"
    max_concurrency: 4  # Match your system resources
    timeout: 300        # 5 minutes per chunk
    tasks:
      - name: "process_chunk_{{ item.index }}"
        type: "llm_task"
        for_each: "{{ prepare_data_chunks.chunks }}"
        model: "gpt-3.5-turbo"  # Faster model for parallel tasks
        prompt: "Process this chunk: {{ item.value }}"
        
  - name: "combine_results"
    type: "python_task"
    script: |
      # Efficiently combine parallel results
      results = []
      for i, chunk in enumerate(context.get_task_output('prepare_data_chunks')['chunks']):
          chunk_result = context.get_task_output(f'process_chunk_{i}')
          results.extend(chunk_result)
      return {"combined_results": results, "total_processed": len(results)}
```

### Optimize Model Selection

**✅ Use appropriate models for tasks:**
```yaml
name: "Model Selection Best Practices"

model_strategy:
  default_model: "gpt-3.5-turbo"
  fallback_model: "claude-3-haiku"
  cost_optimization: true

tasks:
  - name: "simple_classification" 
    type: "llm_task"
    model: "gpt-3.5-turbo"  # Fast, cheap for simple tasks
    prompt: "Classify this text as positive/negative: {{ text }}"
    max_tokens: 10
    
  - name: "complex_analysis"
    type: "llm_task"
    model: "gpt-4"  # More capable for complex reasoning
    prompt: "Provide detailed analysis of: {{ complex_document }}"
    max_tokens: 2000
    
  - name: "creative_writing"
    type: "llm_task"
    model: "claude-3-opus"  # Best for creative tasks
    prompt: "Write a creative story about: {{ theme }}"
    temperature: 0.8
```

## Security & Privacy

### Secure Credential Management

**✅ Proper secret handling:**
```yaml
name: "Secure Pipeline"

security:
  encryption_at_rest: true
  encryption_in_transit: true
  audit_logging: true

tasks:
  - name: "secure_data_access"
    type: "python_task"
    script: |
      import os
      from cryptography.fernet import Fernet
      
      # ✅ Get secrets from environment or secret manager
      api_key = os.getenv('SECURE_API_KEY')
      if not api_key:
          raise ValueError("Missing required API key")
      
      # ✅ Use built-in secret management
      db_password = context.get_secret('database_password')
      
      # ✅ Never log sensitive data
      logger.info("Connecting to database with masked credentials")
      
      # Perform secure operations
      result = secure_database_operation(api_key, db_password)
      
      # ✅ Return sanitized results
      return {
          "status": "success", 
          "records_processed": len(result),
          "connection_hash": hash(api_key)  # Safe to log
      }
```

**❌ Security anti-patterns:**
```yaml
# DON'T DO THIS
tasks:
  - name: "insecure_task"
    type: "python_task"
    script: |
      # ❌ Hard-coded credentials
      api_key = "sk-1234567890abcdef"
      password = "admin123"
      
      # ❌ Logging sensitive data
      print(f"Using API key: {api_key}")
      
      # ❌ Storing credentials in variables
      return {"api_key": api_key, "result": data}
```

### Input Validation and Sanitization

**✅ Comprehensive input validation:**
```yaml
name: "Input Validation Pipeline"

input_validation:
  enabled: true
  strict_mode: true
  max_input_size: "10MB"

input_variables:
  user_content:
    type: string
    validation:
      max_length: 10000
      regex: "^[\\w\\s\\.,!?-]*$"  # Allow only safe characters
      sanitization: true
    description: "User-provided content to process"
    
  file_upload:
    type: file
    validation:
      allowed_extensions: [".txt", ".pdf", ".docx"]
      max_size: "5MB"
      virus_scan: true
    description: "User-uploaded file"

tasks:
  - name: "validate_and_sanitize"
    type: "python_task"
    script: |
      import re
      from html import escape
      import bleach
      
      user_content = "{{ user_content }}"
      
      # ✅ Validate input format
      if not re.match(r'^[\w\s\.,!?-]*$', user_content):
          raise ValueError("Input contains prohibited characters")
      
      # ✅ Sanitize for safe processing  
      sanitized_content = bleach.clean(
          user_content,
          tags=['p', 'br', 'em', 'strong'],  # Allow only safe HTML tags
          strip=True
      )
      
      # ✅ Additional length check
      if len(sanitized_content) > 10000:
          raise ValueError("Input exceeds maximum length")
          
      return {"sanitized_content": sanitized_content}
```

### Data Privacy Compliance

**✅ Privacy-aware processing:**
```yaml
name: "Privacy Compliant Pipeline"

privacy_settings:
  data_residency: "EU"
  retention_period: "90 days"
  anonymization: true
  audit_trail: true

tasks:
  - name: "anonymize_personal_data"
    type: "python_task"
    script: |
      import hashlib
      import re
      
      data = "{{ input_data }}"
      
      # ✅ Remove or hash PII
      # Email addresses
      data = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 
                    '[EMAIL_REDACTED]', data)
      
      # Phone numbers
      data = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE_REDACTED]', data)
      
      # Social Security Numbers
      data = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN_REDACTED]', data)
      
      return {"anonymized_data": data}
      
  - name: "audit_data_processing"
    type: "python_task" 
    script: |
      from datetime import datetime
      
      audit_entry = {
          "timestamp": datetime.now().isoformat(),
          "pipeline_id": context.get_pipeline_id(),
          "data_types_processed": ["text", "user_input"],
          "anonymization_applied": True,
          "user_consent": context.get_variable('user_consent', False)
      }
      
      context.log_audit_event(audit_entry)
      return {"audit_logged": True}
```

## Error Handling & Reliability

### Implement Circuit Breakers

**✅ Robust external service integration:**
```yaml
name: "Reliable External Service Integration"

tasks:
  - name: "external_service_with_circuit_breaker"
    type: "python_task"
    error_handling:
      circuit_breaker:
        failure_threshold: 5
        recovery_timeout: 300  # 5 minutes
        half_open_max_calls: 3
    script: |
      import requests
      from datetime import datetime, timedelta
      
      # Check circuit breaker state
      circuit_state = context.get_circuit_breaker_state('external_api')
      
      if circuit_state == 'open':
          # Use cached data or alternative service
          return context.get_cache('last_successful_response')
      
      try:
          response = requests.get('https://api.example.com/data', timeout=30)
          response.raise_for_status()
          
          result = response.json()
          context.set_cache('last_successful_response', result, ttl=3600)
          context.record_circuit_breaker_success('external_api')
          
          return result
          
      except Exception as e:
          context.record_circuit_breaker_failure('external_api')
          
          # Return fallback data
          return {
              "status": "degraded",
              "data": context.get_cache('last_successful_response'),
              "error": "External service unavailable"
          }
```

### Graceful Degradation

**✅ Multi-tier fallback strategy:**
```yaml
name: "Graceful Degradation Pipeline"

tasks:
  - name: "primary_processing"
    type: "llm_task"
    model: "gpt-4"
    prompt: "Provide detailed analysis: {{ input_text }}"
    error_handling:
      retry_count: 2
      fallback:
        - name: "secondary_processing"
          type: "llm_task"
          model: "gpt-3.5-turbo"
          prompt: "Provide basic analysis: {{ input_text }}"
        - name: "tertiary_processing"
          type: "python_task"
          script: |
            # Simple rule-based fallback
            text = "{{ input_text }}"
            word_count = len(text.split())
            char_count = len(text)
            
            return {
              "analysis": f"Basic analysis: {word_count} words, {char_count} characters",
              "sentiment": "neutral",  # Default when AI unavailable
              "fallback_used": True
            }
        - name: "emergency_response"
          type: "python_task"
          script: |
            return {
              "analysis": "Analysis unavailable - service degraded",
              "error": True,
              "fallback_used": True
            }
```

### Comprehensive Monitoring

**✅ Observable pipeline:**
```yaml
name: "Well-Monitored Pipeline"

monitoring:
  metrics_enabled: true
  tracing_enabled: true
  health_checks: true
  alerting:
    error_threshold: 0.05  # 5% error rate
    latency_threshold: 300 # 5 minutes
    notification_channels: ["email", "slack"]

tasks:
  - name: "monitored_task"
    type: "llm_task"
    monitoring:
      custom_metrics:
        - name: "processing_quality_score"
          type: "gauge"
        - name: "tokens_used"
          type: "counter"
    script: |
      import time
      from datetime import datetime
      
      start_time = time.time()
      
      try:
          result = process_with_llm("{{ input_data }}")
          
          # ✅ Record success metrics
          processing_time = time.time() - start_time
          context.record_metric("processing_latency", processing_time)
          context.record_metric("tokens_used", result.get('tokens', 0))
          
          # ✅ Health check
          if processing_time > 60:  # More than 1 minute
              context.log_warning("Processing time exceeded expected threshold")
              
          return result
          
      except Exception as e:
          # ✅ Record error metrics
          context.record_metric("error_count", 1)
          context.log_error(f"Task failed: {str(e)}")
          
          # ✅ Send alert if needed
          if context.get_error_rate() > 0.1:  # 10% error rate
              context.send_alert("High error rate detected")
              
          raise e
```

## Maintainability & Documentation

### Self-Documenting Pipelines

**✅ Clear documentation:**
```yaml
name: "Content Analysis Pipeline"
version: "3.2.1"
description: |
  Analyzes content for sentiment, key topics, and quality metrics.
  Used by marketing team for content optimization and A/B testing.
  
  Dependencies:
  - OpenAI API access
  - Redis for caching
  - PostgreSQL for result storage
  
  Expected processing time: 2-5 minutes per item
  Cost estimate: $0.10-$0.50 per analysis

author: "Data Science Team"
contact: "data-science@company.com"
documentation_url: "https://wiki.company.com/content-analysis-pipeline"

changelog:
  - version: "3.2.1"
    date: "2024-01-15"
    changes: "Added sentiment confidence scoring"
  - version: "3.2.0" 
    date: "2024-01-10"
    changes: "Integrated topic modeling with caching"

input_variables:
  content_text:
    type: string
    description: "The text content to analyze (blog post, article, etc.)"
    validation:
      max_length: 50000
      min_length: 100
    example: "This is a sample blog post about artificial intelligence..."
    
  analysis_features:
    type: array
    description: "List of analysis features to enable"
    options: ["sentiment", "topics", "quality", "readability"]
    default: ["sentiment", "topics"]
    example: ["sentiment", "topics", "quality"]

tasks:
  - name: "preprocess_content"
    description: |
      Clean and normalize the input content:
      - Remove HTML tags
      - Fix encoding issues  
      - Split into sentences
      - Basic tokenization
    type: "python_task"
    estimated_time: "10 seconds"
    script: |
      # Implementation with clear comments
      import re
      from html import unescape
      
      content = "{{ content_text }}"
      
      # Remove HTML tags
      clean_content = re.sub(r'<[^>]+>', '', content)
      
      # Fix HTML entities
      clean_content = unescape(clean_content)
      
      # Normalize whitespace
      clean_content = re.sub(r'\s+', ' ', clean_content).strip()
      
      return {
          "cleaned_content": clean_content,
          "original_length": len(content),
          "cleaned_length": len(clean_content)
      }
```

### Version Control Best Practices

**✅ Organized pipeline structure:**
```
pipelines/
├── production/
│   ├── content-analysis-v3.2.1.yaml    # Current production
│   └── user-onboarding-v1.5.0.yaml
├── staging/
│   ├── content-analysis-v3.3.0.yaml    # Testing new features
│   └── experimental-workflow-v0.1.0.yaml
├── components/                           # Reusable components
│   ├── data-validation.yaml
│   ├── error-handling.yaml
│   └── monitoring.yaml
├── schemas/                             # Data schemas
│   ├── user-data-schema.json
│   └── content-schema.json
└── README.md                           # Pipeline inventory
```

### Configuration Management

**✅ Environment-specific configs:**
```yaml
# config/base.yaml
name: "Content Analysis Pipeline"
monitoring:
  metrics_enabled: true
  tracing_enabled: true

# config/development.yaml
extends: "base.yaml"
model_settings:
  development_mode: true
  cost_optimization: false
cache:
  enabled: false  # Disable caching in development
logging:
  level: "DEBUG"

# config/production.yaml  
extends: "base.yaml"
model_settings:
  development_mode: false
  cost_optimization: true
cache:
  enabled: true
  ttl: 3600
logging:
  level: "INFO"
alerting:
  enabled: true
  channels: ["pagerduty", "slack"]
```

## Testing & Validation

### Comprehensive Test Coverage

**✅ Multi-layer testing:**
```yaml
# tests/unit/test_content_analysis.yaml
name: "Content Analysis Unit Tests"
type: "test_suite"

tests:
  - name: "test_basic_sentiment_analysis"
    input:
      content_text: "I love this product! It's amazing and works perfectly."
      analysis_features: ["sentiment"]
    expected_output:
      sentiment_score: ">= 0.7"  # Positive sentiment
      sentiment_label: "positive"
    validation:
      response_time: "< 30 seconds"
      
  - name: "test_empty_content_handling"
    input:
      content_text: ""
    expected_error: "ValidationError"
    error_message: "Content cannot be empty"
    
  - name: "test_long_content_handling"
    input:
      content_text: "{{ load_test_data('long_article.txt') }}"
    validation:
      response_time: "< 300 seconds"
      memory_usage: "< 1GB"
```

**✅ Integration testing:**
```yaml
# tests/integration/test_pipeline_integration.yaml
name: "Pipeline Integration Tests"
type: "integration_test"

setup:
  - start_test_database
  - load_test_data
  - configure_test_apis

tests:
  - name: "test_end_to_end_processing"
    pipeline: "content-analysis-v3.2.1.yaml"
    inputs:
      content_text: "{{ test_articles.sample_blog_post }}"
      analysis_features: ["sentiment", "topics", "quality"]
    validate:
      - database_records_created: 3  # sentiment, topics, quality
      - cache_entries_created: ">= 1"
      - processing_time: "< 180 seconds"
      - api_calls_made: "< 10"
      
teardown:
  - cleanup_test_data
  - stop_test_database
```

### Performance Testing

**✅ Load testing:**
```yaml
# tests/performance/load_test.yaml
name: "Content Analysis Load Test"
type: "load_test"

test_scenarios:
  - name: "normal_load"
    concurrent_users: 10
    duration: "10 minutes"
    ramp_up_time: "2 minutes"
    target_throughput: "1 request/second"
    
  - name: "peak_load"
    concurrent_users: 50 
    duration: "5 minutes"
    target_throughput: "5 requests/second"
    
  - name: "stress_test"
    concurrent_users: 100
    duration: "2 minutes"
    acceptable_error_rate: "< 5%"

validation:
  response_time_p95: "< 60 seconds"
  error_rate: "< 1%"
  memory_usage: "< 2GB"
  cpu_usage: "< 80%"
```

## Deployment & Operations

### Blue-Green Deployment

**✅ Zero-downtime deployments:**
```yaml
# deployment/blue-green-config.yaml
name: "Content Analysis Deployment"
strategy: "blue_green"

environments:
  blue:
    pipeline_version: "3.2.1"
    traffic_percentage: 100
    health_check: "https://api.company.com/health/content-analysis"
    
  green:
    pipeline_version: "3.3.0"
    traffic_percentage: 0
    health_check: "https://api-staging.company.com/health/content-analysis"

deployment_process:
  - deploy_to_green
  - run_smoke_tests
  - gradual_traffic_shift:
      - 10%_for_5_minutes
      - 25%_for_5_minutes  
      - 50%_for_10_minutes
      - 100%_if_healthy
  - monitor_metrics:
      - error_rate
      - response_time
      - user_satisfaction
  - rollback_if_issues_detected
```

### Monitoring and Alerting

**✅ Comprehensive observability:**
```yaml
# monitoring/alerts.yaml
name: "Content Analysis Monitoring"

health_checks:
  - name: "pipeline_availability"
    endpoint: "/health/content-analysis"
    frequency: "30 seconds"
    timeout: "10 seconds"
    
  - name: "dependency_health"
    checks:
      - openai_api_status
      - redis_connectivity
      - database_connectivity
    frequency: "1 minute"

alerts:
  - name: "high_error_rate"
    condition: "error_rate > 5%"
    duration: "5 minutes"
    severity: "critical"
    channels: ["pagerduty", "slack"]
    
  - name: "slow_response_time"
    condition: "avg_response_time > 60 seconds"
    duration: "10 minutes"
    severity: "warning"
    channels: ["slack"]
    
  - name: "cost_spike"
    condition: "hourly_cost > $100"
    severity: "warning"
    channels: ["email", "slack"]

dashboards:
  - name: "operational_metrics"
    metrics:
      - request_volume
      - error_rate
      - response_time_percentiles
      - cost_per_hour
      
  - name: "business_metrics"
    metrics:
      - analyses_completed
      - user_satisfaction_score
      - feature_usage_breakdown
```

## Cost Optimization

### Model Selection and Usage

**✅ Cost-effective model strategy:**
```yaml
name: "Cost-Optimized Analysis Pipeline"

cost_management:
  budget_limit: "$1000/month"
  cost_tracking: true
  optimization_mode: "balanced"  # "performance", "cost", "balanced"

model_tier_strategy:
  tier_1_fast:
    models: ["gpt-3.5-turbo", "claude-3-haiku"]
    use_cases: ["simple_classification", "basic_summarization"]
    cost_per_1k_tokens: 0.001
    
  tier_2_balanced:
    models: ["gpt-4", "claude-3-sonnet"]
    use_cases: ["detailed_analysis", "content_generation"]
    cost_per_1k_tokens: 0.01
    
  tier_3_premium:
    models: ["gpt-4-turbo", "claude-3-opus"]
    use_cases: ["complex_reasoning", "creative_tasks"]
    cost_per_1k_tokens: 0.03

tasks:
  - name: "cost_aware_processing"
    type: "python_task"
    script: |
      content_length = len("{{ content_text }}")
      complexity_score = estimate_complexity("{{ content_text }}")
      
      # Choose model tier based on complexity and cost
      if complexity_score < 0.3 and content_length < 1000:
          selected_model = "gpt-3.5-turbo"
          max_tokens = 500
      elif complexity_score < 0.7 and content_length < 5000:
          selected_model = "gpt-4"
          max_tokens = 1000
      else:
          selected_model = "gpt-4-turbo"
          max_tokens = 2000
          
      return {
          "selected_model": selected_model,
          "max_tokens": max_tokens,
          "estimated_cost": calculate_estimated_cost(selected_model, max_tokens)
      }
      
  - name: "adaptive_analysis"
    type: "llm_task"
    model: "{{ cost_aware_processing.selected_model }}"
    max_tokens: "{{ cost_aware_processing.max_tokens }}"
    prompt: |
      {% if cost_aware_processing.selected_model == "gpt-3.5-turbo" %}
      Provide a concise analysis of: {{ content_text }}
      {% else %}
      Provide a detailed analysis of: {{ content_text }}
      {% endif %}
```

### Resource Optimization

**✅ Efficient resource usage:**
```yaml
name: "Resource-Optimized Pipeline"

resource_management:
  auto_scaling: true
  resource_limits:
    memory: "1GB"
    cpu: "1 core"
    timeout: "10 minutes"

optimization_strategies:
  - name: "batch_processing"
    enabled: true
    batch_size: 10
    max_wait_time: "30 seconds"
    
  - name: "intelligent_caching" 
    enabled: true
    cache_similar_inputs: true
    similarity_threshold: 0.9
    
  - name: "request_coalescing"
    enabled: true
    coalesce_window: "5 seconds"

tasks:
  - name: "batch_optimize"
    type: "python_task"
    script: |
      # Batch similar requests together
      batch = context.get_current_batch()
      if len(batch) >= 10:
          # Process full batch
          results = process_batch_efficiently(batch)
          return {"batch_processed": True, "count": len(batch)}
      else:
          # Wait for more items or timeout
          context.wait_for_batch_completion(timeout=30)
```

## Summary Checklist

### Design Phase
- [ ] Pipeline has single, clear responsibility
- [ ] Tasks are modular and reusable  
- [ ] Names and descriptions are meaningful
- [ ] Input validation is comprehensive
- [ ] Error handling strategy is defined

### Implementation Phase
- [ ] Caching strategy is implemented
- [ ] Parallel processing is used where appropriate
- [ ] Security best practices are followed
- [ ] Monitoring and metrics are configured
- [ ] Tests cover critical paths

### Deployment Phase
- [ ] Environment-specific configurations exist
- [ ] Health checks are implemented  
- [ ] Alerting is configured
- [ ] Cost monitoring is enabled
- [ ] Documentation is complete

### Operations Phase
- [ ] Performance metrics are tracked
- [ ] Error rates are monitored
- [ ] Costs are optimized
- [ ] Security audits are regular
- [ ] Pipeline versions are managed

By following these best practices, you'll build Orchestrator pipelines that are reliable, maintainable, secure, and cost-effective. Remember that best practices evolve with your needs - regularly review and update your approaches based on operational experience and changing requirements.