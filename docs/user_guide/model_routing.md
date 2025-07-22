# Intelligent Model Routing

Orchestrator provides sophisticated model routing capabilities that automatically select the best model for each task based on requirements, capabilities, and resource constraints.

## Overview

The intelligent model routing system:
- Automatically selects optimal models based on task requirements
- Balances performance, cost, and resource usage
- Supports fallback to alternative models when needed
- Learns from execution history to improve selections
- Handles both cloud APIs and local models seamlessly

## How Model Routing Works

### 1. Automatic Model Selection

When a task requires AI capabilities, Orchestrator automatically selects the best model:

```yaml
steps:
  - id: analyze_document
    tool: llm-analyze
    action: analyze
    parameters:
      content: "{{ document }}"
      # No model specified - Orchestrator chooses automatically
```

### 2. Task-Based Requirements

The router considers task type when selecting models:

```yaml
steps:
  # Code generation - selects a model optimized for coding
  - id: generate_code
    tool: llm-generate
    action: generate
    parameters:
      prompt: "Write a Python function to sort a list"
      
  # Analysis - selects a model with strong reasoning
  - id: analyze_data
    tool: llm-analyze
    action: analyze
    parameters:
      content: "{{ complex_data }}"
      analysis_type: "statistical"
      
  # Creative writing - selects a creative model
  - id: write_story
    tool: llm-generate
    action: generate
    parameters:
      prompt: "Write a short story about AI"
      temperature: 0.8
```

### 3. Explicit Model Requirements

You can specify requirements for model selection:

```yaml
steps:
  - id: complex_analysis
    tool: llm-analyze
    action: analyze
    parameters:
      content: "{{ large_document }}"
    requires_model:
      min_context_window: 100000  # Needs 100k token context
      expertise: ["reasoning", "analysis"]
      capabilities: ["structured_output"]
```

### 4. Resource-Aware Selection

The router considers available resources:

```yaml
steps:
  - id: local_inference
    tool: llm-generate
    action: generate
    parameters:
      prompt: "{{ prompt }}"
    requires_model:
      prefer_local: true  # Prefer local models
      max_memory_gb: 8   # Memory constraint
      requires_gpu: false # CPU-only execution
```

## Model Selection Algorithm

### Selection Criteria

1. **Task Compatibility** - Model must support the required task type
2. **Context Window** - Model must handle the input size
3. **Capabilities** - Model must have required features
4. **Resource Availability** - Model must fit within resource limits
5. **Performance History** - Historical success rates are considered
6. **Cost Optimization** - Balances quality with API costs

### Upper Confidence Bound (UCB) Algorithm

Orchestrator uses UCB for intelligent model selection:

```python
# Simplified selection logic
score = success_rate + sqrt(2 * log(total_uses) / model_uses)
```

This balances:
- **Exploitation** - Using models with proven success
- **Exploration** - Trying potentially better models

## Model Routing Examples

### Example 1: Multi-Model Pipeline

```yaml
name: multi-model-analysis
description: Different models for different tasks

steps:
  # Fast, small model for simple extraction
  - id: extract_summary
    tool: llm-generate
    action: generate
    parameters:
      prompt: "Extract key points from: {{ text }}"
    requires_model:
      speed: "fast"
      
  # Powerful model for complex reasoning
  - id: deep_analysis
    tool: llm-analyze
    action: analyze
    parameters:
      content: "{{ extract_summary.result }}"
      analysis_type: "reasoning"
    requires_model:
      capabilities: ["advanced_reasoning"]
      
  # Specialized model for code
  - id: generate_implementation
    tool: llm-generate
    action: generate
    parameters:
      prompt: "Implement solution based on: {{ deep_analysis.result }}"
    requires_model:
      expertise: ["coding"]
```

### Example 2: Fallback Strategy

```yaml
name: resilient-processing
description: Fallback to alternative models

steps:
  - id: primary_analysis
    tool: llm-analyze
    action: analyze
    parameters:
      content: "{{ data }}"
    requires_model:
      # Primary requirements
      capabilities: ["structured_output", "json_mode"]
      preference: "cloud"  # Prefer cloud APIs
    on_failure: continue  # Continue on failure
    
  - id: fallback_analysis
    tool: llm-analyze
    action: analyze
    condition: "{{ primary_analysis.status == 'failed' }}"
    parameters:
      content: "{{ data }}"
    requires_model:
      # Relaxed requirements for fallback
      capabilities: ["basic_analysis"]
      preference: "local"  # Fallback to local
```

### Example 3: Cost-Optimized Pipeline

```yaml
name: cost-optimized
description: Minimize API costs

steps:
  # Use cheap model for filtering
  - id: filter_relevant
    tool: llm-analyze
    action: classify
    parameters:
      content: "{{ documents }}"
      categories: ["relevant", "irrelevant"]
    requires_model:
      cost_tier: "low"  # Cheapest tier
      
  # Only process relevant with expensive model
  - id: process_relevant
    tool: llm-analyze
    action: analyze
    parameters:
      content: "{{ filter_relevant.relevant_items }}"
    requires_model:
      cost_tier: "high"  # Best quality for important items
      capabilities: ["advanced_reasoning"]
```

## Using the LLM Router Tool

The router tool provides direct control over model selection:

```yaml
steps:
  - id: smart_routing
    tool: llm-router
    action: route
    parameters:
      task: "complex_reasoning"
      requirements:
        accuracy: "high"
        speed: "medium"
        cost: "optimized"
      prompt: "{{ complex_prompt }}"
      
  # View selected model
  - id: show_selection
    tool: report-generator
    action: generate
    parameters:
      content: |
        Selected model: {{ smart_routing.model_used }}
        Confidence: {{ smart_routing.selection_confidence }}
        Alternatives: {{ smart_routing.alternatives }}
```

## Model Registry Configuration

### Registering Models

Models are registered with capabilities and requirements:

```python
# In model configuration
{
    "model": "gpt-4-turbo",
    "capabilities": ["reasoning", "coding", "analysis"],
    "context_window": 128000,
    "cost_per_1k_tokens": 0.01,
    "average_latency_ms": 500,
    "resource_requirements": {
        "memory_gb": 0,  # Cloud API
        "requires_gpu": False
    }
}
```

### Local Model Configuration

```python
{
    "model": "llama-3.1-8b",
    "capabilities": ["general", "reasoning"],
    "context_window": 8192,
    "cost_per_1k_tokens": 0,  # Free local inference
    "average_latency_ms": 100,
    "resource_requirements": {
        "memory_gb": 16,
        "requires_gpu": True,
        "gpu_memory_gb": 10
    }
}
```

## Advanced Routing Features

### 1. Dynamic Context Splitting

For large inputs exceeding model limits:

```yaml
steps:
  - id: process_large_doc
    tool: llm-analyze
    action: analyze
    parameters:
      content: "{{ huge_document }}"
      # Router automatically splits if needed
      allow_splitting: true
      merge_strategy: "hierarchical"
```

### 2. Ensemble Routing

Use multiple models for consensus:

```yaml
steps:
  - id: ensemble_analysis
    tool: llm-router
    action: ensemble
    parameters:
      prompt: "{{ critical_question }}"
      ensemble_size: 3  # Use 3 different models
      consensus_threshold: 0.7  # 70% agreement required
```

### 3. Adaptive Learning

The router learns from execution results:

```yaml
steps:
  - id: adaptive_task
    tool: llm-generate
    action: generate
    parameters:
      prompt: "{{ task_prompt }}"
    routing_hints:
      learn_from_result: true  # Update model scores
      success_metric: "user_satisfaction"
```

## Monitoring Model Usage

### View Model Performance

```yaml
steps:
  - id: get_metrics
    tool: llm-router
    action: get_metrics
    parameters:
      time_range: "last_24_hours"
      
  - id: show_report
    tool: report-generator
    action: generate
    parameters:
      content: |
        # Model Performance Report
        
        ## Usage by Model
        {% for model, stats in get_metrics.result.items() %}
        ### {{ model }}
        - Total requests: {{ stats.total_requests }}
        - Success rate: {{ stats.success_rate }}%
        - Average latency: {{ stats.avg_latency_ms }}ms
        - Total cost: ${{ stats.total_cost }}
        {% endfor %}
```

## Best Practices

### 1. Let the Router Decide

```yaml
# Good - let router optimize
- id: analyze
  tool: llm-analyze
  action: analyze
  parameters:
    content: "{{ data }}"
    
# Avoid - hardcoding model unless necessary
- id: analyze
  tool: llm-analyze
  action: analyze
  parameters:
    content: "{{ data }}"
    model: "gpt-4"  # Only if specifically needed
```

### 2. Specify Requirements, Not Models

```yaml
# Good - specify what you need
requires_model:
  capabilities: ["coding", "debugging"]
  min_context_window: 32000
  
# Avoid - over-constraining
requires_model:
  model: "specific-model-v2"  # Too specific
```

### 3. Use Appropriate Fallback Strategies

```yaml
steps:
  - id: primary
    tool: llm-analyze
    action: analyze
    parameters:
      content: "{{ data }}"
    on_failure: continue
    
  - id: fallback
    condition: "{{ primary.status == 'failed' }}"
    tool: llm-analyze
    action: analyze
    parameters:
      content: "{{ data }}"
    requires_model:
      relaxed_requirements: true
```

### 4. Monitor and Optimize

```yaml
# Periodically review model performance
- id: weekly_review
  tool: llm-router
  action: get_metrics
  parameters:
    time_range: "last_week"
    include_recommendations: true
```

## Troubleshooting

### Model Not Selected

```yaml
# Debug model selection
- id: debug_routing
  tool: llm-router
  action: explain_selection
  parameters:
    task: "{{ failed_task }}"
    requirements: "{{ task_requirements }}"
```

### High Costs

```yaml
# Analyze cost drivers
- id: cost_analysis
  tool: llm-router
  action: analyze_costs
  parameters:
    time_range: "last_month"
    group_by: ["model", "task_type"]
```

### Performance Issues

```yaml
# Identify bottlenecks
- id: performance_check
  tool: llm-router
  action: get_metrics
  parameters:
    metrics: ["latency", "throughput"]
    identify_outliers: true
```

## Future Enhancements

The model routing system continues to evolve with:

1. **Predictive Routing** - Pre-select models based on task patterns
2. **Multi-Region Support** - Route to nearest API endpoints
3. **Custom Scoring** - Define custom model selection criteria
4. **Model Composition** - Combine multiple models for complex tasks
5. **Budget Controls** - Hard limits on API spending

## Summary

Intelligent model routing in Orchestrator:
- Automatically selects optimal models for each task
- Balances performance, cost, and resource constraints
- Learns from execution history
- Provides fallback and ensemble strategies
- Supports both cloud and local models

By leveraging intelligent routing, your pipelines automatically adapt to use the best available models while optimizing for your specific requirements.