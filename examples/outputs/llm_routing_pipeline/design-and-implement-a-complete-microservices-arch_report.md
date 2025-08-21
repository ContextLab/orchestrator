# LLM Task Routing and Optimization Report

## Task Analysis
- **Original Task**: Design and implement a complete microservices architecture with Kubernetes orchestration, service mesh, distributed tracing, and auto-scaling capabilities
- **Task Type**: code_generation
- **Complexity**: complex

## Model Selection
- **Selected Model**: openai:gpt-5
- **Score**: 49.8875
- **Reasons**: Good at code_generation, Large model for complex task, Large 2000B model, High success rate: 98%
- **Estimated Cost**: $0.075
- **Estimated Latency**: 1.0s

## Prompt Optimization
- **Original Length**: 38 tokens
- **Optimized Length**: 38 tokens
- **Reduction**: 0.0%
- **Applied Optimizations**: None

### Optimized Prompt
```
Design and implement a complete microservices architecture with Kubernetes orchestration, service mesh, distributed tracing, and auto-scaling capabilities
```

## Routing Decision
- **Final Model**: anthropic:claude-sonnet-4-20250514
- **Strategy**: capability_based
- **Routing Reason**: Least loaded (current load: 0)
- **Current Load**: 1

## Recommendations
- Specify desired output format explicitly

## Alternative Models
- **anthropic:claude-sonnet-4-20250514** (Score: 42.3)
  - General purpose model, Large model for complex task, Large 600B model, High success rate: 99%
- **google:gemini-2.5-pro** (Score: 39.5)
  - General purpose model, Large model for complex task, Large 1500B model, High success rate: 98%
- **openai:gpt-5-mini** (Score: 35.277499999999996)
  - Good at code_generation, May be too small for complex task, Moderate 100.0B model, High success rate: 98%
