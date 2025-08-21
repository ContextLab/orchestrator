# LLM Task Routing and Optimization Report

## Task Analysis
- **Original Task**: I would like you to please write a function that can calculate the factorial of a number
- **Task Type**: general
- **Complexity**: simple

## Model Selection
- **Selected Model**: ollama:llama3.2:1b
- **Score**: 56.993
- **Reasons**: Good at general, Optimal small model for simple task, Compact 1.0B model, High success rate: 93%
- **Estimated Cost**: $0.0
- **Estimated Latency**: 0.8s

## Prompt Optimization
- **Original Length**: 22 tokens
- **Optimized Length**: 20 tokens
- **Reduction**: 9.1%
- **Applied Optimizations**: clarity, brevity

### Optimized Prompt
```
Task: please please write a function that can calculate the factorial of a number
```

## Routing Decision
- **Final Model**: ollama:llama3.2:3b
- **Strategy**: capability_based
- **Routing Reason**: Least loaded (current load: 0)
- **Current Load**: 1

## Recommendations
- Consider adding examples of desired output
- Specify desired output format explicitly

## Alternative Models
- **ollama:llama3.2:3b** (Score: 56.979)
  - Good at general, Optimal small model for simple task, Compact 3.0B model, High success rate: 94%
- **openai:gpt-5-nano** (Score: 56.926249999999996)
  - Good at general, Optimal small model for simple task, Compact 10.0B model, High success rate: 98%
- **openai:dall-e-3** (Score: 56.924)
  - Good at general, Optimal small model for simple task, Compact 10.0B model, High success rate: 100%
