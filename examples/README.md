# Orchestrator Framework Examples

This directory contains example pipeline definitions demonstrating various features of the Orchestrator Framework.

## Available Examples

### 1. Simple Pipeline (`simple_pipeline.yaml`)

A basic pipeline demonstrating:
- Text generation with AUTO tag resolution
- Parallel task execution
- Dependency management
- Context variable usage

**Use case**: Simple text processing workflow  
**Complexity**: Beginner  
**Models required**: Any text generation model  

### 2. Multi-Model Pipeline (`multi_model_pipeline.yaml`)

An advanced pipeline demonstrating:
- Multiple AI model integration
- Complex dependency chains
- Resource allocation
- Error handling strategies
- Multi-format output generation

**Use case**: Comprehensive data analysis workflow  
**Complexity**: Advanced  
**Models required**: Multiple models (GPT-4, Claude, Gemini)  

## Running Examples

### Prerequisites

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables**:
   ```bash
   export OPENAI_API_KEY="your-openai-key"
   export ANTHROPIC_API_KEY="your-anthropic-key"
   export GOOGLE_AI_API_KEY="your-google-key"
   ```

3. **Configure models** (optional):
   Edit `config/models.yaml` to customize model settings.

### Running from Python

```python
import asyncio
from orchestrator import Orchestrator

async def run_example():
    orchestrator = Orchestrator()
    
    # Run simple pipeline
    results = await orchestrator.execute_yaml_file(
        "examples/simple_pipeline.yaml",
        context={"input_topic": "machine learning"}
    )
    
    print("Pipeline results:", results)

asyncio.run(run_example())
```

### Running from Command Line

```bash
# Simple pipeline
python -m orchestrator run examples/simple_pipeline.yaml \
    --context input_topic="machine learning"

# Multi-model pipeline
python -m orchestrator run examples/multi_model_pipeline.yaml \
    --context dataset_url="https://example.com/sales_data.csv"
```

## AUTO Tag Examples

The examples demonstrate various AUTO tag patterns:

### Analysis Method Selection
```yaml
analysis_type: <AUTO>What type of analysis is most appropriate for this text?</AUTO>
```

### Format Determination
```yaml
format: <AUTO>Determine the best format for this data source</AUTO>
```

### Method Recommendation
```yaml
methods: <AUTO>Choose the most appropriate statistical methods</AUTO>
```

### Conditional Logic
```yaml
validation_rules: <AUTO>Generate appropriate validation rules for this dataset</AUTO>
```

## Creating Custom Pipelines

### Basic Structure

```yaml
id: my_pipeline
name: My Custom Pipeline
description: Description of what this pipeline does
version: "1.0"

context:
  # Global variables accessible to all tasks
  variable_name: value

steps:
  - id: task1
    name: First Task
    action: generate  # or analyze, transform, etc.
    parameters:
      # Task-specific parameters
      prompt: "Your prompt here"
    metadata:
      # Optional metadata
      requires_model: true
      priority: 1.0
```

### Advanced Features

```yaml
steps:
  - id: advanced_task
    name: Advanced Task
    action: analyze
    parameters:
      data: "{{ results.previous_task }}"  # Reference previous results
      method: <AUTO>Choose best method</AUTO>  # AUTO resolution
    dependencies:
      - previous_task  # Task dependencies
    metadata:
      requires_model: gpt-4  # Specific model requirement
      cpu_cores: 4  # Resource requirements
      memory_mb: 2048
      timeout: 300
      priority: 0.8
      on_failure: continue  # Error handling
```

## Best Practices

1. **Use descriptive IDs and names** for tasks
2. **Specify dependencies explicitly** to ensure correct execution order
3. **Set appropriate timeouts** for long-running tasks
4. **Use AUTO tags** for dynamic parameter resolution
5. **Include error handling** with `on_failure` policies
6. **Set resource requirements** for resource-intensive tasks
7. **Use context variables** for pipeline-wide configuration
8. **Test pipelines** with mock models before using real APIs

## Troubleshooting

### Common Issues

1. **Model not available**: Check API keys and model configuration
2. **Resource allocation failed**: Reduce resource requirements or increase limits
3. **AUTO tag resolution failed**: Ensure models are available for resolution
4. **Dependency cycles**: Check task dependencies for circular references
5. **Timeout errors**: Increase task timeouts for long-running operations

### Debug Mode

Enable debug logging to see detailed execution information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

orchestrator = Orchestrator()
```

### Health Checks

Verify system health before running pipelines:

```python
health = await orchestrator.health_check()
print("System health:", health["overall"])
```

## Contributing

To add new examples:

1. Create a new YAML file in this directory
2. Follow the naming convention: `feature_example.yaml`
3. Include comprehensive comments
4. Add documentation to this README
5. Test the example with different model configurations

For more information, see the [main documentation](../docs/index.rst).