# Declarative Framework Guide

## Overview

The Orchestrator's declarative framework allows you to define AI pipelines using simple YAML files. This approach separates the workflow logic from implementation details, making pipelines easier to create, understand, and maintain.

## Architecture

```
┌─────────────────────────┐
│    YAML Pipeline        │
│    Definition           │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│    YAML Compiler        │
│  (with AUTO tag parser) │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   Pipeline Object       │
│   (Tasks + Dependencies)│
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  ModelBasedControlSystem│
│  (Direct Execution)     │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│    AI Model Output      │
└─────────────────────────┘
```

## Quick Start

### 1. Create a YAML Pipeline

```yaml
# my_pipeline.yaml
name: "Data Analysis Pipeline"
description: "Analyze data and generate insights"

inputs:
  data_source:
    type: string
    description: "Path or description of data"
    required: true
    
  analysis_type:
    type: string
    description: "Type of analysis to perform"
    default: "comprehensive"

steps:
  - id: load_data
    action: |
      Load and prepare data from: {{data_source}}
      
      Extract:
      1. Data structure and format
      2. Key metrics and statistics
      3. Data quality indicators
      
      Return prepared dataset summary
      
  - id: analyze_data
    action: |
      Perform {{analysis_type}} analysis on the data:
      {{load_data.result}}
      
      Include:
      1. Statistical analysis
      2. Pattern identification
      3. Anomaly detection
      4. Trend analysis
      
      Return comprehensive analysis results
    depends_on: [load_data]
    
  - id: generate_insights
    action: |
      Generate actionable insights from analysis:
      {{analyze_data.result}}
      
      Provide:
      1. Key findings
      2. Recommendations
      3. Risk factors
      4. Next steps
      
      Format as executive summary
    depends_on: [analyze_data]

outputs:
  insights: "{{generate_insights.result}}"
  analysis: "{{analyze_data.result}}"
  data_summary: "{{load_data.result}}"
```

### 2. Execute the Pipeline

```python
import asyncio
from pathlib import Path
from orchestrator.compiler.yaml_compiler import YAMLCompiler
from orchestrator.control_systems.model_based_control_system import ModelBasedControlSystem
from orchestrator.models.model_registry import ModelRegistry
from orchestrator.integrations.openai_model import OpenAIModel

async def run_pipeline():
    # Setup
    model_registry = ModelRegistry()
    model = OpenAIModel(model_name="gpt-4o-mini")
    model_registry.register_model(model)
    
    control_system = ModelBasedControlSystem(model_registry)
    compiler = YAMLCompiler()
    
    # Load and compile YAML
    with open('my_pipeline.yaml', 'r') as f:
        yaml_content = f.read()
    
    inputs = {
        "data_source": "sales_data_2024.csv",
        "analysis_type": "quarterly trends"
    }
    
    pipeline = await compiler.compile(yaml_content, inputs)
    
    # Execute
    results = await control_system.execute_pipeline(pipeline)
    
    # Access results
    print("Insights:", results["generate_insights"])
    return results

# Run
asyncio.run(run_pipeline())
```

## Key Features

### 1. Sequential Execution

Tasks execute in dependency order automatically:

```yaml
steps:
  - id: step_a
    action: "First task"
    
  - id: step_b
    action: "Uses result from A: {{step_a.result}}"
    depends_on: [step_a]
    
  - id: step_c
    action: "Uses both: {{step_a.result}} and {{step_b.result}}"
    depends_on: [step_a, step_b]
```

### 2. Context Propagation

Each step automatically receives results from previous steps:

```yaml
steps:
  - id: fetch_data
    action: "Fetch user data for ID: {{user_id}}"
    
  - id: analyze_behavior
    action: |
      Analyze user behavior patterns:
      User Data: {{fetch_data.result}}
      
      Focus on engagement metrics
    depends_on: [fetch_data]
    
  - id: personalize_content
    action: |
      Create personalized recommendations:
      User Profile: {{fetch_data.result}}
      Behavior Analysis: {{analyze_behavior.result}}
      
      Generate top 5 recommendations
    depends_on: [fetch_data, analyze_behavior]
```

### 3. Template Resolution

Use Jinja2 templates for dynamic values:

```yaml
inputs:
  threshold:
    type: float
    default: 0.8

steps:
  - id: filter_data
    action: |
      Filter items with score > {{threshold}}
      {% if threshold > 0.9 %}
      Apply strict quality checks
      {% else %}
      Apply standard quality checks
      {% endif %}
```

### 4. Conditional Execution

Execute steps based on conditions:

```yaml
steps:
  - id: check_data_quality
    action: "Assess data quality and return score (0-1)"
    
  - id: clean_data
    action: "Clean and preprocess low-quality data"
    depends_on: [check_data_quality]
    condition: "{{check_data_quality.result.score}} < 0.7"
    
  - id: enhance_data
    action: "Enhance high-quality data"
    depends_on: [check_data_quality]
    condition: "{{check_data_quality.result.score}} >= 0.7"
```

### 5. Error Handling

Handle errors gracefully:

```yaml
steps:
  - id: external_api_call
    action: "Fetch data from external API"
    timeout: 30.0
    on_error:
      action: "Use cached data instead"
      continue_on_error: true
      fallback_value: {"source": "cache", "data": []}
```

## Working Examples

### Research Assistant

```yaml
name: "Research Assistant"
description: "Research and summarize information"

inputs:
  topic:
    type: string
    required: true
    
  depth:
    type: string
    default: "comprehensive"

steps:
  - id: research_plan
    action: |
      Create research plan for "{{topic}}":
      1. Identify key aspects
      2. Determine information sources
      3. Plan {{depth}} research approach
      
  - id: gather_info
    action: |
      Research "{{topic}}" following plan:
      {{research_plan.result}}
      
      Collect facts, trends, and insights
    depends_on: [research_plan]
    
  - id: synthesize
    action: |
      Synthesize research findings:
      {{gather_info.result}}
      
      Create comprehensive summary with:
      1. Key findings
      2. Analysis
      3. Recommendations
    depends_on: [gather_info]

outputs:
  summary: "{{synthesize.result}}"
  research_plan: "{{research_plan.result}}"
```

### Content Generator

```yaml
name: "Content Generator"
description: "Generate various content types"

inputs:
  topic:
    type: string
    required: true
    
  content_type:
    type: string
    default: "blog"
    
  tone:
    type: string
    default: "professional"

steps:
  - id: content_strategy
    action: |
      Develop content strategy for {{content_type}} about "{{topic}}":
      - Target audience analysis
      - Key messages
      - Content structure
      - {{tone}} tone guidelines
      
  - id: create_outline
    action: |
      Create detailed outline:
      Strategy: {{content_strategy.result}}
      
      Structure with sections and key points
    depends_on: [content_strategy]
    
  - id: write_content
    action: |
      Write {{content_type}} content:
      Topic: {{topic}}
      Outline: {{create_outline.result}}
      Tone: {{tone}}
      
      Create engaging, well-structured content
    depends_on: [create_outline]
    
  - id: polish_content
    action: |
      Polish and refine:
      {{write_content.result}}
      
      Improve clarity, flow, and impact
    depends_on: [write_content]

outputs:
  final_content: "{{polish_content.result}}"
  outline: "{{create_outline.result}}"
```

### Data Processing Workflow

```yaml
name: "Data Processing Workflow"
description: "Process and analyze data"

inputs:
  data_path:
    type: string
    required: true
    
  output_format:
    type: string
    default: "json"

steps:
  - id: load_data
    action: |
      Load data from {{data_path}}:
      - Identify format and structure
      - Extract metadata
      - Perform initial validation
      
  - id: clean_data
    action: |
      Clean and preprocess data:
      {{load_data.result}}
      
      - Handle missing values
      - Standardize formats
      - Remove duplicates
    depends_on: [load_data]
    
  - id: transform_data
    action: |
      Transform data to {{output_format}}:
      {{clean_data.result}}
      
      - Apply transformations
      - Validate output
      - Add metadata
    depends_on: [clean_data]
    
  - id: quality_check
    action: |
      Perform final quality check:
      {{transform_data.result}}
      
      Return validation report
    depends_on: [transform_data]

outputs:
  processed_data: "{{transform_data.result}}"
  quality_report: "{{quality_check.result}}"
```

## Best Practices

### 1. Clear Action Descriptions

Be specific about what each step should do:

```yaml
# Good
action: |
  Analyze customer feedback data:
  1. Identify sentiment (positive, negative, neutral)
  2. Extract key themes and topics
  3. Prioritize issues by frequency and impact
  4. Generate actionable recommendations
  
  Return structured analysis with scores

# Too vague
action: "Analyze feedback"
```

### 2. Proper Context Usage

Pass relevant context between steps:

```yaml
steps:
  - id: extract_requirements
    action: |
      Extract key requirements from: {{project_description}}
      
      Identify:
      - Functional requirements
      - Technical constraints
      - Performance goals
      
  - id: design_solution
    action: |
      Design solution based on requirements:
      {{extract_requirements.result}}
      
      Consider all constraints and goals
    depends_on: [extract_requirements]
```

### 3. Modular Steps

Keep steps focused and modular:

```yaml
# Good - separate concerns
steps:
  - id: validate_input
    action: "Validate and sanitize input data"
    
  - id: process_data
    action: "Process validated data"
    depends_on: [validate_input]
    
  - id: format_output
    action: "Format results for presentation"
    depends_on: [process_data]

# Bad - doing too much in one step
steps:
  - id: do_everything
    action: "Validate, process, and format data"
```

### 4. Error Recovery

Plan for potential failures:

```yaml
steps:
  - id: primary_source
    action: "Fetch data from primary source"
    timeout: 30.0
    on_error:
      action: "Log error and continue"
      continue_on_error: true
      
  - id: backup_source
    action: "Fetch from backup source"
    condition: "{{primary_source.error}} != null"
    
  - id: process_data
    action: |
      Process data from available source:
      Primary: {{primary_source.result}}
      Backup: {{backup_source.result}}
    depends_on: [primary_source, backup_source]
```

## Limitations

### Current Limitations

1. **No Loop Support**: Loops are not currently implemented
2. **Sequential Only**: No parallel execution within steps
3. **No Dynamic Steps**: All steps must be predefined
4. **No External Tools**: Direct file/API access not available

### Workarounds

For loops, use aggregated processing:

```yaml
# Instead of looping over items
steps:
  - id: process_all_items
    action: |
      Process all items in the list:
      {{items}}
      
      For each item:
      1. Validate format
      2. Transform data
      3. Generate summary
      
      Return results for all items
```

## Troubleshooting

### Common Issues

1. **Template Variable Not Found**
   - Check variable names match exactly
   - Ensure inputs are provided
   - Verify step IDs in references

2. **Circular Dependencies**
   - Review depends_on declarations
   - Ensure no cycles in dependency graph

3. **Step Not Executing**
   - Check condition expressions
   - Verify dependencies are satisfied
   - Look for errors in previous steps

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Execute pipeline with debug info
results = await control_system.execute_pipeline(pipeline)
```

## Performance Tips

### 1. Minimize Context Size

Keep context focused:

```yaml
# Good - pass only needed data
action: |
  Process user profile:
  Name: {{user.name}}
  Preferences: {{user.preferences}}

# Bad - passing entire object
action: |
  Process user profile:
  {{user}}
```

### 2. Cache Expensive Operations

Use cache_results for repeated operations:

```yaml
steps:
  - id: expensive_analysis
    action: "Perform complex analysis"
    cache_results: true
    cache_ttl: 3600
```

### 3. Set Appropriate Timeouts

Prevent hanging operations:

```yaml
steps:
  - id: quick_task
    action: "Simple operation"
    timeout: 10.0
    
  - id: complex_task
    action: "Complex analysis"
    timeout: 120.0
```

## Integration Examples

### With FastAPI

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class PipelineRequest(BaseModel):
    topic: str
    options: dict = {}

@app.post("/analyze")
async def analyze(request: PipelineRequest):
    # Load pipeline
    with open("analysis_pipeline.yaml") as f:
        yaml_content = f.read()
    
    # Compile and execute
    pipeline = await compiler.compile(yaml_content, request.dict())
    results = await control_system.execute_pipeline(pipeline)
    
    return {"status": "success", "results": results}
```

### With Celery

```python
from celery import Celery

app = Celery('pipelines')

@app.task
def run_pipeline_task(yaml_file, inputs):
    async def execute():
        with open(yaml_file) as f:
            yaml_content = f.read()
        
        pipeline = await compiler.compile(yaml_content, inputs)
        return await control_system.execute_pipeline(pipeline)
    
    return asyncio.run(execute())
```

## Conclusion

The declarative framework simplifies AI pipeline creation while maintaining flexibility and power. By separating workflow definition from implementation, it enables rapid development and iteration of AI-powered applications.