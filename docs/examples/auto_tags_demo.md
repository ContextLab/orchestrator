# AUTO Tags Demo Pipeline

**Pipeline**: `examples/auto_tags_demo.yaml`  
**Category**: AI & Model Features  
**Complexity**: Intermediate  
**Key Features**: AUTO tags, Dynamic decisions, Conditional processing

## Overview

The AUTO Tags Demo Pipeline showcases the power of AUTO tags for dynamic, AI-driven decision making within pipelines. Instead of hardcoding values, AUTO tags allow the AI to make intelligent choices based on context, content analysis, and processing requirements.

## Key Concepts Demonstrated

### 1. Dynamic Model Selection
```yaml
model: <AUTO task="strategy">Based on content type {{ analyze_content }} and complexity {{ task_complexity }}, select appropriate model for strategic decision</AUTO>
```

### 2. Context-Aware Parameter Resolution
```yaml
max_tokens: <AUTO type="integer" min="100" max="1000">Based on strategy {{ select_processing_strategy.strategy }}, how many tokens needed for good analysis?</AUTO>
```

### 3. Conditional Step Execution  
```yaml
condition: <AUTO>Quality score from assessment is {{ assess_quality }}. Should we improve if score < 7? Answer 'true' or 'false'</AUTO>
```

## Pipeline Structure

### Input Parameters
- **content** (string): Text content to analyze (default: AI healthcare example)
- **task_complexity** (string): Complexity level - "simple", "medium", or "complex" (default: "medium")

### Processing Steps

1. **Content Analysis** - Determines content type (technical/academic/casual/business)
2. **Strategy Selection** - AI chooses processing approach based on content and complexity
3. **Format Decision** - Selects optimal output format (markdown/json/plain)
4. **Content Processing** - Processes content using dynamically selected parameters
5. **Quality Assessment** - Evaluates output quality with AI-chosen criteria
6. **Conditional Improvement** - Improves output if quality score is below threshold
7. **Report Generation** - Creates comprehensive summary of all dynamic decisions

### Key Features

#### AUTO Tag Types Used
- **Model Selection**: Chooses appropriate AI model based on task requirements
- **Parameter Optimization**: Sets token limits and detail levels automatically
- **Format Selection**: Picks best output format for content type
- **Quality Criteria**: Determines appropriate evaluation standards
- **Conditional Logic**: Makes yes/no decisions based on quality thresholds

#### Dynamic Decision Points
1. Content type classification
2. Model selection for strategic decisions
3. Processing strategy (simple/advanced/expert)
4. Output format selection
5. Token allocation optimization
6. Quality assessment criteria
7. Conditional improvement trigger

## Usage Examples

### Basic Usage
```bash
python scripts/run_pipeline.py examples/auto_tags_demo.yaml
```

### With Custom Content
```bash
python scripts/run_pipeline.py examples/auto_tags_demo.yaml \
  -i content="Quantum computing leverages quantum mechanical phenomena to process information in fundamentally new ways" \
  -i task_complexity="complex"
```

### Business Content Example
```bash
python scripts/run_pipeline.py examples/auto_tags_demo.yaml \
  -i content="Our Q4 revenue increased by 23% driven by strong performance in cloud services and enterprise solutions" \
  -i task_complexity="simple"
```

## Expected Outputs

The pipeline produces structured outputs showing all dynamic decisions:

### Output Structure
```yaml
outputs:
  content_type: "technical"           # AI-determined content classification
  processing_strategy: "advanced"    # AI-selected processing approach  
  selected_model: "gpt-4o-mini"     # AI-chosen model for the task
  output_format: "markdown"         # AI-selected format
  final_analysis: "..."             # Processed content (improved if needed)
  quality_score: "8/10"             # AI quality assessment
  demo_report: "..."                # Comprehensive summary report
```

### Sample Decision Flow
For technical healthcare content with medium complexity:
1. **Content Type**: "technical" (high precision medical terminology)
2. **Strategy**: "advanced" (complex analysis needed)
3. **Model**: "gpt-4o-mini" (good balance for technical content)
4. **Format**: "markdown" (structured technical documentation)
5. **Tokens**: 750 (sufficient for advanced analysis)
6. **Quality**: "completeness" (technical content needs comprehensive coverage)

## Advanced Features

### Context-Aware AUTO Tags
AUTO tags can reference previous step outputs:
```yaml
criteria: <AUTO>What quality criteria should we use for {{ analyze_content }} content: "accuracy", "completeness", "clarity", or "all"?</AUTO>
```

### Type-Constrained AUTO Tags
```yaml
max_tokens: <AUTO type="integer" min="100" max="1000">Based on strategy {{ select_processing_strategy.strategy }}, how many tokens needed for good analysis?</AUTO>
```

### Multi-Variable AUTO Tags
```yaml
strategy: <AUTO>Given content type "{{ analyze_content }}" and complexity "{{ task_complexity }}", what processing approach is best: "simple", "advanced", or "expert"?</AUTO>
```

## Best Practices Demonstrated

1. **Progressive Decision Making**: Each AUTO tag builds on previous decisions
2. **Context Preservation**: Template variables carry information between steps
3. **Quality Feedback Loop**: Quality assessment influences conditional improvement
4. **Resource Optimization**: Token allocation adapts to processing requirements
5. **Transparent Decision Trail**: All decisions are captured in outputs

## Common Use Cases

- **Content-Aware Processing**: Adapt processing based on content type and complexity
- **Dynamic Resource Allocation**: Optimize computational resources automatically
- **Quality-Driven Workflows**: Implement feedback loops with quality thresholds
- **Adaptive Model Selection**: Choose models based on task requirements
- **Conditional Pipeline Paths**: Execute steps based on AI-determined conditions

## Troubleshooting

### Common Issues
- **AUTO tag resolution failures**: Ensure prompts are clear and specific
- **Circular dependencies**: Avoid AUTO tags that reference their own outputs
- **Type mismatches**: Use type constraints for numeric AUTO tags

### Debugging AUTO Tags
1. Check the demo report output to see all resolved values
2. Review step-by-step decision making in the execution trace
3. Verify template variables are properly passed between steps

## Related Examples
- [llm_routing_pipeline.md](llm_routing_pipeline.md) - Advanced routing with AUTO tags
- [fact_checker.md](fact_checker.md) - AUTO tags for list generation
- [control_flow_conditional.md](control_flow_conditional.md) - Conditional processing

## Technical Notes

- AUTO tags are resolved during pipeline compilation
- Each AUTO tag creates a separate AI call for decision making
- Template variables are resolved before AUTO tag evaluation
- Type constraints ensure AUTO tag outputs match expected formats

This pipeline serves as a comprehensive introduction to AUTO tags and their applications in creating intelligent, adaptive pipelines.