# Minimal Research Pipeline

**Pipeline**: `examples/research_minimal.yaml`  
**Category**: Research & Analysis  
**Complexity**: Beginner  
**Key Features**: Web search, JSON processing, Template filters, File generation

## Overview

The Minimal Research Pipeline demonstrates the simplest possible research workflow using only web search and text generation. It searches the web for information on a given topic, processes results into structured JSON format, and generates a clean markdown report with sources.

## Key Features Demonstrated

### 1. Web Search Integration
```yaml
- id: search_web
  tool: web-search
  action: search
  parameters:
    query: "{{topic}}"
    max_results: 5
```

### 2. Structured JSON Response
```yaml
response_format: "json_object"
prompt: |
  Return a JSON object with the following structure:
  {
    "topic": "{{topic}}",
    "overview": "Brief one-sentence overview",
    "key_points": ["point1", "point2", "point3"],
    "summary": "Concise 2-3 sentence summary"
  }
```

### 3. JSON Template Processing
```yaml
{% set summary_data = summarize_results.result | from_json %}
{{summary_data.overview}}
```

### 4. Template Loops and Filters
```yaml
{% for result in search_web.results %}
{{loop.index}}. [{{result.title}}]({{result.url}})
{% endfor %}
```

### 5. Dynamic File Naming
```yaml
path: "{{ output_path }}/{{topic | slugify}}_summary.md"
```

## Pipeline Structure

### Input Parameters
- **topic** (required): The research topic to investigate
- **output_path** (optional): Directory for output files (default: "examples/outputs/research_minimal")

### Processing Steps

1. **Web Search** - Searches for information on the topic (5 results)
2. **Summarize Results** - Processes search results into structured JSON format
3. **Save Summary** - Generates markdown report with formatted output

### Dependencies
- Linear flow: search_web → summarize_results → save_summary

## Usage Examples

### Basic Research
```bash
python scripts/run_pipeline.py examples/research_minimal.yaml \
  -i topic="artificial intelligence ethics"
```

### Custom Output Directory
```bash
python scripts/run_pipeline.py examples/research_minimal.yaml \
  -i topic="climate change mitigation" \
  -i output_path="my_research_reports"
```

### Complex Topics
```bash
python scripts/run_pipeline.py examples/research_minimal.yaml \
  -i topic="quantum computing applications in cryptography"
```

## Sample Output

### Example Research Report
**Topic**: "quantum computing basics"  
**Generated File**: [quantum-computing-basics_summary.md](../../examples/outputs/research_minimal/quantum-computing-basics_summary.md)

#### Report Structure:
- **Header**: Research topic and metadata
- **Overview**: One-sentence topic overview
- **Key Findings**: 3 main points extracted from sources
- **Summary**: 2-3 sentence synthesis
- **Sources**: Links to all referenced materials

#### Sample Content:
```markdown
# Research Summary: quantum computing basics
**Date:** 2025-07-31-13:47:58
**Sources Reviewed:** 5 sources reviewed

## Overview
Quantum computing leverages quantum mechanical phenomena to perform computation 
in ways that differ fundamentally from classical computers.

## Key Findings
1. Quantum mechanics describes the behavior of matter and energy at the smallest scales...
2. Quantum computers exploit unique quantum properties such as superposition...
3. The outcomes of quantum computations are inherently probabilistic...

## Summary
Quantum computing is based on the principles of quantum mechanics, where physical 
properties are quantized and particles can exist in multiple states simultaneously...
```

## Advanced Template Features

### JSON Processing
```yaml
# Parse JSON response into template variable
{% set summary_data = summarize_results.result | from_json %}

# Access JSON properties
{{summary_data.overview}}
{{summary_data.summary}}

# Loop through JSON arrays
{% for point in summary_data.key_points %}
{{loop.index}}. {{point}}
{% endfor %}
```

### Template Filters
```yaml
# String formatting
{{topic | slugify}}                    # Convert to filename-safe format
{{ execution.timestamp | date('%Y-%m-%d %H:%M:%S') }}  # Format timestamp

# Data processing
{{search_web.total_results}}           # Access nested properties
{{loop.index}}                         # Loop counter in templates
```

### Search Result Processing
```yaml
{% for result in search_web.results %}
{{loop.index}}. {{result.title}}      # Title from search result
   {{result.snippet}}                  # Description snippet
   Source: {{result.url}}              # Source URL
{% endfor %}
```

## Best Practices Demonstrated

1. **Minimal Dependencies**: Uses only essential tools (web-search, filesystem)
2. **Structured Output**: JSON format ensures consistent data processing
3. **Source Attribution**: All sources properly linked and cited
4. **Clean Formatting**: Professional markdown output with clear sections
5. **Error Prevention**: Template filters handle various data types safely

## Performance Characteristics

- **Speed**: Fast execution with only 3 steps
- **Resource Usage**: Minimal - just web search and text generation
- **Reliability**: Simple pipeline with few failure points
- **Scalability**: Can handle various topic complexities

## Common Use Cases

- **Quick Research**: Fast overview of unfamiliar topics
- **Fact Checking**: Verify basic information about subjects
- **Educational Research**: Student research starting points
- **Content Research**: Background research for articles or presentations
- **Topic Exploration**: Initial investigation of new areas of interest

## Troubleshooting

### Common Issues

#### Web Search Failures
```
Error: Web search returned no results
```
**Solutions**:
- Check internet connectivity
- Verify topic spelling and phrasing
- Try broader or more specific search terms

#### JSON Parsing Errors
```
Error: Invalid JSON in summarize_results
```
**Solutions**:
- Check model response format
- Verify JSON structure in prompt
- Ensure model supports json_object format

#### File Writing Issues
```
Error: Cannot write to output directory
```
**Solutions**:
- Check directory permissions
- Verify output path exists
- Ensure filename is valid (slugify filter helps)

### Optimization Tips

1. **Topic Formulation**: Use clear, specific topics for better results
2. **Search Terms**: Include relevant keywords for comprehensive coverage
3. **Output Organization**: Use descriptive output paths for multiple research sessions
4. **Model Selection**: AUTO tag chooses appropriate model for task complexity

## Extension Examples

### Enhanced Search Parameters
```yaml
parameters:
  query: "{{topic}} recent developments 2024"
  max_results: 10
  time_range: "past_year"
```

### Multiple Search Queries
```yaml
- id: search_general
  tool: web-search
  parameters:
    query: "{{topic}} overview"
    
- id: search_technical
  tool: web-search
  parameters:
    query: "{{topic}} technical details"
```

### Quality Scoring
```yaml
- id: assess_quality
  action: generate_text
  parameters:
    prompt: |
      Rate the quality of this research summary on a scale of 1-10:
      {{summarize_results.result}}
      
      Consider: accuracy, completeness, source quality, clarity
```

## Related Examples
- [research_basic.md](research_basic.md) - More detailed research workflow
- [research_advanced_tools.md](research_advanced_tools.md) - Professional research with PDF output
- [web_research_pipeline.md](web_research_pipeline.md) - Comprehensive web research
- [working_web_search.md](working_web_search.md) - Simple web search example

## Technical Requirements

- **Tools**: web-search, filesystem
- **Models**: AUTO-selected model for summarization (usually fast, efficient model)
- **Network**: Internet access for web search
- **Storage**: Minimal (text files only)
- **Processing**: Low CPU/memory requirements

This pipeline is perfect for beginners learning research automation and serves as a foundation for more complex research workflows. Its simplicity makes it reliable and easy to understand while still producing high-quality, properly sourced research summaries.