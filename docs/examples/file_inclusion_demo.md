# File Inclusion Demo Pipeline

**Pipeline**: `examples/file_inclusion_demo.yaml`  
**Category**: Fundamentals  
**Complexity**: Intermediate  
**Key Features**: File inclusion, External templates, Modular prompts, Multiple syntax options

## Overview

The File Inclusion Demo Pipeline demonstrates the powerful file inclusion capabilities of the orchestrator framework. It shows how to organize prompts, templates, and configuration files externally and include them dynamically in pipeline steps, promoting modularity and reusability.

## Key Features Demonstrated

### 1. Template Variable File Inclusion
```yaml
prompt: |
  {{ file:prompts/research_system_prompt.txt }}
  
  Research Topic: {{ research_topic }}
```

### 2. Angle Bracket File Inclusion
```yaml
prompt: |
  << prompts/analysis_instructions.md >>
  
  Research Topic: {{ research_topic }}
```

### 3. Mixed Content Integration
```yaml
prompt: |
  {{ file:prompts/report_template.md }}
  
  Additional formatting guidelines:
  << templates/formatting_guidelines.txt >>
```

### 4. JSON Configuration Inclusion
```yaml
content: |
  Configuration: {{ file:config/report_config.json }}
```

## Pipeline Architecture

### Input Parameters
- **research_topic** (optional): Topic to research (default: "artificial intelligence ethics")
- **output_format** (optional): Report format (default: "markdown")

### External Files Structure
```
examples/
├── prompts/
│   ├── research_system_prompt.txt
│   ├── analysis_instructions.md
│   └── report_template.md
├── templates/
│   └── formatting_guidelines.txt
└── config/
    └── report_config.json
```

### Processing Flow

1. **Load System Prompt** - Include external research system prompt
2. **Conduct Web Search** - Search based on generated research plan
3. **Analyze Results** - Apply analysis instructions from external file
4. **Generate Report** - Use external template with formatting guidelines
5. **Save Report** - Create output file with included configuration

## File Inclusion Syntax

### Template Variable Syntax
```yaml
# Include file content as template variable
{{ file:path/to/file.txt }}
```

### Angle Bracket Syntax
```yaml
# Include file content directly
<< path/to/file.md >>
```

### Comparison
| Syntax | Use Case | Processing |
|--------|----------|------------|
| `{{ file:path }}` | Template content with variables | Variables resolved before inclusion |
| `<< path >>` | Static content inclusion | Direct file content insertion |

## Usage Examples

### Basic Research Report
```bash
python scripts/run_pipeline.py examples/file_inclusion_demo.yaml \
  -i research_topic="quantum computing applications"
```

### Custom Format Report
```bash
python scripts/run_pipeline.py examples/file_inclusion_demo.yaml \
  -i research_topic="climate change mitigation" \
  -i output_format="html"
```

### Multiple Topic Research
```bash
# Research AI Ethics
python scripts/run_pipeline.py examples/file_inclusion_demo.yaml \
  -i research_topic="AI ethics and governance"

# Research Blockchain
python scripts/run_pipeline.py examples/file_inclusion_demo.yaml \
  -i research_topic="blockchain scalability solutions"
```

## External File Examples

### Research System Prompt (`prompts/research_system_prompt.txt`)
```text
You are an expert research analyst. Your task is to:

1. Analyze the given research topic thoroughly
2. Identify key aspects and subtopics
3. Create a comprehensive search strategy
4. Suggest relevant keywords and phrases

Focus on academic credibility and diverse perspectives.
```

### Analysis Instructions (`prompts/analysis_instructions.md`)
```markdown
# Analysis Guidelines

## Evaluation Criteria
- Source credibility and authority
- Recency and relevance of information
- Diversity of perspectives presented
- Evidence quality and supporting data

## Analysis Structure
1. **Summary**: Key findings overview
2. **Themes**: Major patterns and trends
3. **Gaps**: Areas requiring further research
4. **Recommendations**: Next steps and priorities
```

### Report Template (`prompts/report_template.md`)
```markdown
# Research Report: {topic}

## Executive Summary
[Provide 2-3 paragraph overview]

## Key Findings
[List major discoveries and insights]

## Detailed Analysis
[Comprehensive analysis of findings]

## Sources and References
[List all sources with credibility assessment]

## Recommendations
[Action items and future research directions]
```

## Advanced File Inclusion Patterns

### 1. Conditional Inclusion
```yaml
content: |
  {% if research_topic contains "technical" %}
  {{ file:prompts/technical_guidelines.md }}
  {% else %}
  {{ file:prompts/general_guidelines.md }}
  {% endif %}
```

### 2. Dynamic Path Construction
```yaml
prompt: |
  {{ file:prompts/{{ output_format }}_template.md }}
```

### 3. Nested Inclusions
```yaml
# Main template includes sub-templates
{{ file:templates/main_template.md }}
# Which internally includes:
# << sections/introduction.md >>
# << sections/methodology.md >>
# << sections/conclusion.md >>
```

## Configuration File Integration

### JSON Configuration (`config/report_config.json`)
```json
{
  "report_settings": {
    "include_timestamps": true,
    "citation_format": "APA",
    "max_section_length": 1000,
    "include_metadata": true
  },
  "search_parameters": {
    "max_results": 20,
    "source_types": ["academic", "news", "reports"],
    "date_range": "last_2_years"
  }
}
```

### Using Configuration
```yaml
parameters:
  max_results: "{{ (file:config/report_config.json).search_parameters.max_results }}"
  citation_format: "{{ (file:config/report_config.json).report_settings.citation_format }}"
```

## File Organization Best Practices

### Directory Structure
```
examples/
├── prompts/           # System prompts and instructions
│   ├── system/        # System-level prompts
│   ├── analysis/      # Analysis-specific prompts
│   └── generation/    # Content generation prompts
├── templates/         # Output templates
│   ├── reports/       # Report templates
│   └── formats/       # Format-specific templates
├── config/           # Configuration files
│   ├── models/       # Model configurations
│   └── output/       # Output settings
└── schemas/          # Data validation schemas
```

### Naming Conventions
- **Descriptive names**: `research_system_prompt.txt`
- **Category prefixes**: `analysis_instructions.md`
- **Format suffixes**: `report_template_markdown.md`

## Benefits of File Inclusion

### 1. Modularity
- Reusable prompt components
- Shared templates across pipelines
- Centralized configuration management

### 2. Maintainability
- Single source of truth for prompts
- Easy updates without pipeline changes
- Version control for prompt templates

### 3. Organization
- Clean separation of concerns
- Logical file structure
- Easier collaboration

### 4. Flexibility
- Dynamic content selection
- Environment-specific configurations
- Template variations

## Technical Implementation

### File Resolution Process
1. **Path Resolution**: Relative to pipeline file location
2. **File Loading**: Content read from filesystem
3. **Template Processing**: Variables resolved in content
4. **Inclusion**: Content inserted into target location

### Error Handling
```yaml
# Fallback for missing files
content: |
  {{ file:custom_prompt.txt | default("Default prompt content") }}
```

### Performance Considerations
- Files loaded once per pipeline execution
- Content cached during execution
- Large files may impact memory usage

## Common Use Cases

- **Prompt Libraries**: Shared prompt collections
- **Template Systems**: Standardized output formats
- **Configuration Management**: Environment-specific settings
- **Documentation Integration**: Include external documentation
- **Multi-language Support**: Language-specific templates
- **Brand Guidelines**: Consistent formatting and style

## Troubleshooting

### File Not Found Errors
- Verify file paths relative to pipeline location
- Check file permissions and accessibility
- Use absolute paths if needed

### Template Variable Issues
- Ensure proper syntax: `{{ file:path }}`
- Check for circular inclusions
- Validate file content for template syntax

### Performance Issues
- Monitor file sizes for large inclusions
- Consider file caching strategies
- Use conditional inclusion for optional content

## Related Examples
- [research_basic.md](research_basic.md) - Basic research pipeline
- [research_advanced_tools.md](research_advanced_tools.md) - Advanced research with tools
- [multimodal_processing.md](multimodal_processing.md) - Template-based processing

## Technical Requirements

- **File System**: Access to external files and directories
- **Template Engine**: Variable resolution and file inclusion support
- **Path Resolution**: Relative and absolute path handling
- **Error Handling**: Graceful handling of missing files

This pipeline demonstrates how file inclusion creates more maintainable, modular, and reusable pipeline architectures by separating content from logic.