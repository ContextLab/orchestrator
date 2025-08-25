# Template Examples for POML Integration

This directory contains examples demonstrating the enhanced template system with Microsoft POML integration.

## Directory Structure

- `poml/` - Pure POML template examples
- `hybrid/` - Hybrid templates combining POML and Jinja2 syntax
- Traditional Jinja2 templates continue to work unchanged in their original locations

## Template Formats Supported

### 1. Jinja2 Templates (Existing)
Traditional templates with variables and logic:
```jinja2
Hello {{ user_name }}, please analyze {{ data_file }}
{% if urgent %}This is urgent{% endif %}
```

### 2. POML Templates (New)
Structured markup templates with semantic components:
```xml
<role>Data Analysis Expert</role>
<task>Analyze the provided dataset and generate insights</task>
<example>
  <input>Process sales_data.csv</input>
  <output>Revenue: $50K, Growth: +15%</output>
</example>
<hint>Focus on quantitative metrics</hint>
<output-format>JSON with key findings</output-format>
```

### 3. Hybrid Templates (New)
Combining POML structure with Jinja2 variables:
```xml
<role>{{ analyst_role | default("Data Analyst") }}</role>
<task>Analyze {{ data_source }} for {{ stakeholder }}</task>
<document src="{{ input_file_path }}" type="csv">
  Source data for analysis
</document>
<output-format>{{ output_format | upper }}</output-format>
```

## Key Features

### Automatic Format Detection
The system automatically detects template format:
- **POML**: Contains `<role>`, `<task>`, or other POML tags
- **Hybrid**: Contains both POML tags and Jinja2 variables
- **Jinja2**: Contains `{{ }}` or `{% %}` syntax only  
- **Plain**: No template syntax detected

### Backward Compatibility
All existing Jinja2 templates work unchanged:
- Same variable resolution
- Same filter support
- Same conditional and loop logic
- Enhanced field access for nested objects

### Data Integration Components
POML templates support rich data integration:

#### Document Components
```xml
<document src="{{ results_file }}" type="csv">
  Analysis results data
</document>

<document src="/reports/summary.pdf" parser="pdf">
  Executive summary document  
</document>
```

#### Table Components
```xml
<table src="{{ metrics_table }}">
  Performance metrics data
</table>
```

#### Image Components (Future)
```xml
<img src="{{ chart_image }}" alt="Sales trends chart" />
```

### Context Integration
Both template types have access to:
- Pipeline output data
- Task results and metadata
- Loop context variables
- User-defined context

## Usage Examples

### Basic POML Template
```python
from orchestrator.core.template_resolver import TemplateResolver

resolver = TemplateResolver(output_tracker, enable_poml=True)

template = """
<role>Customer Service Analyst</role>
<task>Analyze customer feedback and identify key issues</task>
<hint>Focus on actionable insights</hint>
"""

result = resolver.resolve_template(template, context)
```

### Hybrid Template with Pipeline Data
```python
template = """
<role>{{ user.role }}</role>
<task>Process results from {{ previous_step }}</task>
<document src="{{ data_processor.output_file }}" type="csv">
  Processed customer data
</document>
<output-format>{{ report_format | default("JSON") }}</output-format>
"""

result = resolver.resolve_template(template, context)
```

### Template Migration
```python
from orchestrator.core.template_migration_tools import migrate_template

# Analyze existing template
analysis = analyze_template(old_jinja_template)
print(f"Suggested strategy: {analysis.suggested_strategy}")

# Migrate template
result = migrate_template(old_jinja_template)
if result.success:
    print("Migration successful!")
    new_template = result.migrated_template
```

## Migration Guide

### Step 1: Analyze Existing Templates
```python
from orchestrator.core.template_migration_tools import analyze_template

analysis = analyze_template(your_template)
print(f"Format: {analysis.original_format}")
print(f"Complexity: {analysis.complexity_score}")
print(f"Strategy: {analysis.suggested_strategy}")
```

### Step 2: Migrate Templates
Choose migration strategy based on analysis:

#### Full POML (Simple templates)
- Convert entirely to POML structure
- Best for templates without complex logic
- Provides maximum semantic structure

#### Hybrid (Medium complexity)
- Add POML structure around existing variables
- Keep Jinja2 variables and simple logic
- Balance of structure and flexibility

#### Enhanced Jinja2 (Complex templates)
- Keep existing Jinja2 structure
- Add POML data components where beneficial
- Minimal disruption to existing logic

### Step 3: Test and Validate
```python
# Test migrated template
resolver = TemplateResolver(output_tracker, enable_poml=True)
result = resolver.resolve_template(migrated_template, test_context)

# Validate template structure
issues = resolver.validate_poml_template(migrated_template)
if not issues:
    print("Template validation passed!")
```

## Best Practices

### 1. Template Structure
- Always include `<role>` and `<task>` in POML templates
- Use `<example>` with `<input>`/`<output>` for clarity
- Add `<hint>` tags for important guidance
- Specify `<output-format>` for structured outputs

### 2. Data Integration
- Use `<document>` for file references
- Specify `type` attribute for proper parsing
- Use template variables in `src` attributes for dynamic paths
- Provide descriptive content for each data component

### 3. Hybrid Templates
- Keep POML structure simple when mixing with Jinja2
- Use POML for semantic structure, Jinja2 for logic
- Test thoroughly with various context values

### 4. Migration Strategy
- Start with low-complexity templates
- Use batch operations for multiple templates
- Validate migrated templates before deployment
- Keep backup copies of original templates

## Troubleshooting

### Common Issues

#### Template Not Recognized as POML
- Ensure POML tags are properly closed: `<role>content</role>`
- Check for typos in tag names
- Verify POML SDK is installed: `pip install poml`

#### Hybrid Template Variables Not Resolving
- Check variable names match context keys
- Use fallback syntax: `{{ var | default("fallback") }}`
- Verify context data is properly structured

#### Data Component Errors
- Check file paths are accessible
- Verify file type matches specified parser
- Use template variables for dynamic paths

#### Migration Issues
- Complex templates may need manual adjustment
- Review migration notes and validation issues
- Consider simpler migration strategy for complex cases

### Getting Help
- Check template format: `resolver.get_template_format(template)`
- Validate POML: `resolver.validate_poml_template(template)`
- Analyze migration: `analyze_template(template)`
- Test with minimal context first