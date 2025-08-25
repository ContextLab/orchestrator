# POML Integration API Reference

## Overview

This document provides comprehensive API reference for the POML (Prompt Markup Language) integration, including template resolution, format detection, migration tools, and advanced template features.

## Core Classes

### TemplateResolver

Main class for resolving templates with POML support.

```python
from src.orchestrator.core.template_resolver import TemplateResolver, TemplateFormat

class TemplateResolver:
    def __init__(
        self,
        enable_poml_processing: bool = True,
        auto_detect_format: bool = True,
        fallback_to_jinja: bool = True,
        enable_advanced_resolution: bool = True,
        cross_task_references: bool = True,
        enable_caching: bool = True,
        cache_ttl_seconds: int = 3600,
        validate_templates: bool = True,
        strict_validation: bool = False
    )
```

**Parameters:**
- `enable_poml_processing`: Enable POML template processing
- `auto_detect_format`: Automatically detect template format
- `fallback_to_jinja`: Fall back to Jinja2 if POML processing fails
- `enable_advanced_resolution`: Enable advanced template resolution features
- `cross_task_references`: Enable cross-task output references
- `enable_caching`: Enable template result caching
- `cache_ttl_seconds`: Cache time-to-live in seconds
- `validate_templates`: Validate template syntax
- `strict_validation`: Use strict validation rules

#### Methods

##### resolve_template_content()

Resolve template content with context variables.

```python
async def resolve_template_content(
    self, 
    template_content: str, 
    context: Dict[str, Any],
    template_format: Optional[TemplateFormat] = None
) -> str
```

**Parameters:**
- `template_content`: Raw template content
- `context`: Variables to substitute in template
- `template_format`: Optional explicit format (auto-detected if None)

**Returns:** Resolved template string

**Example:**
```python
resolver = TemplateResolver()

template = """
<role>{{ role_type }}</role>
<task>{{ task_description }}</task>
"""

context = {
    "role_type": "assistant",
    "task_description": "Explain quantum computing"
}

result = await resolver.resolve_template_content(template, context)
print(result)
# Output:
# <role>assistant</role>
# <task>Explain quantum computing</task>
```

##### resolve_template_file()

Resolve template from file.

```python
async def resolve_template_file(
    self,
    template_path: str,
    context: Dict[str, Any],
    encoding: str = 'utf-8'
) -> str
```

**Parameters:**
- `template_path`: Path to template file
- `context`: Variables to substitute
- `encoding`: File encoding (default: utf-8)

**Returns:** Resolved template string

**Example:**
```python
result = await resolver.resolve_template_file(
    "templates/analysis_template.poml",
    {"data_source": "user_feedback", "analysis_type": "sentiment"}
)
```

##### resolve_with_output_references()

Resolve template with cross-task output references.

```python
async def resolve_with_output_references(
    self,
    template_content: str,
    context: Dict[str, Any],
    output_tracker: OutputTracker
) -> str
```

**Parameters:**
- `template_content`: Template with output references
- `context`: Base context variables
- `output_tracker`: Output tracker for cross-task references

**Returns:** Resolved template with output references

**Example:**
```python
from src.orchestrator.core.output_tracker import OutputTracker

template = """
<role>data analyst</role>
<task>Analyze the processed data: {{ output_refs.data_processing.result }}</task>
<context>Previous summary: {{ output_refs.summary.overview }}</context>
"""

output_tracker = OutputTracker()
result = await resolver.resolve_with_output_references(
    template, context, output_tracker
)
```

##### validate_template()

Validate template syntax and structure.

```python
def validate_template(
    self,
    template_content: str,
    template_format: Optional[TemplateFormat] = None
) -> ValidationResult
```

**Parameters:**
- `template_content`: Template content to validate
- `template_format`: Template format (auto-detected if None)

**Returns:** `ValidationResult` object with validation details

**Example:**
```python
validation = resolver.validate_template(template_content)
if validation.is_valid:
    print("Template is valid")
else:
    print(f"Validation errors: {validation.errors}")
```

### TemplateFormat

Enumeration of supported template formats.

```python
class TemplateFormat(Enum):
    JINJA2 = "jinja2"
    POML = "poml"
    HYBRID = "hybrid"
    PLAIN = "plain"
```

**Format Descriptions:**
- `JINJA2`: Traditional Jinja2 templates with `{{ }}` and `{% %}` syntax
- `POML`: Microsoft POML XML-like structured markup
- `HYBRID`: Mix of POML elements and Jinja2 syntax
- `PLAIN`: Plain text with no template syntax

### TemplateFormatDetector

Automatically detects template format based on content patterns.

```python
class TemplateFormatDetector:
    def __init__(self):
        self.jinja_pattern = re.compile(r'{{\s*[^}]+\s*}}|\{\%\s*[^%]+\s*\%\}')
        self.poml_patterns = [
            re.compile(r'<(role|task|example|examples|hint|output-format|document|table|img|poml)[\s>]'),
            re.compile(r'</(role|task|example|examples|hint|output-format|document|table|img|poml)>'),
        ]
```

#### Methods

##### detect_format()

Detect the format of template content.

```python
def detect_format(self, template_content: str) -> TemplateFormat
```

**Parameters:**
- `template_content`: Template content to analyze

**Returns:** Detected `TemplateFormat`

**Example:**
```python
detector = TemplateFormatDetector()

# Jinja2 template
jinja_template = "Hello {{ name }}! {% if greeting %}{{ greeting }}{% endif %}"
format1 = detector.detect_format(jinja_template)  # Returns: TemplateFormat.JINJA2

# POML template  
poml_template = "<role>assistant</role><task>Help the user</task>"
format2 = detector.detect_format(poml_template)   # Returns: TemplateFormat.POML

# Hybrid template
hybrid_template = "<role>{{ role }}</role><task>Process {{ data }}</task>"
format3 = detector.detect_format(hybrid_template)  # Returns: TemplateFormat.HYBRID
```

##### get_format_confidence()

Get confidence score for format detection.

```python
def get_format_confidence(self, template_content: str, detected_format: TemplateFormat) -> float
```

**Returns:** Confidence score from 0.0 to 1.0

## Migration Tools

### TemplateMigrationTools

Tools for converting between template formats.

```python
from src.orchestrator.core.template_migration_tools import TemplateMigrationTools

class TemplateMigrationTools:
    def __init__(self, preserve_comments: bool = True, validate_output: bool = True)
```

#### Methods

##### convert_jinja_to_poml()

Convert Jinja2 template to POML format.

```python
async def convert_jinja_to_poml(self, jinja_content: str) -> str
```

**Parameters:**
- `jinja_content`: Jinja2 template content

**Returns:** Converted POML template

**Example:**
```python
migration_tools = TemplateMigrationTools()

jinja_template = """
You are {{ role }}.
Task: {{ task }}
{% if examples %}
Examples:
{% for example in examples %}
- {{ example }}
{% endfor %}
{% endif %}
"""

poml_template = await migration_tools.convert_jinja_to_poml(jinja_template)
print(poml_template)
# Output:
# <role>{{ role }}</role>
# <task>{{ task }}</task>
# {% if examples %}
# <examples>
# {% for example in examples %}
# <example>{{ example }}</example>
# {% endfor %}  
# </examples>
# {% endif %}
```

##### convert_poml_to_jinja()

Convert POML template to Jinja2 format.

```python
async def convert_poml_to_jinja(self, poml_content: str) -> str
```

##### batch_convert_directory()

Batch convert all templates in a directory.

```python
async def batch_convert_directory(
    self,
    source_dir: str,
    target_dir: str,
    source_format: str,
    target_format: str,
    file_patterns: List[str] = None,
    preserve_structure: bool = True
) -> BatchConversionResult
```

**Parameters:**
- `source_dir`: Source directory path
- `target_dir`: Target directory path  
- `source_format`: Source format ("jinja2" or "poml")
- `target_format`: Target format ("jinja2" or "poml")
- `file_patterns`: File patterns to match (default: ["*.j2", "*.jinja", "*.template"])
- `preserve_structure`: Preserve directory structure

**Returns:** `BatchConversionResult` with conversion statistics

**Example:**
```python
result = await migration_tools.batch_convert_directory(
    source_dir="templates/jinja/",
    target_dir="templates/poml/",
    source_format="jinja2", 
    target_format="poml",
    file_patterns=["*.j2", "*.jinja"]
)

print(f"Converted {result.successful_conversions} templates")
print(f"Failed: {result.failed_conversions}")
```

##### analyze_template_compatibility()

Analyze template for conversion compatibility.

```python
def analyze_template_compatibility(
    self,
    template_content: str,
    source_format: str,
    target_format: str
) -> CompatibilityAnalysis
```

**Returns:** `CompatibilityAnalysis` with compatibility assessment

### BatchConversionResult

Result of batch template conversion.

```python
@dataclass
class BatchConversionResult:
    total_files: int
    successful_conversions: int
    failed_conversions: int
    skipped_files: int
    conversion_details: List[ConversionDetail]
    errors: List[str]
    warnings: List[str]
```

### CompatibilityAnalysis

Analysis of template conversion compatibility.

```python
@dataclass
class CompatibilityAnalysis:
    compatible: bool
    confidence: float
    potential_issues: List[str]
    required_manual_changes: List[str]
    recommendations: List[str]
```

## POML Template Syntax

### Basic Elements

#### Role Definition
```xml
<role>assistant|user|system</role>
```

**Example:**
```xml
<role>helpful assistant</role>
<role>{{ user_role }}</role>
```

#### Task Description
```xml
<task>Task description</task>
```

**Example:**
```xml
<task>Analyze the following data and provide insights</task>
<task>{{ task_instruction }}</task>
```

#### Examples
```xml
<examples>
  <example>Example content</example>
</examples>

<!-- Or single example -->
<example>Single example</example>
```

**Example:**
```xml
<examples>
  <example>Input: What is AI? Output: Artificial Intelligence is...</example>
  <example>Input: {{ example_input }} Output: {{ example_output }}</example>
</examples>
```

#### Output Format Specification
```xml
<output-format>Format description</output-format>
```

**Example:**
```xml
<output-format>
Return results as JSON with keys: analysis, confidence, recommendations
</output-format>
```

#### Hints
```xml
<hint>Additional guidance</hint>
```

### Advanced Elements

#### Document Structure
```xml
<document type="report|specification|guide">
  <section id="section_id">Content</section>
</document>
```

**Example:**
```xml
<document type="analysis_report">
  <section id="executive_summary">
    <h1>Executive Summary</h1>
    <p>{{ summary_text }}</p>
  </section>
  <section id="methodology">
    <h2>Methodology</h2>
    <p>{{ methodology_description }}</p>
  </section>
</document>
```

#### Tables
```xml
<table>
  <header>Column 1|Column 2|Column 3</header>
  <row>Value 1|Value 2|Value 3</row>
</table>
```

**Example:**
```xml
<table>
  <header>Metric|Value|Unit</header>
  {% for metric in metrics %}
  <row>{{ metric.name }}|{{ metric.value }}|{{ metric.unit }}</row>
  {% endfor %}
</table>
```

#### Images (for multimodal models)
```xml
<img src="path/to/image" alt="Description"/>
```

### Hybrid Syntax

POML elements can be mixed with Jinja2 syntax:

```xml
<role>{{ role_type }}</role>

<task>
Process the following items:
{% for item in items %}
- {{ item.name }}: {{ item.description }}
{% endfor %}
</task>

<examples>
{% if examples %}
  {% for example in examples %}
  <example>{{ example }}</example>
  {% endfor %}
{% else %}
  <example>No examples provided</example>
{% endif %}
</examples>

<output-format>
{% if output_format %}
{{ output_format }}
{% else %}
Return results as structured text
{% endif %}
</output-format>
```

## Advanced Features

### Cross-Task Output References

Reference outputs from previous tasks using the `output_refs` context variable:

```xml
<role>data analyst</role>

<task>
Analyze the processed data from the previous step:
{{ output_refs.data_processing.cleaned_data }}
</task>

<context>
Initial analysis findings: {{ output_refs.initial_analysis.summary }}
Preprocessing notes: {{ output_refs.preprocessing.notes }}
</context>

<output-format>
Build upon the previous analysis and provide:
1. Deeper insights
2. Trend analysis  
3. Actionable recommendations
</output-format>
```

### Template Metadata

Add metadata to templates:

```xml
<poml version="1.0" template-id="analysis_template" author="data_team">

<metadata>
  <title>{{ document_title }}</title>
  <version>{{ template_version }}</version>
  <created>{{ creation_date }}</created>
  <tags>{{ template_tags | join(', ') }}</tags>
</metadata>

<role>{{ role_type }}</role>
<task>{{ task_description }}</task>

</poml>
```

### Conditional POML Elements

Use Jinja2 conditionals with POML elements:

```xml
<role>{{ role_type | default('assistant') }}</role>

<task>{{ base_task }}</task>

{% if include_examples %}
<examples>
  {% for example in examples %}
  <example>{{ example.input }} -> {{ example.output }}</example>
  {% endfor %}
</examples>
{% endif %}

{% if custom_instructions %}
<hint>{{ custom_instructions }}</hint>
{% endif %}

<output-format>
{% if structured_output %}
Return as JSON with schema: {{ output_schema }}
{% else %}
Return as natural language text
{% endif %}
</output-format>
```

## Usage Examples

### Basic POML Template

```python
from src.orchestrator.core.template_resolver import TemplateResolver

async def basic_poml_example():
    resolver = TemplateResolver(enable_poml_processing=True)
    
    template = """
    <role>research assistant</role>
    
    <task>
    Research the topic: {{ research_topic }}
    Focus on: {{ focus_areas | join(', ') }}
    </task>
    
    <examples>
    <example>Topic: Climate Change, Focus: Impact on agriculture</example>
    </examples>
    
    <output-format>
    Provide a structured report with:
    1. Executive Summary
    2. Key Findings
    3. Detailed Analysis
    4. Sources
    </output-format>
    """
    
    context = {
        "research_topic": "Artificial Intelligence in Healthcare",
        "focus_areas": ["diagnosis", "treatment", "patient care"]
    }
    
    result = await resolver.resolve_template_content(template, context)
    print(result)

asyncio.run(basic_poml_example())
```

### Template Format Detection

```python
from src.orchestrator.core.template_resolver import TemplateFormatDetector

def format_detection_example():
    detector = TemplateFormatDetector()
    
    templates = {
        "jinja2": "Hello {{ name }}! {% if greeting %}{{ greeting }}{% endif %}",
        "poml": "<role>assistant</role><task>Help the user with {{ task }}</task>",
        "hybrid": "<role>{{ role }}</role>Process: {{ data | upper }}",
        "plain": "This is just plain text without any template syntax"
    }
    
    for expected, content in templates.items():
        detected = detector.detect_format(content)
        confidence = detector.get_format_confidence(content, detected)
        
        print(f"Expected: {expected}")
        print(f"Detected: {detected.value}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Match: {'✓' if expected == detected.value else '✗'}")
        print()

format_detection_example()
```

### Template Migration

```python
from src.orchestrator.core.template_migration_tools import TemplateMigrationTools

async def migration_example():
    migration_tools = TemplateMigrationTools()
    
    # Convert Jinja2 to POML
    jinja_template = """
    You are a {{ role }}.
    
    Instructions: {{ instructions }}
    
    {% if examples %}
    Here are some examples:
    {% for example in examples %}
    Example {{ loop.index }}: {{ example }}
    {% endfor %}
    {% endif %}
    
    Please respond in {{ output_format }} format.
    """
    
    # Convert to POML
    poml_template = await migration_tools.convert_jinja_to_poml(jinja_template)
    print("Converted POML template:")
    print(poml_template)
    
    # Analyze compatibility
    compatibility = migration_tools.analyze_template_compatibility(
        jinja_template, "jinja2", "poml"
    )
    
    print(f"\nCompatibility: {'✓' if compatibility.compatible else '✗'}")
    print(f"Confidence: {compatibility.confidence:.2f}")
    if compatibility.potential_issues:
        print(f"Issues: {compatibility.potential_issues}")

asyncio.run(migration_example())
```

### Advanced Template with Output References

```python
from src.orchestrator.core.template_resolver import TemplateResolver
from src.orchestrator.core.output_tracker import OutputTracker

async def advanced_template_example():
    resolver = TemplateResolver(
        enable_poml_processing=True,
        cross_task_references=True
    )
    
    output_tracker = OutputTracker()
    
    # Simulate previous task outputs
    output_tracker.save_output("data_collection", {
        "raw_data": "user feedback data...",
        "metadata": {"source": "survey", "count": 1500}
    })
    
    output_tracker.save_output("preprocessing", {
        "cleaned_data": "processed feedback...",
        "issues_found": ["duplicates", "incomplete responses"]
    })
    
    template = """
    <role>senior data analyst</role>
    
    <task>
    Perform sentiment analysis on the preprocessed data:
    {{ output_refs.preprocessing.cleaned_data }}
    
    Consider the data quality issues found:
    {% for issue in output_refs.preprocessing.issues_found %}
    - {{ issue }}
    {% endfor %}
    </task>
    
    <context>
    Original data source: {{ output_refs.data_collection.metadata.source }}
    Sample size: {{ output_refs.data_collection.metadata.count }}
    </context>
    
    <output-format>
    <document type="analysis_report">
      <section id="sentiment_distribution">
        Sentiment distribution analysis
      </section>
      <section id="quality_impact">
        Impact of data quality issues on analysis
      </section>
      <section id="recommendations">
        Recommendations for data collection improvements
      </section>
    </document>
    </output-format>
    """
    
    result = await resolver.resolve_with_output_references(
        template, {}, output_tracker
    )
    print(result)

asyncio.run(advanced_template_example())
```

### Batch Template Conversion

```python
import os
from src.orchestrator.core.template_migration_tools import TemplateMigrationTools

async def batch_conversion_example():
    migration_tools = TemplateMigrationTools()
    
    # Create sample templates for conversion
    os.makedirs("sample_templates", exist_ok=True)
    
    # Sample Jinja2 template
    with open("sample_templates/analysis.j2", "w") as f:
        f.write("""
        You are {{ role }}.
        
        Task: {{ task }}
        
        {% if data %}
        Data to analyze:
        {% for item in data %}
        - {{ item }}
        {% endfor %}
        {% endif %}
        """)
    
    # Another sample
    with open("sample_templates/report.jinja", "w") as f:
        f.write("""
        Generate a {{ report_type }} report.
        
        Include:
        {% for section in sections %}
        {{ loop.index }}. {{ section }}
        {% endfor %}
        """)
    
    # Batch convert
    result = await migration_tools.batch_convert_directory(
        source_dir="sample_templates",
        target_dir="converted_templates",
        source_format="jinja2",
        target_format="poml"
    )
    
    print(f"Conversion Results:")
    print(f"Total files: {result.total_files}")
    print(f"Successful: {result.successful_conversions}")
    print(f"Failed: {result.failed_conversions}")
    
    for detail in result.conversion_details:
        print(f"  {detail.source_file} -> {detail.target_file}")
        if detail.warnings:
            print(f"    Warnings: {detail.warnings}")

asyncio.run(batch_conversion_example())
```

## Configuration

### Environment Variables

```bash
# POML processing
TEMPLATE_ENABLE_POML=true
TEMPLATE_AUTO_DETECT_FORMAT=true
TEMPLATE_FALLBACK_TO_JINJA=true

# Advanced features
TEMPLATE_CROSS_TASK_REFS=true
TEMPLATE_ENABLE_CACHING=true
TEMPLATE_CACHE_TTL=3600

# Validation
TEMPLATE_VALIDATE_TEMPLATES=true
TEMPLATE_STRICT_VALIDATION=false

# Performance
TEMPLATE_MAX_SIZE_KB=1024
TEMPLATE_TIMEOUT_SECONDS=30
```

### YAML Configuration

```yaml
template_config:
  enable_poml_processing: true
  auto_detect_format: true
  fallback_to_jinja: true
  
  advanced_features:
    enable_advanced_resolution: true
    cross_task_references: true
    enable_caching: true
    cache_ttl_seconds: 3600
  
  validation:
    validate_templates: true
    strict_validation: false
  
  performance:
    max_template_size_kb: 1024
    template_timeout_seconds: 30
    
  migration:
    preserve_comments: true
    validate_output: true
    backup_originals: true
```

### Programmatic Configuration

```python
from src.orchestrator.core.template_resolver import TemplateResolver

# Custom configuration
resolver = TemplateResolver(
    enable_poml_processing=True,
    auto_detect_format=True,
    fallback_to_jinja=True,
    enable_advanced_resolution=True,
    cross_task_references=True,
    enable_caching=True,
    cache_ttl_seconds=1800,  # 30 minutes
    validate_templates=True,
    strict_validation=False
)
```

## Error Handling

### POML-Specific Exceptions

```python
class POMLException(Exception):
    """Base exception for POML-related errors."""
    pass

class TemplateParsingError(POMLException):
    """Raised when template cannot be parsed."""
    pass

class TemplateResolutionError(POMLException):
    """Raised when template resolution fails."""
    pass

class TemplateValidationError(POMLException):
    """Raised when template validation fails."""
    pass

class TemplateMigrationError(POMLException):
    """Raised when template migration fails."""
    pass
```

### Error Handling Examples

```python
from src.orchestrator.core.template_resolver import TemplateResolver, TemplateValidationError

async def error_handling_example():
    resolver = TemplateResolver()
    
    # Invalid template
    invalid_template = """
    <role>{{ role }
    <task>{{ task }}</task>
    """  # Missing closing }} in role
    
    try:
        result = await resolver.resolve_template_content(
            invalid_template, 
            {"role": "assistant", "task": "help user"}
        )
    except TemplateValidationError as e:
        print(f"Template validation failed: {e}")
        # Fall back to plain text or fixed template
        
    except TemplateResolutionError as e:
        print(f"Template resolution failed: {e}")
        # Log error and use fallback template
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        # Generic error handling
```

## Performance Considerations

### Template Caching

```python
# Enable caching for better performance
resolver = TemplateResolver(
    enable_caching=True,
    cache_ttl_seconds=3600  # Cache for 1 hour
)

# Cache is automatically used for repeated templates
template = "<role>{{ role }}</role><task>{{ task }}</task>"

# First call - processes and caches
result1 = await resolver.resolve_template_content(template, context1)

# Second call - uses cached parsed template
result2 = await resolver.resolve_template_content(template, context2)
```

### Large Template Handling

```python
# Configure for large templates
resolver = TemplateResolver(
    enable_poml_processing=True,
    validate_templates=False,  # Skip validation for large templates
    enable_caching=True
)

# Monitor template processing time
import time

start_time = time.time()
result = await resolver.resolve_template_content(large_template, context)
duration = time.time() - start_time

if duration > 1.0:  # More than 1 second
    print(f"⚠️  Slow template processing: {duration:.2f}s")
```

This comprehensive API reference provides all the tools needed to work with POML templates, migrate existing templates, and leverage advanced template resolution features in your applications.