# POML Integration Migration Guide

## Overview

This guide provides step-by-step instructions for migrating to Microsoft POML (Prompt Markup Language) integration in the orchestrator framework. POML integration enables enhanced template processing with structured markup, hybrid format support, and advanced template resolution capabilities.

## What is POML?

POML (Prompt Markup Language) is Microsoft's structured markup language designed specifically for prompt engineering and template management. This integration provides:

- **Structured Templates**: XML-like markup for clear prompt organization
- **Hybrid Format Support**: Seamless mixing of POML and Jinja2 templates  
- **Enhanced Resolution**: Advanced cross-task output references
- **Migration Tools**: Automatic conversion from existing templates
- **Backward Compatibility**: Existing templates continue to work unchanged

## Pre-Migration Assessment

### System Requirements

Before starting the migration, ensure your system meets these requirements:

```bash
# Check Python version (3.8+ required)
python --version

# Check if orchestrator is installed and updated
pip show orchestrator-framework

# Check if POML dependencies are available (optional)
pip list | grep poml
```

### Compatibility Checklist

- [ ] **Python 3.8+**: POML integration requires Python 3.8 or higher
- [ ] **Template Files**: Inventory existing template files and formats
- [ ] **Template Usage**: Document current template usage patterns
- [ ] **Dependencies**: Verify template resolution dependencies
- [ ] **Testing**: Identify critical templates for migration testing

### Current Template Assessment

Run this assessment script to analyze your current templates:

```python
# template_assessment.py
import os
import glob
from pathlib import Path
from src.orchestrator.core.template_resolver import TemplateFormatDetector

def assess_current_templates(template_dir="templates/"):
    """Assess current template files for POML migration compatibility."""
    
    detector = TemplateFormatDetector()
    assessment = {
        "jinja2_templates": [],
        "plain_templates": [], 
        "hybrid_templates": [],
        "poml_templates": [],
        "total_files": 0
    }
    
    # Find all template files
    template_patterns = ["*.yaml", "*.yml", "*.j2", "*.jinja", "*.template", "*.txt", "*.md"]
    template_files = []
    
    for pattern in template_patterns:
        template_files.extend(glob.glob(os.path.join(template_dir, "**", pattern), recursive=True))
    
    assessment["total_files"] = len(template_files)
    
    # Analyze each template
    for template_path in template_files:
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            format_type = detector.detect_format(content)
            
            if format_type.value == "jinja2":
                assessment["jinja2_templates"].append(template_path)
            elif format_type.value == "plain":
                assessment["plain_templates"].append(template_path)
            elif format_type.value == "hybrid":
                assessment["hybrid_templates"].append(template_path)
            elif format_type.value == "poml":
                assessment["poml_templates"].append(template_path)
                
        except Exception as e:
            print(f"Warning: Could not analyze {template_path}: {e}")
    
    return assessment

def print_assessment_report(assessment):
    """Print a detailed assessment report."""
    print("Template Assessment Report")
    print("=" * 40)
    print(f"Total template files found: {assessment['total_files']}")
    print()
    
    print(f"Jinja2 templates: {len(assessment['jinja2_templates'])}")
    for template in assessment['jinja2_templates'][:5]:  # Show first 5
        print(f"  - {template}")
    if len(assessment['jinja2_templates']) > 5:
        print(f"  ... and {len(assessment['jinja2_templates']) - 5} more")
    print()
    
    print(f"Plain templates: {len(assessment['plain_templates'])}")
    for template in assessment['plain_templates'][:5]:
        print(f"  - {template}")
    if len(assessment['plain_templates']) > 5:
        print(f"  ... and {len(assessment['plain_templates']) - 5} more")
    print()
    
    print(f"Existing POML templates: {len(assessment['poml_templates'])}")
    print(f"Hybrid templates: {len(assessment['hybrid_templates'])}")

if __name__ == "__main__":
    # Assess templates in common locations
    template_dirs = ["templates/", "examples/templates/", "src/templates/"]
    
    for template_dir in template_dirs:
        if os.path.exists(template_dir):
            print(f"\nAssessing templates in: {template_dir}")
            assessment = assess_current_templates(template_dir)
            print_assessment_report(assessment)
```

## Migration Strategies

### Strategy 1: Gradual Migration (Recommended)

Migrate templates incrementally while maintaining backward compatibility.

**Benefits:**
- Zero downtime migration
- Gradual learning curve
- Easy rollback for specific templates
- Continuous validation

**Process:**
1. Enable POML support (backward compatible)
2. Identify high-impact templates for migration
3. Convert templates one by one
4. Validate each converted template
5. Update documentation and examples

### Strategy 2: Bulk Migration

Convert all templates at once using migration tools.

**Benefits:**
- Consistent template format across system
- One-time migration effort
- Immediate access to all POML features

**Considerations:**
- Requires thorough testing
- Higher risk of issues
- More complex rollback

### Strategy 3: Hybrid Approach

Use both formats side by side indefinitely.

**Benefits:**
- Maximum flexibility
- No forced migration
- Gradual adoption over time

## Migration Steps

### Step 1: Enable POML Support

Enable POML template resolution in your configuration:

```yaml
# config.yaml
template_config:
  # Enable POML template processing
  enable_poml_processing: true
  
  # Template format detection
  auto_detect_format: true
  
  # Fallback behavior
  fallback_to_jinja: true
  
  # Enhanced resolution features
  enable_advanced_resolution: true
  cross_task_references: true
```

Or programmatically:

```python
from src.orchestrator.core.template_resolver import TemplateResolver, TemplateFormat

# Initialize template resolver with POML support
resolver = TemplateResolver(
    enable_poml_processing=True,
    auto_detect_format=True,
    fallback_to_jinja=True
)

# Test POML availability
print(f"POML available: {resolver.poml_available}")
```

### Step 2: Install POML Dependencies (Optional)

For full POML functionality, install the POML library:

```bash
# Install POML library for enhanced features
pip install poml

# Or add to requirements.txt
echo "poml>=1.0.0" >> requirements.txt
pip install -r requirements.txt
```

**Note**: POML integration works without the library installed, providing basic functionality through the built-in parser.

### Step 3: Validate Current Templates

Ensure existing templates continue to work:

```python
# template_validation.py
from src.orchestrator.core.template_resolver import TemplateResolver
import asyncio

async def validate_existing_templates():
    """Validate that existing templates still work after enabling POML."""
    
    resolver = TemplateResolver(enable_poml_processing=True)
    
    # Test templates
    test_cases = [
        {
            "name": "Jinja2 template",
            "content": "Hello {{ name }}! Today is {{ date }}.",
            "context": {"name": "Alice", "date": "2025-08-25"}
        },
        {
            "name": "Plain template", 
            "content": "This is a plain text template without variables.",
            "context": {}
        },
        {
            "name": "POML template",
            "content": "<role>assistant</role><task>Say hello to {{ name }}</task>",
            "context": {"name": "Bob"}
        }
    ]
    
    for test in test_cases:
        try:
            result = await resolver.resolve_template_content(
                test["content"], 
                test["context"]
            )
            print(f"✓ {test['name']}: {result[:50]}...")
        except Exception as e:
            print(f"❌ {test['name']}: {e}")

if __name__ == "__main__":
    asyncio.run(validate_existing_templates())
```

### Step 4: Convert Templates to POML

#### Manual Conversion

Convert templates manually for better control:

**Before (Jinja2):**
```jinja2
You are a helpful assistant. 

Task: {{ task_description }}

Context:
{% for item in context_items %}
- {{ item }}
{% endfor %}

Please provide a detailed response.
```

**After (POML):**
```xml
<role>assistant</role>

<task>{{ task_description }}</task>

<examples>
{% for item in context_items %}
<example>{{ item }}</example>
{% endfor %}
</examples>

<output-format>Please provide a detailed response.</output-format>
```

#### Automated Conversion

Use migration tools for bulk conversion:

```python
# template_converter.py
from src.orchestrator.core.template_migration_tools import TemplateMigrationTools
import asyncio

async def convert_templates():
    """Convert templates using migration tools."""
    
    migration_tools = TemplateMigrationTools()
    
    # Convert single template
    jinja_content = """
    You are {{ role }}.
    Task: {{ task }}
    {% if examples %}
    Examples:
    {% for example in examples %}
    - {{ example }}
    {% endfor %}
    {% endif %}
    """
    
    poml_content = await migration_tools.convert_jinja_to_poml(jinja_content)
    print("Converted POML template:")
    print(poml_content)
    
    # Batch convert directory
    await migration_tools.batch_convert_directory(
        source_dir="templates/",
        target_dir="templates_poml/",
        source_format="jinja2",
        target_format="poml"
    )

if __name__ == "__main__":
    asyncio.run(convert_templates())
```

### Step 5: Test Converted Templates

Thoroughly test converted templates:

```python
# template_testing.py
from src.orchestrator.core.template_resolver import TemplateResolver
import asyncio

async def test_converted_templates():
    """Test converted templates for functionality and output quality."""
    
    resolver = TemplateResolver(enable_poml_processing=True)
    
    # Test cases with the same context
    test_context = {
        "role": "helpful assistant",
        "task": "Explain quantum computing",
        "examples": ["Quantum states", "Superposition", "Entanglement"]
    }
    
    # Original Jinja2 template
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
    
    # Converted POML template  
    poml_template = """
    <role>{{ role }}</role>
    <task>{{ task }}</task>
    {% if examples %}
    <examples>
    {% for example in examples %}
    <example>{{ example }}</example>
    {% endfor %}
    </examples>
    {% endif %}
    """
    
    # Resolve both templates
    jinja_result = await resolver.resolve_template_content(jinja_template, test_context)
    poml_result = await resolver.resolve_template_content(poml_template, test_context)
    
    print("Original Jinja2 result:")
    print(jinja_result)
    print("\nConverted POML result:")
    print(poml_result)
    
    # Compare outputs (basic validation)
    if "quantum computing" in jinja_result.lower() and "quantum computing" in poml_result.lower():
        print("✓ Both templates processed task correctly")
    else:
        print("⚠️ Templates may need adjustment")

if __name__ == "__main__":
    asyncio.run(test_converted_templates())
```

### Step 6: Implement Enhanced Features

Take advantage of POML-specific features:

#### Cross-Task Output References

```xml
<!-- Advanced template with cross-task references -->
<role>data analyst</role>

<task>
Analyze the data from previous task: {{ output_refs.data_processing.processed_data }}
</task>

<context>
Previous analysis: {{ output_refs.initial_analysis.summary }}
</context>

<output-format>
<document>
<section>Executive Summary</section>
<section>Detailed Analysis</section>  
<section>Recommendations</section>
</document>
</output-format>
```

#### Enhanced Document Structure

```xml
<role>technical writer</role>

<task>Create comprehensive documentation</task>

<document type="technical_spec">
  <metadata>
    <title>{{ document_title }}</title>
    <version>{{ version }}</version>
    <date>{{ current_date }}</date>
  </metadata>
  
  <section id="introduction">
    <p>{{ introduction_text }}</p>
  </section>
  
  <section id="requirements">
    {% for requirement in requirements %}
    <requirement id="{{ requirement.id }}">
      <description>{{ requirement.description }}</description>
      <priority>{{ requirement.priority }}</priority>
    </requirement>
    {% endfor %}
  </section>
</document>
```

## Configuration Reference

### Template Resolver Configuration

```python
from src.orchestrator.core.template_resolver import TemplateResolver, TemplateFormat

# Full configuration options
resolver = TemplateResolver(
    # POML processing
    enable_poml_processing=True,
    poml_strict_mode=False,  # Allow mixed content
    
    # Format detection
    auto_detect_format=True,
    default_format=TemplateFormat.HYBRID,
    
    # Fallback behavior
    fallback_to_jinja=True,
    fallback_on_error=True,
    
    # Advanced features
    enable_advanced_resolution=True,
    cross_task_references=True,
    enable_caching=True,
    
    # Validation
    validate_templates=True,
    strict_validation=False
)
```

### Environment Variables

```bash
# POML processing settings
export TEMPLATE_ENABLE_POML=true
export TEMPLATE_AUTO_DETECT_FORMAT=true
export TEMPLATE_FALLBACK_TO_JINJA=true

# Advanced features
export TEMPLATE_CROSS_TASK_REFS=true
export TEMPLATE_ENABLE_CACHING=true

# Validation settings
export TEMPLATE_VALIDATE_TEMPLATES=true
export TEMPLATE_STRICT_VALIDATION=false
```

### YAML Configuration

```yaml
template_config:
  # POML settings
  enable_poml_processing: true
  poml_strict_mode: false
  
  # Detection and fallback
  auto_detect_format: true
  default_format: "hybrid"
  fallback_to_jinja: true
  fallback_on_error: true
  
  # Advanced features
  enable_advanced_resolution: true
  cross_task_references: true
  enable_caching: true
  cache_ttl_seconds: 3600
  
  # Validation
  validate_templates: true
  strict_validation: false
  
  # Performance
  max_template_size_kb: 1024
  template_timeout_seconds: 30
```

## POML Template Syntax Reference

### Basic Structure

```xml
<!-- Role definition -->
<role>assistant|user|system</role>

<!-- Task description -->
<task>Specific task description</task>

<!-- Examples -->
<examples>
  <example>Example input/output</example>
  <example>Another example</example>
</examples>

<!-- Output format specification -->
<output-format>Expected output format</output-format>
```

### Advanced Elements

```xml
<!-- Hints for the model -->
<hint>Additional guidance or constraints</hint>

<!-- Document structure -->
<document type="report|specification|guide">
  <section id="intro">Content</section>
  <section id="details">More content</section>
</document>

<!-- Tables -->
<table>
  <header>Column 1|Column 2|Column 3</header>
  <row>Value 1|Value 2|Value 3</row>
</table>

<!-- Images (for multimodal models) -->
<img src="path/to/image.jpg" alt="Description"/>
```

### Hybrid Templates

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
Return results as:
{{ output_format | default("JSON format") }}
</output-format>
```

## Migration Validation

### Validation Checklist

- [ ] **Template Loading**: All templates load without errors
- [ ] **Format Detection**: Format detection works correctly  
- [ ] **Jinja2 Compatibility**: Existing Jinja2 templates still work
- [ ] **POML Processing**: POML templates process correctly
- [ ] **Hybrid Templates**: Mixed format templates work as expected
- [ ] **Cross-Task References**: Advanced resolution features work
- [ ] **Error Handling**: Fallback mechanisms function properly
- [ ] **Performance**: Template processing performance is acceptable

### Comprehensive Validation Script

```python
# comprehensive_validation.py
import asyncio
import time
from src.orchestrator.core.template_resolver import TemplateResolver, TemplateFormat
from src.orchestrator.core.template_migration_tools import TemplateMigrationTools

async def comprehensive_validation():
    """Comprehensive validation of POML integration."""
    
    print("POML Integration Validation")
    print("=" * 40)
    
    validation_results = {
        "passed": [],
        "failed": [],
        "warnings": []
    }
    
    # Test 1: Resolver initialization
    try:
        resolver = TemplateResolver(enable_poml_processing=True)
        validation_results["passed"].append("Template resolver initialization")
    except Exception as e:
        validation_results["failed"].append(f"Resolver initialization: {e}")
        return validation_results
    
    # Test 2: Format detection
    try:
        detector = resolver.format_detector
        
        test_templates = {
            "jinja2": "Hello {{ name }}!",
            "poml": "<role>assistant</role><task>Say hello</task>", 
            "hybrid": "<role>{{ role }}</role>Hello {{ name }}!",
            "plain": "This is plain text"
        }
        
        for expected_format, template in test_templates.items():
            detected = detector.detect_format(template)
            if detected.value == expected_format:
                validation_results["passed"].append(f"Format detection: {expected_format}")
            else:
                validation_results["warnings"].append(
                    f"Format detection mismatch: expected {expected_format}, got {detected.value}"
                )
    except Exception as e:
        validation_results["failed"].append(f"Format detection: {e}")
    
    # Test 3: Template resolution
    try:
        test_cases = [
            ("Jinja2", "Hello {{ name }}!", {"name": "Alice"}),
            ("POML", "<role>assistant</role><task>Greet {{ name }}</task>", {"name": "Bob"}),
            ("Hybrid", "<role>assistant</role>Hello {{ name }}!", {"name": "Charlie"}),
            ("Plain", "Hello world!", {})
        ]
        
        for test_name, template, context in test_cases:
            start_time = time.time()
            result = await resolver.resolve_template_content(template, context)
            duration = (time.time() - start_time) * 1000  # ms
            
            if result and len(result) > 0:
                validation_results["passed"].append(f"Template resolution: {test_name} ({duration:.1f}ms)")
            else:
                validation_results["failed"].append(f"Template resolution: {test_name} - empty result")
                
    except Exception as e:
        validation_results["failed"].append(f"Template resolution: {e}")
    
    # Test 4: Migration tools
    try:
        migration_tools = TemplateMigrationTools()
        
        # Test conversion
        jinja_template = "You are {{ role }}. Task: {{ task }}"
        poml_result = await migration_tools.convert_jinja_to_poml(jinja_template)
        
        if "<role>" in poml_result and "<task>" in poml_result:
            validation_results["passed"].append("Template conversion")
        else:
            validation_results["warnings"].append("Template conversion may not be optimal")
            
    except Exception as e:
        validation_results["failed"].append(f"Migration tools: {e}")
    
    # Test 5: Performance benchmarks
    try:
        # Benchmark template processing speed
        large_template = "<role>assistant</role><task>" + "x" * 1000 + "{{ variable }}</task>"
        context = {"variable": "test_value"}
        
        start_time = time.time()
        for _ in range(10):
            await resolver.resolve_template_content(large_template, context)
        avg_duration = (time.time() - start_time) / 10 * 1000  # ms
        
        if avg_duration < 100:  # Less than 100ms average
            validation_results["passed"].append(f"Performance: {avg_duration:.1f}ms average")
        else:
            validation_results["warnings"].append(f"Performance: {avg_duration:.1f}ms average (slower than expected)")
            
    except Exception as e:
        validation_results["warnings"].append(f"Performance testing: {e}")
    
    # Display results
    print("\nValidation Results:")
    print("-" * 20)
    
    if validation_results["passed"]:
        print("✓ PASSED:")
        for test in validation_results["passed"]:
            print(f"  ✓ {test}")
    
    if validation_results["warnings"]:
        print("\n⚠️  WARNINGS:")
        for warning in validation_results["warnings"]:
            print(f"  ⚠️  {warning}")
    
    if validation_results["failed"]:
        print("\n❌ FAILED:")
        for failure in validation_results["failed"]:
            print(f"  ❌ {failure}")
    
    # Overall assessment
    total_tests = len(validation_results["passed"]) + len(validation_results["warnings"]) + len(validation_results["failed"])
    success_rate = len(validation_results["passed"]) / total_tests if total_tests > 0 else 0
    
    print(f"\nOverall Success Rate: {success_rate:.1%}")
    
    if len(validation_results["failed"]) == 0:
        print("✓ VALIDATION PASSED")
    else:
        print("❌ VALIDATION FAILED")
    
    return validation_results

if __name__ == "__main__":
    asyncio.run(comprehensive_validation())
```

## Rollback Procedures

### Emergency Rollback

If issues occur, disable POML processing immediately:

```python
# emergency_rollback.py
from src.orchestrator.core.template_resolver import TemplateResolver

def emergency_rollback():
    """Disable POML processing and revert to Jinja2 only."""
    
    # Create resolver with POML disabled
    resolver = TemplateResolver(
        enable_poml_processing=False,
        auto_detect_format=False,
        fallback_to_jinja=True
    )
    
    print("Emergency rollback completed.")
    print("POML processing disabled, using Jinja2 only.")
    
    return resolver

if __name__ == "__main__":
    emergency_rollback()
```

### Configuration Rollback

```yaml
# config.yaml - Emergency rollback configuration
template_config:
  enable_poml_processing: false  # Disable POML
  auto_detect_format: false      # Use default format only
  default_format: "jinja2"       # Force Jinja2
  fallback_to_jinja: true       # Always fallback to Jinja2
```

### Template File Rollback

```bash
# Backup and restore original templates
cp -r templates/ templates_backup/

# If you have original templates in git:
git checkout HEAD~1 -- templates/

# Or restore from backup:
rm -rf templates/
mv templates_backup/ templates/
```

## Best Practices

### Template Design

1. **Start Simple**: Begin with basic POML structures before advanced features
2. **Use Semantic Elements**: Choose POML elements that match your content semantically
3. **Maintain Consistency**: Use consistent element patterns across templates
4. **Document Patterns**: Create style guide for POML usage in your organization

### Migration Strategy

1. **Gradual Adoption**: Migrate high-impact templates first
2. **Test Thoroughly**: Validate each converted template before production use
3. **Monitor Performance**: Track template processing performance after migration
4. **Keep Backups**: Always backup original templates before conversion

### Performance Optimization

1. **Cache Templates**: Enable template caching for frequently used templates
2. **Optimize Large Templates**: Split very large templates into smaller components
3. **Use Appropriate Formats**: Choose the right format (POML/Jinja2/Hybrid) for each use case
4. **Monitor Metrics**: Track template processing times and optimize bottlenecks

## Troubleshooting

### Common Issues

#### Issue: Templates not rendering correctly
```python
# Check format detection
from src.orchestrator.core.template_resolver import TemplateFormatDetector

detector = TemplateFormatDetector()
detected = detector.detect_format(your_template_content)
print(f"Detected format: {detected}")

# Check if POML is available
from src.orchestrator.core.template_resolver import POML_AVAILABLE
print(f"POML library available: {POML_AVAILABLE}")
```

#### Issue: Cross-task references not working
```python
# Check if advanced resolution is enabled
resolver = TemplateResolver(
    enable_poml_processing=True,
    enable_advanced_resolution=True,
    cross_task_references=True
)

# Verify output tracker is configured
from src.orchestrator.core.output_tracker import OutputTracker
tracker = OutputTracker()
print("Output tracker initialized successfully")
```

#### Issue: Performance degradation
```python
# Enable caching
resolver = TemplateResolver(
    enable_poml_processing=True,
    enable_caching=True,
    cache_ttl_seconds=3600
)

# Check template sizes
import os
for template_file in template_files:
    size = os.path.getsize(template_file) / 1024  # KB
    if size > 100:
        print(f"Large template: {template_file} ({size:.1f}KB)")
```

### Support Resources

- **Template Validation**: Use the comprehensive validation script
- **Migration Tools**: Leverage the built-in conversion utilities  
- **Format Detection**: Use format detector for debugging
- **Performance Monitoring**: Enable detailed timing metrics
- **Community Examples**: Check examples/ directory for template patterns

## Expected Benefits

After successful migration, you should see:

- **Enhanced Structure**: Better organized and more readable templates
- **Advanced Features**: Access to cross-task references and enhanced resolution  
- **Flexibility**: Mix POML and Jinja2 syntax as needed
- **Future-Proof**: Ready for advanced prompt engineering features

## Next Steps

1. **Explore Advanced Features**: Try cross-task output references and document structuring
2. **Optimize Templates**: Fine-tune templates for your specific use cases
3. **Share Patterns**: Document successful template patterns for your team
4. **Contribute**: Share improvements and patterns with the community

This POML migration guide provides a comprehensive approach to adopting Microsoft POML integration safely and effectively. The hybrid approach ensures compatibility while enabling advanced features for enhanced prompt engineering.