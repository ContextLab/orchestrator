# Migration Guide

## Overview

This guide helps you migrate existing pipelines and code from older versions of the Orchestrator framework to the current version. It covers breaking changes, deprecated features, and provides step-by-step migration instructions.

**Current Version**: v2.x  
**Previous Versions Covered**: v1.x, v1.5.x

## Breaking Changes Summary

### Major Changes in v2.x

1. **Unified Template Resolver**: Template rendering system redesigned for better reliability
2. **Model Registry Unification**: Singleton pattern for consistent model access
3. **Output Sanitizer**: Automatic removal of conversational markers from AI outputs
4. **Enhanced Loop Variables**: Improved support for while loop variable resolution
5. **Tool Registration**: Updated tool registration patterns
6. **Configuration Format**: Updated YAML schema for better validation

## Model Registry Migration (v1.x → v2.x)

### Issue: Multiple Model Registry Instances

**Problem**: In v1.x, different components created separate model registry instances, causing models registered in one component to be unavailable in others.

### Migration Steps

#### 1. Update Model Registry Imports

**Old Code (v1.x)**:
```python
from orchestrator.models import ModelRegistry

class MyComponent:
    def __init__(self, model_registry=None):
        self.model_registry = model_registry or ModelRegistry()
```

**New Code (v2.x)**:
```python
from orchestrator.models import get_model_registry

class MyComponent:
    def __init__(self, model_registry=None):
        self.model_registry = model_registry or get_model_registry()
```

#### 2. Update Custom Model Registration

**Old Code (v1.x)**:
```python
# Each component had its own registry
registry = ModelRegistry()
registry.register_model("my-custom-model", MyModel())

# Pass registry to orchestrator
orchestrator = Orchestrator(model_registry=registry)
```

**New Code (v2.x)**:
```python
# Register in the global singleton registry
from orchestrator.models import get_model_registry

registry = get_model_registry()
registry.register_model("my-custom-model", MyModel())

# Orchestrator automatically uses the global registry
orchestrator = Orchestrator()
```

#### 3. Update Test Code

**Old Code (v1.x)**:
```python
def test_with_custom_models():
    registry = ModelRegistry()
    registry.register_test_models()
    orchestrator = Orchestrator(model_registry=registry)
```

**New Code (v2.x)**:
```python
from orchestrator.models import reset_model_registry, set_model_registry

def test_with_custom_models():
    # Reset to clean state
    reset_model_registry()
    
    # Or set a custom test registry
    test_registry = ModelRegistry()
    test_registry.register_test_models()
    set_model_registry(test_registry)
    
    orchestrator = Orchestrator()
```

### Compatibility Notes

- **Backward Compatible**: Old initialization patterns still work but are deprecated
- **Performance Impact**: No performance impact, single registry is more efficient
- **Memory Usage**: Reduced memory usage due to single registry instance

## Template Rendering Migration

### Issue: Conditional Step Template Failures

**Problem**: Templates referencing skipped conditional steps would fail with `UndefinedError`.

### Migration Steps

#### 1. Update Conditional Step References

**Old Code (v1.x)** - Would fail if `enhance_text` step was skipped:
```yaml
steps:
  - id: enhance_text
    if: "{{ quality < threshold }}"
    action: enhance_text
    
  - id: use_text
    action: process
    parameters:
      # This would fail if enhance_text was skipped
      text: "{% if enhance_text.result %}{{ enhance_text.result }}{% else %}{{ original_text }}{% endif %}"
```

**New Code (v2.x)** - Works reliably:
```yaml
steps:
  - id: enhance_text
    if: "{{ quality < threshold }}"
    action: enhance_text
    
  - id: use_text
    action: process
    parameters:
      # Safe reference with default filter
      text: "{{ enhance_text.result | default(original_text) }}"
    dependencies: [enhance_text]  # Ensure condition is evaluated
```

#### 2. Alternative Safe Patterns

**Explicit Existence Checking**:
```yaml
text: |
  {% if enhance_text is defined and enhance_text and enhance_text.result %}
    {{ enhance_text.result }}
  {% else %}
    {{ original_text }}
  {% endif %}
```

**Using Default Filters**:
```yaml
# Simple default
text: "{{ (enhance_text.result) | default(original_text) }}"

# Complex default with fallback chain
text: "{{ (enhance_text.result) | default((basic_text.result) | default(original_text)) }}"
```

### Automatic Fixes Applied

The v2.x framework automatically handles:
- **Skipped Steps**: Registered with `None` values in template context
- **Undefined Attributes**: Return `None` instead of raising errors
- **Conditional Dependencies**: Proper dependency resolution for conditional steps

## Loop Variables Migration

### Issue: While Loop Variable Resolution

**Problem**: In v1.x, while loop variables weren't properly resolved in templates within loop iterations.

### Migration Steps

#### 1. Update While Loop Syntax

**Old Code (v1.x)** - Variable resolution issues:
```yaml
- id: iterative_improvement
  while: "{{ quality < 0.8 }}"
  steps:
    - id: improve_content
      parameters:
        # Variables might not resolve correctly
        iteration: "{{ loop_iteration }}"
        current_quality: "{{ quality }}"
```

**New Code (v2.x)** - Reliable variable resolution:
```yaml
- id: iterative_improvement
  while: "{{ quality < 0.8 and iterations < max_iterations }}"
  steps:
    - id: improve_content
      parameters:
        # Variables properly resolved
        iteration: "{{ iterations }}"
        current_quality: "{{ quality }}"
        
  loop_vars:
    iterations: "{{ iterations + 1 }}"
    quality: "{{ improve_content.quality_score }}"
```

#### 2. Enhanced Loop Variable Support

**New Features in v2.x**:
```yaml
# Access to loop metadata
- id: process_with_metadata  
  while: "{{ continue_processing }}"
  steps:
    - id: process_step
      parameters:
        iteration_count: "{{ $iteration }}"    # Current iteration number
        loop_start_time: "{{ $loop_start }}"   # When loop started
        elapsed_time: "{{ $elapsed }}"         # Time since loop start
```

## Output Sanitization Migration

### Issue: Conversational AI Output

**Problem**: AI models often return conversational text like "Certainly! Here is the content you requested:" instead of clean output.

### Migration Steps

#### 1. Update Prompts for Clean Output

**Old Code (v1.x)** - Required manual output cleaning:
```yaml
- id: generate_data
  action: generate_text
  parameters:
    prompt: "Generate a JSON object with user data"
    
# Manual cleaning step required
- id: clean_output
  action: generate_text
  parameters:
    prompt: |
      Clean this response to contain only valid JSON:
      {{ generate_data.result }}
```

**New Code (v2.x)** - Automatic sanitization:
```yaml
- id: generate_data
  action: generate_text
  parameters:
    prompt: |
      Generate a JSON object with user data.
      
      Return ONLY valid JSON without any explanatory text.
      Do not include markdown formatting or code fences.
      Start with { and end with }
  # Output automatically sanitized by OutputSanitizer
```

#### 2. Configure Output Sanitization

**Customizing Sanitizer Behavior**:
```yaml
# Enable/disable sanitization per step
- id: generate_with_custom_sanitization
  action: generate_text
  parameters:
    prompt: "{{ prompt_text }}"
    sanitize_output: true  # Explicit control
    sanitizer_config:
      remove_markers: true      # Remove conversational markers
      clean_json: true          # Clean JSON formatting
      remove_code_fences: true  # Remove ```json fences
```

### Automatic Sanitization Features

The v2.x OutputSanitizer automatically removes:
- Conversational markers ("Certainly!", "Here is", etc.)
- Code fences (```json, ```yaml, etc.)
- Explanatory prefixes and suffixes
- Markdown formatting artifacts
- Extra whitespace and formatting

## Tool Registration Migration

### Issue: Tool Registration Patterns

**Problem**: Tool registration patterns have been updated for better consistency and error handling.

### Migration Steps

#### 1. Update Tool Registration

**Old Code (v1.x)**:
```python
from orchestrator.tools import ToolRegistry

# Manual registry management
registry = ToolRegistry()
registry.register("my_tool", MyTool())

orchestrator = Orchestrator(tool_registry=registry)
```

**New Code (v2.x)**:
```python
from orchestrator.tools import get_tool_registry

# Use global registry
registry = get_tool_registry()
registry.register("my_tool", MyTool())

# Orchestrator uses global registry automatically
orchestrator = Orchestrator()
```

#### 2. Update Tool Definitions

**Enhanced Tool Interface**:
```python
from orchestrator.tools import BaseTool

class MyTool(BaseTool):
    name = "my_tool"
    version = "2.0"
    
    def __init__(self):
        super().__init__()
        # Enhanced initialization
        
    async def execute(self, action: str, parameters: dict) -> dict:
        # Enhanced error handling and validation
        try:
            result = await self._process_action(action, parameters)
            return {
                "success": True,
                "result": result,
                "metadata": {
                    "execution_time": self.execution_time,
                    "tool_version": self.version
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
```

## Configuration Format Updates

### YAML Schema Changes

#### 1. Parameter Definitions

**Old Format (v1.x)**:
```yaml
parameters:
  input_file: "data.json"
  threshold: 0.5
  options:
    - "option1"
    - "option2"
```

**New Format (v2.x)** - Enhanced with validation:
```yaml
parameters:
  input_file:
    type: string
    required: true
    description: "Path to input data file"
    pattern: "^.*\\.(json|csv)$"
    
  threshold:
    type: number
    default: 0.5
    minimum: 0.0
    maximum: 1.0
    description: "Quality threshold for processing"
    
  options:
    type: array
    items:
      type: string
    default: ["option1", "option2"]
    description: "Processing options"
```

#### 2. Step Definitions

**Enhanced Step Configuration**:
```yaml
steps:
  - id: enhanced_step
    action: process_data
    timeout: 120          # New: explicit timeout
    retry:                # New: retry configuration
      attempts: 3
      delay: 5
      backoff_factor: 1.5
    on_error: "continue"  # New: error handling
    cache_key: "step_{{ input_hash }}"  # New: caching
    parameters:
      data: "{{ input_data }}"
```

## API Changes

### Function Signature Updates

#### 1. Orchestrator Initialization

**Old API (v1.x)**:
```python
orchestrator = Orchestrator(
    model_registry=my_registry,
    tool_registry=my_tools,
    template_manager=my_templates
)
```

**New API (v2.x)**:
```python
# Simplified - uses global registries
orchestrator = Orchestrator()

# Or with configuration
orchestrator = Orchestrator(
    config={
        "timeout_default": 300,
        "max_parallel_default": 5,
        "enable_caching": True
    }
)
```

#### 2. Pipeline Execution

**Enhanced Execution Options**:
```python
# New execution parameters
result = await orchestrator.run_pipeline(
    pipeline_path="my_pipeline.yaml",
    parameters={"input": "data.json"},
    execution_config={
        "enable_debug": True,
        "checkpoint_interval": 10,
        "max_execution_time": 3600
    }
)
```

## Automated Migration Tools

### Migration Scripts

We provide scripts to help automate the migration:

#### 1. Pipeline Migration Script

```bash
# Migrate pipeline files to v2.x format
python scripts/migrate_pipeline.py \
  --input my_old_pipeline.yaml \
  --output my_new_pipeline.yaml \
  --version 2.x
```

#### 2. Code Migration Script

```bash
# Update Python code imports and patterns
python scripts/migrate_code.py \
  --directory src/ \
  --from-version 1.x \
  --to-version 2.x \
  --backup
```

### Validation Tools

#### 1. Compatibility Checker

```bash
# Check pipeline compatibility with v2.x
python scripts/check_compatibility.py my_pipeline.yaml
```

Example output:
```
Compatibility Check Results:
✅ Model references: Compatible
⚠️  Template syntax: 2 warnings found
❌ Tool usage: 1 breaking change found

Warnings:
- Line 45: Consider using default filter for {{ optional_step.result }}

Breaking Changes:
- Line 23: Tool 'old_tool_name' not available, use 'new_tool_name' instead

Migration Suggestions:
- Update template on line 45: {{ optional_step.result | default('N/A') }}
- Replace 'old_tool_name' with 'new_tool_name' on line 23
```

#### 2. Pipeline Validator

```bash
# Validate migrated pipeline
python scripts/validate_pipeline.py my_new_pipeline.yaml --strict
```

## Step-by-Step Migration Process

### Phase 1: Pre-Migration Assessment

1. **Inventory Current Pipelines**
   ```bash
   find . -name "*.yaml" -path "*/examples/*" > pipeline_inventory.txt
   ```

2. **Run Compatibility Checks**
   ```bash
   for pipeline in $(cat pipeline_inventory.txt); do
     python scripts/check_compatibility.py "$pipeline"
   done > compatibility_report.txt
   ```

3. **Identify Custom Components**
   ```bash
   grep -r "class.*Tool\|ModelRegistry\|TemplateManager" src/ > custom_components.txt
   ```

### Phase 2: Code Migration

1. **Update Import Statements**
   ```bash
   # Automated import updates
   python scripts/migrate_imports.py --directory src/ --version 2.x
   ```

2. **Update Model Registry Usage**
   ```bash
   # Convert ModelRegistry() to get_model_registry()
   sed -i 's/ModelRegistry()/get_model_registry()/g' src/**/*.py
   ```

3. **Update Tool Registration**
   ```python
   # Manual updates required for tool registration patterns
   ```

### Phase 3: Pipeline Migration

1. **Backup Existing Pipelines**
   ```bash
   cp -r examples/ examples_v1_backup/
   ```

2. **Migrate Pipeline Configurations**
   ```bash
   for pipeline in examples/*.yaml; do
     python scripts/migrate_pipeline.py --input "$pipeline" --output "$pipeline.v2" --version 2.x
     mv "$pipeline.v2" "$pipeline"
   done
   ```

3. **Update Template Syntax**
   - Replace unsafe variable access with default filters
   - Add dependency declarations for conditional steps
   - Update loop variable references

### Phase 4: Testing and Validation

1. **Run Automated Tests**
   ```bash
   pytest tests/ -v --migration-test
   ```

2. **Test Sample Pipelines**
   ```bash
   python scripts/run_pipeline.py examples/simple_pipeline.yaml --test-mode
   ```

3. **Performance Regression Testing**
   ```bash
   python scripts/performance_test.py --compare-with-baseline
   ```

### Phase 5: Deployment

1. **Update Dependencies**
   ```bash
   pip install orchestrator>=2.0.0
   ```

2. **Update Configuration Files**
   - Model configuration files
   - Environment configuration
   - Deployment scripts

3. **Monitor Initial Deployments**
   - Enable debug logging initially
   - Monitor error rates
   - Validate output quality

## Common Migration Issues

### Issue 1: Template Syntax Errors

**Problem**: Old template syntax no longer works
**Solution**: Use migration script and manual review

```bash
python scripts/fix_templates.py my_pipeline.yaml
```

### Issue 2: Model Registry Conflicts

**Problem**: Custom models not available after migration
**Solution**: Update registration to use global registry

```python
# Add to initialization code
from orchestrator.models import get_model_registry
registry = get_model_registry()
registry.register_all_custom_models()
```

### Issue 3: Tool Registration Failures

**Problem**: Tools not found after migration
**Solution**: Update tool registration patterns

```python
from orchestrator.tools import get_tool_registry
registry = get_tool_registry()
# Re-register all custom tools
```

## Rollback Procedures

If migration issues occur, follow these rollback steps:

### 1. Immediate Rollback

```bash
# Restore from backup
cp -r examples_v1_backup/ examples/
git checkout HEAD~1 -- src/
pip install orchestrator==1.5.x
```

### 2. Partial Rollback

```bash
# Rollback specific components
git checkout HEAD~1 -- src/orchestrator/models/
pip install orchestrator==1.5.x --force-reinstall
```

## Post-Migration Checklist

- [ ] All pipelines validate successfully
- [ ] Custom models are registered and available
- [ ] Custom tools work correctly
- [ ] Template rendering produces expected output
- [ ] Performance meets or exceeds baseline
- [ ] Error handling works as expected
- [ ] Tests pass with new version
- [ ] Documentation is updated
- [ ] Team is trained on new features

## Support and Resources

### Documentation
- [Best Practices Guide](best-practices.md) - Updated practices for v2.x
- [Troubleshooting Guide](troubleshooting.md) - Common issues and solutions
- [API Reference](/docs/api_reference.md) - Complete v2.x API documentation

### Migration Support
- **Migration Scripts**: Available in `scripts/migration/`
- **Validation Tools**: Use `scripts/validate_pipeline.py` 
- **Compatibility Checker**: Use `scripts/check_compatibility.py`

### Getting Help
- **GitHub Issues**: Report migration problems
- **Documentation**: Check example pipelines for patterns
- **Community**: Join discussions about migration experiences

## Version Compatibility Matrix

| Feature | v1.x | v1.5.x | v2.x | Notes |
|---------|------|--------|------|-------|
| Model Registry | Manual | Manual | Singleton | Breaking change |
| Template Rendering | Basic | Enhanced | UnifiedResolver | Backward compatible |
| Output Sanitization | Manual | Manual | Automatic | New feature |
| Loop Variables | Limited | Enhanced | Full Support | Enhanced |
| Tool Registration | Manual | Manual | Global Registry | Breaking change |
| Error Handling | Basic | Enhanced | Advanced | Enhanced |
| Caching | None | Basic | Advanced | New feature |

## Summary

Migration to v2.x provides significant improvements in:
- **Reliability**: Better error handling and template resolution
- **Performance**: Unified registries and caching
- **Usability**: Automatic output sanitization and enhanced debugging
- **Maintainability**: Cleaner APIs and better separation of concerns

The migration process is designed to be as smooth as possible with automated tools and comprehensive validation. Most changes are backward compatible, with clear migration paths for breaking changes.

For additional support during migration, refer to the troubleshooting guide and example pipelines that demonstrate v2.x best practices.