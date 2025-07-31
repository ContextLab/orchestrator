# Template Rendering Fix Documentation

## Issue Summary

When using conditional steps in pipelines, template rendering can fail when trying to access results from steps that were skipped due to conditional execution. This causes raw templates like `{% if enhance_text.result %}{{ enhance_text.result }}{% else %}{{ input_text }}{% endif %}` to be passed to LLMs instead of rendered values.

## Root Cause

The issue occurs because:
1. Conditional steps (with `if:` clauses) may be skipped during execution
2. Skipped steps are not registered in the template context
3. Jinja2 templates that reference these skipped steps fail with UndefinedError
4. The template rendering falls back to returning the original template string

## Fix Applied

Modified `src/orchestrator/orchestrator.py` to register skipped tasks in the template manager:

```python
# Skip tasks that are already marked as skipped
if task.status == TaskStatus.SKIPPED:
    results[task_id] = {"status": "skipped"}
    # Register skipped task with None value in template manager
    # This allows Jinja conditionals like {% if enhance_text.result %} to work
    self.template_manager.register_context(task_id, None)
    continue
```

Also modified `src/orchestrator/core/template_manager.py` to return `None` for undefined attributes instead of raising errors:

```python
def __getattr__(self, name):
    if name == 'result':
        return self._value if hasattr(self, '_value') else str(self)
    # For other attributes, return None instead of undefined
    # This allows Jinja2 conditionals to work properly
    return None
```

## Best Practices for Pipeline Authors

### 1. Use Safe Conditional Checks

Instead of:
```yaml
{% if enhance_text.result %}{{ enhance_text.result }}{% else %}{{ input_text }}{% endif %}
```

Consider using:
```yaml
{% if enhance_text is defined and enhance_text and enhance_text.result %}{{ enhance_text.result }}{% else %}{{ input_text }}{% endif %}
```

Or with the default filter:
```yaml
{{ (enhance_text.result) | default(input_text) }}
```

### 2. Design Dependencies Carefully

When a step depends on a conditional step, ensure it handles the case where the conditional step was skipped:

```yaml
steps:
  - id: optional_enhancement
    if: "{{ some_condition }}"
    action: enhance_text
    
  - id: use_enhancement
    action: process
    parameters:
      # Safe access with fallback
      text: "{{ optional_enhancement.result | default(original_text) }}"
    dependencies:
      - optional_enhancement  # Will wait for skip decision
```

### 3. Use Explicit Fallback Steps

For complex conditional flows, consider using explicit fallback steps:

```yaml
steps:
  - id: try_enhance
    if: "{{ quality < threshold }}"
    action: enhance_text
    
  - id: prepare_text
    action: generate_text
    parameters:
      prompt: "Using {% if try_enhance.result %}enhanced{% else %}original{% endif %} text"
    dependencies:
      - try_enhance  # Ensures try_enhance decision is made first
```

## Testing

The fix has been tested with:
- `research_advanced_tools.yaml` - Complex research pipeline with multiple template uses
- `control_flow_advanced.yaml` - Pipeline with conditional steps and loops
- Simple template rendering scenarios

All pipelines now render templates correctly, even when referencing skipped conditional steps.

## Future Improvements

1. **Better Error Messages**: Provide clearer error messages when template rendering fails due to undefined variables
2. **Template Validation**: Add validation during pipeline compilation to detect potential undefined variable access
3. **Documentation**: Update pipeline examples to show best practices for conditional step references
4. **Simplified Syntax**: Consider adding custom Jinja2 functions or filters to make safe access easier