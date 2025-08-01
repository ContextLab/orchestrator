# Template Rendering Debug Summary

## Issue Summary

The `research_advanced_tools.yaml` pipeline is not rendering templates in the final output files. Templates like `{{ topic }}`, `{{ analyze_findings.result }}` appear as raw text in the markdown files.

## Root Causes Identified

### 1. UnboundLocalError in template_manager.py (FIXED)

**Issue**: The error handler in `render()` method tried to access `context` variable that was only defined in the try block.

**Fix**: Moved context creation before the try block (commit 40ea065).

### 2. Jinja2 Syntax Error in Pipeline YAML

**Issue**: The pipeline contains invalid Jinja2 syntax that causes template compilation to fail:
```
{% for result in search_topic.results[:{{ max_results | int }}] %}
```

**Error**: `TemplateSyntaxError: expected token ':', got '}'`

**Problem**: You cannot nest template expressions inside Jinja2 control structures. The `{{ max_results | int }}` inside the for loop is invalid syntax.

**Impact**: When the template fails to compile, the `deep_render` method catches the exception and returns the original unrendered template, which is why templates appear raw in the output.

## How Template Rendering Works

1. **Control System** calls `_render_task_templates()` which creates/retrieves a TemplateManager
2. For filesystem writes, the content parameter is **intentionally preserved** unrendered
3. **FileSystemTool** receives the template_manager via `_template_manager` parameter
4. FileSystemTool calls `template_manager.deep_render(content)` to render at runtime
5. If rendering fails, the original content is written (current behavior)

## Current Status

- ✅ UnboundLocalError fixed
- ✅ Template manager is correctly passed to FileSystemTool
- ✅ FileSystemTool attempts to render templates
- ❌ Template rendering fails due to syntax error in YAML
- ❌ Error handling returns original unrendered content

## Verification

The checkpoint tool shows:
- All step results are available in context
- Templates are present in the content parameter
- No rendered_parameters field exists (templates not rendered)

## Next Steps

1. **Fix the Jinja2 syntax error** in `research_advanced_tools.yaml`:
   - Change `{% for result in search_topic.results[:{{ max_results | int }}] %}`
   - To: `{% for result in search_topic.results[:max_results] %}` or use slice filter

2. **Improve error handling** to make template errors more visible:
   - Consider failing the task when template rendering fails
   - Or at least log a prominent warning

3. **Add validation** during YAML compilation to catch template syntax errors early

## Test Results

When testing the exact template content in isolation with proper context, rendering works perfectly. This confirms the issue is specifically the invalid Jinja2 syntax in the pipeline YAML, not the rendering system itself.