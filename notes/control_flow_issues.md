# Control Flow Pipeline Issues and Solutions

## Issues Found in control_flow_conditional.yaml

### 1. Incorrect Conditional Syntax
**Issue**: Pipeline used `if:` instead of `condition:` for conditional execution
```yaml
# Wrong:
if: "{{ read_file.size > size_threshold }}"

# Correct:
condition: "{{ read_file.size > size_threshold }}"
```

### 2. Unsupported Ternary Operator
**Issue**: Jinja2 doesn't support JavaScript-style ternary operator `? :`
```yaml
# Wrong:
{{ read_file.size > size_threshold ? 'Compressed' : 'Expanded' }}

# Correct:
{% if read_file.size > size_threshold %}Compressed{% else %}Expanded{% endif %}
```

### 3. Condition Evaluation Timing
**Issue**: Conditions are evaluated at compile time before step results are available
- The condition `{{ read_file.size > size_threshold }}` tries to access `read_file` before the step has executed
- This causes: `Failed to evaluate condition: name 'read_file' is not defined`

**Root Cause**: Similar to the template rendering issue - conditions need JIT evaluation

### 4. Template Rendering in Outputs
**Issue**: Even when steps work, final output templates may not render if dependencies failed
- `{{ compress_large.result | default(expand_small.result) }}` becomes `{{result}}` 
- This happens because both steps were skipped, so neither result exists

## Solutions Applied

1. **Fixed conditional syntax**: Changed `if:` to `condition:`
2. **Fixed ternary operator**: Replaced with proper Jinja2 if/else syntax
3. **Condition timing issue**: Still needs JIT condition evaluation (like template rendering)

## Recommendations for Other Control Flow Pipelines

1. **Check conditional syntax**: Ensure all use `condition:` not `if:`
2. **Verify Jinja2 compatibility**: No ternary operators, use proper if/else blocks
3. **Test condition dependencies**: Conditions that reference step results need those steps as dependencies
4. **Handle missing results**: Use `default()` filter or check if variables are defined

## Next Steps

The condition evaluation timing issue needs to be fixed in the control flow compiler, similar to how we fixed template rendering:
- Defer condition evaluation until runtime
- Evaluate conditions only when dependencies are satisfied
- Track condition dependencies like template dependencies

## Update: Current Status

### What's Working
- ✅ Pipeline executes without syntax errors
- ✅ Ternary operator replaced with proper Jinja2 syntax
- ✅ Output file shows correct values for templates that don't depend on conditional steps
- ✅ The file size (150 bytes) and processing type (Expanded) render correctly

### What's Not Working
- ❌ Conditions are evaluated at orchestrator execution time, not when dependencies are ready
- ❌ `read_file` is not in context when conditions are evaluated
- ❌ Both conditional steps (compress_large, expand_small) are skipped with "condition_error"
- ❌ Final template `{{result}}` can't render because neither conditional step executed

### Root Cause Analysis

The condition evaluation happens in `orchestrator.py` at line 464:
```python
should_execute = await task.should_execute(
    context, 
    previous_results,
    resolver
)
```

At this point, `previous_results` contains the step results, but the template manager in the context doesn't have these results registered. The condition template rendering in `conditional.py` tries to use the template manager, but it doesn't have access to the step results.

### Proposed Solution

1. **Register step results with template manager before condition evaluation**
   - In `orchestrator.py`, before calling `should_execute`, ensure all previous_results are registered with the template manager
   - This will make step results available for condition template rendering

2. **Alternative: Pass previous_results to template manager during condition rendering**
   - Modify the condition evaluation to pass previous_results as additional_context to the template render method

### Solution Implemented

After extensive debugging, we found multiple issues:

1. **Conditions were being processed during YAML compilation**
   - Fixed by ensuring conditions are skipped in `_process_templates` method
   - Added check for `current_key == "condition"` to skip processing

2. **Conditions stored in metadata were being removed from task_def**
   - Fixed by popping condition keys after storing in metadata

3. **Template manager didn't have step results during condition evaluation**
   - Fixed by passing `step_results` as additional_context to template_manager.render()
   - This ensures step results are available when rendering condition templates

4. **For loops and while loops need similar handling**
   - Conditions in loops also need to be evaluated at runtime, not compile time
   - Same principles apply: skip during compilation, render at runtime with context