# Issue #159: Template Rendering in control_flow_advanced.yaml

## Problem Summary
Translation files in `control_flow_advanced.yaml` are being written with unrendered template placeholders instead of actual content.

### Example Output (Current - Broken)
```
# Translation to es

## Source Text Used
{% if select_text %}{{ select_text }}{% else %}Template rendering test{% endif %}

## Translated Text
{{ translate }}

## Quality Assessment
{{ validate_translation }}
```

## Root Cause Analysis

### Issue 1: Dependency Not Being Honored
The `translate_text` for_each loop has a dependency on `select_text` (line 129 in YAML), but the expanded tasks (`translate_text_0_translate`) are running at execution level 0, BEFORE `select_text` completes.

**Evidence from logs:**
- `ORCHESTRATOR: Executing level 0 with tasks: ['analyze_text', 'translate_text_0_translate']`
- `'select_text' is undefined` error during template rendering

This means the dependency chain is broken during for_each expansion.

### Issue 2: Template Context Not Available
Even when implementing fixes to register all previous results with the template manager, `select_text` is undefined because it literally hasn't run yet when the translate task executes.

## Technical Details

### For Each Expansion (orchestrator.py lines 1652-1662)
```python
# Add dependencies from the ForEachTask itself
if idx == 0:
    # First iteration depends on ForEachTask dependencies
    task_deps.extend(for_each_task.dependencies)
elif for_each_task.max_parallel == 1:
    # Sequential execution - depend on previous iteration
    prev_task_id = f"{for_each_task.id}_{idx-1}_{step_def['id']}"
    task_deps.append(prev_task_id)
else:
    # Parallel execution - depend on ForEachTask dependencies
    task_deps.extend(for_each_task.dependencies)
```

The code DOES add dependencies, but something in the execution order calculation is wrong.

## Fixes Attempted

### Fix 1: Enhanced Template Registration (PARTIAL SUCCESS)
Modified `orchestrator.py` and `hybrid_control_system.py` to:
- Register ALL previous results before template rendering
- Register results with multiple access patterns (direct, _str, _result)
- Register pipeline parameters and loop variables

**Result:** Templates for filesystem operations have more context, but still missing results from tasks that haven't run yet.

### Fix 2: Deep Template Rendering
The FileSystemTool already uses `deep_render` which should handle complex templates.

**Result:** Still fails because the underlying data (`select_text`) isn't available.

## The Real Problem

The issue is NOT primarily about template rendering - it's about task execution order. The for_each expanded tasks are not properly inheriting dependencies from their parent ForEachTask.

## Solution Needed

### Option 1: Fix Dependency Inheritance
Ensure that ALL tasks expanded from a for_each loop inherit the parent's dependencies, not just the first task or based on parallelization.

### Option 2: Fix Execution Level Calculation
The execution level calculation needs to properly account for for_each task dependencies so that expanded tasks run at the correct level.

### Option 3: Delay For Each Expansion
Instead of expanding for_each at compile time, expand them at runtime AFTER their dependencies complete. This would ensure the context is available.

## Next Steps

1. **Debug Execution Order**: Add logging to understand why `translate_text_0_translate` is being scheduled at level 0
2. **Fix Dependency Propagation**: Ensure all expanded tasks inherit parent dependencies
3. **Test Fix**: Run control_flow_advanced.yaml and verify translation files are properly rendered
4. **Regression Testing**: Ensure other for_each loops still work correctly

## Test Command
```bash
python scripts/run_pipeline.py examples/control_flow_advanced.yaml \
  -i input_text="Test template rendering" \
  -i languages='["es"]' \
  -o examples/outputs/test_159
```

## Success Criteria
The translation file should contain actual translated text, not template placeholders:
```
# Translation to es

## Source Text Used
[Actual enhanced text content]

## Translated Text
[Actual Spanish translation]

## Quality Assessment
[Actual quality assessment text]
```