# Implementation Plan: Fix For-Each Loop Dependency Handling (Issue #159)

## Executive Summary

The core issue is that `for_each` loops with dependencies are not properly waiting for those dependencies to complete before executing. This causes template rendering failures when loop tasks try to reference results from tasks they depend on, but which haven't run yet.

## Problem Analysis

### Current Behavior
1. **YAML Definition**: `translate_text` for_each loop has `dependencies: [select_text]`
2. **Compile Time**: Loop is expanded into individual tasks (`translate_text_0_translate`, etc.)
3. **Execution Time**: Expanded tasks run at level 0, BEFORE `select_text` completes
4. **Result**: Templates fail because `select_text` is undefined

### Root Cause
The dependency inheritance during for_each expansion is incorrect. Currently (orchestrator.py lines 1652-1662):
- Only the FIRST task in FIRST iteration gets parent dependencies (when idx==0)
- OR all tasks get them if max_parallel > 1
- This logic is flawed and doesn't ensure all expanded tasks wait for dependencies

## Affected Components

### Primary Files to Modify
1. **`src/orchestrator/orchestrator.py`** - Fix dependency inheritance in `_expand_for_each_task()`
2. **`src/orchestrator/compiler/control_flow_compiler.py`** - Ensure ForEachTask preserves dependencies
3. **`src/orchestrator/core/pipeline.py`** - Verify execution level calculation handles expanded tasks

### Secondary Files (Already partially fixed)
1. **`src/orchestrator/orchestrator.py`** - Template context registration (lines 1314-1366)
2. **`src/orchestrator/control_systems/hybrid_control_system.py`** - Result registration (lines 222-272)

## Implementation Details

### Fix 1: Correct Dependency Inheritance for All Expanded Tasks

**File**: `src/orchestrator/orchestrator.py`
**Method**: `_expand_for_each_task()` (lines 1640-1696)

**Current Code** (lines 1650-1663):
```python
# Handle dependencies
task_deps = []

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

**Proposed Fix**:
```python
# Handle dependencies
task_deps = []

# ALL tasks from for_each expansion must inherit parent dependencies
# This ensures they wait for required tasks to complete first
task_deps.extend(for_each_task.dependencies)

# For sequential execution, also add dependency on previous iteration
if for_each_task.max_parallel == 1 and idx > 0:
    # Sequential execution - also depend on previous iteration
    prev_task_id = f"{for_each_task.id}_{idx-1}_{step_def['id']}"
    task_deps.append(prev_task_id)

# Add internal dependencies within the loop body
for dep in step_def.get("dependencies", []):
    if dep in [s["id"] for s in for_each_task.loop_steps]:
        # Internal dependency - reference the task from same iteration
        task_deps.append(f"{for_each_task.id}_{idx}_{dep}")
    else:
        # External dependency - use as-is
        task_deps.append(dep)
```

### Fix 2: Ensure ForEachTask Preserves All Dependencies

**File**: `src/orchestrator/compiler/control_flow_compiler.py`
**Method**: `_create_for_each_task()` (lines around where ForEachTask is created)

**Verification Needed**:
```python
# Ensure dependencies are properly passed to ForEachTask
for_each_task = ForEachTask(
    id=task_id,
    name=loop_def.get("name", f"For each: {task_id}"),
    action="for_each_runtime",
    parameters={},
    dependencies=loop_def.get("dependencies", []),  # <-- Ensure this captures all deps
    # ... rest of parameters
)

# Log for debugging
logger.info(f"Created ForEachTask '{task_id}' with dependencies: {for_each_task.dependencies}")
```

### Fix 3: Add Comprehensive Logging

**File**: `src/orchestrator/orchestrator.py`
**Location**: Multiple points in execution flow

Add logging to trace execution order issues:
```python
# In _execute_level() method
self.logger.info(f"Executing level {level} with tasks: {ready_tasks}")
for task_id in ready_tasks:
    task = pipeline.get_task(task_id)
    self.logger.info(f"  Task '{task_id}' deps: {task.dependencies}, status: {task.status}")

# In _expand_for_each_task() method
self.logger.info(f"Expanding ForEachTask '{for_each_task.id}' with dependencies: {for_each_task.dependencies}")
for task in expanded_tasks:
    self.logger.info(f"  Created task '{task.id}' with deps: {task.dependencies}")
```

## Edge Cases to Handle

### 1. Nested For-Each Loops
```yaml
- id: outer_loop
  for_each: "{{ list1 }}"
  steps:
    - id: inner_loop
      for_each: "{{ list2 }}"
      steps:
        - id: process
          action: generate_text
  dependencies:
    - prerequisite_task
```
**Solution**: Each level of nesting must properly inherit dependencies from its parent.

### 2. Mixed Sequential and Parallel Execution
```yaml
- id: loop1
  for_each: "{{ items }}"
  max_parallel: 1  # Sequential
  dependencies: [task_a]

- id: loop2
  for_each: "{{ items }}"
  max_parallel: 5  # Parallel
  dependencies: [loop1]
```
**Solution**: Sequential loops add inter-iteration dependencies IN ADDITION to parent dependencies.

### 3. Conditional Tasks Before Loops
```yaml
- id: conditional_task
  if: "{{ condition }}"
  action: generate_text

- id: loop_task
  for_each: "{{ items }}"
  dependencies: [conditional_task]
```
**Solution**: Even if conditional_task is skipped, the dependency should be honored (loop waits).

### 4. ForEachTask with AUTO Tags (Runtime Expansion)
```yaml
- id: dynamic_loop
  for_each: "<AUTO>generate list of items</AUTO>"
  dependencies: [data_source]
```
**Solution**: ForEachTask itself must wait for dependencies before expansion.

### 5. Multiple Dependencies Including Loops
```yaml
- id: task_a
  action: generate_text

- id: loop_b
  for_each: "{{ items }}"
  
- id: task_c
  dependencies: [task_a, loop_b]
```
**Solution**: task_c waits for BOTH task_a AND all tasks from loop_b expansion.

## Testing Strategy

### Unit Tests
1. **Test Dependency Inheritance**: Verify all expanded tasks get parent dependencies
2. **Test Execution Levels**: Ensure expanded tasks are placed at correct execution level
3. **Test Sequential vs Parallel**: Verify different max_parallel values work correctly

### Integration Tests
1. **control_flow_advanced.yaml**: Should produce properly rendered translation files
2. **fact_checker.yaml**: Loops should wait for extract_sources_list and extract_claims_list
3. **control_flow_for_loop.yaml**: Should wait for create_output_dir before processing files

### Test Commands
```bash
# Test the main issue case
python scripts/run_pipeline.py examples/control_flow_advanced.yaml \
  -i input_text="Test fix for Issue 159" \
  -i languages='["es", "fr"]' \
  -o examples/outputs/issue_159_test

# Verify translation files
cat examples/outputs/issue_159_test/translations/test-fix-for-issue-159_es.txt
# Should NOT contain {{ translate }} or {% if select_text %}

# Test other affected pipelines
python scripts/run_pipeline.py examples/fact_checker.yaml \
  -i document_path="examples/data/test_article.md"

python scripts/run_pipeline.py examples/control_flow_for_loop.yaml \
  -i file_list='["test1.txt", "test2.txt"]'
```

## Implementation Steps

### Phase 1: Core Fix (Day 1)
1. ✅ Backup current code
2. Implement Fix 1 (dependency inheritance)
3. Add comprehensive logging
4. Test with control_flow_advanced.yaml

### Phase 2: Validation (Day 1-2)
1. Run all example pipelines with for_each loops
2. Verify no regressions
3. Check execution order logs
4. Validate template rendering

### Phase 3: Edge Cases (Day 2)
1. Test nested loops
2. Test conditional dependencies
3. Test ForEachTask with AUTO tags
4. Document any limitations

### Phase 4: Documentation (Day 2)
1. Update code comments
2. Add docstring examples
3. Update pipeline examples if needed
4. Close Issue #159 with summary

## Success Criteria

### Primary Goal
✅ Translation files in control_flow_advanced.yaml contain actual content, not template placeholders

### Secondary Goals
✅ All for_each loops properly wait for their dependencies
✅ No regression in other pipelines
✅ Clear logging shows correct execution order
✅ Edge cases are handled gracefully

## Risk Mitigation

### Risk 1: Breaking Existing Pipelines
**Mitigation**: Run full test suite before committing changes

### Risk 2: Performance Impact
**Mitigation**: Dependencies only affect execution order, not parallelization within levels

### Risk 3: Circular Dependencies
**Mitigation**: Existing circular dependency detection should catch any issues

## Alternative Solutions (If Primary Fix Fails)

### Option A: Delay Loop Expansion
Instead of expanding at compile time, expand for_each loops at runtime AFTER dependencies complete. This would be a larger architectural change but would guarantee correct context.

### Option B: Two-Phase Execution
1. Execute all non-loop tasks first
2. Then expand and execute all loops
This is simpler but less flexible.

### Option C: Explicit Dependency Declaration
Require users to explicitly declare dependencies for each step within for_each:
```yaml
- id: translate
  action: generate_text
  dependencies: [select_text]  # Explicit dependency
```
This puts the burden on the user but gives precise control.

## Conclusion

The primary fix (correcting dependency inheritance) is the most straightforward solution that maintains backward compatibility while fixing the issue. It requires minimal code changes and addresses the root cause directly.

The implementation should take 1-2 days including testing and documentation.