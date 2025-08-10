# Template Dependency Migration Guide

## Overview

This guide addresses the systematic template rendering issues identified in Issue #153. The root cause has been identified as **missing dependency declarations** in pipeline steps, not issues with the template rendering system itself.

## The Problem

When pipeline steps use template variables that reference other step results (e.g., `{{ search_step.result }}`), but don't declare proper dependencies, the following occurs:

1. Steps execute in parallel by default
2. The filesystem tool executes before referenced steps complete
3. Template variables remain unrendered as `{{ step_id.result }}`
4. Users see literal template syntax in their output files

## The Solution

**Add explicit `dependencies:` declarations** to any step that uses template variables referencing other steps.

### Before (Broken)
```yaml
steps:
  - id: search_data
    tool: web-search
    action: search
    parameters:
      query: "{{ topic }}"
      
  - id: save_results  # ❌ MISSING dependencies
    tool: filesystem
    action: write
    parameters:
      path: "{{ output_path }}/results.md"
      content: |
        # Search Results
        {{ search_data.result }}  # Will show as {{ search_data.result }}
```

### After (Fixed)
```yaml
steps:
  - id: search_data
    tool: web-search
    action: search
    parameters:
      query: "{{ topic }}"
      
  - id: save_results
    tool: filesystem
    action: write
    dependencies:           # ✅ ADDED dependencies
      - search_data         # ✅ Ensures search_data completes first
    parameters:
      path: "{{ output_path }}/results.md"
      content: |
        # Search Results
        {{ search_data.result }}  # Will render actual search results
```

## Audit Results Summary

Our automated audit of 48 pipelines found:
- **19 pipelines** with dependency issues
- **48 total dependency issues** requiring fixes
- Most common pattern: filesystem operations missing dependencies on content generation steps

### Most Affected Pipelines
1. `research_advanced_tools.yaml` - 5 dependency issues
2. `research_basic.yaml` - 4 dependency issues  
3. `test_validation_pipeline.yaml` - 3 dependency issues

## Migration Steps

### Step 1: Run the Audit Tool
```bash
python scripts/audit_template_dependencies.py --verbose
```

This will identify all pipelines with missing dependencies and provide specific fix suggestions.

### Step 2: Apply Fixes Systematically

For each pipeline with issues, add `dependencies:` declarations following this pattern:

1. **Identify template variables**: Look for `{{ step_id.result }}` or `{{ step_id.data }}`
2. **Extract step IDs**: From `{{ search_results.result }}`, the step ID is `search_results`  
3. **Add dependencies**: Add each referenced step ID to the `dependencies:` list

### Step 3: Test Your Fixes

Run the pipeline and verify:
```bash
# Test the pipeline
python scripts/run_pipeline.py examples/your_pipeline.yaml -i topic="test subject"

# Verify no template variables remain unrendered
grep -r "{{" output_directory/  # Should return no results
```

## Common Patterns and Fixes

### Pattern 1: Research → Save Pattern
```yaml
# BROKEN: Missing dependency
- id: research
  action: generate_text
  parameters:
    prompt: "Research {{ topic }}"
    
- id: save_research
  tool: filesystem
  action: write
  parameters:
    content: "{{ research.result }}"  # ❌ Won't render

# FIXED: Add dependency
- id: save_research
  tool: filesystem
  action: write
  dependencies:
    - research  # ✅ Now research completes first
  parameters:
    content: "{{ research.result }}"  # ✅ Will render
```

### Pattern 2: Search → Analyze → Save Chain
```yaml
# BROKEN: Missing dependencies
- id: search
  tool: web-search
  action: search
  
- id: analyze
  action: analyze_text
  parameters:
    text: "{{ search.results }}"  # ❌ search might not be done
    
- id: final_report
  tool: filesystem
  action: write
  parameters:
    content: |
      Search: {{ search.results }}
      Analysis: {{ analyze.result }}  # ❌ analyze might not be done

# FIXED: Add proper dependency chain  
- id: analyze
  action: analyze_text
  dependencies:
    - search  # ✅ Wait for search
  parameters:
    text: "{{ search.results }}"
    
- id: final_report
  tool: filesystem
  action: write
  dependencies:
    - search   # ✅ Wait for search
    - analyze  # ✅ Wait for analysis  
  parameters:
    content: |
      Search: {{ search.results }}
      Analysis: {{ analyze.result }}
```

### Pattern 3: Multiple Dependencies
```yaml
# BROKEN: Multiple missing dependencies
- id: step1
  action: generate_text
  
- id: step2  
  tool: web-search
  
- id: step3
  action: analyze_text
  
- id: combine_all
  tool: filesystem
  action: write
  parameters:
    content: |
      Generated: {{ step1.result }}
      Search: {{ step2.results }}  
      Analysis: {{ step3.result }}

# FIXED: List all dependencies
- id: combine_all
  tool: filesystem  
  action: write
  dependencies:
    - step1  # ✅ All required steps
    - step2
    - step3
  parameters:
    content: |
      Generated: {{ step1.result }}
      Search: {{ step2.results }}
      Analysis: {{ step3.result }}
```

## What's NOT Broken

The template rendering system itself works perfectly for:

✅ **Pipeline parameters**: `{{ topic }}`, `{{ output_path }}` - Always work  
✅ **Execution metadata**: `{{ execution.timestamp }}` - Always work  
✅ **Step results WITH dependencies**: `{{ step.result }}` when `dependencies: [step]` is declared  
✅ **Complex templates**: Loops, conditionals, filters - All work correctly  

## Validation Testing

After fixing dependencies, your pipelines should:

1. **Render all templates**: No `{{` or `{%` should appear in output files
2. **Execute in correct order**: Steps with dependencies wait for predecessors  
3. **Produce expected content**: Real data appears instead of template placeholders
4. **Pass end-to-end tests**: Complete user workflows work correctly

## Need Help?

1. **Run the audit tool**: `python scripts/audit_template_dependencies.py --verbose`
2. **Check the logs**: Enable debug logging to see template rendering details
3. **Test incrementally**: Fix one pipeline at a time and test
4. **Validate thoroughly**: Run complete user workflows to ensure fixes work

## Impact Assessment

✅ **Low Risk Changes**: Adding `dependencies:` declarations is safe and backwards-compatible  
✅ **No Breaking Changes**: Existing functionality remains intact  
✅ **Immediate Benefits**: Template rendering works as users expect  
✅ **Future Proof**: Prevents similar issues in new pipelines  

The fixes are minimal, safe, and highly effective at resolving the template rendering issues users have been experiencing.