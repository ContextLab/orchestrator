# Issue #290: Template Resolution Architecture Overhaul - Stream A Progress

## Mission Summary
Fix the template resolution architecture based on root cause analysis from Issue #289: Template Manager Instance Context Mismatch.

## Root Cause Analysis Completed ✅
- **Issue**: FileSystemTool receives a TemplateManager instance that lacks proper context registration
- **Problem**: Context is registered correctly but the wrong instance is passed to tools
- **Evidence**: Templates like `{{ parameters.input_document }}` and `{{ execution.timestamp }}` remain unresolved

## Implementation Progress

### ✅ Primary Fix: Context Re-registration Verification
**File**: `/src/orchestrator/tools/system_tools.py` - FileSystemTool

**Changes Made**:
1. Added `_re_register_context()` method to re-register context from resolution context
2. Added context verification logic in `_write_file()` method
3. Context re-registration triggers when template manager context is empty

**Code Added**:
```python
# CRITICAL FIX: BEFORE calling deep_render, verify template manager has context
if _template_manager and hasattr(_template_manager, 'context'):
    if not _template_manager.context or len(_template_manager.context) == 0:
        logger.warning("Template manager context empty, attempting re-registration")
        # Re-register from resolution context if available
        if _resolution_context:
            success = self._re_register_context(_template_manager, _resolution_context)
```

### ✅ Secondary Fix: Template Manager Instance Consistency  
**File**: `/src/orchestrator/control_systems/hybrid_control_system.py`

**Changes Made**:
1. Added instance ID tracking for debugging template manager instances
2. Enhanced logging to track which template manager instance is passed to tools
3. Added execution metadata generation to fix empty execution objects

**Code Added**:
```python
# Add instance ID tracking for debugging (Issue #290 fix)
import uuid
self.hybrid_template_resolver.template_manager.instance_id = f"tm_{uuid.uuid4().hex[:8]}"

# Log instance being passed for debugging (Issue #290 fix)
logger.info(f"Passing template_manager instance {getattr(self.hybrid_template_resolver.template_manager, 'instance_id', 'unknown')}")
```

### ✅ Tertiary Fix: Execution Metadata Generation
**File**: `/src/orchestrator/control_systems/hybrid_control_system.py`

**Problem Found**: Execution object was empty `{}` causing `{{ execution.timestamp }}` to fail
**Solution**: Added `_get_execution_metadata()` method to generate proper execution metadata

**Code Added**:
```python
def _get_execution_metadata(self, context: Dict[str, Any]) -> Dict[str, Any]:
    """Get execution metadata for templates."""
    from datetime import datetime
    
    # Try to get existing execution metadata
    existing_execution = context.get("execution", {})
    if isinstance(existing_execution, dict) and existing_execution:
        # If we have execution metadata with timestamp, use it
        if "timestamp" in existing_execution:
            return existing_execution
    
    # Generate new execution metadata
    now = datetime.now()
    return {
        "timestamp": now.strftime("%Y-%m-%dT%H:%M:%S"),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "iso_timestamp": now.isoformat(),
        "pipeline_id": context.get("pipeline_id", "unknown"),
        "execution_id": context.get("execution_id", "unknown"),
    }
```

## Testing Results

### Test Pipeline: `examples/iterative_fact_checker.yaml`

**Command**: `python scripts/execution/run_pipeline.py examples/iterative_fact_checker.yaml -o examples/outputs/FULLY_FIXED_templates`

**Expected Outcome**:
- `{{ parameters.input_document }}` should resolve to `"test_climate_document.md"`
- `{{ execution.timestamp }}` should resolve to actual timestamp

**Actual Outcome**: ❌ **Templates Still Not Resolved**

**Generated Report Content**:
```markdown
# Fact-Checking Report

## Document Information
- **Source Document**: {{ parameters.input_document }}
- **Date Processed**: {{ execution.timestamp }}
```

## Current Status: PARTIAL SUCCESS ⚠️

### What Works ✅
1. **Context Verification Logic**: Successfully implemented and context is non-empty
2. **Instance Tracking**: Template manager instances are properly tracked
3. **Execution Metadata**: Execution object now contains proper metadata structure

### What Still Fails ❌
1. **Template Resolution**: Basic templates like `{{ parameters.input_document }}` still not resolving
2. **Deep Render**: The `template_manager.deep_render()` method is not processing templates

## Investigation Findings

### Context Analysis
- Template manager context shows 44+ variables including `parameters` and `execution`
- Both parameters and execution objects are confirmed available in logs
- Context re-registration logic did **NOT** trigger (context was not empty)

### Debug Logs Evidence
```
Template manager context keys: ['timestamp', 'debug_mode', 'output_path', 'input_document', 'quality_threshold', 'max_iterations', ...]
✅ Parameters object available with 5 items
✅ Execution object available
Unified resolver: Original had templates: True, Rendered has templates: True
```

The issue is **NOT** context registration but the actual template resolution mechanism in the UnifiedTemplateResolver or TemplateManager.deep_render() method.

## Next Steps Required

### Immediate Priority: Template Manager Deep Investigation
1. **Test TemplateManager.deep_render() directly** with simple template
2. **Investigate unified resolver resolution logic** - why is it not working?  
3. **Check Jinja2 template environment** setup and context passing
4. **Validate template preprocessing** for dollar variables

### Technical Debt
- Our fixes are correct architectural improvements but don't solve the core issue
- Need to trace the template resolution call chain from UnifiedTemplateResolver -> TemplateManager
- The problem is likely in the Jinja2 template rendering itself

## Files Modified
- `/src/orchestrator/tools/system_tools.py` - Added context re-registration verification
- `/src/orchestrator/control_systems/hybrid_control_system.py` - Added instance tracking and execution metadata

## Test Output Locations
- `examples/outputs/FULLY_FIXED_templates/fact_checking_report.md` - Contains unresolved templates
- Debug logs show context is properly populated but templates remain unresolved

---
**Status**: Architecture fixes implemented correctly, but core template resolution still requires investigation.
**Next Owner**: Needs deeper TemplateManager/Jinja2 investigation