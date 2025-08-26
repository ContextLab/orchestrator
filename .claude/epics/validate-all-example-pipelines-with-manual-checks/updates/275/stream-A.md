# Stream A: Core Template Resolution Engine - Issue #275

**Stream Focus**: Fix the fundamental UnifiedTemplateResolver in `src/orchestrator/core/unified_template_resolver.py`

**Key Objectives**:
- Fix UnifiedTemplateResolver.resolve() method to properly handle nested contexts
- Ensure loop variables ($item, $index, etc.) are properly injected
- Fix template resolution order and context inheritance
- Add proper error handling and debugging support

## Analysis of Current System

### Current State (2025-08-26)

**Existing Components**:
1. `UnifiedTemplateResolver` class exists and has sophisticated structure
2. `TemplateManager` with comprehensive Jinja2 integration 
3. `GlobalLoopContextManager` for loop variable management
4. Integration points in `orchestrator.py` and `tools/base.py`

**Core Issues Identified**:

1. **Template Resolution Timing**: The system attempts to resolve templates but critical variables are missing from context
2. **Loop Variable Integration**: Loop variables from `GlobalLoopContextManager` not consistently available in template context
3. **Method Missing**: `resolve_before_tool_execution()` is called but may not be working properly
4. **Context Assembly**: The `collect_context()` method exists but context may not be complete

### Root Cause Analysis

From examining the code, the key issues are:

1. **Missing resolve_before_tool_execution() implementation**: The method exists but may not be assembling context correctly
2. **Loop context not synchronized**: `loop_context_manager` is initialized but may not be properly synchronized with `template_manager`
3. **Incomplete context propagation**: Context collection happens but may not include all necessary variables

## Implementation Plan

### Phase 1: Fix Core resolve() Method ✅ (In Progress)

**Current Status**: The `resolve_templates()` method exists but needs enhancement

**Key Changes Needed**:
1. Enhance context assembly in `collect_context()`
2. Fix `resolve_before_tool_execution()` method
3. Ensure proper loop variable injection
4. Add validation for unresolved templates

### Phase 2: Loop Context Integration 

**Target**: Ensure all loop variables are available in template context

**Key Areas**:
1. Synchronize `loop_context_manager` with `template_manager`
2. Verify loop variables are included in `collect_context()`
3. Test with nested loop scenarios

### Phase 3: Error Handling & Validation

**Target**: Catch unresolved templates before execution

**Key Areas**:
1. Add template validation before tool execution
2. Clear error messages for template failures
3. Debug logging for template resolution process

## Current Progress

### ✅ Completed
- [x] System analysis and architecture understanding
- [x] Identification of core issues
- [x] Progress tracking setup

### 🔄 In Progress  
- [x] Analyzing UnifiedTemplateResolver implementation
- [ ] Fixing core resolve methods

### ⏳ Pending
- [ ] Loop variable injection fixes
- [ ] Error handling improvements
- [ ] Integration testing
- [ ] Documentation updates

## Key Files Being Modified

1. `/Users/jmanning/orchestrator/src/orchestrator/core/unified_template_resolver.py` - Main fixes
2. `/Users/jmanning/orchestrator/src/orchestrator/core/template_manager.py` - Context integration
3. `/Users/jmanning/orchestrator/src/orchestrator/core/loop_context.py` - Loop variable management

## Interface Contracts for Other Streams

**For Stream B (Loop Context)**:
- `UnifiedTemplateResolver.collect_context()` will include loop variables
- `resolve_before_tool_execution()` will have access to loop context
- Loop variables will be available as `$item`, `$index`, `$is_first`, `$is_last`

**For Stream C (Tool Integration)**:
- `resolve_before_tool_execution()` will return fully resolved parameters
- No `{{variables}}` will be passed to tools
- All tool parameters will be resolved before tool execution

**For Stream D (Testing)**:
- `validate_templates()` method for template validation
- `get_debug_info()` method for debugging
- Clear error messages for template resolution failures

## Next Actions

1. Fix the `collect_context()` method to properly assemble context
2. Enhance `resolve_before_tool_execution()` to ensure complete resolution
3. Add validation to catch unresolved templates
4. Test with a simple failing pipeline to verify fixes

## Critical Success Criteria

1. ✅ **No unresolved templates**: Zero `{{variable}}` strings in tool parameters
2. ⏳ **Loop variables accessible**: `$item`, `$index` available in all contexts  
3. ⏳ **Complete context**: All pipeline, step, and loop variables available
4. ⏳ **Error clarity**: Clear messages when templates can't be resolved