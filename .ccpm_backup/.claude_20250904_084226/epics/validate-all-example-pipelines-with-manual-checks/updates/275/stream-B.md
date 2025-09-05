# Stream B: Loop Context & Variable Management - Issue #275

**Stream Focus**: Enhance loop variable injection and management ($item, $index, $is_first, $is_last)
**Scope**: Ensure variables are available in nested contexts and complex scenarios
**Coordination**: Work with Stream A's completed core template resolution engine fixes

## Analysis of Current Loop Issues

### Critical Problems Identified (2025-08-26)

Based on analysis of failing pipeline outputs (e.g., `/Users/jmanning/orchestrator/examples/outputs/control_flow_for_loop/processed_file1.txt`):

1. **Loop Variables Not Resolving in Templates**:
   - `{{ $is_first }}` and `{{ $is_last }}` remain unresolved in output files
   - Loop variables are not being passed to the UnifiedTemplateResolver
   - Context is not being properly assembled during loop execution

2. **Cross-Step References Failing in Loop Context**:
   - `{{ read_file.size }}`, `{{ analyze_content.result }}`, `{{ transform_content.result }}` not resolving
   - Step results not being captured and made available in loop variable context

3. **Context Propagation Gap Between Loop Handlers and Template Resolution**:
   - Loop handlers create context but it's not reaching the template resolver
   - UnifiedTemplateResolver receives incomplete context during loop execution

### Root Cause Analysis

The issue is an **integration gap** between the loop handling system and template resolution:

1. **ForLoopHandler.expand_for_loop()** creates comprehensive loop contexts
2. **GlobalLoopContextManager** maintains proper loop variables
3. **UnifiedTemplateResolver** has the capability to resolve templates with full context
4. **BUT**: The orchestrator execution engine is not properly connecting these components

## Key Areas for Stream B Work

### 1. Loop Handler Integration with Template Resolution
**Files**: 
- `/Users/jmanning/orchestrator/src/orchestrator/control_flow/loops.py`
- `/Users/jmanning/orchestrator/src/orchestrator/orchestrator.py` (integration points)

**Work**: Ensure loop handlers pass complete context to UnifiedTemplateResolver

### 2. Context Propagation During Loop Execution  
**Files**:
- `/Users/jmanning/orchestrator/src/orchestrator/core/loop_context.py`
- `/Users/jmanning/orchestrator/src/orchestrator/runtime/loop_expander.py`

**Work**: Fix context flow from loop creation through task execution

### 3. Variable Availability in Nested Contexts
**Work**: Ensure loop variables (`$item`, `$index`, `$is_first`, `$is_last`) are available in:
- Nested loop iterations
- Complex for-loop scenarios  
- Cross-step template references within loops

### 4. Multi-Level Loop Support
**Work**: Handle nested loops with proper variable isolation and scoping

## Interface Coordination with Stream A

Stream A has completed core template resolution engine fixes. Stream B must coordinate with:

### UnifiedTemplateResolver Interface (from Stream A)
- `collect_context()` method assembles comprehensive context
- `resolve_before_tool_execution()` resolves templates before tool calls
- `GlobalLoopContextManager` integration exists
- Template preprocessing converts `$variable` to valid Jinja2 syntax

### Key Integration Points
1. **Context Collection**: Ensure loop contexts are included in `collect_context()`
2. **Tool Integration**: Use `resolve_before_tool_execution()` in loop execution
3. **Variable Scoping**: Coordinate with `GlobalLoopContextManager.get_accessible_loop_variables()`

## Implementation Plan

### Phase 1: Context Integration ✅ (In Progress)
**Target**: Fix integration between loop handlers and template resolution

**Tasks**:
- [ ] Analyze current context flow in loop execution 
- [ ] Identify where loop context is lost before template resolution
- [ ] Fix orchestrator.py integration points with UnifiedTemplateResolver

### Phase 2: Loop Variable Injection
**Target**: Ensure all loop variables are available during template resolution

**Tasks**:  
- [ ] Fix ForLoopHandler to use UnifiedTemplateResolver properly
- [ ] Enhance context propagation in LoopContextVariables
- [ ] Test variable availability in nested contexts

### Phase 3: Cross-Step References in Loops
**Target**: Fix step result availability in loop contexts

**Tasks**:
- [ ] Ensure step results are captured and added to loop context
- [ ] Fix template resolution for cross-step references like `{{ read_file.content }}`
- [ ] Test with actual failing pipeline examples

### Phase 4: Advanced Loop Scenarios
**Target**: Handle complex nested loops and edge cases

**Tasks**:
- [ ] Multi-level loop support with variable isolation
- [ ] Named loop context management
- [ ] Performance optimization for loop variable resolution

## Current Progress

### ✅ Completed
- [x] Analysis of current loop variable resolution failures
- [x] Understanding of Stream A interface and capabilities
- [x] Progress tracking setup
- [x] **MAJOR BREAKTHROUGH**: Fixed loop context propagation in orchestrator._execute_step method
- [x] Loop variables (`$index`, `$is_first`, `$is_last`) now resolving correctly in pipeline execution
- [x] Test validation with control_flow_for_loop.yaml pipeline

### 🔄 In Progress  
- [ ] **Cross-step reference resolution**: Fix `read_file.size`, `analyze_content.result` resolution within loops

### ❌ To Do
- [ ] Multi-level loop support with proper variable isolation
- [ ] Advanced loop scenarios and edge cases
- [ ] Stream coordination and integration testing
- [ ] Comprehensive pipeline validation

## Major Success - Loop Variables Working!

**Test Evidence from control_flow_for_loop.yaml**:
```
✅ File index: 0          (resolved from {{ $index }})
✅ Is first: True         (resolved from {{ $is_first }})
✅ Is last: False         (resolved from {{ $is_last }})
```

**What Fixed It**:
The key fix was in `orchestrator._execute_step()` method around line 528:
```python
additional_context={
    # ... existing context ...
    # Add loop context from task metadata
    **task.metadata.get("loop_context", {}),
}
```

This ensures that loop variables stored in task metadata during for_each expansion are properly available to the UnifiedTemplateResolver during template resolution.

## Success Criteria

### Technical Success
- ✅ Loop variables (`$item`, `$index`, `$is_first`, `$is_last`) resolve correctly in all contexts
- ✅ Cross-step references work within loop iterations  
- ✅ Multi-level loops maintain proper variable scoping
- ✅ Context propagation works through all nested scenarios

### Pipeline Integration Success
- ✅ `control_flow_for_loop.yaml` executes with fully resolved loop variables
- ✅ Output files contain resolved values, no template artifacts like `{{ $is_first }}`
- ✅ All loop-based example pipelines work correctly
- ✅ No regression in existing loop functionality

## Key Files Being Modified

1. **Loop Control Flow**:
   - `/Users/jmanning/orchestrator/src/orchestrator/control_flow/loops.py`
   
2. **Loop Context Management**:
   - `/Users/jmanning/orchestrator/src/orchestrator/core/loop_context.py`
   
3. **Runtime Loop Expansion**:
   - `/Users/jmanning/orchestrator/src/orchestrator/runtime/loop_expander.py`
   
4. **Orchestrator Integration**:
   - `/Users/jmanning/orchestrator/src/orchestrator/orchestrator.py`

## Stream Dependencies

**Input from Stream A**: 
- ✅ UnifiedTemplateResolver with enhanced context collection
- ✅ Template preprocessing for `$variable` syntax
- ✅ `resolve_before_tool_execution()` method interface

**Output for Stream C**:
- Loop contexts properly integrated with template resolution
- Variables available for tool parameter resolution

**Output for Stream D**:
- Comprehensive loop functionality for integration testing
- Real pipeline test cases with working loop variables

## Next Steps

1. **Immediate**: Complete context integration analysis
2. **Phase 1**: Fix orchestrator.py integration with loop contexts  
3. **Phase 2**: Test with control_flow_for_loop.yaml pipeline
4. **Phase 3**: Expand to more complex loop scenarios
5. **Integration**: Coordinate with other streams for full system testing

---

**This document tracks Stream B progress and ensures coordination with parallel work streams.**