# Stream Core: Template Resolution System (#223)

## Status: Starting Analysis
**Date**: 2025-08-22
**Branch**: epic/template-system

## Tasks
- [ ] Analyze current template resolution implementation
- [ ] Create unified template resolution layer
- [ ] Ensure templates are resolved BEFORE passing to tools
- [ ] Properly expose structured data to template engine
- [ ] Make it work consistently across all components

## Progress Log

### 2025-08-22 - Initial Analysis
- Read requirements from epic documentation
- Analyzed current template resolution implementation in:
  - ✅ src/orchestrator/core/template_manager.py (extensive template system)
  - ✅ src/orchestrator/core/context_manager.py (unified context management)
  - ✅ src/orchestrator/compiler/template_renderer.py (simple template renderer)
  - ✅ src/orchestrator/orchestrator.py (shows how templates are used)
  - ✅ src/orchestrator/tools/base.py (tool template resolution)
  - ✅ src/orchestrator/control_flow/loops.py (loop template handling)

## Current State Analysis

### Identified Issues:
1. **Multiple Template Systems**: We have 3 different template systems:
   - `TemplateManager` (most sophisticated, Jinja2-based)
   - `TemplateRenderer` (simple regex-based)
   - Basic string replacement in various places

2. **Inconsistent Template Resolution**: 
   - Tools use `TemplateManager.deep_render()` in `Tool.execute()`
   - Loop handlers have their own `_process_templates_with_named_loops()`
   - Orchestrator has manual template rendering in `_execute_task()`

3. **Context Fragmentation**:
   - `ContextManager` manages hierarchical context but isn't fully integrated
   - Loop variables handled separately in different places
   - Template context registration happens in multiple places

4. **Template Resolution Timing**:
   - Some templates resolved in orchestrator before tool execution
   - Some templates resolved in tool base class
   - Loop templates resolved in loop handlers
   - Inconsistent timing leads to missing variables

## Implementation

### ✅ Created Unified Template Resolution System
- **UnifiedTemplateResolver**: Centralized template resolution layer
- **TemplateResolutionContext**: Comprehensive context collection from all sources
- **Integration**: Seamlessly integrates with existing TemplateManager and ContextManager

### ✅ Key Features Implemented:
1. **Centralized Context Collection**: Gathers variables from:
   - Pipeline inputs and parameters
   - Step results from previous tasks
   - Loop variables (with proper Jinja2 compatibility)
   - Tool parameters
   - Additional context

2. **Template Resolution Before Tool Execution**: 
   - Templates resolved in orchestrator BEFORE passing to tools
   - Tools receive fully resolved parameters
   - No more missing variables during tool execution

3. **Loop Variable Integration**:
   - Fixed $iteration, $item, $index variable availability
   - Proper loop context sharing between loop handlers and template resolver
   - Both $variable and variable syntax supported for compatibility

4. **Comprehensive Error Handling**:
   - Graceful fallback to legacy template manager
   - Detailed debugging information
   - Maintains backward compatibility

### ✅ Testing:
- 11 comprehensive test cases covering all aspects
- Loop variable resolution
- Nested data structures  
- Tool parameter resolution
- Context hierarchy
- Error handling
- All tests passing

### ✅ Integration Points:
- **Orchestrator**: Updated to use unified resolver for task execution
- **Tool Base Class**: Enhanced to support both unified and legacy resolvers
- **Filesystem Tool**: Updated with runtime template rendering support
- **Loop Handlers**: Share same loop context manager for consistency