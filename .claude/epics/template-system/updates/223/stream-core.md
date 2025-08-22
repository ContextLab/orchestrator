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