# JIT Template Rendering - Handoff Notes

## Current State
The JIT template rendering foundation is implemented and committed. The main components are:

1. **TemplateMetadata** class in `src/orchestrator/core/template_metadata.py` - tracks template dependencies
2. **Template analysis** in `YAMLCompiler._analyze_template()` - identifies which step IDs each template depends on
3. **Task enhancement** - Task class now has `template_metadata` field to store analysis results
4. **Control flow integration** - Fixed to preserve template metadata for conditional tasks

## What's Working
- Template dependency analysis correctly identifies dependencies
- Basic pipelines (like `research_basic.yaml`) work correctly
- The control system renders templates at runtime using the template manager

## Current Issue
The remaining issue is with conditional tasks that have both:
1. Dependencies on other tasks
2. A condition that uses results from those dependencies

Example from `research_advanced_tools.yaml`:
```yaml
- id: extract_content
  dependencies: [search_topic, deep_search]
  condition: "{{ (search_topic.results | length > 0) or (deep_search.results | length > 0) }}"
```

The problem: We can't evaluate the condition until dependencies complete, but the current code tries to evaluate it too early.

## Solution Approach
The solution is to ensure conditional tasks only evaluate their conditions AFTER their declared dependencies are satisfied. This requires modifying the execution flow in `orchestrator.py`.

## Debugging Commands
Test the implementation:
```bash
# Test basic pipeline (should work)
python scripts/run_pipeline.py examples/research_basic.yaml -i topic="test"

# Test advanced pipeline (currently fails on conditional task)
python scripts/run_pipeline.py examples/research_advanced_tools.yaml -i topic="test"
```

## Key Files to Review
1. `src/orchestrator/orchestrator.py` - Lines 483-503 where conditional task evaluation happens
2. `src/orchestrator/control_flow/conditional.py` - The ConditionalTask.should_execute() method
3. `src/orchestrator/core/control_system.py` - The _render_task_templates() method

## Next Steps
1. Ensure conditional tasks wait for dependencies before evaluating conditions
2. Test all example pipelines to ensure they work
3. Add unit tests for template analysis and rendering
4. Update documentation

The implementation is very close - just need to fix the timing of conditional task evaluation.
