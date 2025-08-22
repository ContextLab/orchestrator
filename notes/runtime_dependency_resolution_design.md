# Runtime Dependency Resolution System for Orchestrator

## Problem Statement

The current orchestrator attempts to expand loops at compile time, which is fundamentally flawed because:

1. **While loops**: Number of iterations depends on runtime conditions
2. **For-each loops**: Iterator can be computed from previous step results (e.g., `for_each: "{{ extract_sources.result }}"`)
3. **Dynamic loops**: AUTO tags generate iteration lists at runtime
4. **Template dependencies**: Steps within loops may reference results from tasks that haven't run yet

This leads to template rendering failures where variables are undefined because their source tasks haven't executed yet.

## Proposed Solution: Runtime Dependency Resolution Engine

### Core Architecture

```
┌─────────────────────────────────────────┐
│         Pipeline Execution State        │
├─────────────────────────────────────────┤
│  • Global Variables Registry            │
│  • Task Results Store                   │
│  • Template Context Manager             │
│  • Dependency Graph                     │
│  • Resolution Queue                     │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│    Dependency Resolution Engine         │
├─────────────────────────────────────────┤
│  1. Parse unrendered content            │
│  2. Extract dependencies                │
│  3. Check dependency satisfaction       │
│  4. Render resolvable items            │
│  5. Update global state                 │
│  6. Repeat until convergence           │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│         Task/Loop Executor              │
├─────────────────────────────────────────┤
│  • Execute tasks with resolved params   │
│  • Expand loops at runtime              │
│  • Update execution state               │
└─────────────────────────────────────────┘
```

### Key Components

#### 1. Pipeline Execution State
A centralized state object attached to the pipeline that maintains:

```python
class PipelineExecutionState:
    def __init__(self):
        # All variables and results available globally
        self.global_context = {
            'variables': {},      # User-defined and system variables
            'results': {},         # Task execution results
            'loop_contexts': {},   # Active loop iteration contexts
            'templates': {},       # Resolved template values
        }
        
        # Dependency tracking
        self.dependency_graph = DependencyGraph()
        self.unresolved_items = []  # Items waiting for dependencies
        self.resolution_queue = []   # Items ready to resolve
        
    def register_result(self, task_id: str, result: Any):
        """Register a task result and trigger dependency resolution."""
        self.global_context['results'][task_id] = result
        self._trigger_resolution_check()
        
    def get_available_context(self) -> Dict[str, Any]:
        """Get all available context for template rendering."""
        return {
            **self.global_context['variables'],
            **self.global_context['results'],
            **self._get_current_loop_context(),
        }
```

#### 2. Dependency Resolution Engine
Iteratively resolves dependencies as data becomes available:

```python
class DependencyResolver:
    def __init__(self, execution_state: PipelineExecutionState):
        self.state = execution_state
        self.max_iterations = 100  # Prevent infinite loops
        
    def resolve_item(self, item: UnresolvedItem) -> ResolvedItem:
        """Attempt to resolve a single item with unrendered content."""
        # Parse template to extract dependencies
        dependencies = self.extract_dependencies(item.content)
        
        # Check if all dependencies are satisfied
        available_context = self.state.get_available_context()
        missing_deps = [d for d in dependencies if d not in available_context]
        
        if not missing_deps:
            # All dependencies satisfied - render the item
            rendered = self.render_template(item.content, available_context)
            return ResolvedItem(item.id, rendered, success=True)
        else:
            # Dependencies not satisfied - return unresolved
            return ResolvedItem(item.id, item.content, success=False, 
                              missing=missing_deps)
    
    def resolve_all_pending(self) -> ResolutionResult:
        """Iteratively resolve all pending items until convergence."""
        iterations = 0
        progress_made = True
        
        while progress_made and iterations < self.max_iterations:
            progress_made = False
            unresolved = self.state.unresolved_items.copy()
            
            for item in unresolved:
                result = self.resolve_item(item)
                if result.success:
                    # Item resolved - update state and remove from unresolved
                    self.state.register_resolved(result)
                    self.state.unresolved_items.remove(item)
                    progress_made = True
            
            iterations += 1
        
        # Check for unresolvable items
        if self.state.unresolved_items:
            return ResolutionResult(
                success=False,
                unresolved=self.state.unresolved_items,
                reason="Circular or missing dependencies"
            )
        
        return ResolutionResult(success=True)
```

#### 3. Loop Runtime Expansion
Loops are expanded at runtime when their dependencies are satisfied:

```python
class LoopExpander:
    def __init__(self, resolver: DependencyResolver):
        self.resolver = resolver
        
    def can_expand(self, loop_task: LoopTask) -> bool:
        """Check if a loop can be expanded (all dependencies satisfied)."""
        if loop_task.type == "for_each":
            # Check if iterator expression can be resolved
            iterator_result = self.resolver.resolve_item(
                UnresolvedItem("iterator", loop_task.for_each_expr)
            )
            return iterator_result.success
        elif loop_task.type == "while":
            # Check if condition can be evaluated
            condition_result = self.resolver.resolve_item(
                UnresolvedItem("condition", loop_task.while_condition)
            )
            return condition_result.success
        return False
    
    def expand_loop(self, loop_task: LoopTask) -> List[Task]:
        """Expand a loop into concrete tasks at runtime."""
        expanded_tasks = []
        
        if loop_task.type == "for_each":
            # Resolve iterator to get items
            items = self.resolver.resolve_item(
                UnresolvedItem("iterator", loop_task.for_each_expr)
            ).rendered_value
            
            # Create tasks for each item
            for idx, item in enumerate(items):
                # Set loop context for this iteration
                loop_context = {
                    'item': item,
                    'index': idx,
                    'is_first': idx == 0,
                    'is_last': idx == len(items) - 1
                }
                
                # Create tasks for this iteration
                for step_template in loop_task.step_templates:
                    task = self.create_task_from_template(
                        step_template, 
                        loop_context,
                        iteration=idx
                    )
                    expanded_tasks.append(task)
                    
        elif loop_task.type == "while":
            # While loops expand incrementally
            iteration = 0
            while self.evaluate_condition(loop_task.while_condition):
                loop_context = {'iteration': iteration}
                
                for step_template in loop_task.step_templates:
                    task = self.create_task_from_template(
                        step_template,
                        loop_context,
                        iteration=iteration
                    )
                    expanded_tasks.append(task)
                
                # Execute tasks and update state before next iteration
                self.execute_and_update(expanded_tasks[-len(loop_task.step_templates):])
                iteration += 1
                
                if iteration > loop_task.max_iterations:
                    break
                    
        return expanded_tasks
```

### Execution Flow

```python
class EnhancedOrchestrator:
    def execute_pipeline(self, pipeline: Pipeline) -> ExecutionResult:
        """Execute pipeline with runtime dependency resolution."""
        execution_state = PipelineExecutionState()
        resolver = DependencyResolver(execution_state)
        expander = LoopExpander(resolver)
        
        # Initialize with pipeline parameters
        execution_state.global_context['variables'] = pipeline.parameters
        
        # Get initial task queue
        task_queue = self.get_initial_tasks(pipeline)
        
        while task_queue:
            # Phase 1: Resolve all pending dependencies
            resolution_result = resolver.resolve_all_pending()
            if not resolution_result.success:
                raise DependencyError(f"Cannot resolve: {resolution_result.unresolved}")
            
            # Phase 2: Process ready tasks
            ready_tasks = []
            for task in task_queue:
                if isinstance(task, LoopTask):
                    # Check if loop can be expanded
                    if expander.can_expand(task):
                        expanded = expander.expand_loop(task)
                        ready_tasks.extend(expanded)
                        task_queue.remove(task)
                else:
                    # Check if regular task is ready
                    if self.task_is_ready(task, execution_state):
                        ready_tasks.append(task)
                        task_queue.remove(task)
            
            # Phase 3: Execute ready tasks
            for task in ready_tasks:
                # Final resolution of task parameters
                resolved_params = resolver.resolve_parameters(
                    task.parameters,
                    execution_state.get_available_context()
                )
                
                # Execute task
                result = self.execute_task(task, resolved_params)
                
                # Update execution state
                execution_state.register_result(task.id, result)
                
                # Check for new tasks to add to queue
                new_tasks = self.get_dependent_tasks(task.id, pipeline)
                task_queue.extend(new_tasks)
        
        return ExecutionResult(success=True, results=execution_state.global_context['results'])
```

### Benefits of This Approach

1. **True Runtime Resolution**: Loops expand when their dependencies are available, not at compile time
2. **Progressive Resolution**: Items are resolved as soon as their dependencies are satisfied
3. **Flexible Dependencies**: Any task can depend on any previous result, including loop iterations
4. **Clear Error Messages**: System knows exactly what dependencies are missing
5. **Incremental Execution**: While loops can check conditions after each iteration
6. **Context Propagation**: Loop contexts properly cascade to nested structures

### Implementation Phases

#### Phase 1: Core Infrastructure (Week 1)
- [ ] Implement PipelineExecutionState class
- [ ] Create DependencyResolver with basic template parsing
- [ ] Add dependency extraction from templates
- [ ] Build resolution iteration logic

#### Phase 2: Loop Support (Week 2)
- [ ] Implement LoopExpander for for_each loops
- [ ] Add while loop incremental expansion
- [ ] Handle nested loop contexts
- [ ] Test with existing pipelines

#### Phase 3: Integration (Week 3)
- [ ] Integrate with existing orchestrator
- [ ] Migrate current template rendering to new system
- [ ] Update control flow compiler
- [ ] Comprehensive testing

#### Phase 4: Optimization (Week 4)
- [ ] Add caching for resolved templates
- [ ] Optimize dependency checking
- [ ] Parallel execution where possible
- [ ] Performance benchmarking

### Example: How control_flow_advanced.yaml Would Work

1. **Initial State**: Pipeline starts with parameters `{input_text: "Test", languages: ["es"]}`

2. **Task Queue**: `[analyze_text, check_quality, enhance_text, select_text, translate_text_loop]`

3. **First Resolution Pass**:
   - `analyze_text` has no dependencies → Execute
   - Register result: `{analyze_text: "Analysis of Test..."}`

4. **Second Resolution Pass**:
   - `check_quality` depends on `analyze_text` (now available) → Execute
   - Register result: `{check_quality: "improve"}`

5. **Third Resolution Pass**:
   - `enhance_text` depends on `check_quality` (now available) → Execute
   - Register result: `{enhance_text: "Enhanced version..."}`

6. **Fourth Resolution Pass**:
   - `select_text` depends on `enhance_text` (now available) → Execute
   - Register result: `{select_text: "Final text for translation..."}`

7. **Fifth Resolution Pass**:
   - `translate_text_loop` depends on `select_text` (now available)
   - Iterator `languages` resolves to `["es"]`
   - Loop expands to: `[translate_text_0_translate, translate_text_0_validate, translate_text_0_save]`

8. **Sixth Resolution Pass**:
   - `translate_text_0_translate` template `{% if select_text %}{{ select_text }}{% else %}...`
   - `select_text` is available in context → Renders successfully
   - Execute translation with rendered prompt

9. **Continue** until all tasks complete

### Error Handling Examples

#### Circular Dependency
```yaml
- id: task_a
  action: generate_text
  parameters:
    prompt: "Process {{ task_b.result }}"

- id: task_b  
  action: generate_text
  parameters:
    prompt: "Process {{ task_a.result }}"
```

**Detection**: After max_iterations, both tasks remain unresolved
**Error**: "Circular dependency detected: task_a → task_b → task_a"

#### Missing Dependency
```yaml
- id: process
  action: generate_text
  parameters:
    prompt: "Process {{ nonexistent_task.result }}"
```

**Detection**: `nonexistent_task` never appears in results
**Error**: "Cannot resolve: process depends on undefined task 'nonexistent_task'"

### Migration Path

1. **Compatibility Mode**: New system runs alongside old system
2. **Gradual Migration**: Convert pipelines one at a time
3. **Validation**: Ensure results match between old and new systems
4. **Deprecation**: Phase out old system over 2-3 releases

## Conclusion

This runtime dependency resolution system provides a robust foundation for handling complex pipeline orchestration with dynamic loops, conditional execution, and flexible dependencies. It solves the fundamental issue of compile-time expansion while providing clear visibility into dependency chains and resolution progress.