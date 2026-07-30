# Claude Skills Refactor - Integration Points & Code References

## Overview
This document identifies specific integration points in the codebase that will be affected by the Claude Skills refactor, with direct file paths and class/function references.

---

## 1. Model Management Integration Points

### Location: `src/orchestrator/models/`

#### File: `model_registry.py`
**Current Multi-Provider Implementation** (lines 20-1730):
```python
class ModelRegistry:
    def __init__(self):
        self.models: Dict[str, Model] = {}
        self.model_selector: UCBModelSelector  # Multi-armed bandit
        self._langchain_adapters: Dict[str, LangChainModelAdapter]
        
    async def select_model(self, requirements: Dict) -> Model:
        # 1. Filter by capabilities (complex logic)
        # 2. Filter by health with caching
        # 3. Apply provider preference
        # 4. UCB algorithm selection
        # 5. Cache result
        return selected_model
```

**Refactor Impact**:
- Simplify `select_model()` - Remove UCB, filter, health check logic
- Reduce from 1,915 lines to ~500 lines
- New logic: Map task_type → Anthropic tier (Opus/Sonnet/Haiku)

#### File: `model_selector.py`
**Current UCB Selector** (lines 1731-1915):
```python
class UCBModelSelector:
    def select(self, model_keys: List[str], context: Dict) -> str:
        # Complex UCB algorithm with exploration/exploitation
        # Tracks attempts, successes, rewards
        pass
```

**Refactor Impact**:
- Replace entire UCB algorithm
- New: Simple tier-based selection
- Keep interface compatibility: `select(model_keys, context) -> str`

#### File: `integrations/`
**Current Provider Files**:
- `anthropic_model.py` - Keep ✓
- `openai_model.py` - Remove ✗
- `google_model.py` - Remove ✗
- `ollama_model.py` - Remove ✗
- `huggingface_model.py` - Remove ✗

**Refactor Impact**:
- Remove 4 provider integration files
- Keep only Anthropic implementation
- Update LangChain adapter to support Anthropic only

---

## 2. Tool/Skill Registry Integration Points

### Location: `src/orchestrator/tools/` and `src/orchestrator/skills/` (NEW)

#### File: `tools/registry.py`
**Current Enhanced Tool Registry** (lines 200-864):
```python
class EnhancedToolRegistry(UniversalToolRegistry):
    def register_tool_enhanced(self, tool, version_info, ...):
        # Versioning, compatibility, security, installation management
        pass
    
    def discover_tools_advanced(self, action_description=None, ...):
        # Complex tool discovery with discovery engine
        pass
```

**Refactor Impact**:
- Deprecate or transform this class
- Move core concepts to new SkillRegistry
- Remove complex capability matching

#### File: `tools/discovery.py`
**Current Tool Discovery Engine**:
```python
class ToolDiscoveryEngine:
    def discover_tools_for_action(self, action_description):
        # Match action description to tools
        pass
```

**Refactor Impact**:
- Replace with skill-based discovery
- New: `SkillDiscoveryEngine` in `skills/discovery.py`
- Simpler pattern matching on skill capabilities

#### File: `skills/` (NEW MODULE)

**New File: `skills/registry.py`**
```python
class SkillRegistry:
    def __init__(self, home_dir: Path):
        self.home_dir = home_dir / ".orchestrator" / "skills"
        self.registry_path = self.home_dir / "registry.yaml"
        self.skills: Dict[str, Skill] = {}
    
    def register_skill(self, skill_name: str, skill_def: dict):
        # Add skill to ~/.orchestrator/skills/registry.yaml
        pass
    
    def get_skill(self, skill_name: str) -> Skill:
        # Load skill definition from registry
        pass
    
    def discover_skills(self, capability: str) -> List[Skill]:
        # Find skills providing capability
        pass
```

**New File: `skills/creator.py`** (ROMA Pattern)
```python
class SkillCreator:
    async def create_skill(self, capability: str, context: dict) -> Skill:
        # Stage 1: Atomize - Determine complexity
        complexity = await self.atomize(capability, context)
        
        if complexity == 'simple':
            return await self.create_simple_skill(capability)
        
        # Stage 2: Plan - Decompose into subtasks
        subtasks = await self.plan(capability, context)
        
        # Stage 3: Execute - Create components
        components = []
        for subtask in subtasks:
            comp = await self.execute_subtask(subtask)
            components.append(comp)
        
        # Stage 4: Aggregate - Combine
        skill = await self.aggregate(components)
        
        # Test and review
        await self.review_and_test(skill)
        return skill
```

**New File: `skills/reviewer.py`**
```python
class SkillReviewer:
    async def review_skill(self, skill: Skill) -> ReviewResult:
        # Multi-agent review using Claude
        # Check correctness, efficiency, safety
        pass
    
    async def improve_skill(self, skill: Skill, review: ReviewResult) -> Skill:
        # Generate improvements based on review
        pass
```

---

## 3. Compiler Integration Points

### Location: `src/orchestrator/compiler/`

#### File: `yaml_compiler.py`
**Current Compilation Pipeline** (lines 156-279):
```python
async def compile(self, yaml_content: str, context=None):
    # Step 1: File processing
    # Step 2: YAML parsing
    # Step 3: Schema validation
    # Step 4: Dependency validation
    # Step 5: Error handler validation
    # ...
    # Step 13: Build pipeline
    return pipeline
```

**Refactor Integration Points**:

1. **After Step 4 (Dependency Validation)** - ADD:
```python
# New Step 4.5: Skill Validation
if self.validate_skills:
    await self._validate_skills(raw_pipeline)
```

2. **During Step 13 (Pipeline Building)** - ADD:
```python
# In _build_pipeline()
# Add auto-skill creation
for step in pipeline_def['steps']:
    if step.get('skill') not in registry:
        # Auto-create missing skill
        new_skill = await skill_creator.create_skill(
            step['skill'],
            context=raw_pipeline
        )
        registry.register_skill(step['skill'], new_skill)
```

3. **After Compilation** - ADD:
```python
# Generate help documentation
pipeline.help_text = self._generate_help(pipeline_yaml)

# Return pipeline with .help() method
pipeline.help = lambda: print(pipeline.help_text)
```

#### File: `enhanced_yaml_compiler.py`
**Loop Processing** (for `parallel` flag support):
```python
# In _process_control_flow()
if 'for' in step:
    for_config = step['for']
    
    # NEW: Check for parallel flag
    if for_config.get('parallel', False):
        # Mark for parallel execution
        step['metadata']['parallel_execution'] = True
```

---

## 4. State Management Integration Points

### Location: `src/orchestrator/state/`

#### File: `langgraph_state_manager.py`
**Current LangGraph Integration** (lines 1-400):
```python
class LangGraphStateManager:
    def __init__(self):
        self.checkpointer = ...
        self.config = ...
    
    def invoke(self, graph, state, config):
        # Execute LangGraph
        pass
```

**Refactor Integration**:
- New: Direct StateGraph compilation from Pipeline
- Current: Pipeline → Manual node/edge creation
- Future: Pipeline → StateGraph (direct)

```python
# New method in YAMLCompiler
def compile_to_langgraph(self, pipeline: Pipeline) -> StateGraph:
    """Convert Pipeline object directly to LangGraph StateGraph."""
    graph = StateGraph(state_schema)
    
    # Add nodes for each task
    for task in pipeline.tasks.values():
        graph.add_node(task.id, create_task_node(task))
    
    # Add edges from dependencies
    for task_id, task in pipeline.tasks.items():
        for dep in task.dependencies:
            graph.add_edge(dep, task_id)
    
    # Add parallelization edges
    for task in pipeline.tasks.values():
        if task.metadata.get('parallel_execution'):
            # Handle parallel execution metadata
            pass
    
    return graph.compile(checkpointer=...)
```

---

## 5. Execution Engine Integration Points

### Location: `src/orchestrator/engine/`

#### File: `control_flow_engine.py`
**Current Control Flow** (lines 1-500):
```python
class ControlFlowEngine:
    async def execute_step(self, step_def, state):
        # Handle for/while/if
        # Dynamic task generation
        pass
```

**Refactor Enhancement**:
- Leverage LangGraph's conditional_edge()
- Use LangGraph's StateGraph for graph construction
- Simplify loop handling with graph conditional edges

#### File: `task_executor.py`
**Task Execution** (lines 1-300):
```python
class TaskExecutor:
    async def execute(self, task: Task, state):
        # Invoke tool/skill
        # Handle retries
        # Track results
        pass
```

**Refactor Impact**:
- Skill execution replaces tool execution
- Interface remains same
- Internal: Call skill instead of tool

---

## 6. Validation System Integration Points

### Location: `src/orchestrator/validation/`

#### File: `tool_validator.py`
**Tool Validation** (lines 1-400):
```python
class ToolValidator:
    def validate_pipeline_tools(self, pipeline_def):
        # Check tool availability
        # Validate parameters
        pass
```

**Refactor Impact**:
- Add SkillValidator alongside (or replace)
- Reduce complexity: Check skill registry only
- Simplify parameter validation

#### File: `model_validator.py`
**Model Validation** (lines 1-300):
```python
class ModelValidator:
    def validate_pipeline_models(self, pipeline_def):
        # Check model availability
        pass
```

**Refactor Impact**:
- Significantly simplify: Only check for Anthropic models
- Remove provider-specific logic
- Reduce from ~300 lines to ~100 lines

---

## 7. Core Pipeline Changes

### Location: `src/orchestrator/core/`

#### File: `pipeline.py`
**Pipeline Class** (No major changes expected)
```python
@dataclass
class Pipeline:
    id: str
    name: str
    tasks: Dict[str, Task]
    context: Dict[str, Any]
    metadata: Dict[str, Any]
    
    # Add in refactor:
    help_text: str = ""  # Generated at compile time
```

**Refactor Impact**:
- Add `.help()` method
- Minor: Add help_text storage
- No structural changes

#### File: `task.py`
**Task Class** (Minor enhancement)
```python
@dataclass
class Task:
    # ... existing fields ...
    
    # Enhance in refactor:
    metadata: Dict[str, Any]
    # Add to metadata:
    # - parallel_execution: bool
    # - skill_version: str
    # - auto_created: bool (if skill was auto-created)
```

---

## 8. Test Infrastructure Changes

### Location: `tests/`

#### Remove/Deprecate Files
```
tests/models/test_openai_model.py          - REMOVE
tests/models/test_google_model.py          - REMOVE
tests/models/test_ollama_model.py          - REMOVE
tests/models/test_huggingface_model.py     - REMOVE
tests/models/test_model_selector_ucb.py    - REMOVE (replace with tier-based)
```

#### New Test Files
```
tests/skills/test_skill_creation.py        - NEW (ROMA pattern)
tests/skills/test_skill_registry.py        - NEW
tests/skills/test_skill_reviewer.py        - NEW
tests/skills/test_skill_tester_e2b.py      - NEW
```

#### Modify Existing Files
```
tests/models/test_model_registry.py        - Simplify to Anthropic-only
tests/integration/test_llm_apis.py         - Keep Anthropic, remove others
tests/pipeline_tests/test_control_flow.py  - Add parallel flag tests
tests/core/test_pipeline.py                - Add help() method test
```

---

## 9. Registry YAML Structure

### New File Structure
```
orchestrator/registry/
├── skills/
│   ├── default_registry.yaml
│   │   skills:
│   │     web_search:
│   │       version: "1.0.0"
│   │       capability: "web-search"
│   │       implementation: "default_skills/web_search.py"
│   │     code_executor:
│   │       version: "1.0.0"
│   │       capability: "code-execution"
│   │     data_processor:
│   │       version: "1.0.0"
│   │       capability: "data-processing"
│   │
│   └── default_skills/
│       ├── web_search.py
│       ├── code_executor.py
│       └── data_processor.py
│
└── models/
    └── default_registry.yaml
        models:
          opus:
            provider: "anthropic"
            model: "claude-opus-4.1"
            tier: "high"
          sonnet:
            provider: "anthropic"
            model: "claude-sonnet-4.5"
            tier: "medium"
          haiku:
            provider: "anthropic"
            model: "claude-haiku-4.5"
            tier: "low"
```

### User Directory Structure
```
~/.orchestrator/
├── skills/
│   ├── registry.yaml           # Copied from default
│   ├── [user-skill-1]/
│   │   └── definition.yaml
│   └── [user-skill-2]/
│       └── definition.yaml
│
└── models/
    └── registry.yaml           # Copied from default
```

---

## 10. Integration Checklist

### Phase 1: Registry Setup
- [ ] Create `src/orchestrator/skills/` module
- [ ] Implement `SkillRegistry` class
- [ ] Create default registry YAML files
- [ ] Implement `RegistryManager` for ~/.orchestrator
- [ ] Add installer script for initial setup

### Phase 2: Model Simplification
- [ ] Simplify `ModelRegistry.select_model()`
- [ ] Remove UCB selector complex logic
- [ ] Deprecate/remove non-Anthropic providers
- [ ] Update model validation tests
- [ ] Test Anthropic-only model selection

### Phase 3: Skill System
- [ ] Implement `SkillCreator` (ROMA pattern)
- [ ] Implement `SkillReviewer` with Claude
- [ ] Implement `SkillTester` with E2B sandbox
- [ ] Integrate skill creation into compiler
- [ ] Create skill creation tests

### Phase 4: Compiler Enhancement
- [ ] Add skill validation to compilation pipeline
- [ ] Integrate skill auto-creation
- [ ] Implement compile-time help generation
- [ ] Add `.help()` method to Pipeline
- [ ] Test direct YAML to LangGraph compilation

### Phase 5: Testing & Polish
- [ ] Update all integration tests
- [ ] Remove multi-provider tests
- [ ] Add E2B sandbox tests
- [ ] Performance benchmarking
- [ ] Documentation updates

---

## Key Code Changes Summary

### Lines to Add/Change
- `compiler/yaml_compiler.py`: +~500 lines (skill integration)
- `skills/` (new): ~1,500 lines (creator, reviewer, registry)
- `models/model_registry.py`: -~1,400 lines (simplification)
- `models/model_selector.py`: -~200 lines (UCB removal)
- `core/pipeline.py`: +~50 lines (help support)
- `state/langgraph_state_manager.py`: +~200 lines (direct compilation)

### Files to Remove
- `integrations/openai_model.py`: ~300 lines
- `integrations/google_model.py`: ~300 lines
- `integrations/ollama_model.py`: ~300 lines
- `integrations/huggingface_model.py`: ~400 lines

### Net Change
- **Estimated Addition**: +1,500-2,000 lines (new skills system)
- **Estimated Removal**: ~1,300 lines (provider removal)
- **Net Result**: +200-700 lines (net increase, focused on skills)

---

## Conclusion

This refactor consolidates around Claude Skills, replacing complex multi-provider logic with focused skill creation and management. The integration points are well-defined, allowing for systematic implementation in 5-6 phases.

The core pipeline abstractions remain stable while tooling layers are substantially simplified and enhanced.
