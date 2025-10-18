# Technical Design Document: Claude Skills-Based Orchestrator Refactor (Version 2)

## Executive Summary

This document outlines a comprehensive refactor of the Orchestrator framework to leverage Anthropic's Claude Skills system, incorporating learnings from ROMA and AgnoAgents frameworks. The refactor will dramatically simplify the codebase by:
- Removing support for non-Anthropic models (temporarily)
- Replacing the complex tools registry with a skills-based system
- Implementing automatic skill creation with multi-agent review
- Compiling pipelines directly to LangGraph without LLM prompting
- Supporting advanced control flow (loops, conditionals, parallelization)
- Creating a comprehensive suite of example pipelines

## Integration with ROMA and AgnoAgents

### ROMA Integration Points
From [ROMA](https://github.com/sentient-agi/ROMA), we will adopt (with attribution):

1. **Four-Stage Processing Pattern** for skill creation:
   - **Atomizer**: Determine if skill needs decomposition
   - **Planner**: Break complex skills into subtasks
   - **Executors**: Implement atomic skill components
   - **Aggregator**: Combine components into final skill

2. **E2B Sandbox Integration** for secure code execution during testing:
   ```python
   # Adapted from ROMA's E2B integration
   class SecureSkillTester:
       """E2B sandbox integration for safe skill testing."""
       def test_in_sandbox(self, skill, test_data):
           # Secure isolated execution environment
           # With S3 mounting for artifacts
   ```

3. **Hierarchical State Management** for nested pipeline execution

### AgnoAgents Integration Points
From [AgnoAgents](https://github.com/agno-agi/agno), we will incorporate:

1. **Performance-Optimized Agent Instantiation**:
   - Target: <10ms agent spawn time
   - Minimal memory footprint per skill instance

2. **MCP (Model Context Protocol) Support**:
   ```python
   class MCPSkillAdapter:
       """Bridge between MCP servers and our skills system."""
       # Enable compatibility with existing MCP tools
   ```

3. **Privacy-First Architecture**:
   - All processing within user infrastructure
   - No external telemetry without explicit opt-in

## Latest Anthropic Models (2025)

### Model Selection Strategy
- **Claude Sonnet 4.5** (Released Sep 2025): Primary orchestrator model
  - World's best coding model with 1M token context window
  - Optimal for building complex agents and computer use
  - Default for skill creation and pipeline orchestration

- **Claude Opus 4.1** (Released Aug 2025): Review and analysis model
  - Most powerful Claude model for deep analysis
  - Used for comprehensive skill review and validation
  - Complex reasoning and instruction adherence

- **Claude Haiku 4.5** (Released Oct 2025): High-speed validation
  - 90% of Sonnet 4.5's performance at 1/3 the cost
  - Used for quick validation checks and simple tasks
  - Ideal for high-volume, cost-sensitive operations

## 1. Core Architecture

### 1.1 Simplified Component Structure

```
orchestrator/
├── registry/                   # Template registry files
│   ├── skills/
│   │   ├── default_registry.yaml
│   │   └── default_skills/    # Pre-built skills
│   └── models/
│       └── default_registry.yaml
├── core/
│   ├── __init__.py
│   ├── pipeline.py            # Pipeline definition with control flow
│   ├── compiler.py            # Direct YAML to LangGraph compilation
│   ├── executor.py            # LangGraph execution engine
│   └── state.py               # State management for pipelines
├── models/
│   ├── __init__.py
│   └── anthropic.py           # Anthropic-only model registry
├── skills/
│   ├── __init__.py
│   ├── registry.py            # Skills registry management
│   ├── creator.py             # Skill creation workflow (ROMA-inspired)
│   ├── reviewer.py            # Skill review and validation
│   └── installer.py           # Handles ~/.orchestrator setup
├── control_flow/
│   ├── __init__.py
│   ├── loops.py               # For and while loop implementations
│   ├── conditionals.py        # If/elif/else routing
│   └── parallelization.py     # Parallel execution management
└── examples/
    ├── pipelines/             # Example pipeline definitions
    └── outputs/               # Pipeline execution outputs

# User home directory structure (created on install)
~/.orchestrator/
├── skills/
│   ├── registry.yaml          # User's skills registry
│   └── [skill-name]/          # Individual skill directories
└── models/
    └── registry.yaml          # User's model registry
```

### 1.2 Registry Management System

```python
class RegistryManager:
    """Manages skills and model registries in user home directory."""

    def __init__(self):
        self.home_dir = Path.home() / ".orchestrator"
        self.skills_dir = self.home_dir / "skills"
        self.models_dir = self.home_dir / "models"

    def install_defaults(self):
        """Copy default registries from package to user home."""
        # Copy orchestrator/registry/skills/default_registry.yaml
        # to ~/.orchestrator/skills/registry.yaml

        # Copy orchestrator/registry/models/default_registry.yaml
        # to ~/.orchestrator/models/registry.yaml

        # Copy default skills from orchestrator/registry/skills/default_skills/
        # to ~/.orchestrator/skills/

    def register_skill(self, skill_name: str, skill_data: dict):
        """Add new skill to user's registry."""
        registry_path = self.skills_dir / "registry.yaml"
        skill_path = self.skills_dir / skill_name

        # Create skill directory
        skill_path.mkdir(exist_ok=True)

        # Update registry
        with open(registry_path, 'r+') as f:
            registry = yaml.safe_load(f)
            registry['skills'][skill_name] = skill_data
            yaml.dump(registry, f)

    def update_model_registry(self, model_id: str, model_data: dict):
        """Future support for adding new models."""
        registry_path = self.models_dir / "registry.yaml"
        # Update registry with new model support
```

## 2. Advanced Pipeline Control Flow

### 2.1 Loop Support

```yaml
# For loop example with parallelization
steps:
  - id: process_files
    skill: file-processor
    for:
      items: "{{ file_list }}"
      variable: current_file
      parallel: true  # Process files in parallel
    parameters:
      file: "{{ current_file }}"
      operation: "analyze"

  # Nested loops example
  - id: matrix_operation
    for:
      items: "{{ rows }}"
      variable: row
    steps:
      - id: process_cell
        for:
          items: "{{ columns }}"
          variable: col
        skill: cell-processor
        parameters:
          row: "{{ row }}"
          col: "{{ col }}"

# While loop example
steps:
  - id: retry_until_success
    while:
      condition: "{{ not last_result.success }}"
      max_iterations: 5
    steps:
      - id: attempt_operation
        skill: api-caller
        parameters:
          endpoint: "{{ api_url }}"
      - id: check_result
        skill: validator
        state:
          last_result: "{{ attempt_operation.output }}"
```

### 2.2 Conditionals

```yaml
# If/elif/else routing
steps:
  - id: analyze_data
    skill: data-analyzer
    parameters:
      data: "{{ input_data }}"

  - id: route_based_on_analysis
    if:
      condition: "{{ analyze_data.complexity == 'simple' }}"
      then:
        - id: simple_process
          skill: simple-processor
    elif:
      - condition: "{{ analyze_data.complexity == 'medium' }}"
        then:
          - id: medium_process
            skill: medium-processor
    else:
      - id: complex_process
        skill: complex-processor

  # Conditional execution
  - id: optional_validation
    skill: validator
    condition: "{{ config.enable_validation == true }}"
    parameters:
      data: "{{ previous_step.output }}"
```

### 2.3 State Management

```python
class PipelineState:
    """State management for pipeline execution."""

    def __init__(self):
        self.variables = {}      # User-defined variables
        self.step_outputs = {}   # Outputs from each step
        self.temp_products = {}  # Temporary products
        self.loop_iterators = {} # Loop state tracking
        self.file_references = {} # File paths for artifacts

    def set_variable(self, name: str, value: Any):
        """Set state variable accessible to all steps."""
        self.variables[name] = value

    def get_variable(self, name: str, default=None):
        """Get state variable with template resolution."""
        # Support nested access: {{ step.output.field }}
        return self._resolve_template(name, default)

    def attach_to_node(self, node_id: str, data: dict):
        """Attach data to specific LangGraph node."""
        # State-specific data attached to nodes
        self.step_outputs[node_id] = data

    def save_artifact(self, step_id: str, content: Any, filename: str):
        """Save intermediate product to filesystem."""
        output_dir = Path(f"~/.orchestrator/pipelines/{self.pipeline_id}/artifacts")
        file_path = output_dir / f"{step_id}_{filename}"
        # Save content and store reference
        self.file_references[f"{step_id}.{filename}"] = file_path
```

## 3. Compile-Time Verification

### 3.1 Pipeline Compiler with Verification

```python
class PipelineCompiler:
    """Compiles and verifies YAML pipelines."""

    def compile(self, pipeline_yaml: dict) -> 'Pipeline':
        """
        Compile with comprehensive verification.

        Verification steps:
        1. Syntax validation
        2. Variable reference checking
        3. Model existence verification
        4. Skill availability check
        5. Create missing skills if needed
        """

        # Step 1: Validate syntax
        self._validate_syntax(pipeline_yaml)

        # Step 2: Check variable references
        self._validate_variables(pipeline_yaml)

        # Step 3: Verify models exist
        self._verify_models(pipeline_yaml)

        # Step 4 & 5: Check/create skills
        self._ensure_skills_available(pipeline_yaml)

        # Generate executable pipeline
        pipeline = self._build_pipeline(pipeline_yaml)

        # Generate help documentation
        pipeline.help_text = self._generate_help(pipeline_yaml)

        return pipeline

    def _validate_syntax(self, yaml_data: dict):
        """Validate YAML syntax and structure."""
        required_fields = ['id', 'name', 'steps']

        # Check control flow syntax
        for step in yaml_data.get('steps', []):
            if 'for' in step:
                assert 'items' in step['for'], "For loop requires 'items'"
                assert 'variable' in step['for'], "For loop requires 'variable'"

            if 'while' in step:
                assert 'condition' in step['while'], "While loop requires 'condition'"

            if 'if' in step:
                assert 'condition' in step['if'], "If statement requires 'condition'"
                assert 'then' in step['if'], "If statement requires 'then' block"

    def _validate_variables(self, yaml_data: dict):
        """Check all referenced variables are defined."""
        defined_vars = set(['inputs'] + list(yaml_data.get('inputs', {}).keys()))

        for step in yaml_data.get('steps', []):
            # Add step output to defined vars
            if 'id' in step:
                defined_vars.add(step['id'])

            # Check parameter references
            params = step.get('parameters', {})
            for param_value in params.values():
                if isinstance(param_value, str) and '{{' in param_value:
                    # Extract variable references
                    refs = self._extract_variable_refs(param_value)
                    for ref in refs:
                        # Skip programmatically defined vars
                        if not ref.startswith('item') and not ref.startswith('loop'):
                            base_var = ref.split('.')[0]
                            assert base_var in defined_vars, \
                                f"Undefined variable: {base_var}"

    def _verify_models(self, yaml_data: dict):
        """Verify all specified models exist in registry."""
        model_registry = ModelRegistry()

        for step in yaml_data.get('steps', []):
            if 'model' in step:
                model_id = step['model']
                assert model_registry.exists(model_id), \
                    f"Model not found: {model_id}"

    def _ensure_skills_available(self, yaml_data: dict):
        """Check skills exist or create them."""
        skills_registry = SkillsRegistry()
        skill_creator = SkillCreator()

        for step in yaml_data.get('steps', []):
            skill_name = step.get('skill')
            if skill_name and not skills_registry.exists(skill_name):
                # Create skill using the skill creation workflow
                new_skill = skill_creator.create_skill(
                    pipeline_context=yaml_data,
                    required_capability=skill_name
                )
                skills_registry.register(new_skill)
```

### 3.2 Pipeline Object

```python
class Pipeline:
    """Executable pipeline object."""

    def __init__(self, id: str, workflow: LangGraphWorkflow, metadata: dict):
        self.id = id
        self.workflow = workflow
        self.metadata = metadata
        self.help_text = ""

    def __call__(self, **inputs):
        """Execute pipeline with given inputs."""
        state = PipelineState()
        state.variables.update(inputs)

        # Execute through LangGraph
        result = self.workflow.execute(state)
        return result

    def help(self):
        """Print usage instructions."""
        print(f"""
Pipeline: {self.metadata['name']}
{'-' * 40}
{self.help_text}

Required Inputs:
{self._format_inputs()}

Example Usage:
  pipeline = compile_pipeline('pipeline.yaml')
  result = pipeline(
      {self._format_example_inputs()}
  )
        """)

    def _generate_help_at_compile_time(self, yaml_data: dict):
        """Use Sonnet 4.5 to generate helpful documentation."""
        prompt = f"""
        Generate a brief, helpful description for this pipeline:
        {yaml.dump(yaml_data)}

        Include:
        1. What the pipeline does
        2. Expected inputs and formats
        3. What outputs to expect
        4. Any important notes or limitations

        Keep it concise (3-5 sentences).
        """

        # Call Sonnet 4.5 to generate help text
        model = AnthropicModelRegistry().get_model('orchestrator')
        self.help_text = model.generate(prompt)
```

## 4. Skill Creation Workflow (Enhanced)

### 4.1 ROMA-Inspired Four-Stage Creation

```python
class EnhancedSkillCreator:
    """Four-stage skill creation inspired by ROMA."""

    async def create_skill(self, pipeline_context: dict, capability: str):
        """Create skill using ROMA's four-stage pattern."""

        # Stage 1: Atomize - Determine complexity
        complexity = await self.atomize(capability, pipeline_context)

        if complexity == 'simple':
            # Direct creation for simple skills
            return await self.create_simple_skill(capability)

        # Stage 2: Plan - Decompose complex skill
        subtasks = await self.plan(capability, pipeline_context)

        # Stage 3: Execute - Create skill components
        components = []
        for subtask in subtasks:
            component = await self.execute_subtask(subtask)
            components.append(component)

        # Stage 4: Aggregate - Combine into final skill
        final_skill = await self.aggregate(components)

        # Review and test
        await self.review_and_test(final_skill)

        return final_skill

    async def atomize(self, capability: str, context: dict) -> str:
        """Determine if skill needs decomposition."""
        # Use Haiku 4.5 for quick assessment
        prompt = f"""
        Assess complexity of creating a skill for: {capability}
        Context: {context}

        Return: 'simple' or 'complex'
        """
        return await self.haiku_model.assess(prompt)

    async def plan(self, capability: str, context: dict) -> List[dict]:
        """Decompose complex skill into subtasks."""
        # Use Sonnet 4.5 for planning
        prompt = f"""
        Break down the skill '{capability}' into implementation subtasks.
        Each subtask should be atomic and testable.
        Context: {context}
        """
        return await self.sonnet_model.plan(prompt)
```

## 5. Example Pipelines (Updated with Control Flow)

### 5.1 Code Review Pipeline with Loops

```yaml
id: code-review-pipeline
name: "Comprehensive Code Review"
description: "Analyze code structure with iterative testing"
parameters:
  directory: "./sample_code"
  output_file: "./code_review_report.md"

steps:
  # Analyze all files in parallel
  - id: analyze_files
    for:
      items: "{{ list_files(directory) }}"
      variable: file_path
      parallel: true
    skill: code-analyzer
    parameters:
      file: "{{ file_path }}"
      analysis_type: "comprehensive"
    state:
      file_analyses: "{{ collect_results() }}"

  # Iteratively improve tests until they pass
  - id: test_loop
    while:
      condition: "{{ not all_tests_passing }}"
      max_iterations: 3
    steps:
      - id: generate_tests
        skill: test-generator
        parameters:
          analyses: "{{ file_analyses }}"
          iteration: "{{ loop_iteration }}"

      - id: run_tests
        skill: test-runner
        parameters:
          test_directory: "{{ generate_tests.directory }}"

      - id: fix_tests
        if:
          condition: "{{ run_tests.failures_count > 0 }}"
          then:
            - id: debug_and_fix
              skill: test-fixer
              parameters:
                failures: "{{ run_tests.failures }}"

      - id: update_state
        state:
          all_tests_passing: "{{ run_tests.success }}"
          loop_iteration: "{{ loop_iteration + 1 }}"

  # Generate final report
  - id: generate_report
    skill: report-generator
    parameters:
      analyses: "{{ file_analyses }}"
      test_results: "{{ test_loop.final_results }}"
      output_file: "{{ output_file }}"
```

## 6. Installation and Setup

### 6.1 Installation Process

```python
class OrchestratorInstaller:
    """Handles initial setup and installation."""

    def install(self):
        """Install orchestrator with default registries."""

        # Create ~/.orchestrator directory structure
        home_dir = Path.home() / ".orchestrator"
        home_dir.mkdir(exist_ok=True)

        skills_dir = home_dir / "skills"
        models_dir = home_dir / "models"
        pipelines_dir = home_dir / "pipelines"

        for dir_path in [skills_dir, models_dir, pipelines_dir]:
            dir_path.mkdir(exist_ok=True)

        # Copy default registries from package
        package_dir = Path(__file__).parent

        # Copy skills registry and default skills
        shutil.copy(
            package_dir / "registry/skills/default_registry.yaml",
            skills_dir / "registry.yaml"
        )

        # Copy all default skills
        default_skills = package_dir / "registry/skills/default_skills"
        for skill_dir in default_skills.iterdir():
            if skill_dir.is_dir():
                shutil.copytree(
                    skill_dir,
                    skills_dir / skill_dir.name
                )

        # Copy models registry
        shutil.copy(
            package_dir / "registry/models/default_registry.yaml",
            models_dir / "registry.yaml"
        )

        print(f"Orchestrator installed to {home_dir}")
        print(f"Skills available: {self._count_skills(skills_dir)}")
        print(f"Models configured: 3 (Opus 4.1, Sonnet 4.5, Haiku 4.5)")
```

## 7. Performance Optimizations (from AgnoAgents)

### 7.1 Fast Agent Instantiation

```python
class OptimizedSkillExecutor:
    """Performance-optimized skill execution inspired by AgnoAgents."""

    # Pre-compile skill templates for fast instantiation
    _skill_cache = {}

    @classmethod
    def get_skill(cls, skill_name: str) -> 'Skill':
        """Get skill with <10ms instantiation time."""
        if skill_name not in cls._skill_cache:
            # Load and compile skill once
            skill = cls._load_skill(skill_name)
            cls._skill_cache[skill_name] = skill

        # Return lightweight copy for execution
        return cls._skill_cache[skill_name].lightweight_copy()

    def execute_parallel(self, skills: List[str], inputs: List[dict]):
        """Execute multiple skills in parallel efficiently."""
        # Use asyncio for true parallelization
        loop = asyncio.get_event_loop()
        tasks = [
            loop.create_task(self._execute_skill(skill, input_data))
            for skill, input_data in zip(skills, inputs)
        ]
        return loop.run_until_complete(asyncio.gather(*tasks))
```

## 8. Testing Strategy (NO MOCKS)

### 8.1 E2B Sandbox Integration

```python
class E2BSandboxTester:
    """Secure sandbox testing inspired by ROMA's E2B integration."""

    async def test_skill(self, skill: Skill, test_cases: List[dict]):
        """Test skill in isolated E2B sandbox."""

        # Create sandbox with S3 mounting for artifacts
        sandbox = await self.create_e2b_sandbox()

        results = []
        for test_case in test_cases:
            # Execute in sandbox with real resources
            result = await sandbox.execute(
                skill.implementation,
                test_case['input'],
                mount_s3=True,  # For file operations
                network_access=True,  # For API calls
                timeout=30000  # 30 seconds max
            )

            # Validate against expected output
            validation = await self.validate_real_output(
                result,
                test_case['expected']
            )

            # Capture screenshots if visual
            if result.has_visual_output:
                screenshot = await sandbox.screenshot()
                validation['screenshot'] = screenshot

            results.append(validation)

        # Cleanup sandbox
        await sandbox.destroy()

        return results
```

## 9. Implementation Roadmap (Revised)

### Phase 1: Core Infrastructure (Week 1-2)
- [ ] Implement registry management system with ~/.orchestrator structure
- [ ] Create installer that sets up default registries
- [ ] Simplify to Anthropic-only models
- [ ] Integrate E2B sandbox for testing (from ROMA)

### Phase 2: Control Flow Implementation (Week 2-3)
- [ ] Implement for/while loop support in compiler
- [ ] Add conditional execution (if/elif/else)
- [ ] Build state management system
- [ ] Add parallelization support

### Phase 3: Skill System with ROMA Pattern (Week 3-4)
- [ ] Implement four-stage skill creation (Atomize, Plan, Execute, Aggregate)
- [ ] Build skill reviewer with iterative improvement
- [ ] Create real-world testing framework with E2B
- [ ] Optimize skill instantiation (<10ms target from AgnoAgents)

### Phase 4: Compile-Time Verification (Week 4-5)
- [ ] Implement syntax validation
- [ ] Add variable reference checking
- [ ] Build model verification
- [ ] Create skill availability checker
- [ ] Generate .help() documentation

### Phase 5: Example Pipelines (Week 5-6)
- [ ] Create all example pipelines with control flow
- [ ] Generate comprehensive test suites
- [ ] Document usage patterns
- [ ] Performance benchmarking

## 10. Conclusion

This enhanced design incorporates best practices from ROMA and AgnoAgents while maintaining the simplicity goal of the original refactor. The addition of advanced control flow, state management, and compile-time verification creates a powerful yet user-friendly framework. The ~/.orchestrator registry system ensures clean separation between package and user skills, while the four-stage creation pattern from ROMA ensures robust skill development.

---

**Key Improvements in V2:**
- Registry management in ~/.orchestrator
- ROMA's four-stage skill creation pattern
- AgnoAgents' performance optimizations
- Advanced control flow (loops, conditionals, parallelization)
- State-dependent variables and temporary products
- Compile-time verification with helpful error messages
- Pipeline objects with .help() method
- E2B sandbox integration for secure testing