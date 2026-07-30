# Claude Skills Refactor - Comprehensive Implementation Plan

## Overview

This plan details the complete refactoring of the Orchestrator framework to leverage Anthropic's Claude Skills system. The refactor will dramatically simplify the codebase by focusing exclusively on Anthropic models and replacing the complex tools registry with an automatic skill creation system.

## Key Principles

### 1. NO MOCK TESTS OR SIMULATIONS
- All tests will use real API calls to Anthropic's Claude models
- Real data processing with actual files and web resources
- E2B sandbox for secure code execution during testing
- Manual verification through screenshots and artifacts

### 2. Simplification Through Focus
- Remove support for non-Anthropic models (OpenAI, Google, local models)
- Direct pipeline compilation without LLM prompting
- Automatic skill creation with multi-agent review

### 3. Real-World Validation
- Every skill must have 3+ real-world test cases
- Test outputs must be manually verifiable
- Visual outputs require screenshots
- API responses validated for format and content

## Phase 1: Core Infrastructure Refactor (Days 1-7)

### 1.1 Model Registry Simplification
**Files to modify:**
- `src/orchestrator/models/registry.py` - Simplify to Anthropic-only
- `src/orchestrator/models/anthropic.py` - Update with latest models
- Remove: `openai.py`, `google.py`, `cohere.py`, `together.py`, `local.py`

**Implementation:**
```python
# New simplified registry structure
ANTHROPIC_MODELS = {
    "opus-4.1": {
        "id": "claude-opus-4-1-20250805",
        "role": "review_and_analysis",
        "context_window": 200000,
    },
    "sonnet-4.5": {
        "id": "claude-sonnet-4-5",
        "role": "orchestrator",
        "context_window": 1000000,
    },
    "haiku-4.5": {
        "id": "claude-haiku-4-5",
        "role": "simple_tasks",
        "context_window": 200000,
    }
}
```

**Tests (Real API calls):**
```python
def test_anthropic_models_real():
    """Test real API calls to each model."""
    client = anthropic.Client(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Test Opus 4.1 for review tasks
    opus_response = client.messages.create(
        model="claude-opus-4-1-20250805",
        messages=[{"role": "user", "content": "Analyze this code: print('hello')"}],
        max_tokens=100
    )
    assert opus_response.content

    # Test Sonnet 4.5 for orchestration
    sonnet_response = client.messages.create(
        model="claude-sonnet-4-5",
        messages=[{"role": "user", "content": "Create a simple Python function"}],
        max_tokens=200
    )
    assert sonnet_response.content

    # Test Haiku 4.5 for simple tasks
    haiku_response = client.messages.create(
        model="claude-haiku-4-5",
        messages=[{"role": "user", "content": "Format this JSON: {a:1}"}],
        max_tokens=50
    )
    assert haiku_response.content
```

### 1.2 Registry Management System
**New files to create:**
```
orchestrator/registry/
├── skills/
│   ├── default_registry.yaml
│   └── default_skills/
└── models/
    └── default_registry.yaml

src/orchestrator/skills/installer.py  # Manages ~/.orchestrator
```

**Implementation:**
```python
class RegistryInstaller:
    """Manages registry installation to user home."""

    def __init__(self):
        self.home_dir = Path.home() / ".orchestrator"

    def install(self):
        """Copy default registries to user home."""
        self.home_dir.mkdir(exist_ok=True)

        # Copy default skills
        shutil.copytree(
            "orchestrator/registry/skills",
            self.home_dir / "skills"
        )

        # Copy model registry
        shutil.copy(
            "orchestrator/registry/models/default_registry.yaml",
            self.home_dir / "models/registry.yaml"
        )
```

## Phase 2: Skills System Implementation (Days 8-14)

### 2.1 Skill Creator with ROMA Pattern
**New file: `src/orchestrator/skills/creator.py`**

```python
class SkillCreator:
    """Creates skills using ROMA pattern from research."""

    async def create_skill(self,
                          pipeline_context: dict,
                          required_capability: str):
        """
        ROMA Pattern:
        1. Atomize - Break down into atomic tasks
        2. Plan - Create execution strategy
        3. Execute - Run with real resources
        4. Aggregate - Combine results
        """

        # Step 1: Atomize using Sonnet 4.5
        atomic_tasks = await self.atomize_capability(
            required_capability,
            model="sonnet-4.5"
        )

        # Step 2: Plan skill structure
        skill_plan = await self.plan_skill(
            atomic_tasks,
            model="sonnet-4.5"
        )

        # Step 3: Execute skill creation
        skill_implementation = await self.execute_creation(
            skill_plan,
            model="sonnet-4.5"
        )

        # Step 4: Aggregate and review with Opus 4.1
        reviewed_skill = await self.review_and_aggregate(
            skill_implementation,
            model="opus-4.1"
        )

        # Step 5: Real-world testing (NO MOCKS)
        test_results = await self.test_with_real_data(reviewed_skill)

        return reviewed_skill
```

### 2.2 Skill Testing Framework
**New file: `src/orchestrator/skills/tester.py`**

```python
class RealWorldSkillTester:
    """Tests skills with real resources - NO MOCKS."""

    async def test_skill(self, skill: Skill):
        """Execute skill with real data."""
        test_cases = []

        for example in skill.examples:
            # Use real APIs
            if skill.requires_api:
                result = await self.call_real_api(
                    skill.api_endpoint,
                    example.input
                )

            # Use real files
            if skill.requires_files:
                files = await self.download_real_files(
                    example.file_urls
                )
                result = await skill.execute(files)

            # Capture screenshots for visual validation
            if skill.produces_visual_output:
                screenshot = await self.capture_screenshot(result)
                test_cases.append({
                    "input": example.input,
                    "output": result,
                    "screenshot": screenshot
                })

            # Validate output format
            validation = await self.validate_output(
                result,
                example.expected_format
            )

            test_cases.append({
                "input": example.input,
                "output": result,
                "valid": validation.passed,
                "details": validation.details
            })

        return TestResults(test_cases)
```

### 2.3 Skill Registry Structure
**File: `~/.orchestrator/skills/registry.yaml`**

```yaml
version: "1.0.0"
skills:
  web-search:
    name: "web-search"
    description: "Search web and extract information"
    version: "1.0.0"
    capabilities: ["search", "extract", "summarize"]
    parameters:
      query: "string"
      max_results: "integer"
    examples:
      - input: {query: "Claude 3.5 features", max_results: 5}
        expected_format: "list of results with titles"
    path: "./web-search/"
    tested_on: "2024-01-15"
    test_results: "./web-search/test_results.md"

  code-analyzer:
    name: "code-analyzer"
    description: "Analyze code structure and quality"
    version: "1.0.0"
    # ... more skills
```

## Phase 3: Pipeline Compilation Enhancement (Days 15-21)

### 3.1 Direct Compilation to LangGraph
**Modify: `src/orchestrator/compiler/pipeline_compiler.py`**

```python
class EnhancedPipelineCompiler:
    """Compile YAML directly to LangGraph without LLM."""

    def compile(self, pipeline_yaml: dict) -> CompiledPipeline:
        """Direct compilation with skill auto-creation."""

        # Parse pipeline structure
        parsed = self.parse_pipeline(pipeline_yaml)

        # Check and create missing skills
        for step in parsed.steps:
            if not self.skill_registry.exists(step.skill):
                # Auto-create missing skill
                new_skill = await self.skill_creator.create_skill(
                    pipeline_context=pipeline_yaml,
                    required_capability=step.action
                )
                self.skill_registry.register(new_skill)

        # Generate LangGraph workflow
        workflow = self.generate_langgraph(parsed)

        # Add control flow support
        workflow = self.add_control_flow(workflow, parsed)

        # Compile-time verification
        self.verify_compilation(workflow)

        return CompiledPipeline(workflow)
```

### 3.2 Advanced Control Flow
**New file: `src/orchestrator/control_flow/enhanced.py`**

```python
class EnhancedControlFlow:
    """Advanced control flow with loops and conditionals."""

    def add_for_loop(self, workflow: LangGraphWorkflow,
                     loop_spec: dict):
        """Add for loop with optional parallelization."""
        if loop_spec.get('parallel'):
            return self.add_parallel_for(workflow, loop_spec)
        else:
            return self.add_sequential_for(workflow, loop_spec)

    def add_while_loop(self, workflow: LangGraphWorkflow,
                      loop_spec: dict):
        """Add while loop with condition checking."""
        max_iter = loop_spec.get('max_iterations', 100)
        condition = loop_spec['condition']

        # Create loop node in LangGraph
        loop_node = ConditionalNode(
            condition=condition,
            max_iterations=max_iter
        )
        workflow.add_node(loop_node)
```

## Phase 4: Example Pipelines (Days 22-28)

### 4.1 Code Review Pipeline with Real Testing
**File: `examples/pipelines/code_review.yaml`**

```yaml
id: code-review-pipeline
name: "Comprehensive Code Review with Testing"
description: "Analyze code and create real tests"

steps:
  - id: analyze_structure
    skill: code-analyzer
    parameters:
      directory: "./sample_project"

  - id: create_tests
    skill: test-generator
    parameters:
      components: "{{ analyze_structure.components }}"
      use_mocks: false  # NEVER use mocks

  - id: run_tests
    skill: test-runner
    parameters:
      test_dir: "{{ create_tests.output_dir }}"
      capture_output: true
      use_real_apis: true

  - id: generate_report
    skill: report-generator
    parameters:
      test_results: "{{ run_tests.results }}"
      screenshots: "{{ run_tests.screenshots }}"
```

### 4.2 Research Pipeline with Source Verification
**File: `examples/pipelines/deep_research.yaml`**

```yaml
id: deep-research
name: "Multi-Agent Research with Verification"

steps:
  - id: research_parallel
    for:
      items: "{{ research_topics }}"
      variable: topic
      parallel: true
    skill: research-agent
    parameters:
      query: "{{ topic }}"
      min_sources: 5

  - id: verify_sources
    for:
      items: "{{ research_parallel.sources }}"
      variable: source
    skill: source-verifier
    parameters:
      url: "{{ source.url }}"
      check_existence: true
      download_content: true

  - id: synthesize
    skill: research-synthesizer
    parameters:
      verified_sources: "{{ verify_sources.verified }}"
```

## Phase 5: Testing Strategy (Days 29-35)

### 5.1 Real-World Integration Tests
**File: `tests/integration/test_real_skills.py`**

```python
class TestRealSkillExecution:
    """All tests use real resources."""

    @pytest.mark.real_api
    def test_web_search_skill_real(self):
        """Test web search with real queries."""
        skill = SkillRegistry.get("web-search")

        # Real web search
        result = skill.execute({
            "query": "Anthropic Claude latest features",
            "max_results": 5
        })

        # Verify real results
        assert len(result.results) > 0
        for item in result.results:
            # Verify URL is accessible
            response = requests.get(item.url)
            assert response.status_code == 200

    @pytest.mark.real_files
    def test_code_analyzer_real_repo(self):
        """Test code analysis on real repository."""
        skill = SkillRegistry.get("code-analyzer")

        # Clone real repo for testing
        repo_url = "https://github.com/anthropics/claude-sdk"
        clone_dir = "./test_repos/claude-sdk"
        subprocess.run(["git", "clone", repo_url, clone_dir])

        # Analyze real code
        result = skill.execute({
            "directory": clone_dir,
            "analysis_type": "comprehensive"
        })

        # Verify analysis results
        assert result.file_count > 0
        assert result.complexity_score > 0
        assert len(result.components) > 0
```

### 5.2 E2B Sandbox Testing
**File: `tests/integration/test_e2b_sandbox.py`**

```python
class TestE2BSandbox:
    """Test code execution in E2B sandbox."""

    @pytest.mark.e2b
    async def test_skill_in_sandbox(self):
        """Execute skill code in secure sandbox."""
        sandbox = await E2BSandbox.create()

        try:
            # Upload skill code to sandbox
            await sandbox.upload_file(
                "skill.py",
                skill_code
            )

            # Execute in sandbox
            result = await sandbox.execute(
                "python skill.py",
                timeout=30
            )

            # Verify execution
            assert result.exit_code == 0
            assert "output" in result.stdout

        finally:
            await sandbox.close()
```

## Phase 6: Documentation and Migration (Days 36-42)

### 6.1 User Documentation Structure
```
docs/
├── getting-started.md
├── skill-creation-guide.md
├── pipeline-authoring.md
├── control-flow-reference.md
├── migration-guide.md
└── api-reference/
    ├── skills.md
    ├── pipelines.md
    └── registry.md
```

### 6.2 Migration Tools
**File: `src/orchestrator/migration/converter.py`**

```python
class PipelineMigrator:
    """Convert old pipelines to new format."""

    def migrate_pipeline(self, old_pipeline: dict) -> dict:
        """Convert old format to skills-based."""
        new_pipeline = {
            "id": old_pipeline["id"],
            "name": old_pipeline["name"],
            "steps": []
        }

        for step in old_pipeline["steps"]:
            # Map old tools to new skills
            skill_name = self.map_tool_to_skill(step["tool"])

            new_step = {
                "id": step["id"],
                "skill": skill_name,
                "action": step.get("action", "execute"),
                "parameters": step.get("parameters", {})
            }

            # Preserve control flow
            if "foreach" in step:
                new_step["for"] = {
                    "items": step["foreach"],
                    "variable": step.get("item_var", "item")
                }

            new_pipeline["steps"].append(new_step)

        return new_pipeline
```

## Testing Requirements

### Core Testing Principles
1. **NO MOCKS OR SIMULATIONS** - All tests use real resources
2. **Real API Calls** - Actual calls to Anthropic Claude models
3. **Real Data Processing** - Download and process real files
4. **Manual Verification** - Screenshots and artifacts for validation
5. **Cost Tracking** - Monitor API costs during testing

### Test Coverage Requirements
- Each skill: Minimum 3 real-world test cases
- Each pipeline: Full end-to-end execution test
- Control flow: Test all loop and conditional variants
- Error handling: Test with real failure scenarios
- Performance: Measure with real data loads

### Example Test Suite Structure
```python
# Real API test example
def test_claude_skill_creation_real():
    """Create a skill using real Claude API."""
    creator = SkillCreator(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Real skill creation request
    skill = creator.create_skill(
        capability="Extract tables from PDF",
        test_with_real_pdf=True
    )

    # Download real PDF for testing
    pdf_url = "https://example.com/sample.pdf"
    pdf_content = requests.get(pdf_url).content

    # Execute skill with real PDF
    result = skill.execute(pdf_content)

    # Verify tables extracted
    assert len(result.tables) > 0

    # Save screenshot for manual verification
    screenshot_path = f"./test_outputs/{skill.name}_result.png"
    result.save_screenshot(screenshot_path)

    print(f"Manual verification: {screenshot_path}")
```

## Success Metrics

### Quantitative Metrics
- **Code Reduction**: 70% reduction in codebase size
- **Dependency Reduction**: From ~50 to <10 dependencies
- **Compilation Speed**: 100x faster (no LLM in compilation)
- **Test Coverage**: 100% real-world tests (0% mocks)
- **Skill Creation Success**: 95% first-attempt success rate

### Qualitative Metrics
- **Developer Experience**: Simplified API, clear documentation
- **Maintainability**: Reduced complexity, focused architecture
- **Extensibility**: Easy skill creation, pipeline authoring
- **Reliability**: Real-world validated, no mock failures

## Risk Mitigation

### Technical Risks
1. **Skill Creation Failures**
   - Mitigation: Comprehensive examples, fallback to manual

2. **API Cost Overruns**
   - Mitigation: Rate limiting, cost tracking, budget alerts

3. **Breaking Changes**
   - Mitigation: Compatibility layer, gradual migration

### Implementation Risks
1. **Timeline Slippage**
   - Mitigation: Phased delivery, core features first

2. **Integration Issues**
   - Mitigation: Early testing, incremental integration

## Timeline Summary

- **Week 1**: Core infrastructure refactor
- **Week 2**: Skills system implementation
- **Week 3**: Pipeline compilation enhancement
- **Week 4**: Example pipelines creation
- **Week 5**: Integration testing
- **Week 6**: Documentation and migration tools

## Next Steps

1. Review and approve this implementation plan
2. Set up development branch for refactor
3. Begin Phase 1 implementation
4. Daily progress updates on issue #426
5. Weekly demos of completed phases

---

This implementation plan provides a clear roadmap for the Claude Skills refactor with emphasis on real-world testing and validation. All code will be tested with actual API calls, real data, and manual verification - no mocks or simulations.