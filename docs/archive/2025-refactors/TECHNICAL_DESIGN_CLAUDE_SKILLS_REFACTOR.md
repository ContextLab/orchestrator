# Technical Design Document: Claude Skills-Based Orchestrator Refactor

## Executive Summary

This document outlines a comprehensive refactor of the Orchestrator framework to leverage Anthropic's Claude Skills system. The refactor will dramatically simplify the codebase by:
- Removing support for non-Anthropic models (temporarily)
- Replacing the complex tools registry with a skills-based system
- Implementing automatic skill creation with multi-agent review
- Compiling pipelines directly to LangGraph without LLM prompting
- Creating a comprehensive suite of example pipelines

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

### Leveraging Claude Agent SDK
The implementation will build upon the Claude Agent SDK, which provides the same infrastructure that powers Claude Code, ensuring production-ready agent capabilities from day one.

## 1. Core Architecture

### 1.1 Simplified Component Structure

```
orchestrator/
├── core/
│   ├── __init__.py
│   ├── pipeline.py         # Pipeline definition and structure
│   ├── compiler.py         # Direct YAML to LangGraph compilation
│   └── executor.py         # LangGraph execution engine
├── models/
│   ├── __init__.py
│   └── anthropic.py        # Anthropic-only model registry
├── skills/
│   ├── __init__.py
│   ├── registry.py         # Skills registry management
│   ├── creator.py          # Skill creation workflow
│   ├── reviewer.py         # Skill review and validation
│   └── repository/         # Skill storage directory
│       ├── REGISTRY.yaml   # Master registry file
│       └── [skill-name]/   # Individual skill directories
└── examples/
    ├── pipelines/          # Example pipeline definitions
    └── outputs/            # Pipeline execution outputs
```

### 1.2 Model Registry Simplification

```python
class AnthropicModelRegistry:
    """Simplified registry for Anthropic models only."""

    MODELS = {
        "opus-4.1": {
            "id": "claude-opus-4-1-20250805",
            "role": "review_and_analysis",
            "context_window": 200000,
            "capabilities": ["deep_analysis", "complex_reasoning", "comprehensive_review"],
            "description": "Most powerful Claude model for complex analysis and review tasks",
            "released": "2025-08-05"
        },
        "sonnet-4.5": {
            "id": "claude-sonnet-4-5",
            "role": "orchestrator",  # Default for agents and coding
            "context_window": 1000000,  # 1M tokens for API customers
            "capabilities": ["agent_coordination", "code_generation", "computer_use"],
            "description": "Best coding model in the world, strongest for building agents",
            "released": "2025-09-29",
            "pricing": {"input": "$3/M tokens", "output": "$15/M tokens"}
        },
        "haiku-4.5": {
            "id": "claude-haiku-4-5",
            "role": "simple_tasks",
            "context_window": 200000,
            "capabilities": ["quick_validation", "formatting", "simple_queries"],
            "description": "90% of Sonnet 4.5's coding performance at 1/3 the cost",
            "released": "2025-10-15",
            "pricing": {"input": "$1/M tokens", "output": "$5/M tokens"}
        }
    }

    def get_model(self, role="orchestrator"):
        """Get model by role with automatic fallback."""
        # Sonnet 4.5 is the default for orchestration and agent tasks
        # Opus 4.1 for deep review and analysis
        # Haiku 4.5 for simple, high-volume tasks
        # Implementation details...
```

## 2. Skills System Design

### 2.1 Skills Registry Structure

```yaml
# REGISTRY.yaml
version: "1.0.0"
skills:
  web-search:
    name: "web-search"
    description: "Search the web and extract information"
    version: "1.0.0"
    capabilities:
      - search
      - extract
      - summarize
    parameters:
      query: "string"
      max_results: "integer"
      region: "string"
    examples:
      - description: "Search for AI news"
        input: {query: "AI developments 2024", max_results: 5}
        expected_output_format: "list of results with titles and snippets"
    path: "./repository/web-search/"
    tested_on: "2024-01-15"
    test_results: "./repository/web-search/test_results.md"

  code-analyzer:
    name: "code-analyzer"
    description: "Analyze code structure and quality"
    version: "1.0.0"
    capabilities:
      - analyze
      - review
      - test
    parameters:
      directory: "path"
      analysis_type: "enum[structure|quality|security|all]"
    # ... more skills
```

### 2.2 Skill Directory Structure

```
skills/repository/[skill-name]/
├── SKILL.md                # Skill definition with YAML frontmatter
├── implementation.py       # Python implementation (if needed)
├── examples/              # Example inputs and outputs
│   ├── example_1.yaml
│   └── example_1_output.md
├── tests/                 # Real-world tests (NO MOCKS)
│   ├── test_real_api.py
│   └── test_results.md
└── validation_log.md      # Review and validation history
```

## 3. Skill Creation Workflow

### 3.1 Automatic Skill Creation Process

```python
class SkillCreator:
    """Manages the skill creation and review workflow."""

    async def create_skill(self,
                          pipeline_context: dict,
                          required_capability: str,
                          orchestrator_model="sonnet-4.5",
                          reviewer_model="opus-4.1"):
        """
        Creates a new skill with multi-agent review.

        Models:
        - Sonnet 4.5: Default orchestrator (best for building agents and coding)
        - Opus 4.1: Default reviewer (most powerful for deep analysis)
        - Haiku 4.5: Used for quick validation checks during testing

        Flow:
        1. Orchestrator analyzes pipeline context
        2. Orchestrator uses skill-creator meta-skill
        3. Reviewer validates the created skill
        4. Iterative improvement until approval
        5. Real-world testing with actual data/APIs
        6. Registration in skills registry
        """

        # Step 1: Analyze what's needed
        skill_requirements = await self.analyze_requirements(
            pipeline_context,
            required_capability,
            orchestrator_model
        )

        # Step 2: Create initial skill
        new_skill = await self.generate_skill(
            skill_requirements,
            orchestrator_model
        )

        # Step 3: Review and refine loop
        max_iterations = 5
        for i in range(max_iterations):
            review_result = await self.review_skill(
                new_skill,
                pipeline_context,
                reviewer_model
            )

            if review_result.approved:
                break

            # Apply reviewer's suggested changes
            new_skill = await self.apply_changes(
                new_skill,
                review_result.changes,
                orchestrator_model
            )

        # Step 4: Real-world testing
        test_results = await self.test_skill_real_world(new_skill)

        # Step 5: Final validation
        if not test_results.all_passed:
            # Return to review with test results
            return await self.create_skill(
                pipeline_context,
                required_capability,
                orchestrator_model,
                reviewer_model
            )

        # Step 6: Register skill
        await self.register_skill(new_skill, test_results)

        return new_skill
```

### 3.2 Real-World Testing Requirements

```python
class SkillTester:
    """Tests skills with real data and APIs - NO MOCKS."""

    async def test_skill_real_world(self, skill):
        """
        Execute skill with real-world data.
        Requirements:
        - Use actual API calls (with rate limiting)
        - Download real files
        - Process actual data
        - Create real outputs
        - Manually verify results
        """

        test_cases = []

        # Run each example from skill definition
        for example in skill.examples:
            # Execute with real inputs
            result = await self.execute_skill(
                skill,
                example.input,
                use_real_apis=True,
                use_real_data=True
            )

            # Verify outputs exist and are valid
            verification = await self.verify_output(
                result,
                example.expected_output_format
            )

            test_cases.append({
                "input": example.input,
                "output": result,
                "verification": verification,
                "screenshots": self.capture_if_visual(result)
            })

        # Generate comprehensive test report
        return self.generate_test_report(test_cases)
```

## 4. Pipeline Compilation

### 4.1 Simplified Compilation Process

```python
class PipelineCompiler:
    """Compiles YAML pipelines directly to LangGraph - NO LLM PROMPTING."""

    def compile(self, pipeline_yaml: dict) -> LangGraphWorkflow:
        """
        Direct compilation without LLM involvement.

        Process:
        1. Parse YAML structure
        2. Map steps to skills
        3. Check skill availability
        4. Create missing skills if needed
        5. Generate LangGraph nodes and edges
        6. Return executable workflow
        """

        workflow = LangGraphWorkflow(pipeline_yaml['id'])

        # Parse pipeline steps
        for step in pipeline_yaml['steps']:
            # Check if required skill exists
            skill = self.skills_registry.get(step['tool'])

            if not skill:
                # Create skill automatically
                skill = await self.skill_creator.create_skill(
                    pipeline_context=pipeline_yaml,
                    required_capability=step['action']
                )

            # Create LangGraph node
            node = self.create_node_from_skill(skill, step)
            workflow.add_node(node)

            # Add edges based on dependencies
            if 'dependencies' in step:
                for dep in step['dependencies']:
                    workflow.add_edge(LangGraphEdge(dep, step['id']))

        return workflow
```

## 5. Example Pipelines

### 5.1 Code Review Pipeline

```yaml
id: code-review-pipeline
name: "Comprehensive Code Review"
description: "Analyze code structure, quality, and create/run tests"
parameters:
  directory: "./sample_code"
  output_file: "./code_review_report.md"

steps:
  - id: analyze_structure
    skill: code-analyzer
    action: analyze_structure
    parameters:
      directory: "{{ directory }}"

  - id: identify_components
    skill: code-analyzer
    action: identify_components
    parameters:
      structure: "{{ analyze_structure.output }}"

  - id: create_test_suite
    skill: test-generator
    action: generate_tests
    parameters:
      components: "{{ identify_components.output }}"
      test_type: "comprehensive"
      use_mocks: false

  - id: run_tests
    skill: test-runner
    action: execute_tests
    parameters:
      test_directory: "{{ create_test_suite.test_directory }}"
      capture_output: true

  - id: analyze_results
    skill: test-analyzer
    action: analyze_results
    parameters:
      test_results: "{{ run_tests.output }}"

  - id: generate_report
    skill: report-generator
    action: create_markdown
    parameters:
      sections:
        - structure: "{{ analyze_structure.output }}"
        - components: "{{ identify_components.output }}"
        - test_results: "{{ analyze_results.output }}"
      output_file: "{{ output_file }}"
```

### 5.2 Deep Research Agent Pipeline

```yaml
id: deep-research-pipeline
name: "Multi-Agent Deep Research"
description: "Parallel research with source verification"
parameters:
  topic: "Quantum Computing Applications in Medicine"
  min_sources: 20
  report_length: 5000
  output_file: "./research_report.md"

steps:
  - id: create_research_plan
    skill: research-planner
    action: generate_plan
    parameters:
      topic: "{{ topic }}"
      depth: "comprehensive"

  - id: parallel_research
    skill: research-agent
    action: research
    parallel: true
    foreach: "{{ create_research_plan.subtopics }}"
    parameters:
      query: "{{ item }}"
      min_sources: 5

  - id: verify_sources
    skill: source-verifier
    action: verify
    foreach: "{{ parallel_research.sources }}"
    parameters:
      url: "{{ item.url }}"
      check_existence: true
      verify_content: true

  - id: synthesize_findings
    skill: research-synthesizer
    action: synthesize
    parameters:
      research_results: "{{ parallel_research.output }}"
      verified_sources: "{{ verify_sources.output }}"
      target_length: "{{ report_length }}"

  - id: generate_bibliography
    skill: bibliography-generator
    action: create
    parameters:
      sources: "{{ verify_sources.verified }}"
      format: "APA"

  - id: create_report
    skill: report-generator
    action: create_markdown
    parameters:
      content: "{{ synthesize_findings.output }}"
      bibliography: "{{ generate_bibliography.output }}"
      output_file: "{{ output_file }}"
```

### 5.3 Multi-Agent Problem Solver Pipeline

```yaml
id: multi-agent-problem-solver
name: "Dynamic Team Problem Solving"
description: "Manager agent coordinates team of specialized agents"
parameters:
  task_description: "Design a sustainable city transportation system"
  output_directory: "./problem_solution"

steps:
  - id: initialize_manager
    skill: manager-agent
    action: initialize
    parameters:
      role: "project_manager"
      task: "{{ task_description }}"

  - id: analyze_requirements
    skill: manager-agent
    action: analyze_task
    parameters:
      task: "{{ task_description }}"
      output: "requirements_and_subtasks"

  - id: create_team_pipeline
    skill: pipeline-generator
    action: generate
    parameters:
      requirements: "{{ analyze_requirements.output }}"
      team_size: "dynamic"

  - id: launch_team
    skill: team-executor
    action: execute_pipeline
    parameters:
      pipeline: "{{ create_team_pipeline.output }}"
      coordination: "manager_led"

  - id: review_solutions
    skill: manager-agent
    action: review
    parameters:
      team_outputs: "{{ launch_team.outputs }}"
      criteria: "{{ analyze_requirements.success_criteria }}"

  - id: iterate_if_needed
    skill: manager-agent
    action: iterate
    condition: "{{ review_solutions.needs_improvement }}"
    parameters:
      feedback: "{{ review_solutions.feedback }}"
      team_pipeline: "{{ create_team_pipeline.output }}"

  - id: compile_solution
    skill: solution-compiler
    action: compile
    parameters:
      components: "{{ launch_team.outputs }}"
      review_notes: "{{ review_solutions.notes }}"
      output_directory: "{{ output_directory }}"
```

### 5.4 Additional Example Pipelines

#### 5.4.1 Documentation Generator
```yaml
id: documentation-generator
name: "Automatic Documentation Creation"
description: "Generate comprehensive docs from code"
# Analyzes code, creates API docs, usage examples, and deployment guides
```

#### 5.4.2 Data Pipeline Builder
```yaml
id: data-pipeline-builder
name: "ETL Pipeline Generator"
description: "Create data processing pipelines from requirements"
# Analyzes data sources, designs transformations, creates monitoring
```

#### 5.4.3 Security Audit Pipeline
```yaml
id: security-audit
name: "Comprehensive Security Analysis"
description: "Multi-layer security assessment"
# Code analysis, dependency scanning, penetration testing simulation
```

#### 5.4.4 Content Creation Suite
```yaml
id: content-creation
name: "Multi-format Content Generator"
description: "Create blog posts, social media, and marketing materials"
# Research, writing, image generation, SEO optimization
```

#### 5.4.5 API Integration Builder
```yaml
id: api-integration
name: "Automatic API Integration"
description: "Connect and integrate multiple APIs"
# API discovery, authentication setup, data mapping, error handling
```

## 6. Implementation Roadmap

### Phase 1: Core Refactor (Week 1-2)
- [ ] Remove non-Anthropic model support
- [ ] Simplify model registry to 3 Anthropic models
- [ ] Create basic skills registry structure
- [ ] Implement direct pipeline compilation (no LLM)

### Phase 2: Skills System (Week 2-3)
- [ ] Implement skill creator with orchestrator model
- [ ] Implement skill reviewer with review loops
- [ ] Create real-world testing framework (NO MOCKS)
- [ ] Build skills repository structure

### Phase 3: Core Skills (Week 3-4)
- [ ] Import and adapt Anthropic example skills
- [ ] Create skills for all example pipeline needs
- [ ] Test each skill with real data/APIs
- [ ] Document skill creation patterns

### Phase 4: Example Pipelines (Week 4-5)
- [ ] Implement code review pipeline with test generation
- [ ] Implement deep research pipeline with verification
- [ ] Implement multi-agent problem solver
- [ ] Create 5 additional showcase pipelines
- [ ] Generate example outputs for each

### Phase 5: Testing & Documentation (Week 5-6)
- [ ] Comprehensive integration testing (NO MOCKS)
- [ ] Performance benchmarking
- [ ] Create user documentation
- [ ] Migration guide from old system

## 7. Testing Strategy

### 7.1 Real-World Testing Requirements
```python
class RealWorldTestFramework:
    """All tests use real resources - NO MOCKS."""

    TEST_REQUIREMENTS = {
        "apis": {
            "anthropic": "Real API calls with actual models",
            "web_services": "Real HTTP requests to live services",
            "databases": "Real database connections and operations"
        },
        "files": {
            "creation": "Actually create files on disk",
            "reading": "Read real files with actual content",
            "downloading": "Download real files from internet"
        },
        "models": {
            "execution": "Run actual model inference",
            "validation": "Verify real model outputs",
            "costs": "Track actual API costs"
        }
    }
```

### 7.2 Validation Criteria
- Every skill must have at least 3 real-world test cases
- Test outputs must be manually verifiable
- Visual outputs require screenshots for validation
- API responses must be validated for format and content
- File operations must produce verifiable artifacts

## 8. Migration Plan

### 8.1 Deprecation Strategy
```python
class MigrationHelper:
    """Helps migrate from old system to new."""

    def migrate_pipeline(self, old_pipeline):
        """Convert old pipeline format to new skills-based format."""
        # Map old tools to new skills
        # Convert parameters
        # Update syntax
        return new_pipeline

    def migrate_tool_to_skill(self, old_tool):
        """Convert old tool definition to skill."""
        # Extract capabilities
        # Create SKILL.md
        # Generate examples
        # Add to registry
        return new_skill
```

### 8.2 Backwards Compatibility
- Temporary adapters for critical old pipelines
- Gradual migration with deprecation warnings
- Tool-to-skill mapping for common operations

## 9. Success Metrics

### 9.1 Simplification Metrics
- **Code Reduction**: Target 70% reduction in codebase size
- **Dependency Reduction**: From ~50 dependencies to <10
- **Compilation Speed**: Direct compilation without LLM = 100x faster
- **Maintenance Overhead**: 80% reduction in complexity

### 9.2 Capability Metrics
- **Skill Coverage**: 100% of example pipeline needs covered
- **Creation Success**: 95% of new skills work on first review cycle
- **Test Coverage**: 100% real-world test coverage (0% mocks)
- **Pipeline Success**: All example pipelines execute successfully

## 10. Risk Mitigation

### 10.1 Technical Risks
- **Risk**: Claude Skills API changes
  - **Mitigation**: Version lock, abstraction layer

- **Risk**: Skill creation failures
  - **Mitigation**: Fallback to manual creation, comprehensive examples

- **Risk**: Real-world test costs
  - **Mitigation**: Rate limiting, cost tracking, budget alerts

### 10.2 Migration Risks
- **Risk**: Breaking existing pipelines
  - **Mitigation**: Phased rollout, compatibility layer

- **Risk**: User adoption challenges
  - **Mitigation**: Comprehensive docs, migration tools

## 11. Cost Optimization Strategy

### 11.1 Model Selection for Cost Efficiency
```python
class CostOptimizer:
    """Intelligently route tasks to appropriate models based on complexity."""

    MODEL_COSTS = {
        "haiku-4.5": {"input": 1.0, "output": 5.0},  # Per million tokens
        "sonnet-4.5": {"input": 3.0, "output": 15.0},
        "opus-4.1": {"input": 15.0, "output": 75.0}  # Estimated
    }

    def select_model(self, task_complexity, budget_priority):
        """
        Route tasks intelligently:
        - Haiku 4.5: Simple validations, formatting (90% of Sonnet performance)
        - Sonnet 4.5: Skill creation, agent coordination (best for complex agents)
        - Opus 4.1: Critical reviews, deep analysis (when quality is paramount)
        """
```

### 11.2 Leveraging Platform Features
- **Prompt Caching**: 90% cost savings on repeated contexts
- **Batch Processing**: 50% cost savings for non-time-critical tasks
- **Memory Capabilities**: Reduce redundant processing across sessions
- **Context Window Management**: Sonnet 4.5's 1M token window for complex pipelines

## Conclusion

This refactor represents a fundamental simplification of the Orchestrator framework, leveraging Claude Skills to create a more maintainable, powerful, and user-friendly system. By focusing exclusively on Anthropic models initially, we can deliver a robust implementation quickly, with the option to extend support to other platforms as they adopt similar capabilities.

The emphasis on real-world testing without mocks ensures reliability, while the automatic skill creation workflow enables rapid capability expansion. The comprehensive example pipelines will demonstrate the system's power and provide templates for users to build upon.