# Claude Skills Orchestrator - User Guide

## Overview

The Claude Skills Orchestrator is a powerful framework for building multi-agent workflows using Anthropic's Claude models. It features automatic skill creation, intelligent model selection, and advanced control flow capabilities.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Skills System](#skills-system)
5. [Pipeline Authoring](#pipeline-authoring)
6. [Model Selection](#model-selection)
7. [Examples](#examples)
8. [API Reference](#api-reference)

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ContextLab/orchestrator.git
cd orchestrator

# Install dependencies
pip install -r requirements.txt

# Install orchestrator
pip install -e .
```

### Configuration

Set up your API key:

```bash
# Create configuration directory
mkdir -p ~/.orchestrator

# Add your API key
echo "ANTHROPIC_API_KEY=your-key-here" >> ~/.orchestrator/.env
```

### Run Your First Pipeline

```bash
# Run a simple example
orchestrator run examples/claude_skills_refactor/01_simple_code_review.yaml
```

---

## Installation

### System Requirements

- **Python**: 3.8 or higher
- **Operating System**: macOS or Linux
- **API Access**: Anthropic API key (Claude models)

### Install Steps

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Orchestrator**:
   ```bash
   pip install -e .
   ```

3. **Configure API keys**:
   ```bash
   mkdir -p ~/.orchestrator
   cat > ~/.orchestrator/.env << EOF
   ANTHROPIC_API_KEY=your-anthropic-api-key
   EOF
   ```

4. **Verify installation**:
   ```bash
   orchestrator --version
   ```

### Optional: Docker Setup

Docker is optional but recommended for isolated skill execution:

```bash
# Docker will be automatically installed/started when needed
# Or install manually:
# macOS: brew install --cask docker
# Linux: curl -fsSL https://get.docker.com | sh
```

---

## Configuration

### API Keys

Store API keys in `~/.orchestrator/.env`:

```bash
# Anthropic (Required)
ANTHROPIC_API_KEY=sk-ant-api03-...

# Optional: For future extensions
OPENAI_API_KEY=sk-...
GOOGLE_AI_API_KEY=AIza...
```

**Security Note**: The `.env` file is automatically excluded from git. Never commit API keys!

### Registry Structure

The orchestrator uses `~/.orchestrator/` for configuration:

```
~/.orchestrator/
├── .env                  # API keys (secure, not committed)
├── skills/               # Skills registry
│   ├── registry.yaml     # Skills catalog
│   └── [skill-name]/     # Individual skill directories
│       ├── skill.yaml    # Skill definition
│       ├── implementation.py  # Python implementation
│       └── tests/        # Skill tests
└── models/               # Models registry
    └── registry.yaml     # Model configurations
```

This is automatically created on first run.

---

## Skills System

### What are Skills?

Skills are reusable capabilities that can be automatically created and integrated into pipelines. Think of them as intelligent, self-documenting functions.

### Automatic Skill Creation

When a pipeline references a skill that doesn't exist, the orchestrator automatically creates it using the **ROMA pattern**:

1. **Atomize**: Break capability into discrete tasks
2. **Plan**: Design skill structure and parameters
3. **Execute**: Generate Python implementation
4. **Aggregate**: Review and refine through iterations

### Example: Automatic Creation

```yaml
steps:
  - id: analyze_code
    tool: code-analyzer  # If skill doesn't exist, it's created!
    action: llm_generate
    parameters:
      directory: "./src"
      prompt: "Analyze code structure"
```

The orchestrator will:
1. Detect that `code-analyzer` skill is missing
2. Use Claude Sonnet to create the skill
3. Use Claude Opus to review it
4. Test with real data
5. Register in `~/.orchestrator/skills/`

### Manual Skill Management

```python
from orchestrator.skills import SkillRegistry, SkillCreator

# List available skills
registry = SkillRegistry()
skills = registry.list_skills()

# Create a new skill manually
creator = SkillCreator()
skill = await creator.create_skill(
    capability="Extract tables from PDF files",
    pipeline_context={"purpose": "data_extraction"}
)

# Search for skills
results = registry.search("pdf")
```

---

## Pipeline Authoring

### Basic Pipeline Structure

```yaml
id: my-pipeline
name: "My First Pipeline"
description: "What this pipeline does"
version: "1.0.0"

inputs:
  parameter_name:
    type: string
    description: "Input description"
    default: "default value"

steps:
  - id: step1
    name: "First Step"
    action: llm_generate
    parameters:
      prompt: "Task with {{ parameter_name }}"
      model: claude-3-5-sonnet-20241022
      max_tokens: 1000

  - id: step2
    name: "Second Step"
    action: llm_generate
    dependencies: [step1]  # Runs after step1
    parameters:
      prompt: "Process {{ step1.result }}"
      model: claude-3-5-sonnet-20241022
      max_tokens: 1000
```

### Using Skills in Pipelines

```yaml
steps:
  - id: web_search
    tool: web-searcher  # References a skill
    action: llm_generate
    parameters:
      query: "{{ search_term }}"
      max_results: 10
      prompt: "Search for {{ search_term }}"
```

### Control Flow

#### Sequential Steps (Dependencies)

```yaml
steps:
  - id: step_a
    action: llm_generate
    parameters:
      prompt: "First task"

  - id: step_b
    dependencies: [step_a]  # Waits for step_a
    action: llm_generate
    parameters:
      prompt: "Uses {{ step_a.result }}"
```

#### Parallel Steps

```yaml
steps:
  - id: task_a
    action: llm_generate
    parameters:
      prompt: "Independent task A"

  - id: task_b
    action: llm_generate
    parameters:
      prompt: "Independent task B (runs in parallel with A)"

  - id: combine
    dependencies: [task_a, task_b]  # Waits for both
    action: llm_generate
    parameters:
      prompt: "Combine {{ task_a.result }} and {{ task_b.result }}"
```

#### Conditional Execution

```yaml
steps:
  - id: check_condition
    action: llm_generate
    parameters:
      prompt: "Determine if we should proceed"

  - id: conditional_step
    condition: "{{ check_condition.should_proceed == true }}"
    action: llm_generate
    parameters:
      prompt: "Only runs if condition is true"
```

#### Loops

```yaml
steps:
  - id: process_items
    for_each: "{{ items_list }}"
    steps:
      - id: process_item
        action: llm_generate
        parameters:
          prompt: "Process {{ $item }}"
```

### Output Artifacts

Save step outputs to files:

```yaml
steps:
  - id: generate_report
    action: llm_generate
    parameters:
      prompt: "Generate analysis report"
    produces: markdown_report
    location: "./output/report.md"
```

---

## Model Selection

### Available Models

The orchestrator uses Anthropic Claude models:

| Model | Context | Best For | Cost |
|-------|---------|----------|------|
| **claude-opus-4.1** | 200K | Deep analysis, reviews | $$$ |
| **claude-sonnet-4.5** | 1M | Orchestration, coding | $$ |
| **claude-haiku-4.5** | 200K | Simple tasks, validation | $ |

*Note: 2025 models will automatically be used when available, falling back to current models.*

### Model Selection Strategies

#### Explicit Selection

```yaml
steps:
  - id: analysis
    action: llm_generate
    parameters:
      prompt: "Analyze this data"
      model: claude-opus-4-1-20250805  # Explicit model
```

#### By Task Type

```yaml
# Use Sonnet for coding
- id: generate_code
  action: llm_generate
  parameters:
    prompt: "Write a Python function"
    model: claude-sonnet-4-5  # Best for code

# Use Haiku for simple tasks
- id: format_output
  action: llm_generate
  parameters:
    prompt: "Format this JSON"
    model: claude-haiku-4-5  # Fast and cheap

# Use Opus for critical analysis
- id: review_code
  action: llm_generate
  parameters:
    prompt: "Comprehensive code review"
    model: claude-opus-4.1  # Highest quality
```

#### Conditional Selection

```yaml
parameters:
  model: "{% if complexity == 'high' %}claude-opus-4.1{% else %}claude-sonnet-4.5{% endif %}"
```

---

## Examples

### Example 1: Code Review Pipeline

Analyzes code and generates a review report.

```yaml
id: code-review
name: "Code Review"
version: "1.0.0"

inputs:
  code_directory:
    type: string
    default: "./src"

steps:
  - id: analyze
    action: llm_generate
    parameters:
      prompt: "Analyze code in {{ code_directory }}"
      model: claude-3-5-sonnet-20241022
      max_tokens: 2000

  - id: generate_report
    dependencies: [analyze]
    action: llm_generate
    parameters:
      prompt: "Create review report from {{ analyze.result }}"
      model: claude-3-5-sonnet-20241022
      max_tokens: 3000
    produces: report
    location: "./code_review.md"
```

**Run it:**
```bash
orchestrator run examples/claude_skills_refactor/01_simple_code_review.yaml
```

### Example 2: Research Synthesis

Researches a topic and synthesizes findings.

```yaml
id: research
name: "Research Synthesis"
version: "1.0.0"

inputs:
  topic:
    type: string
    default: "LangGraph agent frameworks"

steps:
  - id: plan
    action: llm_generate
    parameters:
      prompt: "Create research plan for {{ topic }}"
      model: claude-3-5-sonnet-20241022

  - id: research
    dependencies: [plan]
    action: llm_generate
    parameters:
      prompt: "Research based on plan: {{ plan.result }}"
      model: claude-3-5-sonnet-20241022
      max_tokens: 4000

  - id: synthesize
    dependencies: [research]
    action: llm_generate
    parameters:
      prompt: "Synthesize findings: {{ research.result }}"
      model: claude-opus-4-1-20250805  # Use Opus for synthesis
      max_tokens: 5000
    produces: report
    location: "./research_report.md"
```

**Run it:**
```bash
orchestrator run examples/claude_skills_refactor/02_research_synthesis.yaml \
  --input topic="Your research topic"
```

### Example 3: Parallel Data Processing

Processes multiple data sources in parallel.

```yaml
id: parallel-processing
name: "Parallel Data Processing"
version: "1.0.0"

inputs:
  processing_mode:
    type: string
    default: "fast"

steps:
  # These three steps run in parallel (no dependencies)
  - id: process_source_1
    action: llm_generate
    parameters:
      prompt: "Process source 1"
      model: "{% if processing_mode == 'fast' %}claude-haiku-4-5{% else %}claude-3-5-sonnet-20241022{% endif %}"

  - id: process_source_2
    action: llm_generate
    parameters:
      prompt: "Process source 2"
      model: "{% if processing_mode == 'fast' %}claude-haiku-4-5{% else %}claude-3-5-sonnet-20241022{% endif %}"

  - id: process_source_3
    action: llm_generate
    parameters:
      prompt: "Process source 3"
      model: "{% if processing_mode == 'fast' %}claude-haiku-4-5{% else %}claude-3-5-sonnet-20241022{% endif %}"

  # This step waits for all three
  - id: aggregate
    dependencies: [process_source_1, process_source_2, process_source_3]
    action: llm_generate
    parameters:
      prompt: "Aggregate results"
      model: claude-3-5-sonnet-20241022
    produces: results
    location: "./aggregated.json"
```

**Run it:**
```bash
# Fast mode (uses Haiku)
orchestrator run examples/claude_skills_refactor/03_parallel_data_processing.yaml

# Thorough mode (uses Sonnet)
orchestrator run examples/claude_skills_refactor/03_parallel_data_processing.yaml \
  --input processing_mode="thorough"
```

---

## API Reference

### Python API

#### Creating Skills

```python
from orchestrator.skills import SkillCreator

# Create a new skill
creator = SkillCreator()
skill = await creator.create_skill(
    capability="Extract structured data from text",
    pipeline_context={"purpose": "data_extraction"},
    max_iterations=3  # Max review iterations
)
```

#### Managing Skills

```python
from orchestrator.skills import SkillRegistry

# Initialize registry
registry = SkillRegistry()

# List all skills
all_skills = registry.list_skills()

# Search for skills
results = registry.search("web")

# Get skill details
skill = registry.get("web-searcher")

# Check if skill exists
exists = registry.exists("code-analyzer")

# Get statistics
stats = registry.get_statistics()
```

#### Compiling Pipelines

```python
from orchestrator.compiler import EnhancedSkillsCompiler

# Create compiler
compiler = EnhancedSkillsCompiler()

# Compile a pipeline
with open("pipeline.yaml") as f:
    yaml_content = f.read()

pipeline = await compiler.compile(
    yaml_content,
    context={"input_param": "value"},
    auto_create_missing_skills=True
)

# Get compilation stats
stats = compiler.get_compilation_stats()
print(f"Auto-created {stats['skills_auto_created']} skills")
```

#### Model Registry

```python
from orchestrator.models import ModelRegistry
from orchestrator.models.providers import ProviderConfig

# Create registry
registry = ModelRegistry()

# Configure Anthropic provider
registry.configure_provider(
    provider_name="anthropic",
    provider_type="anthropic",
    config={"api_key": "your-key"}
)

# Initialize
await registry.initialize()

# Get available models
models = registry.available_models

# Health check
health = await registry.health_check()
```

---

## Advanced Features

### Template Variables

Use Jinja2 templates in your pipelines:

```yaml
parameters:
  # Variable substitution
  query: "{{ user_input }}"

  # Conditionals
  model: "{% if urgent %}claude-opus-4.1{% else %}claude-haiku-4.5{% endif %}"

  # Loops in prompts
  prompt: |
    Process these items:
    {% for item in item_list %}
    - {{ item }}
    {% endfor %}

  # Step outputs
  text: "{{ previous_step.result }}"

  # Nested access
  data: "{{ step1.output.nested.field }}"
```

### Error Handling

```yaml
steps:
  - id: risky_step
    action: llm_generate
    parameters:
      prompt: "Complex task that might fail"
    on_failure: continue  # Options: continue, fail, retry, skip
    max_retries: 3
    timeout: 300  # seconds
```

### Real-World Testing

All skills are tested with real resources (NO MOCKS):

```python
from orchestrator.skills import RealWorldSkillTester

# Test a skill
tester = RealWorldSkillTester()
results = await tester.test_skill(
    skill,
    test_cases=[
        {
            "description": "Test with real data",
            "input": {"data": "https://example.com/data.json"},
            "expected_format": "object"
        }
    ]
)

print(f"Passed: {results['summary']['passed']}/{results['summary']['total']}")
```

---

## Best Practices

### 1. Model Selection

- **Haiku 4.5**: Simple formatting, validation, quick tasks
- **Sonnet 4.5**: Code generation, analysis, orchestration
- **Opus 4.1**: Critical reviews, complex reasoning, final synthesis

### 2. Pipeline Design

- **Break down complex tasks** into sequential steps
- **Use parallel execution** for independent operations
- **Add dependencies** explicitly for clarity
- **Include descriptions** for maintainability

### 3. Skills

- **Let skills auto-create** for common operations
- **Test skills** with real data before production use
- **Review auto-created skills** in `~/.orchestrator/skills/`
- **Export/share skills** for reuse across projects

### 4. Error Handling

- **Set timeouts** for long-running operations
- **Configure retries** for flaky operations
- **Use on_failure** to control error behavior
- **Test error paths** with real scenarios

### 5. Cost Optimization

- **Use Haiku** for simple, high-volume tasks
- **Use Sonnet** for balanced quality/cost
- **Reserve Opus** for critical decisions
- **Monitor costs** through pipeline stats

---

## Troubleshooting

### API Key Issues

```bash
# Verify API key is set
cat ~/.orchestrator/.env | grep ANTHROPIC_API_KEY

# Test API key
python -c "from orchestrator.utils.api_keys_flexible import ensure_api_key; print(ensure_api_key('anthropic')[:20])"
```

### Docker Issues

Docker starts automatically, but if issues persist:

```bash
# Check Docker status
docker info

# Start Docker Desktop (macOS)
open -a Docker

# Start Docker daemon (Linux)
sudo systemctl start docker
```

### Skill Creation Fails

Check:
1. API key is valid
2. Sufficient API rate limits
3. Network connectivity
4. Logs: `~/.orchestrator/logs/`

### Model Not Found

If 2025 models (Opus 4.1, Sonnet 4.5, Haiku 4.5) return errors:
- They may not be released yet
- System automatically falls back to current models (Claude 3.5 Sonnet, Claude 3 Haiku)

---

## CLI Commands

```bash
# Run a pipeline
orchestrator run <pipeline.yaml> [--input key=value ...]

# Compile without executing
orchestrator compile <pipeline.yaml>

# List available skills
orchestrator skills list

# Show skill details
orchestrator skills show <skill-name>

# Export a skill
orchestrator skills export <skill-name> <output.yaml>

# Check version
orchestrator --version

# View help
orchestrator --help
```

---

## Additional Resources

- **Examples**: See `examples/claude_skills_refactor/` for working pipelines
- **Technical Design**: `TECHNICAL_DESIGN_CLAUDE_SKILLS_REFACTOR_V2.md`
- **API Documentation**: `docs/api_reference.md`
- **GitHub Issues**: Report bugs at github.com/ContextLab/orchestrator/issues

---

## Quick Reference

### Model Codes

```yaml
# Current models (available now)
claude-3-5-sonnet-20241022  # Best current model
claude-3-haiku-20240307     # Fast & cheap

# 2025 models (with fallback)
claude-opus-4.1            # → claude-3-5-sonnet-20241022
claude-sonnet-4.5          # → claude-3-5-sonnet-20241022
claude-haiku-4.5           # → claude-3-haiku-20240307
```

### Common Actions

- `llm_generate`: Generate text with LLM
- `execute_skill`: Execute a registered skill
- `control_flow`: Advanced flow control
- `for_each_runtime`: Runtime loop expansion

### Template Variables

- `{{ variable }}`: Variable substitution
- `{{ step.result }}`: Step output
- `{% if condition %}{% endif %}`: Conditionals
- `{% for item in list %}{% endfor %}`: Loops
- `$item`, `$index`: Loop variables

---

**Version**: 1.0.0 (Claude Skills Refactor)
**Last Updated**: 2024-01-15
**License**: MIT