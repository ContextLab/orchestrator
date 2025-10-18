# Orchestrator Framework - Claude Skills Edition

> **A powerful framework for building multi-agent workflows with Anthropic's Claude models**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

## Overview

Orchestrator is a framework for building sophisticated AI workflows using Anthropic's Claude models. It features **automatic skill creation**, **intelligent model selection**, and **advanced control flow** - all powered by Claude's latest capabilities.

### Why Orchestrator?

- **🎯 Automatic Skill Creation**: Skills are created on-demand using Claude's reasoning
- **🤖 Latest Claude Models**: Opus 4.1, Sonnet 4.5, Haiku 4.5 (with intelligent fallbacks)
- **🔄 ROMA Pattern**: Atomize → Plan → Execute → Aggregate for quality skills
- **📝 YAML Pipelines**: Simple, readable workflow definitions
- **⚡ Real-World Testing**: All skills tested with real APIs (no mocks)
- **🧠 Control Flow**: Loops, conditionals, parallel execution built-in

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/ContextLab/orchestrator.git
cd orchestrator
pip install -e .
```

### 2. Configure

```bash
# Set up your Anthropic API key
mkdir -p ~/.orchestrator
echo "ANTHROPIC_API_KEY=your-key-here" >> ~/.orchestrator/.env
```

### 3. Run

```bash
# Run an example pipeline
python scripts/execution/run_pipeline.py examples/claude_skills_refactor/01_simple_code_review.yaml
```

**That's it!** The orchestrator will automatically:
- Create required skills using Claude
- Select optimal models for each task
- Execute your workflow and save outputs

---

## Claude Models

### Model Lineup

| Model | Context | Role | Best For |
|-------|---------|------|----------|
| **Opus 4.1** | 200K | Review & Analysis | Deep analysis, critical reviews, complex reasoning |
| **Sonnet 4.5** | 1M | Orchestrator | Code generation, agent building, general tasks |
| **Haiku 4.5** | 200K | Simple Tasks | Quick validation, formatting, high-volume operations |

### Automatic Fallbacks

2025 models automatically fall back to current available models:
- `claude-opus-4.1` → `claude-3-5-sonnet-20241022`
- `claude-sonnet-4.5` → `claude-3-5-sonnet-20241022`
- `claude-haiku-4.5` → `claude-3-haiku-20240307`

---

## Skills System

### What Are Skills?

Skills are self-contained capabilities created automatically when your pipeline needs them.

**Example**: Your pipeline references `web-searcher`
→ Orchestrator detects it's missing
→ Uses Claude to create the skill via ROMA pattern
→ Saves to `~/.orchestrator/skills/web-searcher/`
→ Skill is now available for all future pipelines

### ROMA Pattern

Every skill is created through 4 stages:

1. **Atomize**: Claude breaks the capability into atomic tasks
2. **Plan**: Designs skill structure, parameters, outputs
3. **Execute**: Generates Python implementation
4. **Aggregate**: Opus reviews and refines iteratively

**Result**: Production-ready skills validated with real data!

---

## Pipeline Examples

### Example 1: Simple Code Review

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

  - id: report
    dependencies: [analyze]
    action: llm_generate
    parameters:
      prompt: "Create review report: {{ analyze.result }}"
      model: claude-3-5-sonnet-20241022
    produces: report
    location: "./review.md"
```

### Example 2: Parallel Processing

```yaml
id: parallel-tasks
name: "Parallel Processing"
version: "1.0.0"

steps:
  # These run simultaneously
  - id: task_a
    action: llm_generate
    parameters:
      prompt: "Process dataset A"
      model: claude-haiku-4-5

  - id: task_b
    action: llm_generate
    parameters:
      prompt: "Process dataset B"
      model: claude-haiku-4-5

  # Waits for both
  - id: combine
    dependencies: [task_a, task_b]
    action: llm_generate
    parameters:
      prompt: "Combine {{ task_a.result }} and {{ task_b.result }}"
      model: claude-3-5-sonnet-20241022
```

### Example 3: Quality Synthesis

```yaml
id: synthesis
name: "Research Synthesis"
version: "1.0.0"

steps:
  - id: research
    action: llm_generate
    parameters:
      prompt: "Research {{ topic }}"
      model: claude-3-5-sonnet-20241022
      max_tokens: 4000

  - id: synthesize
    dependencies: [research]
    action: llm_generate
    parameters:
      prompt: "Create synthesis: {{ research.result }}"
      model: claude-opus-4-1-20250805  # Use Opus for quality
      max_tokens: 5000
    produces: report
    location: "./synthesis.md"
```

---

## Python API

### Basic Usage

```python
import orchestrator as orc

# Initialize models
orc.init_models()

# Compile pipeline
pipeline = orc.compile("my_pipeline.yaml")

# Run with inputs
result = pipeline.run(topic="AI agents", depth="comprehensive")

print(result)
```

### Skills Management

```python
from orchestrator.skills import SkillCreator, SkillRegistry

# Create a skill
creator = SkillCreator()
skill = await creator.create_skill(
    capability="Extract tables from PDFs"
)

# Manage skills
registry = SkillRegistry()
skills = registry.list_skills()
results = registry.search("pdf")
```

### Advanced Compilation

```python
from orchestrator.compiler import EnhancedSkillsCompiler

# Create compiler
compiler = EnhancedSkillsCompiler()

# Compile with auto-skill-creation
with open("pipeline.yaml") as f:
    pipeline = await compiler.compile(
        f.read(),
        context={"param": "value"},
        auto_create_missing_skills=True
    )

# Check what was created
stats = compiler.get_compilation_stats()
print(f"Auto-created skills: {stats['created_skill_names']}")
```

---

## Features

### ✅ Implemented

- **Automatic Skill Creation** - ROMA pattern with Claude
- **Multi-Agent Workflows** - Orchestrator + Reviewer pattern
- **Advanced Control Flow** - Loops, conditionals, parallel execution
- **Intelligent Model Selection** - Right model for each task
- **Real-World Testing** - All skills tested with real APIs
- **Comprehensive Validation** - 6-layer validation system
- **Output Artifacts** - Save to files (MD, JSON, etc.)
- **Template Engine** - Jinja2 for dynamic configurations
- **Docker Integration** - Auto-install and auto-start
- **Secure API Keys** - Encrypted storage in ~/.orchestrator

### 🔄 Control Flow

- **Sequential**: Dependencies between steps
- **Parallel**: Independent steps run simultaneously
- **For-Each**: Loop over lists of items
- **While**: Conditional loops
- **Conditionals**: If/else logic
- **Dynamic Flow**: Goto and runtime routing

---

## Directory Structure

```
orchestrator/
├── src/orchestrator/
│   ├── models/              # Model management (Anthropic-only)
│   ├── skills/              # Skills system
│   │   ├── creator.py       # ROMA pattern skill creation
│   │   ├── tester.py        # Real-world testing
│   │   └── registry.py      # Skill management
│   ├── compiler/            # Pipeline compilation
│   │   ├── enhanced_skills_compiler.py  # Main compiler
│   │   └── control_flow_compiler.py     # Control flow
│   ├── execution/           # LangGraph execution engine
│   └── utils/               # Utilities
├── examples/claude_skills_refactor/  # Example pipelines
├── tests/                   # Test suite
└── docs/                    # Documentation

~/.orchestrator/             # User configuration
├── .env                     # API keys (secure)
├── skills/                  # Skills registry
└── models/                  # Model configurations
```

---

## Configuration Files

### ~/.orchestrator/.env

```bash
# Your Anthropic API key (required)
ANTHROPIC_API_KEY=sk-ant-api03-...
```

### ~/.orchestrator/skills/registry.yaml

```yaml
version: "1.0.0"
skills:
  web-searcher:
    name: "web-searcher"
    description: "Search and extract web information"
    version: "1.0.0"
    capabilities: ["search", "extract"]
    path: "./web-searcher/"
```

### ~/.orchestrator/models/registry.yaml

```yaml
version: "1.0.0"
models:
  claude-opus-4.1:
    role: "review_and_analysis"
    context_window: 200000

  claude-sonnet-4.5:
    role: "orchestrator"
    context_window: 1000000

  claude-haiku-4.5:
    role: "simple_tasks"
    context_window: 200000
```

---

## Testing

### Run Integration Tests

```bash
# Run all Claude Skills tests
python -m pytest tests/integration/test_claude_skills_refactor.py -v

# Run specific test
python -m pytest tests/integration/test_claude_skills_refactor.py::test_skill_creation_with_roma -v
```

### Test Your Pipelines

```bash
# Compile only (no execution)
orchestrator compile my_pipeline.yaml

# Run with validation
orchestrator run my_pipeline.yaml --validate

# Show compilation stats
orchestrator compile my_pipeline.yaml --show-stats
```

---

## Documentation

- **[User Guide](docs/CLAUDE_SKILLS_USER_GUIDE.md)** - Complete documentation
- **[Quick Start](docs/QUICK_START.md)** - 5-minute tutorial
- **[Examples](examples/claude_skills_refactor/)** - Working pipelines
- **[API Reference](docs/api_reference.md)** - Python API docs

---

## Support

- **📚 Documentation**: `docs/CLAUDE_SKILLS_USER_GUIDE.md`
- **💡 Examples**: `examples/claude_skills_refactor/`
- **🐛 Issues**: https://github.com/ContextLab/orchestrator/issues
- **📧 Email**: contextualdynamics@gmail.com

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Citation

```bibtex
@software{orchestrator2025,
  title = {Orchestrator: Claude Skills Framework},
  author = {Manning, Jeremy R. and {Contextual Dynamics Lab}},
  year = {2025},
  url = {https://github.com/ContextLab/orchestrator},
  organization = {Dartmouth College}
}
```

---

**Built with Claude by the Contextual Dynamics Lab**