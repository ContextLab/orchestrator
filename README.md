# Orchestrator Framework

[![PyPI Version](https://img.shields.io/pypi/v/py-orc)](https://pypi.org/project/py-orc/)
[![Python Versions](https://img.shields.io/pypi/pyversions/py-orc)](https://pypi.org/project/py-orc/)
[![Downloads](https://img.shields.io/pypi/dm/py-orc)](https://pypi.org/project/py-orc/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/ContextLab/orchestrator/blob/main/LICENSE)
[![CI](https://github.com/ContextLab/orchestrator/actions/workflows/ci.yml/badge.svg)](https://github.com/ContextLab/orchestrator/actions/workflows/ci.yml)
[![Documentation](https://readthedocs.org/projects/orc/badge/?version=latest)](https://orc.readthedocs.io/en/latest/?badge=latest)

## Project status: alpha

This project is **alpha** and under active recovery. Treat the supported
surface as small and everything else as experimental.

**Verified** — covered by the blocking CI gate, which runs the
`unit`/`contract`/`e2e` layer with no network, no API keys and no optional
extras installed, plus a smoke test of a golden pipeline through the
*installed wheel*:

- Compiling a YAML pipeline into a task graph
- Executing sequential and parallel steps with dependency ordering
- Jinja-style template interpolation, including cross-step references
- Deterministic local tools (filesystem, data-processing, validation)
- `orchestrator run` / `orchestrator validate` and the equivalent Python API

**Present in the tree but NOT verified**, and therefore not claimed to work:
OpenAI / Google / HuggingFace / Ollama adapters, multimodal tooling, the web
dashboard, monitoring and analytics, MCP integration, and the deployment
tooling. Anthropic is the first provider being brought under live acceptance
tests; provider support is only advertised here once the live-provider
workflow passes.

**The wider legacy test suite is not green.** Only the marked
`unit`/`contract`/`e2e` layer gates the build. The remainder were written
against several superseded architectures, many fail, and some hang; they run
in a separate, deliberately **non-blocking** CI job so the size of that
backlog stays visible instead of being hidden behind a green check. Do not
read "CI passing" as "the whole suite passes". Current counts are reported by
that job — see [#354](https://github.com/ContextLab/orchestrator/issues/354)
rather than a number maintained by hand here, which has been wrong before.

**Free models via Dartmouth Chat.** If you have a
[Dartmouth Chat](https://chat.dartmouth.edu/) account, the
`DartmouthProvider` runs pipelines against real models at **zero cost per
token** — no provider extra needed, since the gateway is OpenAI-compatible and
the adapter speaks it with `aiohttp` (already a core dependency):

```python
from orchestrator import DartmouthProvider

provider = DartmouthProvider()
await provider.initialize()
print(provider.list_free_models())
# ['google.gemma-3-27b-it', 'google.gemma-4-31B-it',
#  'meta.llama-3-2-3b-instruct', 'meta.llama-3.2-11b-vision-instruct',
#  'openai.gpt-oss-120b', 'qwen.qwen3-vl:32b', 'qwen.qwen3.5-122b']

text, model_used = await provider.generate_free("Summarize this in one line: ...")
```

Set `DARTMOUTH_CHAT_API_KEY`, or put it in `~/.orchestrator/.env`. Free/paid
status comes from the live catalog, and **paid models are refused unless
`ORCHESTRATOR_ALLOW_PAID_MODELS=1`**, so a model-name typo cannot quietly
start spending. Prefer `generate_free()` over naming one model: the free
endpoints are individually hosted and go down independently, and it falls
through the free set until one answers.

**Security.** Two confirmed remote-code-execution defects have been fixed
(pipeline conditions and `transform_spec` both reached `eval`). Pipeline
content is now evaluated by a constrained, fail-closed AST evaluator, and
`src/orchestrator` contains no live `eval()` call sites. Pipeline YAML is
still **trusted input**: run only pipelines you would run as code, and see
[CONTRIBUTING.md](CONTRIBUTING.md) before touching the evaluator.

The scope, canonical code path, and the criteria for promoting anything out of
"unverified" are recorded in
[docs/adr/0001-product-contract.md](docs/adr/0001-product-contract.md).

## Overview

Orchestrator is an AI pipeline orchestration framework built around YAML
workflow definitions. It combines a declarative pipeline language with model
selection and `<AUTO>` ambiguity resolution, so workflows can be described
without hand-writing the execution plumbing.

### Key features

- 🎯 **YAML-based pipelines** — declare workflows with full template variable support
- 🔄 **`<AUTO>` ambiguity resolution** — let a model resolve configuration choices
- ⚡ **Parallel execution** — independent steps run concurrently, dependencies are ordered
- 📦 **Modular architecture** — extend with custom models, tools, and control systems
- 🔒 **Fail-closed conditions** — pipeline conditions run in a constrained expression
  language, never `eval()`; a condition that cannot be evaluated does not run its step
- 🪶 **Light import** — `import orchestrator` pulls in 12 dependencies, not 40;
  optional features live behind extras and are imported only when used
- ✅ **Validation framework** — pipelines, dependencies, and data flow are checked
  before execution

## Quick Start

### Installation

```bash
pip install py-orc
```

The base install is deliberately small and is all you need to compile and run
pipelines built from deterministic local tools — no API key required.

Optional features are grouped into extras:

```bash
pip install "py-orc[anthropic]"    # Anthropic provider
pip install "py-orc[openai]"       # OpenAI provider (unverified)
pip install "py-orc[google]"       # Google provider (unverified)
pip install "py-orc[langgraph]"    # LangGraph state/checkpoint backends
pip install "py-orc[web]"          # web search and browser tools
pip install "py-orc[multimedia]"   # image/audio/video tools
pip install "py-orc[viz]"          # plotting and report figures
pip install "py-orc[infra]"        # Docker, Redis, Postgres backends
pip install "py-orc[dev]"          # development and test tooling
pip install "py-orc[all]"          # every runtime extra
```

A missing extra disables only the feature that needs it; it never breaks
`import orchestrator`.

### Run your first pipeline

Save this as `hello.yaml`:

```yaml
id: hello
name: Hello Pipeline

parameters:
  greeting:
    type: string
    default: "hello"

steps:
  - id: write_greeting
    tool: filesystem
    action: write
    parameters:
      path: "./out/greeting.txt"
      content: "{{ greeting }} world"

  - id: read_back
    tool: filesystem
    action: read
    parameters:
      path: "./out/greeting.txt"
    dependencies:
      - write_greeting
```

Then:

```bash
orchestrator validate hello.yaml          # compile only; prints the task graph
orchestrator run hello.yaml -i greeting=hi
cat out/greeting.txt                      # -> hi world
```

Exit codes: `0` success, `1` execution failure, `2` validation failure,
`130` interrupted.

The equivalent Python API:

```python
import asyncio

from orchestrator import Orchestrator
from orchestrator.control_systems.tool_integrated_control_system import (
    ToolIntegratedControlSystem,
)

async def main():
    orchestrator = Orchestrator(control_system=ToolIntegratedControlSystem())
    try:
        results = await orchestrator.execute_yaml_file("hello.yaml", {"greeting": "hi"})
        print(results["read_back"]["result"]["content"])   # -> hi world
    finally:
        await orchestrator.shutdown()

asyncio.run(main())
```

Both surfaces are asserted to produce the same result in
`tests/test_golden_pipelines.py`.

### API Key Configuration

Orchestrator supports multiple AI providers. Configure your API keys using the interactive setup:

```bash
# Interactive API key setup
python scripts/utilities/setup_api_keys.py

# Or set environment variables directly
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_AI_API_KEY="your-google-ai-key"
export HF_TOKEN="your-huggingface-token"
```

API keys are stored securely in `~/.orchestrator/.env` with file permissions set to 600 (owner read/write only).

#### Required Environment Variables

If you prefer to set environment variables manually:

- `OPENAI_API_KEY` - OpenAI API key (for GPT models)
- `ANTHROPIC_API_KEY` - Anthropic API key (for Claude models)
- `GOOGLE_AI_API_KEY` - Google AI API key (for Gemini models)
- `HF_TOKEN` - Hugging Face token (for HuggingFace models)

**Note**: Ollama models run locally and don't require API keys. They will be downloaded automatically on first use.

### Basic Usage

1. **Create a simple pipeline** (`hello_world.yaml`):

```yaml
id: hello_world
name: Hello World Pipeline
description: A simple example pipeline

steps:
  - id: greet
    action: generate_text
    parameters:
      prompt: "Say hello to the world in a creative way!"
      
  - id: translate
    action: generate_text
    parameters:
      prompt: "Translate this greeting to Spanish: {{ greet.result }}"
    dependencies: [greet]

outputs:
  greeting: "{{ greet.result }}"
  spanish: "{{ translate.result }}"
```

2. **Run the pipeline**:

```bash
# Using the CLI script
python scripts/execution/run_pipeline.py hello_world.yaml

# With inputs
python scripts/execution/run_pipeline.py hello_world.yaml -i name=World -i language=Spanish

# From a JSON file
python scripts/execution/run_pipeline.py hello_world.yaml -f inputs.json -o output_dir/

# Or programmatically
import orchestrator as orc

# Initialize models (auto-detects available models)
orc.init_models()

# Compile and run the pipeline
pipeline = orc.compile("hello_world.yaml")
result = pipeline.run()

print(result)
```

### Using AUTO Tags

Orchestrator's `<AUTO>` tags let AI decide configuration details:

```yaml
steps:
  - id: analyze_data
    action: analyze
    parameters:
      data: "{{ input_data }}"
      method: <AUTO>Choose the best analysis method for this data type</AUTO>
      visualization: <AUTO>Decide if we should create a chart</AUTO>
```

## Model Configuration

Configure available models in `models.yaml`:

```yaml
models:
  # Local models (via Ollama) - downloaded on first use
  - source: ollama
    name: deepseek-r1:8b
    expertise: [reasoning, code, math]
    size: 8b
    
  - source: ollama
    name: qwen2.5-coder:7b
    expertise: [code, programming]
    size: 7b
    
  - source: ollama
    name: gemma3:12b
    expertise: [general, reasoning, analysis]
    size: 12b

  # Cloud models  
  - source: openai
    name: gpt-5
    expertise: [general, reasoning, code, analysis, vision, multimodal]
    size: 2000b  # Estimated
    
  - source: anthropic
    name: claude-sonnet-4-20250514
    expertise: [general, reasoning, efficient]
    size: 600b  # Estimated
    
  - source: google
    name: gemini-2.5-flash
    expertise: [general, fast, efficient, thinking]
    size: 80b  # Estimated

defaults:
  expertise_preferences:
    code: qwen2.5-coder:32b
    reasoning: deepseek-r1:32b
    fast: llama3.2:1b
    general: llama3.1:8b
    analysis: gemma3:27b
```

Models are downloaded only when first used, saving disk space and initialization time.

## Advanced Example

Here's a more complex example showing model requirements and parallel execution:

```yaml
id: research_pipeline
name: AI Research Pipeline
description: Research a topic and create a comprehensive report

inputs:
  - name: topic
    type: string
    description: Research topic
    
  - name: depth
    type: string
    default: <AUTO>Determine appropriate research depth</AUTO>

steps:
  # Parallel research from multiple sources
  - id: web_search
    action: search_web
    parameters:
      query: "{{ topic }} latest research 2025"
      count: <AUTO>Decide how many results to fetch</AUTO>
    requires_model:
      expertise: [research, web]
      
  - id: academic_search
    action: search_academic
    parameters:
      query: "{{ topic }}"
      filters: <AUTO>Set appropriate academic filters</AUTO>
    requires_model:
      expertise: [research, academic]
      
  # Analyze findings with specialized model
  - id: analyze_findings
    action: analyze
    parameters:
      web_results: "{{ web_search.results }}"
      academic_results: "{{ academic_search.results }}"
      analysis_focus: <AUTO>Determine key aspects to analyze</AUTO>
    dependencies: [web_search, academic_search]
    requires_model:
      expertise: [analysis, reasoning]
      min_size: 20b  # Require large model for complex analysis
      
  # Generate report
  - id: write_report
    action: generate_document
    parameters:
      topic: "{{ topic }}"
      analysis: "{{ analyze_findings.result }}"
      style: <AUTO>Choose appropriate writing style</AUTO>
      length: <AUTO>Determine optimal report length</AUTO>
    dependencies: [analyze_findings]
    requires_model:
      expertise: [writing, general]

outputs:
  report: "{{ write_report.document }}"
  summary: "{{ analyze_findings.summary }}"
```

## Complete Example: Research Report Generator

Here's a fully functional pipeline that generates research reports:

```yaml
# research_report.yaml
id: research_report
name: Research Report Generator
description: Generate comprehensive research reports with citations

inputs:
  - name: topic
    type: string
    description: Research topic
  - name: instructions
    type: string
    description: Additional instructions for the report

outputs:
  - pdf: <AUTO>Generate appropriate filename for the research report PDF</AUTO>

steps:
  - id: search
    name: Web Search
    action: search_web
    parameters:
      query: <AUTO>Create effective search query for {topic} with {instructions}</AUTO>
      max_results: 10
    requires_model:
      expertise: fast
      
  - id: compile_notes
    name: Compile Research Notes
    action: generate_text
    parameters:
      prompt: |
        Compile comprehensive research notes from these search results:
        {{ search.results }}
        
        Topic: {{ topic }}
        Instructions: {{ instructions }}
        
        Create detailed notes with:
        - Key findings
        - Important quotes
        - Source citations
        - Relevant statistics
    dependencies: [search]
    requires_model:
      expertise: [analysis, reasoning]
      min_size: 7b
      
  - id: write_report
    name: Write Report
    action: generate_document
    parameters:
      content: |
        Write a comprehensive research report on "{{ topic }}"
        
        Research notes:
        {{ compile_notes.result }}
        
        Requirements:
        - Professional academic style
        - Include introduction, body sections, and conclusion
        - Cite sources properly
        - {{ instructions }}
      format: markdown
    dependencies: [compile_notes]
    requires_model:
      expertise: [writing, general]
      min_size: 20b
      
  - id: create_pdf
    name: Create PDF
    action: convert_to_pdf
    parameters:
      markdown: "{{ write_report.document }}"
      filename: "{{ outputs.pdf }}"
    dependencies: [write_report]
```

Run it with:

```python
import orchestrator as orc

# Initialize models
orc.init_models()

# Compile pipeline
pipeline = orc.compile("research_report.yaml")

# Run with inputs
result = pipeline.run(
    topic="quantum computing applications in medicine",
    instructions="Focus on recent breakthroughs and future potential"
)

print(f"Report saved to: {result}")
```

## Examples

The `examples/` directory contains working demonstrations of Orchestrator's capabilities. Here's a highlighted example:

### Simple Data Processing Pipeline

The [simple_data_processing.yaml](examples/simple_data_processing.yaml) pipeline demonstrates fundamental concepts:

**What it does:**
- Reads a CSV file containing project data
- Filters records based on criteria (status = "active")
- Generates both filtered data and an analysis report

**Key concepts demonstrated:**
- File I/O with the `filesystem` tool
- Data processing with the `data-processing` tool
- Template variable usage between pipeline steps
- Multi-format output generation

Browse more examples in the [examples directory](examples/) including web research, model routing, recursive processing, and more.

> **Heads-up:** most shipped examples do not currently pass `orchestrator validate` — a sweep found 44 of 45 failing, some because the example is stale and some because the validator produces false positives. Tracked in
> [#104](https://github.com/ContextLab/orchestrator/issues/104) and
> [#241](https://github.com/ContextLab/orchestrator/issues/241). The three
> pipelines under `tests/golden/` are verified and are the reliable starting
> point.

## Documentation

Comprehensive documentation is available at [orc.readthedocs.io](https://orc.readthedocs.io/), including:

- [Getting Started Guide](https://orc.readthedocs.io/en/latest/getting_started/quickstart.html)
- [YAML Configuration Reference](https://orc.readthedocs.io/en/latest/user_guide/yaml_configuration.html)
- [Model Configuration](https://orc.readthedocs.io/en/latest/user_guide/model_configuration.html)
- [API Reference](https://orc.readthedocs.io/en/latest/api/core.html)
- [Examples and Tutorials](https://orc.readthedocs.io/en/latest/tutorials/examples.html)

## Model support

Adapters for several providers exist in the tree, but "an adapter exists" is
not the same as "the provider works". This table reflects test evidence, not
intent:

| Provider | Extra | Status |
|-|-|-|
| Anthropic | `anthropic` | Being brought under live acceptance tests |
| OpenAI | `openai` | Adapter present, unverified |
| Google | `google` | Adapter present, unverified |
| Ollama (local) | — | Adapter present, unverified |
| HuggingFace | — | Adapter present, unverified |

A provider moves out of "unverified" only when the `live-tests` workflow passes
for it. Until then it is not recommended, and the specific model lists that
previously appeared here have been removed rather than left to rot.

**No model is required** to compile and run pipelines built from deterministic
local tools.

## Requirements

- Python 3.11+ (tested on 3.11, 3.12 and 3.13)
- Optional: an API key for a cloud provider, if your pipeline uses one
- Optional: Ollama, for local model execution

## Contributing

We welcome contributions! Please see our [Contributing Guide](https://github.com/ContextLab/orchestrator/blob/main/CONTRIBUTING.md) for details.

## Support

- 📚 [Documentation](https://orc.readthedocs.io/)
- 🐛 [Issue Tracker](https://github.com/ContextLab/orchestrator/issues)
- 💬 [Discussions](https://github.com/ContextLab/orchestrator/discussions)
- 📧 Email: contextualdynamics@gmail.com

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/ContextLab/orchestrator/blob/main/LICENSE) file for details.

## Citation

If you use Orchestrator in your research, please cite:

```bibtex
@software{orchestrator2025,
  title = {Orchestrator: AI Pipeline Orchestration Framework},
  author = {Manning, Jeremy R. and {Contextual Dynamics Lab}},
  year = {2025},
  url = {https://github.com/ContextLab/orchestrator},
  organization = {Dartmouth College}
}
```

## Repository Organization

```
orchestrator/
├── config/                 # Configuration files
│   ├── models.yaml        # Model definitions and configurations
│   ├── orchestrator.yaml  # Main orchestrator settings
│   └── validation_schema.json  # Schema for YAML validation
├── data/                  # Sample data files
├── docs/                  # Documentation
│   ├── tutorials/         # Step-by-step guides
│   ├── api/              # API reference
│   └── user_guide/       # User documentation
├── examples/              # Example pipelines
│   ├── *.yaml            # All example pipeline YAML files
│   ├── data/             # Example data files
│   ├── outputs/          # Generated outputs (gitignored)
│   └── checkpoints/      # Pipeline checkpoints (gitignored)
├── scripts/               # Organized utility scripts
│   ├── execution/        # Pipeline execution scripts
│   │   ├── run_pipeline.py    # Main pipeline runner
│   │   └── quick_run_pipelines.py # Batch pipeline execution
│   ├── validation/       # Pipeline and configuration validation
│   │   ├── validate_all_pipelines.py # Pipeline validation
│   │   ├── quick_validate.py    # Fast validation checks
│   │   └── audit_pipelines.py   # Comprehensive pipeline auditing
│   ├── testing/          # Pipeline execution testing
│   │   ├── test_all_real_pipelines.py # Real-world pipeline tests
│   │   └── test_all_pipelines_with_wrappers.py # Wrapper testing
│   ├── utilities/        # Repository maintenance and utilities
│   │   ├── setup_api_keys.py    # API key configuration
│   │   ├── repository_organizer.py # Repository organization
│   │   └── generate_sample_data.py # Sample data generation
│   ├── production/       # Production deployment and monitoring
│   │   ├── production_deploy.py  # Production deployment
│   │   ├── performance_monitor.py # Performance monitoring
│   │   └── dashboard_generator.py # Monitoring dashboard
│   └── maintenance/      # Output regeneration and verification
│       ├── regenerate_all_outputs.py # Regenerate pipeline outputs
│       └── verify_all_outputs.py     # Verify output integrity
├── src/orchestrator/      # Source code
│   ├── core/             # Core components (Pipeline, Task, UnifiedTemplateResolver)
│   ├── models/           # Model integrations
│   ├── tools/            # Tool implementations
│   ├── compiler/         # YAML compiler and template engine
│   ├── control_systems/  # Execution control systems
│   ├── validation/       # Validation framework
│   └── utils/            # Utilities (OutputSanitizer, etc.)
├── tests/                 # Test suite
│   ├── integration/      # Integration tests
│   ├── local/           # Tests requiring local resources
│   └── test_*.py        # Unit tests
└── venv/                 # Virtual environment (gitignored)
```

## Acknowledgments

Orchestrator is developed and maintained by the [Contextual Dynamics Lab](https://www.context-lab.com/) at Dartmouth College.

---

*Built with ❤️ by the Contextual Dynamics Lab*