# Orchestrator Framework

[![PyPI Version](https://img.shields.io/pypi/v/py-orc)](https://pypi.org/project/py-orc/)
[![Python Versions](https://img.shields.io/pypi/pyversions/py-orc)](https://pypi.org/project/py-orc/)
[![Downloads](https://img.shields.io/pypi/dm/py-orc)](https://pypi.org/project/py-orc/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/ContextLab/orchestrator/blob/main/LICENSE)
[![Tests](https://github.com/ContextLab/orchestrator/actions/workflows/tests.yml/badge.svg)](https://github.com/ContextLab/orchestrator/actions/workflows/tests.yml)
[![Coverage](https://github.com/ContextLab/orchestrator/actions/workflows/coverage.yml/badge.svg)](https://github.com/ContextLab/orchestrator/actions/workflows/coverage.yml)
[![Documentation](https://readthedocs.org/projects/orc/badge/?version=latest)](https://orc.readthedocs.io/en/latest/?badge=latest)

## Overview

Orchestrator is a powerful, flexible AI pipeline orchestration framework that simplifies the creation and execution of complex AI workflows. By combining YAML-based configuration with intelligent model selection and automatic ambiguity resolution, Orchestrator makes it easy to build sophisticated AI applications without getting bogged down in implementation details.

### Key Features

- 🎯 **YAML-Based Pipelines**: Define complex workflows in simple, readable YAML
- 🤖 **Multi-Model Support**: Seamlessly work with OpenAI, Anthropic, Google, Ollama, and HuggingFace models
- 🧠 **Intelligent Model Selection**: Automatically choose the best model based on task requirements
- 🔄 **Automatic Ambiguity Resolution**: Use `<AUTO>` tags to let AI resolve configuration ambiguities
- 📦 **Modular Architecture**: Extend with custom models, tools, and control systems
- 🛡️ **Production Ready**: Built-in error handling, retries, checkpointing, and monitoring
- ⚡ **Parallel Execution**: Efficient resource management and parallel task execution
- 🐳 **Sandboxed Execution**: Secure code execution in isolated environments
- 💾 **Lazy Model Loading**: Models are downloaded only when needed, saving disk space

## Quick Start

### Installation

```bash
pip install py-orc
```

For additional features:
```bash
pip install py-orc[ollama]      # Ollama model support
pip install py-orc[cloud]        # Cloud model providers
pip install py-orc[dev]          # Development tools
pip install py-orc[all]          # Everything
```

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

```python
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
    name: llama3.1:8b
    expertise: [general, reasoning, multilingual]
    size: 8b
    
  - source: ollama
    name: qwen2.5-coder:7b
    expertise: [code, programming]
    size: 7b

  # Cloud models
  - source: openai
    name: gpt-4o
    expertise: [general, reasoning, code, analysis, vision]
    size: 1760b  # Estimated

defaults:
  expertise_preferences:
    code: qwen2.5-coder:7b
    reasoning: deepseek-r1:8b
    fast: llama3.2:1b
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

## Documentation

Comprehensive documentation is available at [orc.readthedocs.io](https://orc.readthedocs.io/), including:

- [Getting Started Guide](https://orc.readthedocs.io/en/latest/getting_started/quickstart.html)
- [YAML Configuration Reference](https://orc.readthedocs.io/en/latest/user_guide/yaml_configuration.html)
- [Model Configuration](https://orc.readthedocs.io/en/latest/user_guide/model_configuration.html)
- [API Reference](https://orc.readthedocs.io/en/latest/api/core.html)
- [Examples and Tutorials](https://orc.readthedocs.io/en/latest/tutorials/examples.html)

## Available Models

Orchestrator supports a wide range of models:

### Local Models (via Ollama)
- **Llama 3.x**: General purpose, multilingual support
- **DeepSeek-R1**: Advanced reasoning and coding
- **Qwen2.5-Coder**: Specialized for code generation
- **Mistral**: Fast and efficient general purpose
- **Gemma3**: Google's efficient models in various sizes

### Cloud Models
- **OpenAI**: GPT-4.1, GPT-4o, o3, o4-mini (reasoning models)
- **Anthropic**: Claude 4 Opus/Sonnet, Claude 3.7 Sonnet
- **Google**: Gemini 2.5 Pro/Flash, Gemini 2.0 series

### HuggingFace Models
- Llama, Qwen, Phi, and many more
- Automatically downloaded on first use

## Requirements

- Python 3.8+
- Optional: Ollama for local model execution
- Optional: API keys for cloud providers (OpenAI, Anthropic, Google)

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

## Acknowledgments

Orchestrator is developed and maintained by the [Contextual Dynamics Lab](https://www.context-lab.com/) at Dartmouth College.

---

*Built with ❤️ by the Contextual Dynamics Lab*