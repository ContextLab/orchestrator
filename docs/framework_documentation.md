# Orchestrator Framework Documentation

**An AI pipeline orchestration framework with intelligent ambiguity resolution and tool integration**

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Key Features](#key-features)
3. [Installation & Setup](#installation--setup)
4. [Usage Examples](#usage-examples)
5. [Tool System](#tool-system)
6. [API Reference](#api-reference)
7. [Current Capabilities](#current-capabilities)
8. [Limitations](#limitations)
9. [Development](#development)

## Architecture Overview

The Orchestrator framework is built around a modular architecture that separates concerns between pipeline definition, execution, and tool integration. The core components work together to provide a unified interface for AI workflow orchestration.

### Core Components

#### 1. YAMLCompiler (`src/orchestrator/compiler/`)
- **Purpose**: Parses YAML pipeline definitions and compiles them into executable `Pipeline` objects
- **Key Features**:
  - `<AUTO>` tag resolution using AI models for ambiguous values
  - Jinja2 template processing for dynamic content
  - Runtime vs compile-time template resolution
  - Schema validation with detailed error reporting
  - Dependency graph validation (circular dependency detection)

#### 2. Task & Pipeline Abstractions (`src/orchestrator/core/`)
- **Task**: Core unit of work with:
  - Unique ID and action type
  - Parameters (supports templates and references)
  - Dependencies and status tracking
  - Error handling and retry logic
- **Pipeline**: Collection of tasks with:
  - Execution ordering based on dependencies
  - Input/output definitions for parameterization
  - Metadata and configuration
  - State management for checkpointing

#### 3. ModelRegistry (`src/orchestrator/models/`)
- **Purpose**: Manages available AI models and intelligent selection
- **Features**:
  - Multi-provider support (Ollama, HuggingFace, OpenAI, Anthropic)
  - Upper Confidence Bound (UCB) algorithm for model selection
  - Capability-based matching (reasoning, tool_use, code_generation)
  - Resource requirements tracking (GPU, memory, tokens)
  - Fallback to quantized models when resources are limited

#### 4. ToolRegistry & MCP Integration (`src/orchestrator/tools/`)
- **ToolRegistry**: Central repository for tools with:
  - Automatic tool registration and discovery
  - Schema generation for MCP compatibility
  - Parameter validation and type checking
  - Execution delegation to appropriate tool implementations
- **MCP Server**: Model Context Protocol integration:
  - Automatic server startup when tools are required
  - Tool schema exposure to AI models
  - Bidirectional communication for tool execution
  - Support for multiple concurrent tool operations

#### 5. Control System Adapters (`src/orchestrator/core/control_system.py`)
- **Purpose**: Pluggable execution backends
- **Current Implementations**:
  - MockControlSystem: For testing and development
  - ToolIntegratedControlSystem: With real tool execution
  - LangGraph adapter (planned)
  - Custom adapters supported

#### 6. State Management (`src/orchestrator/state/`)
- **Features**:
  - Automatic checkpointing at task boundaries
  - Recovery from last successful checkpoint
  - State persistence to various backends (PostgreSQL, Redis, file)
  - Adaptive checkpointing based on task criticality

### Data Flow

```
YAML Definition → YAMLCompiler → Pipeline Object → Orchestrator → Control System → Tool Execution
                      ↓                ↓               ↓               ↓
                 Auto Tags      Input Validation   Dependency    Tool Registry
                 Resolution     Template           Resolution    MCP Server
                               Processing
```

## Key Features

### 1. Input-Agnostic Pipelines

Pipelines are designed to be reusable with different inputs, making them truly generic:

```yaml
inputs:
  topic:
    type: string
    description: "Research topic to investigate"
    required: true
  instructions:
    type: string
    description: "Specific guidance for the research"
    required: true

outputs:
  pdf:
    type: string
    value: "{{ inputs.topic }}_report.pdf"
```

The same pipeline can generate different outputs based on input parameters:
- `topic: "machine_learning"` → `machine_learning_report.pdf`
- `topic: "quantum_computing"` → `quantum_computing_report.pdf`

### 2. AUTO Tag Ambiguity Resolution

The framework uses `<AUTO>` tags to let AI models resolve ambiguous values:

```yaml
steps:
  - id: analyze_data
    action: analyze
    parameters:
      method: <AUTO>Choose best analysis method for this data type</AUTO>
      depth: <AUTO>Determine analysis depth based on data complexity</AUTO>
```

The ambiguity resolver:
- Uses the best available model for resolution
- Provides context-aware suggestions
- Maintains consistency across related AUTO tags
- Supports conditional resolution based on input parameters

### 3. Runtime Template Resolution

Templates are resolved at different stages:

```yaml
steps:
  - id: search
    action: search_web
    parameters:
      query: "{{ inputs.topic }} recent advances"  # Runtime resolution
      
  - id: compile
    action: compile_results
    parameters:
      content: "$results.search"  # Reference to previous task result
```

### 4. Automatic Tool Detection

The framework automatically detects required tools from pipeline definitions:

```yaml
steps:
  - id: web_search
    action: search_web        # → Requires headless-browser tool
    
  - id: shell_command
    action: "!echo hello"     # → Requires terminal tool
    
  - id: file_operation
    action: write_file        # → Requires filesystem tool
```

### 5. MCP Server Integration

Tools are automatically exposed via Model Context Protocol:
- Server starts automatically when tools are detected
- Tools schemas are generated and exposed
- AI models can discover and use tools dynamically
- Bidirectional communication for complex tool interactions

## Installation & Setup

### Prerequisites

- Python 3.11+
- Optional: Docker (for sandboxed execution)
- Optional: Ollama or other model providers

### Basic Setup

```bash
# Clone the repository
git clone <repository-url>
cd orchestrator

# Install dependencies (when implemented)
pip install -e .

# Initialize models
python -c "import orchestrator; orchestrator.init_models()"
```

### Model Configuration

The framework auto-detects available models:

```python
import orchestrator as orc

# Initialize available models
registry = orc.init_models()

# Check available models
print(registry.list_models())
# Output: ['ollama:gemma2:27b', 'ollama:llama3.2:1b', 'huggingface:distilgpt2']
```

## Usage Examples

### Basic Pipeline Compilation and Execution

```python
import orchestrator as orc

# Initialize models
orc.init_models()

# Compile pipeline
pipeline = orc.compile("examples/pipelines/research-report-template.yaml")

# Execute with different inputs
result1 = pipeline.run(
    topic="machine_learning",
    instructions="Focus on transformer architectures"
)

result2 = pipeline.run(
    topic="renewable_energy", 
    instructions="Emphasize solar and wind technologies"
)

print(f"Generated: {result1}")  # machine_learning_report.pdf
print(f"Generated: {result2}")  # renewable_energy_report.pdf
```

### Advanced Pipeline with Tool Integration

```python
import orchestrator as orc

# Pipeline automatically detects and configures required tools
pipeline = orc.compile("examples/pipelines/research-report-template.yaml")

# Tools are auto-detected: web-search, terminal, filesystem, validation
# MCP server starts automatically

# Execute pipeline - tools are used automatically
result = pipeline.run(
    topic="quantum_computing",
    instructions="Cover error correction and commercial applications"
)
```

### Custom Tool Registration

```python
from orchestrator.tools.base import Tool, default_registry

class CustomAnalysisTool(Tool):
    def __init__(self):
        super().__init__(
            name="custom-analysis",
            description="Perform custom data analysis"
        )
        self.add_parameter("data", "object", "Data to analyze")
        self.add_parameter("method", "string", "Analysis method")
    
    async def execute(self, **kwargs):
        # Implementation here
        return {"result": "analysis complete"}

# Register the tool
default_registry.register(CustomAnalysisTool())
```

### Async Pipeline Execution

```python
import asyncio
import orchestrator as orc

async def run_pipeline():
    # Compile pipeline
    pipeline = await orc.compile_async("pipeline.yaml")
    
    # Execute multiple pipelines concurrently
    tasks = [
        pipeline._run_async(topic="ai", instructions="Focus on ethics"),
        pipeline._run_async(topic="climate", instructions="Focus on solutions"),
        pipeline._run_async(topic="space", instructions="Focus on exploration")
    ]
    
    results = await asyncio.gather(*tasks)
    return results

# Run async
results = asyncio.run(run_pipeline())
```

## Tool System

### Available Tools

The framework provides a comprehensive set of built-in tools:

#### Web Tools (`src/orchestrator/tools/web_tools.py`)
- **HeadlessBrowserTool**: Web scraping and page interaction
- **WebSearchTool**: Search engine integration

#### System Tools (`src/orchestrator/tools/system_tools.py`)  
- **TerminalTool**: Shell command execution
- **FileSystemTool**: File operations (read, write, copy, move)

#### Data Tools (`src/orchestrator/tools/data_tools.py`)
- **DataProcessingTool**: Data transformation and conversion
- **ValidationTool**: Data validation and schema checking

### Tool Schema Example

```python
# Tools automatically generate MCP-compatible schemas
{
  "name": "terminal",
  "description": "Execute terminal commands in a sandboxed environment",
  "inputSchema": {
    "type": "object",
    "properties": {
      "command": {"type": "string", "description": "Command to execute"},
      "working_dir": {"type": "string", "description": "Working directory"},
      "timeout": {"type": "integer", "description": "Timeout in seconds"}
    },
    "required": ["command"]
  }
}
```

### MCP Server Configuration

```python
from orchestrator.tools.mcp_server import default_mcp_server

# Server automatically starts when tools are detected
# Configuration is generated dynamically:
{
  "mcpServers": {
    "orchestrator-tools": {
      "command": "python",
      "args": ["-m", "orchestrator.tools.mcp_server"],
      "env": {"ORCHESTRATOR_TOOLS": "enabled"}
    }
  }
}
```

### Tool Detection Logic

The framework uses intelligent heuristics to detect required tools:

```python
# Action patterns that trigger tool detection:
"search_web" → headless-browser
"!command" → terminal  
"write_file" → filesystem
"validate_data" → validation
"transform_data" → data-processing
```

## API Reference

### Core Functions

```python
# Model initialization
orchestrator.init_models() -> ModelRegistry

# Pipeline compilation
orchestrator.compile(yaml_path: str) -> OrchestratorPipeline
orchestrator.compile_async(yaml_path: str) -> OrchestratorPipeline

# Pipeline execution
pipeline.run(**kwargs) -> Any
pipeline._run_async(**kwargs) -> Any
```

### OrchestratorPipeline Class

```python
class OrchestratorPipeline:
    def run(self, **kwargs) -> Any:
        """Execute pipeline with keyword arguments"""
    
    def _validate_inputs(self, kwargs: dict) -> None:
        """Validate required inputs are provided"""
    
    def _resolve_outputs(self, inputs: dict) -> dict:
        """Resolve output definitions using AUTO tags"""
    
    def _resolve_runtime_templates(self, pipeline: Pipeline, context: dict) -> Pipeline:
        """Apply runtime template resolution"""
```

### Tool Registry

```python
from orchestrator.tools.base import default_registry

# Register tool
default_registry.register(tool_instance)

# Execute tool
result = await default_registry.execute_tool(tool_name, **params)

# List available tools
tools = default_registry.list_tools()

# Get tool by name
tool = default_registry.get_tool(tool_name)
```

## Current Capabilities

### ✅ Implemented Features

1. **Input-Agnostic Pipeline System**
   - Templates with runtime resolution
   - Input validation and type checking
   - Output generation based on inputs
   - Keyword argument support

2. **Tool Integration Framework**
   - Comprehensive tool library (web, system, data)
   - Automatic tool detection from YAML
   - MCP server integration
   - Tool parameter mapping and validation

3. **AI Model Integration** 
   - Multi-provider support (Ollama, HuggingFace)
   - Intelligent model selection
   - AUTO tag resolution using AI
   - Fallback and error handling

4. **Pipeline Execution Engine**
   - Dependency resolution and ordering
   - Task status tracking
   - Error handling with fallbacks
   - Reference resolution ($results.task_id)

5. **YAML Processing**
   - Schema validation
   - Template processing (Jinja2)
   - Runtime vs compile-time resolution
   - Circular dependency detection

### 🚧 Working Features

1. **MCP Server Protocol**
   - Server startup and configuration
   - Tool schema generation
   - Basic tool execution (simulated)

2. **State Management**
   - Basic checkpointing infrastructure
   - Recovery mechanisms (not fully tested)

### 📋 Planned Features

1. **Advanced MCP Integration**
   - Real MCP protocol implementation
   - Tool discovery by AI models
   - Bidirectional tool communication

2. **Sandboxed Execution**
   - Docker container isolation
   - Resource limit enforcement
   - Network access controls

3. **Advanced Control Systems**
   - LangGraph adapter implementation
   - Custom control system plugins
   - Distributed execution support

## Limitations

### Current Limitations

1. **MCP Server**: Currently simulated - needs real MCP protocol implementation
2. **Tool Execution**: Some tools use simulation rather than real execution
3. **State Persistence**: Database backends not fully implemented
4. **Error Recovery**: Advanced error handling strategies need refinement
5. **Performance**: No optimization for large-scale pipeline execution
6. **Security**: Sandboxing and input validation need hardening

### Known Issues

1. **Tool Parameter Mapping**: Some edge cases in parameter conversion
2. **Template Resolution**: Complex nested templates may fail
3. **Dependency Cycles**: Detection works but error messages could be clearer
4. **Resource Management**: No active monitoring of CPU/memory usage

### Scale Limitations

- **Pipeline Size**: Tested with small pipelines (< 20 tasks)
- **Concurrent Execution**: Limited testing of parallel task execution
- **Model Selection**: UCB algorithm needs tuning for production use
- **Tool Discovery**: Heuristic-based detection may miss edge cases

## Development

### Project Structure

```
orchestrator/
├── src/orchestrator/          # Core library
│   ├── compiler/             # YAML parsing and compilation
│   ├── core/                 # Core abstractions (Task, Pipeline, etc.)
│   ├── models/               # Model registry and abstractions
│   ├── tools/                # Tool library and MCP integration
│   ├── state/                # State management
│   ├── integrations/         # Third-party integrations
│   └── orchestrator.py       # Main orchestrator class
├── examples/                 # Example pipelines and tests
│   ├── pipelines/            # YAML pipeline definitions
│   └── test_*.py             # Integration tests
├── tests/                    # Unit tests
├── docs/                     # Documentation
└── config/                   # Configuration schemas
```

### Running Tests

```bash
# Run integration tests
python examples/test_full_integration.py

# Test individual components
python examples/test_orchestrator_coverage_lines_207_272.py
python tests/test_ambiguity_resolver.py
python tests/test_core_pipeline_coverage.py
```

### Adding New Tools

1. Create tool class inheriting from `Tool`
2. Implement `execute` method
3. Register with `default_registry`
4. Add detection logic to `ToolDetector`

```python
from orchestrator.tools.base import Tool

class MyCustomTool(Tool):
    def __init__(self):
        super().__init__(name="my-tool", description="My tool description")
        self.add_parameter("param1", "string", "Parameter description")
    
    async def execute(self, **kwargs):
        # Implementation
        return {"result": "success"}
```

### Contributing

The framework is in active development. Key areas for contribution:

1. **Real MCP Protocol Implementation**
2. **Additional Tool Integrations** 
3. **Performance Optimization**
4. **Security Hardening**
5. **Documentation and Examples**

---

*This documentation reflects the current state of the Orchestrator framework as of July 2024. For the latest updates, see the project repository.*