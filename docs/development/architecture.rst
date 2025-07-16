Architecture Overview
====================

This document describes the architectural design of the py-orc framework, including core components, design patterns, and data flow.

.. contents:: Table of Contents
   :local:
   :depth: 2

System Architecture
-------------------

py-orc follows a modular, plugin-based architecture designed for scalability and extensibility. The framework is organized into several key layers:

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────┐
   │                      User Interface Layer                    │
   │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
   │  │   Python API    │  │   CLI Interface │  │   YAML Pipeline │
   │  │   (orchestrator)│  │   (py-orc)      │  │   Definitions   │
   │  └─────────────────┘  └─────────────────┘  └─────────────────┘
   └─────────────────────────────────────────────────────────────┘
   
   ┌─────────────────────────────────────────────────────────────┐
   │                     Orchestration Layer                      │
   │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
   │  │   Core Pipeline │  │   Task Executor │  │   State Manager │
   │  │   Management    │  │   Engine        │  │   & Checkpoints │
   │  └─────────────────┘  └─────────────────┘  └─────────────────┘
   └─────────────────────────────────────────────────────────────┘
   
   ┌─────────────────────────────────────────────────────────────┐
   │                    Compilation Layer                         │
   │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
   │  │   YAML Compiler │  │   AUTO Tag      │  │   Dependency    │
   │  │   & Parser      │  │   Resolution    │  │   Resolution    │
   │  └─────────────────┘  └─────────────────┘  └─────────────────┘
   └─────────────────────────────────────────────────────────────┘
   
   ┌─────────────────────────────────────────────────────────────┐
   │                     Integration Layer                        │
   │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
   │  │   Model         │  │   Tool          │  │   Control       │
   │  │   Integrations  │  │   Integrations  │  │   System        │
   │  │   (OpenAI, etc) │  │   (MCP Tools)   │  │   Adapters      │
   │  └─────────────────┘  └─────────────────┘  └─────────────────┘
   └─────────────────────────────────────────────────────────────┘

Core Components
---------------

Core Module (``orchestrator.core``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The core module contains the fundamental building blocks of the framework:

**Pipeline** (``core.pipeline``)
   The main execution unit that coordinates tasks and manages their dependencies.

**Task** (``core.task``)
   Individual units of work that can be executed independently or as part of a pipeline.

**Model** (``core.model``)
   Abstract base class for all AI model integrations, providing a unified interface.

**Control System** (``core.control_system``)
   Interface for different orchestration backends (LangGraph, custom implementations).

**Cache** (``core.cache``)
   Multi-level caching system for performance optimization.

**Error Handler** (``core.error_handler``)
   Centralized error handling with recovery strategies.

**Resource Allocator** (``core.resource_allocator``)
   Manages computational resources and prevents resource conflicts.

Compiler Module (``orchestrator.compiler``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The compiler module handles YAML pipeline definitions and ambiguity resolution:

**YAML Compiler** (``compiler.yaml_compiler``)
   Parses YAML pipeline definitions and converts them to executable pipelines.

**AUTO Tag Parser** (``compiler.auto_tag_yaml_parser``)
   Specialized parser for handling ``<AUTO>`` tags in YAML configurations.

**Ambiguity Resolver** (``compiler.ambiguity_resolver``)
   Resolves ambiguous parameters using AI models and context.

State Management (``orchestrator.state``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The state module provides persistence and recovery capabilities:

**State Manager** (``state.state_manager``)
   Abstract interface for state persistence and recovery.

**Simple State Manager** (``state.simple_state_manager``)
   File-based state manager for development and testing.

**Adaptive Checkpoint** (``state.adaptive_checkpoint``)
   Intelligent checkpointing based on task criticality and execution time.

**Backends** (``state.backends``)
   Storage backends for different persistence requirements.

Model Integrations (``orchestrator.integrations``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pre-built integrations for popular AI model providers:

- **OpenAI Model** (``integrations.openai_model``)
- **Anthropic Model** (``integrations.anthropic_model``)
- **Google Model** (``integrations.google_model``)
- **HuggingFace Model** (``integrations.huggingface_model``)
- **Lazy Ollama Model** (``integrations.lazy_ollama_model``)

Tools Module (``orchestrator.tools``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tool integrations for external capabilities:

**Base Tool** (``tools.base``)
   Abstract base class for all tool implementations.

**Web Tools** (``tools.web_tools``)
   Web scraping, search, and interaction capabilities.

**System Tools** (``tools.system_tools``)
   File system operations and command execution.

**Data Tools** (``tools.data_tools``)
   Data processing and transformation utilities.

**MCP Server** (``tools.mcp_server``)
   Model Context Protocol server for tool integration.

Design Patterns
---------------

Factory Pattern
~~~~~~~~~~~~~~~

The framework uses the Factory pattern for creating model instances:

.. code-block:: python

   class ModelFactory:
       @staticmethod
       def create_model(model_type: str, config: dict) -> BaseModel:
           if model_type == "openai":
               return OpenAIModel(config)
           elif model_type == "anthropic":
               return AnthropicModel(config)
           # ... other model types

Observer Pattern
~~~~~~~~~~~~~~~~

The pipeline execution system uses the Observer pattern for event handling:

.. code-block:: python

   class Pipeline:
       def __init__(self):
           self.observers = []
       
       def add_observer(self, observer):
           self.observers.append(observer)
       
       def notify_observers(self, event):
           for observer in self.observers:
               observer.handle_event(event)

Strategy Pattern
~~~~~~~~~~~~~~~~

Different execution strategies are implemented using the Strategy pattern:

.. code-block:: python

   class ExecutionStrategy:
       def execute(self, task: Task) -> Result:
           raise NotImplementedError
   
   class SequentialStrategy(ExecutionStrategy):
       def execute(self, task: Task) -> Result:
           # Sequential execution logic
           pass
   
   class ParallelStrategy(ExecutionStrategy):
       def execute(self, task: Task) -> Result:
           # Parallel execution logic
           pass

Command Pattern
~~~~~~~~~~~~~~~

Tasks are implemented as commands for better encapsulation and undo capabilities:

.. code-block:: python

   class Task:
       def execute(self, context: ExecutionContext) -> Result:
           # Task execution logic
           pass
       
       def undo(self, context: ExecutionContext) -> None:
           # Undo logic if needed
           pass

Data Flow
---------

Pipeline Execution Flow
~~~~~~~~~~~~~~~~~~~~~~~

1. **YAML Parsing**: Pipeline definition is parsed from YAML
2. **Compilation**: Tasks are compiled and dependencies resolved
3. **Validation**: Pipeline structure and parameters are validated
4. **Execution Planning**: Execution order is determined based on dependencies
5. **Task Execution**: Tasks are executed according to the plan
6. **State Management**: Execution state is saved at checkpoints
7. **Result Aggregation**: Results are collected and returned

.. code-block:: text

   YAML File → Compiler → Pipeline → Executor → Results
       ↓           ↓          ↓         ↓          ↓
   Validation → AUTO Tag → Task → Model → Output
               Resolution  Queue  Calls

AUTO Tag Resolution Flow
~~~~~~~~~~~~~~~~~~~~~~~~

1. **Tag Detection**: ``<AUTO>`` tags are identified in YAML
2. **Context Building**: Surrounding context is gathered
3. **Model Selection**: Appropriate model is chosen for resolution
4. **Resolution**: Model generates specific values
5. **Validation**: Generated values are validated
6. **Substitution**: AUTO tags are replaced with resolved values

State Management Flow
~~~~~~~~~~~~~~~~~~~~~

1. **Checkpoint Creation**: State is captured at strategic points
2. **Persistence**: State is saved to storage backend
3. **Recovery**: Failed pipelines can be resumed from checkpoints
4. **Cleanup**: Old checkpoints are cleaned up automatically

Error Handling Flow
~~~~~~~~~~~~~~~~~~~

1. **Error Detection**: Errors are caught and classified
2. **Recovery Strategy**: Appropriate recovery strategy is selected
3. **Retry Logic**: Transient errors are retried with backoff
4. **Fallback**: Alternative models or strategies are used
5. **Graceful Degradation**: System continues with reduced functionality

Extensibility Points
--------------------

Custom Models
~~~~~~~~~~~~~

New model providers can be added by implementing the ``BaseModel`` interface:

.. code-block:: python

   class CustomModel(BaseModel):
       def __init__(self, config: dict):
           self.config = config
       
       async def generate(self, prompt: str) -> str:
           # Custom model implementation
           pass

Custom Tools
~~~~~~~~~~~~

New tools can be added by implementing the ``BaseTool`` interface:

.. code-block:: python

   class CustomTool(BaseTool):
       def __init__(self, config: dict):
           self.config = config
       
       async def execute(self, **kwargs) -> dict:
           # Custom tool implementation
           pass

Custom State Backends
~~~~~~~~~~~~~~~~~~~~~

New storage backends can be implemented:

.. code-block:: python

   class CustomStateBackend(StateBackend):
       async def save(self, key: str, data: dict) -> None:
           # Custom storage implementation
           pass
       
       async def load(self, key: str) -> dict:
           # Custom retrieval implementation
           pass

Performance Considerations
--------------------------

Caching Strategy
~~~~~~~~~~~~~~~~

The framework implements a multi-level caching system:

1. **Memory Cache**: Fast access to frequently used data
2. **Redis Cache**: Shared cache for distributed setups
3. **Disk Cache**: Persistent cache for large objects

Parallel Execution
~~~~~~~~~~~~~~~~~~

Tasks with no dependencies can be executed in parallel:

- Thread-based parallelism for I/O-bound tasks
- Process-based parallelism for CPU-bound tasks
- Async/await for concurrent API calls

Resource Management
~~~~~~~~~~~~~~~~~~~

The resource allocator prevents resource conflicts:

- Memory usage tracking
- GPU resource allocation
- API rate limiting
- Concurrent execution limits

Security Considerations
-----------------------

Input Validation
~~~~~~~~~~~~~~~~

All inputs are validated to prevent injection attacks:

- YAML schema validation
- Parameter type checking
- Model output sanitization

Sandboxing
~~~~~~~~~~

External code execution is sandboxed:

- Docker containers for isolation
- Resource limits enforcement
- Network access controls

API Key Management
~~~~~~~~~~~~~~~~~~

API keys are securely managed:

- Environment variable configuration
- Encrypted storage options
- Key rotation support

Future Enhancements
-------------------

Planned architectural improvements include:

1. **Distributed Execution**: Multi-node pipeline execution
2. **Plugin System**: Dynamic loading of extensions
3. **Workflow Optimization**: Automatic pipeline optimization
4. **Monitoring Integration**: Built-in observability features
5. **Model Fine-tuning**: Automatic model improvement based on usage