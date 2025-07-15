=============
API Reference
=============

Complete API documentation for the Orchestrator framework.

.. toctree::
   :maxdepth: 2
   :caption: API Modules

   core
   compiler
   models
   tools
   orchestrator

Overview
========

The Orchestrator API is organized into several key modules:

Core Module (:doc:`core`)
-------------------------

The foundation of the framework, containing:

- **Task**: Individual units of work
- **Pipeline**: Collections of tasks with dependencies
- **ControlSystem**: Pluggable execution backends
- **Model**: AI model abstraction

Compiler Module (:doc:`compiler`)  
---------------------------------

YAML pipeline compilation and processing:

- **YAMLCompiler**: Main compilation engine
- **AmbiguityResolver**: AUTO tag resolution
- **SchemaValidator**: Pipeline validation
- **TemplateEngine**: Jinja2 template processing

Models Module (:doc:`models`)
-----------------------------

AI model management and selection:

- **ModelRegistry**: Central model repository
- **ModelSelector**: Intelligent model selection
- **ModelCapabilities**: Model capability definitions
- **ResourceManager**: Resource allocation and monitoring

Tools Module (:doc:`tools`)
---------------------------

Tool system and integrations:

- **ToolRegistry**: Tool management
- **MCPServer**: Model Context Protocol integration
- **WebTools**: Internet interaction tools
- **SystemTools**: File and command execution
- **DataTools**: Data processing and validation

Orchestrator Module (:doc:`orchestrator`)
-----------------------------------------

Main orchestration engine:

- **Orchestrator**: Primary orchestration class
- **StateManager**: Pipeline state and checkpointing
- **ExecutionEngine**: Task execution and coordination
- **ErrorHandler**: Error management and recovery

Quick Reference
===============

Core Classes
------------

.. autosummary::
   :toctree: _autosummary

   orchestrator.Task
   orchestrator.Pipeline
   orchestrator.Orchestrator
   orchestrator.YAMLCompiler
   orchestrator.ModelRegistry
   orchestrator.ToolRegistry

Main Functions
--------------

.. autosummary::
   :toctree: _autosummary

   orchestrator.init_models
   orchestrator.compile
   orchestrator.compile_async

Key Exceptions
--------------

.. autosummary::
   :toctree: _autosummary

   orchestrator.OrchestratorError
   orchestrator.CompilationError
   orchestrator.ExecutionError
   orchestrator.ValidationError

Usage Examples
==============

Basic Usage
-----------

.. code-block:: python

   import orchestrator as orc
   
   # Initialize models
   registry = orc.init_models()
   
   # Compile pipeline
   pipeline = orc.compile("my_pipeline.yaml")
   
   # Execute
   result = pipeline.run(input_param="value")

Advanced Usage
--------------

.. code-block:: python

   from orchestrator import Orchestrator
   from orchestrator.core.control_system import MockControlSystem
   from orchestrator.models.model_registry import ModelRegistry
   
   # Create custom orchestrator
   control_system = MockControlSystem()
   orchestrator = Orchestrator(control_system=control_system)
   
   # Use custom model registry
   registry = ModelRegistry()
   # ... configure models
   
   # Compile with custom settings
   pipeline = orchestrator.compile(
       yaml_content,
       config={"timeout": 3600}
   )

Type Annotations
================

The Orchestrator framework uses comprehensive type annotations for better IDE support and type checking:

.. code-block:: python

   from typing import Dict, Any, List, Optional
   from orchestrator import Pipeline, Task
   
   def process_pipeline(
       pipeline: Pipeline,
       inputs: Dict[str, Any],
       timeout: Optional[int] = None
   ) -> Dict[str, Any]:
       return pipeline.run(**inputs)

Environment Variables
====================

The framework recognizes these environment variables:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Description
   * - ``ORCHESTRATOR_HOME``
     - Home directory for configuration and cache
   * - ``ORCHESTRATOR_LOG_LEVEL``
     - Logging level (DEBUG, INFO, WARNING, ERROR)
   * - ``ORCHESTRATOR_MODEL_TIMEOUT``
     - Default timeout for model operations
   * - ``ORCHESTRATOR_TOOL_TIMEOUT``
     - Default timeout for tool operations
   * - ``ORCHESTRATOR_MCP_AUTO_START``
     - Auto-start MCP server when tools detected

Configuration
=============

Default configuration can be overridden using a config file at ``~/.orchestrator/config.yaml``:

.. code-block:: yaml

   models:
     default: "ollama:gemma2:27b"
     fallback: "ollama:llama3.2:1b"
     timeout: 300
   
   tools:
     mcp_port: 8000
     auto_start: true
   
   execution:
     parallel: true
     checkpoint: true
     timeout: 3600
   
   logging:
     level: "INFO"
     format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

Performance Considerations
=========================

Model Loading
-------------

Models are loaded lazily and cached. For better performance:

.. code-block:: python

   # Initialize models once at startup
   orc.init_models()
   
   # Reuse compiled pipelines
   pipeline = orc.compile("pipeline.yaml")
   
   # Multiple executions reuse the same pipeline
   for inputs in input_batches:
       result = pipeline.run(**inputs)

Memory Management
-----------------

Large pipelines and datasets can consume significant memory:

.. code-block:: python

   # Enable checkpointing for long pipelines
   pipeline = orc.compile("pipeline.yaml", config={
       "checkpoint": True,
       "memory_limit": "8GB"
   })
   
   # Process data in batches
   for batch in data_batches:
       result = pipeline.run(data=batch)
       # Results are automatically checkpointed

Error Handling
==============

The framework provides structured error handling:

.. code-block:: python

   from orchestrator import CompilationError, ExecutionError
   
   try:
       pipeline = orc.compile("pipeline.yaml")
       result = pipeline.run(input="value")
   except CompilationError as e:
       print(f"Pipeline compilation failed: {e}")
       print(f"Error details: {e.details}")
   except ExecutionError as e:
       print(f"Pipeline execution failed: {e}")
       print(f"Failed step: {e.step_id}")
       print(f"Error context: {e.context}")

Debugging
=========

Enable detailed logging for debugging:

.. code-block:: python

   import logging
   
   # Enable debug logging
   logging.basicConfig(level=logging.DEBUG)
   
   # Compile with debug information
   pipeline = orc.compile("pipeline.yaml", debug=True)
   
   # Execute with verbose output
   result = pipeline.run(input="value", _verbose=True)

Extension Points
================

The framework provides several extension points:

Custom Control Systems
----------------------

.. code-block:: python

   from orchestrator.core.control_system import ControlSystem
   
   class MyControlSystem(ControlSystem):
       async def execute_task(self, task: Task, context: dict) -> dict:
           # Custom execution logic
           pass

Custom Tools
------------

.. code-block:: python

   from orchestrator.tools.base import Tool
   
   class MyTool(Tool):
       def __init__(self):
           super().__init__("my-tool", "Description")
           
       async def execute(self, **kwargs) -> dict:
           # Tool implementation
           pass

Custom Models
-------------

.. code-block:: python

   from orchestrator.core.model import Model
   
   class MyModel(Model):
       async def generate(self, prompt: str, **kwargs) -> str:
           # Model implementation
           pass

Thread Safety
=============

The framework is designed to be thread-safe:

.. code-block:: python

   import concurrent.futures
   
   # Safe to use across threads
   pipeline = orc.compile("pipeline.yaml")
   
   def process_input(input_data):
       return pipeline.run(**input_data)
   
   # Parallel execution
   with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
       futures = [executor.submit(process_input, data) 
                 for data in input_datasets]
       results = [f.result() for f in futures]

Testing
=======

Testing utilities and patterns:

.. code-block:: python

   from orchestrator.testing import MockModel, TestRunner
   
   def test_my_pipeline():
       # Use mock model for testing
       with MockModel() as mock:
           mock.set_response("test response")
           
           pipeline = orc.compile("test_pipeline.yaml")
           result = pipeline.run(input="test")
           
           assert result == "expected"
   
   # Test runner for pipeline validation
   runner = TestRunner("pipelines/")
   runner.validate_all()  # Validates all YAML files
   runner.test_compilation()  # Tests compilation
   runner.run_smoke_tests()  # Basic execution tests

Migration Guide
===============

From Version 0.1.x to 0.2.x
---------------------------

.. code-block:: python

   # Old way (0.1.x)
   from orchestrator import OrchestratorEngine
   engine = OrchestratorEngine()
   result = engine.run_pipeline("pipeline.yaml", inputs)
   
   # New way (0.2.x)
   import orchestrator as orc
   pipeline = orc.compile("pipeline.yaml")
   result = pipeline.run(**inputs)

Troubleshooting
===============

Common Issues
-------------

**ImportError: No module named 'orchestrator'**

- Ensure the package is installed: ``pip install orchestrator-ai``
- Check virtual environment activation

**Model Loading Failures**

- Verify model availability: ``ollama list``
- Check API keys for cloud models
- Ensure sufficient memory for local models

**Pipeline Compilation Errors**

- Validate YAML syntax
- Check required fields in pipeline definition
- Verify template syntax

**Tool Execution Failures**

- Check tool dependencies (e.g., pandoc for PDF generation)
- Verify network connectivity for web tools
- Check file permissions for system tools

Getting Help
============

- **Documentation**: https://orchestrator.readthedocs.io
- **GitHub Issues**: https://github.com/contextualdynamics/orchestrator/issues
- **Discussions**: https://github.com/contextualdynamics/orchestrator/discussions
- **Discord**: https://discord.gg/orchestrator