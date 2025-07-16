Core API Reference
==================

This section documents the core classes and functions of the Orchestrator Framework.

.. note::
   This is the auto-generated API documentation. For more detailed guides and examples, see the :doc:`../user_guide/yaml_configuration` and :doc:`../tutorials/notebooks`.

Core Classes
------------

The Orchestrator Framework is built around four main abstractions:

Task
~~~~

Represents a single unit of work in a pipeline.

**Key Methods:**
- ``start()`` - Mark task as started
- ``complete(result)`` - Mark task as completed with result
- ``fail(error)`` - Mark task as failed with error
- ``is_ready()`` - Check if task dependencies are satisfied

Pipeline
~~~~~~~~

Represents a collection of tasks with dependencies.

**Key Methods:**
- ``add_task(task)`` - Add a task to the pipeline
- ``get_task(task_id)`` - Get a task by ID
- ``get_execution_order()`` - Get task execution order
- ``set_context(key, value)`` - Set pipeline context

Model
~~~~~

Abstract base class for AI models.

**Key Methods:**
- ``generate(prompt, **kwargs)`` - Generate text
- ``generate_structured(prompt, schema, **kwargs)`` - Generate structured output
- ``can_execute(task)`` - Check if model can handle task

Control System
~~~~~~~~~~~~~~

Abstract base class for control system adapters.

**Key Methods:**
- ``execute_task(task, context)`` - Execute a single task
- ``execute_pipeline(pipeline)`` - Execute entire pipeline
- ``get_capabilities()`` - Get system capabilities

Advanced Components
-------------------

Error Handling
~~~~~~~~~~~~~~

- ``ErrorHandler`` - Central error handling system
- ``RetryStrategy`` - Base class for retry strategies
- ``CircuitBreaker`` - Circuit breaker for fault tolerance

Caching
~~~~~~~

- ``MultiLevelCache`` - Multi-level caching system
- ``MemoryCache`` - In-memory cache backend
- ``DiskCache`` - Disk-based cache backend

Resource Management
~~~~~~~~~~~~~~~~~~~

- ``ResourceAllocator`` - Resource allocation manager
- ``ResourcePool`` - Pool for specific resource types
- ``ResourceRequest`` - Resource allocation request

State Management
~~~~~~~~~~~~~~~~

- ``StateManager`` - Pipeline state persistence
- ``AdaptiveCheckpointManager`` - Intelligent checkpointing

Execution
~~~~~~~~~

- ``ParallelExecutor`` - Parallel task execution
- ``SandboxManager`` - Sandboxed code execution
- ``DockerSandboxExecutor`` - Docker-based sandboxing

Compilation
~~~~~~~~~~~

- ``YAMLCompiler`` - YAML pipeline compilation
- ``SchemaValidator`` - Pipeline schema validation
- ``AmbiguityResolver`` - AUTO tag resolution

Usage Examples
--------------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import asyncio
   from orchestrator import Task, Pipeline, Orchestrator
   
   async def main():
       # Create a task
       task = Task(
           id="hello",
           name="Hello Task",
           action="generate_text",
           parameters={"prompt": "Hello, world!"}
       )
       
       # Create a pipeline
       pipeline = Pipeline(id="demo", name="Demo Pipeline")
       pipeline.add_task(task)
       
       # Execute with orchestrator
       orchestrator = Orchestrator()
       result = await orchestrator.execute_pipeline(pipeline)
       return result
   
   # Run the pipeline
   result = asyncio.run(main())

YAML Configuration
~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   id: demo_pipeline
   name: Demo Pipeline
   
   tasks:
     - id: hello
       name: Hello Task
       action: generate_text
       parameters:
         prompt: "Hello, world!"

Error Handling
~~~~~~~~~~~~~~

.. code-block:: python

   import asyncio
   from orchestrator.core.error_handler import ErrorHandler
   
   async def run_with_error_handling():
       error_handler = ErrorHandler()
       orchestrator = Orchestrator(error_handler=error_handler)
       
       # Tasks will automatically retry on failure
       result = await orchestrator.execute_pipeline(pipeline)
       return result
   
   # Run with error handling
   result = asyncio.run(run_with_error_handling())

For detailed API documentation, explore the source code in the ``src/orchestrator/`` directory.