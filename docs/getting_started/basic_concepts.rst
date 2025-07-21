Basic Concepts
==============

This guide introduces the core concepts of the Orchestrator Framework. Understanding these fundamentals will help you build effective AI workflows.

Core Abstractions
-----------------

The Orchestrator Framework is built around four core abstractions:

.. image:: ../images/core_abstractions.png
   :alt: Core Abstractions
   :align: center

Task
~~~~

A **Task** represents a single unit of work in your pipeline. Each task has:

* **ID** - Unique identifier
* **Name** - Human-readable name
* **Action** - What the task does (e.g., "generate_text", "classify")
* **Parameters** - Input data and configuration
* **Dependencies** - Other tasks that must complete first

.. code-block:: python

   from orchestrator import Task
   
   task = Task(
       id="summarize",
       name="Summarize Document",
       action="generate_text",
       parameters={
           "prompt": "Summarize this document: {document}",
           "max_tokens": 150
       },
       dependencies=["extract_document"]
   )

Pipeline
~~~~~~~~

A **Pipeline** is a collection of tasks with defined dependencies. It represents your complete workflow:

.. code-block:: python

   from orchestrator import Pipeline
   
   pipeline = Pipeline(
       id="document_processing",
       name="Document Processing Pipeline"
   )
   
   # Add tasks to pipeline
   pipeline.add_task(extract_task)
   pipeline.add_task(summarize_task)
   pipeline.add_task(classify_task)

Model
~~~~~

A **Model** represents an AI service that can execute tasks. Models can be:

* **Local models** (running on your machine)
* **API-based models** (OpenAI, Anthropic, etc.)
* **Custom models** (your own implementations)

.. code-block:: python

   import os
   import os
   from orchestrator.models import OpenAIModel
   
   # API key should be set in environment variable or ~/.orchestrator/.env
   model = OpenAIModel(
       name="gpt-4",
       api_key=os.environ.get("OPENAI_API_KEY"),  # Loaded from environment
       model="gpt-4"
   )

Orchestrator
~~~~~~~~~~~~

The **Orchestrator** is the execution engine that:

* Manages model selection
* Handles task dependencies
* Provides error handling and recovery
* Manages state and checkpointing

.. code-block:: python

   import asyncio
   from orchestrator import Orchestrator
   
   async def run_pipeline():
       orchestrator = Orchestrator()
       orchestrator.register_model(model)
       
       result = await orchestrator.execute_pipeline(pipeline)
       return result
   
   # Run the pipeline
   result = asyncio.run(run_pipeline())

Task Dependencies
-----------------

Tasks can depend on other tasks, creating a directed acyclic graph (DAG):

.. code-block:: python

   # Task A (no dependencies)
   task_a = Task(id="a", name="Task A", action="generate_text")
   
   # Task B depends on A
   task_b = Task(id="b", name="Task B", action="generate_text", 
                 dependencies=["a"])
   
   # Task C depends on A and B
   task_c = Task(id="c", name="Task C", action="generate_text",
                 dependencies=["a", "b"])

Execution Order
~~~~~~~~~~~~~~~

The orchestrator automatically determines execution order based on dependencies:

.. code-block:: text

   Level 0: [Task A]           # No dependencies
   Level 1: [Task B]           # Depends on A
   Level 2: [Task C]           # Depends on A and B

Tasks at the same level can execute in parallel for better performance.

Parameter Substitution
~~~~~~~~~~~~~~~~~~~~~

Tasks can reference outputs from other tasks using template syntax:

.. code-block:: python

   task_a = Task(
       id="extract",
       name="Extract Information",
       action="generate_text",
       parameters={"prompt": "Extract key facts from: {document}"}
   )
   
   task_b = Task(
       id="summarize",
       name="Summarize Facts",
       action="generate_text",
       parameters={"prompt": "Summarize these facts: {extract}"},
       dependencies=["extract"]
   )

Pipeline Execution
------------------

When you execute a pipeline, the orchestrator:

1. **Validates** the pipeline structure
2. **Determines** execution order
3. **Selects** appropriate models for each task
4. **Executes** tasks in dependency order
5. **Manages** errors and retries
6. **Returns** results from all tasks

.. code-block:: python

   import asyncio
   
   async def execute_and_process():
       # Execute pipeline
       result = await orchestrator.execute_pipeline(pipeline)
       
       # Access individual task results
       print(result["extract"])    # Output from extract task
       print(result["summarize"])  # Output from summarize task
       return result
   
   # Run the execution
   result = asyncio.run(execute_and_process())

Model Selection
---------------

The orchestrator automatically selects the best model for each task based on:

* **Capabilities** - What the model can do
* **Requirements** - What the task needs
* **Performance** - Historical success rates
* **Cost** - Resource usage and API costs

.. code-block:: python

   import asyncio
   
   async def run_with_model_selection():
       # Register multiple models
       orchestrator.register_model(gpt4_model)
       orchestrator.register_model(claude_model)
       orchestrator.register_model(local_model)
       
       # Orchestrator will select best model for each task
       result = await orchestrator.execute_pipeline(pipeline)
       return result
   
   # Run with model selection
   result = asyncio.run(run_with_model_selection())

Error Handling
--------------

The framework provides comprehensive error handling:

Retry Strategies
~~~~~~~~~~~~~~~~

.. code-block:: python

   import asyncio
   from orchestrator.core.error_handler import ErrorHandler
   
   async def run_with_retry():
       error_handler = ErrorHandler()
       orchestrator = Orchestrator(error_handler=error_handler)
       
       # Tasks will automatically retry on failure
       result = await orchestrator.execute_pipeline(pipeline)
       return result
   
   # Run with retry handling
   result = asyncio.run(run_with_retry())

Circuit Breakers
~~~~~~~~~~~~~~~~

.. code-block:: python

   import asyncio
   
   async def run_with_circuit_breaker():
       # Circuit breaker prevents cascading failures
       breaker = error_handler.get_circuit_breaker("openai_api")
       
       # Executes with circuit breaker protection
       result = await orchestrator.execute_pipeline(pipeline)
       return result
   
   # Run with circuit breaker
   result = asyncio.run(run_with_circuit_breaker())

Fallback Models
~~~~~~~~~~~~~~~

.. code-block:: python

   import asyncio
   
   async def run_with_fallback():
       # Register models in order of preference
       orchestrator.register_model(primary_model)
       orchestrator.register_model(fallback_model)
       
       # Will use fallback if primary fails
       result = await orchestrator.execute_pipeline(pipeline)
       return result
   
   # Run with fallback support
   result = asyncio.run(run_with_fallback())

State Management
---------------

For long-running pipelines, state management ensures reliability:

Checkpointing
~~~~~~~~~~~~~

.. code-block:: python

   import asyncio
   from orchestrator.state import StateManager
   
   async def run_with_checkpointing():
       state_manager = StateManager(storage_path="./checkpoints")
       orchestrator = Orchestrator(state_manager=state_manager)
       
       # Automatically saves checkpoints during execution
       result = await orchestrator.execute_pipeline(pipeline)
       return result
   
   # Run with checkpointing
   result = asyncio.run(run_with_checkpointing())

Recovery
~~~~~~~~

.. code-block:: python

   import asyncio
   
   async def resume_from_checkpoint():
       # Resume from last checkpoint
       result = await orchestrator.resume_pipeline("pipeline_id")
       return result
   
   # Resume execution
   result = asyncio.run(resume_from_checkpoint())

YAML Configuration
-----------------

Define pipelines declaratively in YAML:

.. code-block:: yaml

   id: document_pipeline
   name: Document Processing Pipeline
   
   tasks:
     - id: extract
       name: Extract Information
       action: generate_text
       parameters:
         prompt: "Extract key facts from: {document}"
     
     - id: summarize
       name: Summarize Facts
       action: generate_text
       parameters:
         prompt: "Summarize these facts: {extract}"
       dependencies:
         - extract

Load and execute:

.. code-block:: python

   import asyncio
   from orchestrator.compiler import YAMLCompiler
   
   async def run_yaml_pipeline():
       compiler = YAMLCompiler()
       pipeline = compiler.compile_file("document_pipeline.yaml")
       
       result = await orchestrator.execute_pipeline(pipeline)
       return result
   
   # Run YAML pipeline
   result = asyncio.run(run_yaml_pipeline())

Advanced Features
-----------------

Resource Management
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import asyncio
   from orchestrator.core.resource_allocator import ResourceAllocator
   
   async def run_with_resource_management():
       allocator = ResourceAllocator()
       orchestrator = Orchestrator(resource_allocator=allocator)
       
       # Automatically manages CPU, memory, and API quotas
       result = await orchestrator.execute_pipeline(pipeline)
       return result
   
   # Run with resource management
   result = asyncio.run(run_with_resource_management())

Parallel Execution
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import asyncio
   from orchestrator.executor import ParallelExecutor
   
   async def run_parallel_execution():
       executor = ParallelExecutor(max_workers=4)
       orchestrator = Orchestrator(executor=executor)
       
       # Independent tasks run in parallel
       result = await orchestrator.execute_pipeline(pipeline)
       return result
   
   # Run with parallel execution
   result = asyncio.run(run_parallel_execution())

Caching
~~~~~~~

.. code-block:: python

   import asyncio
   from orchestrator.core.cache import MultiLevelCache
   
   async def run_with_caching():
       cache = MultiLevelCache()
       orchestrator = Orchestrator(cache=cache)
       
       # Results are cached for faster subsequent runs
       result = await orchestrator.execute_pipeline(pipeline)
       return result
   
   # Run with caching
   result = asyncio.run(run_with_caching())

Best Practices
--------------

1. **Keep tasks focused** - Each task should have a single responsibility
2. **Use descriptive names** - Make your pipelines self-documenting
3. **Handle errors gracefully** - Use retry strategies and fallbacks
4. **Test incrementally** - Start with mock models, then switch to real ones
5. **Monitor performance** - Track execution times and resource usage
6. **Use YAML for complex pipelines** - Easier to read and maintain
7. **Version your pipelines** - Track changes over time

Common Patterns
---------------

**Sequential Processing**
   Tasks that build on each other's outputs

**Fan-out/Fan-in**
   One task spawns multiple parallel tasks that later combine

**Conditional Execution**
   Tasks that only run under certain conditions

**Data Transformation**
   Tasks that process and reshape data

**Multi-Model Workflows**
   Using different models for different types of tasks

Next Steps
----------

Now that you understand the core concepts:

* Build :doc:`your_first_pipeline`
* Learn about :doc:`../user_guide/yaml_configuration`
* Explore :doc:`../user_guide/models_and_adapters`
* Try the :doc:`../tutorials/notebooks`

.. tip::
   The best way to learn is by building. Start with simple pipelines and gradually add complexity as you become more comfortable with the framework.