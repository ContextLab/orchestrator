Quickstart Guide
===============

This guide will get you up and running with Orchestrator in 5 minutes. We'll create a simple AI workflow that demonstrates the core concepts.

Your First Pipeline
-------------------

Let's create a simple text generation pipeline:

.. code-block:: python

   from orchestrator import Orchestrator, Task, Pipeline
   from orchestrator.models.mock_model import MockModel
   
   # Create a mock model for testing
   model = MockModel("gpt-test")
   model.set_response("Hello, world!", "Hello! How can I help you today?")
   
   # Create a task
   task = Task(
       id="greeting",
       name="Generate Greeting",
       action="generate_text",
       parameters={"prompt": "Hello, world!"}
   )
   
   # Create a pipeline
   pipeline = Pipeline(id="hello_pipeline", name="Hello Pipeline")
   pipeline.add_task(task)
   
   # Create orchestrator and register model
   orchestrator = Orchestrator()
   orchestrator.register_model(model)
   
   # Execute pipeline
   import asyncio

   

   async def run_pipeline():

       result = await orchestrator.execute_pipeline(pipeline)

       return result

   

   # Run the pipeline

   result = asyncio.run(run_pipeline())
   print(f"Result: {result['greeting']}")

Multi-Task Pipeline
-------------------

Let's create a more complex pipeline with multiple tasks:

.. code-block:: python

   from orchestrator import Task, Pipeline
   
   # Task 1: Generate story outline
   outline_task = Task(
       id="outline",
       name="Generate Story Outline",
       action="generate_text",
       parameters={"prompt": "Create a story outline about space exploration"}
   )
   
   # Task 2: Write story (depends on outline)
   story_task = Task(
       id="story",
       name="Write Story",
       action="generate_text",
       parameters={"prompt": "Write a story based on: {outline}"},
       dependencies=["outline"]
   )
   
   # Task 3: Summarize story (depends on story)
   summary_task = Task(
       id="summary",
       name="Summarize Story",
       action="generate_text",
       parameters={"prompt": "Summarize this story: {story}"},
       dependencies=["story"]
   )
   
   # Create pipeline with all tasks
   pipeline = Pipeline(id="story_pipeline", name="Story Creation Pipeline")
   pipeline.add_task(outline_task)
   pipeline.add_task(story_task)
   pipeline.add_task(summary_task)
   
   # Execute pipeline
   import asyncio

   

   async def run_pipeline():

       result = await orchestrator.execute_pipeline(pipeline)

       return result

   

   # Run the pipeline

   result = asyncio.run(run_pipeline())
   print(f"Outline: {result['outline']}")
   print(f"Story: {result['story']}")
   print(f"Summary: {result['summary']}")

YAML Configuration
-----------------

You can also define pipelines in YAML:

.. code-block:: yaml

   # story_pipeline.yaml
   id: story_pipeline
   name: Story Creation Pipeline
   
   tasks:
     - id: outline
       name: Generate Story Outline
       action: generate_text
       parameters:
         prompt: "Create a story outline about space exploration"
     
     - id: story
       name: Write Story
       action: generate_text
       parameters:
         prompt: "Write a story based on: {outline}"
       dependencies:
         - outline
     
     - id: summary
       name: Summarize Story
       action: generate_text
       parameters:
         prompt: "Summarize this story: {story}"
       dependencies:
         - story

Load and execute the YAML pipeline:

.. code-block:: python

   from orchestrator.compiler import YAMLCompiler
   
   # Load pipeline from YAML
   compiler = YAMLCompiler()
   pipeline = compiler.compile_file("story_pipeline.yaml")
   
   # Execute pipeline
   import asyncio

   

   async def run_pipeline():

       result = await orchestrator.execute_pipeline(pipeline)

       return result

   

   # Run the pipeline

   result = asyncio.run(run_pipeline())

Real AI Models
--------------

Let's use a real AI model instead of the mock:

.. code-block:: python

   import os
   from orchestrator.models.openai_model import OpenAIModel
   
   # API key should be set in environment variable or ~/.orchestrator/.env
   # Create OpenAI model
   openai_model = OpenAIModel(
       name="gpt-4",
       api_key=os.environ.get("OPENAI_API_KEY"),  # Loaded from environment
       model="gpt-4"
   )
   
   # Register model
   orchestrator.register_model(openai_model)
   
   # Execute pipeline (will use OpenAI)
   import asyncio

   

   async def run_pipeline():

       result = await orchestrator.execute_pipeline(pipeline)

       return result

   

   # Run the pipeline

   result = asyncio.run(run_pipeline())

Error Handling
--------------

Orchestrator provides built-in error handling:

.. code-block:: python

   from orchestrator.core.error_handler import ErrorHandler
   
   # Create error handler with retry strategy
   error_handler = ErrorHandler()
   
   # Configure orchestrator with error handling
   orchestrator = Orchestrator(error_handler=error_handler)
   
   # Execute pipeline with automatic retry on failures
   try:
       import asyncio

       

       async def run_pipeline():

           result = await orchestrator.execute_pipeline(pipeline)

           return result

       

       # Run the pipeline

       result = asyncio.run(run_pipeline())
   except Exception as e:
       print(f"Pipeline failed: {e}")

State Management
---------------

Enable checkpointing for long-running pipelines:

.. code-block:: python

   from orchestrator.state import StateManager
   
   # Create state manager
   state_manager = StateManager(storage_path="./checkpoints")
   
   # Configure orchestrator with state management
   orchestrator = Orchestrator(state_manager=state_manager)
   
   # Execute pipeline with automatic checkpointing
   import asyncio

   

   async def run_pipeline():

       result = await orchestrator.execute_pipeline(pipeline)

       return result

   

   # Run the pipeline

   result = asyncio.run(run_pipeline())

Monitoring & Logging
--------------------

Enable monitoring to track pipeline execution:

.. code-block:: python

   import logging
   
   # Enable debug logging
   logging.basicConfig(level=logging.DEBUG)
   
   # Execute pipeline with logging
   import asyncio

   

   async def run_pipeline():

       result = await orchestrator.execute_pipeline(pipeline)

       return result

   

   # Run the pipeline

   result = asyncio.run(run_pipeline())
   
   # Get execution statistics
   stats = orchestrator.get_execution_stats()
   print(f"Execution time: {stats['total_time']:.2f}s")
   print(f"Tasks completed: {stats['completed_tasks']}")

Next Steps
----------

Now that you've created your first pipeline, explore these topics:

**Core Concepts**
   Learn about :doc:`basic_concepts` like Tasks, Pipelines, and Models

**YAML Configuration**
   Deep dive into :doc:`../user_guide/yaml_configuration`

**Model Integration**
   Connect real AI models in :doc:`../user_guide/models_and_adapters`

**Advanced Features**
   Explore :doc:`../advanced/performance_optimization` and :doc:`../advanced/custom_models`

**Interactive Tutorials**
   Try the :doc:`../tutorials/notebooks` for hands-on learning

Common Patterns
---------------

Here are some common patterns to get you started:

**Sequential Processing**
   Tasks that depend on previous results

**Parallel Processing**
   Independent tasks that can run simultaneously

**Conditional Logic**
   Tasks that execute based on conditions

**Data Transformation**
   Tasks that process and transform data

**Multi-Model Orchestration**
   Using different models for different tasks

.. tip::
   Start with simple pipelines and gradually add complexity as you learn the framework. The mock models are perfect for testing and development before switching to real AI models.

.. note::
   Remember to set up your API keys when using real AI models. See :doc:`installation` for configuration details.