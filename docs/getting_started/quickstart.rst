Quickstart Guide
===============

This guide will get you up and running with Orchestrator in 5 minutes. We'll create a simple AI workflow that demonstrates the core concepts.

Prerequisites
-------------

Before you begin, make sure you have:

1. Installed Orchestrator: ``pip install orchestrator``
2. Set up your API keys as environment variables:

   .. code-block:: bash

      export OPENAI_API_KEY="your-openai-api-key"
      # or
      export ANTHROPIC_API_KEY="your-anthropic-api-key"

Your First Pipeline
-------------------

Let's create a simple text generation pipeline with a real AI model:

.. code-block:: python

   import os
   from orchestrator import Orchestrator, Task, Pipeline
   from orchestrator.models.openai_model import OpenAIModel
   from orchestrator.utils.api_keys import load_api_keys
   
   # Load API keys from environment
   load_api_keys()
   
   # Create a real OpenAI model
   model = OpenAIModel(
       name="gpt-3.5-turbo",
       api_key=os.environ.get("OPENAI_API_KEY"),  # Loaded from environment
   )
   
   # Create a task
   task = Task(
       id="greeting",
       name="Generate Greeting",
       action="generate_text",
       parameters={
           "prompt": "Say hello and ask how you can help today",
           "max_tokens": 50
       }
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
       print(f"AI Response: {result['greeting']}")
       return result
   
   # Run the pipeline
   result = asyncio.run(run_pipeline())

That's it! You've just executed your first AI pipeline with a real model.

Using Different Models
----------------------

You can easily switch between different AI providers:

**Using Anthropic's Claude:**

.. code-block:: python

   from orchestrator.models.anthropic_model import AnthropicModel
   
   model = AnthropicModel(
       name="claude-2",
       api_key=os.environ.get("ANTHROPIC_API_KEY"),
   )

**Using OpenAI's GPT-4:**

.. code-block:: python

   model = OpenAIModel(
       name="gpt-4",
       api_key=os.environ.get("OPENAI_API_KEY"),
   )

Building More Complex Pipelines
-------------------------------

Let's create a pipeline with multiple tasks:

.. code-block:: python

   # Task 1: Generate a story idea
   idea_task = Task(
       id="generate_idea",
       name="Generate Story Idea",
       action="generate_text",
       parameters={
           "prompt": "Generate a creative story idea in one sentence",
           "max_tokens": 100
       }
   )
   
   # Task 2: Expand the idea
   expand_task = Task(
       id="expand_story",
       name="Expand Story",
       action="generate_text",
       parameters={
           "prompt": "Expand this idea into a short story: {generate_idea}",
           "max_tokens": 300
       },
       dependencies=["generate_idea"]  # This task depends on the first one
   )
   
   # Create pipeline with both tasks
   story_pipeline = Pipeline(id="story_creator", name="Story Creation Pipeline")
   story_pipeline.add_task(idea_task)
   story_pipeline.add_task(expand_task)
   
   # Execute the pipeline
   async def create_story():
       result = await orchestrator.execute_pipeline(story_pipeline)
       print(f"Story Idea: {result['generate_idea']}")
       print(f"\nFull Story: {result['expand_story']}")
       return result
   
   result = asyncio.run(create_story())

Error Handling
--------------

Add error handling to make your pipelines robust:

.. code-block:: python

   from orchestrator.core.error_handler import ErrorHandler, ExponentialBackoffRetry
   
   # Create error handler with retry strategy
   error_handler = ErrorHandler()
   error_handler.register_retry_strategy(
       "default",
       ExponentialBackoffRetry(max_retries=3, base_delay=1.0)
   )
   
   # Create orchestrator with error handling
   orchestrator = Orchestrator(error_handler=error_handler)
   orchestrator.register_model(model)
   
   try:
       result = await orchestrator.execute_pipeline(pipeline)
       print("Success!")
   except Exception as e:
       print(f"Pipeline failed after retries: {e}")

Working with YAML Pipelines
---------------------------

You can also define pipelines in YAML:

.. code-block:: yaml

   # hello_pipeline.yaml
   name: Hello Pipeline
   description: A simple greeting pipeline
   
   steps:
     - id: greeting
       action: generate_text
       parameters:
         prompt: "Say hello and offer assistance"
         max_tokens: 50
   
   outputs:
     message: "{{ greeting }}"

Load and execute YAML pipelines:

.. code-block:: python

   from orchestrator.compiler import YAMLCompiler
   
   # Load pipeline from YAML
   compiler = YAMLCompiler()
   pipeline = compiler.compile_from_file("hello_pipeline.yaml")
   
   # Execute as before
   result = await orchestrator.execute_pipeline(pipeline)

Next Steps
----------

Now that you've created your first pipelines:

1. **Explore Advanced Features:**
   
   - :doc:`Conditional task execution <../tutorials/conditional_execution>`
   - :doc:`Parallel task processing <../tutorials/parallel_processing>`
   - :doc:`State management and checkpointing <../tutorials/state_management>`

2. **Build Real Applications:**
   
   - Content generation systems
   - Data analysis pipelines
   - Multi-agent workflows
   - Research automation tools

3. **Learn Best Practices:**
   
   - :doc:`Error handling strategies <../best_practices/error_handling>`
   - :doc:`Performance optimization <../best_practices/performance>`
   - :doc:`Security considerations <../best_practices/security>`

Remember: Always use real API keys and models to ensure your pipelines work with actual AI services!