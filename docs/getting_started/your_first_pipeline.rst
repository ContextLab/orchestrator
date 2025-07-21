Your First Pipeline
==================

In this tutorial, we'll build a complete pipeline step by step. You'll learn how to create tasks, manage dependencies, and execute your first AI workflow.

What We'll Build
----------------

We'll create a **Research Assistant Pipeline** that:

1. **Researches** a topic by generating key questions
2. **Analyzes** the questions to identify themes
3. **Synthesizes** findings into a comprehensive report

This pipeline demonstrates task dependencies, parameter passing, and real-world AI orchestration.

Step 1: Setup
-------------

First, let's set up our environment with a real AI model:

.. code-block:: python

   import asyncio
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
   
   # Note: Make sure you have set your OPENAI_API_KEY environment variable
   # You can also use AnthropicModel with ANTHROPIC_API_KEY if preferred

Step 2: Create Tasks
--------------------

Now let's create our three tasks:

.. code-block:: python

   # Task 1: Generate research questions
   research_task = Task(
       id="research_questions",
       name="Generate Research Questions",
       action="generate_text",
       parameters={
           "prompt": "Generate 3 research questions about: {topic}",
           "max_tokens": 200
       }
   )
   
   # Task 2: Analyze questions for themes
   analysis_task = Task(
       id="analyze_themes",
       name="Analyze Key Themes",
       action="generate_text",
       parameters={
           "prompt": "Analyze these questions and identify key themes: {research_questions}",
           "max_tokens": 150
       },
       dependencies=["research_questions"]  # Depends on research task
   )
   
   # Task 3: Write comprehensive report
   report_task = Task(
       id="write_report",
       name="Write Research Report",
       action="generate_text",
       parameters={
           "prompt": "Write a comprehensive report on {topic} covering these themes: {analyze_themes}",
           "max_tokens": 500
       },
       dependencies=["analyze_themes"]  # Depends on analysis task
   )

Step 3: Create Pipeline
-----------------------

Combine tasks into a pipeline:

.. code-block:: python

   # Create pipeline
   pipeline = Pipeline(
       id="research_assistant",
       name="Research Assistant Pipeline",
       description="Generates research questions, analyzes themes, and writes a report"
   )
   
   # Add tasks to pipeline
   pipeline.add_task(research_task)
   pipeline.add_task(analysis_task)
   pipeline.add_task(report_task)
   
   # Set initial context
   pipeline.set_context("topic", "artificial intelligence")
   
   print("Pipeline created successfully!")
   print(f"Tasks: {list(pipeline.tasks.keys())}")
   print(f"Execution order: {pipeline.get_execution_order()}")

Step 4: Execute Pipeline
------------------------

Now let's execute our pipeline with real API calls:

.. code-block:: python

   async def run_pipeline():
       # Create orchestrator
       orchestrator = Orchestrator()
       
       # Register our model
       orchestrator.register_model(model)
       
       print("Starting pipeline execution...")
       
       # Execute pipeline with real API calls
       result = await orchestrator.execute_pipeline(pipeline)
       
       print("\n=== Pipeline Results ===")
       print(f"Research Questions:\n{result['research_questions']}\n")
       print(f"Key Themes:\n{result['analyze_themes']}\n")
       print(f"Final Report:\n{result['write_report']}\n")
       
       return result
   
   # Run the pipeline
   # Note: In Jupyter notebooks, you can use top-level await:
   # result = await run_pipeline()
   
   # In regular Python scripts, use asyncio.run():
   import asyncio
   result = asyncio.run(run_pipeline())

Step 5: Add Error Handling
---------------------------

Let's make our pipeline more robust:

.. code-block:: python

   from orchestrator.core.error_handler import ErrorHandler
   from orchestrator.core.error_handler import ExponentialBackoffRetry
   
   async def run_robust_pipeline():
       # Create error handler with retry strategy
       error_handler = ErrorHandler()
       error_handler.register_retry_strategy(
           "research_retry",
           ExponentialBackoffRetry(max_retries=3, base_delay=1.0)
       )
       
       # Create orchestrator with error handling
       orchestrator = Orchestrator(error_handler=error_handler)
       orchestrator.register_model(model)
       
       try:
           print("Starting robust pipeline execution...")
           result = await orchestrator.execute_pipeline(pipeline)
           print("✅ Pipeline completed successfully!")
           return result
           
       except Exception as e:
           print(f"❌ Pipeline failed: {e}")
           # Get execution statistics
           stats = error_handler.get_error_statistics()
           print(f"Errors encountered: {stats['total_errors']}")
           return None
   
   # Run robust pipeline
   # In Jupyter notebooks: result = await run_robust_pipeline()
   # In regular Python scripts:
   result = asyncio.run(run_robust_pipeline())

Step 6: Add State Management
-----------------------------

For longer pipelines, add checkpointing:

.. code-block:: python

   from orchestrator.state import StateManager
   
   async def run_stateful_pipeline():
       # Create state manager
       state_manager = StateManager(checkpoint_dir="./pipeline_checkpoints")
       
       # Create orchestrator with state management
       orchestrator = Orchestrator(state_manager=state_manager)
       orchestrator.register_model(model)
       
       # Enable checkpointing
       orchestrator.enable_checkpointing(interval_steps=1)
       
       try:
           print("Starting stateful pipeline execution...")
           result = await orchestrator.execute_pipeline(pipeline)
           print("✅ Pipeline completed with checkpointing!")
           return result
           
       except Exception as e:
           print(f"❌ Pipeline failed: {e}")
           print("You can resume from the last checkpoint")
           return None
   
   # Run stateful pipeline
   result = asyncio.run(run_stateful_pipeline())

Complete Example
----------------

Here's the complete code in one place:

.. code-block:: python

   import asyncio
   import os
   from orchestrator import Orchestrator, Task, Pipeline
   from orchestrator.models.openai_model import OpenAIModel
   from orchestrator.utils.api_keys import load_api_keys
   from orchestrator.core.error_handler import ErrorHandler, ExponentialBackoffRetry
   from orchestrator.state import StateManager
   
   async def create_research_pipeline():
       # Load API keys
       load_api_keys()
       
       # Create model with real API
       model = OpenAIModel(
           name="gpt-3.5-turbo",
           api_key=os.environ.get("OPENAI_API_KEY"),
       )
       
       # Create tasks
       research_task = Task(
           id="research_questions",
           name="Generate Research Questions",
           action="generate_text",
           parameters={
               "prompt": "Generate 3 research questions about: artificial intelligence",
               "max_tokens": 200
           }
       )
       
       analysis_task = Task(
           id="analyze_themes",
           name="Analyze Key Themes",
           action="generate_text",
           parameters={
               "prompt": "Analyze these questions and identify key themes: {research_questions}",
               "max_tokens": 150
           },
           dependencies=["research_questions"]
       )
       
       report_task = Task(
           id="write_report",
           name="Write Research Report",
           action="generate_text",
           parameters={
               "prompt": "Write a comprehensive report on artificial intelligence covering these themes: {analyze_themes}",
               "max_tokens": 500
           },
           dependencies=["analyze_themes"]
       )
       
       # Create pipeline
       pipeline = Pipeline(
           id="research_assistant",
           name="Research Assistant Pipeline"
       )
       
       # Add tasks
       pipeline.add_task(research_task)
       pipeline.add_task(analysis_task)
       pipeline.add_task(report_task)
       
       # Set up orchestrator with error handling and state management
       error_handler = ErrorHandler()
       error_handler.register_retry_strategy(
           "default",
           ExponentialBackoffRetry(max_retries=3)
       )
       
       state_manager = StateManager(checkpoint_dir="./checkpoints")
       
       orchestrator = Orchestrator(
           error_handler=error_handler,
           state_manager=state_manager
       )
       orchestrator.register_model(model)
       orchestrator.enable_checkpointing(interval_steps=1)
       
       # Execute pipeline
       print("Starting AI research pipeline...")
       result = await orchestrator.execute_pipeline(pipeline)
       
       print("\n=== Results ===")
       for task_id, output in result.items():
           print(f"\n{task_id}:\n{output}")
       
       return result
   
   # Run the complete pipeline
   if __name__ == "__main__":
       asyncio.run(create_research_pipeline())

Key Takeaways
-------------

You've learned how to:

1. **Create tasks** with parameters and dependencies
2. **Build pipelines** that orchestrate multiple tasks
3. **Use real AI models** with proper API key management
4. **Add error handling** for robust execution
5. **Enable checkpointing** for long-running pipelines

Next Steps
----------

- Try modifying the prompts to research different topics
- Add more tasks to create a deeper analysis
- Experiment with different models (GPT-4, Claude, etc.)
- Explore conditional task execution based on results
- Build your own custom pipelines for your use cases

Remember to always use real API keys and models - this ensures your pipelines work with actual AI services and produce real results!