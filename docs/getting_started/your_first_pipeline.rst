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

First, let's set up our environment:

.. code-block:: python

   import asyncio
   from orchestrator import Orchestrator, Task, Pipeline
   from orchestrator.models.mock_model import MockModel
   
   # Create a mock model for testing
   model = MockModel("research_assistant")
   
   # Set up responses for our mock model
   model.set_response(
       "Generate 3 research questions about: artificial intelligence",
       "1. How does AI impact job markets?\n2. What are the ethical implications of AI?\n3. How can AI be made more accessible?"
   )
   
   model.set_response(
       "Analyze these questions and identify key themes: 1. How does AI impact job markets?\n2. What are the ethical implications of AI?\n3. How can AI be made more accessible?",
       "Key themes identified: Economic Impact, Ethics and Responsibility, Accessibility and Democratization"
   )
   
   model.set_response(
       "Write a comprehensive report on artificial intelligence covering these themes: Economic Impact, Ethics and Responsibility, Accessibility and Democratization",
       "# AI Research Report\n\n## Economic Impact\nAI is reshaping job markets...\n\n## Ethics and Responsibility\nAI systems must be developed responsibly...\n\n## Accessibility and Democratization\nMaking AI tools accessible to all..."
   )

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

Now let's execute our pipeline:

.. code-block:: python

   async def run_pipeline():
       # Create orchestrator
       orchestrator = Orchestrator()
       
       # Register our model
       orchestrator.register_model(model)
       
       print("Starting pipeline execution...")
       
       # Execute pipeline
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
       state_manager = StateManager(storage_path="./checkpoints")
       
       # Create orchestrator with state management
       orchestrator = Orchestrator(state_manager=state_manager)
       orchestrator.register_model(model)
       
       print("Starting stateful pipeline execution...")
       
       # Execute with automatic checkpointing
       result = await orchestrator.execute_pipeline(pipeline)
       
       print("✅ Pipeline completed with checkpointing!")
       
       # List checkpoints created
       checkpoints = await state_manager.list_checkpoints("research_assistant")
       print(f"Checkpoints created: {len(checkpoints)}")
       
       return result
   
   # Run stateful pipeline
   # In Jupyter notebooks: result = await run_stateful_pipeline()
   # In regular Python scripts:
   result = asyncio.run(run_stateful_pipeline())

Step 7: YAML Configuration
--------------------------

Let's convert our pipeline to YAML:

.. code-block:: yaml

   # research_pipeline.yaml
   id: research_assistant
   name: Research Assistant Pipeline
   description: Generates research questions, analyzes themes, and writes a report
   
   context:
     topic: artificial intelligence
   
   tasks:
     - id: research_questions
       name: Generate Research Questions
       action: generate_text
       parameters:
         prompt: "Generate 3 research questions about: {topic}"
         max_tokens: 200
     
     - id: analyze_themes
       name: Analyze Key Themes
       action: generate_text
       parameters:
         prompt: "Analyze these questions and identify key themes: {research_questions}"
         max_tokens: 150
       dependencies:
         - research_questions
     
     - id: write_report
       name: Write Research Report
       action: generate_text
       parameters:
         prompt: "Write a comprehensive report on {topic} covering these themes: {analyze_themes}"
         max_tokens: 500
       dependencies:
         - analyze_themes

Load and execute the YAML pipeline:

.. code-block:: python

   from orchestrator.compiler import YAMLCompiler
   
   async def run_yaml_pipeline():
       # Create compiler and load pipeline
       compiler = YAMLCompiler()
       pipeline = compiler.compile_file("research_pipeline.yaml")
       
       # Create orchestrator
       orchestrator = Orchestrator()
       orchestrator.register_model(model)
       
       print("Starting YAML pipeline execution...")
       
       # Execute pipeline
       result = await orchestrator.execute_pipeline(pipeline)
       
       print("✅ YAML pipeline completed!")
       return result
   
   # Run YAML pipeline
   # In Jupyter notebooks: result = await run_yaml_pipeline()
   # In regular Python scripts:
   result = asyncio.run(run_yaml_pipeline())

Step 8: Real AI Models
----------------------

Replace mock model with real AI:

.. code-block:: python

   from orchestrator.models.openai_model import OpenAIModel
   
   async def run_with_real_ai():
       # Create OpenAI model
       openai_model = OpenAIModel(
           name="gpt-4",
           api_key="your-openai-api-key",
           model="gpt-4"
       )
       
       # Create orchestrator with real AI
       orchestrator = Orchestrator()
       orchestrator.register_model(openai_model)
       
       print("Starting pipeline with real AI...")
       
       # Execute pipeline with real AI
       result = await orchestrator.execute_pipeline(pipeline)
       
       print("✅ Real AI pipeline completed!")
       return result
   
   # Run with real AI (uncomment when you have API keys)
   # In Jupyter notebooks: result = await run_with_real_ai()
   # In regular Python scripts: result = asyncio.run(run_with_real_ai())

Step 9: Monitoring and Analytics
--------------------------------

Add monitoring to track performance:

.. code-block:: python

   import time
   from orchestrator.core.resource_allocator import ResourceAllocator
   
   async def run_monitored_pipeline():
       # Create resource allocator for monitoring
       allocator = ResourceAllocator()
       
       # Create orchestrator with monitoring
       orchestrator = Orchestrator(resource_allocator=allocator)
       orchestrator.register_model(model)
       
       print("Starting monitored pipeline execution...")
       start_time = time.time()
       
       # Execute pipeline
       result = await orchestrator.execute_pipeline(pipeline)
       
       end_time = time.time()
       execution_time = end_time - start_time
       
       print(f"✅ Pipeline completed in {execution_time:.2f} seconds")
       
       # Get resource statistics
       stats = allocator.get_overall_statistics()
       print(f"Resource utilization: {stats['overall_utilization']:.2f}")
       
       return result
   
   # Run monitored pipeline
   # In Jupyter notebooks: result = await run_monitored_pipeline()
   # In regular Python scripts:
   result = asyncio.run(run_monitored_pipeline())

Complete Example
----------------

Here's the complete, production-ready pipeline:

.. code-block:: python

   import asyncio
   import logging
   from orchestrator import Orchestrator, Task, Pipeline
   from orchestrator.models.mock_model import MockModel
   from orchestrator.core.error_handler import ErrorHandler
   from orchestrator.state import StateManager
   from orchestrator.core.resource_allocator import ResourceAllocator
   
   # Configure logging
   logging.basicConfig(level=logging.INFO)
   logger = logging.getLogger(__name__)
   
   async def create_research_pipeline():
       """Create a production-ready research assistant pipeline."""
       
       # Create mock model with responses
       model = MockModel("research_assistant")
       model.set_response(
           "Generate 3 research questions about: artificial intelligence",
           "1. How does AI impact job markets?\n2. What are the ethical implications of AI?\n3. How can AI be made more accessible?"
       )
       model.set_response(
           "Analyze these questions and identify key themes: 1. How does AI impact job markets?\n2. What are the ethical implications of AI?\n3. How can AI be made more accessible?",
           "Key themes: Economic Impact, Ethics and Responsibility, Accessibility"
       )
       model.set_response(
           "Write a comprehensive report on artificial intelligence covering these themes: Economic Impact, Ethics and Responsibility, Accessibility",
           "# AI Research Report\n\n## Economic Impact\nAI is transforming industries...\n\n## Ethics\nResponsible AI development...\n\n## Accessibility\nDemocratizing AI tools..."
       )
       
       # Create tasks
       tasks = [
           Task(
               id="research_questions",
               name="Generate Research Questions",
               action="generate_text",
               parameters={
                   "prompt": "Generate 3 research questions about: {topic}",
                   "max_tokens": 200
               }
           ),
           Task(
               id="analyze_themes",
               name="Analyze Key Themes",
               action="generate_text",
               parameters={
                   "prompt": "Analyze these questions and identify key themes: {research_questions}",
                   "max_tokens": 150
               },
               dependencies=["research_questions"]
           ),
           Task(
               id="write_report",
               name="Write Research Report",
               action="generate_text",
               parameters={
                   "prompt": "Write a comprehensive report on {topic} covering these themes: {analyze_themes}",
                   "max_tokens": 500
               },
               dependencies=["analyze_themes"]
           )
       ]
       
       # Create pipeline
       pipeline = Pipeline(
           id="research_assistant",
           name="Research Assistant Pipeline"
       )
       
       for task in tasks:
           pipeline.add_task(task)
       
       pipeline.set_context("topic", "artificial intelligence")
       
       # Create components
       error_handler = ErrorHandler()
       state_manager = StateManager(storage_path="./checkpoints")
       resource_allocator = ResourceAllocator()
       
       # Create orchestrator
       orchestrator = Orchestrator(
           error_handler=error_handler,
           state_manager=state_manager,
           resource_allocator=resource_allocator
       )
       
       orchestrator.register_model(model)
       
       return orchestrator, pipeline
   
   async def main():
       """Main execution function."""
       logger.info("Creating research assistant pipeline...")
       
       orchestrator, pipeline = await create_research_pipeline()
       
       logger.info("Executing pipeline...")
       
       try:
           result = await orchestrator.execute_pipeline(pipeline)
           
           logger.info("Pipeline completed successfully!")
           
           print("\n=== Results ===")
           for task_id, output in result.items():
               print(f"\n{task_id}:")
               print(f"{output}")
           
       except Exception as e:
           logger.error(f"Pipeline failed: {e}")
           raise
   
   # Run the complete example
   if __name__ == "__main__":
       asyncio.run(main())

What You've Learned
-------------------

Congratulations! You've built a complete AI pipeline with:

✅ **Task Creation** - Defined individual work units  
✅ **Dependencies** - Managed task execution order  
✅ **Parameter Passing** - Connected task outputs to inputs  
✅ **Error Handling** - Added retry strategies and circuit breakers  
✅ **State Management** - Enabled checkpointing for reliability  
✅ **YAML Configuration** - Declarative pipeline definition  
✅ **Monitoring** - Resource tracking and performance analytics  

Next Steps
----------

Now you're ready to:

* **Explore Advanced Features** - :doc:`../advanced/performance_optimization`
* **Learn YAML Configuration** - :doc:`../user_guide/yaml_configuration`
* **Integrate Real Models** - :doc:`../user_guide/models_and_adapters`
* **Try Interactive Tutorials** - :doc:`../tutorials/notebooks`

.. tip::
   Start building your own pipelines! The framework is designed to be flexible - you can adapt this pattern to any AI workflow you need to create.

.. note::
   Remember to replace mock models with real AI models for production use. See the :doc:`../user_guide/models_and_adapters` guide for integration details.