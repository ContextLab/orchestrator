===============
Getting Started
===============

Welcome to Orchestrator! This guide will help you understand the core concepts and get you up and running quickly.

What is Orchestrator?
====================

Orchestrator is an AI pipeline orchestration framework designed to make complex AI workflows simple and reusable. It provides:

- **YAML-based pipeline definitions** that are easy to read and write
- **Input-agnostic pipelines** that can be reused with different parameters
- **Intelligent ambiguity resolution** using AI models to fill in uncertain values
- **Tool integration** for real-world actions via the Model Context Protocol (MCP)
- **Multi-model support** with intelligent selection based on capabilities
- **State management** with checkpointing and recovery

Core Concepts
=============

Pipelines
---------

A pipeline is a collection of tasks that work together to achieve a goal. Pipelines are defined in YAML and can include:

.. code-block:: yaml

   name: research-report
   description: Generate comprehensive research reports
   
   inputs:
     topic:
       type: string
       description: Research topic
       required: true
   
   steps:
     - id: search
       action: search_web
       parameters:
         query: "{{ inputs.topic }} latest research"

Tasks
-----

Tasks are the building blocks of pipelines. Each task has:

- **ID**: Unique identifier within the pipeline
- **Action**: What the task should do
- **Parameters**: Input data for the task
- **Dependencies**: Other tasks that must complete first

AUTO Tags
---------

When you're unsure about a value, use ``<AUTO>`` tags to let AI models decide:

.. code-block:: yaml

   parameters:
     method: <AUTO>Choose best analysis method for this data</AUTO>
     depth: <AUTO>Determine appropriate depth level</AUTO>

Tools
-----

Tools provide real-world capabilities to your pipelines:

- **Web Tools**: Search, scrape, and interact with websites
- **System Tools**: Execute commands and manage files
- **Data Tools**: Process and validate data

Quick Start Example
===================

Let's create a simple research pipeline:

1. **Create a pipeline definition** (``research.yaml``):

.. code-block:: yaml

   name: quick-research
   description: Quick research on any topic
   
   inputs:
     topic:
       type: string
       required: true
   
   outputs:
     report:
       type: string
       value: "{{ inputs.topic }}_report.md"
   
   steps:
     - id: search
       action: search_web
       parameters:
         query: "{{ inputs.topic }}"
         max_results: 5
     
     - id: summarize
       action: generate_summary
       parameters:
         content: "$results.search"
         style: <AUTO>Choose appropriate summary style</AUTO>

2. **Run the pipeline**:

.. code-block:: python

   import orchestrator as orc
   
   # Initialize models
   orc.init_models()
   
   # Compile the pipeline
   pipeline = orc.compile("research.yaml")
   
   # Execute with different topics
   result1 = pipeline.run(topic="artificial intelligence")
   result2 = pipeline.run(topic="climate change")
   
   print(f"Reports generated: {result1}, {result2}")

Key Features in Action
======================

Input-Agnostic Design
---------------------

The same pipeline works with different inputs:

.. tabs::

   .. tab:: Python

      .. code-block:: python

         # One pipeline, many uses
         pipeline = orc.compile("report-template.yaml")
         
         # Generate different reports
         ai_report = pipeline.run(topic="AI", style="technical")
         bio_report = pipeline.run(topic="Biology", style="educational")
         eco_report = pipeline.run(topic="Economics", style="executive")

   .. tab:: YAML

      .. code-block:: yaml

         inputs:
           topic:
             type: string
             required: true
           style:
             type: string
             default: "technical"

Tool Integration
----------------

Tools are automatically detected and made available:

.. code-block:: yaml

   steps:
     - id: fetch_data
       action: search_web        # Auto-detects web tool
       
     - id: save_results
       action: write_file        # Auto-detects filesystem tool
       
     - id: run_analysis
       action: "!python analyze.py"  # Auto-detects terminal tool

Model Selection
---------------

The framework intelligently selects the best model for each task:

.. code-block:: python

   # Models are selected based on:
   # - Task requirements (reasoning, coding, etc.)
   # - Available resources
   # - Performance history
   
   registry = orc.init_models()
   print(registry.list_models())
   # Output: ['ollama:gemma2:27b', 'ollama:llama3.2:1b', 'huggingface:gpt2']

Next Steps
==========

Now that you understand the basics:

1. :doc:`installation` - Set up Orchestrator in your environment
2. :doc:`quickstart` - Build your first pipeline
3. :doc:`tutorials/index` - Learn through hands-on tutorials
4. :doc:`examples/index` - Explore real-world examples

.. tip::

   Join our community on GitHub to ask questions, share pipelines, and contribute to the project!