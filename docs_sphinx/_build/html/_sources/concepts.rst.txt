============
Key Concepts
============

Understanding the fundamental concepts behind Orchestrator will help you build more effective pipelines and make the most of the framework's capabilities.

Pipelines
==========

A **pipeline** is a collection of interconnected tasks that work together to achieve a specific goal. Think of it as a recipe or workflow that can be executed automatically.

Input-Agnostic Design
---------------------

One of Orchestrator's core innovations is **input-agnostic pipelines**. This means a single pipeline definition can work with different inputs to produce different outputs:

.. code-block:: yaml

   # One pipeline definition
   name: research-pipeline
   
   inputs:
     topic: { type: string, required: true }
     depth: { type: string, default: "medium" }
   
   steps:
     - id: research
       action: search_web
       parameters:
         query: "{{ inputs.topic }}"

.. code-block:: python

   # Many different uses
   pipeline = orc.compile("research-pipeline.yaml")
   
   ai_report = pipeline.run(topic="artificial intelligence")
   climate_report = pipeline.run(topic="climate change") 
   space_report = pipeline.run(topic="space exploration")

This design promotes **reusability** and **maintainability** - write once, use many times.

Tasks
=====

A **task** is the fundamental unit of work in a pipeline. Each task represents a single operation or action.

Task Anatomy
------------

Every task has these key components:

.. code-block:: yaml

   - id: unique_identifier        # Required: Unique name
     action: what_to_do           # Required: Action to perform  
     description: "What it does"  # Optional: Human description
     parameters:                  # Optional: Input parameters
       key: value
     depends_on: [other_task]     # Optional: Dependencies
     condition: "when_to_run"     # Optional: Conditional execution

Task Dependencies
-----------------

Tasks can depend on other tasks, creating execution ordering:

.. code-block:: yaml

   steps:
     - id: fetch_data
       action: download_file
       parameters:
         url: "{{ inputs.data_url }}"
     
     - id: process_data
       depends_on: [fetch_data]   # Runs after fetch_data
       action: transform_data
       parameters:
         data: "$results.fetch_data"
     
     - id: save_results
       depends_on: [process_data] # Runs after process_data
       action: write_file
       parameters:
         content: "$results.process_data"

Templates and References
========================

Orchestrator uses **Jinja2 templating** to make pipelines dynamic and data-driven.

Template Syntax
---------------

.. code-block:: yaml

   # Access input values
   query: "{{ inputs.search_term }}"
   
   # Reference results from other tasks
   data: "$results.previous_task"
   
   # Use filters and functions
   filename: "{{ inputs.name | slugify }}.pdf"
   
   # Conditional expressions
   mode: "{{ 'advanced' if inputs.premium else 'basic' }}"

Runtime vs Compile-Time Resolution
----------------------------------

Templates are resolved at different stages:

- **Compile-time**: Static values resolved when pipeline is compiled
- **Runtime**: Dynamic values resolved during execution

.. code-block:: yaml

   steps:
     - id: example
       parameters:
         # Compile-time: resolved once during compilation
         timestamp: "{{ compile_time.now }}"
         
         # Runtime: resolved during each execution
         user_input: "{{ inputs.query }}"
         previous_result: "$results.other_task"

AUTO Tags
=========

**AUTO tags** are Orchestrator's solution to ambiguous or uncertain values. When you're not sure what value to use, let an AI model decide:

.. code-block:: yaml

   parameters:
     # Simple AUTO tag
     style: <AUTO>Choose appropriate writing style</AUTO>
     
     # Context-aware AUTO tag
     method: <AUTO>Based on data type {{ results.data.type }}, choose best analysis method</AUTO>
     
     # Complex AUTO tag with instructions
     sections: |
       <AUTO>
       For a report about {{ inputs.topic }}, determine which sections to include:
       - Executive Summary: yes/no
       - Technical Details: yes/no  
       - Future Outlook: yes/no
       Return as JSON object
       </AUTO>

How AUTO Tags Work
------------------

1. **Detection**: The compiler identifies AUTO tags in the pipeline
2. **Context**: Gathers relevant context (inputs, previous results, etc.)
3. **Resolution**: Sends context to an AI model for decision
4. **Substitution**: Replaces AUTO tag with the model's response

Tools and Actions
=================

**Tools** provide real-world capabilities to your pipelines - they're how pipelines interact with the outside world.

Tool Categories
---------------

**Web Tools**:
- Search the internet
- Scrape websites
- Interact with web pages

**System Tools**:
- Execute commands
- Manage files
- Run scripts

**Data Tools**:
- Process and transform data
- Validate information
- Convert between formats

**AI Tools**:
- Generate content
- Analyze text
- Extract information

Action Names
------------

Actions are how you invoke tools in pipelines:

.. code-block:: yaml

   # Web search
   - action: search_web
     parameters:
       query: "machine learning"
   
   # File operations
   - action: write_file
     parameters:
       path: "output.txt"
       content: "Hello world"
   
   # Shell commands (prefix with !)
   - action: "!ls -la"
   
   # AI generation
   - action: generate_content
     parameters:
       prompt: "Write a summary about {{ topic }}"

Automatic Tool Detection
-----------------------

Orchestrator automatically detects required tools from your pipeline:

.. code-block:: yaml

   steps:
     - action: search_web        # → Requires web tool
     - action: "!python script.py"  # → Requires terminal tool
     - action: write_file        # → Requires filesystem tool

Tools are registered and made available via the **Model Context Protocol (MCP)**.

Models and Intelligence
=======================

**Models** provide the AI capabilities that power AUTO tag resolution and content generation.

Model Types
-----------

**Local Models** (via Ollama):
- Run on your machine
- No API costs
- Privacy and control

**Cloud Models** (OpenAI, Anthropic):
- Powerful capabilities
- API-based
- Pay per use

**Specialized Models**:
- Code generation
- Data analysis
- Specific domains

Intelligent Model Selection
--------------------------

Orchestrator chooses the best model for each task based on:

- **Task requirements** (reasoning, coding, analysis)
- **Available resources** (memory, GPU, time)
- **Performance history** (success rates, quality scores)
- **Cost considerations** (API costs, efficiency)

.. code-block:: python

   # Models are selected automatically
   registry = orc.init_models()
   
   # Available models are ranked by capability
   print(registry.list_models())
   # ['ollama:gemma2:27b', 'ollama:llama3.2:1b', 'huggingface:gpt2']

State Management
================

**State management** ensures pipeline reliability and recovery.

Checkpointing
-------------

Orchestrator can save pipeline state at task boundaries:

.. code-block:: yaml

   config:
     checkpoint: true  # Enable automatic checkpointing
   
   steps:
     - id: expensive_task
       action: long_running_process
       checkpoint: true  # Force checkpoint after this step

Recovery
--------

If a pipeline fails, it can resume from the last checkpoint:

.. code-block:: python

   # Pipeline fails at step 5
   pipeline.run(inputs)  # Fails
   
   # Resume from last checkpoint
   pipeline.resume()  # Continues from step 4

This is especially valuable for:
- Long-running pipelines
- Expensive operations
- Unreliable external services

Control Systems
===============

**Control systems** are the execution engines that run your pipelines.

Built-in Control Systems
------------------------

**MockControlSystem**:
- For testing and development
- Simulates tool execution
- Fast and predictable

**ToolIntegratedControlSystem**:
- Real tool execution
- Full MCP integration
- Production-ready

Custom Control Systems
---------------------

You can create custom control systems for specific needs:

.. code-block:: python

   from orchestrator.core.control_system import ControlSystem
   
   class MyControlSystem(ControlSystem):
       async def execute_task(self, task, context):
           # Custom execution logic
           pass

Pipeline Composition
====================

Complex workflows can be built by composing smaller pipelines.

Pipeline Imports
---------------

.. code-block:: yaml

   imports:
     - common/validation.yaml as validator
     - workflows/analysis.yaml as analyzer
   
   steps:
     - id: validate
       pipeline: validator
       inputs:
         data: "{{ inputs.raw_data }}"
     
     - id: analyze  
       pipeline: analyzer
       inputs:
         validated_data: "$results.validate"

Modular Design
--------------

This enables:
- **Reusability**: Share common patterns
- **Maintainability**: Update once, use everywhere  
- **Collaboration**: Teams can work on different components
- **Testing**: Test pipelines in isolation

Error Handling
==============

Robust pipelines handle errors gracefully.

Error Strategies
---------------

.. code-block:: yaml

   steps:
     - id: risky_task
       action: external_api_call
       error_handling:
         # Retry with backoff
         retry:
           max_attempts: 3
           backoff: exponential
         
         # Fallback action
         fallback:
           action: use_cached_data
         
         # Continue pipeline on error
         continue_on_error: true

Error Types
-----------

- **Network errors**: Connection failures, timeouts
- **Data errors**: Invalid formats, missing fields
- **Logic errors**: Failed validation, impossible conditions
- **Resource errors**: Out of memory, disk space

Performance Concepts
====================

Understanding performance helps you build efficient pipelines.

Parallel Execution
------------------

Tasks without dependencies can run in parallel:

.. code-block:: yaml

   steps:
     # These run in parallel
     - id: source1
       action: fetch_data_a
     
     - id: source2  
       action: fetch_data_b
     
     # This waits for both
     - id: combine
       depends_on: [source1, source2]
       action: merge_data

Caching
-------

Expensive operations can be cached:

.. code-block:: yaml

   steps:
     - id: expensive_computation
       action: complex_analysis
       cache:
         enabled: true
         key: "{{ inputs.data_hash }}"
         ttl: 3600  # 1 hour

Resource Management
------------------

Control resource usage:

.. code-block:: yaml

   config:
     resources:
       max_memory: "8GB"
       max_threads: 4
       gpu_enabled: false

Security Concepts  
=================

Security is built into Orchestrator's design.

Sandboxing
----------

Code execution happens in isolated environments:
- **Docker containers** for full isolation
- **Restricted permissions** for file access
- **Network controls** for external access

Input Validation
----------------

All inputs are validated:

.. code-block:: yaml

   inputs:
     email:
       type: string
       validation:
         pattern: "^[\\w.-]+@[\\w.-]+\\.\\w+$"
     
     amount:
       type: number
       validation:
         min: 0
         max: 10000

Secret Management
-----------------

Sensitive data is handled securely:

.. code-block:: yaml

   parameters:
     api_key: "{{ env.SECRET_API_KEY }}"  # From environment
     password: "{{ vault.db_password }}"   # From secret vault

Best Practices
==============

Design Principles
-----------------

1. **Single Responsibility**: Each task does one thing well
2. **Loose Coupling**: Tasks don't depend on implementation details
3. **High Cohesion**: Related tasks are grouped together
4. **Fail Fast**: Validate inputs and catch errors early
5. **Idempotent**: Running the same pipeline multiple times is safe

Pipeline Organization
--------------------

.. code-block::

   pipelines/
   ├── common/           # Shared components
   │   ├── validation.yaml
   │   └── formatting.yaml
   ├── workflows/        # Complete workflows  
   │   ├── research.yaml
   │   └── analysis.yaml
   └── specialized/      # Domain-specific
       ├── finance.yaml
       └── healthcare.yaml

Testing Strategy
---------------

- **Unit test** individual tasks
- **Integration test** complete pipelines  
- **Smoke test** with real data
- **Performance test** under load

Next Steps
==========

Now that you understand the concepts:

1. **Practice** with the :doc:`tutorials/index`
2. **Explore** the :doc:`api/index` for detailed reference
3. **Build** your own pipelines for real problems
4. **Share** your patterns with the community

.. tip::

   The best way to internalize these concepts is to start building. Begin with simple pipelines and gradually add complexity as you become more comfortable with the framework.