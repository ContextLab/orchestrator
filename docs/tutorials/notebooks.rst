Interactive Tutorials
====================

The Orchestrator Framework comes with comprehensive Jupyter notebook tutorials that provide hands-on learning experiences. These tutorials are designed to take you from beginner to advanced user.

.. note::
   All tutorials are available in the ``notebooks/`` directory of your Orchestrator installation. You can also `view them on GitHub <https://github.com/ContextLab/orchestrator/tree/main/notebooks>`_.

Tutorial Overview
-----------------

.. raw:: html

   <div class="feature-grid">
       <div class="feature-card">
           <h3>📚 01 - Getting Started</h3>
           <p><strong>Duration:</strong> 30-45 minutes<br>
           <strong>Level:</strong> Beginner<br>
           <strong>Prerequisites:</strong> Basic Python knowledge</p>
           <p>Learn core concepts and build your first pipeline</p>
       </div>
       <div class="feature-card">
           <h3>🔧 02 - YAML Configuration</h3>
           <p><strong>Duration:</strong> 45-60 minutes<br>
           <strong>Level:</strong> Intermediate<br>
           <strong>Prerequisites:</strong> Complete tutorial 01</p>
           <p>Master declarative workflow design with YAML</p>
       </div>
       <div class="feature-card">
           <h3>🚀 03 - Advanced Models</h3>
           <p><strong>Duration:</strong> 60-75 minutes<br>
           <strong>Level:</strong> Advanced<br>
           <strong>Prerequisites:</strong> Complete tutorials 01-02</p>
           <p>Multi-model orchestration and optimization</p>
       </div>
   </div>

Getting Started with Tutorials
-------------------------------

Prerequisites
~~~~~~~~~~~~~

Before starting the tutorials, ensure you have:

1. **Python 3.8+** installed
2. **Jupyter Notebook** or **JupyterLab**
3. **Orchestrator Framework** installed

Installation
~~~~~~~~~~~~

.. code-block:: bash

   # Install Orchestrator Framework
   pip install orchestrator-framework
   
   # Install Jupyter (if not already installed)
   pip install jupyter
   
   # Clone the repository for tutorials
   git clone https://github.com/ContextLab/orchestrator.git
   cd orchestrator

Starting Jupyter
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Start Jupyter Notebook
   jupyter notebook
   
   # Or start JupyterLab
   jupyter lab

Navigate to the ``notebooks/`` directory and open the first tutorial.

Tutorial 01: Getting Started
-----------------------------

.. image:: ../images/tutorial_01_preview.png
   :alt: Tutorial 01 Preview
   :align: center

**File:** ``01_getting_started.ipynb``

This tutorial introduces you to the Orchestrator Framework fundamentals:

**What You'll Learn:**
   * Core concepts: Tasks, Pipelines, Models, Orchestrator
   * Creating and executing your first pipeline
   * Working with mock models for development
   * State management and checkpointing
   * Pipeline progress monitoring
   * Basic serialization and deserialization

**Key Topics:**
   * Task creation and configuration
   * Pipeline dependency management
   * Model registration and selection
   * Error handling basics
   * Result processing and analysis

**Hands-On Exercises:**
   * Build a simple text generation pipeline
   * Create a multi-task workflow with dependencies
   * Implement error handling and retries
   * Add state management for reliability

.. code-block:: python

   # Example from Tutorial 01
   from orchestrator import Orchestrator, Task, Pipeline
   from orchestrator.models.mock_model import MockModel
   
   # Create your first task
   task = Task(
       id="hello_world",
       name="Hello World Task",
       action="generate_text",
       parameters={"prompt": "Hello, Orchestrator!"}
   )
   
   # Build and execute pipeline
   pipeline = Pipeline(id="first_pipeline")
   pipeline.add_task(task)
   
   orchestrator = Orchestrator()
   result = await orchestrator.execute_pipeline(pipeline)

Tutorial 02: YAML Configuration
-------------------------------

.. image:: ../images/tutorial_02_preview.png
   :alt: Tutorial 02 Preview
   :align: center

**File:** ``02_yaml_configuration.ipynb``

This tutorial focuses on declarative workflow design:

**What You'll Learn:**
   * Defining workflows declaratively in YAML
   * Template variables and AUTO resolution
   * Pipeline compilation from YAML
   * Advanced YAML features and validation
   * Best practices for YAML workflow design
   * Error handling and debugging

**Key Topics:**
   * YAML syntax for pipelines
   * Template variable substitution
   * Conditional logic in YAML
   * Schema validation
   * Auto-resolution of ambiguous parameters

**Hands-On Exercises:**
   * Convert Python pipelines to YAML
   * Use template variables for dynamic workflows
   * Implement conditional task execution
   * Create reusable pipeline templates

.. code-block:: yaml

   # Example from Tutorial 02
   id: research_pipeline
   name: Research Assistant Pipeline
   
   context:
     topic: artificial intelligence
   
   tasks:
     - id: research
       name: Generate Research Questions
       action: generate_text
       parameters:
         prompt: "Research questions about: {topic}"
     
     - id: analyze
       name: Analyze Themes
       action: generate_text
       parameters:
         prompt: "Analyze themes in: {research}"
       dependencies:
         - research

Tutorial 03: Advanced Model Integration
---------------------------------------

.. image:: ../images/tutorial_03_preview.png
   :alt: Tutorial 03 Preview
   :align: center

**File:** ``03_advanced_model_integration.ipynb``

This tutorial covers production-ready model orchestration:

**What You'll Learn:**
   * Model capabilities and requirements
   * Intelligent model selection algorithms
   * Fallback strategies and error handling
   * Performance monitoring and cost analysis
   * Load balancing and optimization
   * Real-world integration patterns

**Key Topics:**
   * Multi-model workflows
   * Cost optimization strategies
   * Performance monitoring
   * Model health checking
   * Advanced error handling

**Hands-On Exercises:**
   * Integrate multiple AI providers
   * Implement fallback strategies
   * Monitor model performance
   * Optimize for cost and latency

.. code-block:: python

   # Example from Tutorial 03
   from orchestrator.models.openai_model import OpenAIModel
   from orchestrator.models.anthropic_model import AnthropicModel
   
   # Register multiple models
   gpt4 = OpenAIModel(name="gpt-4", api_key="your-key")
   claude = AnthropicModel(name="claude-3", api_key="your-key")
   
   orchestrator.register_model(gpt4)
   orchestrator.register_model(claude)
   
   # Orchestrator automatically selects best model
   result = await orchestrator.execute_pipeline(pipeline)

Running the Tutorials
---------------------

Interactive Execution
~~~~~~~~~~~~~~~~~~~~

Each tutorial is designed to be run interactively:

1. **Read the explanation** in each cell
2. **Run the code examples** step by step
3. **Experiment** with the parameters
4. **Complete the exercises** at the end of each section

Code Examples
~~~~~~~~~~~~~

All code examples are runnable and include:

* **Complete implementations** that you can copy and modify
* **Clear explanations** of what each part does
* **Expected outputs** so you know what to expect
* **Troubleshooting tips** for common issues

Exercises and Challenges
~~~~~~~~~~~~~~~~~~~~~~~

Each tutorial includes:

* **Guided exercises** with step-by-step instructions
* **Challenge problems** to test your understanding
* **Solutions** provided at the end of each notebook
* **Extension ideas** for further exploration

Tutorial Support Files
----------------------

The tutorials come with supporting files:

.. code-block:: text

   notebooks/
   ├── 01_getting_started.ipynb
   ├── 02_yaml_configuration.ipynb
   ├── 03_advanced_model_integration.ipynb
   ├── README.md                           # Tutorial guide
   ├── data/                               # Sample data files
   │   ├── sample_pipeline.yaml
   │   ├── complex_workflow.yaml
   │   └── test_data.json
   ├── images/                             # Tutorial images
   │   ├── architecture_diagram.png
   │   └── workflow_visualization.png
   └── solutions/                          # Exercise solutions
       ├── 01_solutions.ipynb
       ├── 02_solutions.ipynb
       └── 03_solutions.ipynb

Best Practices for Learning
---------------------------

To get the most out of these tutorials:

1. **Follow the order** - Each tutorial builds on the previous one
2. **Run every cell** - Don't just read, execute the code
3. **Experiment** - Modify parameters and see what happens
4. **Complete exercises** - They reinforce key concepts
5. **Ask questions** - Use the discussion forums for help
6. **Build your own** - Apply concepts to your own projects

Common Issues and Solutions
--------------------------

**Jupyter Not Starting**
   .. code-block:: bash

      # Try updating Jupyter
      pip install --upgrade jupyter
      
      # Or install JupyterLab
      pip install jupyterlab

**Import Errors**
   .. code-block:: python

      # Make sure Orchestrator is installed
      pip install orchestrator-framework
      
      # Or install in development mode
      pip install -e .

**Mock Model Issues**
   .. code-block:: python

      # Mock models need explicit responses
      model.set_response("your prompt", "expected response")

**Async/Await Problems**
   .. code-block:: python

      # Use await in notebook cells
      result = await orchestrator.execute_pipeline(pipeline)

Advanced Tutorial Topics
------------------------

After completing the core tutorials, explore these advanced topics:

**Custom Model Development**
   Learn to create your own model adapters

**Control System Integration**
   Integrate with LangGraph and MCP

**Performance Optimization**
   Optimize pipelines for speed and cost

**Production Deployment**
   Deploy pipelines to production environments

**Monitoring and Analytics**
   Track performance and usage metrics

Getting Help
------------

If you need help with the tutorials:

1. **Check the README** - Each tutorial has detailed instructions
2. **Review solutions** - Compare your code with provided solutions
3. **Search issues** - Check GitHub issues for common problems
4. **Ask questions** - Use GitHub discussions for help
5. **Join the community** - Connect with other users

Next Steps
----------

After completing the tutorials:

* **Build your own pipelines** - Apply what you've learned
* **Explore the API** - Read the :doc:`../api/core` documentation
* **Try examples** - Check out the ``examples/`` directory
* **Contribute** - Help improve the framework
* **Share your work** - Show the community what you've built

.. tip::
   The tutorials are continuously updated. Check back regularly for new content and improved examples.

.. note::
   All tutorial code is tested and maintained. If you encounter issues, please report them on our GitHub repository.