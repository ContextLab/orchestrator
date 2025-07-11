Installation Guide
=================

This guide will help you install the Orchestrator Framework and get it running on your system.

Requirements
------------

Before installing Orchestrator, ensure you have the following requirements:

* **Python 3.8+** - The framework requires Python 3.8 or higher
* **pip** - Python package installer
* **Optional dependencies** for specific features:
  
  * **Docker** - For sandboxed execution
  * **Redis** - For distributed caching
  * **PostgreSQL** - For persistent state storage

Basic Installation
------------------

Install Orchestrator using pip:

.. code-block:: bash

   pip install orchestrator-framework

This installs the core framework with all essential dependencies.

Development Installation
------------------------

To install from source for development:

.. code-block:: bash

   git clone https://github.com/orchestrator-framework/orchestrator.git
   cd orchestrator
   pip install -e .

This installs the package in development mode, allowing you to modify the source code.

Optional Dependencies
--------------------

Install additional dependencies for specific features:

.. tabs::

   .. tab:: Docker Support

      For sandboxed execution with Docker:

      .. code-block:: bash

         pip install orchestrator-framework[docker]

   .. tab:: Database Support

      For persistent state storage:

      .. code-block:: bash

         pip install orchestrator-framework[database]

   .. tab:: All Features

      For all optional dependencies:

      .. code-block:: bash

         pip install orchestrator-framework[all]

Verifying Installation
---------------------

Verify your installation by running:

.. code-block:: python

   import orchestrator
   print(f"Orchestrator version: {orchestrator.__version__}")

   # Test basic functionality
   from orchestrator import Task, Pipeline
   
   task = Task(id="test", name="Test Task", action="echo", parameters={"message": "Hello!"})
   pipeline = Pipeline(id="test_pipeline", name="Test Pipeline")
   pipeline.add_task(task)
   
   print("✅ Installation successful!")

Configuration
-------------

After installation, you may want to configure Orchestrator for your environment.

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

Set these environment variables for optimal performance:

.. code-block:: bash

   # Optional: Set cache directory
   export ORCHESTRATOR_CACHE_DIR=/path/to/cache
   
   # Optional: Set checkpoint directory
   export ORCHESTRATOR_CHECKPOINT_DIR=/path/to/checkpoints
   
   # Optional: Set log level
   export ORCHESTRATOR_LOG_LEVEL=INFO

API Keys
~~~~~~~~

Configure API keys for external services:

.. code-block:: bash

   # OpenAI
   export OPENAI_API_KEY=your_openai_key
   
   # Anthropic
   export ANTHROPIC_API_KEY=your_anthropic_key
   
   # Google
   export GOOGLE_API_KEY=your_google_key

Docker Setup
~~~~~~~~~~~~

If using Docker features, ensure Docker is running:

.. code-block:: bash

   docker --version
   docker run hello-world

Troubleshooting
---------------

Common Installation Issues
~~~~~~~~~~~~~~~~~~~~~~~~~

**Permission Errors**
   Use ``--user`` flag: ``pip install --user orchestrator-framework``

**Python Version Issues**
   Ensure Python 3.8+: ``python --version``

**Missing Dependencies**
   Install system dependencies:
   
   .. code-block:: bash
   
      # Ubuntu/Debian
      sudo apt-get update
      sudo apt-get install python3-dev build-essential
      
      # macOS
      brew install python
      
      # Windows
      # Use Python from python.org

**Docker Issues**
   Ensure Docker is installed and running:
   
   .. code-block:: bash
   
      docker --version
      docker info

Getting Help
~~~~~~~~~~~~

If you encounter issues:

1. Check the :doc:`../advanced/troubleshooting` guide
2. Search existing `GitHub issues <https://github.com/orchestrator-framework/orchestrator/issues>`_
3. Create a new issue with your error details

Next Steps
----------

Once installed, proceed to:

* :doc:`quickstart` - Build your first pipeline
* :doc:`basic_concepts` - Learn core concepts
* :doc:`../tutorials/notebooks` - Interactive tutorials