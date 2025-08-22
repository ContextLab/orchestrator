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

   pip install py-orc

This installs the core framework with all essential dependencies.

Development Installation
------------------------

To install from source for development:

.. code-block:: bash

   git clone https://github.com/ContextLab/orchestrator.git
   cd orchestrator
   pip install -e .

This installs the package in development mode, allowing you to modify the source code.

Optional Dependencies
--------------------

Install additional dependencies for specific features:

.. tabs::

   .. tab:: Ollama Support

      For local model execution with Ollama:

      .. code-block:: bash

         pip install py-orc[ollama]

   .. tab:: Cloud Models

      For cloud model providers:

      .. code-block:: bash

         pip install py-orc[cloud]

   .. tab:: Development

      For development tools and testing:

      .. code-block:: bash

         pip install py-orc[dev]

   .. tab:: All Features

      For all optional dependencies:

      .. code-block:: bash

         pip install py-orc[all]

Verifying Installation
---------------------

Verify your installation by running:

.. code-block:: python

   import orchestrator as orc
   print(f"Orchestrator version: {orc.__version__}")

   # Test basic functionality
   try:
       # Initialize models (will detect available models)
       orc.init_models()
       print("✅ Model initialization successful!")
       
       # Test pipeline compilation
       from orchestrator.core.pipeline import Pipeline
       pipeline = Pipeline(id="test", name="Test")
       print("✅ Pipeline creation successful!")
       
       print("✅ Installation verified!")
   except Exception as e:
       print(f"❌ Installation issue: {e}")

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

Use the interactive setup script to configure API keys:

.. code-block:: bash

   python scripts/setup_api_keys.py

Or configure API keys manually:

.. code-block:: bash

   # OpenAI
   export OPENAI_API_KEY=your_openai_key
   
   # Anthropic
   export ANTHROPIC_API_KEY=your_anthropic_key
   
   # Google AI
   export GOOGLE_AI_API_KEY=your_google_ai_key
   
   # Hugging Face (optional)
   export HF_TOKEN=your_huggingface_token

API keys are stored securely in ``~/.orchestrator/.env`` with restricted file permissions.

Ollama Setup (Optional)
~~~~~~~~~~~~~~~~~~~~~~~

For local model execution, install Ollama:

.. code-block:: bash

   # macOS
   brew install ollama
   
   # Linux
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Start Ollama service
   ollama serve

Models are downloaded automatically on first use.

Troubleshooting
---------------

Common Installation Issues
~~~~~~~~~~~~~~~~~~~~~~~~~

**Permission Errors**
   Use ``--user`` flag: ``pip install --user py-orc``

**Python Version Issues**
   Ensure Python 3.8+: ``python --version``

**Missing Dependencies**
   Install system dependencies:
   
   .. code-block:: bash
   
      # Ubuntu/Debian
      sudo apt-get update
      sudo apt-get install python3-dev build-essential curl
      
      # macOS
      brew install python
      
      # Windows
      # Use Python from python.org and ensure curl is available

**Model Download Issues**
   For Ollama models, ensure sufficient disk space and network connectivity:
   
   .. code-block:: bash
   
      ollama list  # Check installed models
      ollama pull llama3.2:1b  # Manually pull a model

Getting Help
~~~~~~~~~~~~

If you encounter issues:

1. Check the :doc:`../advanced/troubleshooting` guide
2. Search existing `GitHub issues <https://github.com/ContextLab/orchestrator/issues>`_
3. Create a new issue with your error details

Next Steps
----------

Once installed, proceed to:

* :doc:`quickstart` - Build your first pipeline
* :doc:`basic_concepts` - Learn core concepts
* :doc:`../tutorials/notebooks` - Interactive tutorials