============
Installation
============

This guide covers the installation of py-orc and its dependencies.

Requirements
============

System Requirements
-------------------

- **Operating System**: Linux, macOS, or Windows (with WSL2)
- **Python**: 3.11 or higher
- **Memory**: 8GB RAM minimum (16GB recommended for larger models)
- **Storage**: 10GB free space for models and dependencies

Optional Requirements
---------------------

- **Docker**: For sandboxed execution environments
- **CUDA**: For GPU acceleration (if using GPU models)
- **Ollama**: For local model execution
- **PostgreSQL**: For production state management

Installation Methods
====================

Using pip (Recommended)
-----------------------

.. code-block:: bash

   # Install from PyPI (when available)
   pip install py-orc
   
   # Or install from source
   git clone https://github.com/ContextLab/orchestrator.git
   cd orchestrator
   pip install -e .

Using conda
-----------

.. code-block:: bash

   # Create conda environment
   conda create -n py-orc python=3.11
   conda activate py-orc
   
   # Install orchestrator
   pip install py-orc

Using Docker
------------

.. code-block:: bash

   # Pull the official image
   docker pull contextlab/py-orc:latest
   
   # Run with volume mount
   docker run -v $(pwd):/workspace contextlab/py-orc

Development Installation
------------------------

For contributors and developers:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/ContextLab/orchestrator.git
   cd orchestrator
   
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install in development mode with extras
   pip install -e ".[dev,test,docs]"
   
   # Install pre-commit hooks
   pre-commit install

Model Setup
===========

Ollama Models (Recommended)
---------------------------

1. **Install Ollama**:

   .. code-block:: bash

      # macOS
      brew install ollama
      
      # Linux
      curl -fsSL https://ollama.ai/install.sh | sh

2. **Pull recommended models**:

   .. code-block:: bash

      # Large model for complex tasks
      ollama pull gemma2:27b
      
      # Small model for simple tasks
      ollama pull llama3.2:1b
      
      # Code-focused model
      ollama pull codellama:7b

3. **Verify installation**:

   .. code-block:: python

      import orchestrator as orc
      
      # Initialize and check models
      registry = orc.init_models()
      print(registry.list_models())

HuggingFace Models
------------------

For HuggingFace models, set up your token:

.. code-block:: bash

   # Set environment variable
   export HUGGINGFACE_TOKEN="your-token-here"
   
   # Or create .env file
   echo "HUGGINGFACE_TOKEN=your-token-here" > .env

OpenAI/Anthropic Models
-----------------------

For cloud models, configure API keys:

.. code-block:: bash

   # OpenAI
   export OPENAI_API_KEY="sk-..."
   
   # Anthropic
   export ANTHROPIC_API_KEY="sk-ant-..."

Tool Dependencies
=================

Web Tools
---------

For headless browser functionality:

.. code-block:: bash

   # Install Playwright
   pip install playwright
   playwright install chromium
   
   # Or use Selenium
   pip install selenium
   # Download appropriate driver

System Tools
------------

No additional setup required for basic system tools.

Data Tools
----------

Install optional data processing libraries:

.. code-block:: bash

   # For advanced data processing
   pip install pandas numpy scipy
   
   # For data validation
   pip install pydantic jsonschema

Configuration
=============

Create a configuration file at ``~/.orchestrator/config.yaml``:

.. code-block:: yaml

   # Model preferences
   models:
     default: "ollama:gemma2:27b"
     fallback: "ollama:llama3.2:1b"
   
   # Resource limits
   resources:
     max_memory: "16GB"
     max_threads: 8
     gpu_enabled: true
   
   # Tool settings
   tools:
     mcp_port: 8000
     sandbox_enabled: true
   
   # State management
   state:
     backend: "postgresql"
     connection: "postgresql://user:pass@localhost/orchestrator"

Environment Variables
---------------------

Set these environment variables for additional configuration:

.. code-block:: bash

   # Core settings
   export ORCHESTRATOR_HOME="$HOME/.orchestrator"
   export ORCHESTRATOR_LOG_LEVEL="INFO"
   
   # Model settings
   export ORCHESTRATOR_MODEL_TIMEOUT="300"
   export ORCHESTRATOR_MODEL_RETRIES="3"
   
   # Tool settings
   export ORCHESTRATOR_TOOL_TIMEOUT="60"
   export ORCHESTRATOR_MCP_AUTO_START="true"

Verifying Installation
======================

Run the verification script:

.. code-block:: python

   import orchestrator as orc
   
   # Check version
   print(f"Orchestrator version: {orc.__version__}")
   
   # Check models
   try:
       registry = orc.init_models()
       models = registry.list_models()
       print(f"Available models: {models}")
   except Exception as e:
       print(f"Model initialization failed: {e}")
   
   # Check tools
   from orchestrator.tools.base import default_registry
   tools = default_registry.list_tools()
   print(f"Available tools: {tools}")
   
   # Run test pipeline
   try:
       pipeline = orc.compile("examples/hello-world.yaml")
       result = pipeline.run(message="Hello, Orchestrator!")
       print(f"Test pipeline result: {result}")
   except Exception as e:
       print(f"Pipeline test failed: {e}")

Troubleshooting
===============

Common Issues
-------------

**Import Error**:

.. code-block:: text

   ModuleNotFoundError: No module named 'orchestrator'

Solution: Ensure you're in the correct environment and have installed the package.

**Model Connection Error**:

.. code-block:: text

   Failed to connect to Ollama at http://localhost:11434

Solution: Start Ollama service with ``ollama serve``.

**Permission Error**:

.. code-block:: text

   Permission denied: '/home/user/.orchestrator'

Solution: Create directory with proper permissions:

.. code-block:: bash

   mkdir -p ~/.orchestrator
   chmod 755 ~/.orchestrator

Getting Help
------------

If you encounter issues:

1. Check the :doc:`troubleshooting` guide
2. Search existing `GitHub issues <https://github.com/contextualdynamics/orchestrator/issues>`_
3. Join our `Discord community <https://discord.gg/orchestrator>`_
4. Create a new issue with detailed information

Next Steps
==========

After installation:

- Continue to :doc:`quickstart` to build your first pipeline
- Explore :doc:`tutorials/index` for hands-on learning
- Review :doc:`examples/index` for real-world use cases