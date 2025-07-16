Models and Adapters
===================

Learn how to integrate AI models and external services with the Orchestrator Framework.

.. note::
   This guide covers model integration. For hands-on practice, see the :doc:`../tutorials/notebooks` tutorial 03.

Model Detection and Availability
---------------------------------

The Orchestrator Framework automatically detects and registers available models when you call ``orc.init_models()``. The system checks for models in the following order:

1. **Ollama Models** (preferred for local execution)
   
   - ``gemma2:27b`` - Large model for complex tasks
   - ``llama3.2:1b`` - Lightweight fallback model
   
2. **HuggingFace Models** (if transformers library is available)
   
   - ``TinyLlama/TinyLlama-1.1B-Chat-v1.0`` - Default lightweight model for testing

3. **Cloud Models** (if API keys are configured)
   
   - OpenAI models (via ``OPENAI_API_KEY``)
   - Anthropic models (via ``ANTHROPIC_API_KEY``)
   - Google models (via ``GOOGLE_API_KEY``)

Initializing Models
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import orchestrator as orc
   
   # Initialize and discover available models
   registry = orc.init_models()
   
   # List all detected models
   available_models = registry.list_models()
   print("Available models:", available_models)
   
   # Check specific model availability
   if any("gemma2:27b" in model for model in available_models):
       print("Large Ollama model available")
   elif any("llama3.2:1b" in model for model in available_models):
       print("Lightweight Ollama model available")
   else:
       print("Using fallback models")

Model Installation
~~~~~~~~~~~~~~~~~~

**Ollama Models (Recommended)**

Install Ollama and pull recommended models:

.. code-block:: bash

   # Install Ollama
   brew install ollama  # macOS
   # or visit https://ollama.ai for other platforms
   
   # Pull recommended models
   ollama pull gemma2:27b    # Large model for complex tasks
   ollama pull llama3.2:1b   # Lightweight fallback

**HuggingFace Models**

Install the transformers library:

.. code-block:: bash

   pip install transformers torch

**Cloud Models**

Set up API keys as environment variables:

.. code-block:: bash

   export OPENAI_API_KEY="sk-..."
   export ANTHROPIC_API_KEY="sk-ant-..."
   export GOOGLE_API_KEY="..."

Supported Models
----------------

The Orchestrator Framework supports various AI models:

OpenAI Models
~~~~~~~~~~~~~

.. code-block:: python

   from orchestrator.models.openai_model import OpenAIModel
   
   model = OpenAIModel(
       name="gpt-4o",
       api_key="your-api-key",
       model="gpt-4o"
   )

Anthropic Models
~~~~~~~~~~~~~~~~

.. code-block:: python

   from orchestrator.models.anthropic_model import AnthropicModel
   
   model = AnthropicModel(
       name="claude-3.5-sonnet",
       api_key="your-api-key",
       model="claude-3.5-sonnet"
   )

Local Models
~~~~~~~~~~~~

.. code-block:: python

   from orchestrator.models.huggingface_model import HuggingFaceModel
   
   model = HuggingFaceModel(
       name="llama-3.2-3b",
       model_path="meta-llama/Llama-3.2-3B-Instruct"
   )

Model Registry
--------------

The model registry manages model selection and load balancing:

.. code-block:: python

   from orchestrator.models.model_registry import ModelRegistry
   
   registry = ModelRegistry()
   registry.register_model(gpt4_model)
   registry.register_model(claude_model)
   
   # Automatic selection based on task requirements
   selected_model = registry.select_model(task)

For complete documentation, see the :doc:`../api/models` reference.