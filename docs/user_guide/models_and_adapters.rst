Models and Adapters
===================

Learn how to integrate AI models and external services with the Orchestrator Framework.

.. note::
   This guide covers model integration. For hands-on practice, see the :doc:`../tutorials/notebooks` tutorial 03.

Supported Models
----------------

The Orchestrator Framework supports various AI models:

OpenAI Models
~~~~~~~~~~~~~

.. code-block:: python

   from orchestrator.models.openai_model import OpenAIModel
   
   model = OpenAIModel(
       name="gpt-4",
       api_key="your-api-key",
       model="gpt-4"
   )

Anthropic Models
~~~~~~~~~~~~~~~~

.. code-block:: python

   from orchestrator.models.anthropic_model import AnthropicModel
   
   model = AnthropicModel(
       name="claude-3-sonnet",
       api_key="your-api-key",
       model="claude-3-sonnet-20240229"
   )

Local Models
~~~~~~~~~~~~

.. code-block:: python

   from orchestrator.models.huggingface_model import HuggingFaceModel
   
   model = HuggingFaceModel(
       name="llama-7b",
       model_path="meta-llama/Llama-2-7b-chat-hf"
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