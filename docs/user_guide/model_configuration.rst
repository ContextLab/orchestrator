Model Configuration and Selection
=================================

The Orchestrator Framework provides a flexible system for configuring and selecting AI models based on task requirements. This guide covers model configuration, automatic installation, and expertise-based selection.

Model Configuration File
------------------------

Models are configured in a ``models.yaml`` file that defines available models, their capabilities, and selection preferences.

Configuration Structure
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   models:
     # Ollama models (automatically installed if not present)
     - source: ollama
       name: gemma2:27b
       expertise: 
         - general
         - reasoning
         - analysis
       size: 27b
       
     - source: ollama
       name: codellama:7b
       expertise:
         - code
         - programming
       size: 7b
       
     # HuggingFace models (automatically downloaded)
     - source: huggingface
       name: microsoft/phi-2
       expertise:
         - reasoning
         - code
       size: 2.7b
       
     # Cloud models (require API keys)
     - source: openai
       name: gpt-4o
       expertise:
         - general
         - reasoning
         - code
         - analysis
         - vision
       size: 1760b
   
   defaults:
     expertise_preferences:
       code: codellama:7b
       reasoning: gemma2:27b
       fast: llama3.2:1b
     fallback_chain:
       - gemma2:27b
       - llama3.2:1b
       - TinyLlama/TinyLlama-1.1B-Chat-v1.0

Model Sources
~~~~~~~~~~~~~

The framework supports models from multiple sources:

- **ollama**: Local models via Ollama (automatically installed if not present)
- **huggingface**: Models from HuggingFace Hub (automatically downloaded)
- **openai**: OpenAI API models (requires ``OPENAI_API_KEY``)
- **anthropic**: Anthropic API models (requires ``ANTHROPIC_API_KEY``)
- **google**: Google AI models (requires ``GOOGLE_API_KEY``)

Model Properties
~~~~~~~~~~~~~~~~

Each model configuration includes:

- **source**: Where the model comes from
- **name**: Model identifier (e.g., "gemma2:27b", "gpt-4o")
- **expertise**: List of areas the model excels at
- **size**: Model size (automatically parsed from name if not specified)

Size Notation
^^^^^^^^^^^^^

Model sizes can be specified using standard notation:

- ``82m`` or ``0.082b``: 82 million parameters
- ``7b``: 7 billion parameters
- ``1.5t``: 1.5 trillion parameters

The framework automatically parses common size patterns from model names (e.g., "llama-7b" → 7B parameters).

Automatic Model Installation
----------------------------

Lazy Loading (On-Demand Downloads)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The framework uses lazy loading for both Ollama and HuggingFace models to avoid downloading large models until they're actually needed:

.. code-block:: python

   import orchestrator as orc
   
   # This registers models but doesn't download them yet
   registry = orc.init_models()
   
   # Models are downloaded only when first used by a pipeline
   pipeline = orc.compile("my_pipeline.yaml")
   result = pipeline.run()  # Model downloads happen here if needed

Ollama Models
~~~~~~~~~~~~~

Ollama models are downloaded on first use:

1. When a task requires an Ollama model, the framework checks if it's available locally
2. If not available, it automatically runs ``ollama pull <model>``
3. The download happens only once - subsequent uses will use the cached model
4. Progress is shown during download: ``📥 Downloading Ollama model: llama3.1:8b``

HuggingFace Models
~~~~~~~~~~~~~~~~~~

HuggingFace models are also downloaded on first use:

.. code-block:: yaml

   - source: huggingface
     name: microsoft/Phi-3.5-mini-instruct
     expertise: [reasoning, code]

The model will be downloaded from HuggingFace Hub and cached locally when first requested by a task. The transformers library handles caching automatically.

.. note::
   
   Model downloads can be large (several GB). Ensure you have:
   
   - Sufficient disk space for model storage
   - A stable internet connection for downloads
   - Time for the initial download (subsequent uses are instant)

Specifying Model Requirements in Pipelines
------------------------------------------

Tasks can specify model requirements using the ``requires_model`` field:

Simple Model Selection
~~~~~~~~~~~~~~~~~~~~~~

Specify a model by name:

.. code-block:: yaml

   steps:
     - id: summarize
       action: generate_text
       parameters:
         prompt: "Summarize this text..."
       requires_model: gemma2:27b  # Use specific model

Expertise-Based Selection
~~~~~~~~~~~~~~~~~~~~~~~~~

Specify requirements and let the framework choose:

.. code-block:: yaml

   steps:
     - id: generate_code
       action: generate_text
       parameters:
         prompt: "Write a Python function..."
       requires_model:
         expertise: code
         min_size: 7b  # At least 7B parameters

Multiple Expertise Areas
~~~~~~~~~~~~~~~~~~~~~~~~

Specify multiple expertise areas (any match will qualify):

.. code-block:: yaml

   steps:
     - id: analyze
       action: analyze
       parameters:
         content: "{input_data}"
       requires_model:
         expertise: 
           - reasoning
           - analysis
         min_size: 20b

Complete Example
----------------

Here's a complete pipeline demonstrating model requirements:

.. code-block:: yaml

   id: multi_model_pipeline
   name: Multi-Model Processing Pipeline
   
   inputs:
     - name: topic
       type: string
   
   steps:
     # Fast task with small model
     - id: quick_check
       action: generate_text
       parameters:
         prompt: "Is this topic related to programming: {topic}?"
       requires_model:
         expertise: fast
         min_size: 0  # Any size
   
     # Code generation with specialized model
     - id: code_example
       action: generate_text
       parameters:
         prompt: "Generate example code for: {topic}"
       requires_model:
         expertise: code
         min_size: 7b
       dependencies: [quick_check]
   
     # Complex reasoning with large model
     - id: deep_analysis
       action: analyze
       parameters:
         content: "{topic} with code: {code_example.result}"
       requires_model:
         expertise: [reasoning, analysis]
         min_size: 27b
       dependencies: [code_example]

Model Selection Algorithm
-------------------------

The framework uses a sophisticated selection algorithm:

1. **Filter by Requirements**:
   
   - Check expertise match
   - Verify minimum size
   - Confirm model capabilities

2. **Health Check**:
   
   - Verify model availability
   - Check API connectivity
   - Monitor error rates

3. **Intelligent Selection**:
   
   - Use Upper Confidence Bound (UCB) algorithm
   - Balance exploration vs exploitation
   - Consider past performance

Default Model Selection
~~~~~~~~~~~~~~~~~~~~~~~

When no specific requirements are given, the framework uses defaults based on task action:

- ``generate_text``, ``generate``: Uses models with "general" expertise
- ``analyze``: Prefers models with "reasoning" and "analysis" expertise
- ``transform``: Uses models with "general" expertise

Monitoring Model Usage
----------------------

Check which models are being used:

.. code-block:: python

   import orchestrator as orc
   
   # Initialize and list available models
   registry = orc.init_models()
   print("Available models:")
   for model_key in registry.list_models():
       print(f"  - {model_key}")
   
   # Run pipeline and check model selection
   pipeline = orc.compile("pipeline.yaml")
   result = pipeline.run(topic="AI agents")

The framework logs model selection decisions:

.. code-block:: text

   >> Using model for task 'quick_check': ollama:llama3.2:1b (fast, 1B params)
   >> Using model for task 'code_example': ollama:codellama:7b (code, 7B params)
   >> Using model for task 'deep_analysis': ollama:gemma2:27b (reasoning, 27B params)

Best Practices
--------------

1. **Specify Expertise for Specialized Tasks**:
   
   - Use ``expertise: code`` for programming tasks
   - Use ``expertise: reasoning`` for complex analysis
   - Use ``expertise: fast`` for simple, quick tasks

2. **Set Appropriate Size Requirements**:
   
   - Small models (< 3B) for simple text generation
   - Medium models (7-13B) for code and moderate complexity
   - Large models (20B+) for complex reasoning and analysis

3. **Provide Fallbacks**:
   
   - Configure multiple models with similar expertise
   - Set up a fallback chain in the configuration
   - Handle model unavailability gracefully

4. **Consider Cost and Performance**:
   
   - Local models (Ollama) for cost-effective processing
   - Cloud models for maximum capability when needed
   - Balance model size with response time requirements

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Model Installation Fails**:

.. code-block:: text

   >> ❌ Failed to install gemma2:27b: connection timeout

Solutions:
- Check internet connectivity
- Verify Ollama is running: ``ollama serve``
- Try manual installation: ``ollama pull gemma2:27b``

**No Models Match Requirements**:

.. code-block:: text

   NoEligibleModelsError: No models meet the specified requirements

Solutions:
- Lower size requirements
- Broaden expertise requirements
- Ensure required models are configured

**API Key Missing**:

.. code-block:: text

   >> ⚠️  OpenAI models configured but OPENAI_API_KEY not set

Solutions:
- Set environment variable: ``export OPENAI_API_KEY="sk-..."``
- Or use local models instead

Configuration Locations
~~~~~~~~~~~~~~~~~~~~~~~

The framework searches for ``models.yaml`` in:

1. Current directory
2. ``~/.orchestrator/models.yaml``
3. Project root directory
4. ``$ORCHESTRATOR_HOME/models.yaml``

Create the file in any of these locations to customize model configuration.