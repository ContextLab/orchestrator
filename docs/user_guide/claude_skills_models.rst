========================================
Claude Skills: Models and Configuration
========================================

.. contents:: Table of Contents
   :depth: 2
   :local:

Overview
========

The Claude Skills Orchestrator uses **Anthropic Claude models exclusively**, providing automatic model selection based on task requirements. This guide covers model configuration, selection strategies, and best practices for the Claude Skills system.

Available Claude Models
=======================

Latest Models (2025)
--------------------

The orchestrator supports the latest Claude models with automatic fallback to current available models:

.. list-table::
   :header-rows: 1
   :widths: 25 15 40 10 10

   * - Model
     - Context
     - Role & Use Cases
     - Speed
     - Cost
   * - **Claude Opus 4.1**
     - 200K
     - Deep analysis, critical reviews, complex reasoning, final synthesis
     - Slow
     - $$$
   * - **Claude Sonnet 4.5**
     - 1M
     - Orchestration, code generation, agent building, general tasks
     - Medium
     - $$
   * - **Claude Haiku 4.5**
     - 200K
     - Simple tasks, quick validation, formatting, high-volume ops
     - Fast
     - $

Current Models (Available Now)
-------------------------------

When 2025 models aren't available yet, the system automatically falls back:

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Requested
     - Fallback
     - Notes
   * - ``claude-opus-4.1``
     - ``claude-3-5-sonnet-20241022``
     - Best available quality
   * - ``claude-sonnet-4.5``
     - ``claude-3-5-sonnet-20241022``
     - Current production model
   * - ``claude-haiku-4-5``
     - ``claude-3-haiku-20240307``
     - Fast and cost-effective

Model Configuration
===================

Registry Location
-----------------

Models are configured in ``~/.orchestrator/models/registry.yaml``:

.. code-block:: yaml

   version: "1.0.0"

   providers:
     anthropic:
       enabled: true
       api_key_env: "ANTHROPIC_API_KEY"
       base_url: null
       timeout: 120
       max_retries: 3

   models:
     claude-opus-4.1:
       provider: "anthropic"
       model_id: "claude-opus-4-1-20250805"
       role: "review_and_analysis"
       context_window: 200000
       max_tokens: 8192

     claude-sonnet-4.5:
       provider: "anthropic"
       model_id: "claude-sonnet-4-5"
       role: "orchestrator"
       context_window: 1000000
       max_tokens: 8192

     claude-haiku-4.5:
       provider: "anthropic"
       model_id: "claude-haiku-4-5"
       role: "simple_tasks"
       context_window: 200000
       max_tokens: 8192

This file is created automatically on first run.

API Key Setup
-------------

Store your API key in ``~/.orchestrator/.env``:

.. code-block:: bash

   # Required for all Claude models
   ANTHROPIC_API_KEY=sk-ant-api03-...

The framework will:
* Automatically load the key
* Validate it on first use
* Securely store it (never committed to git)

Model Selection in Pipelines
=============================

Explicit Model Selection
------------------------

Specify the exact model to use:

.. code-block:: yaml

   steps:
     - id: analyze_code
       action: llm_generate
       parameters:
         prompt: "Analyze this code"
         model: claude-3-5-sonnet-20241022  # Explicit model
         max_tokens: 2000

By Task Role
------------

Use models based on their role:

.. code-block:: yaml

   steps:
     # Simple validation - use Haiku
     - id: validate_input
       action: llm_generate
       parameters:
         prompt: "Validate this input"
         model: claude-haiku-4-5
         max_tokens: 100

     # Code generation - use Sonnet
     - id: generate_code
       action: llm_generate
       parameters:
         prompt: "Generate Python code"
         model: claude-sonnet-4-5
         max_tokens: 2000

     # Critical review - use Opus
     - id: security_review
       action: llm_generate
       parameters:
         prompt: "Security audit"
         model: claude-opus-4-1-20250805
         max_tokens: 3000

Conditional Selection
---------------------

Use Jinja2 templates for dynamic selection:

.. code-block:: yaml

   steps:
     - id: process_data
       action: llm_generate
       parameters:
         model: >
           {% if priority == 'critical' %}
           claude-opus-4.1
           {% elif task_complexity == 'simple' %}
           claude-haiku-4-5
           {% else %}
           claude-3-5-sonnet-20241022
           {% endif %}
         prompt: "Process this data"

Model Selection Strategy
========================

When to Use Each Model
----------------------

Claude Haiku 4.5
~~~~~~~~~~~~~~~~

**Best for:**

* Input validation
* Format conversions
* Simple queries
* Quick checks
* High-volume operations

**Example:**

.. code-block:: yaml

   - id: validate_json
     action: llm_generate
     parameters:
       prompt: "Is this valid JSON? {data}"
       model: claude-haiku-4-5
       max_tokens: 50

Claude Sonnet 4.5
~~~~~~~~~~~~~~~~~

**Best for:**

* Code generation
* Data analysis
* Workflow orchestration
* Complex transformations
* Multi-step reasoning

**Example:**

.. code-block:: yaml

   - id: generate_function
     action: llm_generate
     parameters:
       prompt: |
         Write a Python function that:
         - Parses CSV files
         - Validates data
         - Returns structured JSON
       model: claude-sonnet-4-5
       max_tokens: 2000

Claude Opus 4.1
~~~~~~~~~~~~~~~

**Best for:**

* Comprehensive reviews
* Critical decisions
* Complex analysis
* Final synthesis
* Security audits

**Example:**

.. code-block:: yaml

   - id: security_audit
     action: llm_generate
     parameters:
       prompt: |
         Perform comprehensive security audit:
         {{ codebase_analysis.result }}
       model: claude-opus-4-1-20250805
       max_tokens: 5000

Cost Optimization
=================

Use the Right Model for Each Task
----------------------------------

**Pattern: Haiku → Sonnet → Opus**

.. code-block:: yaml

   steps:
     # Stage 1: Quick filtering with Haiku
     - id: filter_items
       action: llm_generate
       parameters:
         model: claude-haiku-4-5
         prompt: "Filter these items: {{ items }}"

     # Stage 2: Process with Sonnet
     - id: process_filtered
       dependencies: [filter_items]
       action: llm_generate
       parameters:
         model: claude-3-5-sonnet-20241022
         prompt: "Process: {{ filter_items.result }}"

     # Stage 3: Final review with Opus
     - id: final_review
       dependencies: [process_filtered]
       action: llm_generate
       parameters:
         model: claude-opus-4-1-20250805
         prompt: "Final review: {{ process_filtered.result }}"

Pricing Guide (Per 1M Tokens)
------------------------------

.. code-block:: text

   Claude Haiku 4.5:    $1 input  / $5 output   (cheapest)
   Claude Sonnet 4.5:   $3 input  / $15 output  (balanced)
   Claude Opus 4.1:     $15 input / $75 output  (premium)

**Example Cost Calculation:**

For a pipeline that:
- Processes 100 items with Haiku (10K tokens each)
- Aggregates with Sonnet (50K tokens)
- Final review with Opus (20K tokens)

.. code-block:: text

   Haiku:  (100 × 10K × $1/1M) = $1.00 input
   Sonnet: (50K × $3/1M) = $0.15 input
   Opus:   (20K × $15/1M) = $0.30 input
   Total:  ~$1.45 input + output costs

Python API
==========

Model Registry
--------------

.. code-block:: python

   from orchestrator.models import ModelRegistry
   from orchestrator.models.providers import ProviderConfig

   # Create registry
   registry = ModelRegistry()

   # Configure Anthropic provider
   registry.configure_provider(
       provider_name="anthropic",
       provider_type="anthropic",
       config={"api_key": "your-key"}
   )

   # Initialize
   await registry.initialize()

   # List available models
   models = registry.available_models
   print(f"Available: {list(models.keys())}")

   # Health check
   health = await registry.health_check()
   print(f"Health: {health}")

Model Selection in Code
-----------------------

.. code-block:: python

   # Get a specific model
   model = await registry.get_model("claude-3-5-sonnet-20241022")

   # Use the model
   response = await model.generate(
       prompt="Hello, Claude!",
       max_tokens=100
   )

Best Practices
==============

1. **Match Model to Task Complexity**

   - Simple tasks → Haiku
   - Standard tasks → Sonnet
   - Critical tasks → Opus

2. **Use Sonnet as Default**

   - Best balance of quality and cost
   - 1M token context for complex workflows
   - Excellent for most use cases

3. **Reserve Opus for Quality-Critical Steps**

   - Final synthesis
   - Security reviews
   - Critical decision points

4. **Optimize with Haiku for Volume**

   - High-frequency operations
   - Simple validations
   - Quick formatting

5. **Test with Current Models**

   - Use ``claude-3-5-sonnet-20241022`` for testing
   - Fallback behavior is automatic
   - Update to 2025 models when available

Examples from Documentation
============================

See :doc:`../tutorials/claude_skills_quickstart` for complete examples including:

* Code review workflows
* Research synthesis
* Parallel data processing
* Conditional model routing

Troubleshooting
===============

Model Not Found
---------------

If you see:

.. code-block:: text

   Error: Model 'claude-opus-4.1' not found

The 2025 model isn't available yet. The system automatically falls back to:

.. code-block:: text

   Using fallback: claude-3-5-sonnet-20241022

API Key Issues
--------------

If you see:

.. code-block:: text

   Error: ANTHROPIC_API_KEY not set

Fix:

.. code-block:: bash

   echo "ANTHROPIC_API_KEY=your-key" >> ~/.orchestrator/.env

Then restart your Python session.

Rate Limiting
-------------

If you hit rate limits:

.. code-block:: text

   Error: Rate limit exceeded

Solutions:
* Add delays between requests
* Use Haiku for high-volume tasks
* Implement batching in your pipeline

Next Steps
==========

* :doc:`../CLAUDE_SKILLS_USER_GUIDE` - Complete user guide
* :doc:`../QUICK_START` - 5-minute tutorial
* :doc:`claude_skills_quickstart` - Detailed quickstart
* `Examples <../../examples/claude_skills_refactor/>`_ - Working pipelines

---

**Version**: 1.0.0 (Claude Skills)

**Last Updated**: January 2025