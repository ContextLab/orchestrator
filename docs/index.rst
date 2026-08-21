Orchestrator documentation
==========================

Orchestrator compiles YAML pipelines into task graphs and executes them through the command line or Python API. Start with a local, deterministic pipeline: it requires no model provider, API key, or optional dependency.

Project status
--------------

Orchestrator is alpha software under active recovery. The supported surface is deliberately small. The six pipelines in ``examples/supported/`` are exercised through both public interfaces and compared as complete normalized results. Other examples are historical or experimental unless a page says otherwise.

Start here
----------

.. toctree::
   :maxdepth: 2

   getting_started/installation
   tutorials/getting-started
   tutorials/README
   getting_started/cli_reference

Use Orchestrator
----------------

.. toctree::
   :maxdepth: 2

   user_guide/yaml_configuration
   error_handling
   loop_variables
   features/timeout_configuration
   features/resume_restart

Reference
---------

.. toctree::
   :maxdepth: 2

   actions
   template_globals
   reference/tool_catalog
   reference/api_reference
   adr/0001-product-contract

Contribute
----------

.. toctree::
   :maxdepth: 2

   writing-style
   development/contributing
   development/testing
   development/architecture

Unsupported areas
-----------------

OpenAI, Google, Hugging Face, Ollama, multimodal tools, MCP integration, monitoring, analytics, and deployment code are present but not part of the verified product surface. Dartmouth Chat is the first provider to pass live acceptance tests. See the product contract for the precise boundary.
