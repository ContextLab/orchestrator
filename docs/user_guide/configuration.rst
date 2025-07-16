Configuration Guide
===================

The Orchestrator framework uses YAML configuration files to customize behavior, define models, and set system parameters.

Configuration Files
-------------------

Orchestrator uses two main configuration files:

1. **models.yaml** - Defines available AI models and their properties
2. **orchestrator.yaml** - Sets framework behavior and system parameters

Configuration Locations
-----------------------

The framework searches for configuration files in the following order:

1. Current working directory
2. ``~/.orchestrator/`` (user configuration directory)
3. Package installation directory
4. ``$ORCHESTRATOR_HOME`` environment variable location

The first file found in any of these locations will be used.

Installation and Setup
----------------------

When you install Orchestrator via pip, default configuration files are available but not automatically installed to avoid overwriting existing configurations. To install the default configurations:

.. code-block:: bash

   # Install default configs to ~/.orchestrator/
   orchestrator-install-configs

This command will:

- Create ``~/.orchestrator/`` directory if it doesn't exist
- Copy default ``models.yaml`` and ``orchestrator.yaml`` files
- Skip files that already exist (won't overwrite your customizations)
- Create a README with configuration instructions

Customizing Configuration
-------------------------

models.yaml
~~~~~~~~~~~

The models configuration file defines available AI models:

.. code-block:: yaml

   models:
     # Local models (via Ollama)
     - source: ollama
       name: llama3.1:8b
       expertise: [general, reasoning, multilingual]
       size: 8b
       
     # Cloud models
     - source: openai
       name: gpt-4o
       expertise: [general, reasoning, code, analysis, vision]
       size: 1760b  # Estimated
       
     # HuggingFace models
     - source: huggingface
       name: microsoft/Phi-3.5-mini-instruct
       expertise: [reasoning, code, compact]
       size: 3.8b
   
   defaults:
     expertise_preferences:
       code: qwen2.5-coder:7b
       reasoning: deepseek-r1:8b
       fast: llama3.2:1b
     fallback_chain:
       - llama3.1:8b
       - mistral:7b
       - llama3.2:1b

You can add new models by editing this file:

.. code-block:: yaml

   # Add a new Ollama model
   - source: ollama
     name: my-custom-model:13b
     expertise: [domain-specific, analysis]
     size: 13b

orchestrator.yaml
~~~~~~~~~~~~~~~~~

The main configuration file controls framework behavior:

.. code-block:: yaml

   # Execution settings
   execution:
     parallel_tasks: 10
     timeout_seconds: 300
     retry_attempts: 3
     retry_delay: 1.0
   
   # Resource limits
   resources:
     max_memory_mb: 8192
     max_cpu_percent: 80
     gpu_enabled: true
   
   # Caching
   cache:
     enabled: true
     ttl_seconds: 3600
     max_size_mb: 1024
   
   # Monitoring
   monitoring:
     log_level: INFO
     metrics_enabled: true
     trace_enabled: false
   
   # Error handling
   error_handling:
     circuit_breaker_threshold: 5
     circuit_breaker_timeout: 60
     fallback_enabled: true

Environment Variables
---------------------

You can override configuration settings using environment variables:

.. code-block:: bash

   # Set custom config location
   export ORCHESTRATOR_HOME=/path/to/configs
   
   # Override specific settings
   export ORCHESTRATOR_LOG_LEVEL=DEBUG
   export ORCHESTRATOR_PARALLEL_TASKS=20
   export ORCHESTRATOR_CACHE_ENABLED=false

Best Practices
--------------

1. **Version Control**: Keep your custom configurations in version control
2. **Separate Environments**: Use different config directories for dev/staging/prod
3. **Model Management**: Regularly update model definitions as new versions are released
4. **Resource Limits**: Set appropriate limits based on your hardware
5. **Monitoring**: Enable metrics and tracing in production for better observability

Configuration Validation
------------------------

Orchestrator validates configuration files on startup:

.. code-block:: python

   import orchestrator as orc
   
   # Validate configuration files
   config_valid, errors = orc.validate_config()
   if not config_valid:
       print("Configuration errors:", errors)

Common Configuration Scenarios
------------------------------

Development Environment
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # orchestrator.yaml for development
   execution:
     parallel_tasks: 2
     timeout_seconds: 60
   
   monitoring:
     log_level: DEBUG
     trace_enabled: true
   
   cache:
     enabled: false  # Disable cache for testing

Production Environment
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # orchestrator.yaml for production
   execution:
     parallel_tasks: 50
     timeout_seconds: 600
     retry_attempts: 5
   
   monitoring:
     log_level: WARNING
     metrics_enabled: true
   
   error_handling:
     circuit_breaker_threshold: 10
     fallback_enabled: true

Resource-Constrained Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # models.yaml for limited resources
   models:
     # Only small, efficient models
     - source: ollama
       name: llama3.2:1b
       expertise: [general, fast]
       size: 1b
       
     - source: ollama
       name: phi-3-mini:3.8b
       expertise: [reasoning, compact]
       size: 3.8b

High-Performance Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # orchestrator.yaml for high performance
   execution:
     parallel_tasks: 100
     use_gpu: true
   
   resources:
     max_memory_mb: 65536
     gpu_memory_fraction: 0.9
   
   cache:
     backend: redis
     redis_url: redis://localhost:6379

Troubleshooting
---------------

Configuration Not Found
~~~~~~~~~~~~~~~~~~~~~~~

If Orchestrator can't find your configuration:

1. Check file exists in one of the search paths
2. Verify file permissions are readable
3. Set ``ORCHESTRATOR_HOME`` environment variable
4. Run ``orchestrator-install-configs`` to install defaults

Invalid Configuration
~~~~~~~~~~~~~~~~~~~~~

If configuration validation fails:

1. Check YAML syntax is valid
2. Verify all required fields are present
3. Ensure model names are correctly formatted
4. Review error messages for specific issues

Performance Issues
~~~~~~~~~~~~~~~~~~

If experiencing slow performance:

1. Reduce ``parallel_tasks`` if system is overloaded
2. Enable caching for repeated operations
3. Use smaller models for simple tasks
4. Check resource limits aren't too restrictive