Orchestrator Framework Documentation
====================================

.. image:: https://img.shields.io/badge/python-3.8%2B-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.8+

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: https://opensource.org/licenses/MIT
   :alt: MIT License

.. image:: https://img.shields.io/badge/build-passing-brightgreen.svg
   :alt: Build Status

Welcome to the **Orchestrator Framework** - a powerful, flexible, and production-ready system for orchestrating AI workflows. Whether you're building simple task chains or complex multi-model pipelines, Orchestrator provides the tools you need to create, manage, and scale your AI applications.

.. note::
   This documentation covers version 1.0.0 of the Orchestrator Framework. For the latest updates and features, please visit our `GitHub repository <https://github.com/ContextLab/orchestrator>`_.

Quick Start
-----------

Get started with Orchestrator in just a few minutes:

.. code-block:: bash

   pip install py-orc

.. code-block:: python

   from orchestrator import Orchestrator, Task, Pipeline
   
   # Create a simple task
   task = Task(
       id="hello_world",
       name="Hello World Task",
       action="generate_text",
       parameters={"prompt": "Hello, world!"}
   )
   
   # Create a pipeline
   pipeline = Pipeline(id="demo", name="Demo Pipeline")
   pipeline.add_task(task)
   
   # Execute with orchestrator
   orchestrator = Orchestrator()
   result = await orchestrator.execute_pipeline(pipeline)
   print(result)

Key Features
------------

.. raw:: html

   <div class="feature-grid">
       <div class="feature-card">
           <h3>🚀 Easy to Use</h3>
           <p>Intuitive API design with comprehensive documentation and tutorials</p>
       </div>
       <div class="feature-card">
           <h3>⚡ High Performance</h3>
           <p>Parallel execution, caching, and resource optimization</p>
       </div>
       <div class="feature-card">
           <h3>🔒 Production Ready</h3>
           <p>Error handling, monitoring, and enterprise-grade security</p>
       </div>
       <div class="feature-card">
           <h3>🔧 Extensible</h3>
           <p>Plugin architecture for custom models and adapters</p>
       </div>
       <div class="feature-card">
           <h3>📊 YAML Configuration</h3>
           <p>Declarative workflow definition with template support</p>
       </div>
       <div class="feature-card">
           <h3>🤖 Multi-Model</h3>
           <p>Support for OpenAI, Anthropic, Google, and local models</p>
       </div>
   </div>

What Makes Orchestrator Special?
--------------------------------

**Declarative Workflows**
   Define your AI workflows in YAML with automatic dependency resolution and intelligent model selection.

**Production-Grade Features**
   Built-in error handling, circuit breakers, retry strategies, and comprehensive monitoring.

**Resource Management**
   Intelligent resource allocation, parallel execution, and multi-level caching for optimal performance.

**Extensible Architecture**
   Plugin-based design supports custom models, adapters, and execution strategies.

**Developer Experience**
   Comprehensive tutorials, extensive documentation, and intuitive APIs make it easy to get started.

Architecture Overview
--------------------

The Orchestrator Framework is built with a modular architecture that separates concerns and promotes extensibility:

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────┐
   │                    Orchestrator Engine                      │
   └─────────────────────────────────────────────────────────────┘
   
   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
   │  YAML Compiler  │  │ Model Registry  │  │ State Manager   │
   │  - Parser       │  │ - Selection     │  │ - Checkpoints   │
   │  - Validation   │  │ - Load Balance  │  │ - Recovery      │
   │  - Templates    │  │ - Health Check  │  │ - Persistence   │
   └─────────────────┘  └─────────────────┘  └─────────────────┘
   
   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
   │ Execution Layer │  │ Error Handler   │  │ Resource Mgmt   │
   │ - Parallel      │  │ - Circuit Break │  │ - Allocation    │
   │ - Sandboxed     │  │ - Retry Logic   │  │ - Monitoring    │
   │ - Distributed   │  │ - Recovery      │  │ - Optimization  │
   └─────────────────┘  └─────────────────┘  └─────────────────┘

Documentation Structure
-----------------------

This documentation is organized into several sections:

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started/installation
   getting_started/quickstart
   getting_started/basic_concepts
   getting_started/your_first_pipeline
   getting_started/cli_reference

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/yaml_configuration
   user_guide/model_configuration
   user_guide/models_and_adapters
   user_guide/error_handling
   user_guide/resource_management
   user_guide/state_management
   user_guide/monitoring

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/core
   api/compiler
   api/executor
   api/models
   api/state
   api/adapters
   api/utilities
   api/validation

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics

   advanced/custom_models
   advanced/control_systems
   advanced/performance_optimization
   advanced/deployment
   advanced/troubleshooting

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/notebooks
   tutorials/examples
   tutorials/integration_guides

.. toctree::
   :maxdepth: 1
   :caption: Development

   development/contributing
   development/testing
   development/github_actions
   development/architecture

Community & Support
-------------------

.. raw:: html

   <div class="feature-grid">
       <div class="feature-card">
           <h3>📚 Documentation</h3>
           <p>Comprehensive guides and API reference</p>
       </div>
       <div class="feature-card">
           <h3>🐛 Issue Tracker</h3>
           <p>Report bugs and request features on GitHub</p>
       </div>
       <div class="feature-card">
           <h3>💬 Discussions</h3>
           <p>Community support and knowledge sharing</p>
       </div>
       <div class="feature-card">
           <h3>🔧 Contributing</h3>
           <p>Help improve the framework</p>
       </div>
   </div>

License
-------

The Orchestrator Framework is released under the MIT License. See the `LICENSE <https://github.com/ContextLab/orchestrator/blob/main/LICENSE>`_ file for details.

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`