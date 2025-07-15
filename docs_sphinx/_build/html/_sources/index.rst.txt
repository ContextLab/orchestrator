.. Orchestrator documentation master file

==============================================
Orchestrator: AI Pipeline Orchestration Framework
==============================================

.. image:: https://img.shields.io/badge/python-3.11+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License

**Orchestrator** is a powerful AI pipeline orchestration framework that provides a unified interface for executing AI workflows defined in YAML. It serves as an intelligent wrapper around LangGraph, MCP (Model Context Protocol), and other AI agent control systems.

.. raw:: html

   <div class="feature-grid">
      <div class="feature-box">
         <h3>🎯 Input-Agnostic Pipelines</h3>
         <p>Create reusable pipelines that adapt to different inputs dynamically</p>
      </div>
      <div class="feature-box">
         <h3>🤖 Intelligent Ambiguity Resolution</h3>
         <p>Let AI models resolve ambiguous values using <code>&lt;AUTO&gt;</code> tags</p>
      </div>
      <div class="feature-box">
         <h3>🔧 Tool Integration</h3>
         <p>Seamlessly integrate real-world tools via MCP protocol</p>
      </div>
      <div class="feature-box">
         <h3>🔄 State Management</h3>
         <p>Built-in checkpointing and recovery for robust execution</p>
      </div>
   </div>

Quick Example
-------------

.. code-block:: python

   import orchestrator as orc

   # Initialize models
   orc.init_models()

   # Compile a pipeline
   pipeline = orc.compile("pipelines/research-report.yaml")

   # Execute with different inputs
   result = pipeline.run(
       topic="quantum_computing",
       instructions="Focus on error correction"
   )

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   getting_started
   installation
   quickstart
   concepts

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   tutorials/index
   examples/index
   best_practices
   troubleshooting

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api/index
   api/core
   api/compiler
   api/models
   api/tools
   api/orchestrator

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide
   :hidden:

   contributing
   architecture
   extending
   testing

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources
   :hidden:

   changelog
   faq
   glossary
   license

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. raw:: html

   <style>
   .feature-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
      gap: 20px;
      margin: 30px 0;
   }
   .feature-box {
      padding: 20px;
      border: 1px solid #e0e0e0;
      border-radius: 8px;
      background: #f9f9f9;
   }
   .feature-box h3 {
      margin-top: 0;
      color: #333;
   }
   .feature-box p {
      margin-bottom: 0;
      color: #666;
   }
   </style>