YAML Configuration
==================

Learn how to define AI workflows declaratively using YAML configuration files.

.. note::
   This guide covers the YAML configuration system. For hands-on practice, see the :doc:`../tutorials/notebooks` tutorial 02.

Overview
--------

YAML configuration allows you to define pipelines declaratively, separating workflow logic from implementation details. This approach provides several benefits:

* **Readability** - Easy to understand and maintain
* **Reusability** - Templates can be shared and reused
* **Versioning** - Track changes to workflows over time
* **Collaboration** - Non-programmers can modify workflows

Basic YAML Structure
--------------------

A basic pipeline YAML file contains:

.. code-block:: yaml

   id: my_pipeline
   name: My Pipeline
   description: A sample pipeline
   
   tasks:
     - id: task1
       name: First Task
       action: generate_text
       parameters:
         prompt: "Hello, world!"
     
     - id: task2
       name: Second Task
       action: generate_text
       parameters:
         prompt: "Process this: {task1}"
       dependencies:
         - task1

Template Variables
------------------

Use template variables for dynamic content:

.. code-block:: yaml

   id: research_pipeline
   name: Research Pipeline
   
   context:
     topic: artificial intelligence
     depth: detailed
   
   tasks:
     - id: research
       name: Research Task
       action: generate_text
       parameters:
         prompt: "Research {topic} with {depth} analysis"

AUTO Resolution
---------------

The AUTO tag automatically resolves ambiguous parameters:

.. code-block:: yaml

   tasks:
     - id: analysis
       name: Analysis Task
       action: <AUTO>
       parameters:
         data: {previous_task}
         model: <AUTO>

For complete documentation, see the :doc:`../api/compiler` reference.