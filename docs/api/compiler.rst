Compiler API Reference
======================

This section documents the YAML compilation and parsing components.

.. note::
   For hands-on examples of YAML configuration, see the :doc:`../user_guide/yaml_configuration` guide.

YAML Compiler
-------------

The ``YAMLCompiler`` class processes YAML pipeline definitions and converts them to executable ``Pipeline`` objects.

**Key Methods:**
- ``compile(yaml_content)`` - Compile YAML string to Pipeline
- ``compile_file(file_path)`` - Compile YAML file to Pipeline

**Example Usage:**

.. code-block:: python

   from orchestrator.compiler import YAMLCompiler
   
   compiler = YAMLCompiler()
   pipeline = compiler.compile_file("my_pipeline.yaml")

Schema Validator
----------------

The ``SchemaValidator`` class validates pipeline YAML against the expected schema.

**Key Methods:**
- ``validate(pipeline_dict)`` - Validate pipeline dictionary
- ``get_schema()`` - Get the validation schema

Ambiguity Resolver
------------------

The ``AmbiguityResolver`` class resolves ``<AUTO>`` tags in pipeline definitions.

**Key Methods:**
- ``resolve_ambiguity(ambiguity_dict)`` - Resolve ambiguous specification
- ``classify_ambiguity(content)`` - Classify type of ambiguity

**Ambiguity Types:**
- ``ACTION`` - Ambiguous task actions
- ``PARAMETER`` - Ambiguous parameters
- ``MODEL`` - Ambiguous model selection
- ``RESOURCE`` - Ambiguous resource allocation