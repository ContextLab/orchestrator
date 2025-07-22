Validation Tools
================

The validation tools provide comprehensive data validation capabilities including JSON Schema validation, custom format support, and intelligent type coercion.

.. contents:: Table of Contents
   :local:
   :depth: 2

ValidationTool
--------------

.. autoclass:: orchestrator.tools.validation.ValidationTool
   :members:
   :undoc-members:
   :show-inheritance:

   The ValidationTool provides three main actions:

   * **validate** - Validate data against a JSON Schema
   * **infer_schema** - Automatically infer a schema from sample data  
   * **extract_structured** - Extract structured data from text (coming soon)

   **Example usage in YAML:**

   .. code-block:: yaml

      steps:
        - id: validate_user_data
          tool: validation
          action: validate
          parameters:
            data: "{{ user_input }}"
            schema:
              type: object
              properties:
                name:
                  type: string
                  minLength: 1
                email:
                  type: string
                  format: email
                age:
                  type: integer
                  minimum: 0
              required: ["name", "email"]
            mode: strict

   **Example usage in Python:**

   .. code-block:: python

      from orchestrator.tools.validation import ValidationTool
      
      tool = ValidationTool()
      
      # Validate data
      result = await tool.execute(
          action="validate",
          data={"name": "John", "email": "john@example.com", "age": 30},
          schema={
              "type": "object",
              "properties": {
                  "name": {"type": "string"},
                  "email": {"type": "string", "format": "email"},
                  "age": {"type": "integer"}
              },
              "required": ["name", "email"]
          },
          mode="strict"
      )
      
      if result["valid"]:
          print("Data is valid!")
      else:
          print("Validation errors:", result["errors"])

Validation Modes
----------------

.. autoclass:: orchestrator.tools.validation.ValidationMode
   :members:
   :undoc-members:
   :show-inheritance:

   The validation tool supports three modes:

   * **STRICT** - Fail on any validation error (default)
   * **LENIENT** - Attempt to coerce compatible types and warn on minor issues
   * **REPORT_ONLY** - Never fail, only report validation issues

   Type coercion in lenient mode:

   * String to integer: ``"42"`` → ``42``
   * String to number: ``"3.14"`` → ``3.14``  
   * String to boolean: ``"true"`` → ``True``, ``"false"`` → ``False``
   * Number to string: ``42`` → ``"42"``

Schema State
------------

.. autoclass:: orchestrator.tools.validation.SchemaState
   :members:
   :undoc-members:
   :show-inheritance:

   Schema resolution states:

   * **FIXED** - Fully determined at compile time
   * **PARTIAL** - Some parts known, others ambiguous
   * **AMBIGUOUS** - Cannot be determined until runtime

Format Validators
-----------------

.. autoclass:: orchestrator.tools.validation.FormatValidator
   :members:
   :undoc-members:
   :show-inheritance:

   Built-in format validators:

   * **model-id** - AI model identifiers (e.g., ``openai/gpt-4``)
   * **tool-name** - Tool names (e.g., ``web-search``)
   * **file-path** - Valid file system paths
   * **yaml-path** - JSONPath expressions
   * **pipeline-ref** - Pipeline identifiers
   * **task-ref** - Task output references (e.g., ``task1.output``)

   **Registering custom formats:**

   .. code-block:: python

      from orchestrator.tools.validation import ValidationTool
      
      tool = ValidationTool()
      
      # Pattern-based validator
      tool.register_format(
          "order-id",
          r"^ORD-\d{6}$",
          "Order ID format (ORD-XXXXXX)"
      )
      
      # Function-based validator
      def validate_even(value):
          return isinstance(value, int) and value % 2 == 0
          
      tool.register_format(
          "even-number",
          validate_even,
          "Even integer validator"
      )

Schema Validator
----------------

.. autoclass:: orchestrator.tools.validation.SchemaValidator
   :members:
   :undoc-members:
   :show-inheritance:

   Core schema validation engine using JSON Schema Draft 7.

Validation Result
-----------------

.. autoclass:: orchestrator.tools.validation.ValidationResult
   :members:
   :undoc-members:
   :show-inheritance:

   Result object containing validation outcome and details.

Working with AUTO Tags
----------------------

The ValidationTool supports AUTO tags for dynamic schema and mode selection:

.. code-block:: yaml

   steps:
     - id: smart_validation
       tool: validation
       action: validate
       parameters:
         data: "{{ input_data }}"
         schema: <AUTO>Infer appropriate schema based on the data structure</AUTO>
         mode: <AUTO>Choose validation mode based on data quality and criticality</AUTO>

Schema Inference Example
------------------------

.. code-block:: python

   from orchestrator.tools.validation import ValidationTool
   
   tool = ValidationTool()
   
   # Sample data
   sample_data = {
       "users": [
           {
               "name": "Alice",
               "email": "alice@example.com",
               "age": 30,
               "active": True
           }
       ],
       "created": "2024-01-15"
   }
   
   # Infer schema
   result = await tool.execute(
       action="infer_schema",
       data=sample_data
   )
   
   print("Inferred schema:")
   print(json.dumps(result["schema"], indent=2))

This will generate a schema with appropriate types and detected formats (e.g., email format for the email field).

Integration with Pipelines
--------------------------

Data validation can be used as quality gates in pipelines:

.. code-block:: yaml

   steps:
     - id: fetch_data
       tool: web-search
       parameters:
         query: "{{ search_term }}"
     
     - id: validate_results
       tool: validation
       action: validate
       parameters:
         data: "{{ fetch_data.results }}"
         schema:
           type: array
           items:
             type: object
             properties:
               title: {type: string}
               url: {type: string, format: uri}
             required: ["title", "url"]
         mode: strict
       
     - id: process_valid_data
       tool: data-processing
       parameters:
         data: "{{ validate_results.data }}"
       dependencies: [validate_results]
       condition: "{{ validate_results.valid == true }}"

See Also
--------

* :doc:`data_tools` - For data processing and transformation
* :doc:`base` - For creating custom tools
* `JSON Schema Documentation <https://json-schema.org/>`_ - For schema syntax reference