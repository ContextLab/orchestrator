Data Tools
==========

Tools for data processing, transformation, and validation.

.. automodule:: orchestrator.tools.data_tools
   :members:
   :undoc-members:
   :show-inheritance:

DataProcessingTool
------------------

.. autoclass:: orchestrator.tools.data_tools.DataProcessingTool
   :members:
   :undoc-members:
   :show-inheritance:

Comprehensive data processing and transformation tool.

**Parameters:**

* ``action`` (string, required): Processing action ("transform", "filter", "aggregate", "merge", "sort")
* ``data`` (object, required): Input data (list, dict, or pandas-compatible format)
* ``operations`` (array, optional): List of operations to apply
* ``schema`` (object, optional): Data schema for validation
* ``output_format`` (string, optional): Output format ("json", "csv", "parquet")

**Actions:**

transform
~~~~~~~~~

Apply transformations to data:

.. code-block:: python

   result = await data_tool.execute(
       action="transform",
       data=[
           {"name": "John", "age": 30, "salary": 50000},
           {"name": "Jane", "age": 25, "salary": 60000}
       ],
       operations=[
           {"type": "add_column", "name": "tax", "expression": "salary * 0.2"},
           {"type": "rename", "from": "age", "to": "years"},
           {"type": "format", "column": "salary", "format": "currency"}
       ]
   )

**Returns:**

.. code-block:: python

   {
       "success": True,
       "data": [
           {"name": "John", "years": 30, "salary": "$50,000", "tax": 10000},
           {"name": "Jane", "years": 25, "salary": "$60,000", "tax": 12000}
       ],
       "operations_applied": 3,
       "rows_processed": 2,
       "execution_time": 0.15
   }

filter
~~~~~~

Filter data based on conditions:

.. code-block:: python

   result = await data_tool.execute(
       action="filter",
       data=employee_data,
       operations=[
           {"type": "where", "condition": "salary > 45000"},
           {"type": "where", "condition": "age < 35"},
           {"type": "limit", "count": 10}
       ]
   )

aggregate
~~~~~~~~~

Perform aggregations on data:

.. code-block:: python

   result = await data_tool.execute(
       action="aggregate",
       data=sales_data,
       operations=[
           {"type": "group_by", "columns": ["region", "product"]},
           {"type": "sum", "column": "sales", "as": "total_sales"},
           {"type": "avg", "column": "price", "as": "avg_price"},
           {"type": "count", "as": "record_count"}
       ]
   )

merge
~~~~~

Merge multiple datasets:

.. code-block:: python

   result = await data_tool.execute(
       action="merge",
       data={
           "employees": employee_data,
           "departments": department_data
       },
       operations=[
           {
               "type": "inner_join",
               "left": "employees",
               "right": "departments", 
               "on": "dept_id"
           }
       ]
   )

**Example Usage:**

.. code-block:: python

   from orchestrator.tools.data_tools import DataProcessingTool
   import asyncio
   
   async def process_sales_data():
       data_tool = DataProcessingTool()
       
       # Sample sales data
       sales_data = [
           {"date": "2024-01-01", "product": "A", "sales": 100, "region": "North"},
           {"date": "2024-01-01", "product": "B", "sales": 150, "region": "South"},
           {"date": "2024-01-02", "product": "A", "sales": 120, "region": "North"},
           {"date": "2024-01-02", "product": "B", "sales": 180, "region": "South"}
       ]
       
       # Transform and aggregate
       result = await data_tool.execute(
           action="aggregate",
           data=sales_data,
           operations=[
               {"type": "group_by", "columns": ["product"]},
               {"type": "sum", "column": "sales", "as": "total_sales"},
               {"type": "avg", "column": "sales", "as": "avg_sales"}
           ],
           output_format="json"
       )
       
       return result
   
   # Run processing
   asyncio.run(process_sales_data())

**Pipeline Usage:**

.. code-block:: yaml

   steps:
     - id: process_data
       action: transform_data
       parameters:
         action: "transform"
         data: "{{ results.load_data.records }}"
         operations:
           - type: "add_column"
             name: "full_name"
             expression: "first_name + ' ' + last_name"
           - type: "filter"
             condition: "active = true"
         output_format: "json"

ValidationTool
--------------

.. autoclass:: orchestrator.tools.data_tools.ValidationTool
   :members:
   :undoc-members:
   :show-inheritance:

Validates data against schemas and business rules.

**Parameters:**

* ``data`` (object, required): Data to validate
* ``rules`` (array, required): Validation rules to apply
* ``schema`` (object, optional): JSON schema for structure validation
* ``strict`` (boolean, optional): Strict validation mode (default: False)
* ``report_format`` (string, optional): Report format ("summary", "detailed")

**Validation Rules:**

Type Validation
~~~~~~~~~~~~~~~

.. code-block:: python

   rules = [
       {
           "type": "type_check",
           "field": "age", 
           "expected_type": "integer",
           "severity": "error"
       },
       {
           "type": "type_check",
           "field": "email",
           "expected_type": "string",
           "severity": "error"
       }
   ]

Range Validation
~~~~~~~~~~~~~~~~

.. code-block:: python

   rules = [
       {
           "type": "range",
           "field": "age",
           "min": 0,
           "max": 150,
           "severity": "error"
       },
       {
           "type": "range",
           "field": "salary",
           "min": 0,
           "severity": "warning"
       }
   ]

Pattern Validation
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   rules = [
       {
           "type": "pattern",
           "field": "email",
           "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
           "severity": "error",
           "message": "Invalid email format"
       },
       {
           "type": "pattern",
           "field": "phone",
           "pattern": r"^\+?1?[0-9]{10,14}$",
           "severity": "warning"
       }
   ]

Required Fields
~~~~~~~~~~~~~~~

.. code-block:: python

   rules = [
       {
           "type": "required",
           "field": "id",
           "severity": "error"
       },
       {
           "type": "required",
           "field": "name",
           "severity": "error"
       }
   ]

Custom Validation
~~~~~~~~~~~~~~~~~

.. code-block:: python

   rules = [
       {
           "type": "custom",
           "field": "birth_date",
           "function": "validate_birth_date",
           "parameters": {"max_age": 120},
           "severity": "error"
       }
   ]

**Example Usage:**

.. code-block:: python

   from orchestrator.tools.data_tools import ValidationTool
   import asyncio
   
   async def validate_employee_data():
       validator = ValidationTool()
       
       # Employee data to validate
       employee_data = [
           {"id": 1, "name": "John Doe", "age": 30, "email": "john@example.com"},
           {"id": 2, "name": "", "age": -5, "email": "invalid-email"},
           {"id": None, "name": "Jane Smith", "age": 25, "email": "jane@example.com"}
       ]
       
       # Validation rules
       rules = [
           {"type": "required", "field": "id", "severity": "error"},
           {"type": "required", "field": "name", "severity": "error"},
           {"type": "type_check", "field": "age", "expected_type": "integer", "severity": "error"},
           {"type": "range", "field": "age", "min": 0, "max": 150, "severity": "error"},
           {"type": "pattern", "field": "email", "pattern": r".+@.+\..+", "severity": "warning"}
       ]
       
       # Run validation
       result = await validator.execute(
           data=employee_data,
           rules=rules,
           report_format="detailed"
       )
       
       return result
   
   # Run validation
   asyncio.run(validate_employee_data())

**Returns:**

.. code-block:: python

   {
       "success": True,
       "valid": False,
       "summary": {
           "total_records": 3,
           "valid_records": 1,
           "records_with_errors": 2,
           "records_with_warnings": 1,
           "total_issues": 4
       },
       "issues": [
           {
               "record_index": 1,
               "field": "name",
               "rule": "required",
               "severity": "error",
               "message": "Required field 'name' is empty"
           },
           {
               "record_index": 1,
               "field": "age", 
               "rule": "range",
               "severity": "error",
               "message": "Value -5 is below minimum 0"
           },
           {
               "record_index": 2,
               "field": "id",
               "rule": "required", 
               "severity": "error",
               "message": "Required field 'id' is missing"
           }
       ],
       "valid_records": [0],
       "invalid_records": [1, 2]
   }

**Pipeline Usage:**

.. code-block:: yaml

   steps:
     - id: validate_input
       action: validate_data
       parameters:
         data: "{{ inputs.customer_data }}"
         rules:
           - type: "required"
             field: "customer_id"
             severity: "error"
           - type: "pattern"
             field: "email"
             pattern: ".+@.+\\..+"
             severity: "warning"
         strict: false
         report_format: "summary"

Data Formats
------------

Supported Input Formats
~~~~~~~~~~~~~~~~~~~~~~~~

* **JSON**: Native JavaScript objects and arrays
* **CSV**: Comma-separated values (with automatic parsing)
* **Pandas**: DataFrame-compatible dictionaries
* **XML**: Simple XML structures
* **YAML**: YAML data structures

.. code-block:: python

   # JSON format
   json_data = [{"name": "John", "age": 30}]
   
   # CSV format (auto-detected)
   csv_data = "name,age\nJohn,30\nJane,25"
   
   # Pandas-style
   pandas_data = {
       "name": ["John", "Jane"],
       "age": [30, 25]
   }

Supported Output Formats
~~~~~~~~~~~~~~~~~~~~~~~~~

* **JSON**: Standard JSON output
* **CSV**: Comma-separated values
* **Parquet**: Columnar storage format
* **Excel**: XLSX format
* **XML**: Structured XML

.. code-block:: python

   result = await data_tool.execute(
       action="transform",
       data=input_data,
       operations=operations,
       output_format="csv"  # or "json", "parquet", "excel"
   )

Schema Definitions
------------------

JSON Schema
~~~~~~~~~~~

Define data structure using JSON Schema:

.. code-block:: python

   schema = {
       "type": "object",
       "properties": {
           "id": {"type": "integer", "minimum": 1},
           "name": {"type": "string", "minLength": 1},
           "email": {"type": "string", "format": "email"},
           "age": {"type": "integer", "minimum": 0, "maximum": 150}
       },
       "required": ["id", "name", "email"]
   }

Custom Schema
~~~~~~~~~~~~~

Define custom validation schema:

.. code-block:: python

   custom_schema = {
       "fields": {
           "customer_id": {"type": "string", "pattern": "^CUST[0-9]{6}$"},
           "order_date": {"type": "date", "format": "YYYY-MM-DD"},
           "amount": {"type": "decimal", "precision": 2, "min": 0}
       },
       "relationships": [
           {"type": "foreign_key", "field": "customer_id", "references": "customers.id"}
       ]
   }

Performance Optimization
------------------------

Batch Processing
~~~~~~~~~~~~~~~~

Process large datasets in batches:

.. code-block:: python

   async def process_large_dataset(data, batch_size=1000):
       data_tool = DataProcessingTool()
       results = []
       
       for i in range(0, len(data), batch_size):
           batch = data[i:i + batch_size]
           
           result = await data_tool.execute(
               action="transform",
               data=batch,
               operations=operations
           )
           
           results.extend(result["data"])
       
       return results

Memory Management
~~~~~~~~~~~~~~~~~

Handle memory efficiently for large datasets:

.. code-block:: python

   # Use streaming for large files
   async def stream_process_csv(file_path):
       data_tool = DataProcessingTool()
       
       result = await data_tool.execute(
           action="transform",
           data={"source": file_path, "streaming": True},
           operations=operations,
           chunk_size=10000
       )

Parallel Processing
~~~~~~~~~~~~~~~~~~~

Process data in parallel:

.. code-block:: python

   import asyncio
   
   async def parallel_processing(data_chunks):
       data_tool = DataProcessingTool()
       
       tasks = [
           data_tool.execute(
               action="transform",
               data=chunk,
               operations=operations
           )
           for chunk in data_chunks
       ]
       
       results = await asyncio.gather(*tasks)
       return results

Error Handling
--------------

Data Format Errors
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   {
       "success": False,
       "error": "invalid_format",
       "message": "Unable to parse CSV data",
       "details": {
           "line": 15,
           "column": 3,
           "expected": "number",
           "received": "text"
       }
   }

Validation Errors
~~~~~~~~~~~~~~~~~

.. code-block:: python

   {
       "success": False,
       "error": "validation_failed",
       "message": "Data validation failed",
       "issues": [
           {
               "field": "email",
               "message": "Invalid email format",
               "value": "not-an-email"
           }
       ]
   }

Memory Errors
~~~~~~~~~~~~~

.. code-block:: python

   {
       "success": False,
       "error": "memory_limit_exceeded",
       "message": "Dataset too large for available memory",
       "suggestion": "Use streaming mode or reduce batch size"
   }

Best Practices
--------------

Data Processing
~~~~~~~~~~~~~~~

* **Validate First**: Always validate data before processing
* **Handle Nulls**: Plan for missing and null values
* **Type Safety**: Ensure consistent data types
* **Error Recovery**: Implement recovery for partial failures
* **Performance**: Use appropriate batch sizes and streaming

.. code-block:: python

   async def robust_data_processing(data):
       validator = ValidationTool()
       data_tool = DataProcessingTool()
       
       # Validate first
       validation = await validator.execute(
           data=data,
           rules=validation_rules
       )
       
       if not validation["valid"]:
           # Handle validation errors
           clean_data = await handle_validation_errors(data, validation)
       else:
           clean_data = data
       
       # Process in batches
       results = []
       batch_size = 1000
       
       for i in range(0, len(clean_data), batch_size):
           batch = clean_data[i:i + batch_size]
           
           try:
               result = await data_tool.execute(
                   action="transform",
                   data=batch,
                   operations=operations
               )
               results.extend(result["data"])
               
           except Exception as e:
               # Log error and continue with next batch
               logger.error(f"Batch {i} failed: {e}")
               continue
       
       return results

Schema Design
~~~~~~~~~~~~~

* **Clear Types**: Use specific, appropriate data types
* **Meaningful Names**: Use descriptive field names
* **Constraints**: Define appropriate constraints and validations
* **Documentation**: Document schema purpose and usage
* **Versioning**: Version schemas for compatibility

Examples
--------

Customer Data Pipeline
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   async def customer_data_pipeline(raw_data):
       validator = ValidationTool()
       data_tool = DataProcessingTool()
       
       # Define validation rules
       rules = [
           {"type": "required", "field": "customer_id"},
           {"type": "pattern", "field": "email", "pattern": r".+@.+\..+"},
           {"type": "range", "field": "age", "min": 0, "max": 150}
       ]
       
       # Validate data
       validation = await validator.execute(
           data=raw_data,
           rules=rules
       )
       
       # Clean and transform
       cleaned = await data_tool.execute(
           action="transform",
           data=raw_data,
           operations=[
               {"type": "remove_nulls"},
               {"type": "standardize_names"},
               {"type": "add_column", "name": "created_at", "value": "now()"}
           ]
       )
       
       return {
           "validation": validation,
           "processed_data": cleaned["data"]
       }

Sales Analytics
~~~~~~~~~~~~~~~

.. code-block:: python

   async def sales_analytics(sales_data):
       data_tool = DataProcessingTool()
       
       # Aggregate sales by region and product
       regional_sales = await data_tool.execute(
           action="aggregate",
           data=sales_data,
           operations=[
               {"type": "group_by", "columns": ["region", "product"]},
               {"type": "sum", "column": "sales", "as": "total_sales"},
               {"type": "avg", "column": "price", "as": "avg_price"},
               {"type": "count", "as": "transaction_count"}
           ]
       )
       
       # Calculate growth trends
       trends = await data_tool.execute(
           action="transform",
           data=regional_sales["data"],
           operations=[
               {"type": "sort", "by": "total_sales", "order": "desc"},
               {"type": "add_column", "name": "rank", "expression": "row_number()"},
               {"type": "add_column", "name": "market_share", 
                "expression": "total_sales / sum(total_sales)"}
           ]
       )
       
       return trends

For more examples, see :doc:`../../tutorials/examples/data_processing_workflow`.