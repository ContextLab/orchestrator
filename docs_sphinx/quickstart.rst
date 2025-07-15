==========
Quickstart
==========

This quickstart guide will walk you through creating and running your first Orchestrator pipeline in under 5 minutes.

Your First Pipeline
===================

Let's create a simple pipeline that generates a summary of any topic.

Step 1: Create the Pipeline Definition
--------------------------------------

Create a file called ``summarize.yaml``:

.. code-block:: yaml

   name: topic-summarizer
   description: Generate a concise summary of any topic
   
   inputs:
     topic:
       type: string
       description: The topic to summarize
       required: true
     
     length:
       type: integer
       description: Approximate word count for the summary
       default: 200
   
   outputs:
     summary:
       type: string
       value: "{{ inputs.topic }}_summary.txt"
   
   steps:
     - id: research
       action: generate_content
       parameters:
         prompt: |
           Research and provide key information about: {{ inputs.topic }}
           Focus on the most important and interesting aspects.
         max_length: 500
     
     - id: summarize
       action: generate_summary
       parameters:
         content: "$results.research"
         target_length: "{{ inputs.length }}"
         style: <AUTO>Choose appropriate style for the topic</AUTO>
     
     - id: save_summary
       action: write_file
       parameters:
         path: "{{ outputs.summary }}"
         content: "$results.summarize"

Step 2: Run the Pipeline
------------------------

Create a Python script to run your pipeline:

.. code-block:: python

   import orchestrator as orc
   
   # Initialize the model pool
   orc.init_models()
   
   # Compile the pipeline
   pipeline = orc.compile("summarize.yaml")
   
   # Run with different topics
   result1 = pipeline.run(
       topic="quantum computing",
       length=150
   )
   
   result2 = pipeline.run(
       topic="sustainable energy",
       length=250
   )
   
   print(f"Created summaries: {result1}, {result2}")

Step 3: Check the Results
-------------------------

Your pipeline will create two files:
- ``quantum_computing_summary.txt``
- ``sustainable_energy_summary.txt``

Each contains a tailored summary of the specified length.

Building More Complex Pipelines
===============================

Research Report Pipeline
------------------------

Let's create a more sophisticated pipeline that generates research reports:

.. code-block:: yaml

   name: research-report-generator
   description: Generate comprehensive research reports with citations
   
   inputs:
     topic:
       type: string
       required: true
     focus_areas:
       type: array
       description: Specific areas to focus on
       default: []
   
   outputs:
     report_pdf:
       type: string
       value: "reports/{{ inputs.topic }}_report.pdf"
   
   steps:
     # Web search for recent information
     - id: search_recent
       action: search_web
       parameters:
         query: "{{ inputs.topic }} 2024 latest developments"
         max_results: 10
     
     # Search academic sources
     - id: search_academic
       action: search_web
       parameters:
         query: "{{ inputs.topic }} research papers scholarly"
         max_results: 5
     
     # Compile all sources
     - id: compile_sources
       action: compile_markdown
       parameters:
         sources:
           - "$results.search_recent"
           - "$results.search_academic"
         include_citations: true
     
     # Generate the report
     - id: write_report
       action: generate_report
       parameters:
         research: "$results.compile_sources"
         topic: "{{ inputs.topic }}"
         focus_areas: "{{ inputs.focus_areas }}"
         style: "academic"
         sections:
           - "Executive Summary"
           - "Introduction"
           - "Current State"
           - "Recent Developments"
           - "Future Outlook"
           - "Conclusions"
     
     # Quality check
     - id: validate
       action: validate_report
       parameters:
         report: "$results.write_report"
         checks:
           - completeness
           - citation_accuracy
           - readability
     
     # Generate PDF
     - id: create_pdf
       action: "!pandoc -o {{ outputs.report_pdf }} --pdf-engine=xelatex"
       parameters:
         input: "$results.write_report"

Working with Tools
==================

Orchestrator automatically detects and configures tools based on your pipeline actions.

Available Tool Actions
----------------------

**Web Tools**:

.. code-block:: yaml

   # Web search
   - action: search_web
     parameters:
       query: "your search query"
   
   # Scrape webpage
   - action: scrape_page
     parameters:
       url: "https://example.com"

**System Tools**:

.. code-block:: yaml

   # Run shell commands (prefix with !)
   - action: "!ls -la"
   
   # File operations
   - action: read_file
     parameters:
       path: "data.txt"
   
   - action: write_file
     parameters:
       path: "output.txt"
       content: "Your content"

**Data Tools**:

.. code-block:: yaml

   # Process data
   - action: transform_data
     parameters:
       input: "$results.previous_step"
       operations:
         - type: filter
           condition: "value > 100"
   
   # Validate data
   - action: validate_data
     parameters:
       data: "$results.data"
       schema:
         type: object
         required: ["name", "value"]

Using AUTO Tags
===============

AUTO tags let AI models make intelligent decisions:

.. code-block:: yaml

   steps:
     - id: analyze
       action: analyze_data
       parameters:
         data: "$results.fetch"
         method: <AUTO>Choose best analysis method based on data type</AUTO>
         visualization: <AUTO>Determine if visualization would be helpful</AUTO>
         depth: <AUTO>Set analysis depth (shallow/medium/deep)</AUTO>

The AI model will examine the context and make appropriate choices.

Pipeline Composition
====================

You can compose pipelines from smaller, reusable components:

.. code-block:: yaml

   name: composite-pipeline
   
   imports:
     - common/data_fetcher.yaml as fetcher
     - common/validator.yaml as validator
   
   steps:
     # Use imported pipeline
     - id: fetch_data
       pipeline: fetcher
       parameters:
         source: "api"
     
     # Local step
     - id: process
       action: process_data
       parameters:
         data: "$results.fetch_data"
     
     # Use another import
     - id: validate
       pipeline: validator
       parameters:
         data: "$results.process"

Error Handling
==============

Add error handling to make pipelines robust:

.. code-block:: yaml

   steps:
     - id: risky_operation
       action: fetch_external_data
       parameters:
         url: "{{ inputs.data_source }}"
       error_handling:
         retry:
           max_attempts: 3
           backoff: exponential
         fallback:
           action: use_cached_data
           parameters:
             cache_key: "{{ inputs.topic }}"

Debugging Pipelines
===================

Enable debug mode for detailed execution logs:

.. code-block:: python

   import logging
   import orchestrator as orc
   
   # Enable debug logging
   logging.basicConfig(level=logging.DEBUG)
   
   # Compile with debug flag
   pipeline = orc.compile("pipeline.yaml", debug=True)
   
   # Run with verbose output
   result = pipeline.run(
       topic="test",
       _verbose=True,
       _step_callback=lambda step: print(f"Executing: {step.id}")
   )

Best Practices
==============

1. **Use Descriptive IDs**: Make step IDs self-documenting
2. **Leverage Templates**: Use Jinja2 templates for dynamic values
3. **Handle Errors**: Always consider what could go wrong
4. **Validate Inputs**: Define clear input schemas
5. **Document Purpose**: Add descriptions to pipelines and steps

Next Steps
==========

Now that you've built your first pipelines:

- Explore :doc:`tutorials/index` for in-depth tutorials
- Check out :doc:`examples/index` for real-world examples
- Learn about :doc:`concepts` for deeper understanding
- Review the :doc:`api/index` for advanced features

.. tip::

   Try modifying the examples above to create your own custom pipelines. The best way to learn is by experimenting!