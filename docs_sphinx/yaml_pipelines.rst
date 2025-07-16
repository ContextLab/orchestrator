==================
YAML Pipeline Guide
==================

This comprehensive guide covers everything you need to know about creating YAML pipelines in Orchestrator.

Pipeline Structure
==================

A complete pipeline definition consists of several sections, each serving a specific purpose:

.. code-block:: yaml

   # Pipeline metadata
   name: pipeline-name           # Required: Unique identifier
   description: Pipeline purpose # Required: Human-readable description
   version: "1.0.0"             # Optional: Version tracking
   
   # Input definitions
   inputs:
     parameter_name:
       type: string             # Required: string, integer, float, boolean, array, object
       description: Purpose     # Required: What this input does
       required: true          # Optional: Default is false
       default: "value"        # Optional: Default value if not provided
       validation:             # Optional: Input validation rules
         pattern: "^[a-z]+$"   # Regex for strings
         min: 0                # Minimum for numbers
         max: 100              # Maximum for numbers
         enum: ["a", "b"]      # Allowed values
   
   # Output definitions
   outputs:
     result_name:
       type: string            # Required: Output data type
       value: "expression"     # Required: How to generate the output
       description: Purpose    # Optional: What this output represents
   
   # Configuration
   config:
     timeout: 3600             # Optional: Global timeout in seconds
     parallel: true            # Optional: Enable parallel execution
     checkpoint: true          # Optional: Enable checkpointing
     error_mode: "continue"    # Optional: stop|continue|retry
   
   # Resource requirements
   resources:
     gpu: false                # Optional: Require GPU
     memory: "8GB"             # Optional: Memory requirement
     model_size: "large"       # Optional: Preferred model size
   
   # Pipeline steps
   steps:
     - id: step_identifier     # Required: Unique step ID
       action: action_name     # Required: What to do
       description: Purpose    # Optional: Step description
       parameters:             # Optional: Step parameters
         key: value
       depends_on: [step_id]   # Optional: Dependencies
       condition: expression   # Optional: Conditional execution
       error_handling:         # Optional: Error handling
         retry:
           max_attempts: 3
           backoff: exponential
         fallback:
           action: alternate_action

Complete Pipeline Elements
==========================

Metadata Section
----------------

The metadata section identifies and describes your pipeline:

.. code-block:: yaml

   name: advanced-research-pipeline
   description: |
     Multi-stage research pipeline that:
     - Searches multiple sources
     - Validates information
     - Generates comprehensive reports
   version: "2.1.0"
   author: "Your Name"
   tags: ["research", "automation", "reporting"]

Input Definitions
-----------------

Inputs make pipelines reusable. Each input can have:

**Basic Types**:

.. code-block:: yaml

   inputs:
     # String input with validation
     topic:
       type: string
       description: "Research topic to investigate"
       required: true
       validation:
         pattern: "^[A-Za-z0-9 ]+$"
         min_length: 3
         max_length: 100
     
     # Integer with range
     depth:
       type: integer
       description: "Research depth (1-5)"
       default: 3
       validation:
         min: 1
         max: 5
     
     # Boolean flag
     include_images:
       type: boolean
       description: "Include images in report"
       default: false
     
     # Array of strings
     sources:
       type: array
       description: "Preferred information sources"
       default: ["web", "academic"]
       validation:
         min_items: 1
         max_items: 10
         item_type: string
     
     # Complex object
     config:
       type: object
       description: "Advanced configuration"
       default:
         language: "en"
         format: "pdf"
       validation:
         properties:
           language:
             type: string
             enum: ["en", "es", "fr", "de"]
           format:
             type: string
             enum: ["pdf", "html", "markdown"]

Output Definitions
------------------

Outputs define what the pipeline produces:

.. code-block:: yaml

   outputs:
     # Simple file output
     report:
       type: string
       value: "reports/{{ inputs.topic | slugify }}_report.pdf"
       description: "Generated PDF report"
     
     # Dynamic output using AUTO
     summary:
       type: string
       value: <AUTO>Generate filename based on content</AUTO>
       description: "Executive summary document"
     
     # Computed output
     metrics:
       type: object
       value:
         word_count: "{{ results.final_report.word_count }}"
         sources_used: "{{ results.compile_sources.count }}"
         generation_time: "{{ execution.duration }}"
     
     # Multiple file outputs
     artifacts:
       type: array
       value:
         - "{{ outputs.report }}"
         - "data/{{ inputs.topic }}_data.json"
         - "images/{{ inputs.topic }}_charts.png"

Step Definitions
----------------

Steps are the core of your pipeline. Each step can use different patterns:

**Basic Actions**:

.. code-block:: yaml

   steps:
     # Simple action
     - id: fetch_data
       action: fetch_url
       parameters:
         url: "https://api.example.com/data"
     
     # Using input values
     - id: search
       action: search_web
       parameters:
         query: "{{ inputs.topic }} {{ inputs.year }}"
         max_results: "{{ inputs.depth * 5 }}"
     
     # Using previous results
     - id: analyze
       action: analyze_data
       parameters:
         data: "$results.fetch_data"
         method: "statistical"
     
     # Shell command (prefix with !)
     - id: convert
       action: "!pandoc -f markdown -t pdf -o output.pdf input.md"
     
     # Using AUTO tags
     - id: summarize
       action: generate_summary
       parameters:
         content: "$results.analyze"
         style: <AUTO>Choose style based on audience</AUTO>
         length: <AUTO>Determine optimal length</AUTO>

**Dependencies and Flow Control**:

.. code-block:: yaml

   steps:
     # Parallel execution (no dependencies)
     - id: source1
       action: fetch_source_a
     
     - id: source2
       action: fetch_source_b
     
     # Sequential execution
     - id: combine
       action: merge_data
       depends_on: [source1, source2]
       parameters:
         data1: "$results.source1"
         data2: "$results.source2"
     
     # Conditional execution
     - id: premium_analysis
       action: advanced_analysis
       condition: "{{ inputs.tier == 'premium' }}"
       parameters:
         data: "$results.combine"
     
     # Dynamic dependencies
     - id: final_step
       depends_on: "{{ ['combine', 'premium_analysis'] if inputs.tier == 'premium' else ['combine'] }}"

**Error Handling**:

.. code-block:: yaml

   steps:
     - id: risky_operation
       action: external_api_call
       error_handling:
         # Retry configuration
         retry:
           max_attempts: 3
           backoff: exponential  # or: constant, linear
           initial_delay: 1000   # milliseconds
           max_delay: 30000
         
         # Fallback action
         fallback:
           action: use_cached_data
           parameters:
             cache_key: "{{ inputs.topic }}"
         
         # Continue on error
         continue_on_error: true
         
         # Custom error message
         error_message: "Failed to fetch external data, using cache"

Template Expressions
====================

Orchestrator uses Jinja2 templating with extensions:

**Variable Access**:

Variables can be referenced throughout your pipeline using Jinja2-style template expressions:

.. code-block:: yaml

   id: variable_demo
   name: Variable Access Demo
   
   inputs:
     - name: user_topic
       type: string
       description: Topic to research
   
   steps:
     - id: initial_search
       action: search_web
       parameters:
         # Reference input variables
         query: "{{ user_topic }} latest news"
         
     - id: analyze_results
       action: analyze
       parameters:
         # Reference results from previous steps
         data: "{{ initial_search.results }}"
         # Can access nested fields
         first_result: "{{ initial_search.results[0].title }}"
       dependencies: [initial_search]
       
     - id: final_report
       action: generate_text
       parameters:
         # Combine multiple references
         prompt: |
           Create a report about {{ user_topic }}
           Based on: {{ analyze_results.summary }}
           Total results found: {{ initial_search.count }}
       dependencies: [analyze_results]
   
   outputs:
     # Define output variables
     report: "{{ final_report.result }}"
     summary: "{{ analyze_results.summary }}"
     search_count: "{{ initial_search.count }}"

**Filters and Functions**:

.. code-block:: text

   # String manipulation
   "{{ inputs.topic | lower }}"
   "{{ inputs.topic | upper }}"
   "{{ inputs.topic | slugify }}"
   "{{ inputs.topic | replace(' ', '_') }}"
   
   # Date formatting
   "{{ execution.timestamp | strftime('%Y-%m-%d') }}"
   
   # Math operations
   "{{ inputs.count * 2 }}"
   "{{ inputs.value | round(2) }}"
   
   # Conditionals
   "{{ 'premium' if inputs.tier == 'gold' else 'standard' }}"
   
   # Lists and loops
   "{{ inputs.items | join(', ') }}"
   "{{ inputs.sources | length }}"

AUTO Tag Usage
==============

AUTO tags delegate decisions to AI models:

**Basic AUTO Tags**:

.. code-block:: yaml

   parameters:
     # Simple decision
     style: <AUTO>Choose appropriate writing style</AUTO>
     
     # Context-aware decision
     method: <AUTO>Based on the data type {{ results.fetch.type }}, choose the best analysis method</AUTO>
     
     # Multiple choices
     options: 
       visualization: <AUTO>Should we create visualizations for this data?</AUTO>
       format: <AUTO>What's the best output format: json, csv, or parquet?</AUTO>

**Advanced AUTO Patterns**:

.. code-block:: yaml

   # Conditional AUTO
   analysis_depth: |
     <AUTO>
     Given:
     - Data size: {{ results.fetch.size }}
     - Time constraint: {{ inputs.deadline }}
     - Importance: {{ inputs.priority }}
     
     Determine the appropriate analysis depth (1-10)
     </AUTO>
   
   # Structured AUTO
   report_sections: |
     <AUTO>
     For a report about {{ inputs.topic }}, determine which sections to include:
     - Executive Summary: yes/no
     - Technical Details: yes/no
     - Future Outlook: yes/no
     - Recommendations: yes/no
     Return as JSON object
     </AUTO>

Pipeline Compilation Process
============================

Understanding how pipelines are compiled helps you write better definitions:

**Compilation Stages**:

1. **Parsing**: YAML is parsed and validated against schema
2. **Template Resolution**: Compile-time templates are resolved
3. **Dependency Analysis**: Task dependencies are analyzed
4. **Tool Detection**: Required tools are identified
5. **Model Selection**: Appropriate models are chosen
6. **Optimization**: Pipeline is optimized for execution

**User Control Points**:

.. code-block:: python

   import orchestrator as orc
   
   # Control compilation options
   pipeline = orc.compile(
       "pipeline.yaml",
       # Override config values
       config={
           "timeout": 7200,
           "checkpoint": True
       },
       # Set compilation flags
       strict=True,           # Strict validation
       optimize=True,         # Enable optimizations
       dry_run=False,         # Actually compile (not just validate)
       debug=True            # Include debug information
   )
   
   # Inspect compilation result
   print(pipeline.get_required_tools())
   print(pipeline.get_task_graph())
   print(pipeline.get_estimated_cost())

**Runtime vs Compile-Time Resolution**:

.. code-block:: yaml

   # Compile-time (resolved during compilation)
   config:
     timestamp: "{{ compile_time.timestamp }}"
   
   # Runtime (resolved during execution)
   steps:
     - id: dynamic
       parameters:
         query: "{{ inputs.topic }}"  # Runtime
         results: "$results.previous"  # Runtime

Advanced Pipeline Features
==========================

Pipeline Imports
----------------

Reuse common patterns:

.. code-block:: yaml

   imports:
     # Import specific steps
     - common/data_validation.yaml#validate_step as validate
     
     # Import entire pipeline
     - workflows/standard_analysis.yaml as analysis
   
   steps:
     # Use imported step
     - id: validation
       extends: validate
       parameters:
         data: "$results.fetch"
     
     # Use imported pipeline
     - id: analyze
       pipeline: analysis
       inputs:
         data: "$results.validation"

Parallel Execution Groups
-------------------------

.. code-block:: yaml

   steps:
     # Define parallel group
     - id: parallel_fetch
       parallel:
         - id: fetch_api
           action: fetch_url
           parameters:
             url: "{{ inputs.api_url }}"
         
         - id: fetch_db
           action: query_database
           parameters:
             query: "{{ inputs.db_query }}"
         
         - id: fetch_file
           action: read_file
           parameters:
             path: "{{ inputs.file_path }}"
     
     # Use results from parallel group
     - id: merge
       action: combine_data
       depends_on: [parallel_fetch]
       parameters:
         sources:
           - "$results.parallel_fetch.fetch_api"
           - "$results.parallel_fetch.fetch_db"
           - "$results.parallel_fetch.fetch_file"

Loops and Iteration
-------------------

.. code-block:: yaml

   steps:
     # For-each loop
     - id: process_items
       for_each: "{{ inputs.items }}"
       as: item
       action: process_single_item
       parameters:
         data: "{{ item }}"
         index: "{{ loop.index }}"
     
     # While loop
     - id: iterative_refinement
       while: "{{ results.quality_check.score < 0.95 }}"
       max_iterations: 10
       action: refine_result
       parameters:
         current: "$results.previous_iteration"

State Management
----------------

.. code-block:: yaml

   # Enable checkpointing
   config:
     checkpoint:
       enabled: true
       frequency: "after_each_step"  # or: "every_n_steps: 5"
       storage: "postgresql"         # or: "redis", "filesystem"
   
   steps:
     - id: long_running
       action: expensive_computation
       checkpoint: true  # Force checkpoint after this step
       recovery:
         strategy: "retry"  # or: "skip", "use_cached"
         max_attempts: 3

Best Practices
==============

1. **Naming Conventions**:
   - Use descriptive IDs: ``fetch_user_data`` not ``step1``
   - Use snake_case for IDs and parameters
   - Use kebab-case for pipeline names

2. **Input Validation**:
   - Always define input types and descriptions
   - Use validation rules to catch errors early
   - Provide sensible defaults

3. **Error Handling**:
   - Add retry logic for external calls
   - Define fallbacks for critical steps
   - Use conditional execution for optional features

4. **Performance**:
   - Enable parallel execution where possible
   - Use caching for expensive operations
   - Set appropriate timeouts

5. **Maintainability**:
   - Use imports for common patterns
   - Document complex logic
   - Version your pipelines

Common Patterns
===============

Data Processing Pipeline
------------------------

.. code-block:: yaml

   name: data-processing-pipeline
   description: ETL pipeline with validation
   
   inputs:
     source_url:
       type: string
       required: true
     
     output_format:
       type: string
       default: "parquet"
       validation:
         enum: ["csv", "json", "parquet"]
   
   steps:
     # Extract
     - id: extract
       action: fetch_data
       parameters:
         url: "{{ inputs.source_url }}"
         format: <AUTO>Detect format from URL</AUTO>
     
     # Transform
     - id: clean
       action: clean_data
       parameters:
         data: "$results.extract"
         rules:
           - remove_duplicates: true
           - handle_missing: "interpolate"
           - standardize_dates: true
     
     - id: transform
       action: transform_data
       parameters:
         data: "$results.clean"
         operations:
           - type: "aggregate"
             group_by: ["category"]
             metrics: ["sum", "avg"]
     
     # Load
     - id: validate
       action: validate_data
       parameters:
         data: "$results.transform"
         schema:
           type: "dataframe"
           columns:
             - name: "category"
               type: "string"
             - name: "total"
               type: "float"
     
     - id: save
       action: save_data
       parameters:
         data: "$results.validate"
         path: "output/processed_data.{{ inputs.output_format }}"
         format: "{{ inputs.output_format }}"

Multi-Source Research Pipeline
------------------------------

.. code-block:: yaml

   name: comprehensive-research
   description: Research from multiple sources with cross-validation
   
   inputs:
     topic:
       type: string
       required: true
     
     sources:
       type: array
       default: ["web", "academic", "news"]
   
   steps:
     # Parallel source fetching
     - id: fetch_sources
       parallel:
         - id: web_search
           condition: "'web' in inputs.sources"
           action: search_web
           parameters:
             query: "{{ inputs.topic }}"
             max_results: 20
         
         - id: academic_search
           condition: "'academic' in inputs.sources"
           action: search_academic
           parameters:
             query: "{{ inputs.topic }}"
             databases: ["arxiv", "pubmed", "scholar"]
         
         - id: news_search
           condition: "'news' in inputs.sources"
           action: search_news
           parameters:
             query: "{{ inputs.topic }}"
             date_range: "last_30_days"
     
     # Process and validate
     - id: extract_facts
       action: extract_information
       parameters:
         sources: "$results.fetch_sources"
         extract:
           - facts
           - claims
           - statistics
     
     - id: cross_validate
       action: validate_claims
       parameters:
         claims: "$results.extract_facts.claims"
         require_sources: 2  # Need 2+ sources to confirm
     
     # Generate report
     - id: synthesize
       action: generate_synthesis
       parameters:
         validated_facts: "$results.cross_validate"
         style: "analytical"
         include_confidence: true