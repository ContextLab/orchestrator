==============
Tool Reference
==============

This comprehensive guide covers all available tools in Orchestrator, how to use them, and provides examples for each.

Tool System Overview
====================

Orchestrator's tool system provides real-world capabilities to your pipelines. Tools are automatically detected from your pipeline definitions and made available through the Model Context Protocol (MCP).

Tool Categories
---------------

1. **Web Tools**: Internet interaction and data gathering
2. **System Tools**: File and command execution
3. **Data Tools**: Processing and validation
4. **AI Tools**: Model-specific capabilities
5. **Integration Tools**: Third-party service connections

How Tools Work
--------------

1. **Automatic Detection**: Tools are identified from action names
2. **MCP Registration**: Tools are exposed via MCP server
3. **Parameter Validation**: Input parameters are validated
4. **Execution**: Tools perform real-world actions
5. **Result Handling**: Results are returned to the pipeline

Web Tools
=========

HeadlessBrowserTool
-------------------

**Purpose**: Interact with web pages using a headless browser

**Action Names**:
- ``search_web``
- ``scrape_page``
- ``screenshot_page``
- ``interact_with_page``

**Parameters**:

.. code-block:: yaml

   # Web search
   - id: search
     action: search_web
     parameters:
       query: "orchestrator framework tutorial"    # Required: Search query
       max_results: 10                            # Optional: Number of results (default: 10)
       search_engine: "google"                    # Optional: google|bing|duckduckgo (default: google)
       include_snippets: true                     # Optional: Include text snippets (default: true)
       region: "us"                              # Optional: Region code (default: us)
       language: "en"                            # Optional: Language code (default: en)
       safe_search: "moderate"                   # Optional: off|moderate|strict (default: moderate)
   
   # Scrape webpage
   - id: scrape
     action: scrape_page
     parameters:
       url: "https://example.com/article"        # Required: URL to scrape
       selectors:                                # Optional: CSS selectors to extract
         title: "h1.main-title"
         content: "div.article-body"
         author: "span.author-name"
       wait_for: "div.content-loaded"            # Optional: Wait for element
       timeout: 30                               # Optional: Timeout in seconds (default: 30)
       javascript: true                          # Optional: Execute JavaScript (default: true)
       clean_html: true                          # Optional: Clean extracted HTML (default: true)
   
   # Take screenshot
   - id: screenshot
     action: screenshot_page
     parameters:
       url: "https://example.com"                # Required: URL to screenshot
       full_page: true                           # Optional: Capture full page (default: false)
       width: 1920                               # Optional: Viewport width (default: 1920)
       height: 1080                              # Optional: Viewport height (default: 1080)
       wait_for: "img"                           # Optional: Wait for element
       output_path: "screenshots/page.png"       # Optional: Save path
   
   # Interact with page
   - id: interact
     action: interact_with_page
     parameters:
       url: "https://example.com/form"           # Required: URL to interact with
       actions:                                  # Required: List of interactions
         - type: "fill"
           selector: "#username"
           value: "testuser"
         - type: "click"
           selector: "#submit-button"
         - type: "wait"
           duration: 2000
         - type: "extract"
           selector: ".result"

**Example Pipeline**:

.. code-block:: yaml

   name: web-research-pipeline
   description: Comprehensive web research with validation
   
   steps:
     # Search for information
     - id: search_topic
       action: search_web
       parameters:
         query: "{{ inputs.topic }} latest news 2024"
         max_results: 20
         search_engine: "google"
     
     # Scrape top results
     - id: scrape_articles
       for_each: "{{ results.search_topic.results[:5] }}"
       as: result
       action: scrape_page
       parameters:
         url: "{{ result.url }}"
         selectors:
           title: "h1, h2.article-title"
           content: "main, article, div.content"
           date: "time, .date, .published"
         clean_html: true
     
     # Take screenshots for reference
     - id: capture_pages
       for_each: "{{ results.search_topic.results[:3] }}"
       as: result
       action: screenshot_page
       parameters:
         url: "{{ result.url }}"
         output_path: "research/{{ inputs.topic }}/{{ loop.index }}.png"

WebSearchTool
-------------

**Purpose**: Simplified web search without browser overhead

**Action Names**:
- ``quick_search``
- ``search_news``
- ``search_academic``
- ``search_images``

**Parameters**:

.. code-block:: yaml

   # Quick search
   - id: search
     action: quick_search
     parameters:
       query: "machine learning basics"          # Required: Search query
       max_results: 5                           # Optional: Result count (default: 10)
       format: "json"                           # Optional: json|text (default: json)
   
   # News search
   - id: news
     action: search_news
     parameters:
       query: "AI breakthroughs"                # Required: Search query
       date_range: "last_week"                  # Optional: last_day|last_week|last_month|last_year
       sources: ["reuters", "techcrunch"]       # Optional: Preferred sources
       sort_by: "relevance"                     # Optional: relevance|date (default: relevance)
   
   # Academic search
   - id: academic
     action: search_academic
     parameters:
       query: "quantum computing"               # Required: Search query
       databases: ["arxiv", "pubmed"]          # Optional: Databases to search
       year_range: "2020-2024"                 # Optional: Year range
       peer_reviewed: true                      # Optional: Only peer-reviewed (default: false)
   
   # Image search
   - id: images
     action: search_images
     parameters:
       query: "data visualization examples"     # Required: Search query
       max_results: 10                         # Optional: Number of images
       size: "large"                           # Optional: small|medium|large|any
       type: "photo"                           # Optional: photo|clipart|lineart|any
       license: "creative_commons"             # Optional: License filter

System Tools
============

TerminalTool
------------

**Purpose**: Execute shell commands in a sandboxed environment

**Action Names**:
- Commands starting with ``!`` (e.g., ``!ls``, ``!python script.py``)
- ``execute_command``
- ``run_script``

**Parameters**:

.. code-block:: yaml

   # Direct command execution
   - id: list_files
     action: "!ls -la /data"
   
   # Command with parameters
   - id: run_command
     action: execute_command
     parameters:
       command: "python analyze.py"              # Required: Command to execute
       arguments: ["--input", "data.csv"]       # Optional: Command arguments
       working_dir: "/project"                  # Optional: Working directory
       environment:                             # Optional: Environment variables
         PYTHONPATH: "/project/lib"
         DEBUG: "true"
       timeout: 300                             # Optional: Timeout in seconds (default: 60)
       capture_output: true                     # Optional: Capture stdout/stderr (default: true)
       shell: true                              # Optional: Use shell execution (default: true)
   
   # Run script file
   - id: run_analysis
     action: run_script
     parameters:
       script_path: "scripts/analyze.sh"        # Required: Path to script
       arguments: ["{{ inputs.data_file }}"]    # Optional: Script arguments
       interpreter: "bash"                      # Optional: bash|python|node (default: auto-detect)
       working_dir: "{{ execution.temp_dir }}"  # Optional: Working directory

**Example Pipeline**:

.. code-block:: yaml

   name: data-processing-automation
   description: Automated data processing with shell commands
   
   steps:
     # Setup environment
     - id: setup
       action: "!mkdir -p output/{{ inputs.project_name }}"
     
     # Download data
     - id: download
       action: execute_command
       parameters:
         command: "wget"
         arguments: 
           - "-O"
           - "data/raw_data.csv"
           - "{{ inputs.data_url }}"
         timeout: 600
     
     # Process with Python
     - id: process
       action: execute_command
       parameters:
         command: "python"
         arguments:
           - "scripts/process_data.py"
           - "--input"
           - "data/raw_data.csv"
           - "--output"
           - "output/{{ inputs.project_name }}/processed.csv"
         environment:
           DATA_QUALITY: "high"
           PROCESSING_MODE: "{{ inputs.mode }}"
     
     # Generate report with R
     - id: report
       action: "!Rscript reports/generate_report.R output/{{ inputs.project_name }}/processed.csv"
     
     # Package results
     - id: package
       action: execute_command
       parameters:
         command: "tar"
         arguments: ["-czf", "{{ outputs.package }}", "output/{{ inputs.project_name }}"]

FileSystemTool
--------------

**Purpose**: Perform file and directory operations

**Action Names**:
- ``read_file``
- ``write_file``
- ``copy_file``
- ``move_file``
- ``delete_file``
- ``list_directory``
- ``create_directory``
- ``file_exists``

**Parameters**:

.. code-block:: yaml

   # Read file
   - id: read_config
     action: read_file
     parameters:
       path: "config/settings.json"             # Required: File path
       encoding: "utf-8"                        # Optional: File encoding (default: utf-8)
       parse: true                              # Optional: Parse JSON/YAML (default: false)
   
   # Write file
   - id: save_results
     action: write_file
     parameters:
       path: "output/results.json"              # Required: File path
       content: "{{ results.analysis | json }}" # Required: Content to write
       encoding: "utf-8"                        # Optional: File encoding (default: utf-8)
       create_dirs: true                        # Optional: Create parent dirs (default: true)
       overwrite: true                          # Optional: Overwrite existing (default: false)
   
   # Copy file
   - id: backup
     action: copy_file
     parameters:
       source: "data/important.db"              # Required: Source path
       destination: "backup/important_{{ execution.timestamp }}.db"  # Required: Destination
       overwrite: false                         # Optional: Overwrite existing (default: false)
   
   # Move file
   - id: archive
     action: move_file
     parameters:
       source: "temp/processed.csv"             # Required: Source path
       destination: "archive/2024/processed.csv" # Required: Destination
       create_dirs: true                        # Optional: Create parent dirs (default: true)
   
   # Delete file
   - id: cleanup
     action: delete_file
     parameters:
       path: "temp/*"                           # Required: Path or pattern
       recursive: true                          # Optional: Delete recursively (default: false)
       force: false                             # Optional: Force deletion (default: false)
   
   # List directory
   - id: scan_files
     action: list_directory
     parameters:
       path: "data/"                            # Required: Directory path
       pattern: "*.csv"                         # Optional: File pattern
       recursive: true                          # Optional: Search subdirs (default: false)
       include_hidden: false                    # Optional: Include hidden files (default: false)
       details: true                            # Optional: Include file details (default: false)
   
   # Create directory
   - id: setup_dirs
     action: create_directory
     parameters:
       path: "output/{{ inputs.project }}/data" # Required: Directory path
       parents: true                            # Optional: Create parents (default: true)
       exist_ok: true                           # Optional: Ok if exists (default: true)
   
   # Check existence
   - id: check_file
     action: file_exists
     parameters:
       path: "config/custom.yaml"               # Required: Path to check

**Example Pipeline**:

.. code-block:: yaml

   name: file-organization-pipeline
   description: Organize and process files automatically
   
   steps:
     # Check for existing data
     - id: check_existing
       action: file_exists
       parameters:
         path: "data/current_dataset.csv"
     
     # Backup if exists
     - id: backup
       condition: "{{ results.check_existing }}"
       action: copy_file
       parameters:
         source: "data/current_dataset.csv"
         destination: "backups/dataset_{{ execution.timestamp }}.csv"
     
     # Read configuration
     - id: read_config
       action: read_file
       parameters:
         path: "config/processing.yaml"
         parse: true
     
     # Process files based on config
     - id: process_files
       for_each: "{{ results.read_config.file_patterns }}"
       as: pattern
       action: list_directory
       parameters:
         path: "{{ pattern.directory }}"
         pattern: "{{ pattern.glob }}"
         recursive: true
     
     # Organize by type
     - id: organize
       for_each: "{{ results.process_files | flatten }}"
       as: file
       action: move_file
       parameters:
         source: "{{ file.path }}"
         destination: "organized/{{ file.extension }}/{{ file.name }}"
         create_dirs: true

Data Tools
==========

DataProcessingTool
------------------

**Purpose**: Transform and manipulate data in various formats

**Action Names**:
- ``transform_data``
- ``filter_data``
- ``aggregate_data``
- ``merge_data``
- ``convert_format``

**Parameters**:

.. code-block:: yaml

   # Transform data
   - id: transform
     action: transform_data
     parameters:
       data: "$results.load_data"               # Required: Input data or path
       operations:                              # Required: List of operations
         - type: "rename_columns"
           mapping:
             old_name: "new_name"
             price: "cost"
         - type: "add_column"
           name: "total"
           expression: "quantity * cost"
         - type: "drop_columns"
           columns: ["unnecessary_field"]
         - type: "convert_types"
           conversions:
             date: "datetime"
             amount: "float"
   
   # Filter data
   - id: filter
     action: filter_data
     parameters:
       data: "$results.transform"               # Required: Input data
       conditions:                              # Required: Filter conditions
         - field: "status"
           operator: "equals"                   # equals|not_equals|contains|gt|lt|gte|lte
           value: "active"
         - field: "amount"
           operator: "gt"
           value: 1000
       mode: "and"                              # Optional: and|or (default: and)
       
   # Aggregate data
   - id: aggregate
     action: aggregate_data
     parameters:
       data: "$results.filter"                  # Required: Input data
       group_by: ["category", "region"]        # Optional: Grouping columns
       aggregations:                            # Required: Aggregation rules
         total_amount:
           column: "amount"
           function: "sum"                      # sum|mean|median|min|max|count|std
         average_price:
           column: "price"
           function: "mean"
         item_count:
           column: "*"
           function: "count"
   
   # Merge data
   - id: merge
     action: merge_data
     parameters:
       left: "$results.main_data"               # Required: Left dataset
       right: "$results.lookup_data"            # Required: Right dataset
       on: "customer_id"                        # Required: Join column(s)
       how: "left"                              # Optional: left|right|inner|outer (default: left)
       suffixes: ["_main", "_lookup"]          # Optional: Column suffixes
   
   # Convert format
   - id: convert
     action: convert_format
     parameters:
       data: "$results.final_data"              # Required: Input data
       from_format: "json"                      # Optional: Auto-detect if not specified
       to_format: "parquet"                     # Required: Target format
       options:                                 # Optional: Format-specific options
         compression: "snappy"
         index: false

**Example Pipeline**:

.. code-block:: yaml

   name: sales-data-analysis
   description: Process and analyze sales data
   
   steps:
     # Load raw data
     - id: load_sales
       action: read_file
       parameters:
         path: "data/sales_2024.csv"
         parse: true
     
     # Clean and transform
     - id: clean_data
       action: transform_data
       parameters:
         data: "$results.load_sales"
         operations:
           - type: "rename_columns"
             mapping:
               "Sale Date": "sale_date"
               "Customer Name": "customer_name"
               "Product ID": "product_id"
               "Sale Amount": "amount"
           - type: "convert_types"
             conversions:
               sale_date: "datetime"
               amount: "float"
           - type: "add_column"
             name: "quarter"
             expression: "sale_date.quarter"
     
     # Filter valid sales
     - id: filter_valid
       action: filter_data
       parameters:
         data: "$results.clean_data"
         conditions:
           - field: "amount"
             operator: "gt"
             value: 0
           - field: "product_id"
             operator: "not_equals"
             value: null
     
     # Aggregate by quarter
     - id: quarterly_summary
       action: aggregate_data
       parameters:
         data: "$results.filter_valid"
         group_by: ["quarter", "product_id"]
         aggregations:
           total_sales:
             column: "amount"
             function: "sum"
           avg_sale:
             column: "amount"
             function: "mean"
           num_transactions:
             column: "*"
             function: "count"
     
     # Save results
     - id: save_summary
       action: convert_format
       parameters:
         data: "$results.quarterly_summary"
         to_format: "excel"
         options:
           sheet_name: "Quarterly Sales"
           index: false

ValidationTool
--------------

**Purpose**: Validate data against schemas and business rules

**Action Names**:
- ``validate_data``
- ``validate_schema``
- ``check_quality``
- ``validate_report``

**Parameters**:

.. code-block:: yaml

   # Validate against schema
   - id: validate_structure
     action: validate_schema
     parameters:
       data: "$results.processed_data"          # Required: Data to validate
       schema:                                  # Required: Validation schema
         type: "object"
         required: ["id", "name", "email"]
         properties:
           id:
             type: "integer"
             minimum: 1
           name:
             type: "string"
             minLength: 2
             maxLength: 100
           email:
             type: "string"
             format: "email"
           age:
             type: "integer"
             minimum: 0
             maximum: 150
       strict: false                            # Optional: Strict mode (default: false)
   
   # Business rule validation
   - id: validate_rules
     action: validate_data
     parameters:
       data: "$results.transactions"            # Required: Data to validate
       rules:                                   # Required: Validation rules
         - name: "positive_amounts"
           field: "amount"
           condition: "value > 0"
           severity: "error"                    # error|warning|info
           message: "Transaction amounts must be positive"
         
         - name: "valid_date_range"
           field: "transaction_date"
           condition: "value >= '2024-01-01' and value <= today()"
           severity: "error"
         
         - name: "customer_exists"
           field: "customer_id"
           condition: "value in valid_customers"
           severity: "warning"
           context:
             valid_customers: "$results.customer_list"
       
       stop_on_error: false                     # Optional: Stop on first error (default: false)
   
   # Data quality checks
   - id: quality_check
     action: check_quality
     parameters:
       data: "$results.dataset"                 # Required: Data to check
       checks:                                  # Required: Quality checks
         - type: "completeness"
           threshold: 0.95                      # 95% non-null required
           columns: ["id", "name", "email"]
         
         - type: "uniqueness"
           columns: ["id", "email"]
         
         - type: "consistency"
           rules:
             - "start_date <= end_date"
             - "total == sum(line_items)"
         
         - type: "accuracy"
           validations:
             email: "regex:^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
             phone: "regex:^\\+?1?\\d{9,15}$"
         
         - type: "timeliness"
           field: "last_updated"
           max_age_days: 30
   
   # Report validation
   - id: validate_report
     action: validate_report
     parameters:
       report: "$results.generated_report"      # Required: Report to validate
       checks:                                  # Required: Report checks
         - "completeness"                       # All sections present
         - "accuracy"                           # Facts are accurate
         - "consistency"                        # No contradictions
         - "readability"                        # Appropriate reading level
         - "citations"                          # Sources properly cited
       requirements:                            # Optional: Specific requirements
         min_words: 1000
         max_words: 5000
         required_sections: ["intro", "analysis", "conclusion"]
         citation_style: "APA"

**Example Pipeline**:

.. code-block:: yaml

   name: data-quality-pipeline
   description: Comprehensive data validation and quality assurance
   
   steps:
     # Load data
     - id: load
       action: read_file
       parameters:
         path: "{{ inputs.data_file }}"
         parse: true
     
     # Schema validation
     - id: validate_schema
       action: validate_schema
       parameters:
         data: "$results.load"
         schema:
           type: "array"
           items:
             type: "object"
             required: ["order_id", "customer_id", "amount", "date"]
             properties:
               order_id:
                 type: "string"
                 pattern: "^ORD-[0-9]{6}$"
               customer_id:
                 type: "integer"
                 minimum: 1
               amount:
                 type: "number"
                 minimum: 0
               date:
                 type: "string"
                 format: "date"
     
     # Business rules
     - id: validate_business
       action: validate_data
       parameters:
         data: "$results.load"
         rules:
           - name: "valid_amounts"
             field: "amount"
             condition: "value > 0 and value < 10000"
             severity: "error"
           
           - name: "recent_orders"
             field: "date"
             condition: "days_between(value, today()) <= 365"
             severity: "warning"
             message: "Order is older than 1 year"
     
     # Quality assessment
     - id: quality_report
       action: check_quality
       parameters:
         data: "$results.load"
         checks:
           - type: "completeness"
             threshold: 0.98
           - type: "uniqueness"
             columns: ["order_id"]
           - type: "consistency"
             rules:
               - "item_total == quantity * unit_price"
           - type: "accuracy"
             validations:
               email: "regex:^[\\w.-]+@[\\w.-]+\\.\\w+$"
     
     # Generate validation report
     - id: create_report
       action: generate_content
       parameters:
         template: |
           # Data Validation Report
           
           ## Schema Validation
           {{ results.validate_schema.summary }}
           
           ## Business Rules
           {{ results.validate_business.summary }}
           
           ## Quality Metrics
           {{ results.quality_report | format_quality_metrics }}
           
           ## Recommendations
           <AUTO>Based on the validation results, provide recommendations</AUTO>

AI Tools
========

Model-Specific Tools
--------------------

**Purpose**: Leverage specific AI model capabilities

**Action Names**:
- ``generate_content``
- ``analyze_text``
- ``extract_information``
- ``generate_code``
- ``reason_about``

**Parameters**:

.. code-block:: yaml

   # Generate content
   - id: generate
     action: generate_content
     parameters:
       prompt: "{{ inputs.prompt }}"            # Required: Generation prompt
       model: <AUTO>Select best model</AUTO>    # Optional: Model selection
       max_tokens: 1000                         # Optional: Maximum tokens
       temperature: 0.7                         # Optional: Creativity (0-2)
       system_prompt: "You are a helpful AI"    # Optional: System message
       format: "markdown"                       # Optional: Output format
       style: "professional"                    # Optional: Writing style
   
   # Analyze text
   - id: analyze
     action: analyze_text
     parameters:
       text: "$results.document"                # Required: Text to analyze
       analysis_types:                          # Required: Types of analysis
         - sentiment                            # Positive/negative/neutral
         - entities                             # Named entities
         - topics                               # Main topics
         - summary                              # Brief summary
         - key_points                           # Bullet points
         - language                             # Detect language
       output_format: "structured"              # Optional: structured|narrative
   
   # Extract information
   - id: extract
     action: extract_information
     parameters:
       content: "$results.raw_text"             # Required: Source content
       extract:                                 # Required: What to extract
         dates:
           description: "All mentioned dates"
           format: "YYYY-MM-DD"
         people:
           description: "Person names with roles"
           include_context: true
         organizations:
           description: "Company and organization names"
         numbers:
           description: "Numerical values with units"
           categories: ["financial", "metrics"]
       output_format: "json"                    # Optional: json|table|text
   
   # Generate code
   - id: code_gen
     action: generate_code
     parameters:
       description: "{{ inputs.feature_request }}" # Required: What to build
       language: "python"                       # Required: Programming language
       framework: "fastapi"                     # Optional: Framework/library
       include_tests: true                      # Optional: Generate tests
       include_docs: true                       # Optional: Generate docs
       style_guide: "PEP8"                     # Optional: Code style
       example_usage: true                      # Optional: Include examples
   
   # Reasoning task
   - id: reason
     action: reason_about
     parameters:
       question: "{{ inputs.problem }}"         # Required: Problem/question
       context: "$results.research"             # Optional: Additional context
       approach: "step_by_step"                 # Optional: Reasoning approach
       show_work: true                          # Optional: Show reasoning
       confidence_level: true                   # Optional: Include confidence

Integration Tools
=================

DatabaseTool
------------

**Purpose**: Interact with databases

**Action Names**:
- ``query_database``
- ``insert_data``
- ``update_data``
- ``delete_data``

**Parameters**:

.. code-block:: yaml

   # Query database
   - id: fetch_data
     action: query_database
     parameters:
       connection: "postgresql://localhost/mydb" # Required: Connection string
       query: "SELECT * FROM users WHERE active = true" # Required: SQL query
       parameters: []                           # Optional: Query parameters
       fetch_size: 1000                         # Optional: Batch size
       timeout: 30                              # Optional: Query timeout

APITool
-------

**Purpose**: Make HTTP API calls

**Action Names**:
- ``call_api``
- ``rest_request``
- ``graphql_query``

**Parameters**:

.. code-block:: yaml

   # REST API call
   - id: api_call
     action: call_api
     parameters:
       url: "https://api.example.com/data"     # Required: API endpoint
       method: "POST"                           # Required: HTTP method
       headers:                                 # Optional: Headers
         Authorization: "Bearer {{ env.API_TOKEN }}"
         Content-Type: "application/json"
       body:                                    # Optional: Request body
         query: "{{ inputs.search_term }}"
         limit: 100
       timeout: 60                              # Optional: Request timeout
       retry: 3                                 # Optional: Retry attempts

Tool Chaining Examples
======================

Research and Report Pipeline
----------------------------

.. code-block:: yaml

   name: comprehensive-research-tool-chain
   description: Chain multiple tools for research and reporting
   
   steps:
     # 1. Search multiple sources
     - id: web_search
       action: search_web
       parameters:
         query: "{{ inputs.topic }} latest research 2024"
         max_results: 20
     
     # 2. Scrape promising articles
     - id: scrape_articles
       for_each: "{{ results.web_search.results[:5] }}"
       as: article
       action: scrape_page
       parameters:
         url: "{{ article.url }}"
         selectors:
           content: "article, main, .content"
     
     # 3. Extract key information
     - id: extract_facts
       action: extract_information
       parameters:
         content: "$results.scrape_articles"
         extract:
           facts:
             description: "Key facts and findings"
           statistics:
             description: "Numerical data with context"
           quotes:
             description: "Notable quotes with attribution"
     
     # 4. Validate information
     - id: cross_validate
       action: validate_data
       parameters:
         data: "$results.extract_facts"
         rules:
           - name: "source_diversity"
             condition: "count(unique(sources)) >= 3"
             severity: "warning"
     
     # 5. Generate report
     - id: create_report
       action: generate_content
       parameters:
         prompt: |
           Create a comprehensive report about {{ inputs.topic }}
           using the following validated information:
           {{ results.extract_facts | json }}
         style: "academic"
         format: "markdown"
         max_tokens: 2000
     
     # 6. Save report
     - id: save_report
       action: write_file
       parameters:
         path: "reports/{{ inputs.topic }}_{{ execution.date }}.md"
         content: "$results.create_report"
     
     # 7. Generate PDF
     - id: create_pdf
       action: "!pandoc -f markdown -t pdf -o reports/{{ inputs.topic }}.pdf reports/{{ inputs.topic }}_{{ execution.date }}.md"

Data Processing Pipeline
------------------------

.. code-block:: yaml

   name: etl-tool-chain
   description: Extract, transform, and load data using tool chain
   
   steps:
     # Extract from multiple sources
     - id: extract_database
       action: query_database
       parameters:
         connection: "{{ env.DB_CONNECTION }}"
         query: "SELECT * FROM sales WHERE date >= '2024-01-01'"
     
     - id: extract_api
       action: call_api
       parameters:
         url: "https://api.company.com/v2/transactions"
         method: "GET"
         headers:
           Authorization: "Bearer {{ env.API_KEY }}"
         params:
           start_date: "2024-01-01"
           page_size: 1000
     
     - id: extract_files
       action: list_directory
       parameters:
         path: "data/uploads/"
         pattern: "sales_*.csv"
         recursive: true
     
     # Load file data
     - id: load_files
       for_each: "{{ results.extract_files }}"
       as: file
       action: read_file
       parameters:
         path: "{{ file.path }}"
         parse: true
     
     # Transform all data
     - id: merge_all
       action: merge_data
       parameters:
         datasets:
           - "$results.extract_database"
           - "$results.extract_api.data"
           - "$results.load_files"
         key: "transaction_id"
     
     - id: clean_data
       action: transform_data
       parameters:
         data: "$results.merge_all"
         operations:
           - type: "remove_duplicates"
             columns: ["transaction_id"]
           - type: "fill_missing"
             strategy: "forward"
           - type: "standardize_formats"
             columns:
               date: "YYYY-MM-DD"
               amount: "decimal(10,2)"
     
     # Validate
     - id: validate_quality
       action: check_quality
       parameters:
         data: "$results.clean_data"
         checks:
           - type: "completeness"
             threshold: 0.99
           - type: "accuracy"
             validations:
               amount: "range:0,1000000"
               date: "date_range:2024-01-01,today"
     
     # Load to destination
     - id: save_processed
       action: write_file
       parameters:
         path: "processed/sales_cleaned_{{ execution.date }}.parquet"
         content: "$results.clean_data"
         format: "parquet"
     
     - id: update_database
       condition: "{{ results.validate_quality.passed }}"
       action: insert_data
       parameters:
         connection: "{{ env.DW_CONNECTION }}"
         table: "sales_fact"
         data: "$results.clean_data"
         mode: "append"

Tool Development
================

Creating Custom Tools
---------------------

To create your own tools:

.. code-block:: python

   from orchestrator.tools.base import Tool
   
   class MyCustomTool(Tool):
       def __init__(self):
           super().__init__(
               name="my-custom-tool",
               description="Does something special"
           )
           
           # Define parameters
           self.add_parameter(
               name="input_data",
               type="string",
               description="Data to process",
               required=True
           )
           
           self.add_parameter(
               name="mode",
               type="string", 
               description="Processing mode",
               required=False,
               default="standard",
               enum=["standard", "advanced", "expert"]
           )
       
       async def execute(self, **kwargs):
           """Execute the tool action."""
           input_data = kwargs["input_data"]
           mode = kwargs.get("mode", "standard")
           
           # Your tool logic here
           result = process_data(input_data, mode)
           
           return {
               "status": "success",
               "result": result,
               "metadata": {
                   "mode": mode,
                   "timestamp": datetime.now()
               }
           }

Registering Custom Tools
------------------------

Register your tool to make it available:

.. code-block:: python

   from orchestrator.tools.base import default_registry
   
   # Register tool
   tool = MyCustomTool()
   default_registry.register(tool)
   
   # Use in pipeline
   pipeline_yaml = """
   steps:
     - id: custom_step
       action: my-custom-tool
       parameters:
         input_data: "{{ inputs.data }}"
         mode: "advanced"
   """

Best Practices
==============

1. **Tool Selection**:
   - Use the most specific tool for the job
   - Prefer built-in tools over shell commands
   - Chain tools for complex operations

2. **Error Handling**:
   - Always handle tool failures gracefully
   - Provide fallback options
   - Validate tool outputs

3. **Performance**:
   - Batch operations when possible
   - Use appropriate timeouts
   - Cache expensive tool results

4. **Security**:
   - Sanitize inputs to shell commands
   - Use parameterized queries for databases
   - Validate file paths

5. **Monitoring**:
   - Log tool executions
   - Track performance metrics
   - Monitor resource usage