Data Processing Workflow
========================

This example demonstrates how to build a scalable data processing pipeline using the Orchestrator's declarative YAML framework. The pipeline handles data ingestion, validation, transformation, analysis, and export - all defined in pure YAML with no custom Python code required.

.. note::
   **Level:** Advanced  
   **Duration:** 45-60 minutes  
   **Prerequisites:** Orchestrator framework installed, database/cloud credentials configured

Overview
--------

The Data Processing Workflow automates:

1. **Data Discovery**: Automatically find and catalog data sources
2. **Schema Validation**: Validate data structure and types
3. **Quality Profiling**: Analyze data quality and identify issues
4. **Data Cleaning**: Fix quality issues and standardize formats
5. **Data Enrichment**: Add calculated fields and derived metrics
6. **Validation**: Ensure processed data meets requirements
7. **Export**: Save data in optimized formats
8. **Lineage Tracking**: Document data transformations
9. **Monitoring**: Track performance and generate alerts

**Key Features Demonstrated:**
- Declarative YAML pipeline definition
- AUTO tag resolution for natural language task descriptions
- Parallel processing for large datasets
- Automatic data validation and quality checks
- Error recovery with intelligent retry logic
- Real-time monitoring and metrics
- No Python code required

Quick Start
-----------

.. code-block:: bash

   # Set up environment variables
   export OPENAI_API_KEY="your-openai-key"
   export DATABASE_URL="postgresql://user:pass@localhost/db"
   export AWS_ACCESS_KEY_ID="your-aws-key"  # If using S3
   
   # Run the data processing pipeline
   orchestrator run examples/data_processing_workflow.yaml \
     --input source="data/sales/*.csv" \
     --input output_path="processed/sales/" \
     --input output_format="parquet"

Complete YAML Pipeline
----------------------

The complete pipeline is defined in ``examples/data_processing_workflow.yaml``. Here are the key sections:

**Pipeline Structure:**

.. code-block:: yaml

   name: "Data Processing Workflow"
   description: "Scalable data processing pipeline with validation and analysis"

   inputs:
     source:
       type: string
       description: "Data source path or pattern"
       required: true
     
     output_path:
       type: string
       description: "Output path for processed data"
       required: true
     
     output_format:
       type: string
       description: "Output format (parquet, csv, json, database)"
       default: "parquet"
     
     chunk_size:
       type: integer
       description: "Records to process per chunk"
       default: 10000

**Key Processing Steps:**

1. **Discovery and Validation:**

.. code-block:: yaml

   - id: discover_sources
     action: <AUTO>discover all data sources matching pattern {{source}} 
       and gather metadata including schemas and statistics</AUTO>
     
   - id: validate_schema
     action: <AUTO>analyze schema and validate data types, 
       required fields, and relationships</AUTO>
     depends_on: [discover_sources]

2. **Parallel Quality Profiling:**

.. code-block:: yaml

   - id: profile_data
     action: <AUTO>profile data quality including completeness, 
       consistency, and anomalies</AUTO>
     loop:
       foreach: "{{discover_sources.result.sources}}"
       parallel: true
       max_iterations: 10

3. **Data Cleaning with Error Handling:**

.. code-block:: yaml

   - id: clean_data
     action: <AUTO>clean data: handle missing values, standardize formats, 
       remove duplicates, fix inconsistencies</AUTO>
     on_error:
       action: <AUTO>log problematic records and continue</AUTO>
       continue_on_error: true
       retry_count: 2

How It Works
------------

**1. Automatic Tool Discovery**

The framework automatically identifies and uses appropriate tools:

- File operations → ``filesystem`` tool
- Database operations → ``database`` tool
- Cloud storage → ``cloud-storage`` tool
- Data processing → ``data-processing`` tool
- Validation → ``validation`` tool

**2. Intelligent Processing**

The pipeline adapts based on:
- Data source type (files, databases, APIs)
- Data volume (adjusts chunk size and parallelism)
- Quality issues (applies appropriate cleaning strategies)
- Output requirements (optimizes for target format)

**3. Parallel Execution**

Large datasets are processed efficiently:
- Sources profiled in parallel
- Cleaning performed in chunks
- Multiple export streams for partitioned data

Running the Pipeline
--------------------

**File Processing:**

.. code-block:: bash

   # Process CSV files
   orchestrator run data_processing_workflow.yaml \
     --input source="raw_data/*.csv" \
     --input output_path="clean_data/" \
     --input output_format="parquet"

   # Process JSON files with custom chunk size
   orchestrator run data_processing_workflow.yaml \
     --input source="logs/*.json" \
     --input output_path="processed_logs/" \
     --input chunk_size=50000

**Database Processing:**

.. code-block:: bash

   # Process database table
   orchestrator run data_processing_workflow.yaml \
     --input source="postgresql://localhost/mydb/sales_table" \
     --input output_path="s3://bucket/processed/sales/"

   # Process with quality threshold
   orchestrator run data_processing_workflow.yaml \
     --input source="mysql://server/db/customers" \
     --input output_path="refined/customers/" \
     --input quality_threshold=0.95

**Cloud Storage:**

.. code-block:: bash

   # Process S3 data
   orchestrator run data_processing_workflow.yaml \
     --input source="s3://data-lake/raw/events/*.parquet" \
     --input output_path="s3://data-lake/processed/events/"

   # Process with parallel workers
   orchestrator run data_processing_workflow.yaml \
     --input source="gs://bucket/data/*" \
     --input output_path="gs://bucket/clean/" \
     --input parallel_workers=8

Example Output
--------------

**Console Output:**

.. code-block:: text

   📊 Data Processing Workflow
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   ✓ discover_sources: Found 12 data files (3.2 GB total) (2.1s)
   ✓ validate_schema: All schemas validated successfully (1.8s)
   ⟳ profile_data: Profiling 12 sources in parallel...
     ✓ sales_2024_q1.csv: 1.2M records, 94% complete (4.3s)
     ✓ sales_2024_q2.csv: 1.5M records, 96% complete (5.1s)
     ⚠ sales_2024_q3.csv: 0.9M records, 87% complete, 42 issues (3.8s)
   ✓ clean_data: Fixed 156 quality issues (12.4s)
   ✓ enrich_data: Added 8 calculated fields (8.7s)
   ✓ validate_output: All validations passed (2.3s)
   ✓ export_data: Exported to parquet (6.1s)
   ✓ track_lineage: Lineage documented (0.8s)
   ✓ monitor_pipeline: Performance within SLA (0.5s)
   
   ✅ Pipeline completed successfully in 44.1s
   📁 Output: processed/sales/ (2.8 GB)
   📊 Quality Score: 0.98/1.0
   🔧 Issues Fixed: 156/198 (79%)

**Quality Report Example:**

.. code-block:: markdown

   # Data Processing Quality Report
   
   ## Summary
   - **Total Records**: 3,600,000
   - **Processing Time**: 44.1 seconds
   - **Quality Score**: 0.98/1.0
   
   ## Data Quality Metrics
   
   ### Before Processing
   - Completeness: 91.2%
   - Validity: 94.5%
   - Uniqueness: 99.1%
   - Consistency: 88.7%
   
   ### After Processing
   - Completeness: 99.8%
   - Validity: 99.9%
   - Uniqueness: 100%
   - Consistency: 99.7%
   
   ## Issues Resolved
   - Missing values imputed: 89,234
   - Duplicates removed: 1,256
   - Format standardized: 45,123
   - Outliers handled: 567

Advanced Features
-----------------

**1. Custom Validation Rules:**

.. code-block:: yaml

   - id: custom_validation
     action: <AUTO>validate data against business rules:
       - Customer age between 18 and 120
       - Order amount > 0
       - Email format is valid
       - Phone numbers are standardized
       Return violations with severity</AUTO>
     condition: "{{data_type}} == 'customer'"

**2. Machine Learning Integration:**

.. code-block:: yaml

   - id: anomaly_detection
     action: <AUTO>detect anomalies using ML models:
       - Identify unusual patterns
       - Flag potential fraud
       - Detect data drift
       Return anomaly scores and explanations</AUTO>
     condition: "{{enable_ml}} == true"

**3. Incremental Processing:**

.. code-block:: yaml

   - id: detect_changes
     action: <AUTO>identify new or modified records since 
       last processing using {{last_run_timestamp}}</AUTO>
     
   - id: process_incremental
     action: <AUTO>process only changed records while 
       maintaining consistency with existing data</AUTO>
     condition: "{{detect_changes.result.count}} > 0"

Performance Optimization
------------------------

**1. Dynamic Resource Allocation:**

The pipeline automatically adjusts resources based on data volume:
- Small datasets (< 1GB): Single-threaded processing
- Medium datasets (1-10GB): Multi-threaded with 4 workers
- Large datasets (> 10GB): Distributed processing with 8+ workers

**2. Memory Management:**

.. code-block:: yaml

   - id: process_large_file
     action: <AUTO>process file in memory-efficient chunks</AUTO>
     loop:
       foreach: "{{file_chunks}}"
       max_iterations: 1000
     cache_results: false  # Don't cache large intermediate results

**3. Optimization Strategies:**

- Columnar storage for analytical workloads
- Partition by date/category for faster queries
- Compression for storage efficiency
- Indexing for quick lookups

Error Handling
--------------

The pipeline includes comprehensive error handling:

**1. Data Quality Issues:**

.. code-block:: yaml

   on_error:
     action: <AUTO>quarantine bad records and continue processing 
       valid data, generate detailed error report</AUTO>
     continue_on_error: true

**2. Source Unavailability:**

.. code-block:: yaml

   on_error:
     action: <AUTO>try alternative connection methods or 
       wait and retry with exponential backoff</AUTO>
     retry_count: 5
     retry_delay: 10

**3. Export Failures:**

.. code-block:: yaml

   on_error:
     action: <AUTO>switch to alternative format or 
       split into smaller files</AUTO>
     fallback_value: "csv"  # Fallback format

Monitoring and Alerting
-----------------------

Real-time monitoring tracks:

- **Performance Metrics**: Processing speed, memory usage, I/O rates
- **Quality Metrics**: Error rates, validation failures, completeness
- **Business Metrics**: Record counts, value distributions, trends

Alerts are generated for:
- Quality score below threshold
- Processing time exceeding SLA
- Unusual data patterns
- System resource constraints

Integration Examples
--------------------

**1. ETL Pipeline:**

.. code-block:: bash

   # Extract from database, transform, load to data warehouse
   orchestrator run data_processing_workflow.yaml \
     --input source="${SOURCE_DB}" \
     --input output_path="${TARGET_DW}" \
     --input transformations="standardize,aggregate,pivot"

**2. Data Lake Processing:**

.. code-block:: bash

   # Process raw data lake files into curated datasets
   orchestrator run data_processing_workflow.yaml \
     --input source="s3://data-lake/raw/" \
     --input output_path="s3://data-lake/curated/" \
     --input output_format="delta"

**3. Real-time Stream Processing:**

.. code-block:: bash

   # Process streaming data with micro-batches
   orchestrator run data_processing_workflow.yaml \
     --input source="kafka://events-topic" \
     --input output_path="s3://processed-events/" \
     --input mode="streaming" \
     --input batch_interval=60

Key Takeaways
-------------

This example demonstrates the power of Orchestrator's declarative framework:

1. **Zero Code Required**: Complete ETL pipeline in pure YAML
2. **Intelligent Processing**: Automatic optimization based on data characteristics
3. **Robust Error Handling**: Graceful degradation and recovery
4. **Scalable Architecture**: From small files to big data
5. **Production Ready**: Monitoring, alerting, and lineage tracking

The declarative approach makes complex data pipelines maintainable and accessible.

Next Steps
----------

- Try the :doc:`multi_agent_collaboration` for complex AI workflows
- Explore :doc:`code_analysis_suite` for development pipelines
- Read the :doc:`../../advanced/performance_optimization` guide
- Check the :doc:`../../user_guide/data_processing` guide for data-specific features