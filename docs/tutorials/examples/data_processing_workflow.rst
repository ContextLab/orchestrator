Data Processing Workflow
========================

This example demonstrates how to build a scalable data processing pipeline that handles large-scale data ingestion, transformation, validation, and analysis. The workflow showcases parallel processing, error recovery, and intelligent data quality management.

.. note::
   **Level:** Advanced  
   **Duration:** 60-90 minutes  
   **Prerequisites:** Python knowledge, understanding of data processing concepts, familiarity with pandas/numpy

Overview
--------

The Data Processing Workflow automates:

1. **Data Ingestion**: Multi-source data collection (APIs, databases, files)
2. **Data Validation**: Schema validation and quality checks
3. **Data Transformation**: Complex transformations and enrichment
4. **Data Analysis**: Statistical analysis and anomaly detection
5. **ML Processing**: Feature engineering and model predictions
6. **Data Export**: Multi-format export and database loading
7. **Monitoring**: Real-time processing metrics and alerts

**Key Features:**
- Handles structured and unstructured data
- Parallel processing for large datasets
- Automatic error recovery and retry logic
- Data lineage tracking
- Real-time quality monitoring
- Incremental processing support

Quick Start
-----------

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/your-org/orchestrator.git
   cd orchestrator
   
   # Install dependencies
   pip install -r requirements.txt
   pip install pandas numpy pyarrow sqlalchemy
   
   # Set up environment variables
   export OPENAI_API_KEY="your-openai-key"
   export DATABASE_URL="postgresql://user:pass@localhost/db"
   export AWS_ACCESS_KEY_ID="your-aws-key"
   export AWS_SECRET_ACCESS_KEY="your-aws-secret"
   
   # Run the example
   python examples/data_processing_workflow.py \
     --source s3://data-bucket/raw/ \
     --output s3://data-bucket/processed/ \
     --mode batch

Complete Implementation
-----------------------

Pipeline Configuration (YAML)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   # data_processing_pipeline.yaml
   id: data_processing_workflow
   name: Scalable Data Processing Pipeline
   version: "1.0"
   
   metadata:
     description: "Large-scale data processing with validation and ML"
     author: "Data Engineering Team"
     tags: ["etl", "data-processing", "ml", "analytics"]
   
   models:
     data_analyzer:
       provider: "openai"
       model: "gpt-4"
       temperature: 0.1
     anomaly_detector:
       provider: "local"
       model: "isolation_forest"
       path: "models/anomaly_detector.pkl"
   
   context:
     processing_mode: "{{ inputs.mode }}"  # batch or streaming
     chunk_size: 10000
     parallel_workers: 8
     error_threshold: 0.05
   
   tasks:
     - id: discover_sources
       name: "Discover Data Sources"
       action: "scan_data_sources"
       parameters:
         sources: "{{ inputs.sources }}"
         file_patterns: <AUTO>Detect file patterns based on source type</AUTO>
         include_metadata: true
       outputs:
         - source_list
         - total_size
         - file_count
     
     - id: validate_schema
       name: "Schema Validation"
       action: "validate_data_schema"
       parameters:
         sources: "{{ discover_sources.source_list }}"
         schema_path: "{{ inputs.schema_path }}"
         validation_mode: <AUTO>Choose strict or permissive based on data quality</AUTO>
       dependencies:
         - discover_sources
       outputs:
         - valid_sources
         - schema_errors
         - compatibility_report
     
     - id: ingest_data
       name: "Ingest Data"
       action: "ingest_from_sources"
       parallel: true
       max_workers: "{{ context.parallel_workers }}"
       for_each: "{{ validate_schema.valid_sources }}"
       parameters:
         source: "{{ item }}"
         chunk_size: "{{ context.chunk_size }}"
         compression: <AUTO>Detect compression type</AUTO>
         error_handling: "skip_corrupted"
       dependencies:
         - validate_schema
       outputs:
         - raw_data
         - ingestion_stats
     
     - id: clean_data
       name: "Data Cleaning"
       action: "clean_and_standardize"
       parallel: true
       for_each: "{{ ingest_data.raw_data }}"
       parameters:
         data_chunk: "{{ item }}"
         cleaning_rules: <AUTO>Apply appropriate cleaning based on data type</AUTO>
         handle_missing: "intelligent_imputation"
         outlier_detection: true
       dependencies:
         - ingest_data
       outputs:
         - cleaned_data
         - cleaning_report
     
     - id: transform_data
       name: "Data Transformation"
       action: "apply_transformations"
       parallel: true
       for_each: "{{ clean_data.cleaned_data }}"
       parameters:
         data: "{{ item }}"
         transformations: <AUTO>Select transformations based on target schema</AUTO>
         preserve_lineage: true
       dependencies:
         - clean_data
       outputs:
         - transformed_data
         - transformation_log
     
     - id: enrich_data
       name: "Data Enrichment"
       action: "enrich_with_external"
       model: "data_analyzer"
       parameters:
         data: "{{ transform_data.transformed_data }}"
         enrichment_sources: <AUTO>Identify relevant external data sources</AUTO>
         enrichment_fields: ["location", "category", "sentiment"]
       dependencies:
         - transform_data
       outputs:
         - enriched_data
         - enrichment_stats
     
     - id: detect_anomalies
       name: "Anomaly Detection"
       action: "run_anomaly_detection"
       model: "anomaly_detector"
       parameters:
         data: "{{ enrich_data.enriched_data }}"
         sensitivity: <AUTO>Adjust based on data characteristics</AUTO>
         contamination: 0.01
       dependencies:
         - enrich_data
       outputs:
         - anomalies
         - anomaly_scores
         - normal_data
     
     - id: analyze_quality
       name: "Data Quality Analysis"
       action: "analyze_data_quality"
       model: "data_analyzer"
       parameters:
         data: "{{ detect_anomalies.normal_data }}"
         quality_dimensions: ["completeness", "accuracy", "consistency", "timeliness"]
         generate_report: true
       dependencies:
         - detect_anomalies
       outputs:
         - quality_scores
         - quality_report
         - recommendations
     
     - id: feature_engineering
       name: "Feature Engineering"
       action: "engineer_features"
       condition: "inputs.enable_ml == true"
       parameters:
         data: "{{ detect_anomalies.normal_data }}"
         feature_config: <AUTO>Generate features based on data patterns</AUTO>
         target_variable: "{{ inputs.target_variable }}"
       dependencies:
         - detect_anomalies
       outputs:
         - feature_data
         - feature_importance
     
     - id: export_data
       name: "Export Processed Data"
       action: "export_to_targets"
       parallel: true
       parameters:
         data: "{{ feature_engineering.feature_data if inputs.enable_ml else detect_anomalies.normal_data }}"
         targets: "{{ inputs.output_targets }}"
         format: <AUTO>Choose optimal format for each target</AUTO>
         partitioning: "{{ inputs.partition_strategy }}"
       dependencies:
         - feature_engineering
         - detect_anomalies
       outputs:
         - export_locations
         - export_stats

Python Implementation
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # data_processing_workflow.py
   import asyncio
   import pandas as pd
   import numpy as np
   from pathlib import Path
   from typing import Dict, List, Any, Optional, AsyncIterator
   import pyarrow.parquet as pq
   import pyarrow as pa
   from datetime import datetime
   import logging
   
   from orchestrator import Orchestrator
   from orchestrator.tools.data_tools import (
       DataIngestionTool,
       DataValidationTool,
       DataTransformationTool,
       DataQualityTool
   )
   from orchestrator.integrations.storage import S3Storage, DatabaseConnector
   from orchestrator.monitoring import DataPipelineMonitor
   
   
   class DataProcessingWorkflow:
       """
       Scalable data processing workflow with ML capabilities.
       
       Handles large-scale data processing with parallel execution,
       quality monitoring, and intelligent error recovery.
       """
       
       def __init__(self, config: Dict[str, Any]):
           self.config = config
           self.orchestrator = None
           self.monitor = None
           self.storage_backends = {}
           self._setup_workflow()
       
       def _setup_workflow(self):
           """Initialize workflow components."""
           self.orchestrator = Orchestrator()
           
           # Initialize monitoring
           self.monitor = DataPipelineMonitor(
               metrics_backend=self.config.get('metrics_backend', 'prometheus')
           )
           
           # Setup storage backends
           self._setup_storage_backends()
           
           # Initialize data tools
           self.tools = {
               'ingestion': DataIngestionTool(self.storage_backends),
               'validation': DataValidationTool(),
               'transformation': DataTransformationTool(),
               'quality': DataQualityTool()
           }
       
       def _setup_storage_backends(self):
           """Configure storage backends."""
           # S3 backend
           if self.config.get('aws_access_key_id'):
               self.storage_backends['s3'] = S3Storage(
                   access_key=self.config['aws_access_key_id'],
                   secret_key=self.config['aws_secret_access_key'],
                   region=self.config.get('aws_region', 'us-east-1')
               )
           
           # Database backend
           if self.config.get('database_url'):
               self.storage_backends['database'] = DatabaseConnector(
                   connection_string=self.config['database_url']
               )
           
           # Local filesystem
           self.storage_backends['local'] = LocalFileSystem()
       
       async def process_data(
           self,
           sources: List[str],
           output_targets: List[str],
           mode: str = 'batch',
           schema_path: Optional[str] = None,
           enable_ml: bool = False,
           **kwargs
       ) -> Dict[str, Any]:
           """
           Process data from sources to targets.
           
           Args:
               sources: List of data source URIs
               output_targets: List of output target URIs
               mode: Processing mode ('batch' or 'streaming')
               schema_path: Path to schema definition
               enable_ml: Enable ML processing features
               
           Returns:
               Processing results and metrics
           """
           start_time = datetime.now()
           
           logging.info(f"Starting data processing workflow in {mode} mode")
           logging.info(f"Sources: {sources}")
           logging.info(f"Targets: {output_targets}")
           
           # Prepare context
           context = {
               'sources': sources,
               'output_targets': output_targets,
               'mode': mode,
               'schema_path': schema_path,
               'enable_ml': enable_ml,
               'start_time': start_time.isoformat(),
               **kwargs
           }
           
           # Execute pipeline
           try:
               # Start monitoring
               await self.monitor.start_pipeline_monitoring(
                   pipeline_id=f"data_processing_{start_time.strftime('%Y%m%d_%H%M%S')}"
               )
               
               # Execute pipeline
               results = await self.orchestrator.execute_pipeline(
                   'data_processing_pipeline.yaml',
                   context=context,
                   progress_callback=self._progress_callback,
                   error_callback=self._error_callback
               )
               
               # Process results
               processing_report = await self._generate_processing_report(results)
               
               # Stop monitoring and get metrics
               metrics = await self.monitor.stop_pipeline_monitoring()
               processing_report['metrics'] = metrics
               
               return processing_report
               
           except Exception as e:
               logging.error(f"Pipeline failed: {str(e)}")
               await self.monitor.record_pipeline_failure(str(e))
               raise
       
       async def _progress_callback(self, task_id: str, progress: float, message: str):
           """Handle progress updates."""
           await self.monitor.record_task_progress(task_id, progress)
           logging.info(f"{task_id}: {progress:.0%} - {message}")
       
       async def _error_callback(self, task_id: str, error: Exception):
           """Handle task errors."""
           await self.monitor.record_task_error(task_id, str(error))
           logging.error(f"{task_id} failed: {str(error)}")
       
       async def _generate_processing_report(
           self,
           results: Dict[str, Any]
       ) -> Dict[str, Any]:
           """Generate comprehensive processing report."""
           report = {
               'summary': {
                   'total_records_processed': 0,
                   'total_records_failed': 0,
                   'processing_time': 0,
                   'data_quality_score': 0
               },
               'details': {},
               'quality_report': {},
               'anomalies': {},
               'export_info': {}
           }
           
           # Calculate summary statistics
           if 'ingest_data' in results:
               ingestion_stats = results['ingest_data']['ingestion_stats']
               report['summary']['total_records_processed'] = sum(
                   stats.get('records_processed', 0) 
                   for stats in ingestion_stats
               )
               report['summary']['total_records_failed'] = sum(
                   stats.get('records_failed', 0) 
                   for stats in ingestion_stats
               )
           
           # Data quality report
           if 'analyze_quality' in results:
               quality_data = results['analyze_quality']
               report['quality_report'] = {
                   'scores': quality_data['quality_scores'],
                   'report': quality_data['quality_report'],
                   'recommendations': quality_data['recommendations']
               }
               report['summary']['data_quality_score'] = np.mean(
                   list(quality_data['quality_scores'].values())
               )
           
           # Anomaly report
           if 'detect_anomalies' in results:
               anomaly_data = results['detect_anomalies']
               report['anomalies'] = {
                   'count': len(anomaly_data['anomalies']),
                   'percentage': len(anomaly_data['anomalies']) / report['summary']['total_records_processed'] * 100
               }
           
           # Export information
           if 'export_data' in results:
               export_data = results['export_data']
               report['export_info'] = {
                   'locations': export_data['export_locations'],
                   'stats': export_data['export_stats']
               }
           
           # Processing time
           start_time = datetime.fromisoformat(results['context']['start_time'])
           report['summary']['processing_time'] = (datetime.now() - start_time).total_seconds()
           
           return report

Parallel Processing
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   class ParallelDataProcessor:
       """Handle parallel data processing with resource management."""
       
       def __init__(self, max_workers: int = 8):
           self.max_workers = max_workers
           self.semaphore = asyncio.Semaphore(max_workers)
           self.task_queue = asyncio.Queue()
           self.results_queue = asyncio.Queue()
       
       async def process_data_parallel(
           self,
           data_chunks: List[pd.DataFrame],
           processing_func: callable,
           **kwargs
       ) -> List[pd.DataFrame]:
           """Process data chunks in parallel."""
           # Create worker tasks
           workers = [
               asyncio.create_task(self._worker(processing_func, **kwargs))
               for _ in range(self.max_workers)
           ]
           
           # Queue all chunks
           for i, chunk in enumerate(data_chunks):
               await self.task_queue.put((i, chunk))
           
           # Add sentinel values to stop workers
           for _ in range(self.max_workers):
               await self.task_queue.put(None)
           
           # Wait for all workers to complete
           await asyncio.gather(*workers)
           
           # Collect results
           results = []
           while not self.results_queue.empty():
               results.append(await self.results_queue.get())
           
           # Sort by original order
           results.sort(key=lambda x: x[0])
           return [result[1] for result in results]
       
       async def _worker(self, processing_func: callable, **kwargs):
           """Worker coroutine for processing data."""
           while True:
               item = await self.task_queue.get()
               if item is None:
                   break
               
               idx, chunk = item
               
               async with self.semaphore:
                   try:
                       # Process chunk
                       processed = await processing_func(chunk, **kwargs)
                       await self.results_queue.put((idx, processed))
                   except Exception as e:
                       logging.error(f"Error processing chunk {idx}: {e}")
                       # Put error result
                       await self.results_queue.put((idx, None))

Data Quality Management
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   class DataQualityManager:
       """Manage data quality throughout the pipeline."""
       
       def __init__(self):
           self.quality_rules = {}
           self.quality_history = []
       
       async def define_quality_rules(
           self,
           data_schema: Dict[str, Any]
       ) -> Dict[str, Any]:
           """Define quality rules based on schema."""
           rules = {}
           
           for column, dtype in data_schema.items():
               rules[column] = {
                   'completeness': {'min_non_null_ratio': 0.95},
                   'validity': self._get_validity_rules(dtype),
                   'consistency': self._get_consistency_rules(column),
                   'accuracy': self._get_accuracy_rules(column, dtype)
               }
           
           self.quality_rules = rules
           return rules
       
       async def assess_quality(
           self,
           data: pd.DataFrame,
           dimensions: List[str] = None
       ) -> Dict[str, Any]:
           """Assess data quality across multiple dimensions."""
           dimensions = dimensions or ['completeness', 'validity', 'consistency', 'accuracy']
           
           quality_scores = {}
           quality_issues = []
           
           for dimension in dimensions:
               score, issues = await self._assess_dimension(data, dimension)
               quality_scores[dimension] = score
               quality_issues.extend(issues)
           
           # Calculate overall score
           overall_score = np.mean(list(quality_scores.values()))
           
           # Generate recommendations
           recommendations = await self._generate_recommendations(
               quality_scores,
               quality_issues
           )
           
           result = {
               'scores': quality_scores,
               'overall_score': overall_score,
               'issues': quality_issues,
               'recommendations': recommendations,
               'timestamp': datetime.now().isoformat()
           }
           
           # Store in history
           self.quality_history.append(result)
           
           return result
       
       async def _assess_dimension(
           self,
           data: pd.DataFrame,
           dimension: str
       ) -> tuple[float, List[Dict]]:
           """Assess a specific quality dimension."""
           if dimension == 'completeness':
               return await self._assess_completeness(data)
           elif dimension == 'validity':
               return await self._assess_validity(data)
           elif dimension == 'consistency':
               return await self._assess_consistency(data)
           elif dimension == 'accuracy':
               return await self._assess_accuracy(data)
           else:
               raise ValueError(f"Unknown dimension: {dimension}")

Incremental Processing
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   class IncrementalProcessor:
       """Handle incremental data processing."""
       
       def __init__(self, state_backend: str = 'redis'):
           self.state_backend = self._init_state_backend(state_backend)
           self.processed_markers = {}
       
       async def get_incremental_data(
           self,
           source: str,
           full_refresh: bool = False
       ) -> AsyncIterator[pd.DataFrame]:
           """Get only new/changed data since last run."""
           if full_refresh:
               # Process all data
               async for chunk in self._read_all_data(source):
                   yield chunk
           else:
               # Get last processed marker
               last_marker = await self.state_backend.get(
                   f"last_processed:{source}"
               )
               
               # Read only new data
               async for chunk in self._read_incremental_data(
                   source,
                   last_marker
               ):
                   yield chunk
                   
                   # Update marker
                   await self._update_marker(source, chunk)
       
       async def _read_incremental_data(
           self,
           source: str,
           last_marker: Optional[str]
       ) -> AsyncIterator[pd.DataFrame]:
           """Read data incrementally based on marker."""
           # Implementation depends on source type
           if source.startswith('s3://'):
               # List objects modified after marker timestamp
               async for obj in self._list_s3_objects(source, last_marker):
                   yield await self._read_s3_object(obj)
           
           elif source.startswith('jdbc://'):
               # Query with WHERE clause based on marker
               query = f"""
                   SELECT * FROM table
                   WHERE updated_at > '{last_marker}'
                   ORDER BY updated_at
               """
               async for chunk in self._query_database(query):
                   yield chunk

Running the Workflow
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # main.py
   import asyncio
   import argparse
   from data_processing_workflow import DataProcessingWorkflow
   
   async def main():
       parser = argparse.ArgumentParser(description='Data Processing Workflow')
       parser.add_argument('--source', nargs='+', required=True,
                          help='Data source URIs')
       parser.add_argument('--output', nargs='+', required=True,
                          help='Output target URIs')
       parser.add_argument('--mode', choices=['batch', 'streaming'],
                          default='batch')
       parser.add_argument('--schema', help='Schema definition file')
       parser.add_argument('--enable-ml', action='store_true',
                          help='Enable ML features')
       parser.add_argument('--parallel-workers', type=int, default=8)
       parser.add_argument('--chunk-size', type=int, default=10000)
       parser.add_argument('--incremental', action='store_true',
                          help='Process incrementally')
       
       args = parser.parse_args()
       
       # Configuration
       config = {
           'openai_api_key': os.getenv('OPENAI_API_KEY'),
           'database_url': os.getenv('DATABASE_URL'),
           'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
           'aws_secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
           'metrics_backend': 'prometheus'
       }
       
       # Create workflow
       workflow = DataProcessingWorkflow(config)
       
       # Process data
       results = await workflow.process_data(
           sources=args.source,
           output_targets=args.output,
           mode=args.mode,
           schema_path=args.schema,
           enable_ml=args.enable_ml,
           parallel_workers=args.parallel_workers,
           chunk_size=args.chunk_size,
           incremental=args.incremental
       )
       
       # Display results
       print("\n📊 Data Processing Complete!")
       print(f"Total Records: {results['summary']['total_records_processed']:,}")
       print(f"Failed Records: {results['summary']['total_records_failed']:,}")
       print(f"Processing Time: {results['summary']['processing_time']:.2f}s")
       print(f"Data Quality Score: {results['summary']['data_quality_score']:.2%}")
       
       if results['anomalies']:
           print(f"\n⚠️  Anomalies Detected: {results['anomalies']['count']:,} "
                 f"({results['anomalies']['percentage']:.2f}%)")
       
       print("\n📍 Output Locations:")
       for location in results['export_info']['locations']:
           print(f"  - {location}")
       
       # Save detailed report
       report_path = f"processing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
       with open(report_path, 'w') as f:
           json.dump(results, f, indent=2, default=str)
       print(f"\n💾 Detailed report saved to: {report_path}")
   
   if __name__ == "__main__":
       asyncio.run(main())

Advanced Features
-----------------

Stream Processing
^^^^^^^^^^^^^^^^^

.. code-block:: python

   class StreamProcessor:
       """Handle real-time stream processing."""
       
       def __init__(self, kafka_config: Dict[str, Any]):
           self.kafka_consumer = self._init_kafka_consumer(kafka_config)
           self.window_size = timedelta(minutes=5)
           self.windows = {}
       
       async def process_stream(
           self,
           topic: str,
           processing_func: callable
       ):
           """Process streaming data from Kafka."""
           async for message in self.kafka_consumer.subscribe(topic):
               # Parse message
               data = json.loads(message.value)
               timestamp = datetime.fromisoformat(data['timestamp'])
               
               # Assign to window
               window_key = self._get_window_key(timestamp)
               
               if window_key not in self.windows:
                   self.windows[window_key] = []
               
               self.windows[window_key].append(data)
               
               # Process complete windows
               await self._process_complete_windows(processing_func)
       
       async def _process_complete_windows(self, processing_func):
           """Process windows that are complete."""
           current_time = datetime.now()
           
           for window_key, data in list(self.windows.items()):
               window_end = window_key + self.window_size
               
               if window_end < current_time:
                   # Process window
                   df = pd.DataFrame(data)
                   result = await processing_func(df)
                   
                   # Emit result
                   await self._emit_result(window_key, result)
                   
                   # Remove processed window
                   del self.windows[window_key]

Data Lineage Tracking
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   class DataLineageTracker:
       """Track data lineage throughout processing."""
       
       def __init__(self):
           self.lineage_graph = nx.DiGraph()
           self.metadata = {}
       
       async def track_transformation(
           self,
           input_data: str,
           output_data: str,
           transformation: str,
           metadata: Dict[str, Any]
       ):
           """Track a data transformation."""
           # Add nodes
           self.lineage_graph.add_node(input_data, type='dataset')
           self.lineage_graph.add_node(output_data, type='dataset')
           
           # Add edge with transformation
           self.lineage_graph.add_edge(
               input_data,
               output_data,
               transformation=transformation,
               timestamp=datetime.now().isoformat(),
               **metadata
           )
           
           # Store metadata
           self.metadata[output_data] = {
               'source': input_data,
               'transformation': transformation,
               'metadata': metadata,
               'timestamp': datetime.now().isoformat()
           }
       
       async def get_data_lineage(
           self,
           dataset: str
       ) -> Dict[str, Any]:
           """Get complete lineage for a dataset."""
           # Get all ancestors
           ancestors = nx.ancestors(self.lineage_graph, dataset)
           
           # Build lineage tree
           lineage = {
               'dataset': dataset,
               'metadata': self.metadata.get(dataset, {}),
               'ancestors': []
           }
           
           for ancestor in ancestors:
               path = nx.shortest_path(
                   self.lineage_graph,
                   ancestor,
                   dataset
               )
               
               lineage['ancestors'].append({
                   'dataset': ancestor,
                   'path': path,
                   'transformations': [
                       self.lineage_graph[path[i]][path[i+1]]
                       for i in range(len(path)-1)
                   ]
               })
           
           return lineage

Monitoring Dashboard
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   class DataPipelineMonitor:
       """Real-time monitoring for data pipelines."""
       
       def __init__(self, metrics_backend: str = 'prometheus'):
           self.metrics = self._init_metrics(metrics_backend)
           self.alerts = []
       
       async def start_pipeline_monitoring(self, pipeline_id: str):
           """Start monitoring a pipeline execution."""
           self.pipeline_id = pipeline_id
           self.start_time = datetime.now()
           
           # Initialize metrics
           self.metrics.gauge('pipeline_active', 1, {'pipeline': pipeline_id})
           
       async def record_task_progress(self, task_id: str, progress: float):
           """Record task progress."""
           self.metrics.gauge(
               'task_progress',
               progress,
               {'pipeline': self.pipeline_id, 'task': task_id}
           )
           
       async def record_data_quality(
           self,
           quality_scores: Dict[str, float]
       ):
           """Record data quality metrics."""
           for dimension, score in quality_scores.items():
               self.metrics.gauge(
                   'data_quality_score',
                   score,
                   {'pipeline': self.pipeline_id, 'dimension': dimension}
               )
               
               # Check thresholds
               if score < 0.8:
                   await self._trigger_alert(
                       f"Low data quality: {dimension} = {score:.2%}"
                   )

Testing
-------

.. code-block:: python

   # test_data_workflow.py
   import pytest
   import pandas as pd
   from data_processing_workflow import DataProcessingWorkflow
   
   @pytest.mark.asyncio
   async def test_data_validation():
       """Test data validation."""
       # Create test data with issues
       test_data = pd.DataFrame({
           'id': [1, 2, None, 4],
           'value': [100, -50, 200, 'invalid'],
           'date': ['2024-01-01', '2024-01-02', 'invalid', '2024-01-04']
       })
       
       workflow = DataProcessingWorkflow({})
       validator = workflow.tools['validation']
       
       # Define schema
       schema = {
           'id': {'type': 'integer', 'nullable': False},
           'value': {'type': 'numeric', 'min': 0},
           'date': {'type': 'date', 'format': '%Y-%m-%d'}
       }
       
       # Validate
       results = await validator.validate(test_data, schema)
       
       assert not results['is_valid']
       assert len(results['errors']) == 3  # null id, negative value, invalid date
   
   @pytest.mark.asyncio
   async def test_parallel_processing():
       """Test parallel data processing."""
       # Create large dataset
       large_data = pd.DataFrame({
           'id': range(100000),
           'value': np.random.rand(100000)
       })
       
       # Split into chunks
       chunks = np.array_split(large_data, 10)
       
       processor = ParallelDataProcessor(max_workers=4)
       
       # Process in parallel
       async def process_chunk(chunk):
           return chunk['value'].mean()
       
       results = await processor.process_data_parallel(
           chunks,
           process_chunk
       )
       
       assert len(results) == 10
       assert all(isinstance(r, float) for r in results)

Best Practices
--------------

1. **Schema Evolution**: Plan for schema changes with versioning
2. **Error Recovery**: Implement checkpointing for long-running processes
3. **Resource Management**: Monitor memory usage for large datasets
4. **Data Quality**: Implement quality gates at each stage
5. **Performance**: Use appropriate partitioning strategies
6. **Monitoring**: Track key metrics and set up alerts
7. **Documentation**: Maintain data dictionaries and lineage

Summary
-------

The Data Processing Workflow demonstrates:

- Scalable data processing with parallel execution
- Comprehensive data quality management
- ML integration for feature engineering
- Real-time and batch processing modes
- Data lineage tracking
- Production-ready monitoring and alerting

This workflow provides a foundation for building robust data pipelines that can handle enterprise-scale data processing requirements.