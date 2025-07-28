==========
Monitoring
==========

Monitoring is crucial for understanding pipeline performance, tracking costs, debugging issues, and optimizing AI workflows. Orchestrator provides comprehensive monitoring capabilities including metrics collection, logging, tracing, and alerting.

Overview
========

Orchestrator's monitoring system provides:

1. **Metrics Collection**: Performance, cost, and usage metrics
2. **Distributed Tracing**: End-to-end request tracking
3. **Structured Logging**: Detailed execution logs
4. **Real-time Dashboards**: Visual monitoring interfaces
5. **Alerting**: Proactive issue detection
6. **Cost Tracking**: AI model usage and costs

Quick Start
===========

Enable monitoring with minimal configuration:

.. code-block:: yaml

   # orchestrator.yaml
   monitoring:
     enabled: true
     providers:
       - type: "prometheus"
         endpoint: "http://localhost:9090"
       - type: "opentelemetry"
         endpoint: "http://localhost:4317"
     
     logging:
       level: "INFO"
       format: "json"
       output: "stdout"

Metrics Collection
==================

Built-in Metrics
----------------

Orchestrator automatically collects these metrics:

**Pipeline Metrics:**

- ``pipeline_executions_total``: Total pipeline executions
- ``pipeline_duration_seconds``: Pipeline execution time
- ``pipeline_success_rate``: Success percentage
- ``pipeline_error_rate``: Error percentage
- ``pipeline_active_count``: Currently running pipelines

**Task Metrics:**

- ``task_executions_total``: Total task executions
- ``task_duration_seconds``: Task execution time
- ``task_retry_count``: Number of retries
- ``task_queue_size``: Pending tasks in queue
- ``task_concurrency``: Concurrent task execution

**Model Metrics:**

- ``model_requests_total``: Total model API calls
- ``model_tokens_used``: Token consumption
- ``model_cost_dollars``: Cost per model
- ``model_latency_seconds``: Model response time
- ``model_error_rate``: Model API errors

**Resource Metrics:**

- ``memory_usage_bytes``: Memory consumption
- ``cpu_usage_percent``: CPU utilization
- ``disk_io_bytes``: Disk read/write
- ``network_io_bytes``: Network traffic

Custom Metrics
--------------

Define custom metrics for your pipelines:

.. code-block:: python

   from orchestrator.monitoring import metrics
   
   # Counter metric
   processed_items = metrics.Counter(
       'processed_items_total',
       'Total number of processed items',
       labels=['item_type', 'status']
   )
   
   # Gauge metric
   queue_depth = metrics.Gauge(
       'processing_queue_depth',
       'Current queue depth',
       labels=['queue_name']
   )
   
   # Histogram metric
   processing_time = metrics.Histogram(
       'item_processing_duration_seconds',
       'Time to process each item',
       buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
       labels=['item_type']
   )
   
   # Summary metric
   api_latency = metrics.Summary(
       'external_api_latency_seconds',
       'External API call latency',
       labels=['api_endpoint', 'method']
   )

Using Metrics in Pipelines
--------------------------

.. code-block:: yaml

   steps:
     - id: process_data
       action: transform_data
       parameters:
         data: "{{ inputs.data }}"
       monitoring:
         custom_metrics:
           - name: "data_rows_processed"
             type: "counter"
             value: "{{ results.row_count }}"
             labels:
               data_type: "{{ inputs.data_type }}"
           
           - name: "processing_accuracy"
             type: "gauge"
             value: "{{ results.accuracy_score }}"
           
           - name: "transformation_time"
             type: "histogram"
             value: "{{ execution.duration }}"
             buckets: [0.1, 0.5, 1, 5, 10]

Distributed Tracing
===================

OpenTelemetry Integration
-------------------------

Orchestrator supports OpenTelemetry for distributed tracing:

.. code-block:: yaml

   tracing:
     enabled: true
     provider: "opentelemetry"
     config:
       service_name: "orchestrator-pipelines"
       endpoint: "http://localhost:4317"
       sampling_rate: 0.1  # Sample 10% of requests
       exporters:
         - "jaeger"
         - "zipkin"
       propagators:
         - "tracecontext"
         - "baggage"

Trace Context
-------------

Traces include comprehensive context:

.. code-block:: python

   {
       "trace_id": "7d3a4f9b8e2c1a5d",
       "span_id": "3f2e1d4c5b6a",
       "parent_span_id": "1a2b3c4d5e6f",
       "operation": "pipeline.execute",
       "service": "orchestrator",
       "attributes": {
           "pipeline.id": "research_assistant",
           "pipeline.version": "1.0",
           "user.id": "user-123",
           "environment": "production",
           "model.provider": "openai",
           "model.name": "gpt-4"
       },
       "events": [
           {
               "name": "task_started",
               "timestamp": "2024-01-15T10:30:00Z",
               "attributes": {
                   "task.id": "web_search",
                   "task.type": "search"
               }
           }
       ],
       "duration_ms": 1523
   }

Custom Spans
------------

Add custom spans for detailed tracing:

.. code-block:: python

   from orchestrator.monitoring import tracer
   
   class CustomTask:
       async def execute(self, context):
           with tracer.start_span("custom_operation") as span:
               span.set_attribute("operation.type", "data_processing")
               span.set_attribute("data.size", len(context.data))
               
               try:
                   # Process data
                   result = await self.process_data(context.data)
                   span.set_attribute("result.size", len(result))
                   span.set_status(StatusCode.OK)
                   return result
                   
               except Exception as e:
                   span.record_exception(e)
                   span.set_status(StatusCode.ERROR, str(e))
                   raise

Logging
=======

Structured Logging
------------------

All logs are structured for easy parsing:

.. code-block:: json

   {
       "timestamp": "2024-01-15T10:30:45.123Z",
       "level": "INFO",
       "logger": "orchestrator.pipeline",
       "message": "Pipeline execution started",
       "pipeline_id": "research_assistant",
       "execution_id": "exec_12345",
       "user_id": "user_123",
       "context": {
           "input_size": 1024,
           "model": "gpt-4",
           "environment": "production"
       },
       "trace_id": "7d3a4f9b8e2c1a5d",
       "span_id": "3f2e1d4c5b6a"
   }

Log Levels
----------

Configure appropriate log levels:

.. code-block:: yaml

   logging:
     levels:
       default: "INFO"
       modules:
         orchestrator.pipeline: "DEBUG"
         orchestrator.models: "INFO"
         orchestrator.tools: "WARNING"
         orchestrator.state: "ERROR"
     
     filters:
       - type: "sensitive_data"
         fields: ["password", "api_key", "token"]
       - type: "pii"
         enabled: true

Log Aggregation
---------------

Send logs to centralized systems:

.. code-block:: yaml

   logging:
     outputs:
       - type: "elasticsearch"
         config:
           hosts: ["http://localhost:9200"]
           index: "orchestrator-logs"
           bulk_size: 1000
           flush_interval: 5
       
       - type: "cloudwatch"
         config:
           region: "us-east-1"
           log_group: "/aws/orchestrator"
           stream_prefix: "pipeline"
       
       - type: "file"
         config:
           path: "/var/log/orchestrator/pipeline.log"
           rotation: "daily"
           retention: 30

Cost Monitoring
===============

Model Cost Tracking
-------------------

Track AI model usage and costs:

.. code-block:: yaml

   cost_tracking:
     enabled: true
     providers:
       openai:
         models:
           gpt-4:
             input_cost_per_1k: 0.03
             output_cost_per_1k: 0.06
           gpt-3.5-turbo:
             input_cost_per_1k: 0.001
             output_cost_per_1k: 0.002
       
       anthropic:
         models:
           claude-opus-4-20250514:
             input_cost_per_1k: 0.015
             output_cost_per_1k: 0.075
     
     budgets:
       daily_limit: 100.00
       monthly_limit: 2000.00
       per_user_limit: 10.00
       
     alerts:
       - threshold: 0.8  # 80% of budget
         action: "warn"
       - threshold: 0.95  # 95% of budget
         action: "throttle"
       - threshold: 1.0  # 100% of budget
         action: "stop"

Cost Reports
------------

Generate detailed cost reports:

.. code-block:: python

   from orchestrator.monitoring import CostReporter
   
   reporter = CostReporter()
   
   # Daily cost report
   daily_report = await reporter.get_daily_report(date="2024-01-15")
   print(f"Total cost: ${daily_report.total_cost:.2f}")
   print(f"By model: {daily_report.cost_by_model}")
   print(f"By pipeline: {daily_report.cost_by_pipeline}")
   print(f"By user: {daily_report.cost_by_user}")
   
   # Cost trends
   trends = await reporter.get_cost_trends(days=30)
   print(f"Average daily cost: ${trends.avg_daily_cost:.2f}")
   print(f"Cost trend: {trends.trend_direction} ({trends.trend_percentage:.1f}%)")

Real-time Dashboards
====================

Grafana Integration
-------------------

Orchestrator provides pre-built Grafana dashboards:

.. code-block:: yaml

   dashboards:
     grafana:
       url: "http://localhost:3000"
       datasources:
         - name: "prometheus"
           type: "prometheus"
           url: "http://localhost:9090"
         - name: "elasticsearch"
           type: "elasticsearch"
           url: "http://localhost:9200"
       
       dashboards:
         - "pipeline-overview"
         - "model-performance"
         - "cost-analysis"
         - "error-tracking"
         - "resource-utilization"

Custom Dashboards
-----------------

Create custom monitoring dashboards:

.. code-block:: json

   {
       "dashboard": {
           "title": "AI Pipeline Monitor",
           "panels": [
               {
                   "title": "Pipeline Success Rate",
                   "type": "graph",
                   "targets": [
                       {
                           "expr": "rate(pipeline_success_total[5m]) / rate(pipeline_executions_total[5m])",
                           "legend": "Success Rate"
                       }
                   ]
               },
               {
                   "title": "Model Costs (24h)",
                   "type": "piechart",
                   "targets": [
                       {
                           "expr": "sum(increase(model_cost_dollars[24h])) by (model)",
                           "legend": "{{model}}"
                       }
                   ]
               },
               {
                   "title": "Active Pipelines",
                   "type": "singlestat",
                   "targets": [
                       {
                           "expr": "pipeline_active_count",
                           "instant": true
                       }
                   ]
               }
           ]
       }
   }

Alerting
========

Alert Rules
-----------

Define comprehensive alert rules:

.. code-block:: yaml

   alerts:
     rules:
       # Performance alerts
       - name: "high_error_rate"
         expr: "rate(pipeline_errors_total[5m]) > 0.1"
         for: "5m"
         severity: "critical"
         annotations:
           summary: "High pipeline error rate"
           description: "Error rate is {{ $value | humanizePercentage }}"
       
       # Cost alerts
       - name: "cost_spike"
         expr: "increase(model_cost_dollars[1h]) > 50"
         severity: "warning"
         annotations:
           summary: "Unusual cost spike detected"
           description: "Cost increased by ${{ $value }} in last hour"
       
       # Resource alerts
       - name: "memory_pressure"
         expr: "memory_usage_percent > 90"
         for: "10m"
         severity: "warning"
         annotations:
           summary: "High memory usage"
           description: "Memory usage at {{ $value }}%"
       
       # SLA alerts
       - name: "sla_violation"
         expr: "histogram_quantile(0.95, pipeline_duration_seconds) > 300"
         severity: "high"
         annotations:
           summary: "95th percentile latency exceeds SLA"
           description: "P95 latency is {{ $value }}s"

Alert Routing
-------------

Route alerts to appropriate channels:

.. code-block:: yaml

   alerting:
     routes:
       - match:
           severity: "critical"
         receivers:
           - "pagerduty"
           - "email-oncall"
           - "slack-critical"
         
       - match:
           severity: "warning"
           team: "platform"
         receivers:
           - "slack-platform"
           - "email-platform"
       
       - match:
           alert_type: "cost"
         receivers:
           - "email-finance"
           - "slack-cost-alerts"
     
     receivers:
       pagerduty:
         routing_key: "{{ env.PAGERDUTY_KEY }}"
         
       email-oncall:
         to: ["oncall@example.com"]
         
       slack-critical:
         webhook: "{{ env.SLACK_CRITICAL_WEBHOOK }}"
         channel: "#critical-alerts"

Performance Monitoring
======================

Latency Tracking
----------------

Monitor pipeline and task latencies:

.. code-block:: yaml

   performance:
     latency_tracking:
       enabled: true
       percentiles: [0.5, 0.75, 0.9, 0.95, 0.99]
       
       sla_thresholds:
         pipeline:
           p50: 10  # seconds
           p95: 30
           p99: 60
         
         task:
           web_search:
             p50: 2
             p95: 5
           ai_analysis:
             p50: 5
             p95: 15

Resource Utilization
--------------------

Track resource usage patterns:

.. code-block:: python

   from orchestrator.monitoring import ResourceMonitor
   
   monitor = ResourceMonitor()
   
   # Get current utilization
   usage = monitor.get_current_usage()
   print(f"CPU: {usage.cpu_percent}%")
   print(f"Memory: {usage.memory_mb}MB / {usage.memory_limit_mb}MB")
   print(f"Active tasks: {usage.active_tasks}")
   
   # Historical analysis
   history = monitor.get_usage_history(hours=24)
   print(f"Peak CPU: {history.max_cpu_percent}%")
   print(f"Avg Memory: {history.avg_memory_mb}MB")

Bottleneck Detection
--------------------

Identify performance bottlenecks:

.. code-block:: yaml

   performance:
     bottleneck_detection:
       enabled: true
       analysis_interval: 300  # 5 minutes
       
       detectors:
         - type: "slow_tasks"
           threshold: 2.0  # 2x slower than average
         
         - type: "queue_backup"
           threshold: 100  # tasks in queue
         
         - type: "resource_contention"
           cpu_threshold: 80
           memory_threshold: 85
       
       auto_scaling:
         enabled: true
         min_workers: 2
         max_workers: 20
         scale_up_threshold: 0.8
         scale_down_threshold: 0.3

Health Checks
=============

System Health
-------------

Monitor overall system health:

.. code-block:: yaml

   health_checks:
     endpoints:
       - path: "/health"
         checks:
           - "database"
           - "cache"
           - "models"
           - "storage"
       
       - path: "/health/detailed"
         checks:
           - name: "database"
             timeout: 5
             query: "SELECT 1"
           
           - name: "model_availability"
             providers: ["openai", "anthropic"]
             timeout: 10
           
           - name: "disk_space"
             min_free_gb: 10
           
           - name: "memory"
             max_usage_percent: 90

Pipeline Health
---------------

Monitor individual pipeline health:

.. code-block:: python

   from orchestrator.monitoring import PipelineHealth
   
   health = PipelineHealth()
   
   # Check pipeline status
   status = await health.check_pipeline("research_assistant")
   print(f"Status: {status.state}")  # healthy, degraded, unhealthy
   print(f"Success rate: {status.success_rate:.1%}")
   print(f"Avg latency: {status.avg_latency:.2f}s")
   print(f"Error types: {status.recent_errors}")
   
   # Get recommendations
   if status.state != "healthy":
       recommendations = health.get_recommendations(status)
       for rec in recommendations:
           print(f"- {rec}")

Debugging Tools
===============

Debug Mode
----------

Enable detailed debugging:

.. code-block:: yaml

   debug:
     enabled: true
     features:
       - "trace_all_requests"
       - "log_model_prompts"
       - "save_intermediate_results"
       - "extended_error_details"
     
     sampling:
       rate: 0.1  # Debug 10% of requests
       users: ["debug_user_1", "debug_user_2"]  # Always debug these users
       pipelines: ["experimental_pipeline"]      # Always debug these pipelines

Request Replay
--------------

Replay requests for debugging:

.. code-block:: python

   from orchestrator.debugging import RequestReplay
   
   replay = RequestReplay()
   
   # Replay a specific execution
   result = await replay.replay_execution(
       execution_id="exec_12345",
       modifications={
           "log_level": "DEBUG",
           "save_all_states": True,
           "use_test_models": True
       }
   )
   
   # Analyze differences
   diff = replay.compare_executions("exec_12345", result.execution_id)
   print(f"Differences found: {len(diff.differences)}")

Profiling
---------

Profile pipeline performance:

.. code-block:: yaml

   profiling:
     enabled: true
     types:
       - "cpu"
       - "memory"
       - "async_io"
     
     triggers:
       - condition: "execution_time > 60"
         profile_duration: 30
       
       - condition: "memory_usage > 1GB"
         profile_type: "memory"
         heap_dump: true

Integration Examples
====================

Complete Monitoring Setup
-------------------------

.. code-block:: yaml

   # monitoring-config.yaml
   monitoring:
     # Metrics
     metrics:
       providers:
         - type: "prometheus"
           endpoint: "http://prometheus:9090"
           pushgateway: "http://pushgateway:9091"
       
       retention: "15d"
       scrape_interval: "15s"
     
     # Tracing
     tracing:
       provider: "opentelemetry"
       endpoint: "http://otel-collector:4317"
       sampling:
         type: "adaptive"
         target_rate: 100  # 100 traces per second
     
     # Logging
     logging:
       outputs:
         - type: "loki"
           endpoint: "http://loki:3100"
         - type: "file"
           path: "/logs/orchestrator.log"
       
       format: "json"
       level: "INFO"
     
     # Dashboards
     dashboards:
       grafana:
         url: "http://grafana:3000"
         org_id: 1
         
     # Alerting
     alerting:
       alertmanager:
         url: "http://alertmanager:9093"
       
       notification_channels:
         - slack
         - email
         - pagerduty

Python Integration
------------------

.. code-block:: python

   from orchestrator import Orchestrator
   from orchestrator.monitoring import MonitoringConfig
   
   # Configure monitoring
   monitoring_config = MonitoringConfig(
       metrics_enabled=True,
       tracing_enabled=True,
       logging_level="INFO",
       cost_tracking_enabled=True
   )
   
   # Initialize orchestrator with monitoring
   orchestrator = Orchestrator(
       monitoring_config=monitoring_config
   )
   
   # Custom metrics
   @orchestrator.monitor(
       metric_name="custom_operation_duration",
       metric_type="histogram"
   )
   async def custom_operation(data):
       # Your operation here
       return process_data(data)
   
   # Run with monitoring
   async with orchestrator.monitored_execution() as monitor:
       result = await orchestrator.execute_pipeline(
           "my_pipeline",
           inputs={"data": data}
       )
       
       # Access monitoring data
       print(f"Execution time: {monitor.duration}s")
       print(f"Total cost: ${monitor.total_cost:.2f}")
       print(f"Tokens used: {monitor.tokens_used}")

Best Practices
==============

1. **Start Simple**: Begin with basic metrics and expand as needed
2. **Set Meaningful Alerts**: Avoid alert fatigue with well-tuned thresholds
3. **Use Structured Logging**: Make logs searchable and parseable
4. **Monitor Costs**: Track AI model usage to control expenses
5. **Implement SLOs**: Define and monitor service level objectives
6. **Regular Reviews**: Analyze monitoring data for optimization opportunities
7. **Automate Responses**: Use monitoring data to trigger automated actions

Summary
=======

Orchestrator's monitoring system provides comprehensive visibility into your AI pipelines, enabling you to:

- Track performance and identify bottlenecks
- Monitor costs and prevent budget overruns
- Debug issues with detailed tracing and logging
- Set up proactive alerting for issues
- Optimize pipeline efficiency based on real data

By leveraging these monitoring capabilities, you can build reliable, efficient, and cost-effective AI automation systems.