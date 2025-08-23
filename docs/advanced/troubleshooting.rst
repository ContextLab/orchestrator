Troubleshooting
================

This comprehensive troubleshooting guide covers common issues, debugging techniques, and solutions for the Orchestrator system.

Common Issues and Solutions
---------------------------

Model Configuration Issues
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Problem: Model not found (e.g., openai/gpt-3.5-turbo not found)**

.. code-block:: bash

    # Check your models.yaml configuration
    cat models.yaml
    
    # Verify model names match exactly what's in your config
    # For OpenAI, use: gpt-4o, gpt-4o-mini, gpt-5
    # For Anthropic, use: claude-sonnet-4-20250514
    # For Ollama, ensure model is available: ollama list

**Problem: API key not configured**

.. code-block:: bash

    # Use the setup script
    python scripts/setup_api_keys.py
    
    # Or set environment variables
    export OPENAI_API_KEY="your-key-here"
    export ANTHROPIC_API_KEY="your-key-here"
    export GOOGLE_AI_API_KEY="your-key-here"

**Problem: Ollama model download fails**

.. code-block:: bash

    # Check Ollama service status
    ollama serve
    
    # Manually pull the model
    ollama pull deepseek-r1:8b
    
    # Check available models
    ollama list

Template Rendering Issues
^^^^^^^^^^^^^^^^^^^^^^^^^

**Problem: Variables not resolved in templates**

.. code-block:: python

    # Check template context using UnifiedTemplateResolver debug mode
    from orchestrator.core.unified_template_resolver import UnifiedTemplateResolver
    
    resolver = UnifiedTemplateResolver(debug_mode=True)
    debug_info = resolver.get_debug_info()
    print(f"Available context: {debug_info}")
    
    # Common issues:
    # - Variable names don't match (case sensitive)
    # - Variables not in scope (check dependencies)
    # - Loop variables need $ prefix in some contexts

**Problem: Template syntax errors**

.. code-block:: python

    # Validate template syntax
    from orchestrator.core.unified_template_resolver import UnifiedTemplateResolver
    
    resolver = UnifiedTemplateResolver()
    errors = resolver.validate_templates("{{ your_template }}")
    if errors:
        print(f"Template errors: {errors}")

Pipeline Validation Issues
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Problem: Pipeline validation fails**

.. code-block:: python

    # Run comprehensive validation
    from orchestrator.validation.validation_report import ValidationReport
    
    report = ValidationReport()
    validation_results = report.validate_pipeline("your_pipeline.yaml")
    print(validation_results.to_dict())
    
    # Use the validation script
    python scripts/validate_all_pipelines.py your_pipeline.yaml

Pipeline Execution Issues
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Problem: Pipeline fails to start**

.. code-block:: python

    # Check pipeline configuration
    from orchestrator.core.pipeline import Pipeline
    from orchestrator.compiler.schema_validator import SchemaValidator
    
    # Validate pipeline schema
    validator = SchemaValidator()
    try:
        validator.validate_pipeline(pipeline_config)
    except ValidationError as e:
        print(f"Configuration error: {e}")
        # Fix: Check YAML syntax and required fields

**Problem: Tasks fail with dependency errors**

.. code-block:: python

    # Use the validation framework to debug dependencies
    from orchestrator.validation.dependency_validator import DependencyValidator
    
    validator = DependencyValidator()
    validation_result = validator.validate(pipeline)
    
    if not validation_result.is_valid:
        print(f"Dependency issues: {validation_result.errors}")
        for error in validation_result.errors:
            print(f"- {error.message} (severity: {error.severity})")
    
    # Check data flow between tasks
    from orchestrator.validation.data_flow_validator import DataFlowValidator
    
    flow_validator = DataFlowValidator()
    flow_result = flow_validator.validate(pipeline)
    if not flow_result.is_valid:
        print(f"Data flow issues: {flow_result.errors}")

**Problem: AUTO tag resolution fails**

.. code-block:: python

    # Debug AUTO tag parsing
    from orchestrator.compiler.auto_tag_yaml_parser import AutoTagYAMLParser
    from orchestrator.compiler.ambiguity_resolver import AmbiguityResolver
    
    # Check AUTO tag syntax
    auto_parser = AutoTagYAMLParser()
    parsed_content = auto_parser.parse_yaml_file("pipeline.yaml")
    
    # Verify model availability for resolution
    resolver = AmbiguityResolver()
    available_models = resolver.get_available_models()
    print(f"Available models for AUTO resolution: {available_models}")

**Problem: Output contains conversational markers**

.. code-block:: python

    # Check OutputSanitizer configuration
    from orchestrator.utils.output_sanitizer import OutputSanitizer
    
    sanitizer = OutputSanitizer(enabled=True)
    cleaned_output = sanitizer.sanitize("Certainly! Here is your result: actual content")
    print(f"Cleaned: {cleaned_output}")
    
    # Configure custom patterns if needed
    sanitizer.add_custom_pattern(r"^Your custom pattern", "starter")
    
    parser = AutoTagYAMLParser()
    try:
        result = parser.parse_yaml_with_auto_tags(yaml_content)
    except Exception as e:
        print(f"AUTO tag parsing error: {e}")
        # Fix: Check for nested tags and special characters

Model Integration Issues
^^^^^^^^^^^^^^^^^^^^^^^^

**Problem: Model initialization fails**

.. code-block:: python

    # Debug model initialization
    from orchestrator.models.model_registry import ModelRegistry
    
    async def debug_model_init(model_name: str):
        registry = ModelRegistry()
        try:
            model = registry.get_model(model_name)
            health = await model.health_check()
            print(f"Model {model_name} health: {health}")
        except Exception as e:
            print(f"Model initialization error: {e}")
            # Common fixes:
            # - Check API keys
            # - Verify network connectivity
            # - Check model availability

**Problem: High API costs**

.. code-block:: python

    # Monitor model costs
    from orchestrator.monitoring.cost_monitor import CostMonitor
    
    monitor = CostMonitor()
    cost_report = await monitor.generate_cost_report()
    
    print(f"Total cost: ${cost_report.total_cost}")
    print(f"Most expensive model: {cost_report.top_cost_model}")
    print(f"Optimization suggestions: {cost_report.suggestions}")

**Problem: Model responses are slow**

.. code-block:: python

    # Profile model performance
    from orchestrator.profiling.model_profiler import ModelProfiler
    
    profiler = ModelProfiler()
    
    @profiler.profile_model
    async def test_model_performance(model):
        start = time.time()
        response = await model.generate_response("Test prompt")
        duration = time.time() - start
        print(f"Response time: {duration:.2f}s")
        return response

Connection and Network Issues
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Problem: Database connection failures**

.. code-block:: python

    # Debug database connectivity
    from orchestrator.state.backends import create_backend
    
    async def test_database_connection():
        try:
            backend = create_backend("postgres", {
                "url": "postgresql://user:pass@localhost/db"
            })
            await backend.connect()
            print("Database connection successful")
        except Exception as e:
            print(f"Database connection failed: {e}")
            # Common fixes:
            # - Check connection string
            # - Verify database is running
            # - Check firewall settings

**Problem: Redis cache connection issues**

.. code-block:: python

    # Debug Redis connectivity
    import redis.asyncio as redis
    
    async def test_redis_connection():
        try:
            r = redis.Redis(host='localhost', port=6379, db=0)
            await r.ping()
            print("Redis connection successful")
        except Exception as e:
            print(f"Redis connection failed: {e}")
            # Fix: Check Redis server status

**Problem: API rate limiting**

.. code-block:: python

    # Handle rate limiting
    from orchestrator.core.error_handler import ErrorHandler, RetryStrategy
    
    error_handler = ErrorHandler(
        retry_strategy=RetryStrategy(
            max_attempts=5,
            backoff_factor=2.0,
            max_backoff=60.0
        )
    )
    
    # Implement exponential backoff for rate-limited requests
    @error_handler.retry_on_rate_limit
    async def make_api_request(model, prompt):
        return await model.generate_response(prompt)

Memory and Resource Issues
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Problem: Memory leaks**

.. code-block:: python

    # Debug memory usage
    import psutil
    import gc
    
    def debug_memory_usage():
        process = psutil.Process()
        memory_info = process.memory_info()
        print(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
        
        # Force garbage collection
        gc.collect()
        
        # Check for circular references
        import sys
        print(f"Reference count: {sys.getrefcount}")

**Problem: CPU overload**

.. code-block:: python

    # Monitor CPU usage
    from orchestrator.monitoring.resource_monitor import ResourceMonitor
    
    monitor = ResourceMonitor()
    
    @monitor.track_cpu_usage
    async def cpu_intensive_task():
        # Your CPU-intensive code here
        pass
    
    # Get CPU usage report
    cpu_report = monitor.get_cpu_report()
    if cpu_report.usage > 0.8:
        print("High CPU usage detected")
        # Fix: Implement task throttling or scaling

**Problem: Disk space issues**

.. code-block:: python

    # Monitor disk usage
    import shutil
    
    def check_disk_space():
        total, used, free = shutil.disk_usage("/")
        print(f"Free disk space: {free // (2**30)} GB")
        
        if free < 1 * (2**30):  # Less than 1GB
            print("Low disk space warning")
            # Fix: Clean up old logs and checkpoints

Debugging Tools and Techniques
------------------------------

Logging Configuration
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import logging
    from orchestrator.utils.logging import setup_logging
    
    # Configure detailed logging
    setup_logging(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('orchestrator.log'),
            logging.StreamHandler()
        ]
    )
    
    # Add context to log messages
    logger = logging.getLogger(__name__)
    logger.info("Pipeline execution started", extra={
        "pipeline_id": pipeline.id,
        "user_id": user.id,
        "timestamp": datetime.utcnow()
    })

Health Check System
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from orchestrator.monitoring.health_checker import HealthChecker
    
    # Comprehensive health checking
    health_checker = HealthChecker()
    
    async def run_health_checks():
        checks = {
            "database": await health_checker.check_database(),
            "cache": await health_checker.check_cache(),
            "models": await health_checker.check_models(),
            "api_endpoints": await health_checker.check_api_endpoints()
        }
        
        for component, status in checks.items():
            if not status.healthy:
                print(f"Health check failed for {component}: {status.error}")

Performance Profiling
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from orchestrator.profiling.profiler import Profiler
    
    # Profile pipeline execution
    profiler = Profiler()
    
    @profiler.profile_async
    async def debug_pipeline_performance(pipeline):
        start_time = time.time()
        
        # Execute pipeline with profiling
        result = await pipeline.execute()
        
        # Generate performance report
        report = profiler.generate_report()
        print(f"Execution time: {report.total_time}")
        print(f"Memory peak: {report.memory_peak}")
        print(f"Bottlenecks: {report.bottlenecks}")
        
        return result

Error Tracking and Alerting
----------------------------

Exception Monitoring
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from orchestrator.monitoring.exception_tracker import ExceptionTracker
    
    # Track and analyze exceptions
    tracker = ExceptionTracker()
    
    @tracker.track_exceptions
    async def monitored_function():
        try:
            # Your code here
            pass
        except Exception as e:
            tracker.record_exception(e, context={
                "function": "monitored_function",
                "user_id": user.id,
                "timestamp": datetime.utcnow()
            })
            raise

Alerting System
^^^^^^^^^^^^^^^

.. code-block:: python

    from orchestrator.monitoring.alerting import AlertManager
    
    # Configure alerts
    alert_manager = AlertManager()
    
    # Set up alert rules
    alert_manager.add_rule(
        name="high_error_rate",
        condition="error_rate > 0.05",
        action="send_email",
        recipients=["admin@example.com"]
    )
    
    alert_manager.add_rule(
        name="low_disk_space",
        condition="disk_free < 1GB",
        action="send_slack",
        channel="#ops"
    )

Pipeline Debugging
------------------

Step-by-Step Execution
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from orchestrator.debugging.step_debugger import StepDebugger
    
    # Debug pipeline step by step
    debugger = StepDebugger()
    
    async def debug_pipeline_execution(pipeline):
        # Enable step-by-step debugging
        debugger.enable_step_mode()
        
        for task in pipeline.tasks:
            print(f"Executing task: {task.id}")
            
            # Set breakpoint
            await debugger.breakpoint(task.id)
            
            # Execute task
            result = await task.execute()
            
            # Inspect result
            print(f"Task result: {result}")
            
            # Continue or abort
            action = input("Continue? (y/n): ")
            if action.lower() == 'n':
                break

State Inspection
^^^^^^^^^^^^^^^^

.. code-block:: python

    from orchestrator.debugging.state_inspector import StateInspector
    
    # Inspect pipeline state
    inspector = StateInspector()
    
    async def inspect_pipeline_state(pipeline_id: str):
        state = await inspector.get_pipeline_state(pipeline_id)
        
        print(f"Pipeline status: {state.status}")
        print(f"Completed tasks: {len(state.completed_tasks)}")
        print(f"Failed tasks: {len(state.failed_tasks)}")
        print(f"Remaining tasks: {len(state.pending_tasks)}")
        
        # Inspect specific task
        if state.failed_tasks:
            failed_task = state.failed_tasks[0]
            print(f"Failed task error: {failed_task.error}")
            print(f"Failed task logs: {failed_task.logs}")

Common Error Patterns
---------------------

Configuration Errors
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

    # Common YAML configuration errors
    
    # Error: Missing required fields
    pipeline:
      name: "research_pipeline"
      # Missing: tasks, version
    
    # Fix: Add required fields
    pipeline:
      name: "research_pipeline"
      version: "1.0"
      tasks:
        - id: "search"
          action: "web_search"

.. code-block:: python

    # Error: Invalid AUTO tag syntax
    yaml_content = """
    parameters:
      query: <AUTO>Search for: latest AI research</AUTO>  # Missing closing tag
    """
    
    # Fix: Properly close AUTO tags
    yaml_content = """
    parameters:
      query: <AUTO>Search for: latest AI research</AUTO>
    """

Runtime Errors
^^^^^^^^^^^^^^^

.. code-block:: python

    # Error: Model not found
    try:
        model = registry.get_model("nonexistent-model")
    except ModelNotFoundError:
        # Fix: Check available models
        available_models = registry.list_models()
        print(f"Available models: {available_models}")

    # Error: Task dependency cycle
    from orchestrator.core.dependency_resolver import DependencyResolver
    
    resolver = DependencyResolver()
    try:
        execution_order = resolver.resolve_dependencies(tasks)
    except CircularDependencyError as e:
        print(f"Circular dependency detected: {e.cycle}")
        # Fix: Remove circular dependencies

Performance Issues
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Issue: Slow pipeline execution
    from orchestrator.profiling.performance_analyzer import PerformanceAnalyzer
    
    analyzer = PerformanceAnalyzer()
    
    async def analyze_slow_pipeline(pipeline):
        analysis = await analyzer.analyze_pipeline(pipeline)
        
        print(f"Bottlenecks: {analysis.bottlenecks}")
        print(f"Suggestions: {analysis.optimization_suggestions}")
        
        # Common fixes:
        # - Enable caching
        # - Parallelize independent tasks
        # - Optimize model selection

Monitoring and Alerting Setup
------------------------------

Metrics Collection
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from prometheus_client import Counter, Histogram, Gauge
    
    # Define metrics
    pipeline_executions = Counter('pipeline_executions_total', 'Total pipeline executions')
    execution_duration = Histogram('pipeline_execution_duration_seconds', 'Pipeline execution time')
    active_pipelines = Gauge('active_pipelines', 'Number of active pipelines')
    
    # Collect metrics
    @execution_duration.time()
    async def execute_pipeline_with_metrics(pipeline):
        pipeline_executions.inc()
        active_pipelines.inc()
        
        try:
            result = await pipeline.execute()
            return result
        finally:
            active_pipelines.dec()

Log Analysis
^^^^^^^^^^^^

.. code-block:: python

    from orchestrator.monitoring.log_analyzer import LogAnalyzer
    
    # Analyze logs for patterns
    analyzer = LogAnalyzer()
    
    # Find error patterns
    error_patterns = analyzer.find_error_patterns(
        log_file="orchestrator.log",
        time_window="1h"
    )
    
    for pattern in error_patterns:
        print(f"Error pattern: {pattern.message}")
        print(f"Frequency: {pattern.count}")
        print(f"First occurrence: {pattern.first_seen}")

Recovery Procedures
-------------------

Pipeline Recovery
^^^^^^^^^^^^^^^^^

.. code-block:: python

    from orchestrator.recovery.pipeline_recovery import PipelineRecovery
    
    # Recover failed pipeline
    recovery = PipelineRecovery()
    
    async def recover_failed_pipeline(pipeline_id: str):
        # Get latest checkpoint
        checkpoint = await recovery.get_latest_checkpoint(pipeline_id)
        
        # Restore pipeline state
        pipeline = await recovery.restore_pipeline(checkpoint)
        
        # Resume execution from last successful task
        await pipeline.resume_from_checkpoint()

Data Recovery
^^^^^^^^^^^^^

.. code-block:: python

    from orchestrator.recovery.data_recovery import DataRecovery
    
    # Recover lost data
    recovery = DataRecovery()
    
    async def recover_lost_data(backup_path: str):
        # Restore from backup
        await recovery.restore_from_backup(backup_path)
        
        # Verify data integrity
        integrity_check = await recovery.verify_data_integrity()
        
        if not integrity_check.passed:
            print(f"Data integrity issues: {integrity_check.issues}")

Emergency Procedures
--------------------

System Shutdown
^^^^^^^^^^^^^^^

.. code-block:: python

    from orchestrator.emergency.emergency_manager import EmergencyManager
    
    # Graceful shutdown
    emergency = EmergencyManager()
    
    async def emergency_shutdown():
        # Stop accepting new pipelines
        await emergency.stop_new_requests()
        
        # Wait for current pipelines to complete
        await emergency.wait_for_completion(timeout=300)
        
        # Force shutdown if needed
        if not emergency.all_pipelines_completed():
            await emergency.force_shutdown()

Disaster Recovery
^^^^^^^^^^^^^^^^^

.. code-block:: python

    from orchestrator.recovery.disaster_recovery import DisasterRecovery
    
    # Full system recovery
    recovery = DisasterRecovery()
    
    async def disaster_recovery():
        # Assess damage
        damage_assessment = await recovery.assess_system_damage()
        
        # Restore from backups
        await recovery.restore_database()
        await recovery.restore_configuration()
        
        # Verify system health
        health_check = await recovery.verify_system_health()
        
        if health_check.healthy:
            print("System recovery successful")
        else:
            print(f"Recovery issues: {health_check.issues}")

Performance Issues
^^^^^^^^^^^^^^^^^^

**Problem: Slow pipeline execution**

.. code-block:: python

    # Enable profiling and monitoring
    import orchestrator as orc
    
    # Use performance monitoring
    from orchestrator.monitoring.performance_monitor import PerformanceMonitor
    
    monitor = PerformanceMonitor()
    
    # Run pipeline with monitoring
    with monitor.profile_execution():
        result = pipeline.run()
    
    # Analyze results
    performance_report = monitor.get_report()
    print(f"Execution time: {performance_report.total_time}")
    print(f"Bottlenecks: {performance_report.bottlenecks}")
    
    # Check template resolution performance
    from orchestrator.core.unified_template_resolver import UnifiedTemplateResolver
    
    resolver = UnifiedTemplateResolver(debug_mode=True)
    debug_info = resolver.get_debug_info()
    print(f"Template resolution stats: {debug_info}")

**Problem: High memory usage**

.. code-block:: python

    # Monitor memory usage during pipeline execution
    import psutil
    import os
    
    def check_memory_usage():
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        print(f"RSS: {memory_info.rss / 1024 / 1024:.2f} MB")
        print(f"VMS: {memory_info.vms / 1024 / 1024:.2f} MB")
    
    # Check before pipeline
    check_memory_usage()
    
    # Run pipeline
    result = pipeline.run()
    
    # Check after pipeline
    check_memory_usage()
    
    # Use memory-efficient model loading
    orc.init_models(lazy_loading=True)

Debugging Techniques
^^^^^^^^^^^^^^^^^^^^

**Enable Debug Mode**

.. code-block:: python

    import logging
    import orchestrator as orc
    
    # Set debug logging level
    logging.basicConfig(level=logging.DEBUG)
    
    # Initialize with debug mode
    orc.init_models(debug_mode=True)
    
    # Use debug mode for template resolution
    from orchestrator.core.unified_template_resolver import UnifiedTemplateResolver
    
    resolver = UnifiedTemplateResolver(debug_mode=True)

**Validate Pipeline Before Execution**

.. code-block:: python

    # Use the validation script before running pipelines
    from orchestrator.validation.validation_report import ValidationReport
    
    # Comprehensive validation
    report = ValidationReport()
    result = report.validate_pipeline("your_pipeline.yaml")
    
    # Check all validation aspects
    if not result.is_valid:
        print("Validation failed:")
        for category, errors in result.errors_by_category.items():
            print(f"  {category}: {len(errors)} errors")
            for error in errors[:3]:  # Show first 3 errors per category
                print(f"    - {error.message}")
    
    # Use command line validation
    # python scripts/validate_all_pipelines.py examples/

**Inspect Pipeline State**

.. code-block:: python

    # Access pipeline execution state
    from orchestrator.core.context_manager import ContextManager
    
    # Get current execution context
    context_manager = ContextManager()
    current_context = context_manager.get_current_context()
    
    print(f"Pipeline ID: {current_context.get('pipeline_id')}")
    print(f"Current Step: {current_context.get('current_step')}")
    print(f"Available Variables: {list(current_context.keys())}")
    
    # Check loop contexts
    from orchestrator.core.loop_context import GlobalLoopContextManager
    
    loop_manager = GlobalLoopContextManager()
    active_loops = loop_manager.active_loops
    print(f"Active loops: {list(active_loops.keys())}")

Best Practices for Troubleshooting
-----------------------------------

1. **Enable Comprehensive Logging**: Use structured logging with appropriate levels
2. **Implement Health Checks**: Monitor all system components continuously
3. **Use Metrics and Monitoring**: Track key performance indicators
4. **Set Up Alerting**: Get notified of issues before they become critical
5. **Document Issues**: Keep a record of common problems and solutions
6. **Test Recovery Procedures**: Regularly test backup and recovery processes
7. **Monitor Resource Usage**: Track CPU, memory, and disk usage
8. **Implement Circuit Breakers**: Prevent cascading failures
9. **Use Staging Environment**: Test changes in a non-production environment
10. **Keep Dependencies Updated**: Regularly update libraries and dependencies

This comprehensive troubleshooting guide should help you identify, diagnose, and resolve issues in your Orchestrator deployment effectively.
