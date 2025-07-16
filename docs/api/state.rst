State API Reference
===================

This section documents the state management and checkpointing components that provide persistence, recovery, and resumption capabilities.

.. note::
   For deployment and troubleshooting guides, see the :doc:`../advanced/deployment` and :doc:`../advanced/troubleshooting` documentation.

.. currentmodule:: orchestrator.state

Overview
--------

The Orchestrator state management system provides comprehensive persistence and recovery capabilities for pipeline execution. It includes intelligent checkpointing, state persistence, and automatic recovery from failures.

**Key Features:**
- **Persistent State**: Store pipeline state across executions
- **Adaptive Checkpointing**: Intelligent checkpoint timing based on criticality
- **Multi-Backend Support**: File, memory, PostgreSQL, and Redis backends
- **Automatic Recovery**: Resume pipelines from last checkpoint
- **State Compression**: Efficient storage with optional compression
- **Concurrent Access**: Thread-safe state operations

**Usage Pattern:**

.. code-block:: python

    from orchestrator.state.state_manager import StateManager
    
    # Initialize state manager
    state_manager = StateManager(
        backend_type="postgres",
        checkpoint_strategy="adaptive",
        compression_enabled=True
    )
    
    # Save pipeline state
    await state_manager.save_pipeline_state(pipeline_id, state)
    
    # Load pipeline state
    state = await state_manager.load_pipeline_state(pipeline_id)
    
    # Resume pipeline from checkpoint
    await state_manager.resume_pipeline(pipeline_id)

State Manager
-------------

The StateManager is the core component that handles all state persistence and recovery operations.

**Key Capabilities:**
- **Pipeline State Persistence**: Save and restore complete pipeline states
- **Checkpoint Management**: Create and manage execution checkpoints
- **Recovery Operations**: Automatic recovery from failures
- **State Querying**: Query pipeline states and execution history
- **Cleanup Operations**: Automatic cleanup of old checkpoints

**Example Usage:**

.. code-block:: python

    from orchestrator.state.state_manager import StateManager
    
    # Initialize with PostgreSQL backend
    state_manager = StateManager(
        backend_type="postgres",
        backend_config={
            "url": "postgresql://user:pass@localhost/orchestrator",
            "pool_size": 10
        },
        checkpoint_strategy="adaptive",
        compression_enabled=True
    )
    
    # Save pipeline state
    pipeline_state = {
        "pipeline_id": "research_pipeline",
        "status": "running",
        "completed_tasks": ["web_search", "data_analysis"],
        "current_task": "report_generation",
        "context": {"search_results": results}
    }
    
    await state_manager.save_pipeline_state(
        pipeline_id="research_pipeline",
        state=pipeline_state
    )
    
    # Create checkpoint
    checkpoint_id = await state_manager.create_checkpoint(
        pipeline_id="research_pipeline",
        checkpoint_data={"task_results": task_results}
    )
    
    # Load pipeline state
    restored_state = await state_manager.load_pipeline_state(
        pipeline_id="research_pipeline"
    )
    
    # Resume from checkpoint
    await state_manager.resume_from_checkpoint(
        pipeline_id="research_pipeline",
        checkpoint_id=checkpoint_id
    )

**Classes:**

.. autoclass:: orchestrator.state.state_manager.StateManager
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: orchestrator.state.state_manager.StateManagerError
   :members:
   :undoc-members:
   :show-inheritance:

Storage Backends
----------------

The state system supports multiple storage backends for different deployment scenarios:

**Available Backends:**
- **File Backend**: Local file system storage
- **Memory Backend**: In-memory storage for testing
- **PostgreSQL Backend**: Production-ready database storage
- **Redis Backend**: High-performance caching and state storage

**Backend Configuration:**

.. code-block:: python

    # File backend
    file_config = {
        "path": "./checkpoints",
        "compression": True,
        "max_file_size": "100MB"
    }
    
    # PostgreSQL backend
    postgres_config = {
        "url": "postgresql://user:pass@localhost/db",
        "pool_size": 20,
        "max_overflow": 30
    }
    
    # Redis backend
    redis_config = {
        "host": "localhost",
        "port": 6379,
        "db": 0,
        "password": "redis_password"
    }

.. autoclass:: orchestrator.state.backends.StorageBackend
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: orchestrator.state.backends.FileBackend
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: orchestrator.state.backends.MemoryBackend
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: orchestrator.state.backends.PostgreSQLBackend
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: orchestrator.state.backends.RedisBackend
   :members:
   :undoc-members:
   :show-inheritance:

Adaptive Checkpointing
----------------------

The adaptive checkpointing system automatically determines optimal checkpoint timing based on pipeline characteristics and execution patterns.

**Checkpointing Strategies:**
- **Time-Based**: Checkpoint at regular time intervals
- **Progress-Based**: Checkpoint based on task completion
- **Adaptive**: Intelligent checkpointing based on criticality
- **Event-Driven**: Checkpoint on specific events

**Key Features:**
- **Criticality Analysis**: Assess task importance for checkpoint timing
- **Resource Optimization**: Balance checkpoint frequency with performance
- **Failure Recovery**: Fast recovery from optimally placed checkpoints
- **Cost Optimization**: Minimize storage costs while maintaining reliability

**Example Usage:**

.. code-block:: python

    from orchestrator.state.adaptive_checkpoint import AdaptiveCheckpointManager
    
    # Create adaptive checkpoint manager
    checkpoint_manager = AdaptiveCheckpointManager(
        min_checkpoint_interval=300,  # 5 minutes
        max_checkpoint_interval=3600,  # 1 hour
        criticality_threshold=0.7,
        storage_budget="1GB"
    )
    
    # Analyze task criticality
    criticality = await checkpoint_manager.analyze_task_criticality(task)
    
    # Determine if checkpoint is needed
    should_checkpoint = await checkpoint_manager.should_checkpoint(
        pipeline_id="research_pipeline",
        current_task=task,
        time_since_last_checkpoint=600
    )
    
    if should_checkpoint:
        await checkpoint_manager.create_checkpoint(
            pipeline_id="research_pipeline",
            checkpoint_data=current_state
        )

**Classes:**

.. autoclass:: orchestrator.state.adaptive_checkpoint.AdaptiveCheckpointManager
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: orchestrator.state.adaptive_checkpoint.CheckpointStrategy
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: orchestrator.state.adaptive_checkpoint.AdaptiveStrategy
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: orchestrator.state.adaptive_checkpoint.TimeBasedStrategy
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: orchestrator.state.adaptive_checkpoint.ProgressBasedStrategy
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: orchestrator.state.adaptive_checkpoint.EventDrivenStrategy
   :members:
   :undoc-members:
   :show-inheritance:

Simple State Management
-----------------------

For lightweight use cases, the Orchestrator provides a simplified state management interface:

.. autoclass:: orchestrator.state.simple_state_manager.SimpleStateManager
   :members:
   :undoc-members:
   :show-inheritance:

State Recovery
--------------

The state system includes comprehensive recovery capabilities for handling failures and resuming execution:

**Recovery Features:**
- **Automatic Detection**: Detect interrupted pipelines
- **State Validation**: Verify checkpoint integrity
- **Selective Recovery**: Recover specific parts of pipeline state
- **Rollback Support**: Rollback to previous checkpoints
- **Conflict Resolution**: Handle concurrent state modifications

**Example Usage:**

.. code-block:: python

    from orchestrator.state.recovery import StateRecovery
    
    # Initialize recovery system
    recovery = StateRecovery(state_manager)
    
    # Detect interrupted pipelines
    interrupted_pipelines = await recovery.detect_interrupted_pipelines()
    
    for pipeline_id in interrupted_pipelines:
        # Validate checkpoint integrity
        is_valid = await recovery.validate_checkpoint(pipeline_id)
        
        if is_valid:
            # Recover pipeline
            await recovery.recover_pipeline(pipeline_id)
            print(f"Pipeline {pipeline_id} recovered successfully")
        else:
            # Rollback to previous checkpoint
            await recovery.rollback_to_previous_checkpoint(pipeline_id)
            print(f"Pipeline {pipeline_id} rolled back to safe state")

**Classes:**

.. autoclass:: orchestrator.state.recovery.StateRecovery
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: orchestrator.state.recovery.RecoveryError
   :members:
   :undoc-members:
   :show-inheritance:

State Monitoring
----------------

Monitor state management operations and performance:

**Monitoring Features:**
- **State Metrics**: Track checkpoint creation and recovery times
- **Storage Usage**: Monitor storage backend utilization
- **Performance Analytics**: Analyze checkpoint overhead
- **Error Tracking**: Track state operation failures

**Example Usage:**

.. code-block:: python

    from orchestrator.state.monitoring import StateMonitor
    
    # Create state monitor
    monitor = StateMonitor(
        metrics_backend="prometheus",
        alert_thresholds={
            "checkpoint_time": 30.0,  # seconds
            "storage_usage": 0.8,     # 80%
            "error_rate": 0.05        # 5%
        }
    )
    
    # Monitor state operations
    state_manager = StateManager(monitor=monitor)
    
    # Get state metrics
    metrics = monitor.get_state_metrics()
    print(f"Average checkpoint time: {metrics.avg_checkpoint_time}")
    print(f"Storage utilization: {metrics.storage_utilization:.1%}")
    print(f"Recovery success rate: {metrics.recovery_success_rate:.1%}")

**Classes:**

.. autoclass:: orchestrator.state.monitoring.StateMonitor
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: orchestrator.state.monitoring.StateMetrics
   :members:
   :undoc-members:
   :show-inheritance:

Best Practices
--------------

**State Management Best Practices:**

1. **Choose Appropriate Backend**: Use PostgreSQL for production, Redis for high-performance caching
2. **Configure Checkpointing**: Use adaptive checkpointing for optimal performance
3. **Monitor Storage**: Set up monitoring for storage usage and performance
4. **Test Recovery**: Regularly test recovery procedures
5. **Optimize Compression**: Enable compression for large states
6. **Set Retention Policies**: Implement automatic cleanup of old checkpoints
7. **Handle Concurrency**: Use proper locking for concurrent access
8. **Validate State**: Always validate checkpoint integrity before recovery

**Performance Optimization:**

.. code-block:: python

    # Optimized state manager configuration
    state_manager = StateManager(
        backend_type="postgres",
        backend_config={
            "url": "postgresql://user:pass@localhost/db",
            "pool_size": 20,
            "connection_timeout": 30
        },
        checkpoint_strategy="adaptive",
        compression_enabled=True,
        compression_level=6,
        batch_size=100,
        async_operations=True
    )

For detailed deployment and troubleshooting guides, see the :doc:`../advanced/deployment` and :doc:`../advanced/troubleshooting` documentation.