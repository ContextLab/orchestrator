Executor API Reference
======================

This section documents the execution engine components that handle task execution, parallel processing, and sandboxed code execution.

.. note::
   For performance optimization guides, see the :doc:`../advanced/performance_optimization` documentation.

.. currentmodule:: orchestrator.executor

Overview
--------

The Orchestrator execution engine provides multiple execution strategies to optimize performance and security:

**Execution Modes:**
- **Sequential**: Execute tasks one at a time in dependency order
- **Parallel**: Execute independent tasks concurrently 
- **Sandboxed**: Execute code in isolated environments for security

**Key Features:**
- **Task Parallelization**: Automatic parallel execution of independent tasks
- **Resource Management**: CPU and memory resource allocation
- **Error Handling**: Robust error recovery and retry mechanisms
- **Security**: Sandboxed execution with configurable permissions
- **Performance Monitoring**: Real-time execution metrics

**Usage Pattern:**

.. code-block:: python

    from orchestrator.executor.parallel_executor import ParallelExecutor
    
    # Configure parallel execution
    executor = ParallelExecutor(
        max_workers=8,
        max_concurrent_tasks=20
    )
    
    # Execute tasks
    results = await executor.execute_tasks(tasks)

Parallel Executor
-----------------

The ParallelExecutor manages concurrent task execution while respecting dependencies and resource constraints.

**Key Capabilities:**
- **Dependency Resolution**: Automatically orders tasks based on dependencies
- **Concurrent Execution**: Runs independent tasks in parallel
- **Resource Throttling**: Limits CPU and memory usage
- **Error Isolation**: Prevents failures from affecting other tasks

**Example Usage:**

.. code-block:: python

    from orchestrator.executor.parallel_executor import ParallelExecutor
    
    # Create executor with configuration
    executor = ParallelExecutor(
        max_workers=8,
        max_concurrent_tasks=20,
        task_timeout=300,
        resource_limits={
            "max_memory": "4GB",
            "max_cpu_cores": 4
        }
    )
    
    # Execute tasks in parallel
    results = await executor.execute_tasks_parallel(tasks)
    
    # Check results
    for task_id, result in results.items():
        if result.success:
            print(f"Task {task_id} completed: {result.output}")
        else:
            print(f"Task {task_id} failed: {result.error}")

**Classes:**

.. autoclass:: orchestrator.executor.parallel_executor.ParallelExecutor
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: orchestrator.executor.parallel_executor.ExecutionMode
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: orchestrator.executor.parallel_executor.ExecutionConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: orchestrator.executor.parallel_executor.ExecutionResult
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: orchestrator.executor.parallel_executor.TaskQueue
   :members:
   :undoc-members:
   :show-inheritance:

Sandboxed Executor
------------------

The SandboxExecutor provides secure code execution in isolated environments, preventing malicious code from affecting the host system.

**Key Capabilities:**
- **Security Isolation**: Execute code in containerized environments
- **Resource Limits**: Configurable CPU, memory, and time limits
- **Network Isolation**: Control network access for security
- **File System Isolation**: Restrict file system access
- **Multi-Runtime Support**: Support for Docker, processes, and VMs

**Example Usage:**

.. code-block:: python

    from orchestrator.executor.sandboxed_executor import SandboxExecutor, SandboxConfig
    
    # Configure sandbox
    config = SandboxConfig(
        memory_limit="512MB",
        cpu_quota=50000,  # 50% of one CPU
        time_limit=30,    # 30 seconds
        network_disabled=True,
        allowed_packages=["requests", "pandas"]
    )
    
    # Create executor
    executor = SandboxExecutor(config)
    
    # Execute code safely
    code = """
    import pandas as pd
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = df.sum()
    """
    
    result = await executor.execute_code(code, language="python")
    
    if result.success:
        print(f"Code output: {result.output}")
    else:
        print(f"Execution failed: {result.error}")

**Security Features:**
- **Container Isolation**: Docker-based sandboxing
- **Resource Quotas**: Prevent resource exhaustion attacks
- **Network Restrictions**: Block unauthorized network access
- **File System Limits**: Restrict file operations
- **Time Limits**: Prevent infinite loops and hanging processes

**Classes:**

.. autoclass:: orchestrator.executor.sandboxed_executor.SandboxManager
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: orchestrator.executor.sandboxed_executor.SandboxExecutor
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: orchestrator.executor.sandboxed_executor.DockerSandboxExecutor
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: orchestrator.executor.sandboxed_executor.ProcessSandboxExecutor
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: orchestrator.executor.sandboxed_executor.SandboxConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: orchestrator.executor.sandboxed_executor.ExecutionResult
   :members:
   :undoc-members:
   :show-inheritance:

Performance Optimization
-------------------------

The execution engine includes several performance optimization features:

**Optimization Strategies:**
- **Task Batching**: Group similar tasks for efficient execution
- **Resource Pooling**: Reuse expensive resources like model instances
- **Caching**: Cache results to avoid redundant computations
- **Load Balancing**: Distribute work across available resources

**Example Configuration:**

.. code-block:: python

    from orchestrator.executor.optimized_executor import OptimizedExecutor
    
    # Create high-performance executor
    executor = OptimizedExecutor(
        enable_caching=True,
        enable_batching=True,
        batch_size=10,
        cache_ttl=3600,
        resource_pooling=True
    )
    
    # Execute with optimization
    results = await executor.execute_optimized(tasks)

**Monitoring and Metrics:**

.. code-block:: python

    from orchestrator.executor.execution_monitor import ExecutionMonitor
    
    # Monitor execution performance
    monitor = ExecutionMonitor()
    executor = ParallelExecutor(monitor=monitor)
    
    # Get execution metrics
    metrics = monitor.get_metrics()
    print(f"Average execution time: {metrics.avg_execution_time}")
    print(f"Task success rate: {metrics.success_rate}")
    print(f"Resource utilization: {metrics.resource_utilization}")

For detailed performance tuning guides, see the :doc:`../advanced/performance_optimization` documentation.