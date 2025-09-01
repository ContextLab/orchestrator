# Troubleshooting Guide for Refactored Orchestrator

This comprehensive guide helps you diagnose and resolve common issues in the refactored Orchestrator system. Use this guide to quickly identify problems and implement solutions.

## Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Common Issues](#common-issues)
3. [Migration Problems](#migration-problems)
4. [API Issues](#api-issues)
5. [Pipeline Compilation Errors](#pipeline-compilation-errors)
6. [Pipeline Execution Errors](#pipeline-execution-errors)
7. [Performance Issues](#performance-issues)
8. [Tool and Task Problems](#tool-and-task-problems)
9. [Model Integration Issues](#model-integration-issues)
10. [Debugging Techniques](#debugging-techniques)

## Quick Diagnostics

### System Health Check

Run this diagnostic script to quickly identify issues:

```python
#!/usr/bin/env python3
"""Orchestrator System Health Check"""

import asyncio
import sys
import os
from datetime import datetime
from pathlib import Path

async def health_check():
    """Comprehensive system health check."""
    
    print("🔍 Orchestrator System Health Check")
    print("=" * 50)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    # Python version check
    python_version = sys.version_info
    print(f"Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    if python_version < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    print("✅ Python version OK")
    
    # Import test
    try:
        from orchestrator.api import PipelineAPI
        print("✅ Core imports successful")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # API instantiation test
    try:
        api = PipelineAPI(development_mode=True)
        print("✅ API instantiation successful")
    except Exception as e:
        print(f"❌ API instantiation failed: {e}")
        return False
    
    # Basic pipeline compilation test
    try:
        test_yaml = """
        name: "Health Check Pipeline"
        tasks:
          - name: "test_task"
            type: "python_task"
            script: "return {'status': 'healthy'}"
        """
        
        pipeline = await api.compile_pipeline(test_yaml)
        print("✅ Pipeline compilation successful")
    except Exception as e:
        print(f"❌ Pipeline compilation failed: {e}")
        return False
    
    # Basic pipeline execution test
    try:
        result = await api.execute_pipeline(pipeline)
        if result and result.get('test_task', {}).get('status') == 'healthy':
            print("✅ Pipeline execution successful")
        else:
            print(f"⚠️ Pipeline execution completed but unexpected result: {result}")
    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        return False
    
    # Environment variables check
    print("\n🔧 Environment Variables:")
    env_vars = [
        'OPENAI_API_KEY',
        'ANTHROPIC_API_KEY', 
        'ORCHESTRATOR_DEVELOPMENT_MODE',
        'ORCHESTRATOR_CACHE_DIR'
    ]
    
    for var in env_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {'*' * min(len(value), 8)}...")
        else:
            print(f"⚪ {var}: Not set")
    
    print("\n✅ System health check completed successfully!")
    return True

if __name__ == "__main__":
    try:
        result = asyncio.run(health_check())
        sys.exit(0 if result else 1)
    except Exception as e:
        print(f"❌ Health check failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
```

Save and run:
```bash
python health_check.py
```

## Common Issues

### Issue: "ModuleNotFoundError: No module named 'orchestrator'"

**Symptoms:**
```
ImportError: No module named 'orchestrator'
ModuleNotFoundError: No module named 'orchestrator.api'
```

**Solutions:**

1. **Install/Reinstall Orchestrator:**
```bash
pip install --upgrade orchestrator-framework
# or if from source
pip install -e .
```

2. **Check Virtual Environment:**
```bash
which python
which pip
pip list | grep orchestrator
```

3. **Python Path Issues:**
```python
import sys
print(sys.path)
# Add project directory if needed
sys.path.insert(0, '/path/to/orchestrator')
```

### Issue: "AsyncioRuntimeError: asyncio.run() cannot be called from a running event loop"

**Symptoms:**
```
RuntimeError: asyncio.run() cannot be called from a running event loop
```

**Solutions:**

1. **Use await instead of asyncio.run():**
```python
# ❌ Don't do this inside async context
async def my_function():
    result = asyncio.run(api.execute_pipeline(pipeline))

# ✅ Do this instead
async def my_function():
    result = await api.execute_pipeline(pipeline)
```

2. **Use sync wrapper when needed:**
```python
from orchestrator.compat import SyncPipelineAPI

def sync_function():
    sync_api = SyncPipelineAPI()
    result = sync_api.execute_pipeline_file("pipeline.yaml")
    return result
```

3. **Get current event loop:**
```python
import asyncio

def run_in_existing_loop():
    loop = asyncio.get_event_loop()
    task = loop.create_task(api.execute_pipeline(pipeline))
    return task
```

### Issue: "Pipeline validation failed" Errors

**Symptoms:**
```
PipelineCompilationError: Pipeline validation failed
ValidationError: Invalid task configuration
```

**Solutions:**

1. **Check YAML Syntax:**
```bash
# Use YAML validator
python -c "import yaml; yaml.safe_load(open('pipeline.yaml'))"
```

2. **Use Development Mode:**
```python
api = PipelineAPI(
    development_mode=True,
    validation_level="permissive"
)
```

3. **Validate Individual Components:**
```python
# Check task syntax
from orchestrator.core.foundation.validation import TaskValidator

validator = TaskValidator()
errors = validator.validate_task(task_config)
if errors:
    print("Validation errors:", errors)
```

## Migration Problems

### Issue: "AttributeError: 'Orchestrator' object has no attribute..."

**Symptoms:**
```
AttributeError: 'Orchestrator' object has no attribute 'execute_pipeline_from_file'
```

**Cause:** Using legacy v1 API calls with refactored system.

**Solution:** Update to new API pattern:

```python
# ❌ Old way
from orchestrator import Orchestrator
orchestrator = Orchestrator()
result = orchestrator.execute_pipeline_from_file("pipeline.yaml")

# ✅ New way  
from orchestrator.api import PipelineAPI
import asyncio

async def main():
    api = PipelineAPI()
    pipeline = await api.compile_pipeline("pipeline.yaml")
    result = await api.execute_pipeline(pipeline)
    return result

result = asyncio.run(main())
```

### Issue: Custom Tasks Not Working After Migration

**Symptoms:**
```
TypeError: 'MyCustomTask' object is not callable
AttributeError: 'MyCustomTask' object has no attribute 'execute'
```

**Solutions:**

1. **Use Legacy Task Adapter:**
```python
from orchestrator.compat import LegacyTaskAdapter
from mymodule import MyCustomTask

api = PipelineAPI()
api.register_tool('my_task', LegacyTaskAdapter(MyCustomTask))
```

2. **Convert to New Tool Format:**
```python
from orchestrator.tools.base_tool import BaseTool

class MyCustomTool(BaseTool):
    async def execute(self, context):
        # Convert your old sync execute method
        result = self.old_execute_method(context)
        return result
        
    def old_execute_method(self, context):
        # Your original logic here
        return {"result": "success"}

api.register_tool('my_task', MyCustomTool)
```

### Issue: Configuration Parameters Not Recognized

**Symptoms:**
```
TypeError: PipelineAPI.__init__() got an unexpected keyword argument 'debug'
```

**Solution:** Update configuration parameters:

```python
# ❌ Old configuration
api = PipelineAPI(
    debug=True,
    enable_parallel_execution=True,
    max_retries=3
)

# ✅ New configuration
api = PipelineAPI(
    development_mode=True,  # Replaces debug
)

# Execution options go in config
config = {
    "parallel_execution": True,
    "max_retries": 3
}

result = await api.execute_pipeline(pipeline, config=config)
```

## API Issues

### Issue: "PipelineAPI object has no attribute 'run_pipeline'"

**Symptoms:**
```
AttributeError: 'PipelineAPI' object has no attribute 'run_pipeline'
```

**Solution:** Use correct method names:

```python
# ❌ Wrong method names
result = api.run_pipeline(pipeline)
pipeline = api.load_pipeline("file.yaml")

# ✅ Correct method names
pipeline = await api.compile_pipeline("file.yaml")
result = await api.execute_pipeline(pipeline)
```

### Issue: Type Errors with Pipeline Objects

**Symptoms:**
```
TypeError: argument of type 'Pipeline' is not iterable
TypeError: expected str, got Pipeline
```

**Solution:** Understand Pipeline vs. string distinction:

```python
# Pipeline compilation returns Pipeline object
pipeline = await api.compile_pipeline("pipeline.yaml")  # Returns Pipeline

# Don't try to use Pipeline object as string
# ❌ Wrong
result = await api.execute_pipeline("pipeline.yaml")

# ✅ Correct
result = await api.execute_pipeline(pipeline)
```

## Pipeline Compilation Errors

### Issue: "Template rendering failed"

**Symptoms:**
```
TemplateRenderingError: Undefined variable 'variable_name'
jinja2.exceptions.UndefinedError: 'variable_name' is undefined
```

**Solutions:**

1. **Check Variable Definitions:**
```yaml
# Make sure all variables are defined
input_variables:
  required_var:
    type: string
    description: "This variable is required"

tasks:
  - name: "use_variable"
    type: "python_task"
    script: "return {'value': '{{ required_var }}'}"
```

2. **Use Default Values:**
```yaml
input_variables:
  optional_var:
    type: string
    default: "default_value"
```

3. **Debug Template Rendering:**
```python
# Enable template debugging
api = PipelineAPI(development_mode=True)

# Check available variables
print("Available variables:", context.get_all_variables())
```

### Issue: "Circular dependency detected"

**Symptoms:**
```
CircularDependencyError: Circular dependency detected in tasks
DependencyError: Task 'task_a' depends on 'task_b' which depends on 'task_a'
```

**Solutions:**

1. **Check Task Dependencies:**
```yaml
tasks:
  - name: "task_a"
    depends_on: ["task_b"]  # ❌ Creates circular dependency
    
  - name: "task_b" 
    depends_on: ["task_a"]  # ❌ Creates circular dependency
```

2. **Use Dependency Visualization:**
```python
from orchestrator.core.foundation.visualization import DependencyGraph

graph = DependencyGraph(pipeline)
graph.visualize()  # Shows dependency tree
graph.detect_cycles()  # Returns circular dependencies
```

3. **Restructure Dependencies:**
```yaml
tasks:
  - name: "base_task"
    type: "python_task"
    script: "return {'base': 'data'}"
    
  - name: "task_a"
    type: "python_task" 
    depends_on: ["base_task"]
    script: "return {'a': '{{ base_task.base }}'}"
    
  - name: "task_b"
    type: "python_task"
    depends_on: ["task_a"]
    script: "return {'b': '{{ task_a.a }}'}"
```

## Pipeline Execution Errors

### Issue: Task Timeout Errors

**Symptoms:**
```
TaskTimeoutError: Task 'long_running_task' exceeded timeout of 300 seconds
asyncio.TimeoutError: Task timed out
```

**Solutions:**

1. **Increase Task Timeout:**
```yaml
tasks:
  - name: "long_task"
    type: "llm_task"
    timeout: 600  # 10 minutes
    model: "gpt-4"
    prompt: "This is a complex task..."
```

2. **Configure Global Timeouts:**
```python
config = {
    "task_timeout": 900,      # 15 minutes per task
    "pipeline_timeout": 3600  # 1 hour total
}

result = await api.execute_pipeline(pipeline, config=config)
```

3. **Add Progress Monitoring:**
```yaml
- name: "monitored_task"
  type: "python_task"
  script: |
    import time
    for i in range(10):
        print(f"Progress: {i*10}%")
        context.update_progress(i * 10)
        time.sleep(5)  # Simulate work
    return {"completed": True}
```

### Issue: Memory Errors During Execution

**Symptoms:**
```
MemoryError: Unable to allocate memory
ResourceExhaustionError: System memory exhausted
```

**Solutions:**

1. **Set Memory Limits:**
```python
config = {
    "memory_limit": "2GB",
    "cleanup_intermediate": True
}

result = await api.execute_pipeline(pipeline, config=config)
```

2. **Process Data in Chunks:**
```yaml
- name: "chunked_processing"
  type: "python_task"
  script: |
    import gc
    
    data = context.get_variable('large_dataset')
    chunk_size = 1000
    results = []
    
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        chunk_result = process_chunk(chunk)
        results.append(chunk_result)
        
        # Clean up memory
        del chunk
        gc.collect()
    
    return {"results": results}
```

3. **Monitor Memory Usage:**
```python
import psutil
import os

def monitor_memory():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory_mb:.1f} MB")
    return memory_mb
```

## Performance Issues

### Issue: Slow Pipeline Compilation

**Symptoms:**
- Pipeline compilation takes several minutes
- High CPU usage during compilation

**Solutions:**

1. **Use Caching:**
```python
api = PipelineAPI()
api.enable_compilation_cache(ttl=3600)  # Cache for 1 hour

# Compiled pipelines are cached automatically
pipeline = await api.compile_pipeline("pipeline.yaml")
```

2. **Optimize Pipeline Structure:**
```yaml
# ❌ Avoid deeply nested structures
tasks:
  - name: "complex_task"
    type: "python_task" 
    script: |
      # 1000+ lines of complex logic here

# ✅ Break into smaller tasks
tasks:
  - name: "prepare_data"
    type: "python_task"
    script: "return prepare_data_function()"
    
  - name: "process_data"
    type: "python_task"
    depends_on: ["prepare_data"]
    script: "return process_data_function({{ prepare_data }})"
```

3. **Profile Compilation:**
```python
import time
from orchestrator.api import PipelineAPI

start_time = time.time()
api = PipelineAPI(development_mode=True)
compilation_start = time.time()

pipeline = await api.compile_pipeline("pipeline.yaml")
compilation_end = time.time()

print(f"API setup: {compilation_start - start_time:.2f}s")
print(f"Compilation: {compilation_end - compilation_start:.2f}s")
```

### Issue: Slow Pipeline Execution

**Symptoms:**
- Pipeline execution is much slower than expected
- Tasks take longer than they should

**Solutions:**

1. **Enable Parallel Execution:**
```python
config = {
    "parallel_execution": True,
    "max_concurrent_tasks": 4
}

result = await api.execute_pipeline(pipeline, config=config)
```

2. **Optimize Task Dependencies:**
```yaml
# ❌ Sequential execution (slow)
tasks:
  - name: "task_1"
    type: "llm_task"
    
  - name: "task_2"
    type: "llm_task"
    depends_on: ["task_1"]  # Unnecessary dependency
    
  - name: "task_3" 
    type: "llm_task"
    depends_on: ["task_2"]  # Unnecessary dependency

# ✅ Parallel execution (fast)
tasks:
  - name: "task_1"
    type: "llm_task"
    
  - name: "task_2"
    type: "llm_task"  # No dependency - runs in parallel
    
  - name: "task_3"
    type: "llm_task"  # No dependency - runs in parallel
```

3. **Use Performance Monitoring:**
```python
from orchestrator.core.foundation.monitoring import PerformanceMonitor

monitor = PerformanceMonitor()
api.add_monitor(monitor)

result = await api.execute_pipeline(pipeline)

# Get performance metrics
metrics = monitor.get_metrics()
print(f"Total execution time: {metrics.total_time}")
print(f"Task breakdown: {metrics.task_times}")
```

## Tool and Task Problems

### Issue: "Tool 'tool_name' not found"

**Symptoms:**
```
ToolNotFoundError: Tool 'my_custom_tool' not found
KeyError: 'unknown_task_type'
```

**Solutions:**

1. **Check Tool Registration:**
```python
# List available tools
api = PipelineAPI()
print("Available tools:", api.list_tools())

# Register custom tool
api.register_tool('my_custom_tool', MyCustomTool)
```

2. **Check Task Type in YAML:**
```yaml
tasks:
  - name: "my_task"
    type: "my_custom_tool"  # Must match registered name
```

3. **Use Built-in Tools:**
```yaml
# List of always-available built-in tools
tasks:
  - name: "python_task"
    type: "python_task"
    
  - name: "llm_task" 
    type: "llm_task"
    
  - name: "web_search_task"
    type: "web_search_task"
    
  - name: "file_task"
    type: "file_task"
```

### Issue: Tool Execution Failures

**Symptoms:**
```
ToolExecutionError: Tool 'llm_task' failed to execute
APIError: Model request failed
```

**Solutions:**

1. **Check API Keys:**
```bash
# Verify environment variables
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
```

2. **Add Error Handling to Tools:**
```yaml
- name: "robust_llm_task"
  type: "llm_task"
  model: "gpt-4"
  prompt: "{{ user_prompt }}"
  error_handling:
    retry_count: 3
    retry_delay: 5
    fallback:
      type: "llm_task"
      model: "gpt-3.5-turbo"
      prompt: "{{ user_prompt }}"
```

3. **Debug Tool Inputs:**
```python
# Enable tool debugging
api = PipelineAPI(development_mode=True)

class DebuggingTool(BaseTool):
    async def execute(self, context):
        print(f"Tool inputs: {context.get_all_variables()}")
        # Your tool logic here
        return {"result": "success"}
```

## Model Integration Issues

### Issue: Model Authentication Failures

**Symptoms:**
```
AuthenticationError: Invalid API key
PermissionError: Access denied to model
```

**Solutions:**

1. **Verify API Keys:**
```python
import os

# Check if keys are set
api_keys = {
    'OpenAI': os.getenv('OPENAI_API_KEY'),
    'Anthropic': os.getenv('ANTHROPIC_API_KEY'),
    'Google': os.getenv('GOOGLE_API_KEY')
}

for provider, key in api_keys.items():
    if key:
        print(f"✅ {provider}: {'*' * min(8, len(key))}...")
    else:
        print(f"❌ {provider}: Not set")
```

2. **Test API Connectivity:**
```python
async def test_model_access():
    api = PipelineAPI()
    
    test_pipeline = """
    name: "Model Test"
    tasks:
      - name: "test_openai"
        type: "llm_task"
        model: "gpt-3.5-turbo"
        prompt: "Say hello"
    """
    
    try:
        pipeline = await api.compile_pipeline(test_pipeline)
        result = await api.execute_pipeline(pipeline)
        print("✅ Model access successful")
        return True
    except Exception as e:
        print(f"❌ Model access failed: {e}")
        return False
```

3. **Use Model Fallbacks:**
```yaml
- name: "resilient_task"
  type: "llm_task"
  model: "gpt-4"
  prompt: "{{ user_prompt }}"
  fallback_models:
    - "gpt-3.5-turbo"
    - "claude-3-haiku"
  error_handling:
    retry_count: 2
```

### Issue: Model Rate Limiting

**Symptoms:**
```
RateLimitError: Rate limit exceeded for model
TooManyRequestsError: API quota exceeded
```

**Solutions:**

1. **Add Rate Limiting:**
```python
config = {
    "rate_limiting": {
        "requests_per_minute": 60,
        "tokens_per_minute": 90000
    }
}

result = await api.execute_pipeline(pipeline, config=config)
```

2. **Use Request Batching:**
```yaml
- name: "batch_requests"
  type: "python_task"
  script: |
    import asyncio
    
    requests = context.get_variable('requests')
    batch_size = 5
    results = []
    
    for i in range(0, len(requests), batch_size):
        batch = requests[i:i+batch_size]
        # Process batch with delay
        batch_results = await process_batch(batch)
        results.extend(batch_results)
        
        # Wait between batches
        if i + batch_size < len(requests):
            await asyncio.sleep(1)
    
    return {"results": results}
```

3. **Monitor Usage:**
```python
from orchestrator.core.foundation.monitoring import UsageMonitor

usage_monitor = UsageMonitor()
api.add_monitor(usage_monitor)

result = await api.execute_pipeline(pipeline)

# Check usage
usage = usage_monitor.get_usage()
print(f"Tokens used: {usage.total_tokens}")
print(f"API calls made: {usage.api_calls}")
```

## Debugging Techniques

### Enable Comprehensive Logging

```python
import logging

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Enable orchestrator-specific logging
orchestrator_logger = logging.getLogger('orchestrator')
orchestrator_logger.setLevel(logging.DEBUG)

api = PipelineAPI(development_mode=True)
```

### Use Interactive Debugging

```python
# Add breakpoints to your tools
from orchestrator.tools.base_tool import BaseTool

class DebuggingTool(BaseTool):
    async def execute(self, context):
        import pdb; pdb.set_trace()  # Breakpoint
        
        # Your code here
        result = {"debug": "info"}
        return result
```

### Pipeline State Inspection

```python
# Inspect pipeline state during execution
class StateInspector:
    def __init__(self, api):
        self.api = api
        
    async def execute_with_inspection(self, pipeline):
        # Add state inspection callbacks
        def on_task_start(task_name):
            print(f"Starting task: {task_name}")
            
        def on_task_complete(task_name, result):
            print(f"Completed task: {task_name}")
            print(f"Result: {result}")
            
        self.api.add_callback('task_start', on_task_start)
        self.api.add_callback('task_complete', on_task_complete)
        
        return await self.api.execute_pipeline(pipeline)
```

### Performance Profiling

```python
import cProfile
import pstats

def profile_pipeline_execution():
    """Profile pipeline execution for performance bottlenecks."""
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Your pipeline execution code
    result = asyncio.run(api.execute_pipeline(pipeline))
    
    profiler.disable()
    
    # Print profiling results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
    
    return result
```

### Error Context Preservation

```python
async def execute_with_context_preservation(api, pipeline):
    """Execute pipeline while preserving error context."""
    
    try:
        result = await api.execute_pipeline(pipeline)
        return result
    except Exception as e:
        # Preserve full context
        import traceback
        error_context = {
            'exception': str(e),
            'traceback': traceback.format_exc(),
            'pipeline_state': api.get_pipeline_state(),
            'system_info': api.get_system_info()
        }
        
        # Log detailed error context
        logging.error(f"Pipeline execution failed with context: {error_context}")
        
        # Re-raise with additional context
        raise type(e)(f"{str(e)}\nContext: {error_context}") from e
```

## Getting Additional Help

### Documentation Resources
- **API Reference**: [docs/api/core.md](api/core.md)
- **Migration Guide**: [docs/migration/from-v1.md](migration/from-v1.md) 
- **Best Practices**: [docs/tutorials/best-practices.md](tutorials/best-practices.md)

### Community Support
- **GitHub Issues**: Report bugs and get help
- **Discord/Slack**: Real-time community support
- **Stack Overflow**: Tag questions with `orchestrator`

### Professional Support
- **Consulting Services**: For complex migrations
- **Priority Support**: For enterprise users
- **Training Programs**: Hands-on learning sessions

This troubleshooting guide should help you resolve most issues quickly. If you encounter problems not covered here, please file an issue on GitHub with full error details and system information.