# Migration Guide: From v1 to Refactored Architecture

This guide helps you migrate from the legacy Orchestrator v1 architecture to the new refactored system. The refactor introduces significant improvements in modularity, maintainability, and performance while maintaining backward compatibility for most use cases.

## Table of Contents

1. [Overview of Changes](#overview-of-changes)
2. [Pre-Migration Assessment](#pre-migration-assessment)
3. [Step-by-Step Migration](#step-by-step-migration)
4. [Code Migration Examples](#code-migration-examples)
5. [Common Migration Issues](#common-migration-issues)
6. [Testing Your Migration](#testing-your-migration)
7. [Rollback Strategy](#rollback-strategy)

## Overview of Changes

### Architecture Transformation

The refactor moves from a monolithic `Orchestrator` class to a modular, API-driven architecture:

**v1 Architecture (Legacy):**
```
orchestrator.py (4,000+ lines)
├── Direct class instantiation
├── Tightly coupled components
├── Mixed responsibilities
└── Hard to extend/test
```

**New Architecture (Refactored):**
```
api/
├── PipelineAPI (Main entry point)
├── Core Foundation (30+ specialized modules)
├── Execution Engine (Modular task execution)
├── Tools System (40+ specialized tools)
└── Variables System (Advanced templating)
```

### Key Improvements

1. **Modular Design**: Components are now independent and testable
2. **Clean APIs**: Well-defined interfaces between components
3. **Better Error Handling**: Comprehensive error recovery and reporting  
4. **Performance**: Optimized execution paths and resource management
5. **Extensibility**: Easy to add new task types and capabilities

## Pre-Migration Assessment

### Compatibility Check

Run this assessment script to analyze your current setup:

```python
# migration_assessment.py
import os
import sys
from pathlib import Path

def assess_migration_requirements():
    """Assess your current setup for migration readiness."""
    
    print("🔍 Orchestrator Migration Assessment")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    if python_version < (3, 8):
        print("❌ Python 3.8+ required. Please upgrade.")
        return False
    else:
        print("✅ Python version compatible")
    
    # Check current orchestrator installation
    try:
        import orchestrator
        print(f"✅ Current orchestrator version: {getattr(orchestrator, '__version__', 'unknown')}")
    except ImportError:
        print("❌ Orchestrator not found. Please install first.")
        return False
    
    # Analyze pipeline files
    pipeline_files = list(Path(".").rglob("*.yaml")) + list(Path(".").rglob("*.yml"))
    print(f"📄 Found {len(pipeline_files)} YAML files")
    
    # Check for legacy patterns
    legacy_patterns = [
        "orchestrator = Orchestrator(",
        "from orchestrator import Orchestrator",
        ".execute_pipeline_from_file(",
        ".run_pipeline(",
    ]
    
    legacy_usage = []
    for file_path in Path(".").rglob("*.py"):
        try:
            content = file_path.read_text(encoding='utf-8')
            for pattern in legacy_patterns:
                if pattern in content:
                    legacy_usage.append(f"{file_path}: {pattern}")
        except Exception:
            continue
    
    if legacy_usage:
        print(f"⚠️  Found {len(legacy_usage)} legacy usage patterns:")
        for usage in legacy_usage[:5]:  # Show first 5
            print(f"   - {usage}")
        if len(legacy_usage) > 5:
            print(f"   ... and {len(legacy_usage) - 5} more")
    else:
        print("✅ No legacy usage patterns detected")
    
    return True

if __name__ == "__main__":
    assess_migration_requirements()
```

Run the assessment:
```bash
python migration_assessment.py
```

## Step-by-Step Migration

### Step 1: Install the Refactored Version

```bash
# Backup current version
pip freeze > requirements_backup.txt

# Install refactored orchestrator
pip install --upgrade orchestrator-framework

# Verify installation
python -c "from orchestrator.api import PipelineAPI; print('✅ Installation successful')"
```

### Step 2: Update Import Statements

**Old imports (v1):**
```python
from orchestrator import Orchestrator
from orchestrator.core.task import Task
from orchestrator.models.model_registry import ModelRegistry
```

**New imports (Refactored):**
```python
from orchestrator.api import PipelineAPI
from orchestrator.core.foundation.task import Task
from orchestrator.core.foundation.model_registry import ModelRegistry
```

### Step 3: Update Pipeline Execution Code

**Old pattern (v1):**
```python
from orchestrator import Orchestrator

# Legacy instantiation
orchestrator = Orchestrator(
    model_registry=model_registry,
    debug=True,
    enable_parallel_execution=True
)

# Legacy execution
result = orchestrator.execute_pipeline_from_file(
    "my_pipeline.yaml",
    input_variables={"topic": "AI ethics"}
)
```

**New pattern (Refactored):**
```python
from orchestrator.api import PipelineAPI
import asyncio

async def main():
    # New API instantiation
    api = PipelineAPI(
        model_registry=model_registry,
        development_mode=True,
        validation_level="strict"
    )
    
    # New execution pattern
    pipeline = await api.compile_pipeline("my_pipeline.yaml")
    result = await api.execute_pipeline(
        pipeline,
        inputs={"topic": "AI ethics"}
    )
    return result

# Run async
result = asyncio.run(main())
```

### Step 4: Update Configuration

**Old configuration (v1):**
```python
orchestrator = Orchestrator(
    debug=True,
    enable_parallel_execution=True,
    max_retries=3,
    model_fallback=True,
    cache_enabled=True
)
```

**New configuration (Refactored):**
```python
api = PipelineAPI(
    development_mode=True,  # Replaces debug=True
    validation_level="permissive",  # For development
)

# Configure execution options
execution_config = {
    "parallel_execution": True,
    "max_retries": 3,
    "enable_fallback": True,
    "cache_enabled": True
}

result = await api.execute_pipeline(
    pipeline, 
    inputs=inputs,
    config=execution_config
)
```

### Step 5: Update Error Handling

**Old error handling (v1):**
```python
try:
    result = orchestrator.run_pipeline(pipeline_path)
except Exception as e:
    logger.error(f"Pipeline failed: {e}")
```

**New error handling (Refactored):**
```python
from orchestrator.core.foundation.exceptions import (
    PipelineCompilationError,
    PipelineExecutionError,
    TaskExecutionError
)

try:
    pipeline = await api.compile_pipeline(pipeline_path)
    result = await api.execute_pipeline(pipeline)
except PipelineCompilationError as e:
    logger.error(f"Pipeline compilation failed: {e}")
    # Handle compilation errors (syntax, validation, etc.)
except PipelineExecutionError as e:
    logger.error(f"Pipeline execution failed: {e}")
    # Handle runtime errors
except TaskExecutionError as e:
    logger.error(f"Task failed: {e.task_name} - {e}")
    # Handle specific task failures
```

## Code Migration Examples

### Example 1: Basic Pipeline Execution

**Before (v1):**
```python
#!/usr/bin/env python3
import logging
from orchestrator import Orchestrator
from orchestrator.models.model_registry import ModelRegistry

# Setup logging
logging.basicConfig(level=logging.INFO)

# Initialize
model_registry = ModelRegistry()
orchestrator = Orchestrator(
    model_registry=model_registry,
    debug=True
)

# Execute
result = orchestrator.execute_pipeline_from_file(
    "content_analysis.yaml",
    input_variables={
        "content": "Sample text to analyze",
        "analysis_type": "sentiment"
    }
)

print(f"Analysis result: {result}")
```

**After (Refactored):**
```python
#!/usr/bin/env python3
import asyncio
import logging
from orchestrator.api import PipelineAPI
from orchestrator.core.foundation.model_registry import ModelRegistry

# Setup logging
logging.basicConfig(level=logging.INFO)

async def main():
    # Initialize
    model_registry = ModelRegistry()
    api = PipelineAPI(
        model_registry=model_registry,
        development_mode=True
    )
    
    # Compile and execute
    pipeline = await api.compile_pipeline("content_analysis.yaml")
    result = await api.execute_pipeline(
        pipeline,
        inputs={
            "content": "Sample text to analyze", 
            "analysis_type": "sentiment"
        }
    )
    
    print(f"Analysis result: {result}")
    return result

if __name__ == "__main__":
    result = asyncio.run(main())
```

### Example 2: Custom Task Implementation

**Before (v1):**
```python
from orchestrator.core.task import Task

class CustomAnalysisTask(Task):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def execute(self, context):
        # Custom implementation
        data = context.get_variable('input_data')
        result = self.analyze_data(data)
        return {"analysis": result}

# Register with orchestrator
orchestrator.register_custom_task('custom_analysis', CustomAnalysisTask)
```

**After (Refactored):**
```python
from orchestrator.core.foundation.task import Task
from orchestrator.tools.base_tool import BaseTool

class CustomAnalysisTool(BaseTool):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    async def execute(self, context):
        # Custom implementation
        data = context.get_variable('input_data')
        result = await self.analyze_data(data)  # Now async
        return {"analysis": result}
        
    async def analyze_data(self, data):
        # Your analysis logic here
        return f"Analyzed: {data}"

# Register with API
api = PipelineAPI()
api.register_tool('custom_analysis', CustomAnalysisTool)
```

### Example 3: Pipeline Monitoring

**Before (v1):**
```python
# Limited monitoring in v1
def monitor_pipeline():
    orchestrator.set_debug(True)  # Basic debugging
    result = orchestrator.run_pipeline("pipeline.yaml")
    return result
```

**After (Refactored):**
```python
from orchestrator.api import PipelineAPI
from orchestrator.core.foundation.monitoring import PipelineMonitor

async def monitor_pipeline():
    api = PipelineAPI()
    
    # Set up monitoring
    monitor = PipelineMonitor()
    api.add_monitor(monitor)
    
    # Execute with monitoring
    pipeline = await api.compile_pipeline("pipeline.yaml")
    result = await api.execute_pipeline(pipeline)
    
    # Access detailed metrics
    metrics = monitor.get_metrics()
    print(f"Execution time: {metrics.execution_time}")
    print(f"Tasks executed: {metrics.tasks_completed}")
    print(f"Errors encountered: {metrics.error_count}")
    
    return result
```

## Common Migration Issues

### Issue 1: Synchronous to Asynchronous

**Problem:** The new API is async-first, but your code is synchronous.

**Solution:** Wrap in async functions or use sync utilities:

```python
# Option 1: Async wrapper
import asyncio
from orchestrator.api import PipelineAPI

def sync_execute_pipeline(pipeline_path, inputs=None):
    """Sync wrapper for async pipeline execution."""
    async def _execute():
        api = PipelineAPI()
        pipeline = await api.compile_pipeline(pipeline_path)
        return await api.execute_pipeline(pipeline, inputs=inputs)
    
    return asyncio.run(_execute())

# Option 2: Use sync utilities (if available)
from orchestrator.api.sync import SyncPipelineAPI

sync_api = SyncPipelineAPI()
result = sync_api.execute_pipeline_file("pipeline.yaml", inputs)
```

### Issue 2: Changed Import Paths

**Problem:** Import errors due to module reorganization.

**Solution:** Use the migration mapping:

```python
# Common migration mappings
IMPORT_MIGRATIONS = {
    "orchestrator.Orchestrator": "orchestrator.api.PipelineAPI",
    "orchestrator.core.task.Task": "orchestrator.core.foundation.task.Task",
    "orchestrator.models.model_registry.ModelRegistry": "orchestrator.core.foundation.model_registry.ModelRegistry",
    "orchestrator.core.exceptions": "orchestrator.core.foundation.exceptions",
    "orchestrator.utils": "orchestrator.core.foundation.utils",
}
```

### Issue 3: Configuration Changes

**Problem:** Old configuration options don't work.

**Solution:** Use the configuration mapping:

```python
# Old -> New configuration mapping
CONFIG_MIGRATIONS = {
    "debug": "development_mode",
    "enable_parallel_execution": ("config", "parallel_execution"),
    "max_retries": ("config", "max_retries"), 
    "model_fallback": ("config", "enable_fallback"),
    "cache_enabled": ("config", "cache_enabled"),
}
```

### Issue 4: Custom Task Registration

**Problem:** Custom task registration API changed.

**Solution:** 

```python
# Old way (v1)
orchestrator.register_custom_task('my_task', MyTaskClass)

# New way (Refactored)
api = PipelineAPI()
api.register_tool('my_task', MyToolClass)  # Tools, not tasks

# Or use the plugin system
from orchestrator.core.foundation.plugins import register_plugin
register_plugin('my_task', MyToolClass)
```

## Testing Your Migration

### Automated Migration Testing

Create a test suite to verify your migration:

```python
# test_migration.py
import unittest
import asyncio
from orchestrator.api import PipelineAPI

class MigrationTest(unittest.TestCase):
    def setUp(self):
        self.api = PipelineAPI(development_mode=True)
        
    def test_basic_pipeline_execution(self):
        """Test basic pipeline execution works."""
        async def run_test():
            pipeline = await self.api.compile_pipeline("test_pipeline.yaml")
            result = await self.api.execute_pipeline(pipeline)
            self.assertIsNotNone(result)
            
        asyncio.run(run_test())
        
    def test_error_handling(self):
        """Test error handling works correctly."""
        from orchestrator.core.foundation.exceptions import PipelineCompilationError
        
        async def run_test():
            with self.assertRaises(PipelineCompilationError):
                await self.api.compile_pipeline("nonexistent_pipeline.yaml")
                
        asyncio.run(run_test())
        
    def test_custom_tools(self):
        """Test custom tool registration."""
        from orchestrator.tools.base_tool import BaseTool
        
        class TestTool(BaseTool):
            async def execute(self, context):
                return {"test": "success"}
                
        self.api.register_tool('test_tool', TestTool)
        
        # Test that tool is registered
        self.assertTrue(self.api.has_tool('test_tool'))

if __name__ == "__main__":
    unittest.main()
```

Run your migration tests:
```bash
python test_migration.py
```

### Manual Testing Checklist

- [ ] All pipeline files compile without errors
- [ ] Pipeline execution produces expected results  
- [ ] Custom tasks/tools work correctly
- [ ] Error handling behaves as expected
- [ ] Performance is acceptable
- [ ] Logging and monitoring work
- [ ] Configuration options are respected

## Rollback Strategy

### Emergency Rollback

If migration fails and you need to rollback immediately:

```bash
# 1. Reinstall previous version
pip install orchestrator-framework==1.x.x  # Your previous version

# 2. Restore backed up requirements
pip install -r requirements_backup.txt

# 3. Revert code changes
git checkout HEAD~1  # Or your backup branch

# 4. Verify old system works
python -c "from orchestrator import Orchestrator; print('Rollback successful')"
```

### Gradual Rollback

For a controlled rollback:

1. **Identify Issues**: Document what's not working
2. **Partial Migration**: Keep working parts, revert problematic parts
3. **Gradual Fix**: Address issues one by one
4. **Test Incrementally**: Verify each fix

### Version Pinning

Pin to specific versions during migration:

```bash
# requirements.txt
orchestrator-framework==2.0.0  # Specific version
# Don't use orchestrator-framework>=2.0.0 during migration
```

## Post-Migration Optimization

### Performance Tuning

After successful migration:

```python
# Optimize for your use case
api = PipelineAPI(
    validation_level="permissive",  # Faster compilation
    cache_enabled=True,             # Better performance
    parallel_execution=True,        # Faster execution
)

# Configure resource limits
execution_config = {
    "max_concurrent_tasks": 4,      # Tune for your system
    "memory_limit": "2GB",          # Set appropriate limits
    "timeout": 300,                 # 5 minute timeout
}
```

### Monitoring Setup

Add comprehensive monitoring:

```python
from orchestrator.core.foundation.monitoring import (
    PipelineMonitor,
    MetricsCollector,
    AlertManager
)

# Set up monitoring
monitor = PipelineMonitor()
metrics = MetricsCollector()
alerts = AlertManager()

api.add_monitor(monitor)
api.add_metrics_collector(metrics)
api.add_alert_manager(alerts)
```

## Migration Support

### Getting Help

- **Documentation**: Check the [API Reference](../api/core.md)
- **Examples**: See updated examples in `/docs/tutorials/`  
- **Issues**: Report migration issues on GitHub
- **Community**: Join the Orchestrator Discord/Slack

### Migration Script

Use our automated migration script:

```bash
# Download migration helper
curl -o migrate.py https://raw.githubusercontent.com/.../migrate.py

# Run migration
python migrate.py --source-dir . --target-version 2.0.0
```

This migration guide should help you successfully transition from the legacy v1 architecture to the new refactored system. Take it step by step, test thoroughly, and don't hesitate to ask for help if you encounter issues.