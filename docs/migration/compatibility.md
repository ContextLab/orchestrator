# Backward Compatibility Guide

This guide explains what remains compatible between the legacy v1 Orchestrator and the refactored architecture, helping you understand what will continue to work without changes during migration.

## Table of Contents

1. [Full Compatibility](#full-compatibility)
2. [Partial Compatibility](#partial-compatibility)
3. [Compatibility Layers](#compatibility-layers)
4. [Migration Strategies](#migration-strategies)
5. [Version Support Policy](#version-support-policy)

## Full Compatibility

These components work exactly the same in both versions without any changes required.

### YAML Pipeline Files ✅

**Status: 100% Compatible**

All existing YAML pipeline files work unchanged in the refactored system:

```yaml
name: "Content Analysis Pipeline"
description: "Analyzes text content for sentiment and topics"
version: "1.0.0"

input_variables:
  text_content:
    type: string
    description: "Text to analyze"

tasks:
  - name: "sentiment_analysis"
    type: "llm_task"
    model: "gpt-3.5-turbo"
    prompt: "Analyze the sentiment of: {{ text_content }}"
    
  - name: "topic_extraction"
    type: "llm_task"
    model: "gpt-4"
    prompt: "Extract main topics from: {{ text_content }}"
    depends_on: ["sentiment_analysis"]

output:
  sentiment: "{{ sentiment_analysis }}"
  topics: "{{ topic_extraction }}"
```

**What works identically:**
- YAML syntax and structure
- Task definitions and types
- Template variables (`{{ variable }}`)
- Dependencies (`depends_on`)
- Input/output specifications
- All built-in task types
- Control flow constructs
- Error handling specifications

### Built-in Task Types ✅

**Status: 100% Compatible**

All built-in task types work identically:

| Task Type | Compatibility | Notes |
|-----------|---------------|--------|
| `llm_task` | ✅ 100% | All parameters and features work |
| `python_task` | ✅ 100% | Script execution unchanged |
| `web_search_task` | ✅ 100% | All search providers supported |
| `file_task` | ✅ 100% | File operations work identically |
| `api_task` | ✅ 100% | HTTP requests unchanged |
| `database_task` | ✅ 100% | All database operations supported |
| `email_task` | ✅ 100% | Email sending works the same |
| `conditional_task` | ✅ 100% | Conditional logic unchanged |
| `loop_task` | ✅ 100% | Loops work identically |

**Example - llm_task compatibility:**
```yaml
# This works identically in both versions
- name: "generate_content"
  type: "llm_task"
  model: "gpt-4"
  prompt: "Write about {{ topic }}"
  max_tokens: 500
  temperature: 0.7
  system_message: "You are a helpful assistant"
  error_handling:
    retry_count: 3
    fallback_response: "Content generation failed"
```

### Template System ✅

**Status: 100% Compatible**

The Jinja2-based template system works identically:

```yaml
# All these template features work unchanged
tasks:
  - name: "templating_example"
    type: "python_task"
    script: |
      # Variable substitution
      user_name = "{{ user_name }}"
      
      # Conditional templates
      {% if premium_user %}
      access_level = "premium"
      {% else %}
      access_level = "basic"
      {% endif %}
      
      # Loop templates
      items = [
      {% for item in item_list %}
        "{{ item }}",
      {% endfor %}
      ]
      
      # Filter usage
      formatted_name = "{{ user_name | upper }}"
      
      return {
          "user": user_name,
          "access": access_level,
          "items": items,
          "formatted": formatted_name
      }
```

**Compatible template features:**
- Variable substitution: `{{ variable }}`
- Conditionals: `{% if condition %}`
- Loops: `{% for item in list %}`
- Filters: `{{ value | filter }}`
- Functions: `{{ function(args) }}`
- Comments: `{# comment #}`

### Pipeline Configuration ✅

**Status: 100% Compatible**

Pipeline-level configuration works unchanged:

```yaml
name: "Data Processing Pipeline"
version: "2.1.0"
description: "Processes customer data"

# All these work identically
timeout: 1800
retry_policy:
  max_retries: 3
  retry_delay: 5
  backoff_multiplier: 2

cache_config:
  enabled: true
  ttl: 3600

parallel_config:
  max_concurrent_tasks: 4
  task_timeout: 300

error_handling:
  on_failure: "continue"
  log_errors: true
  send_alerts: true

monitoring:
  track_performance: true
  collect_metrics: true
```

## Partial Compatibility

These components work but may have limitations or require minor adjustments.

### Custom Task Classes ⚠️

**Status: 80% Compatible with Adapter**

Custom task classes need to use the compatibility adapter:

**v1 Custom Task (Still Works):**
```python
from orchestrator.core.task import Task

class MyCustomTask(Task):
    def execute(self, context):
        # Synchronous implementation
        data = context.get_variable('input_data')
        result = self.process_data(data)
        return {"processed": result}
        
    def process_data(self, data):
        return f"Processed: {data}"
```

**Using with Refactored System:**
```python
from orchestrator.api import PipelineAPI
from orchestrator.compat import LegacyTaskAdapter

# Wrap legacy task in adapter
api = PipelineAPI()
api.register_tool('my_custom_task', LegacyTaskAdapter(MyCustomTask))

# Use in pipeline normally
pipeline = await api.compile_pipeline("""
tasks:
  - name: "process_data"
    type: "my_custom_task"
    input_data: "{{ raw_data }}"
""")
```

**Limitations:**
- Performance overhead from sync/async conversion
- Some advanced context features may not be available
- Debugging may be more complex

### Model Configuration ⚠️

**Status: 90% Compatible**

Most model configurations work, with some new options available:

```yaml
# This works in both versions
tasks:
  - name: "text_generation"
    type: "llm_task"
    model: "gpt-4"                    # ✅ Compatible
    temperature: 0.7                  # ✅ Compatible
    max_tokens: 1000                  # ✅ Compatible
    top_p: 0.9                        # ✅ Compatible
    frequency_penalty: 0.1            # ✅ Compatible
    presence_penalty: 0.1             # ✅ Compatible
    stop_sequences: ["END", "STOP"]   # ✅ Compatible
    
    # New options (ignored in v1, used in refactored)
    model_routing: "cost_optimized"   # 🆕 New feature
    fallback_models: ["gpt-3.5-turbo"] # 🆕 New feature
    caching_strategy: "aggressive"    # 🆕 New feature
```

### Environment Variables ⚠️

**Status: 95% Compatible**

Most environment variables work the same:

| Environment Variable | v1 Support | Refactored Support | Notes |
|---------------------|------------|-------------------|--------|
| `OPENAI_API_KEY` | ✅ | ✅ | Works identically |
| `ANTHROPIC_API_KEY` | ✅ | ✅ | Works identically |
| `ORCHESTRATOR_DEBUG` | ✅ | ⚠️ | Use `ORCHESTRATOR_DEVELOPMENT_MODE` |
| `ORCHESTRATOR_CACHE_DIR` | ✅ | ✅ | Works identically |
| `ORCHESTRATOR_LOG_LEVEL` | ✅ | ⚠️ | Use standard Python logging |
| `ORCHESTRATOR_PARALLEL_WORKERS` | ✅ | ✅ | Works identically |

## Compatibility Layers

The refactored system includes compatibility layers to ease migration.

### Legacy Import Compatibility

**Temporary import compatibility:**
```python
# These imports work but are deprecated
from orchestrator import Orchestrator  # ⚠️ Deprecated
from orchestrator.core.task import Task  # ⚠️ Deprecated

# Use these instead for new code
from orchestrator.api import PipelineAPI
from orchestrator.tools.base_tool import BaseTool
```

### Sync Wrapper Layer

**For gradual async migration:**
```python
from orchestrator.compat import SyncPipelineAPI

# Synchronous interface that wraps async calls
sync_api = SyncPipelineAPI()

# Works like the old API
result = sync_api.execute_pipeline_file(
    "pipeline.yaml",
    inputs={"data": "test"}
)

# Internally uses async API
```

### Configuration Adapter

**For legacy configuration:**
```python
from orchestrator.compat import LegacyConfigAdapter

# Convert old-style config
old_config = {
    "debug": True,
    "enable_parallel_execution": True,
    "max_retries": 3
}

new_config = LegacyConfigAdapter.convert(old_config)
api = PipelineAPI(**new_config)
```

## Migration Strategies

### Strategy 1: Gradual Migration ⭐ Recommended

Migrate components incrementally while maintaining compatibility:

```python
# Phase 1: Use compatibility layer
from orchestrator.compat import SyncPipelineAPI
sync_api = SyncPipelineAPI()

# Phase 2: Move to async gradually
from orchestrator.api import PipelineAPI
import asyncio

def migrate_one_function():
    # Wrap async in sync for now
    return asyncio.run(async_pipeline_execution())

async def async_pipeline_execution():
    api = PipelineAPI()
    pipeline = await api.compile_pipeline("pipeline.yaml")
    return await api.execute_pipeline(pipeline)

# Phase 3: Full async adoption
async def fully_migrated():
    api = PipelineAPI()
    # All async from here
```

### Strategy 2: Side-by-Side Migration

Run both systems in parallel during migration:

```python
import os
from orchestrator.api import PipelineAPI
from orchestrator.compat import SyncPipelineAPI

def execute_pipeline_with_fallback(pipeline_path, inputs):
    """Execute with new API, fallback to old on issues."""
    
    if os.getenv('USE_LEGACY_ORCHESTRATOR'):
        # Use compatibility layer
        sync_api = SyncPipelineAPI()
        return sync_api.execute_pipeline_file(pipeline_path, inputs)
    
    try:
        # Try new async API
        return asyncio.run(execute_with_new_api(pipeline_path, inputs))
    except Exception as e:
        logger.warning(f"New API failed, using legacy: {e}")
        # Fallback to compatibility layer
        sync_api = SyncPipelineAPI()
        return sync_api.execute_pipeline_file(pipeline_path, inputs)

async def execute_with_new_api(pipeline_path, inputs):
    api = PipelineAPI()
    pipeline = await api.compile_pipeline(pipeline_path)
    return await api.execute_pipeline(pipeline, inputs=inputs)
```

### Strategy 3: Feature Flag Migration

Use feature flags to control migration:

```python
import os
from orchestrator.api import PipelineAPI
from orchestrator.compat import SyncPipelineAPI

class OrchestrationService:
    def __init__(self):
        self.use_new_api = os.getenv('ENABLE_REFACTORED_ORCHESTRATOR', 'false').lower() == 'true'
        
        if self.use_new_api:
            self.api = PipelineAPI()
        else:
            self.api = SyncPipelineAPI()
    
    async def execute_pipeline(self, pipeline_path, inputs):
        if self.use_new_api:
            pipeline = await self.api.compile_pipeline(pipeline_path)
            return await self.api.execute_pipeline(pipeline, inputs=inputs)
        else:
            # Sync API call
            return self.api.execute_pipeline_file(pipeline_path, inputs)
```

## Version Support Policy

### Current Support Matrix

| Version | Status | Support Level | End of Life |
|---------|--------|---------------|-------------|
| v1.x | Legacy | Security fixes only | June 2025 |
| v2.x | Current | Full support | TBD |
| v3.x | Future | In development | TBD |

### Deprecation Timeline

**Phase 1 (Now - Dec 2024): Compatibility Layer**
- Full compatibility layer available
- All v1 features work with warnings
- Migration tools provided

**Phase 2 (Jan 2025 - June 2025): Deprecation Warnings**
- Compatibility layer remains but warns
- New features only in v2+ API
- Migration strongly encouraged

**Phase 3 (July 2025+): Legacy Removal**
- Compatibility layer removed
- v1 import paths removed
- Only v2+ API supported

### LTS Policy

**Long Term Support versions:**
- v2.0 will be LTS until at least 2026
- Security updates for 3 years
- Bug fixes for 2 years
- Migration support for 1 year

## Testing Compatibility

### Compatibility Test Suite

```python
# test_compatibility.py
import unittest
import asyncio
from orchestrator.api import PipelineAPI
from orchestrator.compat import SyncPipelineAPI

class CompatibilityTest(unittest.TestCase):
    
    def test_yaml_pipeline_compatibility(self):
        """Test that YAML pipelines work in both systems."""
        yaml_content = """
        name: "Test Pipeline"
        tasks:
          - name: "simple_task"
            type: "python_task" 
            script: "return {'result': 42}"
        """
        
        # Test with sync API
        sync_api = SyncPipelineAPI()
        sync_result = sync_api.execute_pipeline_string(yaml_content)
        
        # Test with async API
        async def test_async():
            api = PipelineAPI()
            pipeline = await api.compile_pipeline(yaml_content)
            return await api.execute_pipeline(pipeline)
            
        async_result = asyncio.run(test_async())
        
        # Results should be identical
        self.assertEqual(sync_result, async_result)
    
    def test_custom_task_compatibility(self):
        """Test custom task compatibility layer."""
        from orchestrator.core.task import Task
        from orchestrator.compat import LegacyTaskAdapter
        
        class TestTask(Task):
            def execute(self, context):
                return {"test": "success"}
        
        # Test adapter works
        adapter = LegacyTaskAdapter(TestTask)
        self.assertIsNotNone(adapter)
        
    def test_configuration_compatibility(self):
        """Test configuration compatibility."""
        from orchestrator.compat import LegacyConfigAdapter
        
        old_config = {
            "debug": True,
            "enable_parallel_execution": False
        }
        
        new_config = LegacyConfigAdapter.convert(old_config)
        self.assertIn("development_mode", new_config)
        self.assertTrue(new_config["development_mode"])

if __name__ == "__main__":
    unittest.main()
```

### Pipeline Compatibility Validator

```bash
# Run compatibility validation on your pipelines
python -m orchestrator.compat.validator --pipeline-dir ./pipelines/

# Output:
# ✅ pipeline1.yaml - Fully compatible
# ⚠️  pipeline2.yaml - Compatible with warnings
# ❌ pipeline3.yaml - Requires migration
```

## Best Practices for Compatibility

### DO ✅

1. **Use compatibility layers initially** - Ease into migration
2. **Test both systems side-by-side** - Verify equivalent behavior
3. **Migrate incrementally** - One component at a time
4. **Keep YAML pipelines unchanged** - They're fully compatible
5. **Use feature flags** - Control migration rollout

### DON'T ❌

1. **Rush the migration** - Take time to test thoroughly
2. **Ignore deprecation warnings** - They indicate future breaking changes
3. **Mix v1 and v2 APIs carelessly** - Use compatibility layers
4. **Modify working YAML pipelines** - They don't need changes
5. **Forget to update documentation** - Keep team informed

## Getting Help

### Resources
- **Migration Guide**: [from-v1.md](from-v1.md)
- **Breaking Changes**: [breaking-changes.md](breaking-changes.md) 
- **API Reference**: [../api/core.md](../api/core.md)
- **Community Forum**: GitHub Discussions
- **Support Chat**: Discord/Slack

### Migration Support
- Free migration consultation available
- Professional services for large migrations
- Community volunteers for open source projects

The compatibility layer ensures a smooth transition while you migrate at your own pace. Take advantage of the gradual migration path to minimize risk and ensure thorough testing.