# Migration Guide: Legacy to LangGraph State Management

This guide helps you migrate from the legacy state management system to the new LangGraph-based state management introduced in orchestrator v2.0.

## Overview

The orchestrator framework has migrated from a custom state management system to LangGraph-based state management, providing:

- **Enhanced Performance**: Built on LangGraph's optimized infrastructure
- **Better Scalability**: Support for multiple storage backends (Memory, SQLite, PostgreSQL)
- **Rich State Schema**: Comprehensive state tracking with TypedDict schemas
- **Global Context**: Cross-session state persistence and semantic search
- **Backward Compatibility**: Seamless transition through compatibility adapters

## Migration Paths

### Path 1: New Projects (Recommended)

For new projects, use the LangGraph-based system directly:

```python
from orchestrator import Orchestrator

# Use LangGraph state management (default in v2.0+)
orchestrator = Orchestrator(use_langgraph_state=True)

# All existing pipeline functionality works the same
results = await orchestrator.execute_pipeline(pipeline, checkpoint_enabled=True)
```

### Path 2: Gradual Migration

For existing projects, gradually migrate using the compatibility adapter:

```python
from orchestrator import Orchestrator
from orchestrator.state.legacy_compatibility import LegacyStateManagerAdapter

# Start with legacy compatibility
orchestrator = Orchestrator(use_langgraph_state=False)

# When ready, enable LangGraph with backward compatibility
orchestrator = Orchestrator(use_langgraph_state=True)
# The LegacyStateManagerAdapter automatically handles compatibility
```

### Path 3: Direct LangGraph Integration

For advanced use cases requiring direct LangGraph access:

```python
from orchestrator.state.langgraph_state_manager import LangGraphGlobalContextManager
from orchestrator.state.global_context import create_initial_pipeline_state

# Create LangGraph manager directly
langgraph_manager = LangGraphGlobalContextManager(
    storage_backend="sqlite",  # or "memory", "postgresql"
    database_url="./langgraph_state.db"
)

# Initialize pipeline state
initial_state = create_initial_pipeline_state(
    pipeline_id="my_pipeline",
    thread_id="my_thread",
    execution_id="my_execution",
    inputs={"key": "value"}
)
```

## API Changes and Deprecations

### Deprecated Modules

The following modules are deprecated and will be removed in v3.0:

- `orchestrator.state.simple_state_manager` - Use `LangGraphGlobalContextManager` instead
- `orchestrator.state.adaptive_checkpoint` - Built into LangGraph system

### Legacy Modules (Backward Compatible)

These modules remain for backward compatibility but new features focus on LangGraph:

- `orchestrator.state.state_manager` - Use `orchestrator.Orchestrator(use_langgraph_state=True)`
- `orchestrator.state.backends` - Use LangGraph's native backends

## Feature Mapping

### Legacy → LangGraph Feature Equivalents

| Legacy Feature | LangGraph Equivalent | Benefits |
|---|---|---|
| `StateManager.save_checkpoint()` | `LangGraphGlobalContextManager.save_checkpoint()` | Enhanced metadata, better performance |
| `StateManager.restore_checkpoint()` | `LangGraphGlobalContextManager.restore_checkpoint()` | Faster restoration, data integrity |
| `AdaptiveCheckpointStrategy` | Built-in optimization | Automatic optimization based on pipeline complexity |
| File/Memory backends | MemorySaver/SqliteSaver/PostgresSaver | Production-ready, scalable backends |
| Basic state tracking | Global context with TypedDict schema | Rich state schema with validation |

### New LangGraph-Only Features

Features only available with LangGraph state management:

```python
orchestrator = Orchestrator(use_langgraph_state=True)

# Enhanced features
global_state = await orchestrator.get_pipeline_global_state(execution_id)
named_checkpoint = await orchestrator.create_named_checkpoint(execution_id, "milestone", "Description")
metrics = await orchestrator.get_pipeline_metrics(execution_id)
```

## Resume Manager Migration

The pipeline resume manager has been enhanced to support both systems:

```python
from orchestrator.core.pipeline_resume_manager import ResumeStrategy

# Works with both legacy and LangGraph
resume_strategy = ResumeStrategy(
    retry_failed_tasks=True,
    max_retry_attempts=3,
    checkpoint_interval_seconds=60.0
)

# Enhanced features with LangGraph
if orchestrator.resume_manager._use_langgraph:
    # Create named resume checkpoints
    named_checkpoint = await orchestrator.resume_manager.create_named_resume_checkpoint(
        execution_id, "before_critical_task", "Checkpoint before critical processing"
    )
    
    # Get enhanced metrics
    metrics = await orchestrator.resume_manager.get_enhanced_resume_metrics(execution_id)
    
    # Optimize storage
    await orchestrator.resume_manager.optimize_checkpoint_storage(execution_id, keep_last_n=5)
```

## Testing Migration

### Legacy Tests

Legacy tests continue to work with the compatibility layer:

```python
# Existing tests work unchanged
def test_legacy_functionality():
    orchestrator = Orchestrator(use_langgraph_state=False)
    # ... existing test code
```

### Enhanced Testing

New tests can leverage LangGraph features:

```python
def test_enhanced_features():
    orchestrator = Orchestrator(use_langgraph_state=True)
    
    # Test global state access
    global_state = await orchestrator.get_pipeline_global_state(execution_id)
    assert global_state is not None
    
    # Test state validation
    from orchestrator.state.global_context import validate_pipeline_state
    errors = validate_pipeline_state(global_state)
    assert len(errors) == 0
```

## Performance Considerations

### Migration Performance Impact

- **Initialization**: LangGraph orchestrator ~15-50% slower to initialize (enhanced features)
- **State Operations**: LangGraph operations comparable or faster than legacy
- **Memory Usage**: LangGraph uses ~20-100% more memory (richer state tracking)
- **Checkpointing**: LangGraph checkpointing 2-5x faster for large states

### Optimization Tips

1. **Use SQLite for Production**: Better performance than file-based storage
2. **Enable Checkpoint Cleanup**: Regularly clean old checkpoints
3. **Monitor Memory Usage**: LangGraph tracks more state data
4. **Batch Operations**: Use concurrent operations for multiple pipelines

## Troubleshooting

### Common Migration Issues

#### Import Errors
```bash
ModuleNotFoundError: No module named 'langgraph'
```
**Solution**: Install LangGraph dependencies:
```bash
pip install langgraph langgraph-checkpoint langgraph-checkpoint-sqlite
```

#### State Schema Validation Errors
```python
ValidationError: Invalid pipeline state structure
```
**Solution**: Use the state validation utilities:
```python
from orchestrator.state.global_context import validate_pipeline_state, create_initial_pipeline_state

# Create valid initial state
state = create_initial_pipeline_state(pipeline_id, thread_id, execution_id, inputs)

# Validate existing state
errors = validate_pipeline_state(state)
if errors:
    print(f"State validation errors: {errors}")
```

#### Backward Compatibility Issues
```python
AttributeError: 'LegacyStateManagerAdapter' object has no attribute 'new_feature'
```
**Solution**: Check if the feature is LangGraph-only:
```python
if hasattr(orchestrator, 'get_langgraph_manager'):
    langgraph_manager = orchestrator.get_langgraph_manager()
    if langgraph_manager:
        # Use LangGraph-specific feature
        result = await langgraph_manager.new_feature()
```

### Performance Issues

#### Slow Initialization
- **Cause**: LangGraph setup overhead
- **Solution**: Cache orchestrator instances, use connection pooling

#### High Memory Usage  
- **Cause**: Rich state tracking in LangGraph
- **Solution**: Enable checkpoint cleanup, monitor state size

#### Checkpoint Failures
- **Cause**: Database connection issues (SQLite/PostgreSQL)
- **Solution**: Check database permissions, use connection retry logic

## Support and Resources

### Documentation
- [LangGraph State Management API](../api/langgraph_state_manager.md)
- [Global Context Schema](../api/global_context.md)
- [Legacy Compatibility Guide](../api/legacy_compatibility.md)

### Examples
- [Basic Migration Example](../examples/migration_basic.py)
- [Advanced LangGraph Features](../examples/langgraph_advanced.py)
- [Performance Comparison](../examples/performance_comparison.py)

### Getting Help

1. **Check Compatibility**: Use `orchestrator.get_state_manager_type()` to verify mode
2. **Enable Debug Logging**: Set `logging.getLogger('orchestrator.state').setLevel(logging.DEBUG)`
3. **Test with Memory Backend**: Use `storage_backend="memory"` for testing
4. **Report Issues**: File issues at [GitHub Issues](https://github.com/orchestrator/orchestrator/issues)

---

**Note**: This migration guide will be updated as new features are added and based on community feedback. Check the latest version at docs/migration/langgraph-state-management.md.