# MCP Memory Context Management Pipeline

**Pipeline**: `examples/mcp_memory_workflow.yaml`  
**Category**: MCP Integration  
**Complexity**: Advanced  
**Key Features**: MCP memory management, Context persistence, TTL expiration, Namespace organization

## Overview

The MCP Memory Context Management Pipeline demonstrates advanced usage of Model Context Protocol (MCP) memory capabilities for maintaining context across pipeline steps. It showcases storing, retrieving, and managing conversational context with TTL expiration, namespace organization, and persistent exports.

## Key Features Demonstrated

### 1. Context Initialization and Storage
```yaml
- id: init_context
  tool: mcp-memory
  parameters:
    action: "store"
    namespace: "conversation"
    key: "user_profile"
    value:
      name: "{{ user_name }}"
      task: "{{ task_description }}"
      started_at: "{{ execution['timestamp'] }}"
    ttl: 7200  # 2 hours
```

### 2. Structured Data Storage
```yaml
parameters:
  action: "store"
  key: "task_steps"
  value:
    - "Load and validate data"
    - "Perform statistical analysis"
    - "Create visualizations"
    - "Generate report"
```

### 3. Progress Tracking
```yaml
value:
  current_step: 1
  completed_steps: ["Data loading plan created"]
  next_action: "Execute data loading"
```

### 4. Memory Listing and Export
```yaml
parameters:
  action: "list"
  namespace: "conversation"
```

## Pipeline Architecture

### Input Parameters
- **user_name** (optional): User identifier for context (default: "User")
- **task_description** (optional): Description of task being tracked (default: sales data analysis)
- **output_path** (optional): Directory for output files (default: examples/outputs/mcp_memory_workflow)

### Processing Flow

1. **Initialize Context** - Store user profile and task information
2. **Store Task Structure** - Define task breakdown and steps
3. **Process First Step** - Execute initial task delegation
4. **Update Progress** - Track completion status and next actions
5. **Retrieve Full Context** - List all stored memory keys
6. **Build Context Summary** - Create human-readable context overview
7. **Export Memory** - Persist memory state to JSON file
8. **Check Expiration** - Validate TTL functionality
9. **Optional Cleanup** - Clear temporary workspace (disabled by default)

### Memory Operations Demonstrated

#### Storage Operations
```yaml
# Store simple key-value
action: "store"
key: "user_profile"
value: { name: "Alice", role: "analyst" }

# Store complex structured data
action: "store"
key: "task_steps"
value: ["step1", "step2", "step3"]
```

#### Retrieval Operations
```yaml
# Retrieve specific key
action: "retrieve"
key: "user_profile"

# List all keys in namespace
action: "list"
namespace: "conversation"
```

#### Management Operations
```yaml
# Clear namespace
action: "clear"
namespace: "temporary_workspace"
```

## Usage Examples

### Basic Context Management
```bash
python scripts/run_pipeline.py examples/mcp_memory_workflow.yaml \
  -i user_name="Alice Johnson" \
  -i task_description="Financial report analysis"
```

### Multi-User Context
```bash
# User 1
python scripts/run_pipeline.py examples/mcp_memory_workflow.yaml \
  -i user_name="Bob Smith" \
  -i task_description="Customer segmentation analysis"

# User 2  
python scripts/run_pipeline.py examples/mcp_memory_workflow.yaml \
  -i user_name="María García López" \
  -i task_description="Product performance review"
```

### Custom Output Location
```bash
python scripts/run_pipeline.py examples/mcp_memory_workflow.yaml \
  -i user_name="DataTeam" \
  -i output_path="custom_outputs/team_context"
```

## Memory Management Features

### TTL (Time To Live) Management
```yaml
ttl: 7200  # 2 hours in seconds
```

Benefits:
- **Automatic Expiration**: Prevents memory bloat
- **Configurable Lifetime**: Adjust based on use case
- **Resource Management**: Efficient memory usage

### Namespace Organization
```yaml
namespace: "conversation"      # User interaction context
namespace: "temporary_workspace"  # Temporary processing data
namespace: "session_data"      # Session-specific information
```

Benefits:
- **Logical Separation**: Organize related data
- **Bulk Operations**: Clear entire namespaces
- **Access Control**: Namespace-based permissions

### Structured Data Storage
```yaml
value:
  current_step: 1
  completed_steps: ["Data loading plan created"]  
  next_action: "Execute data loading"
  metadata:
    priority: "high"
    estimated_duration: "30 minutes"
```

## Sample Output Files

### Context Summary (`alice-johnson_context_summary.md`)
```markdown
# Context Summary

**Generated on:** 2024-08-23T10:30:00Z

## Current Context State

**Namespace**: conversation
**Active Keys**: user_profile, task_steps, progress

### Details:
- **user_profile**: Stored in memory
- **task_steps**: Stored in memory  
- **progress**: Stored in memory
```

### Memory Export (`alice-johnson_memory_export.json`)
```json
{
  "namespace": "conversation",
  "exported_at": "2024-08-23T10:30:00Z", 
  "keys": ["user_profile", "task_steps", "progress"],
  "metadata": {
    "user": "Alice Johnson",
    "task": "Financial report analysis"
  }
}
```

## Advanced MCP Memory Patterns

### Progressive Context Building
```yaml
# Step 1: Initialize
store: user_profile

# Step 2: Add task structure  
store: task_steps

# Step 3: Track progress
store: progress

# Step 4: Add results
store: results
```

### Context Retrieval Patterns
```yaml
# Get specific context item
retrieve: user_profile

# Get all context for user
list: namespace="user_123"

# Get context matching pattern
search: pattern="task_*"
```

### Memory Lifecycle Management
```yaml
# Short-term memory (5 minutes)
ttl: 300

# Session memory (2 hours)
ttl: 7200  

# Persistent memory (24 hours)
ttl: 86400

# Permanent storage (no TTL)
# ttl: null
```

## Context Management Use Cases

### 1. Multi-Step Workflows
Store intermediate results and track progress across complex pipelines:
```yaml
store:
  workflow_state:
    current_step: 3
    completed_steps: ["analysis", "validation"]
    next_steps: ["reporting", "delivery"]
```

### 2. User Session Management
Maintain user preferences and interaction history:
```yaml
store:
  user_session:
    preferences: { format: "markdown", detail: "high" }
    interaction_count: 5
    last_activity: "2024-08-23T10:30:00Z"
```

### 3. Error Recovery Context
Store context for error recovery and retry logic:
```yaml
store:
  error_context:
    failed_step: "data_processing"
    error_details: "Connection timeout"
    retry_count: 2
    recovery_strategy: "use_cached_data"
```

### 4. Collaborative Workflows
Share context between multiple users or systems:
```yaml
store:
  shared_context:
    project_id: "proj_123"
    contributors: ["alice", "bob", "charlie"]
    shared_data: { dataset: "sales_q3.csv" }
```

## Technical Implementation

### Memory Operations
```yaml
# Available actions
actions:
  - "store"     # Save key-value pair
  - "retrieve"  # Get value by key
  - "list"      # List keys in namespace
  - "clear"     # Clear namespace
  - "delete"    # Delete specific key
  - "exists"    # Check key existence
```

### Data Types Supported
```yaml
supported_types:
  - strings
  - numbers  
  - booleans
  - arrays
  - objects (nested JSON)
  - null values
```

### Template Integration
```yaml
# Use memory values in templates
content: "User {{ memory.user_profile.name }} completed step {{ memory.progress.current_step }}"

# Store template results
value: "{{ some_calculation_result }}"
```

## Best Practices Demonstrated

1. **Namespace Organization**: Use meaningful namespace names
2. **TTL Management**: Set appropriate expiration times
3. **Structured Storage**: Use consistent data structures
4. **Progress Tracking**: Maintain workflow state information
5. **Export Capability**: Create persistent backups of memory state
6. **Error Handling**: Check for key existence and retrieval success
7. **Resource Cleanup**: Clear temporary data when no longer needed

## Common Use Cases

- **Conversational AI**: Maintain chat context across interactions
- **Workflow Management**: Track multi-step process progress
- **User Personalization**: Store user preferences and history
- **Error Recovery**: Preserve context for failure recovery
- **Collaborative Systems**: Share context between multiple agents
- **Session Management**: Handle user sessions and state
- **Caching Layer**: Store frequently accessed data

## Troubleshooting

### Memory Not Persisting
- Check TTL settings and expiration times
- Verify namespace and key names are correct
- Ensure MCP memory service is running and accessible

### Context Retrieval Issues
- Validate key existence before retrieval
- Check namespace access permissions
- Verify memory service connectivity

### Performance Concerns
- Monitor memory usage and namespace sizes
- Use appropriate TTL values for cleanup
- Consider memory limits and quotas

## Related Examples
- [mcp_simple_test.md](mcp_simple_test.md) - Basic MCP integration
- [mcp_integration_pipeline.md](mcp_integration_pipeline.md) - Advanced MCP features
- [interactive_pipeline.md](interactive_pipeline.md) - User context management

## Technical Requirements

- **MCP Service**: Active Model Context Protocol service
- **Memory Backend**: Persistent storage for memory operations
- **Namespace Support**: Multi-tenant memory organization
- **TTL Support**: Automatic expiration capabilities
- **JSON Serialization**: Structured data storage and retrieval

This pipeline demonstrates enterprise-grade context management suitable for complex multi-step workflows requiring persistent memory and state management across pipeline executions.