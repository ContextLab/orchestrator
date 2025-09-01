# Variable Management API Reference

The Variable Management System provides comprehensive variable storage, retrieval, scoping, and dependency tracking for pipeline execution. It supports context isolation, template resolution, and event-driven updates.

## VariableManager

The `VariableManager` is the core class for managing variables throughout pipeline execution.

### Constructor

```python
from orchestrator.execution import VariableManager

variable_manager = VariableManager(pipeline_id="my_pipeline")
```

**Parameters:**
- `pipeline_id` (str): Unique identifier for the pipeline execution (default: "default")

### Variable Operations

#### set_variable()

Set a variable with full metadata support.

```python
def set_variable(
    name: str,
    value: Any,
    scope: VariableScope = VariableScope.GLOBAL,
    var_type: VariableType = VariableType.INTERMEDIATE,
    source_step: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[Set[str]] = None,
    context_id: Optional[str] = None
) -> None
```

**Parameters:**
- `name` (str): Variable name
- `value` (Any): Variable value
- `scope` (VariableScope): Variable scope (GLOBAL, STEP, LOOP, TEMPORARY)
- `var_type` (VariableType): Type classification (INPUT, OUTPUT, INTERMEDIATE, CONFIGURATION, SYSTEM)
- `source_step` (Optional[str]): Step that created this variable
- `description` (Optional[str]): Human-readable description
- `tags` (Optional[Set[str]]): Tags for categorization
- `context_id` (Optional[str]): Context identifier for isolation

**Example:**
```python
from orchestrator.execution import VariableScope, VariableType

# Set a simple variable
variable_manager.set_variable("customer_count", 1500)

# Set with full metadata
variable_manager.set_variable(
    name="processed_data",
    value={"records": 45000, "valid": 43500, "errors": 1500},
    scope=VariableScope.GLOBAL,
    var_type=VariableType.OUTPUT,
    source_step="data_processing",
    description="Results from data processing step",
    tags={"output", "metrics", "validation"}
)

# Set configuration variable
variable_manager.set_variable(
    name="batch_size",
    value=1000,
    var_type=VariableType.CONFIGURATION,
    description="Processing batch size"
)
```

#### get_variable()

Get a variable value with optional template resolution.

```python
def get_variable(
    name: str,
    default: Any = None,
    context_id: Optional[str] = None,
    resolve_templates: bool = True
) -> Any
```

**Parameters:**
- `name` (str): Variable name
- `default` (Any): Default value if variable not found
- `context_id` (Optional[str]): Context to search in addition to global
- `resolve_templates` (bool): Whether to resolve template expressions

**Returns:**
- Variable value or default if not found

**Example:**
```python
# Get simple variable
customer_count = variable_manager.get_variable("customer_count", default=0)

# Get with template resolution
template_value = variable_manager.get_variable("output_path")  # Resolves ${base_path}/output

# Get without template resolution
raw_value = variable_manager.get_variable("template_string", resolve_templates=False)

# Get from specific context
context_value = variable_manager.get_variable("temp_var", context_id="processing_ctx")
```

#### get_variable_metadata()

Get detailed metadata for a variable.

```python
def get_variable_metadata(
    name: str,
    context_id: Optional[str] = None
) -> Optional[VariableMetadata]
```

**Parameters:**
- `name` (str): Variable name
- `context_id` (Optional[str]): Context to search in

**Returns:**
- `VariableMetadata` object or None if not found

**Example:**
```python
metadata = variable_manager.get_variable_metadata("processed_data")
if metadata:
    print(f"Variable: {metadata.name}")
    print(f"Type: {metadata.var_type.value}")
    print(f"Scope: {metadata.scope.value}")
    print(f"Created: {metadata.created_at}")
    print(f"Updated: {metadata.updated_at}")
    print(f"Source: {metadata.source_step}")
    print(f"Description: {metadata.description}")
    print(f"Tags: {metadata.tags}")
    print(f"Version: {metadata.version}")
```

#### has_variable()

Check if a variable exists.

```python
def has_variable(
    name: str,
    context_id: Optional[str] = None
) -> bool
```

**Example:**
```python
if variable_manager.has_variable("customer_data"):
    data = variable_manager.get_variable("customer_data")
    # Process data...
```

#### delete_variable()

Delete a variable from storage.

```python
def delete_variable(
    name: str,
    context_id: Optional[str] = None
) -> bool
```

**Returns:**
- `True` if variable was deleted successfully

**Example:**
```python
# Delete temporary variable
success = variable_manager.delete_variable("temp_processing_data")

# Delete from specific context
success = variable_manager.delete_variable("context_var", context_id="temp_ctx")
```

### Variable Listing and Filtering

#### list_variables()

List variables with optional filtering.

```python
def list_variables(
    scope: Optional[VariableScope] = None,
    var_type: Optional[VariableType] = None,
    context_id: Optional[str] = None,
    include_metadata: bool = False
) -> Dict[str, Any]
```

**Parameters:**
- `scope` (Optional[VariableScope]): Filter by scope
- `var_type` (Optional[VariableType]): Filter by variable type
- `context_id` (Optional[str]): Context to list from
- `include_metadata` (bool): Whether to include metadata in results

**Returns:**
- Dictionary of variable names to values (or values + metadata)

**Example:**
```python
# List all variables
all_vars = variable_manager.list_variables()

# List only output variables
outputs = variable_manager.list_variables(var_type=VariableType.OUTPUT)

# List with metadata
vars_with_metadata = variable_manager.list_variables(include_metadata=True)
for name, info in vars_with_metadata.items():
    print(f"{name}: {info['value']} (type: {info['metadata'].var_type.value})")

# List global configuration variables
config_vars = variable_manager.list_variables(
    scope=VariableScope.GLOBAL,
    var_type=VariableType.CONFIGURATION
)
```

## Variable Scoping and Context Management

### Context Operations

#### create_context()

Create a new isolated context for variable management.

```python
def create_context() -> str
```

**Returns:**
- Context identifier string

**Example:**
```python
# Create context for loop iteration
loop_context = variable_manager.create_context()
print(f"Created context: {loop_context}")
```

#### destroy_context()

Destroy a context and all its variables.

```python
def destroy_context(context_id: str) -> None
```

**Example:**
```python
# Clean up context when done
variable_manager.destroy_context(loop_context)
```

### Scope Stack Management

#### push_scope() / pop_scope()

Manage scope contexts for nested variable resolution.

```python
def push_scope(scope_mapping: Dict[str, str]) -> None
def pop_scope() -> Optional[Dict[str, str]]
```

**Example:**
```python
# Push scope for loop variables
scope_mapping = {
    "current_item": "loop_item_5",
    "index": "loop_index_5"
}
variable_manager.push_scope(scope_mapping)

# Variables in this scope will map local names to global variables
item = variable_manager.get_variable("current_item")  # Gets "loop_item_5"

# Pop scope when exiting loop
variable_manager.pop_scope()
```

## Template Resolution

#### set_template()

Set a variable template for dynamic resolution.

```python
def set_template(name: str, template: str) -> None
```

**Parameters:**
- `name` (str): Variable name
- `template` (str): Template string with ${var} placeholders

**Example:**
```python
# Set template variables
variable_manager.set_variable("base_path", "/data/projects")
variable_manager.set_variable("project_name", "customer_analytics")

# Create template
variable_manager.set_template("output_path", "${base_path}/${project_name}/output")

# Template resolves automatically when accessed
output_path = variable_manager.get_variable("output_path")
# Returns: "/data/projects/customer_analytics/output"
```

#### resolve_template()

Manually resolve a template string.

```python
def resolve_template(
    template: str,
    context: Optional[Dict[str, Any]] = None
) -> Any
```

**Parameters:**
- `template` (str): Template string with ${var} placeholders
- `context` (Optional[Dict[str, Any]]): Additional context variables

**Example:**
```python
# Resolve template with current variables
result = variable_manager.resolve_template("Processing ${customer_count} customers")

# Resolve with additional context
context = {"batch_size": 500}
result = variable_manager.resolve_template(
    "Processing ${customer_count} customers in batches of ${batch_size}",
    context
)
```

## Dependency Tracking

#### add_dependency()

Add dependency relationships between variables.

```python
def add_dependency(variable_name: str, depends_on: Set[str]) -> None
```

**Example:**
```python
# Variable dependencies
variable_manager.add_dependency("final_report", {"processed_data", "summary_stats"})
variable_manager.add_dependency("summary_stats", {"raw_data", "validation_results"})
```

#### get_dependencies() / get_dependents()

Get dependency information for variables.

```python
def get_dependencies(variable_name: str) -> Set[str]
def get_dependents(variable_name: str) -> Set[str]
```

**Example:**
```python
# Get what a variable depends on
deps = variable_manager.get_dependencies("final_report")
print(f"final_report depends on: {deps}")

# Get what depends on a variable  
dependents = variable_manager.get_dependents("raw_data")
print(f"Variables that depend on raw_data: {dependents}")
```

## Event Handling

#### on_variable_created() / on_variable_changed()

Register event handlers for variable operations.

```python
def on_variable_created(handler: Callable[[str, Any], None]) -> None
def on_variable_changed(handler: Callable[[str, Any, Any], None]) -> None
```

**Example:**
```python
def log_variable_creation(name: str, value: Any) -> None:
    print(f"Variable created: {name} = {value}")

def log_variable_change(name: str, old_value: Any, new_value: Any) -> None:
    print(f"Variable changed: {name} from {old_value} to {new_value}")

# Register handlers
variable_manager.on_variable_created(log_variable_creation)
variable_manager.on_variable_changed(log_variable_change)
```

## State Persistence

#### export_state() / import_state()

Export and import variable manager state for persistence.

```python
def export_state(context_id: Optional[str] = None) -> Dict[str, Any]
def import_state(state: Dict[str, Any]) -> None
```

**Example:**
```python
# Export complete state
state = variable_manager.export_state()
print(f"Exported {len(state['global_variables'])} global variables")

# Export specific context
context_state = variable_manager.export_state(context_id="processing_ctx")

# Import state (typically for recovery)
variable_manager.import_state(state)
```

## VariableContext (Context Manager)

The `VariableContext` provides a clean way to manage variable scopes with automatic cleanup.

### Usage

```python
from orchestrator.execution import VariableContext

with VariableContext(variable_manager) as ctx:
    # Set variables in this context
    ctx.set_variable("temp_data", processing_data)
    ctx.set_variable("batch_id", "batch_001")
    
    # Variables are automatically cleaned up on exit
    temp_value = ctx.get_variable("temp_data")

# Context variables are automatically destroyed
```

### With Scope Mapping

```python
# Context with scope mapping
scope_mapping = {"item": "current_loop_item", "index": "current_loop_index"}

with VariableContext(variable_manager).with_scope(scope_mapping) as ctx:
    # Local names map to global variables
    current_item = ctx.get_variable("item")  # Gets "current_loop_item"
    current_index = ctx.get_variable("index")  # Gets "current_loop_index"
    
    # Set context-specific variables
    ctx.set_variable("processing_result", result_data)

# Scope is automatically popped and context cleaned up
```

## Variable Types and Scopes

### VariableScope Enum

- `GLOBAL`: Available across entire pipeline
- `STEP`: Available only within specific step
- `LOOP`: Available only within loop iteration
- `TEMPORARY`: Available only for specific operation

### VariableType Enum

- `INPUT`: Input parameter to pipeline/step
- `OUTPUT`: Output result from step
- `INTERMEDIATE`: Intermediate computation result
- `CONFIGURATION`: Configuration parameter
- `SYSTEM`: System-generated variable

## Complete Usage Example

```python
import asyncio
from orchestrator.execution import VariableManager, VariableScope, VariableType, VariableContext

async def main():
    # Create variable manager
    vm = VariableManager(pipeline_id="customer_analytics")
    
    # Set configuration
    vm.set_variable("batch_size", 1000, var_type=VariableType.CONFIGURATION)
    vm.set_variable("output_dir", "/data/output", var_type=VariableType.CONFIGURATION)
    
    # Set up template
    vm.set_template("output_path", "${output_dir}/results_${timestamp}.json")
    
    # Add event handlers
    def log_changes(name, old_val, new_val):
        print(f"Variable {name} changed: {old_val} -> {new_val}")
    vm.on_variable_changed(log_changes)
    
    # Simulate pipeline execution
    print("=== Pipeline Execution ===")
    
    # Step 1: Load data
    vm.set_variable(
        "raw_data",
        {"customers": 1500, "records": 45000},
        var_type=VariableType.OUTPUT,
        source_step="load_data",
        description="Raw customer data from database"
    )
    
    # Step 2: Process data with loop context
    batch_size = vm.get_variable("batch_size")
    total_records = vm.get_variable("raw_data")["records"]
    batches = (total_records + batch_size - 1) // batch_size
    
    processed_records = 0
    
    for i in range(batches):
        # Create context for this batch
        with VariableContext(vm) as batch_ctx:
            batch_ctx.set_variable("batch_id", f"batch_{i:03d}")
            batch_ctx.set_variable("batch_start", i * batch_size)
            batch_ctx.set_variable("batch_end", min((i + 1) * batch_size, total_records))
            
            # Process batch (simulated)
            batch_records = batch_ctx.get_variable("batch_end") - batch_ctx.get_variable("batch_start")
            processed_records += batch_records
            
            print(f"Processed {batch_ctx.get_variable('batch_id')}: {batch_records} records")
            
            # Store intermediate result
            batch_ctx.set_variable(
                "batch_result",
                {"processed": batch_records, "valid": int(batch_records * 0.95)},
                scope=VariableScope.TEMPORARY
            )
    
    # Step 3: Store final results
    vm.set_variable("timestamp", "2024-08-31T19:00:00")
    
    final_results = {
        "total_customers": vm.get_variable("raw_data")["customers"],
        "total_records": total_records,
        "processed_records": processed_records,
        "success_rate": processed_records / total_records
    }
    
    vm.set_variable(
        "final_results",
        final_results,
        var_type=VariableType.OUTPUT,
        source_step="finalize_processing",
        description="Final processing results and metrics"
    )
    
    # Get output path (template resolution)
    output_path = vm.get_variable("output_path")
    print(f"Results will be saved to: {output_path}")
    
    # Show variable summary
    print("\n=== Variable Summary ===")
    all_vars = vm.list_variables(include_metadata=True)
    
    for name, info in all_vars.items():
        metadata = info['metadata']
        print(f"{name}:")
        print(f"  Value: {info['value']}")
        print(f"  Type: {metadata.var_type.value}")
        print(f"  Scope: {metadata.scope.value}")
        print(f"  Source: {metadata.source_step or 'N/A'}")
        print()
    
    # Export state for checkpointing
    state = vm.export_state()
    print(f"Exported state with {len(state['global_variables'])} variables")

if __name__ == "__main__":
    asyncio.run(main())
```

## Best Practices

1. **Use Appropriate Types**: Choose correct `VariableType` and `VariableScope` for proper organization.

2. **Provide Metadata**: Include descriptions, source steps, and tags for better debugging and monitoring.

3. **Use Context Managers**: Use `VariableContext` for temporary variables that should be automatically cleaned up.

4. **Template Resolution**: Leverage template variables for dynamic path and configuration resolution.

5. **Event Handlers**: Use event handlers for debugging, logging, and cross-system synchronization.

6. **Dependency Tracking**: Track variable dependencies for better change management and validation.

7. **Context Isolation**: Use contexts for loop iterations and parallel processing to avoid variable conflicts.

8. **State Persistence**: Export/import state for checkpointing and recovery scenarios.