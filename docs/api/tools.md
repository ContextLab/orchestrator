# Tools Registry API Reference

The Tools Registry provides comprehensive tool discovery, management, version control, and security features. The enhanced registry supports tool registration, compatibility checking, installation management, and extensible design patterns.

## EnhancedToolRegistry

The `EnhancedToolRegistry` is the main interface for tool management with advanced features beyond basic tool registration.

### Constructor

```python
from orchestrator.tools import EnhancedToolRegistry, get_enhanced_registry

# Get global registry instance (recommended)
registry = get_enhanced_registry()

# Or create new instance
registry = EnhancedToolRegistry()
```

### Tool Registration

#### register_tool_enhanced()

Register a tool with comprehensive metadata and feature support.

```python
def register_tool_enhanced(
    tool: OrchestratorTool,
    version_info: Optional[VersionInfo] = None,
    category: ToolCategory = ToolCategory.CUSTOM,
    security_level: SecurityLevel = SecurityLevel.MODERATE,
    installation_requirements: Optional[List[InstallationRequirement]] = None,
    compatibility_requirements: Optional[List[CompatibilityRequirement]] = None,
    provides: Optional[List[str]] = None,
    requires: Optional[List[str]] = None,
    conflicts_with: Optional[List[str]] = None,
    **kwargs
) -> bool
```

**Parameters:**
- `tool` (OrchestratorTool): The tool instance to register
- `version_info` (Optional[VersionInfo]): Version information
- `category` (ToolCategory): Tool category classification
- `security_level` (SecurityLevel): Security level for execution
- `installation_requirements` (Optional[List]): Installation dependencies
- `compatibility_requirements` (Optional[List]): Compatibility constraints
- `provides` (Optional[List[str]]): Capabilities this tool provides
- `requires` (Optional[List[str]]): Capabilities this tool needs
- `conflicts_with` (Optional[List[str]]): Tools that conflict

**Returns:**
- `bool`: True if registration was successful

**Example:**
```python
from orchestrator.tools import (
    EnhancedToolRegistry, VersionInfo, SecurityLevel, 
    ToolCategory, InstallationRequirement, CompatibilityRequirement
)
from orchestrator.tools.base import Tool, ToolParameter

# Define a custom tool
class DataProcessorTool(Tool):
    def __init__(self):
        super().__init__(
            name="data_processor",
            description="Process and transform data files",
            parameters=[
                ToolParameter("input_file", str, "Input data file path"),
                ToolParameter("output_format", str, "Output format (csv, json, parquet)")
            ]
        )
    
    async def execute(self, input_file: str, output_format: str = "csv"):
        # Tool implementation...
        return {"processed": True, "output_file": f"processed_data.{output_format}"}

# Create tool instance
data_tool = DataProcessorTool()

# Define version and requirements
version = VersionInfo(major=2, minor=1, patch=0)

installation_reqs = [
    InstallationRequirement(
        package_manager="pip",
        package_name="pandas",
        version_spec=">=1.5.0"
    ),
    InstallationRequirement(
        package_manager="pip", 
        package_name="pyarrow",
        version_spec=">=10.0.0"
    )
]

compatibility_reqs = [
    CompatibilityRequirement(
        name="python",
        min_version=VersionInfo.parse("3.8.0"),
        required=True,
        reason="Uses modern Python features"
    )
]

# Register with enhanced features
registry = get_enhanced_registry()
success = registry.register_tool_enhanced(
    tool=data_tool,
    version_info=version,
    category=ToolCategory.DATA_PROCESSING,
    security_level=SecurityLevel.MODERATE,
    installation_requirements=installation_reqs,
    compatibility_requirements=compatibility_reqs,
    provides=["data_processing", "format_conversion"],
    requires=["file_system_access"]
)

if success:
    print("Tool registered successfully!")
```

### Simple Registration

#### register_tool_simple()

Simplified tool registration for basic use cases.

```python
from orchestrator.tools import register_tool_simple

success = register_tool_simple(
    tool=data_tool,
    version="2.1.0",
    category="data_processing",
    security_level="moderate"
)
```

### Tool Discovery

#### discover_tools_advanced()

Advanced tool discovery with comprehensive filtering options.

```python
def discover_tools_advanced(
    action_description: Optional[str] = None,
    category: Optional[ToolCategory] = None,
    version_constraints: Optional[Dict[str, str]] = None,
    security_level: Optional[SecurityLevel] = None,
    installation_status: Optional[InstallationStatus] = None,
    capabilities: Optional[List[str]] = None,
    exclude_deprecated: bool = True
) -> List[Dict[str, Any]]
```

**Parameters:**
- `action_description` (Optional[str]): Natural language description of desired action
- `category` (Optional[ToolCategory]): Filter by tool category
- `version_constraints` (Optional[Dict]): Version requirements per tool
- `security_level` (Optional[SecurityLevel]): Required security level
- `installation_status` (Optional[InstallationStatus]): Installation status filter
- `capabilities` (Optional[List[str]]): Required capabilities
- `exclude_deprecated` (bool): Whether to exclude deprecated tools

**Returns:**
- List of tool information dictionaries

**Example:**
```python
# Find tools for data processing
tools = registry.discover_tools_advanced(
    action_description="process CSV data files",
    category=ToolCategory.DATA_PROCESSING,
    security_level=SecurityLevel.MODERATE,
    capabilities=["data_processing", "format_conversion"]
)

for tool_info in tools:
    metadata = tool_info["metadata"]
    print(f"Tool: {metadata['name']}")
    print(f"Description: {metadata['description']}")
    print(f"Version: {metadata['version_info']}")
    print(f"Status: {tool_info['installation_status']}")
    print(f"Capabilities: {metadata['provides']}")
    print()
```

#### discover_tools_for_action()

Simple tool discovery by action description.

```python
from orchestrator.tools import discover_tools_for_action

tool_names = discover_tools_for_action("convert CSV to JSON")
print(f"Recommended tools: {tool_names}")
```

### Version Management

#### get_tool_versions()

Get all available versions of a tool.

```python
def get_tool_versions(tool_name: str) -> List[VersionInfo]
```

**Example:**
```python
versions = registry.get_tool_versions("data_processor")
for version in versions:
    print(f"Available version: {version}")
```

#### get_latest_version()

Get the latest version of a tool.

```python
def get_latest_version(tool_name: str) -> Optional[VersionInfo]
```

**Example:**
```python
latest = registry.get_latest_version("data_processor")
if latest:
    print(f"Latest version: {latest}")
```

#### is_version_compatible()

Check if a specific version is compatible with the current system.

```python
def is_version_compatible(tool_name: str, required_version: str) -> bool
```

**Example:**
```python
compatible = registry.is_version_compatible("data_processor", "2.1.0")
if not compatible:
    print("Version 2.1.0 is not compatible with current system")
```

#### upgrade_tool()

Upgrade a tool to a specific or latest version.

```python
def upgrade_tool(tool_name: str, target_version: Optional[str] = None) -> bool
```

**Example:**
```python
# Upgrade to latest version
success = registry.upgrade_tool("data_processor")

# Upgrade to specific version
success = registry.upgrade_tool("data_processor", "2.1.0")

if success:
    print("Tool upgraded successfully")
```

### Tool Chain Discovery

#### get_tool_chain_enhanced()

Get an enhanced tool chain with dependency resolution for complex actions.

```python
def get_tool_chain_enhanced(
    action_description: str,
    constraints: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]
```

**Example:**
```python
# Get tool chain for complex data processing
chain = registry.get_tool_chain_enhanced(
    "load CSV data, validate it, transform to JSON, and save to database",
    constraints={"security_level": "strict", "max_execution_time": 300}
)

for step in chain:
    tool_match = step["tool"]
    metadata = step["metadata"] 
    dependencies = step["dependencies"]
    
    print(f"Step: {tool_match.tool_name}")
    print(f"Confidence: {tool_match.confidence}")
    print(f"Dependencies: {[dep['name'] for dep in dependencies]}")
    print(f"Installation Status: {step['installation_status']}")
    print()
```

### Installation Management

#### install_tool_requirements()

Install requirements for a tool.

```python
def install_tool_requirements(tool_name: str) -> bool
```

**Example:**
```python
# Install requirements for a tool
success = registry.install_tool_requirements("data_processor")
if success:
    print("Requirements installed successfully")
```

#### register_installation_callback()

Register callback for installation completion.

```python
def register_installation_callback(tool_name: str, callback: Callable) -> None
```

**Example:**
```python
def on_installation_complete(tool_name: str, status):
    print(f"Installation of {tool_name} completed with status: {status}")

registry.register_installation_callback("data_processor", on_installation_complete)
```

### Security Management

#### validate_security_policy()

Validate if an operation is allowed by a tool's security policy.

```python
def validate_security_policy(tool_name: str, operation: str) -> bool
```

**Example:**
```python
# Check if tool can perform file write operation
can_write = registry.validate_security_policy("data_processor", "file_write")
if not can_write:
    print("Tool is not allowed to write files")
```

#### block_tool() / unblock_tool()

Block or unblock tools from execution.

```python
def block_tool(tool_name: str, reason: str) -> bool
def unblock_tool(tool_name: str) -> bool
```

**Example:**
```python
# Block a problematic tool
registry.block_tool("unreliable_tool", "Security vulnerability detected")

# Later, unblock when fixed
registry.unblock_tool("unreliable_tool")
```

### Tool Execution with Monitoring

#### execute_tool_monitored()

Execute a tool with enhanced monitoring and security checks.

```python
async def execute_tool_monitored(
    tool_name: str,
    **kwargs
) -> ToolExecutionResult
```

**Example:**
```python
# Execute tool with monitoring
result = await registry.execute_tool_monitored(
    "data_processor",
    input_file="customer_data.csv",
    output_format="json"
)

if result.success:
    print(f"Tool executed successfully in {result.execution_time}s")
    print(f"Result: {result.output}")
else:
    print(f"Tool execution failed: {result.error}")
```

### Performance Monitoring

#### update_performance_metrics()

Update performance metrics for a tool after execution.

```python
def update_performance_metrics(tool_name: str, execution_time: float, success: bool) -> None
```

This is typically called automatically by `execute_tool_monitored()`, but can be used manually:

```python
# Manual performance tracking
start_time = time.time()
try:
    # Execute tool manually...
    execution_time = time.time() - start_time
    registry.update_performance_metrics("data_processor", execution_time, True)
except Exception:
    execution_time = time.time() - start_time
    registry.update_performance_metrics("data_processor", execution_time, False)
```

### Extension Management

#### register_plugin_interface()

Register a plugin interface for extensibility.

```python
def register_plugin_interface(interface_name: str, interface_class: Type) -> None
```

**Example:**
```python
from abc import ABC, abstractmethod

class DataProcessorPlugin(ABC):
    @abstractmethod
    def process_data(self, data: Any) -> Any:
        pass

# Register the interface
registry.register_plugin_interface("data_processor_plugin", DataProcessorPlugin)
```

#### register_extension()

Register an extension for a tool.

```python
def register_extension(tool_name: str, extension_name: str, extension_config: Dict[str, Any]) -> bool
```

**Example:**
```python
# Register extension configuration
extension_config = {
    "class_name": "AdvancedDataProcessor",
    "module_path": "extensions.advanced_processor",
    "config": {
        "enable_caching": True,
        "cache_size": "100MB"
    }
}

success = registry.register_extension(
    "data_processor",
    "advanced_processing",
    extension_config
)
```

### Compatibility Checking

#### check_system_compatibility()

Check compatibility of all registered tools with the current system.

```python
def check_system_compatibility() -> Dict[str, Any]
```

**Example:**
```python
compatibility_report = registry.check_system_compatibility()

print(f"Compatible tools: {len(compatibility_report['compatible'])}")
print(f"Incompatible tools: {len(compatibility_report['incompatible'])}")

if compatibility_report['missing_dependencies']:
    print("Missing dependencies:")
    for issue in compatibility_report['missing_dependencies']:
        print(f"  {issue['tool']}: missing {issue['missing']}")
```

#### check_tool_compatibility()

Check compatibility of a specific tool version.

```python
from orchestrator.tools import check_tool_compatibility

compatible = check_tool_compatibility("data_processor", "2.1.0")
```

### State Management

#### export_registry_state() / import_registry_state()

Export and import complete registry state.

```python
def export_registry_state() -> Dict[str, Any]
def import_registry_state(state: Dict[str, Any]) -> bool
```

**Example:**
```python
# Export registry state for backup
state = registry.export_registry_state()
print(f"Exported state at {state['export_timestamp']}")

# Import state for restoration
success = registry.import_registry_state(state)
if success:
    print("Registry state imported successfully")
```

## Version Information Classes

### VersionInfo

Represents semantic version information.

```python
from orchestrator.tools import VersionInfo

# Create version info
version = VersionInfo(major=2, minor=1, patch=0, pre_release="beta1")

# Parse from string
version = VersionInfo.parse("2.1.0-beta1+build123")

# Version comparison
v1 = VersionInfo.parse("1.0.0")
v2 = VersionInfo.parse("2.0.0")
print(v1 < v2)  # True
```

### CompatibilityRequirement

Defines compatibility constraints for tools.

```python
from orchestrator.tools import CompatibilityRequirement

req = CompatibilityRequirement(
    name="python",
    min_version=VersionInfo.parse("3.8.0"),
    max_version=VersionInfo.parse("3.11.999"),
    required=True,
    reason="Uses Python 3.8+ features"
)

# Check compatibility
python_version = VersionInfo.parse("3.9.0")
is_compatible = req.is_compatible(python_version)
```

### InstallationRequirement

Defines installation dependencies for tools.

```python
from orchestrator.tools import InstallationRequirement

req = InstallationRequirement(
    package_manager="pip",
    package_name="pandas",
    version_spec=">=1.5.0,<2.0.0",
    install_command="pip install pandas>=1.5.0,<2.0.0",
    dependencies=["numpy", "python-dateutil"],
    environment_setup={"PANDAS_CONFIG": "/path/to/config"}
)
```

## Security Policy Management

### SecurityPolicy

Defines security constraints for tool execution.

```python
from orchestrator.tools import SecurityPolicy, SecurityLevel

policy = SecurityPolicy(
    level=SecurityLevel.MODERATE,
    allowed_operations=["read", "write", "process"],
    blocked_operations=["system_call", "admin_access"],
    sandboxed=False,
    network_access=True,
    file_system_access=True,
    max_execution_time=300,  # 5 minutes
    max_memory_usage=1024,   # 1GB
    environment_variables={"TOOL_MODE": "safe"}
)
```

## Complete Usage Example

```python
import asyncio
from orchestrator.tools import (
    get_enhanced_registry, VersionInfo, SecurityLevel, ToolCategory,
    InstallationRequirement, CompatibilityRequirement
)
from orchestrator.tools.base import Tool, ToolParameter

class CustomerAnalyticsTool(Tool):
    def __init__(self):
        super().__init__(
            name="customer_analytics",
            description="Analyze customer data and generate insights",
            parameters=[
                ToolParameter("data_file", str, "Customer data file"),
                ToolParameter("analysis_type", str, "Type of analysis to perform"),
                ToolParameter("output_format", str, "Output format", default="json")
            ]
        )
    
    async def execute(self, data_file: str, analysis_type: str, output_format: str = "json"):
        # Simulate analysis
        await asyncio.sleep(2)
        return {
            "analysis_type": analysis_type,
            "customer_count": 1500,
            "insights": ["High retention rate", "Growth in premium segment"],
            "output_format": output_format
        }

async def main():
    # Get the enhanced registry
    registry = get_enhanced_registry()
    
    # Create and register tool
    analytics_tool = CustomerAnalyticsTool()
    
    # Define comprehensive registration
    version = VersionInfo(1, 2, 0)
    
    installation_reqs = [
        InstallationRequirement(
            package_manager="pip",
            package_name="pandas",
            version_spec=">=1.5.0"
        )
    ]
    
    compatibility_reqs = [
        CompatibilityRequirement(
            name="python",
            min_version=VersionInfo.parse("3.8.0"),
            required=True
        )
    ]
    
    print("Registering customer analytics tool...")
    success = registry.register_tool_enhanced(
        tool=analytics_tool,
        version_info=version,
        category=ToolCategory.DATA_ANALYSIS,
        security_level=SecurityLevel.MODERATE,
        installation_requirements=installation_reqs,
        compatibility_requirements=compatibility_reqs,
        provides=["customer_analytics", "data_insights"],
        requires=["data_processing"]
    )
    
    if not success:
        print("Tool registration failed!")
        return
    
    print("Tool registered successfully!")
    
    # Check system compatibility
    compatibility = registry.check_system_compatibility()
    print(f"System compatibility check:")
    print(f"  Compatible tools: {len(compatibility['compatible'])}")
    print(f"  Issues: {len(compatibility['incompatible'])}")
    
    # Discover tools for analytics
    print("\nDiscovering analytics tools...")
    tools = registry.discover_tools_advanced(
        action_description="analyze customer data",
        category=ToolCategory.DATA_ANALYSIS,
        capabilities=["customer_analytics"]
    )
    
    for tool_info in tools:
        metadata = tool_info["metadata"]
        print(f"Found tool: {metadata['name']} v{metadata['version_info']}")
        print(f"  Provides: {metadata['provides']}")
        print(f"  Status: {tool_info['installation_status']}")
    
    # Execute tool with monitoring
    print("\nExecuting analytics tool...")
    result = await registry.execute_tool_monitored(
        "customer_analytics",
        data_file="customer_data.csv",
        analysis_type="retention_analysis",
        output_format="json"
    )
    
    if result.success:
        print(f"Execution completed in {result.execution_time:.2f}s")
        print(f"Results: {result.output}")
        
        # Check performance metrics
        print("\nPerformance metrics updated")
        
    else:
        print(f"Execution failed: {result.error}")
    
    # Get tool chain for complex workflow
    print("\nGenerating tool chain for complex workflow...")
    chain = registry.get_tool_chain_enhanced(
        "load customer data, clean it, analyze retention patterns, and generate report"
    )
    
    print("Recommended tool chain:")
    for i, step in enumerate(chain, 1):
        tool_match = step["tool"]
        print(f"  Step {i}: {tool_match.tool_name} (confidence: {tool_match.confidence:.2f})")
    
    # Export registry state
    state = registry.export_registry_state()
    print(f"\nRegistry state exported with {len(state['enhanced_metadata'])} tools")

if __name__ == "__main__":
    asyncio.run(main())
```

## Best Practices

1. **Use Enhanced Registration**: Register tools with `register_tool_enhanced()` for full feature support.

2. **Version Management**: Always specify version information and compatibility requirements.

3. **Security Considerations**: Choose appropriate security levels and validate operations before execution.

4. **Capability Tracking**: Define clear `provides` and `requires` capabilities for better discovery.

5. **Installation Management**: Specify installation requirements for automated dependency handling.

6. **Performance Monitoring**: Use `execute_tool_monitored()` for automatic performance tracking.

7. **Compatibility Checking**: Regularly check system compatibility, especially after tool updates.

8. **Tool Chains**: Use tool chain discovery for complex multi-step operations.

9. **Extension Points**: Design tools with extension interfaces for modularity.

10. **State Management**: Export registry state for backup and disaster recovery scenarios.