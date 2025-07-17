# Tool Definition Guide

## Overview

The Orchestrator framework allows you to define custom tools that can be used within your AI pipelines. Tools extend the capabilities of your pipelines by providing access to external systems, specialized processing, or custom logic.

## Tool Architecture

Tools in the Orchestrator framework follow a plugin-based architecture:

```
┌─────────────────┐
│   Pipeline      │
│   Definition    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Control System  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     Tools       │
├─────────────────┤
│ - File System   │
│ - Database      │
│ - API Client    │
│ - Custom Logic  │
└─────────────────┘
```

## Creating a Custom Tool

### Basic Tool Structure

```python
from orchestrator.tools.base_tool import BaseTool
from typing import Any, Dict, Optional

class MyCustomTool(BaseTool):
    """A custom tool for specific functionality."""
    
    def __init__(self, name: str = "my_custom_tool", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        # Initialize tool-specific resources
        
    async def execute(self, action: str, parameters: Dict[str, Any]) -> Any:
        """
        Execute the tool action.
        
        Args:
            action: The action to perform
            parameters: Parameters for the action
            
        Returns:
            The result of the action
        """
        # Implement your tool logic here
        pass
        
    async def validate_parameters(self, action: str, parameters: Dict[str, Any]) -> bool:
        """Validate parameters before execution."""
        # Implement validation logic
        return True
        
    def get_capabilities(self) -> Dict[str, Any]:
        """Return tool capabilities."""
        return {
            "actions": ["read", "write", "process"],
            "description": "Custom tool for X functionality",
            "version": "1.0.0"
        }
```

### Example: File System Tool

```python
import aiofiles
from pathlib import Path
from typing import Any, Dict, Optional

class FileSystemTool(BaseTool):
    """Tool for file system operations."""
    
    def __init__(self, name: str = "filesystem", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.allowed_paths = config.get("allowed_paths", []) if config else []
        
    async def execute(self, action: str, parameters: Dict[str, Any]) -> Any:
        """Execute file system operations."""
        if action == "read":
            return await self._read_file(parameters)
        elif action == "write":
            return await self._write_file(parameters)
        elif action == "list":
            return await self._list_directory(parameters)
        else:
            raise ValueError(f"Unknown action: {action}")
            
    async def _read_file(self, params: Dict[str, Any]) -> str:
        """Read a file."""
        file_path = Path(params["path"])
        
        # Security check
        if not self._is_path_allowed(file_path):
            raise PermissionError(f"Access denied to path: {file_path}")
            
        async with aiofiles.open(file_path, 'r') as f:
            content = await f.read()
        return content
        
    async def _write_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Write to a file."""
        file_path = Path(params["path"])
        content = params["content"]
        
        # Security check
        if not self._is_path_allowed(file_path):
            raise PermissionError(f"Access denied to path: {file_path}")
            
        async with aiofiles.open(file_path, 'w') as f:
            await f.write(content)
            
        return {"success": True, "path": str(file_path)}
        
    async def _list_directory(self, params: Dict[str, Any]) -> list:
        """List directory contents."""
        dir_path = Path(params["path"])
        
        if not self._is_path_allowed(dir_path):
            raise PermissionError(f"Access denied to path: {dir_path}")
            
        return [str(p) for p in dir_path.iterdir()]
        
    def _is_path_allowed(self, path: Path) -> bool:
        """Check if path access is allowed."""
        if not self.allowed_paths:
            return True  # No restrictions
            
        path = path.resolve()
        for allowed in self.allowed_paths:
            allowed_path = Path(allowed).resolve()
            try:
                path.relative_to(allowed_path)
                return True
            except ValueError:
                continue
        return False
        
    async def validate_parameters(self, action: str, parameters: Dict[str, Any]) -> bool:
        """Validate parameters."""
        if action in ["read", "write", "list"]:
            return "path" in parameters
        return False
        
    def get_capabilities(self) -> Dict[str, Any]:
        """Return tool capabilities."""
        return {
            "actions": ["read", "write", "list"],
            "description": "File system operations tool",
            "version": "1.0.0",
            "security": {
                "sandboxed": True,
                "allowed_paths": self.allowed_paths
            }
        }
```

## Registering Tools

### With the Control System

```python
from orchestrator.control_systems.model_based_control_system import ModelBasedControlSystem
from my_tools import FileSystemTool, DatabaseTool

# Create control system
control_system = ModelBasedControlSystem(model_registry)

# Register tools
control_system.register_tool(FileSystemTool(config={"allowed_paths": ["/data"]}))
control_system.register_tool(DatabaseTool(config={"connection_string": "..."}))
```

### Tool Registry Pattern

```python
from orchestrator.tools.tool_registry import ToolRegistry

# Create tool registry
tool_registry = ToolRegistry()

# Register tools
tool_registry.register("filesystem", FileSystemTool)
tool_registry.register("database", DatabaseTool)
tool_registry.register("http", HTTPTool)

# Get tool instance
fs_tool = tool_registry.get_tool("filesystem", config={...})
```

## Using Tools in YAML Pipelines

### Direct Tool Actions

```yaml
steps:
  - id: read_data
    tool: filesystem
    action: read
    parameters:
      path: "/data/input.json"
      
  - id: process_data
    action: |
      Process the data from file:
      {{read_data.result}}
      
      Transform and analyze the content
    depends_on: [read_data]
    
  - id: save_results
    tool: filesystem
    action: write
    parameters:
      path: "/data/output.json"
      content: "{{process_data.result}}"
    depends_on: [process_data]
```

### Tool-Augmented Actions

```yaml
steps:
  - id: analyze_with_tools
    action: |
      Analyze the data using available tools:
      1. Read configuration from filesystem
      2. Query database for additional context
      3. Process and combine results
      Return comprehensive analysis
    tools:
      - filesystem
      - database
    parameters:
      config_path: "/config/analysis.yaml"
      db_query: "SELECT * FROM metrics WHERE date > '2024-01-01'"
```

## Built-in Tools

### 1. HTTP Tool

```python
class HTTPTool(BaseTool):
    """Tool for making HTTP requests."""
    
    async def execute(self, action: str, parameters: Dict[str, Any]) -> Any:
        if action == "get":
            return await self._http_get(parameters["url"], parameters.get("headers", {}))
        elif action == "post":
            return await self._http_post(
                parameters["url"], 
                parameters.get("data", {}),
                parameters.get("headers", {})
            )
```

Usage in YAML:

```yaml
- id: fetch_api_data
  tool: http
  action: get
  parameters:
    url: "https://api.example.com/data"
    headers:
      Authorization: "Bearer {{api_token}}"
```

### 2. Database Tool

```python
class DatabaseTool(BaseTool):
    """Tool for database operations."""
    
    async def execute(self, action: str, parameters: Dict[str, Any]) -> Any:
        if action == "query":
            return await self._execute_query(parameters["sql"], parameters.get("params", []))
        elif action == "insert":
            return await self._insert_data(parameters["table"], parameters["data"])
```

Usage in YAML:

```yaml
- id: query_users
  tool: database
  action: query
  parameters:
    sql: "SELECT * FROM users WHERE created_at > ?"
    params: ["2024-01-01"]
```

### 3. Cache Tool

```python
class CacheTool(BaseTool):
    """Tool for caching operations."""
    
    async def execute(self, action: str, parameters: Dict[str, Any]) -> Any:
        if action == "get":
            return await self._get_cache(parameters["key"])
        elif action == "set":
            return await self._set_cache(
                parameters["key"], 
                parameters["value"],
                parameters.get("ttl", 3600)
            )
```

## Security Considerations

### 1. Sandboxing

Always implement proper sandboxing for tools that access system resources:

```python
class SecureFileTool(BaseTool):
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.sandbox_root = Path(config["sandbox_root"]).resolve()
        
    def _validate_path(self, path: str) -> Path:
        """Ensure path is within sandbox."""
        target = (self.sandbox_root / path).resolve()
        try:
            target.relative_to(self.sandbox_root)
            return target
        except ValueError:
            raise PermissionError("Path outside sandbox")
```

### 2. Input Validation

Always validate and sanitize inputs:

```python
async def validate_parameters(self, action: str, parameters: Dict[str, Any]) -> bool:
    """Validate parameters with strict rules."""
    if action == "query":
        sql = parameters.get("sql", "")
        # Prevent dangerous operations
        forbidden = ["DROP", "DELETE", "TRUNCATE", "ALTER"]
        if any(word in sql.upper() for word in forbidden):
            raise ValueError("Forbidden SQL operation")
    return True
```

### 3. Rate Limiting

Implement rate limiting for resource-intensive operations:

```python
from asyncio import Semaphore

class RateLimitedTool(BaseTool):
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.semaphore = Semaphore(config.get("max_concurrent", 10))
        
    async def execute(self, action: str, parameters: Dict[str, Any]) -> Any:
        async with self.semaphore:
            return await self._execute_internal(action, parameters)
```

## Error Handling

Implement comprehensive error handling:

```python
class RobustTool(BaseTool):
    async def execute(self, action: str, parameters: Dict[str, Any]) -> Any:
        try:
            # Validate first
            if not await self.validate_parameters(action, parameters):
                raise ValueError("Invalid parameters")
                
            # Execute with timeout
            return await asyncio.wait_for(
                self._execute_action(action, parameters),
                timeout=self.config.get("timeout", 30.0)
            )
            
        except asyncio.TimeoutError:
            return {"error": "Operation timed out", "action": action}
        except PermissionError as e:
            return {"error": f"Permission denied: {e}", "action": action}
        except Exception as e:
            # Log error
            self.logger.error(f"Tool error: {e}", exc_info=True)
            return {"error": str(e), "action": action}
```

## Testing Tools

### Unit Testing

```python
import pytest
from my_tools import FileSystemTool

@pytest.mark.asyncio
async def test_filesystem_tool():
    # Create tool with test config
    tool = FileSystemTool(config={"allowed_paths": ["/tmp/test"]})
    
    # Test read action
    params = {"path": "/tmp/test/file.txt"}
    result = await tool.execute("read", params)
    assert isinstance(result, str)
    
    # Test validation
    assert await tool.validate_parameters("read", params)
    assert not await tool.validate_parameters("read", {})
```

### Integration Testing

```python
@pytest.mark.asyncio
async def test_tool_in_pipeline():
    # Create pipeline with tool
    yaml_content = """
    steps:
      - id: read_config
        tool: filesystem
        action: read
        parameters:
          path: "config.json"
    """
    
    # Compile and execute
    pipeline = await compiler.compile(yaml_content, {})
    results = await control_system.execute_pipeline(pipeline)
    
    assert "read_config" in results
```

## Best Practices

### 1. Async-First Design
Always use async methods for I/O operations:

```python
async def execute(self, action: str, parameters: Dict[str, Any]) -> Any:
    # Use async libraries
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
```

### 2. Resource Management
Properly manage resources:

```python
class DatabaseTool(BaseTool):
    async def __aenter__(self):
        self.connection = await asyncpg.connect(self.config["dsn"])
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.connection.close()
```

### 3. Logging and Monitoring
Add comprehensive logging:

```python
import logging

class MonitoredTool(BaseTool):
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.logger = logging.getLogger(f"tool.{name}")
        
    async def execute(self, action: str, parameters: Dict[str, Any]) -> Any:
        start_time = time.time()
        self.logger.info(f"Executing {action} with {parameters}")
        
        try:
            result = await self._execute_internal(action, parameters)
            self.logger.info(f"Completed {action} in {time.time() - start_time:.2f}s")
            return result
        except Exception as e:
            self.logger.error(f"Failed {action}: {e}")
            raise
```

### 4. Configuration Management
Use structured configuration:

```python
from pydantic import BaseModel

class FileSystemConfig(BaseModel):
    allowed_paths: List[str]
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_extensions: List[str] = [".txt", ".json", ".yaml"]
    
class ConfiguredFileTool(BaseTool):
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.config = FileSystemConfig(**config)
```

## Advanced Patterns

### Tool Composition

```python
class ComposedTool(BaseTool):
    """Tool that combines multiple tools."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.tools = {
            "fs": FileSystemTool(config=config.get("fs_config", {})),
            "http": HTTPTool(config=config.get("http_config", {}))
        }
        
    async def execute(self, action: str, parameters: Dict[str, Any]) -> Any:
        if action == "fetch_and_save":
            # Fetch data via HTTP
            data = await self.tools["http"].execute("get", {
                "url": parameters["url"]
            })
            
            # Save to file system
            return await self.tools["fs"].execute("write", {
                "path": parameters["output_path"],
                "content": json.dumps(data)
            })
```

### Tool Middleware

```python
class ToolMiddleware:
    """Middleware for tool execution."""
    
    async def __call__(self, tool: BaseTool, action: str, parameters: Dict[str, Any]):
        # Pre-execution
        self.log_request(tool.name, action, parameters)
        
        # Execute
        result = await tool.execute(action, parameters)
        
        # Post-execution
        self.log_response(tool.name, action, result)
        self.update_metrics(tool.name, action)
        
        return result
```

## Conclusion

Tools extend the Orchestrator framework with custom functionality while maintaining security and reliability. Follow these guidelines to create robust, reusable tools that enhance your AI pipelines.