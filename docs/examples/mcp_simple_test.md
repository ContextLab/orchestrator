# Simple MCP Test Pipeline

**Pipeline**: `examples/mcp_simple_test.yaml`  
**Category**: MCP Integration  
**Complexity**: Beginner  
**Key Features**: MCP server connection, Tool discovery, DuckDuckGo search, Connection lifecycle

## Overview

The Simple MCP Test Pipeline demonstrates basic Model Context Protocol (MCP) server integration. It connects to a DuckDuckGo MCP server, discovers available tools, performs web searches, and manages the connection lifecycle, serving as an introduction to MCP capabilities.

## Key Features Demonstrated

### 1. MCP Server Connection
```yaml
- id: connect
  tool: mcp-server
  parameters:
    action: "connect"
    server_name: "duckduckgo"
    server_config:
      command: "python"
      args: ["src/orchestrator/tools/mcp_servers/duckduckgo_server.py"]
```

### 2. Tool Discovery
```yaml
- id: list_tools
  tool: mcp-server
  parameters:
    action: "list_tools"
    server_name: "duckduckgo"
```

### 3. Tool Execution
```yaml
- id: search
  tool: mcp-server
  parameters:
    action: "execute_tool"
    server_name: "duckduckgo"
    tool_name: "search"
    tool_params:
      query: "{{ parameters.search_query }}"
      max_results: 3
```

### 4. Connection Management
```yaml
- id: disconnect
  tool: mcp-server
  parameters:
    action: "disconnect"
    server_name: "duckduckgo"
```

## Pipeline Architecture

### Input Parameters
- **search_query** (optional): Search terms for DuckDuckGo (default: "Python programming")

### Processing Flow

1. **Connect** - Establish connection to DuckDuckGo MCP server
2. **List Tools** - Discover available tools and capabilities
3. **Search** - Execute search with specified query
4. **Save Results** - Store search results to JSON file
5. **Disconnect** - Cleanly close MCP server connection

### MCP Server Configuration

#### DuckDuckGo Server Setup
```yaml
server_config:
  command: "python"
  args: ["src/orchestrator/tools/mcp_servers/duckduckgo_server.py"]
  env: {}  # Environment variables if needed
```

#### Connection Parameters
- **server_name**: Unique identifier for this MCP server instance
- **command**: Executable to launch the MCP server
- **args**: Command-line arguments for the server
- **env**: Environment variables for the server process

## Usage Examples

### Basic Search Test
```bash
python scripts/run_pipeline.py examples/mcp_simple_test.yaml \
  -i search_query="artificial intelligence"
```

### Technology Search
```bash
python scripts/run_pipeline.py examples/mcp_simple_test.yaml \
  -i search_query="machine learning frameworks"
```

### News Search
```bash
python scripts/run_pipeline.py examples/mcp_simple_test.yaml \
  -i search_query="climate change solutions"
```

### Programming Topics
```bash
python scripts/run_pipeline.py examples/mcp_simple_test.yaml \
  -i search_query="JavaScript best practices"
```

## MCP Operations Detailed

### Connection Lifecycle
```yaml
# 1. Connect
action: "connect"
server_name: "duckduckgo"

# 2. Use server tools
action: "list_tools"
action: "execute_tool"

# 3. Disconnect
action: "disconnect"
server_name: "duckduckgo"
```

### Tool Discovery Process
```yaml
# List all available tools
action: "list_tools"
server_name: "duckduckgo"

# Typical response:
tools:
  - name: "search"
    description: "Search DuckDuckGo for information"
    input_schema:
      type: "object"
      properties:
        query: {type: "string"}
        max_results: {type: "integer", default: 10}
```

### Tool Execution Pattern
```yaml
# Execute specific tool
action: "execute_tool"
server_name: "duckduckgo"
tool_name: "search"
tool_params:
  query: "search terms here"
  max_results: 5
```

## Sample Output Structure

### Search Results JSON
```json
{
  "results": [
    {
      "title": "Python Programming - Official Site",
      "url": "https://www.python.org/",
      "snippet": "The official home of the Python Programming Language..."
    },
    {
      "title": "Learn Python Programming",
      "url": "https://www.learnpython.org/",
      "snippet": "Free interactive Python tutorial for beginners..."
    },
    {
      "title": "Python Programming Tutorial",
      "url": "https://www.tutorialspoint.com/python/",
      "snippet": "Python is a general-purpose interpreted, interactive..."
    }
  ],
  "query": "Python programming",
  "total_results": 3
}
```

### Pipeline Output Values
```yaml
outputs:
  connected: true
  tools_count: 1
  search_results: {...}  # Full search response
```

## File Output Pattern

### Generated Files
```
examples/outputs/mcp_simple_test/
├── python-programming_results.json      # For "Python programming"
├── artificial-intellige_results.json    # For "artificial intelligence"
└── machine-learning-fra_results.json    # For "machine learning frameworks"
```

### Filename Generation
```yaml
path: "{{ parameters.search_query[:20] | slugify }}_results.json"
# Truncates query to 20 chars and creates URL-safe filename
```

## Advanced MCP Features

### Error Handling
```yaml
# Connection validation
connected: "{{ connect.connected }}"

# Tool availability check
tools_count: "{{ list_tools.tools | length if list_tools.tools else 0 }}"
```

### Multiple Server Management
```yaml
# Connect to multiple MCP servers
servers:
  - name: "duckduckgo"
    config: {...}
  - name: "wikipedia"
    config: {...}
```

### Tool Parameter Validation
```yaml
tool_params:
  query: "{{ parameters.search_query }}"
  max_results: 3
  # Parameters validated against tool schema
```

## MCP Server Development

### Custom Server Structure
```python
# Basic MCP server implementation
class CustomMCPServer:
    def list_tools(self):
        return [
            {
                "name": "search",
                "description": "Search functionality",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_results": {"type": "integer"}
                    }
                }
            }
        ]
    
    def execute_tool(self, name, params):
        if name == "search":
            return self.search(params["query"], params.get("max_results", 10))
```

### Server Configuration Options
```yaml
server_config:
  command: "node"                    # Node.js server
  args: ["server.js", "--port=3000"] # Custom arguments
  env:
    API_KEY: "secret_key"            # Environment variables
    DEBUG: "true"                    # Debug mode
  timeout: 30                        # Connection timeout (seconds)
  retries: 3                         # Connection retry attempts
```

## Integration Patterns

### Sequential Tool Usage
```yaml
# Step 1: Search for information
- tool_name: "search"
  tool_params: {query: "topic"}

# Step 2: Use search results in next tool
- tool_name: "analyze"
  tool_params: {content: "{{ search.result }}"}
```

### Conditional Tool Execution
```yaml
condition: "{{ list_tools.tools | length > 0 }}"
# Only execute if tools are available
```

### Result Processing
```yaml
# Transform results for next steps
processed_results: "{{ search.result.results | map(attribute='title') | list }}"
```

## Common Use Cases

- **MCP Server Testing**: Validate new MCP server implementations
- **Tool Discovery**: Explore available MCP capabilities
- **Integration Validation**: Verify MCP server connectivity
- **Search Automation**: Automated web search capabilities  
- **Data Collection**: Gather information from multiple sources
- **Service Health Checks**: Monitor MCP server availability

## Troubleshooting

### Connection Issues
- Verify MCP server executable path is correct
- Check server process starts without errors
- Ensure required dependencies are installed
- Validate environment variables if needed

### Tool Execution Failures
- Confirm tool name matches server capabilities
- Validate tool parameters against schema
- Check server logs for detailed error messages
- Verify network connectivity for external APIs

### Result Processing Problems
- Ensure result structure matches expectations
- Handle cases where searches return no results
- Check JSON serialization for complex objects

## Related Examples
- [mcp_memory_workflow.md](mcp_memory_workflow.md) - Advanced MCP memory usage
- [mcp_integration_pipeline.md](mcp_integration_pipeline.md) - Complex MCP workflows
- [web_research_pipeline.md](web_research_pipeline.md) - Research using multiple sources

## Technical Requirements

- **Python Environment**: For DuckDuckGo MCP server
- **MCP Protocol**: Model Context Protocol implementation
- **Network Access**: Internet connectivity for web searches  
- **File System**: Write access for storing results
- **Process Management**: Ability to spawn and manage server processes

This pipeline serves as the foundation for understanding MCP integration and provides a template for building more complex MCP-powered workflows.