# MCP Integration Pipeline

**Pipeline**: `examples/mcp_integration_pipeline.yaml`  
**Category**: External Integrations & APIs  
**Complexity**: Advanced  
**Key Features**: MCP server connections, External tool execution, Memory management, Web search integration, Service orchestration

## Overview

The MCP Integration Pipeline demonstrates how to connect to and utilize Model Context Protocol (MCP) servers for external service integration. It showcases connecting to a DuckDuckGo search MCP server, executing search operations, managing results in memory, and handling the full lifecycle of external service interactions.

## Key Features Demonstrated

### 1. MCP Server Connection
```yaml
- id: connect_mcp
  tool: mcp-server
  action: execute
  parameters:
    action: "connect"
    server_name: "duckduckgo-search"
    server_config:
      command: "python"
      args: ["src/orchestrator/tools/mcp_servers/duckduckgo_server.py"]
      env: {}
```

### 2. Tool Discovery
```yaml
- id: list_tools
  tool: mcp-server
  action: execute
  parameters:
    action: "list_tools"
    server_name: "duckduckgo-search"
```

### 3. Memory Management
```yaml
- id: store_context
  tool: mcp-memory
  action: execute
  parameters:
    action: "store"
    namespace: "search_session"
    key: "query"
    value: "{{ parameters.search_query }}"
    ttl: 3600  # 1 hour
```

### 4. External Tool Execution
```yaml
- id: search_web
  tool: mcp-server
  action: execute
  parameters:
    action: "execute_tool"
    server_name: "duckduckgo-search"
    tool_name: "search"
    tool_params:
      query: "{{ parameters.search_query }}"
      max_results: 5
```

### 5. Service Lifecycle Management
```yaml
# Proper cleanup
- id: disconnect_mcp
  tool: mcp-server
  action: execute
  parameters:
    action: "disconnect"
    server_name: "duckduckgo-search"
```

## Pipeline Architecture

### Input Parameters
- **search_query** (optional): Search query to execute (default: "AI orchestration frameworks")

### Processing Flow

1. **MCP Server Connection** - Establishes connection to DuckDuckGo MCP server
2. **Tool Discovery** - Lists available tools from the connected server
3. **Context Storage** - Stores search query in memory for session tracking
4. **Search Execution** - Executes web search using MCP server tool
5. **Result Storage** - Saves search results in memory with TTL
6. **Memory Inspection** - Lists stored memory contents for verification
7. **Result Persistence** - Saves formatted results to JSON file
8. **Service Cleanup** - Properly disconnects from MCP server

## Usage Examples

### Basic Web Search
```bash
python scripts/run_pipeline.py examples/mcp_integration_pipeline.yaml \
  -i search_query="artificial intelligence ethics"
```

### Technology Research
```bash
python scripts/run_pipeline.py examples/mcp_integration_pipeline.yaml \
  -i search_query="quantum computing breakthroughs"
```

### Market Research
```bash
python scripts/run_pipeline.py examples/mcp_integration_pipeline.yaml \
  -i search_query="blockchain cryptocurrency trends"
```

## Sample Output Structure

### JSON Results File
```json
{
  "query": "artificial intelligence ethics governance",
  "timestamp": "2025-08-20T15:48:23.554084",
  "total_results": 5,
  "results": [
    {
      "title": "AI Ethics Guidelines for Governance",
      "url": "https://example.com/ai-ethics",
      "snippet": "Comprehensive guide to AI ethics and governance frameworks",
      "rank": 1
    }
  ],
  "memory_keys": ["query", "results"]
}
```

### Generated Files
Check actual search results in: [*_results.json](../../examples/outputs/mcp_integration/)

## Technical Implementation

### MCP Server Configuration
```yaml
server_config:
  command: "python"
  args: ["src/orchestrator/tools/mcp_servers/duckduckgo_server.py"]
  env: {}
# Configures the external MCP server process
```

### Memory Management with TTL
```yaml
parameters:
  action: "store"
  namespace: "search_session"
  key: "results"
  value: "{{ search_web.result }}"
  ttl: 3600  # Results expire after 1 hour
```

### Dynamic File Naming
```yaml
path: "examples/outputs/mcp_integration/{{ parameters.search_query[:25] | slugify }}_results.json"
# Creates unique filenames based on search query
```

## Advanced Features

### Service Discovery
The pipeline discovers available tools before execution:
- Lists all tools provided by the MCP server
- Validates tool availability before execution
- Enables dynamic tool selection

### Session Management
- Namespaced memory storage for session isolation
- TTL-based automatic cleanup of stored data
- Memory key listing for debugging and validation

### Error Handling
- Proper connection/disconnection lifecycle
- Dependency management ensures proper execution order
- Graceful handling of search failures

### Structured Output
```yaml
content: "{{ {'query': parameters.search_query, 'timestamp': execution.timestamp, 'results': search_web.result.results if search_web.result else [], 'total': search_web.result.total if search_web.result else 0} | tojson(indent=2) }}"
```

## Common Use Cases

- **Research Automation**: Automated web search for research topics
- **Market Intelligence**: Competitive intelligence and market research
- **Content Discovery**: Finding relevant content and sources
- **Data Collection**: Gathering information from web sources
- **Knowledge Base Building**: Building searchable knowledge repositories
- **Trend Analysis**: Monitoring trends and developments in specific domains

## Best Practices Demonstrated

1. **Proper Lifecycle Management**: Connect → Use → Disconnect pattern
2. **Memory Management**: Organized storage with TTL for automatic cleanup
3. **Service Discovery**: Dynamic tool discovery before execution
4. **Error Resilience**: Defensive programming with existence checks
5. **Structured Output**: Consistent, machine-readable result formatting
6. **Resource Cleanup**: Explicit service disconnection

## Troubleshooting

### Common Issues
- **Connection Failures**: Ensure MCP server is accessible and configured correctly
- **Tool Discovery**: Verify server provides expected tools
- **Memory Issues**: Check namespace and key naming for conflicts
- **Permission Errors**: Verify file system permissions for output directories

### Performance Considerations
- **Connection Overhead**: MCP server connections have setup/teardown costs
- **Memory Usage**: Consider TTL settings for long-running pipelines
- **Search Limits**: Adjust max_results based on processing requirements
- **Network Dependencies**: External services may have rate limits

## Related Examples
- [mcp_memory_workflow.md](mcp_memory_workflow.md) - Advanced memory management patterns
- [mcp_simple_test.md](mcp_simple_test.md) - Basic MCP server testing
- [web_research_pipeline.md](web_research_pipeline.md) - Web research workflows
- [working_web_search.yaml](working_web_search.md) - Alternative web search implementations

## Technical Requirements

- **Tools**: mcp-server, mcp-memory, filesystem
- **External Services**: DuckDuckGo MCP server
- **Python Environment**: MCP server dependencies
- **Network Access**: Internet connectivity for web searches

This pipeline provides a foundation for integrating external services and APIs through the MCP protocol, enabling powerful service orchestration and data collection workflows.