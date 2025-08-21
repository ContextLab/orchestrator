#!/usr/bin/env python3
"""DuckDuckGo MCP Server - Real implementation using duckduckgo-search library."""

import asyncio
import json
import sys
import logging
from typing import Any, Dict, List
from ddgs import DDGS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DuckDuckGoMCPServer:
    """MCP server that provides DuckDuckGo search capabilities."""
    
    def __init__(self):
        self.request_id_counter = 0
        self.capabilities = {
            "tools": True,
            "resources": False,
            "prompts": False
        }
        self.tools = [
            {
                "name": "search",
                "description": "Search DuckDuckGo for information",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results",
                            "default": 10
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "search_news",
                "description": "Search DuckDuckGo news",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "News search query"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results",
                            "default": 10
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "search_images",
                "description": "Search DuckDuckGo images",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Image search query"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results",
                            "default": 10
                        }
                    },
                    "required": ["query"]
                }
            }
        ]
    
    async def handle_initialize(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialization request."""
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {
                "protocolVersion": "1.0",
                "capabilities": self.capabilities,
                "serverInfo": {
                    "name": "duckduckgo-search",
                    "version": "1.0.0"
                }
            }
        }
    
    async def handle_tools_list(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/list request."""
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {
                "tools": self.tools
            }
        }
    
    async def handle_tools_call(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request to execute a tool."""
        params = request.get("params", {})
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        try:
            if tool_name == "search":
                result = await self.search_web(
                    arguments.get("query"),
                    arguments.get("max_results", 10)
                )
            elif tool_name == "search_news":
                result = await self.search_news(
                    arguments.get("query"),
                    arguments.get("max_results", 10)
                )
            elif tool_name == "search_images":
                result = await self.search_images(
                    arguments.get("query"),
                    arguments.get("max_results", 10)
                )
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "error": {
                        "code": -32601,
                        "message": f"Unknown tool: {tool_name}"
                    }
                }
            
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "result": result
            }
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {
                    "code": -32603,
                    "message": str(e)
                }
            }
    
    async def search_web(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """Perform real DuckDuckGo web search."""
        loop = asyncio.get_event_loop()
        
        def _search():
            with DDGS() as ddgs:
                results = list(ddgs.text(
                    query,
                    region="us-en",
                    safesearch="moderate",
                    max_results=max_results
                ))
                return results
        
        # Run synchronous DuckDuckGo search in executor
        ddg_results = await loop.run_in_executor(None, _search)
        
        # Format results
        formatted_results = []
        for idx, result in enumerate(ddg_results):
            formatted_results.append({
                "title": result.get("title", ""),
                "url": result.get("href", ""),
                "snippet": result.get("body", ""),
                "rank": idx + 1
            })
        
        return {
            "results": formatted_results,
            "total": len(formatted_results),
            "query": query
        }
    
    async def search_news(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """Perform real DuckDuckGo news search."""
        loop = asyncio.get_event_loop()
        
        def _search():
            with DDGS() as ddgs:
                results = list(ddgs.news(
                    query,
                    region="us-en",
                    safesearch="moderate",
                    max_results=max_results
                ))
                return results
        
        # Run synchronous DuckDuckGo search in executor
        ddg_results = await loop.run_in_executor(None, _search)
        
        # Format results
        formatted_results = []
        for idx, result in enumerate(ddg_results):
            formatted_results.append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "snippet": result.get("body", ""),
                "date": result.get("date", ""),
                "source": result.get("source", ""),
                "rank": idx + 1
            })
        
        return {
            "results": formatted_results,
            "total": len(formatted_results),
            "query": query
        }
    
    async def search_images(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """Perform real DuckDuckGo image search."""
        loop = asyncio.get_event_loop()
        
        def _search():
            with DDGS() as ddgs:
                results = list(ddgs.images(
                    query,
                    region="us-en",
                    safesearch="moderate",
                    max_results=max_results
                ))
                return results
        
        # Run synchronous DuckDuckGo search in executor
        ddg_results = await loop.run_in_executor(None, _search)
        
        # Format results
        formatted_results = []
        for idx, result in enumerate(ddg_results):
            formatted_results.append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "image": result.get("image", ""),
                "thumbnail": result.get("thumbnail", ""),
                "width": result.get("width", 0),
                "height": result.get("height", 0),
                "source": result.get("source", ""),
                "rank": idx + 1
            })
        
        return {
            "results": formatted_results,
            "total": len(formatted_results),
            "query": query
        }
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Route requests to appropriate handlers."""
        method = request.get("method")
        
        if method == "initialize":
            return await self.handle_initialize(request)
        elif method == "tools/list":
            return await self.handle_tools_list(request)
        elif method == "tools/call":
            return await self.handle_tools_call(request)
        elif method == "shutdown":
            # Acknowledge shutdown
            return {"jsonrpc": "2.0", "result": {}}
        else:
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            }
    
    async def run(self):
        """Run the MCP server, reading from stdin and writing to stdout."""
        logger.info("DuckDuckGo MCP server starting...")
        
        # Use line-buffered streams
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)
        
        while True:
            try:
                # Read a line from stdin
                line = await reader.readline()
                if not line:
                    break
                
                # Parse JSON-RPC request
                request = json.loads(line.decode().strip())
                logger.info(f"Received request: {request.get('method')}")
                
                # Handle the request
                response = await self.handle_request(request)
                
                # Write response to stdout
                response_str = json.dumps(response) + "\n"
                sys.stdout.write(response_str)
                sys.stdout.flush()
                
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON received: {e}")
                error_response = {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32700,
                        "message": "Parse error"
                    }
                }
                sys.stdout.write(json.dumps(error_response) + "\n")
                sys.stdout.flush()
            except Exception as e:
                logger.error(f"Server error: {e}")
                break
        
        logger.info("DuckDuckGo MCP server shutting down")


if __name__ == "__main__":
    server = DuckDuckGoMCPServer()
    asyncio.run(server.run())