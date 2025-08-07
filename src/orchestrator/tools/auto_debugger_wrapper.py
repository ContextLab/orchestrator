"""
AutoDebugger Tool Wrapper for Orchestrator Integration

This module provides a wrapper to integrate the LangChain-based AutoDebuggerTool
with the orchestrator's Tool system, enabling it to be used in pipelines.
"""

import json
import logging
from typing import Any, Dict, List

from .base import Tool
from .auto_debugger import AutoDebuggerTool as LangChainAutoDebuggerTool

logger = logging.getLogger(__name__)


class AutoDebuggerTool(Tool):
    """
    Orchestrator wrapper for the LangChain AutoDebuggerTool.
    
    This wrapper integrates the universal debugging capabilities of the AutoDebugger
    with the orchestrator's pipeline system, allowing it to be used as a standard
    pipeline tool for debugging any content or process.
    """
    
    def __init__(self):
        super().__init__(
            name="auto_debugger",
            description="""Universal debugging tool that analyzes problems, executes fixes, and validates results.
            
Use this tool when you need to:
- Debug and fix generated code (Python, JavaScript, SQL, etc.)
- Resolve document compilation errors (LaTeX, Markdown, HTML)
- Fix data format issues (JSON, YAML, CSV)
- Debug API integration problems
- Correct configuration errors
- Fix test failures
- Debug any other content or process

The tool automatically analyzes the problem, suggests fixes, executes them using real tools and models, and validates the results until the issue is resolved.

Input parameters:
- task_description: What you're trying to accomplish (required)
- content_to_debug: The content that needs debugging (optional)
- error_context: Error messages or descriptions (optional)
- expected_outcome: What should happen when fixed (optional)
- available_tools: Specific tools to use (optional)

Returns structured JSON with debug results and fixed content."""
        )
        
        # Initialize the LangChain AutoDebugger tool lazily
        self._langchain_tool = None
        
        # Define parameters for the orchestrator tool system
        self.add_parameter(
            name="task_description",
            type="string",
            description="What you're trying to accomplish",
            required=True
        )
        
        self.add_parameter(
            name="content_to_debug",
            type="string", 
            description="Code, document, data, or other content to debug",
            required=False,
            default=""
        )
        
        self.add_parameter(
            name="error_context",
            type="string",
            description="Error messages or failure descriptions",
            required=False,
            default=""
        )
        
        self.add_parameter(
            name="expected_outcome",
            type="string",
            description="What should happen when fixed",
            required=False,
            default=""
        )
        
        self.add_parameter(
            name="available_tools",
            type="array",
            description="Specific tools to use (optional list of strings)",
            required=False,
            default=None
        )
        
        logger.info("AutoDebuggerTool wrapper initialized for orchestrator integration")
    
    async def _execute_impl(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the AutoDebugger tool.
        
        This method wraps the LangChain AutoDebuggerTool's _arun method and
        provides structured output for the orchestrator pipeline system.
        """
        # Extract parameters
        task_description = kwargs.get("task_description", "")
        content_to_debug = kwargs.get("content_to_debug", "")
        error_context = kwargs.get("error_context", "")
        expected_outcome = kwargs.get("expected_outcome", "")
        available_tools = kwargs.get("available_tools", None)
        
        # Validate required parameters
        if not task_description.strip():
            raise ValueError("task_description is required and cannot be empty")
        
        logger.info(f"AutoDebugger executing: {task_description}")
        
        try:
            # Initialize LangChain tool if not already done
            if self._langchain_tool is None:
                self._langchain_tool = LangChainAutoDebuggerTool()
            
            # Execute the LangChain AutoDebugger tool
            result_json = await self._langchain_tool._arun(
                task_description=task_description,
                content_to_debug=content_to_debug,
                error_context=error_context,
                expected_outcome=expected_outcome,
                available_tools=available_tools
            )
            
            # Parse the JSON result from the LangChain tool
            debug_result = json.loads(result_json)
            
            # Structure the result for orchestrator pipelines
            orchestrator_result = {
                "success": debug_result.get("success", False),
                "session_id": debug_result.get("session_id", ""),
                "task_description": debug_result.get("task_description", task_description),
                "total_iterations": debug_result.get("total_iterations", 0),
                "debug_summary": debug_result.get("debug_summary", ""),
                "modifications_made": debug_result.get("modifications_made", []),
                "tools_used": debug_result.get("tools_used", []),
                "execution_time": debug_result.get("execution_time", 0.0),
            }
            
            # Add success-specific or failure-specific fields
            if debug_result.get("success", False):
                orchestrator_result.update({
                    "final_content": debug_result.get("final_content", ""),
                    "validation": debug_result.get("validation", {}),
                })
                
                logger.info(f"AutoDebugger succeeded after {debug_result.get('total_iterations', 0)} iterations")
                
            else:
                orchestrator_result.update({
                    "error_message": debug_result.get("error_message", "Debugging failed"),
                    "final_error": debug_result.get("final_error", ""),
                })
                
                logger.warning(f"AutoDebugger failed: {debug_result.get('error_message', 'Unknown error')}")
            
            # Always include the raw debug result for advanced users
            orchestrator_result["raw_debug_result"] = debug_result
            
            return orchestrator_result
            
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse AutoDebugger result as JSON: {e}"
            logger.error(error_msg)
            return {
                "success": False,
                "error_message": error_msg,
                "task_description": task_description,
                "total_iterations": 0,
                "debug_summary": "JSON parsing error",
                "modifications_made": [],
                "tools_used": [],
                "execution_time": 0.0,
            }
            
        except Exception as e:
            error_msg = f"AutoDebugger execution failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "success": False,
                "error_message": error_msg,
                "task_description": task_description,
                "total_iterations": 0,
                "debug_summary": f"Execution error: {str(e)}",
                "modifications_made": [],
                "tools_used": [],
                "execution_time": 0.0,
            }
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get AutoDebugger capabilities for tool discovery.
        
        Returns:
            Dictionary describing the tool's capabilities
        """
        return {
            "categories": [
                "debugging", 
                "code_analysis", 
                "error_resolution", 
                "content_fixing",
                "validation",
                "automation"
            ],
            "content_types": [
                "python", "javascript", "sql", "html", "css",
                "latex", "markdown", "yaml", "json", "xml", 
                "csv", "text", "configuration"
            ],
            "operations": [
                "analyze", "fix", "validate", "debug", "repair",
                "resolve", "correct", "improve", "test"
            ],
            "features": [
                "multi_iteration_debugging",
                "real_llm_analysis", 
                "real_tool_execution",
                "automatic_validation",
                "comprehensive_error_handling",
                "universal_content_support",
                "no_mock_implementations"
            ]
        }
    
    def is_suitable_for_action(self, action_description: str) -> bool:
        """
        Determine if this tool is suitable for a given action.
        
        Args:
            action_description: Description of the action to perform
            
        Returns:
            True if this tool can handle the action
        """
        action_lower = action_description.lower()
        
        # Keywords that indicate debugging/fixing needs
        debug_keywords = [
            "debug", "fix", "repair", "resolve", "correct", "error",
            "issue", "problem", "broken", "failing", "fail", "bug",
            "syntax error", "runtime error", "compilation error",
            "validation error", "format error", "parse error"
        ]
        
        return any(keyword in action_lower for keyword in debug_keywords)
    
    def get_usage_examples(self) -> List[Dict[str, Any]]:
        """
        Get usage examples for documentation and help.
        
        Returns:
            List of usage examples
        """
        return [
            {
                "name": "Debug Python Syntax Errors",
                "description": "Fix Python code with syntax errors",
                "parameters": {
                    "task_description": "Fix Python syntax errors in data processing script",
                    "content_to_debug": "def process_data(data):\n    for item in data\n        result.append(item.upper())",
                    "error_context": "SyntaxError: invalid syntax (missing colon)",
                    "expected_outcome": "Working Python code that processes data correctly"
                }
            },
            {
                "name": "Debug API Integration",
                "description": "Fix problems with API calls and error handling",
                "parameters": {
                    "task_description": "Fix API integration issues in user data fetcher",
                    "content_to_debug": "response = requests.get(url)\ndata = response.json()",
                    "error_context": "No error handling, possible 404 errors",
                    "expected_outcome": "Robust API code with proper error handling"
                }
            },
            {
                "name": "Debug LaTeX Compilation",
                "description": "Fix LaTeX document compilation errors",
                "parameters": {
                    "task_description": "Fix LaTeX compilation errors to generate PDF",
                    "content_to_debug": "\\documentclass{article}\n\\begin{document}\nHello world\n\\end{document",
                    "error_context": "Missing closing brace in \\end{document}",
                    "expected_outcome": "LaTeX document that compiles successfully"
                }
            },
            {
                "name": "Debug YAML Configuration",
                "description": "Fix YAML syntax and structure errors",
                "parameters": {
                    "task_description": "Fix YAML configuration syntax errors",
                    "content_to_debug": "name: test\nconfig:\n  port: 8080\n   timeout: 30",
                    "error_context": "YAML indentation error",
                    "expected_outcome": "Valid YAML configuration file"
                }
            },
            {
                "name": "Debug Data Processing",
                "description": "Fix data format and processing issues",
                "parameters": {
                    "task_description": "Fix CSV processing code to handle malformed data",
                    "content_to_debug": "data = pd.read_csv('file.csv')\nresult = data.groupby('category').sum()",
                    "error_context": "KeyError: 'category' column missing in some files",
                    "expected_outcome": "Robust data processing with error handling"
                }
            }
        ]