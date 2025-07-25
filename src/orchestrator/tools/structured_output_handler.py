"""Structured output handler using LangChain for consistent tool responses."""

import re
from typing import Any, Dict, List, Optional, Union

from langchain.schema import BaseOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

# Try to import pydantic, install if needed
try:
    from pydantic import BaseModel, Field
except ImportError:
    import subprocess
    import sys
    import logging

    logger = logging.getLogger(__name__)
    logger.info("Pydantic not found. Installing...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pydantic"])
        from pydantic import BaseModel, Field

        logger.info("Pydantic installed successfully")
    except Exception as e:
        raise ImportError(
            f"Failed to install pydantic: {e}. Please install manually with: pip install pydantic"
        )


class ToolCallResponse(BaseModel):
    """Structured response for tool calls."""

    action: str = Field(
        description="The action to perform (e.g., 'save_file', 'execute_code', 'analyze')"
    )
    parameters: Dict[str, Any] = Field(description="Parameters for the action")
    reasoning: Optional[str] = Field(description="Brief reasoning for this action")

    class Config:
        json_schema_extra = {
            "example": {
                "action": "save_file",
                "parameters": {
                    "path": "output/report.md",
                    "content": "# Report\n\nContent here...",
                },
                "reasoning": "Saving the generated report to the specified file",
            }
        }


class FileOperationResponse(BaseModel):
    """Structured response for file operations."""

    operation: str = Field(
        description="Type of operation: 'read', 'write', 'append', 'delete'"
    )
    filepath: str = Field(description="Full path to the file")
    content: Optional[str] = Field(description="Content for write/append operations")
    success: bool = Field(
        default=True, description="Whether the operation should succeed"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "operation": "write",
                "filepath": "examples/output/result.md",
                "content": "# Results\n\nProcessed data...",
                "success": True,
            }
        }


class CodeExecutionResponse(BaseModel):
    """Structured response for code execution."""

    language: str = Field(description="Programming language")
    code: str = Field(description="Code to execute")
    timeout: Optional[int] = Field(
        default=30, description="Execution timeout in seconds"
    )
    capture_output: bool = Field(default=True, description="Whether to capture output")

    class Config:
        json_schema_extra = {
            "example": {
                "language": "python",
                "code": "print('Hello, World!')",
                "timeout": 30,
                "capture_output": True,
            }
        }


class AnalysisResponse(BaseModel):
    """Structured response for analysis tasks."""

    analysis_type: str = Field(description="Type of analysis performed")
    findings: List[str] = Field(description="Key findings from the analysis")
    recommendations: Optional[List[str]] = Field(
        description="Recommendations based on analysis"
    )
    data: Optional[Dict[str, Any]] = Field(description="Structured data from analysis")

    class Config:
        json_schema_extra = {
            "example": {
                "analysis_type": "code_quality",
                "findings": [
                    "Code follows PEP8 standards",
                    "Good test coverage at 85%",
                ],
                "recommendations": [
                    "Add more edge case tests",
                    "Consider refactoring large functions",
                ],
                "data": {"coverage": 0.85, "complexity": "medium"},
            }
        }


class StructuredOutputHandler:
    """Handles structured output parsing for tool responses."""

    def __init__(self):
        self.parsers = {
            "tool_call": PydanticOutputParser(pydantic_object=ToolCallResponse),
            "file_operation": PydanticOutputParser(
                pydantic_object=FileOperationResponse
            ),
            "code_execution": PydanticOutputParser(
                pydantic_object=CodeExecutionResponse
            ),
            "analysis": PydanticOutputParser(pydantic_object=AnalysisResponse),
        }

    def get_parser(self, output_type: str) -> BaseOutputParser:
        """Get the appropriate parser for the output type."""
        return self.parsers.get(output_type, self.parsers["tool_call"])

    def create_prompt_with_format(
        self, base_prompt: str, output_type: str = "tool_call"
    ) -> str:
        """Create a prompt with format instructions."""
        parser = self.get_parser(output_type)
        format_instructions = parser.get_format_instructions()

        prompt_template = PromptTemplate(
            template="{base_prompt}\n\n{format_instructions}",
            input_variables=["base_prompt"],
            partial_variables={"format_instructions": format_instructions},
        )

        return prompt_template.format(base_prompt=base_prompt)

    def parse_response(
        self, response: str, output_type: str = "tool_call"
    ) -> Union[BaseModel, Dict[str, Any]]:
        """Parse a response into structured output."""
        parser = self.get_parser(output_type)

        try:
            # Try to parse directly
            return parser.parse(response)
        except Exception:
            # If direct parsing fails, try to extract JSON
            json_match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL)
            if json_match:
                try:
                    json_str = json_match.group(1)
                    return parser.parse(json_str)
                except Exception:
                    pass

            # If JSON extraction fails, try to find JSON-like structure
            json_match = re.search(r"\{[^{}]*\}", response)
            if json_match:
                try:
                    json_str = json_match.group(0)
                    return parser.parse(json_str)
                except Exception:
                    pass

            # If all parsing fails, return a default structure
            if output_type == "file_operation":
                return FileOperationResponse(
                    operation="write",
                    filepath="output/default.txt",
                    content=response,
                    success=False,
                )
            else:
                return {"error": "Failed to parse response", "raw_response": response}

    def ensure_tool_execution(self, action: str, response: str) -> Dict[str, Any]:
        """Ensure that a tool action is properly formatted for execution."""
        # Map common action patterns to structured responses
        action_patterns = {
            r"save.*file|write.*file|create.*file": "file_operation",
            r"execute.*code|run.*script": "code_execution",
            r"analyze|examine|inspect": "analysis",
        }

        output_type = "tool_call"
        for pattern, otype in action_patterns.items():
            if re.search(pattern, action, re.IGNORECASE):
                output_type = otype
                break

        parsed = self.parse_response(response, output_type)

        # Convert to execution format
        if isinstance(parsed, FileOperationResponse):
            return {
                "tool": "filesystem",
                "action": "write",
                "parameters": {
                    "path": parsed.filepath,
                    "content": parsed.content or "",
                },
            }
        elif isinstance(parsed, CodeExecutionResponse):
            return {
                "tool": "terminal",
                "action": "execute",
                "parameters": {
                    "command": parsed.code,
                    "timeout": parsed.timeout,
                    "capture_output": parsed.capture_output,
                },
            }
        elif isinstance(parsed, AnalysisResponse):
            return {
                "tool": "analysis",
                "action": "analyze",
                "parameters": {
                    "type": parsed.analysis_type,
                    "findings": parsed.findings,
                    "recommendations": parsed.recommendations,
                    "data": parsed.data,
                },
            }
        elif isinstance(parsed, ToolCallResponse):
            return {
                "tool": parsed.action.split("_")[0],
                "action": parsed.action,
                "parameters": parsed.parameters,
            }
        else:
            return {
                "tool": "unknown",
                "action": action,
                "parameters": {"response": str(parsed)},
            }
