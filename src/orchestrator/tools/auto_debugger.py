"""
AutoDebugger Tool - Universal debugging tool for pipeline execution.

This tool implements the AutoDebugger from Issue #201 - a comprehensive pipeline tool
that can debug and fix ANY content or process using a three-step loop:
Analyze → Execute → Fix

Key Features:
- Universal debugging: code, documents, data, configurations, APIs, etc.
- Real LLM analysis with multiple specialized models
- Real tool execution using all orchestrator tools
- Pattern recognition and learning from debugging history
- Integration with model and tool registries
- NO MOCKS - all functionality uses real systems or raises exceptions
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


@dataclass
class AutoDebugContext:
    """Context for debugging session."""
    task_description: str
    content: str
    error_context: str
    expected_outcome: str
    available_tools: List[str]
    iteration: int = 0
    debug_history: List[Dict[str, Any]] = field(default_factory=list)
    modifications_made: List[str] = field(default_factory=list)
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class AnalysisResult:
    """Result of LLM analysis of debugging situation."""
    root_cause: str
    suggested_action: str
    tool_to_use: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_outcome: str = ""
    risk_assessment: str = "medium"
    confidence: float = 0.5
    reasoning: str = ""


@dataclass
class ExecutionResult:
    """Result of executing a debugging action."""
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    error_message: str = ""
    traceback: str = ""
    tool_used: Optional[str] = None
    model_used: Optional[str] = None
    parameters_used: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    fixed_content: Optional[str] = None
    analysis_used: Optional[AnalysisResult] = None


@dataclass
class ValidationResult:
    """Result of validating a debugging solution."""
    is_resolved: bool
    remaining_issues: List[str] = field(default_factory=list)
    meets_expectations: bool = False
    quality_score: float = 0.0
    reasoning: str = ""
    validation_details: Dict[str, Any] = field(default_factory=dict)


class AutoDebuggerInput(BaseModel):
    """Input schema for AutoDebugger tool."""
    task_description: str = Field(description="What you're trying to accomplish")
    content_to_debug: str = Field(default="", description="Code, document, data, or other content to debug")
    error_context: str = Field(default="", description="Error messages or failure descriptions")
    expected_outcome: str = Field(default="", description="What should happen when fixed")
    available_tools: Optional[List[str]] = Field(default=None, description="Specific tools to use (optional)")


class AutoDebuggerTool(BaseTool):
    """
    Universal debugging tool that can debug and fix ANY content or process.
    
    This tool implements a three-step loop: Analyze → Execute → Validate
    and can be used as a pipeline step to debug:
    - Generated code (Python, JavaScript, SQL, etc.)
    - Document compilation (LaTeX, Markdown, HTML)
    - Data processing issues
    - API integration problems
    - Configuration files
    - System setup and installation
    - And much more...
    
    NO MOCKS - All functionality uses real models, tools, and systems.
    Raises exceptions when resources are unavailable.
    """
    
    name: str = "auto_debugger"
    description: str = """
    Universal debugging tool that analyzes problems, executes fixes, and validates results.
    
    Use this tool when you need to:
    - Debug and fix generated code
    - Resolve document compilation errors  
    - Fix data format issues
    - Debug API integration problems
    - Correct configuration errors
    - Fix test failures
    - Debug any other content or process
    
    The tool will automatically analyze the problem, suggest fixes, execute them,
    and validate the results until the issue is resolved.
    
    Input parameters:
    - task_description: What you're trying to accomplish
    - content_to_debug: The content that needs debugging (optional)
    - error_context: Error messages or descriptions (optional)
    - expected_outcome: What should happen when fixed (optional)
    - available_tools: Specific tools to use (optional)
    
    Returns JSON with debug results and fixed content.
    """
    
    args_schema: type[AutoDebuggerInput] = AutoDebuggerInput
    
    # Use __init_subclass__ to avoid pydantic field validation issues
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
    
    def __init__(self):
        super().__init__()
        
        # Initialize model configurations with smart defaults (use object.__setattr__ to bypass pydantic)
        object.__setattr__(self, 'model_configurations', self._initialize_model_config())
        
        # Initialize model instances (will be set up on first use)
        object.__setattr__(self, 'analyzer_model', None)
        object.__setattr__(self, 'fixer_model', None)
        object.__setattr__(self, 'validator_model', None)
        
        # Tool and model registries (will be injected)
        object.__setattr__(self, 'tool_registry', None)
        object.__setattr__(self, 'model_registry', None)
        
        # Debugging configuration
        object.__setattr__(self, 'max_debug_iterations', 10)
        
        logger.info("AutoDebuggerTool initialized")
    
    def _initialize_model_config(self) -> Dict[str, Any]:
        """Initialize model configuration with smart defaults and fallbacks."""
        return {
            "analyzer_model": {
                "default": {
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-20250514",
                    "temperature": 0.1,
                    "max_tokens": 4000
                },
                "fallbacks": [
                    {
                        "provider": "openai", 
                        "model": "gpt-4o",
                        "temperature": 0.1,
                        "max_tokens": 4000
                    },
                    {
                        "provider": "ollama",
                        "model": "llama3.1:70b",
                        "temperature": 0.1,
                        "max_tokens": 4000
                    }
                ]
            },
            "fixer_model": {
                "default": {
                    "provider": "openai",
                    "model": "gpt-4o", 
                    "temperature": 0.2,
                    "max_tokens": 8000
                },
                "fallbacks": [
                    {
                        "provider": "anthropic",
                        "model": "claude-sonnet-4-20250514",
                        "temperature": 0.2, 
                        "max_tokens": 8000
                    }
                ]
            },
            "validator_model": {
                "default": {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "temperature": 0.0,
                    "max_tokens": 2000
                },
                "fallbacks": [
                    {
                        "provider": "anthropic", 
                        "model": "claude-sonnet-4-20250514",
                        "temperature": 0.0,
                        "max_tokens": 2000
                    }
                ]
            }
        }
    
    async def _arun(self, 
                   task_description: str,
                   content_to_debug: str = "",
                   error_context: str = "",
                   expected_outcome: str = "",
                   available_tools: Optional[List[str]] = None) -> str:
        """
        Main AutoDebugger execution - can debug ANY content or process.
        
        Args:
            task_description: What you're trying to accomplish
            content_to_debug: Code, document, data, or other content to debug
            error_context: Error messages or failure descriptions
            expected_outcome: What should happen when fixed
            available_tools: Specific tools to use (optional)
            
        Returns:
            JSON string with debug results and fixed content
        """
        logger.info(f"AutoDebugger starting: {task_description}")
        
        # Initialize registries - REAL ONLY, no mocks
        await self._initialize_registries()
        
        # Create debugging context
        debug_context = AutoDebugContext(
            task_description=task_description,
            content=content_to_debug,
            error_context=error_context,
            expected_outcome=expected_outcome,
            available_tools=available_tools or await self._get_available_tools()
        )
        
        # Execute the three-step debugging loop
        return await self._execute_debug_loop(debug_context)
    
    def _run(self, 
             task_description: str,
             content_to_debug: str = "",
             error_context: str = "",
             expected_outcome: str = "",
             available_tools: Optional[List[str]] = None) -> str:
        """
        Synchronous run method required by LangChain BaseTool.
        
        This method provides a sync wrapper around the async implementation.
        """
        import asyncio
        
        # If we're already in an async context, we need to handle it properly
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, safe to use asyncio.run
            return asyncio.run(self._arun(
                task_description=task_description,
                content_to_debug=content_to_debug,
                error_context=error_context,
                expected_outcome=expected_outcome,
                available_tools=available_tools
            ))
        else:
            # We're in an async context, need to create a new thread
            import concurrent.futures
            import threading
            
            def run_in_thread():
                return asyncio.run(self._arun(
                    task_description=task_description,
                    content_to_debug=content_to_debug,
                    error_context=error_context,
                    expected_outcome=expected_outcome,
                    available_tools=available_tools
                ))
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result()
    
    async def _initialize_registries(self):
        """Initialize tool and model registries - REAL ONLY."""
        if self.tool_registry is None:
            # Get REAL tool registry from orchestrator
            try:
                from .base import default_registry
                object.__setattr__(self, 'tool_registry', default_registry)
                if self.tool_registry is None:
                    raise RuntimeError("Tool registry not available")
            except ImportError as e:
                raise RuntimeError(f"Cannot import tool registry: {e}")
        
        if self.model_registry is None:
            # Get REAL model registry from orchestrator
            try:
                from ..models.registry_singleton import get_model_registry
                registry = get_model_registry()
                object.__setattr__(self, 'model_registry', registry)
                if self.model_registry is None:
                    # Model registry is optional - we can work without it
                    logger.warning("Model registry not available, using direct model instantiation")
            except ImportError as e:
                logger.warning(f"Cannot import model registry: {e}, using direct model instantiation")
        
        # Initialize REAL models
        if self.analyzer_model is None:
            analyzer = await self._get_model("analyzer_model")
            object.__setattr__(self, 'analyzer_model', analyzer)
        if self.fixer_model is None:
            fixer = await self._get_model("fixer_model")
            object.__setattr__(self, 'fixer_model', fixer)
        if self.validator_model is None:
            validator = await self._get_model("validator_model")
            object.__setattr__(self, 'validator_model', validator)
    
    async def _get_model(self, model_type: str):
        """Get REAL model instance with fallback handling."""
        config = self.model_configurations[model_type]
        
        # Try default model first
        try:
            model = await self._create_model_instance(config["default"])
            if model:
                return model
        except Exception as e:
            logger.warning(f"Default model failed for {model_type}: {e}")
        
        # Try fallbacks
        for fallback_config in config["fallbacks"]:
            try:
                model = await self._create_model_instance(fallback_config)
                if model:
                    logger.info(f"Using fallback model for {model_type}: {fallback_config['model']}")
                    return model
            except Exception as e:
                logger.warning(f"Fallback model failed: {e}")
        
        # NO MOCKS - raise exception
        raise RuntimeError(f"No available models for {model_type}. All configured models failed.")
    
    async def _create_model_instance(self, config: Dict[str, Any]):
        """Create REAL model instance from configuration."""
        provider = config["provider"]
        model_name = config["model"]
        
        # Get API keys using existing infrastructure
        from ..utils.api_keys_flexible import load_api_keys_optional
        available_keys = load_api_keys_optional()
        
        if provider == "anthropic":
            try:
                from langchain_anthropic import ChatAnthropic
                api_key = available_keys.get("anthropic")
                if not api_key:
                    raise RuntimeError("Anthropic API key not found - check ~/.orchestrator/ configuration")
                return ChatAnthropic(
                    model=model_name,
                    api_key=api_key,
                    temperature=config["temperature"],
                    max_tokens=config["max_tokens"]
                )
            except ImportError:
                raise RuntimeError("Anthropic not available - install langchain-anthropic")
                
        elif provider == "openai":
            try:
                from langchain_openai import ChatOpenAI
                api_key = available_keys.get("openai")
                if not api_key:
                    raise RuntimeError("OpenAI API key not found - check ~/.orchestrator/ configuration")
                return ChatOpenAI(
                    model=model_name,
                    api_key=api_key,
                    temperature=config["temperature"],
                    max_tokens=config["max_tokens"]
                )
            except ImportError:
                raise RuntimeError("OpenAI not available - install langchain-openai")
                
        elif provider == "ollama":
            try:
                from langchain_community.chat_models import ChatOllama
                return ChatOllama(
                    model=model_name,
                    temperature=config["temperature"]
                )
            except ImportError:
                raise RuntimeError("Ollama not available - install langchain-community")
        
        raise RuntimeError(f"Unknown provider: {provider}")
    
    async def _get_available_tools(self) -> List[str]:
        """Get list of available debugging tools - REAL ONLY."""
        if not self.tool_registry:
            raise RuntimeError("Tool registry not initialized")
        
        # Use the standard list_tools method from ToolRegistry
        if hasattr(self.tool_registry, 'list_tools'):
            tools = self.tool_registry.list_tools()
            if not tools:
                raise RuntimeError("No tools available in tool registry")
            return tools
        else:
            raise RuntimeError("Tool registry does not support listing tools")
    
    async def _execute_debug_loop(self, context: AutoDebugContext) -> str:
        """Execute the three-step debugging loop until resolution."""
        
        while context.iteration < self.max_debug_iterations:
            logger.info(f"AutoDebugger iteration {context.iteration + 1}/{self.max_debug_iterations}")
            
            try:
                # STEP 1: ANALYZE - Use real LLM to understand the problem
                analysis = await self._analyze_problem(context)
                logger.debug(f"Analysis suggested: {analysis.suggested_action}")
                
                # STEP 2: EXECUTE - Take real action to fix the problem
                execution_result = await self._execute_fix_action(analysis, context)
                logger.debug(f"Execution success: {execution_result.success}")
                
                # STEP 3: VALIDATE - Check if the problem is resolved
                if execution_result.success:
                    validation = await self._validate_solution(execution_result, context)
                    
                    if validation.is_resolved:
                        # SUCCESS! Return complete results
                        return self._create_success_result(
                            context=context,
                            final_result=execution_result,
                            validation=validation
                        )
                    else:
                        # Fix worked but didn't fully resolve - continue debugging
                        context = self._update_context_for_next_iteration(
                            context, analysis, execution_result, validation
                        )
                else:
                    # Fix failed - analyze why and try different approach
                    context = self._update_context_after_failure(
                        context, analysis, execution_result
                    )
                    
            except Exception as e:
                logger.error(f"Debug iteration failed: {e}")
                context.error_context = f"Debug iteration error: {str(e)}"
                
            context.iteration += 1
            
        # Max iterations reached - return partial results
        return self._create_failure_result(context)
    
    async def _analyze_problem(self, context: AutoDebugContext) -> AnalysisResult:
        """Use REAL LLM to analyze the problem and suggest fixes."""
        
        if not self.analyzer_model:
            raise RuntimeError("Analyzer model not available")
        
        # Build comprehensive analysis prompt
        analysis_prompt = self._build_analysis_prompt(context)
        
        # REAL LLM call - no mocks
        if hasattr(self.analyzer_model, 'ainvoke'):
            response = await self.analyzer_model.ainvoke(analysis_prompt)
        else:
            response = await asyncio.to_thread(self.analyzer_model.invoke, analysis_prompt)
        
        # Extract JSON from response
        analysis_json = self._extract_json_from_response(response.content)
        
        return AnalysisResult(
            root_cause=analysis_json.get("root_cause", "Unknown issue"),
            suggested_action=analysis_json.get("suggested_action", "No action suggested"),
            tool_to_use=analysis_json.get("tool_to_use"),
            parameters=analysis_json.get("parameters", {}),
            expected_outcome=analysis_json.get("expected_outcome", ""),
            risk_assessment=analysis_json.get("risk_assessment", "medium"),
            confidence=float(analysis_json.get("confidence", 0.5)),
            reasoning=analysis_json.get("reasoning", "")
        )
    
    def _build_analysis_prompt(self, context: AutoDebugContext) -> str:
        """Build comprehensive analysis prompt for LLM."""
        
        content_type = self._detect_content_type(context.content)
        
        prompt = f"""
AUTODEBUGGER ANALYSIS REQUEST

TASK DESCRIPTION: {context.task_description}
CONTENT TYPE: {content_type}

CONTENT TO DEBUG:
```{content_type}
{context.content}
```

ERROR/PROBLEM CONTEXT:
{context.error_context}

EXPECTED OUTCOME:
{context.expected_outcome}

AVAILABLE TOOLS: {', '.join(context.available_tools)}

PREVIOUS ATTEMPTS:
{self._format_debug_history(context.debug_history)}

Please analyze this problem and provide:
1. Root cause of the issue
2. Specific action to take (which tool to use and how)
3. Expected outcome of the fix
4. Risk assessment
5. Confidence level (0-1)

Respond in JSON format:
{{
    "root_cause": "Clear explanation of what's wrong",
    "suggested_action": "Specific action to take",
    "tool_to_use": "exact_tool_name or null",
    "parameters": {{"key": "value parameters for tool"}},
    "expected_outcome": "What should happen after fix",
    "risk_assessment": "low|medium|high",
    "confidence": 0.8,
    "reasoning": "Step-by-step reasoning"
}}
"""
        return prompt
    
    def _detect_content_type(self, content: str) -> str:
        """Detect the type of content being debugged."""
        content_lower = content.lower().strip()
        
        # Check for common patterns
        if content_lower.startswith("def ") or "import " in content_lower:
            return "python"
        elif content_lower.startswith("function ") or "const " in content_lower:
            return "javascript"
        elif "\\documentclass" in content_lower or "\\begin{" in content_lower:
            return "latex"
        elif content_lower.startswith("select ") or "from " in content_lower:
            return "sql"
        elif "version:" in content_lower or "steps:" in content_lower:
            return "yaml"
        elif content_lower.startswith("{") or content_lower.startswith("["):
            return "json"
        elif "<!DOCTYPE" in content_lower or "<html" in content_lower:
            return "html"
        else:
            return "text"
    
    def _format_debug_history(self, debug_history: List[Dict[str, Any]]) -> str:
        """Format debugging history for context."""
        if not debug_history:
            return "No previous attempts"
        
        formatted = []
        for i, attempt in enumerate(debug_history):
            formatted.append(f"Attempt {i+1}: {attempt.get('summary', 'No summary')}")
        
        return "\n".join(formatted)
    
    def _extract_json_from_response(self, response_content: str) -> Dict[str, Any]:
        """Extract JSON from LLM response."""
        try:
            # Try to find JSON in the response
            import re
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # If no JSON found, raise error - no fallbacks
                raise ValueError("No valid JSON found in LLM response")
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from analysis response: {e}")
    
    async def _execute_fix_action(self, analysis: AnalysisResult, context: AutoDebugContext) -> ExecutionResult:
        """Execute the suggested fix using REAL tools and operations."""
        
        start_time = time.time()
        
        try:
            if analysis.tool_to_use:
                # Execute using specific tool from orchestrator toolkit
                result = await self._execute_tool_action(analysis, context)
                
            elif "write_file" in analysis.suggested_action.lower() or "create_file" in analysis.suggested_action.lower():
                # Direct file operations for fixing content
                result = await self._execute_file_fix(analysis, context)
                
            elif "run_command" in analysis.suggested_action.lower() or "execute" in analysis.suggested_action.lower():
                # Execute system commands for fixes
                result = await self._execute_command_fix(analysis, context)
                
            elif "generate" in analysis.suggested_action.lower() or "create" in analysis.suggested_action.lower():
                # Use LLM to generate fixed content
                result = await self._execute_llm_fix(analysis, context)
                
            else:
                # Generic execution attempt
                result = await self._execute_generic_fix(analysis, context)
            
            result.execution_time = time.time() - start_time
            result.analysis_used = analysis
            return result
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=e,
                error_message=str(e),
                traceback=traceback.format_exc(),
                execution_time=time.time() - start_time,
                analysis_used=analysis
            )

    async def _execute_tool_action(self, analysis: AnalysisResult, context: AutoDebugContext) -> ExecutionResult:
        """Execute using specific tool from orchestrator toolkit - REAL ONLY."""
        
        tool_name = analysis.tool_to_use
        parameters = analysis.parameters
        
        logger.info(f"Executing {tool_name} with parameters: {parameters}")
        
        # Get REAL tool from registry
        if not self.tool_registry:
            raise RuntimeError("Tool registry not available")
        
        if not hasattr(self.tool_registry, 'get_tool'):
            raise RuntimeError("Tool registry does not support getting tools")
        
        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            raise RuntimeError(f"Tool '{tool_name}' not found in registry")
        
        # Execute tool with parameters - REAL execution only
        if hasattr(tool, 'arun'):
            result = await tool.arun(**parameters)
        elif hasattr(tool, 'run'):
            result = await asyncio.to_thread(tool.run, **parameters)
        else:
            raise RuntimeError(f"Tool '{tool_name}' has no run method")
        
        return ExecutionResult(
            success=True,
            result=result,
            tool_used=tool_name,
            parameters_used=parameters,
            fixed_content=parameters.get("content") if "content" in parameters else None
        )

    async def _execute_file_fix(self, analysis: AnalysisResult, context: AutoDebugContext) -> ExecutionResult:
        """Execute file-based fixes using REAL filesystem operations."""
        
        # Extract file information from analysis
        file_path = analysis.parameters.get("file_path", "temp_debug_file.txt")
        fixed_content = analysis.parameters.get("content", context.content)
        
        # Use filesystem tool if available
        if self.tool_registry and hasattr(self.tool_registry, 'get_tool'):
            filesystem_tool = self.tool_registry.get_tool("filesystem")
            if filesystem_tool:
                result = await filesystem_tool.arun(
                    action="write",
                    path=file_path,
                    content=fixed_content
                )
                
                return ExecutionResult(
                    success=True,
                    result={"file_path": file_path, "content_written": len(fixed_content)},
                    tool_used="filesystem",
                    parameters_used={"action": "write", "path": file_path},
                    fixed_content=fixed_content
                )
        
        # Direct filesystem operations as fallback
        import os
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        return ExecutionResult(
            success=True,
            result={"file_path": file_path, "content_written": len(fixed_content)},
            tool_used="direct_filesystem",
            parameters_used={"path": file_path},
            fixed_content=fixed_content
        )

    async def _execute_command_fix(self, analysis: AnalysisResult, context: AutoDebugContext) -> ExecutionResult:
        """Execute system command fixes using REAL command execution."""
        
        command = analysis.parameters.get("command", "")
        working_directory = analysis.parameters.get("cwd", ".")
        
        if not command:
            raise RuntimeError("No command specified for execution")
        
        # Use system_tools if available
        if self.tool_registry and hasattr(self.tool_registry, 'get_tool'):
            system_tool = self.tool_registry.get_tool("system_tools")
            if system_tool:
                result = await system_tool.arun(
                    action="execute_command", 
                    command=command,
                    cwd=working_directory,
                    capture_output=True
                )
                
                return ExecutionResult(
                    success=result.get("return_code", 1) == 0,
                    result=result,
                    tool_used="system_tools",
                    parameters_used={"command": command, "cwd": working_directory}
                )
        
        # Direct subprocess execution as fallback
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=working_directory,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        result = {
            "return_code": process.returncode,
            "stdout": stdout.decode('utf-8', errors='ignore'),
            "stderr": stderr.decode('utf-8', errors='ignore')
        }
        
        return ExecutionResult(
            success=process.returncode == 0,
            result=result,
            tool_used="direct_subprocess",
            parameters_used={"command": command, "cwd": working_directory}
        )

    async def _execute_llm_fix(self, analysis: AnalysisResult, context: AutoDebugContext) -> ExecutionResult:
        """Use REAL LLM to generate fixed content."""
        
        if not self.fixer_model:
            raise RuntimeError("Fixer model not available")
        
        fix_prompt = f"""
CONTENT FIXING REQUEST

ORIGINAL CONTENT:
```
{context.content}
```

PROBLEM: {analysis.root_cause}

FIX NEEDED: {analysis.suggested_action}

EXPECTED OUTCOME: {analysis.expected_outcome}

Please provide the corrected version of the content that addresses the problem.
Return ONLY the fixed content, no explanations or additional text.
"""
        
        # REAL LLM call for content fixing
        if hasattr(self.fixer_model, 'ainvoke'):
            response = await self.fixer_model.ainvoke(fix_prompt)
        else:
            response = await asyncio.to_thread(self.fixer_model.invoke, fix_prompt)
        
        fixed_content = response.content.strip()
        
        # Remove markdown code blocks if present
        import re
        code_block_match = re.match(r'```[\w]*\n(.*)\n```', fixed_content, re.DOTALL)
        if code_block_match:
            fixed_content = code_block_match.group(1)
        
        return ExecutionResult(
            success=True,
            result={"fixed_content": fixed_content},
            model_used=getattr(self.fixer_model, 'model_name', 'fixer_model'),
            fixed_content=fixed_content
        )

    async def _execute_generic_fix(self, analysis: AnalysisResult, context: AutoDebugContext) -> ExecutionResult:
        """Generic fix execution when specific approach isn't clear."""
        
        # Try to determine the best approach based on content and error
        content_type = self._detect_content_type(context.content)
        
        if content_type in ["python", "javascript", "sql"]:
            # For code, try to create a fixed version using LLM
            return await self._execute_llm_fix(analysis, context)
        
        elif context.content.strip():
            # For other content with actual content, use LLM fix
            return await self._execute_llm_fix(analysis, context)
        
        else:
            # For tasks without content, try tool execution
            if analysis.parameters:
                return await self._execute_tool_action(analysis, context)
            else:
                raise RuntimeError("Cannot determine how to execute fix - no content or parameters provided")

    async def _validate_solution(self, execution_result: ExecutionResult, context: AutoDebugContext) -> ValidationResult:
        """Validate that the fix actually resolves the problem using REAL LLM."""
        
        if not self.validator_model:
            raise RuntimeError("Validator model not available")
        
        validation_prompt = f"""
SOLUTION VALIDATION REQUEST

ORIGINAL PROBLEM: {context.task_description}
ERROR CONTEXT: {context.error_context}
EXPECTED OUTCOME: {context.expected_outcome}

FIX THAT WAS APPLIED:
Tool Used: {execution_result.tool_used or 'LLM Generation'}
Result: {execution_result.result}

FIXED CONTENT (if applicable):
```
{execution_result.fixed_content if execution_result.fixed_content else 'N/A'}
```

Please validate if this fix resolves the original problem:
1. Is the problem fully resolved? (true/false)
2. Are there any remaining issues?
3. Does the solution meet the expected outcome?
4. Quality score (0-1)
5. Validation reasoning

Respond in JSON format:
{{
    "is_resolved": true,
    "remaining_issues": ["list", "of", "issues"],
    "meets_expectations": true,
    "quality_score": 0.9,
    "validation_reasoning": "Detailed explanation"
}}
"""
        
        # REAL LLM call for validation
        if hasattr(self.validator_model, 'ainvoke'):
            response = await self.validator_model.ainvoke(validation_prompt)
        else:
            response = await asyncio.to_thread(self.validator_model.invoke, validation_prompt)
        
        validation_json = self._extract_json_from_response(response.content)
        
        return ValidationResult(
            is_resolved=validation_json.get("is_resolved", False),
            remaining_issues=validation_json.get("remaining_issues", []),
            meets_expectations=validation_json.get("meets_expectations", False),
            quality_score=float(validation_json.get("quality_score", 0.0)),
            reasoning=validation_json.get("validation_reasoning", ""),
            validation_details=validation_json
        )
    
    def _create_success_result(self, 
                             context: AutoDebugContext,
                             final_result: ExecutionResult,
                             validation: ValidationResult) -> str:
        """Create successful debugging result."""
        
        result = {
            "success": True,
            "session_id": context.session_id,
            "task_description": context.task_description,
            "total_iterations": context.iteration + 1,
            "final_content": final_result.fixed_content or final_result.result,
            "validation": {
                "is_resolved": validation.is_resolved,
                "quality_score": validation.quality_score,
                "reasoning": validation.reasoning
            },
            "debug_summary": self._generate_debug_summary(context),
            "modifications_made": context.modifications_made,
            "tools_used": self._extract_tools_used(context),
            "execution_time": time.time() - context.debug_history[0].get("timestamp", time.time()) if context.debug_history else 0
        }
        
        return json.dumps(result, indent=2)
    
    def _create_failure_result(self, context: AutoDebugContext) -> str:
        """Create failure result when max iterations reached."""
        
        result = {
            "success": False,
            "session_id": context.session_id,
            "task_description": context.task_description,
            "total_iterations": context.iteration,
            "error_message": "Max debugging iterations reached without resolution",
            "debug_summary": self._generate_debug_summary(context),
            "modifications_made": context.modifications_made,
            "tools_used": self._extract_tools_used(context),
            "final_error": context.error_context
        }
        
        return json.dumps(result, indent=2)
    
    def _generate_debug_summary(self, context: AutoDebugContext) -> str:
        """Generate human-readable summary of debugging session."""
        
        if not context.debug_history:
            return "No debugging attempts made"
        
        summary_parts = [
            f"Debugging session for: {context.task_description}",
            f"Total iterations: {context.iteration}",
            f"Modifications made: {len(context.modifications_made)}"
        ]
        
        if context.modifications_made:
            summary_parts.append("Key changes:")
            for i, mod in enumerate(context.modifications_made[:3]):
                summary_parts.append(f"  {i+1}. {mod}")
        
        return "\n".join(summary_parts)
    
    def _extract_tools_used(self, context: AutoDebugContext) -> List[str]:
        """Extract list of tools used during debugging."""
        tools_used = set()
        
        for attempt in context.debug_history:
            if "tool_used" in attempt:
                tools_used.add(attempt["tool_used"])
        
        return list(tools_used)
    
    def _update_context_for_next_iteration(self,
                                         context: AutoDebugContext,
                                         analysis: AnalysisResult,
                                         execution_result: ExecutionResult,
                                         validation: ValidationResult) -> AutoDebugContext:
        """Update context for next debugging iteration."""
        
        # Update content if we have fixed content
        if execution_result.fixed_content:
            context.content = execution_result.fixed_content
        
        # Update error context with remaining issues
        if validation.remaining_issues:
            context.error_context = "; ".join(validation.remaining_issues)
        
        # Add to debug history
        context.debug_history.append({
            "iteration": context.iteration,
            "analysis": analysis,
            "execution_result": execution_result,
            "validation": validation,
            "timestamp": time.time(),
            "summary": f"Attempted {analysis.suggested_action} - Partial success"
        })
        
        # Add modification summary
        if execution_result.tool_used:
            context.modifications_made.append(
                f"Used {execution_result.tool_used}: {analysis.suggested_action}"
            )
        
        return context
    
    def _update_context_after_failure(self,
                                     context: AutoDebugContext,
                                     analysis: AnalysisResult,
                                     execution_result: ExecutionResult) -> AutoDebugContext:
        """Update context after failed execution."""
        
        # Update error context with execution failure
        context.error_context = f"Previous attempt failed: {execution_result.error_message}"
        
        # Add to debug history
        context.debug_history.append({
            "iteration": context.iteration,
            "analysis": analysis,
            "execution_result": execution_result,
            "validation": None,
            "timestamp": time.time(),
            "summary": f"Failed attempt: {analysis.suggested_action}"
        })
        
        return context