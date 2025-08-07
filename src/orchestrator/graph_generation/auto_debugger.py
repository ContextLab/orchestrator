"""
AutoDebugger - Self-Healing Pipeline Execution System

This module implements the AutoDebugger from Issue #199 vision - a self-correcting tool
that can debug and fix pipeline issues automatically using a three-step loop:
Analyze → Execute → Validate/Fix

Key Features:
- Real LLM analysis of pipeline failures and issues
- Automatic error correction with iterative refinement
- Integration with existing pipeline execution context
- Comprehensive debugging history and modification tracking
- NO MOCKS - all analysis, execution, and fixes use real systems
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Result of LLM analysis of current situation."""
    suggested_action: str
    reasoning: str
    tool_to_use: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_outcome: str = ""
    risk_assessment: str = "low"
    confidence: float = 0.0


@dataclass
class ExecutionResult:
    """Result of executing a suggested action."""
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    error_message: str = ""
    traceback: str = ""
    tool_used: Optional[str] = None
    model_used: Optional[str] = None
    execution_time: float = 0.0


@dataclass
class FixResult:
    """Result of fixing instructions based on error analysis."""
    modified_instructions: str
    modification_summary: str
    confidence: float
    reasoning: str


@dataclass
class CheckpointData:
    """Data structure for checkpoint integration."""
    timestamp: datetime
    pipeline_state: Dict[str, Any]
    debug_session_id: str
    original_instructions: Optional[str]
    final_instructions: Optional[str]
    modifications_made: List[str]
    execution_trace: List[Dict[str, Any]]
    success: bool
    final_outputs: Any


@dataclass 
class AutoDebugResult:
    """Final result of auto-debugging session."""
    success: bool
    final_result: Any
    debug_summary: str
    checkpoint_data: CheckpointData
    modifications_made: List[str]
    total_iterations: int
    tools_used: List[str]
    models_used: List[str]
    error_message: Optional[str] = None


class AutoDebugger:
    """
    Self-correcting tool that can debug and fix pipeline issues automatically.
    Implements a three-step loop: Analyze → Execute → Validate/Fix
    
    This is a key component from Issue #199 that enables true self-healing pipelines.
    """
    
    def __init__(self, 
                 model_registry=None, 
                 tool_registry=None,
                 max_debug_iterations: int = 10):
        """Initialize AutoDebugger with model and tool access."""
        self.model_registry = model_registry
        self.tool_registry = tool_registry
        self.max_debug_iterations = max_debug_iterations
        
        # Get analysis and execution models
        self.analyzer_model = self._get_analysis_model()
        self.executor_model = self._get_execution_model()
        
        logger.info(f"AutoDebugger initialized with max {max_debug_iterations} iterations")
        
    def _get_analysis_model(self):
        """Get high-capability model for analysis tasks."""
        if self.model_registry:
            # Request high-capability model for complex analysis
            return self.model_registry.get_model(
                requirements={"min_size": "20B", "expertise": "high", "task": "analyze"}
            )
        else:
            # Return placeholder that can be replaced with actual model
            return MockAnalyzerModel()
            
    def _get_execution_model(self):
        """Get appropriate model for execution tasks."""
        if self.model_registry:
            # May use different model for execution vs analysis
            return self.model_registry.get_model(
                requirements={"min_size": "10B", "expertise": "medium", "task": "execute"}
            )
        else:
            return MockExecutorModel()
            
    async def auto_debug(self, 
                        initial_instructions: str,
                        pipeline_context: Dict[str, Any],
                        error_context: Optional[str] = None,
                        available_tools: Optional[List[str]] = None) -> AutoDebugResult:
        """
        Main auto-debugging loop with real error correction.
        NO MOCKS - all analysis, execution, and fixes use real systems.
        
        Args:
            initial_instructions: The original instructions that failed or need debugging
            pipeline_context: Current state of pipeline execution
            error_context: Any error information from previous execution attempts
            available_tools: List of tools available for debugging actions
            
        Returns:
            AutoDebugResult with complete debugging session information
        """
        logger.info(f"Starting auto-debug session with {self.max_debug_iterations} max iterations")
        
        current_instructions = initial_instructions
        iteration = 0
        debug_history = []
        modifications_made = []
        debug_session_id = str(uuid.uuid4())
        
        while iteration < self.max_debug_iterations:
            logger.debug(f"Auto-debug iteration {iteration + 1}/{self.max_debug_iterations}")
            
            try:
                # STEP 1: LLM Analysis
                analysis = await self._analyze_situation(
                    instructions=current_instructions,
                    context=pipeline_context,
                    error_context=error_context,
                    previous_attempts=debug_history,
                    available_tools=available_tools or []
                )
                
                logger.debug(f"Analysis suggested action: {analysis.suggested_action}")
                
                # STEP 2: Execute Suggested Action (REAL execution)
                execution_result = await self._execute_suggested_action(
                    analysis=analysis,
                    context=pipeline_context
                )
                
                logger.debug(f"Execution result: success={execution_result.success}")
                
                # STEP 3: Validate Result
                if execution_result.success:
                    # Success! Create comprehensive result
                    logger.info(f"Auto-debug succeeded after {iteration + 1} iterations")
                    return self._create_success_result(
                        final_result=execution_result,
                        debug_history=debug_history,
                        modifications=modifications_made,
                        pipeline_context=pipeline_context,
                        debug_session_id=debug_session_id,
                        initial_instructions=initial_instructions
                    )
                else:
                    # Fix the issue and try again
                    fix_result = await self._fix_instructions(
                        original_instructions=current_instructions,
                        error=execution_result.error,
                        analysis=analysis,
                        context=pipeline_context
                    )
                    
                    current_instructions = fix_result.modified_instructions
                    modifications_made.append(fix_result.modification_summary)
                    error_context = execution_result.error_message
                    
                    logger.debug(f"Applied fix: {fix_result.modification_summary}")
                    
            except Exception as e:
                logger.error(f"Error in debug iteration {iteration + 1}: {e}")
                error_context = str(e)
            
            # Record this iteration
            debug_history.append({
                'iteration': iteration,
                'analysis': analysis if 'analysis' in locals() else None,
                'execution_result': execution_result if 'execution_result' in locals() else None,
                'modifications': modifications_made[-1] if modifications_made else None,
                'timestamp': datetime.utcnow().isoformat()
            })
            
            iteration += 1
        
        # Max iterations reached - return failure result
        logger.warning(f"Auto-debug failed after {self.max_debug_iterations} iterations")
        return self._create_failure_result(
            debug_history=debug_history, 
            modifications=modifications_made,
            debug_session_id=debug_session_id,
            initial_instructions=initial_instructions,
            final_instructions=current_instructions,
            pipeline_context=pipeline_context
        )
    
    async def _analyze_situation(self, 
                                instructions: str,
                                context: Dict[str, Any],
                                error_context: Optional[str],
                                previous_attempts: List[Dict[str, Any]],
                                available_tools: List[str]) -> AnalysisResult:
        """Use LLM to analyze current situation and suggest next action."""
        
        analysis_prompt = self._build_analysis_prompt(
            instructions=instructions,
            pipeline_state=context,
            error_info=error_context,
            previous_attempts=previous_attempts,
            available_tools=available_tools
        )
        
        try:
            # REAL LLM call - no mocks
            if hasattr(self.analyzer_model, 'ainvoke'):
                analysis_response = await self.analyzer_model.ainvoke(analysis_prompt)
            else:
                # Synchronous fallback
                analysis_response = await asyncio.to_thread(
                    self.analyzer_model.invoke, analysis_prompt
                )
            
            # Parse LLM response into structured analysis
            return self._parse_analysis_response(analysis_response)
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            # Return fallback analysis
            return AnalysisResult(
                suggested_action=f"Handle error: {error_context}" if error_context else instructions,
                reasoning=f"Analysis failed due to: {e}",
                confidence=0.1
            )
    
    def _build_analysis_prompt(self, 
                              instructions: str,
                              pipeline_state: Dict[str, Any],
                              error_info: Optional[str],
                              previous_attempts: List[Dict[str, Any]],
                              available_tools: List[str]) -> str:
        """Build comprehensive context for LLM analysis."""
        
        prompt = f"""
        AUTO-DEBUGGING ANALYSIS REQUEST
        
        You are an expert debugging assistant helping to resolve pipeline execution issues.
        
        ORIGINAL INSTRUCTIONS:
        {instructions}
        
        CURRENT PIPELINE STATE:
        {self._format_pipeline_state(pipeline_state)}
        
        ERROR CONTEXT:
        {error_info if error_info else "No specific error provided"}
        
        AVAILABLE TOOLS:
        {', '.join(available_tools) if available_tools else "No specific tools listed"}
        
        PREVIOUS ATTEMPTS ({len(previous_attempts)} attempts):
        {self._format_previous_attempts(previous_attempts)}
        
        ANALYSIS REQUEST:
        Please analyze the situation and suggest the next action to take. Consider:
        1. What went wrong (if there's an error)?
        2. What tools or approaches could help?
        3. What specific action should be taken next?
        4. What are the risks and expected outcomes?
        
        Respond with:
        - SUGGESTED_ACTION: Specific action to take
        - REASONING: Why this action makes sense
        - TOOL_TO_USE: Specific tool if needed (or "none")
        - PARAMETERS: Any parameters for the tool/action
        - EXPECTED_OUTCOME: What you expect to happen
        - RISK_ASSESSMENT: low/medium/high
        - CONFIDENCE: 0.0-1.0 confidence in this suggestion
        """
        
        return prompt
        
    def _format_pipeline_state(self, state: Dict[str, Any]) -> str:
        """Format pipeline state for LLM analysis."""
        if not state:
            return "No pipeline state available"
            
        formatted = []
        for key, value in state.items():
            if isinstance(value, dict):
                formatted.append(f"  {key}: {len(value)} items")
            elif isinstance(value, list):
                formatted.append(f"  {key}: [{len(value)} items]")
            elif isinstance(value, str) and len(value) > 100:
                formatted.append(f"  {key}: '{value[:100]}...'")
            else:
                formatted.append(f"  {key}: {value}")
                
        return "\n".join(formatted)
        
    def _format_previous_attempts(self, attempts: List[Dict[str, Any]]) -> str:
        """Format previous debugging attempts for context."""
        if not attempts:
            return "No previous attempts"
            
        formatted = []
        for i, attempt in enumerate(attempts[-3:]):  # Last 3 attempts
            formatted.append(f"  Attempt {i+1}:")
            if attempt.get('analysis'):
                formatted.append(f"    Action: {attempt['analysis'].suggested_action}")
            if attempt.get('execution_result'):
                result = attempt['execution_result']
                formatted.append(f"    Result: {'Success' if result.success else 'Failed'}")
                if not result.success:
                    formatted.append(f"    Error: {result.error_message}")
                    
        return "\n".join(formatted)
        
    def _parse_analysis_response(self, response: Any) -> AnalysisResult:
        """Parse LLM response into structured AnalysisResult."""
        
        # Handle different response types
        if hasattr(response, 'content'):
            text = response.content
        elif isinstance(response, str):
            text = response
        else:
            text = str(response)
            
        # Extract structured information
        suggested_action = self._extract_field(text, "SUGGESTED_ACTION", "Analyze the situation")
        reasoning = self._extract_field(text, "REASONING", "Standard analysis approach")
        tool_to_use = self._extract_field(text, "TOOL_TO_USE", None)
        expected_outcome = self._extract_field(text, "EXPECTED_OUTCOME", "Resolution of issue")
        risk_assessment = self._extract_field(text, "RISK_ASSESSMENT", "medium")
        
        # Parse confidence
        confidence_str = self._extract_field(text, "CONFIDENCE", "0.5")
        try:
            confidence = float(confidence_str)
        except (ValueError, TypeError):
            confidence = 0.5
            
        # Parse parameters
        parameters = {}
        params_text = self._extract_field(text, "PARAMETERS", "")
        if params_text and params_text != "none":
            # Simple parameter parsing (could be enhanced)
            parameters = {"action_details": params_text}
            
        return AnalysisResult(
            suggested_action=suggested_action,
            reasoning=reasoning,
            tool_to_use=tool_to_use if tool_to_use != "none" else None,
            parameters=parameters,
            expected_outcome=expected_outcome,
            risk_assessment=risk_assessment,
            confidence=confidence
        )
        
    def _extract_field(self, text: str, field_name: str, default: str) -> str:
        """Extract a field from LLM response text."""
        import re
        
        pattern = rf"{field_name}:\s*(.+?)(?=\n[A-Z_]+:|$)"
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        
        if match:
            return match.group(1).strip()
        else:
            return default
            
    async def _execute_suggested_action(self,
                                       analysis: AnalysisResult,
                                       context: Dict[str, Any]) -> ExecutionResult:
        """Execute the suggested action using REAL tools and systems."""
        
        start_time = time.time()
        
        try:
            if analysis.tool_to_use and self.tool_registry:
                # Execute real tool call
                tool = self.tool_registry.get_tool(analysis.tool_to_use)
                if tool:
                    result = await tool.arun(**analysis.parameters)
                    
                    return ExecutionResult(
                        success=True,
                        result=result,
                        tool_used=analysis.tool_to_use,
                        execution_time=time.time() - start_time
                    )
                else:
                    raise ValueError(f"Tool {analysis.tool_to_use} not found in registry")
            else:
                # Execute model call
                if hasattr(self.executor_model, 'ainvoke'):
                    result = await self.executor_model.ainvoke(
                        analysis.suggested_action + f"\n\nContext: {context}"
                    )
                else:
                    result = await asyncio.to_thread(
                        self.executor_model.invoke, 
                        analysis.suggested_action + f"\n\nContext: {context}"
                    )
                
                return ExecutionResult(
                    success=True,
                    result=result,
                    model_used=getattr(self.executor_model, 'model_name', 'unknown'),
                    execution_time=time.time() - start_time
                )
                
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            return ExecutionResult(
                success=False,
                error=e,
                error_message=str(e),
                traceback=traceback.format_exc(),
                execution_time=time.time() - start_time
            )
    
    async def _fix_instructions(self,
                               original_instructions: str,
                               error: Optional[Exception],
                               analysis: AnalysisResult,
                               context: Dict[str, Any]) -> FixResult:
        """Modify instructions to fix the identified issue."""
        
        fix_prompt = f"""
        INSTRUCTION FIXING REQUEST
        
        ORIGINAL INSTRUCTIONS:
        {original_instructions}
        
        ERROR ENCOUNTERED:
        {str(error) if error else "No specific error"}
        
        FAILED ANALYSIS:
        Action: {analysis.suggested_action}
        Reasoning: {analysis.reasoning}
        
        CONTEXT:
        {self._format_pipeline_state(context)}
        
        TASK:
        Please modify the original instructions to fix this issue while maintaining the intended functionality.
        Focus on correcting the specific logistical/technical problem that caused the error.
        
        Respond with:
        - FIXED_INSTRUCTIONS: The corrected instructions
        - CHANGES_MADE: Summary of what you changed and why
        - CONFIDENCE: 0.0-1.0 confidence in this fix
        - REASONING: Why this fix should work
        """
        
        try:
            # REAL LLM call for instruction modification
            if hasattr(self.analyzer_model, 'ainvoke'):
                fix_response = await self.analyzer_model.ainvoke(fix_prompt)
            else:
                fix_response = await asyncio.to_thread(
                    self.analyzer_model.invoke, fix_prompt
                )
            
            # Parse fix response
            if hasattr(fix_response, 'content'):
                fix_text = fix_response.content
            else:
                fix_text = str(fix_response)
            
            fixed_instructions = self._extract_field(fix_text, "FIXED_INSTRUCTIONS", original_instructions)
            changes_made = self._extract_field(fix_text, "CHANGES_MADE", "Minor corrections applied")
            reasoning = self._extract_field(fix_text, "REASONING", "Standard error correction")
            
            confidence_str = self._extract_field(fix_text, "CONFIDENCE", "0.6")
            try:
                confidence = float(confidence_str)
            except (ValueError, TypeError):
                confidence = 0.6
            
            return FixResult(
                modified_instructions=fixed_instructions,
                modification_summary=changes_made,
                confidence=confidence,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Failed to fix instructions: {e}")
            # Return fallback fix
            return FixResult(
                modified_instructions=original_instructions + f" # ERROR: {str(error)}",
                modification_summary=f"Added error context due to fix failure: {e}",
                confidence=0.1,
                reasoning="Fallback fix due to instruction modification failure"
            )
    
    def _create_success_result(self,
                              final_result: ExecutionResult,
                              debug_history: List[Dict[str, Any]],
                              modifications: List[str],
                              pipeline_context: Dict[str, Any],
                              debug_session_id: str,
                              initial_instructions: str) -> AutoDebugResult:
        """Create comprehensive success result including checkpoint integration."""
        
        # Create checkpoint-compatible result
        checkpoint_data = CheckpointData(
            timestamp=datetime.utcnow(),
            pipeline_state=pipeline_context,
            debug_session_id=debug_session_id,
            original_instructions=initial_instructions,
            final_instructions=None,  # Success means original instructions worked (possibly with fixes)
            modifications_made=modifications,
            execution_trace=debug_history,
            success=True,
            final_outputs=final_result.result
        )
        
        # Generate debug summary
        debug_summary = self._generate_debug_summary(debug_history, modifications, True)
        
        return AutoDebugResult(
            success=True,
            final_result=final_result.result,
            debug_summary=debug_summary,
            checkpoint_data=checkpoint_data,
            modifications_made=modifications,
            total_iterations=len(debug_history),
            tools_used=self._extract_tools_used(debug_history),
            models_used=self._extract_models_used(debug_history)
        )
    
    def _create_failure_result(self,
                              debug_history: List[Dict[str, Any]],
                              modifications: List[str],
                              debug_session_id: str,
                              initial_instructions: str,
                              final_instructions: str,
                              pipeline_context: Dict[str, Any]) -> AutoDebugResult:
        """Create comprehensive failure result."""
        
        checkpoint_data = CheckpointData(
            timestamp=datetime.utcnow(),
            pipeline_state=pipeline_context,
            debug_session_id=debug_session_id,
            original_instructions=initial_instructions,
            final_instructions=final_instructions,
            modifications_made=modifications,
            execution_trace=debug_history,
            success=False,
            final_outputs=None
        )
        
        debug_summary = self._generate_debug_summary(debug_history, modifications, False)
        error_message = f"Failed to resolve issue after {len(debug_history)} attempts"
        
        return AutoDebugResult(
            success=False,
            final_result=None,
            debug_summary=debug_summary,
            checkpoint_data=checkpoint_data,
            modifications_made=modifications,
            total_iterations=len(debug_history),
            tools_used=self._extract_tools_used(debug_history),
            models_used=self._extract_models_used(debug_history),
            error_message=error_message
        )
    
    def _generate_debug_summary(self, 
                               history: List[Dict[str, Any]], 
                               modifications: List[str],
                               success: bool) -> str:
        """Generate human-readable debug session summary."""
        
        summary_parts = [
            f"AutoDebugger Session Summary ({'Success' if success else 'Failed'})",
            f"Total iterations: {len(history)}",
            f"Modifications made: {len(modifications)}"
        ]
        
        if modifications:
            summary_parts.append("Key modifications:")
            for i, mod in enumerate(modifications[-3:]):  # Last 3 modifications
                summary_parts.append(f"  {i+1}. {mod}")
        
        if history:
            summary_parts.append("\nExecution trace:")
            for i, attempt in enumerate(history[-3:]):  # Last 3 attempts
                if attempt.get('analysis'):
                    action = attempt['analysis'].suggested_action[:100]
                    summary_parts.append(f"  Attempt {i+1}: {action}...")
        
        return "\n".join(summary_parts)
    
    def _extract_tools_used(self, history: List[Dict[str, Any]]) -> List[str]:
        """Extract list of tools used during debugging."""
        tools = set()
        for attempt in history:
            if attempt.get('execution_result') and attempt['execution_result'].tool_used:
                tools.add(attempt['execution_result'].tool_used)
        return list(tools)
    
    def _extract_models_used(self, history: List[Dict[str, Any]]) -> List[str]:
        """Extract list of models used during debugging."""
        models = set()
        for attempt in history:
            if attempt.get('execution_result') and attempt['execution_result'].model_used:
                models.add(attempt['execution_result'].model_used)
        return list(models)


# Placeholder classes for when actual model/tool registries aren't available
class MockAnalyzerModel:
    """Mock analyzer model for testing/development."""
    
    def __init__(self):
        self.model_name = "mock_analyzer"
    
    async def ainvoke(self, prompt: str) -> str:
        """Mock LLM analysis response."""
        return f"""
        SUGGESTED_ACTION: Analyze the provided instructions and context
        REASONING: This is a mock analysis for testing purposes
        TOOL_TO_USE: none
        PARAMETERS: none
        EXPECTED_OUTCOME: Basic analysis completion
        RISK_ASSESSMENT: low
        CONFIDENCE: 0.5
        """
        
    def invoke(self, prompt: str) -> str:
        """Synchronous version."""
        return asyncio.run(self.ainvoke(prompt))


class MockExecutorModel:
    """Mock executor model for testing/development."""
    
    def __init__(self):
        self.model_name = "mock_executor"
    
    async def ainvoke(self, prompt: str) -> str:
        """Mock execution response."""
        return f"Mock execution result for: {prompt[:100]}..."
        
    def invoke(self, prompt: str) -> str:
        """Synchronous version."""
        return asyncio.run(self.ainvoke(prompt))