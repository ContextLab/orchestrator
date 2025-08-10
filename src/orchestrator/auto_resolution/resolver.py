"""Core lazy AUTO tag resolver with multi-pass LLM orchestration."""

import asyncio
import time
from typing import Any, Dict, List, Optional

from .models import (
    AutoTagContext,
    AutoTagResolution,
    AutoTagConfig,
    RequirementsAnalysis,
    PromptConstruction,
    ActionPlan,
    ResolutionError,
    AutoTagResolutionError,
    ParseError,
    ValidationError,
)
from .requirements_analyzer import RequirementsAnalyzer
from .prompt_constructor import PromptConstructor
from .resolution_executor import ResolutionExecutor
from .action_determiner import ActionDeterminer
from .resolution_logger import ResolutionLogger
from .context_discovery import ContextDiscoveryEngine, DiscoveredContext
from .template_injector import TemplateInjector


class LazyAutoTagResolver:
    """Multi-pass AUTO tag resolver with LLM orchestration."""
    
    def __init__(
        self,
        config: Optional[AutoTagConfig] = None,
        model_registry: Optional[Any] = None,
        requirements_analyzer: Optional[RequirementsAnalyzer] = None,
        prompt_constructor: Optional[PromptConstructor] = None,
        resolution_executor: Optional[ResolutionExecutor] = None,
        action_determiner: Optional[ActionDeterminer] = None,
        logger: Optional[ResolutionLogger] = None,
        context_discovery: Optional[ContextDiscoveryEngine] = None,
        template_injector: Optional[TemplateInjector] = None
    ):
        self.config = config or AutoTagConfig()
        self.model_registry = model_registry
        
        # Initialize components
        self.requirements_analyzer = requirements_analyzer or RequirementsAnalyzer()
        self.prompt_constructor = prompt_constructor or PromptConstructor()
        self.resolution_executor = resolution_executor or ResolutionExecutor(model_registry)
        self.action_determiner = action_determiner or ActionDeterminer()
        self.logger = logger or ResolutionLogger(self.config.checkpoint_resolutions)
        
        # Initialize new intelligent resolution components
        self.context_discovery = context_discovery or ContextDiscoveryEngine()
        self.template_injector = template_injector or TemplateInjector()
    
    async def resolve(
        self,
        auto_tag: str,
        context: AutoTagContext,
        model_override: Optional[List[str]] = None
    ) -> AutoTagResolution:
        """Resolve AUTO tag through multi-pass process.
        
        Args:
            auto_tag: The AUTO tag content (without <AUTO> markers)
            context: Full context for resolution
            model_override: Optional list of models to use instead of config
            
        Returns:
            Complete resolution result
            
        Raises:
            AutoTagResolutionError: If resolution fails after all attempts
        """
        start_time = time.time()
        
        # Log resolution start
        self.logger.log_resolution_start(auto_tag, context)
        
        # Determine models to try
        models_to_try = model_override or self.config.model_escalation
        
        # Track attempts
        models_attempted = []
        all_errors = []
        
        # Try resolution with escalating models
        for model_idx, model in enumerate(models_to_try):
            models_attempted.append(model)
            self.logger.log_model_attempt(model, model_idx)
            
            # Try multiple times with same model
            for retry in range(self.config.max_retries_per_model):
                try:
                    resolution = await self._resolve_with_model(
                        auto_tag, context, model
                    )
                    
                    # Update metadata
                    resolution.total_time_ms = int((time.time() - start_time) * 1000)
                    resolution.retry_count = model_idx * self.config.max_retries_per_model + retry
                    resolution.models_attempted = models_attempted.copy()
                    resolution.final_model_used = model
                    
                    # Log success
                    self.logger.log_resolution_complete(resolution)
                    
                    return resolution
                    
                except (ParseError, ValidationError, ResolutionError) as e:
                    error_info = {
                        "model": model,
                        "retry": retry,
                        "error_type": type(e).__name__,
                        "error_message": str(e)
                    }
                    all_errors.append(error_info)
                    
                    # Log retry
                    if retry < self.config.max_retries_per_model - 1:
                        self.logger.log_pass_result(
                            f"retry_{retry}",
                            None,
                            0,
                            success=False,
                            error=str(e)
                        )
                    
                    # Don't retry if this is a fundamental error
                    if isinstance(e, ResolutionError) and "timeout" not in str(e).lower():
                        break
                        
                except Exception as e:
                    # Unexpected error - log and try next model
                    error_info = {
                        "model": model,
                        "retry": retry,
                        "error_type": "unexpected",
                        "error_message": str(e)
                    }
                    all_errors.append(error_info)
                    break
        
        # All attempts failed
        self.logger.log_resolution_error(
            auto_tag,
            context.tag_location,
            AutoTagResolutionError(f"Failed after {len(models_attempted)} models"),
            models_attempted
        )
        
        raise AutoTagResolutionError(
            f"Failed to resolve AUTO tag after trying models: {models_attempted}. "
            f"Errors: {all_errors}"
        )
    
    async def _discover_and_inject_context(
        self,
        auto_tag: str,
        context: AutoTagContext
    ) -> DiscoveredContext:
        """
        Discover relevant context for AUTO tag using natural language understanding.
        
        This is the key innovation: AUTO tags no longer need explicit variable references.
        We automatically discover what data the user is referring to based on their intent.
        
        Args:
            auto_tag: Natural language AUTO tag content
            context: Full pipeline context
            
        Returns:
            DiscoveredContext with relevant data and metadata
        """
        # Use context discovery to find relevant data based on intent
        discovered = self.context_discovery.discover_relevant_data(
            intent=auto_tag,
            step_results=context.step_results,
            variables=context.variables
        )
        
        # Log discovery results
        if discovered.relevant_data:
            self.logger.log_pass_result(
                "context_discovery",
                {
                    "discovered_paths": discovered.discovered_paths,
                    "confidence_scores": discovered.confidence_scores,
                    "keywords_matched": discovered.keywords_matched
                },
                0  # No duration tracking for this
            )
        
        return discovered
    
    async def _resolve_with_model(
        self,
        auto_tag: str,
        context: AutoTagContext,
        model: str
    ) -> AutoTagResolution:
        """Execute the enhanced multi-pass resolution process with intelligent context discovery."""
        
        # Initialize resolution object with placeholder prompt construction
        resolution = AutoTagResolution(
            original_tag=auto_tag,
            tag_location=context.tag_location,
            requirements=RequirementsAnalysis(),
            prompt_construction=PromptConstruction(prompt="<placeholder>"),
            resolved_value=None,
            action_plan=ActionPlan(action_type="return_value")
        )
        
        # NEW: Context Discovery Phase (before requirements analysis)
        # This is the key innovation - discover relevant data based on natural language intent
        discovered_context = await self._discover_and_inject_context(auto_tag, context)
        
        # Pass 1: Requirements Analysis
        requirements_start = time.time()
        self.logger.log_pass_start("requirements_analysis", model, {
            "tag": auto_tag,
            "location": context.tag_location
        })
        
        try:
            requirements = await asyncio.wait_for(
                self.requirements_analyzer.analyze(auto_tag, context, model),
                timeout=self.config.pass_timeouts.requirements_analysis
            )
            resolution.requirements = requirements
            requirements_duration = int((time.time() - requirements_start) * 1000)
            
            self.logger.log_pass_result(
                "requirements_analysis",
                requirements,
                requirements_duration
            )
            self.logger.log_requirements_analysis(
                requirements.__dict__,
                requirements_duration
            )
            
        except asyncio.TimeoutError:
            raise ResolutionError(f"Requirements analysis timed out after {self.config.pass_timeouts.requirements_analysis}s")
        
        # Pass 2: Prompt Construction (Enhanced with discovered context)
        prompt_start = time.time()
        self.logger.log_pass_start("prompt_construction", model, {
            "tag": auto_tag,
            "requirements": requirements,
            "discovered_context": len(discovered_context.relevant_data) if discovered_context else 0
        })
        
        try:
            # Use template injector if we have discovered context
            if discovered_context and discovered_context.relevant_data:
                # Create enriched prompt using discovered context
                injection_result = self.template_injector.inject_context(
                    auto_tag,
                    discovered_context,
                    context.variables
                )
                
                # Create a modified context with the enriched prompt
                enriched_auto_tag = injection_result.enriched_prompt
                
                # Pass the enriched tag to prompt constructor
                prompt_data = await asyncio.wait_for(
                    self.prompt_constructor.construct(enriched_auto_tag, context, requirements, model),
                    timeout=self.config.pass_timeouts.prompt_construction
                )
                
                # Store injection metadata in prompt construction
                prompt_data.resolved_context = injection_result.injected_variables
            else:
                # Fallback to original behavior if no context discovered
                prompt_data = await asyncio.wait_for(
                    self.prompt_constructor.construct(auto_tag, context, requirements, model),
                    timeout=self.config.pass_timeouts.prompt_construction
                )
            resolution.prompt_construction = prompt_data
            prompt_duration = int((time.time() - prompt_start) * 1000)
            
            self.logger.log_pass_result(
                "prompt_construction",
                prompt_data,
                prompt_duration
            )
            self.logger.log_prompt_construction(
                prompt_data.__dict__,
                prompt_duration
            )
            
        except asyncio.TimeoutError:
            raise ResolutionError(f"Prompt construction timed out after {self.config.pass_timeouts.prompt_construction}s")
        
        # Pass 3: Resolution Execution
        execution_start = time.time()
        self.logger.log_pass_start("resolution_execution", 
            prompt_data.target_model or model, 
            {"prompt_length": len(prompt_data.prompt)}
        )
        
        try:
            resolved_value = await asyncio.wait_for(
                self.resolution_executor.execute(prompt_data, context, requirements),
                timeout=self.config.pass_timeouts.resolution_execution
            )
            resolution.resolved_value = resolved_value
            resolution.resolution_time_ms = int((time.time() - execution_start) * 1000)
            
            self.logger.log_pass_result(
                "resolution_execution",
                resolved_value,
                resolution.resolution_time_ms
            )
            self.logger.log_resolution_execution(
                resolved_value,
                resolution.resolution_time_ms
            )
            
        except asyncio.TimeoutError:
            raise ResolutionError(f"Resolution execution timed out after {self.config.pass_timeouts.resolution_execution}s")
        
        # Post-resolution: Determine Action
        action_start = time.time()
        self.logger.log_pass_start("action_determination", model, {
            "resolved_type": type(resolved_value).__name__
        })
        
        try:
            action_plan = await asyncio.wait_for(
                self.action_determiner.determine(resolved_value, requirements, context, model),
                timeout=self.config.pass_timeouts.action_determination
            )
            resolution.action_plan = action_plan
            action_duration = int((time.time() - action_start) * 1000)
            
            self.logger.log_pass_result(
                "action_determination",
                action_plan,
                action_duration
            )
            self.logger.log_action_determination(
                action_plan.__dict__,
                action_duration
            )
            
        except asyncio.TimeoutError:
            raise ResolutionError(f"Action determination timed out after {self.config.pass_timeouts.action_determination}s")
        
        return resolution
    
    def extract_auto_tag_content(self, text: str) -> Optional[str]:
        """Extract content from AUTO tag markers.
        
        Args:
            text: Text that may contain <AUTO>...</AUTO>
            
        Returns:
            Content between AUTO tags, or None if not found
        """
        import re
        
        pattern = r'<AUTO>(.*?)</AUTO>'
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        return None