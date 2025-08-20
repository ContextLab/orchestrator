"""LLM interaction tools for managing model routing, delegation, and optimization."""

import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .base import Tool
from ..models import get_model_registry
from ..core.model import Model


@dataclass
class TaskRequirements:
    """Requirements for a specific task."""

    task_type: str  # e.g., 'code_generation', 'analysis', 'creative_writing'
    complexity: str  # 'simple', 'moderate', 'complex'
    required_capabilities: List[str] = field(default_factory=list)
    max_tokens: Optional[int] = None
    response_format: Optional[str] = None  # 'text', 'json', 'structured'
    language: Optional[str] = None
    domain: Optional[str] = None  # e.g., 'medical', 'legal', 'technical'


@dataclass
class ModelScore:
    """Score for a model's suitability for a task."""

    model_name: str
    score: float
    reasons: List[str]
    estimated_cost: float
    estimated_latency: float


class TaskDelegationTool(Tool):
    """Analyzes tasks and delegates them to the most appropriate model."""

    def __init__(self):
        super().__init__(
            name="task-delegation",
            description="Analyze task requirements and select the best model for execution",
        )
        self.add_parameter("task", "string", "The task to be executed")
        self.add_parameter(
            "requirements",
            "object",
            "Task requirements and constraints",
            required=False,
        )
        self.add_parameter(
            "fallback_enabled",
            "boolean",
            "Enable fallback to alternative models",
            default=True,
        )
        self.add_parameter(
            "cost_weight", "number", "Weight for cost optimization (0-1)", default=0.3
        )
        self.add_parameter(
            "quality_weight",
            "number",
            "Weight for quality optimization (0-1)",
            default=0.7,
        )

        self.logger = logging.getLogger(__name__)
        self.model_registry = get_model_registry()

        # Task type mappings
        self.task_type_keywords = {
            "code_generation": [
                "write code",
                "implement",
                "create function",
                "generate code",
            ],
            "code_analysis": ["analyze code", "review code", "debug", "find bugs"],
            "creative_writing": ["write story", "creative", "poem", "narrative"],
            "analysis": ["analyze", "examine", "investigate", "evaluate"],
            "summarization": ["summarize", "tldr", "brief", "overview"],
            "question_answering": ["what", "why", "how", "when", "where", "who"],
            "translation": ["translate", "convert to", "in spanish", "in french"],
            "data_processing": ["process data", "transform", "extract", "parse"],
        }

        # Model expertise mapping
        self.model_expertise = {
            "gpt-4": ["code_generation", "analysis", "creative_writing", "general"],
            "gpt-4o": ["code_generation", "analysis", "multimodal", "general"],
            "claude-sonnet-4-20250514": [
                "code_analysis",
                "creative_writing",
                "analysis",
                "general",
            ],
            "claude-opus-4-20250514": ["complex_reasoning", "creative_writing", "analysis"],
            "gemini-2.5-pro": ["multimodal", "analysis", "data_processing", "general"],
            "deepseek-r1": ["code_generation", "reasoning", "technical"],
            "qwen2.5-coder": ["code_generation", "code_analysis", "technical"],
        }

    def _analyze_task(
        self, task: str, requirements: Optional[Dict[str, Any]] = None
    ) -> TaskRequirements:
        """Analyze a task to determine its requirements."""
        task_lower = task.lower()

        # Determine task type
        task_type = "general"
        for t_type, keywords in self.task_type_keywords.items():
            if any(keyword in task_lower for keyword in keywords):
                task_type = t_type
                break

        # Determine complexity based on task length and keywords
        complexity = "simple"
        if len(task) > 500 or any(
            word in task_lower for word in ["complex", "advanced", "detailed"]
        ):
            complexity = "complex"
        elif len(task) > 200 or any(
            word in task_lower for word in ["analyze", "implement", "design"]
        ):
            complexity = "moderate"

        # Extract requirements
        req = TaskRequirements(task_type=task_type, complexity=complexity)

        # Check for specific requirements
        if "json" in task_lower or "structured" in task_lower:
            req.response_format = "json"
            req.required_capabilities.append("json_mode")

        if any(word in task_lower for word in ["image", "picture", "visual", "see"]):
            req.required_capabilities.append("vision")

        # Override with explicit requirements
        if requirements:
            req.task_type = requirements.get("task_type", req.task_type)
            req.complexity = requirements.get("complexity", req.complexity)
            req.required_capabilities.extend(requirements.get("capabilities", []))
            req.max_tokens = requirements.get("max_tokens")
            req.domain = requirements.get("domain")

        return req

    def _score_model(
        self,
        model: Model,
        requirements: TaskRequirements,
        cost_weight: float,
        quality_weight: float,
    ) -> ModelScore:
        """Score a model based on task requirements."""
        score = 0.0
        reasons = []

        # Get model key for expertise lookup
        model_key = model.name.lower().replace(":", "-")

        # Check task type match
        model_expertises = self.model_expertise.get(model_key, ["general"])
        if requirements.task_type in model_expertises:
            score += 30
            reasons.append(f"Good at {requirements.task_type}")
        elif "general" in model_expertises:
            score += 15
            reasons.append("General purpose model")

        # Check required capabilities
        if requirements.required_capabilities:
            for cap in requirements.required_capabilities:
                if model.capabilities and hasattr(model.capabilities, cap):
                    if getattr(model.capabilities, cap):
                        score += 20
                        reasons.append(f"Supports {cap}")
                    else:
                        score -= 50
                        reasons.append(f"Missing required {cap}")

        # Size/complexity matching
        model_size = getattr(model, "_size_billions", 10)
        if requirements.complexity == "complex" and model_size > 100:
            score += 20
            reasons.append("Large model for complex task")
        elif requirements.complexity == "simple" and model_size < 50:
            score += 15
            reasons.append("Efficient model for simple task")
        elif requirements.complexity == "moderate" and 20 <= model_size <= 100:
            score += 15
            reasons.append("Balanced model for moderate task")

        # Performance metrics
        if hasattr(model, "metrics"):
            success_rate = model.metrics.success_rate
            if success_rate > 0.9:
                score += 10
                reasons.append(f"High success rate: {success_rate:.0%}")
            elif success_rate < 0.7:
                score -= 10
                reasons.append(f"Low success rate: {success_rate:.0%}")

        # Cost estimation (simplified)
        estimated_cost = (
            getattr(model, "_cost_per_1k_tokens", 0.01) * 2
        )  # Assume 2k tokens
        cost_score = 100 - (estimated_cost * 10)  # Lower cost = higher score

        # Latency estimation
        estimated_latency = (
            getattr(model.metrics, "latency_p50", 1.0)
            if hasattr(model, "metrics")
            else 1.0
        )
        latency_score = 100 - (estimated_latency * 10)  # Lower latency = higher score

        # Weighted final score
        quality_score = score
        final_score = (
            quality_weight * quality_score
            + cost_weight * cost_score * 0.5  # Cost less important than quality
            + (1 - quality_weight - cost_weight) * latency_score * 0.5
        )

        return ModelScore(
            model_name=f"{model.provider}:{model.name}",
            score=final_score,
            reasons=reasons,
            estimated_cost=estimated_cost,
            estimated_latency=estimated_latency,
        )

    async def _execute_impl(self, **kwargs) -> Dict[str, Any]:
        """Execute task delegation."""
        task = kwargs["task"]
        requirements = kwargs.get("requirements", {})
        fallback_enabled = kwargs.get("fallback_enabled", True)
        cost_weight = kwargs.get("cost_weight", 0.3)
        quality_weight = kwargs.get("quality_weight", 0.7)

        # Analyze task
        task_req = self._analyze_task(task, requirements)

        # Get available models
        available_models = await self.model_registry.get_available_models()
        if not available_models:
            return {
                "success": False,
                "error": "No models available",
                "task_analysis": task_req.__dict__,
            }

        # Score all models
        scores = []
        for model_key in available_models:
            try:
                # Parse model key to get provider and name
                if ":" in model_key:
                    provider, name = model_key.split(":", 1)
                    model = self.model_registry.get_model(name, provider)
                else:
                    model = self.model_registry.get_model(model_key)
                score = self._score_model(model, task_req, cost_weight, quality_weight)
                scores.append(score)
            except Exception as e:
                self.logger.warning(f"Error scoring model {model_key}: {e}")

        # Sort by score
        scores.sort(key=lambda x: x.score, reverse=True)

        # Select best model
        if not scores:
            return {
                "success": False,
                "error": "No suitable models found",
                "task_analysis": task_req.__dict__,
            }

        best_model = scores[0]

        # Prepare fallback options
        fallback_models = []
        if fallback_enabled and len(scores) > 1:
            fallback_models = [s.model_name for s in scores[1:4]]  # Top 3 alternatives

        return {
            "success": True,
            "selected_model": best_model.model_name,
            "score": best_model.score,
            "reasons": best_model.reasons,
            "estimated_cost": best_model.estimated_cost,
            "estimated_latency": best_model.estimated_latency,
            "task_analysis": task_req.__dict__,
            "fallback_models": fallback_models,
            "all_scores": [
                {"model": s.model_name, "score": s.score, "reasons": s.reasons}
                for s in scores[:5]  # Top 5
            ],
        }


class MultiModelRoutingTool(Tool):
    """Routes requests across multiple models for load balancing and optimization."""

    def __init__(self):
        super().__init__(
            name="multi-model-routing",
            description="Route requests across multiple models with load balancing and cost optimization",
        )
        self.add_parameter("request", "string", "The request to route")
        self.add_parameter(
            "models", "array", "List of models to route between", required=False
        )
        self.add_parameter(
            "strategy",
            "string",
            "Routing strategy: round_robin, least_loaded, cost_optimized, capability_based",
            default="capability_based",
        )
        self.add_parameter(
            "max_concurrent",
            "integer",
            "Maximum concurrent requests per model",
            default=5,
        )
        self.add_parameter(
            "timeout", "number", "Request timeout in seconds", default=30.0
        )

        self.logger = logging.getLogger(__name__)
        self.model_registry = get_model_registry()

        # Track model loads
        self.model_loads: Dict[str, int] = {}
        self.model_request_times: Dict[str, List[float]] = {}
        self.round_robin_index = 0

    def _get_model_load(self, model_key: str) -> int:
        """Get current load for a model."""
        return self.model_loads.get(model_key, 0)

    def _update_model_load(self, model_key: str, delta: int):
        """Update model load."""
        current = self.model_loads.get(model_key, 0)
        self.model_loads[model_key] = max(0, current + delta)

    def _get_average_latency(self, model_key: str) -> float:
        """Get average latency for a model."""
        times = self.model_request_times.get(model_key, [])
        if not times:
            return 1.0  # Default 1 second
        return sum(times) / len(times)

    def _record_request_time(self, model_key: str, duration: float):
        """Record request duration for a model."""
        if model_key not in self.model_request_times:
            self.model_request_times[model_key] = []

        times = self.model_request_times[model_key]
        times.append(duration)

        # Keep only last 100 times
        if len(times) > 100:
            self.model_request_times[model_key] = times[-100:]

    async def _route_round_robin(
        self, models: List[str], request: str
    ) -> Tuple[str, str]:
        """Round-robin routing strategy."""
        if not models:
            raise ValueError("No models available for routing")

        selected = models[self.round_robin_index % len(models)]
        self.round_robin_index += 1

        return selected, "Round-robin selection"

    async def _route_least_loaded(
        self, models: List[str], request: str
    ) -> Tuple[str, str]:
        """Route to least loaded model."""
        if not models:
            raise ValueError("No models available for routing")

        # Find model with lowest load
        loads = [(model, self._get_model_load(model)) for model in models]
        loads.sort(key=lambda x: x[1])

        selected = loads[0][0]
        return selected, f"Least loaded (current load: {loads[0][1]})"

    async def _route_cost_optimized(
        self, models: List[str], request: str
    ) -> Tuple[str, str]:
        """Route based on cost optimization."""
        if not models:
            raise ValueError("No models available for routing")

        # Get cost estimates
        costs = []
        for model_key in models:
            try:
                # Parse model key to get provider and name
                if ":" in model_key:
                    provider, name = model_key.split(":", 1)
                    model = self.model_registry.get_model(name, provider)
                else:
                    model = self.model_registry.get_model(model_key)
                cost = getattr(model, "_cost_per_1k_tokens", 0.01)
                costs.append((model_key, cost))
            except Exception:
                costs.append((model_key, 0.01))  # Default cost

        # Sort by cost
        costs.sort(key=lambda x: x[1])

        selected = costs[0][0]
        return selected, f"Lowest cost (${costs[0][1]:.3f}/1k tokens)"

    async def _route_capability_based(
        self, models: List[str], request: str
    ) -> Tuple[str, str]:
        """Route based on model capabilities and request analysis."""
        if not models:
            raise ValueError("No models available for routing")

        # Use TaskDelegationTool for analysis
        delegation_tool = TaskDelegationTool()
        delegation_tool.model_registry = self.model_registry
        result = await delegation_tool._execute_impl(
            task=request,
            fallback_enabled=True,
            cost_weight=0.3,
            quality_weight=0.7
        )

        if result["success"] and result["selected_model"] in models:
            return (
                result["selected_model"],
                f"Best for task: {', '.join(result['reasons'][:2])}",
            )

        # Fallback to least loaded
        return await self._route_least_loaded(models, request)

    async def _execute_impl(self, **kwargs) -> Dict[str, Any]:
        """Execute request routing."""
        request = kwargs["request"]
        models = kwargs.get("models")
        strategy = kwargs.get("strategy", "capability_based")
        max_concurrent = kwargs.get("max_concurrent", 5)
        kwargs.get("timeout", 30.0)

        # Get available models if not specified
        if not models:
            models = await self.model_registry.get_available_models()

        if not models:
            return {"success": False, "error": "No models available for routing"}

        # Filter models by load
        available_models = [
            m for m in models if self._get_model_load(m) < max_concurrent
        ]

        if not available_models:
            return {
                "success": False,
                "error": "All models at capacity",
                "model_loads": {m: self._get_model_load(m) for m in models},
            }

        # Route based on strategy
        try:
            if strategy == "round_robin":
                selected_model, reason = await self._route_round_robin(
                    available_models, request
                )
            elif strategy == "least_loaded":
                selected_model, reason = await self._route_least_loaded(
                    available_models, request
                )
            elif strategy == "cost_optimized":
                selected_model, reason = await self._route_cost_optimized(
                    available_models, request
                )
            else:  # capability_based
                selected_model, reason = await self._route_capability_based(
                    available_models, request
                )

            # Update load
            self._update_model_load(selected_model, 1)

            return {
                "success": True,
                "selected_model": selected_model,
                "routing_reason": reason,
                "strategy": strategy,
                "current_load": self._get_model_load(selected_model),
                "average_latency": self._get_average_latency(selected_model),
                "all_loads": {m: self._get_model_load(m) for m in models},
            }

        except Exception as e:
            return {"success": False, "error": str(e), "strategy": strategy}


class PromptOptimizationTool(Tool):
    """Optimizes prompts for specific models and use cases."""

    def __init__(self):
        super().__init__(
            name="prompt-optimization",
            description="Optimize prompts for specific models, including formatting and token management",
        )
        self.add_parameter("prompt", "string", "The prompt to optimize")
        self.add_parameter(
            "model", "string", "Target model for optimization", required=False
        )
        self.add_parameter(
            "optimization_goals",
            "array",
            "Goals: clarity, brevity, specificity, model_specific",
            default=["clarity", "model_specific"],
        )
        self.add_parameter(
            "max_tokens", "integer", "Maximum token limit", required=False
        )
        self.add_parameter(
            "preserve_intent", "boolean", "Preserve original intent", default=True
        )

        self.logger = logging.getLogger(__name__)
        self.model_registry = get_model_registry()

        # Model-specific optimization templates
        self.model_templates = {
            "gpt-4": {"prefix": "", "suffix": "", "style": "clear and direct"},
            "claude": {
                "prefix": "",
                "suffix": "\n\nPlease provide a detailed response.",
                "style": "conversational with context",
            },
            "gemini": {
                "prefix": "Task: ",
                "suffix": "\n\nProvide a comprehensive answer.",
                "style": "structured and comprehensive",
            },
        }

        # Optimization strategies
        self.strategies = {
            "clarity": self._optimize_clarity,
            "brevity": self._optimize_brevity,
            "specificity": self._optimize_specificity,
            "model_specific": self._optimize_model_specific,
        }

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        # Rough estimate: 1 token ≈ 4 characters
        return len(text) // 4

    def _optimize_clarity(self, prompt: str, context: Dict[str, Any]) -> str:
        """Optimize prompt for clarity."""
        # Remove ambiguous pronouns
        clarified = prompt

        # Add structure if missing
        if not any(
            marker in prompt.lower() for marker in ["step", "first", "then", "finally"]
        ):
            if "?" in prompt:
                clarified = f"Question: {prompt}"
            elif any(
                word in prompt.lower() for word in ["create", "write", "generate"]
            ):
                clarified = f"Task: {prompt}"

        # Add explicit output format if missing
        if "format" not in prompt.lower() and "output" not in prompt.lower():
            if "list" in prompt.lower():
                clarified += "\n\nProvide the output as a bulleted list."
            elif "code" in prompt.lower():
                clarified += (
                    "\n\nProvide the code with appropriate syntax highlighting."
                )

        return clarified

    def _optimize_brevity(self, prompt: str, context: Dict[str, Any]) -> str:
        """Optimize prompt for brevity."""
        # Remove redundant words
        brevity_map = {
            "please could you": "please",
            "i would like you to": "please",
            "can you please": "please",
            "in order to": "to",
            "as well as": "and",
            "at this point in time": "now",
            "due to the fact that": "because",
        }

        optimized = prompt.lower()
        for long_form, short_form in brevity_map.items():
            optimized = optimized.replace(long_form, short_form)

        # Preserve original casing for words that weren't replaced
        if optimized != prompt.lower():
            # Simple approach: capitalize first letter and after periods
            optimized = ". ".join(s.strip().capitalize() for s in optimized.split(". "))
        else:
            optimized = prompt

        return optimized

    def _optimize_specificity(self, prompt: str, context: Dict[str, Any]) -> str:
        """Optimize prompt for specificity."""
        specific = prompt

        # Add specifics for common vague requests
        vague_patterns = {
            "analyze this": "analyze the following data/code/text",
            "explain it": "explain the concept/process/code in detail",
            "make it better": "improve the code/text by addressing specific issues",
            "fix this": "identify and fix the errors in the following",
        }

        lower_prompt = prompt.lower()
        for vague, specific_version in vague_patterns.items():
            if vague in lower_prompt:
                specific = prompt.replace(vague, specific_version)
                break

        # Add context if missing
        if not any(
            word in specific.lower()
            for word in ["following", "below", "above", "attached"]
        ):
            if context.get("has_context"):
                specific += "\n\nUse the provided context to inform your response."

        return specific

    def _optimize_model_specific(self, prompt: str, context: Dict[str, Any]) -> str:
        """Optimize prompt for specific model."""
        model = context.get("model", "gpt-4")

        # Extract base model name
        model_base = model.split("-")[0].split("/")[-1].lower()

        # Get template
        template = self.model_templates.get(model_base, self.model_templates["gpt-4"])

        # Apply template
        optimized = template["prefix"] + prompt + template["suffix"]

        # Model-specific optimizations
        if "claude" in model_base:
            # Claude prefers conversational style
            if prompt.startswith(("Create", "Write", "Generate")):
                optimized = f"I need help with the following task: {prompt}"

        elif "gemini" in model_base:
            # Gemini works well with structured prompts
            if "steps" not in prompt.lower():
                optimized = (
                    f"Objective: {prompt}\n\nPlease approach this systematically."
                )

        return optimized

    async def _execute_impl(self, **kwargs) -> Dict[str, Any]:
        """Execute prompt optimization."""
        prompt = kwargs["prompt"]
        model = kwargs.get("model")
        optimization_goals = kwargs.get(
            "optimization_goals", ["clarity", "model_specific"]
        )
        max_tokens = kwargs.get("max_tokens")
        preserve_intent = kwargs.get("preserve_intent", True)

        # Detect model if not specified
        if not model and self.model_registry:
            # Use task delegation to find best model
            delegation_tool = TaskDelegationTool()
            delegation_tool.model_registry = self.model_registry
            delegation_result = await delegation_tool._execute_impl(
                task=prompt,
                fallback_enabled=True,
                cost_weight=0.3,
                quality_weight=0.7
            )
            if delegation_result["success"]:
                model = delegation_result["selected_model"]

        # Build context
        context = {
            "model": model,
            "has_context": "following" in prompt.lower() or "below" in prompt.lower(),
            "original_prompt": prompt,
        }

        # Apply optimizations
        optimized = prompt
        applied_optimizations = []

        for goal in optimization_goals:
            if goal in self.strategies:
                old_optimized = optimized
                optimized = self.strategies[goal](optimized, context)
                if optimized != old_optimized:
                    applied_optimizations.append(goal)

        # Check token limit
        estimated_tokens = self._estimate_tokens(optimized)
        if max_tokens and estimated_tokens > max_tokens:
            # Truncate while preserving meaning
            if preserve_intent:
                # Keep beginning and end
                allowed_chars = max_tokens * 4  # Rough estimate
                if len(optimized) > allowed_chars:
                    mid = allowed_chars // 2
                    optimized = (
                        optimized[:mid] + "\n[...truncated...]\n" + optimized[-mid:]
                    )
            else:
                optimized = optimized[: max_tokens * 4]
            applied_optimizations.append("truncation")

        # Calculate improvement metrics
        original_tokens = self._estimate_tokens(prompt)
        optimized_tokens = self._estimate_tokens(optimized)

        return {
            "success": True,
            "original_prompt": prompt,
            "optimized_prompt": optimized,
            "model": model,
            "applied_optimizations": applied_optimizations,
            "metrics": {
                "original_tokens": original_tokens,
                "optimized_tokens": optimized_tokens,
                "token_reduction": original_tokens - optimized_tokens,
                "reduction_percentage": (
                    ((original_tokens - optimized_tokens) / original_tokens * 100)
                    if original_tokens > 0
                    else 0
                ),
            },
            "recommendations": self._get_recommendations(prompt, optimized, context),
        }

    def _get_recommendations(
        self, original: str, optimized: str, context: Dict[str, Any]
    ) -> List[str]:
        """Get additional recommendations for prompt improvement."""
        recommendations = []

        # Check for missing elements
        if "example" not in original.lower() and "for example" not in original.lower():
            if any(
                word in original.lower() for word in ["create", "write", "generate"]
            ):
                recommendations.append("Consider adding examples of desired output")

        if "format" not in original.lower() and "json" not in original.lower():
            recommendations.append("Specify desired output format explicitly")

        if len(original) > 1000:
            recommendations.append("Consider breaking complex prompts into steps")

        if context.get("model") and "claude" in context["model"]:
            recommendations.append(
                "Claude works well with conversational, context-rich prompts"
            )

        return recommendations
