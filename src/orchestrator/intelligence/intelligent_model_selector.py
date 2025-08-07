"""Intelligent Model Selector - Phase 3 Advanced Features

Implements multi-dimensional optimization for intelligent model selection,
building on the Phase 2 service integration and Phase 1 LangChain foundation.
"""

import time
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

from ..models.model_registry import ModelRegistry
from ..core.model import Model
from ..core.exceptions import ModelNotFoundError, NoEligibleModelsError

logger = logging.getLogger(__name__)


class OptimizationObjective(Enum):
    """Optimization objectives for model selection."""
    PERFORMANCE = "performance"
    COST = "cost" 
    LATENCY = "latency"
    ACCURACY = "accuracy"
    BALANCED = "balanced"


@dataclass
class ModelRequirements:
    """Enhanced model requirements for intelligent selection."""
    # Basic requirements (from existing system)
    capabilities: List[str] = None
    min_parameters: Optional[str] = None
    max_parameters: Optional[str] = None
    expertise_level: Optional[str] = None
    
    # Advanced optimization requirements
    max_cost_per_token: Optional[float] = None
    max_latency_ms: Optional[int] = None
    min_accuracy_score: Optional[float] = None
    preferred_providers: List[str] = None
    optimization_objective: OptimizationObjective = OptimizationObjective.BALANCED
    
    # Context and workload information
    expected_tokens: Optional[int] = None
    batch_size: Optional[int] = None
    concurrent_requests: Optional[int] = None
    workload_priority: str = "normal"  # high, normal, low
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []
        if self.preferred_providers is None:
            self.preferred_providers = []


@dataclass 
class ModelScore:
    """Multi-dimensional model scoring."""
    model_key: str
    performance_score: float  # 0-1, based on UCB data
    cost_score: float        # 0-1, lower cost = higher score  
    latency_score: float     # 0-1, lower latency = higher score
    accuracy_score: float    # 0-1, based on success rate
    availability_score: float  # 0-1, service health and model availability
    
    # Composite scores
    weighted_score: float
    confidence: float        # 0-1, confidence in the scoring
    
    # Metadata
    provider: str
    model_name: str
    estimated_cost: Optional[float] = None
    estimated_latency_ms: Optional[int] = None


class IntelligentModelSelector:
    """Advanced model selection with multi-dimensional optimization.
    
    Builds on the existing UCB algorithm and model registry from Phase 1 & 2,
    adding intelligent optimization for performance, cost, latency, and accuracy.
    """
    
    def __init__(self, model_registry: ModelRegistry):
        """Initialize intelligent model selector.
        
        Args:
            model_registry: Enhanced model registry from Phase 2
        """
        self.model_registry = model_registry
        self.performance_cache: Dict[str, Dict[str, float]] = {}
        self._cache_ttl = 300  # 5 minutes
        self._last_cache_update = {}
        
        # Cost data (tokens per dollar) - real provider pricing
        self.cost_data = {
            "openai:gpt-4": {"input_cost": 0.03, "output_cost": 0.06},  # per 1K tokens
            "openai:gpt-3.5-turbo": {"input_cost": 0.001, "output_cost": 0.002},
            "anthropic:claude-sonnet-4": {"input_cost": 0.003, "output_cost": 0.015},
            "anthropic:claude-haiku": {"input_cost": 0.00025, "output_cost": 0.00125},
            "ollama:llama3.2:3b": {"input_cost": 0.0, "output_cost": 0.0},  # Free local
            "ollama:gemma2:9b": {"input_cost": 0.0, "output_cost": 0.0},
        }
        
        # Latency baselines (ms) - typical response times
        self.latency_baselines = {
            "openai:gpt-4": 2000,
            "openai:gpt-3.5-turbo": 800,
            "anthropic:claude-sonnet-4": 1500,
            "anthropic:claude-haiku": 500,
            "ollama:llama3.2:3b": 300,  # Local models faster
            "ollama:gemma2:9b": 500,
        }
        
        # Optimization weights for different objectives
        self.optimization_weights = {
            OptimizationObjective.PERFORMANCE: {
                "performance": 0.5, "accuracy": 0.3, "availability": 0.2, 
                "cost": 0.0, "latency": 0.0
            },
            OptimizationObjective.COST: {
                "cost": 0.6, "performance": 0.2, "availability": 0.1,
                "accuracy": 0.1, "latency": 0.0
            },
            OptimizationObjective.LATENCY: {
                "latency": 0.5, "availability": 0.2, "performance": 0.2,
                "accuracy": 0.1, "cost": 0.0
            },
            OptimizationObjective.ACCURACY: {
                "accuracy": 0.5, "performance": 0.3, "availability": 0.2,
                "cost": 0.0, "latency": 0.0
            },
            OptimizationObjective.BALANCED: {
                "performance": 0.25, "cost": 0.2, "latency": 0.2,
                "accuracy": 0.2, "availability": 0.15
            }
        }
    
    def select_optimal_model(
        self, 
        requirements: ModelRequirements,
        available_models: Optional[List[str]] = None
    ) -> str:
        """Select optimal model using intelligent multi-dimensional optimization.
        
        Args:
            requirements: Enhanced model requirements
            available_models: Optional list to filter from
            
        Returns:
            Model key of selected optimal model
            
        Raises:
            NoEligibleModelsError: If no models meet requirements
        """
        start_time = time.time()
        
        # Get candidate models (builds on existing registry capabilities)
        if available_models is None:
            candidate_models = self._get_candidate_models(requirements)
        else:
            candidate_models = self._filter_candidates(available_models, requirements)
            
        if not candidate_models:
            raise NoEligibleModelsError("No models meet the specified requirements")
        
        # Score all candidates using multi-dimensional analysis
        scored_models = self._score_models(candidate_models, requirements)
        
        # Select best model based on optimization objective
        optimal_model = self._select_best_score(scored_models, requirements)
        
        selection_time = time.time() - start_time
        logger.info(
            f"Intelligent model selection completed in {selection_time:.3f}s: "
            f"{optimal_model.model_key} (score: {optimal_model.weighted_score:.3f}, "
            f"confidence: {optimal_model.confidence:.3f})"
        )
        
        return optimal_model.model_key
    
    def get_model_recommendations(
        self,
        requirements: ModelRequirements,
        top_k: int = 3
    ) -> List[ModelScore]:
        """Get top-k model recommendations with detailed scoring.
        
        Args:
            requirements: Model requirements
            top_k: Number of recommendations to return
            
        Returns:
            List of ModelScore objects, sorted by weighted score
        """
        candidate_models = self._get_candidate_models(requirements)
        if not candidate_models:
            return []
            
        scored_models = self._score_models(candidate_models, requirements)
        
        # Sort by weighted score and return top-k
        scored_models.sort(key=lambda x: x.weighted_score, reverse=True)
        return scored_models[:top_k]
    
    def explain_selection(
        self,
        model_key: str, 
        requirements: ModelRequirements
    ) -> Dict[str, Any]:
        """Provide detailed explanation of why a model was selected.
        
        Args:
            model_key: Selected model key
            requirements: Requirements used for selection
            
        Returns:
            Dictionary with detailed explanation
        """
        scored_models = self._score_models([model_key], requirements)
        if not scored_models:
            return {"error": "Model not found or not scoreable"}
            
        score = scored_models[0]
        weights = self.optimization_weights[requirements.optimization_objective]
        
        return {
            "model_key": model_key,
            "overall_score": score.weighted_score,
            "confidence": score.confidence,
            "optimization_objective": requirements.optimization_objective.value,
            "score_breakdown": {
                "performance": {"score": score.performance_score, "weight": weights["performance"]},
                "cost": {"score": score.cost_score, "weight": weights["cost"]},
                "latency": {"score": score.latency_score, "weight": weights["latency"]},
                "accuracy": {"score": score.accuracy_score, "weight": weights["accuracy"]},
                "availability": {"score": score.availability_score, "weight": weights["availability"]},
            },
            "estimated_cost": score.estimated_cost,
            "estimated_latency_ms": score.estimated_latency_ms,
            "provider": score.provider,
            "model_name": score.model_name,
        }
    
    def _get_candidate_models(self, requirements: ModelRequirements) -> List[str]:
        """Get candidate models based on basic requirements.
        
        Uses existing model registry capabilities from Phase 1 & 2.
        """
        # Use existing capability-based filtering
        if requirements.capabilities:
            # Find models that match any of the required capabilities
            candidates = set()
            for capability in requirements.capabilities:
                models_for_capability = self.model_registry.find_models_by_capability(capability)
                # Convert model objects to model keys
                for model in models_for_capability:
                    model_key = f"{model.provider}:{model.name}"
                    candidates.add(model_key)
            candidates = list(candidates)
        else:
            candidates = list(self.model_registry.models.keys())
        
        # Apply basic filters using existing registry methods
        filtered_candidates = []
        for model_key in candidates:
            try:
                model = self.model_registry.get_model(model_key.split(':', 1)[1], 
                                                    model_key.split(':', 1)[0])
                
                # Apply parameter size filters
                if self._meets_parameter_requirements(model, requirements):
                    # Apply provider preferences
                    if self._meets_provider_preferences(model_key, requirements):
                        filtered_candidates.append(model_key)
                        
            except (ModelNotFoundError, IndexError):
                continue
                
        return filtered_candidates
    
    def _filter_candidates(
        self, 
        available_models: List[str], 
        requirements: ModelRequirements
    ) -> List[str]:
        """Filter available models based on requirements."""
        candidates = self._get_candidate_models(requirements)
        return [model for model in available_models if model in candidates]
    
    def _score_models(
        self,
        model_keys: List[str],
        requirements: ModelRequirements
    ) -> List[ModelScore]:
        """Score models using multi-dimensional analysis."""
        scores = []
        
        for model_key in model_keys:
            try:
                score = self._score_single_model(model_key, requirements)
                if score:
                    scores.append(score)
            except Exception as e:
                logger.warning(f"Failed to score model {model_key}: {e}")
                continue
                
        return scores
    
    def _score_single_model(
        self,
        model_key: str,
        requirements: ModelRequirements
    ) -> Optional[ModelScore]:
        """Score a single model across all dimensions."""
        try:
            # Parse model key
            provider, model_name = model_key.split(':', 1)
            
            # Get model instance for metadata
            model = self.model_registry.get_model(model_name, provider)
            
            # Calculate individual scores
            performance_score = self._calculate_performance_score(model_key)
            cost_score = self._calculate_cost_score(model_key, requirements)
            latency_score = self._calculate_latency_score(model_key, requirements)  
            accuracy_score = self._calculate_accuracy_score(model_key)
            availability_score = self._calculate_availability_score(model_key)
            
            # Calculate weighted composite score
            weights = self.optimization_weights[requirements.optimization_objective]
            weighted_score = (
                performance_score * weights["performance"] +
                cost_score * weights["cost"] +
                latency_score * weights["latency"] +
                accuracy_score * weights["accuracy"] +
                availability_score * weights["availability"]
            )
            
            # Calculate confidence based on data availability
            confidence = self._calculate_confidence(model_key)
            
            return ModelScore(
                model_key=model_key,
                performance_score=performance_score,
                cost_score=cost_score,
                latency_score=latency_score,
                accuracy_score=accuracy_score,
                availability_score=availability_score,
                weighted_score=weighted_score,
                confidence=confidence,
                provider=provider,
                model_name=model_name,
                estimated_cost=self._estimate_cost(model_key, requirements),
                estimated_latency_ms=self._estimate_latency(model_key, requirements)
            )
            
        except Exception as e:
            logger.error(f"Error scoring model {model_key}: {e}")
            return None
    
    def _calculate_performance_score(self, model_key: str) -> float:
        """Calculate performance score based on UCB algorithm data."""
        # Use existing UCB statistics from model selector
        if hasattr(self.model_registry, 'model_selector'):
            stats = self.model_registry.model_selector.model_stats.get(model_key)
            if stats:
                # Convert UCB average reward to 0-1 score
                return min(1.0, max(0.0, stats.get("average_reward", 0.5)))
        
        # Default score for models without history
        return 0.5
    
    def _calculate_cost_score(self, model_key: str, requirements: ModelRequirements) -> float:
        """Calculate cost score (higher score = lower cost)."""
        cost_info = self.cost_data.get(model_key, {"input_cost": 0.1, "output_cost": 0.1})
        
        if requirements.expected_tokens:
            # Estimate total cost (assume 50/50 input/output split)
            input_tokens = requirements.expected_tokens * 0.5
            output_tokens = requirements.expected_tokens * 0.5
            total_cost = (
                (input_tokens / 1000) * cost_info["input_cost"] +
                (output_tokens / 1000) * cost_info["output_cost"]
            )
            
            # Apply budget constraint if specified
            if requirements.max_cost_per_token:
                cost_per_token = total_cost / requirements.expected_tokens
                if cost_per_token > requirements.max_cost_per_token:
                    return 0.0  # Exceeds budget
                    
            # Convert to 0-1 score (lower cost = higher score)
            # Use sigmoid-like transformation for better scaling
            if total_cost <= 0.0:
                return 1.0  # Free models get perfect score
            else:
                # Scale based on typical cost ranges (0.001 to 1.0 dollars)
                normalized_cost = min(1.0, total_cost / 0.1)  # Normalize to 0-1 range
                return max(0.0, 1.0 - normalized_cost)
        else:
            # Use average cost as baseline
            avg_cost = (cost_info["input_cost"] + cost_info["output_cost"]) / 2
            if avg_cost <= 0.0:
                return 1.0  # Free models
            else:
                # Normalize against expensive baseline (0.1 per 1K tokens)
                normalized_cost = min(1.0, avg_cost / 0.1)
                return max(0.0, 1.0 - normalized_cost)
    
    def _calculate_latency_score(self, model_key: str, requirements: ModelRequirements) -> float:
        """Calculate latency score (higher score = lower latency)."""
        baseline_latency = self.latency_baselines.get(model_key, 1000)  # ms
        
        # Apply latency constraint if specified
        if requirements.max_latency_ms:
            if baseline_latency > requirements.max_latency_ms:
                return 0.0  # Exceeds latency requirement
        
        # Convert to 0-1 score (lower latency = higher score)
        # Use log scale for latency sensitivity  
        max_acceptable_latency = 5000  # 5 seconds
        return max(0.0, 1.0 - np.log10(baseline_latency) / np.log10(max_acceptable_latency))
    
    def _calculate_accuracy_score(self, model_key: str) -> float:
        """Calculate accuracy score based on success rate."""
        # Use existing UCB statistics for success rate
        if hasattr(self.model_registry, 'model_selector'):
            stats = self.model_registry.model_selector.model_stats.get(model_key)
            if stats and stats.get("attempts", 0) > 0:
                success_rate = stats.get("successes", 0) / stats["attempts"]
                return success_rate
        
        # Default accuracy for models without history
        return 0.8  # Assume reasonable accuracy
    
    def _calculate_availability_score(self, model_key: str) -> float:
        """Calculate availability score based on service health."""
        provider = model_key.split(':', 1)[0]
        
        # Check service health using Phase 2 service managers
        from ..utils.service_manager import SERVICE_MANAGERS
        
        if provider == "ollama":
            ollama_manager = SERVICE_MANAGERS.get("ollama")
            if ollama_manager:
                if ollama_manager.is_running():
                    model_name = model_key.split(':', 1)[1]
                    if ollama_manager.is_model_available(model_name):
                        return 1.0
                    else:
                        return 0.3  # Service running but model not available
                else:
                    return 0.1  # Service not running
        elif provider in ["openai", "anthropic", "google"]:
            # External APIs - assume good availability but not perfect
            return 0.9
        
        # Default availability
        return 0.7
    
    def _calculate_confidence(self, model_key: str) -> float:
        """Calculate confidence in scoring based on data availability."""
        confidence_factors = []
        
        # UCB data availability
        if hasattr(self.model_registry, 'model_selector'):
            stats = self.model_registry.model_selector.model_stats.get(model_key)
            if stats:
                attempts = stats.get("attempts", 0)
                # More attempts = higher confidence
                confidence_factors.append(min(1.0, attempts / 10.0))
            else:
                confidence_factors.append(0.2)  # Low confidence without data
        
        # Cost data availability
        if model_key in self.cost_data:
            confidence_factors.append(1.0)
        else:
            confidence_factors.append(0.5)
            
        # Latency data availability  
        if model_key in self.latency_baselines:
            confidence_factors.append(1.0)
        else:
            confidence_factors.append(0.5)
        
        # Return average confidence
        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
    
    def _select_best_score(
        self,
        scored_models: List[ModelScore],
        requirements: ModelRequirements
    ) -> ModelScore:
        """Select best model from scored candidates."""
        if not scored_models:
            raise NoEligibleModelsError("No valid model scores available")
        
        # Filter by hard constraints first
        valid_models = []
        for score in scored_models:
            if self._meets_hard_constraints(score, requirements):
                valid_models.append(score)
                
        if not valid_models:
            # Relax constraints and use best available
            logger.warning("No models meet hard constraints, using best available")
            valid_models = scored_models
        
        # Sort by weighted score and confidence
        valid_models.sort(
            key=lambda x: (x.weighted_score * x.confidence),
            reverse=True
        )
        
        return valid_models[0]
    
    def _meets_hard_constraints(
        self,
        score: ModelScore,
        requirements: ModelRequirements
    ) -> bool:
        """Check if model meets hard constraints."""
        # Cost constraint
        if requirements.max_cost_per_token and score.estimated_cost:
            expected_tokens = requirements.expected_tokens or 1000
            cost_per_token = score.estimated_cost / expected_tokens
            if cost_per_token > requirements.max_cost_per_token:
                return False
        
        # Latency constraint
        if requirements.max_latency_ms and score.estimated_latency_ms:
            if score.estimated_latency_ms > requirements.max_latency_ms:
                return False
        
        # Accuracy constraint
        if requirements.min_accuracy_score:
            if score.accuracy_score < requirements.min_accuracy_score:
                return False
                
        return True
    
    def _meets_parameter_requirements(self, model: Model, requirements: ModelRequirements) -> bool:
        """Check if model meets parameter size requirements."""
        # Implementation depends on model metadata structure
        # This is a placeholder that can be enhanced based on actual model attributes
        return True
    
    def _meets_provider_preferences(self, model_key: str, requirements: ModelRequirements) -> bool:
        """Check if model meets provider preferences."""
        if not requirements.preferred_providers:
            return True
            
        provider = model_key.split(':', 1)[0]
        return provider in requirements.preferred_providers
    
    def _estimate_cost(self, model_key: str, requirements: ModelRequirements) -> Optional[float]:
        """Estimate total cost for the request."""
        if not requirements.expected_tokens:
            return None
            
        cost_info = self.cost_data.get(model_key, {"input_cost": 0.1, "output_cost": 0.1})
        
        # Assume 50/50 input/output split
        input_tokens = requirements.expected_tokens * 0.5
        output_tokens = requirements.expected_tokens * 0.5
        
        return (
            (input_tokens / 1000) * cost_info["input_cost"] +
            (output_tokens / 1000) * cost_info["output_cost"]
        )
    
    def _estimate_latency(self, model_key: str, requirements: ModelRequirements) -> Optional[int]:
        """Estimate latency for the request."""
        baseline = self.latency_baselines.get(model_key, 1000)
        
        # Adjust for batch size and concurrent requests
        if requirements.batch_size and requirements.batch_size > 1:
            baseline = int(baseline * (1 + 0.2 * (requirements.batch_size - 1)))
            
        if requirements.concurrent_requests and requirements.concurrent_requests > 1:
            baseline = int(baseline * (1 + 0.1 * (requirements.concurrent_requests - 1)))
            
        return baseline


# Helper functions for integration with existing system
def create_intelligent_selector(model_registry: ModelRegistry) -> IntelligentModelSelector:
    """Create intelligent selector from existing model registry."""
    return IntelligentModelSelector(model_registry)


def select_optimal_model_for_task(
    task_description: str,
    optimization_objective: OptimizationObjective = OptimizationObjective.BALANCED,
    model_registry: Optional[ModelRegistry] = None
) -> str:
    """High-level function to select optimal model for a task description.
    
    Args:
        task_description: Natural language description of the task
        optimization_objective: What to optimize for
        model_registry: Optional registry instance
        
    Returns:
        Selected model key
    """
    if model_registry is None:
        # Use global registry if available
        from ..models.model_registry import ModelRegistry
        model_registry = ModelRegistry()
    
    # Create basic requirements from task description
    # This could be enhanced with NLP analysis of the task
    requirements = ModelRequirements(
        optimization_objective=optimization_objective,
        expected_tokens=1000,  # Default estimate
    )
    
    # Add capability inference based on task keywords
    if any(keyword in task_description.lower() for keyword in ["code", "programming", "python"]):
        if "code_generation" not in requirements.capabilities:
            requirements.capabilities.append("code_generation")
    if any(keyword in task_description.lower() for keyword in ["analysis", "reasoning"]):
        if "analysis" not in requirements.capabilities:
            requirements.capabilities.append("analysis")
    if any(keyword in task_description.lower() for keyword in ["creative", "story", "write"]):
        if "creative_writing" not in requirements.capabilities:
            requirements.capabilities.append("creative_writing")
    
    selector = IntelligentModelSelector(model_registry)
    return selector.select_optimal_model(requirements)