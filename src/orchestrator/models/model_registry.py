"""Model registry for managing and selecting models."""

from __future__ import annotations

import asyncio
import math
from typing import Any, Dict, List, Optional

from ..core.model import Model, ModelMetrics
from ..core.exceptions import ModelNotFoundError, NoEligibleModelsError
from .performance_optimizations import ModelRegistryOptimizer, BatchModelProcessor
from .memory_optimization import MemoryOptimizedRegistry, MemoryMonitor, optimize_model_registry_memory
from .advanced_caching import CacheManager, background_cache_maintenance
from .langchain_adapter import LangChainModelAdapter


class ModelRegistry:
    """
    Central registry for all available models.

    Manages model registration, selection, and performance tracking
    using multi-armed bandit algorithms.
    """

    def __init__(self, 
                 enable_memory_optimization: bool = True, 
                 memory_warning_threshold: float = 80.0,
                 enable_advanced_caching: bool = True) -> None:
        """Initialize model registry."""
        self.models: Dict[str, Model] = {}
        self.model_selector = UCBModelSelector()
        self._model_health_cache: Dict[str, bool] = {}
        self._cache_ttl = 300  # 5 minutes
        self._last_health_check = 0  # Will be set to current time on first use
        self._auto_registrar = None  # Will be set up after initialization
        
        # Performance optimizations
        self.optimizer = ModelRegistryOptimizer()
        self.batch_processor = BatchModelProcessor()
        self._index_rebuild_threshold = 10  # Rebuild index after N model changes
        
        # Memory optimization
        self.enable_memory_optimization = enable_memory_optimization
        if enable_memory_optimization:
            self.memory_monitor = MemoryMonitor(memory_warning_threshold)
            self._memory_optimization_enabled = True
            # Set up memory callbacks for automatic optimization
            self.memory_monitor.add_callback('warning', self._on_memory_warning)
            self.memory_monitor.add_callback('critical', self._on_memory_critical)
        else:
            self.memory_monitor = None
            self._memory_optimization_enabled = False
        
        # Advanced caching
        self.enable_advanced_caching = enable_advanced_caching
        if enable_advanced_caching:
            self.cache_manager = CacheManager(registry=self)
            self._advanced_caching_enabled = True
            self._background_maintenance_task = None
        else:
            self.cache_manager = None
            self._advanced_caching_enabled = False
            
        # LangChain integration
        self._langchain_adapters: Dict[str, LangChainModelAdapter] = {}
        self._langchain_enabled = True

    def register_model(self, model: Model) -> None:
        """
        Register a new model.

        Args:
            model: Model to register

        Raises:
            ValueError: If model with same name already exists
        """
        model_key = self._get_model_key(model)

        if model_key in self.models:
            raise ValueError(f"Model '{model_key}' already registered")

        self.models[model_key] = model

        # Initialize model in selector
        self.model_selector.initialize_model(model_key, model.metrics)
        
        # Update performance index
        self.optimizer.update_model_index(model_key, model)
        
        # Check memory usage after registration
        if self._memory_optimization_enabled and self.memory_monitor:
            self.memory_monitor.check_memory()
        
        # Invalidate advanced caches for this model
        if self._advanced_caching_enabled and self.cache_manager:
            self.cache_manager.invalidate_model(model_key)

    def unregister_model(self, model_name: str, provider: str = "") -> None:
        """
        Unregister a model.

        Args:
            model_name: Name of model to unregister
            provider: Provider name (optional)
        """
        if provider:
            model_key = f"{provider}:{model_name}"
        else:
            # Find model by name only
            model_key = None
            for key in self.models:
                if key.split(":")[-1] == model_name:
                    model_key = key
                    break

        if model_key and model_key in self.models:
            del self.models[model_key]
            self.model_selector.remove_model(model_key)
            
            # Mark index as dirty (will be rebuilt on next use)
            self.optimizer.index_dirty = True
        else:
            raise ModelNotFoundError(f"Model '{model_name}' not found")

    def register_langchain_model(self, provider: str, model_name: str, **config: Any) -> str:
        """
        Register a LangChain model adapter.
        
        Args:
            provider: Provider name (e.g., "openai", "anthropic", "ollama")  
            model_name: Model name
            **config: Additional configuration for the model
            
        Returns:
            Model key for the registered adapter
            
        Raises:
            ValueError: If LangChain integration is disabled or adapter creation fails
        """
        if not self._langchain_enabled:
            raise ValueError("LangChain integration is disabled")
            
        try:
            # Create LangChain adapter
            adapter = LangChainModelAdapter(provider, model_name, **config)
            adapter_key = self._get_model_key(adapter)
            
            # Register the adapter as a regular model (preserving all UCB/caching logic)
            self.register_model(adapter)
            
            # Store reference to the adapter
            self._langchain_adapters[adapter_key] = adapter
            
            return adapter_key
            
        except Exception as e:
            raise ValueError(f"Failed to create LangChain adapter for {provider}:{model_name}: {str(e)}")
    
    def unregister_langchain_model(self, provider: str, model_name: str) -> None:
        """
        Unregister a LangChain model adapter.
        
        Args:
            provider: Provider name
            model_name: Model name
        """
        adapter_key = f"{provider}:{model_name}"
        
        # Remove from LangChain adapters
        if adapter_key in self._langchain_adapters:
            del self._langchain_adapters[adapter_key]
            
        # Remove from general registry
        if adapter_key in self.models:
            self.unregister_model(model_name, provider)
    
    def get_langchain_adapters(self) -> Dict[str, LangChainModelAdapter]:
        """
        Get all registered LangChain adapters.
        
        Returns:
            Dictionary of adapter_key -> LangChainModelAdapter
        """
        return self._langchain_adapters.copy()
    
    def is_langchain_model(self, model_key: str) -> bool:
        """
        Check if a model is a LangChain adapter.
        
        Args:
            model_key: Model key to check
            
        Returns:
            True if model is a LangChain adapter
        """
        return model_key in self._langchain_adapters
    
    def enable_langchain_integration(self) -> None:
        """Enable LangChain integration."""
        self._langchain_enabled = True
        
    def disable_langchain_integration(self) -> None:
        """Disable LangChain integration (existing adapters remain)."""
        self._langchain_enabled = False
    
    def auto_register_langchain_models(self, config: Dict[str, Any]) -> List[str]:
        """
        Auto-register LangChain models from configuration.
        
        Args:
            config: Configuration dictionary with model definitions
            
        Returns:
            List of registered model keys
            
        Example config:
        {
            "models": [
                {"provider": "openai", "model": "gpt-4-turbo", "auto_install": true},
                {"provider": "anthropic", "model": "claude-3-sonnet", "auto_install": true},
                {"provider": "ollama", "model": "llama3.2:3b", "ensure_running": true}
            ]
        }
        """
        registered_keys = []
        
        if not self._langchain_enabled:
            raise ValueError("LangChain integration is disabled")
            
        models_config = config.get("models", [])
        
        for model_config in models_config:
            try:
                provider = model_config["provider"]
                model_name = model_config["model"]
                
                # Handle service requirements
                if model_config.get("ensure_running", False):
                    self._ensure_service_running(provider)
                    
                # Handle auto-installation
                if model_config.get("auto_install", False):
                    self._auto_install_dependencies(provider)
                    
                # Register the model
                model_key = self.register_langchain_model(
                    provider, 
                    model_name, 
                    **{k: v for k, v in model_config.items() 
                       if k not in ["provider", "model", "auto_install", "ensure_running"]}
                )
                
                registered_keys.append(model_key)
                
            except Exception as e:
                # Log error but continue with other models
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Failed to auto-register {model_config}: {e}")
                continue
                
        return registered_keys
    
    def _ensure_service_running(self, provider: str) -> None:
        """Ensure required service is running for provider."""
        try:
            from ..utils.service_manager import ensure_service_running
            
            service_map = {
                "ollama": "ollama",
                "docker": "docker",
            }
            
            service_name = service_map.get(provider)
            if service_name:
                ensure_service_running(service_name)
                
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to ensure service running for {provider}: {e}")
    
    def _auto_install_dependencies(self, provider: str) -> None:
        """Auto-install dependencies for provider."""
        try:
            from ..utils.auto_install import ensure_packages
            
            package_map = {
                "openai": ["langchain-openai"],
                "anthropic": ["langchain-anthropic"], 
                "google": ["langchain-google-genai"],
                "ollama": ["langchain-community"],
                "huggingface": ["langchain-huggingface"],
            }
            
            packages = package_map.get(provider, [])
            if packages:
                ensure_packages(packages)
                
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to auto-install dependencies for {provider}: {e}")

    def enable_auto_registration(self) -> None:
        """Enable automatic model registration for new models."""
        if self._auto_registrar is None:
            from .auto_register import AutoModelRegistrar

            self._auto_registrar = AutoModelRegistrar(self)

    async def get_model_async(self, model_name: str, provider: str = "") -> Model:
        """
        Get a model by name, attempting auto-registration if not found.

        Args:
            model_name: Model name
            provider: Provider name (optional)

        Returns:
            Model instance

        Raises:
            ModelNotFoundError: If model not found and auto-registration fails
        """
        try:
            return self.get_model(model_name, provider)
        except ModelNotFoundError:
            # Try auto-registration if enabled
            if self._auto_registrar and provider:
                model = await self._auto_registrar.try_register_model(
                    model_name, provider
                )
                if model:
                    return model
            elif self._auto_registrar and not provider:
                # Try to guess provider from model name
                from .auto_register import get_provider_from_model_name

                guessed_provider = get_provider_from_model_name(model_name)
                if guessed_provider:
                    model = await self._auto_registrar.try_register_model(
                        model_name, guessed_provider
                    )
                    if model:
                        return model
            raise

    def get_model(self, model_name: str, provider: str = "") -> Model:
        """
        Get a model by name.

        Args:
            model_name: Model name (can include provider prefix like "openai:gpt-4")
            provider: Provider name (optional, ignored if model_name includes provider)

        Returns:
            Model instance

        Raises:
            ModelNotFoundError: If model not found
        """
        # Check if model_name already includes provider prefix
        if ":" in model_name and not provider:
            # Model name includes provider, use it directly
            model_key = model_name
        elif provider:
            # Provider specified separately
            model_key = f"{provider}:{model_name}"
        else:
            # Find model by name only
            model_key = None
            for key in self.models:
                if key.split(":")[-1] == model_name:
                    model_key = key
                    break

        if model_key and model_key in self.models:
            return self.models[model_key]
        else:
            raise ModelNotFoundError(f"Model '{model_name}' not found")

    def list_models(self, provider: Optional[str] = None) -> List[str]:
        """
        List all registered models.

        Args:
            provider: Filter by provider (optional)

        Returns:
            List of model names
        """
        if provider:
            return [key for key in self.models.keys() if key.startswith(f"{provider}:")]
        return list(self.models.keys())

    def list_providers(self) -> List[str]:
        """
        List all providers.

        Returns:
            List of provider names
        """
        providers = set()
        for key in self.models.keys():
            if ":" in key:
                providers.add(key.split(":")[0])
        return sorted(list(providers))

    async def get_available_models(self) -> List[str]:
        """
        Get list of available (healthy) models.

        Returns:
            List of available model keys
        """
        all_models = list(self.models.values())
        healthy_models = await self._filter_by_health(all_models)
        return [self._get_model_key(model) for model in healthy_models]

    async def select_model(self, requirements: Dict[str, Any]) -> Model:
        """
        Select best model for given requirements.

        Args:
            requirements: Requirements dictionary

        Returns:
            Selected model

        Raises:
            NoEligibleModelsError: If no models meet requirements
        """
        print(f">> DEBUG ModelRegistry.select_model called with: {requirements}")

        # Check advanced cache first
        if self._advanced_caching_enabled and self.cache_manager:
            requirements_hash = self._hash_requirements(requirements)
            cached_model_key = self.cache_manager.get_cached_selection(requirements_hash)
            
            if cached_model_key and cached_model_key in self.models:
                # Verify cached model is still healthy and eligible
                cached_model = self.models[cached_model_key]
                if await self._is_model_healthy(cached_model):
                    print(f">> DEBUG: Using cached selection: {cached_model_key}")
                    return cached_model

        # Step 1: Filter by capabilities
        eligible_models = await self._filter_by_capabilities(requirements)

        if not eligible_models:
            print(
                f">> DEBUG: No models passed capability filter. Total models: {len(self.models)}"
            )
            raise NoEligibleModelsError("No models meet the specified requirements")

        # Step 2: Filter by health
        healthy_models = await self._filter_by_health(eligible_models)

        if not healthy_models:
            raise NoEligibleModelsError("No healthy models available")

        # Step 2.5: Apply provider preference for test stability
        # Prefer models less likely to have authentication issues
        prioritized_models = self._apply_provider_preference(healthy_models)

        # Step 3: Use bandit algorithm for selection
        selected_key = self.model_selector.select(
            [self._get_model_key(model) for model in prioritized_models], requirements
        )

        # Cache the selection result
        if self._advanced_caching_enabled and self.cache_manager:
            requirements_hash = self._hash_requirements(requirements)
            self.cache_manager.cache_selection(requirements_hash, selected_key)

        return self.models[selected_key]

    def _apply_provider_preference(self, models: List[Model]) -> List[Model]:
        """Apply provider preference to prioritize stable models for tests.
        
        Priority order (highest to lowest):
        1. OpenAI - most reliable API, rarely gated
        2. Anthropic - reliable API, rarely gated  
        3. Google - reliable API, rarely gated
        4. HuggingFace - can have authentication issues with gated models
        5. Ollama - local models, but may require download
        
        Args:
            models: List of models to prioritize
            
        Returns:
            Reordered list with preferred providers first
        """
        provider_priority = {
            'openai': 1,
            'anthropic': 2,
            'google': 3,
            'huggingface': 4,
            'ollama': 5,
        }
        
        def get_provider_priority(model: Model) -> tuple:
            """Get priority tuple for sorting (lower is better)"""
            model_name = self._get_model_key(model)
            
            # Extract provider from model name (format: "provider:model_name")
            if ':' in model_name:
                provider = model_name.split(':', 1)[0].lower()
            else:
                provider = 'unknown'
            
            # Get priority (lower is better), default to high number for unknown
            priority = provider_priority.get(provider, 99)
            
            return (priority, model_name)  # Secondary sort by name for consistency
        
        # Sort models by provider preference
        sorted_models = sorted(models, key=get_provider_priority)
        
        # Log the prioritization for debugging
        if sorted_models:
            first_model = self._get_model_key(sorted_models[0])
            print(f">> DEBUG: Prioritized model selection, chose: {first_model}")
        
        return sorted_models

    async def _filter_by_capabilities(
        self, requirements: Dict[str, Any]
    ) -> List[Model]:
        """Filter models by capabilities and expertise (with performance optimizations)."""
        # Rebuild index if necessary
        if self.optimizer.index_dirty or len(self.optimizer.index.size_index) != len(self.models):
            self.optimizer.build_index(self.models)
        
        # Try to use fast filtering if we have enough models to justify it
        if len(self.models) > 50:
            try:
                # Convert requirements to ModelSelectionCriteria for fast filtering
                criteria = self._requirements_to_criteria(requirements)
                
                # Get candidate model keys using optimized indexes
                candidate_keys = self.optimizer.fast_filter_by_criteria(criteria)
                
                # Convert back to model objects
                candidates = [self.models[key] for key in candidate_keys if key in self.models]
                
                # Apply any remaining filters not handled by fast filtering
                eligible = self._apply_detailed_filters(candidates, requirements)
                
                return eligible
            except Exception:
                # Fall back to original method if fast filtering fails
                pass
        
        # Original filtering logic for smaller registries or fallback
        eligible = []

        # Extract all requirement types
        required_expertise = requirements.get("expertise", [])
        if isinstance(required_expertise, str):
            required_expertise = [required_expertise]

        # Size constraints
        min_size_str = requirements.get("min_size", "0")
        max_size_str = requirements.get("max_size")
        from ..utils.model_utils import parse_model_size
        min_size_billions = parse_model_size("", min_size_str)
        max_size_billions = None
        if max_size_str:
            max_size_billions = parse_model_size("", max_size_str)

        # Performance constraints
        min_tokens_per_second = requirements.get("min_tokens_per_second")
        cost_limit = requirements.get("cost_limit")
        budget_period = requirements.get("budget_period", "per-task")

        for model in self.models.values():
            # Check basic requirements first
            if not model.meets_requirements(requirements):
                continue

            # Check expertise level if specified
            if required_expertise:
                if isinstance(required_expertise, list) and len(required_expertise) > 0:
                    # Handle legacy list format and new expertise level format
                    if isinstance(required_expertise[0], str) and required_expertise[0] in ["low", "medium", "high", "very-high"]:
                        # New expertise level format
                        if not self._meets_expertise_level(model, required_expertise[0]):
                            continue
                    else:
                        # Legacy list format
                        model_expertise = getattr(model, "_expertise", ["general"])
                        if not any(exp in model_expertise for exp in required_expertise):
                            continue

            # Check size constraints
            model_size = getattr(model, "_size_billions", 1.0)
            if model_size < min_size_billions:
                continue
            if max_size_billions and model_size > max_size_billions:
                continue

            # Check performance constraints
            if min_tokens_per_second:
                model_throughput = getattr(model.metrics, "throughput", 0)
                if model_throughput < min_tokens_per_second:
                    continue

            # Check cost constraints
            if cost_limit and not self._meets_cost_constraint(model, cost_limit, budget_period):
                continue

            eligible.append(model)

        # Sort eligible models to prefer API models over local models for reliability
        # This helps avoid timeout issues with HuggingFace models that need downloading
        def model_priority(model: Model) -> int:
            """Return priority score (lower is better)."""
            provider = model.provider.lower()
            # API models get highest priority
            if provider in ["openai", "anthropic", "google"]:
                return 0
            # Ollama models are local but fast if already installed
            elif provider == "ollama":
                return 1
            # HuggingFace models may need downloading
            elif provider == "huggingface":
                return 2
            # Unknown providers
            else:
                return 3

        eligible.sort(key=model_priority)

        return eligible

    def _requirements_to_criteria(self, requirements: Dict[str, Any]) -> 'ModelSelectionCriteria':
        """Convert requirements dict to ModelSelectionCriteria for fast filtering."""
        from .model_selector import ModelSelectionCriteria
        
        criteria = ModelSelectionCriteria()
        
        # Map requirements to criteria fields
        expertise = requirements.get("expertise")
        if isinstance(expertise, str):
            criteria.expertise = expertise
        elif isinstance(expertise, list) and expertise:
            # For legacy list format, try to infer level
            if any(e in ["low", "medium", "high", "very-high"] for e in expertise):
                criteria.expertise = expertise[0]
        
        # Size constraints
        min_size_str = requirements.get("min_size")
        if min_size_str:
            from ..utils.model_utils import parse_model_size
            criteria.min_model_size = parse_model_size("", min_size_str)
        
        max_size_str = requirements.get("max_size")
        if max_size_str:
            from ..utils.model_utils import parse_model_size
            criteria.max_model_size = parse_model_size("", max_size_str)
        
        # Cost constraints
        criteria.cost_limit = requirements.get("cost_limit")
        criteria.budget_period = requirements.get("budget_period", "per-task")
        
        # Capabilities
        if requirements.get("vision_capable"):
            criteria.modalities.append("vision")
        if requirements.get("code_specialized"):
            criteria.modalities.append("code")
        
        return criteria

    def _apply_detailed_filters(self, candidates: List[Model], requirements: Dict[str, Any]) -> List[Model]:
        """Apply detailed filters not handled by fast filtering."""
        filtered = []
        
        for model in candidates:
            # Apply any complex logic that couldn't be optimized
            if not model.meets_requirements(requirements):
                continue
            
            # Additional detailed checks can go here
            filtered.append(model)
        
        return filtered

    def _meets_expertise_level(self, model: Model, required_level: str) -> bool:
        """
        Check if model meets expertise level requirement.
        
        Args:
            model: Model to check
            required_level: Required expertise level ("low", "medium", "high", "very-high")
            
        Returns:
            True if model meets the requirement
        """
        # Define expertise hierarchy
        expertise_levels = {"low": 1, "medium": 2, "high": 3, "very-high": 4}
        
        # Get model's expertise level
        model_expertise_attrs = getattr(model, "_expertise", ["general"])
        if model_expertise_attrs is None:
            model_expertise_attrs = ["general"]
        
        # Map model expertise to level (check highest level first)
        model_level = "medium"  # Default level
        if "analysis" in model_expertise_attrs or "research" in model_expertise_attrs:
            model_level = "very-high"
        elif "code" in model_expertise_attrs or "reasoning" in model_expertise_attrs:
            model_level = "high"
        elif "fast" in model_expertise_attrs or "compact" in model_expertise_attrs:
            model_level = "low"
        
        # Check if model level meets requirement
        required_score = expertise_levels.get(required_level, 2)
        model_score = expertise_levels.get(model_level, 2)
        
        return model_score >= required_score

    def _meets_cost_constraint(self, model: Model, cost_limit: float, budget_period: str) -> bool:
        """
        Check if model meets cost constraint.
        
        Args:
            model: Model to check
            cost_limit: Maximum cost limit
            budget_period: Budget period ("per-task", "per-pipeline", "per-hour")
            
        Returns:
            True if model meets the cost constraint
        """
        # For now, do basic cost check based on model's cost per token
        # This can be enhanced later with usage tracking
        
        if model.cost.is_free:
            return True  # Free models always meet cost constraints
        
        # Estimate cost based on average usage
        if budget_period == "per-task":
            # Assume 1000 tokens average per task
            estimated_cost = model.cost.calculate_cost(500, 500)  # 500 input + 500 output
        elif budget_period == "per-pipeline":
            # Assume 5000 tokens average per pipeline
            estimated_cost = model.cost.calculate_cost(2500, 2500)
        elif budget_period == "per-hour":
            # Assume 50000 tokens per hour
            estimated_cost = model.cost.calculate_cost(25000, 25000)
        else:
            # Default to per-task
            estimated_cost = model.cost.calculate_cost(500, 500)
        
        return estimated_cost <= cost_limit

    def detect_model_capabilities(self, model: Model) -> Dict[str, Any]:
        """
        Detect and analyze model capabilities.
        
        Args:
            model: Model to analyze
            
        Returns:
            Dictionary with detected capabilities and analysis
        """
        capabilities = model.capabilities
        analysis = {
            "basic_capabilities": {
                "text_generation": "text-generation" in capabilities.supported_tasks,
                "chat": "chat" in capabilities.supported_tasks,
                "completion": "completion" in capabilities.supported_tasks,
            },
            "advanced_capabilities": {
                "vision": capabilities.vision_capable,
                "code": capabilities.code_specialized,
                "function_calling": capabilities.supports_function_calling,
                "structured_output": capabilities.supports_structured_output,
                "json_mode": capabilities.supports_json_mode,
            },
            "performance_metrics": {
                "accuracy_score": capabilities.accuracy_score,
                "speed_rating": capabilities.speed_rating,
                "context_window": capabilities.context_window,
            },
            "expertise_analysis": self._analyze_model_expertise(model),
            "cost_analysis": self._analyze_model_cost(model),
            "suitability_scores": self._calculate_suitability_scores(model)
        }
        
        return analysis

    def _analyze_model_expertise(self, model: Model) -> Dict[str, Any]:
        """
        Analyze model expertise areas and levels.
        
        Args:
            model: Model to analyze
            
        Returns:
            Dictionary with expertise analysis
        """
        expertise_attrs = getattr(model, "_expertise", ["general"])
        if expertise_attrs is None:
            expertise_attrs = ["general"]
        
        # Determine expertise level using existing logic
        level = "medium"  # Default
        if "analysis" in expertise_attrs or "research" in expertise_attrs:
            level = "very-high"
        elif "code" in expertise_attrs or "reasoning" in expertise_attrs:
            level = "high"
        elif "fast" in expertise_attrs or "compact" in expertise_attrs:
            level = "low"
        
        # Categorize expertise areas
        categories = {
            "coding": any(attr in expertise_attrs for attr in ["code", "programming", "software"]),
            "analysis": any(attr in expertise_attrs for attr in ["analysis", "research", "data"]),
            "creative": any(attr in expertise_attrs for attr in ["creative", "writing", "content"]),
            "reasoning": any(attr in expertise_attrs for attr in ["reasoning", "logic", "math"]),
            "general": "general" in expertise_attrs or not any(expertise_attrs),
            "specialized": len([attr for attr in expertise_attrs if attr not in ["general", "fast", "compact"]]) > 2
        }
        
        return {
            "level": level,
            "areas": expertise_attrs,
            "categories": categories,
            "specialization_score": len([cat for cat, has_cat in categories.items() if has_cat and cat != "general"])
        }

    def _analyze_model_cost(self, model: Model) -> Dict[str, Any]:
        """
        Analyze model cost characteristics.
        
        Args:
            model: Model to analyze
            
        Returns:
            Dictionary with cost analysis
        """
        cost = model.cost
        
        if cost.is_free:
            return {
                "type": "free",
                "cost_tier": "free",
                "efficiency_score": 100.0,
                "budget_friendly": True,
                "cost_per_1k_avg": 0.0
            }
        
        # Calculate average cost per 1k tokens
        avg_cost = (cost.input_cost_per_1k_tokens + cost.output_cost_per_1k_tokens) / 2
        
        # Determine cost tier
        if avg_cost < 0.001:
            tier = "very_low"
        elif avg_cost < 0.01:
            tier = "low"
        elif avg_cost < 0.1:
            tier = "medium"
        elif avg_cost < 1.0:
            tier = "high"
        else:
            tier = "very_high"
        
        # Calculate efficiency (performance per dollar)
        performance_score = model.capabilities.accuracy_score
        efficiency = cost.get_cost_efficiency_score(performance_score)
        
        return {
            "type": "paid",
            "cost_tier": tier,
            "efficiency_score": efficiency,
            "budget_friendly": avg_cost < 0.01,
            "cost_per_1k_avg": avg_cost,
            "base_cost": cost.base_cost_per_request,
            "estimated_task_cost": cost.estimate_cost_for_budget_period("per-task")
        }

    def _calculate_suitability_scores(self, model: Model) -> Dict[str, float]:
        """
        Calculate suitability scores for different use cases.
        
        Args:
            model: Model to analyze
            
        Returns:
            Dictionary with suitability scores (0-1)
        """
        caps = model.capabilities
        expertise = getattr(model, "_expertise", ["general"])
        if expertise is None:
            expertise = ["general"]
        
        scores = {}
        
        # Code tasks
        code_score = 0.5  # Base score
        if caps.code_specialized:
            code_score += 0.3
        if "code" in expertise or "programming" in expertise:
            code_score += 0.2
        scores["coding"] = min(code_score, 1.0)
        
        # Analysis tasks  
        analysis_score = caps.accuracy_score * 0.6  # Base on accuracy
        if "analysis" in expertise or "research" in expertise:
            analysis_score += 0.4
        scores["analysis"] = min(analysis_score, 1.0)
        
        # Creative tasks
        creative_score = 0.6  # Most models can do basic creative work
        if "creative" in expertise or "writing" in expertise:
            creative_score += 0.3
        if caps.context_window > 32000:  # Long context helps with creative work
            creative_score += 0.1
        scores["creative"] = min(creative_score, 1.0)
        
        # Chat/conversation
        chat_score = 0.7  # Most models handle chat reasonably
        if "chat" in caps.supported_tasks:
            chat_score += 0.2
        if "general" in expertise:
            chat_score += 0.1
        scores["chat"] = min(chat_score, 1.0)
        
        # Vision tasks
        vision_score = 0.1 if not caps.vision_capable else 0.8
        if caps.vision_capable and "vision" in expertise:
            vision_score += 0.2
        scores["vision"] = min(vision_score, 1.0)
        
        # Speed-critical tasks
        speed_scores = {"fast": 1.0, "medium": 0.6, "slow": 0.2}
        scores["speed_critical"] = speed_scores.get(caps.speed_rating, 0.5)
        
        # Budget-constrained tasks
        if model.cost.is_free:
            scores["budget_constrained"] = 1.0
        else:
            avg_cost = (model.cost.input_cost_per_1k_tokens + model.cost.output_cost_per_1k_tokens) / 2
            # Inverse relationship with cost
            scores["budget_constrained"] = max(0.1, min(1.0, 0.01 / max(avg_cost, 0.001)))
        
        return scores

    def find_models_by_capability(self, capability: str, threshold: float = 0.7) -> List[Model]:
        """
        Find models that excel at a specific capability.
        
        Args:
            capability: Capability to search for ("coding", "analysis", "creative", etc.)
            threshold: Minimum suitability score (0-1)
            
        Returns:
            List of suitable models sorted by suitability score
        """
        suitable_models = []
        
        for model in self.models.values():
            analysis = self.detect_model_capabilities(model)
            suitability_scores = analysis["suitability_scores"]
            
            if capability in suitability_scores:
                score = suitability_scores[capability]
                if score >= threshold:
                    suitable_models.append((model, score))
        
        # Sort by score descending
        suitable_models.sort(key=lambda x: x[1], reverse=True)
        return [model for model, score in suitable_models]

    def get_capability_matrix(self) -> Dict[str, Dict[str, float]]:
        """
        Get capability matrix for all registered models.
        
        Returns:
            Dictionary mapping model names to capability scores
        """
        matrix = {}
        
        for model_key, model in self.models.items():
            analysis = self.detect_model_capabilities(model)
            matrix[model_key] = analysis["suitability_scores"]
        
        return matrix

    def recommend_models_for_task(self, task_description: str, max_recommendations: int = 3) -> List[Dict[str, Any]]:
        """
        Recommend models for a given task description.
        
        Args:
            task_description: Description of the task
            max_recommendations: Maximum number of models to recommend
            
        Returns:
            List of model recommendations with reasoning
        """
        # Simple keyword-based matching (could be enhanced with NLP)
        task_lower = task_description.lower()
        
        # Determine primary capability needed
        capability_keywords = {
            "coding": ["code", "program", "script", "debug", "software", "development"],
            "analysis": ["analyze", "research", "study", "examine", "investigate", "data"],
            "creative": ["write", "story", "creative", "content", "blog", "article"],
            "vision": ["image", "visual", "picture", "photo", "see", "vision"],
            "chat": ["chat", "conversation", "talk", "discuss", "help"]
        }
        
        # Find matching capabilities
        relevant_capabilities = []
        for capability, keywords in capability_keywords.items():
            if any(keyword in task_lower for keyword in keywords):
                relevant_capabilities.append(capability)
        
        # Default to general chat if no specific capability identified
        if not relevant_capabilities:
            relevant_capabilities = ["chat"]
        
        # Get recommendations for each capability
        all_recommendations = []
        for capability in relevant_capabilities:
            models = self.find_models_by_capability(capability, threshold=0.5)
            for model in models[:max_recommendations]:
                analysis = self.detect_model_capabilities(model)
                recommendation = {
                    "model": model,
                    "model_key": self._get_model_key(model),
                    "capability": capability,
                    "suitability_score": analysis["suitability_scores"][capability],
                    "reasoning": self._generate_recommendation_reasoning(model, capability, analysis),
                    "cost_analysis": analysis["cost_analysis"],
                    "expertise_level": analysis["expertise_analysis"]["level"]
                }
                all_recommendations.append(recommendation)
        
        # Remove duplicates and sort by suitability
        seen_models = set()
        unique_recommendations = []
        for rec in all_recommendations:
            model_key = rec["model_key"]
            if model_key not in seen_models:
                seen_models.add(model_key)
                unique_recommendations.append(rec)
        
        # Sort by suitability score
        unique_recommendations.sort(key=lambda x: x["suitability_score"], reverse=True)
        
        return unique_recommendations[:max_recommendations]

    def _generate_recommendation_reasoning(self, model: Model, capability: str, analysis: Dict[str, Any]) -> str:
        """
        Generate reasoning for why a model is recommended.
        
        Args:
            model: Recommended model
            capability: Capability it's recommended for
            analysis: Model analysis results
            
        Returns:
            Human-readable reasoning string
        """
        reasons = []
        
        # Expertise-based reasoning
        expertise_level = analysis["expertise_analysis"]["level"]
        if expertise_level in ["high", "very-high"]:
            reasons.append(f"has {expertise_level} expertise level")
        
        # Capability-specific reasoning
        if capability == "coding" and model.capabilities.code_specialized:
            reasons.append("specialized for code generation")
        elif capability == "vision" and model.capabilities.vision_capable:
            reasons.append("supports vision/image processing")
        elif capability == "analysis" and "analysis" in analysis["expertise_analysis"]["areas"]:
            reasons.append("optimized for analytical tasks")
        
        # Cost reasoning
        cost_analysis = analysis["cost_analysis"]
        if cost_analysis["type"] == "free":
            reasons.append("free to use")
        elif cost_analysis["budget_friendly"]:
            reasons.append("budget-friendly pricing")
        
        # Performance reasoning
        if model.capabilities.accuracy_score > 0.9:
            reasons.append("high accuracy score")
        elif model.capabilities.speed_rating == "fast":
            reasons.append("fast response times")
        
        if not reasons:
            reasons.append("good general-purpose model")
        
        return f"Recommended because it {' and '.join(reasons)}."

    async def _filter_by_health(self, models: List[Model]) -> List[Model]:
        """Filter models by health status."""
        # Check if cache is stale or if any models don't have cached health status
        import time

        current_time = time.time()

        # Only consider cache stale if we've checked before and enough time has passed
        cache_is_stale = (
            self._last_health_check > 0
            and current_time - self._last_health_check > self._cache_ttl
        )

        # Check if any models are missing from cache
        missing_models = []
        for model in models:
            model_key = self._get_model_key(model)
            if model_key not in self._model_health_cache:
                missing_models.append(model)

        # Only refresh if cache is stale OR if we have missing models
        if cache_is_stale or missing_models:
            # If cache is stale, refresh all models; otherwise just refresh missing ones
            models_to_refresh = models if cache_is_stale else missing_models
            await self._refresh_health_cache(models_to_refresh)
            self._last_health_check = current_time

        # Return healthy models
        healthy = []
        for model in models:
            model_key = self._get_model_key(model)
            if self._model_health_cache.get(model_key, False):
                healthy.append(model)

        return healthy

    async def _refresh_health_cache(self, models: List[Model]) -> None:
        """Refresh health cache for models (with batch processing optimization)."""
        # Use batch processing for large numbers of models
        if len(models) > 5:
            try:
                health_results = await self.batch_processor.batch_health_check(
                    models, batch_size=10, timeout=15.0
                )
                
                # Update cache with batch results
                for model_key, is_healthy in health_results.items():
                    self._model_health_cache[model_key] = is_healthy
                
                return
            except Exception:
                # Fall back to individual checks if batch processing fails
                pass
        
        # Original individual check method for small numbers or fallback
        tasks = []
        for model in models:
            model_key = self._get_model_key(model)
            tasks.append(self._check_model_health(model_key, model))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _check_model_health(self, model_key: str, model: Model) -> None:
        """Check health of a single model."""
        try:
            # Add timeout to individual health checks
            is_healthy = await asyncio.wait_for(
                model.health_check(), timeout=15.0  # 15 second timeout per model
            )
            self._model_health_cache[model_key] = is_healthy
        except Exception:
            self._model_health_cache[model_key] = False

    def update_model_performance(
        self, model: Model, success: bool, latency: float = 0.0, cost: float = 0.0
    ) -> None:
        """
        Update model performance metrics.

        Args:
            model: Model that was used
            success: Whether the operation was successful
            latency: Operation latency in seconds
            cost: Operation cost in USD
        """
        model_key = self._get_model_key(model)

        # Update bandit algorithm
        reward = self._calculate_reward(success, latency, cost)
        self.model_selector.update_reward(model_key, reward)

        # Update model metrics
        self._update_model_metrics(model, success, latency, cost)

    def _calculate_reward(self, success: bool, latency: float, cost: float) -> float:
        """Calculate reward for bandit algorithm."""
        if not success:
            return 0.0

        # Reward formula: base reward - latency penalty - cost penalty
        base_reward = 1.0
        latency_penalty = min(latency / 10.0, 0.5)  # Cap at 0.5
        cost_penalty = min(cost * 100, 0.3)  # Cap at 0.3

        return max(base_reward - latency_penalty - cost_penalty, 0.1)

    def _update_model_metrics(
        self, model: Model, success: bool, latency: float, cost: float
    ) -> None:
        """Update model metrics with new data."""
        metrics = model.metrics

        # Update success rate (simple exponential moving average)
        alpha = 0.1
        metrics.success_rate = (
            alpha * (1.0 if success else 0.0) + (1 - alpha) * metrics.success_rate
        )

        # Update latency (only if successful)
        if success and latency > 0:
            metrics.latency_p50 = alpha * latency + (1 - alpha) * metrics.latency_p50

        # Update cost per token (simple average)
        if cost > 0:
            metrics.cost_per_token = alpha * cost + (1 - alpha) * metrics.cost_per_token

    def _get_model_key(self, model: Model) -> str:
        """Get unique key for model."""
        return f"{model.provider}:{model.name}"

    def get_model_statistics(self) -> Dict[str, Any]:
        """
        Get registry statistics.

        Returns:
            Dictionary of statistics
        """
        stats = {
            "total_models": len(self.models),
            "providers": len(self.list_providers()),
            "healthy_models": sum(1 for h in self._model_health_cache.values() if h),
            "selection_stats": self.model_selector.get_statistics(),
        }

        # Provider breakdown
        provider_counts = {}
        for key in self.models.keys():
            if ":" in key:
                provider = key.split(":")[0]
                provider_counts[provider] = provider_counts.get(provider, 0) + 1

        stats["provider_breakdown"] = provider_counts

        return stats

    def reset_statistics(self) -> None:
        """Reset all performance statistics."""
        self.model_selector.reset_statistics()
        self._model_health_cache.clear()
        self._last_health_check = 0

    # Memory optimization methods
    def _on_memory_warning(self, profile) -> None:
        """Handle memory warning by reducing cache sizes and cleaning up."""
        if not self._memory_optimization_enabled:
            return
        
        # Reduce optimizer cache sizes
        self.optimizer.set_cache_ttl(150)  # Reduce from 300 to 150 seconds
        
        # Clear some caches
        self.optimizer._cleanup_selection_cache()
        self.optimizer._cleanup_capability_cache()
        
        # Clear health cache (will be rebuilt as needed)
        self._model_health_cache.clear()
        
        print(f"Memory warning: {profile.memory_percent:.1f}% - optimized caches")

    def _on_memory_critical(self, profile) -> None:
        """Handle critical memory by aggressive cleanup."""
        if not self._memory_optimization_enabled:
            return
        
        # Aggressive cache cleanup
        self.optimizer.clear_caches()
        self._model_health_cache.clear()
        
        # Force garbage collection
        self.memory_monitor.force_garbage_collection()
        
        # Reduce cache TTL further
        self.optimizer.set_cache_ttl(60)  # Very short TTL
        
        print(f"Critical memory: {profile.memory_percent:.1f}% - aggressive cleanup")

    def optimize_memory_usage(self, target_memory_mb: Optional[float] = None) -> Dict[str, Any]:
        """
        Optimize memory usage of the registry.
        
        Args:
            target_memory_mb: Target memory usage in MB (optional)
            
        Returns:
            Optimization results
        """
        if not self._memory_optimization_enabled:
            return {'error': 'Memory optimization not enabled'}
        
        # Check current memory status
        profile = self.memory_monitor.check_memory()
        
        # Get memory optimization suggestions
        suggestions = self.memory_monitor.suggest_optimizations()
        
        # Perform registry-specific optimization
        registry_optimization = optimize_model_registry_memory(self, target_memory_mb or 100.0)
        
        # Clear caches
        cache_stats = {
            'selection_cache_cleared': len(self.optimizer.selection_cache),
            'capability_cache_cleared': len(self.optimizer.capability_cache),
            'health_cache_cleared': len(self._model_health_cache)
        }
        
        self.optimizer.clear_caches()
        self._model_health_cache.clear()
        
        # Force garbage collection
        gc_stats = self.memory_monitor.force_garbage_collection()
        
        # Get updated profile
        updated_profile = self.memory_monitor.check_memory()
        
        return {
            'before_optimization': profile,
            'after_optimization': updated_profile,
            'memory_freed_mb': profile.process_memory_mb - updated_profile.process_memory_mb,
            'cache_cleanup': cache_stats,
            'gc_collected': gc_stats,
            'registry_analysis': registry_optimization,
            'suggestions': suggestions
        }

    def get_memory_report(self) -> Dict[str, Any]:
        """
        Get comprehensive memory usage report.
        
        Returns:
            Memory usage report
        """
        if not self._memory_optimization_enabled:
            return {'error': 'Memory optimization not enabled'}
        
        # Get current memory profile
        profile = self.memory_monitor.check_memory()
        
        # Get memory trend
        trend = self.memory_monitor.get_memory_trend()
        
        # Get optimization suggestions
        suggestions = self.memory_monitor.suggest_optimizations()
        
        # Get optimizer statistics
        optimizer_stats = self.optimizer.get_statistics()
        
        # Calculate registry-specific metrics
        model_count = len(self.models)
        avg_model_size = sum(
            self._estimate_model_memory_usage(model) 
            for model in self.models.values()
        ) / max(model_count, 1)
        
        return {
            'memory_profile': profile,
            'memory_trend': trend,
            'optimization_suggestions': suggestions,
            'optimizer_statistics': optimizer_stats,
            'registry_metrics': {
                'total_models': model_count,
                'average_model_size_mb': avg_model_size,
                'health_cache_size': len(self._model_health_cache),
                'memory_optimization_enabled': self._memory_optimization_enabled
            }
        }

    def _estimate_model_memory_usage(self, model: Model) -> float:
        """Estimate memory usage of a model in MB."""
        from .memory_optimization import estimate_model_memory_usage
        return estimate_model_memory_usage(model)

    def set_memory_thresholds(self, warning: float = 80.0, critical: float = 90.0) -> None:
        """
        Set memory warning and critical thresholds.
        
        Args:
            warning: Warning threshold percentage
            critical: Critical threshold percentage
        """
        if self._memory_optimization_enabled and self.memory_monitor:
            self.memory_monitor.warning_threshold = warning
            self.memory_monitor.critical_threshold = critical

    def enable_memory_monitoring(self, warning_threshold: float = 80.0) -> None:
        """
        Enable memory monitoring if it was disabled.
        
        Args:
            warning_threshold: Memory warning threshold
        """
        if not self._memory_optimization_enabled:
            self.memory_monitor = MemoryMonitor(warning_threshold)
            self._memory_optimization_enabled = True
            self.memory_monitor.add_callback('warning', self._on_memory_warning)
            self.memory_monitor.add_callback('critical', self._on_memory_critical)

    def disable_memory_monitoring(self) -> None:
        """Disable memory monitoring."""
        self._memory_optimization_enabled = False
        self.memory_monitor = None

    # Advanced caching methods
    def _hash_requirements(self, requirements: Dict[str, Any]) -> str:
        """Generate hash for requirements dictionary for caching."""
        import json
        import hashlib
        
        # Create consistent string representation
        req_str = json.dumps(requirements, sort_keys=True)
        return hashlib.md5(req_str.encode()).hexdigest()
    
    async def _is_model_healthy(self, model: Model) -> bool:
        """Check if a specific model is healthy."""
        try:
            return await model.health_check()
        except Exception:
            return False
    
    def detect_model_capabilities(self, model: Model) -> Dict[str, Any]:
        """
        Detect and analyze model capabilities with advanced caching.
        
        Args:
            model: Model to analyze
            
        Returns:
            Capability analysis dictionary
        """
        model_key = self._get_model_key(model)
        
        # Check advanced cache first
        if self._advanced_caching_enabled and self.cache_manager:
            cached_analysis = self.cache_manager.get_cached_capability_analysis(model_key)
            if cached_analysis:
                return cached_analysis
        
        # Perform capability analysis
        analysis = self._perform_capability_analysis(model)
        
        # Cache the result
        if self._advanced_caching_enabled and self.cache_manager:
            self.cache_manager.cache_capability_analysis(model_key, analysis)
        
        return analysis
    
    def _perform_capability_analysis(self, model: Model) -> Dict[str, Any]:
        """Perform the actual capability analysis."""
        # Enhanced capability analysis based on Issue 194 features
        analysis = {
            "basic_capabilities": {},
            "advanced_capabilities": {},
            "performance_metrics": {},
            "expertise_analysis": {},
            "cost_analysis": {},
            "suitability_scores": {}
        }
        
        # Basic capabilities
        capabilities = model.capabilities
        analysis["basic_capabilities"] = {
            "supports_function_calling": capabilities.supports_function_calling,
            "supports_structured_output": capabilities.supports_structured_output,
            "vision_capable": capabilities.vision_capable,
            "code_specialized": capabilities.code_specialized,
            "context_window": capabilities.context_window,
            "max_tokens": capabilities.max_tokens,
            "supported_tasks": capabilities.supported_tasks,
            "languages": capabilities.languages
        }
        
        # Advanced capabilities based on Issue 194 enhancements
        expertise = getattr(model, '_expertise', ['general'])
        size_billions = getattr(model, '_size_billions', 1.0)
        
        analysis["advanced_capabilities"] = {
            "expertise_areas": expertise,
            "model_size_billions": size_billions,
            "parameter_efficiency": self._calculate_parameter_efficiency(model),
            "specialized_domains": self._identify_specialized_domains(expertise)
        }
        
        # Performance metrics
        metrics = model.metrics
        analysis["performance_metrics"] = {
            "latency_p50": metrics.latency_p50,
            "latency_p95": metrics.latency_p95,
            "throughput": metrics.throughput,
            "accuracy": metrics.accuracy,
            "success_rate": metrics.success_rate,
            "cost_per_token": metrics.cost_per_token
        }
        
        # Expertise analysis based on Issue 194 hierarchy
        analysis["expertise_analysis"] = {
            "level": self._determine_expertise_level(expertise, size_billions),
            "confidence": self._calculate_expertise_confidence(model),
            "specialized_score": self._calculate_specialization_score(expertise)
        }
        
        # Cost analysis
        cost = model.cost
        analysis["cost_analysis"] = {
            "type": "free" if cost.is_free else "paid",
            "cost_per_1k_avg": (cost.input_cost_per_1k_tokens + cost.output_cost_per_1k_tokens) / 2 if not cost.is_free else 0.0,
            "budget_friendly": cost.is_free or analysis["cost_analysis"].get("cost_per_1k_avg", 0) < 0.01,
            "efficiency_score": cost.get_cost_efficiency_score(metrics.accuracy)
        }
        
        # Suitability scores for different use cases
        analysis["suitability_scores"] = {
            "general": self._calculate_general_suitability(model),
            "code": self._calculate_code_suitability(model),
            "analysis": self._calculate_analysis_suitability(model),
            "creative": self._calculate_creative_suitability(model),
            "speed_critical": self._calculate_speed_suitability(model),
            "budget_constrained": 1.0 if cost.is_free else max(0.0, 1.0 - analysis["cost_analysis"]["cost_per_1k_avg"] * 100),
            "accuracy_critical": metrics.accuracy,
            "high_volume": min(1.0, metrics.throughput / 50.0)  # Normalize around 50 tokens/sec
        }
        
        return analysis
    
    def _determine_expertise_level(self, expertise: List[str], size_billions: float) -> str:
        """Determine expertise level based on Issue 194 hierarchy."""
        if any(e in ["analysis", "research"] for e in expertise) or size_billions > 100:
            return "very-high"
        elif any(e in ["code", "reasoning"] for e in expertise) or size_billions > 10:
            return "high"
        elif any(e in ["fast", "compact"] for e in expertise) or size_billions < 3:
            return "low"
        else:
            return "medium"
    
    def _calculate_parameter_efficiency(self, model: Model) -> float:
        """Calculate parameter efficiency score."""
        size_billions = getattr(model, '_size_billions', 1.0)
        accuracy = model.metrics.accuracy
        
        # Higher accuracy per parameter = better efficiency
        return accuracy / (size_billions ** 0.5)  # Square root scaling
    
    def _identify_specialized_domains(self, expertise: List[str]) -> List[str]:
        """Identify specialized domains from expertise."""
        domain_mapping = {
            "code": ["programming", "software_development", "debugging"],
            "analysis": ["data_analysis", "research", "reasoning"],
            "creative": ["writing", "storytelling", "brainstorming"],
            "reasoning": ["logic", "problem_solving", "mathematics"],
            "fast": ["real_time", "interactive", "quick_response"]
        }
        
        domains = []
        for exp in expertise:
            if exp in domain_mapping:
                domains.extend(domain_mapping[exp])
        
        return list(set(domains))
    
    def _calculate_expertise_confidence(self, model: Model) -> float:
        """Calculate confidence in expertise assessment."""
        # Based on metrics quality and experience
        metrics = model.metrics
        
        # More attempts = higher confidence in metrics
        attempts_factor = min(1.0, getattr(metrics, 'attempts', 10) / 50.0)
        
        # Higher success rate = higher confidence
        success_factor = metrics.success_rate
        
        return (attempts_factor + success_factor) / 2
    
    def _calculate_specialization_score(self, expertise: List[str]) -> float:
        """Calculate how specialized vs generalist the model is."""
        if not expertise:
            return 0.0
        
        # More specific expertise areas = higher specialization
        specialized_terms = ["code", "analysis", "research", "reasoning", "creative"]
        general_terms = ["general", "chat", "assistant"]
        
        specialized_count = sum(1 for exp in expertise if exp in specialized_terms)
        general_count = sum(1 for exp in expertise if exp in general_terms)
        
        if specialized_count + general_count == 0:
            return 0.5  # Unknown
        
        return specialized_count / (specialized_count + general_count)
    
    def _calculate_general_suitability(self, model: Model) -> float:
        """Calculate suitability for general tasks."""
        expertise = getattr(model, '_expertise', ['general'])
        general_score = 1.0 if "general" in expertise else 0.7
        
        # Balance of size and performance
        size_billions = getattr(model, '_size_billions', 1.0)
        size_score = 1.0 - abs(size_billions - 7.0) / 20.0  # Optimal around 7B
        size_score = max(0.3, min(1.0, size_score))
        
        return (general_score + size_score + model.metrics.accuracy) / 3
    
    def _calculate_code_suitability(self, model: Model) -> float:
        """Calculate suitability for code tasks."""
        expertise = getattr(model, '_expertise', [])
        code_score = 1.0 if "code" in expertise else 0.3
        
        # Code models benefit from larger size and high accuracy
        size_billions = getattr(model, '_size_billions', 1.0)
        size_score = min(1.0, size_billions / 10.0)  # Better with more parameters
        
        return (code_score + size_score + model.metrics.accuracy) / 3
    
    def _calculate_analysis_suitability(self, model: Model) -> float:
        """Calculate suitability for analysis tasks."""
        expertise = getattr(model, '_expertise', [])
        analysis_score = 1.0 if any(e in ["analysis", "research", "reasoning"] for e in expertise) else 0.4
        
        # Analysis benefits from larger models and high accuracy
        size_billions = getattr(model, '_size_billions', 1.0)
        size_score = min(1.0, size_billions / 20.0)  # Very large models preferred
        
        return (analysis_score + size_score + model.metrics.accuracy) / 3
    
    def _calculate_creative_suitability(self, model: Model) -> float:
        """Calculate suitability for creative tasks."""
        expertise = getattr(model, '_expertise', [])
        creative_score = 1.0 if "creative" in expertise else 0.6  # Most models can be creative
        
        # Creative tasks benefit from medium-large models
        size_billions = getattr(model, '_size_billions', 1.0)
        size_score = 1.0 - abs(size_billions - 15.0) / 30.0  # Optimal around 15B
        size_score = max(0.3, min(1.0, size_score))
        
        return (creative_score + size_score + model.metrics.accuracy) / 3
    
    def _calculate_speed_suitability(self, model: Model) -> float:
        """Calculate suitability for speed-critical tasks."""
        expertise = getattr(model, '_expertise', [])
        speed_score = 1.0 if "fast" in expertise else 0.5
        
        # Speed is inversely related to size but directly to throughput
        size_billions = getattr(model, '_size_billions', 1.0)
        size_penalty = max(0.2, 1.0 - size_billions / 20.0)  # Smaller is faster
        
        throughput_score = min(1.0, model.metrics.throughput / 100.0)  # Normalize to 100 tok/sec
        
        return (speed_score + size_penalty + throughput_score) / 3
    
    def start_background_maintenance(self) -> None:
        """Start background cache maintenance task."""
        if not self._advanced_caching_enabled or not self.cache_manager:
            return
        
        if self._background_maintenance_task is None:
            import asyncio
            self._background_maintenance_task = asyncio.create_task(
                background_cache_maintenance(self.cache_manager, interval=300)
            )
    
    def stop_background_maintenance(self) -> None:
        """Stop background cache maintenance task."""
        if self._background_maintenance_task:
            self._background_maintenance_task.cancel()
            self._background_maintenance_task = None
    
    def get_caching_statistics(self) -> Dict[str, Any]:
        """Get comprehensive caching statistics."""
        if not self._advanced_caching_enabled or not self.cache_manager:
            return {'error': 'Advanced caching not enabled'}
        
        return self.cache_manager.get_comprehensive_statistics()
    
    def warm_caches(self) -> Dict[str, int]:
        """Manually trigger cache warming."""
        if not self._advanced_caching_enabled or not self.cache_manager:
            return {'error': 'Advanced caching not enabled'}
        
        return self.cache_manager.warm_caches()
    
    def clear_all_caches(self) -> None:
        """Clear all caches."""
        if self._advanced_caching_enabled and self.cache_manager:
            self.cache_manager.clear_all_caches()
        
        # Also clear legacy caches
        self.optimizer.clear_caches()
        self._model_health_cache.clear()
    
    def enable_cache_warming(self) -> None:
        """Enable predictive cache warming."""
        if self._advanced_caching_enabled and self.cache_manager:
            self.cache_manager.enable_warming()
    
    def disable_cache_warming(self) -> None:
        """Disable predictive cache warming."""
        if self._advanced_caching_enabled and self.cache_manager:
            self.cache_manager.disable_warming()


class UCBModelSelector:
    """
    Upper Confidence Bound algorithm for model selection.

    Balances exploration and exploitation when selecting models
    based on their performance history.
    """

    def __init__(self, exploration_factor: float = 2.0) -> None:
        """
        Initialize UCB selector.

        Args:
            exploration_factor: Exploration parameter (higher = more exploration)
        """
        self.exploration_factor = exploration_factor
        self.model_stats: Dict[str, Dict[str, float]] = {}
        self.total_attempts = 0
        self._pending_attempts: set = set()  # Track models with pending attempts

    def initialize_model(self, model_key: str, metrics: ModelMetrics) -> None:
        """
        Initialize model in selector.

        Args:
            model_key: Unique model key
            metrics: Model performance metrics
        """
        self.model_stats[model_key] = {
            "attempts": 0,
            "successes": 0,
            "total_reward": 0.0,
            "average_reward": metrics.success_rate,
        }

    def select(self, model_keys: List[str], context: Dict[str, Any]) -> str:
        """
        Select model using UCB algorithm.

        Args:
            model_keys: List of available model keys
            context: Selection context

        Returns:
            Selected model key

        Raises:
            ValueError: If no models available
        """
        if not model_keys:
            raise ValueError("No models available for selection")

        # Initialize any new models
        for key in model_keys:
            if key not in self.model_stats:
                self.model_stats[key] = {
                    "attempts": 0,
                    "successes": 0,
                    "total_reward": 0.0,
                    "average_reward": 0.5,  # Neutral starting point
                }

        # Calculate UCB scores
        scores = {}

        for key in model_keys:
            stats = self.model_stats[key]

            if stats["attempts"] == 0:
                # Explore untried models first
                scores[key] = float("inf")
            else:
                # UCB formula: average_reward + exploration_bonus
                average_reward = stats["average_reward"]
                exploration_bonus = self.exploration_factor * math.sqrt(
                    math.log(self.total_attempts + 1) / stats["attempts"]
                )
                scores[key] = average_reward + exploration_bonus

        # Select model with highest score
        selected_key = max(scores, key=scores.get)

        # Update attempt count immediately (as expected by tests)
        self.model_stats[selected_key]["attempts"] += 1
        self.total_attempts += 1

        # Mark as having a pending attempt (to avoid double counting in update_reward)
        self._pending_attempts.add(selected_key)

        return selected_key

    def update_reward(self, model_key: str, reward: float) -> None:
        """
        Update model statistics after execution.

        Args:
            model_key: Model key
            reward: Reward value (0-1)
        """
        if model_key not in self.model_stats:
            return

        stats = self.model_stats[model_key]

        # Only increment attempts if this wasn't already counted by select()
        if model_key not in self._pending_attempts:
            stats["attempts"] += 1
            self.total_attempts += 1
        else:
            # Remove from pending attempts (already counted)
            self._pending_attempts.remove(model_key)

        # Update statistics
        stats["total_reward"] += reward
        if reward > 0:
            stats["successes"] += 1

        # Update average reward
        stats["average_reward"] = stats["total_reward"] / stats["attempts"]

    def remove_model(self, model_key: str) -> None:
        """
        Remove model from selector.

        Args:
            model_key: Model key to remove
        """
        if model_key in self.model_stats:
            del self.model_stats[model_key]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get selection statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "total_attempts": self.total_attempts,
            "models_tracked": len(self.model_stats),
            "model_performance": {
                key: {
                    "attempts": stats["attempts"],
                    "successes": stats["successes"],
                    "success_rate": stats["successes"] / max(stats["attempts"], 1),
                    "average_reward": stats["average_reward"],
                }
                for key, stats in self.model_stats.items()
            },
        }

    def reset_statistics(self) -> None:
        """Reset all selection statistics."""
        for stats in self.model_stats.values():
            stats["attempts"] = 0
            stats["successes"] = 0
            stats["total_reward"] = 0.0
            stats["average_reward"] = 0.5

        self.total_attempts = 0
        self._pending_attempts.clear()

    def get_model_confidence(self, model_key: str) -> float:
        """
        Get confidence score for a model.

        Args:
            model_key: Model key

        Returns:
            Confidence score (0-1)
        """
        if model_key not in self.model_stats:
            return 0.0

        stats = self.model_stats[model_key]

        if stats["attempts"] == 0:
            return 0.0

        # Confidence increases with more attempts and higher success rate
        attempt_factor = min(stats["attempts"] / 10.0, 1.0)  # Cap at 10 attempts
        success_factor = stats["average_reward"]

        return attempt_factor * success_factor
