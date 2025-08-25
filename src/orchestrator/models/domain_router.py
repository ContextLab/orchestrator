"""Domain-specific model routing with RouteLLM integration."""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from ..core.model import Model
from .model_registry import ModelRegistry
from .model_selector import ModelSelector, ModelSelectionCriteria
from .routellm_integration import (
    RouteLLMConfig,
    RoutingDecision,
    RoutingMetrics,
    CostTracker,
    FeatureFlags,
    RouterType,
)

logger = logging.getLogger(__name__)


@dataclass
class DomainConfig:
    """Configuration for a specific domain."""

    name: str  # Domain name (e.g., "medical", "legal", "creative")
    keywords: List[str] = field(
        default_factory=list
    )  # Keywords that indicate this domain
    patterns: List[str] = field(default_factory=list)  # Regex patterns to match
    preferred_models: List[str] = field(default_factory=list)  # Preferred model IDs
    required_capabilities: List[str] = field(
        default_factory=list
    )  # Required capabilities
    required_certifications: List[str] = field(
        default_factory=list
    )  # e.g., ["HIPAA", "SOC2"]
    min_accuracy_score: float = 0.0  # Minimum accuracy requirement

    def matches_text(self, text: str) -> float:
        """
        Check if text matches this domain.

        Args:
            text: Text to check

        Returns:
            Confidence score (0-1) that text belongs to this domain
        """
        text_lower = text.lower()
        score = 0.0

        # Check keywords
        keyword_matches = 0
        for keyword in self.keywords:
            if keyword.lower() in text_lower:
                keyword_matches += 1

        if self.keywords:
            keyword_score = keyword_matches / len(self.keywords)
            score = max(score, keyword_score)

        # Check patterns
        pattern_matches = 0
        for pattern in self.patterns:
            try:
                if re.search(pattern, text, re.IGNORECASE):
                    pattern_matches += 1
            except re.error:
                continue

        if self.patterns:
            pattern_score = pattern_matches / len(self.patterns)
            score = max(score, pattern_score * 1.2)  # Patterns are more specific

        return min(score, 1.0)


class DomainRouter:
    """
    Routes model selection based on content domain with RouteLLM integration.

    Analyzes input text to determine domain and selects appropriate models
    with domain expertise. Optionally uses RouteLLM for intelligent routing
    and cost optimization while maintaining full backward compatibility.
    """

    def __init__(
        self, 
        registry: ModelRegistry,
        routellm_config: Optional[RouteLLMConfig] = None,
        feature_flags: Optional[FeatureFlags] = None,
    ):
        """
        Initialize domain router with optional RouteLLM integration.

        Args:
            registry: Model registry to use
            routellm_config: Optional RouteLLM configuration for intelligent routing
            feature_flags: Optional feature flags for gradual rollout control
        """
        self.registry = registry
        self.selector = ModelSelector(registry)
        self.domains: Dict[str, DomainConfig] = {}

        # RouteLLM integration components
        self.routellm_config = routellm_config or RouteLLMConfig()
        self.feature_flags = feature_flags or FeatureFlags()
        self.cost_tracker = (
            CostTracker(self.routellm_config.metrics_retention_days)
            if self.routellm_config.cost_tracking_enabled
            else None
        )
        
        # RouteLLM controller (lazy initialization)
        self._routellm_controller = None
        self._controller_initialized = False
        self._controller_error: Optional[Exception] = None

        # Initialize default domains
        self._init_default_domains()

    def _init_default_domains(self) -> None:
        """Initialize default domain configurations."""

        # Medical domain
        medical = DomainConfig(
            name="medical",
            keywords=[
                "patient",
                "diagnosis",
                "treatment",
                "symptom",
                "medication",
                "clinical",
                "medical",
                "health",
                "disease",
                "therapy",
                "prescription",
                "dosage",
                "condition",
                "physician",
                "hospital",
            ],
            patterns=[
                r"\b(diagnos\w+|treat\w+|medicat\w+|clinical|medical)\b",
                r"\b(patient|physician|doctor|nurse|hospital|clinic)\b",
                r"\b(symptom|disease|condition|disorder|syndrome)\b",
            ],
            preferred_models=["gpt-4", "claude-3-opus"],
            required_capabilities=[],
            required_certifications=["HIPAA"],
            min_accuracy_score=0.8,
        )

        # Legal domain
        legal = DomainConfig(
            name="legal",
            keywords=[
                "legal",
                "law",
                "contract",
                "agreement",
                "liability",
                "regulation",
                "compliance",
                "court",
                "attorney",
                "clause",
                "jurisdiction",
                "statute",
                "litigation",
                "precedent",
            ],
            patterns=[
                r"\b(legal|law|contract|agreement|liabilit\w+)\b",
                r"\b(regulat\w+|complian\w+|jurisdict\w+)\b",
                r"\b(court|attorney|lawyer|counsel)\b",
                r"\bsection\s+\d+\.\d+\b",  # Legal section references
            ],
            preferred_models=["gpt-4", "claude-3-opus"],
            required_capabilities=[],
            min_accuracy_score=0.8,
        )

        # Creative domain
        creative = DomainConfig(
            name="creative",
            keywords=[
                "creative",
                "story",
                "poem",
                "art",
                "design",
                "narrative",
                "character",
                "plot",
                "scene",
                "dialogue",
                "imagination",
                "artistic",
                "aesthetic",
                "composition",
            ],
            patterns=[
                r"\b(creat\w+|story|poem|art\w+|design)\b",
                r"\b(write|compose|craft|imagine)\s+(a|an|the)\s+\w+",
                r"\b(character|plot|scene|dialogue|narrative)\b",
            ],
            preferred_models=["claude-3-opus", "gpt-4"],
            required_capabilities=[],
            min_accuracy_score=0.85,
        )

        # Technical/Engineering domain
        technical = DomainConfig(
            name="technical",
            keywords=[
                "technical",
                "engineering",
                "system",
                "architecture",
                "implementation",
                "algorithm",
                "optimization",
                "performance",
                "infrastructure",
                "deployment",
                "integration",
                "api",
            ],
            patterns=[
                r"\b(technical|engineer\w+|architect\w+)\b",
                r"\b(implement\w+|algorithm|optimiz\w+)\b",
                r"\b(system|infrastructure|deployment|integration)\b",
                r"\b(API|SDK|framework|library|database)\b",
            ],
            preferred_models=["gpt-4", "claude-3-opus"],
            required_capabilities=[],
            min_accuracy_score=0.8,
        )

        # Scientific/Research domain
        scientific = DomainConfig(
            name="scientific",
            keywords=[
                "scientific",
                "research",
                "hypothesis",
                "experiment",
                "analysis",
                "data",
                "methodology",
                "results",
                "conclusion",
                "peer-review",
                "publication",
                "citation",
                "theory",
            ],
            patterns=[
                r"\b(scientif\w+|research|hypothesis|experiment)\b",
                r"\b(analy\w+|data|methodolog\w+|result\w+)\b",
                r"\b(peer[\s-]?review|publicat\w+|citat\w+)\b",
                r"\b(theory|theorem|proof|evidence)\b",
            ],
            preferred_models=["gpt-4", "claude-3-opus"],
            required_capabilities=[],
            min_accuracy_score=0.8,
        )

        # Financial domain
        financial = DomainConfig(
            name="financial",
            keywords=[
                "financial",
                "investment",
                "portfolio",
                "market",
                "trading",
                "risk",
                "return",
                "asset",
                "equity",
                "revenue",
                "profit",
                "loss",
                "accounting",
                "audit",
            ],
            patterns=[
                r"\b(financ\w+|invest\w+|portfolio|market)\b",
                r"\b(trad\w+|risk|return|asset|equity)\b",
                r"\b(revenue|profit|loss|accounting|audit)\b",
                r"\$\d+\.?\d*[KMB]?\b",  # Dollar amounts
            ],
            preferred_models=["gpt-4", "claude-3-opus"],
            required_capabilities=[],
            min_accuracy_score=0.8,
        )

        # Educational domain
        educational = DomainConfig(
            name="educational",
            keywords=[
                "educational",
                "learning",
                "teaching",
                "student",
                "curriculum",
                "lesson",
                "tutorial",
                "explain",
                "understand",
                "concept",
                "example",
                "practice",
            ],
            patterns=[
                r"\b(educat\w+|learn\w+|teach\w+|student)\b",
                r"\b(explain|understand|concept|example)\b",
                r"\b(lesson|tutorial|curriculum|course)\b",
                r"(?i)how\s+(to|do|does|can)\b",  # Educational questions
            ],
            preferred_models=["gpt-4", "claude-3-opus", "gpt-3.5-turbo"],
            required_capabilities=[],
            min_accuracy_score=0.85,
        )

        # Add domains to router
        self.domains = {
            "medical": medical,
            "legal": legal,
            "creative": creative,
            "technical": technical,
            "scientific": scientific,
            "financial": financial,
            "educational": educational,
        }

    def _initialize_routellm_controller(self) -> None:
        """Initialize RouteLLM controller with error handling."""
        if self._controller_initialized:
            return
            
        try:
            # Only import RouteLLM when actually needed to avoid import errors
            from routellm.controller import Controller
            
            self._routellm_controller = Controller(
                routers=[self.routellm_config.router_type.value],
                strong_model=self.routellm_config.strong_model,
                weak_model=self.routellm_config.weak_model,
            )
            
            logger.info(f"RouteLLM controller initialized with router: {self.routellm_config.router_type.value}")
            
        except ImportError as e:
            self._controller_error = e
            logger.warning(f"RouteLLM not available, falling back to domain selector: {e}")
        except Exception as e:
            self._controller_error = e
            logger.error(f"Failed to initialize RouteLLM controller: {e}")
        finally:
            self._controller_initialized = True

    async def _should_use_routellm(self, text: str, domains: List[Tuple[str, float]]) -> bool:
        """
        Determine if RouteLLM should be used for this request.
        
        Args:
            text: Input text to analyze
            domains: Detected domains with confidence scores
            
        Returns:
            True if RouteLLM should be used, False otherwise
        """
        # Check global feature flag
        if not self.feature_flags.is_enabled(FeatureFlags.ROUTELLM_ENABLED):
            return False
            
        # Check if RouteLLM is configured and available
        if not self.routellm_config.enabled:
            return False
            
        # Check domain-specific feature flags
        if domains and self.routellm_config.domain_specific_routing:
            primary_domain = domains[0][0]  # Highest confidence domain
            if not self.feature_flags.is_domain_enabled(primary_domain):
                return False
        
        # Initialize controller if needed
        self._initialize_routellm_controller()
        
        # Check if controller is available
        return self._routellm_controller is not None and self._controller_error is None

    async def _route_with_routellm(
        self, text: str, domains: List[Tuple[str, float]]
    ) -> RoutingDecision:
        """
        Use RouteLLM to make routing decisions.
        
        Args:
            text: Input text for routing
            domains: Detected domains with confidence scores
            
        Returns:
            RoutingDecision with recommendation
        """
        start_time = time.time()
        
        try:
            if not self._routellm_controller:
                return RoutingDecision(
                    should_use_routellm=False,
                    fallback_reason="controller_not_available"
                )
            
            # Build routing context from domains
            routing_context = self._build_routing_context(text, domains)
            
            # Make RouteLLM API call
            response = await self._make_routellm_request(text, routing_context)
            
            # Extract routing decision from response
            decision = self._parse_routellm_response(response, domains)
            decision.domains = [domain for domain, _ in domains]
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            logger.debug(f"RouteLLM routing completed in {latency_ms:.2f}ms")
            
            return decision
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.warning(f"RouteLLM routing failed after {latency_ms:.2f}ms: {e}")
            
            return RoutingDecision(
                should_use_routellm=False,
                fallback_reason=f"routing_error: {str(e)}"
            )

    def _build_routing_context(
        self, text: str, domains: List[Tuple[str, float]]
    ) -> Dict[str, Any]:
        """Build context for RouteLLM routing decision."""
        context = {
            "text_length": len(text),
            "detected_domains": [{"domain": d, "confidence": c} for d, c in domains],
            "primary_domain": domains[0][0] if domains else None,
        }
        
        # Add domain-specific context
        if domains:
            primary_domain = domains[0][0]
            domain_override = self.routellm_config.get_domain_override(primary_domain)
            if domain_override:
                context.update(domain_override)
        
        return context

    async def _make_routellm_request(
        self, text: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Make request to RouteLLM controller.
        
        This is a placeholder for the actual RouteLLM API call.
        The real implementation would use the RouteLLM controller
        to get routing recommendations.
        """
        # For now, simulate a RouteLLM decision based on text complexity
        # In the real implementation, this would be:
        # response = await self._routellm_controller.chat.completions.create(
        #     model=self.routellm_config.get_router_model_string(),
        #     messages=[{"role": "user", "content": text}]
        # )
        
        # Simulate routing decision based on text complexity
        complexity_score = self._estimate_text_complexity(text)
        
        if complexity_score > 0.7:  # Complex text - use strong model
            recommended_model = self.routellm_config.strong_model
            confidence = 0.9
        else:  # Simple text - use weak model
            recommended_model = self.routellm_config.weak_model
            confidence = 0.8
        
        return {
            "model_used": recommended_model,
            "confidence": confidence,
            "complexity_score": complexity_score,
            "reasoning": f"Text complexity: {complexity_score:.2f}",
        }

    def _estimate_text_complexity(self, text: str) -> float:
        """
        Estimate text complexity for routing decisions.
        
        This is a simple heuristic - in production, RouteLLM
        would use sophisticated models for this determination.
        """
        # Simple complexity heuristics
        word_count = len(text.split())
        sentence_count = len([s for s in text.split('.') if s.strip()])
        avg_word_length = sum(len(word) for word in text.split()) / max(word_count, 1)
        
        # Technical terms increase complexity
        technical_patterns = [
            r'\b\w+\(\)',  # Function calls
            r'\b[A-Z]{2,}',  # Acronyms
            r'\b\d+\.\d+',  # Version numbers
            r'[{}[\]()]',  # Code-like brackets
        ]
        
        technical_score = 0
        for pattern in technical_patterns:
            technical_score += len(re.findall(pattern, text))
        
        # Normalize scores
        length_score = min(word_count / 100, 1.0)  # 100+ words = complex
        structure_score = min(sentence_count / 10, 1.0)  # 10+ sentences = complex  
        vocab_score = min(avg_word_length / 8, 1.0)  # 8+ char avg = complex
        tech_score = min(technical_score / 20, 1.0)  # 20+ technical terms = complex
        
        # Weighted combination
        complexity = (
            length_score * 0.3 +
            structure_score * 0.2 +
            vocab_score * 0.2 +
            tech_score * 0.3
        )
        
        return min(complexity, 1.0)

    def _parse_routellm_response(
        self, response: Dict[str, Any], domains: List[Tuple[str, float]]
    ) -> RoutingDecision:
        """Parse RouteLLM response into a routing decision."""
        return RoutingDecision(
            should_use_routellm=True,
            recommended_model=response.get("model_used"),
            confidence=response.get("confidence", 0.0),
            estimated_cost=0.0,  # Would be calculated based on model costs
            reasoning=response.get("reasoning", ""),
        )

    async def _execute_routellm_routing(
        self, text: str, routing_decision: RoutingDecision
    ) -> Model:
        """
        Execute RouteLLM routing decision with fallback handling.
        
        Args:
            text: Input text
            routing_decision: RouteLLM routing decision
            
        Returns:
            Selected model
        """
        try:
            # Map RouteLLM model recommendation to our model registry
            model = await self._get_model_from_recommendation(
                routing_decision.recommended_model
            )
            
            # Validate model availability
            if not await self._validate_model_availability(model):
                raise ValueError(f"Recommended model {routing_decision.recommended_model} not available")
            
            # Track successful RouteLLM routing
            if self.cost_tracker:
                tracking_id = self.cost_tracker.track_routing_decision(
                    text=text,
                    domains=routing_decision.domains,
                    routing_method="routellm",
                    selected_model=f"{model.provider}:{model.name}",
                    estimated_cost=routing_decision.estimated_cost,
                    routing_confidence=routing_decision.confidence,
                    success=True,
                )
                logger.debug(f"Tracked RouteLLM routing: {tracking_id}")
            
            return model
            
        except Exception as e:
            logger.warning(f"RouteLLM routing execution failed: {e}")
            
            # Track failed RouteLLM attempt
            if self.cost_tracker:
                self.cost_tracker.track_routing_decision(
                    text=text,
                    domains=routing_decision.domains,
                    routing_method="routellm",
                    selected_model="",
                    estimated_cost=0.0,
                    success=False,
                    error_message=str(e),
                )
            
            # Fall back to traditional domain routing
            return await self._route_with_domain_selector(text, None, None)

    async def _get_model_from_recommendation(self, model_name: str) -> Model:
        """
        Get model instance from RouteLLM recommendation.
        
        Args:
            model_name: Model name from RouteLLM
            
        Returns:
            Model instance from registry
        """
        # Try to find exact match first
        for model_id, model in self.registry.models.items():
            if model_name in model_id or model.name == model_name:
                return model
        
        # Fall back to using model selector with preference
        criteria = ModelSelectionCriteria(
            preferred_models=[model_name],
            selection_strategy="balanced"
        )
        
        return await self.selector.select_model(criteria)

    async def _validate_model_availability(self, model: Model) -> bool:
        """
        Validate that a model is available for use.
        
        Args:
            model: Model to validate
            
        Returns:
            True if model is available
        """
        # Check if model has availability attribute
        if hasattr(model, 'is_available'):
            return getattr(model, 'is_available', True)
        
        # Assume available if no availability info
        return True

    def register_domain(self, domain: DomainConfig) -> None:
        """
        Register a custom domain configuration.

        Args:
            domain: Domain configuration to register
        """
        self.domains[domain.name] = domain

    def detect_domains(
        self, text: str, threshold: float = 0.3
    ) -> List[Tuple[str, float]]:
        """
        Detect domains in text.

        Args:
            text: Text to analyze
            threshold: Minimum confidence threshold

        Returns:
            List of (domain_name, confidence) tuples sorted by confidence
        """
        detections = []

        for domain_name, domain_config in self.domains.items():
            confidence = domain_config.matches_text(text)
            if confidence >= threshold:
                detections.append((domain_name, confidence))

        # Sort by confidence descending
        detections.sort(key=lambda x: x[1], reverse=True)
        return detections

    async def route_by_domain(
        self,
        text: str,
        base_criteria: Optional[ModelSelectionCriteria] = None,
        domain_override: Optional[str] = None,
    ) -> Model:
        """
        Select model based on detected domain with optional RouteLLM integration.

        This method maintains complete backward compatibility while optionally
        using RouteLLM for intelligent routing and cost optimization.

        Args:
            text: Text to analyze for domain
            base_criteria: Base selection criteria to extend
            domain_override: Force specific domain instead of detecting

        Returns:
            Selected model
        """
        # Detect domain if not overridden
        if domain_override:
            domains = [(domain_override, 1.0)]
        else:
            domains = self.detect_domains(text)

        # Try RouteLLM routing if enabled and available
        if await self._should_use_routellm(text, domains):
            try:
                routing_decision = await self._route_with_routellm(text, domains)
                
                if routing_decision.should_use_routellm:
                    logger.debug(f"Using RouteLLM routing: {routing_decision.reasoning}")
                    return await self._execute_routellm_routing(text, routing_decision)
                else:
                    logger.debug(f"RouteLLM declined routing: {routing_decision.fallback_reason}")
            
            except Exception as e:
                logger.warning(f"RouteLLM routing failed, falling back to domain selector: {e}")
                
                # Track the fallback
                if self.cost_tracker:
                    self.cost_tracker.track_routing_decision(
                        text=text,
                        domains=[d for d, _ in domains],
                        routing_method="routellm_fallback",
                        selected_model="",
                        estimated_cost=0.0,
                        success=False,
                        error_message=str(e),
                    )

        # Fall back to traditional domain-based routing
        return await self._route_with_domain_selector(text, domains, base_criteria)

    async def _route_with_domain_selector(
        self,
        text: str,
        domains: Optional[List[Tuple[str, float]]],
        base_criteria: Optional[ModelSelectionCriteria],
    ) -> Model:
        """
        Traditional domain-based routing using ModelSelector.
        
        This preserves the original routing logic for fallback and compatibility.
        """
        # Use detected domains or detect them now
        if domains is None:
            domains = self.detect_domains(text)

        # Start with base criteria or create new
        criteria = base_criteria or ModelSelectionCriteria()

        if domains:
            # Use highest confidence domain
            domain_name, confidence = domains[0]
            domain_config = self.domains.get(domain_name)

            if domain_config:
                # Update criteria based on domain
                criteria.required_domains = [domain_name]
                # Don't set preferred_models as it's too restrictive in testing
                # criteria.preferred_models.extend(domain_config.preferred_models)

                # Don't add domain-specific capabilities as hard requirements
                # They will be used for scoring instead

                if domain_config.min_accuracy_score > criteria.min_accuracy_score:
                    criteria.min_accuracy_score = domain_config.min_accuracy_score

        # Use model selector with domain-enhanced criteria
        selected_model = await self.selector.select_model(criteria)
        
        # Track traditional routing
        if self.cost_tracker:
            self.cost_tracker.track_routing_decision(
                text=text,
                domains=[d for d, _ in domains] if domains else [],
                routing_method="domain_selector",
                selected_model=f"{selected_model.provider}:{selected_model.name}",
                estimated_cost=0.0,  # Would calculate based on model costs
                success=True,
            )
        
        return selected_model

    def get_domain_info(self, domain_name: str) -> Optional[DomainConfig]:
        """
        Get information about a specific domain.

        Args:
            domain_name: Domain name

        Returns:
            Domain configuration or None
        """
        return self.domains.get(domain_name)

    def list_domains(self) -> List[str]:
        """
        List all registered domains.

        Returns:
            List of domain names
        """
        return list(self.domains.keys())

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for domain information.

        Args:
            text: Text to analyze

        Returns:
            Analysis results including detected domains
        """
        domains = self.detect_domains(text, threshold=0.1)

        analysis_result = {
            "text_length": len(text),
            "detected_domains": [
                {"domain": domain, "confidence": conf} for domain, conf in domains
            ],
            "primary_domain": domains[0][0] if domains else None,
            "multi_domain": len(domains) > 1,
        }
        
        # Add RouteLLM analysis if enabled
        if self.routellm_config.enabled and self.feature_flags.is_enabled(FeatureFlags.ROUTELLM_ENABLED):
            complexity_score = self._estimate_text_complexity(text)
            analysis_result.update({
                "routellm_enabled": True,
                "complexity_score": complexity_score,
                "recommended_routing": "strong_model" if complexity_score > 0.7 else "weak_model",
                "routing_confidence": min(0.8 + complexity_score * 0.2, 1.0),
            })
        
        return analysis_result

    # RouteLLM-specific methods for external access
    
    def get_routellm_config(self) -> RouteLLMConfig:
        """Get current RouteLLM configuration."""
        return self.routellm_config
    
    def update_routellm_config(self, config: RouteLLMConfig) -> None:
        """Update RouteLLM configuration."""
        self.routellm_config = config
        # Reset controller to pick up new config
        self._controller_initialized = False
        self._routellm_controller = None
        self._controller_error = None
        logger.info("RouteLLM configuration updated")
    
    def get_feature_flags(self) -> FeatureFlags:
        """Get current feature flags."""
        return self.feature_flags
    
    def update_feature_flags(self, flags: Dict[str, bool]) -> None:
        """Update feature flags."""
        self.feature_flags.update_flags(flags)
        logger.info(f"Feature flags updated: {flags}")
    
    def get_cost_savings_report(self, period_days: int = 30) -> Optional[Dict[str, Any]]:
        """
        Get cost savings report from RouteLLM usage.
        
        Args:
            period_days: Number of days to include in report
            
        Returns:
            Cost savings report or None if tracking disabled
        """
        if not self.cost_tracker:
            return None
            
        report = self.cost_tracker.get_cost_savings_report(period_days)
        return {
            "period_days": report.period_days,
            "total_requests": report.total_requests,
            "routellm_requests": report.routellm_requests,
            "traditional_requests": report.traditional_requests,
            "estimated_savings": report.estimated_savings,
            "savings_percentage": report.savings_percentage,
            "success_rate": report.success_rate,
            "average_latency_ms": report.average_routing_latency_ms,
            "average_quality_score": report.average_quality_score,
        }
    
    def get_routing_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of routing metrics."""
        if not self.cost_tracker:
            return {"error": "Cost tracking disabled"}
        
        return self.cost_tracker.get_metrics_summary()
    
    def is_routellm_enabled(self) -> bool:
        """Check if RouteLLM is currently enabled."""
        return (
            self.routellm_config.enabled
            and self.feature_flags.is_enabled(FeatureFlags.ROUTELLM_ENABLED)
        )
    
    def get_routellm_status(self) -> Dict[str, Any]:
        """Get detailed RouteLLM integration status."""
        return {
            "config_enabled": self.routellm_config.enabled,
            "feature_flag_enabled": self.feature_flags.is_enabled(FeatureFlags.ROUTELLM_ENABLED),
            "controller_initialized": self._controller_initialized,
            "controller_available": self._routellm_controller is not None,
            "controller_error": str(self._controller_error) if self._controller_error else None,
            "cost_tracking_enabled": self.cost_tracker is not None,
            "router_type": self.routellm_config.router_type.value,
            "strong_model": self.routellm_config.strong_model,
            "weak_model": self.routellm_config.weak_model,
            "threshold": self.routellm_config.threshold,
        }
