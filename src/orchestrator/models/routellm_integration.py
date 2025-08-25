"""RouteLLM integration module for intelligent model routing and cost optimization."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class RouterType(Enum):
    """Available RouteLLM router types."""
    
    MATRIX_FACTORIZATION = "mf"
    BERT_CLASSIFIER = "bert"
    CAUSAL_LLM = "causal_llm"
    SIMILARITY_WEIGHTED = "sw_ranking"
    RANDOM = "random"


@dataclass
class RouteLLMConfig:
    """Configuration for RouteLLM integration."""
    
    # Core routing settings
    enabled: bool = False
    router_type: RouterType = RouterType.MATRIX_FACTORIZATION
    threshold: float = 0.11593  # Default RouteLLM threshold
    
    # Model configuration
    strong_model: str = "gpt-4-1106-preview"
    weak_model: str = "gpt-3.5-turbo"
    
    # Fallback and reliability settings
    fallback_enabled: bool = True
    max_retry_attempts: int = 3
    timeout_seconds: float = 30.0
    
    # Cost optimization settings
    cost_tracking_enabled: bool = True
    cost_optimization_target: float = 0.5  # Target 50% cost reduction
    
    # Performance monitoring
    performance_monitoring_enabled: bool = True
    metrics_retention_days: int = 30
    
    # Domain-specific settings
    domain_specific_routing: bool = True
    domain_routing_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def get_router_model_string(self) -> str:
        """Get the router model string for RouteLLM API calls."""
        return f"router-{self.router_type.value}-{self.threshold}"
    
    def get_domain_override(self, domain: str) -> Optional[Dict[str, Any]]:
        """Get domain-specific routing overrides."""
        return self.domain_routing_overrides.get(domain)


@dataclass
class RoutingDecision:
    """Result of a RouteLLM routing decision."""
    
    should_use_routellm: bool
    recommended_model: Optional[str] = None
    confidence: float = 0.0
    estimated_cost: float = 0.0
    reasoning: str = ""
    domains: List[str] = field(default_factory=list)
    fallback_reason: Optional[str] = None
    
    @property
    def is_high_confidence(self) -> bool:
        """Check if this is a high-confidence routing decision."""
        return self.confidence >= 0.8
    
    @property
    def is_cost_effective(self) -> bool:
        """Check if this routing decision is cost-effective."""
        return self.estimated_cost > 0 and self.should_use_routellm


@dataclass
class RoutingMetrics:
    """Metrics for tracking routing decisions and performance."""
    
    # Request identification
    tracking_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Input characteristics
    input_text_length: int = 0
    detected_domains: List[str] = field(default_factory=list)
    domain_confidence: float = 0.0
    
    # Routing decision
    routing_method: str = ""  # 'routellm' or 'domain_selector'
    selected_model: str = ""
    routing_confidence: float = 0.0
    
    # Cost metrics
    estimated_cost: float = 0.0
    actual_cost: Optional[float] = None
    cost_savings_vs_baseline: Optional[float] = None
    
    # Performance metrics
    routing_latency_ms: float = 0.0
    model_response_time_ms: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    
    # Quality metrics
    response_quality_score: Optional[float] = None
    user_satisfaction: Optional[float] = None


@dataclass
class CostSavingsReport:
    """Report on cost savings achieved through RouteLLM routing."""
    
    period_days: int
    total_requests: int
    routellm_requests: int
    traditional_requests: int
    
    # Cost metrics
    total_estimated_cost: float
    routellm_estimated_cost: float
    traditional_estimated_cost: float
    estimated_savings: float
    savings_percentage: float
    
    # Performance metrics
    average_routing_latency_ms: float
    success_rate: float
    
    # Quality metrics
    average_quality_score: Optional[float] = None
    quality_maintained_percentage: Optional[float] = None
    
    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        if self.traditional_estimated_cost > 0:
            self.savings_percentage = (
                self.estimated_savings / self.traditional_estimated_cost * 100
            )
        else:
            self.savings_percentage = 0.0


class FeatureFlags:
    """Feature flags for RouteLLM integration rollout."""
    
    # Core feature flags
    ROUTELLM_ENABLED = "routellm_enabled"
    ROUTELLM_COST_TRACKING = "routellm_cost_tracking"
    ROUTELLM_PERFORMANCE_MONITORING = "routellm_performance_monitoring"
    
    # Domain-specific flags
    ROUTELLM_MEDICAL_DOMAIN = "routellm_medical_domain"
    ROUTELLM_LEGAL_DOMAIN = "routellm_legal_domain"
    ROUTELLM_TECHNICAL_DOMAIN = "routellm_technical_domain"
    ROUTELLM_CREATIVE_DOMAIN = "routellm_creative_domain"
    ROUTELLM_SCIENTIFIC_DOMAIN = "routellm_scientific_domain"
    ROUTELLM_FINANCIAL_DOMAIN = "routellm_financial_domain"
    ROUTELLM_EDUCATIONAL_DOMAIN = "routellm_educational_domain"
    
    # Experimental flags
    ROUTELLM_DYNAMIC_THRESHOLD = "routellm_dynamic_threshold"
    ROUTELLM_A_B_TESTING = "routellm_a_b_testing"
    ROUTELLM_QUALITY_FEEDBACK = "routellm_quality_feedback"
    
    def __init__(self):
        """Initialize feature flags with default values."""
        self._flags: Dict[str, bool] = {
            # Conservative defaults for safe rollout
            self.ROUTELLM_ENABLED: False,
            self.ROUTELLM_COST_TRACKING: True,
            self.ROUTELLM_PERFORMANCE_MONITORING: True,
            
            # Domain flags - start with technical domain only
            self.ROUTELLM_TECHNICAL_DOMAIN: False,
            self.ROUTELLM_MEDICAL_DOMAIN: False,
            self.ROUTELLM_LEGAL_DOMAIN: False,
            self.ROUTELLM_CREATIVE_DOMAIN: False,
            self.ROUTELLM_SCIENTIFIC_DOMAIN: False,
            self.ROUTELLM_FINANCIAL_DOMAIN: False,
            self.ROUTELLM_EDUCATIONAL_DOMAIN: False,
            
            # Experimental features - disabled by default
            self.ROUTELLM_DYNAMIC_THRESHOLD: False,
            self.ROUTELLM_A_B_TESTING: False,
            self.ROUTELLM_QUALITY_FEEDBACK: False,
        }
    
    def is_enabled(self, flag: str) -> bool:
        """Check if a feature flag is enabled."""
        return self._flags.get(flag, False)
    
    def enable(self, flag: str) -> None:
        """Enable a feature flag."""
        self._flags[flag] = True
        logger.info(f"Feature flag enabled: {flag}")
    
    def disable(self, flag: str) -> None:
        """Disable a feature flag."""
        self._flags[flag] = False
        logger.info(f"Feature flag disabled: {flag}")
    
    def update_flags(self, flags: Dict[str, bool]) -> None:
        """Update multiple feature flags at once."""
        self._flags.update(flags)
        logger.info(f"Updated feature flags: {flags}")
    
    def get_all_flags(self) -> Dict[str, bool]:
        """Get all current feature flag values."""
        return self._flags.copy()
    
    def is_domain_enabled(self, domain: str) -> bool:
        """Check if RouteLLM is enabled for a specific domain."""
        domain_flag_map = {
            "medical": self.ROUTELLM_MEDICAL_DOMAIN,
            "legal": self.ROUTELLM_LEGAL_DOMAIN,
            "technical": self.ROUTELLM_TECHNICAL_DOMAIN,
            "creative": self.ROUTELLM_CREATIVE_DOMAIN,
            "scientific": self.ROUTELLM_SCIENTIFIC_DOMAIN,
            "financial": self.ROUTELLM_FINANCIAL_DOMAIN,
            "educational": self.ROUTELLM_EDUCATIONAL_DOMAIN,
        }
        
        domain_flag = domain_flag_map.get(domain.lower())
        if domain_flag:
            return self.is_enabled(domain_flag)
        
        # For unknown domains, use the global flag
        return self.is_enabled(self.ROUTELLM_ENABLED)


class CostTracker:
    """Track routing decisions and calculate cost savings."""
    
    def __init__(self, retention_days: int = 30):
        """Initialize cost tracker with metrics retention policy."""
        self.retention_days = retention_days
        self.metrics: List[RoutingMetrics] = []
    
    def track_routing_decision(
        self,
        text: str,
        domains: List[str],
        routing_method: str,
        selected_model: str,
        estimated_cost: float,
        routing_latency_ms: float = 0.0,
        routing_confidence: float = 0.0,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> str:
        """Track a routing decision and return tracking ID."""
        metric = RoutingMetrics(
            input_text_length=len(text),
            detected_domains=domains,
            routing_method=routing_method,
            selected_model=selected_model,
            estimated_cost=estimated_cost,
            routing_latency_ms=routing_latency_ms,
            routing_confidence=routing_confidence,
            success=success,
            error_message=error_message,
        )
        
        self.metrics.append(metric)
        self._cleanup_old_metrics()
        
        logger.debug(f"Tracked routing decision: {metric.tracking_id}")
        return metric.tracking_id
    
    def update_actual_cost(self, tracking_id: str, actual_cost: float) -> None:
        """Update the actual cost for a tracked request."""
        for metric in self.metrics:
            if metric.tracking_id == tracking_id:
                metric.actual_cost = actual_cost
                # Calculate savings vs baseline (assuming GPT-4 as baseline)
                baseline_cost = actual_cost * 3.0  # Rough 3x cost difference
                metric.cost_savings_vs_baseline = baseline_cost - actual_cost
                logger.debug(f"Updated actual cost for {tracking_id}: ${actual_cost}")
                return
        
        logger.warning(f"Could not find tracking ID {tracking_id} to update cost")
    
    def update_quality_score(self, tracking_id: str, quality_score: float) -> None:
        """Update the response quality score for a tracked request."""
        for metric in self.metrics:
            if metric.tracking_id == tracking_id:
                metric.response_quality_score = quality_score
                logger.debug(f"Updated quality score for {tracking_id}: {quality_score}")
                return
        
        logger.warning(f"Could not find tracking ID {tracking_id} to update quality score")
    
    def get_cost_savings_report(self, period_days: int = 30) -> CostSavingsReport:
        """Generate comprehensive cost savings report."""
        cutoff = datetime.utcnow() - timedelta(days=period_days)
        recent_metrics = [m for m in self.metrics if m.timestamp >= cutoff]
        
        if not recent_metrics:
            return CostSavingsReport(
                period_days=period_days,
                total_requests=0,
                routellm_requests=0,
                traditional_requests=0,
                total_estimated_cost=0.0,
                routellm_estimated_cost=0.0,
                traditional_estimated_cost=0.0,
                estimated_savings=0.0,
                savings_percentage=0.0,
                average_routing_latency_ms=0.0,
                success_rate=1.0,
            )
        
        routellm_metrics = [m for m in recent_metrics if m.routing_method == "routellm"]
        traditional_metrics = [m for m in recent_metrics if m.routing_method == "domain_selector"]
        
        # Cost calculations
        routellm_cost = sum(m.estimated_cost for m in routellm_metrics)
        traditional_cost = sum(m.estimated_cost for m in traditional_metrics)
        total_cost = routellm_cost + traditional_cost
        estimated_savings = traditional_cost - routellm_cost if traditional_cost > routellm_cost else 0
        
        # Performance calculations
        successful_requests = [m for m in recent_metrics if m.success]
        success_rate = len(successful_requests) / len(recent_metrics) if recent_metrics else 1.0
        avg_latency = sum(m.routing_latency_ms for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0.0
        
        # Quality calculations
        quality_scores = [m.response_quality_score for m in recent_metrics if m.response_quality_score is not None]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else None
        
        return CostSavingsReport(
            period_days=period_days,
            total_requests=len(recent_metrics),
            routellm_requests=len(routellm_metrics),
            traditional_requests=len(traditional_metrics),
            total_estimated_cost=total_cost,
            routellm_estimated_cost=routellm_cost,
            traditional_estimated_cost=traditional_cost,
            estimated_savings=estimated_savings,
            savings_percentage=0.0,  # Calculated in __post_init__
            average_routing_latency_ms=avg_latency,
            success_rate=success_rate,
            average_quality_score=avg_quality,
        )
    
    def _cleanup_old_metrics(self) -> None:
        """Remove metrics older than retention period."""
        cutoff = datetime.utcnow() - timedelta(days=self.retention_days)
        original_count = len(self.metrics)
        self.metrics = [m for m in self.metrics if m.timestamp >= cutoff]
        
        cleaned_count = original_count - len(self.metrics)
        if cleaned_count > 0:
            logger.debug(f"Cleaned up {cleaned_count} old routing metrics")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of all tracked metrics."""
        if not self.metrics:
            return {"total_requests": 0, "success_rate": 1.0}
        
        successful = len([m for m in self.metrics if m.success])
        routellm_count = len([m for m in self.metrics if m.routing_method == "routellm"])
        
        return {
            "total_requests": len(self.metrics),
            "success_rate": successful / len(self.metrics),
            "routellm_usage_rate": routellm_count / len(self.metrics),
            "average_cost": sum(m.estimated_cost for m in self.metrics) / len(self.metrics),
            "average_latency_ms": sum(m.routing_latency_ms for m in self.metrics) / len(self.metrics),
        }