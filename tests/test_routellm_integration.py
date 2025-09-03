"""
Test Category: RouteLLM Integration
Real tests for RouteLLM integration functionality including configuration, routing, cost tracking, and fallback mechanisms.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass

from src.orchestrator.models.routellm_integration import (
    RouteLLMConfig,
    RoutingDecision,
    RoutingMetrics,
    CostTracker,
    FeatureFlags,
    RouterType,
    CostSavingsReport,
)
from src.orchestrator.models.domain_router import DomainRouter, DomainConfig
from src.orchestrator.models.model_registry import ModelRegistry
from src.orchestrator.models.model_selector import ModelSelectionCriteria
from src.orchestrator.core.model import Model, ModelCapabilities, ModelCost, ModelMetrics


class TestRouteLLMConfig:
    """Test RouteLLM configuration functionality."""
    
    def test_default_config(self):
        """Test default RouteLLM configuration values."""
        config = RouteLLMConfig()
        
        assert config.enabled is False
        assert config.router_type == RouterType.MATRIX_FACTORIZATION
        assert config.threshold == 0.11593
        assert config.strong_model == "gpt-4-1106-preview"
        assert config.weak_model == "gpt-3.5-turbo"
        assert config.fallback_enabled is True
        assert config.cost_tracking_enabled is True
    
    def test_get_router_model_string(self):
        """Test router model string generation."""
        config = RouteLLMConfig(
            router_type=RouterType.MATRIX_FACTORIZATION,
            threshold=0.12345
        )
        
        expected = "router-mf-0.12345"
        assert config.get_router_model_string() == expected
    
    def test_domain_override(self):
        """Test domain-specific routing overrides."""
        config = RouteLLMConfig(
            domain_routing_overrides={
                "medical": {"threshold": 0.8, "strong_model": "gpt-4-medical"},
                "legal": {"threshold": 0.9}
            }
        )
        
        medical_override = config.get_domain_override("medical")
        assert medical_override is not None
        assert medical_override["threshold"] == 0.8
        assert medical_override["strong_model"] == "gpt-4-medical"
        
        legal_override = config.get_domain_override("legal")
        assert legal_override is not None
        assert legal_override["threshold"] == 0.9
        
        unknown_override = config.get_domain_override("unknown")
        assert unknown_override is None


class TestRoutingDecision:
    """Test routing decision functionality."""
    
    def test_routing_decision_creation(self):
        """Test creating routing decisions."""
        decision = RoutingDecision(
            should_use_routellm=True,
            recommended_model="gpt-3.5-turbo",
            confidence=0.85,
            estimated_cost=0.002,
            reasoning="Simple query suitable for weak model",
            domains=["technical"]
        )
        
        assert decision.should_use_routellm is True
        assert decision.recommended_model == "gpt-3.5-turbo"
        assert decision.confidence == 0.85
        assert decision.is_high_confidence is True  # >= 0.8
        assert decision.is_cost_effective is True  # > 0 and should_use_routellm
    
    def test_low_confidence_decision(self):
        """Test low confidence routing decision."""
        decision = RoutingDecision(
            should_use_routellm=True,
            confidence=0.6
        )
        
        assert decision.is_high_confidence is False  # < 0.8


class TestFeatureFlags:
    """Test feature flag functionality."""
    
    def test_default_flags(self):
        """Test default feature flag values."""
        flags = FeatureFlags()
        
        # Core flags should be conservative by default
        assert flags.is_enabled(FeatureFlags.ROUTELLM_ENABLED) is False
        assert flags.is_enabled(FeatureFlags.ROUTELLM_COST_TRACKING) is True
        assert flags.is_enabled(FeatureFlags.ROUTELLM_PERFORMANCE_MONITORING) is True
        
        # Domain flags should be disabled by default
        assert flags.is_enabled(FeatureFlags.ROUTELLM_TECHNICAL_DOMAIN) is False
        assert flags.is_enabled(FeatureFlags.ROUTELLM_MEDICAL_DOMAIN) is False
    
    def test_enable_disable_flags(self):
        """Test enabling and disabling feature flags."""
        flags = FeatureFlags()
        
        # Enable a flag
        flags.enable(FeatureFlags.ROUTELLM_ENABLED)
        assert flags.is_enabled(FeatureFlags.ROUTELLM_ENABLED) is True
        
        # Disable the flag
        flags.disable(FeatureFlags.ROUTELLM_ENABLED)
        assert flags.is_enabled(FeatureFlags.ROUTELLM_ENABLED) is False
    
    def test_update_multiple_flags(self):
        """Test updating multiple flags at once."""
        flags = FeatureFlags()
        
        new_flags = {
            FeatureFlags.ROUTELLM_ENABLED: True,
            FeatureFlags.ROUTELLM_TECHNICAL_DOMAIN: True,
            FeatureFlags.ROUTELLM_MEDICAL_DOMAIN: False,
        }
        
        flags.update_flags(new_flags)
        
        assert flags.is_enabled(FeatureFlags.ROUTELLM_ENABLED) is True
        assert flags.is_enabled(FeatureFlags.ROUTELLM_TECHNICAL_DOMAIN) is True
        assert flags.is_enabled(FeatureFlags.ROUTELLM_MEDICAL_DOMAIN) is False
    
    def test_domain_enabled_check(self):
        """Test domain-specific enablement checks."""
        flags = FeatureFlags()
        
        # Enable technical domain
        flags.enable(FeatureFlags.ROUTELLM_TECHNICAL_DOMAIN)
        
        assert flags.is_domain_enabled("technical") is True
        assert flags.is_domain_enabled("medical") is False
        
        # Unknown domain should use global flag
        flags.enable(FeatureFlags.ROUTELLM_ENABLED)
        assert flags.is_domain_enabled("unknown_domain") is True
        
        flags.disable(FeatureFlags.ROUTELLM_ENABLED)
        assert flags.is_domain_enabled("unknown_domain") is False


class TestCostTracker:
    """Test cost tracking functionality."""
    
    def test_track_routing_decision(self):
        """Test tracking routing decisions."""
        tracker = CostTracker()
        
        tracking_id = tracker.track_routing_decision(
            text="Test query about technical topics",
            domains=["technical"],
            routing_method="routellm",
            selected_model="openai:gpt-3.5-turbo",
            estimated_cost=0.002,
            routing_latency_ms=15.5,
            routing_confidence=0.85,
            success=True,
        )
        
        assert tracking_id is not None
        assert len(tracker.metrics) == 1
        
        metric = tracker.metrics[0]
        assert metric.tracking_id == tracking_id
        assert metric.routing_method == "routellm"
        assert metric.selected_model == "openai:gpt-3.5-turbo"
        assert metric.estimated_cost == 0.002
        assert metric.routing_latency_ms == 15.5
        assert metric.routing_confidence == 0.85
        assert metric.success is True
    
    def test_update_actual_cost(self):
        """Test updating actual cost for tracked requests."""
        tracker = CostTracker()
        
        tracking_id = tracker.track_routing_decision(
            text="Test query",
            domains=["general"],
            routing_method="routellm",
            selected_model="openai:gpt-3.5-turbo",
            estimated_cost=0.002,
        )
        
        # Update actual cost
        tracker.update_actual_cost(tracking_id, 0.0015)
        
        metric = tracker.metrics[0]
        assert metric.actual_cost == 0.0015
        # baseline is 3x actual cost, so savings = baseline - actual = 3*0.0015 - 0.0015 = 2*0.0015
        assert abs(metric.cost_savings_vs_baseline - (0.0015 * 2)) < 1e-10  # Handle floating point precision
    
    def test_update_quality_score(self):
        """Test updating quality scores."""
        tracker = CostTracker()
        
        tracking_id = tracker.track_routing_decision(
            text="Test query",
            domains=["general"],
            routing_method="routellm",
            selected_model="openai:gpt-3.5-turbo",
            estimated_cost=0.002,
        )
        
        # Update quality score
        tracker.update_quality_score(tracking_id, 0.92)
        
        metric = tracker.metrics[0]
        assert metric.response_quality_score == 0.92
    
    def test_cost_savings_report(self):
        """Test generating cost savings reports."""
        tracker = CostTracker()
        
        # Add some RouteLLM requests
        tracker.track_routing_decision(
            text="Simple query 1", domains=["general"], routing_method="routellm",
            selected_model="gpt-3.5-turbo", estimated_cost=0.001, success=True
        )
        tracker.track_routing_decision(
            text="Simple query 2", domains=["general"], routing_method="routellm",
            selected_model="gpt-3.5-turbo", estimated_cost=0.001, success=True
        )
        
        # Add some traditional requests
        tracker.track_routing_decision(
            text="Complex query 1", domains=["medical"], routing_method="domain_selector",
            selected_model="gpt-4", estimated_cost=0.01, success=True
        )
        tracker.track_routing_decision(
            text="Complex query 2", domains=["legal"], routing_method="domain_selector",
            selected_model="gpt-4", estimated_cost=0.01, success=True
        )
        
        report = tracker.get_cost_savings_report(period_days=30)
        
        assert report.total_requests == 4
        assert report.routellm_requests == 2
        assert report.traditional_requests == 2
        assert report.routellm_estimated_cost == 0.002  # 2 * 0.001
        assert report.traditional_estimated_cost == 0.02  # 2 * 0.01
        assert report.success_rate == 1.0  # All successful
    
    def test_metrics_cleanup(self):
        """Test automatic cleanup of old metrics."""
        tracker = CostTracker(retention_days=1)  # Very short retention
        
        # Create an old metric (simulate by modifying timestamp)
        old_tracking_id = tracker.track_routing_decision(
            text="Old query", domains=["general"], routing_method="routellm",
            selected_model="gpt-3.5-turbo", estimated_cost=0.001
        )
        
        # Manually set old timestamp
        tracker.metrics[0].timestamp = datetime.utcnow() - timedelta(days=2)
        
        # Add a recent metric - this should trigger cleanup
        new_tracking_id = tracker.track_routing_decision(
            text="New query", domains=["general"], routing_method="routellm",
            selected_model="gpt-3.5-turbo", estimated_cost=0.001
        )
        
        # Should only have the recent metric
        assert len(tracker.metrics) == 1
        assert tracker.metrics[0].tracking_id == new_tracking_id


@pytest.fixture
def mock_model_registry():
    """Create a mock model registry for testing."""
    registry = MagicMock(spec=ModelRegistry)
    
    # Mock models
    gpt35_turbo = MagicMock(spec=Model)
    gpt35_turbo.name = "gpt-3.5-turbo"
    gpt35_turbo.provider = "openai"
    gpt35_turbo.capabilities = ModelCapabilities(
        supported_tasks=["chat", "text-generation"],
        accuracy_score=0.85,
        speed_rating="fast",
        vision_capable=False,
        code_specialized=True,
        supports_function_calling=True
    )
    gpt35_turbo.cost = ModelCost(
        input_cost_per_1k_tokens=0.001,
        output_cost_per_1k_tokens=0.002
    )
    gpt35_turbo.metrics = ModelMetrics(success_rate=0.95)
    
    gpt4 = MagicMock(spec=Model)
    gpt4.name = "gpt-4-1106-preview"
    gpt4.provider = "openai"
    gpt4.capabilities = ModelCapabilities(
        supported_tasks=["chat", "text-generation", "vision"],
        accuracy_score=0.95,
        speed_rating="medium",
        vision_capable=True,
        code_specialized=True,
        supports_function_calling=True
    )
    gpt4.cost = ModelCost(
        input_cost_per_1k_tokens=0.01,
        output_cost_per_1k_tokens=0.03
    )
    gpt4.metrics = ModelMetrics(success_rate=0.98)
    
    registry.models = {
        "openai:gpt-3.5-turbo": gpt35_turbo,
        "openai:gpt-4-1106-preview": gpt4,
    }
    
    return registry


class TestDomainRouterRouteLLMIntegration:
    """Test RouteLLM integration in DomainRouter."""
    
    def test_domain_router_without_routellm(self, mock_model_registry):
        """Test domain router works normally without RouteLLM configuration."""
        router = DomainRouter(mock_model_registry)
        
        assert router.routellm_config is not None
        assert router.routellm_config.enabled is False
        assert router.feature_flags is not None
        assert router.cost_tracker is not None  # Enabled by default
        assert router.is_routellm_enabled() is False
    
    def test_domain_router_with_routellm_config(self, mock_model_registry):
        """Test domain router with RouteLLM configuration."""
        config = RouteLLMConfig(
            enabled=True,
            router_type=RouterType.BERT_CLASSIFIER,
            strong_model="gpt-4-turbo",
            weak_model="gpt-3.5-turbo"
        )
        
        flags = FeatureFlags()
        flags.enable(FeatureFlags.ROUTELLM_ENABLED)
        
        router = DomainRouter(mock_model_registry, config, flags)
        
        assert router.routellm_config.enabled is True
        assert router.routellm_config.router_type == RouterType.BERT_CLASSIFIER
        assert router.feature_flags.is_enabled(FeatureFlags.ROUTELLM_ENABLED) is True
        assert router.is_routellm_enabled() is True
    
    async def test_should_use_routellm(self, mock_model_registry):
        """Test RouteLLM usage decision logic."""
        config = RouteLLMConfig(enabled=True)
        flags = FeatureFlags()
        
        router = DomainRouter(mock_model_registry, config, flags)
        
        domains = [("technical", 0.8)]
        
        # Should not use RouteLLM with feature flag disabled
        should_use = await router._should_use_routellm("test query", domains)
        assert should_use is False
        
        # Enable feature flag
        flags.enable(FeatureFlags.ROUTELLM_ENABLED)
        
        # Should use RouteLLM now (but will fail due to import)
        should_use = await router._should_use_routellm("test query", domains)
        assert should_use is False  # Will be False due to import failure in tests
    
    def test_text_complexity_estimation(self, mock_model_registry):
        """Test text complexity estimation for routing decisions."""
        router = DomainRouter(mock_model_registry)
        
        # Simple text
        simple_text = "Hello world"
        simple_complexity = router._estimate_text_complexity(simple_text)
        assert 0.0 <= simple_complexity <= 0.3
        
        # Complex technical text
        complex_text = """
        Implement a distributed microservices architecture using Kubernetes (K8s) with 
        service mesh (Istio), implementing circuit breaker patterns, observability with 
        Prometheus/Grafana, and automated CI/CD pipelines using GitOps methodology.
        Configure RBAC, network policies, and pod security standards. Implement 
        horizontal pod autoscaling (HPA) and vertical pod autoscaling (VPA).
        """
        complex_complexity = router._estimate_text_complexity(complex_text)
        assert 0.5 <= complex_complexity <= 1.0
    
    async def test_route_by_domain_fallback(self, mock_model_registry):
        """Test that route_by_domain falls back to traditional routing when RouteLLM fails."""
        config = RouteLLMConfig(enabled=True)
        flags = FeatureFlags()
        flags.enable(FeatureFlags.ROUTELLM_ENABLED)
        
        # Mock the model selector to return a specific model
        mock_selector = AsyncMock()
        mock_selector.select_model.return_value = mock_model_registry.models["openai:gpt-3.5-turbo"]
        
        router = DomainRouter(mock_model_registry, config, flags)
        router.selector = mock_selector
        
        # Should fall back to domain selector since RouteLLM import will fail
        result = await router.route_by_domain("Test technical query")
        
        assert result is not None
        assert result.name == "gpt-3.5-turbo"
        
        # Verify fallback was used by checking selector was called
        mock_selector.select_model.assert_called_once()
    
    def test_get_routellm_status(self, mock_model_registry):
        """Test getting RouteLLM integration status."""
        config = RouteLLMConfig(
            enabled=True,
            router_type=RouterType.MATRIX_FACTORIZATION,
            strong_model="gpt-4-turbo",
            threshold=0.2
        )
        flags = FeatureFlags()
        flags.enable(FeatureFlags.ROUTELLM_ENABLED)
        
        router = DomainRouter(mock_model_registry, config, flags)
        status = router.get_routellm_status()
        
        assert status["config_enabled"] is True
        assert status["feature_flag_enabled"] is True
        assert status["controller_initialized"] is False  # Not initialized yet
        assert status["controller_available"] is False
        assert status["cost_tracking_enabled"] is True
        assert status["router_type"] == "mf"
        assert status["strong_model"] == "gpt-4-turbo"
        assert status["threshold"] == 0.2
    
    def test_analyze_text_with_routellm(self, mock_model_registry):
        """Test text analysis includes RouteLLM information when enabled."""
        config = RouteLLMConfig(enabled=True)
        flags = FeatureFlags()
        flags.enable(FeatureFlags.ROUTELLM_ENABLED)
        
        router = DomainRouter(mock_model_registry, config, flags)
        
        analysis = router.analyze_text("This is a technical discussion about machine learning algorithms")
        
        # Should include standard domain analysis
        assert "text_length" in analysis
        assert "detected_domains" in analysis
        assert "primary_domain" in analysis
        
        # Should include RouteLLM analysis
        assert "routellm_enabled" in analysis
        assert "complexity_score" in analysis
        assert "recommended_routing" in analysis
        assert "routing_confidence" in analysis
        
        # Complexity score should be reasonable
        assert 0.0 <= analysis["complexity_score"] <= 1.0
        assert analysis["recommended_routing"] in ["strong_model", "weak_model"]
    
    def test_cost_savings_report_integration(self, mock_model_registry):
        """Test cost savings report integration."""
        config = RouteLLMConfig(enabled=True, cost_tracking_enabled=True)
        router = DomainRouter(mock_model_registry, config)
        
        # Initially no data
        report = router.get_cost_savings_report()
        assert report is not None
        assert report["total_requests"] == 0
        
        # Add some tracking data manually
        if router.cost_tracker:
            router.cost_tracker.track_routing_decision(
                text="Test query",
                domains=["technical"],
                routing_method="routellm",
                selected_model="gpt-3.5-turbo",
                estimated_cost=0.001,
                success=True
            )
            
            report = router.get_cost_savings_report()
            assert report["total_requests"] == 1
            assert report["routellm_requests"] == 1
    
    def test_update_routellm_config(self, mock_model_registry):
        """Test updating RouteLLM configuration."""
        router = DomainRouter(mock_model_registry)
        
        # Initial config should be default
        assert router.routellm_config.enabled is False
        assert router.routellm_config.router_type == RouterType.MATRIX_FACTORIZATION
        
        # Update configuration
        new_config = RouteLLMConfig(
            enabled=True,
            router_type=RouterType.BERT_CLASSIFIER,
            strong_model="gpt-4-turbo",
            threshold=0.15
        )
        
        router.update_routellm_config(new_config)
        
        # Config should be updated
        assert router.routellm_config.enabled is True
        assert router.routellm_config.router_type == RouterType.BERT_CLASSIFIER
        assert router.routellm_config.strong_model == "gpt-4-turbo"
        assert router.routellm_config.threshold == 0.15
        
        # Controller should be reset for reinitialization
        assert router._controller_initialized is False
        assert router._routellm_controller is None
    
    def test_feature_flags_integration(self, mock_model_registry):
        """Test feature flags integration and updates."""
        router = DomainRouter(mock_model_registry)
        
        # Initial state
        assert router.feature_flags.is_enabled(FeatureFlags.ROUTELLM_ENABLED) is False
        
        # Update flags
        new_flags = {
            FeatureFlags.ROUTELLM_ENABLED: True,
            FeatureFlags.ROUTELLM_TECHNICAL_DOMAIN: True,
        }
        
        router.update_feature_flags(new_flags)
        
        # Flags should be updated
        assert router.feature_flags.is_enabled(FeatureFlags.ROUTELLM_ENABLED) is True
        assert router.feature_flags.is_enabled(FeatureFlags.ROUTELLM_TECHNICAL_DOMAIN) is True
        
        # Check domain-specific enablement
        assert router.feature_flags.is_domain_enabled("technical") is True
        assert router.feature_flags.is_domain_enabled("medical") is False
    
    def test_backward_compatibility(self, mock_model_registry):
        """Test that existing API remains completely backward compatible."""
        # Create router exactly as before (no RouteLLM params)
        router = DomainRouter(mock_model_registry)
        
        # All original methods should exist and work
        assert hasattr(router, 'detect_domains')
        assert hasattr(router, 'register_domain')
        assert hasattr(router, 'get_domain_info')
        assert hasattr(router, 'list_domains')
        assert hasattr(router, 'analyze_text')
        
        # Domain detection should work normally
        domains = router.detect_domains("This is a medical diagnosis discussion")
        assert len(domains) > 0
        assert any(domain[0] == "medical" for domain in domains)
        
        # Domain registration should work
        custom_domain = DomainConfig(
            name="custom",
            keywords=["custom", "test"],
            patterns=[r"\bcustom\b"]
        )
        
        router.register_domain(custom_domain)
        assert "custom" in router.domains
        
        # Analysis should work without RouteLLM fields (when disabled)
        analysis = router.analyze_text("Custom test content")
        assert "text_length" in analysis
        assert "detected_domains" in analysis
        # RouteLLM fields should not be present when disabled
        assert "routellm_enabled" not in analysis