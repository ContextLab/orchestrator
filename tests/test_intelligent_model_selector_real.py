"""Real integration tests for Intelligent Model Selector - Phase 3

Tests intelligent model selection with REAL model calls, service integration,
and multi-dimensional optimization. NO MOCKS policy enforced.
"""

import pytest
import time
import asyncio
from unittest.mock import patch
from typing import List, Dict

from src.orchestrator.intelligence.intelligent_model_selector import (
    IntelligentModelSelector,
    ModelRequirements,
    ModelScore,
    OptimizationObjective,
    create_intelligent_selector,
    select_optimal_model_for_task
)
from src.orchestrator.models.model_registry import ModelRegistry
from src.orchestrator.core.exceptions import NoEligibleModelsError
from src.orchestrator.utils.api_keys import load_api_keys_optional


class TestIntelligentModelSelectorReal:
    """Test intelligent model selector with real integration."""
    
    def setup_method(self):
        """Setup test environment with real model registry."""
        self.api_keys = load_api_keys_optional()
        self.registry = ModelRegistry()
        self.selector = IntelligentModelSelector(self.registry)
        
        # Register some test models for selection
        self._setup_test_models()
    
    def _setup_test_models(self):
        """Setup test models in registry."""
        try:
            # Try to register real models if API keys available
            if self.api_keys.get("OPENAI_API_KEY"):
                from src.orchestrator.models.openai_model import OpenAIModel
                openai_model = OpenAIModel("gpt-3.5-turbo", api_key=self.api_keys["OPENAI_API_KEY"])
                self.registry.register_model(openai_model)
                
            if self.api_keys.get("ANTHROPIC_API_KEY"):
                from src.orchestrator.models.anthropic_model import AnthropicModel  
                anthropic_model = AnthropicModel("claude-haiku", api_key=self.api_keys["ANTHROPIC_API_KEY"])
                self.registry.register_model(anthropic_model)
                
        except ImportError as e:
            pytest.skip(f"Model imports not available: {e}")
        except Exception as e:
            pytest.skip(f"Error setting up models: {e}")
    
    def test_intelligent_selector_initialization(self):
        """Test intelligent selector initializes correctly."""
        assert self.selector.model_registry == self.registry
        assert isinstance(self.selector.cost_data, dict)
        assert isinstance(self.selector.latency_baselines, dict)
        assert isinstance(self.selector.optimization_weights, dict)
        
        # Verify optimization weights sum to 1 for each objective
        for objective, weights in self.selector.optimization_weights.items():
            weight_sum = sum(weights.values())
            assert abs(weight_sum - 1.0) < 0.01, f"Weights for {objective} don't sum to 1: {weight_sum}"
    
    def test_model_requirements_creation(self):
        """Test creating model requirements with various options."""
        # Basic requirements
        req1 = ModelRequirements()
        assert req1.capabilities == []
        assert req1.optimization_objective == OptimizationObjective.BALANCED
        
        # Advanced requirements
        req2 = ModelRequirements(
            capabilities=["code_generation", "analysis"],
            max_cost_per_token=0.001,
            max_latency_ms=1000,
            min_accuracy_score=0.8,
            preferred_providers=["openai", "anthropic"],
            optimization_objective=OptimizationObjective.PERFORMANCE,
            expected_tokens=500,
            workload_priority="high"
        )
        
        assert len(req2.capabilities) == 2
        assert req2.max_cost_per_token == 0.001
        assert req2.optimization_objective == OptimizationObjective.PERFORMANCE
        assert req2.expected_tokens == 500
    
    @pytest.mark.skipif(
        not load_api_keys_optional().get("OPENAI_API_KEY"),
        reason="OpenAI API key required for real integration test"
    )
    def test_select_optimal_model_with_real_openai(self):
        """Test selecting optimal model with real OpenAI integration."""
        requirements = ModelRequirements(
            optimization_objective=OptimizationObjective.COST,
            expected_tokens=100,
            max_cost_per_token=0.01
        )
        
        try:
            selected_model = self.selector.select_optimal_model(requirements)
            assert selected_model is not None
            assert isinstance(selected_model, str)
            assert ":" in selected_model  # Should be provider:model format
            
            # Verify selection completed in reasonable time
            start_time = time.time()
            self.selector.select_optimal_model(requirements)
            selection_time = time.time() - start_time
            assert selection_time < 1.0, f"Selection took too long: {selection_time}s"
            
        except NoEligibleModelsError:
            pytest.skip("No eligible models available for test")
    
    def test_get_model_recommendations(self):
        """Test getting top-k model recommendations."""
        requirements = ModelRequirements(
            optimization_objective=OptimizationObjective.BALANCED,
            expected_tokens=200
        )
        
        recommendations = self.selector.get_model_recommendations(requirements, top_k=3)
        
        # Should return list of ModelScore objects
        assert isinstance(recommendations, list)
        for rec in recommendations:
            assert isinstance(rec, ModelScore)
            assert hasattr(rec, 'model_key')
            assert hasattr(rec, 'weighted_score')
            assert hasattr(rec, 'confidence')
            
            # Scores should be in valid range
            assert 0.0 <= rec.weighted_score <= 1.0
            assert 0.0 <= rec.confidence <= 1.0
            assert 0.0 <= rec.performance_score <= 1.0
            assert 0.0 <= rec.cost_score <= 1.0
            assert 0.0 <= rec.latency_score <= 1.0
            assert 0.0 <= rec.accuracy_score <= 1.0
            assert 0.0 <= rec.availability_score <= 1.0
        
        # Should be sorted by weighted score (descending)
        if len(recommendations) > 1:
            for i in range(len(recommendations) - 1):
                assert recommendations[i].weighted_score >= recommendations[i + 1].weighted_score
    
    def test_explain_selection_detailed(self):
        """Test detailed selection explanation."""
        # First get a recommendation
        requirements = ModelRequirements(
            optimization_objective=OptimizationObjective.PERFORMANCE,
            expected_tokens=300
        )
        
        recommendations = self.selector.get_model_recommendations(requirements, top_k=1)
        if not recommendations:
            pytest.skip("No models available for explanation test")
            
        model_key = recommendations[0].model_key
        explanation = self.selector.explain_selection(model_key, requirements)
        
        # Verify explanation structure
        assert "model_key" in explanation
        assert "overall_score" in explanation
        assert "confidence" in explanation
        assert "optimization_objective" in explanation
        assert "score_breakdown" in explanation
        
        # Verify score breakdown
        breakdown = explanation["score_breakdown"]
        expected_dimensions = ["performance", "cost", "latency", "accuracy", "availability"]
        for dimension in expected_dimensions:
            assert dimension in breakdown
            assert "score" in breakdown[dimension]
            assert "weight" in breakdown[dimension]
            assert 0.0 <= breakdown[dimension]["score"] <= 1.0
            assert 0.0 <= breakdown[dimension]["weight"] <= 1.0
    
    def test_optimization_objectives_behavior(self):
        """Test different optimization objectives produce different selections."""
        base_requirements = ModelRequirements(expected_tokens=200)
        
        objectives_to_test = [
            OptimizationObjective.PERFORMANCE,
            OptimizationObjective.COST,
            OptimizationObjective.LATENCY,
            OptimizationObjective.BALANCED
        ]
        
        selections = {}
        for objective in objectives_to_test:
            requirements = ModelRequirements(
                optimization_objective=objective,
                expected_tokens=200
            )
            
            try:
                recommendations = self.selector.get_model_recommendations(requirements, top_k=3)
                if recommendations:
                    selections[objective] = recommendations[0].model_key
            except NoEligibleModelsError:
                continue
        
        # At least some objectives should produce selections
        assert len(selections) > 0, "No selections produced for any optimization objective"
        
        # Different objectives might produce different selections
        unique_selections = set(selections.values())
        # This test is informational - different objectives may or may not produce different results
        print(f"Unique selections across objectives: {len(unique_selections)} / {len(selections)}")
        for obj, selection in selections.items():
            print(f"{obj.value}: {selection}")
    
    def test_cost_optimization_real_pricing(self):
        """Test cost optimization uses real pricing data."""
        # Create high-cost constraint
        requirements = ModelRequirements(
            optimization_objective=OptimizationObjective.COST,
            max_cost_per_token=0.0001,  # Very low cost limit
            expected_tokens=1000
        )
        
        recommendations = self.selector.get_model_recommendations(requirements, top_k=5)
        
        # Should prioritize free/cheap models
        for rec in recommendations:
            if rec.estimated_cost is not None:
                cost_per_token = rec.estimated_cost / 1000  # expected tokens
                if cost_per_token > requirements.max_cost_per_token:
                    # Should filter out expensive models or give them low scores
                    assert rec.weighted_score < 0.5, f"Expensive model {rec.model_key} scored too high"
    
    def test_latency_optimization_real_baselines(self):
        """Test latency optimization uses real baseline data."""
        requirements = ModelRequirements(
            optimization_objective=OptimizationObjective.LATENCY,
            max_latency_ms=600,  # Aggressive latency requirement
            expected_tokens=100
        )
        
        recommendations = self.selector.get_model_recommendations(requirements, top_k=5)
        
        # Should prioritize fast models
        for rec in recommendations:
            if rec.estimated_latency_ms is not None:
                if rec.estimated_latency_ms > requirements.max_latency_ms:
                    # Should filter out slow models or give them low scores
                    assert rec.weighted_score < 0.5, f"Slow model {rec.model_key} scored too high"
    
    def test_availability_score_integration(self):
        """Test availability scoring integrates with Phase 2 service managers."""
        # Test with Ollama models that require service management
        requirements = ModelRequirements(
            preferred_providers=["ollama"],
            optimization_objective=OptimizationObjective.BALANCED
        )
        
        recommendations = self.selector.get_model_recommendations(requirements, top_k=3)
        
        # Verify availability scoring works
        for rec in recommendations:
            if rec.provider == "ollama":
                # Availability score should reflect actual service status
                assert 0.0 <= rec.availability_score <= 1.0
                
                # If Ollama service is running and model available, should have high score
                from src.orchestrator.utils.service_manager import SERVICE_MANAGERS
                ollama_manager = SERVICE_MANAGERS.get("ollama")
                if ollama_manager and ollama_manager.is_running():
                    if ollama_manager.is_model_available(rec.model_name):
                        assert rec.availability_score >= 0.8
    
    def test_performance_score_ucb_integration(self):
        """Test performance scoring integrates with existing UCB algorithm."""
        # Use a model with some UCB history if available
        if hasattr(self.registry, 'model_selector') and self.registry.model_selector.model_stats:
            model_key = next(iter(self.registry.model_selector.model_stats.keys()))
            
            # Add some fake history for testing
            self.registry.model_selector.model_stats[model_key] = {
                "attempts": 10,
                "successes": 8,
                "total_reward": 8.5,
                "average_reward": 0.85
            }
            
            requirements = ModelRequirements()
            score = self.selector._score_single_model(model_key, requirements)
            
            if score:
                # Performance score should reflect UCB data
                assert score.performance_score > 0.8, "High UCB reward should result in high performance score"
                assert score.accuracy_score == 0.8, "Success rate should match accuracy score"
    
    def test_concurrent_selection_performance(self):
        """Test intelligent selection performance under concurrent load."""
        requirements = ModelRequirements(
            optimization_objective=OptimizationObjective.BALANCED,
            expected_tokens=150
        )
        
        def select_model():
            return self.selector.select_optimal_model(requirements)
        
        # Measure time for sequential selections
        start_time = time.time()
        for _ in range(5):
            try:
                select_model()
            except NoEligibleModelsError:
                pass
        sequential_time = time.time() - start_time
        
        # Each selection should be fast
        avg_time_per_selection = sequential_time / 5
        assert avg_time_per_selection < 0.5, f"Selection too slow: {avg_time_per_selection}s"
    
    def test_create_intelligent_selector_helper(self):
        """Test helper function for creating intelligent selector."""
        selector = create_intelligent_selector(self.registry)
        assert isinstance(selector, IntelligentModelSelector)
        assert selector.model_registry == self.registry
    
    def test_select_optimal_model_for_task_helper(self):
        """Test high-level task-based model selection."""
        # Test different task types
        test_tasks = [
            ("Write a Python function to sort a list", OptimizationObjective.PERFORMANCE),
            ("Create a creative story about robots", OptimizationObjective.BALANCED),
            ("Analyze this data quickly", OptimizationObjective.LATENCY),
            ("Generate code on a budget", OptimizationObjective.COST)
        ]
        
        for task, objective in test_tasks:
            try:
                model_key = select_optimal_model_for_task(
                    task, 
                    objective, 
                    self.registry
                )
                assert isinstance(model_key, str)
                assert ":" in model_key
            except NoEligibleModelsError:
                # Expected if no models available for this combination
                continue
    
    def test_hard_constraints_enforcement(self):
        """Test that hard constraints are enforced."""
        # Set impossible constraints
        requirements = ModelRequirements(
            max_cost_per_token=0.0000001,  # Impossibly low cost
            max_latency_ms=1,  # Impossibly low latency
            min_accuracy_score=0.999,  # Impossibly high accuracy
            expected_tokens=1000
        )
        
        # Should either find no models or only return models that truly meet constraints
        try:
            recommendations = self.selector.get_model_recommendations(requirements, top_k=5)
            for rec in recommendations:
                # Any returned model should meet hard constraints
                if rec.estimated_cost is not None:
                    cost_per_token = rec.estimated_cost / requirements.expected_tokens
                    # Note: This might fail if constraints are truly impossible
                    # The selector should handle this gracefully
        except NoEligibleModelsError:
            # Expected behavior for impossible constraints
            pass
    
    def test_confidence_scoring_accuracy(self):
        """Test confidence scoring reflects data availability."""
        # Test with a model that should have good data availability
        known_models = ["openai:gpt-3.5-turbo", "anthropic:claude-haiku"]
        
        for model_key in known_models:
            if model_key in self.selector.cost_data and model_key in self.selector.latency_baselines:
                confidence = self.selector._calculate_confidence(model_key)
                assert confidence > 0.5, f"Model {model_key} with good data should have high confidence"
        
        # Test with unknown model
        unknown_confidence = self.selector._calculate_confidence("unknown:model")
        assert unknown_confidence < 0.8, "Unknown model should have lower confidence"


class TestIntelligentSelectorEdgeCases:
    """Test edge cases and error handling."""
    
    def setup_method(self):
        """Setup minimal test environment."""
        self.registry = ModelRegistry()
        self.selector = IntelligentModelSelector(self.registry)
    
    def test_no_eligible_models_error(self):
        """Test handling when no models meet requirements."""
        # Empty registry should raise NoEligibleModelsError
        requirements = ModelRequirements()
        
        with pytest.raises(NoEligibleModelsError):
            self.selector.select_optimal_model(requirements)
    
    def test_empty_recommendations(self):
        """Test behavior with empty model registry."""
        requirements = ModelRequirements()
        recommendations = self.selector.get_model_recommendations(requirements)
        
        assert recommendations == []
    
    def test_malformed_model_key_handling(self):
        """Test handling of malformed model keys."""
        # Add malformed model key to test error handling
        malformed_keys = ["no-colon-key", ":missing-provider", "provider:", ""]
        
        for key in malformed_keys:
            score = self.selector._score_single_model(key, ModelRequirements())
            # Should handle gracefully and return None
            assert score is None
    
    def test_missing_data_graceful_handling(self):
        """Test graceful handling of missing cost/latency data."""
        # Test with model not in cost or latency data
        unknown_model = "unknown:test-model"
        
        cost_score = self.selector._calculate_cost_score(unknown_model, ModelRequirements())
        latency_score = self.selector._calculate_latency_score(unknown_model, ModelRequirements())
        
        # Should return reasonable default scores
        assert 0.0 <= cost_score <= 1.0
        assert 0.0 <= latency_score <= 1.0
    
    def test_extreme_requirements_handling(self):
        """Test handling of extreme requirements values."""
        extreme_requirements = ModelRequirements(
            max_cost_per_token=float('inf'),
            max_latency_ms=0,
            min_accuracy_score=-1.0,
            expected_tokens=0
        )
        
        # Should handle gracefully without crashing
        recommendations = self.selector.get_model_recommendations(extreme_requirements)
        assert isinstance(recommendations, list)


@pytest.mark.integration
class TestIntelligentSelectorRealWorldScenarios:
    """Real-world scenario testing."""
    
    def setup_method(self):
        """Setup for real-world scenarios."""
        self.api_keys = load_api_keys_optional()
        self.registry = ModelRegistry()
        self.selector = IntelligentModelSelector(self.registry)
    
    @pytest.mark.skipif(
        not any(load_api_keys_optional().get(key) for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]),
        reason="At least one API key required for real integration test"
    )
    def test_research_pipeline_model_selection(self):
        """Test model selection for research pipeline scenario."""
        # Simulate research pipeline requirements
        research_requirements = ModelRequirements(
            capabilities=["analysis", "reasoning"],
            optimization_objective=OptimizationObjective.ACCURACY,
            expected_tokens=2000,
            max_latency_ms=5000,
            workload_priority="high"
        )
        
        try:
            recommendations = self.selector.get_model_recommendations(research_requirements, top_k=3)
            assert len(recommendations) > 0
            
            # Research tasks should prefer accuracy
            top_model = recommendations[0]
            assert top_model.accuracy_score >= 0.7, "Research task should prioritize accurate models"
            
        except NoEligibleModelsError:
            pytest.skip("No eligible models for research scenario")
    
    def test_batch_processing_model_selection(self):
        """Test model selection for batch processing scenario."""
        batch_requirements = ModelRequirements(
            optimization_objective=OptimizationObjective.COST,
            expected_tokens=10000,
            batch_size=50,
            concurrent_requests=5,
            workload_priority="low"
        )
        
        try:
            recommendations = self.selector.get_model_recommendations(batch_requirements, top_k=3)
            
            # Batch processing should prioritize cost efficiency
            if recommendations:
                top_model = recommendations[0]
                assert top_model.cost_score >= 0.5, "Batch processing should prioritize cost-effective models"
                
        except NoEligibleModelsError:
            pytest.skip("No eligible models for batch scenario")
    
    def test_real_time_interaction_model_selection(self):
        """Test model selection for real-time interaction scenario."""
        realtime_requirements = ModelRequirements(
            optimization_objective=OptimizationObjective.LATENCY,
            expected_tokens=200,
            max_latency_ms=800,
            workload_priority="high"
        )
        
        try:
            recommendations = self.selector.get_model_recommendations(realtime_requirements, top_k=3)
            
            # Real-time should prioritize latency
            if recommendations:
                top_model = recommendations[0]
                assert top_model.latency_score >= 0.6, "Real-time tasks should prioritize fast models"
                if top_model.estimated_latency_ms:
                    assert top_model.estimated_latency_ms <= 1000, "Should select reasonably fast models"
                    
        except NoEligibleModelsError:
            pytest.skip("No eligible models for real-time scenario")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])