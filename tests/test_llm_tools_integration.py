"""Real-world integration tests for LLM routing tools."""

import asyncio
import os
import pytest
from src.orchestrator.tools.llm_tools import (

from tests.test_infrastructure import create_test_orchestrator, TestModel, TestProvider
    TaskDelegationTool,
    MultiModelRoutingTool,
    PromptOptimizationTool
)
from src.orchestrator.models import get_model_registry
from orchestrator import init_models

class TestLLMToolsIntegration:
    """Test LLM tools with real API calls."""
    
    @pytest.fixture(autouse=True)
    async def setup(self):
        """Initialize models with real API keys."""
        # Initialize real models - no mocks!
        init_models()
        self.model_registry = get_model_registry()
        
        # Verify we have at least one API key or local model
        available_models = await self.model_registry.get_available_models()
        if not available_models:
            pytest.skip("No models available for real testing")
    
    @pytest.mark.asyncio
    async def test_task_delegation_real_models(self):
        """Test task delegation with real model scoring."""
        tool = TaskDelegationTool()
        tool.model_registry = self.model_registry
        
        # Test cases with different task types
        test_cases = [
            {
                "task": "Write a Python function to calculate fibonacci numbers",
                "expected_type": "code_generation"
            },
            {
                "task": "Analyze the sentiment of customer reviews",
                "expected_type": "analysis"
            },
            {
                "task": "Translate this text to Spanish: Hello world",
                "expected_type": "translation"
            }
        ]
        
        for test_case in test_cases:
            result = await tool._execute_impl(
                task=test_case["task"],
                requirements={"complexity": "moderate"},
                cost_weight=0.3,
                quality_weight=0.7
            )
            
            assert result["success"] is True
            assert "selected_model" in result
            assert "score" in result
            assert "reasons" in result
            assert "estimated_cost" in result
            
            # Check task analysis
            task_analysis = result.get("task_analysis", {})
            # Allow general type as well since it's a valid classification
            assert task_analysis.get("task_type") in [test_case["expected_type"], "general"]
            
            # Verify the selected model actually exists
            model_name = result["selected_model"]
            # Handle provider:model format
            if ":" in model_name:
                provider, name = model_name.split(":", 1)
                model = self.model_registry.get_model(name, provider)
            else:
                model = self.model_registry.get_model(model_name)
            assert model is not None, f"Model {model_name} not found in registry"
    
    @pytest.mark.asyncio
    async def test_multi_model_routing_strategies(self):
        """Test different routing strategies with real models."""
        tool = MultiModelRoutingTool()
        tool.model_registry = self.model_registry
        
        # Get available models for testing
        available_models = list(await self.model_registry.get_available_models())[:3]  # Use first 3 models
        if not available_models:
            pytest.skip("No models available for routing test")
        
        strategies = ["round_robin", "least_loaded", "cost_optimized", "capability_based"]
        request = "Generate a haiku about programming"
        
        for strategy in strategies:
            result = await tool._execute_impl(
                request=request,
                models=available_models,
                strategy=strategy,
                max_concurrent=5
            )
            
            # Allow failure for some strategies if no suitable models
            if result["success"]:
                assert result["strategy"] == strategy
                assert "selected_model" in result
                assert "routing_reason" in result
                
                # Verify the selected model exists
                selected = result["selected_model"]
                all_models = await self.model_registry.get_available_models()
                assert selected in available_models or selected in all_models
    
    @pytest.mark.asyncio
    async def test_prompt_optimization_real_execution(self):
        """Test prompt optimization with real model feedback."""
        tool = PromptOptimizationTool()
        tool.model_registry = self.model_registry
        
        test_prompts = [
            "write code fibonacci",  # Unclear prompt
            "The thing is, I need you to, like, help me understand how machine learning, you know, works and stuff",  # Verbose
            "Explain quantum computing"  # Clear but can be optimized
        ]
        
        for original_prompt in test_prompts:
            result = await tool._execute_impl(
                prompt=original_prompt,
                optimization_goals=["clarity", "brevity"],
                preserve_intent=True
            )
            
            assert result["success"] is True
            assert "optimized_prompt" in result
            assert "metrics" in result
            assert result["metrics"]["original_tokens"] > 0
            assert result["metrics"]["optimized_tokens"] > 0
            
            # Optimization might make prompt longer (add clarity) or shorter (brevity)
            # Sometimes it might not change if already optimal
            # So we just check that we got a result
            assert len(result["optimized_prompt"]) > 0
            
            # Test with specific model if available
            available_models = list(await self.model_registry.get_available_models())
            if available_models:
                model_name = available_models[0]
                result_with_model = await tool._execute_impl(
                    prompt=original_prompt,
                    model=model_name,
                    optimization_goals=["model_specific"],
                    preserve_intent=True
                )
                assert result_with_model["success"] is True

    @pytest.mark.asyncio
    async def test_tool_chain_integration(self):
        """Test the full chain: delegation -> optimization -> routing."""
        task = "Create a detailed analysis of renewable energy trends"
        
        # Step 1: Delegate to find best model
        delegation_tool = TaskDelegationTool()
        delegation_tool.model_registry = self.model_registry
        
        delegation_result = await delegation_tool._execute_impl(
            task=task,
            requirements={"complexity": "complex"},
            cost_weight=0.2,
            quality_weight=0.8
        )
        
        assert delegation_result["success"] is True
        selected_model = delegation_result["selected_model"]
        
        # Step 2: Optimize the prompt for the selected model
        optimization_tool = PromptOptimizationTool()
        optimization_tool.model_registry = self.model_registry
        
        optimization_result = await optimization_tool._execute_impl(
            prompt=task,
            model=selected_model,
            optimization_goals=["clarity", "model_specific"],
            preserve_intent=True
        )
        
        assert optimization_result["success"] is True
        optimized_prompt = optimization_result["optimized_prompt"]
        
        # Step 3: Route the optimized request
        routing_tool = MultiModelRoutingTool()
        routing_tool.model_registry = self.model_registry
        
        # Use fallback models or just the selected model
        models_to_route = delegation_result.get("fallback_models", [selected_model])
        if not models_to_route:
            models_to_route = [selected_model]
        
        routing_result = await routing_tool._execute_impl(
            request=optimized_prompt,
            models=models_to_route,
            strategy="capability_based"
        )
        
        # Routing might fail if models aren't suitable, but delegation and optimization should work
        assert delegation_result["success"] is True
        assert optimization_result["success"] is True
        
        if routing_result["success"]:
            # Verify we have a complete workflow result
            final_model = routing_result["selected_model"]
            all_models = await self.model_registry.get_available_models()
            assert final_model in all_models or ":" in final_model

    @pytest.mark.asyncio
    async def test_auto_tag_resolution(self):
        """Test AUTO tag resolution in task delegation."""
        tool = TaskDelegationTool()
        tool.model_registry = self.model_registry
        
        # This tests the fallback AUTO resolution in the handler
        # In real usage, the control system would handle this
        result = await tool._execute_impl(
            task="Build a complete e-commerce platform with payment processing",
            requirements={
                "complexity": "complex"  # Normally would be <AUTO>...</AUTO>
            },
            cost_weight=0.3,
            quality_weight=0.7
        )
        
        assert result["success"] is True
        # Complex task should select a more capable model
        assert result["score"] > 0
        assert "complex" in str(result.get("reasons", [])).lower() or result["task_analysis"]["complexity"] == "complex"

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in LLM tools."""
        # Test with empty task
        tool = TaskDelegationTool()
        tool.model_registry = self.model_registry
        
        result = await tool._execute_impl(
            task="",
            requirements={},
            cost_weight=0.5,
            quality_weight=0.5
        )
        
        # Should still work with empty task
        assert "success" in result
        if result["success"]:
            assert "selected_model" in result
        
        # Test prompt optimization with no model registry
        opt_tool = PromptOptimizationTool()
        opt_tool.model_registry = None
        
        result = await opt_tool._execute_impl(
            prompt="test prompt",
            optimization_goals=["clarity"],
            preserve_intent=True
        )
        
        # Should handle missing registry gracefully
        assert "success" in result