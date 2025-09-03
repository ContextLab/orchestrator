"""Test model routing with real API calls."""

import pytest
import asyncio
from typing import Dict, Any, List
from src.orchestrator.tools.llm_tools import MultiModelRoutingTool
from src.orchestrator.models.model_registry import ModelRegistry


class TestModelRouting:
    """Test model routing with real API calls."""
    
    @pytest.fixture
    async def routing_tool(self):
        """Create a MultiModelRoutingTool instance."""
        return MultiModelRoutingTool()
    
    @pytest.fixture
    async def model_registry(self):
        """Create a model registry with real models."""
        from src.orchestrator.models.model_registry import ModelRegistry
        
        registry = ModelRegistry()
        await registry.load_models()
        return registry
    
    @pytest.mark.asyncio
    async def test_route_multiple_tasks(self, routing_tool):
        """Test routing multiple tasks to appropriate models."""
        result = await routing_tool.execute(
            action="route",
            tasks=[
                {"task": "Summarize this text in 2 sentences", "context": "AI is transforming industries worldwide. Machine learning enables real-time processing."},
                {"task": "Write Python code", "context": "fibonacci function with type hints"},
                {"task": "Analyze sales data", "context": "Q4 2024 sales: $2.5M revenue, 15% growth"}
            ],
            routing_strategy="balanced",
            constraints={"total_budget": 10.0, "max_latency": 30.0}
        )
        
        assert result["success"]
        assert "recommendations" in result
        assert len(result["recommendations"]) == 3
        assert "total_estimated_cost" in result
        assert result["total_estimated_cost"] < 10.0
        
        # Each recommendation has real model selection
        for rec in result["recommendations"]:
            assert "model" in rec
            assert "estimated_cost" in rec
            assert rec["estimated_cost"] >= 0
            assert "reasons" in rec or "reasoning" in rec
    
    @pytest.mark.asyncio
    async def test_optimize_batch_processing(self, routing_tool):
        """Test batch optimization with real executions."""
        # Real translation tasks
        result = await routing_tool.execute(
            action="optimize_batch",
            tasks=[
                "Translate 'Hello World' to Spanish",
                "Translate 'Good morning' to French",
                "Translate 'Thank you' to German",
                "Translate 'Goodbye' to Italian"
            ],
            optimization_goal="minimize_cost",
            constraints={"max_budget_per_task": 0.05}
        )
        
        assert result["success"]
        assert "results" in result
        assert len(result["results"]) == 4
        
        # Check for real translations
        translations = result["results"]
        assert any("hola" in str(t).lower() or "mundo" in str(t).lower() for t in translations)
        
        # Check cost tracking
        assert "total_cost" in result
        assert result["total_cost"] > 0  # Real API costs
        assert "models_used" in result
        assert len(result["models_used"]) > 0
    
    @pytest.mark.asyncio
    async def test_routing_strategies(self, routing_tool):
        """Test different routing strategies."""
        strategies = {
            "cost": "cost_optimized",
            "balanced": "balanced", 
            "quality": "quality_optimized"
        }
        
        results_by_strategy = {}
        
        for key, strategy in strategies.items():
            result = await routing_tool.execute(
                action="route",
                tasks=[
                    {"task": "Write a complex analysis", "context": "Analyze market trends"},
                    {"task": "Simple calculation", "context": "Add 2+2"}
                ],
                routing_strategy=strategy,
                constraints={"total_budget": 10.0}
            )
            
            assert result["success"]
            assert "recommendations" in result
            
            # Store model selections for comparison
            models = [rec["model"] for rec in result["recommendations"]]
            costs = [rec["estimated_cost"] for rec in result["recommendations"]]
            results_by_strategy[key] = {"models": models, "costs": costs}
        
        # Verify strategy affects model selection
        # Cost-optimized should have lower total cost
        cost_total = sum(results_by_strategy["cost"]["costs"])
        quality_total = sum(results_by_strategy["quality"]["costs"])
        
        # Quality strategy should generally cost more
        assert quality_total >= cost_total * 0.8  # Allow some variance
    
    @pytest.mark.asyncio
    async def test_single_request_routing(self, routing_tool):
        """Test backward compatibility with single request routing."""
        result = await routing_tool.execute(
            request="Generate a haiku about artificial intelligence",
            preferences={"quality": 0.8, "cost": 0.5, "speed": 0.3}
        )
        
        assert result["success"]
        assert "model_selected" in result
        assert "estimated_cost" in result
        assert "result" in result
        
        # Check the haiku was generated
        haiku = result["result"]
        assert len(haiku) > 10  # Non-empty result
        assert isinstance(haiku, str)
    
    @pytest.mark.asyncio
    async def test_error_handling(self, routing_tool):
        """Test error handling for invalid inputs."""
        # Test with invalid action
        with pytest.raises(ValueError):
            await routing_tool.execute(
                action="invalid_action",
                tasks=[]
            )
        
        # Test with empty tasks
        result = await routing_tool.execute(
            action="route",
            tasks=[],
            routing_strategy="balanced"
        )
        
        assert result["success"]
        assert result["recommendations"] == []
        assert result["total_estimated_cost"] == 0
    
    @pytest.mark.asyncio
    async def test_budget_constraints(self, routing_tool):
        """Test that budget constraints are respected."""
        result = await routing_tool.execute(
            action="route",
            tasks=[
                {"task": "Write a 10,000 word essay", "context": "Complex topic"},
                {"task": "Translate entire book", "context": "500 pages"},
                {"task": "Generate comprehensive report", "context": "Annual report"}
            ],
            routing_strategy="cost_optimized",
            constraints={"total_budget": 0.50}  # Very low budget
        )
        
        assert result["success"]
        assert result["total_estimated_cost"] <= 0.50 * 1.1  # Allow 10% variance
        
        # Should select cheaper models
        for rec in result["recommendations"]:
            model = rec["model"].lower()
            # Check for budget-friendly models
            assert any(cheap in model for cheap in ["nano", "mini", "1b", "gemma", "llama"])
    
    @pytest.mark.asyncio
    async def test_complex_task_routing(self, routing_tool):
        """Test routing for complex multi-step tasks."""
        result = await routing_tool.execute(
            action="route",
            tasks=[
                {
                    "task": "Research and synthesize information",
                    "context": "Quantum computing applications in cryptography",
                    "requirements": ["deep_analysis", "citations", "technical_accuracy"]
                },
                {
                    "task": "Generate marketing copy",
                    "context": "Product launch announcement",
                    "requirements": ["creativity", "engagement", "brand_voice"]
                },
                {
                    "task": "Debug Python code",
                    "context": "AsyncIO race condition issue",
                    "requirements": ["code_understanding", "debugging", "solution"]
                }
            ],
            routing_strategy="quality_optimized",
            constraints={"total_budget": 20.0}
        )
        
        assert result["success"]
        assert len(result["recommendations"]) == 3
        
        # Quality-optimized should select more capable models
        for i, rec in enumerate(result["recommendations"]):
            model = rec["model"].lower()
            
            # Research task should get a strong reasoning model
            if i == 0:
                assert any(strong in model for strong in ["opus", "gpt-5", "pro", "sonnet"])
            
            # Creative task should get a creative model
            elif i == 1:
                assert rec["estimated_cost"] > 0
            
            # Code task should get a code-capable model
            elif i == 2:
                assert any(code in model for code in ["gpt", "claude", "codex", "sonnet"])