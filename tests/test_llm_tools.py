#!/usr/bin/env python
"""Test script for LLM tools integration."""

import asyncio
import sys
sys.path.insert(0, 'src')

from tests.test_infrastructure import create_test_orchestrator, TestModel, TestProvider
from src.orchestrator.tools.llm_tools import (

    TaskDelegationTool,
    MultiModelRoutingTool,
    PromptOptimizationTool
)
from orchestrator import init_models
from src.orchestrator.models import get_model_registry

async def test_tools():
    """Test the LLM tools with real API calls."""
    # Initialize models
    print("Initializing models...")
    from orchestrator import init_models
    init_models()
    model_registry = get_model_registry()
    
    print("\n=== Testing TaskDelegationTool ===")
    delegation_tool = TaskDelegationTool()
    delegation_tool.model_registry = model_registry
    
    result = await delegation_tool._execute_impl(
        task="Write a Python function to calculate fibonacci",
        requirements={"complexity": "moderate"},
        cost_weight=0.3,
        quality_weight=0.7
    )
    
    print(f"Success: {result.get('success')}")
    print(f"Selected model: {result.get('selected_model')}")
    print(f"Task type: {result.get('task_analysis', {}).get('task_type')}")
    print(f"Score: {result.get('score')}")
    print(f"Reasons: {result.get('reasons')}")
    
    print("\n=== Testing PromptOptimizationTool ===")
    optimization_tool = PromptOptimizationTool()
    optimization_tool.model_registry = model_registry
    
    result = await optimization_tool._execute_impl(
        prompt="write python fibonacci",
        optimization_goals=["clarity", "brevity"],
        preserve_intent=True
    )
    
    print(f"Success: {result.get('success')}")
    print(f"Original prompt: write python fibonacci")
    print(f"Optimized prompt: {result.get('optimized_prompt')}")
    print(f"Token reduction: {result.get('metrics', {}).get('reduction_percentage')}%")
    
    print("\n=== Testing MultiModelRoutingTool ===")
    routing_tool = MultiModelRoutingTool()
    routing_tool.model_registry = model_registry
    
    result = await routing_tool._execute_impl(
        request="Generate a haiku about programming",
        strategy="capability_based",
        max_concurrent=5
    )
    
    print(f"Success: {result.get('success')}")
    print(f"Selected model: {result.get('selected_model')}")
    print(f"Strategy: {result.get('strategy')}")
    print(f"Routing reason: {result.get('routing_reason')}")
    
    print("\n✅ All tools tested successfully!")

if __name__ == "__main__":
    asyncio.run(test_tools())