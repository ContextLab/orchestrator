"""Test evaluate_condition with actual pipelines."""

import asyncio
import yaml
from pathlib import Path

from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator.models.model_registry import ModelRegistry
from src.orchestrator.control_systems.hybrid_control_system import HybridControlSystem


async def test_control_flow_while_loop():
    """Test the control_flow_while_loop.yaml pipeline with evaluate_condition."""
    # Load the pipeline
    pipeline_path = Path("examples/control_flow_while_loop.yaml")
    with open(pipeline_path, "r") as f:
        pipeline_yaml = f.read()
    
    # Set up the orchestrator
    model_registry = ModelRegistry()
    control_system = HybridControlSystem(model_registry)
    
    orchestrator = Orchestrator(
        control_systems={"hybrid": control_system},
        model_registry=model_registry
    )
    
    # Execute the pipeline
    print(f"Executing pipeline: {pipeline_path}")
    
    try:
        # Create a pipeline spec from YAML
        pipeline_spec = yaml.safe_load(pipeline_yaml)
        
        # Execute with a timeout
        result = await asyncio.wait_for(
            orchestrator.execute_pipeline(pipeline_spec),
            timeout=30.0  # 30 second timeout
        )
        
        print(f"Pipeline executed successfully!")
        print(f"Result: {result}")
        
        # Check that the pipeline completed
        assert result is not None
        print("\n✅ Test passed!")
        
    except asyncio.TimeoutError:
        print("\n❌ Pipeline timed out after 30 seconds")
        raise
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        raise


async def test_simple_condition():
    """Test a simple condition evaluation."""
    from src.orchestrator.actions.condition_evaluator import get_condition_evaluator
    
    # Test various conditions
    test_cases = [
        ("5 > 3", {}, True),
        ("x > 10", {"x": 15}, True),
        ("x > 10", {"x": 5}, False),
        ("enabled and x > 5", {"enabled": True, "x": 10}, True),
        ("enabled and x > 5", {"enabled": False, "x": 10}, False),
        ("(a > 0 or b < 0) and enabled", {"a": 5, "b": 10, "enabled": True}, True),
    ]
    
    print("\nTesting simple conditions:")
    for condition, context, expected in test_cases:
        evaluator = get_condition_evaluator(condition, context)
        result = await evaluator.execute(condition=condition, context=context)
        success = result["result"] == expected
        status = "✅" if success else "❌"
        print(f"{status} {condition} with {context} => {result['result']} (expected {expected})")
        assert success, f"Failed: {condition}"


if __name__ == "__main__":
    print("Testing evaluate_condition integration...\n")
    
    # Run simple condition tests first
    asyncio.run(test_simple_condition())
    
    # Then try the full pipeline
    print("\n" + "="*60 + "\n")
    asyncio.run(test_control_flow_while_loop())