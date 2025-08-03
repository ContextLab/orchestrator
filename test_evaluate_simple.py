"""Simple test to verify evaluate_condition is working."""

import asyncio
from src.orchestrator.control_systems.hybrid_control_system import HybridControlSystem
from src.orchestrator.models.model_registry import ModelRegistry
from src.orchestrator.core.task import Task


async def test_evaluate_condition_handler():
    """Test the _handle_evaluate_condition method directly."""
    print("Testing evaluate_condition handler in HybridControlSystem...\n")
    
    # Create control system
    model_registry = ModelRegistry()
    control_system = HybridControlSystem(model_registry)
    
    # Test cases
    test_cases = [
        # (condition, context, expected_result)
        ("5 > 3", {}, True),
        ("x > 10", {"x": 15}, True),
        ("x > 10", {"x": 5}, False),
        ("enabled", {"enabled": True}, True),
        ("enabled", {"enabled": False}, False),
        ("count == 0", {"count": 0}, True),
        ("count == 0", {"count": 1}, False),
        ("guess < target", {"guess": 25, "target": 50}, True),
        ("guess > target", {"guess": 75, "target": 50}, True),
        ("guess == target", {"guess": 50, "target": 50}, True),
    ]
    
    for condition, context, expected in test_cases:
        # Create a task for evaluate_condition
        task = Task(
            id=f"test_condition_{condition.replace(' ', '_')}",
            name=f"Test: {condition}",
            action="evaluate_condition",
            parameters={"condition": condition}
        )
        
        # Execute through control system
        try:
            result = await control_system.execute_task(task, context)
            actual = result.get("result", False)
            success = actual == expected
            status = "✅" if success else "❌"
            print(f"{status} {condition} with {context} => {actual} (expected {expected})")
            
            if not success:
                print(f"   Full result: {result}")
        except Exception as e:
            print(f"❌ {condition} with {context} => ERROR: {e}")
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    asyncio.run(test_evaluate_condition_handler())