"""Test control flow while loop pipeline."""
import asyncio
import yaml
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.orchestrator.models import get_model_registry, set_model_registry, ModelRegistry
from src.orchestrator.control_systems import HybridControlSystem
from src.orchestrator.engine.control_flow_engine import ControlFlowEngine
from src.orchestrator.core.model import ModelCapabilities, ModelRequirements, ModelMetrics, ModelCost

async def test_while_loop():
    """Test while loop pipeline."""
    # Load pipeline
    pipeline_path = Path("examples/control_flow_while_loop.yaml")
    with open(pipeline_path) as f:
        pipeline_def = yaml.safe_load(f)
    
    # Create a simple mock model
    class SimpleMockModel:
        def __init__(self):
            self.name = "test-llm"
            self.model_name = "test-llm"
            self.provider = "mock"
            self.capabilities = ModelCapabilities(supported_tasks=["generate"])
            self.requirements = ModelRequirements()
            self.metrics = ModelMetrics()
            self.cost = ModelCost(is_free=True)
            self._is_available = True
            self._expertise = ["general"]
            
        async def generate(self, **kwargs):
            """Generate mock response."""
            prompt = kwargs.get("prompt", "")
            # Return the target number to make the loop end
            if "guess" in prompt.lower():
                return "42"
            if "attempt" in prompt.lower():
                return "Attempt complete"
            return "Starting number guessing game. Target is 42"
            
        def meets_requirements(self, requirements: dict) -> bool:
            """Check if model meets requirements."""
            return True
    
    # Create model registry with mock models
    model_registry = ModelRegistry()
    mock_llm = SimpleMockModel()
    model_registry.register_model(mock_llm)
    
    # Set the global registry
    set_model_registry(model_registry)
    
    # Create control flow engine
    engine = ControlFlowEngine(model_registry=model_registry)
    
    # Test inputs
    test_cases = [
        {"target_number": 42, "max_attempts": 10},
        {"target_number": 7, "max_attempts": 5},
    ]
    
    for inputs in test_cases:
        print(f"\n{'='*60}")
        print(f"Testing while loop with inputs: {inputs}")
        print('='*60)
        
        try:
            # Run pipeline using engine
            result = await engine.execute_yaml(
                yaml.dump(pipeline_def),
                inputs=inputs
            )
            
            print(f"\nPipeline completed!")
            print(f"Result: {result}")
            
            # Check outputs
            output_file = Path("examples/outputs/control_flow_while_loop/result.txt")
            if output_file.exists():
                print(f"\nOutput file content:")
                print(output_file.read_text())
            
            # Check log files
            log_dir = Path("examples/outputs/control_flow_while_loop/logs")
            if log_dir.exists():
                log_files = list(log_dir.glob("*.txt"))
                print(f"\nFound {len(log_files)} log files")
                for log_file in sorted(log_files):
                    print(f"\n{log_file.name}:")
                    print(log_file.read_text())
                    
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_while_loop())