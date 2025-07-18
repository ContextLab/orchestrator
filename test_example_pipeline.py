#!/usr/bin/env python3
"""Test script to run an example pipeline with proper initialization."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.orchestrator import init_models, Orchestrator
from src.orchestrator.control_systems.model_based_control_system import ModelBasedControlSystem
from src.orchestrator.control_systems.tool_integrated_control_system import ToolIntegratedControlSystem


async def test_research_assistant():
    """Test the research assistant pipeline."""
    print("🚀 Testing Research Assistant Pipeline")
    print("=" * 60)
    
    # Initialize models from models.yaml
    print("\n📦 Initializing models...")
    model_registry = init_models()
    
    # Create a control system that handles both model-based and tool-based tasks
    class HybridControlSystem(ModelBasedControlSystem):
        """Control system that handles both AI models and file operations."""
        
        def __init__(self, model_registry):
            super().__init__(model_registry)
            # Extend supported actions
            self.config["capabilities"]["supported_actions"].extend([
                "save_output", "save_to_file", "write_file", "read_file"
            ])
            
        async def execute_task(self, task, context):
            """Execute task with extended support for file operations."""
            # Handle save_output action
            if task.action == "save_output":
                return await self._handle_save_output(task, context)
            
            # Default to parent implementation
            return await super().execute_task(task, context)
            
        async def _handle_save_output(self, task, context):
            """Handle save_output action."""
            import json
            from pathlib import Path
            
            # Get parameters
            content = task.parameters.get("content", "")
            filename = task.parameters.get("filename", "output.md")
            output_dir = task.parameters.get("output_dir", "examples/output")
            
            # Resolve content from previous results if needed
            if isinstance(content, str) and content.startswith("{{") and content.endswith("}}"):
                # Extract step reference
                ref = content.strip("{{").strip("}}")
                if "." in ref:
                    step_id, field = ref.split(".", 1)
                    if "previous_results" in context and step_id in context["previous_results"]:
                        result = context["previous_results"][step_id]
                        # Navigate nested fields
                        for part in field.split("."):
                            if isinstance(result, dict) and part in result:
                                result = result[part]
                            else:
                                result = str(result)
                                break
                        content = str(result)
            
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Write file
            file_path = output_path / filename
            
            # If content is a dict, convert to formatted string
            if isinstance(content, dict):
                if "content" in content:
                    content = content["content"]
                else:
                    content = json.dumps(content, indent=2)
            
            file_path.write_text(str(content))
            
            return {
                "success": True,
                "filepath": str(file_path),
                "size": len(str(content)),
                "message": f"Saved output to {file_path}"
            }
    
    # Create orchestrator with hybrid control system
    control_system = HybridControlSystem(model_registry)
    orchestrator = Orchestrator(
        model_registry=model_registry,
        control_system=control_system
    )
    
    # Load and execute pipeline
    pipeline_path = Path("examples/research_assistant.yaml")
    print(f"\n📄 Loading pipeline: {pipeline_path}")
    
    # Define inputs
    inputs = {
        "query": "quantum computing applications in cryptography",
        "output_dir": "examples/output"
    }
    
    print(f"\n📥 Inputs: {inputs}")
    
    try:
        # Execute pipeline
        print("\n⚡ Executing pipeline...")
        results = await orchestrator.execute_yaml_file(
            str(pipeline_path),
            context=inputs
        )
        
        print("\n✅ Pipeline completed successfully!")
        print("\n📊 Results:")
        for task_id, result in results.items():
            if isinstance(result, dict) and "filepath" in result:
                print(f"  - {task_id}: Saved to {result['filepath']}")
            elif isinstance(result, str) and len(result) > 100:
                print(f"  - {task_id}: {result[:100]}...")
            else:
                print(f"  - {task_id}: {result}")
                
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_research_assistant())