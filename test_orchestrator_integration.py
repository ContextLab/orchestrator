#!/usr/bin/env python3
"""
Simple test to verify orchestrator template integration.
"""

import asyncio
from orchestrator.orchestrator import Orchestrator
from orchestrator.core.pipeline import Pipeline
from orchestrator.core.task import Task
from orchestrator.core.template_manager import TemplateManager
from orchestrator.models.model_registry import ModelRegistry
from orchestrator.control_systems.hybrid_control_system import HybridControlSystem

async def test_orchestrator_templates():
    """Test orchestrator with template integration."""
    print("Testing orchestrator template integration...")
    
    # Create a simple pipeline with templates
    pipeline = Pipeline(
        id="test-pipeline",
        name="Test Pipeline",
        context={"title": "Research Report", "author": "AI Assistant"}
    )
    
    # Add a filesystem task that uses templates
    task = Task(
        id="save-file",
        name="Save File Task",
        action="write",  # FileSystem action
        parameters={
            "path": "/tmp/{{title | slugify}}.md",
            "content": "# {{title}}\n\nAuthor: {{author}}\n\nThis is a test report."
        },
        metadata={"tool": "filesystem"}  # Specify to use filesystem tool
    )
    pipeline.add_task(task)
    
    # Create orchestrator with minimal setup
    try:
        # Create model registry (empty is fine for tool-only tasks)
        model_registry = ModelRegistry()
        
        # Create control system
        control_system = HybridControlSystem(model_registry)
        
        # Create template manager
        template_manager = TemplateManager(debug_mode=True)
        
        # Create orchestrator
        orchestrator = Orchestrator(
            model_registry=model_registry,
            control_system=control_system,
            template_manager=template_manager
        )
        
        # Execute pipeline
        results = await orchestrator.execute_pipeline(pipeline)
        
        print(f"Pipeline results: {results}")
        
        # Check if the task succeeded
        if "save-file" in results:
            result = results["save-file"]
            if isinstance(result, dict) and result.get("success"):
                print("✅ Pipeline execution successful!")
                
                # Check file content
                filepath = result.get("path") or result.get("filepath")
                if filepath:
                    with open(filepath, "r") as f:
                        content = f.read()
                        print(f"Generated file content:\n{content}")
                        if "Research Report" in content and "AI Assistant" in content:
                            print("✅ Templates were rendered correctly in pipeline!")
                        else:
                            print("❌ Templates were not rendered correctly in pipeline!")
                else:
                    print("❌ No file path returned!")
            else:
                print(f"❌ Task failed: {result}")
        else:
            print("❌ Task results not found!")
            
    except Exception as e:
        print(f"❌ Error testing orchestrator: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_orchestrator_templates())