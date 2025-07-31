#!/usr/bin/env python3
"""Test ContextManager integration with actual pipelines."""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.orchestrator.core.context_manager import ContextManager
from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator.core.pipeline import Pipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


async def test_research_minimal():
    """Test ContextManager integration with research_minimal pipeline."""
    print("\n=== Testing research_minimal with ContextManager ===")
    
    # Load pipeline
    pipeline_path = Path("examples/research_minimal.yaml")
    yaml_content = pipeline_path.read_text()
    
    # Initialize models
    model_registry = init_models()
    
    # Create orchestrator
    orchestrator = Orchestrator(model_registry=model_registry)
    
    # Replace the template manager with our context manager
    context_manager = ContextManager()
    context_manager.initialize_template_manager(orchestrator.template_manager)
    
    # Monkey-patch to test integration
    original_execute = orchestrator.execute_yaml
    
    async def execute_yaml_with_context(yaml_content, inputs=None):
        """Execute pipeline with context manager."""
        inputs = inputs or {}
        
        # Parse pipeline ID from YAML
        import yaml
        pipeline_data = yaml.safe_load(yaml_content)
        pipeline_id = pipeline_data.get('id', 'unknown')
        
        # Use context manager for pipeline execution
        with context_manager.pipeline_context(pipeline_id, inputs, {}):
            # Log initial context
            print(f"Initial context: {list(context_manager.get_merged_context().keys())}")
            
            # Call original execute
            result = await original_execute(yaml_content, inputs)
            
            # Log final context
            print(f"Final context: {list(context_manager.get_merged_context().keys())}")
            
            return result
    
    orchestrator.execute_yaml = execute_yaml_with_context
    
    # Test inputs
    inputs = {
        "topic": "quantum computing applications"
    }
    
    try:
        # Execute pipeline
        print(f"Executing pipeline with topic: {inputs['topic']}")
        result = await orchestrator.execute_yaml(yaml_content, inputs)
        
        # Check results
        print("\nPipeline completed!")
        print(f"Outputs: {json.dumps(result.get('outputs', {}), indent=2)}")
        
        # Verify template rendering
        if "summary" in result.get("outputs", {}):
            summary = result["outputs"]["summary"]
            # Check that templates were rendered
            assert "{{" not in summary, f"Unrendered template in summary: {summary[:100]}..."
            assert inputs["topic"] in summary.lower(), f"Topic not found in summary"
            print("✓ Templates properly rendered in output")
        
        # Check output file
        output_file = result.get("outputs", {}).get("output_file")
        if output_file and Path(output_file).exists():
            content = Path(output_file).read_text()
            assert "{{" not in content, "Unrendered template in output file"
            assert inputs["topic"] in content.lower(), "Topic not found in output file"
            print("✓ Output file properly rendered")
            
    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True


async def test_control_flow_advanced():
    """Test ContextManager with control_flow_advanced pipeline."""
    print("\n=== Testing control_flow_advanced with ContextManager ===")
    
    # Load pipeline
    pipeline_path = Path("examples/control_flow_advanced.yaml")
    yaml_content = pipeline_path.read_text()
    
    # Initialize models
    model_registry = init_models()
    
    # Create orchestrator
    orchestrator = Orchestrator(model_registry=model_registry)
    
    # Test inputs
    inputs = {
        "input_text": "The impact of artificial intelligence on healthcare",
        "languages": ["es", "fr"],
        "output": "test_outputs/control_flow_advanced"
    }
    
    try:
        # Execute pipeline
        print(f"Executing pipeline with input: {inputs['input_text'][:50]}...")
        result = await orchestrator.execute_yaml(yaml_content, inputs)
        
        print("\nPipeline completed!")
        
        # Check main report
        report_pattern = Path(inputs["output"]) / "*_report.md"
        report_files = list(Path(".").glob(str(report_pattern)))
        if report_files:
            print(f"Found {len(report_files)} report files")
            for report_file in report_files:
                content = report_file.read_text()
                # Check for unrendered templates
                if "{{input_text}}" in content:
                    print(f"❌ Unrendered {{{{input_text}}}} in {report_file}")
                elif "{{output}}" in content:
                    print(f"❌ Unrendered {{{{output}}}} in {report_file}")
                else:
                    print(f"✓ Report properly rendered: {report_file}")
                    
        # Check translations
        trans_pattern = Path(inputs["output"]) / "translations" / "*.txt"
        trans_files = list(Path(".").glob(str(trans_pattern)))
        if trans_files:
            print(f"Found {len(trans_files)} translation files")
            for trans_file in trans_files:
                content = trans_file.read_text()
                if "{{" in content:
                    print(f"❌ Unrendered template in {trans_file}")
                elif "please provide the text" in content.lower():
                    print(f"❌ Generic AI response in {trans_file}")
                else:
                    print(f"✓ Translation properly rendered: {trans_file}")
                    
    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True


async def main():
    """Run integration tests."""
    print("Starting ContextManager integration tests...")
    
    # Test with research_minimal
    minimal_ok = await test_research_minimal()
    
    # Test with control_flow_advanced
    advanced_ok = await test_control_flow_advanced()
    
    print("\n=== Summary ===")
    print(f"research_minimal: {'✅ PASSED' if minimal_ok else '❌ FAILED'}")
    print(f"control_flow_advanced: {'✅ PASSED' if advanced_ok else '❌ FAILED'}")
    
    return minimal_ok and advanced_ok


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)