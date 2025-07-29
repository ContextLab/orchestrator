#!/usr/bin/env python3
"""Test all example pipelines and verify their outputs."""

import asyncio
import os
import sys
import json
import glob
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import orchestrator


class ExampleTester:
    def __init__(self):
        self.results = {}
        self.output_dir = Path("test_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
    async def test_example(self, yaml_file: Path) -> Dict[str, Any]:
        """Test a single example pipeline."""
        print(f"\n{'='*60}")
        print(f"Testing: {yaml_file.name}")
        print(f"{'='*60}")
        
        result = {
            "file": str(yaml_file),
            "name": yaml_file.stem,
            "status": "pending",
            "error": None,
            "outputs": [],
            "execution_time": 0
        }
        
        try:
            # Initialize models
            orchestrator.init_models()
            
            # Compile pipeline
            pipeline = await orchestrator.compile_async(str(yaml_file))
            
            # Prepare inputs based on pipeline requirements
            inputs = self.get_default_inputs(yaml_file.stem)
            
            # Run pipeline
            import time
            start_time = time.time()
            
            print(f"Running with inputs: {inputs}")
            outputs = await pipeline.run_async(**inputs)
            
            execution_time = time.time() - start_time
            result["execution_time"] = execution_time
            
            # Save outputs
            output_file = self.output_dir / f"{yaml_file.stem}_output.json"
            with open(output_file, 'w') as f:
                if isinstance(outputs, dict):
                    json.dump(outputs, f, indent=2, default=str)
                else:
                    json.dump({"result": str(outputs)}, f, indent=2)
            
            result["outputs"].append(str(output_file))
            result["status"] = "success"
            
            print(f"✅ Success! Execution time: {execution_time:.2f}s")
            print(f"   Output saved to: {output_file}")
            
            # Check for any generated files
            for pattern in ["*.pdf", "*.md", "*.txt", "*.html", "*.png", "*.jpg"]:
                for generated_file in glob.glob(pattern):
                    if os.path.getmtime(generated_file) > start_time:
                        result["outputs"].append(generated_file)
                        print(f"   Generated file: {generated_file}")
            
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            print(f"❌ Failed: {e}")
            
        return result
    
    def get_default_inputs(self, pipeline_name: str) -> Dict[str, Any]:
        """Get default inputs for different pipeline types."""
        defaults = {
            # Data processing examples
            "simple_data_processing": {
                "data_file": "examples/test_data/sample_data.csv"
            },
            "data_processing_pipeline": {
                "input_file": "examples/test_data/sample_data.csv",
                "output_format": "json"
            },
            "recursive_data_processing": {
                "data_path": "examples/test_data/customers.json",
                "max_depth": 2
            },
            
            # Research examples
            "web_research_pipeline": {
                "topic": "quantum computing applications",
                "max_results": 3
            },
            "working_web_search": {
                "query": "latest AI developments 2025",
                "num_results": 3
            },
            "research_pipeline": {
                "topic": "sustainable energy",
                "depth": "basic"
            },
            
            # Creative examples
            "creative_image_pipeline": {
                "prompt": "a futuristic city with flying cars",
                "style": "digital art"
            },
            "multimodal_processing": {
                "text_input": "Analyze this data",
                "mode": "text"
            },
            
            # Interactive examples (skip interactive prompts)
            "interactive_pipeline": {
                "skip_interaction": True,
                "default_response": "yes"
            },
            
            # Control flow examples
            "control_flow_conditional": {
                "value": 75,
                "threshold": 50
            },
            "control_flow_for_loop": {
                "items": ["apple", "banana", "orange"]
            },
            "control_flow_while_loop": {
                "max_iterations": 3,
                "target_value": 10
            },
            "control_flow_dynamic": {
                "mode": "fast",
                "skip_validation": False
            },
            "control_flow_advanced": {
                "data": [1, 2, 3, 4, 5],
                "operation": "sum"
            },
            
            # Validation examples
            "validation_pipeline": {
                "data": {"name": "test", "age": 25},
                "schema": {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}
            },
            "test_validation_pipeline": {
                "test_data": {"value": 42},
                "expected": 42
            },
            
            # Other examples
            "auto_tags_demo": {
                "task": "summarize",
                "content": "AI is transforming how we work and live."
            },
            "model_routing_demo": {
                "task_type": "creative",
                "budget": 0.10
            },
            "llm_routing_pipeline": {
                "prompt": "Explain quantum physics simply",
                "complexity": "low"
            },
            "terminal_automation": {
                "command": "echo 'Hello from Orchestrator'",
                "safe_mode": True
            },
            "modular_analysis_pipeline": {
                "data_source": "examples/test_data/sample_data.csv",
                "analysis_type": "basic"
            }
        }
        
        # Return defaults for known pipelines, empty dict for others
        return defaults.get(pipeline_name, {})
    
    async def test_all_examples(self):
        """Test all example pipelines."""
        examples_dir = Path("examples")
        yaml_files = list(examples_dir.glob("*.yaml"))
        yaml_files.extend(examples_dir.glob("pipelines/*.yaml"))
        yaml_files.extend(examples_dir.glob("sub_pipelines/*.yaml"))
        
        # Skip MCP examples if MCP is not configured
        skip_patterns = ["mcp_integration", "mcp_memory"]
        
        print(f"Found {len(yaml_files)} example pipelines")
        
        for yaml_file in sorted(yaml_files):
            # Skip certain examples that require special setup
            if any(pattern in yaml_file.name for pattern in skip_patterns):
                print(f"\n⏭️  Skipping {yaml_file.name} (requires special setup)")
                continue
                
            result = await self.test_example(yaml_file)
            self.results[yaml_file.stem] = result
        
        # Print summary
        self.print_summary()
        
        # Save detailed results
        results_file = self.output_dir / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nDetailed results saved to: {results_file}")
    
    def print_summary(self):
        """Print test summary."""
        print(f"\n{'='*60}")
        print("TEST SUMMARY")
        print(f"{'='*60}")
        
        total = len(self.results)
        success = sum(1 for r in self.results.values() if r["status"] == "success")
        failed = sum(1 for r in self.results.values() if r["status"] == "failed")
        
        print(f"Total examples tested: {total}")
        print(f"✅ Successful: {success}")
        print(f"❌ Failed: {failed}")
        
        if failed > 0:
            print("\nFailed examples:")
            for name, result in self.results.items():
                if result["status"] == "failed":
                    print(f"  - {name}: {result['error']}")
        
        print(f"\nSuccess rate: {success/total*100:.1f}%")


async def main():
    """Main entry point."""
    tester = ExampleTester()
    await tester.test_all_examples()


if __name__ == "__main__":
    asyncio.run(main())