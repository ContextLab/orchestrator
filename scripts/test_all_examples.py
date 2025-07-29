#!/usr/bin/env python3
"""Test all example pipelines systematically."""

import asyncio
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Reduce logging
os.environ['ORCHESTRATOR_LOG_LEVEL'] = 'WARNING'

import orchestrator
from orchestrator.utils.api_keys_flexible import load_api_keys_optional


class ExampleTester:
    def __init__(self):
        self.results = []
        self.output_dir = Path("test_outputs") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def get_test_inputs(self, example_name: str) -> Dict[str, Any]:
        """Get appropriate test inputs for each example."""
        
        # Create test data directory
        test_data_dir = Path("examples/test_data")
        test_data_dir.mkdir(exist_ok=True)
        
        # Create test files if needed
        self._create_test_files(test_data_dir)
        
        # Map example names to test inputs
        inputs = {
            # AUTO tags examples
            "auto_tags_demo": {
                "data_file": str(test_data_dir / "sales_data.csv"),
                "analysis_goal": "identify top selling products and customer patterns",
                "output_format_preference": "executive_summary"
            },
            
            # Model routing
            "model_routing_demo": {
                "task_budget": 5.00,
                "priority": "balanced"
            },
            
            "llm_routing_pipeline": {
                "task": "Analyze the environmental impact of electric vehicles",
                "optimization_goals": ["clarity", "accuracy"],
                "routing_strategy": "capability_based"
            },
            
            # Control flow examples
            "control_flow_conditional": {
                "value": 85,
                "threshold": 70
            },
            
            "control_flow_for_loop": {
                "items": ["apple", "banana", "cherry", "date"]
            },
            
            "control_flow_while_loop": {
                "max_iterations": 5,
                "target_value": 100
            },
            
            "control_flow_dynamic": {
                "mode": "fast",
                "skip_validation": False
            },
            
            "control_flow_advanced": {
                "data": [10, 25, 30, 45, 50],
                "operation": "average"
            },
            
            # Data processing
            "simple_data_processing": {
                "data_file": str(test_data_dir / "sales_data.csv")
            },
            
            "data_processing_pipeline": {
                "input_file": str(test_data_dir / "sales_data.csv"),
                "output_format": "json",
                "quality_threshold": 0.90,
                "enable_profiling": True
            },
            
            "recursive_data_processing": {
                "data_path": str(test_data_dir / "customers.json"),
                "max_depth": 3
            },
            
            "modular_analysis_pipeline": {
                "data_source": str(test_data_dir / "sales_data.csv"),
                "analysis_type": "statistical"
            },
            
            # Web and research
            "web_research_pipeline": {
                "research_topic": "renewable energy storage solutions",
                "max_sources": 3,
                "output_format": "markdown",
                "research_depth": "standard"
            },
            
            "working_web_search": {
                "query": "latest breakthroughs in quantum computing 2024",
                "num_results": 3
            },
            
            "research_pipeline": {
                "topic": "artificial intelligence ethics",
                "depth": "quick"
            },
            
            # Creative and multimodal
            "creative_image_pipeline": {
                "prompt": "a serene mountain landscape at sunset",
                "style": "impressionist",
                "resolution": "512x512"
            },
            
            "multimodal_processing": {
                "text_input": "Analyze this quarterly sales report",
                "mode": "text_only"
            },
            
            # Interactive (non-interactive defaults)
            "interactive_pipeline": {
                "skip_prompts": True,
                "default_choice": "yes"
            },
            
            # Validation
            "validation_pipeline": {
                "data": {"name": "Test User", "email": "test@example.com", "age": 30},
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "email": {"type": "string", "format": "email"},
                        "age": {"type": "integer", "minimum": 0, "maximum": 150}
                    },
                    "required": ["name", "email"]
                }
            },
            
            "test_validation_pipeline": {
                "test_data": {"value": 42, "flag": True},
                "expected": {"value": 42}
            },
            
            # System tools
            "terminal_automation": {
                "commands": ["echo 'Test automation'", "date", "pwd"],
                "safe_mode": True
            },
            
            # Sub-pipelines
            "statistical_analysis": {
                "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "confidence_level": 0.95
            },
            
            # Code optimization
            "code_optimization": {
                "code": '''def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total''',
                "language": "python",
                "optimization_level": "basic"
            },
            
            # Simple research
            "simple_research": {
                "topic": "machine learning basics",
                "max_results": 3
            },
            
            # Research report template
            "research-report-template": {
                "topic": "blockchain technology",
                "sections": ["introduction", "overview", "conclusion"]
            }
        }
        
        return inputs.get(example_name, {})
    
    def _create_test_files(self, test_data_dir: Path):
        """Create test data files."""
        
        # Sales data CSV
        sales_csv = test_data_dir / "sales_data.csv"
        if not sales_csv.exists():
            with open(sales_csv, 'w') as f:
                f.write("order_id,customer_id,product_name,quantity,unit_price,order_date,status\n")
                f.write("ORD-000001,CUST-001,Widget A,5,19.99,2024-01-15,delivered\n")
                f.write("ORD-000002,CUST-002,Widget B,3,29.99,2024-01-16,shipped\n")
                f.write("ORD-000003,CUST-001,Widget C,1,49.99,2024-01-17,processing\n")
                f.write("ORD-000004,CUST-003,Widget A,10,19.99,2024-01-18,delivered\n")
                f.write("ORD-000005,CUST-002,Widget D,2,39.99,2024-01-19,pending\n")
        
        # Customers JSON
        customers_json = test_data_dir / "customers.json"
        if not customers_json.exists():
            customers = {
                "customers": [
                    {"id": "CUST-001", "name": "Alice Smith", "orders": 15, "total_spent": 1250.50},
                    {"id": "CUST-002", "name": "Bob Jones", "orders": 8, "total_spent": 750.25},
                    {"id": "CUST-003", "name": "Carol White", "orders": 22, "total_spent": 2100.00}
                ]
            }
            with open(customers_json, 'w') as f:
                json.dump(customers, f, indent=2)
    
    async def test_example(self, yaml_path: Path) -> Dict[str, Any]:
        """Test a single example pipeline."""
        example_name = yaml_path.stem
        
        print(f"\n{'='*70}")
        print(f"Testing: {example_name}")
        print(f"File: {yaml_path}")
        print(f"{'='*70}")
        
        result = {
            "name": example_name,
            "file": str(yaml_path),
            "status": "pending",
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "duration": None,
            "error": None,
            "outputs": [],
            "output_files": []
        }
        
        # Create example-specific output directory
        example_output_dir = self.output_dir / example_name
        example_output_dir.mkdir(exist_ok=True)
        
        try:
            # Get test inputs
            inputs = self.get_test_inputs(example_name)
            print(f"Inputs: {json.dumps(inputs, indent=2)}")
            
            # Compile pipeline
            print("Compiling pipeline...")
            pipeline = await orchestrator.compile_async(str(yaml_path.absolute()))
            
            # Set environment variable for output directory
            os.environ['ORCHESTRATOR_OUTPUT_DIR'] = str(example_output_dir.absolute())
            
            # Run pipeline
            print("Running pipeline...")
            start = datetime.now()
            
            outputs = await pipeline.run_async(**inputs)
            
            end = datetime.now()
            duration = (end - start).total_seconds()
            
            # Clear environment variable
            os.environ.pop('ORCHESTRATOR_OUTPUT_DIR', None)
            
            # Save outputs
            output_file = example_output_dir / "pipeline_output.json"
            with open(output_file, 'w') as f:
                if isinstance(outputs, dict):
                    json.dump(outputs, f, indent=2, default=str)
                else:
                    json.dump({"result": str(outputs)}, f, indent=2)
            
            result["status"] = "success"
            result["end_time"] = end.isoformat()
            result["duration"] = duration
            result["outputs"] = outputs if isinstance(outputs, dict) else {"result": str(outputs)}
            
            # Find generated files
            for file_path in example_output_dir.glob("*"):
                if file_path.name != "pipeline_output.json":
                    result["output_files"].append(str(file_path))
            
            print(f"✅ Success! Duration: {duration:.2f}s")
            print(f"   Output files: {len(result['output_files'])}")
            
        except Exception as e:
            result["status"] = "failed"
            result["end_time"] = datetime.now().isoformat()
            result["error"] = {
                "type": type(e).__name__,
                "message": str(e)
            }
            print(f"❌ Failed: {type(e).__name__}: {str(e)}")
            
            # Clear environment variable on error
            os.environ.pop('ORCHESTRATOR_OUTPUT_DIR', None)
        
        return result
    
    async def test_all(self):
        """Test all example pipelines."""
        # Find all example YAML files
        examples_dir = Path("examples")
        yaml_files = list(examples_dir.glob("*.yaml"))
        yaml_files.extend(examples_dir.glob("pipelines/*.yaml"))
        yaml_files.extend(examples_dir.glob("sub_pipelines/*.yaml"))
        
        # Skip MCP examples if not configured
        skip_patterns = ["mcp_integration", "mcp_memory"]
        
        # Filter examples
        examples_to_test = []
        for yaml_file in sorted(yaml_files):
            if any(pattern in yaml_file.name for pattern in skip_patterns):
                print(f"⏭️  Skipping {yaml_file.name} (requires MCP setup)")
                continue
            examples_to_test.append(yaml_file)
        
        print(f"\nFound {len(examples_to_test)} examples to test")
        print(f"Output directory: {self.output_dir}")
        
        # Initialize models
        print("\nInitializing models...")
        orchestrator.init_models()
        
        # Test each example
        for yaml_file in examples_to_test:
            result = await self.test_example(yaml_file)
            self.results.append(result)
        
        # Generate summary
        self.generate_summary()
    
    def generate_summary(self):
        """Generate test summary."""
        print(f"\n{'='*70}")
        print("TEST SUMMARY")
        print(f"{'='*70}")
        
        total = len(self.results)
        success = sum(1 for r in self.results if r["status"] == "success")
        failed = sum(1 for r in self.results if r["status"] == "failed")
        
        print(f"Total examples: {total}")
        print(f"✅ Successful: {success} ({success/total*100:.1f}%)")
        print(f"❌ Failed: {failed} ({failed/total*100:.1f}%)")
        
        # Examples with output files
        with_files = sum(1 for r in self.results if r["output_files"])
        print(f"\nExamples with output files: {with_files}")
        
        # Failed examples
        if failed > 0:
            print("\nFailed examples:")
            for r in self.results:
                if r["status"] == "failed":
                    error = r["error"]
                    print(f"  - {r['name']}: {error['type']} - {error['message']}")
        
        # Examples without output files
        without_files = [r for r in self.results if r["status"] == "success" and not r["output_files"]]
        if without_files:
            print("\nSuccessful examples without output files:")
            for r in without_files:
                print(f"  - {r['name']}")
        
        # Save detailed report
        report_file = self.output_dir / "test_summary.json"
        with open(report_file, 'w') as f:
            json.dump({
                "test_time": datetime.now().isoformat(),
                "summary": {
                    "total": total,
                    "success": success,
                    "failed": failed,
                    "success_rate": success/total if total > 0 else 0,
                    "with_output_files": with_files
                },
                "results": self.results
            }, f, indent=2)
        
        print(f"\nDetailed report saved to: {report_file}")
        
        # Output quality report
        quality_file = self.output_dir / "output_quality.md"
        self.generate_quality_report(quality_file)
        print(f"Output quality report saved to: {quality_file}")
    
    def generate_quality_report(self, output_file: Path):
        """Generate output quality report."""
        with open(output_file, 'w') as f:
            f.write("# Output Quality Report\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            
            for result in self.results:
                if result["status"] == "success":
                    f.write(f"## {result['name']}\n\n")
                    f.write(f"- Status: ✅ Success\n")
                    f.write(f"- Duration: {result['duration']:.2f}s\n")
                    f.write(f"- Output files: {len(result['output_files'])}\n")
                    
                    if result['output_files']:
                        f.write("\n### Generated Files:\n")
                        for file_path in result['output_files']:
                            f.write(f"- {Path(file_path).name}\n")
                    
                    f.write("\n---\n\n")


async def main():
    """Main entry point."""
    tester = ExampleTester()
    await tester.test_all()


if __name__ == "__main__":
    asyncio.run(main())