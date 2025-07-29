#!/usr/bin/env python3
"""Validate all example pipelines by running them with real data."""

import asyncio
import json
import os
import sys
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import orchestrator
from orchestrator.utils.api_keys_flexible import load_api_keys_optional


class ExampleValidator:
    def __init__(self):
        self.results = {}
        self.output_dir = Path("example_validation_outputs")
        self.output_dir.mkdir(exist_ok=True)
        self.start_time = datetime.now()
        
    def get_test_inputs(self, example_name: str) -> Dict[str, Any]:
        """Get appropriate test inputs for each example."""
        
        # Create test data files if needed
        test_data_dir = Path("examples/test_data")
        test_data_dir.mkdir(exist_ok=True)
        
        # Create sample CSV data
        sample_csv = test_data_dir / "sample_data.csv"
        if not sample_csv.exists():
            with open(sample_csv, 'w') as f:
                f.write("order_id,customer_id,product_name,quantity,unit_price,order_date,status\n")
                f.write("ORD-000001,CUST-001,Widget A,5,19.99,2024-01-15,delivered\n")
                f.write("ORD-000002,CUST-002,Widget B,3,29.99,2024-01-16,shipped\n")
                f.write("ORD-000003,CUST-001,Widget C,1,49.99,2024-01-17,processing\n")
                f.write("ORD-000004,CUST-003,Widget A,10,19.99,2024-01-18,delivered\n")
                f.write("ORD-000005,CUST-002,Widget D,2,39.99,2024-01-19,pending\n")
        
        # Create sample JSON data
        sample_json = test_data_dir / "customers.json"
        if not sample_json.exists():
            customers = {
                "customers": [
                    {"id": "CUST-001", "name": "Alice Smith", "orders": 15, "total_spent": 1250.50},
                    {"id": "CUST-002", "name": "Bob Jones", "orders": 8, "total_spent": 750.25},
                    {"id": "CUST-003", "name": "Carol White", "orders": 22, "total_spent": 2100.00}
                ]
            }
            with open(sample_json, 'w') as f:
                json.dump(customers, f, indent=2)
        
        # Map example names to test inputs
        inputs = {
            # AUTO tags examples
            "auto_tags_demo": {
                "data_file": str(sample_csv),
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
                "data_file": str(sample_csv)
            },
            
            "data_processing_pipeline": {
                "input_file": str(sample_csv),
                "output_format": "json",
                "quality_threshold": 0.90,
                "enable_profiling": True
            },
            
            "recursive_data_processing": {
                "data_path": str(sample_json),
                "max_depth": 3
            },
            
            "modular_analysis_pipeline": {
                "data_source": str(sample_csv),
                "analysis_type": "statistical"
            },
            
            # Web and research
            "web_research_pipeline": {
                "research_topic": "renewable energy storage solutions",
                "max_sources": 5,
                "output_format": "markdown",
                "research_depth": "standard"
            },
            
            "working_web_search": {
                "query": "latest breakthroughs in quantum computing",
                "num_results": 3
            },
            
            "research_pipeline": {
                "topic": "artificial intelligence ethics",
                "depth": "comprehensive"
            },
            
            # Creative and multimodal
            "creative_image_pipeline": {
                "prompt": "a serene mountain landscape at sunset",
                "style": "impressionist",
                "resolution": "1024x768"
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
            }
        }
        
        return inputs.get(example_name, {})
    
    async def validate_example(self, yaml_path: Path) -> Dict[str, Any]:
        """Validate a single example pipeline."""
        example_name = yaml_path.stem
        
        print(f"\n{'='*70}")
        print(f"Validating: {example_name}")
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
            "metrics": {}
        }
        
        try:
            # Get test inputs
            inputs = self.get_test_inputs(example_name)
            print(f"Inputs: {json.dumps(inputs, indent=2)}")
            
            # Compile pipeline
            print("Compiling pipeline...")
            pipeline = await orchestrator.compile_async(str(yaml_path))
            
            # Run pipeline
            print("Running pipeline...")
            start = datetime.now()
            
            outputs = await pipeline.run_async(**inputs)
            
            end = datetime.now()
            duration = (end - start).total_seconds()
            
            # Save outputs
            output_file = self.output_dir / f"{example_name}_output.json"
            with open(output_file, 'w') as f:
                if isinstance(outputs, dict):
                    json.dump(outputs, f, indent=2, default=str)
                else:
                    json.dump({"result": str(outputs)}, f, indent=2)
            
            result["status"] = "success"
            result["end_time"] = end.isoformat()
            result["duration"] = duration
            result["outputs"].append(str(output_file))
            result["metrics"]["execution_time"] = duration
            
            # Check for generated files
            for ext in ['.pdf', '.md', '.txt', '.html', '.png', '.jpg', '.csv', '.json']:
                pattern = f"*{ext}"
                for generated in Path('.').glob(pattern):
                    if generated.stat().st_mtime > start.timestamp():
                        result["outputs"].append(str(generated))
            
            print(f"✅ Success! Duration: {duration:.2f}s")
            print(f"   Output: {output_file}")
            
            # Validate output quality
            quality = self.validate_output_quality(outputs, example_name)
            result["metrics"]["quality_score"] = quality["score"]
            result["metrics"]["quality_notes"] = quality["notes"]
            
            if quality["score"] < 0.7:
                print(f"⚠️  Warning: Low quality score: {quality['score']}")
                print(f"   Notes: {quality['notes']}")
            
        except Exception as e:
            result["status"] = "failed"
            result["end_time"] = datetime.now().isoformat()
            result["error"] = {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc()
            }
            print(f"❌ Failed: {e}")
            print(f"   Type: {type(e).__name__}")
            
        return result
    
    def validate_output_quality(self, outputs: Any, example_name: str) -> Dict[str, Any]:
        """Validate the quality of pipeline outputs."""
        quality = {"score": 1.0, "notes": []}
        
        # Check if outputs exist
        if outputs is None:
            quality["score"] = 0.0
            quality["notes"].append("No output produced")
            return quality
        
        # Example-specific validation
        if example_name == "web_research_pipeline":
            if isinstance(outputs, dict):
                if not outputs.get("report_path"):
                    quality["score"] -= 0.3
                    quality["notes"].append("No report path in output")
                if not outputs.get("sources_analyzed"):
                    quality["score"] -= 0.2
                    quality["notes"].append("No sources analyzed count")
        
        elif example_name == "data_processing_pipeline":
            if isinstance(outputs, dict):
                if not outputs.get("processed_rows"):
                    quality["score"] -= 0.2
                    quality["notes"].append("No processed rows count")
                if outputs.get("quality_score", 0) < 0.8:
                    quality["score"] -= 0.1
                    quality["notes"].append("Data quality score below threshold")
        
        elif "control_flow" in example_name:
            # Control flow examples should produce structured results
            if not isinstance(outputs, (dict, list)):
                quality["score"] -= 0.2
                quality["notes"].append("Output not structured")
        
        # General checks
        if isinstance(outputs, str) and len(outputs) < 10:
            quality["score"] -= 0.1
            quality["notes"].append("Output too short")
        
        if isinstance(outputs, dict) and len(outputs) == 0:
            quality["score"] -= 0.2
            quality["notes"].append("Empty output dictionary")
        
        quality["score"] = max(0.0, quality["score"])
        return quality
    
    async def validate_all(self):
        """Validate all example pipelines."""
        # Find all example YAML files
        examples_dir = Path("examples")
        yaml_files = list(examples_dir.glob("*.yaml"))
        yaml_files.extend(examples_dir.glob("pipelines/*.yaml"))
        yaml_files.extend(examples_dir.glob("sub_pipelines/*.yaml"))
        
        # Skip MCP examples if not configured
        skip_patterns = ["mcp_integration", "mcp_memory"]
        if not os.environ.get("MCP_SERVER_URL"):
            skip_patterns.extend(["mcp_"])
        
        # Filter examples
        examples_to_test = []
        for yaml_file in sorted(yaml_files):
            if any(pattern in yaml_file.name for pattern in skip_patterns):
                print(f"⏭️  Skipping {yaml_file.name} (requires special setup)")
                continue
            examples_to_test.append(yaml_file)
        
        print(f"\nFound {len(examples_to_test)} examples to validate")
        print(f"Output directory: {self.output_dir}")
        
        # Initialize models once
        print("\nInitializing models...")
        orchestrator.init_models()
        
        # Validate each example
        for yaml_file in examples_to_test:
            result = await self.validate_example(yaml_file)
            self.results[yaml_file.stem] = result
        
        # Generate summary
        self.generate_summary()
    
    def generate_summary(self):
        """Generate validation summary."""
        print(f"\n{'='*70}")
        print("VALIDATION SUMMARY")
        print(f"{'='*70}")
        
        total = len(self.results)
        success = sum(1 for r in self.results.values() if r["status"] == "success")
        failed = sum(1 for r in self.results.values() if r["status"] == "failed")
        
        print(f"Total examples: {total}")
        print(f"✅ Successful: {success} ({success/total*100:.1f}%)")
        print(f"❌ Failed: {failed} ({failed/total*100:.1f}%)")
        
        # Quality metrics
        quality_scores = [
            r["metrics"].get("quality_score", 0) 
            for r in self.results.values() 
            if r["status"] == "success"
        ]
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            print(f"\nAverage quality score: {avg_quality:.2f}/1.0")
        
        # Execution times
        exec_times = [
            r["metrics"].get("execution_time", 0) 
            for r in self.results.values() 
            if r["status"] == "success"
        ]
        if exec_times:
            print(f"Average execution time: {sum(exec_times)/len(exec_times):.2f}s")
            print(f"Total execution time: {sum(exec_times):.2f}s")
        
        # Failed examples
        if failed > 0:
            print("\nFailed examples:")
            for name, result in self.results.items():
                if result["status"] == "failed":
                    error = result["error"]
                    print(f"  - {name}: {error['type']} - {error['message']}")
        
        # Low quality examples
        low_quality = [
            (name, r["metrics"]["quality_score"]) 
            for name, r in self.results.items() 
            if r["status"] == "success" and r["metrics"].get("quality_score", 1) < 0.7
        ]
        if low_quality:
            print("\nLow quality outputs:")
            for name, score in low_quality:
                notes = self.results[name]["metrics"].get("quality_notes", [])
                print(f"  - {name}: {score:.2f} - {', '.join(notes)}")
        
        # Save detailed report
        report_file = self.output_dir / "validation_report.json"
        with open(report_file, 'w') as f:
            json.dump({
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "summary": {
                    "total": total,
                    "success": success,
                    "failed": failed,
                    "success_rate": success/total if total > 0 else 0,
                    "average_quality": avg_quality if quality_scores else 0,
                    "total_execution_time": sum(exec_times) if exec_times else 0
                },
                "results": self.results
            }, f, indent=2)
        
        print(f"\nDetailed report saved to: {report_file}")
        
        # Check API keys
        keys = load_api_keys_optional()
        if keys:
            print(f"\nAPI keys available: {', '.join(keys.keys())}")
        else:
            print("\n⚠️  No API keys found - only local models available")


async def main():
    """Main entry point."""
    validator = ExampleValidator()
    await validator.validate_all()


if __name__ == "__main__":
    asyncio.run(main())