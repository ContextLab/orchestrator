#!/usr/bin/env python3
"""Test all real example pipelines for Issue #243."""

import asyncio
import sys
import os
from pathlib import Path
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator import Orchestrator, init_models
from orchestrator.compiler.yaml_compiler import YAMLCompiler
from orchestrator.control_systems.hybrid_control_system import HybridControlSystem

class RealPipelineValidator:
    """Test real pipelines for actual execution issues."""
    
    def __init__(self):
        self.results = {}
        self.model_registry = None
        self.control_system = None
        
    async def test_pipeline(self, pipeline_path: Path) -> dict:
        """Test a single pipeline."""
        result = {
            "name": pipeline_path.name,
            "compile": "pending",
            "execute": "pending",
            "issues": []
        }
        
        try:
            # Read pipeline
            with open(pipeline_path) as f:
                yaml_content = f.read()
            
            # Compile
            compiler = YAMLCompiler(development_mode=True)
            pipeline = await compiler.compile(yaml_content)
            result["compile"] = "success"
            
            # Get minimal test inputs
            inputs = self._get_minimal_inputs(pipeline_path.name)
            
            # Try to execute with timeout
            orchestrator = Orchestrator(
                model_registry=self.model_registry,
                control_system=self.control_system
            )
            
            # Execute with short timeout
            try:
                results = await asyncio.wait_for(
                    orchestrator.execute_yaml(yaml_content, inputs),
                    timeout=30.0  # 30 second timeout per pipeline
                )
                
                # Check results for issues
                result_str = str(results)
                if "{{" in result_str and "}}" in result_str:
                    result["issues"].append("unrendered_templates")
                if "$item" in result_str or "$index" in result_str:
                    result["issues"].append("unrendered_loop_vars")
                if "error" in result_str.lower():
                    result["issues"].append("error_in_output")
                    
                result["execute"] = "success" if not result["issues"] else "with_issues"
                
            except asyncio.TimeoutError:
                result["execute"] = "timeout"
                result["issues"].append("execution_timeout")
                
        except Exception as e:
            error_msg = str(e)[:100]
            if "compile" in result and result["compile"] == "pending":
                result["compile"] = "failed"
                result["issues"].append(f"compile_error: {error_msg}")
            else:
                result["execute"] = "failed"
                result["issues"].append(f"execute_error: {error_msg}")
        
        self.results[pipeline_path.name] = result
        return result
    
    def _get_minimal_inputs(self, pipeline_name: str) -> dict:
        """Get minimal inputs to test pipeline."""
        # Very minimal inputs just to test execution
        return {
            "topic": "test",
            "input_text": "Test text",
            "query": "test query",
            "data": {"test": "data"},
            "output_path": f"examples/outputs/test_{pipeline_name}",
            "input_file": "test.txt",
            "url": "https://example.com",
            "prompt": "test prompt"
        }
    
    async def test_all(self):
        """Test all pipelines."""
        # Initialize models once
        print("Initializing models...")
        self.model_registry = init_models()
        
        if not self.model_registry or not self.model_registry.models:
            print("❌ No models available")
            return
        
        self.control_system = HybridControlSystem(self.model_registry)
        
        # Get real pipeline files
        examples_dir = Path("examples")
        
        # Priority pipelines to test (ones likely to have issues)
        priority_pipelines = [
            "research_minimal.yaml",
            "simple_data_processing.yaml",
            "control_flow_advanced.yaml",
            "control_flow_for_loop.yaml",
            "control_flow_while_loop.yaml",
            "data_processing.yaml",
            "web_research_pipeline.yaml",
            "llm_routing_pipeline.yaml",
            "model_routing_demo.yaml",
            "multimodal_processing.yaml",
            "fact_checker.yaml",
            "iterative_fact_checker.yaml",
            "enhanced_research_pipeline.yaml",
            "mcp_integration_pipeline.yaml"
        ]
        
        # Test each pipeline
        print(f"\nTesting {len(priority_pipelines)} priority pipelines...")
        print("=" * 60)
        
        for pipeline_name in priority_pipelines:
            pipeline_path = examples_dir / pipeline_name
            if not pipeline_path.exists():
                print(f"⏭️  {pipeline_name}: Skipped (not found)")
                continue
                
            print(f"Testing {pipeline_name}...", end=" ")
            result = await self.test_pipeline(pipeline_path)
            
            # Print result
            if result["compile"] == "failed":
                print(f"❌ Compile failed")
            elif result["execute"] == "failed":
                print(f"❌ Execute failed")
            elif result["execute"] == "timeout":
                print(f"⏱️  Timeout")
            elif result["issues"]:
                print(f"⚠️  Issues: {', '.join(result['issues'][:2])}")
            else:
                print(f"✅ Success")
        
        # Summary
        self.print_summary()
        
    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        
        compile_success = sum(1 for r in self.results.values() if r["compile"] == "success")
        compile_failed = sum(1 for r in self.results.values() if r["compile"] == "failed")
        
        execute_success = sum(1 for r in self.results.values() if r["execute"] == "success")
        execute_issues = sum(1 for r in self.results.values() if r["execute"] == "with_issues")
        execute_failed = sum(1 for r in self.results.values() if r["execute"] == "failed")
        execute_timeout = sum(1 for r in self.results.values() if r["execute"] == "timeout")
        
        print(f"Total tested: {len(self.results)}")
        print(f"\nCompilation:")
        print(f"  ✅ Success: {compile_success}")
        print(f"  ❌ Failed: {compile_failed}")
        
        print(f"\nExecution:")
        print(f"  ✅ Success: {execute_success}")
        print(f"  ⚠️  With Issues: {execute_issues}")
        print(f"  ❌ Failed: {execute_failed}")
        print(f"  ⏱️  Timeout: {execute_timeout}")
        
        # List main issues
        all_issues = []
        for result in self.results.values():
            all_issues.extend(result["issues"])
        
        if all_issues:
            print(f"\nCommon issues found:")
            issue_counts = {}
            for issue in all_issues:
                base_issue = issue.split(":")[0]
                issue_counts[base_issue] = issue_counts.get(base_issue, 0) + 1
            
            for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1])[:5]:
                print(f"  - {issue}: {count} occurrences")
        
        # Save detailed results
        output_file = Path("examples/outputs/pipeline_test_results.json")
        output_file.parent.mkdir(exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\nDetailed results saved to: {output_file}")

async def main():
    validator = RealPipelineValidator()
    await validator.test_all()

if __name__ == "__main__":
    asyncio.run(main())