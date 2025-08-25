#!/usr/bin/env python3
"""
Validate all example pipelines for Issue #243.
Runs each pipeline and checks for common issues.
"""

import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add orchestrator to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator import Orchestrator, init_models
from orchestrator.models import get_model_registry
from orchestrator.compiler.yaml_compiler import YAMLCompiler
from orchestrator.control_systems.hybrid_control_system import HybridControlSystem


class PipelineValidator:
    """Validates example pipelines for quality issues."""
    
    def __init__(self):
        self.results = {}
        self.issues = []
        self.examples_dir = Path("examples")
        self.output_dir = Path("examples/outputs/validation_run")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_registry = None
        self.control_system = None
        
    async def validate_pipeline(self, pipeline_path: Path) -> Dict[str, Any]:
        """Validate a single pipeline."""
        print(f"\n{'='*60}")
        print(f"Validating: {pipeline_path.name}")
        print(f"{'='*60}")
        
        result = {
            "pipeline": pipeline_path.name,
            "status": "pending",
            "issues": [],
            "execution_time": 0,
            "output_quality": 0
        }
        
        start_time = time.time()
        
        try:
            # Load pipeline
            with open(pipeline_path) as f:
                yaml_content = f.read()
            
            # Compile pipeline
            compiler = YAMLCompiler(development_mode=True)
            pipeline = await compiler.compile(yaml_content)
            
            # Get appropriate inputs
            inputs = self._get_test_inputs(pipeline_path.name)
            
            # Setup orchestrator with models and control system
            orchestrator = Orchestrator(
                model_registry=self.model_registry,
                control_system=self.control_system
            )
            
            # Run pipeline
            output_path = self.output_dir / pipeline_path.stem
            output_path.mkdir(exist_ok=True)
            
            # Add output_path to inputs if not present
            if 'output_path' not in inputs:
                inputs['output_path'] = str(output_path)
            
            results = await orchestrator.execute_yaml(yaml_content, inputs)
            
            # Check for issues
            issues = self._check_for_issues(results, output_path)
            
            result["status"] = "success" if not issues else "issues_found"
            result["issues"] = issues
            result["execution_time"] = time.time() - start_time
            result["output_quality"] = self._calculate_quality_score(results, issues)
            
            print(f"✅ Pipeline executed successfully")
            if issues:
                print(f"⚠️  Found {len(issues)} issues:")
                for issue in issues:
                    print(f"   - {issue}")
                    
        except Exception as e:
            result["status"] = "failed"
            result["issues"].append(f"Execution error: {str(e)}")
            result["execution_time"] = time.time() - start_time
            print(f"❌ Pipeline failed: {str(e)}")
            traceback.print_exc()
            
        self.results[pipeline_path.name] = result
        return result
    
    def _get_test_inputs(self, pipeline_name: str) -> Dict[str, Any]:
        """Get appropriate test inputs for a pipeline."""
        # Default inputs
        inputs = {
            "input_text": "Climate change is affecting global weather patterns.",
            "topic": "renewable energy",
            "query": "machine learning applications",
            "url": "https://example.com",
            "file_path": "test.txt",
            "data": {"key": "value", "count": 10}
        }
        
        # Pipeline-specific inputs
        if "research" in pipeline_name:
            inputs["topic"] = "artificial intelligence in healthcare"
            inputs["depth"] = "comprehensive"
        elif "data" in pipeline_name:
            inputs["data_path"] = "examples/data/sample_data.csv"
            inputs["format"] = "csv"
        elif "image" in pipeline_name:
            inputs["prompt"] = "futuristic city skyline"
            inputs["style"] = "digital art"
        elif "timeout" in pipeline_name:
            inputs["delay"] = 1
        elif "validation" in pipeline_name:
            inputs["data"] = {"name": "Test", "email": "test@example.com"}
            inputs["schema"] = {"type": "object", "properties": {"name": {"type": "string"}}}
            
        return inputs
    
    def _check_for_issues(self, results: Any, output_path: Path) -> List[str]:
        """Check for common issues in pipeline results."""
        issues = []
        
        # Convert results to string for checking
        result_str = str(results)
        
        # Check for unrendered templates
        if "{{" in result_str or "}}" in result_str:
            issues.append("Unrendered template variables found")
            
        # Check for loop variables
        if "$item" in result_str or "$index" in result_str or "$iteration" in result_str:
            issues.append("Unrendered loop variables found")
            
        # Check for conversational markers
        conversational_markers = [
            "Certainly!", "Sure!", "I'd be happy to",
            "Let me", "I'll create", "Here's"
        ]
        for marker in conversational_markers:
            if marker in result_str:
                issues.append(f"Conversational marker found: '{marker}'")
                break
                
        # Check for error indicators
        if "error" in result_str.lower() or "failed" in result_str.lower():
            if "error" not in str(results).lower():  # Not an actual error field
                issues.append("Error indicators in output")
                
        # Check output files
        output_files = list(output_path.glob("**/*"))
        if len(output_files) == 1:  # Only the directory itself
            issues.append("No output files generated")
            
        # Check for empty content
        for file_path in output_files:
            if file_path.is_file():
                content = file_path.read_text()
                if not content.strip():
                    issues.append(f"Empty output file: {file_path.name}")
                elif "{{" in content or "}}" in content:
                    issues.append(f"Unrendered templates in file: {file_path.name}")
                    
        return issues
    
    def _calculate_quality_score(self, results: Any, issues: List[str]) -> float:
        """Calculate a quality score for the output."""
        score = 100.0
        
        # Deduct for each issue
        score -= len(issues) * 10
        
        # Bonus for successful execution
        if results:
            score += 10
            
        # Ensure score is between 0 and 100
        return max(0, min(100, score))
    
    async def validate_all(self):
        """Validate all example pipelines."""
        # Initialize models first
        print("Initializing models...")
        self.model_registry = init_models()
        
        if not self.model_registry or not self.model_registry.models:
            print("❌ No models available. Please check your API keys and models.yaml")
            return
        
        # Create control system with models
        self.control_system = HybridControlSystem(self.model_registry)
        
        # Get all pipeline files
        pipelines = sorted(self.examples_dir.glob("*.yaml"))
        
        # Filter to focus on the 25 main pipelines
        priority_pipelines = [
            "auto_tags_demo.yaml",
            "control_flow_advanced.yaml",
            "control_flow_conditional.yaml",
            "control_flow_dynamic.yaml",
            "creative_image_pipeline.yaml",
            "data_processing.yaml",
            "data_processing_pipeline.yaml",
            "interactive_pipeline.yaml",
            "llm_routing_pipeline.yaml",
            "mcp_integration_pipeline.yaml",
            "mcp_memory_workflow.yaml",
            "model_routing_demo.yaml",
            "modular_analysis_pipeline.yaml",
            "multimodal_processing.yaml",
            "recursive_data_processing.yaml",
            "research_minimal.yaml",
            "simple_data_processing.yaml",
            "simple_timeout_test.yaml",
            "statistical_analysis.yaml",
            "terminal_automation.yaml",
            "test_timeout.yaml",
            "test_timeout_websearch.yaml",
            "test_validation_pipeline.yaml",
            "validation_pipeline.yaml",
            "web_research_pipeline.yaml",
            "working_web_search.yaml"
        ]
        
        # Filter pipelines
        pipelines = [p for p in pipelines if p.name in priority_pipelines]
        
        print(f"Found {len(pipelines)} priority pipelines to validate")
        
        # Validate each pipeline
        for pipeline_path in pipelines:
            await self.validate_pipeline(pipeline_path)
            
        # Generate report
        self.generate_report()
        
    def generate_report(self):
        """Generate validation report."""
        print("\n" + "="*70)
        print("VALIDATION REPORT")
        print("="*70)
        
        # Statistics
        total = len(self.results)
        successful = sum(1 for r in self.results.values() if r["status"] == "success")
        with_issues = sum(1 for r in self.results.values() if r["status"] == "issues_found")
        failed = sum(1 for r in self.results.values() if r["status"] == "failed")
        
        print(f"\nTotal Pipelines: {total}")
        print(f"✅ Successful: {successful}")
        print(f"⚠️  With Issues: {with_issues}")
        print(f"❌ Failed: {failed}")
        
        # Average quality score
        quality_scores = [r["output_quality"] for r in self.results.values()]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        print(f"\nAverage Quality Score: {avg_quality:.1f}%")
        
        # Detailed results
        print("\n" + "-"*70)
        print("DETAILED RESULTS")
        print("-"*70)
        
        for name, result in sorted(self.results.items()):
            status_icon = "✅" if result["status"] == "success" else "⚠️" if result["status"] == "issues_found" else "❌"
            print(f"\n{status_icon} {name}")
            print(f"   Status: {result['status']}")
            print(f"   Quality: {result['output_quality']:.0f}%")
            print(f"   Time: {result['execution_time']:.1f}s")
            if result["issues"]:
                print(f"   Issues:")
                for issue in result["issues"]:
                    print(f"      - {issue}")
                    
        # Save report
        report_path = self.output_dir / "validation_report.json"
        with open(report_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\n📄 Report saved to: {report_path}")
        
        # Summary for Issue #243
        print("\n" + "="*70)
        print("ISSUE #243 SUMMARY")
        print("="*70)
        
        if failed > 0:
            print(f"❌ {failed} pipelines need fixes")
        elif with_issues > 0:
            print(f"⚠️  {with_issues} pipelines have minor issues")
        else:
            print("✅ All pipelines validated successfully!")
            
        print("\nNext Steps:")
        if failed > 0:
            print("1. Fix failed pipelines first")
        if with_issues > 0:
            print("2. Address quality issues in pipelines with warnings")
        print("3. Run comprehensive test suite with: python tests/pipeline_tests/run_all.py")
        

async def main():
    """Main validation function."""
    validator = PipelineValidator()
    await validator.validate_all()


if __name__ == "__main__":
    asyncio.run(main())