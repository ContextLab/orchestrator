#!/usr/bin/env python3
"""
Comprehensive pipeline validation and execution script.
Runs ALL example pipelines with appropriate inputs and saves outputs.
"""

import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator import Orchestrator, init_models
from orchestrator.compiler.yaml_compiler import YAMLCompiler
from orchestrator.control_systems.hybrid_control_system import HybridControlSystem


class ComprehensivePipelineValidator:
    """Validates and runs all example pipelines with quality checks."""
    
    def __init__(self):
        self.results = {}
        self.model_registry = None
        self.control_system = None
        self.examples_dir = Path("examples")
        self.base_output_dir = Path("examples/outputs")
        
    def get_pipeline_inputs(self, pipeline_name: str) -> Dict[str, Any]:
        """Get appropriate inputs for each pipeline based on its requirements."""
        
        # Default inputs that work for most pipelines
        default_inputs = {
            "output_path": str(self.base_output_dir / pipeline_name.replace('.yaml', '')),
            "topic": "artificial intelligence in healthcare",
            "input_text": "Artificial intelligence is transforming healthcare through improved diagnostics and personalized treatment.",
            "query": "machine learning applications in medicine",
            "data": {"patients": 100, "treatments": ["AI-assisted", "traditional"], "success_rate": 0.85},
            "prompt": "Create an innovative healthcare solution",
            "url": "https://example.com/healthcare-ai",
            "file_path": "data/sample.txt",
            "input_file": "data/input.csv"
        }
        
        # Pipeline-specific inputs
        specific_inputs = {
            "auto_tags_demo.yaml": {
                "content": "This article discusses machine learning, artificial intelligence, deep learning, neural networks, and healthcare applications.",
                "task_complexity": "detailed"
            },
            "code_optimization.yaml": {
                "code_snippet": "def calculate_sum(numbers):\n    total = 0\n    for n in numbers:\n        total = total + n\n    return total",
                "optimization_level": "aggressive",
                "target_language": "python"
            },
            "control_flow_advanced.yaml": {
                "languages": ["Spanish", "French", "German"],
                "quality_threshold": 0.7
            },
            "control_flow_conditional.yaml": {
                "input_file": "test.txt",
                "size_threshold": 1000
            },
            "control_flow_dynamic.yaml": {
                "operation": "data_processing",
                "retry_limit": 3
            },
            "control_flow_for_loop.yaml": {
                "items": ["apple", "banana", "orange", "grape", "mango"],
                "processing_type": "analyze"
            },
            "control_flow_while_loop.yaml": {
                "max_iterations": 5,
                "target_quality": 0.8,
                "initial_value": 0.5
            },
            "creative_image_pipeline.yaml": {
                "base_prompt": "futuristic city with flying cars",
                "num_variations": 3,
                "art_styles": ["cyberpunk", "neo-tokyo", "blade runner"]
            },
            "data_processing.yaml": {
                "data_source": "csv",
                "output_format": "json"
            },
            "data_processing_pipeline.yaml": {
                "input_file": "data/sample.csv",
                "quality_threshold": 0.8,
                "enable_profiling": True
            },
            "enhanced_research_pipeline.yaml": {
                "research_topic": "quantum computing applications",
                "depth": "comprehensive",
                "max_sources": 10
            },
            "fact_checker.yaml": {
                "claim": "The Earth orbits around the Sun",
                "confidence_threshold": 0.7
            },
            "interactive_pipeline.yaml": {
                "input_file": "data.json",
                "output_dir": str(self.base_output_dir / "interactive_pipeline")
            },
            "iterative_fact_checker.yaml": {
                "claims": [
                    "Water boils at 100°C at sea level",
                    "The speed of light is constant",
                    "DNA has a double helix structure"
                ],
                "max_iterations": 3
            },
            "llm_routing_pipeline.yaml": {
                "task": "Explain quantum entanglement in simple terms",
                "optimization_goals": ["accuracy", "clarity"],
                "routing_strategy": "capability_based"
            },
            "mcp_integration_pipeline.yaml": {
                "search_query": "latest AI breakthroughs 2024"
            },
            "mcp_memory_workflow.yaml": {
                "user_name": "TestUser",
                "task_description": "Analyze healthcare data trends"
            },
            "model_routing_demo.yaml": {
                "task_budget": 1.0,
                "priority": "high"
            },
            "modular_analysis_pipeline.yaml": {
                "dataset": {"sales": [100, 150, 200, 175, 225], "months": ["Jan", "Feb", "Mar", "Apr", "May"]},
                "analysis_types": ["statistical", "trend", "sentiment"]
            },
            "multimodal_processing.yaml": {
                "input_image": "sample.jpg",
                "input_audio": "sample.mp3",
                "input_video": "sample.mp4",
                "output_dir": str(self.base_output_dir / "multimodal_processing")
            },
            "research_minimal.yaml": {
                "topic": "renewable energy innovations"
            },
            "simple_data_processing.yaml": {
                # Uses default inputs
            },
            "statistical_analysis.yaml": {
                "data": [23, 45, 67, 89, 12, 34, 56, 78, 90, 21],
                "analysis_type": "comprehensive"
            },
            "validation_pipeline.yaml": {
                "data": {"name": "John Doe", "email": "john@example.com", "age": 30},
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string", "format": "email"},
                        "age": {"type": "number", "minimum": 0}
                    }
                }
            },
            "web_research_pipeline.yaml": {
                "research_topic": "sustainable technology",
                "max_sources": 5,
                "output_format": "markdown",
                "research_depth": "standard"
            }
        }
        
        # Merge default and specific inputs
        inputs = default_inputs.copy()
        if pipeline_name in specific_inputs:
            inputs.update(specific_inputs[pipeline_name])
            
        # Ensure output_path is set correctly for this specific pipeline
        inputs["output_path"] = str(self.base_output_dir / pipeline_name.replace('.yaml', ''))
        
        return inputs
    
    async def validate_pipeline(self, pipeline_path: Path) -> Dict[str, Any]:
        """Validate and run a single pipeline."""
        pipeline_name = pipeline_path.name
        print(f"\n{'='*60}")
        print(f"Validating: {pipeline_name}")
        print(f"{'='*60}")
        
        result = {
            "pipeline": pipeline_name,
            "status": "pending",
            "compile_time": 0,
            "execution_time": 0,
            "output_quality": 0,
            "issues": [],
            "outputs": []
        }
        
        # Create output directory
        output_dir = self.base_output_dir / pipeline_name.replace('.yaml', '')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Read pipeline
            with open(pipeline_path) as f:
                yaml_content = f.read()
            
            # Compile pipeline
            print(f"📝 Compiling pipeline...")
            compile_start = time.time()
            compiler = YAMLCompiler(development_mode=True)
            pipeline = await compiler.compile(yaml_content)
            result["compile_time"] = time.time() - compile_start
            print(f"✅ Compiled in {result['compile_time']:.2f}s")
            
            # Get appropriate inputs
            inputs = self.get_pipeline_inputs(pipeline_name)
            print(f"📊 Using inputs: {json.dumps({k: str(v)[:50] if isinstance(v, str) else v for k, v in inputs.items()}, indent=2)}")
            
            # Setup orchestrator
            orchestrator = Orchestrator(
                model_registry=self.model_registry,
                control_system=self.control_system
            )
            
            # Run pipeline
            print(f"🚀 Executing pipeline...")
            exec_start = time.time()
            
            try:
                # Execute with reasonable timeout
                results = await asyncio.wait_for(
                    orchestrator.execute_yaml(yaml_content, inputs),
                    timeout=120.0  # 2 minute timeout
                )
                result["execution_time"] = time.time() - exec_start
                print(f"✅ Executed in {result['execution_time']:.2f}s")
                
                # Analyze results
                quality_score, issues = self.analyze_results(results, output_dir)
                result["output_quality"] = quality_score
                result["issues"] = issues
                
                # List output files
                output_files = list(output_dir.glob("**/*"))
                result["outputs"] = [str(f.relative_to(output_dir)) for f in output_files if f.is_file()]
                
                if issues:
                    print(f"⚠️  Quality issues found: {', '.join(issues[:3])}")
                    result["status"] = "completed_with_issues"
                else:
                    print(f"✅ Pipeline completed successfully with quality score: {quality_score}%")
                    result["status"] = "success"
                    
                # Save results summary
                summary_file = output_dir / "validation_summary.json"
                with open(summary_file, "w") as f:
                    json.dump({
                        "pipeline": pipeline_name,
                        "timestamp": datetime.now().isoformat(),
                        "status": result["status"],
                        "quality_score": quality_score,
                        "execution_time": result["execution_time"],
                        "issues": issues,
                        "outputs": result["outputs"]
                    }, f, indent=2)
                    
            except asyncio.TimeoutError:
                result["status"] = "timeout"
                result["issues"].append("execution_timeout")
                print(f"⏱️  Pipeline execution timed out")
                
        except Exception as e:
            result["status"] = "failed"
            result["issues"].append(f"error: {str(e)[:100]}")
            print(f"❌ Pipeline failed: {str(e)[:100]}")
            traceback.print_exc()
        
        self.results[pipeline_name] = result
        return result
    
    def analyze_results(self, results: Any, output_dir: Path) -> tuple[float, list]:
        """Analyze pipeline results for quality issues."""
        issues = []
        quality_score = 100.0
        
        # Convert results to string for analysis
        result_str = str(results)
        
        # Check for unrendered templates
        if "{{" in result_str and "}}" in result_str:
            issues.append("unrendered_templates")
            quality_score -= 20
        
        # Check for loop variables
        if "$item" in result_str or "$index" in result_str or "$iteration" in result_str:
            issues.append("unrendered_loop_variables")
            quality_score -= 15
        
        # Check for conversational markers (AI responses that weren't cleaned)
        conversational_markers = [
            "Certainly!", "Sure!", "I'd be happy to",
            "Let me", "I'll create", "Here's", "Here is"
        ]
        for marker in conversational_markers:
            if marker in result_str:
                issues.append(f"conversational_marker: {marker}")
                quality_score -= 10
                break
        
        # Check for error indicators
        if "error" in result_str.lower() or "failed" in result_str.lower():
            # Check if it's an actual error or just mentioned in content
            if "Error:" in result_str or "Failed:" in result_str:
                issues.append("error_in_output")
                quality_score -= 25
        
        # Check output files
        output_files = list(output_dir.glob("**/*"))
        if len([f for f in output_files if f.is_file()]) == 0:
            issues.append("no_output_files")
            quality_score -= 30
        
        # Check for empty files
        for file_path in output_files:
            if file_path.is_file():
                try:
                    content = file_path.read_text()
                    if not content.strip():
                        issues.append(f"empty_file: {file_path.name}")
                        quality_score -= 10
                    elif "{{" in content and "}}" in content:
                        issues.append(f"templates_in_file: {file_path.name}")
                        quality_score -= 10
                except:
                    pass  # Binary files or unreadable files
        
        # Ensure score is between 0 and 100
        quality_score = max(0, min(100, quality_score))
        
        return quality_score, issues
    
    async def validate_all(self):
        """Validate all example pipelines."""
        # Initialize models
        print("🔧 Initializing models...")
        self.model_registry = init_models()
        
        if not self.model_registry or not self.model_registry.models:
            print("❌ No models available. Please check API keys and models.yaml")
            return
        
        print(f"✅ Initialized {len(self.model_registry.models)} models")
        
        # Create control system
        self.control_system = HybridControlSystem(self.model_registry)
        
        # Get all pipeline files
        pipelines = sorted(self.examples_dir.glob("*.yaml"))
        
        # Filter out backup and test files
        pipelines = [p for p in pipelines if not any(x in p.name for x in ["backup", "fixed", "test", "simple_test"])]
        
        print(f"\n📋 Found {len(pipelines)} pipelines to validate")
        
        # Process each pipeline
        for i, pipeline_path in enumerate(pipelines, 1):
            print(f"\n[{i}/{len(pipelines)}] Processing {pipeline_path.name}")
            await self.validate_pipeline(pipeline_path)
            
            # Add small delay between pipelines to avoid overwhelming the system
            await asyncio.sleep(1)
        
        # Generate final report
        self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive validation report."""
        print("\n" + "="*70)
        print("VALIDATION REPORT")
        print("="*70)
        
        # Calculate statistics
        total = len(self.results)
        successful = sum(1 for r in self.results.values() if r["status"] == "success")
        with_issues = sum(1 for r in self.results.values() if r["status"] == "completed_with_issues")
        failed = sum(1 for r in self.results.values() if r["status"] == "failed")
        timeout = sum(1 for r in self.results.values() if r["status"] == "timeout")
        
        print(f"\n📊 Overall Statistics:")
        print(f"  Total Pipelines: {total}")
        print(f"  ✅ Successful: {successful} ({successful/total*100:.1f}%)")
        print(f"  ⚠️  With Issues: {with_issues} ({with_issues/total*100:.1f}%)")
        print(f"  ❌ Failed: {failed} ({failed/total*100:.1f}%)")
        print(f"  ⏱️  Timeout: {timeout} ({timeout/total*100:.1f}%)")
        
        # Average metrics
        quality_scores = [r["output_quality"] for r in self.results.values() if r["output_quality"] > 0]
        exec_times = [r["execution_time"] for r in self.results.values() if r["execution_time"] > 0]
        
        if quality_scores:
            print(f"\n📈 Quality Metrics:")
            print(f"  Average Quality Score: {sum(quality_scores)/len(quality_scores):.1f}%")
            print(f"  Best Quality: {max(quality_scores):.1f}%")
            print(f"  Worst Quality: {min(quality_scores):.1f}%")
        
        if exec_times:
            print(f"\n⏱️  Performance Metrics:")
            print(f"  Average Execution Time: {sum(exec_times)/len(exec_times):.1f}s")
            print(f"  Fastest: {min(exec_times):.1f}s")
            print(f"  Slowest: {max(exec_times):.1f}s")
        
        # Common issues
        all_issues = []
        for result in self.results.values():
            all_issues.extend(result["issues"])
        
        if all_issues:
            print(f"\n⚠️  Common Issues:")
            issue_counts = {}
            for issue in all_issues:
                base_issue = issue.split(":")[0]
                issue_counts[base_issue] = issue_counts.get(base_issue, 0) + 1
            
            for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1])[:10]:
                print(f"  - {issue}: {count} occurrences")
        
        # Detailed results
        print(f"\n📝 Detailed Results:")
        print("-"*70)
        
        for name, result in sorted(self.results.items()):
            status_icon = {
                "success": "✅",
                "completed_with_issues": "⚠️",
                "failed": "❌",
                "timeout": "⏱️",
                "pending": "⏸️"
            }.get(result["status"], "❓")
            
            print(f"\n{status_icon} {name}")
            print(f"   Status: {result['status']}")
            print(f"   Quality: {result['output_quality']:.0f}%")
            print(f"   Execution: {result['execution_time']:.1f}s")
            
            if result["outputs"]:
                print(f"   Output Files: {len(result['outputs'])}")
                for output in result["outputs"][:3]:
                    print(f"     - {output}")
                if len(result["outputs"]) > 3:
                    print(f"     ... and {len(result['outputs'])-3} more")
            
            if result["issues"]:
                print(f"   Issues:")
                for issue in result["issues"][:3]:
                    print(f"     - {issue}")
                if len(result["issues"]) > 3:
                    print(f"     ... and {len(result['issues'])-3} more")
        
        # Save full report
        report_path = self.base_output_dir / "full_validation_report.json"
        with open(report_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\n💾 Full report saved to: {report_path}")
        
        # Save summary
        summary_path = self.base_output_dir / "validation_summary.md"
        with open(summary_path, "w") as f:
            f.write("# Pipeline Validation Summary\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            f.write("## Statistics\n\n")
            f.write(f"- Total: {total}\n")
            f.write(f"- Success: {successful}\n")
            f.write(f"- With Issues: {with_issues}\n")
            f.write(f"- Failed: {failed}\n")
            f.write(f"- Timeout: {timeout}\n\n")
            
            f.write("## Pipeline Status\n\n")
            f.write("| Pipeline | Status | Quality | Time | Issues |\n")
            f.write("|----------|--------|---------|------|--------|\n")
            
            for name, result in sorted(self.results.items()):
                status = result["status"]
                quality = f"{result['output_quality']:.0f}%" if result["output_quality"] > 0 else "N/A"
                time_str = f"{result['execution_time']:.1f}s" if result["execution_time"] > 0 else "N/A"
                issues = len(result["issues"])
                f.write(f"| {name} | {status} | {quality} | {time_str} | {issues} |\n")
        
        print(f"📄 Summary saved to: {summary_path}")


async def main():
    """Main entry point."""
    validator = ComprehensivePipelineValidator()
    await validator.validate_all()


if __name__ == "__main__":
    # Set up environment
    os.environ["LOG_LEVEL"] = "WARNING"  # Reduce noise
    
    # Run validation
    asyncio.run(main())