#!/usr/bin/env python3
"""
Template Resolution Validation Script - Issue #275 Stream D

Comprehensive validation of template resolution fixes across all orchestrator components.
Tests all aspects of template resolution and provides detailed reporting.
"""

import asyncio
import json
import os
import sys
import time
import traceback
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Add orchestrator to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator import Orchestrator, init_models
from orchestrator.models import get_model_registry
from orchestrator.compiler.yaml_compiler import YAMLCompiler
from orchestrator.control_systems.hybrid_control_system import HybridControlSystem
from src.orchestrator.core.unified_template_resolver import (
    UnifiedTemplateResolver,
    TemplateResolutionContext
)


class TemplateResolutionValidator:
    """Comprehensive template resolution validation system."""
    
    def __init__(self):
        self.results = {}
        self.issues = []
        self.examples_dir = Path("examples")
        self.output_dir = Path("examples/outputs/template_validation")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_registry = None
        self.orchestrator = None
        
    async def initialize(self):
        """Initialize models and orchestrator."""
        print("Initializing models and orchestrator...")
        self.model_registry = init_models()
        
        if not self.model_registry or not self.model_registry.models:
            print("❌ No models available. Template resolution testing will be limited.")
            return False
        
        control_system = HybridControlSystem(self.model_registry)
        self.orchestrator = Orchestrator(
            model_registry=self.model_registry,
            control_system=control_system
        )
        
        print(f"✅ Initialized with {len(self.model_registry.models)} models available")
        return True
    
    async def validate_core_template_resolution(self) -> Dict[str, Any]:
        """Test core template resolution functionality (Stream A validation)."""
        
        print("\n" + "="*60)
        print("VALIDATING CORE TEMPLATE RESOLUTION (Stream A)")
        print("="*60)
        
        resolver = UnifiedTemplateResolver(debug_mode=True)
        results = {
            "status": "pending",
            "tests_passed": 0,
            "tests_total": 0,
            "issues": []
        }
        
        test_scenarios = [
            {
                "name": "Jinja2 $variable syntax preprocessing",
                "template": "Item: {{ $item }}, Index: {{ $index }}",
                "context": {"item": "test_value", "index": 5},
                "expected": "Item: test_value, Index: 5"
            },
            {
                "name": "Cross-step reference resolution",
                "template": "Content: {{ read_file.content }}, Size: {{ read_file.size }}",
                "context": {"read_file": {"content": "sample content", "size": 1024}},
                "expected": "Content: sample content, Size: 1024"
            },
            {
                "name": "Nested data structure resolution",
                "template": "{{ data.nested.value }} from {{ data.source }}",
                "context": {"data": {"nested": {"value": "deep_value"}, "source": "test"}},
                "expected": "deep_value from test"
            },
            {
                "name": "Complex mixed template",
                "template": "Processing {{ $item }} ({{ $index }}) with {{ result.status }}",
                "context": {"item": "file.txt", "index": 1, "result": {"status": "success"}},
                "expected": "Processing file.txt (1) with success"
            }
        ]
        
        results["tests_total"] = len(test_scenarios)
        
        for scenario in test_scenarios:
            try:
                context = resolver.collect_context(
                    pipeline_id="validation_test",
                    additional_context=scenario["context"]
                )
                
                result = resolver.resolve_templates(scenario["template"], context)
                
                if result == scenario["expected"]:
                    results["tests_passed"] += 1
                    print(f"✅ {scenario['name']}: PASSED")
                else:
                    results["issues"].append(f"{scenario['name']}: Expected '{scenario['expected']}', got '{result}'")
                    print(f"❌ {scenario['name']}: FAILED - Expected '{scenario['expected']}', got '{result}'")
                
                # Check for unresolved templates
                if "{{" in result or "}}" in result:
                    results["issues"].append(f"{scenario['name']}: Unresolved templates remain: '{result}'")
                    print(f"⚠️  {scenario['name']}: Unresolved templates detected")
                    
            except Exception as e:
                results["issues"].append(f"{scenario['name']}: Exception - {str(e)}")
                print(f"❌ {scenario['name']}: EXCEPTION - {str(e)}")
        
        # Test debug capabilities
        try:
            context = resolver.collect_context(pipeline_id="debug_test")
            resolver.register_context(context)
            debug_info = resolver.get_debug_info()
            
            if debug_info and "has_current_context" in debug_info:
                results["tests_passed"] += 1
                print(f"✅ Debug info retrieval: PASSED")
            else:
                results["issues"].append("Debug info retrieval failed")
                print(f"❌ Debug info retrieval: FAILED")
                
            results["tests_total"] += 1
            
        except Exception as e:
            results["issues"].append(f"Debug capabilities: Exception - {str(e)}")
            print(f"❌ Debug capabilities: EXCEPTION - {str(e)}")
            results["tests_total"] += 1
        
        results["status"] = "success" if results["tests_passed"] == results["tests_total"] else "partial"
        
        print(f"\nCore Template Resolution: {results['tests_passed']}/{results['tests_total']} tests passed")
        return results
    
    def create_test_data_files(self, temp_dir: Path) -> Path:
        """Create test data files for pipeline validation."""
        data_dir = temp_dir / "data"
        data_dir.mkdir(exist_ok=True)
        
        # Create test files with meaningful content
        test_files = {
            "file1.txt": "Renewable energy sources are becoming increasingly important for sustainable development.",
            "file2.txt": "Machine learning algorithms can analyze large datasets to find meaningful patterns.",
            "file3.txt": "Climate change mitigation requires coordinated global efforts and innovative solutions."
        }
        
        for filename, content in test_files.items():
            (data_dir / filename).write_text(content)
        
        return data_dir
    
    async def validate_control_flow_for_loop(self) -> Dict[str, Any]:
        """Validate template resolution in control_flow_for_loop.yaml pipeline."""
        
        print("\n" + "="*60)
        print("VALIDATING CONTROL FLOW FOR LOOP PIPELINE")
        print("="*60)
        
        results = {
            "status": "pending",
            "execution_success": False,
            "template_issues": [],
            "loop_variables": {"resolved": 0, "total": 4},  # $item, $index, $is_first, $is_last
            "cross_step_references": {"resolved": 0, "total": 0},
            "ai_model_issues": []
        }
        
        if not self.orchestrator:
            results["status"] = "skipped"
            results["template_issues"].append("No orchestrator available")
            return results
        
        pipeline_path = self.examples_dir / "control_flow_for_loop.yaml"
        if not pipeline_path.exists():
            results["status"] = "skipped"
            results["template_issues"].append("Pipeline file not found")
            return results
        
        try:
            # Create test data
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                data_dir = self.create_test_data_files(temp_path)
                output_dir = temp_path / "output"
                
                # Load and modify pipeline
                with open(pipeline_path, 'r') as f:
                    pipeline_yaml = f.read()
                
                # Modify to use test data
                pipeline_yaml = pipeline_yaml.replace(
                    'path: "data/{{ $item }}"',
                    f'path: "{data_dir}/{{{{ $item }}}}"'
                )
                
                # Execute pipeline
                inputs = {
                    "file_list": ["file1.txt", "file2.txt", "file3.txt"],
                    "output_dir": str(output_dir)
                }
                
                print(f"Executing control_flow_for_loop pipeline...")
                execution_results = await self.orchestrator.execute_yaml(pipeline_yaml, inputs)
                
                results["execution_success"] = execution_results is not None
                
                if output_dir.exists():
                    # Validate output files
                    processed_files = sorted(output_dir.glob("processed_*.txt"))
                    
                    if len(processed_files) == 3:
                        print(f"✅ Generated {len(processed_files)} processed files")
                        
                        for i, file_path in enumerate(processed_files):
                            content = file_path.read_text()
                            
                            # Check loop variables
                            loop_vars_in_file = {
                                "$item": f"file{i+1}.txt" in content,
                                "$index": f"Index: {i}" in content,
                                "$is_first": f"Is first: {i == 0}" in content,
                                "$is_last": f"Is last: {i == 2}" in content
                            }
                            
                            for var, resolved in loop_vars_in_file.items():
                                if resolved:
                                    results["loop_variables"]["resolved"] += 1 / 3  # Divide by number of files
                            
                            # Check for unresolved templates
                            if "{{" in content or "}}" in content:
                                results["template_issues"].append(f"Unresolved templates in {file_path.name}")
                            
                            # Check for loop variable artifacts
                            if any(var in content for var in ["$item", "$index", "$is_first", "$is_last"]):
                                results["template_issues"].append(f"Unresolved loop variables in {file_path.name}")
                            
                            # Check cross-step references
                            if "File Size:" in content:
                                results["cross_step_references"]["total"] += 1
                                if "None bytes" not in content and "{{ read_file.size }}" not in content:
                                    results["cross_step_references"]["resolved"] += 1
                            
                            # Check for AI model issues
                            ai_confusion_markers = [
                                "I don't have access to",
                                "placeholder didn't load",
                                "variable was not provided"
                            ]
                            
                            for marker in ai_confusion_markers:
                                if marker in content:
                                    results["ai_model_issues"].append(f"AI confusion in {file_path.name}: {marker}")
                    else:
                        results["template_issues"].append(f"Expected 3 processed files, got {len(processed_files)}")
                else:
                    results["template_issues"].append("Output directory not created")
                
        except Exception as e:
            results["template_issues"].append(f"Execution error: {str(e)}")
            print(f"❌ Pipeline execution failed: {str(e)}")
        
        # Determine overall status
        if results["execution_success"] and not results["template_issues"]:
            results["status"] = "success"
        elif results["execution_success"]:
            results["status"] = "partial"
        else:
            results["status"] = "failed"
        
        # Report results
        print(f"Execution Success: {results['execution_success']}")
        print(f"Loop Variables Resolved: {results['loop_variables']['resolved']:.1f}/{results['loop_variables']['total']}")
        print(f"Cross-step References: {results['cross_step_references']['resolved']}/{results['cross_step_references']['total']}")
        print(f"Template Issues: {len(results['template_issues'])}")
        print(f"AI Model Issues: {len(results['ai_model_issues'])}")
        
        return results
    
    async def validate_multiple_pipelines(self) -> Dict[str, Any]:
        """Validate template resolution across multiple priority pipelines."""
        
        print("\n" + "="*60)
        print("VALIDATING MULTIPLE PIPELINES")
        print("="*60)
        
        priority_pipelines = [
            "control_flow_for_loop.yaml",
            "control_flow_conditional.yaml",
            "simple_data_processing.yaml", 
            "data_processing_pipeline.yaml",
            "control_flow_advanced.yaml"
        ]
        
        results = {
            "pipelines_tested": 0,
            "pipelines_executed": 0,
            "pipelines_clean": 0,
            "pipeline_results": {}
        }
        
        if not self.orchestrator:
            print("⚠️ No orchestrator available - skipping pipeline validation")
            return results
        
        for pipeline_name in priority_pipelines:
            pipeline_path = self.examples_dir / pipeline_name
            
            if not pipeline_path.exists():
                results["pipeline_results"][pipeline_name] = {
                    "status": "not_found",
                    "issues": ["Pipeline file not found"]
                }
                continue
            
            results["pipelines_tested"] += 1
            print(f"\nTesting {pipeline_name}...")
            
            try:
                with open(pipeline_path, 'r') as f:
                    pipeline_yaml = f.read()
                
                # Simple test inputs
                inputs = {
                    "input_text": "Test input for template validation",
                    "data": [{"name": "test", "value": 123}],
                    "output_path": str(self.output_dir / f"test_{pipeline_name}"),
                    "file_list": ["file1.txt", "file2.txt"],
                    "topic": "template validation"
                }
                
                # Execute with timeout
                try:
                    execution_results = await asyncio.wait_for(
                        self.orchestrator.execute_yaml(pipeline_yaml, inputs),
                        timeout=30.0  # 30 second timeout for validation
                    )
                    
                    results["pipelines_executed"] += 1
                    
                    # Analyze results for template issues
                    issues = []
                    result_str = str(execution_results)
                    
                    if "{{" in result_str or "}}" in result_str:
                        issues.append("Unresolved Jinja2 templates")
                    
                    if any(var in result_str for var in ["$item", "$index", "$is_first", "$is_last"]):
                        issues.append("Unresolved loop variables")
                    
                    if "I don't have access to" in result_str or "placeholder" in result_str.lower():
                        issues.append("AI model template confusion")
                    
                    results["pipeline_results"][pipeline_name] = {
                        "status": "executed",
                        "issues": issues
                    }
                    
                    if not issues:
                        results["pipelines_clean"] += 1
                        print(f"✅ {pipeline_name}: Clean execution")
                    else:
                        print(f"⚠️  {pipeline_name}: {len(issues)} template issues")
                        
                except asyncio.TimeoutError:
                    results["pipeline_results"][pipeline_name] = {
                        "status": "timeout", 
                        "issues": ["Execution timeout"]
                    }
                    print(f"⏱️  {pipeline_name}: Timeout")
                    
            except Exception as e:
                results["pipeline_results"][pipeline_name] = {
                    "status": "error",
                    "issues": [f"Error: {str(e)}"]
                }
                print(f"❌ {pipeline_name}: Error - {str(e)}")
        
        return results
    
    async def run_comprehensive_validation(self):
        """Run comprehensive template resolution validation."""
        
        print("🔍 COMPREHENSIVE TEMPLATE RESOLUTION VALIDATION")
        print("=" * 80)
        print("Issue #275 - Stream D: Integration Testing & Validation")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print("=" * 80)
        
        # Initialize
        init_success = await self.initialize()
        
        validation_results = {
            "timestamp": datetime.now().isoformat(),
            "initialization": {"success": init_success},
            "core_resolution": {},
            "control_flow_for_loop": {},
            "multiple_pipelines": {},
            "summary": {}
        }
        
        # Run validation phases
        
        # Phase 1: Core template resolution
        validation_results["core_resolution"] = await self.validate_core_template_resolution()
        
        # Phase 2: Control flow for loop (key test case)
        if init_success:
            validation_results["control_flow_for_loop"] = await self.validate_control_flow_for_loop()
        
        # Phase 3: Multiple pipelines
        if init_success:
            validation_results["multiple_pipelines"] = await self.validate_multiple_pipelines()
        
        # Generate summary
        self.generate_validation_report(validation_results)
        
        return validation_results
    
    def generate_validation_report(self, results: Dict[str, Any]):
        """Generate comprehensive validation report."""
        
        print("\n" + "=" * 80)
        print("TEMPLATE RESOLUTION VALIDATION REPORT")
        print("=" * 80)
        
        # Core resolution summary
        core = results["core_resolution"]
        print(f"\n📋 CORE TEMPLATE RESOLUTION (Stream A Validation)")
        print(f"   Status: {core.get('status', 'unknown')}")
        print(f"   Tests: {core.get('tests_passed', 0)}/{core.get('tests_total', 0)} passed")
        if core.get("issues"):
            print(f"   Issues: {len(core['issues'])}")
            for issue in core["issues"][:3]:  # Show first 3
                print(f"      - {issue}")
        
        # Control flow validation
        if "control_flow_for_loop" in results:
            cf = results["control_flow_for_loop"]
            print(f"\n🔄 CONTROL FLOW FOR LOOP VALIDATION")
            print(f"   Status: {cf.get('status', 'unknown')}")
            print(f"   Execution: {'✅' if cf.get('execution_success') else '❌'}")
            print(f"   Loop Variables: {cf.get('loop_variables', {}).get('resolved', 0):.1f}/{cf.get('loop_variables', {}).get('total', 4)}")
            print(f"   Template Issues: {len(cf.get('template_issues', []))}")
            print(f"   AI Model Issues: {len(cf.get('ai_model_issues', []))}")
        
        # Multiple pipelines summary
        if "multiple_pipelines" in results:
            mp = results["multiple_pipelines"]
            print(f"\n🔀 MULTIPLE PIPELINES VALIDATION")
            print(f"   Tested: {mp.get('pipelines_tested', 0)}")
            print(f"   Executed: {mp.get('pipelines_executed', 0)}")
            print(f"   Clean: {mp.get('pipelines_clean', 0)}")
        
        # Overall assessment
        print(f"\n📊 OVERALL ASSESSMENT")
        
        # Calculate progress indicators
        core_progress = core.get('tests_passed', 0) / max(core.get('tests_total', 1), 1) * 100
        
        template_health = "🟢 EXCELLENT" if core_progress >= 90 else "🟡 GOOD" if core_progress >= 70 else "🔴 NEEDS_WORK"
        
        print(f"   Core Resolution: {core_progress:.0f}% - {template_health}")
        
        if "control_flow_for_loop" in results:
            cf = results["control_flow_for_loop"]
            loop_success = cf.get("execution_success", False) and len(cf.get("template_issues", [])) == 0
            loop_status = "🟢 SUCCESS" if loop_success else "🟡 PARTIAL" if cf.get("execution_success") else "🔴 FAILED"
            print(f"   Loop Processing: {loop_status}")
        
        if "multiple_pipelines" in results:
            mp = results["multiple_pipelines"]
            clean_rate = mp.get('pipelines_clean', 0) / max(mp.get('pipelines_tested', 1), 1) * 100
            pipeline_health = "🟢 EXCELLENT" if clean_rate >= 80 else "🟡 GOOD" if clean_rate >= 50 else "🔴 NEEDS_WORK"
            print(f"   Pipeline Health: {clean_rate:.0f}% clean - {pipeline_health}")
        
        # Stream progress assessment
        print(f"\n🚀 STREAM PROGRESS ASSESSMENT")
        print(f"   Stream A (Core Resolution): {'🟢 COMPLETE' if core_progress >= 90 else '🟡 IN_PROGRESS'}")
        
        if "control_flow_for_loop" in results:
            cf = results["control_flow_for_loop"]
            loop_vars_complete = cf.get('loop_variables', {}).get('resolved', 0) >= 3
            print(f"   Stream B (Loop Context): {'🟢 MAJOR_PROGRESS' if loop_vars_complete else '🟡 IN_PROGRESS'}")
        
        print(f"   Stream C (Tool Integration): 🟡 IN_PROGRESS (based on AI model issues)")
        print(f"   Stream D (Testing): 🟢 ACTIVE (this validation)")
        
        # Next steps
        print(f"\n📋 RECOMMENDED NEXT STEPS")
        if core_progress < 90:
            print(f"   1. Complete Stream A core template resolution fixes")
        
        if "control_flow_for_loop" in results:
            cf = results["control_flow_for_loop"]
            if cf.get("cross_step_references", {}).get("resolved", 0) == 0:
                print(f"   2. Fix cross-step reference resolution (Stream B)")
            if cf.get("ai_model_issues"):
                print(f"   3. Fix AI model parameter resolution (Stream C)")
        
        print(f"   4. Continue comprehensive pipeline validation")
        print(f"   5. Set up regression prevention framework")
        
        # Save report
        report_path = self.output_dir / "template_resolution_validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Detailed report saved to: {report_path}")


async def main():
    """Run template resolution validation."""
    validator = TemplateResolutionValidator()
    await validator.run_comprehensive_validation()


if __name__ == "__main__":
    asyncio.run(main())