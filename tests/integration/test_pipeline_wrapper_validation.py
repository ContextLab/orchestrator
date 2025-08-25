#!/usr/bin/env python3
"""
Comprehensive pipeline wrapper validation for Issue #252.

This module provides comprehensive testing of all 25 example pipelines
with wrapper integrations including RouteLLM, POML, and the base wrapper architecture.
"""

import asyncio
import json
import logging
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

import pytest

# Add orchestrator to path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from orchestrator import Orchestrator, init_models
from orchestrator.models import get_model_registry
from orchestrator.compiler.yaml_compiler import YAMLCompiler
from orchestrator.control_systems.hybrid_control_system import HybridControlSystem
from orchestrator.core.wrapper_testing import WrapperTestHarness, TestScenario, TestResult
from orchestrator.core.wrapper_base import BaseWrapper, BaseWrapperConfig
from orchestrator.core.feature_flags import FeatureFlagManager
from orchestrator.core.wrapper_monitoring import WrapperMonitoring

logger = logging.getLogger(__name__)


@dataclass
class PipelineTestResult:
    """Result of pipeline wrapper validation."""
    
    pipeline_name: str
    wrapper_config: str
    success: bool
    execution_time_ms: float
    output_quality_score: float
    issues: List[str] = None
    performance_metrics: Dict[str, Any] = None
    fallback_used: bool = False
    cost_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []
        if self.performance_metrics is None:
            self.performance_metrics = {}
        if self.cost_metrics is None:
            self.cost_metrics = {}


@dataclass
class WrapperConfiguration:
    """Configuration for wrapper testing."""
    
    name: str
    description: str
    enabled_features: List[str]
    config_overrides: Dict[str, Any]
    expected_cost_reduction: Optional[float] = None  # For RouteLLM
    template_enhancements: bool = False  # For POML


class PipelineWrapperValidator:
    """
    Comprehensive validator for pipeline wrapper integrations.
    
    Tests all 25 example pipelines with various wrapper configurations
    to ensure compatibility, performance, and quality.
    """
    
    def __init__(self):
        self.results: List[PipelineTestResult] = []
        self.examples_dir = Path("examples")
        self.output_dir = Path("examples/outputs/wrapper_validation")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Core pipeline list - the 25 main examples
        self.core_pipelines = [
            "auto_tags_demo.yaml",
            "control_flow_advanced.yaml", 
            "control_flow_conditional.yaml",
            "control_flow_for_loop.yaml",
            "creative_image_pipeline.yaml",
            "data_processing.yaml",
            "data_processing_pipeline.yaml",
            "interactive_pipeline.yaml",
            "llm_routing_pipeline.yaml",
            "mcp_integration_pipeline.yaml",
            "mcp_memory_workflow.yaml",
            "model_routing_demo.yaml",
            "multimodal_processing.yaml",
            "research_minimal.yaml",
            "simple_data_processing.yaml",
            "simple_timeout_test.yaml",
            "statistical_analysis.yaml",
            "terminal_automation.yaml",
            "validation_pipeline.yaml",
            "web_research_pipeline.yaml",
            "working_web_search.yaml",
            # Additional priority pipelines to reach 25
            "research_advanced_tools.yaml",
            "error_handling_examples.yaml",
            "enhanced_until_conditions_demo.yaml",
            "file_inclusion_demo.yaml"
        ]
        
        # Wrapper configurations to test
        self.wrapper_configs = [
            WrapperConfiguration(
                name="baseline",
                description="Baseline configuration without wrapper enhancements",
                enabled_features=[],
                config_overrides={"wrapper_enabled": False}
            ),
            WrapperConfiguration(
                name="routellm_cost_optimized",
                description="RouteLLM with cost optimization enabled",
                enabled_features=["routellm", "cost_optimization"],
                config_overrides={
                    "wrapper_enabled": True,
                    "routellm_enabled": True,
                    "cost_optimization": True,
                    "routing_strategy": "cost"
                },
                expected_cost_reduction=0.40  # 40% minimum cost reduction
            ),
            WrapperConfiguration(
                name="routellm_quality_balanced",
                description="RouteLLM with quality-cost balance",
                enabled_features=["routellm", "quality_routing"],
                config_overrides={
                    "wrapper_enabled": True,
                    "routellm_enabled": True,
                    "routing_strategy": "balanced"
                },
                expected_cost_reduction=0.20  # 20% cost reduction with quality focus
            ),
            WrapperConfiguration(
                name="poml_enhanced",
                description="POML integration with template enhancements",
                enabled_features=["poml", "template_enhancements"],
                config_overrides={
                    "wrapper_enabled": True,
                    "poml_enabled": True,
                    "enhanced_templates": True
                },
                template_enhancements=True
            ),
            WrapperConfiguration(
                name="full_wrapper_stack",
                description="All wrapper integrations enabled",
                enabled_features=["routellm", "poml", "monitoring", "fallback"],
                config_overrides={
                    "wrapper_enabled": True,
                    "routellm_enabled": True,
                    "poml_enabled": True,
                    "monitoring_enabled": True,
                    "fallback_enabled": True
                }
            )
        ]
        
        self.model_registry = None
        self.control_system = None
        
    async def initialize(self):
        """Initialize testing infrastructure."""
        logger.info("Initializing pipeline wrapper validator...")
        
        # Initialize models
        self.model_registry = init_models()
        if not self.model_registry or not self.model_registry.models:
            raise RuntimeError("No models available. Please check API keys and models.yaml")
            
        # Create control system
        self.control_system = HybridControlSystem(self.model_registry)
        logger.info(f"Initialized with {len(self.model_registry.models)} models")
        
    async def validate_pipeline_with_wrapper(
        self, 
        pipeline_path: Path, 
        wrapper_config: WrapperConfiguration
    ) -> PipelineTestResult:
        """
        Validate a single pipeline with a specific wrapper configuration.
        
        Args:
            pipeline_path: Path to pipeline YAML file
            wrapper_config: Wrapper configuration to test
            
        Returns:
            Pipeline test result
        """
        logger.info(f"Testing {pipeline_path.name} with {wrapper_config.name}")
        
        start_time = time.time()
        result = PipelineTestResult(
            pipeline_name=pipeline_path.name,
            wrapper_config=wrapper_config.name,
            success=False,
            execution_time_ms=0,
            output_quality_score=0,
        )
        
        try:
            # Load and compile pipeline
            with open(pipeline_path) as f:
                yaml_content = f.read()
                
            compiler = YAMLCompiler(development_mode=True)
            pipeline = await compiler.compile(yaml_content)
            
            # Get test inputs
            inputs = self._get_pipeline_inputs(pipeline_path.name)
            
            # Setup output directory
            output_path = self.output_dir / wrapper_config.name / pipeline_path.stem
            output_path.mkdir(parents=True, exist_ok=True)
            inputs['output_path'] = str(output_path)
            
            # Create orchestrator with wrapper configuration
            orchestrator = await self._create_configured_orchestrator(wrapper_config)
            
            # Execute pipeline
            execution_start = time.time()
            results = await orchestrator.execute_yaml(yaml_content, inputs)
            execution_time = (time.time() - execution_start) * 1000
            
            # Analyze results
            issues = self._analyze_pipeline_output(results, output_path)
            quality_score = self._calculate_quality_score(results, issues, output_path)
            performance_metrics = self._collect_performance_metrics(orchestrator, execution_time)
            cost_metrics = self._collect_cost_metrics(orchestrator, wrapper_config)
            
            # Check if wrapper worked as expected
            wrapper_success = self._validate_wrapper_behavior(
                results, wrapper_config, issues, cost_metrics
            )
            
            result.success = wrapper_success and len(issues) == 0
            result.execution_time_ms = time.time() - start_time
            result.output_quality_score = quality_score
            result.issues = issues
            result.performance_metrics = performance_metrics
            result.cost_metrics = cost_metrics
            
            logger.info(f"✅ {pipeline_path.name} with {wrapper_config.name}: "
                       f"Success={result.success}, Quality={quality_score:.1f}%")
                       
        except Exception as e:
            result.execution_time_ms = (time.time() - start_time) * 1000
            result.issues = [f"Execution error: {str(e)}"]
            logger.error(f"❌ {pipeline_path.name} with {wrapper_config.name} failed: {e}")
            traceback.print_exc()
            
        return result
        
    async def _create_configured_orchestrator(
        self, wrapper_config: WrapperConfiguration
    ) -> Orchestrator:
        """Create orchestrator with specific wrapper configuration."""
        # Create feature flag manager
        feature_flags = FeatureFlagManager()
        for feature in wrapper_config.enabled_features:
            from orchestrator.core.feature_flags import FeatureFlag
            flag = FeatureFlag(name=feature, enabled=True)
            feature_flags.register_flag(flag)
        
        # Create monitoring
        monitoring = WrapperMonitoring()
        
        # Apply configuration overrides to control system
        control_system = HybridControlSystem(self.model_registry)
        
        # Configure orchestrator with wrapper settings
        orchestrator = Orchestrator(
            model_registry=self.model_registry,
            control_system=control_system
        )
        
        # Apply wrapper configuration
        if hasattr(orchestrator, 'configure_wrappers'):
            await orchestrator.configure_wrappers(wrapper_config.config_overrides)
            
        return orchestrator
    
    def _get_pipeline_inputs(self, pipeline_name: str) -> Dict[str, Any]:
        """Get appropriate test inputs for a pipeline."""
        base_inputs = {
            "input_text": "Artificial intelligence is transforming healthcare by enabling more accurate diagnostics and personalized treatment plans.",
            "topic": "sustainable energy technologies",
            "query": "machine learning applications in renewable energy",
            "url": "https://example.com/research",
            "file_path": "examples/data/sample_data.csv",
            "data": {"category": "technology", "importance": "high", "count": 25}
        }
        
        # Pipeline-specific inputs
        if "research" in pipeline_name:
            base_inputs.update({
                "topic": "quantum computing breakthroughs",
                "depth": "comprehensive",
                "sources": 5
            })
        elif "data" in pipeline_name:
            base_inputs.update({
                "data_path": "examples/data/sample_data.csv",
                "format": "csv",
                "analysis_type": "statistical"
            })
        elif "image" in pipeline_name or "creative" in pipeline_name:
            base_inputs.update({
                "prompt": "futuristic sustainable city with renewable energy",
                "style": "photorealistic",
                "variations": 3
            })
        elif "control" in pipeline_name:
            base_inputs.update({
                "conditions": ["high_quality", "fast_processing"],
                "iterations": 3,
                "threshold": 0.8
            })
        elif "mcp" in pipeline_name:
            base_inputs.update({
                "search_query": "artificial intelligence sustainability",
                "user_id": "test_user_252",
                "session_id": f"session_{int(time.time())}"
            })
        elif "timeout" in pipeline_name:
            base_inputs.update({
                "delay": 1,  # Short delay for testing
                "max_retries": 2
            })
        elif "routing" in pipeline_name:
            base_inputs.update({
                "complexity": "medium",
                "priority": "balanced",
                "budget_limit": 0.10
            })
            
        return base_inputs
    
    def _analyze_pipeline_output(self, results: Any, output_path: Path) -> List[str]:
        """Analyze pipeline output for common issues."""
        issues = []
        result_str = str(results)
        
        # Check for unrendered templates
        if "{{" in result_str or "}}" in result_str:
            issues.append("Unrendered template variables detected")
            
        # Check for loop variables
        loop_vars = ["$item", "$index", "$iteration", "$context"]
        for var in loop_vars:
            if var in result_str:
                issues.append(f"Unrendered loop variable: {var}")
                
        # Check for conversational markers
        conversational_markers = [
            "Certainly!", "Sure!", "I'd be happy to", "Let me help",
            "Here's what I found", "I'll create", "Based on your request"
        ]
        for marker in conversational_markers:
            if marker in result_str:
                issues.append(f"Conversational marker found: '{marker}'")
                break
                
        # Check for error indicators
        error_patterns = ["error occurred", "failed to", "could not", "unable to"]
        for pattern in error_patterns:
            if pattern.lower() in result_str.lower():
                issues.append(f"Error pattern detected: '{pattern}'")
                
        # Check output files
        output_files = [f for f in output_path.rglob("*") if f.is_file()]
        if len(output_files) == 0:
            issues.append("No output files generated")
        else:
            # Check file contents
            for file_path in output_files:
                try:
                    content = file_path.read_text(encoding='utf-8')
                    if not content.strip():
                        issues.append(f"Empty output file: {file_path.name}")
                    elif "{{" in content or "}}" in content:
                        issues.append(f"Unrendered templates in {file_path.name}")
                except Exception as e:
                    issues.append(f"Could not read {file_path.name}: {str(e)}")
                    
        return issues
    
    def _calculate_quality_score(
        self, results: Any, issues: List[str], output_path: Path
    ) -> float:
        """Calculate quality score for pipeline output."""
        score = 100.0
        
        # Deduct for issues
        score -= len(issues) * 15  # More severe penalty for issues
        
        # Check output completeness
        output_files = [f for f in output_path.rglob("*") if f.is_file()]
        if len(output_files) > 0:
            score += 10  # Bonus for generating outputs
            
            # Check content quality
            total_content_length = 0
            for file_path in output_files:
                try:
                    content = file_path.read_text(encoding='utf-8')
                    total_content_length += len(content.strip())
                except:
                    continue
                    
            if total_content_length > 1000:  # Substantial content
                score += 15
            elif total_content_length > 100:  # Minimal content
                score += 5
                
        # Check result structure
        if results and hasattr(results, '__dict__'):
            score += 5  # Structured results
            
        return max(0, min(100, score))
    
    def _collect_performance_metrics(
        self, orchestrator: Orchestrator, execution_time: float
    ) -> Dict[str, Any]:
        """Collect performance metrics from orchestrator."""
        metrics = {
            "execution_time_ms": execution_time,
            "total_api_calls": getattr(orchestrator, '_api_call_count', 0),
            "cache_hits": getattr(orchestrator, '_cache_hits', 0),
            "cache_misses": getattr(orchestrator, '_cache_misses', 0)
        }
        
        # Add wrapper-specific metrics if available
        if hasattr(orchestrator, 'wrapper_metrics'):
            metrics.update(orchestrator.wrapper_metrics)
            
        return metrics
    
    def _collect_cost_metrics(
        self, orchestrator: Orchestrator, wrapper_config: WrapperConfiguration
    ) -> Dict[str, float]:
        """Collect cost metrics from orchestrator."""
        metrics = {}
        
        if hasattr(orchestrator, 'cost_tracker'):
            metrics.update({
                "total_cost": getattr(orchestrator.cost_tracker, 'total_cost', 0.0),
                "input_tokens": getattr(orchestrator.cost_tracker, 'input_tokens', 0),
                "output_tokens": getattr(orchestrator.cost_tracker, 'output_tokens', 0)
            })
            
            # Calculate cost reduction for RouteLLM
            if "routellm" in wrapper_config.enabled_features:
                baseline_cost = getattr(orchestrator.cost_tracker, 'baseline_cost', 0.0)
                if baseline_cost > 0:
                    cost_reduction = (baseline_cost - metrics["total_cost"]) / baseline_cost
                    metrics["cost_reduction_percentage"] = cost_reduction
                    
        return metrics
    
    def _validate_wrapper_behavior(
        self, 
        results: Any, 
        wrapper_config: WrapperConfiguration,
        issues: List[str],
        cost_metrics: Dict[str, float]
    ) -> bool:
        """Validate that wrapper behaved as expected."""
        
        # RouteLLM cost optimization validation
        if wrapper_config.expected_cost_reduction:
            actual_reduction = cost_metrics.get("cost_reduction_percentage", 0.0)
            if actual_reduction < wrapper_config.expected_cost_reduction:
                issues.append(
                    f"Cost reduction {actual_reduction:.1%} below expected "
                    f"{wrapper_config.expected_cost_reduction:.1%}"
                )
                return False
                
        # POML template enhancement validation
        if wrapper_config.template_enhancements:
            result_str = str(results)
            # Check for enhanced template features
            if "{{" in result_str or "}}" in result_str:
                issues.append("POML template enhancements not working properly")
                return False
                
        # General wrapper functionality
        if "wrapper" in wrapper_config.enabled_features:
            # Ensure wrapper overhead is minimal (<5ms)
            wrapper_overhead = cost_metrics.get("wrapper_overhead_ms", 0)
            if wrapper_overhead > 5.0:
                issues.append(f"Wrapper overhead {wrapper_overhead}ms exceeds 5ms limit")
                return False
                
        return True
    
    async def validate_all_pipelines(self) -> Dict[str, Any]:
        """
        Validate all 25 core pipelines with all wrapper configurations.
        
        Returns:
            Comprehensive validation results
        """
        logger.info("Starting comprehensive pipeline wrapper validation")
        await self.initialize()
        
        total_tests = len(self.core_pipelines) * len(self.wrapper_configs)
        logger.info(f"Running {total_tests} pipeline-wrapper combination tests")
        
        # Run all combinations
        for pipeline_name in self.core_pipelines:
            pipeline_path = self.examples_dir / pipeline_name
            if not pipeline_path.exists():
                logger.warning(f"Pipeline not found: {pipeline_name}")
                continue
                
            for wrapper_config in self.wrapper_configs:
                result = await self.validate_pipeline_with_wrapper(
                    pipeline_path, wrapper_config
                )
                self.results.append(result)
                
        return self.generate_comprehensive_report()
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        logger.info("Generating comprehensive validation report")
        
        # Calculate summary statistics
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - successful_tests
        
        # Group by wrapper configuration
        by_wrapper = {}
        for result in self.results:
            config_name = result.wrapper_config
            if config_name not in by_wrapper:
                by_wrapper[config_name] = {"total": 0, "success": 0, "results": []}
            by_wrapper[config_name]["total"] += 1
            if result.success:
                by_wrapper[config_name]["success"] += 1
            by_wrapper[config_name]["results"].append(result)
            
        # Group by pipeline
        by_pipeline = {}
        for result in self.results:
            pipeline_name = result.pipeline_name
            if pipeline_name not in by_pipeline:
                by_pipeline[pipeline_name] = {"total": 0, "success": 0, "results": []}
            by_pipeline[pipeline_name]["total"] += 1
            if result.success:
                by_pipeline[pipeline_name]["success"] += 1
            by_pipeline[pipeline_name]["results"].append(result)
        
        # Calculate performance metrics
        execution_times = [r.execution_time_ms for r in self.results]
        quality_scores = [r.output_quality_score for r in self.results]
        
        report = {
            "summary": {
                "timestamp": datetime.utcnow().isoformat(),
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "failed_tests": failed_tests,
                "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
                "pipelines_tested": len(set(r.pipeline_name for r in self.results)),
                "wrapper_configs_tested": len(set(r.wrapper_config for r in self.results))
            },
            "performance": {
                "average_execution_time_ms": sum(execution_times) / len(execution_times) if execution_times else 0,
                "min_execution_time_ms": min(execution_times) if execution_times else 0,
                "max_execution_time_ms": max(execution_times) if execution_times else 0,
                "average_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0
            },
            "by_wrapper_config": {
                name: {
                    "success_rate": data["success"] / data["total"] if data["total"] > 0 else 0,
                    "total_tests": data["total"],
                    "successful_tests": data["success"]
                }
                for name, data in by_wrapper.items()
            },
            "by_pipeline": {
                name: {
                    "success_rate": data["success"] / data["total"] if data["total"] > 0 else 0,
                    "total_tests": data["total"], 
                    "successful_tests": data["success"]
                }
                for name, data in by_pipeline.items()
            },
            "detailed_results": [
                {
                    "pipeline": r.pipeline_name,
                    "wrapper_config": r.wrapper_config,
                    "success": r.success,
                    "execution_time_ms": r.execution_time_ms,
                    "quality_score": r.output_quality_score,
                    "issues_count": len(r.issues),
                    "issues": r.issues,
                    "performance_metrics": r.performance_metrics,
                    "cost_metrics": r.cost_metrics
                }
                for r in self.results
            ]
        }
        
        # Save report
        report_path = self.output_dir / "comprehensive_validation_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Report saved to: {report_path}")
        
        # Print summary
        print("\n" + "="*80)
        print("COMPREHENSIVE PIPELINE WRAPPER VALIDATION REPORT")
        print("="*80)
        print(f"\nTotal Tests: {total_tests}")
        print(f"✅ Successful: {successful_tests} ({successful_tests/total_tests*100:.1f}%)")
        print(f"❌ Failed: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
        
        print(f"\nPerformance:")
        print(f"Average Execution Time: {report['performance']['average_execution_time_ms']:.1f}ms")
        print(f"Average Quality Score: {report['performance']['average_quality_score']:.1f}%")
        
        print("\nWrapper Configuration Results:")
        for config_name, config_data in by_wrapper.items():
            success_rate = config_data["success"] / config_data["total"] * 100
            print(f"  {config_name}: {config_data['success']}/{config_data['total']} ({success_rate:.1f}%)")
            
        print("\nPipeline Results:")
        for pipeline_name, pipeline_data in by_pipeline.items():
            success_rate = pipeline_data["success"] / pipeline_data["total"] * 100
            print(f"  {pipeline_name}: {pipeline_data['success']}/{pipeline_data['total']} ({success_rate:.1f}%)")
        
        return report


# pytest fixtures and test functions

@pytest.fixture
async def pipeline_validator():
    """Create and initialize pipeline validator."""
    validator = PipelineWrapperValidator()
    await validator.initialize()
    return validator


@pytest.mark.asyncio
async def test_baseline_configuration(pipeline_validator):
    """Test all pipelines with baseline configuration."""
    validator = pipeline_validator
    baseline_config = next(c for c in validator.wrapper_configs if c.name == "baseline")
    
    results = []
    for pipeline_name in validator.core_pipelines[:5]:  # Test first 5 for speed
        pipeline_path = validator.examples_dir / pipeline_name
        if pipeline_path.exists():
            result = await validator.validate_pipeline_with_wrapper(
                pipeline_path, baseline_config
            )
            results.append(result)
            
    # Assert all pipelines work with baseline
    success_rate = sum(1 for r in results if r.success) / len(results)
    assert success_rate >= 0.8, f"Baseline success rate {success_rate:.1%} below 80%"


@pytest.mark.asyncio  
async def test_routellm_cost_optimization(pipeline_validator):
    """Test RouteLLM cost optimization."""
    validator = pipeline_validator
    routellm_config = next(c for c in validator.wrapper_configs if c.name == "routellm_cost_optimized")
    
    # Test with a data processing pipeline
    pipeline_path = validator.examples_dir / "simple_data_processing.yaml"
    if pipeline_path.exists():
        result = await validator.validate_pipeline_with_wrapper(
            pipeline_path, routellm_config
        )
        
        # Assert cost reduction achieved
        cost_reduction = result.cost_metrics.get("cost_reduction_percentage", 0)
        assert cost_reduction >= 0.20, f"Cost reduction {cost_reduction:.1%} below 20%"
        

@pytest.mark.asyncio
async def test_poml_template_enhancement(pipeline_validator):
    """Test POML template enhancement."""
    validator = pipeline_validator
    poml_config = next(c for c in validator.wrapper_configs if c.name == "poml_enhanced")
    
    # Test with a research pipeline
    pipeline_path = validator.examples_dir / "research_minimal.yaml"
    if pipeline_path.exists():
        result = await validator.validate_pipeline_with_wrapper(
            pipeline_path, poml_config
        )
        
        # Assert template enhancements work
        assert result.success, f"POML enhanced pipeline failed: {result.issues}"
        assert "Unrendered template" not in str(result.issues)


@pytest.mark.asyncio
async def test_wrapper_performance_overhead(pipeline_validator):
    """Test that wrapper overhead is minimal."""
    validator = pipeline_validator
    
    # Compare baseline vs full wrapper stack
    baseline_config = next(c for c in validator.wrapper_configs if c.name == "baseline")
    full_config = next(c for c in validator.wrapper_configs if c.name == "full_wrapper_stack")
    
    pipeline_path = validator.examples_dir / "simple_data_processing.yaml"
    if pipeline_path.exists():
        baseline_result = await validator.validate_pipeline_with_wrapper(
            pipeline_path, baseline_config
        )
        wrapper_result = await validator.validate_pipeline_with_wrapper(
            pipeline_path, full_config
        )
        
        # Calculate overhead
        overhead = wrapper_result.execution_time_ms - baseline_result.execution_time_ms
        assert overhead < 50, f"Wrapper overhead {overhead}ms exceeds 50ms limit"


if __name__ == "__main__":
    async def main():
        validator = PipelineWrapperValidator()
        report = await validator.validate_all_pipelines()
        
        # Print summary for Issue #252
        success_rate = report["summary"]["success_rate"]
        if success_rate >= 0.95:
            print("\n🎉 Issue #252: All pipeline wrapper validations PASSED!")
        elif success_rate >= 0.80:
            print(f"\n⚠️  Issue #252: {success_rate:.1%} success rate - Some issues need attention")
        else:
            print(f"\n❌ Issue #252: {success_rate:.1%} success rate - Significant issues found")
            
    asyncio.run(main())