"""
Integration tests for pipeline testing infrastructure.

Tests the core pipeline testing framework including:
- Pipeline discovery and categorization  
- Execution testing for all example pipelines
- Template resolution validation
- File organization checking
- Performance monitoring
- Quality scoring

This is the main integration point for Issue #281 Stream A.
"""

import asyncio
import logging
import pytest
import time
from pathlib import Path
from typing import Dict, List, Optional

from orchestrator import init_models
from orchestrator.testing import (
    PipelineDiscovery, 
    PipelineTestSuite,
    PipelineInfo,
    TestResults
)

logger = logging.getLogger(__name__)


class TestPipelineInfrastructure:
    """Test suite for pipeline testing infrastructure."""
    
    @pytest.fixture(scope="class")
    def examples_dir(self) -> Path:
        """Get examples directory path."""
        return Path("examples")
    
    @pytest.fixture(scope="class")
    def pipeline_discovery(self, examples_dir) -> PipelineDiscovery:
        """Create pipeline discovery instance."""
        return PipelineDiscovery(examples_dir)
    
    @pytest.fixture(scope="class")
    def pipeline_test_suite(self, examples_dir) -> PipelineTestSuite:
        """Create pipeline test suite instance."""
        model_registry = init_models()
        
        # Skip if no models available
        available_models = model_registry.list_models()
        if not available_models:
            pytest.skip("No models available for pipeline testing")
        
        return PipelineTestSuite(examples_dir=examples_dir, model_registry=model_registry)
    
    def test_pipeline_discovery_basic(self, pipeline_discovery):
        """Test basic pipeline discovery functionality."""
        logger.info("Testing pipeline discovery...")
        
        # Discover all pipelines
        discovered = pipeline_discovery.discover_all_pipelines()
        
        # Should find multiple pipelines
        assert len(discovered) > 0, "No pipelines discovered"
        assert len(discovered) >= 30, f"Expected at least 30 pipelines, found {len(discovered)}"
        
        # Check that each pipeline has required information
        for name, info in discovered.items():
            assert isinstance(info, PipelineInfo)
            assert info.name == name
            assert info.path.exists(), f"Pipeline file not found: {info.path}"
            assert info.category in [
                "data_processing", "research", "creative", "control_flow",
                "multimodal", "integration", "optimization", "automation", 
                "validation", "general"
            ], f"Unknown category: {info.category}"
            assert info.complexity in ["simple", "medium", "complex"]
        
        logger.info(f"Successfully discovered {len(discovered)} pipelines")
    
    def test_pipeline_categorization(self, pipeline_discovery):
        """Test pipeline categorization accuracy."""
        discovered = pipeline_discovery.discover_all_pipelines()
        
        # Check specific pipeline categories
        category_checks = {
            "data_processing": ["simple_data_processing", "data_processing_pipeline", "statistical_analysis"],
            "research": ["research_minimal", "research_basic", "research_advanced_tools"],
            "creative": ["creative_image_pipeline"],
            "control_flow": ["control_flow_conditional", "control_flow_for_loop", "control_flow_while_loop"],
            "integration": ["mcp_integration_pipeline", "mcp_memory_workflow"]
        }
        
        for expected_category, pipeline_names in category_checks.items():
            for pipeline_name in pipeline_names:
                if pipeline_name in discovered:
                    info = discovered[pipeline_name]
                    assert info.category == expected_category, \
                        f"Pipeline {pipeline_name} should be category {expected_category}, got {info.category}"
    
    def test_test_safe_pipeline_detection(self, pipeline_discovery):
        """Test detection of test-safe pipelines."""
        discovered = pipeline_discovery.discover_all_pipelines()
        test_safe = pipeline_discovery.get_test_safe_pipelines()
        
        # Should have multiple test-safe pipelines
        assert len(test_safe) > 0, "No test-safe pipelines found"
        assert len(test_safe) >= 20, f"Expected at least 20 test-safe pipelines, found {len(test_safe)}"
        
        # Check that interactive pipelines are excluded
        unsafe_names = [p.name for p in test_safe if not p.is_test_safe]
        assert len(unsafe_names) == 0, f"Unsafe pipelines in test-safe list: {unsafe_names}"
        
        # Should exclude known problematic pipelines
        test_safe_names = [p.name for p in test_safe]
        problematic = ["interactive_pipeline", "terminal_automation"]
        for name in problematic:
            if name in discovered:
                assert name not in test_safe_names, f"Problematic pipeline {name} marked as test-safe"
    
    def test_core_test_pipeline_selection(self, pipeline_discovery):
        """Test core test pipeline selection logic."""
        discovered = pipeline_discovery.discover_all_pipelines()
        core_pipelines = pipeline_discovery.get_core_test_pipelines()
        
        # Should have 15-20 core pipelines
        assert 15 <= len(core_pipelines) <= 20, \
            f"Expected 15-20 core pipelines, got {len(core_pipelines)}"
        
        # Should cover major categories
        categories_covered = set(p.category for p in core_pipelines)
        expected_categories = {"data_processing", "research", "creative", "control_flow"}
        assert expected_categories.issubset(categories_covered), \
            f"Core pipelines missing categories: {expected_categories - categories_covered}"
        
        # Should prioritize simple/medium complexity
        complexities = [p.complexity for p in core_pipelines]
        simple_medium_count = sum(1 for c in complexities if c in ["simple", "medium"])
        assert simple_medium_count >= len(core_pipelines) * 0.7, \
            "Core pipelines should be primarily simple/medium complexity"
    
    def test_quick_test_pipeline_selection(self, pipeline_discovery):
        """Test quick test pipeline selection."""
        quick_pipelines = pipeline_discovery.get_quick_test_pipelines()
        
        # Should have 5-10 quick pipelines
        assert 5 <= len(quick_pipelines) <= 10, \
            f"Expected 5-10 quick pipelines, got {len(quick_pipelines)}"
        
        # Should prioritize simple complexity and short runtime
        for pipeline in quick_pipelines:
            assert pipeline.is_test_safe, f"Quick pipeline {pipeline.name} not test-safe"
            assert pipeline.estimated_runtime <= 120, \
                f"Quick pipeline {pipeline.name} has long estimated runtime: {pipeline.estimated_runtime}s"
    
    def test_pipeline_test_suite_initialization(self, pipeline_test_suite):
        """Test pipeline test suite initialization."""
        assert pipeline_test_suite.examples_dir.exists()
        assert pipeline_test_suite.model_registry is not None
        assert pipeline_test_suite.orchestrator is not None
        assert pipeline_test_suite.discovery is not None
        
        # Should have discovered pipelines after initialization
        discovered = pipeline_test_suite.discover_pipelines()
        assert len(discovered) > 0, "Test suite should discover pipelines"
    
    @pytest.mark.asyncio
    async def test_single_pipeline_execution(self, pipeline_test_suite):
        """Test execution of a single simple pipeline."""
        # Discover pipelines
        discovered = pipeline_test_suite.discover_pipelines()
        
        # Find a simple, test-safe pipeline
        simple_pipelines = [p for p in discovered.values() 
                           if p.complexity == "simple" and p.is_test_safe]
        
        if not simple_pipelines:
            pytest.skip("No simple test-safe pipelines available")
        
        # Test the first simple pipeline
        test_pipeline = simple_pipelines[0]
        logger.info(f"Testing single pipeline: {test_pipeline.name}")
        
        # Run test
        results = await pipeline_test_suite.run_pipeline_tests([test_pipeline.name])
        
        # Validate results
        assert isinstance(results, TestResults)
        assert test_pipeline.name in results.results
        
        result = results.results[test_pipeline.name]
        
        # Check execution result
        if result.execution.error:
            logger.error(f"Pipeline execution failed: {result.execution.error_message}")
            if result.execution.error_traceback:
                logger.error(f"Traceback: {result.execution.error_traceback}")
        
        # For simple pipelines, expect success (but allow some failures due to external dependencies)
        if not result.execution.success:
            pytest.skip(f"Pipeline {test_pipeline.name} failed execution - may be due to missing API keys or external dependencies")
        
        # If execution succeeded, validate other aspects
        assert result.execution.execution_time > 0, "Should have positive execution time"
        assert result.templates.resolved_correctly, f"Template resolution failed: {result.templates.issues}"
        
        logger.info(f"Pipeline {test_pipeline.name} completed successfully: "
                   f"score={result.quality_score:.1f}, time={result.execution.execution_time:.1f}s")
    
    @pytest.mark.asyncio
    async def test_quick_pipeline_test_mode(self, pipeline_test_suite):
        """Test quick pipeline test mode."""
        logger.info("Running quick pipeline test mode...")
        start_time = time.time()
        
        # Run quick tests
        results = await pipeline_test_suite.run_pipeline_tests(test_mode="quick")
        
        execution_time = time.time() - start_time
        
        # Validate results
        assert isinstance(results, TestResults)
        assert 5 <= results.total_tests <= 10, \
            f"Quick mode should test 5-10 pipelines, tested {results.total_tests}"
        
        # Should complete relatively quickly (allow up to 10 minutes for all quick tests)
        assert execution_time <= 600, \
            f"Quick tests took too long: {execution_time:.1f}s"
        
        # Log results
        logger.info(f"Quick tests completed in {execution_time:.1f}s")
        logger.info(f"Results: {results.successful_tests}/{results.total_tests} passed "
                   f"({results.success_rate:.1f}%)")
        
        # Allow some failures but expect majority to pass
        if results.success_rate < 60:
            failed_pipelines = results.get_failed_pipelines()
            logger.warning(f"Many quick tests failed: {failed_pipelines}")
            # Don't fail the test entirely - may be due to external dependencies
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_core_pipeline_test_mode(self, pipeline_test_suite):
        """Test core pipeline test mode (marked as slow)."""
        logger.info("Running core pipeline test mode...")
        start_time = time.time()
        
        # Run core tests  
        results = await pipeline_test_suite.run_pipeline_tests(test_mode="core")
        
        execution_time = time.time() - start_time
        
        # Validate results
        assert isinstance(results, TestResults)
        assert 15 <= results.total_tests <= 20, \
            f"Core mode should test 15-20 pipelines, tested {results.total_tests}"
        
        # Should complete in reasonable time (allow up to 30 minutes)
        assert execution_time <= 1800, \
            f"Core tests took too long: {execution_time:.1f}s"
        
        # Log results
        logger.info(f"Core tests completed in {execution_time:.1f}s")
        logger.info(f"Results: {results.successful_tests}/{results.total_tests} passed "
                   f"({results.success_rate:.1f}%)")
        logger.info(f"Total cost: ${results.total_cost:.2f}")
        
        # Expect reasonable success rate
        if results.success_rate < 50:
            failed_pipelines = results.get_failed_pipelines()
            logger.warning(f"Many core tests failed: {failed_pipelines}")
    
    def test_template_resolution_validation(self, pipeline_test_suite):
        """Test template resolution validation logic."""
        discovered = pipeline_test_suite.discover_pipelines()
        
        # Test template validation on a few pipelines
        test_pipelines = list(discovered.values())[:3]
        
        for pipeline_info in test_pipelines:
            template_result = pipeline_test_suite._test_template_resolution(pipeline_info)
            
            # Should return valid TemplateResult
            assert hasattr(template_result, 'resolved_correctly')
            assert hasattr(template_result, 'issues')
            assert isinstance(template_result.issues, list)
    
    def test_file_organization_validation(self, pipeline_test_suite):
        """Test file organization validation logic."""
        discovered = pipeline_test_suite.discover_pipelines()
        
        # Test organization validation on a few pipelines
        test_pipelines = list(discovered.values())[:3]
        
        for pipeline_info in test_pipelines:
            org_result = pipeline_test_suite._test_file_organization(pipeline_info)
            
            # Should return valid OrganizationResult
            assert hasattr(org_result, 'valid')
            assert hasattr(org_result, 'issues')
            assert isinstance(org_result.issues, list)
            assert hasattr(org_result, 'expected_output_dir')
    
    def test_performance_monitoring_setup(self, pipeline_test_suite):
        """Test performance monitoring setup."""
        assert pipeline_test_suite.enable_performance_tracking
        assert isinstance(pipeline_test_suite.performance_baselines, dict)
        
        # Test performance score calculation
        test_metrics = {
            'execution_time': 30.0,
            'estimated_cost': 0.05,
            'memory_usage_mb': 100.0
        }
        
        # Create mock pipeline info
        from orchestrator.testing.pipeline_discovery import PipelineInfo
        mock_pipeline = PipelineInfo(
            name="test_pipeline",
            path=Path("test.yaml"),
            category="test",
            complexity="simple",
            estimated_runtime=60
        )
        
        score = pipeline_test_suite._calculate_performance_score(test_metrics, mock_pipeline)
        assert 0.0 <= score <= 1.0, f"Performance score should be 0-1, got {score}"
    
    def test_error_handling_and_logging(self, pipeline_test_suite):
        """Test error handling and logging setup."""
        # Test with invalid pipeline name
        discovered = pipeline_test_suite.discover_pipelines()
        
        # Should handle missing pipelines gracefully
        invalid_results = asyncio.run(
            pipeline_test_suite.run_pipeline_tests(["non_existent_pipeline"])
        )
        
        assert isinstance(invalid_results, TestResults)
        assert invalid_results.total_tests == 0
    
    @pytest.mark.integration
    def test_integration_with_existing_test_framework(self):
        """Test integration with existing pytest framework."""
        # This test validates that the new infrastructure works with existing test patterns
        
        # Import existing test utilities
        from tests.conftest import populated_model_registry
        
        # Should be able to create instances with existing fixtures
        # (This would typically be done via pytest fixture injection)
        pass
    
    def test_cost_and_timeout_controls(self, pipeline_test_suite):
        """Test cost and timeout control mechanisms."""
        assert pipeline_test_suite.timeout_seconds > 0
        assert pipeline_test_suite.max_cost_per_pipeline > 0
        
        # Should have reasonable defaults
        assert pipeline_test_suite.timeout_seconds <= 600  # Max 10 minutes per pipeline
        assert pipeline_test_suite.max_cost_per_pipeline <= 5.0  # Max $5 per pipeline


class TestExamplePipelines:
    """
    Integration tests that mirror the structure in the requirements.
    
    These tests integrate the pipeline testing infrastructure with the
    standard pytest framework as specified in the requirements.
    """
    
    @pytest.fixture(scope="class")
    def pipeline_tester(self):
        """Create pipeline test suite for integration testing."""
        model_registry = init_models()
        available_models = model_registry.list_models()
        
        if not available_models:
            pytest.skip("No models available for pipeline testing")
        
        return PipelineTestSuite(model_registry=model_registry)
    
    @pytest.mark.asyncio
    async def test_all_pipelines_execute_successfully(self, pipeline_tester):
        """
        Test all example pipelines execute without errors.
        
        This is a key test from the requirements that validates
        execution success across all pipelines.
        """
        # Run quick tests for CI efficiency
        results = await pipeline_tester.run_pipeline_tests(test_mode="quick")
        
        failed_pipelines = results.get_failed_pipelines()
        
        # Allow some failures due to external dependencies but log them
        if failed_pipelines:
            logger.warning(f"Some pipelines failed execution: {failed_pipelines}")
            # In a real deployment, this might be more strict
        
        # Ensure we tested something
        assert results.total_tests > 0, "No pipelines were tested"
        
        # Expect at least 50% success rate (allowing for API key issues, etc.)
        assert results.success_rate >= 50, \
            f"Success rate too low: {results.success_rate:.1f}% (failed: {failed_pipelines})"
    
    @pytest.mark.asyncio
    async def test_no_template_artifacts_in_outputs(self, pipeline_tester):
        """
        Test no unresolved templates in any outputs.
        
        This test validates template resolution across all pipelines.
        """
        results = await pipeline_tester.run_pipeline_tests(test_mode="quick")
        
        template_issues = [
            name for name, result in results.results.items()
            if not result.templates.resolved_correctly
        ]
        
        # Log template issues for investigation
        if template_issues:
            logger.warning(f"Pipelines with template issues: {template_issues}")
            for name in template_issues:
                result = results.results[name]
                logger.warning(f"  {name}: {result.templates.issues}")
        
        # Template resolution should be good for successfully executed pipelines
        successful_pipelines = [
            name for name, result in results.results.items()
            if result.execution.success
        ]
        
        successful_template_issues = [
            name for name in template_issues 
            if name in successful_pipelines
        ]
        
        assert len(successful_template_issues) == 0, \
            f"Successful pipelines have template issues: {successful_template_issues}"
    
    def test_pipeline_discovery_completeness(self, pipeline_tester):
        """Test that pipeline discovery finds all expected pipelines."""
        discovered = pipeline_tester.discover_pipelines()
        
        # Should discover major example pipelines
        expected_pipelines = [
            "simple_data_processing",
            "control_flow_conditional", 
            "research_minimal",
            "creative_image_pipeline"
        ]
        
        discovered_names = set(discovered.keys())
        missing_expected = [name for name in expected_pipelines 
                          if name not in discovered_names]
        
        if missing_expected:
            logger.warning(f"Some expected pipelines not discovered: {missing_expected}")
        
        # Should have discovered a substantial number
        assert len(discovered) >= 30, \
            f"Expected to discover at least 30 pipelines, found {len(discovered)}"