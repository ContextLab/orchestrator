#!/usr/bin/env python3
"""
CLI script for running pipeline tests.

Usage:
    python scripts/run_pipeline_tests.py --mode quick
    python scripts/run_pipeline_tests.py --mode core --report-dir results/
    python scripts/run_pipeline_tests.py --mode full --pipeline simple_data_processing
    python scripts/run_pipeline_tests.py --discover-only
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from orchestrator import init_models
from orchestrator.testing import (
    PipelineTestSuite, TestResults, TestModeManager, TestMode,
    CIIntegrationManager, create_ci_config_from_environment,
    ReleaseValidator, determine_release_type_from_version, ReleaseType
)
from orchestrator.testing.test_reporter import PipelineTestReporter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run pipeline tests for orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run quick tests (5-10 pipelines)
    python scripts/run_pipeline_tests.py --mode quick
    
    # Run core tests (15-20 pipelines) 
    python scripts/run_pipeline_tests.py --mode core
    
    # Run all tests
    python scripts/run_pipeline_tests.py --mode full
    
    # Test specific pipeline
    python scripts/run_pipeline_tests.py --pipeline simple_data_processing
    
    # Just discover pipelines
    python scripts/run_pipeline_tests.py --discover-only
    
    # Generate reports in custom directory
    python scripts/run_pipeline_tests.py --mode quick --report-dir custom_results/
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["smoke", "quick", "core", "full", "regression"],
        default="quick",
        help="Test mode: smoke (3 min), quick (8 min), core (25 min), full (90 min), regression (30 min)"
    )
    
    parser.add_argument(
        "--pipeline",
        type=str,
        help="Run tests for specific pipeline only"
    )
    
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="Only discover pipelines, don't run tests"
    )
    
    parser.add_argument(
        "--examples-dir",
        type=Path,
        default=Path("examples"),
        help="Directory containing example pipelines"
    )
    
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("pipeline_test_results"),
        help="Directory for test reports"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout per pipeline in seconds"
    )
    
    parser.add_argument(
        "--max-cost",
        type=float,
        default=1.0,
        help="Maximum cost per pipeline in dollars"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first failure"
    )
    
    # Quality validation options (Stream B)
    parser.add_argument(
        "--enable-llm-quality",
        action="store_true",
        default=False,
        help="Enable LLM-powered quality assessment (requires API keys)"
    )
    
    parser.add_argument(
        "--enable-template-validation",
        action="store_true", 
        default=True,
        help="Enable enhanced template validation"
    )
    
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=85.0,
        help="Minimum quality score for production readiness (0-100)"
    )
    
    parser.add_argument(
        "--production-ready-only",
        action="store_true",
        help="Only report pipelines that meet production readiness criteria"
    )
    
    # Stream D: CI/CD Integration & Test Modes
    parser.add_argument(
        "--time-budget",
        type=int,
        help="Time budget in minutes for test execution (enables smart mode selection)"
    )
    
    parser.add_argument(
        "--ci-mode",
        action="store_true",
        help="Enable CI/CD mode with artifact generation and status reporting"
    )
    
    parser.add_argument(
        "--release-validation",
        type=str,
        help="Perform release validation for specified version (e.g., '1.2.3')"
    )
    
    parser.add_argument(
        "--release-type",
        choices=["major", "minor", "patch", "hotfix", "prerelease"],
        help="Override release type for validation (auto-detected from version if not specified)"
    )
    
    parser.add_argument(
        "--ci-artifacts-dir",
        type=Path,
        default=Path("ci_artifacts"),
        help="Directory for CI/CD artifacts (reports, status files, etc.)"
    )
    
    parser.add_argument(
        "--smart-selection",
        action="store_true",
        help="Enable smart pipeline selection based on historical performance"
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("orchestrator").setLevel(logging.DEBUG)
    
    logger.info("Starting pipeline test runner")
    logger.info(f"Mode: {args.mode if not args.pipeline else 'single'}")
    logger.info(f"Examples directory: {args.examples_dir}")
    logger.info(f"Report directory: {args.report_dir}")
    
    try:
        # Initialize model registry
        logger.info("Initializing model registry...")
        model_registry = init_models()
        
        available_models = model_registry.list_models()
        logger.info(f"Available models: {len(available_models)}")
        
        if not available_models:
            logger.error("No models available. Check API keys and configuration.")
            return 1
        
        # Initialize test suite with quality validation and Stream D features
        logger.info("Initializing pipeline test suite...")
        logger.info(f"Quality validation - LLM: {args.enable_llm_quality}, "
                   f"Templates: {args.enable_template_validation}, "
                   f"Threshold: {args.quality_threshold}")
        
        # Stream D: Enable performance monitoring and regression detection
        enable_performance = args.ci_mode or args.release_validation or args.smart_selection
        enable_regression = args.ci_mode or args.release_validation
        
        test_suite = PipelineTestSuite(
            examples_dir=args.examples_dir,
            model_registry=model_registry,
            enable_llm_quality_review=args.enable_llm_quality,
            enable_enhanced_template_validation=args.enable_template_validation,
            quality_threshold=args.quality_threshold,
            enable_performance_monitoring=enable_performance,
            enable_regression_detection=enable_regression
        )
        
        # Stream D: Initialize test mode manager for smart selection
        mode_manager = None
        if args.smart_selection or args.time_budget:
            mode_manager = TestModeManager(test_suite.performance_tracker)
            logger.info("Initialized test mode manager for smart pipeline selection")
        
        # Stream D: Initialize CI/CD integration if requested
        ci_manager = None
        if args.ci_mode:
            ci_config = create_ci_config_from_environment()
            ci_manager = CIIntegrationManager(ci_config)
            logger.info(f"Initialized CI/CD integration for {ci_config.system.value}")
        
        # Stream D: Initialize release validator if requested
        release_validator = None
        if args.release_validation:
            release_validator = ReleaseValidator(
                performance_tracker=test_suite.performance_tracker,
                quality_reviewer=test_suite.quality_validator.quality_reviewer if hasattr(test_suite, 'quality_validator') else None
            )
            logger.info(f"Initialized release validator for version {args.release_validation}")
        
        # Configure test suite
        test_suite.timeout_seconds = args.timeout
        test_suite.max_cost_per_pipeline = args.max_cost
        
        # Discover pipelines
        logger.info("Discovering pipelines...")
        discovered = test_suite.discover_pipelines()
        
        logger.info(f"Discovered {len(discovered)} pipelines:")
        for name, info in list(discovered.items())[:10]:  # Show first 10
            logger.info(f"  - {name} ({info.category}, {info.complexity})")
        if len(discovered) > 10:
            logger.info(f"  ... and {len(discovered) - 10} more")
        
        # Just discovery mode
        if args.discover_only:
            logger.info("Discovery complete. Exiting.")
            
            # Print categorized summary
            categories = {}
            for info in discovered.values():
                if info.category not in categories:
                    categories[info.category] = []
                categories[info.category].append(info.name)
            
            print("\n=== Pipeline Discovery Summary ===")
            for category, pipelines in categories.items():
                print(f"\n{category.title()} ({len(pipelines)}):")
                for pipeline in sorted(pipelines):
                    print(f"  - {pipeline}")
            
            test_safe_count = len(test_suite.discovery.get_test_safe_pipelines())
            core_count = len(test_suite.discovery.get_core_test_pipelines()) 
            quick_count = len(test_suite.discovery.get_quick_test_pipelines())
            
            print(f"\nTest Categories:")
            print(f"  - Test-safe pipelines: {test_safe_count}")
            print(f"  - Core test pipelines: {core_count}")
            print(f"  - Quick test pipelines: {quick_count}")
            
            return 0
        
        # Stream D: Smart mode selection based on time budget
        final_mode = args.mode
        selected_pipelines = None
        
        if args.time_budget and mode_manager:
            # Use time budget to select optimal mode
            recommended_mode = mode_manager.get_recommended_mode_for_time_budget(args.time_budget)
            logger.info(f"Time budget: {args.time_budget} minutes")
            logger.info(f"Recommended mode: {recommended_mode.value} (was: {args.mode})")
            final_mode = recommended_mode.value
            
            # Get optimal pipeline selection
            available_pipelines = list(discovered.keys())
            composition = mode_manager.select_optimal_pipeline_suite(
                recommended_mode, available_pipelines, args.time_budget
            )
            selected_pipelines = composition.selected_pipelines
            
            logger.info(f"Smart selection: {len(selected_pipelines)} pipelines")
            logger.info(f"Estimated time: {composition.estimated_total_time_minutes:.1f} minutes")
            logger.info(f"Coverage: {composition.coverage_percentage:.1f}%")
        
        # Run tests
        logger.info("Starting pipeline tests...")
        
        if args.pipeline:
            # Test specific pipeline
            if args.pipeline not in discovered:
                logger.error(f"Pipeline '{args.pipeline}' not found")
                return 1
            
            logger.info(f"Testing single pipeline: {args.pipeline}")
            results = await test_suite.run_pipeline_tests([args.pipeline])
        elif selected_pipelines:
            # Use smart selection
            logger.info(f"Running smart selection: {len(selected_pipelines)} pipelines")
            results = await test_suite.run_pipeline_tests(selected_pipelines)
        else:
            # Test by mode
            logger.info(f"Running {final_mode} mode tests...")
            results = await test_suite.run_pipeline_tests(test_mode=final_mode)
        
        # Log results summary
        logger.info("=" * 60)
        logger.info("TEST RESULTS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total tests: {results.total_tests}")
        logger.info(f"Passed: {results.successful_tests}")
        logger.info(f"Failed: {results.failed_tests}")
        logger.info(f"Success rate: {results.success_rate:.1f}%")
        logger.info(f"Total time: {results.total_time:.1f}s")
        logger.info(f"Total cost: ${results.total_cost:.4f}")
        logger.info(f"Average quality score: {results.average_quality_score:.1f}")
        
        # Quality validation reporting (Stream B)
        production_ready_pipelines = results.get_production_ready_pipelines()
        quality_issues = results.get_quality_issues_summary()
        
        logger.info(f"Production ready: {len(production_ready_pipelines)}/{results.total_tests}")
        if quality_issues['critical_issues'] > 0:
            logger.warning(f"Critical quality issues: {quality_issues['critical_issues']}")
        if quality_issues['major_issues'] > 0:
            logger.info(f"Major quality issues: {quality_issues['major_issues']}")
        if quality_issues['template_artifacts'] > 0:
            logger.warning(f"Pipelines with template artifacts: {quality_issues['template_artifacts']}")
        
        # Production readiness filter
        if args.production_ready_only:
            if not production_ready_pipelines:
                logger.warning("No pipelines meet production readiness criteria")
            else:
                logger.info("Production ready pipelines:")
                for pipeline_name in production_ready_pipelines:
                    result = results.results[pipeline_name]
                    score = result.quality_score
                    logger.info(f"  - {pipeline_name}: Score {score:.1f}")
        
        # Show failed pipelines
        failed_pipelines = results.get_failed_pipelines()
        if failed_pipelines:
            logger.warning(f"Failed pipelines ({len(failed_pipelines)}):")
            for pipeline_name in failed_pipelines:
                result = results.results[pipeline_name]
                error_msg = result.execution.error_message or "Unknown error"
                logger.warning(f"  - {pipeline_name}: {error_msg}")
        
        # Generate reports
        logger.info("Generating test reports...")
        reporter = PipelineTestReporter(output_dir=args.report_dir)
        
        test_mode = args.mode if not args.pipeline else "single"
        report_files = reporter.generate_comprehensive_report(results, test_mode)
        
        logger.info("Generated reports:")
        for report_type, file_path in report_files.items():
            logger.info(f"  - {report_type}: {file_path}")
        
        # Stream D: CI/CD integration and release validation
        exit_code = 0
        
        if ci_manager:
            logger.info("Processing CI/CD integration...")
            ci_results, ci_summary = ci_manager.convert_test_results_to_ci_format(results)
            
            # Generate CI artifacts
            ci_artifacts = ci_manager.generate_ci_artifacts(
                ci_results, ci_summary, 
                additional_data={
                    "mode": final_mode,
                    "time_budget": args.time_budget,
                    "smart_selection": bool(selected_pipelines)
                }
            )
            
            logger.info(f"Generated {len(ci_artifacts)} CI artifacts")
            for artifact in ci_artifacts:
                logger.info(f"  - {artifact}")
            
            # Set CI exit code
            exit_code = ci_manager.determine_exit_code(ci_summary)
            
            logger.info(f"CI/CD Status: {'PASSED' if ci_summary.quality_gate_passed else 'FAILED'}")
            logger.info(f"Release Ready: {'YES' if ci_summary.release_ready else 'NO'}")
        
        # Stream D: Release validation
        if args.release_validation and release_validator:
            logger.info(f"Performing release validation for version {args.release_validation}...")
            
            # Determine release type
            if args.release_type:
                release_type = ReleaseType(args.release_type)
            else:
                release_type = determine_release_type_from_version(args.release_validation)
            
            logger.info(f"Release type: {release_type.value}")
            
            # Validate release readiness
            validation_result = release_validator.validate_release_readiness(
                results, release_type
            )
            
            logger.info("=" * 60)
            logger.info("RELEASE VALIDATION RESULTS")
            logger.info("=" * 60)
            logger.info(f"Validation Level: {validation_result.validation_level.value}")
            logger.info(f"Overall Score: {validation_result.overall_score:.1f}/100")
            logger.info(f"Validation Passed: {'✅ YES' if validation_result.validation_passed else '❌ NO'}")
            logger.info(f"Release Ready: {'✅ YES' if validation_result.release_ready else '❌ NO'}")
            
            if validation_result.blocking_issues:
                logger.error("Blocking Issues:")
                for issue in validation_result.blocking_issues:
                    logger.error(f"  - {issue}")
            
            if validation_result.warning_issues:
                logger.warning("Warning Issues:")
                for issue in validation_result.warning_issues:
                    logger.warning(f"  - {issue}")
            
            if validation_result.recommendations:
                logger.info("Recommendations:")
                for rec in validation_result.recommendations:
                    logger.info(f"  - {rec}")
            
            # Override exit code for release validation
            if not validation_result.validation_passed:
                exit_code = 1
            elif not validation_result.release_ready:
                exit_code = 2 if not args.fail_fast else 1
        
        # Default exit code logic if no CI/CD or release validation
        if not ci_manager and not args.release_validation:
            if results.success_rate >= 80:
                logger.info("Pipeline tests PASSED")
                exit_code = 0
            elif results.success_rate >= 50:
                logger.warning("Pipeline tests passed with warnings")
                exit_code = 0 if not args.fail_fast else 1
            else:
                logger.error("Pipeline tests FAILED")
                exit_code = 1
        
        return exit_code
    
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Test run failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)