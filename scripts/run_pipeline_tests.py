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
from orchestrator.testing import PipelineTestSuite, TestResults
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
        choices=["quick", "core", "full"],
        default="quick",
        help="Test mode: quick (5-10 pipelines), core (15-20 pipelines), full (all pipelines)"
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
        
        # Initialize test suite with quality validation
        logger.info("Initializing pipeline test suite...")
        logger.info(f"Quality validation - LLM: {args.enable_llm_quality}, "
                   f"Templates: {args.enable_template_validation}, "
                   f"Threshold: {args.quality_threshold}")
        
        test_suite = PipelineTestSuite(
            examples_dir=args.examples_dir,
            model_registry=model_registry,
            enable_llm_quality_review=args.enable_llm_quality,
            enable_enhanced_template_validation=args.enable_template_validation,
            quality_threshold=args.quality_threshold
        )
        
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
        
        # Run tests
        logger.info("Starting pipeline tests...")
        
        if args.pipeline:
            # Test specific pipeline
            if args.pipeline not in discovered:
                logger.error(f"Pipeline '{args.pipeline}' not found")
                return 1
            
            logger.info(f"Testing single pipeline: {args.pipeline}")
            results = await test_suite.run_pipeline_tests([args.pipeline])
        else:
            # Test by mode
            logger.info(f"Running {args.mode} mode tests...")
            results = await test_suite.run_pipeline_tests(test_mode=args.mode)
        
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
        
        # Return appropriate exit code
        if results.success_rate >= 80:
            logger.info("Pipeline tests PASSED")
            return 0
        elif results.success_rate >= 50:
            logger.warning("Pipeline tests passed with warnings")
            return 0 if not args.fail_fast else 1
        else:
            logger.error("Pipeline tests FAILED")
            return 1
    
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