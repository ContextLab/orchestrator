#!/usr/bin/env python3
"""
Comprehensive pipeline testing with wrapper integrations - Issue #252.

This script runs all 25 example pipelines with various wrapper configurations
to validate RouteLLM, POML, and wrapper architecture integrations.
"""

import asyncio
import json
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add orchestrator to path
sys.path.insert(0, Path(__file__).parent.parent)

from tests.integration.test_pipeline_wrapper_validation import PipelineWrapperValidator
from tests.performance.test_wrapper_performance_regression import PerformanceRegressionTester
from tests.quality.test_output_quality_validation import OutputQualityValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline_wrapper_testing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ComprehensivePipelineTester:
    """
    Comprehensive tester that orchestrates all wrapper validation tests.
    
    Combines pipeline validation, performance regression testing,
    and quality validation into a single comprehensive test suite.
    """
    
    def __init__(self):
        self.results_dir = Path("tests/results/comprehensive")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Test components
        self.pipeline_validator = PipelineWrapperValidator()
        self.performance_tester = PerformanceRegressionTester()
        self.quality_validator = OutputQualityValidator()
        
        # Overall results
        self.comprehensive_results = {}
        self.start_time = None
        self.end_time = None
        
    async def initialize_all_testers(self):
        """Initialize all testing components."""
        logger.info("Initializing comprehensive pipeline tester...")
        
        try:
            await self.pipeline_validator.initialize()
            logger.info("✅ Pipeline validator initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize pipeline validator: {e}")
            raise
            
        try:
            await self.performance_tester.initialize()
            logger.info("✅ Performance tester initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize performance tester: {e}")
            raise
            
        try:
            await self.quality_validator.initialize()
            logger.info("✅ Quality validator initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize quality validator: {e}")
            raise
            
        logger.info("🎉 All testers initialized successfully")
        
    async def run_comprehensive_tests(
        self, 
        include_performance: bool = True,
        include_quality: bool = True,
        performance_iterations: int = 2
    ) -> Dict[str, Any]:
        """
        Run comprehensive wrapper validation tests.
        
        Args:
            include_performance: Whether to run performance regression tests
            include_quality: Whether to run quality validation tests
            performance_iterations: Number of performance test iterations
            
        Returns:
            Comprehensive test results
        """
        self.start_time = datetime.utcnow()
        logger.info("🚀 Starting comprehensive wrapper validation tests")
        
        results = {
            "test_start_time": self.start_time.isoformat(),
            "pipeline_validation": {},
            "performance_testing": {},
            "quality_validation": {},
            "summary": {}
        }
        
        # 1. Pipeline Wrapper Validation
        logger.info("\n" + "="*60)
        logger.info("PHASE 1: PIPELINE WRAPPER VALIDATION")
        logger.info("="*60)
        
        try:
            pipeline_results = await self.pipeline_validator.validate_all_pipelines()
            results["pipeline_validation"] = pipeline_results
            logger.info("✅ Pipeline wrapper validation completed")
        except Exception as e:
            logger.error(f"❌ Pipeline wrapper validation failed: {e}")
            results["pipeline_validation"] = {"error": str(e)}
            traceback.print_exc()
            
        # 2. Performance Regression Testing
        if include_performance:
            logger.info("\n" + "="*60)
            logger.info("PHASE 2: PERFORMANCE REGRESSION TESTING")
            logger.info("="*60)
            
            try:
                # Use performance test wrapper configurations
                from tests.performance.test_wrapper_performance_regression import PERFORMANCE_TEST_CONFIGS
                
                performance_results = await self.performance_tester.run_performance_benchmarks(
                    PERFORMANCE_TEST_CONFIGS, 
                    iterations=performance_iterations
                )
                
                performance_report = self.performance_tester.generate_performance_report()
                results["performance_testing"] = performance_report
                logger.info("✅ Performance regression testing completed")
            except Exception as e:
                logger.error(f"❌ Performance regression testing failed: {e}")
                results["performance_testing"] = {"error": str(e)}
                traceback.print_exc()
        else:
            logger.info("⏭️  Skipping performance regression testing")
            
        # 3. Quality Validation Testing
        if include_quality:
            logger.info("\n" + "="*60)
            logger.info("PHASE 3: OUTPUT QUALITY VALIDATION")
            logger.info("="*60)
            
            try:
                # Use quality test wrapper configurations
                from tests.quality.test_output_quality_validation import QUALITY_TEST_CONFIGS
                
                quality_report = await self.quality_validator.validate_all_pipeline_quality(
                    QUALITY_TEST_CONFIGS
                )
                
                results["quality_validation"] = quality_report
                logger.info("✅ Quality validation testing completed")
            except Exception as e:
                logger.error(f"❌ Quality validation testing failed: {e}")
                results["quality_validation"] = {"error": str(e)}
                traceback.print_exc()
        else:
            logger.info("⏭️  Skipping quality validation testing")
            
        # 4. Generate Comprehensive Summary
        self.end_time = datetime.utcnow()
        results["test_end_time"] = self.end_time.isoformat()
        results["total_duration_seconds"] = (self.end_time - self.start_time).total_seconds()
        
        summary = self._generate_comprehensive_summary(results)
        results["summary"] = summary
        
        # Save comprehensive results
        self.comprehensive_results = results
        await self._save_comprehensive_report(results)
        
        return results
        
    def _generate_comprehensive_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary of all test results."""
        summary = {
            "overall_status": "unknown",
            "critical_issues": [],
            "warnings": [],
            "successes": [],
            "recommendations": []
        }
        
        critical_issues = []
        warnings = []
        successes = []
        
        # Analyze pipeline validation results
        pipeline_results = results.get("pipeline_validation", {})
        if "error" in pipeline_results:
            critical_issues.append("Pipeline validation failed to complete")
        elif "summary" in pipeline_results:
            pipeline_summary = pipeline_results["summary"]
            success_rate = pipeline_summary.get("success_rate", 0)
            
            if success_rate >= 0.95:
                successes.append(f"Excellent pipeline success rate: {success_rate:.1%}")
            elif success_rate >= 0.80:
                warnings.append(f"Good pipeline success rate: {success_rate:.1%}")
            else:
                critical_issues.append(f"Poor pipeline success rate: {success_rate:.1%}")
                
        # Analyze performance results
        performance_results = results.get("performance_testing", {})
        if "error" in performance_results:
            warnings.append("Performance testing failed to complete")
        elif "summary" in performance_results:
            perf_summary = performance_results["summary"]
            regression_rate = perf_summary.get("regression_rate", 0)
            
            if regression_rate == 0:
                successes.append("No performance regressions detected")
            elif regression_rate <= 0.20:  # 20% threshold
                warnings.append(f"Minor performance regressions: {regression_rate:.1%}")
            else:
                critical_issues.append(f"Significant performance regressions: {regression_rate:.1%}")
                
        # Analyze quality results  
        quality_results = results.get("quality_validation", {})
        if "error" in quality_results:
            warnings.append("Quality validation failed to complete")
        elif "summary" in quality_results:
            quality_summary = quality_results["summary"]
            improvements = quality_summary.get("quality_improvements", 0)
            degradations = quality_summary.get("quality_degradations", 0)
            
            if degradations == 0:
                successes.append("No quality degradations detected")
            elif improvements >= degradations:
                warnings.append(f"Quality changes: {improvements} improvements vs {degradations} degradations")
            else:
                critical_issues.append(f"Quality degradation: {degradations} issues vs {improvements} improvements")
                
        # Determine overall status
        if critical_issues:
            overall_status = "CRITICAL_ISSUES"
        elif warnings:
            overall_status = "WARNINGS" 
        else:
            overall_status = "SUCCESS"
            
        # Generate recommendations
        recommendations = []
        if critical_issues:
            recommendations.append("Address critical issues before proceeding with wrapper deployment")
        if warnings:
            recommendations.append("Review warnings and consider improvements")
        if not successes:
            recommendations.append("No successes detected - investigate test configuration")
        else:
            recommendations.append("Continue with wrapper integration deployment")
            
        summary.update({
            "overall_status": overall_status,
            "critical_issues": critical_issues,
            "warnings": warnings,
            "successes": successes,
            "recommendations": recommendations
        })
        
        return summary
        
    async def _save_comprehensive_report(self, results: Dict[str, Any]):
        """Save comprehensive test report to file."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_file = self.results_dir / f"comprehensive_wrapper_validation_{timestamp}.json"
        
        with open(report_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
            
        logger.info(f"📄 Comprehensive report saved to: {report_file}")
        
        # Also save a summary report
        summary_file = self.results_dir / f"validation_summary_{timestamp}.json"
        summary_data = {
            "timestamp": results.get("test_start_time"),
            "duration_seconds": results.get("total_duration_seconds"),
            "overall_status": results["summary"]["overall_status"],
            "critical_issues": results["summary"]["critical_issues"],
            "warnings": results["summary"]["warnings"],
            "successes": results["summary"]["successes"],
            "recommendations": results["summary"]["recommendations"]
        }
        
        with open(summary_file, "w") as f:
            json.dump(summary_data, f, indent=2, default=str)
            
        logger.info(f"📄 Summary report saved to: {summary_file}")
        
    def print_final_report(self):
        """Print final comprehensive test report."""
        if not self.comprehensive_results:
            logger.error("No comprehensive results to report")
            return
            
        results = self.comprehensive_results
        summary = results.get("summary", {})
        
        print("\n" + "="*80)
        print("COMPREHENSIVE WRAPPER VALIDATION REPORT - Issue #252")
        print("="*80)
        
        # Overall status
        status = summary.get("overall_status", "UNKNOWN")
        status_icon = "🎉" if status == "SUCCESS" else "⚠️" if status == "WARNINGS" else "❌"
        print(f"\n{status_icon} Overall Status: {status}")
        
        # Duration
        duration = results.get("total_duration_seconds", 0)
        print(f"⏱️  Total Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        
        # Pipeline validation summary
        pipeline_results = results.get("pipeline_validation", {})
        if "summary" in pipeline_results:
            pipeline_summary = pipeline_results["summary"]
            print(f"\n📋 Pipeline Validation:")
            print(f"   Success Rate: {pipeline_summary.get('success_rate', 0):.1%}")
            print(f"   Pipelines Tested: {pipeline_summary.get('pipelines_tested', 0)}")
            print(f"   Wrapper Configs Tested: {pipeline_summary.get('wrapper_configs_tested', 0)}")
            
        # Performance testing summary
        performance_results = results.get("performance_testing", {})
        if "summary" in performance_results:
            perf_summary = performance_results["summary"]
            print(f"\n⚡ Performance Testing:")
            print(f"   Regression Rate: {perf_summary.get('regression_rate', 0):.1%}")
            print(f"   Total Tests: {perf_summary.get('total_tests', 0)}")
            
            if "performance_statistics" in performance_results:
                perf_stats = performance_results["performance_statistics"]
                avg_delta = perf_stats.get("average_performance_delta_percent", 0)
                print(f"   Average Performance Impact: {avg_delta:+.1f}%")
                
        # Quality validation summary
        quality_results = results.get("quality_validation", {})
        if "summary" in quality_results:
            quality_summary = quality_results["summary"]
            print(f"\n🎯 Quality Validation:")
            print(f"   Quality Improvements: {quality_summary.get('quality_improvements', 0)}")
            print(f"   Quality Degradations: {quality_summary.get('quality_degradations', 0)}")
            
            avg_delta = quality_summary.get("average_quality_delta", 0)
            if avg_delta != 0:
                print(f"   Average Quality Impact: {avg_delta:+.1f}%")
                
        # Issues and successes
        critical_issues = summary.get("critical_issues", [])
        if critical_issues:
            print(f"\n❌ Critical Issues ({len(critical_issues)}):")
            for issue in critical_issues:
                print(f"   - {issue}")
                
        warnings = summary.get("warnings", [])
        if warnings:
            print(f"\n⚠️  Warnings ({len(warnings)}):")
            for warning in warnings:
                print(f"   - {warning}")
                
        successes = summary.get("successes", [])
        if successes:
            print(f"\n✅ Successes ({len(successes)}):")
            for success in successes:
                print(f"   - {success}")
                
        # Recommendations
        recommendations = summary.get("recommendations", [])
        if recommendations:
            print(f"\n💡 Recommendations:")
            for rec in recommendations:
                print(f"   - {rec}")
                
        # Issue #252 specific conclusion
        print("\n" + "="*80)
        print("ISSUE #252 CONCLUSION")
        print("="*80)
        
        if status == "SUCCESS":
            print("🎉 All wrapper integrations validated successfully!")
            print("✅ Ready for production deployment")
        elif status == "WARNINGS":
            print("⚠️  Wrapper integrations mostly successful with minor issues")
            print("🔍 Review warnings before production deployment")
        else:
            print("❌ Critical issues detected in wrapper integrations")
            print("🛠️  Address critical issues before proceeding")
            
        print(f"\n📄 Detailed reports available in: {self.results_dir}")


async def main():
    """Main function for comprehensive testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive wrapper validation testing")
    parser.add_argument("--skip-performance", action="store_true", help="Skip performance testing")
    parser.add_argument("--skip-quality", action="store_true", help="Skip quality validation")
    parser.add_argument("--performance-iterations", type=int, default=2, help="Performance test iterations")
    parser.add_argument("--quick", action="store_true", help="Quick test mode (fewer iterations)")
    
    args = parser.parse_args()
    
    # Adjust for quick mode
    if args.quick:
        args.performance_iterations = 1
        logger.info("🏃 Running in quick test mode")
        
    # Create comprehensive tester
    tester = ComprehensivePipelineTester()
    
    try:
        # Initialize all testers
        await tester.initialize_all_testers()
        
        # Run comprehensive tests
        results = await tester.run_comprehensive_tests(
            include_performance=not args.skip_performance,
            include_quality=not args.skip_quality,
            performance_iterations=args.performance_iterations
        )
        
        # Print final report
        tester.print_final_report()
        
        # Exit with appropriate code
        summary = results.get("summary", {})
        status = summary.get("overall_status", "UNKNOWN")
        
        if status == "SUCCESS":
            sys.exit(0)  # Success
        elif status == "WARNINGS":
            sys.exit(1)  # Warnings
        else:
            sys.exit(2)  # Critical issues
            
    except KeyboardInterrupt:
        logger.info("❌ Testing interrupted by user")
        sys.exit(3)
    except Exception as e:
        logger.error(f"❌ Comprehensive testing failed: {e}")
        traceback.print_exc()
        sys.exit(4)


if __name__ == "__main__":
    asyncio.run(main())