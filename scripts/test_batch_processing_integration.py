#!/usr/bin/env python3
"""
Comprehensive Test Suite for Batch Processing Integration

Tests all components of the Stream D batch processing system:
- Batch reviewer functionality
- Report generation
- Integration with validation tools
- Production automation capabilities
- Performance optimization

Usage:
    python scripts/test_batch_processing_integration.py
    python scripts/test_batch_processing_integration.py --component batch_reviewer
    python scripts/test_batch_processing_integration.py --component report_generator
    python scripts/test_batch_processing_integration.py --performance-test
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator.core.credential_manager import create_credential_manager
from orchestrator.core.quality_assessment import PipelineQualityReview, QualityIssue, IssueSeverity, IssueCategory
from orchestrator.quality.report_generator import QualityReportGenerator
from quality_review.batch_reviewer import ComprehensiveBatchReviewer, BatchReviewConfig
from quality_review.integrated_validation import IntegratedValidationSystem
from quality_review.production_automation import ProductionAutomationSystem


class TestResult:
    """Represents a test result."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.success = False
        self.duration = 0
        self.start_time = None
        self.end_time = None
        self.error = None
        self.warnings = []
        self.data = {}
    
    def start(self):
        """Mark test as started."""
        self.start_time = time.time()
    
    def complete(self, success: bool = True, error: str = None, data: Dict[str, Any] = None):
        """Mark test as completed."""
        self.end_time = time.time()
        if self.start_time:
            self.duration = self.end_time - self.start_time
        self.success = success
        self.error = error
        self.data = data or {}
    
    def add_warning(self, warning: str):
        """Add warning to test result."""
        self.warnings.append(warning)


class BatchProcessingTestSuite:
    """Comprehensive test suite for batch processing integration."""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.test_results = []
        
        # Test configuration
        self.test_output_dir = Path("test_batch_processing_output")
        self.test_output_dir.mkdir(exist_ok=True)
        
        # Initialize components with test configuration
        self.batch_config = BatchReviewConfig(
            max_concurrent_reviews=2,
            timeout_per_pipeline=30,  # Shorter timeout for tests
            enable_caching=True,
            output_directory=str(self.test_output_dir / "quality_reports")
        )
        
        self.logger.info("Initialized Batch Processing Test Suite")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup test logging."""
        logger = logging.getLogger("BatchProcessingTest")
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def create_test(self, name: str, description: str = "") -> TestResult:
        """Create and register a test."""
        test = TestResult(name, description)
        self.test_results.append(test)
        return test
    
    async def test_batch_reviewer_initialization(self) -> TestResult:
        """Test batch reviewer initialization."""
        test = self.create_test(
            "batch_reviewer_initialization",
            "Test initialization of ComprehensiveBatchReviewer"
        )
        test.start()
        
        try:
            # Test basic initialization
            reviewer = ComprehensiveBatchReviewer(self.batch_config)
            
            # Verify components
            assert reviewer.config is not None, "Configuration not set"
            assert reviewer.logger is not None, "Logger not initialized"
            assert reviewer.available_pipelines is not None, "Pipeline discovery failed"
            
            # Test pipeline discovery
            discovered_count = len(reviewer.available_pipelines)
            if discovered_count == 0:
                test.add_warning("No pipelines discovered - check examples/outputs directory")
            
            test.complete(
                success=True,
                data={
                    "discovered_pipelines": discovered_count,
                    "available_pipelines": reviewer.available_pipelines[:10]  # First 10 for testing
                }
            )
            
            self.logger.info(f"✅ Batch reviewer initialized with {discovered_count} pipelines")
            
        except Exception as e:
            test.complete(success=False, error=str(e))
            self.logger.error(f"❌ Batch reviewer initialization failed: {e}")
        
        return test
    
    async def test_single_pipeline_review(self) -> TestResult:
        """Test reviewing a single pipeline."""
        test = self.create_test(
            "single_pipeline_review",
            "Test quality review of a single pipeline"
        )
        test.start()
        
        try:
            reviewer = ComprehensiveBatchReviewer(self.batch_config)
            
            # Find a test pipeline
            if not reviewer.available_pipelines:
                test.complete(success=False, error="No pipelines available for testing")
                return test
            
            test_pipeline = reviewer.available_pipelines[0]
            self.logger.info(f"Testing single pipeline review: {test_pipeline}")
            
            # Review the pipeline
            pipeline_name, review, error = await reviewer.review_pipeline_with_timeout(
                test_pipeline,
                reviewer.progress_tracker if hasattr(reviewer, 'progress_tracker') else None
            )
            
            if review is not None:
                test.complete(
                    success=True,
                    data={
                        "pipeline_name": pipeline_name,
                        "overall_score": review.overall_score,
                        "production_ready": review.production_ready,
                        "total_issues": review.total_issues,
                        "files_reviewed": len(review.files_reviewed)
                    }
                )
                
                self.logger.info(f"✅ Pipeline {test_pipeline} reviewed: {review.overall_score}/100")
            else:
                test.complete(success=False, error=str(error) if error else "Unknown error")
                self.logger.error(f"❌ Pipeline {test_pipeline} review failed: {error}")
        
        except Exception as e:
            test.complete(success=False, error=str(e))
            self.logger.error(f"❌ Single pipeline review test failed: {e}")
        
        return test
    
    async def test_batch_review_functionality(self) -> TestResult:
        """Test batch review of multiple pipelines."""
        test = self.create_test(
            "batch_review_functionality", 
            "Test batch review of multiple pipelines"
        )
        test.start()
        
        try:
            reviewer = ComprehensiveBatchReviewer(self.batch_config)
            
            # Select test pipelines (limit to 3 for faster testing)
            test_pipelines = reviewer.available_pipelines[:3]
            if len(test_pipelines) == 0:
                test.complete(success=False, error="No pipelines available for batch testing")
                return test
            
            self.logger.info(f"Testing batch review of {len(test_pipelines)} pipelines")
            
            # Run batch review
            batch_report = await reviewer.batch_review_pipelines(
                test_pipelines,
                show_progress=False
            )
            
            # Validate batch report structure
            assert "batch_review_summary" in batch_report, "Missing batch review summary"
            assert "quality_metrics" in batch_report, "Missing quality metrics"
            assert "detailed_reviews" in batch_report, "Missing detailed reviews"
            
            summary = batch_report["batch_review_summary"]
            quality_metrics = batch_report["quality_metrics"]
            
            test.complete(
                success=True,
                data={
                    "pipelines_tested": len(test_pipelines),
                    "successful_reviews": summary["successful_reviews"],
                    "failed_reviews": summary["failed_reviews"],
                    "success_rate": summary["success_rate"],
                    "average_quality_score": quality_metrics["average_score"],
                    "production_ready_count": quality_metrics["production_ready_count"]
                }
            )
            
            self.logger.info(f"✅ Batch review completed: {summary['success_rate']:.1f}% success rate")
            
        except Exception as e:
            test.complete(success=False, error=str(e))
            self.logger.error(f"❌ Batch review test failed: {e}")
        
        return test
    
    async def test_report_generation(self) -> TestResult:
        """Test report generation functionality."""
        test = self.create_test(
            "report_generation",
            "Test comprehensive report generation"
        )
        test.start()
        
        try:
            # Create test report generator
            report_generator = QualityReportGenerator(
                output_directory=self.test_output_dir / "reports"
            )
            
            # Create mock review data for testing
            mock_review = self._create_mock_review("test_pipeline")
            
            # Test individual report generation
            individual_reports = report_generator.generate_individual_report(
                mock_review,
                formats=["json", "markdown", "html"]
            )
            
            # Verify files were created
            for format_type, file_path in individual_reports.items():
                assert file_path.exists(), f"Report file not created: {format_type}"
                assert file_path.stat().st_size > 0, f"Report file is empty: {format_type}"
            
            # Test batch report generation  
            mock_reviews = {
                "test_pipeline_1": self._create_mock_review("test_pipeline_1"),
                "test_pipeline_2": self._create_mock_review("test_pipeline_2")
            }
            
            batch_reports = report_generator.generate_batch_report(
                mock_reviews,
                failed_reviews={"failed_pipeline": "Test error"},
                metadata={"test_run": True}
            )
            
            # Verify batch reports
            for format_type, file_path in batch_reports.items():
                assert file_path.exists(), f"Batch report file not created: {format_type}"
                assert file_path.stat().st_size > 0, f"Batch report file is empty: {format_type}"
            
            # Test dashboard generation
            dashboard_path = report_generator.generate_dashboard()
            assert dashboard_path.exists(), "Dashboard file not created"
            assert dashboard_path.stat().st_size > 0, "Dashboard file is empty"
            
            test.complete(
                success=True,
                data={
                    "individual_reports": len(individual_reports),
                    "batch_reports": len(batch_reports),
                    "dashboard_generated": dashboard_path.exists(),
                    "report_files": [str(p) for p in individual_reports.values()]
                }
            )
            
            self.logger.info("✅ Report generation test completed")
            
        except Exception as e:
            test.complete(success=False, error=str(e))
            self.logger.error(f"❌ Report generation test failed: {e}")
        
        return test
    
    async def test_integrated_validation(self) -> TestResult:
        """Test integrated validation system."""
        test = self.create_test(
            "integrated_validation",
            "Test integration with validation tools"
        )
        test.start()
        
        try:
            # Create validation system
            validation_system = IntegratedValidationSystem()
            
            # Test health check functionality
            health_status = await validation_system.health_check()
            assert "overall_health" in health_status, "Missing overall health status"
            assert "components" in health_status, "Missing component health status"
            
            # Verify health check structure
            assert health_status["overall_health"] in ["healthy", "degraded", "critical"], "Invalid health status"
            
            test.complete(
                success=True,
                data={
                    "health_status": health_status["overall_health"],
                    "components_checked": len(health_status.get("components", {})),
                    "alerts": len(health_status.get("alerts", []))
                }
            )
            
            self.logger.info(f"✅ Integrated validation test completed: {health_status['overall_health']}")
            
        except Exception as e:
            test.complete(success=False, error=str(e))
            self.logger.error(f"❌ Integrated validation test failed: {e}")
        
        return test
    
    async def test_production_automation(self) -> TestResult:
        """Test production automation system."""
        test = self.create_test(
            "production_automation",
            "Test production automation capabilities"
        )
        test.start()
        
        try:
            # Create temporary config for testing
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as config_file:
                test_config = {
                    "check_interval_minutes": 1,
                    "max_concurrent_reviews": 1,
                    "timeout_per_pipeline": 30,
                    "enable_caching": True,
                    "output_directory": str(self.test_output_dir / "production_reports"),
                    "max_consecutive_failures": 2,
                    "quality_thresholds": {
                        "average_score_warning": 70,
                        "average_score_critical": 50
                    },
                    "notifications": {
                        "email": {"enabled": False},
                        "webhook": {"enabled": False}
                    }
                }
                json.dump(test_config, config_file)
                config_path = Path(config_file.name)
            
            try:
                # Initialize production system
                production_system = ProductionAutomationSystem(config_file=config_path)
                
                # Test health check
                health_check = await production_system.health_check()
                assert "overall_health" in health_check, "Missing health check data"
                
                # Test performance report generation
                performance_report = await production_system.generate_performance_report(hours=1)
                assert "trends" in performance_report, "Missing performance trends"
                assert "recommendations" in performance_report, "Missing recommendations"
                
                test.complete(
                    success=True,
                    data={
                        "health_status": health_check["overall_health"],
                        "performance_metrics": len(performance_report.get("trends", {})),
                        "recommendations": len(performance_report.get("recommendations", []))
                    }
                )
                
                self.logger.info("✅ Production automation test completed")
                
            finally:
                # Cleanup config file
                if config_path.exists():
                    config_path.unlink()
            
        except Exception as e:
            test.complete(success=False, error=str(e))
            self.logger.error(f"❌ Production automation test failed: {e}")
        
        return test
    
    async def test_performance_optimization(self) -> TestResult:
        """Test performance optimization features."""
        test = self.create_test(
            "performance_optimization",
            "Test performance optimization and caching"
        )
        test.start()
        
        try:
            # Test with caching enabled
            cached_config = BatchReviewConfig(
                max_concurrent_reviews=3,
                timeout_per_pipeline=30,
                enable_caching=True,
                output_directory=str(self.test_output_dir / "cached_reports")
            )
            
            reviewer = ComprehensiveBatchReviewer(cached_config)
            test_pipelines = reviewer.available_pipelines[:2]  # Test with 2 pipelines
            
            if len(test_pipelines) == 0:
                test.complete(success=False, error="No pipelines available for performance testing")
                return test
            
            # First run (no cache)
            start_time = time.time()
            first_run = await reviewer.batch_review_pipelines(test_pipelines, show_progress=False)
            first_duration = time.time() - start_time
            
            # Second run (with cache)
            start_time = time.time()
            second_run = await reviewer.batch_review_pipelines(test_pipelines, show_progress=False)
            second_duration = time.time() - start_time
            
            # Cache should improve performance
            performance_improvement = (first_duration - second_duration) / first_duration * 100
            
            test.complete(
                success=True,
                data={
                    "first_run_duration": first_duration,
                    "second_run_duration": second_duration,
                    "performance_improvement_percent": performance_improvement,
                    "caching_effective": second_duration < first_duration,
                    "concurrent_reviews": cached_config.max_concurrent_reviews
                }
            )
            
            self.logger.info(f"✅ Performance test: {performance_improvement:.1f}% improvement with caching")
            
        except Exception as e:
            test.complete(success=False, error=str(e))
            self.logger.error(f"❌ Performance optimization test failed: {e}")
        
        return test
    
    def _create_mock_review(self, pipeline_name: str) -> PipelineQualityReview:
        """Create mock quality review for testing."""
        mock_issues = [
            QualityIssue(
                category=IssueCategory.CONTENT_QUALITY,
                severity=IssueSeverity.MINOR,
                description=f"Mock issue in {pipeline_name}",
                file_path=f"examples/outputs/{pipeline_name}/test.md",
                suggestion="Fix mock issue"
            )
        ]
        
        return PipelineQualityReview(
            pipeline_name=pipeline_name,
            overall_score=85,
            files_reviewed=[f"examples/outputs/{pipeline_name}/test.md"],
            critical_issues=[],
            major_issues=[],
            minor_issues=mock_issues,
            recommendations=["Test recommendation"],
            production_ready=True,
            reviewer_model="test_model",
            review_duration_seconds=1.5
        )
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all batch processing tests."""
        self.logger.info("🚀 Starting Comprehensive Batch Processing Tests")
        start_time = time.time()
        
        # Run all tests
        tests_to_run = [
            self.test_batch_reviewer_initialization(),
            self.test_single_pipeline_review(),
            self.test_batch_review_functionality(),
            self.test_report_generation(),
            self.test_integrated_validation(),
            self.test_production_automation(),
            self.test_performance_optimization()
        ]
        
        # Execute tests
        for test_coro in tests_to_run:
            try:
                await test_coro
            except Exception as e:
                self.logger.error(f"Test execution failed: {e}")
        
        total_duration = time.time() - start_time
        
        # Compile results
        results = {
            "test_summary": {
                "total_tests": len(self.test_results),
                "successful_tests": sum(1 for t in self.test_results if t.success),
                "failed_tests": sum(1 for t in self.test_results if not t.success),
                "total_duration_seconds": total_duration,
                "success_rate": sum(1 for t in self.test_results if t.success) / len(self.test_results) * 100
            },
            "detailed_results": [
                {
                    "name": test.name,
                    "description": test.description,
                    "success": test.success,
                    "duration": test.duration,
                    "error": test.error,
                    "warnings": test.warnings,
                    "data": test.data
                }
                for test in self.test_results
            ]
        }
        
        return results
    
    async def run_specific_test(self, test_name: str) -> Optional[TestResult]:
        """Run a specific test by name."""
        test_map = {
            "batch_reviewer": self.test_batch_reviewer_initialization,
            "single_review": self.test_single_pipeline_review,
            "batch_review": self.test_batch_review_functionality,
            "report_generator": self.test_report_generation,
            "integrated_validation": self.test_integrated_validation,
            "production_automation": self.test_production_automation,
            "performance": self.test_performance_optimization
        }
        
        if test_name not in test_map:
            self.logger.error(f"Unknown test: {test_name}")
            return None
        
        self.logger.info(f"🔍 Running specific test: {test_name}")
        try:
            await test_map[test_name]()
            return self.test_results[-1] if self.test_results else None
        except Exception as e:
            self.logger.error(f"Test {test_name} failed: {e}")
            return None


async def main():
    """Main test execution function."""
    parser = argparse.ArgumentParser(
        description="Comprehensive test suite for batch processing integration"
    )
    parser.add_argument(
        "--component",
        type=str,
        choices=["batch_reviewer", "single_review", "batch_review", "report_generator", 
                "integrated_validation", "production_automation", "performance"],
        help="Run specific component test"
    )
    parser.add_argument(
        "--performance-test",
        action="store_true",
        help="Run performance optimization test only"
    )
    
    args = parser.parse_args()
    
    # Initialize test suite
    test_suite = BatchProcessingTestSuite()
    
    try:
        if args.component:
            # Run specific test
            result = await test_suite.run_specific_test(args.component)
            if result:
                status = "✅ PASSED" if result.success else "❌ FAILED"
                print(f"\n{status} {result.name} ({result.duration:.2f}s)")
                if result.error:
                    print(f"Error: {result.error}")
                if result.data:
                    print(f"Data: {json.dumps(result.data, indent=2)}")
            
        elif args.performance_test:
            # Run performance test only
            await test_suite.test_performance_optimization()
            result = test_suite.test_results[-1]
            status = "✅ PASSED" if result.success else "❌ FAILED"
            print(f"\n{status} Performance Test ({result.duration:.2f}s)")
            if result.data:
                print(f"Performance Data: {json.dumps(result.data, indent=2)}")
        
        else:
            # Run all tests
            results = await test_suite.run_all_tests()
            
            # Print summary
            summary = results["test_summary"]
            print(f"\n📊 Test Summary:")
            print(f"   Total Tests: {summary['total_tests']}")
            print(f"   Successful: {summary['successful_tests']}")
            print(f"   Failed: {summary['failed_tests']}")
            print(f"   Success Rate: {summary['success_rate']:.1f}%")
            print(f"   Total Duration: {summary['total_duration_seconds']:.2f}s")
            
            # Print detailed results
            print(f"\n📋 Detailed Results:")
            for test_result in results["detailed_results"]:
                status = "✅" if test_result["success"] else "❌"
                print(f"   {status} {test_result['name']} ({test_result['duration']:.2f}s)")
                if test_result["error"]:
                    print(f"      Error: {test_result['error']}")
                if test_result["warnings"]:
                    for warning in test_result["warnings"]:
                        print(f"      Warning: {warning}")
            
            # Save results
            results_file = Path("test_batch_processing_results.json")
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n📄 Full results saved to: {results_file}")
            
            # Exit with appropriate code
            if summary["failed_tests"] > 0:
                print(f"\n⚠️  {summary['failed_tests']} tests failed")
                sys.exit(1)
            else:
                print(f"\n🎉 All tests passed!")
    
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())