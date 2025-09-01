#!/usr/bin/env python3
"""
Comprehensive platform and performance test runner.

Executes all multi-platform compatibility and performance tests,
generates reports, and provides CI/CD integration.
"""

import asyncio
import argparse
import json
import logging
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add orchestrator to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import test classes
try:
    from tests.platform.test_platform_detection import PlatformCompatibilityTester
    from tests.platform.compatibility.test_macos_compatibility import MacOSCompatibilityTester
    from tests.platform.compatibility.test_linux_compatibility import LinuxCompatibilityTester
    from tests.platform.compatibility.test_windows_compatibility import WindowsCompatibilityTester
    from tests.platform.cross_platform.test_path_handling import CrossPlatformPathTester
    from tests.platform.cross_platform.test_external_dependencies import ExternalDependencyTester
    from tests.platform.cross_platform.test_api_connectivity import APIConnectivityTester
    from tests.performance.test_multi_platform_performance import MultiPlatformPerformanceTester
    from tests.performance.test_performance_alerts import PerformanceMonitor
except ImportError as e:
    logger.error(f"Failed to import test modules: {e}")
    sys.exit(1)


class PlatformTestRunner:
    """Comprehensive test runner for platform and performance testing."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.current_platform = platform.system()
        self.output_dir = output_dir or Path("tests/platform/results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.test_results = {}
        self.performance_results = {}
        self.start_time = None
        self.end_time = None
        
        # Test configuration
        self.test_suites = {
            "platform_detection": {
                "class": PlatformCompatibilityTester,
                "async": False,
                "critical": True,
                "description": "Basic platform detection and capabilities"
            },
            "platform_specific": {
                "class": self._get_platform_specific_tester(),
                "async": False,
                "critical": True,
                "description": f"{self.current_platform}-specific compatibility tests"
            },
            "path_handling": {
                "class": CrossPlatformPathTester,
                "async": False,
                "critical": True,
                "description": "Cross-platform path and file system handling"
            },
            "external_dependencies": {
                "class": ExternalDependencyTester,
                "async": False,
                "critical": True,
                "description": "External dependency loading and functionality"
            },
            "api_connectivity": {
                "class": APIConnectivityTester,
                "async": True,
                "critical": False,
                "description": "API endpoint connectivity and functionality"
            },
            "performance_benchmarks": {
                "class": MultiPlatformPerformanceTester,
                "async": True,
                "critical": False,
                "description": "Multi-platform performance benchmarking"
            }
        }
        
    def _get_platform_specific_tester(self):
        """Get platform-specific tester class."""
        if self.current_platform == "Darwin":
            return MacOSCompatibilityTester
        elif self.current_platform == "Linux":
            return LinuxCompatibilityTester
        elif self.current_platform == "Windows":
            return WindowsCompatibilityTester
        else:
            # Return base platform tester for unknown platforms
            return PlatformCompatibilityTester
    
    async def run_test_suite(self, suite_name: str, suite_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test suite."""
        logger.info(f"Running {suite_name}: {suite_config['description']}")
        
        start_time = time.time()
        
        try:
            # Initialize tester
            tester_class = suite_config["class"]
            tester = tester_class()
            
            # Initialize if needed
            if hasattr(tester, "initialize") and suite_config["async"]:
                await tester.initialize()
            elif hasattr(tester, "initialize"):
                # Handle sync initialization
                if asyncio.iscoroutinefunction(tester.initialize):
                    await tester.initialize()
                else:
                    tester.initialize()
            
            # Run tests
            if suite_config["async"]:
                if suite_name == "performance_benchmarks":
                    # Special handling for performance tests
                    await tester.initialize() if hasattr(tester, "initialize") else None
                    performance_results = await tester.run_comprehensive_benchmarks(iterations=2)
                    report = tester.generate_platform_report()
                    results = {
                        "performance_results": performance_results,
                        "report": report,
                        "platform": tester.current_platform
                    }
                else:
                    results = await tester.run_all_tests()
            else:
                results = tester.run_all_tests()
            
            execution_time = time.time() - start_time
            
            # Add metadata
            test_result = {
                "suite_name": suite_name,
                "description": suite_config["description"],
                "platform": self.current_platform,
                "execution_time_seconds": execution_time,
                "timestamp": datetime.utcnow().isoformat(),
                "critical": suite_config["critical"],
                "status": "completed",
                "results": results
            }
            
            # Determine success
            if "overall" in results:
                test_result["success"] = results["overall"]["success_rate"] >= 0.7
                test_result["success_rate"] = results["overall"]["success_rate"]
            else:
                # For performance tests or other formats
                test_result["success"] = not results.get("skipped", False)
                test_result["success_rate"] = 1.0 if test_result["success"] else 0.0
            
            logger.info(f"Completed {suite_name}: {test_result['success']} (rate: {test_result['success_rate']:.1%}, time: {execution_time:.1f}s)")
            
            return test_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Failed {suite_name}: {e}")
            
            return {
                "suite_name": suite_name,
                "description": suite_config["description"],
                "platform": self.current_platform,
                "execution_time_seconds": execution_time,
                "timestamp": datetime.utcnow().isoformat(),
                "critical": suite_config["critical"],
                "status": "failed",
                "success": False,
                "success_rate": 0.0,
                "error": str(e)
            }
    
    async def run_all_tests(self, test_filter: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run all configured test suites."""
        logger.info(f"Starting comprehensive platform testing on {self.current_platform}")
        self.start_time = datetime.utcnow()
        
        # Filter tests if specified
        suites_to_run = self.test_suites
        if test_filter:
            suites_to_run = {name: config for name, config in self.test_suites.items() 
                           if name in test_filter}
        
        # Run test suites
        test_results = {}
        for suite_name, suite_config in suites_to_run.items():
            result = await self.run_test_suite(suite_name, suite_config)
            test_results[suite_name] = result
        
        self.end_time = datetime.utcnow()
        self.test_results = test_results
        
        # Generate summary
        summary = self._generate_summary(test_results)
        
        return {
            "platform": self.current_platform,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "total_duration_seconds": (self.end_time - self.start_time).total_seconds(),
            "test_results": test_results,
            "summary": summary
        }
    
    def _generate_summary(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate test summary statistics."""
        total_tests = len(test_results)
        successful_tests = sum(1 for result in test_results.values() if result.get("success", False))
        critical_tests = [result for result in test_results.values() if result.get("critical", False)]
        critical_failures = sum(1 for result in critical_tests if not result.get("success", False))
        
        # Calculate overall success rates
        success_rates = [result.get("success_rate", 0.0) for result in test_results.values()]
        avg_success_rate = sum(success_rates) / len(success_rates) if success_rates else 0.0
        
        # Execution time statistics
        execution_times = [result.get("execution_time_seconds", 0.0) for result in test_results.values()]
        total_execution_time = sum(execution_times)
        
        # Overall status
        if critical_failures > 0:
            overall_status = "FAILED"
        elif successful_tests == total_tests:
            overall_status = "PASSED"
        else:
            overall_status = "PARTIAL"
        
        return {
            "overall_status": overall_status,
            "total_test_suites": total_tests,
            "successful_test_suites": successful_tests,
            "failed_test_suites": total_tests - successful_tests,
            "critical_test_suites": len(critical_tests),
            "critical_failures": critical_failures,
            "average_success_rate": avg_success_rate,
            "total_execution_time_seconds": total_execution_time,
            "platform_compatible": critical_failures == 0 and avg_success_rate >= 0.8
        }
    
    async def run_performance_analysis(self) -> Optional[Dict[str, Any]]:
        """Run performance analysis and generate alerts."""
        if "performance_benchmarks" not in self.test_results:
            logger.warning("No performance benchmark results available for analysis")
            return None
        
        performance_result = self.test_results["performance_benchmarks"]
        
        if not performance_result.get("success", False):
            logger.warning("Performance benchmarks failed, skipping analysis")
            return None
        
        try:
            # Extract performance metrics
            performance_data = performance_result.get("results", {}).get("report", {})
            
            if not performance_data:
                logger.warning("No performance data available for analysis")
                return None
            
            # Run performance monitoring
            monitor = PerformanceMonitor()
            ci_cd_report = monitor.generate_ci_cd_report(performance_data)
            
            # Save performance analysis
            analysis_file = self.output_dir / f"performance_analysis_{int(time.time())}.json"
            with open(analysis_file, 'w') as f:
                json.dump(ci_cd_report, f, indent=2, default=str)
            
            logger.info(f"Performance analysis saved to {analysis_file}")
            logger.info(f"Performance status: {ci_cd_report['summary']['overall_status']}")
            
            return ci_cd_report
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            return None
    
    def save_results(self) -> Path:
        """Save test results to file."""
        results_file = self.output_dir / f"platform_test_results_{self.current_platform.lower()}_{int(time.time())}.json"
        
        full_results = {
            "platform": self.current_platform,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "test_results": self.test_results,
            "summary": self._generate_summary(self.test_results) if self.test_results else {}
        }
        
        with open(results_file, 'w') as f:
            json.dump(full_results, f, indent=2, default=str)
        
        logger.info(f"Test results saved to {results_file}")
        return results_file
    
    def print_summary(self):
        """Print test summary to console."""
        if not self.test_results:
            print("No test results available")
            return
        
        summary = self._generate_summary(self.test_results)
        
        print("\n" + "="*80)
        print(f"PLATFORM COMPATIBILITY TEST SUMMARY - {self.current_platform}")
        print("="*80)
        
        print(f"\nOverall Status: {summary['overall_status']}")
        print(f"Platform Compatible: {'Yes' if summary['platform_compatible'] else 'No'}")
        
        print(f"\nTest Suites: {summary['successful_test_suites']}/{summary['total_test_suites']} passed")
        print(f"Critical Tests: {len([r for r in self.test_results.values() if r.get('critical', False)]) - summary['critical_failures']}/{len([r for r in self.test_results.values() if r.get('critical', False)])} passed")
        print(f"Average Success Rate: {summary['average_success_rate']:.1%}")
        print(f"Total Execution Time: {summary['total_execution_time_seconds']:.1f} seconds")
        
        print(f"\nDetailed Results:")
        for suite_name, result in self.test_results.items():
            status_emoji = "✅" if result.get("success", False) else "❌"
            critical_mark = " [CRITICAL]" if result.get("critical", False) else ""
            
            print(f"  {status_emoji} {suite_name}{critical_mark}: {result.get('success_rate', 0):.1%} "
                  f"({result.get('execution_time_seconds', 0):.1f}s)")
            
            if not result.get("success", False) and "error" in result:
                print(f"    Error: {result['error']}")


async def main():
    """Main entry point for platform testing."""
    parser = argparse.ArgumentParser(description="Comprehensive platform and performance testing")
    parser.add_argument("--output-dir", type=Path, help="Output directory for results")
    parser.add_argument("--filter", nargs="+", help="Filter specific test suites to run")
    parser.add_argument("--performance-only", action="store_true", help="Run only performance tests")
    parser.add_argument("--skip-performance", action="store_true", help="Skip performance tests")
    parser.add_argument("--ci-mode", action="store_true", help="CI/CD mode with appropriate exit codes")
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = PlatformTestRunner(output_dir=args.output_dir)
    
    # Determine which tests to run
    test_filter = args.filter
    if args.performance_only:
        test_filter = ["performance_benchmarks"]
    elif args.skip_performance:
        test_filter = [name for name in runner.test_suites.keys() if name != "performance_benchmarks"]
    
    try:
        # Run tests
        results = await runner.run_all_tests(test_filter=test_filter)
        
        # Run performance analysis if performance tests were executed
        performance_analysis = None
        if "performance_benchmarks" in runner.test_results:
            performance_analysis = await runner.run_performance_analysis()
        
        # Save results
        results_file = runner.save_results()
        
        # Print summary
        runner.print_summary()
        
        # CI/CD mode handling
        if args.ci_mode:
            summary = runner._generate_summary(runner.test_results)
            
            if summary["overall_status"] == "FAILED":
                print(f"\n❌ CI/CD: FAILED - Critical test failures detected")
                sys.exit(1)
            elif not summary["platform_compatible"]:
                print(f"\n⚠️  CI/CD: WARNING - Platform compatibility issues detected")
                sys.exit(2)
            elif performance_analysis and performance_analysis["summary"]["overall_status"] == "FAIL":
                print(f"\n⚠️  CI/CD: WARNING - Performance issues detected")
                sys.exit(2)
            else:
                print(f"\n✅ CI/CD: PASSED - All tests successful")
                sys.exit(0)
        
        print(f"\nResults saved to: {results_file}")
        
    except KeyboardInterrupt:
        logger.info("Testing interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Testing failed with error: {e}")
        if args.ci_mode:
            sys.exit(1)
        raise


if __name__ == "__main__":
    asyncio.run(main())