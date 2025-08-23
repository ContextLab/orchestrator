#!/usr/bin/env python3
"""
Main test runner script for pipeline tests with advanced features.

This script discovers and runs all pipeline tests from Streams 2-6, providing:
- Configurable parallelism
- Performance reporting
- Cost tracking
- Pass/fail summaries
- --fast flag to skip slow tests
- Comprehensive result analysis

Usage:
    python run_all.py                    # Run all tests
    python run_all.py --fast             # Skip slow tests
    python run_all.py --parallel 4       # Use 4 parallel workers
    python run_all.py --output report/   # Save reports to report/
    python run_all.py --help             # Show help
"""

import argparse
import asyncio
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import importlib.util
import pytest

# Performance tracking
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@dataclass
class TestRunConfiguration:
    """Configuration for test run execution."""
    
    parallel_workers: int = 1
    skip_slow_tests: bool = False
    output_directory: Optional[Path] = None
    verbose: bool = False
    timeout_per_test: int = 300  # 5 minutes per test
    max_cost_per_test: float = 1.0  # $1 max per test
    total_cost_limit: float = 30.0  # $30 total limit
    include_patterns: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)


@dataclass
class TestExecutionResult:
    """Result of executing a single test."""
    
    test_name: str
    module_name: str
    success: bool
    execution_time: float
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    estimated_cost: float = 0.0
    memory_usage_mb: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    output_summary: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class TestSuiteReport:
    """Comprehensive report of test suite execution."""
    
    # Basic statistics
    total_tests: int
    successful_tests: int
    failed_tests: int
    skipped_tests: int
    
    # Performance metrics
    total_execution_time: float
    average_execution_time: float
    fastest_test_time: float
    slowest_test_time: float
    
    # Cost tracking
    total_estimated_cost: float
    average_cost_per_test: float
    highest_cost_test: float
    cost_by_test_module: Dict[str, float] = field(default_factory=dict)
    
    # Resource usage
    peak_memory_usage_mb: Optional[float] = None
    total_memory_usage_mb: Optional[float] = None
    
    # Detailed results
    test_results: List[TestExecutionResult] = field(default_factory=list)
    failed_test_details: List[TestExecutionResult] = field(default_factory=list)
    warnings_summary: List[str] = field(default_factory=list)
    
    # Configuration
    run_configuration: TestRunConfiguration = field(default_factory=TestRunConfiguration)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_tests == 0:
            return 0.0
        return (self.successful_tests / self.total_tests) * 100.0
    
    @property
    def cost_efficiency(self) -> float:
        """Calculate cost efficiency (tests per dollar)."""
        if self.total_estimated_cost == 0.0:
            return float('inf')
        return self.successful_tests / self.total_estimated_cost


class TestDiscovery:
    """Utility for discovering available pipeline tests."""
    
    @staticmethod
    def discover_test_modules(test_directory: Path = None) -> List[Path]:
        """
        Discover all test modules in the pipeline_tests directory.
        
        Args:
            test_directory: Directory to search (defaults to current directory)
            
        Returns:
            List[Path]: Discovered test module paths
        """
        if test_directory is None:
            test_directory = Path(__file__).parent
        
        # Find all test_*.py files except test_base.py, run_all.py, and test_runner.py
        test_files = []
        for test_file in test_directory.glob("test_*.py"):
            if test_file.name in ["test_base.py", "run_all.py", "test_runner.py"]:
                continue
            test_files.append(test_file)
        
        return sorted(test_files)
    
    @staticmethod
    def get_test_metadata(test_file: Path) -> Dict[str, Any]:
        """
        Extract metadata from a test module.
        
        Args:
            test_file: Path to test module
            
        Returns:
            Dict[str, Any]: Test metadata
        """
        try:
            # Read first 50 lines to extract docstring and basic info
            with open(test_file, 'r') as f:
                lines = [f.readline() for _ in range(50)]
            
            content = ''.join(lines)
            
            # Basic metadata
            metadata = {
                'module': test_file.stem,
                'file_path': str(test_file),
                'estimated_slow': 'slow' in content.lower(),
                'estimated_cost_high': any(keyword in content.lower() for keyword in [
                    'expensive', 'cost', 'api calls', 'model'
                ]),
                'requires_network': any(keyword in content.lower() for keyword in [
                    'network', 'api', 'http', 'web'
                ]),
                'description': '',
                'tags': set()
            }
            
            # Extract docstring if available
            if '"""' in content:
                start = content.find('"""')
                end = content.find('"""', start + 3)
                if end != -1:
                    metadata['description'] = content[start+3:end].strip()
            
            # Extract tags from pytest markers
            for line in lines:
                if 'pytest.mark.' in line:
                    if 'slow' in line:
                        metadata['tags'].add('slow')
                    if 'integration' in line:
                        metadata['tags'].add('integration')
                    if 'unit' in line:
                        metadata['tags'].add('unit')
            
            metadata['tags'] = list(metadata['tags'])
            return metadata
            
        except Exception:
            return {
                'module': test_file.stem,
                'file_path': str(test_file),
                'description': f'Test module: {test_file.name}',
                'tags': []
            }


class TestExecutor:
    """Main test execution engine with parallel processing support."""
    
    def __init__(self, config: TestRunConfiguration):
        """Initialize test executor with configuration."""
        self.config = config
        self.results: List[TestExecutionResult] = []
        self.start_time = time.time()
        self.peak_memory = 0.0
    
    async def run_all_tests(self, test_modules: List[Path]) -> TestSuiteReport:
        """
        Execute all discovered tests with parallel processing.
        
        Args:
            test_modules: List of test module paths to execute
            
        Returns:
            TestSuiteReport: Comprehensive test results
        """
        print(f"🚀 Starting test execution with {len(test_modules)} modules")
        print(f"⚙️  Configuration: {self.config.parallel_workers} workers, "
              f"fast_mode={self.config.skip_slow_tests}")
        
        # Filter test modules based on configuration
        filtered_modules = self._filter_test_modules(test_modules)
        
        if self.config.parallel_workers > 1:
            results = await self._run_tests_parallel(filtered_modules)
        else:
            results = await self._run_tests_sequential(filtered_modules)
        
        # Generate comprehensive report
        return self._generate_report(results)
    
    def _filter_test_modules(self, test_modules: List[Path]) -> List[Path]:
        """Filter test modules based on configuration."""
        filtered = []
        
        for module in test_modules:
            metadata = TestDiscovery.get_test_metadata(module)
            
            # Skip slow tests if fast mode enabled
            if self.config.skip_slow_tests and metadata.get('estimated_slow', False):
                print(f"⏩ Skipping slow test: {module.name}")
                continue
            
            # Apply include/exclude patterns
            module_name = module.stem
            
            if self.config.exclude_patterns:
                if any(pattern in module_name for pattern in self.config.exclude_patterns):
                    print(f"⏩ Excluding test: {module.name}")
                    continue
            
            if self.config.include_patterns:
                if not any(pattern in module_name for pattern in self.config.include_patterns):
                    continue
            
            filtered.append(module)
        
        print(f"📊 Running {len(filtered)} tests (filtered from {len(test_modules)})")
        return filtered
    
    async def _run_tests_sequential(self, test_modules: List[Path]) -> List[TestExecutionResult]:
        """Run tests sequentially."""
        results = []
        
        for i, module in enumerate(test_modules, 1):
            print(f"▶️  [{i}/{len(test_modules)}] Running {module.name}")
            result = await self._execute_test_module(module)
            results.append(result)
            
            # Progress reporting
            if result.success:
                print(f"✅ {module.name} passed ({result.execution_time:.1f}s, ${result.estimated_cost:.3f})")
            else:
                print(f"❌ {module.name} failed ({result.execution_time:.1f}s)")
                if result.error_message:
                    print(f"   Error: {result.error_message}")
        
        return results
    
    async def _run_tests_parallel(self, test_modules: List[Path]) -> List[TestExecutionResult]:
        """Run tests in parallel using ThreadPoolExecutor."""
        print(f"🔧 Using {self.config.parallel_workers} parallel workers")
        
        results = []
        with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            # Submit all tasks
            future_to_module = {
                executor.submit(self._execute_test_module_sync, module): module
                for module in test_modules
            }
            
            # Process completed tasks
            completed = 0
            for future in as_completed(future_to_module):
                module = future_to_module[future]
                completed += 1
                
                try:
                    result = future.result(timeout=self.config.timeout_per_test)
                    results.append(result)
                    
                    status = "✅" if result.success else "❌"
                    print(f"{status} [{completed}/{len(test_modules)}] {module.name} "
                          f"({result.execution_time:.1f}s, ${result.estimated_cost:.3f})")
                    
                except Exception as e:
                    # Create error result for failed execution
                    error_result = TestExecutionResult(
                        test_name=module.name,
                        module_name=module.stem,
                        success=False,
                        execution_time=0.0,
                        error_message=f"Execution failed: {e}",
                        error_traceback=str(e)
                    )
                    results.append(error_result)
                    print(f"❌ [{completed}/{len(test_modules)}] {module.name} execution failed: {e}")
        
        return results
    
    def _execute_test_module_sync(self, module_path: Path) -> TestExecutionResult:
        """Synchronous wrapper for test execution (for ThreadPoolExecutor)."""
        import asyncio
        return asyncio.run(self._execute_test_module(module_path))
    
    async def _execute_test_module(self, module_path: Path) -> TestExecutionResult:
        """
        Execute a single test module using pytest.
        
        Args:
            module_path: Path to test module
            
        Returns:
            TestExecutionResult: Execution results
        """
        start_time = time.time()
        memory_start = self._get_memory_usage()
        
        try:
            # Run pytest on the specific module
            test_args = [
                str(module_path),
                "-v",  # Verbose output
                "--tb=short",  # Short traceback format
                "-x",  # Stop on first failure
                "--disable-warnings",  # Reduce noise
            ]
            
            # Add timeout if configured
            if self.config.timeout_per_test:
                test_args.extend(["--timeout", str(self.config.timeout_per_test)])
            
            # Execute pytest programmatically
            exit_code = pytest.main(test_args)
            
            execution_time = time.time() - start_time
            memory_end = self._get_memory_usage()
            memory_used = memory_end - memory_start if memory_start and memory_end else None
            
            # Update peak memory tracking
            if memory_end and memory_end > self.peak_memory:
                self.peak_memory = memory_end
            
            # Analyze results
            success = exit_code == 0
            
            result = TestExecutionResult(
                test_name=module_path.name,
                module_name=module_path.stem,
                success=success,
                execution_time=execution_time,
                memory_usage_mb=memory_used,
                estimated_cost=self._estimate_test_cost(module_path, execution_time),
                error_message=None if success else f"pytest exit code: {exit_code}"
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return TestExecutionResult(
                test_name=module_path.name,
                module_name=module_path.stem,
                success=False,
                execution_time=execution_time,
                error_message=str(e),
                error_traceback=str(e),
                estimated_cost=0.0
            )
    
    def _estimate_test_cost(self, module_path: Path, execution_time: float) -> float:
        """
        Estimate cost of test execution based on module and execution time.
        
        Args:
            module_path: Path to test module
            execution_time: Test execution time in seconds
            
        Returns:
            float: Estimated cost in USD
        """
        # Base cost estimation factors
        base_cost_per_minute = 0.10  # $0.10 per minute baseline
        
        # Module-specific cost multipliers
        cost_multipliers = {
            'model_pipelines': 3.0,    # Model tests are expensive
            'integration': 2.0,        # Integration tests use more resources
            'data_processing': 1.5,    # Data processing tests
            'control_flow': 1.0,       # Control flow tests
            'validation': 1.0,         # Validation tests
            'base': 0.5               # Base infrastructure tests
        }
        
        # Determine multiplier based on module name
        multiplier = 1.0
        for module_type, mult in cost_multipliers.items():
            if module_type in module_path.stem:
                multiplier = mult
                break
        
        # Calculate estimated cost
        cost = (execution_time / 60.0) * base_cost_per_minute * multiplier
        return round(cost, 4)
    
    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage in MB."""
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                return process.memory_info().rss / 1024 / 1024  # Convert to MB
            except Exception:
                return None
        return None
    
    def _generate_report(self, results: List[TestExecutionResult]) -> TestSuiteReport:
        """Generate comprehensive test suite report."""
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r.success)
        failed_tests = total_tests - successful_tests
        
        execution_times = [r.execution_time for r in results]
        costs = [r.estimated_cost for r in results]
        
        # Calculate cost by module
        cost_by_module = {}
        for result in results:
            module_base = result.module_name.replace('test_', '')
            if module_base not in cost_by_module:
                cost_by_module[module_base] = 0.0
            cost_by_module[module_base] += result.estimated_cost
        
        report = TestSuiteReport(
            total_tests=total_tests,
            successful_tests=successful_tests,
            failed_tests=failed_tests,
            skipped_tests=0,  # Not implemented yet
            total_execution_time=time.time() - self.start_time,
            average_execution_time=sum(execution_times) / len(execution_times) if execution_times else 0.0,
            fastest_test_time=min(execution_times) if execution_times else 0.0,
            slowest_test_time=max(execution_times) if execution_times else 0.0,
            total_estimated_cost=sum(costs),
            average_cost_per_test=sum(costs) / len(costs) if costs else 0.0,
            highest_cost_test=max(costs) if costs else 0.0,
            cost_by_test_module=cost_by_module,
            peak_memory_usage_mb=self.peak_memory if self.peak_memory > 0 else None,
            test_results=results,
            failed_test_details=[r for r in results if not r.success],
            run_configuration=self.config
        )
        
        return report


class ReportGenerator:
    """Utility for generating comprehensive test reports."""
    
    @staticmethod
    def print_summary_report(report: TestSuiteReport):
        """Print a summary report to console."""
        print("\n" + "="*80)
        print("🎯 PIPELINE TEST SUITE SUMMARY")
        print("="*80)
        
        # Basic statistics
        print(f"📊 Test Results:")
        print(f"   Total Tests:     {report.total_tests}")
        print(f"   Passed:          {report.successful_tests} ({report.success_rate:.1f}%)")
        print(f"   Failed:          {report.failed_tests}")
        print(f"   Skipped:         {report.skipped_tests}")
        
        # Performance metrics
        print(f"\n⏱️  Performance:")
        print(f"   Total Time:      {report.total_execution_time:.1f}s")
        print(f"   Average Time:    {report.average_execution_time:.1f}s per test")
        print(f"   Fastest Test:    {report.fastest_test_time:.1f}s")
        print(f"   Slowest Test:    {report.slowest_test_time:.1f}s")
        
        # Cost analysis
        print(f"\n💰 Cost Analysis:")
        print(f"   Total Cost:      ${report.total_estimated_cost:.3f}")
        print(f"   Average Cost:    ${report.average_cost_per_test:.3f} per test")
        print(f"   Highest Cost:    ${report.highest_cost_test:.3f}")
        print(f"   Cost Efficiency: {report.cost_efficiency:.1f} tests per $")
        
        # Memory usage
        if report.peak_memory_usage_mb:
            print(f"\n🖥️  Memory Usage:")
            print(f"   Peak Memory:     {report.peak_memory_usage_mb:.1f} MB")
        
        # Failed tests details
        if report.failed_test_details:
            print(f"\n❌ Failed Tests:")
            for failure in report.failed_test_details:
                print(f"   • {failure.test_name}: {failure.error_message}")
        
        # Cost breakdown by module
        if report.cost_by_test_module:
            print(f"\n💸 Cost by Test Module:")
            sorted_costs = sorted(report.cost_by_test_module.items(), 
                                 key=lambda x: x[1], reverse=True)
            for module, cost in sorted_costs:
                print(f"   {module:20} ${cost:.3f}")
        
        # Configuration
        print(f"\n⚙️  Configuration:")
        print(f"   Parallel Workers: {report.run_configuration.parallel_workers}")
        print(f"   Fast Mode:       {report.run_configuration.skip_slow_tests}")
        print(f"   Timeout/Test:    {report.run_configuration.timeout_per_test}s")
        
        print("="*80)
        
        # Final status
        if report.failed_tests == 0:
            print("🎉 ALL TESTS PASSED! 🎉")
        else:
            print(f"⚠️  {report.failed_tests} TESTS FAILED")
        
        print("="*80 + "\n")
    
    @staticmethod
    def save_detailed_report(report: TestSuiteReport, output_path: Path):
        """Save detailed report to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert report to dictionary
        report_data = asdict(report)
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"📄 Detailed report saved to: {output_path}")
    
    @staticmethod
    def save_html_report(report: TestSuiteReport, output_path: Path):
        """Generate and save HTML report."""
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pipeline Test Suite Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .success {{ color: #28a745; }}
        .failure {{ color: #dc3545; }}
        .warning {{ color: #ffc107; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f8f9fa; }}
        .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
        .section {{ margin: 30px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 Pipeline Test Suite Report</h1>
        <div class="metric"><strong>Success Rate:</strong> 
            <span class="{'success' if report.success_rate > 80 else 'warning' if report.success_rate > 60 else 'failure'}">
                {report.success_rate:.1f}%
            </span>
        </div>
        <div class="metric"><strong>Total Time:</strong> {report.total_execution_time:.1f}s</div>
        <div class="metric"><strong>Total Cost:</strong> ${report.total_estimated_cost:.3f}</div>
        <div class="metric"><strong>Tests:</strong> {report.total_tests}</div>
    </div>

    <div class="section">
        <h2>📊 Test Results Overview</h2>
        <table>
            <tr>
                <th>Test Module</th>
                <th>Status</th>
                <th>Execution Time</th>
                <th>Estimated Cost</th>
                <th>Memory Usage</th>
            </tr>
"""
        
        for result in report.test_results:
            status_class = "success" if result.success else "failure"
            status_text = "✅ PASS" if result.success else "❌ FAIL"
            memory_text = f"{result.memory_usage_mb:.1f} MB" if result.memory_usage_mb else "N/A"
            
            html_content += f"""
            <tr>
                <td>{result.module_name}</td>
                <td class="{status_class}">{status_text}</td>
                <td>{result.execution_time:.1f}s</td>
                <td>${result.estimated_cost:.3f}</td>
                <td>{memory_text}</td>
            </tr>
"""
        
        html_content += """
        </table>
    </div>
"""
        
        if report.failed_test_details:
            html_content += """
    <div class="section">
        <h2>❌ Failed Tests</h2>
        <table>
            <tr>
                <th>Test</th>
                <th>Error Message</th>
            </tr>
"""
            for failure in report.failed_test_details:
                html_content += f"""
            <tr>
                <td>{failure.test_name}</td>
                <td class="failure">{failure.error_message or 'Unknown error'}</td>
            </tr>
"""
            html_content += """
        </table>
    </div>
"""
        
        html_content += """
</body>
</html>
"""
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"🌐 HTML report saved to: {output_path}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Advanced test runner for pipeline tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Run all tests
  %(prog)s --fast                    # Skip slow tests
  %(prog)s --parallel 4              # Use 4 parallel workers
  %(prog)s --output reports/         # Save reports to reports/
  %(prog)s --include model           # Only run model-related tests
  %(prog)s --exclude integration     # Skip integration tests
  %(prog)s --timeout 600             # Set 10 minute timeout per test
        """
    )
    
    parser.add_argument(
        "--parallel", "-p",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)"
    )
    
    parser.add_argument(
        "--fast", "-f",
        action="store_true",
        help="Skip slow tests for faster execution"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output directory for reports (default: ./test_reports)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout per test in seconds (default: 300)"
    )
    
    parser.add_argument(
        "--max-cost",
        type=float,
        default=30.0,
        help="Maximum total cost limit in USD (default: 30.0)"
    )
    
    parser.add_argument(
        "--include",
        type=str,
        action="append",
        help="Include tests matching pattern (can be used multiple times)"
    )
    
    parser.add_argument(
        "--exclude", 
        type=str,
        action="append",
        help="Exclude tests matching pattern (can be used multiple times)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what tests would run without executing them"
    )
    
    return parser.parse_args()


async def main():
    """Main entry point for test runner."""
    args = parse_arguments()
    
    # Create configuration
    config = TestRunConfiguration(
        parallel_workers=args.parallel,
        skip_slow_tests=args.fast,
        output_directory=args.output or Path("test_reports"),
        verbose=args.verbose,
        timeout_per_test=args.timeout,
        total_cost_limit=args.max_cost,
        include_patterns=args.include or [],
        exclude_patterns=args.exclude or []
    )
    
    # Discover test modules
    print("🔍 Discovering test modules...")
    test_modules = TestDiscovery.discover_test_modules()
    
    if not test_modules:
        print("❌ No test modules found!")
        return 1
    
    print(f"📁 Found {len(test_modules)} test modules:")
    for module in test_modules:
        metadata = TestDiscovery.get_test_metadata(module)
        tags = ", ".join(metadata.get('tags', []))
        print(f"   • {module.name} ({tags if tags else 'no tags'})")
    
    # Dry run mode
    if args.dry_run:
        print("\n🔍 DRY RUN - Would execute these tests:")
        executor = TestExecutor(config)
        filtered_modules = executor._filter_test_modules(test_modules)
        for module in filtered_modules:
            print(f"   ✓ {module.name}")
        print(f"\nTotal: {len(filtered_modules)} tests would run")
        return 0
    
    # Execute tests
    executor = TestExecutor(config)
    try:
        report = await executor.run_all_tests(test_modules)
    except KeyboardInterrupt:
        print("\n⚠️  Test execution interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        return 1
    
    # Generate reports
    ReportGenerator.print_summary_report(report)
    
    # Save detailed reports
    if config.output_directory:
        timestamp = int(time.time())
        json_report_path = config.output_directory / f"test_report_{timestamp}.json"
        html_report_path = config.output_directory / f"test_report_{timestamp}.html"
        
        ReportGenerator.save_detailed_report(report, json_report_path)
        ReportGenerator.save_html_report(report, html_report_path)
    
    # Return appropriate exit code
    return 0 if report.failed_tests == 0 else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)