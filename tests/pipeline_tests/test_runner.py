"""Test runner utilities for pipeline testing with advanced analysis and reporting."""

import asyncio
import json
import shutil
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import yaml

from orchestrator import Orchestrator
from src.orchestrator.models.model_registry import ModelRegistry

from .test_base import BasePipelineTest, PipelineExecutionResult, PipelineTestConfiguration


class _ConcretePipelineTest(BasePipelineTest):
    """Concrete implementation of BasePipelineTest for use in test runner."""
    
    def test_basic_execution(self):
        """Not used in test runner context."""
        pass
    
    def test_error_handling(self):
        """Not used in test runner context."""
        pass


@dataclass
class PipelineTestCase:
    """Definition of a single pipeline test case."""
    
    name: str
    yaml_content: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    expected_outputs: Dict[str, Any] = field(default_factory=dict)
    expected_success: bool = True
    description: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    timeout_override: Optional[int] = None
    cost_limit_override: Optional[float] = None


@dataclass
class TestSuiteResult:
    """Results from running a complete test suite."""
    
    suite_name: str
    total_tests: int
    successful_tests: int
    failed_tests: int
    skipped_tests: int
    total_execution_time: float
    total_cost: float
    test_results: List[Tuple[PipelineTestCase, PipelineExecutionResult]] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_tests == 0:
            return 0.0
        return (self.successful_tests / self.total_tests) * 100
    
    @property
    def average_execution_time(self) -> float:
        """Calculate average execution time per test."""
        if not self.test_results:
            return 0.0
        return sum(result.execution_time for _, result in self.test_results) / len(self.test_results)
    
    @property
    def average_cost(self) -> float:
        """Calculate average cost per test."""
        if not self.test_results:
            return 0.0
        return sum(result.estimated_cost for _, result in self.test_results) / len(self.test_results)


class PipelineLoader:
    """Utility for loading and parsing pipeline YAML files."""
    
    @staticmethod
    def load_from_file(file_path: Union[str, Path]) -> str:
        """
        Load pipeline YAML from file.
        
        Args:
            file_path: Path to YAML file
            
        Returns:
            str: YAML content
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file cannot be parsed
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Pipeline file not found: {path}")
        
        try:
            return path.read_text(encoding='utf-8')
        except Exception as e:
            raise ValueError(f"Failed to read pipeline file {path}: {e}")
    
    @staticmethod
    def validate_yaml_syntax(yaml_content: str) -> bool:
        """
        Validate YAML syntax without executing.
        
        Args:
            yaml_content: YAML content to validate
            
        Returns:
            bool: True if valid YAML
        """
        try:
            yaml.safe_load(yaml_content)
            return True
        except yaml.YAMLError:
            return False
    
    @staticmethod
    def extract_pipeline_metadata(yaml_content: str) -> Dict[str, Any]:
        """
        Extract metadata from pipeline YAML.
        
        Args:
            yaml_content: YAML content
            
        Returns:
            Dict[str, Any]: Pipeline metadata
        """
        try:
            data = yaml.safe_load(yaml_content)
            if not isinstance(data, dict):
                return {}
            
            metadata = {
                'name': data.get('name', 'Unknown'),
                'description': data.get('description', ''),
                'version': data.get('version', '1.0.0'),
                'task_count': len(data.get('tasks', [])),
                'has_outputs': 'outputs' in data,
                'has_inputs': 'inputs' in data,
            }
            
            # Extract task types
            task_types = set()
            for task in data.get('tasks', []):
                if isinstance(task, dict) and 'type' in task:
                    task_types.add(task['type'])
            metadata['task_types'] = list(task_types)
            
            return metadata
            
        except Exception:
            return {}


class OutputDirectoryManager:
    """Utility for managing pipeline output directories during testing."""
    
    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize output directory manager.
        
        Args:
            base_dir: Base directory for test outputs (defaults to temp)
        """
        self.base_dir = base_dir or Path.cwd() / "test_outputs"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.created_dirs: Set[Path] = set()
    
    def create_test_directory(self, test_name: str, timestamp: bool = True) -> Path:
        """
        Create a directory for test outputs.
        
        Args:
            test_name: Name of the test
            timestamp: Whether to include timestamp in directory name
            
        Returns:
            Path: Created directory path
        """
        dir_name = test_name.replace(' ', '_').replace('/', '_')
        if timestamp:
            dir_name += f"_{int(time.time())}"
        
        test_dir = self.base_dir / dir_name
        test_dir.mkdir(parents=True, exist_ok=True)
        self.created_dirs.add(test_dir)
        
        return test_dir
    
    def cleanup_test_directories(self, keep_failed: bool = True):
        """
        Clean up created test directories.
        
        Args:
            keep_failed: Whether to keep directories from failed tests
        """
        for test_dir in self.created_dirs:
            if test_dir.exists():
                # If keep_failed is True, check for error indicators
                if keep_failed:
                    error_files = list(test_dir.glob("*error*")) + list(test_dir.glob("*fail*"))
                    if error_files:
                        continue  # Keep this directory
                
                try:
                    shutil.rmtree(test_dir)
                except Exception:
                    pass  # Ignore cleanup errors
        
        self.created_dirs.clear()
    
    def save_test_result(self, test_dir: Path, result: PipelineExecutionResult, test_case: PipelineTestCase):
        """
        Save test result to directory.
        
        Args:
            test_dir: Directory to save to
            result: Test execution result
            test_case: Test case definition
        """
        # Save result as JSON
        result_data = asdict(result)
        # Convert non-serializable fields
        if result.error:
            result_data['error'] = str(result.error)
        
        result_file = test_dir / "result.json"
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2, default=str)
        
        # Save test case definition
        case_file = test_dir / "test_case.yaml"
        case_data = {
            'name': test_case.name,
            'description': test_case.description,
            'inputs': test_case.inputs,
            'expected_outputs': test_case.expected_outputs,
            'expected_success': test_case.expected_success,
            'tags': list(test_case.tags)
        }
        with open(case_file, 'w') as f:
            yaml.dump(case_data, f, indent=2)
        
        # Save pipeline YAML
        pipeline_file = test_dir / "pipeline.yaml"
        pipeline_file.write_text(test_case.yaml_content)
        
        # Save outputs if available
        if result.outputs:
            outputs_file = test_dir / "outputs.json"
            with open(outputs_file, 'w') as f:
                json.dump(result.outputs, f, indent=2, default=str)


class ResultComparator:
    """Utility for comparing pipeline execution results."""
    
    @staticmethod
    def compare_outputs(actual: Dict[str, Any], 
                       expected: Dict[str, Any],
                       fuzzy_match: bool = True,
                       similarity_threshold: float = 0.8) -> Dict[str, bool]:
        """
        Compare actual outputs with expected outputs.
        
        Args:
            actual: Actual pipeline outputs
            expected: Expected outputs
            fuzzy_match: Whether to use fuzzy string matching
            similarity_threshold: Minimum similarity for fuzzy matching
            
        Returns:
            Dict[str, bool]: Comparison results for each key
        """
        comparison_results = {}
        
        # Check that all expected keys are present
        for key in expected.keys():
            if key not in actual:
                comparison_results[key] = False
                continue
            
            actual_value = actual[key]
            expected_value = expected[key]
            
            # Exact match check
            if actual_value == expected_value:
                comparison_results[key] = True
                continue
            
            # Type-specific comparison
            if isinstance(expected_value, str) and isinstance(actual_value, str):
                if fuzzy_match:
                    similarity = ResultComparator._calculate_string_similarity(
                        actual_value, expected_value
                    )
                    comparison_results[key] = similarity >= similarity_threshold
                else:
                    comparison_results[key] = actual_value == expected_value
            
            elif isinstance(expected_value, (int, float)) and isinstance(actual_value, (int, float)):
                # Numeric comparison with small tolerance
                tolerance = abs(expected_value * 0.01)  # 1% tolerance
                comparison_results[key] = abs(actual_value - expected_value) <= tolerance
            
            else:
                # Fallback to exact match
                comparison_results[key] = actual_value == expected_value
        
        return comparison_results
    
    @staticmethod
    def _calculate_string_similarity(str1: str, str2: str) -> float:
        """
        Calculate similarity between two strings using a simple metric.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            float: Similarity score (0-1)
        """
        if not str1 and not str2:
            return 1.0
        if not str1 or not str2:
            return 0.0
        
        # Simple word-based similarity
        words1 = set(str1.lower().split())
        words2 = set(str2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)


class QualityScorer:
    """Utility for scoring pipeline execution quality."""
    
    @staticmethod
    def calculate_output_quality_score(outputs: Dict[str, Any]) -> float:
        """
        Calculate quality score for pipeline outputs.
        
        Args:
            outputs: Pipeline outputs to score
            
        Returns:
            float: Quality score (0-1)
        """
        if not outputs:
            return 0.0
        
        score = 0.0
        total_weight = 0.0
        
        for key, value in outputs.items():
            weight = 1.0
            item_score = 0.0
            
            if isinstance(value, str):
                # String quality assessment
                if value.strip():  # Non-empty
                    item_score += 0.3
                
                # Length assessment (prefer meaningful content)
                if 10 <= len(value) <= 1000:  # Reasonable length
                    item_score += 0.3
                
                # Content quality indicators
                if not any(indicator in value.lower() for indicator in [
                    'error', 'failed', 'null', 'undefined', 'none'
                ]):
                    item_score += 0.4
            
            elif isinstance(value, (int, float)):
                # Numeric values are generally good
                item_score = 0.8
            
            elif isinstance(value, (list, dict)):
                # Complex data structures
                if value:  # Non-empty
                    item_score = 0.9
                else:
                    item_score = 0.2
            
            else:
                # Other types get moderate score
                item_score = 0.5
            
            score += item_score * weight
            total_weight += weight
        
        return score / total_weight if total_weight > 0 else 0.0
    
    @staticmethod
    def calculate_performance_score(execution_time: float,
                                  cost: float,
                                  api_calls: int,
                                  success: bool) -> float:
        """
        Calculate performance score based on execution metrics.
        
        Args:
            execution_time: Time taken in seconds
            cost: Estimated cost in dollars
            api_calls: Number of API calls made
            success: Whether execution was successful
            
        Returns:
            float: Performance score (0-1)
        """
        if not success:
            return 0.0
        
        score = 0.4  # Base score for success
        
        # Time performance (faster is better)
        if execution_time < 10:
            score += 0.3
        elif execution_time < 30:
            score += 0.2
        elif execution_time < 60:
            score += 0.1
        
        # Cost performance (cheaper is better)
        if cost < 0.01:
            score += 0.2
        elif cost < 0.05:
            score += 0.15
        elif cost < 0.10:
            score += 0.1
        elif cost < 0.20:
            score += 0.05
        
        # API efficiency (fewer calls for same result is better)
        if api_calls <= 2:
            score += 0.1
        elif api_calls <= 5:
            score += 0.05
        
        return min(1.0, score)


class PipelineTestRunner:
    """Main test runner for executing pipeline test suites."""
    
    def __init__(self,
                 orchestrator: Orchestrator,
                 model_registry: ModelRegistry,
                 config: Optional[PipelineTestConfiguration] = None,
                 output_manager: Optional[OutputDirectoryManager] = None):
        """
        Initialize test runner.
        
        Args:
            orchestrator: Orchestrator instance
            model_registry: Model registry
            config: Test configuration
            output_manager: Output directory manager
        """
        self.orchestrator = orchestrator
        self.model_registry = model_registry
        self.config = config or PipelineTestConfiguration()
        self.output_manager = output_manager or OutputDirectoryManager()
        
        # Store components for creating test instances when needed
        self._orchestrator = orchestrator
        self._model_registry = model_registry
        self._config = config
    
    async def run_test_suite_async(self,
                                 test_cases: List[PipelineTestCase],
                                 suite_name: str = "Pipeline Test Suite",
                                 parallel: bool = False,
                                 max_workers: int = 3) -> TestSuiteResult:
        """
        Run a complete test suite asynchronously.
        
        Args:
            test_cases: List of test cases to execute
            suite_name: Name of the test suite
            parallel: Whether to run tests in parallel
            max_workers: Maximum parallel workers
            
        Returns:
            TestSuiteResult: Complete test suite results
        """
        start_time = time.time()
        
        if parallel and len(test_cases) > 1:
            results = await self._run_tests_parallel(test_cases, max_workers)
        else:
            results = await self._run_tests_sequential(test_cases)
        
        end_time = time.time()
        
        # Compile suite results
        successful = sum(1 for _, result in results if result.success)
        failed = sum(1 for _, result in results if not result.success)
        total_cost = sum(result.estimated_cost for _, result in results)
        
        return TestSuiteResult(
            suite_name=suite_name,
            total_tests=len(test_cases),
            successful_tests=successful,
            failed_tests=failed,
            skipped_tests=0,  # Not implemented yet
            total_execution_time=end_time - start_time,
            total_cost=total_cost,
            test_results=results
        )
    
    def run_test_suite_sync(self,
                          test_cases: List[PipelineTestCase],
                          suite_name: str = "Pipeline Test Suite",
                          parallel: bool = False,
                          max_workers: int = 3) -> TestSuiteResult:
        """
        Run test suite synchronously (wrapper for async method).
        
        Args:
            test_cases: List of test cases to execute
            suite_name: Name of the test suite
            parallel: Whether to run tests in parallel
            max_workers: Maximum parallel workers
            
        Returns:
            TestSuiteResult: Complete test suite results
        """
        return asyncio.run(self.run_test_suite_async(
            test_cases, suite_name, parallel, max_workers
        ))
    
    async def _run_tests_sequential(self,
                                  test_cases: List[PipelineTestCase]) -> List[Tuple[PipelineTestCase, PipelineExecutionResult]]:
        """Run tests sequentially."""
        results = []
        
        for test_case in test_cases:
            print(f"Running test: {test_case.name}")
            result = await self._execute_single_test(test_case)
            results.append((test_case, result))
            
            # Save results immediately
            if self.config.save_intermediate_outputs:
                test_dir = self.output_manager.create_test_directory(test_case.name)
                self.output_manager.save_test_result(test_dir, result, test_case)
        
        return results
    
    async def _run_tests_parallel(self,
                                test_cases: List[PipelineTestCase],
                                max_workers: int) -> List[Tuple[PipelineTestCase, PipelineExecutionResult]]:
        """Run tests in parallel."""
        semaphore = asyncio.Semaphore(max_workers)
        
        async def run_with_semaphore(test_case):
            async with semaphore:
                return test_case, await self._execute_single_test(test_case)
        
        tasks = [run_with_semaphore(test_case) for test_case in test_cases]
        results = await asyncio.gather(*tasks)
        
        # Save all results
        if self.config.save_intermediate_outputs:
            for test_case, result in results:
                test_dir = self.output_manager.create_test_directory(test_case.name)
                self.output_manager.save_test_result(test_dir, result, test_case)
        
        return results
    
    async def _execute_single_test(self, test_case: PipelineTestCase) -> PipelineExecutionResult:
        """Execute a single test case."""
        # Create output directory for this test
        test_output_dir = self.output_manager.create_test_directory(test_case.name, timestamp=False)
        
        # Create a concrete test instance for execution
        test_instance = _ConcretePipelineTest(
            self._orchestrator,
            self._model_registry, 
            self._config
        )
        
        # Execute the pipeline
        result = await test_instance.execute_pipeline_async(
            test_case.yaml_content,
            test_case.inputs,
            test_output_dir
        )
        
        # Additional validation based on test case expectations
        if test_case.expected_outputs:
            comparison_results = ResultComparator.compare_outputs(
                result.outputs,
                test_case.expected_outputs
            )
            result.validation_results.update(comparison_results)
            
            # Update output validation based on comparison
            result.output_validation = all(comparison_results.values())
        
        # Check if result matches expected success
        if result.success != test_case.expected_success:
            result.warnings.append(
                f"Expected success={test_case.expected_success}, got success={result.success}"
            )
        
        return result
    
    def generate_test_report(self, suite_result: TestSuiteResult) -> str:
        """
        Generate a comprehensive test report.
        
        Args:
            suite_result: Test suite results
            
        Returns:
            str: Formatted test report
        """
        report_lines = [
            f"# Pipeline Test Report: {suite_result.suite_name}",
            "",
            "## Summary",
            f"- Total Tests: {suite_result.total_tests}",
            f"- Successful: {suite_result.successful_tests}",
            f"- Failed: {suite_result.failed_tests}",
            f"- Success Rate: {suite_result.success_rate:.1f}%",
            f"- Total Execution Time: {suite_result.total_execution_time:.2f}s",
            f"- Average Execution Time: {suite_result.average_execution_time:.2f}s",
            f"- Total Cost: ${suite_result.total_cost:.4f}",
            f"- Average Cost per Test: ${suite_result.average_cost:.4f}",
            "",
            "## Individual Test Results",
            ""
        ]
        
        for test_case, result in suite_result.test_results:
            status = "✅ PASS" if result.success else "❌ FAIL"
            report_lines.extend([
                f"### {test_case.name} - {status}",
                f"- Execution Time: {result.execution_time:.2f}s",
                f"- Cost: ${result.estimated_cost:.4f}",
                f"- Performance Score: {result.performance_score:.2f}",
                f"- Quality Score: {QualityScorer.calculate_output_quality_score(result.outputs):.2f}",
                ""
            ])
            
            if not result.success and result.error_message:
                report_lines.extend([
                    "**Error Details:**",
                    f"```",
                    result.error_message,
                    "```",
                    ""
                ])
            
            if result.warnings:
                report_lines.extend([
                    "**Warnings:**",
                    *[f"- {warning}" for warning in result.warnings],
                    ""
                ])
        
        return "\n".join(report_lines)