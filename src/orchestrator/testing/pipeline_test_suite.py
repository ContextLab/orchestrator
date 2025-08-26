"""Core pipeline testing suite with comprehensive execution and validation capabilities."""

import asyncio
import json
import logging
import os
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from orchestrator import Orchestrator, init_models
from orchestrator.models.model_registry import ModelRegistry
from orchestrator.utils.api_keys_flexible import load_api_keys_optional

from .pipeline_discovery import PipelineDiscovery, PipelineInfo
from .quality_validator import QualityValidator, QualityValidationResult
from .template_validator import TemplateValidator, TemplateValidationResult

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of a single pipeline execution."""
    
    success: bool
    execution_time: float
    outputs: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Exception] = None
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    
    # Performance metrics
    api_calls_count: int = 0
    tokens_used: int = 0
    estimated_cost: float = 0.0
    memory_usage_mb: float = 0.0


@dataclass
class TemplateResult:
    """Result of template resolution validation."""
    
    resolved_correctly: bool
    issues: List[str] = field(default_factory=list)
    unresolved_templates: List[str] = field(default_factory=list)


@dataclass
class OrganizationResult:
    """Result of file organization validation."""
    
    valid: bool
    issues: List[str] = field(default_factory=list)
    output_files_found: List[str] = field(default_factory=list)
    expected_output_dir: Optional[str] = None


@dataclass
class PerformanceResult:
    """Result of performance monitoring."""
    
    metrics: Dict[str, float] = field(default_factory=dict)
    regression_detected: bool = False
    historical_comparison: Dict[str, float] = field(default_factory=dict)
    performance_score: float = 1.0


@dataclass
class PipelineTestResult:
    """Comprehensive result of pipeline testing."""
    
    pipeline_name: str
    execution: ExecutionResult
    templates: TemplateResult
    organization: OrganizationResult
    performance: PerformanceResult
    
    # Quality validation results (Stream B integration)
    quality_validation: Optional[QualityValidationResult] = None
    template_validation: Optional[TemplateValidationResult] = None
    
    # Overall assessment
    overall_success: bool = False
    quality_score: float = 0.0
    test_duration: float = 0.0
    warnings: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Calculate overall assessment after initialization."""
        # Enhanced success criteria including quality validation
        success_criteria = [
            self.execution.success,
            self.templates.resolved_correctly,
            self.organization.valid
        ]
        
        # Add quality validation criteria if available
        if self.quality_validation:
            success_criteria.extend([
                self.quality_validation.production_ready,
                not self.quality_validation.has_critical_issues
            ])
        
        if self.template_validation:
            success_criteria.extend([
                self.template_validation.all_templates_resolved,
                not self.template_validation.has_critical_issues
            ])
        
        self.overall_success = all(success_criteria)
        
        # Calculate enhanced quality score (0-100) with quality validation
        score = 0.0
        
        # Core execution components (60 points)
        if self.execution.success:
            score += 25
        if self.templates.resolved_correctly:
            score += 20
        if self.organization.valid:
            score += 15
        
        # Quality validation components (30 points)
        if self.quality_validation:
            # Scale LLM quality score to 20 points max
            score += (self.quality_validation.overall_score / 100.0) * 20
        else:
            # Fallback scoring without quality validation
            score += 15
        
        # Template validation component (10 points)
        if self.template_validation:
            score += (self.template_validation.template_score / 100.0) * 10
        else:
            # Fallback scoring without template validation
            score += 5
        
        # Performance bonus (up to 10 points)
        if self.performance.performance_score > 0.8:
            score += 10
        elif self.performance.performance_score > 0.6:
            score += 5
        
        self.quality_score = min(100.0, score)


@dataclass
class TestResults:
    """Collection of pipeline test results."""
    
    results: Dict[str, PipelineTestResult] = field(default_factory=dict)
    total_tests: int = 0
    successful_tests: int = 0
    failed_tests: int = 0
    total_time: float = 0.0
    total_cost: float = 0.0
    
    def __post_init__(self):
        """Calculate summary statistics."""
        self.total_tests = len(self.results)
        self.successful_tests = sum(1 for r in self.results.values() if r.overall_success)
        self.failed_tests = self.total_tests - self.successful_tests
        self.total_time = sum(r.test_duration for r in self.results.values())
        self.total_cost = sum(r.execution.estimated_cost for r in self.results.values())
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_tests == 0:
            return 0.0
        return (self.successful_tests / self.total_tests) * 100
    
    @property
    def average_quality_score(self) -> float:
        """Calculate average quality score."""
        if not self.results:
            return 0.0
        return sum(r.quality_score for r in self.results.values()) / len(self.results)
    
    def get_failed_pipelines(self) -> List[str]:
        """Get list of failed pipeline names."""
        return [name for name, result in self.results.items() 
                if not result.overall_success]
    
    def get_high_quality_pipelines(self, threshold: float = 80.0) -> List[str]:
        """Get pipelines with quality score above threshold."""
        return [name for name, result in self.results.items()
                if result.quality_score >= threshold]
    
    def get_production_ready_pipelines(self) -> List[str]:
        """Get pipelines that are production ready based on quality validation."""
        return [name for name, result in self.results.items()
                if (result.quality_validation and result.quality_validation.production_ready) or
                   (not result.quality_validation and result.overall_success and result.quality_score >= 85.0)]
    
    def get_quality_issues_summary(self) -> Dict[str, int]:
        """Get summary of quality issues found across all pipelines."""
        summary = {
            'critical_issues': 0,
            'major_issues': 0,
            'minor_issues': 0,
            'template_artifacts': 0,
            'content_quality_issues': 0
        }
        
        for result in self.results.values():
            if result.quality_validation:
                summary['critical_issues'] += len(result.quality_validation.critical_issues)
                summary['major_issues'] += len(result.quality_validation.major_issues)
                summary['minor_issues'] += len(result.quality_validation.minor_issues)
                if result.quality_validation.template_artifacts_found:
                    summary['template_artifacts'] += 1
            
            if result.template_validation:
                summary['critical_issues'] += len(result.template_validation.critical_issues)
                summary['major_issues'] += len(result.template_validation.major_issues)
                summary['minor_issues'] += len(result.template_validation.minor_issues)
        
        return summary


class PipelineTestSuite:
    """
    Core pipeline testing suite with comprehensive validation.
    
    Provides automated testing for all example pipelines with:
    - Execution validation
    - Template resolution checking  
    - File organization validation
    - Performance monitoring
    - Quality scoring
    """
    
    def __init__(self, 
                 examples_dir: Optional[Path] = None,
                 model_registry: Optional[ModelRegistry] = None,
                 orchestrator: Optional[Orchestrator] = None,
                 enable_llm_quality_review: bool = True,
                 enable_enhanced_template_validation: bool = True,
                 quality_threshold: float = 85.0):
        """
        Initialize pipeline test suite.
        
        Args:
            examples_dir: Directory containing example pipelines
            model_registry: Model registry for testing
            orchestrator: Orchestrator instance for execution
            enable_llm_quality_review: Enable LLM-powered quality assessment
            enable_enhanced_template_validation: Enable advanced template validation
            quality_threshold: Minimum quality score for production readiness
        """
        self.examples_dir = examples_dir or Path("examples")
        self.model_registry = model_registry or init_models()
        self.orchestrator = orchestrator or Orchestrator(model_registry=self.model_registry)
        
        # Initialize pipeline discovery
        self.discovery = PipelineDiscovery(self.examples_dir)
        self.discovered_pipelines: Dict[str, PipelineInfo] = {}
        
        # Test configuration
        self.timeout_seconds = 300  # 5 minutes per pipeline
        self.max_cost_per_pipeline = 1.0  # $1.00 per pipeline
        self.enable_performance_tracking = True
        
        # Quality validation configuration (Stream B)
        self.enable_llm_quality_review = enable_llm_quality_review
        self.enable_enhanced_template_validation = enable_enhanced_template_validation
        self.quality_threshold = quality_threshold
        
        # Initialize quality validation components
        self.quality_validator: Optional[QualityValidator] = None
        self.template_validator: Optional[TemplateValidator] = None
        
        if self.enable_llm_quality_review:
            try:
                self.quality_validator = QualityValidator(
                    quality_threshold=self.quality_threshold
                )
                logger.info("Initialized LLM quality validator")
            except Exception as e:
                logger.warning(f"Failed to initialize quality validator: {e}")
                self.enable_llm_quality_review = False
        
        if self.enable_enhanced_template_validation:
            try:
                self.template_validator = TemplateValidator()
                logger.info("Initialized enhanced template validator")
            except Exception as e:
                logger.warning(f"Failed to initialize template validator: {e}")
                self.enable_enhanced_template_validation = False
        
        # Results tracking
        self.execution_history: List[PipelineTestResult] = []
        self.performance_baselines: Dict[str, Dict[str, float]] = {}
        
        logger.info(f"Initialized PipelineTestSuite (Quality: {self.enable_llm_quality_review}, "
                   f"Templates: {self.enable_enhanced_template_validation})")
    
    def discover_pipelines(self) -> Dict[str, PipelineInfo]:
        """
        Discover all available pipelines.
        
        Returns:
            Dict[str, PipelineInfo]: Discovered pipeline information
        """
        logger.info("Discovering pipelines...")
        self.discovered_pipelines = self.discovery.discover_all_pipelines()
        
        logger.info(f"Discovered {len(self.discovered_pipelines)} pipelines:")
        for name, info in self.discovered_pipelines.items():
            logger.info(f"  - {name} ({info.category}, {info.complexity})")
        
        return self.discovered_pipelines
    
    async def run_pipeline_tests(self, 
                                pipeline_list: Optional[List[str]] = None,
                                test_mode: str = "full") -> TestResults:
        """
        Run comprehensive pipeline test suite.
        
        Args:
            pipeline_list: Specific pipelines to test (None for all)
            test_mode: Test mode (full, core, quick, single)
            
        Returns:
            TestResults: Comprehensive test results
        """
        logger.info(f"Running pipeline tests (mode: {test_mode})")
        start_time = time.time()
        
        # Discover pipelines if not already done
        if not self.discovered_pipelines:
            self.discover_pipelines()
        
        # Determine which pipelines to test
        if pipeline_list:
            pipelines_to_test = [self.discovered_pipelines[name] for name in pipeline_list
                               if name in self.discovered_pipelines]
        else:
            pipelines_to_test = self._get_pipelines_for_mode(test_mode)
        
        logger.info(f"Testing {len(pipelines_to_test)} pipelines")
        
        # Run tests
        test_results = TestResults()
        
        for pipeline_info in pipelines_to_test:
            try:
                logger.info(f"Testing pipeline: {pipeline_info.name}")
                result = await self._test_pipeline_comprehensive(pipeline_info)
                test_results.results[pipeline_info.name] = result
                
                # Log result
                status = "PASS" if result.overall_success else "FAIL"
                logger.info(f"  {status}: {pipeline_info.name} "
                          f"(score: {result.quality_score:.1f}, "
                          f"time: {result.execution.execution_time:.1f}s)")
                
            except Exception as e:
                logger.error(f"Failed to test pipeline {pipeline_info.name}: {e}")
                # Create failed result
                failed_result = PipelineTestResult(
                    pipeline_name=pipeline_info.name,
                    execution=ExecutionResult(success=False, execution_time=0.0, 
                                            error=e, error_message=str(e)),
                    templates=TemplateResult(resolved_correctly=False),
                    organization=OrganizationResult(valid=False),
                    performance=PerformanceResult()
                )
                test_results.results[pipeline_info.name] = failed_result
        
        # Update results summary
        test_results.__post_init__()
        
        total_time = time.time() - start_time
        logger.info(f"Pipeline testing completed in {total_time:.1f}s")
        logger.info(f"Results: {test_results.successful_tests}/{test_results.total_tests} "
                   f"passed ({test_results.success_rate:.1f}%)")
        
        return test_results
    
    async def _test_pipeline_comprehensive(self, pipeline_info: PipelineInfo) -> PipelineTestResult:
        """
        Run comprehensive testing on a single pipeline.
        
        Args:
            pipeline_info: Pipeline information
            
        Returns:
            PipelineTestResult: Comprehensive test result
        """
        test_start = time.time()
        
        # 1. Execution Testing
        execution_result = await self._test_pipeline_execution(pipeline_info)
        
        # 2. Template Resolution Validation (Basic)
        template_result = self._test_template_resolution(pipeline_info)
        
        # 3. File Organization Validation
        organization_result = self._test_file_organization(pipeline_info)
        
        # 4. Performance Monitoring
        performance_result = self._test_performance_metrics(pipeline_info, execution_result)
        
        # 5. Quality Validation (Stream B) - LLM-powered assessment
        quality_validation_result = None
        if self.enable_llm_quality_review and self.quality_validator:
            try:
                quality_validation_result = await self.quality_validator.validate_pipeline_quality(
                    pipeline_info.name
                )
            except Exception as e:
                logger.warning(f"Quality validation failed for {pipeline_info.name}: {e}")
        
        # 6. Enhanced Template Validation (Stream B) - Advanced template checking
        template_validation_result = None
        if self.enable_enhanced_template_validation and self.template_validator:
            try:
                output_dir = self.examples_dir / "outputs" / pipeline_info.name
                template_validation_result = self.template_validator.validate_pipeline_templates(
                    pipeline_info.path, 
                    output_directory=output_dir if output_dir.exists() else None
                )
            except Exception as e:
                logger.warning(f"Template validation failed for {pipeline_info.name}: {e}")
        
        # Create comprehensive result
        result = PipelineTestResult(
            pipeline_name=pipeline_info.name,
            execution=execution_result,
            templates=template_result,
            organization=organization_result,
            performance=performance_result,
            quality_validation=quality_validation_result,
            template_validation=template_validation_result,
            test_duration=time.time() - test_start
        )
        
        # Add to execution history
        self.execution_history.append(result)
        
        return result
    
    async def _test_pipeline_execution(self, pipeline_info: PipelineInfo) -> ExecutionResult:
        """
        Test pipeline execution without errors.
        
        Args:
            pipeline_info: Pipeline information
            
        Returns:
            ExecutionResult: Execution result
        """
        start_time = time.time()
        
        try:
            # Get test inputs for this pipeline
            inputs = self._get_test_inputs_for_pipeline(pipeline_info)
            
            # Create temporary output directory
            with tempfile.TemporaryDirectory(prefix=f"test_{pipeline_info.name}_") as temp_dir:
                output_dir = Path(temp_dir)
                
                # Add output path to inputs
                context = inputs.copy()
                context["output_path"] = str(output_dir)
                
                # Execute pipeline with timeout
                execution_task = asyncio.create_task(
                    self.orchestrator.execute_yaml_file(str(pipeline_info.path), context=context)
                )
                
                outputs = await asyncio.wait_for(execution_task, timeout=self.timeout_seconds)
                
                execution_time = time.time() - start_time
                
                return ExecutionResult(
                    success=True,
                    execution_time=execution_time,
                    outputs=outputs or {},
                    api_calls_count=self._extract_api_calls_count(outputs),
                    tokens_used=self._extract_tokens_used(outputs),
                    estimated_cost=self._estimate_cost(outputs),
                    memory_usage_mb=self._extract_memory_usage(outputs)
                )
                
        except asyncio.TimeoutError:
            error = TimeoutError(f"Pipeline execution timed out after {self.timeout_seconds}s")
            return ExecutionResult(
                success=False,
                execution_time=time.time() - start_time,
                error=error,
                error_message=str(error)
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                execution_time=time.time() - start_time,
                error=e,
                error_message=str(e),
                error_traceback=traceback.format_exc()
            )
    
    def _test_template_resolution(self, pipeline_info: PipelineInfo) -> TemplateResult:
        """
        Test template resolution in pipeline outputs.
        
        Args:
            pipeline_info: Pipeline information
            
        Returns:
            TemplateResult: Template validation result
        """
        issues = []
        unresolved_templates = []
        
        # Check pipeline YAML for template syntax
        try:
            with open(pipeline_info.path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Basic template syntax validation
            open_braces = content.count('{{')
            close_braces = content.count('}}')
            
            if open_braces != close_braces:
                issues.append(f"Unbalanced template braces: {open_braces} {{ vs {close_braces} }}")
            
            # Check for common template patterns that might not resolve
            import re
            template_pattern = r'\{\{([^}]+)\}\}'
            templates = re.findall(template_pattern, content)
            
            for template in templates:
                template_clean = template.strip()
                # Check for potentially problematic templates
                if '.' not in template_clean and '|' not in template_clean and template_clean not in ['item', 'index']:
                    if template_clean not in ['topic', 'model', 'theme', 'count', 'input_file']:
                        unresolved_templates.append(template_clean)
            
            # Check output directory for template artifacts (if it exists)
            output_dir = self.examples_dir / "outputs" / pipeline_info.name
            if output_dir.exists():
                template_artifacts = self._check_output_files_for_templates(output_dir)
                issues.extend(template_artifacts)
            
        except Exception as e:
            issues.append(f"Failed to analyze templates: {e}")
        
        return TemplateResult(
            resolved_correctly=len(issues) == 0,
            issues=issues,
            unresolved_templates=unresolved_templates
        )
    
    def _test_file_organization(self, pipeline_info: PipelineInfo) -> OrganizationResult:
        """
        Test file organization and naming conventions.
        
        Args:
            pipeline_info: Pipeline information
            
        Returns:
            OrganizationResult: Organization validation result
        """
        issues = []
        output_files = []
        expected_output_dir = f"examples/outputs/{pipeline_info.name}/"
        
        # Check if output directory exists
        output_dir = Path(expected_output_dir)
        
        if not output_dir.exists():
            # For new pipelines, this might be expected
            issues.append(f"Output directory does not exist: {expected_output_dir}")
        else:
            # Check files in output directory
            try:
                for file_path in output_dir.rglob("*"):
                    if file_path.is_file():
                        file_name = file_path.name
                        output_files.append(str(file_path))
                        
                        # Check for generic/poor naming conventions
                        generic_names = ['output', 'result', 'temp', 'test', 'file']
                        if any(generic in file_name.lower() for generic in generic_names):
                            if not file_name.startswith(('validation_', 'archive/')):  # Allow some generic names
                                issues.append(f"Generic filename: {file_name}")
                        
                        # Check for timestamp-based naming (which can be okay but should be documented)
                        import re
                        if re.search(r'\d{4}-\d{2}-\d{2}', file_name) or re.search(r'\d{13,}', file_name):
                            # This is often okay for output files, just note it
                            pass
            
            except Exception as e:
                issues.append(f"Failed to analyze output directory: {e}")
        
        return OrganizationResult(
            valid=len(issues) == 0,
            issues=issues,
            output_files_found=output_files,
            expected_output_dir=expected_output_dir
        )
    
    def _test_performance_metrics(self, 
                                pipeline_info: PipelineInfo,
                                execution_result: ExecutionResult) -> PerformanceResult:
        """
        Test and monitor performance metrics.
        
        Args:
            pipeline_info: Pipeline information
            execution_result: Execution result with performance data
            
        Returns:
            PerformanceResult: Performance monitoring result
        """
        metrics = {
            'execution_time': execution_result.execution_time,
            'estimated_cost': execution_result.estimated_cost,
            'api_calls': execution_result.api_calls_count,
            'tokens_used': execution_result.tokens_used,
            'memory_usage_mb': execution_result.memory_usage_mb
        }
        
        # Get historical data for regression detection
        historical = self.performance_baselines.get(pipeline_info.name, {})
        
        # Detect regression (simple threshold-based)
        regression_detected = False
        if historical and execution_result.success:
            # Check if execution time increased significantly
            if 'execution_time' in historical:
                time_increase = (metrics['execution_time'] - historical['execution_time']) / historical['execution_time']
                if time_increase > 0.5:  # 50% increase threshold
                    regression_detected = True
            
            # Check if cost increased significantly
            if 'estimated_cost' in historical and historical['estimated_cost'] > 0:
                cost_increase = (metrics['estimated_cost'] - historical['estimated_cost']) / historical['estimated_cost']
                if cost_increase > 0.3:  # 30% increase threshold
                    regression_detected = True
        
        # Calculate performance score
        performance_score = self._calculate_performance_score(metrics, pipeline_info)
        
        # Update baseline if this execution was successful
        if execution_result.success:
            self.performance_baselines[pipeline_info.name] = metrics.copy()
        
        return PerformanceResult(
            metrics=metrics,
            regression_detected=regression_detected,
            historical_comparison=historical,
            performance_score=performance_score
        )
    
    def _get_pipelines_for_mode(self, mode: str) -> List[PipelineInfo]:
        """
        Get pipelines to test based on test mode.
        
        Args:
            mode: Test mode (full, core, quick)
            
        Returns:
            List[PipelineInfo]: Pipelines to test
        """
        if mode == "full":
            return list(self.discovered_pipelines.values())
        elif mode == "core":
            return self.discovery.get_core_test_pipelines()
        elif mode == "quick":
            return self.discovery.get_quick_test_pipelines()
        else:
            # Default to test-safe pipelines
            return self.discovery.get_test_safe_pipelines()
    
    def _get_test_inputs_for_pipeline(self, pipeline_info: PipelineInfo) -> Dict[str, Any]:
        """
        Get appropriate test inputs for pipeline.
        
        Args:
            pipeline_info: Pipeline information
            
        Returns:
            Dict[str, Any]: Test input parameters
        """
        # Use inputs from pipeline discovery
        inputs = pipeline_info.input_requirements.copy()
        
        # Add common defaults
        defaults = {
            "model": "anthropic:claude-sonnet-4-20250514",  # Use available fast model
            "timeout": 180,  # 3 minutes default timeout
            "max_cost": 0.50  # $0.50 max cost per pipeline
        }
        
        # Merge defaults with specific inputs
        for key, value in defaults.items():
            if key not in inputs:
                inputs[key] = value
        
        return inputs
    
    def _extract_api_calls_count(self, outputs: Dict[str, Any]) -> int:
        """Extract API calls count from outputs."""
        if isinstance(outputs, dict):
            metadata = outputs.get('execution_metadata', {})
            if isinstance(metadata, dict):
                return metadata.get('api_calls', 0)
        return 0
    
    def _extract_tokens_used(self, outputs: Dict[str, Any]) -> int:
        """Extract tokens used from outputs."""
        if isinstance(outputs, dict):
            metadata = outputs.get('execution_metadata', {})
            if isinstance(metadata, dict):
                return metadata.get('tokens_used', 0)
        return 0
    
    def _estimate_cost(self, outputs: Dict[str, Any]) -> float:
        """Estimate execution cost."""
        if isinstance(outputs, dict):
            metadata = outputs.get('execution_metadata', {})
            if isinstance(metadata, dict):
                return metadata.get('total_cost', 0.0)
        return 0.001  # Minimal default cost
    
    def _extract_memory_usage(self, outputs: Dict[str, Any]) -> float:
        """Extract memory usage from outputs."""
        if isinstance(outputs, dict):
            metadata = outputs.get('execution_metadata', {})
            if isinstance(metadata, dict):
                return metadata.get('memory_peak_mb', 0.0)
        return 0.0
    
    def _check_output_files_for_templates(self, output_dir: Path) -> List[str]:
        """Check output files for unresolved template artifacts."""
        issues = []
        
        try:
            for file_path in output_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix in ['.md', '.txt', '.json', '.yaml', '.html']:
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        # Check for unresolved templates
                        if '{{' in content and '}}' in content:
                            import re
                            unresolved = re.findall(r'\{\{[^}]+\}\}', content)
                            if unresolved:
                                issues.append(f"Unresolved templates in {file_path.name}: {unresolved[:3]}")
                    
                    except Exception:
                        # Skip files that can't be read
                        pass
        
        except Exception as e:
            issues.append(f"Failed to check output files: {e}")
        
        return issues
    
    def _calculate_performance_score(self, metrics: Dict[str, float], 
                                   pipeline_info: PipelineInfo) -> float:
        """
        Calculate performance score (0.0-1.0).
        
        Args:
            metrics: Performance metrics
            pipeline_info: Pipeline information
            
        Returns:
            float: Performance score
        """
        score = 1.0
        
        # Penalize long execution times
        exec_time = metrics.get('execution_time', 0)
        if exec_time > pipeline_info.estimated_runtime * 2:
            score -= 0.3
        elif exec_time > pipeline_info.estimated_runtime * 1.5:
            score -= 0.1
        
        # Penalize high costs
        cost = metrics.get('estimated_cost', 0)
        if cost > 0.50:  # Over $0.50
            score -= 0.3
        elif cost > 0.25:  # Over $0.25
            score -= 0.1
        
        # Penalize high memory usage
        memory = metrics.get('memory_usage_mb', 0)
        if memory > 1000:  # Over 1GB
            score -= 0.2
        elif memory > 500:  # Over 500MB
            score -= 0.1
        
        return max(0.0, score)