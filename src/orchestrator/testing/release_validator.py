"""
Release validation system for automated quality gates and release readiness assessment.

This module provides comprehensive release validation including quality gates,
regression testing, performance validation, and automated release blocking.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple
from statistics import mean

logger = logging.getLogger(__name__)


class ReleaseType(Enum):
    """Different types of releases with varying validation requirements."""
    MAJOR = "major"          # x.0.0 - Strictest validation
    MINOR = "minor"          # x.y.0 - Standard validation  
    PATCH = "patch"          # x.y.z - Focused validation
    HOTFIX = "hotfix"        # Emergency fix - Minimal validation
    PRERELEASE = "prerelease"  # Alpha/beta - Relaxed validation


class ValidationLevel(Enum):
    """Validation strictness levels."""
    MINIMAL = "minimal"      # Basic execution tests only
    STANDARD = "standard"    # Standard quality gates
    STRICT = "strict"        # Enhanced quality gates
    CRITICAL = "critical"    # Maximum validation for major releases


@dataclass
class ReleaseValidationCriteria:
    """Validation criteria for a specific release type."""
    
    # Test execution requirements
    min_success_rate: float = 0.95
    required_test_coverage: float = 0.80  # Percentage of total pipelines
    max_execution_time_minutes: int = 120
    max_cost_dollars: float = 10.0
    
    # Quality requirements
    min_quality_score: float = 90.0
    max_critical_quality_issues: int = 0
    max_major_quality_issues: int = 2
    allow_template_artifacts: bool = False
    
    # Performance requirements
    max_performance_regression_percent: float = 10.0
    max_critical_regression_alerts: int = 0
    max_major_regression_alerts: int = 1
    
    # Specific pipeline requirements
    required_pipelines: Set[str] = field(default_factory=set)
    critical_pipelines: Set[str] = field(default_factory=set)
    performance_critical_pipelines: Set[str] = field(default_factory=set)
    
    # Temporal requirements
    min_historical_data_days: int = 7
    require_baseline_comparison: bool = True
    
    # Additional gates
    require_documentation_update: bool = False
    require_manual_approval: bool = False
    block_on_security_issues: bool = True


@dataclass
class ValidationResult:
    """Result of release validation assessment."""
    
    validation_passed: bool
    release_ready: bool
    validation_level: ValidationLevel
    overall_score: float  # 0-100
    
    # Detailed results
    execution_passed: bool
    quality_passed: bool
    performance_passed: bool
    coverage_passed: bool
    
    # Issues and recommendations
    blocking_issues: List[str] = field(default_factory=list)
    warning_issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Metrics
    success_rate: float = 0.0
    quality_score: float = 0.0
    coverage_percentage: float = 0.0
    execution_time_minutes: float = 0.0
    regression_alerts: int = 0
    
    # Artifacts
    validation_report_path: Optional[Path] = None
    detailed_results_path: Optional[Path] = None


class ReleaseValidator:
    """
    Comprehensive release validation system.
    
    Features:
    - Multi-level validation based on release type
    - Automated quality gate enforcement  
    - Performance regression detection
    - Historical trend analysis
    - Configurable validation criteria
    - Release readiness assessment
    """
    
    def __init__(self, performance_tracker=None, quality_reviewer=None):
        """Initialize release validator."""
        self.performance_tracker = performance_tracker
        self.quality_reviewer = quality_reviewer
        self._validation_criteria = self._initialize_validation_criteria()
        
    def _initialize_validation_criteria(self) -> Dict[ReleaseType, ReleaseValidationCriteria]:
        """Initialize validation criteria for different release types."""
        return {
            ReleaseType.MAJOR: ReleaseValidationCriteria(
                min_success_rate=0.98,
                required_test_coverage=0.95,
                max_execution_time_minutes=180,
                max_cost_dollars=15.0,
                min_quality_score=95.0,
                max_critical_quality_issues=0,
                max_major_quality_issues=0,
                max_performance_regression_percent=5.0,
                max_critical_regression_alerts=0,
                max_major_regression_alerts=0,
                required_pipelines={
                    "simple_data_processing", 
                    "data_processing_pipeline",
                    "control_flow_conditional",
                    "control_flow_for_loop", 
                    "research_minimal",
                    "creative_image_pipeline"
                },
                critical_pipelines={
                    "simple_data_processing",
                    "data_processing_pipeline"
                },
                min_historical_data_days=14,
                require_documentation_update=True,
                require_manual_approval=True
            ),
            
            ReleaseType.MINOR: ReleaseValidationCriteria(
                min_success_rate=0.95,
                required_test_coverage=0.85,
                max_execution_time_minutes=120,
                max_cost_dollars=10.0,
                min_quality_score=90.0,
                max_critical_quality_issues=0,
                max_major_quality_issues=2,
                max_performance_regression_percent=10.0,
                max_critical_regression_alerts=0,
                max_major_regression_alerts=1,
                required_pipelines={
                    "simple_data_processing",
                    "data_processing_pipeline", 
                    "control_flow_conditional",
                    "research_minimal"
                },
                critical_pipelines={
                    "simple_data_processing"
                },
                min_historical_data_days=7
            ),
            
            ReleaseType.PATCH: ReleaseValidationCriteria(
                min_success_rate=0.90,
                required_test_coverage=0.75,
                max_execution_time_minutes=90,
                max_cost_dollars=8.0,
                min_quality_score=85.0,
                max_critical_quality_issues=0,
                max_major_quality_issues=3,
                max_performance_regression_percent=15.0,
                max_critical_regression_alerts=0,
                max_major_regression_alerts=2,
                required_pipelines={
                    "simple_data_processing",
                    "control_flow_conditional"
                },
                min_historical_data_days=3
            ),
            
            ReleaseType.HOTFIX: ReleaseValidationCriteria(
                min_success_rate=0.85,
                required_test_coverage=0.60,
                max_execution_time_minutes=60,
                max_cost_dollars=5.0,
                min_quality_score=80.0,
                max_critical_quality_issues=1,
                max_major_quality_issues=5,
                max_performance_regression_percent=20.0,
                max_critical_regression_alerts=1,
                max_major_regression_alerts=3,
                required_pipelines={
                    "simple_data_processing"
                },
                min_historical_data_days=1,
                allow_template_artifacts=True
            ),
            
            ReleaseType.PRERELEASE: ReleaseValidationCriteria(
                min_success_rate=0.80,
                required_test_coverage=0.70,
                max_execution_time_minutes=150,
                max_cost_dollars=12.0,
                min_quality_score=75.0,
                max_critical_quality_issues=2,
                max_major_quality_issues=8,
                max_performance_regression_percent=25.0,
                max_critical_regression_alerts=2,
                max_major_regression_alerts=5,
                required_pipelines={
                    "simple_data_processing",
                    "control_flow_conditional"
                },
                min_historical_data_days=1,
                allow_template_artifacts=True
            )
        }
    
    def get_validation_criteria(self, release_type: ReleaseType) -> ReleaseValidationCriteria:
        """Get validation criteria for a specific release type."""
        return self._validation_criteria[release_type]
    
    def determine_validation_level(self, release_type: ReleaseType) -> ValidationLevel:
        """Determine appropriate validation level for release type."""
        level_mapping = {
            ReleaseType.MAJOR: ValidationLevel.CRITICAL,
            ReleaseType.MINOR: ValidationLevel.STRICT,
            ReleaseType.PATCH: ValidationLevel.STANDARD,
            ReleaseType.HOTFIX: ValidationLevel.MINIMAL,
            ReleaseType.PRERELEASE: ValidationLevel.STANDARD
        }
        return level_mapping[release_type]
    
    def validate_release_readiness(
        self,
        test_results,
        release_type: ReleaseType,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Validate release readiness based on test results and release type.
        
        Args:
            test_results: TestResults from pipeline test suite
            release_type: Type of release being validated
            additional_context: Additional validation context
            
        Returns:
            ValidationResult with detailed assessment
        """
        criteria = self.get_validation_criteria(release_type)
        validation_level = self.determine_validation_level(release_type)
        
        logger.info(f"Validating {release_type.value} release readiness")
        logger.info(f"Validation level: {validation_level.value}")
        logger.info(f"Criteria: success_rate >= {criteria.min_success_rate:.0%}, "
                   f"quality >= {criteria.min_quality_score}")
        
        # Initialize result
        result = ValidationResult(
            validation_passed=False,
            release_ready=False,
            validation_level=validation_level,
            overall_score=0.0,
            execution_passed=False,
            quality_passed=False,
            performance_passed=False,
            coverage_passed=False
        )
        
        # 1. Execution validation
        execution_result = self._validate_execution_requirements(test_results, criteria, result)
        
        # 2. Quality validation  
        quality_result = self._validate_quality_requirements(test_results, criteria, result)
        
        # 3. Performance validation
        performance_result = self._validate_performance_requirements(test_results, criteria, result)
        
        # 4. Coverage validation
        coverage_result = self._validate_coverage_requirements(test_results, criteria, result)
        
        # 5. Historical validation (if performance tracker available)
        historical_result = self._validate_historical_requirements(test_results, criteria, result)
        
        # Calculate overall score
        scores = []
        if execution_result: scores.append(execution_result)
        if quality_result: scores.append(quality_result)
        if performance_result: scores.append(performance_result) 
        if coverage_result: scores.append(coverage_result)
        if historical_result: scores.append(historical_result)
        
        result.overall_score = mean(scores) if scores else 0.0
        
        # Determine final validation status
        result.validation_passed = (
            result.execution_passed and
            result.quality_passed and 
            result.performance_passed and
            result.coverage_passed
        )
        
        # Release readiness requires higher standards
        result.release_ready = (
            result.validation_passed and
            result.overall_score >= 85.0 and
            len(result.blocking_issues) == 0
        )
        
        # Add final recommendations
        self._add_final_recommendations(result, criteria, release_type)
        
        logger.info(f"Validation result: {'PASSED' if result.validation_passed else 'FAILED'}")
        logger.info(f"Release ready: {'YES' if result.release_ready else 'NO'}")
        logger.info(f"Overall score: {result.overall_score:.1f}/100")
        
        return result
    
    def _validate_execution_requirements(
        self, 
        test_results, 
        criteria: ReleaseValidationCriteria,
        result: ValidationResult
    ) -> float:
        """Validate execution requirements."""
        score = 0.0
        
        # Success rate validation
        success_rate = test_results.success_rate / 100.0
        result.success_rate = success_rate
        
        if success_rate >= criteria.min_success_rate:
            score += 25.0
            result.execution_passed = True
        else:
            result.blocking_issues.append(
                f"Success rate {success_rate:.1%} below minimum {criteria.min_success_rate:.1%}"
            )
        
        # Execution time validation
        execution_time_minutes = test_results.total_time / 60.0
        result.execution_time_minutes = execution_time_minutes
        
        if execution_time_minutes <= criteria.max_execution_time_minutes:
            score += 25.0
        else:
            result.warning_issues.append(
                f"Execution time {execution_time_minutes:.1f}min exceeds {criteria.max_execution_time_minutes}min"
            )
            score += 15.0  # Partial credit
        
        # Cost validation
        if test_results.total_cost <= criteria.max_cost_dollars:
            score += 25.0
        else:
            result.warning_issues.append(
                f"Cost ${test_results.total_cost:.2f} exceeds ${criteria.max_cost_dollars:.2f}"
            )
            score += 15.0  # Partial credit
        
        # Required pipelines validation
        if criteria.required_pipelines:
            tested_pipelines = set(test_results.results.keys())
            missing_required = criteria.required_pipelines - tested_pipelines
            
            if missing_required:
                result.blocking_issues.append(
                    f"Missing required pipelines: {', '.join(missing_required)}"
                )
            else:
                score += 25.0
        else:
            score += 25.0  # No specific requirements
        
        return score
    
    def _validate_quality_requirements(
        self,
        test_results,
        criteria: ReleaseValidationCriteria, 
        result: ValidationResult
    ) -> float:
        """Validate quality requirements."""
        score = 0.0
        
        # Overall quality score
        quality_score = test_results.average_quality_score
        result.quality_score = quality_score
        
        if quality_score >= criteria.min_quality_score:
            score += 40.0
            result.quality_passed = True
        else:
            result.blocking_issues.append(
                f"Quality score {quality_score:.1f} below minimum {criteria.min_quality_score}"
            )
        
        # Quality issue analysis
        if hasattr(test_results, 'get_quality_issues_summary'):
            issues = test_results.get_quality_issues_summary()
            
            critical_issues = issues.get('critical_issues', 0)
            major_issues = issues.get('major_issues', 0)
            template_artifacts = issues.get('template_artifacts', 0)
            
            # Critical issues
            if critical_issues <= criteria.max_critical_quality_issues:
                score += 20.0
            else:
                result.blocking_issues.append(
                    f"Critical quality issues: {critical_issues} > {criteria.max_critical_quality_issues}"
                )
            
            # Major issues  
            if major_issues <= criteria.max_major_quality_issues:
                score += 20.0
            else:
                result.warning_issues.append(
                    f"Major quality issues: {major_issues} > {criteria.max_major_quality_issues}"
                )
                score += 10.0  # Partial credit
            
            # Template artifacts
            if template_artifacts == 0 or criteria.allow_template_artifacts:
                score += 20.0
            else:
                result.blocking_issues.append(
                    f"Template artifacts detected in {template_artifacts} pipelines"
                )
        else:
            score += 60.0  # No detailed analysis available
        
        return score
    
    def _validate_performance_requirements(
        self,
        test_results,
        criteria: ReleaseValidationCriteria,
        result: ValidationResult
    ) -> float:
        """Validate performance requirements.""" 
        score = 50.0  # Default if no performance data
        
        if not self.performance_tracker:
            result.warning_issues.append("No performance tracker available for regression analysis")
            return score
            
        try:
            # Get regression alerts if available
            regression_alerts = []
            if hasattr(test_results, 'get_regression_alerts'):
                regression_alerts = test_results.get_regression_alerts()
            
            critical_alerts = len([a for a in regression_alerts if a.severity == "CRITICAL"])
            major_alerts = len([a for a in regression_alerts if a.severity in ["HIGH", "MEDIUM"]])
            
            result.regression_alerts = len(regression_alerts)
            
            # Critical regression validation
            if critical_alerts <= criteria.max_critical_regression_alerts:
                score += 25.0
                if critical_alerts == 0:
                    result.performance_passed = True
            else:
                result.blocking_issues.append(
                    f"Critical performance regressions: {critical_alerts} > {criteria.max_critical_regression_alerts}"
                )
            
            # Major regression validation
            if major_alerts <= criteria.max_major_regression_alerts:
                score += 25.0
            else:
                result.warning_issues.append(
                    f"Major performance regressions: {major_alerts} > {criteria.max_major_regression_alerts}"
                )
                score += 15.0  # Partial credit
            
            # Performance critical pipelines
            if criteria.performance_critical_pipelines:
                critical_pipeline_alerts = [
                    a for a in regression_alerts
                    if a.pipeline_name in criteria.performance_critical_pipelines and
                    a.severity in ["CRITICAL", "HIGH"]
                ]
                
                if critical_pipeline_alerts:
                    result.blocking_issues.append(
                        f"Performance regressions in critical pipelines: "
                        f"{', '.join(a.pipeline_name for a in critical_pipeline_alerts)}"
                    )
                else:
                    score += 25.0
            else:
                score += 25.0  # No critical pipelines specified
        
        except Exception as e:
            logger.warning(f"Performance validation error: {e}")
            result.warning_issues.append(f"Performance validation incomplete: {e}")
            score = 50.0
        
        return score
    
    def _validate_coverage_requirements(
        self,
        test_results,
        criteria: ReleaseValidationCriteria,
        result: ValidationResult
    ) -> float:
        """Validate test coverage requirements."""
        score = 0.0
        
        # This would ideally integrate with pipeline discovery to get total available pipelines
        # For now, we'll estimate based on test results
        tested_pipelines = len(test_results.results)
        
        # Rough estimate of total pipelines (could be made more accurate with discovery integration)
        estimated_total_pipelines = max(50, int(tested_pipelines / 0.7))  # Assume we test ~70% normally
        coverage_percentage = (tested_pipelines / estimated_total_pipelines) * 100
        result.coverage_percentage = coverage_percentage / 100.0
        
        required_coverage_percentage = criteria.required_test_coverage * 100
        
        if coverage_percentage >= required_coverage_percentage:
            score = 100.0
            result.coverage_passed = True
        else:
            result.warning_issues.append(
                f"Test coverage {coverage_percentage:.1f}% below required {required_coverage_percentage:.1f}%"
            )
            # Partial credit based on how close we are
            score = max(0, (coverage_percentage / required_coverage_percentage) * 100)
        
        return score
    
    def _validate_historical_requirements(
        self,
        test_results,
        criteria: ReleaseValidationCriteria, 
        result: ValidationResult
    ) -> Optional[float]:
        """Validate historical data requirements."""
        if not self.performance_tracker:
            return None
            
        score = 100.0  # Default to pass if no specific requirements
        
        if criteria.require_baseline_comparison:
            try:
                # Check if we have sufficient historical data
                cutoff_date = datetime.now() - timedelta(days=criteria.min_historical_data_days)
                
                # This would need to be implemented in performance tracker
                # For now, assume we have sufficient data if performance tracker exists
                result.recommendations.append(
                    f"Historical data validation requires {criteria.min_historical_data_days} days of data"
                )
                
            except Exception as e:
                logger.warning(f"Historical validation error: {e}")
                result.warning_issues.append(f"Historical validation incomplete: {e}")
                score = 75.0
        
        return score
    
    def _add_final_recommendations(
        self,
        result: ValidationResult,
        criteria: ReleaseValidationCriteria,
        release_type: ReleaseType
    ):
        """Add final recommendations based on validation results."""
        if not result.validation_passed:
            result.recommendations.append("Address all blocking issues before proceeding with release")
        
        if result.warning_issues:
            result.recommendations.append("Review and address warning issues to improve release quality")
        
        if result.overall_score < 90 and release_type in [ReleaseType.MAJOR, ReleaseType.MINOR]:
            result.recommendations.append("Consider improving test quality before major/minor release")
        
        if criteria.require_manual_approval and result.validation_passed:
            result.recommendations.append("Manual approval required before release")
        
        if criteria.require_documentation_update:
            result.recommendations.append("Ensure documentation is updated for this release")
        
        # Performance-specific recommendations
        if result.regression_alerts > 0:
            result.recommendations.append("Review performance regression alerts and optimize if needed")
        
        # Quality-specific recommendations
        if result.quality_score < criteria.min_quality_score + 5:  # Close to threshold
            result.recommendations.append("Quality score is close to minimum - consider additional validation")


def determine_release_type_from_version(version: str) -> ReleaseType:
    """Determine release type from version string (e.g., '1.2.3')."""
    if "alpha" in version.lower() or "beta" in version.lower() or "rc" in version.lower():
        return ReleaseType.PRERELEASE
    
    try:
        parts = version.split('.')
        if len(parts) >= 3:
            major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
            
            if patch > 0 and "hotfix" in version.lower():
                return ReleaseType.HOTFIX
            elif patch > 0:
                return ReleaseType.PATCH
            elif minor > 0:
                return ReleaseType.MINOR
            else:
                return ReleaseType.MAJOR
    except (ValueError, IndexError):
        pass
    
    return ReleaseType.MINOR  # Default fallback


def create_release_validation_report(
    validation_result: ValidationResult,
    release_version: str,
    output_path: Path
) -> Path:
    """Create detailed release validation report."""
    content = [
        f"# Release Validation Report",
        f"",
        f"**Version:** {release_version}",
        f"**Validation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Validation Level:** {validation_result.validation_level.value}",
        f"**Overall Score:** {validation_result.overall_score:.1f}/100",
        f"",
        f"## Validation Summary",
        f"",
        f"| Component | Status | Score |",
        f"|-----------|--------|-------|",
        f"| Execution | {'✅ PASS' if validation_result.execution_passed else '❌ FAIL'} | - |",
        f"| Quality | {'✅ PASS' if validation_result.quality_passed else '❌ FAIL'} | {validation_result.quality_score:.1f} |",
        f"| Performance | {'✅ PASS' if validation_result.performance_passed else '❌ FAIL'} | - |",
        f"| Coverage | {'✅ PASS' if validation_result.coverage_passed else '❌ FAIL'} | {validation_result.coverage_percentage:.1%} |",
        f"",
        f"**Final Result:** {'🎉 RELEASE READY' if validation_result.release_ready else '🚫 NOT READY FOR RELEASE'}",
        f""
    ]
    
    # Blocking issues
    if validation_result.blocking_issues:
        content.extend([
            "## 🚨 Blocking Issues",
            ""
        ])
        for issue in validation_result.blocking_issues:
            content.append(f"- ❌ {issue}")
        content.append("")
    
    # Warning issues
    if validation_result.warning_issues:
        content.extend([
            "## ⚠️ Warning Issues", 
            ""
        ])
        for issue in validation_result.warning_issues:
            content.append(f"- ⚠️ {issue}")
        content.append("")
    
    # Recommendations
    if validation_result.recommendations:
        content.extend([
            "## 💡 Recommendations",
            ""
        ])
        for rec in validation_result.recommendations:
            content.append(f"- 💡 {rec}")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(content))
        
    return output_path


if __name__ == "__main__":
    # Example usage
    print("Release Validator - Example Usage")
    print("=" * 40)
    
    validator = ReleaseValidator()
    
    for release_type in ReleaseType:
        criteria = validator.get_validation_criteria(release_type)
        level = validator.determine_validation_level(release_type)
        print(f"\n{release_type.value.upper()} Release:")
        print(f"  Validation Level: {level.value}")
        print(f"  Min Success Rate: {criteria.min_success_rate:.0%}")
        print(f"  Min Quality Score: {criteria.min_quality_score}")
        print(f"  Required Pipelines: {len(criteria.required_pipelines)}")