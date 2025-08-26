"""Quality validation integration for pipeline testing.

Integrates LLM-powered quality assessment from Issue #277 into the pipeline
testing infrastructure, providing sophisticated content quality evaluation.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Import quality reviewer from Issue #277 
from orchestrator.core.llm_quality_reviewer import LLMQualityReviewer, LLMQualityError
from orchestrator.core.quality_assessment import (
    PipelineQualityReview, QualityIssue, IssueSeverity, IssueCategory
)

logger = logging.getLogger(__name__)


@dataclass 
class QualityValidationResult:
    """Result of quality validation for a pipeline."""
    
    pipeline_name: str
    overall_score: float  # 0-100
    production_ready: bool
    
    # Issue categorization
    critical_issues: List[QualityIssue] = field(default_factory=list)
    major_issues: List[QualityIssue] = field(default_factory=list) 
    minor_issues: List[QualityIssue] = field(default_factory=list)
    
    # Quality assessment details
    template_artifacts_found: bool = False
    content_quality_score: float = 0.0
    visual_quality_score: float = 0.0
    completeness_score: float = 0.0
    
    # Files and processing info
    files_reviewed: List[str] = field(default_factory=list)
    review_duration: float = 0.0
    reviewer_model: str = ""
    recommendations: List[str] = field(default_factory=list)
    
    # Error handling
    validation_errors: List[str] = field(default_factory=list)
    
    @property
    def total_issues(self) -> int:
        """Total number of quality issues found."""
        return len(self.critical_issues) + len(self.major_issues) + len(self.minor_issues)
    
    @property
    def has_critical_issues(self) -> bool:
        """Whether critical issues were found."""
        return len(self.critical_issues) > 0
    
    @property
    def quality_grade(self) -> str:
        """Letter grade based on quality score."""
        if self.overall_score >= 90:
            return "A"
        elif self.overall_score >= 80:
            return "B"
        elif self.overall_score >= 70:
            return "C"
        elif self.overall_score >= 60:
            return "D"
        else:
            return "F"


class QualityValidator:
    """
    Quality validation integration for pipeline testing.
    
    Integrates LLM quality review system with pipeline testing infrastructure
    to provide sophisticated content quality assessment and validation.
    """
    
    # Quality thresholds
    PRODUCTION_READY_THRESHOLD = 85.0
    ACCEPTABLE_THRESHOLD = 70.0
    CRITICAL_ISSUE_FAIL_THRESHOLD = 0  # Any critical issues = fail
    
    def __init__(self, 
                 quality_reviewer: Optional[LLMQualityReviewer] = None,
                 enable_llm_review: bool = True,
                 enable_visual_review: bool = True,
                 quality_threshold: float = 85.0):
        """
        Initialize quality validator.
        
        Args:
            quality_reviewer: LLM quality reviewer instance
            enable_llm_review: Whether to use LLM-based quality assessment
            enable_visual_review: Whether to assess visual content
            quality_threshold: Minimum quality score for production readiness
        """
        self.quality_reviewer = quality_reviewer
        self.enable_llm_review = enable_llm_review
        self.enable_visual_review = enable_visual_review
        self.quality_threshold = max(quality_threshold, self.PRODUCTION_READY_THRESHOLD)
        
        # Initialize LLM reviewer if enabled
        if self.enable_llm_review and not self.quality_reviewer:
            try:
                self.quality_reviewer = LLMQualityReviewer()
                logger.info("Initialized LLM quality reviewer for validation")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM quality reviewer: {e}")
                self.enable_llm_review = False
        
        # Quality assessment configuration
        self.max_review_time = 300.0  # 5 minutes max per pipeline
        self.fallback_to_rules = True  # Use rule-based assessment if LLM fails
        
        logger.info(f"Quality validator initialized (LLM: {self.enable_llm_review}, "
                   f"Visual: {self.enable_visual_review}, Threshold: {self.quality_threshold})")
    
    async def validate_pipeline_quality(self, 
                                      pipeline_name: str,
                                      output_directory: Optional[Path] = None) -> QualityValidationResult:
        """
        Validate quality of pipeline outputs.
        
        Args:
            pipeline_name: Name of the pipeline
            output_directory: Directory containing pipeline outputs
            
        Returns:
            QualityValidationResult: Comprehensive quality assessment
        """
        start_time = time.time()
        logger.info(f"Starting quality validation for pipeline: {pipeline_name}")
        
        # Determine output directory
        if not output_directory:
            output_directory = Path("examples/outputs") / pipeline_name
        
        # Initialize result
        result = QualityValidationResult(
            pipeline_name=pipeline_name,
            overall_score=0.0,
            production_ready=False
        )
        
        try:
            # Check if output directory exists
            if not output_directory.exists():
                result.validation_errors.append(f"Output directory not found: {output_directory}")
                logger.warning(f"No output directory found for {pipeline_name}: {output_directory}")
                return result
            
            # Perform LLM-based quality review if enabled
            if self.enable_llm_review and self.quality_reviewer:
                try:
                    llm_review = await self._perform_llm_quality_review(pipeline_name)
                    self._integrate_llm_review_results(result, llm_review)
                    
                except Exception as e:
                    error_msg = f"LLM quality review failed: {e}"
                    logger.warning(error_msg)
                    result.validation_errors.append(error_msg)
                    
                    # Fall back to rule-based assessment if enabled
                    if self.fallback_to_rules:
                        rule_based_result = await self._perform_rule_based_assessment(output_directory)
                        self._integrate_rule_based_results(result, rule_based_result)
            
            # If LLM review is disabled, use rule-based assessment
            elif self.fallback_to_rules:
                rule_based_result = await self._perform_rule_based_assessment(output_directory)
                self._integrate_rule_based_results(result, rule_based_result)
            
            # Finalize quality assessment
            self._finalize_quality_assessment(result)
            
        except Exception as e:
            error_msg = f"Quality validation failed for {pipeline_name}: {e}"
            logger.error(error_msg)
            result.validation_errors.append(error_msg)
        
        result.review_duration = time.time() - start_time
        
        logger.info(f"Quality validation completed for {pipeline_name}: "
                   f"Score {result.overall_score:.1f}, Grade {result.quality_grade}, "
                   f"Issues: {result.total_issues}, Production ready: {result.production_ready}")
        
        return result
    
    async def _perform_llm_quality_review(self, pipeline_name: str) -> PipelineQualityReview:
        """Perform LLM-based quality review."""
        try:
            # Set timeout for LLM review
            review_task = asyncio.create_task(
                self.quality_reviewer.review_pipeline_outputs(pipeline_name)
            )
            return await asyncio.wait_for(review_task, timeout=self.max_review_time)
            
        except asyncio.TimeoutError:
            raise LLMQualityError(f"LLM review timed out after {self.max_review_time}s")
        except Exception as e:
            raise LLMQualityError(f"LLM review failed: {e}")
    
    async def _perform_rule_based_assessment(self, 
                                           output_directory: Path) -> Dict[str, Any]:
        """Perform rule-based quality assessment as fallback."""
        logger.info("Performing rule-based quality assessment")
        
        issues = []
        files_reviewed = []
        
        # Scan all files in output directory
        try:
            for file_path in output_directory.rglob("*"):
                if file_path.is_file():
                    file_issues = await self._assess_file_rule_based(file_path)
                    issues.extend(file_issues)
                    files_reviewed.append(str(file_path))
        
        except Exception as e:
            logger.warning(f"Rule-based assessment failed: {e}")
        
        return {
            "issues": issues,
            "files_reviewed": files_reviewed,
            "overall_score": self._calculate_rule_based_score(issues),
            "reviewer_model": "rule-based"
        }
    
    async def _assess_file_rule_based(self, file_path: Path) -> List[QualityIssue]:
        """Assess a single file using rule-based methods."""
        issues = []
        
        try:
            # Only assess text files
            if file_path.suffix.lower() in ['.md', '.txt', '.csv', '.json', '.html']:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                
                # Check for template artifacts
                template_issues = self._check_template_artifacts(content, str(file_path))
                issues.extend(template_issues)
                
                # Check for debug/conversational content
                content_issues = self._check_content_quality_rules(content, str(file_path))
                issues.extend(content_issues)
                
        except Exception as e:
            logger.warning(f"Failed to assess file {file_path}: {e}")
        
        return issues
    
    def _check_template_artifacts(self, content: str, file_path: str) -> List[QualityIssue]:
        """Check for unresolved template artifacts."""
        issues = []
        
        # Check for {{ }} patterns
        import re
        template_patterns = [
            r'\{\{[^}]+\}\}',  # Standard Jinja2 templates
            r'\{\{\s*\$[^}]+\}\}',  # $ variable templates
            r'\{\{[^}]*\.[^}]+\}\}',  # Cross-step references
        ]
        
        for pattern in template_patterns:
            matches = re.findall(pattern, content)
            if matches:
                # Filter out common acceptable patterns
                problematic_matches = []
                for match in matches:
                    # Skip common acceptable patterns
                    if not any(acceptable in match.lower() for acceptable in 
                              ['date', 'now', 'timestamp', 'url']):
                        problematic_matches.append(match)
                
                if problematic_matches:
                    issues.append(QualityIssue(
                        category=IssueCategory.TEMPLATE_ARTIFACT,
                        severity=IssueSeverity.CRITICAL,
                        description=f"Unresolved template artifacts found: {problematic_matches[:3]}",
                        file_path=file_path,
                        suggestion="Ensure all template variables are properly resolved before output"
                    ))
        
        return issues
    
    def _check_content_quality_rules(self, content: str, file_path: str) -> List[QualityIssue]:
        """Check for content quality issues using rule-based methods."""
        issues = []
        
        # Check for conversational/debug content
        conversational_patterns = [
            r'certainly[!,.]',
            r'here\s+(is|are)\s+the',
            r'i\'ll\s+help\s+you',
            r'let\s+me\s+',
            r'as\s+an\s+ai\s+',
            r'i\s+can\s+help',
            r'here\'s\s+what'
        ]
        
        for pattern in conversational_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                issues.append(QualityIssue(
                    category=IssueCategory.CONTENT_QUALITY,
                    severity=IssueSeverity.MAJOR,
                    description="Conversational tone detected in output content",
                    file_path=file_path,
                    suggestion="Remove conversational phrases for professional output"
                ))
                break  # Only report once per file
        
        # Check for incomplete content
        incomplete_patterns = [
            r'\.\.\.+\s*$',  # Trailing ellipsis
            r'\[to\s+be\s+completed\]',
            r'\[todo\]',
            r'\[placeholder\]',
            r'lorem\s+ipsum'
        ]
        
        for pattern in incomplete_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                issues.append(QualityIssue(
                    category=IssueCategory.COMPLETENESS,
                    severity=IssueSeverity.MAJOR,
                    description="Incomplete or placeholder content detected",
                    file_path=file_path,
                    suggestion="Complete all content and remove placeholders"
                ))
                break
        
        return issues
    
    def _integrate_llm_review_results(self, 
                                    result: QualityValidationResult,
                                    llm_review: PipelineQualityReview):
        """Integrate LLM review results into validation result."""
        result.overall_score = llm_review.overall_score
        result.critical_issues = llm_review.critical_issues
        result.major_issues = llm_review.major_issues
        result.minor_issues = llm_review.minor_issues
        result.files_reviewed = llm_review.files_reviewed
        result.reviewer_model = llm_review.reviewer_model
        result.recommendations = llm_review.recommendations
        
        # Calculate component scores
        result.template_artifacts_found = any(
            issue.category == IssueCategory.TEMPLATE_ARTIFACT 
            for issue in llm_review.critical_issues + llm_review.major_issues
        )
        
        # Calculate quality component scores
        result.content_quality_score = self._calculate_content_quality_score(
            llm_review.critical_issues + llm_review.major_issues + llm_review.minor_issues
        )
        result.visual_quality_score = self._calculate_visual_quality_score(
            llm_review.critical_issues + llm_review.major_issues + llm_review.minor_issues
        )
        result.completeness_score = self._calculate_completeness_score(
            llm_review.critical_issues + llm_review.major_issues + llm_review.minor_issues
        )
    
    def _integrate_rule_based_results(self, 
                                    result: QualityValidationResult,
                                    rule_results: Dict[str, Any]):
        """Integrate rule-based assessment results."""
        result.overall_score = rule_results["overall_score"]
        result.files_reviewed = rule_results["files_reviewed"]
        result.reviewer_model = rule_results["reviewer_model"]
        
        # Categorize issues by severity
        all_issues = rule_results["issues"]
        result.critical_issues = [i for i in all_issues if i.severity == IssueSeverity.CRITICAL]
        result.major_issues = [i for i in all_issues if i.severity == IssueSeverity.MAJOR]
        result.minor_issues = [i for i in all_issues if i.severity == IssueSeverity.MINOR]
        
        # Generate basic recommendations
        result.recommendations = self._generate_rule_based_recommendations(all_issues)
        
        # Calculate component scores
        result.template_artifacts_found = any(
            issue.category == IssueCategory.TEMPLATE_ARTIFACT for issue in all_issues
        )
        result.content_quality_score = self._calculate_content_quality_score(all_issues)
        result.completeness_score = self._calculate_completeness_score(all_issues)
    
    def _finalize_quality_assessment(self, result: QualityValidationResult):
        """Finalize quality assessment and determine production readiness."""
        # Production readiness assessment
        result.production_ready = (
            result.overall_score >= self.quality_threshold and
            len(result.critical_issues) <= self.CRITICAL_ISSUE_FAIL_THRESHOLD and
            not result.template_artifacts_found
        )
    
    def _calculate_rule_based_score(self, issues: List[QualityIssue]) -> float:
        """Calculate quality score based on rule-based assessment."""
        base_score = 100.0
        
        for issue in issues:
            if issue.severity == IssueSeverity.CRITICAL:
                base_score -= 25.0
            elif issue.severity == IssueSeverity.MAJOR:
                base_score -= 10.0
            elif issue.severity == IssueSeverity.MINOR:
                base_score -= 5.0
        
        return max(0.0, base_score)
    
    def _calculate_content_quality_score(self, issues: List[QualityIssue]) -> float:
        """Calculate content quality component score."""
        content_issues = [i for i in issues if i.category == IssueCategory.CONTENT_QUALITY]
        base_score = 100.0
        
        for issue in content_issues:
            if issue.severity == IssueSeverity.CRITICAL:
                base_score -= 30.0
            elif issue.severity == IssueSeverity.MAJOR:
                base_score -= 15.0
            else:
                base_score -= 5.0
        
        return max(0.0, base_score)
    
    def _calculate_visual_quality_score(self, issues: List[QualityIssue]) -> float:
        """Calculate visual quality component score."""
        visual_issues = [i for i in issues if i.category == IssueCategory.VISUAL_QUALITY]
        
        if not visual_issues:
            return 100.0  # No visual content or no issues
        
        base_score = 100.0
        for issue in visual_issues:
            if issue.severity == IssueSeverity.CRITICAL:
                base_score -= 40.0
            elif issue.severity == IssueSeverity.MAJOR:
                base_score -= 20.0
            else:
                base_score -= 10.0
        
        return max(0.0, base_score)
    
    def _calculate_completeness_score(self, issues: List[QualityIssue]) -> float:
        """Calculate completeness component score."""
        completeness_issues = [i for i in issues if i.category == IssueCategory.COMPLETENESS]
        base_score = 100.0
        
        for issue in completeness_issues:
            if issue.severity == IssueSeverity.CRITICAL:
                base_score -= 35.0
            elif issue.severity == IssueSeverity.MAJOR:
                base_score -= 20.0
            else:
                base_score -= 10.0
        
        return max(0.0, base_score)
    
    def _generate_rule_based_recommendations(self, issues: List[QualityIssue]) -> List[str]:
        """Generate recommendations from rule-based issues."""
        recommendations = []
        
        # Group by category
        categories = {}
        for issue in issues:
            if issue.category not in categories:
                categories[issue.category] = []
            categories[issue.category].append(issue)
        
        # Generate category-specific recommendations
        if IssueCategory.TEMPLATE_ARTIFACT in categories:
            recommendations.append("Fix all unresolved template variables before deployment")
        
        if IssueCategory.CONTENT_QUALITY in categories:
            recommendations.append("Remove conversational tone and debug content from outputs")
        
        if IssueCategory.COMPLETENESS in categories:
            recommendations.append("Complete all content and remove placeholder text")
        
        critical_count = len([i for i in issues if i.severity == IssueSeverity.CRITICAL])
        if critical_count > 0:
            recommendations.insert(0, f"Address {critical_count} critical issues immediately")
        
        return recommendations
    
    def supports_llm_review(self) -> bool:
        """Check if LLM review is available and enabled."""
        return self.enable_llm_review and self.quality_reviewer is not None
    
    def supports_visual_review(self) -> bool:
        """Check if visual content review is available."""
        return (self.enable_visual_review and 
                self.quality_reviewer is not None and
                hasattr(self.quality_reviewer, '_get_vision_client'))