"""
Quality assessment framework for pipeline output analysis.

This module provides the core data structures and classes for assessing
the quality of pipeline outputs using LLM-powered analysis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class IssueSeverity(Enum):
    """Severity levels for quality issues."""
    CRITICAL = "critical"  # Must fix - prevents production use
    MAJOR = "major"       # Should fix - impacts user experience  
    MINOR = "minor"       # Nice to fix - minor improvements
    ACCEPTABLE = "acceptable"  # No issues found


class IssueCategory(Enum):
    """Categories of quality issues."""
    TEMPLATE_ARTIFACT = "template_artifact"
    CONTENT_QUALITY = "content_quality"
    FILE_ORGANIZATION = "file_organization"
    VISUAL_QUALITY = "visual_quality"
    COMPLETENESS = "completeness"
    PROFESSIONAL_STANDARDS = "professional_standards"


@dataclass
class QualityIssue:
    """Represents a specific quality issue found in output analysis."""
    
    category: IssueCategory
    severity: IssueSeverity
    description: str
    file_path: str
    line_number: Optional[int] = None
    suggestion: str = ""
    confidence: float = 1.0  # 0.0 to 1.0
    detected_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert issue to dictionary for serialization."""
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "description": self.description,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "suggestion": self.suggestion,
            "confidence": self.confidence,
            "detected_at": self.detected_at.isoformat()
        }


@dataclass
class ContentQuality:
    """Assessment of text content quality."""
    
    rating: IssueSeverity
    issues: List[QualityIssue] = field(default_factory=list)
    feedback: str = ""
    template_artifacts_detected: bool = False
    debug_artifacts_detected: bool = False
    incomplete_content_detected: bool = False
    conversational_tone_detected: bool = False
    
    def has_critical_issues(self) -> bool:
        """Check if content has critical quality issues."""
        return any(issue.severity == IssueSeverity.CRITICAL for issue in self.issues)
    
    def has_template_artifacts(self) -> bool:
        """Check if content contains unrendered template variables."""
        return self.template_artifacts_detected or any(
            issue.category == IssueCategory.TEMPLATE_ARTIFACT 
            for issue in self.issues
        )


@dataclass 
class VisualQuality:
    """Assessment of visual content quality (images, charts)."""
    
    rating: IssueSeverity
    issues: List[QualityIssue] = field(default_factory=list)
    feedback: str = ""
    image_renders_correctly: bool = True
    charts_have_labels: bool = True
    professional_appearance: bool = True
    appropriate_styling: bool = True
    
    def has_visual_issues(self) -> bool:
        """Check if visual content has quality issues."""
        return not all([
            self.image_renders_correctly,
            self.charts_have_labels, 
            self.professional_appearance,
            self.appropriate_styling
        ])


@dataclass
class OrganizationReview:
    """Assessment of file organization and naming conventions."""
    
    issues: List[QualityIssue] = field(default_factory=list)
    correct_location: bool = True
    appropriate_naming: bool = True
    expected_files_present: bool = True
    
    def has_organization_issues(self) -> bool:
        """Check if file organization has issues."""
        return not all([
            self.correct_location,
            self.appropriate_naming,
            self.expected_files_present
        ]) or len(self.issues) > 0


@dataclass
class PipelineQualityReview:
    """Comprehensive quality review for a complete pipeline."""
    
    pipeline_name: str
    overall_score: int  # 0-100 quality score
    files_reviewed: List[str] = field(default_factory=list)
    critical_issues: List[QualityIssue] = field(default_factory=list)
    major_issues: List[QualityIssue] = field(default_factory=list)
    minor_issues: List[QualityIssue] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    production_ready: bool = False
    reviewed_at: datetime = field(default_factory=datetime.utcnow)
    reviewer_model: str = ""
    review_duration_seconds: float = 0.0
    
    @property
    def total_issues(self) -> int:
        """Total number of issues across all severities."""
        return len(self.critical_issues) + len(self.major_issues) + len(self.minor_issues)
    
    def get_issues_by_category(self, category: IssueCategory) -> List[QualityIssue]:
        """Get all issues of a specific category."""
        all_issues = self.critical_issues + self.major_issues + self.minor_issues
        return [issue for issue in all_issues if issue.category == category]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert review to dictionary for serialization."""
        return {
            "pipeline_name": self.pipeline_name,
            "overall_score": self.overall_score,
            "files_reviewed": self.files_reviewed,
            "critical_issues": [issue.to_dict() for issue in self.critical_issues],
            "major_issues": [issue.to_dict() for issue in self.major_issues], 
            "minor_issues": [issue.to_dict() for issue in self.minor_issues],
            "recommendations": self.recommendations,
            "production_ready": self.production_ready,
            "reviewed_at": self.reviewed_at.isoformat(),
            "reviewer_model": self.reviewer_model,
            "review_duration_seconds": self.review_duration_seconds,
            "total_issues": self.total_issues
        }


class QualityScorer:
    """Calculates quality scores based on issues found."""
    
    @staticmethod
    def calculate_score(issues: List[QualityIssue]) -> int:
        """Calculate 0-100 quality score based on issues found."""
        base_score = 100
        
        for issue in issues:
            if issue.severity == IssueSeverity.CRITICAL:
                base_score -= 25
            elif issue.severity == IssueSeverity.MAJOR:
                base_score -= 10
            elif issue.severity == IssueSeverity.MINOR:
                base_score -= 3
                
        return max(0, base_score)
    
    @staticmethod
    def determine_production_readiness(score: int) -> bool:
        """Determine if pipeline output is production ready based on score."""
        return score >= 80  # Minimum threshold for production readiness


class TemplateArtifactDetector:
    """Detects unrendered template variables in content."""
    
    # Common template variable patterns
    TEMPLATE_PATTERNS = [
        r'\{\{[^}]+\}\}',           # Jinja2: {{variable}}
        r'\$\{[^}]+\}',            # Shell/JS: ${variable}  
        r'%\{[^}]+\}%',            # Custom: %{variable}%
        r'\[\[[^\]]+\]\]',         # Wiki-style: [[variable]]
        r'<[^>]+>',                # Angle brackets: <variable> (careful with HTML)
        r'\{\%[^%]+\%\}',          # Jinja2 statements: {% statement %}
    ]
    
    def detect_template_artifacts(self, content: str, file_path: str = "") -> List[QualityIssue]:
        """Detect unrendered template variables in content."""
        issues = []
        
        for pattern in self.TEMPLATE_PATTERNS:
            matches = re.finditer(pattern, content, re.MULTILINE)
            
            for match in matches:
                # Skip common HTML tags and valid content
                matched_text = match.group(0)
                if self._is_likely_template_artifact(matched_text, content):
                    line_number = content[:match.start()].count('\n') + 1
                    
                    issues.append(QualityIssue(
                        category=IssueCategory.TEMPLATE_ARTIFACT,
                        severity=IssueSeverity.CRITICAL,
                        description=f"Unrendered template variable: {matched_text}",
                        file_path=file_path,
                        line_number=line_number,
                        suggestion=f"Ensure template variable '{matched_text}' is properly resolved before output generation"
                    ))
        
        return issues
    
    def _is_likely_template_artifact(self, text: str, full_content: str) -> bool:
        """Determine if matched text is likely an unrendered template variable."""
        # Skip common HTML tags
        html_tags = ['<html>', '<head>', '<body>', '<div>', '<span>', '<p>', '<a>', '<img>', '<br>']
        if any(tag in text.lower() for tag in html_tags):
            return False
            
        # Skip markdown links
        if text.startswith('[[') and '|' in text:
            return False
            
        # If it contains typical variable names, likely a template artifact
        variable_indicators = ['name', 'value', 'data', 'file', 'path', 'model', 'input', 'output']
        return any(indicator in text.lower() for indicator in variable_indicators)


class ContentQualityAssessor:
    """Assesses content quality using pattern matching and heuristics."""
    
    # Debug/conversational phrases to detect
    CONVERSATIONAL_PATTERNS = [
        r"(?i)\bcertainly!?\b",
        r"(?i)\bhere'?s?\s+(?:the|your|a)\b",
        r"(?i)\bi'?ll\s+help\s+you\b",
        r"(?i)\blet\s+me\s+(?:help|assist|show)\b",
        r"(?i)\bof\s+course!?\b",
        r"(?i)\bfeel\s+free\s+to\b",
        r"(?i)\bplease\s+(?:find|see|note)\b",
        r"(?i)\bas\s+(?:an\s+)?ai\b",
        r"(?i)\bi\s+(?:can|will|would|should)\s+help\b",
    ]
    
    # Placeholder text patterns
    PLACEHOLDER_PATTERNS = [
        r"(?i)\b(?:lorem\s+ipsum|placeholder|sample\s+text|example\s+content)\b",
        r"(?i)\b(?:todo|tbd|to\s+be\s+determined|coming\s+soon)\b",
        r"(?i)\b(?:insert\s+\w+\s+here|add\s+\w+\s+here)\b",
        r"\[.*?\]",  # Bracketed placeholders
    ]
    
    def assess_content_quality(self, content: str, file_path: str = "") -> ContentQuality:
        """Assess the quality of text content."""
        issues = []
        
        # Check for template artifacts
        template_detector = TemplateArtifactDetector()
        template_issues = template_detector.detect_template_artifacts(content, file_path)
        issues.extend(template_issues)
        
        # Check for conversational tone
        conversational_issues = self._detect_conversational_tone(content, file_path)
        issues.extend(conversational_issues)
        
        # Check for placeholder content
        placeholder_issues = self._detect_placeholder_content(content, file_path)
        issues.extend(placeholder_issues)
        
        # Check for content completeness
        completeness_issues = self._assess_completeness(content, file_path)
        issues.extend(completeness_issues)
        
        # Determine overall rating
        if any(issue.severity == IssueSeverity.CRITICAL for issue in issues):
            rating = IssueSeverity.CRITICAL
        elif any(issue.severity == IssueSeverity.MAJOR for issue in issues):
            rating = IssueSeverity.MAJOR  
        elif any(issue.severity == IssueSeverity.MINOR for issue in issues):
            rating = IssueSeverity.MINOR
        else:
            rating = IssueSeverity.ACCEPTABLE
        
        return ContentQuality(
            rating=rating,
            issues=issues,
            template_artifacts_detected=len(template_issues) > 0,
            debug_artifacts_detected=len(conversational_issues) > 0,
            incomplete_content_detected=any(
                issue.category == IssueCategory.COMPLETENESS for issue in issues
            ),
            conversational_tone_detected=len(conversational_issues) > 0
        )
    
    def _detect_conversational_tone(self, content: str, file_path: str) -> List[QualityIssue]:
        """Detect conversational or debug-like tone in content."""
        issues = []
        
        for pattern in self.CONVERSATIONAL_PATTERNS:
            matches = re.finditer(pattern, content, re.MULTILINE)
            
            for match in matches:
                line_number = content[:match.start()].count('\n') + 1
                
                issues.append(QualityIssue(
                    category=IssueCategory.CONTENT_QUALITY,
                    severity=IssueSeverity.MAJOR,
                    description=f"Conversational tone detected: '{match.group(0)}'",
                    file_path=file_path,
                    line_number=line_number,
                    suggestion="Replace conversational language with direct, professional content"
                ))
        
        return issues
    
    def _detect_placeholder_content(self, content: str, file_path: str) -> List[QualityIssue]:
        """Detect placeholder or template content."""
        issues = []
        
        for pattern in self.PLACEHOLDER_PATTERNS:
            matches = re.finditer(pattern, content, re.MULTILINE)
            
            for match in matches:
                line_number = content[:match.start()].count('\n') + 1
                
                issues.append(QualityIssue(
                    category=IssueCategory.COMPLETENESS,
                    severity=IssueSeverity.MAJOR,
                    description=f"Placeholder content detected: '{match.group(0)}'",
                    file_path=file_path,
                    line_number=line_number,
                    suggestion="Replace placeholder content with actual information"
                ))
        
        return issues
    
    def _assess_completeness(self, content: str, file_path: str) -> List[QualityIssue]:
        """Assess content completeness."""
        issues = []
        
        # Check for very short content
        if len(content.strip()) < 100:
            issues.append(QualityIssue(
                category=IssueCategory.COMPLETENESS,
                severity=IssueSeverity.MINOR,
                description="Content appears very brief - may be incomplete",
                file_path=file_path,
                suggestion="Verify that all expected content is present and complete"
            ))
        
        # Check for cut-off indicators
        cutoff_patterns = [
            r"(?i)\.\.\.$",           # Ends with ellipsis
            r"(?i)\[truncated\]",     # Truncation notice
            r"(?i)\[continued\]",     # Continuation notice
        ]
        
        for pattern in cutoff_patterns:
            if re.search(pattern, content):
                issues.append(QualityIssue(
                    category=IssueCategory.COMPLETENESS,
                    severity=IssueSeverity.MAJOR,
                    description="Content appears to be cut off or truncated",
                    file_path=file_path,
                    suggestion="Ensure content generation completed successfully and is not truncated"
                ))
        
        return issues