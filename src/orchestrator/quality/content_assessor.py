"""
Advanced content quality assessment with LLM-powered semantic analysis.

This module extends the basic content assessment with sophisticated LLM prompts
for semantic analysis, professional tone validation, and content appropriateness
evaluation beyond simple pattern matching.
"""

import re
import asyncio
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from ..core.quality_assessment import (
    QualityIssue, IssueCategory, IssueSeverity, ContentQuality
)


class ContentType(Enum):
    """Types of content for specialized assessment."""
    MARKDOWN_DOCUMENTATION = "markdown_documentation"
    CSV_DATA = "csv_data"
    JSON_DATA = "json_data"
    CODE_OUTPUT = "code_output"
    REPORT_NARRATIVE = "report_narrative"
    TECHNICAL_ANALYSIS = "technical_analysis"
    DATA_VISUALIZATION_DESC = "data_visualization_description"
    GENERAL_TEXT = "general_text"


@dataclass
class ContentAssessmentPrompt:
    """Structured prompt for LLM-based content assessment."""
    
    content_type: ContentType
    system_prompt: str
    analysis_prompt: str
    specific_checks: List[str]
    expected_json_schema: Dict[str, Any]


class AdvancedContentAssessor:
    """
    Advanced content quality assessment using LLM-powered semantic analysis
    and content-type-specific validation criteria.
    """
    
    def __init__(self):
        """Initialize advanced content assessor."""
        self.assessment_prompts = self._initialize_assessment_prompts()
        
        # Enhanced pattern detection beyond basic conversational tone
        self.advanced_debug_patterns = [
            # Development environment traces
            r"(?i)\b(debug|trace|console\.log|print\(|dump\(|var_dump)\b",
            r"(?i)\b(stack\s+trace|backtrace|exception\s+in)\b",
            r"(?i)\b(error\s+on\s+line|fatal\s+error|warning:)\b",
            
            # AI model artifacts
            r"(?i)\b(as\s+an\s+ai|i'm\s+an\s+ai|ai\s+language\s+model)\b",
            r"(?i)\b(i\s+(?:don't|can't|cannot)\s+(?:access|see|view|browse))\b",
            r"(?i)\b(my\s+knowledge\s+cutoff|training\s+data)\b",
            r"(?i)\b(i\s+(?:apologize|sorry)\s+for\s+(?:any|the))\b",
            
            # Conversational artifacts  
            r"(?i)\b(certainly!?|of\s+course!?|absolutely!?|definitely!?)\b",
            r"(?i)\b(here\s+(?:is|are|you\s+go))\b",
            r"(?i)\b(i\s+hope\s+this\s+helps?|hope\s+this\s+helps?)\b",
            r"(?i)\b(let\s+me\s+(?:know|help|assist|show|explain))\b",
            r"(?i)\b(feel\s+free\s+to|don't\s+hesitate\s+to)\b",
            
            # Instructional artifacts
            r"(?i)\b(step\s+\d+:|first,?|second,?|third,?|finally,?)\b",
            r"(?i)\b(in\s+(?:this|the)\s+(?:example|tutorial|guide))\b",
            r"(?i)\b(for\s+(?:this|your)\s+(?:task|project|use\s+case))\b",
            
            # Placeholder and incomplete content
            r"(?i)\b(to\s+be\s+(?:determined|completed|implemented))\b",
            r"(?i)\b(coming\s+soon|work\s+in\s+progress|under\s+construction)\b",
            r"(?i)\b(example\s+(?:content|text|data|output))\b",
            r"(?i)\b(sample\s+(?:content|text|data|output))\b",
            
            # Technical debugging artifacts
            r"(?i)\b(test\s+(?:data|content|output|file))\b",
            r"(?i)\b(mock\s+(?:data|content|output|response))\b",
            r"(?i)\b(dummy\s+(?:data|content|text|values?))\b",
            
            # Meta-commentary
            r"(?i)\b(note\s+that|please\s+note|important\s+note)\b",
            r"(?i)\b(as\s+(?:mentioned|noted|discussed)\s+(?:above|below|earlier))\b",
            r"(?i)\b((?:this|the)\s+(?:above|following)\s+(?:shows?|demonstrates?))\b",
        ]
        
        # Professional tone indicators
        self.unprofessional_patterns = [
            # Casual language
            r"(?i)\b(gonna|wanna|kinda|sorta|yeah|ok|awesome|cool)\b",
            r"(?i)\b(super\s+(?:easy|simple|quick|fast))\b",
            r"(?i)\b(really\s+(?:good|great|awesome|cool))\b",
            
            # Emotional language  
            r"(?i)\b(excited|thrilled|amazing|fantastic|incredible)\b",
            r"(?i)\b(love\s+(?:this|that|it)|hate\s+(?:this|that|it))\b",
            
            # Uncertain language
            r"(?i)\b(maybe|perhaps|probably|likely|might\s+be|could\s+be)\b",
            r"(?i)\b(i\s+think|i\s+believe|i\s+guess|i\s+suppose)\b",
            r"(?i)\b(sort\s+of|kind\s+of|more\s+or\s+less)\b",
        ]
        
        # Content completeness indicators
        self.completeness_issues = [
            # Truncation indicators
            r"\.\.\.$",
            r"\[truncated\]",
            r"\[continued\]",
            r"\[more\]",
            r"(?i)\b(and\s+so\s+on|etc\.?|and\s+more)$",
            
            # Missing content indicators
            r"(?i)\b(see\s+(?:above|below|attachment|link))\b",
            r"(?i)\b(refer\s+to\s+(?:the|section|page|document))\b", 
            r"(?i)\b(as\s+shown\s+in\s+(?:the|figure|chart|table))\b",
            
            # Incomplete processing
            r"(?i)\b(processing\.\.\.|loading\.\.\.|generating\.\.\.)\b",
            r"(?i)\b(analysis\s+in\s+progress|calculation\s+pending)\b",
        ]
    
    def _initialize_assessment_prompts(self) -> Dict[ContentType, ContentAssessmentPrompt]:
        """Initialize content-type-specific assessment prompts."""
        prompts = {}
        
        # Markdown Documentation Assessment
        prompts[ContentType.MARKDOWN_DOCUMENTATION] = ContentAssessmentPrompt(
            content_type=ContentType.MARKDOWN_DOCUMENTATION,
            system_prompt="""You are a technical documentation quality expert. 
            Assess markdown documentation for professional standards, clarity, completeness, and production readiness.""",
            analysis_prompt="""Analyze this markdown documentation for:

1. CRITICAL ISSUES:
   - Unrendered template variables ({{variable}}, ${var}, etc.)
   - Conversational AI artifacts ("Certainly!", "Here's how...", "I hope this helps")
   - Debug statements or development artifacts
   - Incomplete sections or placeholder content
   - Broken internal references or links

2. PROFESSIONAL STANDARDS:
   - Clear, professional tone and language
   - Proper markdown formatting and structure
   - Complete information without gaps
   - Appropriate technical depth
   - Consistent terminology and style

3. TECHNICAL ACCURACY:
   - Accurate code examples and syntax
   - Correct technical terminology
   - Valid file paths and references
   - Properly formatted data examples

4. COMPLETENESS:
   - All sections fully developed
   - No placeholder or sample content
   - Complete explanations and context
   - Proper conclusions and summaries""",
            specific_checks=[
                "template_artifacts", "conversational_tone", "debug_artifacts", 
                "incomplete_content", "professional_tone", "technical_accuracy",
                "markdown_formatting", "content_completeness"
            ],
            expected_json_schema={
                "overall_rating": "CRITICAL|MAJOR|MINOR|ACCEPTABLE",
                "issues": [{"category": "str", "severity": "str", "description": "str", "suggestion": "str"}],
                "professional_standards": {"score": "int", "feedback": "str"},
                "technical_accuracy": {"score": "int", "feedback": "str"},
                "completeness_assessment": {"score": "int", "feedback": "str"}
            }
        )
        
        # CSV Data Assessment  
        prompts[ContentType.CSV_DATA] = ContentAssessmentPrompt(
            content_type=ContentType.CSV_DATA,
            system_prompt="""You are a data quality expert. Assess CSV data files for structure, completeness, and professional presentation standards.""",
            analysis_prompt="""Analyze this CSV data for:

1. CRITICAL ISSUES:
   - Template variables in headers or data ({{column}}, ${value})
   - Debug or sample data markers
   - Incomplete data processing artifacts
   - Malformed CSV structure
   - Missing or placeholder values

2. DATA QUALITY:
   - Consistent data formatting
   - Appropriate data types for columns
   - No test or dummy data
   - Complete records without gaps
   - Proper handling of special characters

3. PROFESSIONAL STANDARDS:
   - Clear, descriptive column headers
   - Consistent naming conventions
   - Appropriate data precision
   - Production-ready data values""",
            specific_checks=[
                "template_artifacts", "data_structure", "completeness", 
                "professional_formatting", "test_data_artifacts"
            ],
            expected_json_schema={
                "overall_rating": "CRITICAL|MAJOR|MINOR|ACCEPTABLE", 
                "data_structure_valid": "bool",
                "has_test_data": "bool",
                "completeness_score": "int"
            }
        )
        
        # JSON Data Assessment
        prompts[ContentType.JSON_DATA] = ContentAssessmentPrompt(
            content_type=ContentType.JSON_DATA,
            system_prompt="""You are a JSON data quality expert. Assess JSON files for structure, completeness, and production readiness.""",
            analysis_prompt="""Analyze this JSON data for:

1. CRITICAL ISSUES:
   - Template variables in keys or values
   - Invalid JSON syntax or structure
   - Debug or development data
   - Incomplete processing artifacts
   - Placeholder or example values

2. DATA QUALITY:
   - Valid JSON schema and structure  
   - Appropriate data types for values
   - Consistent naming conventions
   - Complete data objects
   - No test or mock data

3. PROFESSIONAL STANDARDS:
   - Clean, readable formatting
   - Meaningful key names
   - Production-appropriate values
   - Proper data organization""",
            specific_checks=[
                "json_validity", "template_artifacts", "test_data", 
                "completeness", "professional_structure"
            ],
            expected_json_schema={
                "overall_rating": "CRITICAL|MAJOR|MINOR|ACCEPTABLE",
                "json_valid": "bool", 
                "has_template_artifacts": "bool",
                "professional_quality": "int"
            }
        )
        
        # Report Narrative Assessment
        prompts[ContentType.REPORT_NARRATIVE] = ContentAssessmentPrompt(
            content_type=ContentType.REPORT_NARRATIVE,
            system_prompt="""You are a professional report writing expert. Assess report narratives for professional tone, completeness, and business readiness.""",
            analysis_prompt="""Analyze this report narrative for:

1. CRITICAL ISSUES:
   - Conversational AI language and artifacts
   - Template variables or placeholders
   - Incomplete analysis or conclusions  
   - Debug or development content
   - Unprofessional tone or language

2. PROFESSIONAL STANDARDS:
   - Business-appropriate language and tone
   - Clear, objective presentation of findings
   - Complete analysis with proper conclusions
   - Appropriate level of technical detail
   - Professional formatting and structure

3. CONTENT QUALITY:
   - Coherent narrative flow
   - Supporting evidence and data
   - Actionable insights and recommendations
   - Proper executive summary level content""",
            specific_checks=[
                "professional_tone", "conversational_artifacts", "completeness",
                "business_readiness", "analytical_depth"
            ],
            expected_json_schema={
                "overall_rating": "CRITICAL|MAJOR|MINOR|ACCEPTABLE",
                "professional_tone_score": "int",
                "business_readiness": "bool", 
                "analytical_completeness": "int"
            }
        )
        
        return prompts
    
    async def assess_content_advanced(
        self, 
        content: str, 
        file_path: str = "", 
        llm_client=None
    ) -> ContentQuality:
        """
        Perform advanced content quality assessment using both rule-based 
        and LLM-powered semantic analysis.
        """
        issues = []
        
        # Determine content type
        content_type = self._determine_content_type(content, file_path)
        
        # Rule-based assessment (enhanced patterns)
        rule_based_issues = self._assess_with_enhanced_patterns(content, file_path, content_type)
        issues.extend(rule_based_issues)
        
        # LLM-based semantic assessment (if client available)
        if llm_client:
            try:
                llm_issues = await self._assess_with_llm(content, file_path, content_type, llm_client)
                issues.extend(llm_issues)
            except Exception as e:
                # Fall back to rule-based only if LLM fails
                issues.append(QualityIssue(
                    category=IssueCategory.CONTENT_QUALITY,
                    severity=IssueSeverity.MINOR,
                    description=f"LLM assessment unavailable: {str(e)}",
                    file_path=file_path,
                    suggestion="Manual review recommended for comprehensive quality assessment"
                ))
        
        # Determine overall rating
        rating = self._determine_overall_rating(issues)
        
        # Create enhanced content quality assessment
        return ContentQuality(
            rating=rating,
            issues=issues,
            template_artifacts_detected=any(
                issue.category == IssueCategory.TEMPLATE_ARTIFACT for issue in issues
            ),
            debug_artifacts_detected=any(
                "debug" in issue.description.lower() or "trace" in issue.description.lower() 
                for issue in issues
            ),
            incomplete_content_detected=any(
                issue.category == IssueCategory.COMPLETENESS for issue in issues
            ),
            conversational_tone_detected=any(
                "conversational" in issue.description.lower() or "ai artifact" in issue.description.lower()
                for issue in issues
            )
        )
    
    def _determine_content_type(self, content: str, file_path: str) -> ContentType:
        """Determine the type of content for specialized assessment."""
        file_ext = Path(file_path).suffix.lower() if file_path else ""
        
        # File extension-based detection
        if file_ext == ".md":
            return ContentType.MARKDOWN_DOCUMENTATION
        elif file_ext == ".csv":
            return ContentType.CSV_DATA  
        elif file_ext == ".json":
            return ContentType.JSON_DATA
        
        # Content-based detection
        content_lower = content.lower()
        
        # Check for CSV structure
        if self._looks_like_csv(content):
            return ContentType.CSV_DATA
        
        # Check for JSON structure
        if self._looks_like_json(content):
            return ContentType.JSON_DATA
        
        # Check for markdown
        if any(marker in content for marker in ['#', '```', '[', '](', '**', '*']):
            return ContentType.MARKDOWN_DOCUMENTATION
        
        # Check for code output
        if any(term in content_lower for term in ['function', 'class', 'def', 'import', 'return']):
            return ContentType.CODE_OUTPUT
        
        # Check for report-like content
        report_indicators = ['analysis', 'results', 'findings', 'conclusion', 'summary', 'recommendation']
        if any(term in content_lower for term in report_indicators):
            return ContentType.REPORT_NARRATIVE
        
        # Check for data visualization descriptions
        viz_indicators = ['chart', 'graph', 'plot', 'visualization', 'figure', 'axis', 'legend']
        if any(term in content_lower for term in viz_indicators):
            return ContentType.DATA_VISUALIZATION_DESC
        
        return ContentType.GENERAL_TEXT
    
    def _looks_like_csv(self, content: str) -> bool:
        """Check if content appears to be CSV formatted."""
        lines = content.strip().split('\n')
        if len(lines) < 2:
            return False
        
        # Check if first few lines have consistent comma separation
        first_line_commas = lines[0].count(',')
        if first_line_commas == 0:
            return False
        
        consistent_commas = all(
            line.count(',') == first_line_commas 
            for line in lines[:min(5, len(lines))]
        )
        
        return consistent_commas
    
    def _looks_like_json(self, content: str) -> bool:
        """Check if content appears to be JSON formatted."""
        try:
            import json
            json.loads(content.strip())
            return True
        except (json.JSONDecodeError, ValueError):
            return False
    
    def _assess_with_enhanced_patterns(
        self, 
        content: str, 
        file_path: str, 
        content_type: ContentType
    ) -> List[QualityIssue]:
        """Assess content using enhanced pattern matching."""
        issues = []
        
        # Advanced debug artifact detection
        for pattern in self.advanced_debug_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                line_number = content[:match.start()].count('\n') + 1
                issues.append(QualityIssue(
                    category=IssueCategory.CONTENT_QUALITY,
                    severity=IssueSeverity.MAJOR,
                    description=f"Debug/development artifact detected: '{match.group(0)}'",
                    file_path=file_path,
                    line_number=line_number,
                    suggestion="Remove debug statements and development artifacts from production output"
                ))
        
        # Professional tone assessment
        for pattern in self.unprofessional_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                line_number = content[:match.start()].count('\n') + 1
                issues.append(QualityIssue(
                    category=IssueCategory.PROFESSIONAL_STANDARDS,
                    severity=IssueSeverity.MINOR,
                    description=f"Unprofessional language detected: '{match.group(0)}'",
                    file_path=file_path,
                    line_number=line_number,
                    suggestion="Use more formal, professional language appropriate for production documentation"
                ))
        
        # Content completeness assessment
        for pattern in self.completeness_issues:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                line_number = content[:match.start()].count('\n') + 1
                issues.append(QualityIssue(
                    category=IssueCategory.COMPLETENESS,
                    severity=IssueSeverity.MAJOR,
                    description=f"Content completeness issue: '{match.group(0)}'",
                    file_path=file_path,
                    line_number=line_number,
                    suggestion="Complete all content sections and remove placeholder references"
                ))
        
        # Content-type specific assessments
        if content_type == ContentType.CSV_DATA:
            issues.extend(self._assess_csv_specific(content, file_path))
        elif content_type == ContentType.JSON_DATA:
            issues.extend(self._assess_json_specific(content, file_path))
        elif content_type == ContentType.MARKDOWN_DOCUMENTATION:
            issues.extend(self._assess_markdown_specific(content, file_path))
        
        return issues
    
    def _assess_csv_specific(self, content: str, file_path: str) -> List[QualityIssue]:
        """CSV-specific quality assessment."""
        issues = []
        
        lines = content.strip().split('\n')
        if not lines:
            return issues
        
        # Check for test data indicators
        test_data_patterns = [
            r"(?i)\b(test|sample|example|dummy|mock|placeholder)\b",
            r"(?i)\b(john\s+doe|jane\s+smith|test\s*user)\b",
            r"(?i)\b(example\.com|test\.com|sample\.org)\b",
        ]
        
        for i, line in enumerate(lines):
            for pattern in test_data_patterns:
                if re.search(pattern, line):
                    issues.append(QualityIssue(
                        category=IssueCategory.CONTENT_QUALITY,
                        severity=IssueSeverity.MAJOR,
                        description=f"Test/sample data detected in CSV: '{line[:50]}...'",
                        file_path=file_path,
                        line_number=i + 1,
                        suggestion="Replace test data with real production data"
                    ))
        
        return issues
    
    def _assess_json_specific(self, content: str, file_path: str) -> List[QualityIssue]:
        """JSON-specific quality assessment.""" 
        issues = []
        
        # Check JSON validity
        try:
            import json
            json.loads(content)
        except json.JSONDecodeError as e:
            issues.append(QualityIssue(
                category=IssueCategory.CONTENT_QUALITY,
                severity=IssueSeverity.CRITICAL,
                description=f"Invalid JSON structure: {str(e)}",
                file_path=file_path,
                suggestion="Fix JSON syntax errors to ensure valid structure"
            ))
        
        # Check for test data in JSON
        test_indicators = ["test", "sample", "example", "mock", "dummy", "placeholder"]
        if any(indicator in content.lower() for indicator in test_indicators):
            issues.append(QualityIssue(
                category=IssueCategory.CONTENT_QUALITY,
                severity=IssueSeverity.MAJOR,
                description="Test or sample data detected in JSON content",
                file_path=file_path,
                suggestion="Replace test data with production-appropriate values"
            ))
        
        return issues
    
    def _assess_markdown_specific(self, content: str, file_path: str) -> List[QualityIssue]:
        """Markdown-specific quality assessment."""
        issues = []
        
        # Check for broken links
        broken_link_pattern = r'\[([^\]]*)\]\(([^)]*)\)'
        matches = re.finditer(broken_link_pattern, content)
        
        for match in matches:
            link_text, link_url = match.groups()
            if not link_url or link_url.startswith('#') and len(link_url) == 1:
                line_number = content[:match.start()].count('\n') + 1
                issues.append(QualityIssue(
                    category=IssueCategory.COMPLETENESS,
                    severity=IssueSeverity.MINOR,
                    description=f"Incomplete or empty link: [{link_text}]({link_url})",
                    file_path=file_path,
                    line_number=line_number,
                    suggestion="Complete link URLs or remove incomplete links"
                ))
        
        # Check for TODO or placeholder sections
        todo_patterns = [
            r"(?i)#{1,6}\s*(todo|tbd|placeholder|coming\s+soon)",
            r"(?i)\*\*\s*(todo|tbd|placeholder|coming\s+soon)\s*\*\*",
        ]
        
        for pattern in todo_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                line_number = content[:match.start()].count('\n') + 1
                issues.append(QualityIssue(
                    category=IssueCategory.COMPLETENESS,
                    severity=IssueSeverity.MAJOR,
                    description=f"Incomplete section detected: {match.group(0)}",
                    file_path=file_path,
                    line_number=line_number,
                    suggestion="Complete all documentation sections before production release"
                ))
        
        return issues
    
    async def _assess_with_llm(
        self, 
        content: str, 
        file_path: str, 
        content_type: ContentType,
        llm_client
    ) -> List[QualityIssue]:
        """Perform LLM-based semantic content assessment."""
        issues = []
        
        # Get content-type specific prompt
        prompt_config = self.assessment_prompts.get(content_type, 
                                                  self.assessment_prompts[ContentType.GENERAL_TEXT])
        
        # Truncate content if too long for LLM
        content_sample = content[:3000] if len(content) > 3000 else content
        
        # Create assessment prompt
        full_prompt = f"""{prompt_config.system_prompt}

{prompt_config.analysis_prompt}

CONTENT TO ANALYZE:
File: {file_path}
Type: {content_type.value}
Content: {content_sample}

Provide assessment as JSON with this structure:
{prompt_config.expected_json_schema}

Focus on identifying issues that would prevent this content from being suitable for production use in a professional software demonstration or showcase."""
        
        try:
            # Call LLM for assessment
            response = await llm_client.assess_content_quality(full_prompt, file_path)
            
            # Parse LLM response into issues
            llm_issues = self._parse_llm_assessment(response, file_path)
            issues.extend(llm_issues)
            
        except Exception as e:
            # LLM assessment failed, return empty list
            pass
        
        return issues
    
    def _parse_llm_assessment(self, response: Dict[str, Any], file_path: str) -> List[QualityIssue]:
        """Parse LLM assessment response into QualityIssue objects."""
        issues = []
        
        # Parse issues from LLM response
        for issue_data in response.get('issues', []):
            try:
                # Map category strings to enums
                category_map = {
                    'template_artifact': IssueCategory.TEMPLATE_ARTIFACT,
                    'content_quality': IssueCategory.CONTENT_QUALITY,
                    'completeness': IssueCategory.COMPLETENESS,
                    'professional_standards': IssueCategory.PROFESSIONAL_STANDARDS,
                    'visual_quality': IssueCategory.VISUAL_QUALITY,
                    'file_organization': IssueCategory.FILE_ORGANIZATION
                }
                
                severity_map = {
                    'critical': IssueSeverity.CRITICAL,
                    'major': IssueSeverity.MAJOR,
                    'minor': IssueSeverity.MINOR
                }
                
                category = category_map.get(
                    issue_data.get('category', '').lower(),
                    IssueCategory.CONTENT_QUALITY
                )
                
                severity = severity_map.get(
                    issue_data.get('severity', '').lower(),
                    IssueSeverity.MAJOR
                )
                
                issues.append(QualityIssue(
                    category=category,
                    severity=severity,
                    description=issue_data.get('description', 'LLM-identified quality issue'),
                    file_path=file_path,
                    suggestion=issue_data.get('suggestion', 'Review and improve content quality'),
                    confidence=0.9  # High confidence for LLM assessment
                ))
                
            except Exception as e:
                # Skip malformed issue data
                continue
        
        return issues
    
    def _determine_overall_rating(self, issues: List[QualityIssue]) -> IssueSeverity:
        """Determine overall content quality rating based on issues found."""
        if any(issue.severity == IssueSeverity.CRITICAL for issue in issues):
            return IssueSeverity.CRITICAL
        elif any(issue.severity == IssueSeverity.MAJOR for issue in issues):
            return IssueSeverity.MAJOR
        elif any(issue.severity == IssueSeverity.MINOR for issue in issues):
            return IssueSeverity.MINOR
        else:
            return IssueSeverity.ACCEPTABLE