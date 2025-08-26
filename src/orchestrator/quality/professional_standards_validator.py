"""
Professional standards validation system.

This module validates content against professional standards for documentation,
code quality, business communication, and production-ready presentation.
Ensures consistency and appropriateness for professional software demonstrations.
"""

import re
from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from ..core.quality_assessment import QualityIssue, IssueCategory, IssueSeverity


class ProfessionalStandard(Enum):
    """Types of professional standards."""
    DOCUMENTATION_COMPLETENESS = "documentation_completeness"
    BUSINESS_COMMUNICATION = "business_communication"
    TECHNICAL_ACCURACY = "technical_accuracy"
    CONSISTENCY = "consistency"
    ACCESSIBILITY = "accessibility"
    BRAND_COMPLIANCE = "brand_compliance"
    DATA_PRESENTATION = "data_presentation"
    CODE_QUALITY = "code_quality"


@dataclass
class StandardCheck:
    """Represents a professional standard check."""
    
    standard: ProfessionalStandard
    description: str
    check_function: str  # Name of method to call
    severity: IssueSeverity
    applies_to_types: Set[str]  # File extensions or content types this applies to


class ProfessionalStandardsValidator:
    """
    Validates content against professional standards for production readiness,
    ensuring consistency, completeness, and appropriate presentation quality.
    """
    
    def __init__(self):
        """Initialize professional standards validator."""
        self.standard_checks = self._initialize_standard_checks()
        
        # Professional terminology and style guidelines
        self.professional_vocabulary = {
            # Prefer professional alternatives
            'utilize': 'use',
            'leverage': 'use',
            'facilitate': 'enable',
            'implement': 'create', 
            'functionality': 'features',
            'utilize': 'use',
        }
        
        # Casual language to flag
        self.casual_expressions = [
            r'(?i)\b(gonna|wanna|kinda|sorta|yeah|ok|cool|awesome)\b',
            r'(?i)\b(super\s+(?:easy|simple|quick|fast))\b',
            r'(?i)\b(really\s+(?:good|great|nice|neat))\b',
            r'(?i)\b(pretty\s+(?:good|nice|cool|neat))\b',
        ]
        
        # Business communication standards
        self.business_tone_violations = [
            r'(?i)\b(love\s+(?:this|that|it)|hate\s+(?:this|that|it))\b',
            r'(?i)\b(excited|thrilled|pumped|stoked)\b',
            r'(?i)\b(amazing|incredible|fantastic|mind-blowing)\b',
            r'(?i)\b(tons\s+of|loads\s+of|heaps\s+of)\b',
        ]
        
        # Technical accuracy patterns
        self.technical_precision_issues = [
            r'(?i)\b(probably|maybe|perhaps|likely|might\s+be)\b',
            r'(?i)\b(i\s+think|i\s+believe|i\s+guess|i\s+assume)\b',
            r'(?i)\b(sort\s+of|kind\s+of|more\s+or\s+less)\b',
            r'(?i)\b(around|about|approximately)\s+\d+%',
        ]
    
    def _initialize_standard_checks(self) -> List[StandardCheck]:
        """Initialize comprehensive professional standard checks."""
        return [
            StandardCheck(
                standard=ProfessionalStandard.DOCUMENTATION_COMPLETENESS,
                description="Documentation sections are complete and comprehensive",
                check_function="_check_documentation_completeness",
                severity=IssueSeverity.MAJOR,
                applies_to_types={'.md', '.txt', '.rst'}
            ),
            StandardCheck(
                standard=ProfessionalStandard.BUSINESS_COMMUNICATION,
                description="Content uses appropriate business communication tone",
                check_function="_check_business_communication",
                severity=IssueSeverity.MINOR,
                applies_to_types={'.md', '.txt', '.rst', '.html'}
            ),
            StandardCheck(
                standard=ProfessionalStandard.TECHNICAL_ACCURACY,
                description="Technical content is precise and accurate",
                check_function="_check_technical_accuracy", 
                severity=IssueSeverity.MAJOR,
                applies_to_types={'.md', '.txt', '.py', '.js', '.html'}
            ),
            StandardCheck(
                standard=ProfessionalStandard.CONSISTENCY,
                description="Content maintains consistent terminology and style",
                check_function="_check_consistency",
                severity=IssueSeverity.MINOR,
                applies_to_types={'.md', '.txt', '.csv', '.json'}
            ),
            StandardCheck(
                standard=ProfessionalStandard.DATA_PRESENTATION,
                description="Data is presented professionally with proper formatting",
                check_function="_check_data_presentation",
                severity=IssueSeverity.MAJOR,
                applies_to_types={'.csv', '.json', '.xlsx'}
            ),
            StandardCheck(
                standard=ProfessionalStandard.ACCESSIBILITY,
                description="Content follows accessibility best practices",
                check_function="_check_accessibility",
                severity=IssueSeverity.MINOR,
                applies_to_types={'.md', '.html', '.rst'}
            ),
        ]
    
    def validate_professional_standards(self, content: str, file_path: str = "") -> List[QualityIssue]:
        """
        Validate content against comprehensive professional standards.
        
        Args:
            content: Content to validate
            file_path: Path to file being validated
            
        Returns:
            List of quality issues found
        """
        issues = []
        file_ext = Path(file_path).suffix.lower() if file_path else ''
        
        # Run applicable standard checks
        for check in self.standard_checks:
            if not file_ext or file_ext in check.applies_to_types:
                check_method = getattr(self, check.check_function)
                check_issues = check_method(content, file_path)
                issues.extend(check_issues)
        
        # Run general professional language checks
        language_issues = self._check_professional_language(content, file_path)
        issues.extend(language_issues)
        
        return issues
    
    def _check_documentation_completeness(self, content: str, file_path: str) -> List[QualityIssue]:
        """Check for documentation completeness and structure."""
        issues = []
        
        # Check for essential documentation sections
        if file_path.endswith('.md'):
            required_sections = self._get_required_sections(content, file_path)
            missing_sections = []
            
            for section in required_sections:
                if not re.search(rf'(?i)#{1,6}\s*{re.escape(section)}', content):
                    missing_sections.append(section)
            
            if missing_sections:
                issues.append(QualityIssue(
                    category=IssueCategory.COMPLETENESS,
                    severity=IssueSeverity.MAJOR,
                    description=f"Missing documentation sections: {', '.join(missing_sections)}",
                    file_path=file_path,
                    suggestion="Add missing sections to complete documentation structure"
                ))
        
        # Check for incomplete sections
        incomplete_patterns = [
            r'(?i)#{1,6}\s+[^\n]+\n\s*(?:$|#)',  # Header with no content
            r'(?i)#{1,6}\s+(?:todo|tbd|coming\s+soon|placeholder)',  # Placeholder headers
        ]
        
        for pattern in incomplete_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                line_number = content[:match.start()].count('\n') + 1
                issues.append(QualityIssue(
                    category=IssueCategory.COMPLETENESS,
                    severity=IssueSeverity.MAJOR,
                    description=f"Incomplete documentation section: {match.group(0).strip()}",
                    file_path=file_path,
                    line_number=line_number,
                    suggestion="Complete documentation section with appropriate content"
                ))
        
        return issues
    
    def _get_required_sections(self, content: str, file_path: str) -> List[str]:
        """Determine required sections based on document type and content."""
        file_name = Path(file_path).stem.lower()
        
        # README files
        if 'readme' in file_name:
            return ['Overview', 'Installation', 'Usage', 'Examples']
        
        # API documentation
        if 'api' in file_name or 'reference' in file_name:
            return ['Methods', 'Parameters', 'Response', 'Examples']
        
        # Tutorial/guide files
        if any(word in file_name for word in ['tutorial', 'guide', 'howto']):
            return ['Prerequisites', 'Steps', 'Examples', 'Troubleshooting']
        
        # General documentation
        return ['Overview', 'Details', 'Examples']
    
    def _check_business_communication(self, content: str, file_path: str) -> List[QualityIssue]:
        """Check for appropriate business communication tone."""
        issues = []
        
        # Check for casual expressions
        for pattern in self.casual_expressions:
            matches = re.finditer(pattern, content)
            for match in matches:
                line_number = content[:match.start()].count('\n') + 1
                issues.append(QualityIssue(
                    category=IssueCategory.PROFESSIONAL_STANDARDS,
                    severity=IssueSeverity.MINOR,
                    description=f"Casual language detected: '{match.group(0)}'",
                    file_path=file_path,
                    line_number=line_number,
                    suggestion="Replace with more formal business language"
                ))
        
        # Check for business tone violations
        for pattern in self.business_tone_violations:
            matches = re.finditer(pattern, content)
            for match in matches:
                line_number = content[:match.start()].count('\n') + 1
                issues.append(QualityIssue(
                    category=IssueCategory.PROFESSIONAL_STANDARDS,
                    severity=IssueSeverity.MINOR,
                    description=f"Inappropriate business tone: '{match.group(0)}'",
                    file_path=file_path,
                    line_number=line_number,
                    suggestion="Use neutral, professional language appropriate for business context"
                ))
        
        return issues
    
    def _check_technical_accuracy(self, content: str, file_path: str) -> List[QualityIssue]:
        """Check for technical precision and accuracy."""
        issues = []
        
        # Check for imprecise language
        for pattern in self.technical_precision_issues:
            matches = re.finditer(pattern, content)
            for match in matches:
                line_number = content[:match.start()].count('\n') + 1
                issues.append(QualityIssue(
                    category=IssueCategory.CONTENT_QUALITY,
                    severity=IssueSeverity.MINOR,
                    description=f"Imprecise technical language: '{match.group(0)}'",
                    file_path=file_path,
                    line_number=line_number,
                    suggestion="Use precise, definitive language for technical accuracy"
                ))
        
        # Check for undefined acronyms (first occurrence should be spelled out)
        acronym_pattern = r'\b[A-Z]{2,}\b'
        acronyms = re.findall(acronym_pattern, content)
        
        for acronym in set(acronyms):  # Remove duplicates
            first_occurrence = content.find(acronym)
            if first_occurrence != -1:
                # Check if it's defined (has parentheses around it or before it)
                context_before = content[max(0, first_occurrence-50):first_occurrence]
                context_after = content[first_occurrence:first_occurrence+len(acronym)+50]
                
                if not (re.search(r'\([^)]*$', context_before) or 
                       re.search(r'^\s*\(', context_after)):
                    line_number = content[:first_occurrence].count('\n') + 1
                    issues.append(QualityIssue(
                        category=IssueCategory.PROFESSIONAL_STANDARDS,
                        severity=IssueSeverity.MINOR,
                        description=f"Undefined acronym: {acronym}",
                        file_path=file_path,
                        line_number=line_number,
                        suggestion=f"Define acronym on first use: 'Full Term ({acronym})'"
                    ))
        
        return issues
    
    def _check_consistency(self, content: str, file_path: str) -> List[QualityIssue]:
        """Check for consistent terminology and style."""
        issues = []
        
        # Check for inconsistent terminology
        terminology_variations = {
            'email': ['e-mail', 'Email', 'E-mail'],
            'website': ['web site', 'Website', 'Web site'],  
            'setup': ['set-up', 'Set up', 'Set-up'],
            'username': ['user name', 'Username', 'User name'],
        }
        
        for preferred, variations in terminology_variations.items():
            for variation in variations:
                if variation in content and preferred not in content.lower():
                    # Find first occurrence for line number
                    match = re.search(re.escape(variation), content)
                    if match:
                        line_number = content[:match.start()].count('\n') + 1
                        issues.append(QualityIssue(
                            category=IssueCategory.PROFESSIONAL_STANDARDS,
                            severity=IssueSeverity.MINOR,
                            description=f"Inconsistent terminology: '{variation}' (prefer '{preferred}')",
                            file_path=file_path,
                            line_number=line_number,
                            suggestion=f"Use consistent terminology: '{preferred}' instead of '{variation}'"
                        ))
        
        # Check for mixed date formats
        date_formats = [
            r'\d{1,2}/\d{1,2}/\d{4}',  # MM/DD/YYYY
            r'\d{4}-\d{2}-\d{2}',      # YYYY-MM-DD  
            r'\d{1,2}-\d{1,2}-\d{4}',  # MM-DD-YYYY
        ]
        
        found_formats = []
        for pattern in date_formats:
            if re.search(pattern, content):
                found_formats.append(pattern)
        
        if len(found_formats) > 1:
            issues.append(QualityIssue(
                category=IssueCategory.PROFESSIONAL_STANDARDS,
                severity=IssueSeverity.MINOR,
                description="Mixed date formats detected",
                file_path=file_path,
                suggestion="Use consistent date format throughout document (recommend ISO 8601: YYYY-MM-DD)"
            ))
        
        return issues
    
    def _check_data_presentation(self, content: str, file_path: str) -> List[QualityIssue]:
        """Check for professional data presentation standards."""
        issues = []
        
        if file_path.endswith('.csv'):
            lines = content.strip().split('\n')
            if not lines:
                return issues
            
            # Check for professional column headers
            headers = lines[0].split(',') if lines else []
            
            for i, header in enumerate(headers):
                header = header.strip().strip('"\'')
                
                # Check for poor header naming
                if header.lower() in ['col1', 'col2', 'column1', 'column2', 'field1', 'field2']:
                    issues.append(QualityIssue(
                        category=IssueCategory.PROFESSIONAL_STANDARDS,
                        severity=IssueSeverity.MAJOR,
                        description=f"Generic column header: '{header}'",
                        file_path=file_path,
                        line_number=1,
                        suggestion="Use descriptive, meaningful column headers"
                    ))
                
                # Check for inconsistent header formatting
                if header and (header.isupper() or header.islower()) and len(headers) > 1:
                    other_headers = [h.strip().strip('"\'') for j, h in enumerate(headers) if j != i]
                    if other_headers and not all(h.isupper() == header.isupper() for h in other_headers):
                        issues.append(QualityIssue(
                            category=IssueCategory.PROFESSIONAL_STANDARDS,
                            severity=IssueSeverity.MINOR,
                            description=f"Inconsistent header case: '{header}'",
                            file_path=file_path,
                            line_number=1,
                            suggestion="Use consistent capitalization for all headers"
                        ))
        
        elif file_path.endswith('.json'):
            # Check for professional JSON key naming
            try:
                import json
                data = json.loads(content)
                self._check_json_key_standards(data, issues, file_path)
            except json.JSONDecodeError:
                pass  # JSON validation happens elsewhere
        
        return issues
    
    def _check_json_key_standards(self, data: Any, issues: List[QualityIssue], file_path: str, path: str = "") -> None:
        """Recursively check JSON keys for professional standards."""
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                
                # Check for poor key naming
                if key.lower() in ['key1', 'key2', 'field1', 'field2', 'item1', 'item2']:
                    issues.append(QualityIssue(
                        category=IssueCategory.PROFESSIONAL_STANDARDS,
                        severity=IssueSeverity.MAJOR,
                        description=f"Generic JSON key: '{key}'",
                        file_path=file_path,
                        suggestion="Use descriptive, meaningful key names"
                    ))
                
                # Check for inconsistent key naming convention
                if '_' in key and any(k for k in data.keys() if k != key and '-' in k):
                    issues.append(QualityIssue(
                        category=IssueCategory.PROFESSIONAL_STANDARDS,
                        severity=IssueSeverity.MINOR,
                        description=f"Mixed key naming conventions: '{key}' (underscores vs hyphens)",
                        file_path=file_path,
                        suggestion="Use consistent naming convention (prefer snake_case or kebab-case)"
                    ))
                
                # Recurse into nested structures
                self._check_json_key_standards(value, issues, file_path, current_path)
        
        elif isinstance(data, list):
            for i, item in enumerate(data):
                self._check_json_key_standards(item, issues, file_path, f"{path}[{i}]")
    
    def _check_accessibility(self, content: str, file_path: str) -> List[QualityIssue]:
        """Check for accessibility best practices."""
        issues = []
        
        if file_path.endswith('.md') or file_path.endswith('.html'):
            # Check for images without alt text (markdown)
            img_pattern = r'!\[([^\]]*)\]\([^)]+\)'
            matches = re.finditer(img_pattern, content)
            
            for match in matches:
                alt_text = match.group(1)
                if not alt_text.strip():
                    line_number = content[:match.start()].count('\n') + 1
                    issues.append(QualityIssue(
                        category=IssueCategory.PROFESSIONAL_STANDARDS,
                        severity=IssueSeverity.MINOR,
                        description="Image missing alt text (accessibility)",
                        file_path=file_path,
                        line_number=line_number,
                        suggestion="Add descriptive alt text for accessibility"
                    ))
            
            # Check for proper heading hierarchy
            headers = re.findall(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE)
            prev_level = 0
            
            for i, (hashes, title) in enumerate(headers):
                current_level = len(hashes)
                
                if current_level > prev_level + 1:
                    # Find line number
                    header_text = f"{hashes} {title}"
                    match = re.search(re.escape(header_text), content)
                    line_number = content[:match.start()].count('\n') + 1 if match else 0
                    
                    issues.append(QualityIssue(
                        category=IssueCategory.PROFESSIONAL_STANDARDS,
                        severity=IssueSeverity.MINOR,
                        description=f"Header hierarchy skip (accessibility): {hashes} {title}",
                        file_path=file_path,
                        line_number=line_number,
                        suggestion="Maintain proper heading hierarchy (don't skip levels)"
                    ))
                
                prev_level = current_level
        
        return issues
    
    def _check_professional_language(self, content: str, file_path: str) -> List[QualityIssue]:
        """Check for professional language usage."""
        issues = []
        
        # Check for buzzword overuse
        buzzwords = ['leverage', 'utilize', 'facilitate', 'synergy', 'paradigm', 'disruptive']
        buzzword_count = {}
        
        for buzzword in buzzwords:
            matches = re.findall(rf'(?i)\b{buzzword}\b', content)
            if len(matches) > 2:  # Allow some usage, but flag overuse
                buzzword_count[buzzword] = len(matches)
        
        for buzzword, count in buzzword_count.items():
            issues.append(QualityIssue(
                category=IssueCategory.PROFESSIONAL_STANDARDS,
                severity=IssueSeverity.MINOR,
                description=f"Overuse of buzzword '{buzzword}' ({count} times)",
                file_path=file_path,
                suggestion=f"Reduce usage of '{buzzword}' and use clearer alternatives"
            ))
        
        # Check for excessive use of superlatives
        superlatives = ['best', 'greatest', 'most', 'perfect', 'ultimate', 'revolutionary']
        superlative_pattern = r'(?i)\b(' + '|'.join(superlatives) + r')\b'
        matches = re.findall(superlative_pattern, content)
        
        if len(matches) > 5:
            issues.append(QualityIssue(
                category=IssueCategory.PROFESSIONAL_STANDARDS,
                severity=IssueSeverity.MINOR,
                description=f"Excessive use of superlatives ({len(matches)} instances)",
                file_path=file_path,
                suggestion="Use moderate language and focus on specific, measurable benefits"
            ))
        
        return issues
    
    def generate_professional_standards_summary(self, issues: List[QualityIssue]) -> Dict[str, Any]:
        """Generate summary of professional standards assessment."""
        standards_summary = {}
        
        # Count issues by standard type
        for standard in ProfessionalStandard:
            standards_summary[standard.value] = 0
        
        # Categorize issues
        for issue in issues:
            if issue.category == IssueCategory.PROFESSIONAL_STANDARDS:
                if 'casual language' in issue.description.lower() or 'business tone' in issue.description.lower():
                    standards_summary[ProfessionalStandard.BUSINESS_COMMUNICATION.value] += 1
                elif 'inconsistent' in issue.description.lower() or 'mixed' in issue.description.lower():
                    standards_summary[ProfessionalStandard.CONSISTENCY.value] += 1
                elif 'imprecise' in issue.description.lower() or 'technical' in issue.description.lower():
                    standards_summary[ProfessionalStandard.TECHNICAL_ACCURACY.value] += 1
                elif 'accessibility' in issue.description.lower():
                    standards_summary[ProfessionalStandard.ACCESSIBILITY.value] += 1
                elif 'generic' in issue.description.lower() or 'professional' in issue.description.lower():
                    standards_summary[ProfessionalStandard.DATA_PRESENTATION.value] += 1
                else:
                    # Default to general professional standards
                    standards_summary['general_professional'] = standards_summary.get('general_professional', 0) + 1
            
            elif issue.category == IssueCategory.COMPLETENESS:
                standards_summary[ProfessionalStandard.DOCUMENTATION_COMPLETENESS.value] += 1
            
            elif issue.category == IssueCategory.CONTENT_QUALITY and 'imprecise' in issue.description.lower():
                standards_summary[ProfessionalStandard.TECHNICAL_ACCURACY.value] += 1
        
        # Calculate overall professional readiness score
        total_issues = sum(standards_summary.values())
        professional_score = max(0, 100 - (total_issues * 10))  # Deduct 10 points per issue
        
        return {
            'standards_summary': standards_summary,
            'total_professional_issues': total_issues,
            'professional_readiness_score': professional_score,
            'production_ready': professional_score >= 80
        }