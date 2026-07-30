"""
Debug artifact and conversational text identification system.

This module provides comprehensive detection of debug statements, development artifacts,
conversational AI language, and other non-production content that should not appear
in professional pipeline outputs.
"""

import re
from typing import List, Dict, Set, Tuple, Optional, Pattern
from dataclasses import dataclass
from enum import Enum

from ..core.quality_assessment import QualityIssue, IssueCategory, IssueSeverity


class ArtifactType(Enum):
    """Types of debug/development artifacts."""
    DEBUG_STATEMENT = "debug_statement"
    STACK_TRACE = "stack_trace"
    CONSOLE_OUTPUT = "console_output"
    CONVERSATIONAL_AI = "conversational_ai"
    DEVELOPMENT_COMMENT = "development_comment"
    TEST_DATA_MARKER = "test_data_marker"
    ERROR_MESSAGE = "error_message"
    LOGGING_STATEMENT = "logging_statement"
    DEVELOPMENT_PLACEHOLDER = "development_placeholder"
    META_COMMENTARY = "meta_commentary"


@dataclass
class ArtifactPattern:
    """Represents a debug artifact pattern with metadata."""
    
    pattern: Pattern[str]
    artifact_type: ArtifactType
    description: str
    severity: IssueSeverity
    confidence: float
    context_sensitive: bool = False
    

class DebugArtifactDetector:
    """
    Comprehensive detector for debug artifacts, conversational AI language,
    and development content that should not appear in production outputs.
    """
    
    def __init__(self):
        """Initialize debug artifact detector with comprehensive pattern library."""
        self.artifact_patterns = self._initialize_artifact_patterns()
        
        # Context patterns to help with accurate detection
        self.code_context_indicators = [
            r'(function|class|def|var|let|const|import|export)',
            r'(if|else|for|while|switch|try|catch)',
            r'[{};]',  # Code structure indicators
        ]
        
        # Whitelist patterns for legitimate use of certain terms
        self.whitelisted_contexts = [
            r'(?i)(class|function|variable)\s+name',  # "debug" as legitimate name
            r'(?i)debug\s+(mode|flag|option|setting)',  # Configuration context
            r'(?i)(about|regarding|concerning)\s+debug',  # Documentation context
        ]
    
    def _initialize_artifact_patterns(self) -> List[ArtifactPattern]:
        """Initialize comprehensive library of debug artifact patterns."""
        patterns = []
        
        # Debug Statements
        patterns.extend([
            ArtifactPattern(
                pattern=re.compile(r'(?i)\b(print|console\.log|echo|var_dump|dump|debug|trace)\s*\(', re.MULTILINE),
                artifact_type=ArtifactType.DEBUG_STATEMENT,
                description="Debug print/output statement",
                severity=IssueSeverity.MAJOR,
                confidence=0.9,
                context_sensitive=True
            ),
            ArtifactPattern(
                pattern=re.compile(r'(?i)\b(debugger|pdb\.set_trace|breakpoint)\s*\(', re.MULTILINE),
                artifact_type=ArtifactType.DEBUG_STATEMENT,
                description="Debugger breakpoint statement",
                severity=IssueSeverity.CRITICAL,
                confidence=0.95
            ),
            ArtifactPattern(
                pattern=re.compile(r'(?i)#\s*(debug|trace|temp|temporary|todo|fixme|hack)', re.MULTILINE),
                artifact_type=ArtifactType.DEVELOPMENT_COMMENT,
                description="Development comment marker",
                severity=IssueSeverity.MAJOR,
                confidence=0.85
            ),
        ])
        
        # Stack Traces and Error Messages
        patterns.extend([
            ArtifactPattern(
                pattern=re.compile(r'(?i)\b(traceback|stack\s+trace|backtrace):', re.MULTILINE),
                artifact_type=ArtifactType.STACK_TRACE,
                description="Stack trace header",
                severity=IssueSeverity.CRITICAL,
                confidence=0.95
            ),
            ArtifactPattern(
                pattern=re.compile(r'(?i)\b(error\s+on\s+line|fatal\s+error|exception\s+in):', re.MULTILINE),
                artifact_type=ArtifactType.ERROR_MESSAGE,
                description="Error message with location",
                severity=IssueSeverity.CRITICAL,
                confidence=0.9
            ),
            ArtifactPattern(
                pattern=re.compile(r'(?i)at\s+[\w\.]+\([^)]*:\d+:\d+\)', re.MULTILINE),
                artifact_type=ArtifactType.STACK_TRACE,
                description="Stack trace frame",
                severity=IssueSeverity.CRITICAL,
                confidence=0.9
            ),
            ArtifactPattern(
                pattern=re.compile(r'(?i)(warning|error):\s+[^\n]+\s+in\s+[^\s]+\.py', re.MULTILINE),
                artifact_type=ArtifactType.ERROR_MESSAGE,
                description="Python error with file reference",
                severity=IssueSeverity.CRITICAL,
                confidence=0.9
            ),
        ])
        
        # Console Output Artifacts
        patterns.extend([
            ArtifactPattern(
                pattern=re.compile(r'(?i)^\$\s+[a-zA-Z][\w\-]*(\s+[^\n]*)?$', re.MULTILINE),
                artifact_type=ArtifactType.CONSOLE_OUTPUT,
                description="Shell command prompt",
                severity=IssueSeverity.MAJOR,
                confidence=0.8
            ),
            ArtifactPattern(
                pattern=re.compile(r'(?i)>>>\s+[^\n]+$', re.MULTILINE),
                artifact_type=ArtifactType.CONSOLE_OUTPUT,
                description="Python interactive prompt",
                severity=IssueSeverity.MAJOR,
                confidence=0.9
            ),
            ArtifactPattern(
                pattern=re.compile(r'(?i)\[INFO\]|\[DEBUG\]|\[WARN\]|\[ERROR\]|\[TRACE\]', re.MULTILINE),
                artifact_type=ArtifactType.LOGGING_STATEMENT,
                description="Logging level indicator",
                severity=IssueSeverity.MAJOR,
                confidence=0.85
            ),
        ])
        
        # Conversational AI Artifacts
        patterns.extend([
            ArtifactPattern(
                pattern=re.compile(r'(?i)\b(certainly!?|of\s+course!?|absolutely!?|definitely!?)\b', re.MULTILINE),
                artifact_type=ArtifactType.CONVERSATIONAL_AI,
                description="AI conversational affirmation",
                severity=IssueSeverity.MAJOR,
                confidence=0.9
            ),
            ArtifactPattern(
                pattern=re.compile(r'(?i)\b(here\s+(?:is|are|you\s+go)|here\'?s\s+(?:the|your|a|an))\b', re.MULTILINE),
                artifact_type=ArtifactType.CONVERSATIONAL_AI,
                description="AI presentation phrase",
                severity=IssueSeverity.MAJOR,
                confidence=0.85
            ),
            ArtifactPattern(
                pattern=re.compile(r'(?i)\b(i\s+hope\s+this\s+helps?|hope\s+this\s+helps?)\b', re.MULTILINE),
                artifact_type=ArtifactType.CONVERSATIONAL_AI,
                description="AI helpful closing phrase",
                severity=IssueSeverity.MAJOR,
                confidence=0.95
            ),
            ArtifactPattern(
                pattern=re.compile(r'(?i)\b(let\s+me\s+(?:help|assist|show|explain|demonstrate))\b', re.MULTILINE),
                artifact_type=ArtifactType.CONVERSATIONAL_AI,
                description="AI assistance offer",
                severity=IssueSeverity.MAJOR,
                confidence=0.9
            ),
            ArtifactPattern(
                pattern=re.compile(r'(?i)\b(as\s+an\s+ai|i\'?m\s+an\s+ai|ai\s+(?:language\s+)?model)\b', re.MULTILINE),
                artifact_type=ArtifactType.CONVERSATIONAL_AI,
                description="AI self-identification",
                severity=IssueSeverity.CRITICAL,
                confidence=0.95
            ),
            ArtifactPattern(
                pattern=re.compile(r'(?i)\b(i\s+(?:don\'?t|can\'?t|cannot)\s+(?:access|see|view|browse|visit))\b', re.MULTILINE),
                artifact_type=ArtifactType.CONVERSATIONAL_AI,
                description="AI capability limitation",
                severity=IssueSeverity.CRITICAL,
                confidence=0.95
            ),
            ArtifactPattern(
                pattern=re.compile(r'(?i)\b(my\s+(?:knowledge\s+cutoff|training\s+data)|knowledge\s+cutoff)\b', re.MULTILINE),
                artifact_type=ArtifactType.CONVERSATIONAL_AI,
                description="AI knowledge limitation reference",
                severity=IssueSeverity.CRITICAL,
                confidence=0.95
            ),
            ArtifactPattern(
                pattern=re.compile(r'(?i)\b(i\s+(?:apologize|sorry)\s+for\s+(?:any|the|that))\b', re.MULTILINE),
                artifact_type=ArtifactType.CONVERSATIONAL_AI,
                description="AI apology phrase",
                severity=IssueSeverity.MAJOR,
                confidence=0.9
            ),
        ])
        
        # Development Comments and Meta-Commentary
        patterns.extend([
            ArtifactPattern(
                pattern=re.compile(r'(?i)\b(note\s+that|please\s+note|important\s+note)\b', re.MULTILINE),
                artifact_type=ArtifactType.META_COMMENTARY,
                description="Meta-commentary note",
                severity=IssueSeverity.MINOR,
                confidence=0.7
            ),
            ArtifactPattern(
                pattern=re.compile(r'(?i)\b(as\s+(?:mentioned|noted|discussed)\s+(?:above|below|earlier|previously))\b', re.MULTILINE),
                artifact_type=ArtifactType.META_COMMENTARY,
                description="Reference to other content",
                severity=IssueSeverity.MINOR,
                confidence=0.7
            ),
            ArtifactPattern(
                pattern=re.compile(r'(?i)\b((?:this|the)\s+(?:above|following)\s+(?:shows?|demonstrates?|illustrates?))\b', re.MULTILINE),
                artifact_type=ArtifactType.META_COMMENTARY,
                description="Instructional reference",
                severity=IssueSeverity.MINOR,
                confidence=0.7
            ),
        ])
        
        # Test Data and Development Placeholders
        patterns.extend([
            ArtifactPattern(
                pattern=re.compile(r'(?i)\b(test\s+(?:data|content|output|file|case)|sample\s+(?:data|content|output))\b', re.MULTILINE),
                artifact_type=ArtifactType.TEST_DATA_MARKER,
                description="Test or sample data marker",
                severity=IssueSeverity.MAJOR,
                confidence=0.8
            ),
            ArtifactPattern(
                pattern=re.compile(r'(?i)\b(mock\s+(?:data|content|output|response)|dummy\s+(?:data|content|text|values?))\b', re.MULTILINE),
                artifact_type=ArtifactType.TEST_DATA_MARKER,
                description="Mock or dummy data marker",
                severity=IssueSeverity.MAJOR,
                confidence=0.85
            ),
            ArtifactPattern(
                pattern=re.compile(r'(?i)\b(placeholder|lorem\s+ipsum|coming\s+soon|work\s+in\s+progress)\b', re.MULTILINE),
                artifact_type=ArtifactType.DEVELOPMENT_PLACEHOLDER,
                description="Development placeholder content",
                severity=IssueSeverity.MAJOR,
                confidence=0.9
            ),
            ArtifactPattern(
                pattern=re.compile(r'(?i)\b(to\s+be\s+(?:determined|completed|implemented|done)|tbd|todo)\b', re.MULTILINE),
                artifact_type=ArtifactType.DEVELOPMENT_PLACEHOLDER,
                description="Incomplete development marker",
                severity=IssueSeverity.MAJOR,
                confidence=0.9
            ),
        ])
        
        # Processing and Generation Artifacts
        patterns.extend([
            ArtifactPattern(
                pattern=re.compile(r'(?i)\b(processing\.\.\.|loading\.\.\.|generating\.\.\.|calculating\.\.\.)\b', re.MULTILINE),
                artifact_type=ArtifactType.DEVELOPMENT_PLACEHOLDER,
                description="Processing status indicator",
                severity=IssueSeverity.MAJOR,
                confidence=0.85
            ),
            ArtifactPattern(
                pattern=re.compile(r'(?i)\b(analysis\s+in\s+progress|calculation\s+pending|waiting\s+for\s+\w+)\b', re.MULTILINE),
                artifact_type=ArtifactType.DEVELOPMENT_PLACEHOLDER,
                description="Pending operation indicator",
                severity=IssueSeverity.MAJOR,
                confidence=0.85
            ),
        ])
        
        # Instructional and Tutorial Artifacts  
        patterns.extend([
            ArtifactPattern(
                pattern=re.compile(r'(?i)\b(step\s+\d+:|first,?\s+|second,?\s+|third,?\s+|finally,?\s+)\b', re.MULTILINE),
                artifact_type=ArtifactType.META_COMMENTARY,
                description="Step-by-step instruction marker",
                severity=IssueSeverity.MINOR,
                confidence=0.75
            ),
            ArtifactPattern(
                pattern=re.compile(r'(?i)\b(in\s+(?:this|the)\s+(?:example|tutorial|guide|walkthrough))\b', re.MULTILINE),
                artifact_type=ArtifactType.META_COMMENTARY,
                description="Tutorial context reference",
                severity=IssueSeverity.MINOR,
                confidence=0.8
            ),
            ArtifactPattern(
                pattern=re.compile(r'(?i)\b(for\s+(?:this|your)\s+(?:task|project|use\s+case|assignment))\b', re.MULTILINE),
                artifact_type=ArtifactType.META_COMMENTARY,
                description="Task-specific instruction",
                severity=IssueSeverity.MINOR,
                confidence=0.75
            ),
        ])
        
        return patterns
    
    def detect_debug_artifacts(self, content: str, file_path: str = "") -> List[QualityIssue]:
        """
        Detect debug artifacts, conversational AI language, and development
        content using comprehensive pattern matching.
        """
        issues = []
        
        for pattern_obj in self.artifact_patterns:
            matches = pattern_obj.pattern.finditer(content)
            
            for match in matches:
                matched_text = match.group(0)
                match_start = match.start()
                
                # Apply context-sensitive filtering if enabled
                if pattern_obj.context_sensitive:
                    if self._is_whitelisted_context(content, match_start, matched_text):
                        continue
                    
                    # Adjust confidence based on context
                    confidence = self._calculate_context_confidence(
                        content, match_start, matched_text, pattern_obj.confidence
                    )
                else:
                    confidence = pattern_obj.confidence
                
                # Skip if confidence is too low
                if confidence < 0.5:
                    continue
                
                line_number = content[:match_start].count('\n') + 1
                
                # Generate contextual suggestion based on artifact type
                suggestion = self._generate_suggestion(pattern_obj.artifact_type, matched_text)
                
                issues.append(QualityIssue(
                    category=IssueCategory.CONTENT_QUALITY,
                    severity=pattern_obj.severity,
                    description=f"{pattern_obj.description}: '{matched_text.strip()}'",
                    file_path=file_path,
                    line_number=line_number,
                    suggestion=suggestion,
                    confidence=confidence
                ))
        
        # Additional context-aware detection
        additional_issues = self._detect_context_specific_artifacts(content, file_path)
        issues.extend(additional_issues)
        
        return self._deduplicate_issues(issues)
    
    def _is_whitelisted_context(self, content: str, match_start: int, matched_text: str) -> bool:
        """Check if the match appears in a whitelisted context."""
        # Get surrounding context (50 characters before and after)
        context_start = max(0, match_start - 50)
        context_end = min(len(content), match_start + len(matched_text) + 50)
        context = content[context_start:context_end]
        
        # Check against whitelist patterns
        for whitelist_pattern in self.whitelisted_contexts:
            if re.search(whitelist_pattern, context, re.IGNORECASE):
                return True
        
        return False
    
    def _calculate_context_confidence(self, content: str, match_start: int, matched_text: str, base_confidence: float) -> float:
        """Calculate confidence based on surrounding context."""
        confidence = base_confidence
        
        # Get surrounding context
        context_start = max(0, match_start - 100)
        context_end = min(len(content), match_start + len(matched_text) + 100)
        context = content[context_start:context_end]
        
        # Reduce confidence if in code-like context
        if any(re.search(pattern, context) for pattern in self.code_context_indicators):
            confidence *= 0.7
        
        # Increase confidence if in documentation context
        if any(marker in context for marker in ['#', '**', '*', '```', '---']):
            confidence *= 1.1
        
        # Reduce confidence if term appears to be part of legitimate text
        if re.search(r'\b(about|regarding|concerning|documentation)\s+' + re.escape(matched_text.lower()), context, re.IGNORECASE):
            confidence *= 0.5
        
        return min(1.0, confidence)
    
    def _generate_suggestion(self, artifact_type: ArtifactType, matched_text: str) -> str:
        """Generate contextual suggestion based on artifact type."""
        suggestions = {
            ArtifactType.DEBUG_STATEMENT: f"Remove debug statement '{matched_text.strip()}' from production output",
            ArtifactType.STACK_TRACE: "Remove stack trace information from production output",
            ArtifactType.CONSOLE_OUTPUT: f"Remove console/terminal output '{matched_text.strip()}' from production content",
            ArtifactType.CONVERSATIONAL_AI: f"Replace conversational AI language '{matched_text.strip()}' with direct, professional content",
            ArtifactType.DEVELOPMENT_COMMENT: f"Remove development comment '{matched_text.strip()}' from production documentation",
            ArtifactType.TEST_DATA_MARKER: f"Replace test data marker '{matched_text.strip()}' with production data",
            ArtifactType.ERROR_MESSAGE: "Remove error messages from production output",
            ArtifactType.LOGGING_STATEMENT: f"Remove logging statement '{matched_text.strip()}' from production content",
            ArtifactType.DEVELOPMENT_PLACEHOLDER: f"Complete development placeholder '{matched_text.strip()}' with actual content",
            ArtifactType.META_COMMENTARY: f"Remove meta-commentary '{matched_text.strip()}' and provide direct information"
        }
        
        return suggestions.get(artifact_type, "Review and improve content for production readiness")
    
    def _detect_context_specific_artifacts(self, content: str, file_path: str) -> List[QualityIssue]:
        """Detect artifacts that require contextual analysis."""
        issues = []
        
        # Detect common names that suggest test data
        test_name_patterns = [
            r'(?i)\b(john\s+doe|jane\s+smith|test\s*user\d*|example\s+user)\b',
            r'(?i)\b(alice|bob|charlie|dave)\b(?=\s+(?:smith|jones|brown|wilson))',
            r'(?i)\b(user\d+|test\d+|sample\d+|example\d+)\b'
        ]
        
        for pattern in test_name_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                line_number = content[:match.start()].count('\n') + 1
                issues.append(QualityIssue(
                    category=IssueCategory.CONTENT_QUALITY,
                    severity=IssueSeverity.MAJOR,
                    description=f"Test/example name detected: '{match.group(0)}'",
                    file_path=file_path,
                    line_number=line_number,
                    suggestion="Replace test names with production-appropriate data or anonymized references",
                    confidence=0.8
                ))
        
        # Detect common test email addresses
        test_email_pattern = r'(?i)\b[a-zA-Z0-9._%+-]+@(?:example|test|sample|demo)\.(?:com|org|net)\b'
        matches = re.finditer(test_email_pattern, content)
        for match in matches:
            line_number = content[:match.start()].count('\n') + 1
            issues.append(QualityIssue(
                category=IssueCategory.CONTENT_QUALITY,
                severity=IssueSeverity.MAJOR,
                description=f"Test email address detected: '{match.group(0)}'",
                file_path=file_path,
                line_number=line_number,
                suggestion="Replace test email addresses with production examples or placeholder format",
                confidence=0.9
            ))
        
        # Detect URL artifacts that suggest development/testing
        dev_url_patterns = [
            r'(?i)\b(?:https?://)?(?:localhost|127\.0\.0\.1|0\.0\.0\.0)(?::\d+)?(?:/[^\s]*)?',
            r'(?i)\b(?:https?://)?[a-zA-Z0-9.-]*(?:\.local|\.test|\.dev|\.localhost)(?::\d+)?(?:/[^\s]*)?'
        ]
        
        for pattern in dev_url_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                line_number = content[:match.start()].count('\n') + 1
                issues.append(QualityIssue(
                    category=IssueCategory.CONTENT_QUALITY,
                    severity=IssueSeverity.MAJOR,
                    description=f"Development URL detected: '{match.group(0)}'",
                    file_path=file_path,
                    line_number=line_number,
                    suggestion="Replace development URLs with production examples or generic placeholders",
                    confidence=0.85
                ))
        
        return issues
    
    def _deduplicate_issues(self, issues: List[QualityIssue]) -> List[QualityIssue]:
        """Remove duplicate issues based on description and line number."""
        seen = set()
        deduplicated = []
        
        for issue in issues:
            key = (issue.description, issue.line_number, issue.file_path)
            if key not in seen:
                seen.add(key)
                deduplicated.append(issue)
        
        return deduplicated
    
    def get_artifact_summary(self, issues: List[QualityIssue]) -> Dict[ArtifactType, int]:
        """Generate summary of detected artifacts by type."""
        summary = {artifact_type: 0 for artifact_type in ArtifactType}
        
        for issue in issues:
            # Parse artifact type from issue description
            for artifact_type in ArtifactType:
                type_keywords = {
                    ArtifactType.DEBUG_STATEMENT: ['debug', 'print', 'console.log'],
                    ArtifactType.STACK_TRACE: ['stack trace', 'traceback'],
                    ArtifactType.CONSOLE_OUTPUT: ['command prompt', 'interactive prompt'],
                    ArtifactType.CONVERSATIONAL_AI: ['conversational', 'ai artifact'],
                    ArtifactType.DEVELOPMENT_COMMENT: ['development comment'],
                    ArtifactType.TEST_DATA_MARKER: ['test', 'sample', 'mock', 'dummy'],
                    ArtifactType.ERROR_MESSAGE: ['error message'],
                    ArtifactType.LOGGING_STATEMENT: ['logging'],
                    ArtifactType.DEVELOPMENT_PLACEHOLDER: ['placeholder', 'processing'],
                    ArtifactType.META_COMMENTARY: ['meta-commentary', 'instructional']
                }
                
                keywords = type_keywords.get(artifact_type, [])
                if any(keyword in issue.description.lower() for keyword in keywords):
                    summary[artifact_type] += 1
                    break
        
        return summary