"""
Enhanced template artifact detection engine.

This module extends the basic template detection capabilities with support for
additional template systems, context-aware filtering, and advanced pattern matching
to reduce false positives while catching more sophisticated template artifacts.
"""

import re
import json
from typing import List, Dict, Set, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass

from ..core.quality_assessment import QualityIssue, IssueCategory, IssueSeverity


@dataclass
class TemplatePattern:
    """Represents a template pattern with metadata."""
    
    pattern: str
    template_system: str
    description: str
    severity: IssueSeverity = IssueSeverity.CRITICAL
    confidence: float = 1.0
    context_aware: bool = True


class EnhancedTemplateDetector:
    """
    Advanced template artifact detection with support for multiple template systems
    and context-aware filtering to reduce false positives.
    """
    
    # Extended template patterns covering more systems
    ENHANCED_TEMPLATE_PATTERNS = [
        # Jinja2/Django Templates
        TemplatePattern(
            pattern=r'\{\{[^}]*\w+[^}]*\}\}',
            template_system="Jinja2/Django",
            description="Jinja2/Django template variable",
            confidence=0.95
        ),
        TemplatePattern(
            pattern=r'\{\%[^%]*\w+[^%]*\%\}',
            template_system="Jinja2/Django", 
            description="Jinja2/Django template statement",
            confidence=0.95
        ),
        
        # Handlebars/Mustache Templates
        TemplatePattern(
            pattern=r'\{\{[^}]*\w+[^}]*\}\}',
            template_system="Handlebars/Mustache",
            description="Handlebars/Mustache template variable",
            confidence=0.9
        ),
        TemplatePattern(
            pattern=r'\{\{\{[^}]*\w+[^}]*\}\}\}',
            template_system="Handlebars",
            description="Handlebars unescaped variable",
            confidence=0.95
        ),
        TemplatePattern(
            pattern=r'\{\{#[^}]*\w+[^}]*\}\}',
            template_system="Handlebars",
            description="Handlebars block helper",
            confidence=0.95
        ),
        TemplatePattern(
            pattern=r'\{\{/[^}]*\w+[^}]*\}\}',
            template_system="Handlebars",
            description="Handlebars closing block",
            confidence=0.95
        ),
        
        # Shell/JavaScript/ES6 Templates
        TemplatePattern(
            pattern=r'\$\{[^}]*\w+[^}]*\}',
            template_system="Shell/JavaScript",
            description="Shell/JavaScript variable substitution",
            confidence=0.9
        ),
        TemplatePattern(
            pattern=r'`[^`]*\$\{[^}]*\w+[^}]*\}[^`]*`',
            template_system="JavaScript ES6",
            description="JavaScript template literal",
            confidence=0.95
        ),
        
        # ERB (Ruby) Templates
        TemplatePattern(
            pattern=r'<%=\s*[^%]*\w+[^%]*\s*%>',
            template_system="ERB",
            description="ERB output expression",
            confidence=0.95
        ),
        TemplatePattern(
            pattern=r'<%\s*[^%]*\w+[^%]*\s*%>',
            template_system="ERB",
            description="ERB code expression",
            confidence=0.95
        ),
        
        # PHP Templates
        TemplatePattern(
            pattern=r'<\?php\s+echo\s+\$\w+\s*\?>',
            template_system="PHP",
            description="PHP variable echo",
            confidence=0.95
        ),
        TemplatePattern(
            pattern=r'<\?=\s*\$\w+\s*\?>',
            template_system="PHP",
            description="PHP short echo tag",
            confidence=0.95
        ),
        
        # Twig Templates
        TemplatePattern(
            pattern=r'\{\{[^}]*\w+[^}]*\}\}',
            template_system="Twig",
            description="Twig template variable",
            confidence=0.9
        ),
        TemplatePattern(
            pattern=r'\{\%[^%]*\w+[^%]*\%\}',
            template_system="Twig",
            description="Twig template statement",
            confidence=0.9
        ),
        
        # Go Templates
        TemplatePattern(
            pattern=r'\{\{[^}]*\.[A-Z]\w*[^}]*\}\}',
            template_system="Go",
            description="Go template field access",
            confidence=0.95
        ),
        TemplatePattern(
            pattern=r'\{\{[^}]*range\s+\.[^}]*\}\}',
            template_system="Go",
            description="Go template range statement",
            confidence=0.95
        ),
        
        # Angular Templates
        TemplatePattern(
            pattern=r'\{\{[^}]*\w+[^}]*\}\}',
            template_system="Angular",
            description="Angular interpolation binding",
            confidence=0.8
        ),
        TemplatePattern(
            pattern=r'\[\([^)]*\w+[^)]*\)\]',
            template_system="Angular",
            description="Angular two-way binding",
            confidence=0.95
        ),
        TemplatePattern(
            pattern=r'\*ng[A-Z]\w*="[^"]*\w+[^"]*"',
            template_system="Angular",
            description="Angular structural directive",
            confidence=0.95
        ),
        
        # Vue.js Templates
        TemplatePattern(
            pattern=r'\{\{[^}]*\w+[^}]*\}\}',
            template_system="Vue.js",
            description="Vue.js interpolation",
            confidence=0.8
        ),
        TemplatePattern(
            pattern=r'v-\w+="[^"]*\w+[^"]*"',
            template_system="Vue.js",
            description="Vue.js directive",
            confidence=0.95
        ),
        
        # React JSX (less common but possible)
        TemplatePattern(
            pattern=r'\{[^}]*\w+[^}]*\}',
            template_system="React JSX",
            description="React JSX expression",
            confidence=0.7,
            context_aware=True
        ),
        
        # Custom/Generic Patterns
        TemplatePattern(
            pattern=r'%\{[^}]*\w+[^}]*\}%',
            template_system="Custom",
            description="Custom template variable with % delimiters",
            confidence=0.9
        ),
        TemplatePattern(
            pattern=r'\[\[[^\]]*\w+[^\]]*\]\]',
            template_system="Wiki-style",
            description="Wiki-style template variable",
            confidence=0.8,
            context_aware=True
        ),
        
        # Configuration placeholders
        TemplatePattern(
            pattern=r'\$\([^)]*\w+[^)]*\)',
            template_system="Configuration",
            description="Configuration placeholder",
            confidence=0.85
        ),
        
        # Build system variables
        TemplatePattern(
            pattern=r'@[A-Z_][A-Z0-9_]*@',
            template_system="Build System",
            description="Autotools/CMake variable",
            confidence=0.9
        ),
        
        # Environment variable patterns in templates
        TemplatePattern(
            pattern=r'\$\{?[A-Z_][A-Z0-9_]*\}?',
            template_system="Environment",
            description="Environment variable reference",
            confidence=0.8,
            context_aware=True
        ),
    ]
    
    def __init__(self):
        """Initialize enhanced template detector."""
        self.patterns = self.ENHANCED_TEMPLATE_PATTERNS.copy()
        
        # Context patterns to help filter false positives
        self.html_context_patterns = [
            r'<[^>]+>',  # HTML tags
            r'&[a-zA-Z]+;',  # HTML entities
            r'<!DOCTYPE[^>]*>',  # DOCTYPE declarations
        ]
        
        self.code_context_patterns = [
            r'(function|class|def|var|let|const)\s+\w+',  # Function/class definitions
            r'(if|else|for|while|switch)\s*\(',  # Control structures
            r'\/\*[\s\S]*?\*\/',  # Block comments
            r'\/\/.*$',  # Line comments
            r'(import|export|require)\s+',  # Module imports
        ]
        
        self.markdown_context_patterns = [
            r'^#{1,6}\s+',  # Headers
            r'```[\s\S]*?```',  # Code blocks
            r'`[^`]+`',  # Inline code
            r'\[.*?\]\(.*?\)',  # Links
            r'!\[.*?\]\(.*?\)',  # Images
        ]
        
        self.valid_html_tags = {
            'html', 'head', 'body', 'div', 'span', 'p', 'a', 'img', 'br', 'hr',
            'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'li', 'table',
            'tr', 'td', 'th', 'form', 'input', 'button', 'script', 'style'
        }
        
    def detect_template_artifacts(self, content: str, file_path: str = "") -> List[QualityIssue]:
        """
        Detect unrendered template variables with enhanced pattern matching
        and context-aware filtering.
        """
        issues = []
        
        # Pre-analyze content context
        content_context = self._analyze_content_context(content, file_path)
        
        for pattern_obj in self.patterns:
            matches = re.finditer(pattern_obj.pattern, content, re.MULTILINE | re.DOTALL)
            
            for match in matches:
                matched_text = match.group(0)
                
                # Apply context-aware filtering
                if pattern_obj.context_aware:
                    if self._is_likely_false_positive(matched_text, content, match.start(), content_context):
                        continue
                
                # Additional validation for template artifacts
                if not self._is_likely_template_artifact(matched_text, content, pattern_obj):
                    continue
                
                line_number = content[:match.start()].count('\n') + 1
                
                # Calculate confidence based on context and pattern
                confidence = self._calculate_confidence(
                    matched_text, pattern_obj, content_context, content, match.start()
                )
                
                issues.append(QualityIssue(
                    category=IssueCategory.TEMPLATE_ARTIFACT,
                    severity=pattern_obj.severity,
                    description=f"Unrendered {pattern_obj.template_system} template: {matched_text}",
                    file_path=file_path,
                    line_number=line_number,
                    suggestion=f"Ensure {pattern_obj.template_system} template variable '{matched_text}' is properly resolved during pipeline processing",
                    confidence=confidence
                ))
        
        # Detect multi-line template blocks
        multiline_issues = self._detect_multiline_templates(content, file_path)
        issues.extend(multiline_issues)
        
        # Detect nested template structures
        nested_issues = self._detect_nested_templates(content, file_path)
        issues.extend(nested_issues)
        
        return issues
    
    def _analyze_content_context(self, content: str, file_path: str) -> Dict[str, bool]:
        """Analyze the content to determine its context (HTML, Markdown, Code, etc.)."""
        context = {
            'is_html': bool(re.search(r'<!DOCTYPE|<html|<head|<body', content, re.IGNORECASE)),
            'is_markdown': bool(re.search(r'(^#{1,6}\s+|```|^\*\s+)', content, re.MULTILINE)),
            'is_code': bool(re.search(r'(function|class|def|import|export)', content)),
            'is_json': self._is_likely_json(content),
            'is_yaml': bool(re.search(r'^[a-zA-Z_][a-zA-Z0-9_]*:\s*', content, re.MULTILINE)),
            'is_config': bool(re.search(r'(config|settings|\.ini|\.conf)', file_path, re.IGNORECASE)),
            'file_extension': Path(file_path).suffix.lower() if file_path else '',
        }
        return context
    
    def _is_likely_json(self, content: str) -> bool:
        """Check if content appears to be JSON."""
        try:
            json.loads(content.strip())
            return True
        except (json.JSONDecodeError, ValueError):
            return False
    
    def _is_likely_false_positive(self, matched_text: str, content: str, match_start: int, context: Dict[str, bool]) -> bool:
        """Check if a match is likely a false positive based on context."""
        
        # Check if it's a valid HTML tag
        if context['is_html'] and self._is_valid_html_tag(matched_text):
            return True
        
        # Check if it's in a code comment
        if self._is_in_comment(content, match_start):
            return True
        
        # Check if it's in a code block (markdown)
        if context['is_markdown'] and self._is_in_code_block(content, match_start):
            return True
        
        # Check if it's a JavaScript template literal (valid syntax)
        if self._is_valid_template_literal(matched_text, content, match_start):
            return True
        
        # Check if it's a legitimate JSX expression
        if context['is_code'] and self._is_valid_jsx_expression(matched_text, content, match_start):
            return True
        
        # Check for common non-template patterns that might match
        if self._is_common_non_template_pattern(matched_text):
            return True
        
        return False
    
    def _is_valid_html_tag(self, text: str) -> bool:
        """Check if text is a valid HTML tag."""
        # Remove angle brackets and extract tag name
        tag_match = re.match(r'</?([a-zA-Z][a-zA-Z0-9]*)', text)
        if tag_match:
            tag_name = tag_match.group(1).lower()
            return tag_name in self.valid_html_tags
        return False
    
    def _is_in_comment(self, content: str, position: int) -> bool:
        """Check if position is within a comment."""
        # Check for block comments /* */
        block_comment_pattern = r'\/\*[\s\S]*?\*\/'
        for match in re.finditer(block_comment_pattern, content):
            if match.start() <= position <= match.end():
                return True
        
        # Check for line comments //
        lines_before = content[:position].split('\n')
        current_line = lines_before[-1] if lines_before else ""
        if '//' in current_line:
            comment_start = current_line.find('//')
            position_in_line = len(current_line) - (len(lines_before[-1]) if lines_before else 0)
            if position_in_line > comment_start:
                return True
        
        return False
    
    def _is_in_code_block(self, content: str, position: int) -> bool:
        """Check if position is within a markdown code block."""
        code_block_pattern = r'```[\s\S]*?```'
        for match in re.finditer(code_block_pattern, content):
            if match.start() <= position <= match.end():
                return True
        return False
    
    def _is_valid_template_literal(self, text: str, content: str, position: int) -> bool:
        """Check if this is a valid JavaScript template literal."""
        # Look for backticks around the expression
        start_context = content[max(0, position-10):position]
        end_context = content[position+len(text):position+len(text)+10]
        
        return '`' in start_context and '`' in end_context
    
    def _is_valid_jsx_expression(self, text: str, content: str, position: int) -> bool:
        """Check if this is a valid JSX expression."""
        # Look for JSX context indicators
        surrounding_context = content[max(0, position-50):position+len(text)+50]
        jsx_indicators = ['return (', 'render()', '<div', '<span', 'className=', 'onClick=']
        
        return any(indicator in surrounding_context for indicator in jsx_indicators)
    
    def _is_common_non_template_pattern(self, text: str) -> bool:
        """Check for common patterns that aren't template variables."""
        non_template_patterns = [
            r'^\{[0-9]+\}$',  # Numbered placeholders like {1}, {2}
            r'^\{[a-f0-9-]{8,}\}$',  # UUIDs or hashes in braces
            r'^\{[\s]*\}$',  # Empty braces
            r'^<[^>]*id=',  # HTML attributes
            r'^<[^>]*class=',  # HTML attributes
        ]
        
        return any(re.match(pattern, text) for pattern in non_template_patterns)
    
    def _is_likely_template_artifact(self, text: str, content: str, pattern_obj: TemplatePattern) -> bool:
        """Advanced validation to determine if text is likely an unrendered template."""
        # Check for variable-like patterns
        variable_patterns = [
            r'\w+',  # Simple variable names
            r'\w+\.\w+',  # Object property access
            r'\w+\[\w+\]',  # Array/object indexing
            r'\w+\|\w+',  # Pipe filters (common in templates)
        ]
        
        inner_content = self._extract_inner_content(text, pattern_obj.template_system)
        if not inner_content:
            return False
        
        # Check if inner content looks like a variable
        variable_like = any(re.search(pattern, inner_content.strip()) for pattern in variable_patterns)
        
        # Additional checks for specific template systems
        if pattern_obj.template_system in ["Jinja2/Django", "Twig"]:
            # Look for common template constructs
            template_constructs = ['for', 'if', 'endif', 'endfor', 'else', 'elif', '|', 'default', 'length']
            has_constructs = any(construct in inner_content for construct in template_constructs)
            return variable_like or has_constructs
        
        elif pattern_obj.template_system == "Handlebars":
            # Look for Handlebars helpers and expressions
            handlebars_constructs = ['each', 'if', 'unless', 'with', '#', '/', '@', 'this.']
            has_constructs = any(construct in inner_content for construct in handlebars_constructs)
            return variable_like or has_constructs
        
        return variable_like
    
    def _extract_inner_content(self, text: str, template_system: str) -> str:
        """Extract the inner content from a template variable."""
        # Remove common delimiters based on template system
        delimiters = {
            'Jinja2/Django': [(r'^\{\{', r'\}\}$'), (r'^\{\%', r'\%\}$')],
            'Handlebars/Mustache': [(r'^\{\{', r'\}\}$'), (r'^\{\{\{', r'\}\}\}$')],
            'ERB': [(r'^<%=?', r'%>$')],
            'PHP': [(r'^<\?php\s+echo\s+', r'\s*\?>$'), (r'^<\?=', r'\?>$')],
            'Shell/JavaScript': [(r'^\$\{', r'\}$')],
            'Custom': [(r'^%\{', r'\}%$')],
            'Wiki-style': [(r'^\[\[', r'\]\]$')],
        }
        
        system_delimiters = delimiters.get(template_system, [])
        for start_pattern, end_pattern in system_delimiters:
            cleaned = re.sub(start_pattern, '', text)
            cleaned = re.sub(end_pattern, '', cleaned)
            if cleaned != text:  # If substitution happened
                return cleaned.strip()
        
        # Fallback: try to extract anything between common delimiters
        common_patterns = [
            (r'^\{\{', r'\}\}$'),
            (r'^\$\{', r'\}$'), 
            (r'^<%', r'%>$'),
            (r'^<\?', r'\?>$'),
        ]
        
        for start_pattern, end_pattern in common_patterns:
            cleaned = re.sub(start_pattern, '', text)
            cleaned = re.sub(end_pattern, '', cleaned)
            if cleaned != text:
                return cleaned.strip()
        
        return text.strip()
    
    def _calculate_confidence(self, text: str, pattern_obj: TemplatePattern, context: Dict[str, bool], content: str, position: int) -> float:
        """Calculate confidence score for template artifact detection."""
        confidence = pattern_obj.confidence
        
        # Adjust based on content context
        if context['is_html'] and not self._is_valid_html_tag(text):
            confidence *= 1.1  # Higher confidence in HTML context if not valid HTML
        
        if context['is_config']:
            confidence *= 1.2  # Higher confidence in config files
        
        # Adjust based on variable name quality
        inner_content = self._extract_inner_content(text, pattern_obj.template_system)
        if inner_content:
            # Good variable names increase confidence
            if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', inner_content.strip()):
                confidence *= 1.1
            
            # Template-specific constructs increase confidence
            template_indicators = ['default', 'length', 'upper', 'lower', 'join', 'split', 'format']
            if any(indicator in inner_content for indicator in template_indicators):
                confidence *= 1.2
        
        # Reduce confidence if in a comment or documentation
        if self._is_in_comment(content, position):
            confidence *= 0.5
        
        return min(1.0, confidence)  # Cap at 1.0
    
    def _detect_multiline_templates(self, content: str, file_path: str) -> List[QualityIssue]:
        """Detect multi-line template blocks that might span multiple lines."""
        issues = []
        
        # Multi-line template patterns
        multiline_patterns = [
            # Jinja2/Django blocks
            (r'\{\%\s*(for|if|block|with)\s+[^%]*\%\}[\s\S]*?\{\%\s*end\1\s*\%\}', "Jinja2/Django", "Multi-line template block"),
            # Handlebars blocks  
            (r'\{\{#\s*\w+[^}]*\}\}[\s\S]*?\{\{/\w+\}\}', "Handlebars", "Multi-line template block"),
            # ERB blocks
            (r'<%[^%]*%>[\s\S]*?<%[^%]*%>', "ERB", "Multi-line template block"),
        ]
        
        for pattern, template_system, description in multiline_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)
            
            for match in matches:
                matched_text = match.group(0)
                line_number = content[:match.start()].count('\n') + 1
                
                issues.append(QualityIssue(
                    category=IssueCategory.TEMPLATE_ARTIFACT,
                    severity=IssueSeverity.CRITICAL,
                    description=f"Unrendered {template_system} {description.lower()}: {matched_text[:100]}{'...' if len(matched_text) > 100 else ''}",
                    file_path=file_path,
                    line_number=line_number,
                    suggestion=f"Ensure {template_system} template block is properly processed during pipeline execution",
                    confidence=0.95
                ))
        
        return issues
    
    def _detect_nested_templates(self, content: str, file_path: str) -> List[QualityIssue]:
        """Detect nested template structures that indicate complex unrendered templates."""
        issues = []
        
        # Nested template patterns
        nested_patterns = [
            # Nested Jinja2 variables
            r'\{\{[^}]*\{\{[^}]*\}\}[^}]*\}\}',
            # Nested shell variables
            r'\$\{[^}]*\$\{[^}]*\}[^}]*\}',
            # Complex Handlebars expressions
            r'\{\{[^}]*\([^)]*\{\{[^}]*\}\}[^)]*\)[^}]*\}\}',
        ]
        
        for pattern in nested_patterns:
            matches = re.finditer(pattern, content)
            
            for match in matches:
                matched_text = match.group(0)
                line_number = content[:match.start()].count('\n') + 1
                
                issues.append(QualityIssue(
                    category=IssueCategory.TEMPLATE_ARTIFACT,
                    severity=IssueSeverity.CRITICAL,
                    description=f"Complex nested template structure: {matched_text}",
                    file_path=file_path,
                    line_number=line_number,
                    suggestion="Review and resolve nested template variable dependencies",
                    confidence=0.9
                ))
        
        return issues