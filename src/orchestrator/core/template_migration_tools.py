"""
Template Migration Tools for POML Integration.

Provides utilities for migrating existing Jinja2 templates to POML format
and creating hybrid templates that combine both approaches.
"""

from __future__ import annotations

import re
import logging
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

from .template_resolver import TemplateFormatDetector, TemplateFormat

logger = logging.getLogger(__name__)


class MigrationStrategy(Enum):
    """Migration strategies for template conversion."""
    FULL_POML = "full_poml"          # Convert entirely to POML
    HYBRID = "hybrid"                # Keep Jinja2 variables, add POML structure  
    ENHANCED_JINJA2 = "enhanced_jinja2"  # Add POML components to existing Jinja2
    PRESERVE = "preserve"            # Keep as-is, no migration needed


@dataclass
class TemplateAnalysis:
    """Analysis results for a template."""
    original_format: TemplateFormat
    complexity_score: float
    jinja2_features: List[str]
    suggested_strategy: MigrationStrategy
    migration_notes: List[str]
    poml_benefits: List[str]
    migration_effort: str  # "low", "medium", "high"


@dataclass
class MigrationResult:
    """Result of template migration."""
    original_template: str
    migrated_template: str
    strategy_used: MigrationStrategy
    changes_made: List[str]
    validation_issues: List[str]
    success: bool


class TemplateMigrationAnalyzer:
    """
    Analyzes existing templates and recommends migration strategies.
    """
    
    def __init__(self):
        self.format_detector = TemplateFormatDetector()
        
        # Patterns for detecting Jinja2 features
        self.jinja2_patterns = {
            'variables': re.compile(r'{{\s*([^}]+)\s*}}'),
            'filters': re.compile(r'{{\s*[^}]*\s*\|\s*([^}\s]+)'),
            'loops': re.compile(r'{\%\s*(for|endfor)\s*[^%]*\%}'),
            'conditionals': re.compile(r'{\%\s*(if|elif|else|endif)\s*[^%]*\%}'),
            'macros': re.compile(r'{\%\s*(macro|endmacro)\s*[^%]*\%}'),
            'includes': re.compile(r'{\%\s*include\s+[^%]*\%}'),
            'comments': re.compile(r'{#[^#]*#}'),
        }
    
    def analyze_template(self, template_content: str, context: Optional[Dict[str, Any]] = None) -> TemplateAnalysis:
        """
        Analyze a template and suggest migration strategy.
        
        Args:
            template_content: The template to analyze
            context: Optional context for understanding variable usage
            
        Returns:
            TemplateAnalysis with recommendations
        """
        original_format = self.format_detector.detect_format(template_content)
        
        # Extract Jinja2 features
        jinja2_features = self._extract_jinja2_features(template_content)
        
        # Calculate complexity score
        complexity_score = self._calculate_complexity(template_content, jinja2_features)
        
        # Determine suggested strategy
        suggested_strategy = self._suggest_migration_strategy(
            original_format, jinja2_features, complexity_score
        )
        
        # Generate migration notes
        migration_notes = self._generate_migration_notes(
            template_content, jinja2_features, suggested_strategy
        )
        
        # Identify POML benefits
        poml_benefits = self._identify_poml_benefits(template_content, jinja2_features)
        
        # Estimate migration effort
        migration_effort = self._estimate_effort(complexity_score, jinja2_features)
        
        return TemplateAnalysis(
            original_format=original_format,
            complexity_score=complexity_score,
            jinja2_features=jinja2_features,
            suggested_strategy=suggested_strategy,
            migration_notes=migration_notes,
            poml_benefits=poml_benefits,
            migration_effort=migration_effort
        )
    
    def _extract_jinja2_features(self, template_content: str) -> List[str]:
        """Extract Jinja2 features used in the template."""
        features = []
        
        for feature_name, pattern in self.jinja2_patterns.items():
            if pattern.search(template_content):
                features.append(feature_name)
        
        return features
    
    def _calculate_complexity(self, template_content: str, features: List[str]) -> float:
        """Calculate template complexity score (0.0 to 1.0)."""
        base_score = 0.0
        
        # Length factor
        length_factor = min(len(template_content) / 1000.0, 0.3)
        base_score += length_factor
        
        # Feature complexity
        feature_weights = {
            'variables': 0.1,
            'filters': 0.15,
            'loops': 0.2,
            'conditionals': 0.2,
            'macros': 0.3,
            'includes': 0.25,
            'comments': 0.05
        }
        
        for feature in features:
            if feature in feature_weights:
                base_score += feature_weights[feature]
        
        return min(base_score, 1.0)
    
    def _suggest_migration_strategy(self, 
                                  original_format: TemplateFormat, 
                                  features: List[str], 
                                  complexity: float) -> MigrationStrategy:
        """Suggest the best migration strategy."""
        
        if original_format == TemplateFormat.POML:
            return MigrationStrategy.PRESERVE
        
        if original_format == TemplateFormat.PLAIN:
            return MigrationStrategy.FULL_POML
        
        # For Jinja2 templates, decide based on complexity and features
        if complexity < 0.3 and not any(f in features for f in ['loops', 'conditionals', 'macros']):
            return MigrationStrategy.FULL_POML
        elif complexity < 0.6:
            return MigrationStrategy.HYBRID
        else:
            return MigrationStrategy.ENHANCED_JINJA2
    
    def _generate_migration_notes(self, 
                                template_content: str,
                                features: List[str], 
                                strategy: MigrationStrategy) -> List[str]:
        """Generate specific migration notes."""
        notes = []
        
        if strategy == MigrationStrategy.FULL_POML:
            notes.append("Template can be fully converted to POML structure")
            if 'variables' in features:
                notes.append("Jinja2 variables will be converted to POML programmatic approach")
        
        elif strategy == MigrationStrategy.HYBRID:
            notes.append("Recommended: Add POML structure while keeping Jinja2 variables")
            if 'filters' in features:
                notes.append("Jinja2 filters will be preserved in variable references")
            if 'loops' in features or 'conditionals' in features:
                notes.append("Complex logic will remain as Jinja2 syntax")
        
        elif strategy == MigrationStrategy.ENHANCED_JINJA2:
            notes.append("Template complexity suggests keeping as enhanced Jinja2")
            notes.append("Consider adding POML data components for better data integration")
        
        # Feature-specific notes
        if 'macros' in features:
            notes.append("Macros need manual conversion - consider POML template composition")
        
        if 'includes' in features:
            notes.append("Template includes should be converted to POML document components")
        
        return notes
    
    def _identify_poml_benefits(self, template_content: str, features: List[str]) -> List[str]:
        """Identify potential benefits of POML conversion."""
        benefits = []
        
        # General benefits
        benefits.append("Structured semantic markup for better LLM understanding")
        benefits.append("Clear separation of role, task, and examples")
        
        # Specific benefits based on content
        if 'role' in template_content.lower():
            benefits.append("Existing role definition will be properly structured")
        
        if 'task' in template_content.lower() or 'instruction' in template_content.lower():
            benefits.append("Task definitions will be semantically marked")
        
        if 'example' in template_content.lower():
            benefits.append("Examples will be structured with input/output format")
        
        if re.search(r'\bdata\b|\bfile\b|\btable\b', template_content.lower()):
            benefits.append("Data integration components will improve file handling")
        
        if 'format' in template_content.lower():
            benefits.append("Output format specifications will be properly structured")
        
        return benefits
    
    def _estimate_effort(self, complexity: float, features: List[str]) -> str:
        """Estimate migration effort level."""
        if complexity < 0.3:
            return "low"
        elif complexity < 0.6:
            return "medium"
        else:
            return "high"


class TemplateMigrationEngine:
    """
    Performs actual template migration based on analysis recommendations.
    """
    
    def __init__(self):
        self.analyzer = TemplateMigrationAnalyzer()
        self.format_detector = TemplateFormatDetector()
    
    def migrate_template(self, 
                        template_content: str,
                        strategy: Optional[MigrationStrategy] = None,
                        context: Optional[Dict[str, Any]] = None) -> MigrationResult:
        """
        Migrate a template using the specified or recommended strategy.
        
        Args:
            template_content: Template to migrate
            strategy: Migration strategy to use (auto-detect if None)
            context: Optional context for migration
            
        Returns:
            MigrationResult with success status and details
        """
        # Analyze template if strategy not specified
        if strategy is None:
            analysis = self.analyzer.analyze_template(template_content, context)
            strategy = analysis.suggested_strategy
        
        changes_made = []
        validation_issues = []
        
        try:
            if strategy == MigrationStrategy.FULL_POML:
                migrated_template = self._migrate_to_full_poml(template_content)
                changes_made.append("Converted to full POML structure")
                
            elif strategy == MigrationStrategy.HYBRID:
                migrated_template = self._migrate_to_hybrid(template_content)
                changes_made.append("Added POML structure with Jinja2 variables")
                
            elif strategy == MigrationStrategy.ENHANCED_JINJA2:
                migrated_template = self._enhance_jinja2_template(template_content)
                changes_made.append("Enhanced Jinja2 with POML data components")
                
            else:  # PRESERVE
                migrated_template = template_content
                changes_made.append("No migration performed - template preserved as-is")
            
            # Validate result
            validation_issues = self._validate_migrated_template(migrated_template)
            
            success = len(validation_issues) == 0
            
        except Exception as e:
            logger.error(f"Template migration failed: {e}")
            migrated_template = template_content
            validation_issues.append(f"Migration failed: {e}")
            success = False
        
        return MigrationResult(
            original_template=template_content,
            migrated_template=migrated_template,
            strategy_used=strategy,
            changes_made=changes_made,
            validation_issues=validation_issues,
            success=success
        )
    
    def _migrate_to_full_poml(self, template_content: str) -> str:
        """Convert template to full POML format."""
        # Extract potential components from content
        role_content = self._extract_role_content(template_content)
        task_content = self._extract_task_content(template_content)
        examples = self._extract_examples(template_content)
        hints = self._extract_hints(template_content)
        output_format = self._extract_output_format(template_content)
        
        # Build POML structure
        poml_parts = []
        
        if role_content:
            poml_parts.append(f"<role>{role_content}</role>")
        else:
            # Default role if none found
            poml_parts.append("<role>Assistant</role>")
        
        if task_content:
            poml_parts.append(f"<task>{task_content}</task>")
        else:
            # Use entire content as task if no specific task found
            clean_content = self._clean_template_content(template_content)
            poml_parts.append(f"<task>{clean_content}</task>")
        
        # Add examples
        for example in examples:
            poml_parts.append("<example>")
            if 'input' in example:
                poml_parts.append(f"  <input>{example['input']}</input>")
            if 'output' in example:
                poml_parts.append(f"  <output>{example['output']}</output>")
            poml_parts.append("</example>")
        
        # Add hints
        for hint in hints:
            poml_parts.append(f"<hint>{hint}</hint>")
        
        if output_format:
            poml_parts.append(f"<output-format>{output_format}</output-format>")
        
        return "\n".join(poml_parts)
    
    def _migrate_to_hybrid(self, template_content: str) -> str:
        """Convert template to hybrid POML/Jinja2 format."""
        # For hybrid, we wrap sections in POML tags but preserve Jinja2 variables
        
        # Try to identify existing structure
        if re.search(r'role|assistant', template_content, re.IGNORECASE):
            # Has role-like content
            template_content = re.sub(
                r'(role\s*:?\s*)(.*?)(\n|$)', 
                r'<role>\2</role>\n', 
                template_content, 
                flags=re.IGNORECASE
            )
        
        if re.search(r'task|instruction|objective', template_content, re.IGNORECASE):
            # Has task-like content
            template_content = re.sub(
                r'(task|instruction|objective)\s*:?\s*(.*?)(?=\n\n|\n[A-Z]|\n#|$)',
                r'<task>\2</task>\n',
                template_content,
                flags=re.IGNORECASE | re.DOTALL
            )
        
        # If no structure detected, wrap everything in a task
        if not re.search(r'<(role|task)', template_content):
            template_content = f"<task>{template_content}</task>"
        
        return template_content
    
    def _enhance_jinja2_template(self, template_content: str) -> str:
        """Enhance Jinja2 template with POML data components."""
        enhanced = template_content
        
        # Look for file references and convert to document components
        file_patterns = [
            r'(\w+\.csv)',
            r'(\w+\.json)',  
            r'(\w+\.txt)',
            r'(\w+\.pdf)',
        ]
        
        for pattern in file_patterns:
            matches = re.finditer(pattern, enhanced)
            for match in matches:
                file_ref = match.group(1)
                # Replace with POML document component if appropriate
                if not re.search(rf'<document[^>]*{re.escape(file_ref)}', enhanced):
                    # Add document component (this is a simple heuristic)
                    doc_component = f'<document src="{file_ref}">Data file</document>\n'
                    enhanced = doc_component + enhanced
        
        return enhanced
    
    def _extract_role_content(self, content: str) -> Optional[str]:
        """Extract role information from template content."""
        patterns = [
            r'role\s*:?\s*([^\n]+)',
            r'you\s+are\s+(?:a\s+|an\s+)?([^\n.]+)',
            r'assistant\s+type\s*:?\s*([^\n]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_task_content(self, content: str) -> Optional[str]:
        """Extract task information from template content."""
        patterns = [
            r'task\s*:?\s*([^\n]+(?:\n(?!\n)[^\n]+)*)',
            r'instruction\s*:?\s*([^\n]+(?:\n(?!\n)[^\n]+)*)',
            r'objective\s*:?\s*([^\n]+(?:\n(?!\n)[^\n]+)*)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_examples(self, content: str) -> List[Dict[str, str]]:
        """Extract examples from template content."""
        examples = []
        
        # Look for example patterns
        example_patterns = [
            r'example\s*:?\s*([^\n]+(?:\n(?!\n)[^\n]+)*)',
            r'input\s*:?\s*([^\n]+).*?output\s*:?\s*([^\n]+)',
        ]
        
        for pattern in example_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
            for match in matches:
                if len(match.groups()) == 1:
                    examples.append({'content': match.group(1).strip()})
                else:
                    examples.append({
                        'input': match.group(1).strip(),
                        'output': match.group(2).strip()
                    })
        
        return examples
    
    def _extract_hints(self, content: str) -> List[str]:
        """Extract hints from template content."""
        hints = []
        
        hint_patterns = [
            r'hint\s*:?\s*([^\n]+)',
            r'note\s*:?\s*([^\n]+)',
            r'remember\s*:?\s*([^\n]+)',
        ]
        
        for pattern in hint_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                hints.append(match.group(1).strip())
        
        return hints
    
    def _extract_output_format(self, content: str) -> Optional[str]:
        """Extract output format information."""
        patterns = [
            r'output\s+format\s*:?\s*([^\n]+(?:\n(?!\n)[^\n]+)*)',
            r'format\s*:?\s*([^\n]+(?:\n(?!\n)[^\n]+)*)',
            r'respond\s+(?:in|with|as)\s+([^\n]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _clean_template_content(self, content: str) -> str:
        """Clean template content for use in POML."""
        # Remove common template artifacts
        cleaned = re.sub(r'{#.*?#}', '', content, flags=re.DOTALL)  # Remove comments
        cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)  # Normalize line breaks
        return cleaned.strip()
    
    def _validate_migrated_template(self, template: str) -> List[str]:
        """Validate migrated template for common issues."""
        issues = []
        
        # Check for unclosed tags
        open_tags = re.findall(r'<(\w+)(?:\s[^>]*)?>(?!.*</\1>)', template)
        for tag in open_tags:
            if tag not in ['br', 'img', 'input', 'meta']:  # Self-closing tags
                issues.append(f"Unclosed tag: <{tag}>")
        
        # Check for invalid nesting
        if '<task>' in template and '<role>' in template:
            role_pos = template.find('<role>')
            task_pos = template.find('<task>')
            if task_pos < role_pos:
                issues.append("Task should come after role in POML structure")
        
        return issues


# Convenience functions for easy usage
def analyze_template(template_content: str) -> TemplateAnalysis:
    """Analyze a template and get migration recommendations."""
    analyzer = TemplateMigrationAnalyzer()
    return analyzer.analyze_template(template_content)


def migrate_template(template_content: str, 
                    strategy: Optional[MigrationStrategy] = None) -> MigrationResult:
    """Migrate a template to POML format."""
    engine = TemplateMigrationEngine()
    return engine.migrate_template(template_content, strategy)


def batch_analyze_templates(templates: Dict[str, str]) -> Dict[str, TemplateAnalysis]:
    """Analyze multiple templates at once."""
    analyzer = TemplateMigrationAnalyzer()
    return {name: analyzer.analyze_template(content) 
            for name, content in templates.items()}


def batch_migrate_templates(templates: Dict[str, str], 
                           strategy: Optional[MigrationStrategy] = None) -> Dict[str, MigrationResult]:
    """Migrate multiple templates at once."""
    engine = TemplateMigrationEngine()
    return {name: engine.migrate_template(content, strategy)
            for name, content in templates.items()}