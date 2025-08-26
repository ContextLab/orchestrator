#!/usr/bin/env python3
"""
Quality Metrics Analyzer for Issue #262.

Advanced quality analysis system that integrates with enhanced validation (#256)
and LLM quality review (#257) to provide detailed quality insights, scoring,
and trend analysis for pipeline validation.
"""

import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class QualityIssue:
    """Represents a quality issue found in pipeline output."""
    issue_type: str
    severity: str  # critical, high, medium, low
    description: str
    pipeline: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    suggestion: Optional[str] = None
    frequency: int = 1


@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for a pipeline."""
    pipeline_name: str
    overall_score: float
    category_scores: Dict[str, float]
    issues: List[QualityIssue]
    output_quality: Dict[str, Any]
    llm_assessment: Optional[Dict[str, Any]] = None
    validation_results: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)


class QualityAnalyzer:
    """Advanced quality analyzer for pipeline validation."""
    
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path).resolve()
        self.outputs_dir = self.root_path / "examples" / "outputs"
        
        # Quality scoring weights
        self.quality_weights = {
            'content_completeness': 0.25,
            'format_correctness': 0.20,
            'template_rendering': 0.15,
            'conversational_markers': 0.15,
            'error_indicators': 0.10,
            'output_consistency': 0.10,
            'file_generation': 0.05
        }
        
        # Issue severity scoring
        self.severity_scores = {
            'critical': 0,    # Blocks functionality
            'high': 25,       # Major quality issues  
            'medium': 50,     # Minor issues
            'low': 75         # Cosmetic issues
        }
        
        # Regex patterns for quality detection
        self.quality_patterns = {
            'template_variables': re.compile(r'\{\{[^}]+\}\}'),
            'loop_variables': re.compile(r'\$(?:item|index|iteration)\b'),
            'conversational_markers': re.compile(r'\b(?:Certainly!|Sure!|I\'d be happy to|Let me|I\'ll create|Here\'s)\b'),
            'error_indicators': re.compile(r'\b(?:error|failed|exception|traceback)\b', re.IGNORECASE),
            'placeholder_text': re.compile(r'\b(?:placeholder|todo|fixme|xxx|tbd)\b', re.IGNORECASE),
            'hardcoded_paths': re.compile(r'(?:/Users/[^/\s]+|C:\\Users\\[^\\s]+|/home/[^/\s]+)'),
            'debug_output': re.compile(r'\b(?:console\.log|print\(|debug|DEBUG)\b'),
            'incomplete_sentences': re.compile(r'[.!?]\s*$', re.MULTILINE)
        }
        
        logger.info(f"Quality Analyzer initialized for: {self.root_path}")

    def analyze_pipeline_quality(self, pipeline_name: str) -> QualityMetrics:
        """Analyze quality metrics for a specific pipeline."""
        logger.info(f"Analyzing quality for pipeline: {pipeline_name}")
        
        pipeline_dir = self.outputs_dir / pipeline_name.replace('.yaml', '')
        
        if not pipeline_dir.exists():
            logger.warning(f"Pipeline output directory not found: {pipeline_dir}")
            return QualityMetrics(
                pipeline_name=pipeline_name,
                overall_score=0.0,
                category_scores={},
                issues=[QualityIssue("missing_output", "critical", "No output directory found", pipeline_name)],
                output_quality={}
            )
        
        # Load validation summary if available
        validation_summary = self._load_validation_summary(pipeline_dir)
        
        # Analyze all output files
        output_files = self._get_output_files(pipeline_dir)
        quality_results = self._analyze_output_files(output_files, pipeline_name)
        
        # Calculate category scores
        category_scores = self._calculate_category_scores(quality_results, validation_summary)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(category_scores)
        
        # Collect all issues
        all_issues = self._collect_quality_issues(quality_results, pipeline_name)
        
        # Load LLM assessment if available
        llm_assessment = self._load_llm_assessment(pipeline_dir)
        
        return QualityMetrics(
            pipeline_name=pipeline_name,
            overall_score=overall_score,
            category_scores=category_scores,
            issues=all_issues,
            output_quality=quality_results,
            llm_assessment=llm_assessment,
            validation_results=validation_summary
        )

    def _load_validation_summary(self, pipeline_dir: Path) -> Optional[Dict[str, Any]]:
        """Load validation summary from pipeline output directory."""
        validation_file = pipeline_dir / "validation_summary.json"
        
        if validation_file.exists():
            try:
                with open(validation_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load validation summary from {validation_file}: {e}")
        
        return None

    def _load_llm_assessment(self, pipeline_dir: Path) -> Optional[Dict[str, Any]]:
        """Load LLM quality assessment if available."""
        llm_file = pipeline_dir / "llm_quality_assessment.json"
        
        if llm_file.exists():
            try:
                with open(llm_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load LLM assessment from {llm_file}: {e}")
        
        return None

    def _get_output_files(self, pipeline_dir: Path) -> List[Path]:
        """Get all output files from pipeline directory."""
        output_files = []
        
        # Common output file patterns
        patterns = ['*.md', '*.txt', '*.json', '*.csv', '*.html', '*.py', '*.js']
        
        for pattern in patterns:
            output_files.extend(pipeline_dir.glob(f"**/{pattern}"))
        
        # Filter out system files
        output_files = [f for f in output_files if not f.name.startswith('.')]
        
        return output_files

    def _analyze_output_files(self, output_files: List[Path], pipeline_name: str) -> Dict[str, Any]:
        """Analyze quality of output files."""
        results = {
            'files_analyzed': len(output_files),
            'content_analysis': {},
            'format_analysis': {},
            'issues_by_file': {},
            'file_sizes': {},
            'file_types': defaultdict(int)
        }
        
        for file_path in output_files:
            try:
                file_analysis = self._analyze_single_file(file_path, pipeline_name)
                
                file_key = str(file_path.relative_to(file_path.parents[1]))
                results['content_analysis'][file_key] = file_analysis['content']
                results['format_analysis'][file_key] = file_analysis['format']
                results['issues_by_file'][file_key] = file_analysis['issues']
                results['file_sizes'][file_key] = file_path.stat().st_size
                results['file_types'][file_path.suffix] += 1
                
            except Exception as e:
                logger.warning(f"Could not analyze file {file_path}: {e}")
        
        return results

    def _analyze_single_file(self, file_path: Path, pipeline_name: str) -> Dict[str, Any]:
        """Analyze quality of a single output file."""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
        except Exception as e:
            logger.warning(f"Could not read file {file_path}: {e}")
            return {
                'content': {'readable': False},
                'format': {'valid': False},
                'issues': ['file_read_error']
            }
        
        analysis = {
            'content': self._analyze_content_quality(content),
            'format': self._analyze_format_quality(file_path, content),
            'issues': []
        }
        
        # Detect quality issues
        issues = self._detect_file_issues(content, file_path, pipeline_name)
        analysis['issues'] = [issue.issue_type for issue in issues]
        
        return analysis

    def _analyze_content_quality(self, content: str) -> Dict[str, Any]:
        """Analyze content quality metrics."""
        if not content.strip():
            return {
                'completeness': 0.0,
                'readable': False,
                'word_count': 0,
                'line_count': 0,
                'empty_file': True
            }
        
        lines = content.split('\n')
        words = content.split()
        
        # Content completeness indicators
        completeness_score = 100.0
        
        # Check for template variables (should be rendered)
        template_vars = len(self.quality_patterns['template_variables'].findall(content))
        if template_vars > 0:
            completeness_score -= min(template_vars * 10, 50)
        
        # Check for loop variables
        loop_vars = len(self.quality_patterns['loop_variables'].findall(content))
        if loop_vars > 0:
            completeness_score -= min(loop_vars * 15, 40)
        
        # Check for placeholder text
        placeholders = len(self.quality_patterns['placeholder_text'].findall(content))
        if placeholders > 0:
            completeness_score -= min(placeholders * 5, 20)
        
        return {
            'completeness': max(0, completeness_score),
            'readable': True,
            'word_count': len(words),
            'line_count': len(lines),
            'empty_file': False,
            'template_variables': template_vars,
            'loop_variables': loop_vars,
            'placeholders': placeholders
        }

    def _analyze_format_quality(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Analyze format-specific quality metrics."""
        file_extension = file_path.suffix.lower()
        
        format_analysis = {
            'extension': file_extension,
            'valid_format': True,
            'format_specific_issues': []
        }
        
        if file_extension == '.json':
            format_analysis.update(self._validate_json_format(content))
        elif file_extension == '.csv':
            format_analysis.update(self._validate_csv_format(content))
        elif file_extension in ['.md', '.txt']:
            format_analysis.update(self._validate_text_format(content))
        elif file_extension == '.html':
            format_analysis.update(self._validate_html_format(content))
        
        return format_analysis

    def _validate_json_format(self, content: str) -> Dict[str, Any]:
        """Validate JSON format quality."""
        try:
            json.loads(content)
            return {'valid_json': True, 'parse_error': None}
        except json.JSONDecodeError as e:
            return {'valid_json': False, 'parse_error': str(e)}

    def _validate_csv_format(self, content: str) -> Dict[str, Any]:
        """Validate CSV format quality."""
        lines = content.strip().split('\n')
        
        if not lines:
            return {'valid_csv': False, 'issue': 'empty_file'}
        
        # Check for consistent column count
        first_line_cols = len(lines[0].split(','))
        inconsistent_rows = []
        
        for i, line in enumerate(lines[1:], 1):
            if len(line.split(',')) != first_line_cols:
                inconsistent_rows.append(i)
        
        return {
            'valid_csv': len(inconsistent_rows) == 0,
            'total_rows': len(lines),
            'column_count': first_line_cols,
            'inconsistent_rows': inconsistent_rows
        }

    def _validate_text_format(self, content: str) -> Dict[str, Any]:
        """Validate text/markdown format quality."""
        analysis = {
            'has_structure': False,
            'has_headers': False,
            'line_length_issues': 0
        }
        
        lines = content.split('\n')
        
        # Check for markdown headers
        if any(line.startswith('#') for line in lines):
            analysis['has_headers'] = True
            analysis['has_structure'] = True
        
        # Check for very long lines (> 120 characters)
        long_lines = [i for i, line in enumerate(lines) if len(line) > 120]
        analysis['line_length_issues'] = len(long_lines)
        
        return analysis

    def _validate_html_format(self, content: str) -> Dict[str, Any]:
        """Validate HTML format quality."""
        analysis = {
            'has_html_structure': False,
            'has_doctype': False,
            'unclosed_tags': []
        }
        
        # Basic HTML structure checks
        if '<html>' in content.lower() and '</html>' in content.lower():
            analysis['has_html_structure'] = True
        
        if '<!doctype' in content.lower():
            analysis['has_doctype'] = True
        
        # Simple tag matching (not comprehensive)
        import re
        open_tags = re.findall(r'<(\w+)[^>]*>', content)
        close_tags = re.findall(r'</(\w+)>', content)
        
        # Self-closing tags to ignore
        self_closing = {'img', 'br', 'hr', 'input', 'meta', 'link'}
        
        open_counts = defaultdict(int)
        close_counts = defaultdict(int)
        
        for tag in open_tags:
            if tag.lower() not in self_closing:
                open_counts[tag.lower()] += 1
        
        for tag in close_tags:
            close_counts[tag.lower()] += 1
        
        # Find unclosed tags
        for tag, count in open_counts.items():
            if count > close_counts.get(tag, 0):
                analysis['unclosed_tags'].append(tag)
        
        return analysis

    def _detect_file_issues(self, content: str, file_path: Path, pipeline_name: str) -> List[QualityIssue]:
        """Detect quality issues in file content."""
        issues = []
        
        # Template variable issues
        template_matches = self.quality_patterns['template_variables'].findall(content)
        if template_matches:
            issues.append(QualityIssue(
                issue_type="unrendered_templates",
                severity="high",
                description=f"Found {len(template_matches)} unrendered template variables",
                pipeline=pipeline_name,
                file_path=str(file_path),
                suggestion="Ensure all template variables are properly rendered"
            ))
        
        # Loop variable issues
        loop_matches = self.quality_patterns['loop_variables'].findall(content)
        if loop_matches:
            issues.append(QualityIssue(
                issue_type="unrendered_loops",
                severity="high",
                description=f"Found {len(loop_matches)} unrendered loop variables",
                pipeline=pipeline_name,
                file_path=str(file_path),
                suggestion="Ensure loop variables are properly processed"
            ))
        
        # Conversational marker issues
        conv_matches = self.quality_patterns['conversational_markers'].findall(content)
        if conv_matches:
            issues.append(QualityIssue(
                issue_type="conversational_markers",
                severity="medium",
                description=f"Found {len(conv_matches)} conversational markers",
                pipeline=pipeline_name,
                file_path=str(file_path),
                suggestion="Remove conversational language from outputs"
            ))
        
        # Error indicator issues
        error_matches = self.quality_patterns['error_indicators'].findall(content)
        if error_matches:
            issues.append(QualityIssue(
                issue_type="error_indicators",
                severity="high",
                description=f"Found {len(error_matches)} error indicators",
                pipeline=pipeline_name,
                file_path=str(file_path),
                suggestion="Review and fix errors in pipeline output"
            ))
        
        # Placeholder text issues
        placeholder_matches = self.quality_patterns['placeholder_text'].findall(content)
        if placeholder_matches:
            issues.append(QualityIssue(
                issue_type="placeholder_text",
                severity="medium",
                description=f"Found {len(placeholder_matches)} placeholder texts",
                pipeline=pipeline_name,
                file_path=str(file_path),
                suggestion="Replace placeholder text with actual content"
            ))
        
        # Hardcoded path issues
        path_matches = self.quality_patterns['hardcoded_paths'].findall(content)
        if path_matches:
            issues.append(QualityIssue(
                issue_type="hardcoded_paths",
                severity="medium",
                description=f"Found {len(path_matches)} hardcoded file paths",
                pipeline=pipeline_name,
                file_path=str(file_path),
                suggestion="Use relative paths or configuration variables"
            ))
        
        # Debug output issues
        debug_matches = self.quality_patterns['debug_output'].findall(content)
        if debug_matches:
            issues.append(QualityIssue(
                issue_type="debug_output",
                severity="low",
                description=f"Found {len(debug_matches)} debug statements",
                pipeline=pipeline_name,
                file_path=str(file_path),
                suggestion="Remove debug output from production files"
            ))
        
        # Empty file issue
        if not content.strip():
            issues.append(QualityIssue(
                issue_type="empty_file",
                severity="critical",
                description="Output file is empty",
                pipeline=pipeline_name,
                file_path=str(file_path),
                suggestion="Investigate why no content was generated"
            ))
        
        return issues

    def _calculate_category_scores(self, quality_results: Dict[str, Any], 
                                 validation_summary: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate scores for each quality category."""
        scores = {}
        
        # Content completeness score
        content_scores = []
        for file_analysis in quality_results['content_analysis'].values():
            content_scores.append(file_analysis.get('completeness', 0))
        scores['content_completeness'] = np.mean(content_scores) if content_scores else 0
        
        # Format correctness score
        format_scores = []
        for file_analysis in quality_results['format_analysis'].values():
            if file_analysis.get('valid_format', True):
                format_scores.append(100.0)
            else:
                format_scores.append(0.0)
        scores['format_correctness'] = np.mean(format_scores) if format_scores else 0
        
        # Template rendering score (based on template variables found)
        template_issues = sum(
            len([issue for issue in file_issues if 'template' in issue or 'loop' in issue])
            for file_issues in quality_results['issues_by_file'].values()
        )
        scores['template_rendering'] = max(0, 100 - (template_issues * 20))
        
        # Conversational markers score
        conv_issues = sum(
            len([issue for issue in file_issues if 'conversational' in issue])
            for file_issues in quality_results['issues_by_file'].values()
        )
        scores['conversational_markers'] = max(0, 100 - (conv_issues * 15))
        
        # Error indicators score
        error_issues = sum(
            len([issue for issue in file_issues if 'error' in issue])
            for file_issues in quality_results['issues_by_file'].values()
        )
        scores['error_indicators'] = max(0, 100 - (error_issues * 25))
        
        # Output consistency score (based on validation summary)
        if validation_summary:
            base_score = validation_summary.get('quality_score', 50)
            scores['output_consistency'] = base_score
        else:
            scores['output_consistency'] = 50  # Default when no validation available
        
        # File generation score
        expected_files = quality_results['files_analyzed']
        if expected_files > 0:
            empty_files = sum(
                1 for file_issues in quality_results['issues_by_file'].values()
                if 'empty_file' in file_issues
            )
            scores['file_generation'] = max(0, 100 - (empty_files / expected_files * 100))
        else:
            scores['file_generation'] = 0
        
        return scores

    def _calculate_overall_score(self, category_scores: Dict[str, float]) -> float:
        """Calculate weighted overall quality score."""
        if not category_scores:
            return 0.0
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for category, score in category_scores.items():
            weight = self.quality_weights.get(category, 0.1)
            weighted_sum += score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _collect_quality_issues(self, quality_results: Dict[str, Any], pipeline_name: str) -> List[QualityIssue]:
        """Collect all quality issues found during analysis."""
        all_issues = []
        
        # Count issue frequencies
        issue_counts = defaultdict(int)
        
        for file_path, file_issues in quality_results['issues_by_file'].items():
            for issue_type in file_issues:
                issue_counts[issue_type] += 1
        
        # Create consolidated issues with frequencies
        for issue_type, count in issue_counts.items():
            # Determine severity based on issue type
            if issue_type in ['empty_file', 'unrendered_templates', 'unrendered_loops']:
                severity = 'critical' if issue_type == 'empty_file' else 'high'
            elif issue_type in ['error_indicators', 'placeholder_text']:
                severity = 'high' if issue_type == 'error_indicators' else 'medium'
            else:
                severity = 'medium'
            
            all_issues.append(QualityIssue(
                issue_type=issue_type,
                severity=severity,
                description=f"Found {count} instances across output files",
                pipeline=pipeline_name,
                frequency=count,
                suggestion=self._get_issue_suggestion(issue_type)
            ))
        
        return sorted(all_issues, key=lambda x: (
            self.severity_scores.get(x.severity, 100),
            -x.frequency
        ))

    def _get_issue_suggestion(self, issue_type: str) -> str:
        """Get suggestion for fixing a specific issue type."""
        suggestions = {
            'unrendered_templates': 'Ensure all template variables are properly rendered with actual values',
            'unrendered_loops': 'Verify loop processing and variable substitution in pipeline logic',
            'conversational_markers': 'Configure LLM to produce direct, non-conversational output',
            'error_indicators': 'Investigate and fix errors in pipeline execution or output processing',
            'placeholder_text': 'Replace all placeholder text with actual generated content',
            'hardcoded_paths': 'Use relative paths or configuration variables instead of hardcoded paths',
            'debug_output': 'Remove debug statements from production pipeline outputs',
            'empty_file': 'Investigate pipeline logic to ensure content generation is working',
            'file_read_error': 'Check file permissions and encoding issues'
        }
        
        return suggestions.get(issue_type, 'Review and address this quality issue')

    def analyze_all_pipelines(self) -> Dict[str, QualityMetrics]:
        """Analyze quality metrics for all available pipelines."""
        logger.info("Analyzing quality for all available pipelines...")
        
        all_metrics = {}
        
        # Find all pipeline output directories
        if self.outputs_dir.exists():
            for pipeline_dir in self.outputs_dir.iterdir():
                if pipeline_dir.is_dir() and not pipeline_dir.name.startswith('.'):
                    pipeline_name = pipeline_dir.name
                    
                    try:
                        metrics = self.analyze_pipeline_quality(pipeline_name)
                        all_metrics[pipeline_name] = metrics
                    except Exception as e:
                        logger.error(f"Failed to analyze pipeline {pipeline_name}: {e}")
                        # Create error metrics
                        all_metrics[pipeline_name] = QualityMetrics(
                            pipeline_name=pipeline_name,
                            overall_score=0.0,
                            category_scores={},
                            issues=[QualityIssue("analysis_error", "critical", str(e), pipeline_name)],
                            output_quality={}
                        )
        
        logger.info(f"Completed quality analysis for {len(all_metrics)} pipelines")
        return all_metrics

    def generate_quality_report(self, metrics: Dict[str, QualityMetrics]) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        if not metrics:
            return {'error': 'No quality metrics available'}
        
        # Overall statistics
        overall_scores = [m.overall_score for m in metrics.values()]
        category_scores = defaultdict(list)
        all_issues = []
        
        for pipeline_metrics in metrics.values():
            all_issues.extend(pipeline_metrics.issues)
            for category, score in pipeline_metrics.category_scores.items():
                category_scores[category].append(score)
        
        # Issue analysis
        issue_types = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for issue in all_issues:
            issue_types[issue.issue_type] += issue.frequency
            severity_counts[issue.severity] += issue.frequency
        
        # Quality distribution
        score_ranges = {
            'excellent': len([s for s in overall_scores if s >= 90]),
            'good': len([s for s in overall_scores if 80 <= s < 90]),
            'fair': len([s for s in overall_scores if 60 <= s < 80]),
            'poor': len([s for s in overall_scores if s < 60])
        }
        
        return {
            'summary': {
                'total_pipelines_analyzed': len(metrics),
                'average_overall_score': np.mean(overall_scores),
                'median_overall_score': np.median(overall_scores),
                'score_standard_deviation': np.std(overall_scores),
                'total_issues_found': sum(issue.frequency for issue in all_issues)
            },
            'score_distribution': score_ranges,
            'category_averages': {
                category: np.mean(scores) for category, scores in category_scores.items()
            },
            'top_issues': dict(sorted(issue_types.items(), key=lambda x: x[1], reverse=True)[:10]),
            'severity_breakdown': dict(severity_counts),
            'pipeline_rankings': {
                'best_quality': sorted(metrics.items(), key=lambda x: x[1].overall_score, reverse=True)[:10],
                'needs_improvement': sorted(metrics.items(), key=lambda x: x[1].overall_score)[:10]
            },
            'recommendations': self._generate_quality_recommendations(metrics)
        }

    def _generate_quality_recommendations(self, metrics: Dict[str, QualityMetrics]) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        # Analyze common issues
        all_issues = []
        for pipeline_metrics in metrics.values():
            all_issues.extend(pipeline_metrics.issues)
        
        issue_types = defaultdict(int)
        for issue in all_issues:
            issue_types[issue.issue_type] += issue.frequency
        
        # Generate recommendations based on most common issues
        if issue_types.get('unrendered_templates', 0) > 5:
            recommendations.append("Implement template validation to ensure all variables are rendered")
        
        if issue_types.get('conversational_markers', 0) > 3:
            recommendations.append("Configure LLM system prompts to eliminate conversational language")
        
        if issue_types.get('empty_file', 0) > 2:
            recommendations.append("Add output validation to detect and prevent empty file generation")
        
        if issue_types.get('error_indicators', 0) > 2:
            recommendations.append("Implement comprehensive error handling and logging")
        
        # Score-based recommendations
        overall_scores = [m.overall_score for m in metrics.values()]
        avg_score = np.mean(overall_scores)
        
        if avg_score < 70:
            recommendations.append("Overall quality is below target - implement systematic quality improvements")
        elif avg_score < 85:
            recommendations.append("Quality is good but has room for improvement - focus on top issue categories")
        
        # Category-specific recommendations
        category_scores = defaultdict(list)
        for pipeline_metrics in metrics.values():
            for category, score in pipeline_metrics.category_scores.items():
                category_scores[category].append(score)
        
        for category, scores in category_scores.items():
            avg_category_score = np.mean(scores)
            if avg_category_score < 75:
                recommendations.append(f"Improve {category.replace('_', ' ')} - currently scoring {avg_category_score:.1f}%")
        
        if not recommendations:
            recommendations.append("Quality metrics are within acceptable ranges - maintain current standards")
        
        return recommendations


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quality Metrics Analyzer")
    parser.add_argument("--root", default=".", help="Repository root path")
    parser.add_argument("--pipeline", help="Analyze specific pipeline")
    parser.add_argument("--all", action='store_true', help="Analyze all pipelines")
    parser.add_argument("--report", action='store_true', help="Generate quality report")
    parser.add_argument("--output", help="Output file for report")
    parser.add_argument("--verbose", action='store_true', help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    analyzer = QualityAnalyzer(args.root)
    
    if args.pipeline:
        # Analyze specific pipeline
        metrics = analyzer.analyze_pipeline_quality(args.pipeline)
        print(f"Quality Analysis for {args.pipeline}")
        print(f"Overall Score: {metrics.overall_score:.1f}%")
        print(f"Issues Found: {len(metrics.issues)}")
        
        if metrics.issues:
            print("\nTop Issues:")
            for issue in metrics.issues[:5]:
                print(f"  - {issue.severity.upper()}: {issue.description}")
        
    elif args.all:
        # Analyze all pipelines
        all_metrics = analyzer.analyze_all_pipelines()
        
        if args.report:
            # Generate comprehensive report
            report = analyzer.generate_quality_report(all_metrics)
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                print(f"Quality report saved to: {args.output}")
            else:
                print(json.dumps(report, indent=2, default=str))
        else:
            # Simple summary
            print(f"Analyzed {len(all_metrics)} pipelines")
            scores = [m.overall_score for m in all_metrics.values()]
            print(f"Average Quality Score: {np.mean(scores):.1f}%")
            print(f"Best: {max(scores):.1f}%, Worst: {min(scores):.1f}%")
    
    else:
        print("Use --pipeline <name> to analyze a specific pipeline")
        print("Use --all to analyze all pipelines")
        print("Add --report for comprehensive quality report")


if __name__ == "__main__":
    main()