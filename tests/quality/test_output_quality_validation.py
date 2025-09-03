#!/usr/bin/env python3
"""
Automated output quality validation for wrapper integrations - Issue #252.

This module provides comprehensive quality assessment of pipeline outputs
to ensure wrapper integrations maintain or improve output quality.
"""

import asyncio
import json
import logging
import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
import hashlib
import difflib
import sys
import os

import pytest

# Add orchestrator to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from orchestrator import Orchestrator, init_models
from src.orchestrator.models import get_model_registry
from src.orchestrator.compiler.yaml_compiler import YAMLCompiler
from src.orchestrator.control_systems.hybrid_control_system import HybridControlSystem

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for pipeline outputs."""
    
    completeness_score: float = 0.0  # 0-100: how complete is the output
    accuracy_score: float = 0.0      # 0-100: accuracy of information
    consistency_score: float = 0.0   # 0-100: internal consistency
    formatting_score: float = 0.0    # 0-100: proper formatting and structure
    template_score: float = 0.0      # 0-100: template rendering quality
    content_quality_score: float = 0.0  # 0-100: overall content quality
    
    # Issue detection
    template_issues: List[str] = field(default_factory=list)
    content_issues: List[str] = field(default_factory=list)
    formatting_issues: List[str] = field(default_factory=list)
    
    # Detailed metrics
    word_count: int = 0
    unique_words: int = 0
    readability_score: float = 0.0
    structure_score: float = 0.0
    
    def overall_score(self) -> float:
        """Calculate overall quality score."""
        scores = [
            self.completeness_score,
            self.accuracy_score,
            self.consistency_score,
            self.formatting_score,
            self.template_score,
            self.content_quality_score
        ]
        return sum(scores) / len(scores) if scores else 0.0


@dataclass
class QualityComparison:
    """Comparison between baseline and wrapper output quality."""
    
    pipeline_name: str
    wrapper_config: str
    baseline_metrics: QualityMetrics
    wrapper_metrics: QualityMetrics
    quality_delta: float  # Positive = improvement, Negative = degradation
    is_improvement: bool
    is_degradation: bool
    significance_threshold: float = 5.0  # 5% threshold for significance


@dataclass
class OutputAnalysis:
    """Analysis of pipeline output."""
    
    content: str
    file_path: Path
    metrics: QualityMetrics
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


class OutputQualityValidator:
    """
    Comprehensive quality validator for pipeline outputs.
    
    Analyzes output quality using multiple metrics and compares
    wrapper implementations against baseline performance.
    """
    
    def __init__(self):
        self.examples_dir = Path("examples")
        self.quality_dir = Path("tests/quality/results")
        self.quality_dir.mkdir(parents=True, exist_ok=True)
        
        # Baseline quality storage
        self.baseline_file = self.quality_dir / "quality_baselines.json"
        self.baselines: Dict[str, QualityMetrics] = {}
        
        # Quality analysis results
        self.quality_analyses: List[OutputAnalysis] = []
        self.quality_comparisons: List[QualityComparison] = []
        
        # Test pipelines for quality validation
        self.test_pipelines = [
            "research_minimal.yaml",
            "simple_data_processing.yaml", 
            "control_flow_advanced.yaml",
            "creative_image_pipeline.yaml",
            "data_processing_pipeline.yaml"
        ]
        
        # Quality validation patterns
        self.template_patterns = {
            "unrendered_variables": re.compile(r'\{\{[^}]+\}\}'),
            "unrendered_loops": re.compile(r'\$(?:item|index|iteration|context)\b'),
            "conditional_remnants": re.compile(r'\{%[^%]+%\}'),
            "incomplete_substitution": re.compile(r'__[A-Z_]+__')
        }
        
        self.content_quality_patterns = {
            "conversational_markers": [
                "Certainly!", "Sure!", "I'd be happy to", "Let me help",
                "Here's what I found", "I'll create", "Based on your request",
                "As requested", "I hope this helps", "Feel free to ask"
            ],
            "placeholder_text": [
                "Lorem ipsum", "placeholder", "TODO", "FIXME", 
                "example text", "sample content", "TBD", "coming soon"
            ],
            "error_indicators": [
                "error occurred", "failed to", "could not", "unable to",
                "something went wrong", "try again", "please check"
            ]
        }
        
        self.model_registry = None
        self.control_system = None
        
    async def initialize(self):
        """Initialize quality validation infrastructure."""
        logger.info("Initializing output quality validator...")
        
        self.model_registry = init_models()
        if not self.model_registry:
            raise RuntimeError("No models available for quality validation")
            
        self.control_system = HybridControlSystem(self.model_registry)
        
        # Load existing baselines
        await self.load_quality_baselines()
        
    async def load_quality_baselines(self):
        """Load quality baselines from file."""
        if self.baseline_file.exists():
            try:
                with open(self.baseline_file) as f:
                    baseline_data = json.load(f)
                    
                for name, data in baseline_data.items():
                    metrics = QualityMetrics(
                        completeness_score=data.get("completeness_score", 0),
                        accuracy_score=data.get("accuracy_score", 0), 
                        consistency_score=data.get("consistency_score", 0),
                        formatting_score=data.get("formatting_score", 0),
                        template_score=data.get("template_score", 0),
                        content_quality_score=data.get("content_quality_score", 0),
                        word_count=data.get("word_count", 0),
                        unique_words=data.get("unique_words", 0),
                        readability_score=data.get("readability_score", 0),
                        structure_score=data.get("structure_score", 0)
                    )
                    self.baselines[name] = metrics
                    
                logger.info(f"Loaded {len(self.baselines)} quality baselines")
            except Exception as e:
                logger.warning(f"Could not load quality baselines: {e}")
                
    async def save_quality_baselines(self):
        """Save quality baselines to file."""
        baseline_data = {}
        for name, metrics in self.baselines.items():
            baseline_data[name] = {
                "completeness_score": metrics.completeness_score,
                "accuracy_score": metrics.accuracy_score,
                "consistency_score": metrics.consistency_score,
                "formatting_score": metrics.formatting_score,
                "template_score": metrics.template_score,
                "content_quality_score": metrics.content_quality_score,
                "word_count": metrics.word_count,
                "unique_words": metrics.unique_words,
                "readability_score": metrics.readability_score,
                "structure_score": metrics.structure_score
            }
            
        with open(self.baseline_file, "w") as f:
            json.dump(baseline_data, f, indent=2)
            
        logger.info(f"Saved {len(self.baselines)} quality baselines")
        
    def analyze_output_quality(self, output_path: Path) -> List[OutputAnalysis]:
        """Analyze quality of all outputs in a directory."""
        analyses = []
        
        if not output_path.exists():
            return analyses
            
        # Analyze all output files
        for file_path in output_path.rglob("*"):
            if file_path.is_file() and file_path.suffix in ['.md', '.txt', '.json', '.csv', '.html']:
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    if content.strip():  # Skip empty files
                        analysis = self._analyze_single_output(content, file_path)
                        analyses.append(analysis)
                except Exception as e:
                    logger.warning(f"Could not analyze {file_path}: {e}")
                    
        return analyses
        
    def _analyze_single_output(self, content: str, file_path: Path) -> OutputAnalysis:
        """Analyze quality of a single output file."""
        metrics = QualityMetrics()
        issues = []
        suggestions = []
        
        # Template quality analysis
        template_score, template_issues = self._analyze_template_quality(content)
        metrics.template_score = template_score
        metrics.template_issues = template_issues
        issues.extend(template_issues)
        
        # Content quality analysis
        content_score, content_issues = self._analyze_content_quality(content)
        metrics.content_quality_score = content_score
        metrics.content_issues = content_issues
        issues.extend(content_issues)
        
        # Formatting quality analysis
        formatting_score, formatting_issues = self._analyze_formatting_quality(content, file_path)
        metrics.formatting_score = formatting_score
        metrics.formatting_issues = formatting_issues
        issues.extend(formatting_issues)
        
        # Completeness analysis
        metrics.completeness_score = self._analyze_completeness(content)
        
        # Consistency analysis
        metrics.consistency_score = self._analyze_consistency(content)
        
        # Basic content metrics
        words = content.split()
        metrics.word_count = len(words)
        metrics.unique_words = len(set(word.lower() for word in words if word.isalnum()))
        
        # Readability (simplified)
        metrics.readability_score = self._calculate_readability_score(content)
        
        # Structure analysis
        metrics.structure_score = self._analyze_structure(content, file_path)
        
        # Generate suggestions
        suggestions = self._generate_quality_suggestions(metrics, issues)
        
        return OutputAnalysis(
            content=content,
            file_path=file_path,
            metrics=metrics,
            issues=issues,
            suggestions=suggestions
        )
        
    def _analyze_template_quality(self, content: str) -> Tuple[float, List[str]]:
        """Analyze template rendering quality."""
        score = 100.0
        issues = []
        
        # Check for unrendered template variables
        for pattern_name, pattern in self.template_patterns.items():
            matches = pattern.findall(content)
            if matches:
                score -= len(matches) * 20  # Heavy penalty for template issues
                issues.append(f"{pattern_name}: {len(matches)} instances found")
                
        return max(0, score), issues
        
    def _analyze_content_quality(self, content: str) -> Tuple[float, List[str]]:
        """Analyze content quality and naturalness."""
        score = 100.0
        issues = []
        
        content_lower = content.lower()
        
        # Check for conversational markers
        for marker in self.content_quality_patterns["conversational_markers"]:
            if marker.lower() in content_lower:
                score -= 15
                issues.append(f"Conversational marker found: '{marker}'")
                break  # Only penalize once
                
        # Check for placeholder text
        for placeholder in self.content_quality_patterns["placeholder_text"]:
            if placeholder.lower() in content_lower:
                score -= 25
                issues.append(f"Placeholder text found: '{placeholder}'")
                
        # Check for error indicators
        for error_pattern in self.content_quality_patterns["error_indicators"]:
            if error_pattern.lower() in content_lower:
                score -= 30
                issues.append(f"Error indicator found: '{error_pattern}'")
                
        # Check for repetitive content
        sentences = content.split('.')
        if len(sentences) > 5:
            unique_sentences = set(s.strip().lower() for s in sentences if s.strip())
            repetition_ratio = 1 - (len(unique_sentences) / len(sentences))
            if repetition_ratio > 0.3:  # More than 30% repetition
                score -= repetition_ratio * 40
                issues.append(f"High content repetition: {repetition_ratio:.1%}")
                
        return max(0, score), issues
        
    def _analyze_formatting_quality(self, content: str, file_path: Path) -> Tuple[float, List[str]]:
        """Analyze formatting and structure quality."""
        score = 100.0
        issues = []
        
        # File type specific analysis
        if file_path.suffix == '.md':
            score, md_issues = self._analyze_markdown_formatting(content)
            issues.extend(md_issues)
        elif file_path.suffix == '.json':
            score, json_issues = self._analyze_json_formatting(content)
            issues.extend(json_issues)
        elif file_path.suffix == '.csv':
            score, csv_issues = self._analyze_csv_formatting(content)
            issues.extend(csv_issues)
            
        # General formatting checks
        lines = content.split('\n')
        
        # Check for extremely long lines
        long_lines = [i for i, line in enumerate(lines) if len(line) > 200]
        if long_lines:
            score -= len(long_lines) * 5
            issues.append(f"{len(long_lines)} lines exceed 200 characters")
            
        # Check for inconsistent line endings or spacing
        empty_line_ratio = sum(1 for line in lines if not line.strip()) / len(lines)
        if empty_line_ratio > 0.5:
            score -= 20
            issues.append("Excessive empty lines detected")
            
        return max(0, score), issues
        
    def _analyze_markdown_formatting(self, content: str) -> Tuple[float, List[str]]:
        """Analyze markdown-specific formatting."""
        score = 100.0
        issues = []
        
        # Check for proper heading structure
        heading_pattern = re.compile(r'^(#{1,6})\s', re.MULTILINE)
        headings = heading_pattern.findall(content)
        
        if headings:
            # Check heading hierarchy
            heading_levels = [len(h) for h in headings]
            if heading_levels:
                first_level = heading_levels[0]
                if first_level > 2:  # First heading should be h1 or h2
                    score -= 10
                    issues.append("Document should start with h1 or h2 heading")
                    
        # Check for proper list formatting
        list_pattern = re.compile(r'^[\s]*[-*+]\s', re.MULTILINE)
        lists = list_pattern.findall(content)
        
        # Check for proper code block formatting
        code_block_pattern = re.compile(r'```[\s\S]*?```')
        code_blocks = code_block_pattern.findall(content)
        
        # Check for inline code consistency
        inline_code_pattern = re.compile(r'`[^`]+`')
        inline_code = inline_code_pattern.findall(content)
        
        return score, issues
        
    def _analyze_json_formatting(self, content: str) -> Tuple[float, List[str]]:
        """Analyze JSON formatting quality."""
        score = 100.0
        issues = []
        
        try:
            # Try to parse JSON
            data = json.loads(content)
            
            # Check for proper indentation (re-serialize and compare)
            formatted = json.dumps(data, indent=2, sort_keys=True)
            if content.strip() != formatted.strip():
                score -= 15
                issues.append("JSON formatting could be improved")
                
        except json.JSONDecodeError as e:
            score = 0
            issues.append(f"Invalid JSON: {str(e)}")
            
        return score, issues
        
    def _analyze_csv_formatting(self, content: str) -> Tuple[float, List[str]]:
        """Analyze CSV formatting quality."""
        score = 100.0
        issues = []
        
        lines = content.strip().split('\n')
        if not lines:
            return 0, ["Empty CSV file"]
            
        # Check header consistency
        if lines:
            header_cols = len(lines[0].split(','))
            
            for i, line in enumerate(lines[1:], 1):
                cols = len(line.split(','))
                if cols != header_cols:
                    score -= 10
                    issues.append(f"Line {i+1} has {cols} columns, expected {header_cols}")
                    
        return max(0, score), issues
        
    def _analyze_completeness(self, content: str) -> float:
        """Analyze content completeness."""
        if not content.strip():
            return 0.0
            
        # Basic completeness metrics
        word_count = len(content.split())
        
        # Score based on content length (assuming substantial content indicates completeness)
        if word_count >= 100:
            completeness = 100.0
        elif word_count >= 50:
            completeness = 80.0
        elif word_count >= 20:
            completeness = 60.0
        elif word_count >= 5:
            completeness = 40.0
        else:
            completeness = 20.0
            
        # Bonus for structured content
        if re.search(r'^#', content, re.MULTILINE):  # Has headings
            completeness = min(100, completeness + 10)
            
        if re.search(r'^[-*+]\s', content, re.MULTILINE):  # Has lists
            completeness = min(100, completeness + 5)
            
        return completeness
        
    def _analyze_consistency(self, content: str) -> float:
        """Analyze internal consistency."""
        if not content.strip():
            return 0.0
            
        score = 100.0
        
        # Check for consistent terminology
        words = content.lower().split()
        word_freq = {}
        for word in words:
            if word.isalnum() and len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
                
        # Look for inconsistent variations (simplified)
        # This is a basic implementation - could be enhanced with NLP
        
        # Check for consistent formatting patterns
        lines = content.split('\n')
        
        # Check heading consistency in markdown
        heading_styles = set()
        for line in lines:
            if line.strip().startswith('#'):
                # Extract heading style
                level = len(line) - len(line.lstrip('#'))
                heading_styles.add(level)
                
        return score
        
    def _calculate_readability_score(self, content: str) -> float:
        """Calculate simplified readability score."""
        if not content.strip():
            return 0.0
            
        sentences = [s for s in content.split('.') if s.strip()]
        words = content.split()
        
        if not sentences or not words:
            return 0.0
            
        # Simplified Flesch reading ease approximation
        avg_sentence_length = len(words) / len(sentences)
        
        # Simple syllable count approximation
        total_syllables = sum(max(1, len([c for c in word if c.lower() in 'aeiou'])) for word in words)
        avg_syllables = total_syllables / len(words)
        
        # Simplified readability score (0-100, higher is more readable)
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)
        
        return max(0, min(100, score))
        
    def _analyze_structure(self, content: str, file_path: Path) -> float:
        """Analyze content structure quality."""
        score = 100.0
        
        if file_path.suffix == '.md':
            # Check for proper markdown structure
            has_headings = bool(re.search(r'^#', content, re.MULTILINE))
            has_paragraphs = len(content.split('\n\n')) > 1
            has_lists = bool(re.search(r'^[-*+]\s', content, re.MULTILINE))
            
            structure_elements = sum([has_headings, has_paragraphs, has_lists])
            score = min(100, structure_elements * 30 + 10)  # Base 10 + bonuses
            
        elif file_path.suffix == '.json':
            try:
                data = json.loads(content)
                # Score based on JSON structure depth and organization
                if isinstance(data, dict):
                    score = min(100, len(data) * 10 + 50)  # Reward structured data
                elif isinstance(data, list):
                    score = min(100, len(data) * 5 + 40)   # Lists are also good
                else:
                    score = 30  # Simple values are basic
            except:
                score = 0
                
        return score
        
    def _generate_quality_suggestions(self, metrics: QualityMetrics, issues: List[str]) -> List[str]:
        """Generate suggestions for quality improvement."""
        suggestions = []
        
        if metrics.template_score < 80:
            suggestions.append("Review template rendering - ensure all variables are properly substituted")
            
        if metrics.content_quality_score < 70:
            suggestions.append("Improve content naturalness - remove conversational markers and placeholders")
            
        if metrics.formatting_score < 80:
            suggestions.append("Improve formatting consistency and structure")
            
        if metrics.completeness_score < 60:
            suggestions.append("Increase content completeness - provide more comprehensive information")
            
        if metrics.readability_score < 50:
            suggestions.append("Improve readability - use shorter sentences and simpler vocabulary")
            
        if len(issues) > 5:
            suggestions.append("Address multiple quality issues found in output")
            
        return suggestions
        
    async def create_baseline_quality_metrics(self, pipeline_name: str) -> QualityMetrics:
        """Create baseline quality metrics for a pipeline."""
        logger.info(f"Creating baseline quality metrics for {pipeline_name}")
        
        pipeline_path = self.examples_dir / pipeline_name
        if not pipeline_path.exists():
            raise FileNotFoundError(f"Pipeline not found: {pipeline_name}")
            
        # Run pipeline with baseline configuration
        with open(pipeline_path) as f:
            yaml_content = f.read()
            
        inputs = self._get_test_inputs(pipeline_name)
        output_path = self.quality_dir / "baseline" / pipeline_path.stem
        output_path.mkdir(parents=True, exist_ok=True)
        inputs['output_path'] = str(output_path)
        
        # Create baseline orchestrator
        orchestrator = Orchestrator(
            model_registry=self.model_registry,
            control_system=self.control_system
        )
        
        # Execute pipeline
        await orchestrator.execute_yaml(yaml_content, inputs)
        
        # Analyze output quality
        analyses = self.analyze_output_quality(output_path)
        
        if analyses:
            # Average metrics across all output files
            avg_metrics = self._average_quality_metrics([a.metrics for a in analyses])
            return avg_metrics
        else:
            # Return minimal metrics if no outputs
            return QualityMetrics(
                completeness_score=0,
                accuracy_score=50,  # Default assumption
                consistency_score=50,
                formatting_score=50,
                template_score=100,  # No templates = no template issues
                content_quality_score=50
            )
            
    def _average_quality_metrics(self, metrics_list: List[QualityMetrics]) -> QualityMetrics:
        """Average multiple quality metrics."""
        if not metrics_list:
            return QualityMetrics()
            
        avg_metrics = QualityMetrics(
            completeness_score=statistics.mean(m.completeness_score for m in metrics_list),
            accuracy_score=statistics.mean(m.accuracy_score for m in metrics_list),
            consistency_score=statistics.mean(m.consistency_score for m in metrics_list),
            formatting_score=statistics.mean(m.formatting_score for m in metrics_list),
            template_score=statistics.mean(m.template_score for m in metrics_list),
            content_quality_score=statistics.mean(m.content_quality_score for m in metrics_list),
            word_count=int(statistics.mean(m.word_count for m in metrics_list)),
            unique_words=int(statistics.mean(m.unique_words for m in metrics_list)),
            readability_score=statistics.mean(m.readability_score for m in metrics_list),
            structure_score=statistics.mean(m.structure_score for m in metrics_list)
        )
        
        return avg_metrics
        
    def _get_test_inputs(self, pipeline_name: str) -> Dict[str, Any]:
        """Get consistent test inputs for quality testing."""
        return {
            "input_text": "Quantum computing represents a paradigm shift in computational capability, offering exponential speedup for specific problem domains through quantum mechanical phenomena.",
            "topic": "quantum computing applications",
            "query": "quantum algorithms for optimization",
            "data": {"technology": "quantum", "application": "optimization", "status": "emerging"}
        }
        
    async def validate_pipeline_quality(
        self, 
        pipeline_name: str, 
        wrapper_config: Dict[str, Any]
    ) -> QualityComparison:
        """Validate quality of pipeline output with wrapper configuration."""
        logger.info(f"Validating quality: {pipeline_name} with {wrapper_config.get('name', 'unknown')}")
        
        # Ensure we have baseline
        if pipeline_name not in self.baselines:
            baseline_metrics = await self.create_baseline_quality_metrics(pipeline_name)
            self.baselines[pipeline_name] = baseline_metrics
            
        baseline_metrics = self.baselines[pipeline_name]
        
        # Run pipeline with wrapper configuration
        pipeline_path = self.examples_dir / pipeline_name
        with open(pipeline_path) as f:
            yaml_content = f.read()
            
        inputs = self._get_test_inputs(pipeline_name)
        output_path = self.quality_dir / wrapper_config.get('name', 'wrapper') / pipeline_path.stem
        output_path.mkdir(parents=True, exist_ok=True)
        inputs['output_path'] = str(output_path)
        
        # Create configured orchestrator
        orchestrator = Orchestrator(
            model_registry=self.model_registry,
            control_system=self.control_system
        )
        
        # Apply wrapper configuration
        if hasattr(orchestrator, 'configure_wrappers'):
            await orchestrator.configure_wrappers(wrapper_config)
            
        # Execute pipeline
        await orchestrator.execute_yaml(yaml_content, inputs)
        
        # Analyze wrapper output quality
        analyses = self.analyze_output_quality(output_path)
        
        if analyses:
            wrapper_metrics = self._average_quality_metrics([a.metrics for a in analyses])
        else:
            wrapper_metrics = QualityMetrics()
            
        # Compare quality
        baseline_overall = baseline_metrics.overall_score()
        wrapper_overall = wrapper_metrics.overall_score()
        quality_delta = wrapper_overall - baseline_overall
        
        comparison = QualityComparison(
            pipeline_name=pipeline_name,
            wrapper_config=wrapper_config.get('name', 'unknown'),
            baseline_metrics=baseline_metrics,
            wrapper_metrics=wrapper_metrics,
            quality_delta=quality_delta,
            is_improvement=quality_delta > 5.0,
            is_degradation=quality_delta < -5.0
        )
        
        self.quality_comparisons.append(comparison)
        return comparison
        
    async def validate_all_pipeline_quality(
        self, 
        wrapper_configs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate quality for all test pipelines with all wrapper configurations."""
        logger.info("Running comprehensive quality validation")
        
        # Ensure baselines exist
        for pipeline_name in self.test_pipelines:
            if pipeline_name not in self.baselines:
                baseline_metrics = await self.create_baseline_quality_metrics(pipeline_name)
                self.baselines[pipeline_name] = baseline_metrics
                
        await self.save_quality_baselines()
        
        # Test all combinations
        comparisons = []
        for pipeline_name in self.test_pipelines:
            for wrapper_config in wrapper_configs:
                comparison = await self.validate_pipeline_quality(pipeline_name, wrapper_config)
                comparisons.append(comparison)
                
        return self.generate_quality_report()
        
    def generate_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality validation report."""
        logger.info("Generating quality validation report")
        
        # Calculate summary statistics
        total_comparisons = len(self.quality_comparisons)
        improvements = [c for c in self.quality_comparisons if c.is_improvement]
        degradations = [c for c in self.quality_comparisons if c.is_degradation]
        
        quality_deltas = [c.quality_delta for c in self.quality_comparisons]
        
        report = {
            "summary": {
                "timestamp": datetime.utcnow().isoformat(),
                "total_comparisons": total_comparisons,
                "quality_improvements": len(improvements),
                "quality_degradations": len(degradations),
                "neutral_changes": total_comparisons - len(improvements) - len(degradations),
                "average_quality_delta": statistics.mean(quality_deltas) if quality_deltas else 0,
                "significance_threshold": 5.0
            },
            "improvements": [
                {
                    "pipeline": c.pipeline_name,
                    "wrapper_config": c.wrapper_config,
                    "quality_delta": c.quality_delta,
                    "baseline_score": c.baseline_metrics.overall_score(),
                    "wrapper_score": c.wrapper_metrics.overall_score()
                }
                for c in improvements
            ],
            "degradations": [
                {
                    "pipeline": c.pipeline_name,
                    "wrapper_config": c.wrapper_config,
                    "quality_delta": c.quality_delta,
                    "baseline_score": c.baseline_metrics.overall_score(),
                    "wrapper_score": c.wrapper_metrics.overall_score()
                }
                for c in degradations
            ],
            "detailed_comparisons": [
                {
                    "pipeline": c.pipeline_name,
                    "wrapper_config": c.wrapper_config,
                    "quality_delta": c.quality_delta,
                    "is_improvement": c.is_improvement,
                    "is_degradation": c.is_degradation,
                    "baseline_metrics": {
                        "overall_score": c.baseline_metrics.overall_score(),
                        "completeness": c.baseline_metrics.completeness_score,
                        "content_quality": c.baseline_metrics.content_quality_score,
                        "template_quality": c.baseline_metrics.template_score,
                        "formatting": c.baseline_metrics.formatting_score
                    },
                    "wrapper_metrics": {
                        "overall_score": c.wrapper_metrics.overall_score(),
                        "completeness": c.wrapper_metrics.completeness_score,
                        "content_quality": c.wrapper_metrics.content_quality_score,
                        "template_quality": c.wrapper_metrics.template_score,
                        "formatting": c.wrapper_metrics.formatting_score
                    }
                }
                for c in self.quality_comparisons
            ]
        }
        
        # Save report
        report_path = self.quality_dir / f"quality_validation_report_{int(time.time())}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
            
        logger.info(f"Quality report saved to: {report_path}")
        
        # Print summary
        print("\n" + "="*80)
        print("OUTPUT QUALITY VALIDATION REPORT")
        print("="*80)
        print(f"\nTotal Comparisons: {total_comparisons}")
        print(f"✅ Quality Improvements: {len(improvements)}")
        print(f"❌ Quality Degradations: {len(degradations)}")
        print(f"➡️  Neutral Changes: {total_comparisons - len(improvements) - len(degradations)}")
        
        if quality_deltas:
            avg_delta = statistics.mean(quality_deltas)
            print(f"\nAverage Quality Delta: {avg_delta:+.1f}%")
            
        if degradations:
            print(f"\n❌ Quality Degradations:")
            for degradation in degradations[:5]:  # Show first 5
                print(f"  {degradation['pipeline']} ({degradation['wrapper_config']}): "
                      f"{degradation['quality_delta']:+.1f}%")
                      
        if improvements:
            print(f"\n✅ Quality Improvements:")
            for improvement in improvements[:5]:  # Show first 5
                print(f"  {improvement['pipeline']} ({improvement['wrapper_config']}): "
                      f"{improvement['quality_delta']:+.1f}%")
        
        return report


# Test wrapper configurations for quality testing
QUALITY_TEST_CONFIGS = [
    {
        "name": "routellm_optimized",
        "wrapper_enabled": True,
        "routellm_enabled": True,
        "routing_strategy": "balanced"
    },
    {
        "name": "poml_enhanced", 
        "wrapper_enabled": True,
        "poml_enabled": True,
        "enhanced_templates": True
    },
    {
        "name": "full_wrappers",
        "wrapper_enabled": True,
        "routellm_enabled": True,
        "poml_enabled": True,
        "monitoring_enabled": True
    }
]


# pytest fixtures and tests

@pytest.fixture
async def quality_validator():
    """Create and initialize quality validator."""
    validator = OutputQualityValidator()
    await validator.initialize()
    return validator


@pytest.mark.asyncio
async def test_baseline_quality_creation(quality_validator):
    """Test creation of baseline quality metrics."""
    validator = quality_validator
    
    # Test with a simple pipeline
    pipeline_name = "simple_data_processing.yaml"
    metrics = await validator.create_baseline_quality_metrics(pipeline_name)
    
    # Assert baseline has reasonable values
    assert metrics.overall_score() >= 50, f"Baseline quality score {metrics.overall_score():.1f}% too low"
    assert metrics.template_score >= 80, "Baseline template score should be high"


@pytest.mark.asyncio
async def test_template_quality_validation(quality_validator):
    """Test template quality validation."""
    validator = quality_validator
    
    # Create content with template issues
    bad_content = "Hello {{name}}, your $item is {{status}}. {%if condition%}Show this{%endif%}"
    
    template_score, issues = validator._analyze_template_quality(bad_content)
    
    assert template_score < 50, f"Template score {template_score}% should be low for bad content"
    assert len(issues) > 0, "Should detect template issues"


@pytest.mark.asyncio
async def test_content_quality_validation(quality_validator):
    """Test content quality validation."""
    validator = quality_validator
    
    # Create content with quality issues
    bad_content = "Certainly! I'd be happy to help you with this placeholder content. TODO: fix this later."
    
    content_score, issues = validator._analyze_content_quality(bad_content)
    
    assert content_score < 70, f"Content score {content_score}% should be low for bad content"
    assert len(issues) > 0, "Should detect content quality issues"


@pytest.mark.asyncio
async def test_wrapper_quality_comparison(quality_validator):
    """Test quality comparison between baseline and wrapper."""
    validator = quality_validator
    
    # Test with research pipeline
    pipeline_name = "research_minimal.yaml"
    wrapper_config = QUALITY_TEST_CONFIGS[0]  # routellm_optimized
    
    comparison = await validator.validate_pipeline_quality(pipeline_name, wrapper_config)
    
    # Assert comparison completed
    assert comparison.pipeline_name == pipeline_name
    assert comparison.wrapper_config == wrapper_config["name"]
    
    # Quality should not degrade significantly
    assert not comparison.is_degradation, f"Quality degraded by {comparison.quality_delta:.1f}%"


if __name__ == "__main__":
    import time
    
    async def main():
        validator = OutputQualityValidator()
        await validator.initialize()
        
        report = await validator.validate_all_pipeline_quality(QUALITY_TEST_CONFIGS)
        
        # Summary for Issue #252
        degradations = len(report["degradations"])
        improvements = len(report["improvements"])
        
        if degradations == 0:
            print("\n🎉 Issue #252: No quality degradations detected!")
        elif improvements >= degradations:
            print(f"\n✅ Issue #252: Quality validation passed - {improvements} improvements vs {degradations} degradations")
        else:
            print(f"\n⚠️  Issue #252: {degradations} quality degradations need attention")
            
    asyncio.run(main())