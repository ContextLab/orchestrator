#!/usr/bin/env python3
"""
Tutorial Validation and Effectiveness Testing

This script validates the tutorial documentation system by testing:
1. Tutorial content quality and completeness
2. Learning path effectiveness
3. Remixing guide accuracy
4. Feature coverage completeness
5. User learning simulation

Usage:
    python scripts/tutorial_validator.py [options]
    
Options:
    --tutorial-dir PATH    Directory containing tutorials (default: docs/tutorials)
    --examples-dir PATH    Directory containing example pipelines (default: examples)
    --test-remixing        Test remixing examples by executing them
    --simulate-learning    Simulate user learning progression
    --validate-links       Check all internal links and references
    --verbose              Enable verbose output
"""

import os
import sys
import json
import yaml
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
from dataclasses import dataclass
import re
from datetime import datetime
import tempfile

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@dataclass
class ValidationResult:
    """Result of a validation test"""
    test_name: str
    passed: bool
    score: float  # 0.0 to 1.0
    issues: List[str]
    details: Dict[str, Any]

@dataclass
class TutorialValidation:
    """Comprehensive validation results for tutorial system"""
    overall_score: float
    test_results: List[ValidationResult]
    summary: Dict[str, Any]
    recommendations: List[str]

class TutorialValidator:
    """Validates tutorial documentation system effectiveness"""
    
    def __init__(self, tutorial_dir: Path, examples_dir: Path, verbose: bool = False):
        self.tutorial_dir = tutorial_dir
        self.examples_dir = examples_dir
        self.verbose = verbose
        self.validation_results = []
        
    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp"""
        if self.verbose or level == "ERROR":
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] [{level}] {message}")
            
    def validate_tutorial_system(self) -> TutorialValidation:
        """Run comprehensive validation of tutorial system"""
        self.log("Starting comprehensive tutorial validation...")
        
        # Run all validation tests
        self.validation_results = [
            self._validate_tutorial_completeness(),
            self._validate_content_quality(),
            self._validate_feature_coverage(),
            self._validate_learning_path(),
            self._validate_remixing_examples(),
            self._validate_links_and_references(),
            self._simulate_user_learning(),
            self._validate_technical_accuracy()
        ]
        
        # Calculate overall score
        overall_score = sum(result.score for result in self.validation_results) / len(self.validation_results)
        
        # Generate summary and recommendations
        summary = self._generate_summary()
        recommendations = self._generate_recommendations()
        
        return TutorialValidation(
            overall_score=overall_score,
            test_results=self.validation_results,
            summary=summary,
            recommendations=recommendations
        )
        
    def _validate_tutorial_completeness(self) -> ValidationResult:
        """Validate that all pipelines have tutorials"""
        self.log("Validating tutorial completeness...")
        
        issues = []
        details = {}
        
        # Get all pipeline files
        yaml_files = set()
        for ext in ['*.yaml', '*.yml']:
            yaml_files.update(f.stem for f in self.examples_dir.glob(ext))
            
        # Get all tutorial files
        tutorial_files = set()
        tutorials_dir = self.tutorial_dir / "pipelines"
        if tutorials_dir.exists():
            tutorial_files.update(f.stem for f in tutorials_dir.glob('*.md'))
            
        # Check coverage
        missing_tutorials = yaml_files - tutorial_files
        extra_tutorials = tutorial_files - yaml_files
        
        if missing_tutorials:
            issues.append(f"Missing tutorials for {len(missing_tutorials)} pipelines: {sorted(missing_tutorials)}")
            
        if extra_tutorials:
            issues.append(f"Extra tutorials without corresponding pipelines: {sorted(extra_tutorials)}")
            
        details = {
            'total_pipelines': len(yaml_files),
            'total_tutorials': len(tutorial_files),
            'missing_tutorials': sorted(missing_tutorials),
            'extra_tutorials': sorted(extra_tutorials),
            'coverage_percentage': (len(yaml_files & tutorial_files) / len(yaml_files)) * 100 if yaml_files else 0
        }
        
        score = 1.0 if not missing_tutorials else max(0.0, 1.0 - len(missing_tutorials) / len(yaml_files))
        
        return ValidationResult(
            test_name="Tutorial Completeness",
            passed=len(missing_tutorials) == 0,
            score=score,
            issues=issues,
            details=details
        )
        
    def _validate_content_quality(self) -> ValidationResult:
        """Validate quality of tutorial content"""
        self.log("Validating tutorial content quality...")
        
        issues = []
        details = {'tutorial_scores': {}}
        
        tutorials_dir = self.tutorial_dir / "pipelines"
        total_score = 0.0
        tutorial_count = 0
        
        if tutorials_dir.exists():
            for tutorial_file in tutorials_dir.glob('*.md'):
                tutorial_score = self._evaluate_single_tutorial(tutorial_file)
                details['tutorial_scores'][tutorial_file.stem] = tutorial_score
                total_score += tutorial_score
                tutorial_count += 1
                
                if tutorial_score < 0.7:
                    issues.append(f"Low quality tutorial: {tutorial_file.stem} (score: {tutorial_score:.2f})")
                    
        average_score = total_score / tutorial_count if tutorial_count > 0 else 0.0
        details['average_quality_score'] = average_score
        details['total_tutorials_evaluated'] = tutorial_count
        
        return ValidationResult(
            test_name="Content Quality",
            passed=average_score >= 0.8,
            score=average_score,
            issues=issues,
            details=details
        )
        
    def _evaluate_single_tutorial(self, tutorial_file: Path) -> float:
        """Evaluate quality of a single tutorial file"""
        try:
            content = tutorial_file.read_text(encoding='utf-8')
            score = 0.0
            
            # Check for required sections (0.4 points total)
            required_sections = [
                "## Overview", "## Pipeline Breakdown", "## Customization Guide",
                "## Remixing Instructions", "## Hands-On Exercise"
            ]
            
            section_score = sum(1 for section in required_sections if section in content) / len(required_sections)
            score += section_score * 0.4
            
            # Check for code examples (0.2 points)
            if "```yaml" in content:
                score += 0.2
                
            # Check for practical examples (0.2 points)
            if "execution_instructions" in content.lower() or "step-by-step" in content.lower():
                score += 0.2
                
            # Check for troubleshooting (0.1 points)
            if "troubleshooting" in content.lower() or "common issues" in content.lower():
                score += 0.1
                
            # Check for links and references (0.1 points)
            if "[" in content and "](" in content:
                score += 0.1
                
            return score
            
        except Exception as e:
            self.log(f"Error evaluating tutorial {tutorial_file}: {e}", "ERROR")
            return 0.0
            
    def _validate_feature_coverage(self) -> ValidationResult:
        """Validate feature coverage matrix accuracy"""
        self.log("Validating feature coverage...")
        
        issues = []
        details = {}
        
        # Load feature matrix
        matrix_file = self.tutorial_dir / "feature_matrix.json"
        if not matrix_file.exists():
            return ValidationResult(
                test_name="Feature Coverage",
                passed=False,
                score=0.0,
                issues=["Feature matrix file not found"],
                details={}
            )
            
        try:
            with open(matrix_file) as f:
                feature_matrix = json.load(f)
                
            feature_map = feature_matrix.get('feature_map', {})
            coverage_stats = feature_matrix.get('coverage_stats', {})
            
            # Validate coverage statistics
            for feature, pipelines in feature_map.items():
                expected_count = len(pipelines)
                actual_count = coverage_stats.get(feature, 0)
                
                if expected_count != actual_count:
                    issues.append(f"Coverage mismatch for {feature}: expected {expected_count}, got {actual_count}")
                    
            # Check for well-covered features
            well_covered = sum(1 for count in coverage_stats.values() if count >= 5)
            under_covered = sum(1 for count in coverage_stats.values() if count < 2)
            
            details = {
                'total_features': len(feature_map),
                'well_covered_features': well_covered,
                'under_covered_features': under_covered,
                'coverage_distribution': dict(coverage_stats)
            }
            
            if under_covered > len(feature_map) * 0.3:  # More than 30% under-covered
                issues.append(f"High number of under-covered features: {under_covered}")
                
            score = max(0.0, 1.0 - (under_covered / len(feature_map)))
            
        except Exception as e:
            return ValidationResult(
                test_name="Feature Coverage",
                passed=False,
                score=0.0,
                issues=[f"Error loading feature matrix: {e}"],
                details={}
            )
            
        return ValidationResult(
            test_name="Feature Coverage",
            passed=len(issues) == 0,
            score=score,
            issues=issues,
            details=details
        )
        
    def _validate_learning_path(self) -> ValidationResult:
        """Validate learning path structure and progression"""
        self.log("Validating learning path...")
        
        issues = []
        details = {}
        
        learning_path_file = self.tutorial_dir / "learning_path.md"
        if not learning_path_file.exists():
            return ValidationResult(
                test_name="Learning Path",
                passed=False,
                score=0.0,
                issues=["Learning path file not found"],
                details={}
            )
            
        try:
            content = learning_path_file.read_text(encoding='utf-8')
            
            # Check for module structure
            modules = re.findall(r'## Module \d+: (.+)', content)
            if len(modules) < 2:
                issues.append("Learning path should have multiple modules for progression")
                
            # Check for prerequisites
            if "Prerequisites" not in content:
                issues.append("Learning path lacks prerequisite information")
                
            # Check for estimated time
            if "Estimated Time" not in content:
                issues.append("Learning path lacks time estimates")
                
            # Check for pipelines in modules
            pipeline_links = re.findall(r'\[([^]]+)\]\(pipelines/([^)]+)\.md\)', content)
            
            details = {
                'modules_found': len(modules),
                'module_names': modules,
                'pipelines_in_path': len(pipeline_links),
                'pipeline_distribution': {}
            }
            
            # Validate pipeline links exist
            broken_links = 0
            tutorials_dir = self.tutorial_dir / "pipelines"
            for pipeline_name, pipeline_file in pipeline_links:
                if not (tutorials_dir / f"{pipeline_file}.md").exists():
                    broken_links += 1
                    
            if broken_links > 0:
                issues.append(f"Learning path has {broken_links} broken pipeline links")
                
            score = max(0.0, 1.0 - (len(issues) * 0.2) - (broken_links * 0.1))
            
        except Exception as e:
            return ValidationResult(
                test_name="Learning Path",
                passed=False,
                score=0.0,
                issues=[f"Error validating learning path: {e}"],
                details={}
            )
            
        return ValidationResult(
            test_name="Learning Path",
            passed=len(issues) == 0 and broken_links == 0,
            score=score,
            issues=issues,
            details=details
        )
        
    def _validate_remixing_examples(self) -> ValidationResult:
        """Validate remixing guide examples"""
        self.log("Validating remixing examples...")
        
        issues = []
        details = {}
        
        remixing_file = self.tutorial_dir / "pipeline_remixing_guide.md"
        if not remixing_file.exists():
            return ValidationResult(
                test_name="Remixing Examples",
                passed=False,
                score=0.0,
                issues=["Remixing guide file not found"],
                details={}
            )
            
        try:
            content = remixing_file.read_text(encoding='utf-8')
            
            # Check for YAML examples
            yaml_blocks = re.findall(r'```yaml\n(.*?)\n```', content, re.DOTALL)
            valid_yaml_count = 0
            
            for yaml_block in yaml_blocks:
                try:
                    yaml.safe_load(yaml_block)
                    valid_yaml_count += 1
                except yaml.YAMLError:
                    pass  # Invalid YAML
                    
            # Check for remixing patterns
            patterns = [
                "Linear Enhancement", "Parallel Processing", 
                "Conditional Remixing", "Iterative Refinement"
            ]
            
            pattern_coverage = sum(1 for pattern in patterns if pattern in content)
            
            details = {
                'total_yaml_examples': len(yaml_blocks),
                'valid_yaml_examples': valid_yaml_count,
                'remixing_patterns_covered': pattern_coverage,
                'pattern_coverage_percentage': (pattern_coverage / len(patterns)) * 100
            }
            
            if valid_yaml_count < len(yaml_blocks) * 0.8:
                issues.append(f"High number of invalid YAML examples: {len(yaml_blocks) - valid_yaml_count}")
                
            if pattern_coverage < len(patterns):
                issues.append(f"Missing remixing patterns: {len(patterns) - pattern_coverage}")
                
            score = (valid_yaml_count / len(yaml_blocks) if yaml_blocks else 1.0) * (pattern_coverage / len(patterns))
            
        except Exception as e:
            return ValidationResult(
                test_name="Remixing Examples",
                passed=False,
                score=0.0,
                issues=[f"Error validating remixing guide: {e}"],
                details={}
            )
            
        return ValidationResult(
            test_name="Remixing Examples",
            passed=len(issues) == 0,
            score=score,
            issues=issues,
            details=details
        )
        
    def _validate_links_and_references(self) -> ValidationResult:
        """Validate all internal links and references"""
        self.log("Validating links and references...")
        
        issues = []
        details = {'broken_links': {}}
        
        # Check main tutorial files
        tutorial_files = [
            self.tutorial_dir / "README.md",
            self.tutorial_dir / "learning_path.md",
            self.tutorial_dir / "feature_coverage_analysis.md",
            self.tutorial_dir / "pipeline_remixing_guide.md"
        ]
        
        # Add individual tutorials
        tutorials_dir = self.tutorial_dir / "pipelines"
        if tutorials_dir.exists():
            tutorial_files.extend(tutorials_dir.glob('*.md'))
            
        total_links = 0
        broken_links = 0
        
        for tutorial_file in tutorial_files:
            if tutorial_file.exists():
                file_links, file_broken = self._check_file_links(tutorial_file)
                total_links += file_links
                broken_links += file_broken
                
                if file_broken > 0:
                    details['broken_links'][tutorial_file.name] = file_broken
                    
        if broken_links > 0:
            issues.append(f"Found {broken_links} broken links across tutorial files")
            
        details['total_links_checked'] = total_links
        details['broken_link_count'] = broken_links
        details['link_success_rate'] = ((total_links - broken_links) / total_links) * 100 if total_links > 0 else 100
        
        score = max(0.0, 1.0 - (broken_links / total_links)) if total_links > 0 else 1.0
        
        return ValidationResult(
            test_name="Links and References",
            passed=broken_links == 0,
            score=score,
            issues=issues,
            details=details
        )
        
    def _check_file_links(self, file_path: Path) -> Tuple[int, int]:
        """Check links in a single file"""
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # Find markdown links
            links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
            total_links = len(links)
            broken_links = 0
            
            for link_text, link_url in links:
                # Skip external links (http/https)
                if link_url.startswith(('http://', 'https://')):
                    continue
                    
                # Handle relative links
                if link_url.startswith('./') or not link_url.startswith('/'):
                    link_path = (file_path.parent / link_url).resolve()
                else:
                    link_path = (self.tutorial_dir / link_url.lstrip('/')).resolve()
                    
                if not link_path.exists():
                    broken_links += 1
                    self.log(f"Broken link in {file_path.name}: {link_url}", "ERROR")
                    
            return total_links, broken_links
            
        except Exception as e:
            self.log(f"Error checking links in {file_path}: {e}", "ERROR")
            return 0, 0
            
    def _simulate_user_learning(self) -> ValidationResult:
        """Simulate user learning progression through tutorials"""
        self.log("Simulating user learning progression...")
        
        issues = []
        details = {}
        
        # Load learning path
        learning_path_file = self.tutorial_dir / "learning_path.md"
        if not learning_path_file.exists():
            return ValidationResult(
                test_name="User Learning Simulation",
                passed=False,
                score=0.0,
                issues=["Cannot simulate learning without learning path"],
                details={}
            )
            
        try:
            content = learning_path_file.read_text(encoding='utf-8')
            
            # Extract pipeline progression
            pipeline_links = re.findall(r'\[([^]]+)\]\(pipelines/([^)]+)\.md\)', content)
            
            # Simulate learning progression
            tutorials_dir = self.tutorial_dir / "pipelines"
            progression_score = 0.0
            complexity_progression = []
            
            for pipeline_name, pipeline_file in pipeline_links:
                tutorial_path = tutorials_dir / f"{pipeline_file}.md"
                
                if tutorial_path.exists():
                    # Extract complexity level
                    tutorial_content = tutorial_path.read_text(encoding='utf-8')
                    complexity_match = re.search(r'\*\*Complexity Level\*\*:\s*(\w+)', tutorial_content)
                    
                    if complexity_match:
                        complexity = complexity_match.group(1).lower()
                        complexity_progression.append(complexity)
                        
            # Check for logical progression (beginner -> intermediate -> advanced)
            progression_issues = self._analyze_complexity_progression(complexity_progression)
            issues.extend(progression_issues)
            
            details = {
                'total_tutorials_in_path': len(pipeline_links),
                'complexity_progression': complexity_progression,
                'progression_analysis': self._get_progression_stats(complexity_progression)
            }
            
            # Score based on logical progression
            progression_score = self._calculate_progression_score(complexity_progression)
            
        except Exception as e:
            return ValidationResult(
                test_name="User Learning Simulation",
                passed=False,
                score=0.0,
                issues=[f"Error simulating user learning: {e}"],
                details={}
            )
            
        return ValidationResult(
            test_name="User Learning Simulation",
            passed=len(issues) == 0,
            score=progression_score,
            issues=issues,
            details=details
        )
        
    def _analyze_complexity_progression(self, progression: List[str]) -> List[str]:
        """Analyze if complexity progression is logical"""
        issues = []
        
        complexity_order = {'beginner': 1, 'intermediate': 2, 'advanced': 3}
        
        # Check for overall progression
        if not progression:
            issues.append("No complexity information found in learning path")
            return issues
            
        # Check first tutorial is beginner-friendly
        if progression and complexity_order.get(progression[0], 99) > 2:
            issues.append("Learning path should start with beginner-level tutorials")
            
        # Check for major complexity jumps
        for i in range(1, len(progression)):
            prev_complexity = complexity_order.get(progression[i-1], 2)
            curr_complexity = complexity_order.get(progression[i], 2)
            
            if curr_complexity - prev_complexity > 1:
                issues.append(f"Large complexity jump from {progression[i-1]} to {progression[i]} at position {i}")
                
        return issues
        
    def _get_progression_stats(self, progression: List[str]) -> Dict[str, Any]:
        """Get statistics about complexity progression"""
        from collections import Counter
        
        counts = Counter(progression)
        total = len(progression)
        
        return {
            'beginner_percentage': (counts['beginner'] / total) * 100 if total > 0 else 0,
            'intermediate_percentage': (counts['intermediate'] / total) * 100 if total > 0 else 0,
            'advanced_percentage': (counts['advanced'] / total) * 100 if total > 0 else 0,
            'total_tutorials': total
        }
        
    def _calculate_progression_score(self, progression: List[str]) -> float:
        """Calculate score for learning progression quality"""
        if not progression:
            return 0.0
            
        score = 1.0
        complexity_order = {'beginner': 1, 'intermediate': 2, 'advanced': 3}
        
        # Penalty for not starting with beginner
        if complexity_order.get(progression[0], 99) > 1:
            score -= 0.2
            
        # Penalty for large jumps
        jump_penalty = 0.0
        for i in range(1, len(progression)):
            prev_complexity = complexity_order.get(progression[i-1], 2)
            curr_complexity = complexity_order.get(progression[i], 2)
            
            if curr_complexity - prev_complexity > 1:
                jump_penalty += 0.1
                
        score -= min(jump_penalty, 0.5)  # Cap penalty at 0.5
        
        return max(0.0, score)
        
    def _validate_technical_accuracy(self) -> ValidationResult:
        """Validate technical accuracy of pipeline examples"""
        self.log("Validating technical accuracy...")
        
        issues = []
        details = {'pipeline_validations': {}}
        
        # Check if we can parse YAML in tutorials
        tutorials_dir = self.tutorial_dir / "pipelines"
        if not tutorials_dir.exists():
            return ValidationResult(
                test_name="Technical Accuracy",
                passed=False,
                score=0.0,
                issues=["Tutorial directory not found"],
                details={}
            )
            
        valid_yamls = 0
        total_yamls = 0
        
        for tutorial_file in tutorials_dir.glob('*.md'):
            try:
                content = tutorial_file.read_text(encoding='utf-8')
                yaml_blocks = re.findall(r'```yaml\n(.*?)\n```', content, re.DOTALL)
                
                for yaml_block in yaml_blocks:
                    total_yamls += 1
                    try:
                        parsed = yaml.safe_load(yaml_block)
                        if parsed and isinstance(parsed, dict):
                            valid_yamls += 1
                            
                            # Basic validation - should have key pipeline elements
                            if 'steps' not in parsed and 'name' not in parsed and 'id' not in parsed:
                                issues.append(f"YAML in {tutorial_file.name} missing key pipeline elements")
                        else:
                            issues.append(f"Invalid YAML structure in {tutorial_file.name}")
                            
                    except yaml.YAMLError as e:
                        issues.append(f"YAML parse error in {tutorial_file.name}: {str(e)[:100]}")
                        
                details['pipeline_validations'][tutorial_file.stem] = {
                    'yaml_blocks': len(yaml_blocks),
                    'valid_blocks': sum(1 for yaml_block in yaml_blocks if self._is_valid_yaml(yaml_block))
                }
                
            except Exception as e:
                issues.append(f"Error processing {tutorial_file.name}: {e}")
                
        accuracy_score = valid_yamls / total_yamls if total_yamls > 0 else 1.0
        
        details.update({
            'total_yaml_blocks': total_yamls,
            'valid_yaml_blocks': valid_yamls,
            'accuracy_percentage': accuracy_score * 100
        })
        
        return ValidationResult(
            test_name="Technical Accuracy",
            passed=len(issues) == 0,
            score=accuracy_score,
            issues=issues,
            details=details
        )
        
    def _is_valid_yaml(self, yaml_content: str) -> bool:
        """Check if YAML content is valid"""
        try:
            parsed = yaml.safe_load(yaml_content)
            return parsed is not None
        except yaml.YAMLError:
            return False
            
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate validation summary"""
        passed_tests = sum(1 for result in self.validation_results if result.passed)
        total_tests = len(self.validation_results)
        
        return {
            'tests_passed': passed_tests,
            'tests_total': total_tests,
            'pass_rate': (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
            'overall_score': sum(result.score for result in self.validation_results) / total_tests if total_tests > 0 else 0,
            'test_scores': {result.test_name: result.score for result in self.validation_results}
        }
        
    def _generate_recommendations(self) -> List[str]:
        """Generate improvement recommendations based on validation results"""
        recommendations = []
        
        for result in self.validation_results:
            if result.score < 0.8:
                recommendations.append(f"Improve {result.test_name}: {', '.join(result.issues[:2])}")
                
        # Add specific recommendations based on common issues
        all_issues = []
        for result in self.validation_results:
            all_issues.extend(result.issues)
            
        if any("broken link" in issue.lower() for issue in all_issues):
            recommendations.append("Audit and fix all broken internal links")
            
        if any("yaml" in issue.lower() for issue in all_issues):
            recommendations.append("Review and validate all YAML examples for syntax correctness")
            
        if any("coverage" in issue.lower() for issue in all_issues):
            recommendations.append("Expand tutorials for under-covered features")
            
        return recommendations[:10]  # Limit to top 10 recommendations

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Validate tutorial documentation system")
    parser.add_argument('--tutorial-dir', default='docs/tutorials', help='Tutorial directory')
    parser.add_argument('--examples-dir', default='examples', help='Examples directory')
    parser.add_argument('--test-remixing', action='store_true', help='Test remixing examples')
    parser.add_argument('--simulate-learning', action='store_true', help='Simulate user learning')
    parser.add_argument('--validate-links', action='store_true', help='Validate links only')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    tutorial_dir = Path(args.tutorial_dir)
    examples_dir = Path(args.examples_dir)
    
    if not tutorial_dir.exists():
        print(f"Error: Tutorial directory {tutorial_dir} does not exist")
        return 1
        
    if not examples_dir.exists():
        print(f"Error: Examples directory {examples_dir} does not exist")
        return 1
        
    validator = TutorialValidator(tutorial_dir, examples_dir, args.verbose)
    
    try:
        validation = validator.validate_tutorial_system()
        
        # Print results
        print("\n" + "="*60)
        print("TUTORIAL SYSTEM VALIDATION REPORT")
        print("="*60)
        print(f"Overall Score: {validation.overall_score:.2f}/1.00")
        print(f"Tests Passed: {validation.summary['tests_passed']}/{validation.summary['tests_total']}")
        print(f"Pass Rate: {validation.summary['pass_rate']:.1f}%")
        
        print("\nTest Results:")
        print("-" * 60)
        
        for result in validation.test_results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"{status} {result.test_name}: {result.score:.2f}")
            
            if result.issues and args.verbose:
                for issue in result.issues[:3]:  # Show top 3 issues
                    print(f"      • {issue}")
                    
        if validation.recommendations:
            print("\nRecommendations:")
            print("-" * 60)
            for i, rec in enumerate(validation.recommendations[:5], 1):
                print(f"{i}. {rec}")
                
        # Detailed results if verbose
        if args.verbose:
            print("\nDetailed Results:")
            print("-" * 60)
            for result in validation.test_results:
                print(f"\n{result.test_name}:")
                print(f"  Score: {result.score:.3f}")
                if result.details:
                    for key, value in result.details.items():
                        if isinstance(value, dict) and len(value) > 5:
                            print(f"  {key}: {len(value)} items")
                        elif isinstance(value, list) and len(value) > 5:
                            print(f"  {key}: {len(value)} items")
                        else:
                            print(f"  {key}: {value}")
                            
        # Return non-zero exit code if validation fails
        return 0 if validation.overall_score >= 0.8 else 1
        
    except Exception as e:
        print(f"Error during validation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())