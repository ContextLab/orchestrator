"""
File organization validation for pipeline outputs.

This module provides comprehensive validation of file structure, naming
conventions, and organizational standards for pipeline outputs to ensure
consistent and professional organization.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core.quality_assessment import (
    IssueCategory, IssueSeverity, OrganizationReview, QualityIssue
)

logger = logging.getLogger(__name__)


class NamingConventionValidator:
    """Validates file naming conventions for consistency and descriptiveness."""
    
    # Professional naming patterns
    PROFESSIONAL_PATTERNS = {
        'snake_case': re.compile(r'^[a-z0-9_]+$'),
        'kebab_case': re.compile(r'^[a-z0-9\-]+$'),
        'camelCase': re.compile(r'^[a-z][a-zA-Z0-9]*$'),
        'PascalCase': re.compile(r'^[A-Z][a-zA-Z0-9]*$'),
        'descriptive': re.compile(r'^[a-zA-Z][a-zA-Z0-9_\-\s]*[a-zA-Z0-9]$')
    }
    
    # Generic/poor naming patterns to flag
    GENERIC_NAMES = {
        'output', 'result', 'file', 'data', 'content', 'document', 'report',
        'image', 'picture', 'chart', 'graph', 'plot', 'generated', 'processed'
    }
    
    # Temporary/debug naming patterns
    TEMPORARY_PATTERNS = [
        re.compile(r'temp', re.IGNORECASE),
        re.compile(r'test', re.IGNORECASE),
        re.compile(r'debug', re.IGNORECASE),
        re.compile(r'sample', re.IGNORECASE),
        re.compile(r'^(new|old)_', re.IGNORECASE),
        re.compile(r'_copy\d*$', re.IGNORECASE),
        re.compile(r'backup', re.IGNORECASE)
    ]
    
    def __init__(self):
        """Initialize naming convention validator."""
        pass
    
    def validate_filename(self, file_path: str, context: Dict[str, Any] = None) -> List[QualityIssue]:
        """Validate a single filename for naming convention compliance."""
        issues = []
        path = Path(file_path)
        stem = path.stem
        
        # Check for empty or very short names
        if len(stem) <= 1:
            issues.append(QualityIssue(
                category=IssueCategory.FILE_ORGANIZATION,
                severity=IssueSeverity.MAJOR,
                description=f"Filename too short or empty: '{stem}'",
                file_path=file_path,
                suggestion="Use descriptive filename that indicates content purpose"
            ))
            return issues
        
        # Check for generic naming
        issues.extend(self._check_generic_naming(stem, file_path))
        
        # Check for temporary/debug naming
        issues.extend(self._check_temporary_naming(stem, file_path))
        
        # Check for professional naming patterns
        issues.extend(self._check_naming_patterns(stem, file_path))
        
        # Check for timestamp appropriateness
        issues.extend(self._check_timestamp_usage(stem, file_path, context))
        
        # Check for special characters and spaces
        issues.extend(self._check_special_characters(stem, file_path))
        
        return issues
    
    def _check_generic_naming(self, stem: str, file_path: str) -> List[QualityIssue]:
        """Check for generic, non-descriptive naming."""
        issues = []
        
        stem_lower = stem.lower()
        
        # Check if name is purely generic
        if stem_lower in self.GENERIC_NAMES:
            issues.append(QualityIssue(
                category=IssueCategory.FILE_ORGANIZATION,
                severity=IssueSeverity.MAJOR,
                description=f"Generic filename: '{stem}' - not descriptive of content",
                file_path=file_path,
                suggestion="Use descriptive filename that indicates specific content or purpose"
            ))
        
        # Check for generic prefixes/suffixes
        generic_starts = any(stem_lower.startswith(generic) for generic in self.GENERIC_NAMES)
        if generic_starts and len(stem_lower.split('_')) == 1:  # Only flag if no additional description
            issues.append(QualityIssue(
                category=IssueCategory.FILE_ORGANIZATION,
                severity=IssueSeverity.MINOR,
                description=f"Filename starts with generic term: '{stem}'",
                file_path=file_path,
                suggestion="Consider more specific naming that describes the actual content"
            ))
        
        return issues
    
    def _check_temporary_naming(self, stem: str, file_path: str) -> List[QualityIssue]:
        """Check for temporary or debug naming patterns."""
        issues = []
        
        for pattern in self.TEMPORARY_PATTERNS:
            if pattern.search(stem):
                issues.append(QualityIssue(
                    category=IssueCategory.FILE_ORGANIZATION,
                    severity=IssueSeverity.MAJOR,
                    description=f"Filename contains temporary/debug pattern: '{stem}'",
                    file_path=file_path,
                    suggestion="Use production-appropriate filename without temporary indicators"
                ))
                break  # Only report once
        
        return issues
    
    def _check_naming_patterns(self, stem: str, file_path: str) -> List[QualityIssue]:
        """Check for consistent professional naming patterns."""
        issues = []
        
        # Check if follows any professional pattern
        follows_pattern = any(pattern.match(stem) for pattern in self.PROFESSIONAL_PATTERNS.values())
        
        if not follows_pattern:
            # Check for common issues
            has_spaces = ' ' in stem
            has_special_chars = re.search(r'[^a-zA-Z0-9_\-\s]', stem)
            starts_with_number = stem[0].isdigit()
            
            if has_spaces and len(stem.split()) > 4:
                issues.append(QualityIssue(
                    category=IssueCategory.FILE_ORGANIZATION,
                    severity=IssueSeverity.MINOR,
                    description=f"Very long filename with spaces: '{stem}'",
                    file_path=file_path,
                    suggestion="Consider shorter, more concise filename with underscores or hyphens"
                ))
            
            if has_special_chars:
                issues.append(QualityIssue(
                    category=IssueCategory.FILE_ORGANIZATION,
                    severity=IssueSeverity.MINOR,
                    description=f"Filename contains special characters: '{stem}'",
                    file_path=file_path,
                    suggestion="Use only letters, numbers, underscores, and hyphens in filenames"
                ))
            
            if starts_with_number and not re.match(r'^\d{4}', stem):  # Allow years
                issues.append(QualityIssue(
                    category=IssueCategory.FILE_ORGANIZATION,
                    severity=IssueSeverity.MINOR,
                    description=f"Filename starts with number: '{stem}'",
                    file_path=file_path,
                    suggestion="Consider starting filename with descriptive text rather than numbers"
                ))
        
        return issues
    
    def _check_timestamp_usage(self, stem: str, file_path: str, context: Dict[str, Any] = None) -> List[QualityIssue]:
        """Check for appropriate timestamp usage in filenames."""
        issues = []
        
        # Look for timestamp patterns
        timestamp_patterns = [
            r'\d{4}-\d{2}-\d{2}',           # YYYY-MM-DD
            r'\d{4}\d{2}\d{2}',             # YYYYMMDD
            r'\d{8}',                        # 8 digits (could be date)
            r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',  # ISO datetime
            r'\d{4}-\d{2}-\d{2}t\d{14}',    # Custom timestamp format
        ]
        
        has_timestamp = any(re.search(pattern, stem) for pattern in timestamp_patterns)
        
        if context:
            pipeline_type = context.get('pipeline_type', '')
            is_versioned_output = context.get('is_versioned', False)
            
            # For certain pipeline types, timestamps might be inappropriate
            if has_timestamp and 'report' in pipeline_type.lower():
                issues.append(QualityIssue(
                    category=IssueCategory.FILE_ORGANIZATION,
                    severity=IssueSeverity.MINOR,
                    description=f"Timestamp in report filename may indicate poor versioning: '{stem}'",
                    file_path=file_path,
                    suggestion="Consider using semantic versioning or clear content-based naming"
                ))
        
        return issues
    
    def _check_special_characters(self, stem: str, file_path: str) -> List[QualityIssue]:
        """Check for problematic special characters."""
        issues = []
        
        # Characters that can cause issues in different systems
        problematic_chars = ['<', '>', ':', '"', '|', '?', '*', '\\', '/']
        found_problematic = [char for char in problematic_chars if char in stem]
        
        if found_problematic:
            issues.append(QualityIssue(
                category=IssueCategory.FILE_ORGANIZATION,
                severity=IssueSeverity.MAJOR,
                description=f"Filename contains problematic characters: {found_problematic}",
                file_path=file_path,
                suggestion="Remove characters that may cause issues on different file systems"
            ))
        
        return issues
    
    def validate_directory_naming_consistency(self, directory_path: str) -> List[QualityIssue]:
        """Validate naming consistency across files in a directory."""
        issues = []
        
        try:
            path = Path(directory_path)
            if not path.exists() or not path.is_dir():
                return issues
            
            # Get all files (excluding hidden files)
            files = [f for f in path.iterdir() if f.is_file() and not f.name.startswith('.')]
            
            if len(files) < 2:
                return issues  # Need at least 2 files to check consistency
            
            # Analyze naming patterns
            stems = [f.stem for f in files]
            
            # Check separator consistency
            issues.extend(self._check_separator_consistency(stems, directory_path))
            
            # Check case consistency
            issues.extend(self._check_case_consistency(stems, directory_path))
            
            # Check naming length consistency
            issues.extend(self._check_length_consistency(stems, directory_path))
            
        except Exception as e:
            logger.error(f"Error validating directory naming consistency: {e}")
        
        return issues
    
    def _check_separator_consistency(self, stems: List[str], directory_path: str) -> List[QualityIssue]:
        """Check for consistent use of separators."""
        issues = []
        
        has_underscores = sum(1 for stem in stems if '_' in stem)
        has_hyphens = sum(1 for stem in stems if '-' in stem)
        has_spaces = sum(1 for stem in stems if ' ' in stem)
        
        total_files = len(stems)
        inconsistency_threshold = 0.3  # 30% different is inconsistent
        
        separator_types = sum(1 for count in [has_underscores, has_hyphens, has_spaces] 
                            if count > total_files * inconsistency_threshold)
        
        if separator_types > 1:
            issues.append(QualityIssue(
                category=IssueCategory.FILE_ORGANIZATION,
                severity=IssueSeverity.MINOR,
                description="Inconsistent filename separators across directory (mix of spaces, hyphens, underscores)",
                file_path=directory_path,
                suggestion="Standardize on one separator type (recommend underscores for most cases)"
            ))
        
        return issues
    
    def _check_case_consistency(self, stems: List[str], directory_path: str) -> List[QualityIssue]:
        """Check for consistent case usage."""
        issues = []
        
        all_lower = sum(1 for stem in stems if stem.islower())
        all_upper = sum(1 for stem in stems if stem.isupper())
        mixed_case = sum(1 for stem in stems if not stem.islower() and not stem.isupper())
        
        total_files = len(stems)
        
        # Check if there's a mix without clear pattern
        if all_lower > 0 and (all_upper > 0 or mixed_case > total_files * 0.5):
            issues.append(QualityIssue(
                category=IssueCategory.FILE_ORGANIZATION,
                severity=IssueSeverity.MINOR,
                description="Inconsistent case usage across filenames",
                file_path=directory_path,
                suggestion="Use consistent case convention (recommend lowercase with underscores)"
            ))
        
        return issues
    
    def _check_length_consistency(self, stems: List[str], directory_path: str) -> List[QualityIssue]:
        """Check for reasonable length consistency."""
        issues = []
        
        lengths = [len(stem) for stem in stems]
        avg_length = sum(lengths) / len(lengths)
        
        # Flag if there are very short and very long names mixed
        very_short = sum(1 for length in lengths if length < 5)
        very_long = sum(1 for length in lengths if length > avg_length * 2)
        
        if very_short > 0 and very_long > 0:
            issues.append(QualityIssue(
                category=IssueCategory.FILE_ORGANIZATION,
                severity=IssueSeverity.MINOR,
                description="Large variation in filename lengths may indicate inconsistent naming approach",
                file_path=directory_path,
                suggestion="Use more consistent naming approach with similar levels of descriptiveness"
            ))
        
        return issues


class DirectoryStructureValidator:
    """Validates directory structure and organization patterns."""
    
    # Expected directory patterns for different pipeline types
    PIPELINE_DIRECTORY_PATTERNS = {
        'data_processing': ['data', 'reports', 'output'],
        'image_generation': ['images', 'variations', 'styles'],
        'analysis': ['charts', 'reports', 'data'],
        'research': ['summaries', 'sources', 'analysis'],
        'creative': ['variations', 'styles', 'originals']
    }
    
    def __init__(self):
        """Initialize directory structure validator."""
        pass
    
    def validate_pipeline_structure(self, pipeline_path: str, pipeline_type: str = None) -> List[QualityIssue]:
        """Validate the overall structure of a pipeline output directory."""
        issues = []
        
        try:
            path = Path(pipeline_path)
            if not path.exists():
                issues.append(QualityIssue(
                    category=IssueCategory.FILE_ORGANIZATION,
                    severity=IssueSeverity.CRITICAL,
                    description=f"Pipeline output directory does not exist: {pipeline_path}",
                    file_path=pipeline_path,
                    suggestion="Ensure pipeline execution completed and created output directory"
                ))
                return issues
            
            # Get directory contents
            subdirs = [d for d in path.iterdir() if d.is_dir() and not d.name.startswith('.')]
            files = [f for f in path.iterdir() if f.is_file() and not f.name.startswith('.')]
            
            # Check for empty directories
            if not subdirs and not files:
                issues.append(QualityIssue(
                    category=IssueCategory.FILE_ORGANIZATION,
                    severity=IssueSeverity.CRITICAL,
                    description="Pipeline output directory is empty",
                    file_path=pipeline_path,
                    suggestion="Verify pipeline execution completed successfully"
                ))
                return issues
            
            # Validate directory organization
            issues.extend(self._validate_directory_organization(path, subdirs, files))
            
            # Validate against pipeline type patterns if known
            if pipeline_type:
                issues.extend(self._validate_pipeline_type_structure(path, pipeline_type, subdirs, files))
            
            # Check for README/documentation
            issues.extend(self._check_documentation_presence(path, files))
            
        except Exception as e:
            issues.append(QualityIssue(
                category=IssueCategory.FILE_ORGANIZATION,
                severity=IssueSeverity.MAJOR,
                description=f"Error validating pipeline structure: {str(e)}",
                file_path=pipeline_path,
                suggestion="Check file permissions and directory access"
            ))
        
        return issues
    
    def _validate_directory_organization(self, path: Path, subdirs: List[Path], files: List[Path]) -> List[QualityIssue]:
        """Validate general directory organization principles."""
        issues = []
        
        # Check for too many root-level files
        if len(files) > 15:
            issues.append(QualityIssue(
                category=IssueCategory.FILE_ORGANIZATION,
                severity=IssueSeverity.MINOR,
                description=f"Many files ({len(files)}) in root directory - consider organizing in subdirectories",
                file_path=str(path),
                suggestion="Group related files into logical subdirectories"
            ))
        
        # Check for very deep nesting
        max_depth = self._calculate_max_depth(path)
        if max_depth > 4:
            issues.append(QualityIssue(
                category=IssueCategory.FILE_ORGANIZATION,
                severity=IssueSeverity.MINOR,
                description=f"Very deep directory nesting (depth {max_depth})",
                file_path=str(path),
                suggestion="Consider flattening directory structure for better accessibility"
            ))
        
        # Check for single-file subdirectories
        for subdir in subdirs:
            subdir_files = list(subdir.iterdir())
            if len(subdir_files) == 1 and subdir_files[0].is_file():
                issues.append(QualityIssue(
                    category=IssueCategory.FILE_ORGANIZATION,
                    severity=IssueSeverity.MINOR,
                    description=f"Subdirectory contains only one file: {subdir.name}",
                    file_path=str(subdir),
                    suggestion="Consider moving single files to parent directory"
                ))
        
        return issues
    
    def _validate_pipeline_type_structure(
        self, 
        path: Path, 
        pipeline_type: str, 
        subdirs: List[Path], 
        files: List[Path]
    ) -> List[QualityIssue]:
        """Validate structure against pipeline type expectations."""
        issues = []
        
        # Get expected patterns for this pipeline type
        expected_dirs = None
        for pattern_type, dirs in self.PIPELINE_DIRECTORY_PATTERNS.items():
            if pattern_type in pipeline_type.lower():
                expected_dirs = dirs
                break
        
        if not expected_dirs:
            return issues  # No specific expectations
        
        # Check if any expected directories exist
        existing_dir_names = [d.name.lower() for d in subdirs]
        expected_found = any(expected.lower() in existing_dir_names for expected in expected_dirs)
        
        # Only suggest if structure seems ad-hoc
        if len(subdirs) > 1 and not expected_found:
            issues.append(QualityIssue(
                category=IssueCategory.FILE_ORGANIZATION,
                severity=IssueSeverity.MINOR,
                description=f"Directory structure doesn't follow common patterns for {pipeline_type} pipelines",
                file_path=str(path),
                suggestion=f"Consider organizing with directories like: {', '.join(expected_dirs)}"
            ))
        
        return issues
    
    def _check_documentation_presence(self, path: Path, files: List[Path]) -> List[QualityIssue]:
        """Check for appropriate documentation files."""
        issues = []
        
        # Look for README files
        readme_files = [f for f in files if f.name.lower().startswith('readme')]
        
        # Count significant output files
        significant_files = [f for f in files if f.suffix.lower() in ['.csv', '.json', '.md', '.html', '.png', '.jpg']]
        
        # If there are many output files but no README, suggest one
        if len(significant_files) > 5 and not readme_files:
            issues.append(QualityIssue(
                category=IssueCategory.FILE_ORGANIZATION,
                severity=IssueSeverity.MINOR,
                description=f"Directory has {len(significant_files)} output files but no README documentation",
                file_path=str(path),
                suggestion="Consider adding README file to explain the outputs and their purpose"
            ))
        
        # Check for validation summary (good practice)
        validation_files = [f for f in files if 'validation' in f.name.lower()]
        if not validation_files and len(significant_files) > 3:
            issues.append(QualityIssue(
                category=IssueCategory.FILE_ORGANIZATION,
                severity=IssueSeverity.MINOR,
                description="No validation summary found - consider adding quality metrics",
                file_path=str(path),
                suggestion="Add validation_summary.json to track output quality and completeness"
            ))
        
        return issues
    
    def _calculate_max_depth(self, path: Path) -> int:
        """Calculate maximum directory depth."""
        max_depth = 0
        
        for item in path.rglob('*'):
            if item.is_dir():
                depth = len(item.relative_to(path).parts)
                max_depth = max(max_depth, depth)
        
        return max_depth


class FileLocationValidator:
    """Validates that files are in appropriate locations."""
    
    # Expected file types for different directories
    DIRECTORY_FILE_EXPECTATIONS = {
        'charts': ['.png', '.jpg', '.svg', '.pdf'],
        'images': ['.png', '.jpg', '.jpeg', '.gif', '.webp'],
        'data': ['.csv', '.json', '.xlsx', '.parquet'],
        'reports': ['.md', '.html', '.pdf', '.docx'],
        'logs': ['.txt', '.log'],
        'config': ['.json', '.yaml', '.yml', '.toml'],
        'archive': ['.*']  # Archive can contain anything
    }
    
    def __init__(self):
        """Initialize file location validator."""
        pass
    
    def validate_file_locations(self, pipeline_path: str) -> List[QualityIssue]:
        """Validate that files are in appropriate locations."""
        issues = []
        
        try:
            path = Path(pipeline_path)
            if not path.exists():
                return issues
            
            # Check files in subdirectories
            for subdir in path.iterdir():
                if subdir.is_dir() and not subdir.name.startswith('.'):
                    issues.extend(self._validate_subdirectory_contents(subdir))
            
            # Check root level files for appropriate placement
            root_files = [f for f in path.iterdir() if f.is_file() and not f.name.startswith('.')]
            issues.extend(self._validate_root_level_files(path, root_files))
            
        except Exception as e:
            logger.error(f"Error validating file locations: {e}")
        
        return issues
    
    def _validate_subdirectory_contents(self, subdir: Path) -> List[QualityIssue]:
        """Validate contents of a specific subdirectory."""
        issues = []
        
        subdir_name = subdir.name.lower()
        expected_extensions = None
        
        # Find matching expectations
        for dir_pattern, extensions in self.DIRECTORY_FILE_EXPECTATIONS.items():
            if dir_pattern in subdir_name:
                expected_extensions = extensions
                break
        
        if expected_extensions and '.*' not in expected_extensions:
            # Check files in this directory
            for file in subdir.iterdir():
                if file.is_file() and not file.name.startswith('.'):
                    if file.suffix.lower() not in expected_extensions:
                        issues.append(QualityIssue(
                            category=IssueCategory.FILE_ORGANIZATION,
                            severity=IssueSeverity.MINOR,
                            description=f"File type {file.suffix} unexpected in {subdir_name} directory",
                            file_path=str(file),
                            suggestion=f"Consider moving to appropriate directory or verify file type is correct"
                        ))
        
        return issues
    
    def _validate_root_level_files(self, path: Path, root_files: List[Path]) -> List[QualityIssue]:
        """Validate root level file placement."""
        issues = []
        
        # Count different file types at root level
        image_files = [f for f in root_files if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif']]
        data_files = [f for f in root_files if f.suffix.lower() in ['.csv', '.json', '.xlsx']]
        
        # If there are many images at root level, suggest organization
        if len(image_files) > 5:
            issues.append(QualityIssue(
                category=IssueCategory.FILE_ORGANIZATION,
                severity=IssueSeverity.MINOR,
                description=f"Many image files ({len(image_files)}) in root directory",
                file_path=str(path),
                suggestion="Consider organizing images in 'images' or 'charts' subdirectory"
            ))
        
        # If there are many data files at root level, suggest organization
        if len(data_files) > 5:
            issues.append(QualityIssue(
                category=IssueCategory.FILE_ORGANIZATION,
                severity=IssueSeverity.MINOR,
                description=f"Many data files ({len(data_files)}) in root directory",
                file_path=str(path),
                suggestion="Consider organizing data files in 'data' subdirectory"
            ))
        
        return issues


class OrganizationQualityValidator:
    """Main validator that orchestrates all organization quality checks."""
    
    def __init__(self):
        """Initialize the organization quality validator."""
        self.naming_validator = NamingConventionValidator()
        self.structure_validator = DirectoryStructureValidator()
        self.location_validator = FileLocationValidator()
    
    def validate_pipeline_organization(
        self, 
        pipeline_path: str, 
        pipeline_name: str = None,
        pipeline_type: str = None
    ) -> OrganizationReview:
        """Comprehensive organization validation for a pipeline output directory."""
        
        all_issues = []
        correct_location = True
        appropriate_naming = True
        expected_files_present = True
        
        try:
            path = Path(pipeline_path)
            
            # Validate overall directory structure
            structure_issues = self.structure_validator.validate_pipeline_structure(
                pipeline_path, pipeline_type
            )
            all_issues.extend(structure_issues)
            
            # Validate file locations
            location_issues = self.location_validator.validate_file_locations(pipeline_path)
            all_issues.extend(location_issues)
            
            if path.exists() and path.is_dir():
                # Validate naming conventions for all files
                for file_path in path.rglob('*'):
                    if file_path.is_file() and not file_path.name.startswith('.'):
                        context = {
                            'pipeline_type': pipeline_type,
                            'pipeline_name': pipeline_name
                        }
                        naming_issues = self.naming_validator.validate_filename(
                            str(file_path), context
                        )
                        all_issues.extend(naming_issues)
                
                # Validate directory-level naming consistency
                for dir_path in path.rglob('*'):
                    if dir_path.is_dir() and not dir_path.name.startswith('.'):
                        consistency_issues = self.naming_validator.validate_directory_naming_consistency(
                            str(dir_path)
                        )
                        all_issues.extend(consistency_issues)
            
            # Determine overall status flags
            critical_location_issues = [i for i in all_issues 
                                     if i.severity == IssueSeverity.CRITICAL and 
                                        'location' in i.description.lower()]
            if critical_location_issues:
                correct_location = False
            
            naming_issues = [i for i in all_issues 
                           if i.category == IssueCategory.FILE_ORGANIZATION and
                              ('naming' in i.description.lower() or 'filename' in i.description.lower())]
            if any(issue.severity in [IssueSeverity.CRITICAL, IssueSeverity.MAJOR] 
                   for issue in naming_issues):
                appropriate_naming = False
            
            if not path.exists() or (path.exists() and not list(path.iterdir())):
                expected_files_present = False
            
        except Exception as e:
            logger.error(f"Error during organization validation: {e}")
            all_issues.append(QualityIssue(
                category=IssueCategory.FILE_ORGANIZATION,
                severity=IssueSeverity.MAJOR,
                description=f"Organization validation failed: {str(e)}",
                file_path=pipeline_path,
                suggestion="Check pipeline output directory and file permissions"
            ))
        
        return OrganizationReview(
            issues=all_issues,
            correct_location=correct_location,
            appropriate_naming=appropriate_naming,
            expected_files_present=expected_files_present
        )
    
    def suggest_improvements(self, review: OrganizationReview, pipeline_path: str) -> List[str]:
        """Generate specific improvement suggestions based on organization review."""
        suggestions = []
        
        if not review.expected_files_present:
            suggestions.append("Ensure pipeline execution completes and generates expected output files")
        
        if not review.correct_location:
            suggestions.append("Verify files are placed in correct directories according to organizational standards")
        
        if not review.appropriate_naming:
            suggestions.append("Standardize filename conventions to use descriptive, professional naming")
        
        # Add specific suggestions based on issue patterns
        issue_categories = {}
        for issue in review.issues:
            category = issue.category.value
            if category not in issue_categories:
                issue_categories[category] = 0
            issue_categories[category] += 1
        
        if issue_categories.get('file_organization', 0) > 5:
            suggestions.append("Consider restructuring directory organization to group related files logically")
        
        # Count severe issues
        severe_issues = [i for i in review.issues if i.severity in [IssueSeverity.CRITICAL, IssueSeverity.MAJOR]]
        if len(severe_issues) > 3:
            suggestions.append(f"Address {len(severe_issues)} high-priority organization issues before production use")
        
        return suggestions