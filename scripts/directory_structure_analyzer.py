#!/usr/bin/env python3
"""
Directory Structure Analyzer for Issue #255 Stream B.

Specialized tool for analyzing and standardizing directory structures,
particularly focusing on examples/outputs directory issues:
- Malformed directory names (JSON-like strings)
- Inconsistent naming conventions
- Improper file organization
"""

import json
import logging
import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class DirectoryIssue:
    """Represents a directory structure issue."""
    path: Path
    issue_type: str
    current_name: str
    suggested_name: str
    severity: str  # 'critical', 'major', 'minor'
    description: str
    fix_commands: List[str]


class DirectoryStructureAnalyzer:
    """Analyzer for directory structure standardization."""
    
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path).resolve()
        self.issues = []
        
        # Naming convention patterns
        self.naming_patterns = {
            'snake_case': r'^[a-z][a-z0-9_]*[a-z0-9]$',
            'kebab_case': r'^[a-z][a-z0-9-]*[a-z0-9]$',
            'camelCase': r'^[a-z][a-zA-Z0-9]*$',
            'PascalCase': r'^[A-Z][a-zA-Z0-9]*$',
        }
        
        # Standard directory structure for examples/outputs
        self.output_standards = {
            'naming_convention': 'snake_case',
            'max_depth': 4,
            'required_files': ['README.md'],  # Optional but recommended
            'forbidden_chars': ['{', '}', ':', '*', '?', '<', '>', '|', '"'],
            'reserved_names': ['con', 'prn', 'aux', 'nul'],
            'max_name_length': 50
        }
        
        # File naming standards
        self.file_standards = {
            'timestamped_pattern': r'.*(\d{4}-\d{2}-\d{2})[t_](\d{6}|\d{9}|\d{12}).*',
            'report_pattern': r'.*_report\.(md|json|html)$',
            'output_pattern': r'^(output|result|processed)_.*',
            'preferred_extensions': {'.md', '.json', '.csv', '.txt', '.html', '.png', '.jpg'}
        }
    
    def analyze_directory_structure(self, target_path: str = "examples/outputs") -> List[DirectoryIssue]:
        """Main analysis method."""
        logger.info(f"Analyzing directory structure: {target_path}")
        
        target_dir = self.root_path / target_path
        if not target_dir.exists():
            logger.warning(f"Target directory does not exist: {target_dir}")
            return []
        
        # Analyze directories
        self._analyze_directories(target_dir)
        
        # Analyze files
        self._analyze_files(target_dir)
        
        # Sort issues by severity
        self.issues.sort(key=lambda x: {'critical': 0, 'major': 1, 'minor': 2}[x.severity])
        
        logger.info(f"Found {len(self.issues)} directory structure issues")
        return self.issues
    
    def _analyze_directories(self, base_path: Path):
        """Analyze directory naming and structure."""
        for dir_path in base_path.rglob('*'):
            if not dir_path.is_dir():
                continue
            
            relative_path = dir_path.relative_to(base_path)
            dir_name = dir_path.name
            
            # Skip hidden directories
            if dir_name.startswith('.'):
                continue
            
            # Check for malformed names (JSON-like structures)
            if self._is_malformed_name(dir_name):
                suggested_name = self._suggest_directory_name(dir_name, dir_path)
                self.issues.append(DirectoryIssue(
                    path=relative_path,
                    issue_type='malformed_name',
                    current_name=dir_name,
                    suggested_name=suggested_name,
                    severity='critical',
                    description=f"Directory name contains invalid characters or structure: {dir_name}",
                    fix_commands=[
                        f"# Rename directory: {dir_name} -> {suggested_name}",
                        f"mv '{dir_path}' '{dir_path.parent / suggested_name}'"
                    ]
                ))
            
            # Check naming convention compliance
            elif not self._follows_naming_convention(dir_name):
                suggested_name = self._convert_to_snake_case(dir_name)
                self.issues.append(DirectoryIssue(
                    path=relative_path,
                    issue_type='naming_convention',
                    current_name=dir_name,
                    suggested_name=suggested_name,
                    severity='major',
                    description=f"Directory name doesn't follow snake_case convention: {dir_name}",
                    fix_commands=[
                        f"# Rename directory: {dir_name} -> {suggested_name}",
                        f"mv '{dir_path}' '{dir_path.parent / suggested_name}'"
                    ]
                ))
            
            # Check directory depth
            depth = len(relative_path.parts)
            if depth > self.output_standards['max_depth']:
                self.issues.append(DirectoryIssue(
                    path=relative_path,
                    issue_type='excessive_depth',
                    current_name=dir_name,
                    suggested_name='',
                    severity='minor',
                    description=f"Directory depth ({depth}) exceeds maximum ({self.output_standards['max_depth']})",
                    fix_commands=[
                        f"# Consider restructuring to reduce depth"
                    ]
                ))
    
    def _analyze_files(self, base_path: Path):
        """Analyze file naming and organization within directories."""
        for file_path in base_path.rglob('*'):
            if not file_path.is_file():
                continue
            
            relative_path = file_path.relative_to(base_path)
            file_name = file_path.name
            
            # Check for problematic file names
            if self._has_problematic_filename(file_name):
                suggested_name = self._suggest_filename(file_name)
                self.issues.append(DirectoryIssue(
                    path=relative_path,
                    issue_type='problematic_filename',
                    current_name=file_name,
                    suggested_name=suggested_name,
                    severity='major',
                    description=f"File name contains problematic characters or format: {file_name}",
                    fix_commands=[
                        f"# Rename file: {file_name} -> {suggested_name}",
                        f"mv '{file_path}' '{file_path.parent / suggested_name}'"
                    ]
                ))
            
            # Check for timestamped files that should be organized
            if self._is_timestamped_file(file_name):
                if not self._is_in_appropriate_location(file_path):
                    suggested_location = self._suggest_file_location(file_path)
                    self.issues.append(DirectoryIssue(
                        path=relative_path,
                        issue_type='misplaced_timestamped',
                        current_name=file_name,
                        suggested_name=suggested_location,
                        severity='minor',
                        description=f"Timestamped file should be organized: {file_name}",
                        fix_commands=[
                            f"# Move timestamped file to appropriate location",
                            f"mkdir -p '{file_path.parent / 'archive'}' || true",
                            f"mv '{file_path}' '{file_path.parent / 'archive' / file_name}'"
                        ]
                    ))
    
    def _is_malformed_name(self, name: str) -> bool:
        """Check if directory name is malformed (contains invalid characters)."""
        # Check for JSON-like structures
        if name.startswith('{') or name.startswith("{'"):
            return True
        
        # Check for forbidden characters
        forbidden_chars = self.output_standards['forbidden_chars']
        if any(char in name for char in forbidden_chars):
            return True
        
        # Check for excessively long names
        if len(name) > self.output_standards['max_name_length']:
            return True
        
        # Check for reserved names
        if name.lower() in self.output_standards['reserved_names']:
            return True
        
        return False
    
    def _follows_naming_convention(self, name: str) -> bool:
        """Check if name follows the preferred naming convention."""
        pattern = self.naming_patterns[self.output_standards['naming_convention']]
        return re.match(pattern, name) is not None
    
    def _suggest_directory_name(self, current_name: str, dir_path: Path) -> str:
        """Suggest a proper directory name based on current name and contents."""
        # Handle JSON-like malformed names
        if current_name.startswith('{') or current_name.startswith("{'"):
            # Try to extract meaningful parts
            if 'result' in current_name:
                # Look for result value
                result_match = re.search(r"'result':\s*'([^']+)'", current_name)
                if result_match:
                    return self._convert_to_snake_case(result_match.group(1))
            
            # Fallback: analyze directory contents
            return self._generate_name_from_contents(dir_path)
        
        # Convert problematic names to snake_case
        return self._convert_to_snake_case(current_name)
    
    def _generate_name_from_contents(self, dir_path: Path) -> str:
        """Generate directory name based on contents."""
        try:
            files = list(dir_path.iterdir())
            if files:
                # Look for README or descriptive files
                for file_path in files:
                    if file_path.name.lower() == 'readme.md':
                        # Could analyze README content for better name
                        pass
                    elif file_path.suffix == '.png':
                        # Image directory - use generic name
                        return 'generated_images'
                
                # Fallback to generic name with timestamp
                return f"output_dir_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            return "empty_directory"
        except:
            return "unknown_content"
    
    def _convert_to_snake_case(self, name: str) -> str:
        """Convert name to snake_case."""
        # Remove invalid characters
        clean_name = re.sub(r'[^a-zA-Z0-9_\s-]', '', name)
        
        # Replace spaces and hyphens with underscores
        clean_name = re.sub(r'[\s-]+', '_', clean_name)
        
        # Convert to lowercase
        clean_name = clean_name.lower()
        
        # Remove multiple underscores
        clean_name = re.sub(r'_+', '_', clean_name)
        
        # Remove leading/trailing underscores
        clean_name = clean_name.strip('_')
        
        # Ensure it starts with a letter
        if clean_name and clean_name[0].isdigit():
            clean_name = f"dir_{clean_name}"
        
        # Fallback if empty
        if not clean_name:
            clean_name = f"renamed_dir_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Truncate if too long
        if len(clean_name) > self.output_standards['max_name_length']:
            clean_name = clean_name[:self.output_standards['max_name_length']].rstrip('_')
        
        return clean_name
    
    def _has_problematic_filename(self, filename: str) -> bool:
        """Check if filename has problematic characters."""
        # Check for colons (problematic on some systems)
        if ':' in filename:
            return True
        
        # Check for other forbidden characters in filenames
        forbidden_in_files = ['<', '>', ':', '"', '|', '?', '*']
        if any(char in filename for char in forbidden_in_files):
            return True
        
        return False
    
    def _suggest_filename(self, filename: str) -> str:
        """Suggest a corrected filename."""
        # Replace colons with hyphens in timestamps
        if ':' in filename:
            suggested = filename.replace(':', '-')
            return suggested
        
        return filename
    
    def _is_timestamped_file(self, filename: str) -> bool:
        """Check if file appears to be timestamped."""
        return re.search(self.file_standards['timestamped_pattern'], filename) is not None
    
    def _is_in_appropriate_location(self, file_path: Path) -> bool:
        """Check if timestamped file is in an appropriate subdirectory."""
        # Check if it's in an 'archive', 'history', or similar subdirectory
        parent_names = [p.name.lower() for p in file_path.parents]
        appropriate_parents = {'archive', 'history', 'timestamped', 'old', 'backup'}
        return any(parent in appropriate_parents for parent in parent_names)
    
    def _suggest_file_location(self, file_path: Path) -> str:
        """Suggest appropriate location for a file."""
        return str(file_path.parent / 'archive' / file_path.name)
    
    def generate_fix_plan(self) -> Dict[str, Any]:
        """Generate a comprehensive fix plan."""
        plan = {
            'timestamp': datetime.now().isoformat(),
            'total_issues': len(self.issues),
            'issues_by_severity': defaultdict(int),
            'issues_by_type': defaultdict(int),
            'fix_commands': [],
            'safety_checks': [],
            'rollback_plan': []
        }
        
        # Categorize issues
        for issue in self.issues:
            plan['issues_by_severity'][issue.severity] += 1
            plan['issues_by_type'][issue.issue_type] += 1
            plan['fix_commands'].extend(issue.fix_commands)
        
        # Add safety checks
        plan['safety_checks'] = [
            "# Before executing any renames:",
            "# 1. Create backup with: git add -A && git commit -m 'Backup before directory restructuring'",
            "# 2. Verify all file paths are accessible",
            "# 3. Check for any running processes that might be using these directories",
            "# 4. Test renames on a small subset first"
        ]
        
        # Add rollback plan
        plan['rollback_plan'] = [
            "# If issues occur during restructuring:",
            "# 1. Stop all operations immediately",
            "# 2. Run: git reset --hard HEAD~1",
            "# 3. Check directory integrity",
            "# 4. Report issues for manual resolution"
        ]
        
        return dict(plan)
    
    def save_analysis_report(self, output_path: str):
        """Save detailed analysis report."""
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'repository_path': str(self.root_path),
            'total_issues': len(self.issues),
            'issues': [asdict(issue) for issue in self.issues],
            'standards': self.output_standards,
            'fix_plan': self.generate_fix_plan()
        }
        
        # Convert Path objects to strings for JSON serialization
        def convert_paths(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            return obj
        
        report = convert_paths(report)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Analysis report saved to: {output_file}")
        return output_file


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Directory Structure Analyzer")
    parser.add_argument("--root", default=".", help="Repository root path")
    parser.add_argument("--target", default="examples/outputs", help="Target directory to analyze")
    parser.add_argument("--output", default="temp/directory_analysis.json", help="Output report file")
    parser.add_argument("--verbose", action='store_true', help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and run analyzer
    analyzer = DirectoryStructureAnalyzer(args.root)
    issues = analyzer.analyze_directory_structure(args.target)
    
    # Save report
    output_file = analyzer.save_analysis_report(args.output)
    
    # Print summary
    print(f"\n{'='*60}")
    print("DIRECTORY STRUCTURE ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Issues found: {len(issues)}")
    
    severity_counts = defaultdict(int)
    type_counts = defaultdict(int)
    
    for issue in issues:
        severity_counts[issue.severity] += 1
        type_counts[issue.issue_type] += 1
    
    print(f"\nBy severity:")
    for severity in ['critical', 'major', 'minor']:
        count = severity_counts[severity]
        if count > 0:
            print(f"  {severity}: {count}")
    
    print(f"\nBy type:")
    for issue_type, count in sorted(type_counts.items()):
        print(f"  {issue_type}: {count}")
    
    print(f"\nDetailed report saved to: {output_file}")
    
    if issues:
        print(f"\nFirst few critical issues:")
        critical_issues = [i for i in issues if i.severity == 'critical'][:3]
        for issue in critical_issues:
            print(f"  - {issue.current_name} -> {issue.suggested_name}")
            print(f"    {issue.description}")


if __name__ == "__main__":
    main()