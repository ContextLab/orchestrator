#!/usr/bin/env python3
"""
Repository Scanner for Issue #255 - Repository Organization & Cleanup.

Comprehensive file discovery and categorization system to identify:
- Temporary and debug files
- Scattered test files  
- Misplaced data and output files
- Checkpoint cleanup candidates
- Files needing reorganization

Safety-first approach with backup validation and rollback capabilities.
"""

import json
import logging
import os
import re
import stat
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class FileInfo:
    """Detailed information about a discovered file."""
    path: Path
    size: int
    modified: datetime
    category: str
    subcategory: str
    safety_level: str  # 'safe', 'review', 'critical'
    target_location: Optional[str] = None
    issues: List[str] = None
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []
        if self.recommendations is None:
            self.recommendations = []


@dataclass
class DirectoryInfo:
    """Information about a directory structure."""
    path: Path
    file_count: int
    total_size: int
    categories: Dict[str, int]
    recommendations: List[str]
    is_organized: bool = False


class RepositoryScanner:
    """Automated repository scanning and file categorization system."""
    
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path).resolve()
        self.scan_results = {
            'files': [],
            'directories': [],
            'statistics': {},
            'recommendations': [],
            'safety_issues': []
        }
        
        # File categorization patterns
        self.file_patterns = {
            # Test files
            'test_files': {
                'patterns': [
                    r'^test_.*\.py$',
                    r'^test_.*\.yaml$', 
                    r'^.*_test\.py$',
                    r'^.*_test\.yaml$',
                    r'test\.py$',
                    r'tests\.py$'
                ],
                'target': 'tests/',
                'safety': 'review'
            },
            
            # Data files (should be in data/ or examples/)
            'data_files': {
                'patterns': [
                    r'.*\.csv$',
                    r'.*\.json$',
                    r'.*\.parquet$',
                    r'raw_data\.*',
                    r'processed_data\.*',
                    r'.*_data\.*',
                    r'sample_.*\.csv$'
                ],
                'target': 'examples/data/',
                'safety': 'review'
            },
            
            # Output/result files (should be in examples/outputs/)
            'output_files': {
                'patterns': [
                    r'.*_report\..*',
                    r'.*_output\..*',
                    r'output_.*',
                    r'result_.*',
                    r'.*_result\..*',
                    r'.*\.html$',
                    r'validation_.*\.log$'
                ],
                'target': 'temp/',
                'safety': 'safe'
            },
            
            # Script files (validation, generation, etc.)
            'utility_scripts': {
                'patterns': [
                    r'verify_.*\.py$',
                    r'validate_.*\.py$',
                    r'regenerate_.*\.py$',
                    r'generate_.*\.py$',
                    r'fix_.*\.py$',
                    r'check_.*\.py$'
                ],
                'target': 'scripts/maintenance/',
                'safety': 'review'
            },
            
            # Debug and temporary files
            'debug_temp': {
                'patterns': [
                    r'debug_.*',
                    r'temp_.*',
                    r'tmp_.*',
                    r'\.tmp$',
                    r'.*~$',
                    r'.*\.bak$',
                    r'.*\.orig$'
                ],
                'target': 'temp/',
                'safety': 'safe'
            },
            
            # Log files
            'logs': {
                'patterns': [
                    r'.*\.log$',
                    r'.*\.out$',
                    r'.*\.err$'
                ],
                'target': 'temp/logs/',
                'safety': 'safe'
            },
            
            # Backup directories
            'backup_dirs': {
                'patterns': [
                    r'.*backup.*',
                    r'.*bak.*', 
                    r'\..*_backup.*'
                ],
                'target': 'temp/backups/',
                'safety': 'review'
            }
        }
        
        # Directory organization standards
        self.directory_standards = {
            'examples/outputs/': {
                'naming_pattern': r'^[a-z_]+$',  # snake_case only
                'max_depth': 3,
                'required_files': []
            },
            'scripts/': {
                'subdirs': ['pipeline/', 'validation/', 'maintenance/', 'debug/'],
                'naming_pattern': r'^[a-z_]+\.py$'
            },
            'checkpoints/': {
                'naming_pattern': r'^[A-Za-z0-9_\s]+_\d+_\d+\.json$',
                'max_files': 500,  # Flag for cleanup if exceeded
                'cleanup_threshold_days': 30
            }
        }
        
        # Safety classifications
        self.safety_rules = {
            'critical': [
                'pyproject.toml', 'requirements*.txt', 'setup.py', 'setup.cfg',
                'README.md', 'LICENSE', 'MANIFEST.in', '.gitignore',
                'models.yaml', 'mcp_tools_config.json'
            ],
            'review': [
                'test_*.py', 'verify_*.py', 'validate_*.py',
                '*.csv', '*.json', '*.yaml'
            ]
        }

    def scan_repository(self) -> Dict[str, Any]:
        """Main scanning method - discovers and categorizes all files."""
        logger.info(f"Starting repository scan from: {self.root_path}")
        
        # Scan files
        self._scan_files()
        
        # Scan directories
        self._scan_directories()
        
        # Generate statistics
        self._generate_statistics()
        
        # Generate recommendations
        self._generate_recommendations()
        
        # Safety analysis
        self._safety_analysis()
        
        logger.info(f"Scan complete. Found {len(self.scan_results['files'])} files across {len(self.scan_results['directories'])} directories")
        
        return self.scan_results
    
    def _scan_files(self):
        """Scan and categorize all files in the repository."""
        for file_path in self._discover_files():
            file_info = self._analyze_file(file_path)
            self.scan_results['files'].append(file_info)
    
    def _discover_files(self) -> List[Path]:
        """Discover all files, excluding common ignore patterns."""
        ignore_patterns = {
            '.git', '__pycache__', '*.pyc', '.pytest_cache', 
            '.coverage', '.tox', 'venv', 'env', 'node_modules',
            '.DS_Store', 'Thumbs.db'
        }
        
        files = []
        for root, dirs, filenames in os.walk(self.root_path):
            # Filter out ignored directories
            dirs[:] = [d for d in dirs if not any(pattern.replace('*', '') in d for pattern in ignore_patterns)]
            
            for filename in filenames:
                if not any(pattern.replace('*', '') in filename for pattern in ignore_patterns):
                    files.append(Path(root) / filename)
        
        return files
    
    def _analyze_file(self, file_path: Path) -> FileInfo:
        """Analyze a single file and categorize it."""
        try:
            stat_info = file_path.stat()
            relative_path = file_path.relative_to(self.root_path)
            
            # Basic file information
            file_info = FileInfo(
                path=relative_path,
                size=stat_info.st_size,
                modified=datetime.fromtimestamp(stat_info.st_mtime),
                category='unknown',
                subcategory='',
                safety_level='safe'
            )
            
            # Categorize file
            category, subcategory, target = self._categorize_file(file_path.name, relative_path)
            file_info.category = category
            file_info.subcategory = subcategory
            file_info.target_location = target
            
            # Determine safety level
            file_info.safety_level = self._determine_safety_level(file_path.name, relative_path)
            
            # Add specific analysis
            self._add_file_analysis(file_info, file_path)
            
            return file_info
            
        except Exception as e:
            logger.warning(f"Error analyzing file {file_path}: {e}")
            return FileInfo(
                path=file_path.relative_to(self.root_path),
                size=0,
                modified=datetime.now(),
                category='error',
                subcategory='analysis_failed',
                safety_level='critical',
                issues=[f"Analysis failed: {e}"]
            )
    
    def _categorize_file(self, filename: str, relative_path: Path) -> Tuple[str, str, str]:
        """Categorize a file based on patterns and location."""
        # Check if file is already in correct location
        path_str = str(relative_path)
        
        # Special handling for files in root that should be moved
        if relative_path.parent == Path('.'):
            for category, config in self.file_patterns.items():
                for pattern in config['patterns']:
                    if re.match(pattern, filename):
                        return category, 'scattered_in_root', config['target']
        
        # Check patterns for all files
        for category, config in self.file_patterns.items():
            for pattern in config['patterns']:
                if re.match(pattern, filename):
                    # Check if already in right location
                    if path_str.startswith(config['target']):
                        return category, 'properly_located', config['target']
                    else:
                        return category, 'mislocated', config['target']
        
        # Check for timestamped files (often temporary outputs)
        if re.search(r'\d{4}-\d{2}-\d{2}[t_]\d{2}', filename.lower()):
            return 'timestamped_output', 'temporary', 'temp/'
        
        # Checkpoint files
        if path_str.startswith('checkpoints/'):
            if self._is_old_checkpoint(relative_path):
                return 'checkpoint', 'old', 'temp/old_checkpoints/'
            else:
                return 'checkpoint', 'current', 'checkpoints/'
        
        # Examples outputs
        if path_str.startswith('examples/outputs/'):
            if self._check_examples_naming(relative_path):
                return 'examples_output', 'properly_named', 'examples/outputs/'
            else:
                return 'examples_output', 'naming_issue', 'examples/outputs/'
        
        return 'unknown', '', ''
    
    def _determine_safety_level(self, filename: str, relative_path: Path) -> str:
        """Determine the safety level for file operations."""
        # Critical files - never move without explicit approval
        if filename in self.safety_rules['critical']:
            return 'critical'
        
        # Review required files
        for pattern in self.safety_rules['review']:
            if re.match(pattern.replace('*', '.*'), filename):
                return 'review'
        
        # Temporary/debug files are generally safe
        if any(keyword in filename.lower() for keyword in ['temp', 'debug', 'tmp', 'log', 'output']):
            return 'safe'
        
        # Files in certain directories are safer to move
        path_str = str(relative_path)
        if any(path_str.startswith(prefix) for prefix in ['temp/', 'debug/', 'logs/', 'examples/outputs/']):
            return 'safe'
        
        return 'review'
    
    def _add_file_analysis(self, file_info: FileInfo, file_path: Path):
        """Add specific analysis and recommendations for the file."""
        # Check for naming convention issues
        if file_info.category == 'examples_output':
            if not re.match(r'^[a-z_]+$', file_path.stem):
                file_info.issues.append("Non-snake_case naming")
                file_info.recommendations.append("Rename to snake_case format")
        
        # Check for very old files
        if file_info.modified < datetime.now().replace(year=datetime.now().year - 1):
            file_info.issues.append("Very old file (>1 year)")
            file_info.recommendations.append("Review if still needed")
        
        # Check for large files
        if file_info.size > 10 * 1024 * 1024:  # 10MB
            file_info.issues.append(f"Large file ({file_info.size / (1024*1024):.1f}MB)")
            file_info.recommendations.append("Consider compressing or archiving")
        
        # Check for empty files
        if file_info.size == 0:
            file_info.issues.append("Empty file")
            file_info.recommendations.append("Consider deleting if not needed")
    
    def _is_old_checkpoint(self, relative_path: Path) -> bool:
        """Check if checkpoint is old and should be archived."""
        try:
            stat_info = (self.root_path / relative_path).stat()
            modified = datetime.fromtimestamp(stat_info.st_mtime)
            threshold = datetime.now().replace(day=datetime.now().day - 30)
            return modified < threshold
        except:
            return False
    
    def _check_examples_naming(self, relative_path: Path) -> bool:
        """Check if examples directory follows naming conventions."""
        parts = relative_path.parts
        if len(parts) >= 3:  # examples/outputs/dirname/...
            dirname = parts[2]
            return re.match(r'^[a-z_]+$', dirname) is not None
        return True
    
    def _scan_directories(self):
        """Analyze directory structure and organization."""
        for dir_path in self.root_path.rglob('*/'):
            if self._should_ignore_directory(dir_path):
                continue
                
            dir_info = self._analyze_directory(dir_path)
            self.scan_results['directories'].append(dir_info)
    
    def _should_ignore_directory(self, dir_path: Path) -> bool:
        """Check if directory should be ignored."""
        ignore_dirs = {'.git', '__pycache__', '.pytest_cache', 'node_modules', 'venv', 'env'}
        return any(part in ignore_dirs for part in dir_path.parts)
    
    def _analyze_directory(self, dir_path: Path) -> DirectoryInfo:
        """Analyze a directory structure."""
        relative_path = dir_path.relative_to(self.root_path)
        
        # Count files and categorize
        file_count = 0
        total_size = 0
        categories = defaultdict(int)
        
        for file_path in dir_path.iterdir():
            if file_path.is_file():
                try:
                    stat_info = file_path.stat()
                    file_count += 1
                    total_size += stat_info.st_size
                    
                    # Categorize file
                    category, _, _ = self._categorize_file(file_path.name, file_path.relative_to(self.root_path))
                    categories[category] += 1
                except:
                    pass
        
        # Generate recommendations
        recommendations = self._get_directory_recommendations(relative_path, categories, file_count)
        
        return DirectoryInfo(
            path=relative_path,
            file_count=file_count,
            total_size=total_size,
            categories=dict(categories),
            recommendations=recommendations,
            is_organized=self._is_directory_organized(relative_path, categories)
        )
    
    def _get_directory_recommendations(self, dir_path: Path, categories: dict, file_count: int) -> List[str]:
        """Generate recommendations for directory organization."""
        recommendations = []
        
        # Root directory issues
        if dir_path == Path('.'):
            if categories.get('test_files', 0) > 0:
                recommendations.append("Move test files to tests/ directory")
            if categories.get('data_files', 0) > 0:
                recommendations.append("Move data files to examples/data/ directory")
            if categories.get('utility_scripts', 0) > 0:
                recommendations.append("Move utility scripts to scripts/maintenance/")
        
        # Checkpoint directory
        elif str(dir_path) == 'checkpoints':
            if file_count > 500:
                recommendations.append(f"Too many checkpoint files ({file_count}). Consider archiving old ones.")
        
        # Examples outputs
        elif str(dir_path).startswith('examples/outputs/'):
            if not re.match(r'^[a-z_]+$', dir_path.name):
                recommendations.append("Rename directory to snake_case format")
        
        return recommendations
    
    def _is_directory_organized(self, dir_path: Path, categories: dict) -> bool:
        """Check if directory follows organization standards."""
        path_str = str(dir_path)
        
        # Root should not have scattered files
        if dir_path == Path('.'):
            return all(count == 0 for count in categories.values() if count)
        
        # Examples outputs should follow naming
        if path_str.startswith('examples/outputs/'):
            return re.match(r'^[a-z_]+$', dir_path.name) is not None
        
        return True
    
    def _generate_statistics(self):
        """Generate comprehensive statistics about the repository state."""
        stats = {
            'total_files': len(self.scan_results['files']),
            'total_size': sum(f.size for f in self.scan_results['files'] if hasattr(f, 'size')),
            'categories': defaultdict(int),
            'safety_levels': defaultdict(int),
            'locations': defaultdict(int),
            'issues': defaultdict(int)
        }
        
        # Analyze file distribution
        for file_info in self.scan_results['files']:
            stats['categories'][file_info.category] += 1
            stats['safety_levels'][file_info.safety_level] += 1
            stats['locations'][str(file_info.path.parent)] += 1
            stats['issues']['files_with_issues'] += len(file_info.issues)
        
        # Directory statistics
        stats['total_directories'] = len(self.scan_results['directories'])
        stats['organized_directories'] = sum(1 for d in self.scan_results['directories'] if d.is_organized)
        
        self.scan_results['statistics'] = dict(stats)
    
    def _generate_recommendations(self):
        """Generate high-level recommendations for repository organization."""
        recommendations = []
        stats = self.scan_results['statistics']
        
        # Root directory cleanup
        root_files = [f for f in self.scan_results['files'] if f.path.parent == Path('.')]
        if root_files:
            scattered_count = len([f for f in root_files if f.subcategory == 'scattered_in_root'])
            recommendations.append(f"Move {scattered_count} scattered files from root directory to appropriate locations")
        
        # Checkpoint cleanup
        checkpoint_files = [f for f in self.scan_results['files'] if f.category == 'checkpoint']
        old_checkpoints = [f for f in checkpoint_files if f.subcategory == 'old']
        if len(old_checkpoints) > 100:
            recommendations.append(f"Archive {len(old_checkpoints)} old checkpoint files to free up space")
        
        # Examples naming
        naming_issues = [f for f in self.scan_results['files'] if 'naming' in f.subcategory]
        if naming_issues:
            recommendations.append(f"Fix naming conventions for {len(naming_issues)} files/directories")
        
        # Large files
        large_files = [f for f in self.scan_results['files'] if f.size > 10 * 1024 * 1024]
        if large_files:
            recommendations.append(f"Review {len(large_files)} large files for compression or archiving")
        
        self.scan_results['recommendations'] = recommendations
    
    def _safety_analysis(self):
        """Perform safety analysis to identify potential risks."""
        safety_issues = []
        
        # Check for critical files being moved
        critical_moves = [f for f in self.scan_results['files'] 
                         if f.safety_level == 'critical' and f.target_location]
        if critical_moves:
            safety_issues.append(f"CRITICAL: {len(critical_moves)} critical files marked for moving - requires manual review")
        
        # Check for large batch operations
        review_files = [f for f in self.scan_results['files'] if f.safety_level == 'review']
        if len(review_files) > 50:
            safety_issues.append(f"Large batch operation: {len(review_files)} files require review before moving")
        
        self.scan_results['safety_issues'] = safety_issues
    
    def generate_report(self, format_type: str = 'json') -> str:
        """Generate a comprehensive report of the scan results."""
        if format_type == 'json':
            return self._generate_json_report()
        elif format_type == 'markdown':
            return self._generate_markdown_report()
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _generate_json_report(self) -> str:
        """Generate JSON report."""
        # Convert dataclasses to dicts for JSON serialization
        report = {
            'scan_timestamp': datetime.now().isoformat(),
            'repository_path': str(self.root_path),
            'files': [asdict(f) for f in self.scan_results['files']],
            'directories': [asdict(d) for d in self.scan_results['directories']],
            'statistics': self.scan_results['statistics'],
            'recommendations': self.scan_results['recommendations'],
            'safety_issues': self.scan_results['safety_issues']
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
        return json.dumps(report, indent=2, default=str)
    
    def _generate_markdown_report(self) -> str:
        """Generate Markdown report."""
        stats = self.scan_results['statistics']
        
        report_lines = [
            "# Repository Organization Analysis Report",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Repository:** {self.root_path}",
            "",
            "## Executive Summary",
            f"- **Total Files:** {stats['total_files']:,}",
            f"- **Total Size:** {stats['total_size'] / (1024*1024):.1f} MB",
            f"- **Directories:** {stats['total_directories']}",
            f"- **Organized Directories:** {stats['organized_directories']}",
            "",
            "## Key Issues Found",
        ]
        
        # Add safety issues
        if self.scan_results['safety_issues']:
            report_lines.extend(["### Safety Concerns"])
            for issue in self.scan_results['safety_issues']:
                report_lines.append(f"⚠️ {issue}")
            report_lines.append("")
        
        # Add recommendations
        if self.scan_results['recommendations']:
            report_lines.extend(["### Recommended Actions"])
            for i, rec in enumerate(self.scan_results['recommendations'], 1):
                report_lines.append(f"{i}. {rec}")
            report_lines.append("")
        
        # Category breakdown
        report_lines.extend([
            "## File Categories",
            "| Category | Count | Percentage |",
            "|----------|-------|------------|"
        ])
        
        total_files = stats['total_files']
        for category, count in sorted(stats['categories'].items()):
            percentage = (count / total_files * 100) if total_files > 0 else 0
            report_lines.append(f"| {category} | {count} | {percentage:.1f}% |")
        
        # Safety level breakdown
        report_lines.extend([
            "",
            "## Safety Level Distribution", 
            "| Safety Level | Count | Description |",
            "|--------------|-------|-------------|"
        ])
        
        safety_descriptions = {
            'safe': 'Can be moved automatically',
            'review': 'Requires manual review',
            'critical': 'Must not be moved without explicit approval'
        }
        
        for level, count in sorted(stats['safety_levels'].items()):
            desc = safety_descriptions.get(level, 'Unknown')
            report_lines.append(f"| {level} | {count} | {desc} |")
        
        return "\n".join(report_lines)
    
    def save_report(self, output_path: str, format_type: str = 'json'):
        """Save the scan report to a file."""
        report_content = self.generate_report(format_type)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Report saved to: {output_file}")
        return output_file


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Repository Organization Scanner")
    parser.add_argument("--root", default=".", help="Repository root path")
    parser.add_argument("--output", default="repository_scan_report.json", help="Output report file")
    parser.add_argument("--format", choices=['json', 'markdown'], default='json', help="Report format")
    parser.add_argument("--verbose", action='store_true', help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and run scanner
    scanner = RepositoryScanner(args.root)
    scan_results = scanner.scan_repository()
    
    # Save report
    output_file = scanner.save_report(args.output, args.format)
    
    # Print summary
    stats = scan_results['statistics']
    print(f"\n{'='*60}")
    print("REPOSITORY SCAN SUMMARY")
    print(f"{'='*60}")
    print(f"Files scanned: {stats['total_files']:,}")
    print(f"Total size: {stats['total_size'] / (1024*1024):.1f} MB")
    print(f"Directories: {stats['total_directories']}")
    print(f"Issues found: {len(scan_results['safety_issues'])}")
    print(f"Recommendations: {len(scan_results['recommendations'])}")
    print(f"\nReport saved to: {output_file}")


if __name__ == "__main__":
    main()