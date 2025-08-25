#!/usr/bin/env python3
"""
Directory Structure Standardizer for Issue #255 Stream B.

Safe execution system for standardizing directory structures based on
the analysis from directory_structure_analyzer.py.

Builds on Stream A's proven safety framework with:
- Multi-layer safety checks
- Automated backup creation
- Rollback capabilities
- Git integration
"""

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import safety validator from Stream A
import sys
sys.path.append(str(Path(__file__).parent))
from safety_validator import SafetyValidator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class StandardizationResult:
    """Result of a directory standardization operation."""
    issue_type: str
    original_path: str
    new_path: str
    success: bool
    error_message: Optional[str] = None
    backup_created: bool = False


class DirectoryStructureStandardizer:
    """Safe directory structure standardization system."""
    
    def __init__(self, root_path: str = ".", backup_dir: str = "temp/backup"):
        self.root_path = Path(root_path).resolve()
        self.backup_dir = Path(backup_dir)
        self.safety_validator = SafetyValidator()
        self.results = []
        self.backup_id = None
        
        # Create backup directory
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def load_analysis_report(self, report_path: str) -> Dict[str, Any]:
        """Load analysis report from directory_structure_analyzer."""
        report_file = Path(report_path)
        if not report_file.exists():
            raise FileNotFoundError(f"Analysis report not found: {report_path}")
        
        with open(report_file, 'r') as f:
            return json.load(f)
    
    def standardize_directory_structure(self, analysis_report_path: str, 
                                      execute: bool = False) -> List[StandardizationResult]:
        """Main standardization method."""
        logger.info("Starting directory structure standardization")
        
        # Load analysis report
        report = self.load_analysis_report(analysis_report_path)
        issues = report.get('issues', [])
        
        if not issues:
            logger.info("No directory structure issues found")
            return []
        
        logger.info(f"Processing {len(issues)} directory structure issues")
        
        # Create backup if executing
        if execute:
            self.backup_id = self._create_backup()
            logger.info(f"Created backup with ID: {self.backup_id}")
        
        # Process issues by severity (critical first)
        critical_issues = [i for i in issues if i['severity'] == 'critical']
        major_issues = [i for i in issues if i['severity'] == 'major'] 
        minor_issues = [i for i in issues if i['severity'] == 'minor']
        
        # Process in order of severity
        for issue_list, severity in [(critical_issues, 'critical'), 
                                   (major_issues, 'major'), 
                                   (minor_issues, 'minor')]:
            if issue_list:
                logger.info(f"Processing {len(issue_list)} {severity} issues")
                self._process_issues(issue_list, execute)
        
        # Generate summary
        success_count = sum(1 for r in self.results if r.success)
        logger.info(f"Standardization complete: {success_count}/{len(self.results)} operations successful")
        
        return self.results
    
    def _create_backup(self) -> str:
        """Create backup using git commit."""
        try:
            # Create backup ID
            backup_id = f"dir_struct_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Add all changes to git
            subprocess.run(['git', 'add', '-A'], cwd=self.root_path, check=True)
            
            # Create commit
            commit_message = f"Issue #255: Backup before directory structure standardization ({backup_id})"
            subprocess.run(['git', 'commit', '-m', commit_message], 
                         cwd=self.root_path, check=True)
            
            logger.info(f"Created git backup commit: {backup_id}")
            return backup_id
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create git backup: {e}")
            raise
    
    def _process_issues(self, issues: List[Dict], execute: bool = False):
        """Process a list of directory structure issues."""
        for issue in issues:
            result = self._process_single_issue(issue, execute)
            self.results.append(result)
            
            if not result.success:
                logger.error(f"Failed to process issue: {result.error_message}")
                # For critical issues, consider stopping
                if issue['severity'] == 'critical':
                    logger.error("Critical issue failed - stopping execution")
                    break
    
    def _process_single_issue(self, issue: Dict, execute: bool = False) -> StandardizationResult:
        """Process a single directory structure issue."""
        issue_type = issue['issue_type']
        current_path = Path(issue['path'])
        
        # Convert relative path to absolute
        if not current_path.is_absolute():
            full_current_path = self.root_path / "examples/outputs" / current_path
        else:
            full_current_path = current_path
        
        logger.info(f"Processing {issue_type}: {current_path}")
        
        try:
            if issue_type == 'malformed_name':
                return self._fix_malformed_directory_name(issue, execute)
            elif issue_type == 'naming_convention':
                return self._fix_naming_convention(issue, execute)
            elif issue_type == 'problematic_filename':
                return self._fix_problematic_filename(issue, execute)
            elif issue_type == 'misplaced_timestamped':
                return self._organize_timestamped_file(issue, execute)
            else:
                return StandardizationResult(
                    issue_type=issue_type,
                    original_path=str(current_path),
                    new_path="",
                    success=False,
                    error_message=f"Unknown issue type: {issue_type}"
                )
                
        except Exception as e:
            logger.error(f"Error processing issue {issue_type}: {e}")
            return StandardizationResult(
                issue_type=issue_type,
                original_path=str(current_path),
                new_path="",
                success=False,
                error_message=str(e)
            )
    
    def _fix_malformed_directory_name(self, issue: Dict, execute: bool) -> StandardizationResult:
        """Fix malformed directory names (JSON-like structures)."""
        original_name = issue['current_name']
        suggested_name = issue['suggested_name']
        
        # Reconstruct paths
        path_parts = issue['path'].split('/')
        parent_path = self.root_path / "examples/outputs" / '/'.join(path_parts[:-1])
        original_full_path = parent_path / original_name
        new_full_path = parent_path / suggested_name
        
        logger.info(f"Renaming malformed directory: {original_name} -> {suggested_name}")
        
        # Safety checks
        if not original_full_path.exists():
            return StandardizationResult(
                issue_type='malformed_name',
                original_path=str(original_full_path),
                new_path=str(new_full_path),
                success=False,
                error_message="Original directory does not exist"
            )
        
        if new_full_path.exists():
            return StandardizationResult(
                issue_type='malformed_name',
                original_path=str(original_full_path),
                new_path=str(new_full_path),
                success=False,
                error_message="Target directory already exists"
            )
        
        if execute:
            try:
                original_full_path.rename(new_full_path)
                logger.info(f"Successfully renamed: {original_full_path} -> {new_full_path}")
                return StandardizationResult(
                    issue_type='malformed_name',
                    original_path=str(original_full_path),
                    new_path=str(new_full_path),
                    success=True
                )
            except Exception as e:
                return StandardizationResult(
                    issue_type='malformed_name',
                    original_path=str(original_full_path),
                    new_path=str(new_full_path),
                    success=False,
                    error_message=f"Rename failed: {e}"
                )
        else:
            # Dry run
            return StandardizationResult(
                issue_type='malformed_name',
                original_path=str(original_full_path),
                new_path=str(new_full_path),
                success=True  # Would succeed in dry run
            )
    
    def _fix_naming_convention(self, issue: Dict, execute: bool) -> StandardizationResult:
        """Fix directory naming convention issues."""
        # Note: The analyzer suggested "test_simple_data_processingyaml" which looks wrong
        # Let's use a better name for this case
        original_name = issue['current_name']
        if original_name == "test_simple_data_processing.yaml":
            # This is actually a directory that was named like a file
            suggested_name = "test_simple_data_processing_yaml"
        else:
            suggested_name = issue['suggested_name']
        
        original_full_path = self.root_path / "examples/outputs" / original_name
        new_full_path = self.root_path / "examples/outputs" / suggested_name
        
        logger.info(f"Fixing naming convention: {original_name} -> {suggested_name}")
        
        # Safety checks
        if not original_full_path.exists():
            return StandardizationResult(
                issue_type='naming_convention',
                original_path=str(original_full_path),
                new_path=str(new_full_path),
                success=False,
                error_message="Original directory does not exist"
            )
        
        if new_full_path.exists():
            return StandardizationResult(
                issue_type='naming_convention',
                original_path=str(original_full_path),
                new_path=str(new_full_path),
                success=False,
                error_message="Target directory already exists"
            )
        
        if execute:
            try:
                original_full_path.rename(new_full_path)
                logger.info(f"Successfully renamed: {original_full_path} -> {new_full_path}")
                return StandardizationResult(
                    issue_type='naming_convention',
                    original_path=str(original_full_path),
                    new_path=str(new_full_path),
                    success=True
                )
            except Exception as e:
                return StandardizationResult(
                    issue_type='naming_convention',
                    original_path=str(original_full_path),
                    new_path=str(new_full_path),
                    success=False,
                    error_message=f"Rename failed: {e}"
                )
        else:
            return StandardizationResult(
                issue_type='naming_convention',
                original_path=str(original_full_path),
                new_path=str(new_full_path),
                success=True
            )
    
    def _fix_problematic_filename(self, issue: Dict, execute: bool) -> StandardizationResult:
        """Fix problematic filenames (e.g., containing colons)."""
        original_name = issue['current_name']
        suggested_name = issue['suggested_name']
        
        # Reconstruct full path
        path_parts = issue['path'].split('/')
        parent_path = self.root_path / "examples/outputs" / '/'.join(path_parts[:-1])
        original_full_path = parent_path / original_name
        new_full_path = parent_path / suggested_name
        
        logger.info(f"Fixing problematic filename: {original_name} -> {suggested_name}")
        
        # Safety checks
        if not original_full_path.exists():
            return StandardizationResult(
                issue_type='problematic_filename',
                original_path=str(original_full_path),
                new_path=str(new_full_path),
                success=False,
                error_message="Original file does not exist"
            )
        
        if new_full_path.exists():
            return StandardizationResult(
                issue_type='problematic_filename',
                original_path=str(original_full_path),
                new_path=str(new_full_path),
                success=False,
                error_message="Target file already exists"
            )
        
        if execute:
            try:
                original_full_path.rename(new_full_path)
                logger.info(f"Successfully renamed file: {original_full_path} -> {new_full_path}")
                return StandardizationResult(
                    issue_type='problematic_filename',
                    original_path=str(original_full_path),
                    new_path=str(new_full_path),
                    success=True
                )
            except Exception as e:
                return StandardizationResult(
                    issue_type='problematic_filename',
                    original_path=str(original_full_path),
                    new_path=str(new_full_path),
                    success=False,
                    error_message=f"Rename failed: {e}"
                )
        else:
            return StandardizationResult(
                issue_type='problematic_filename',
                original_path=str(original_full_path),
                new_path=str(new_full_path),
                success=True
            )
    
    def _organize_timestamped_file(self, issue: Dict, execute: bool) -> StandardizationResult:
        """Organize timestamped files into archive subdirectories."""
        original_name = issue['current_name']
        
        # Reconstruct paths
        path_parts = issue['path'].split('/')
        parent_path = self.root_path / "examples/outputs" / '/'.join(path_parts[:-1])
        original_full_path = parent_path / original_name
        archive_dir = parent_path / 'archive'
        new_full_path = archive_dir / original_name
        
        logger.info(f"Organizing timestamped file: {original_name} -> archive/{original_name}")
        
        # Safety checks
        if not original_full_path.exists():
            return StandardizationResult(
                issue_type='misplaced_timestamped',
                original_path=str(original_full_path),
                new_path=str(new_full_path),
                success=False,
                error_message="Original file does not exist"
            )
        
        if execute:
            try:
                # Create archive directory if it doesn't exist
                archive_dir.mkdir(exist_ok=True)
                
                # Move file to archive
                original_full_path.rename(new_full_path)
                logger.info(f"Successfully moved to archive: {original_full_path} -> {new_full_path}")
                return StandardizationResult(
                    issue_type='misplaced_timestamped',
                    original_path=str(original_full_path),
                    new_path=str(new_full_path),
                    success=True
                )
            except Exception as e:
                return StandardizationResult(
                    issue_type='misplaced_timestamped',
                    original_path=str(original_full_path),
                    new_path=str(new_full_path),
                    success=False,
                    error_message=f"Move failed: {e}"
                )
        else:
            return StandardizationResult(
                issue_type='misplaced_timestamped',
                original_path=str(original_full_path),
                new_path=str(new_full_path),
                success=True
            )
    
    def generate_execution_report(self, output_path: str):
        """Generate comprehensive execution report."""
        report = {
            'execution_timestamp': datetime.now().isoformat(),
            'backup_id': self.backup_id,
            'total_operations': len(self.results),
            'successful_operations': sum(1 for r in self.results if r.success),
            'failed_operations': sum(1 for r in self.results if not r.success),
            'operations_by_type': {},
            'results': [],
            'summary': {
                'critical_issues_fixed': 0,
                'major_issues_fixed': 0,
                'minor_issues_fixed': 0,
                'files_renamed': 0,
                'directories_renamed': 0,
                'files_moved_to_archive': 0
            }
        }
        
        # Categorize results
        for result in self.results:
            issue_type = result.issue_type
            if issue_type not in report['operations_by_type']:
                report['operations_by_type'][issue_type] = {'success': 0, 'failed': 0}
            
            if result.success:
                report['operations_by_type'][issue_type]['success'] += 1
                
                # Update summary
                if issue_type == 'malformed_name':
                    report['summary']['critical_issues_fixed'] += 1
                    report['summary']['directories_renamed'] += 1
                elif issue_type == 'naming_convention':
                    report['summary']['major_issues_fixed'] += 1
                    report['summary']['directories_renamed'] += 1
                elif issue_type == 'problematic_filename':
                    report['summary']['major_issues_fixed'] += 1
                    report['summary']['files_renamed'] += 1
                elif issue_type == 'misplaced_timestamped':
                    report['summary']['minor_issues_fixed'] += 1
                    report['summary']['files_moved_to_archive'] += 1
            else:
                report['operations_by_type'][issue_type]['failed'] += 1
            
            # Add result details
            report['results'].append({
                'issue_type': result.issue_type,
                'original_path': result.original_path,
                'new_path': result.new_path,
                'success': result.success,
                'error_message': result.error_message
            })
        
        # Save report
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Execution report saved to: {output_file}")
        return output_file


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Directory Structure Standardizer")
    parser.add_argument("--analysis-report", required=True, 
                       help="Path to analysis report from directory_structure_analyzer")
    parser.add_argument("--root", default=".", help="Repository root path")
    parser.add_argument("--execute", action='store_true', 
                       help="Execute standardization (default is dry run)")
    parser.add_argument("--report", default="temp/standardization_report.json", 
                       help="Output execution report file")
    parser.add_argument("--verbose", action='store_true', help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and run standardizer
    standardizer = DirectoryStructureStandardizer(args.root)
    
    if not args.execute:
        logger.info("Running in DRY RUN mode - no changes will be made")
        logger.info("Use --execute flag to perform actual standardization")
    
    results = standardizer.standardize_directory_structure(
        args.analysis_report, 
        execute=args.execute
    )
    
    # Generate report
    report_file = standardizer.generate_execution_report(args.report)
    
    # Print summary
    success_count = sum(1 for r in results if r.success)
    total_count = len(results)
    
    print(f"\n{'='*60}")
    print("DIRECTORY STRUCTURE STANDARDIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"Mode: {'EXECUTE' if args.execute else 'DRY RUN'}")
    print(f"Operations: {success_count}/{total_count} successful")
    
    if standardizer.backup_id and args.execute:
        print(f"Backup ID: {standardizer.backup_id}")
    
    # Show failed operations
    failed_results = [r for r in results if not r.success]
    if failed_results:
        print(f"\nFailed operations:")
        for result in failed_results:
            print(f"  - {result.issue_type}: {result.error_message}")
    
    print(f"\nDetailed report: {report_file}")
    
    if args.execute and success_count > 0:
        print(f"\n✅ Successfully standardized {success_count} directory structure issues!")
        if standardizer.backup_id:
            print(f"💾 Backup created: {standardizer.backup_id}")
            print("🔄 Rollback: git reset --hard HEAD~1 (if needed)")


if __name__ == "__main__":
    main()