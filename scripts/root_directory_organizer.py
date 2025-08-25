#!/usr/bin/env python3
"""
Root Directory Organizer for Issue #255 - Focus on scattered files in root.

Focused organizer that handles only the critical scattered files in the root directory.
This is a safer, more targeted approach than the full repository reorganization.

Targets specifically:
- Test files (test_*.py, test_*.yaml) -> tests/
- Utility scripts (verify_*.py, regenerate_*.py) -> scripts/maintenance/  
- Data files (*.csv, *.html, *.parquet) -> examples/data/
- Log files (*.log) -> temp/logs/
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RootDirectoryOrganizer:
    """Focused organizer for scattered files in root directory."""
    
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path).resolve()
        self.temp_dir = self.root_path / "temp"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Define specific file patterns and their targets
        self.file_mapping = {
            # Test files
            'test_files': {
                'patterns': [
                    'test_*.py',
                    'test_*.yaml', 
                    '*_test.py',
                    '*_test.yaml'
                ],
                'target_dir': 'tests',
                'description': 'Test files'
            },
            
            # Utility/maintenance scripts
            'utility_scripts': {
                'patterns': [
                    'verify_*.py',
                    'validate_*.py',
                    'regenerate_*.py',
                    'generate_*.py',
                    'fix_*.py',
                    'check_*.py'
                ],
                'target_dir': 'scripts/maintenance',
                'description': 'Utility scripts'
            },
            
            # Data files
            'data_files': {
                'patterns': [
                    '*.csv',
                    '*.parquet',
                    'processed_data.*',
                    'raw_data.*'
                ],
                'target_dir': 'examples/data',
                'description': 'Data files'
            },
            
            # Output/report files
            'output_files': {
                'patterns': [
                    '*_report.html',
                    '*_report.md',
                    'data_processing_report.*'
                ],
                'target_dir': 'temp',
                'description': 'Output/report files'
            },
            
            # Log files
            'log_files': {
                'patterns': [
                    '*.log'
                ],
                'target_dir': 'temp/logs',
                'description': 'Log files'
            }
        }
        
        # Critical files that should NEVER be moved
        self.protected_files = {
            'pyproject.toml', 'setup.py', 'setup.cfg', 'requirements.txt',
            'requirements-web.txt', 'models.yaml', 'mcp_tools_config.json',
            'README.md', 'CHANGELOG.md', 'LICENSE', 'MANIFEST.in',
            'CONTRIBUTING.md', '.gitignore', '.env', 'pytest.ini',
            'coverage.xml'
        }
    
    def scan_root_directory(self) -> Dict[str, Any]:
        """Scan root directory for files that need organization."""
        logger.info("Scanning root directory for scattered files...")
        
        root_files = []
        for file_path in self.root_path.iterdir():
            if file_path.is_file() and file_path.name not in self.protected_files:
                root_files.append(file_path)
        
        # Categorize files
        categorized_files = {
            'organized': {},
            'unmatched': [],
            'protected': [],
            'summary': {}
        }
        
        for file_path in root_files:
            filename = file_path.name
            matched = False
            
            for category, config in self.file_mapping.items():
                for pattern in config['patterns']:
                    if file_path.match(pattern):
                        if category not in categorized_files['organized']:
                            categorized_files['organized'][category] = []
                        
                        categorized_files['organized'][category].append({
                            'path': str(file_path.relative_to(self.root_path)),
                            'name': filename,
                            'size': file_path.stat().st_size,
                            'target_dir': config['target_dir'],
                            'target_path': str(Path(config['target_dir']) / filename)
                        })
                        matched = True
                        break
                
                if matched:
                    break
            
            if not matched:
                categorized_files['unmatched'].append({
                    'path': str(file_path.relative_to(self.root_path)),
                    'name': filename,
                    'size': file_path.stat().st_size
                })
        
        # Add protected files info
        for file_path in self.root_path.iterdir():
            if file_path.is_file() and file_path.name in self.protected_files:
                categorized_files['protected'].append(file_path.name)
        
        # Generate summary
        total_to_organize = sum(len(files) for files in categorized_files['organized'].values())
        categorized_files['summary'] = {
            'total_root_files': len(root_files) + len(categorized_files['protected']),
            'files_to_organize': total_to_organize,
            'unmatched_files': len(categorized_files['unmatched']),
            'protected_files': len(categorized_files['protected']),
            'categories': {cat: len(files) for cat, files in categorized_files['organized'].items()}
        }
        
        logger.info(f"Found {total_to_organize} files to organize in {len(categorized_files['organized'])} categories")
        return categorized_files
    
    def create_organization_plan(self, scan_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a detailed plan for organizing root directory files."""
        if scan_results is None:
            scan_results = self.scan_root_directory()
        
        logger.info("Creating organization plan...")
        
        plan = {
            'timestamp': datetime.now().isoformat(),
            'operations': [],
            'directories_to_create': set(),
            'safety_checks': {
                'critical_files_affected': 0,
                'large_files': [],
                'potential_conflicts': []
            },
            'summary': scan_results['summary']
        }
        
        # Create operations for each categorized file
        for category, files in scan_results['organized'].items():
            config = self.file_mapping[category]
            
            for file_info in files:
                operation = {
                    'operation': 'move',
                    'source': file_info['path'],
                    'target': file_info['target_path'],
                    'category': category,
                    'description': config['description'],
                    'size': file_info['size']
                }
                
                plan['operations'].append(operation)
                plan['directories_to_create'].add(config['target_dir'])
                
                # Safety checks
                if file_info['size'] > 10 * 1024 * 1024:  # 10MB
                    plan['safety_checks']['large_files'].append({
                        'file': file_info['name'],
                        'size_mb': file_info['size'] / (1024 * 1024)
                    })
        
        plan['directories_to_create'] = sorted(list(plan['directories_to_create']))
        
        # Check for potential target conflicts
        target_files = {}
        for op in plan['operations']:
            target = op['target']
            if target in target_files:
                plan['safety_checks']['potential_conflicts'].append({
                    'target': target,
                    'sources': [target_files[target], op['source']]
                })
            else:
                target_files[target] = op['source']
        
        logger.info(f"Plan created: {len(plan['operations'])} operations, {len(plan['directories_to_create'])} directories to create")
        return plan
    
    def validate_plan(self, plan: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate the organization plan for safety."""
        issues = []
        
        # Check for critical files
        if plan['safety_checks']['critical_files_affected'] > 0:
            issues.append(f"CRITICAL: {plan['safety_checks']['critical_files_affected']} critical files affected")
        
        # Check for conflicts
        if plan['safety_checks']['potential_conflicts']:
            issues.append(f"File conflicts: {len(plan['safety_checks']['potential_conflicts'])} conflicts detected")
        
        # Check for large files
        if plan['safety_checks']['large_files']:
            issues.append(f"Large files: {len(plan['safety_checks']['large_files'])} files >10MB")
        
        # Check target directory accessibility
        for target_dir in plan['directories_to_create']:
            target_path = self.root_path / target_dir
            try:
                target_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                issues.append(f"Cannot create directory {target_dir}: {e}")
        
        is_safe = len(issues) == 0 or all('Large files' in issue or 'CRITICAL' not in issue for issue in issues)
        
        return is_safe, issues
    
    def create_backup(self, operations: List[Dict[str, Any]]) -> str:
        """Create a backup of files before moving them."""
        backup_id = f"root_org_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_dir = self.temp_dir / "safety_backups" / backup_id
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating backup: {backup_id}")
        
        backup_manifest = {
            'backup_id': backup_id,
            'timestamp': datetime.now().isoformat(),
            'operations': operations,
            'files_backed_up': []
        }
        
        for operation in operations:
            source_path = self.root_path / operation['source']
            if source_path.exists():
                backup_file_path = backup_dir / operation['source']
                shutil.copy2(source_path, backup_file_path)
                backup_manifest['files_backed_up'].append(operation['source'])
        
        # Save manifest
        manifest_path = backup_dir / "backup_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(backup_manifest, f, indent=2, default=str)
        
        logger.info(f"Backup complete: {len(backup_manifest['files_backed_up'])} files backed up")
        return backup_id
    
    def execute_plan(self, plan: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
        """Execute the organization plan."""
        operations = plan['operations']
        
        logger.info(f"{'DRY RUN: ' if dry_run else ''}Executing organization plan with {len(operations)} operations")
        
        # Create backup if not dry run
        backup_id = None
        if not dry_run:
            backup_id = self.create_backup(operations)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'dry_run': dry_run,
            'backup_id': backup_id,
            'operations_attempted': len(operations),
            'operations_completed': 0,
            'operations_failed': 0,
            'failures': [],
            'directories_created': []
        }
        
        # Create target directories
        for target_dir in plan['directories_to_create']:
            target_path = self.root_path / target_dir
            if dry_run:
                logger.info(f"DRY RUN: Would create directory: {target_path}")
                results['directories_created'].append(target_dir)
            else:
                try:
                    target_path.mkdir(parents=True, exist_ok=True)
                    results['directories_created'].append(target_dir)
                    logger.info(f"Created directory: {target_path}")
                except Exception as e:
                    logger.error(f"Failed to create directory {target_path}: {e}")
                    results['failures'].append(f"Directory creation failed: {target_dir}")
        
        # Execute file moves by category for better organization
        operations_by_category = {}
        for op in operations:
            category = op['category']
            if category not in operations_by_category:
                operations_by_category[category] = []
            operations_by_category[category].append(op)
        
        for category, category_ops in operations_by_category.items():
            logger.info(f"Processing {category}: {len(category_ops)} files")
            
            for operation in category_ops:
                source_path = self.root_path / operation['source']
                target_path = self.root_path / operation['target']
                
                if dry_run:
                    logger.info(f"DRY RUN: Would move {source_path.name} -> {target_path}")
                    results['operations_completed'] += 1
                else:
                    try:
                        if not source_path.exists():
                            raise FileNotFoundError(f"Source file not found: {source_path}")
                        
                        if target_path.exists():
                            raise FileExistsError(f"Target file already exists: {target_path}")
                        
                        # Move the file
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        source_path.rename(target_path)
                        
                        logger.info(f"Moved: {source_path.name} -> {target_path.parent.name}/{target_path.name}")
                        results['operations_completed'] += 1
                        
                    except Exception as e:
                        logger.error(f"Failed to move {source_path} -> {target_path}: {e}")
                        results['operations_failed'] += 1
                        results['failures'].append({
                            'operation': operation,
                            'error': str(e)
                        })
        
        # Determine final status
        if results['operations_failed'] == 0:
            results['status'] = 'success'
        elif results['operations_completed'] > 0:
            results['status'] = 'partial_success'
        else:
            results['status'] = 'failed'
        
        logger.info(f"Execution complete: {results['operations_completed']}/{results['operations_attempted']} operations successful")
        
        # Save execution report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.temp_dir / f"root_org_execution_{timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Execution report saved: {report_path}")
        
        return results
    
    def print_summary(self, scan_results: Dict[str, Any] = None):
        """Print a summary of root directory organization status."""
        if scan_results is None:
            scan_results = self.scan_root_directory()
        
        summary = scan_results['summary']
        
        print(f"\n{'='*60}")
        print("ROOT DIRECTORY ORGANIZATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total files in root: {summary['total_root_files']}")
        print(f"Protected files: {summary['protected_files']}")
        print(f"Files to organize: {summary['files_to_organize']}")
        print(f"Unmatched files: {summary['unmatched_files']}")
        
        if summary['categories']:
            print(f"\nFiles by category:")
            for category, count in summary['categories'].items():
                desc = self.file_mapping[category]['description']
                target = self.file_mapping[category]['target_dir']
                print(f"  {desc}: {count} files -> {target}/")
        
        if scan_results['unmatched']:
            print(f"\nUnmatched files requiring manual review:")
            for file_info in scan_results['unmatched'][:10]:  # Show first 10
                print(f"  - {file_info['name']} ({file_info['size']} bytes)")
            if len(scan_results['unmatched']) > 10:
                print(f"  ... and {len(scan_results['unmatched']) - 10} more")


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Root Directory Organization Tool")
    parser.add_argument("--root", default=".", help="Repository root path")
    parser.add_argument("--scan", action='store_true', help="Scan root directory")
    parser.add_argument("--plan", action='store_true', help="Create organization plan")
    parser.add_argument("--execute", action='store_true', help="Execute organization")
    parser.add_argument("--dry-run", action='store_true', help="Preview changes only")
    parser.add_argument("--auto-approve", action='store_true', help="Skip confirmations")
    
    args = parser.parse_args()
    
    organizer = RootDirectoryOrganizer(args.root)
    
    try:
        if args.scan or not any([args.plan, args.execute]):
            scan_results = organizer.scan_root_directory()
            organizer.print_summary(scan_results)
        
        if args.plan:
            scan_results = organizer.scan_root_directory()
            plan = organizer.create_organization_plan(scan_results)
            is_safe, issues = organizer.validate_plan(plan)
            
            print(f"\nOrganization Plan:")
            print(f"Operations: {len(plan['operations'])}")
            print(f"Directories to create: {len(plan['directories_to_create'])}")
            print(f"Safety status: {'SAFE' if is_safe else 'NEEDS REVIEW'}")
            
            if issues:
                print(f"\nSafety issues:")
                for issue in issues:
                    print(f"  ⚠️  {issue}")
            
            if not is_safe and not args.auto_approve and not args.dry_run:
                response = input("\nProceed anyway? [y/N]: ")
                if response.lower() != 'y':
                    print("Operation cancelled")
                    return
            
            if args.execute or args.dry_run:
                results = organizer.execute_plan(plan, dry_run=args.dry_run)
                
                print(f"\nExecution Results:")
                print(f"Status: {results['status']}")
                print(f"Operations: {results['operations_completed']}/{results['operations_attempted']}")
                
                if results['operations_failed'] > 0:
                    print(f"Failures: {results['operations_failed']}")
                    for failure in results['failures'][:3]:  # Show first 3 failures
                        print(f"  - {failure['operation']['source']}: {failure['error']}")
                
                if results.get('backup_id') and not args.dry_run:
                    print(f"Backup: {results['backup_id']}")
        
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        raise


if __name__ == "__main__":
    main()