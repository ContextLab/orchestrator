#!/usr/bin/env python3
"""
Repository Organizer for Issue #255 - Repository Organization & Cleanup.

Main orchestrator script that combines file discovery, safety validation, and 
automated file organization operations. Provides safe, atomic operations with 
rollback capabilities.

Usage:
    python scripts/utilities/repository_organizer.py --scan           # Analyze repository
    python scripts/utilities/repository_organizer.py --plan          # Generate organization plan  
    python scripts/utilities/repository_organizer.py --execute       # Execute organization
    python scripts/utilities/repository_organizer.py --dry-run       # Preview changes only
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add current directory to path for local imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from repository_scanner import RepositoryScanner, FileInfo
from safety_validator import SafetyValidator, SafetyCheck

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RepositoryOrganizer:
    """Main orchestrator for repository organization operations."""
    
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path).resolve()
        self.scanner = RepositoryScanner(str(self.root_path))
        self.validator = SafetyValidator(str(self.root_path))
        self.organization_plan = None
        self.scan_results = None
        
        # Create necessary directories
        self.temp_dir = self.root_path / "temp"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Organization mapping
        self.target_structure = {
            'test_files': 'tests/',
            'utility_scripts': 'scripts/maintenance/', 
            'data_files': 'examples/data/',
            'output_files': 'temp/',
            'debug_temp': 'temp/',
            'logs': 'temp/logs/',
            'backup_dirs': 'temp/backups/',
            'timestamped_output': 'temp/'
        }
    
    def analyze_repository(self, save_report: bool = True) -> Dict[str, Any]:
        """Perform comprehensive repository analysis."""
        logger.info("Starting comprehensive repository analysis...")
        
        # Run scanner
        self.scan_results = self.scanner.scan_repository()
        
        if save_report:
            # Save reports
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            json_report = self.temp_dir / f"repository_analysis_{timestamp}.json"
            md_report = self.temp_dir / f"repository_analysis_{timestamp}.md"
            
            self.scanner.save_report(str(json_report), 'json')
            self.scanner.save_report(str(md_report), 'markdown')
            
            logger.info(f"Analysis reports saved: {json_report}, {md_report}")
        
        return self.scan_results
    
    def generate_organization_plan(self) -> Dict[str, Any]:
        """Generate a detailed organization plan based on scan results."""
        if not self.scan_results:
            logger.info("Running repository scan first...")
            self.analyze_repository()
        
        logger.info("Generating organization plan...")
        
        plan = {
            'timestamp': datetime.now().isoformat(),
            'operations': [],
            'statistics': {
                'total_files': 0,
                'files_to_move': 0,
                'directories_to_create': set(),
                'safety_breakdown': {'safe': 0, 'review': 0, 'critical': 0}
            },
            'safety_summary': {
                'can_execute_automatically': False,
                'requires_user_approval': [],
                'potential_issues': []
            }
        }
        
        # Process files that need organization
        files_to_organize = [f for f in self.scan_results['files'] 
                           if f.subcategory in ['scattered_in_root', 'mislocated'] 
                           and f.target_location]
        
        plan['statistics']['total_files'] = len(self.scan_results['files'])
        plan['statistics']['files_to_move'] = len(files_to_organize)
        
        for file_info in files_to_organize:
            # Create operation
            operation = {
                'operation': 'move',
                'source': str(file_info.path),
                'target': str(Path(file_info.target_location) / file_info.path.name),
                'category': file_info.category,
                'safety_level': file_info.safety_level,
                'size': file_info.size,
                'issues': file_info.issues,
                'recommendations': file_info.recommendations
            }
            
            plan['operations'].append(operation)
            plan['statistics']['directories_to_create'].add(file_info.target_location)
            plan['statistics']['safety_breakdown'][file_info.safety_level] += 1
        
        # Convert set to list for JSON serialization
        plan['statistics']['directories_to_create'] = sorted(list(plan['statistics']['directories_to_create']))
        
        # Analyze safety requirements
        critical_operations = [op for op in plan['operations'] if op['safety_level'] == 'critical']
        review_operations = [op for op in plan['operations'] if op['safety_level'] == 'review']
        
        if critical_operations:
            plan['safety_summary']['requires_user_approval'].append(
                f"{len(critical_operations)} critical files require explicit approval"
            )
        
        if review_operations:
            plan['safety_summary']['requires_user_approval'].append(
                f"{len(review_operations)} files require review before moving"
            )
        
        # Can execute automatically only if no critical files and few review files
        plan['safety_summary']['can_execute_automatically'] = (
            len(critical_operations) == 0 and len(review_operations) <= 10
        )
        
        self.organization_plan = plan
        
        # Save plan
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plan_file = self.temp_dir / f"organization_plan_{timestamp}.json"
        with open(plan_file, 'w') as f:
            json.dump(plan, f, indent=2, default=str)
        
        logger.info(f"Organization plan generated: {len(plan['operations'])} operations planned")
        logger.info(f"Plan saved to: {plan_file}")
        
        return plan
    
    def validate_plan(self, plan: Optional[Dict[str, Any]] = None) -> tuple[bool, List[SafetyCheck]]:
        """Validate the organization plan for safety."""
        if plan is None:
            plan = self.organization_plan
        
        if not plan:
            raise ValueError("No organization plan available. Run generate_organization_plan() first.")
        
        logger.info(f"Validating organization plan with {len(plan['operations'])} operations...")
        
        # Run safety validation
        is_safe, safety_checks = self.validator.validate_operation(
            plan['operations'], 
            operation_type='move'
        )
        
        logger.info(f"Plan validation complete: {'SAFE' if is_safe else 'NOT SAFE'}")
        
        return is_safe, safety_checks
    
    def execute_plan(self, plan: Optional[Dict[str, Any]] = None, 
                    dry_run: bool = False, auto_approve: bool = False) -> Dict[str, Any]:
        """Execute the organization plan with safety measures."""
        if plan is None:
            plan = self.organization_plan
        
        if not plan:
            raise ValueError("No organization plan available. Run generate_organization_plan() first.")
        
        operations = plan['operations']
        
        logger.info(f"{'DRY RUN: ' if dry_run else ''}Executing organization plan with {len(operations)} operations")
        
        # Pre-execution validation
        is_safe, safety_checks = self.validate_plan(plan)
        
        if not is_safe and not auto_approve:
            logger.error("Plan failed safety validation. Use --auto-approve to override.")
            return {'status': 'failed', 'reason': 'safety_validation_failed', 'safety_checks': safety_checks}
        
        # Create backup if not dry run
        backup_manifest = None
        if not dry_run:
            source_files = [op['source'] for op in operations]
            backup_manifest = self.validator.create_backup(source_files)
            logger.info(f"Backup created: {backup_manifest.backup_id}")
        
        # Execute operations
        results = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'dry_run': dry_run,
            'backup_id': backup_manifest.backup_id if backup_manifest else None,
            'operations_attempted': len(operations),
            'operations_completed': 0,
            'operations_failed': 0,
            'failures': [],
            'directories_created': []
        }
        
        # Group operations by target directory for efficient processing
        operations_by_target = {}
        for op in operations:
            target_dir = str(Path(op['target']).parent)
            if target_dir not in operations_by_target:
                operations_by_target[target_dir] = []
            operations_by_target[target_dir].append(op)
        
        # Create target directories first
        for target_dir in operations_by_target.keys():
            target_path = self.root_path / target_dir
            if dry_run:
                logger.info(f"DRY RUN: Would create directory: {target_path}")
                results['directories_created'].append(str(target_dir))
            else:
                try:
                    target_path.mkdir(parents=True, exist_ok=True)
                    results['directories_created'].append(str(target_dir))
                    logger.info(f"Created directory: {target_path}")
                except Exception as e:
                    logger.error(f"Failed to create directory {target_path}: {e}")
                    results['failures'].append(f"Directory creation failed: {target_dir} - {e}")
        
        # Execute file operations
        for target_dir, ops in operations_by_target.items():
            for op in ops:
                source_path = self.root_path / op['source']
                target_path = self.root_path / op['target']
                
                if dry_run:
                    logger.info(f"DRY RUN: Would move {source_path} -> {target_path}")
                    results['operations_completed'] += 1
                else:
                    try:
                        # Move file
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        source_path.rename(target_path)
                        
                        logger.info(f"Moved: {source_path} -> {target_path}")
                        results['operations_completed'] += 1
                        
                    except Exception as e:
                        logger.error(f"Failed to move {source_path} -> {target_path}: {e}")
                        results['operations_failed'] += 1
                        results['failures'].append({
                            'operation': op,
                            'error': str(e)
                        })
        
        # Update status based on results
        if results['operations_failed'] > 0:
            results['status'] = 'partial_success'
            
        if results['operations_failed'] == results['operations_attempted']:
            results['status'] = 'failed'
        
        # Save execution report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.temp_dir / f"execution_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Execution complete: {results['operations_completed']}/{results['operations_attempted']} operations successful")
        logger.info(f"Execution report saved to: {report_file}")
        
        return results
    
    def rollback_to_backup(self, backup_id: str) -> bool:
        """Rollback to a previous backup."""
        logger.info(f"Rolling back to backup: {backup_id}")
        
        success = self.validator.restore_backup(backup_id)
        
        if success:
            logger.info("Rollback completed successfully")
        else:
            logger.error("Rollback failed")
        
        return success
    
    def print_summary(self):
        """Print a summary of the current repository state."""
        if not self.scan_results:
            logger.info("Running repository scan first...")
            self.analyze_repository()
        
        stats = self.scan_results['statistics']
        
        print(f"\n{'='*60}")
        print("REPOSITORY ORGANIZATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total files: {stats['total_files']:,}")
        print(f"Total size: {stats['total_size'] / (1024*1024):.1f} MB")
        print(f"Directories: {stats['total_directories']}")
        print(f"Organized directories: {stats['organized_directories']}")
        
        print(f"\nFiles requiring organization:")
        scattered_files = len([f for f in self.scan_results['files'] 
                              if f.subcategory == 'scattered_in_root'])
        mislocated_files = len([f for f in self.scan_results['files'] 
                               if f.subcategory == 'mislocated'])
        
        print(f"  Scattered in root: {scattered_files}")
        print(f"  Mislocated: {mislocated_files}")
        
        print(f"\nTop recommendations:")
        for i, rec in enumerate(self.scan_results.get('recommendations', [])[:5], 1):
            print(f"  {i}. {rec}")
        
        if self.scan_results.get('safety_issues'):
            print(f"\n⚠️  Safety concerns:")
            for issue in self.scan_results['safety_issues']:
                print(f"  - {issue}")


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Repository Organization System")
    parser.add_argument("--root", default=".", help="Repository root path")
    
    # Main operations
    parser.add_argument("--scan", action='store_true', help="Analyze repository structure")
    parser.add_argument("--plan", action='store_true', help="Generate organization plan")
    parser.add_argument("--execute", action='store_true', help="Execute organization plan")
    parser.add_argument("--rollback", metavar='BACKUP_ID', help="Rollback to backup")
    
    # Execution options
    parser.add_argument("--dry-run", action='store_true', help="Preview changes without executing")
    parser.add_argument("--auto-approve", action='store_true', help="Skip safety confirmations")
    parser.add_argument("--summary", action='store_true', help="Show repository summary")
    
    # Backup management
    parser.add_argument("--list-backups", action='store_true', help="List available backups")
    
    args = parser.parse_args()
    
    organizer = RepositoryOrganizer(args.root)
    
    try:
        if args.scan:
            organizer.analyze_repository()
            organizer.print_summary()
        
        elif args.plan:
            plan = organizer.generate_organization_plan()
            is_safe, checks = organizer.validate_plan(plan)
            
            print(f"\nOrganization Plan Generated:")
            print(f"Operations planned: {len(plan['operations'])}")
            print(f"Directories to create: {len(plan['statistics']['directories_to_create'])}")
            print(f"Safety status: {'SAFE' if is_safe else 'REQUIRES REVIEW'}")
            
            if not is_safe:
                print("\nSafety issues:")
                for check in checks:
                    if not check.passed:
                        print(f"  ⚠️  {check.check_name}: {check.message}")
        
        elif args.execute:
            if not organizer.organization_plan:
                organizer.generate_organization_plan()
            
            results = organizer.execute_plan(
                dry_run=args.dry_run,
                auto_approve=args.auto_approve
            )
            
            print(f"\nExecution Results:")
            print(f"Status: {results['status']}")
            print(f"Operations completed: {results['operations_completed']}/{results['operations_attempted']}")
            
            if results['operations_failed'] > 0:
                print(f"Operations failed: {results['operations_failed']}")
            
            if results.get('backup_id'):
                print(f"Backup created: {results['backup_id']}")
        
        elif args.rollback:
            success = organizer.rollback_to_backup(args.rollback)
            print(f"Rollback {'successful' if success else 'failed'}")
        
        elif args.list_backups:
            backups = organizer.validator.list_backups()
            if backups:
                print("Available backups:")
                for backup in backups:
                    print(f"  {backup['backup_id']} ({backup['timestamp']}) - {backup['file_count']} files")
            else:
                print("No backups found")
        
        elif args.summary:
            organizer.print_summary()
        
        else:
            organizer.print_summary()
            print("\nUse --scan, --plan, --execute, or --summary for specific operations")
            print("Use --help for full usage information")
    
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        raise


if __name__ == "__main__":
    main()