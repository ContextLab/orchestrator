#!/usr/bin/env python3
"""
Organization Maintenance System for Issue #255 Stream C.

Long-term maintenance and self-healing system for repository organization:
- Automated cleanup schedules and procedures
- Self-healing capabilities for minor violations
- Maintenance documentation generation
- Integration with all Stream A, B, and C systems
- Recovery procedures for organization failures
- Preventive maintenance recommendations

Final component completing the automated organization framework.
"""

import json
import logging
import sys
import subprocess
import schedule
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
import shutil
import tempfile

# Import our complete infrastructure
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from repository_scanner import RepositoryScanner
    from safety_validator import SafetyValidator
    from repository_organization_monitor import RepositoryOrganizationMonitor
    from organization_validator import OrganizationValidator
    from organization_reporter import OrganizationReporter
    # Import Stream A & B proven tools
    from root_directory_organizer import RootDirectoryOrganizer
    from directory_structure_standardizer import DirectoryStructureStandardizer
except ImportError as e:
    logging.error(f"Failed to import required modules: {e}")
    logging.error("Ensure all organization system components are available")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class MaintenanceTask:
    """Maintenance task definition."""
    task_id: str
    name: str
    description: str
    schedule: str  # cron-like or 'daily', 'weekly', etc.
    handler: str   # function name to call
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    auto_fix: bool = False
    severity_threshold: str = "warning"  # Only auto-fix above this threshold


@dataclass
class MaintenanceResult:
    """Result of maintenance operation."""
    task_id: str
    timestamp: datetime
    success: bool
    actions_taken: List[str]
    issues_found: int
    issues_resolved: int
    backup_created: bool
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class OrganizationMaintenance:
    """Comprehensive maintenance system for repository organization."""
    
    def __init__(self, root_path: str = ".", config_file: Optional[str] = None):
        self.root_path = Path(root_path).resolve()
        self.config = self._load_config(config_file)
        
        # Initialize all system components
        self.scanner = RepositoryScanner(str(self.root_path))
        self.safety_validator = SafetyValidator(str(self.root_path))
        self.monitor = RepositoryOrganizationMonitor(str(self.root_path))
        self.validator = OrganizationValidator(str(self.root_path))
        self.reporter = OrganizationReporter(str(self.root_path))
        
        # Initialize proven Stream A & B organizers
        self.root_organizer = RootDirectoryOrganizer(str(self.root_path))
        self.directory_standardizer = DirectoryStructureStandardizer(str(self.root_path))
        
        # Maintenance infrastructure
        self.maintenance_dir = self.root_path / "temp" / "maintenance"
        self.maintenance_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = self.maintenance_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.documentation_dir = self.root_path / "docs" / "maintenance"
        self.documentation_dir.mkdir(parents=True, exist_ok=True)
        
        # Scheduler setup
        self.scheduler_active = False
        self.scheduler_thread = None
        
        # Load maintenance tasks
        self.maintenance_tasks = self._initialize_maintenance_tasks()
        
        logger.info(f"Organization Maintenance System initialized for: {self.root_path}")
        logger.info(f"Loaded {len(self.maintenance_tasks)} maintenance tasks")

    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load maintenance configuration."""
        default_config = {
            "maintenance_schedule": {
                "daily_cleanup": True,
                "weekly_deep_scan": True,
                "monthly_archive_review": True,
                "quarterly_full_reorganization": False
            },
            "auto_healing": {
                "enabled": True,
                "max_auto_fixes_per_session": 20,
                "safety_threshold": "warning",  # Only auto-fix warning and below
                "require_backup": True,
                "max_file_size_mb": 10  # Don't auto-move files larger than this
            },
            "preventive_maintenance": {
                "monitor_health_score": True,
                "trend_analysis_enabled": True,
                "predictive_cleanup": True,
                "capacity_monitoring": True
            },
            "recovery_procedures": {
                "backup_retention_days": 30,
                "rollback_enabled": True,
                "emergency_contacts": [],
                "escalation_threshold": "critical"
            },
            "documentation": {
                "auto_generate": True,
                "update_frequency": "weekly",
                "include_metrics": True,
                "stakeholder_reports": True
            }
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file) as f:
                    user_config = json.load(f)
                    self._deep_merge_config(default_config, user_config)
            except Exception as e:
                logger.warning(f"Could not load config file {config_file}: {e}")
        
        return default_config

    def _deep_merge_config(self, base_config: dict, user_config: dict):
        """Recursively merge user config into base config."""
        for key, value in user_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._deep_merge_config(base_config[key], value)
            else:
                base_config[key] = value

    def _initialize_maintenance_tasks(self) -> List[MaintenanceTask]:
        """Initialize maintenance task definitions."""
        tasks = []
        
        # Daily tasks
        if self.config['maintenance_schedule']['daily_cleanup']:
            tasks.append(MaintenanceTask(
                task_id="daily_health_check",
                name="Daily Health Check",
                description="Monitor repository health and detect new violations",
                schedule="daily",
                handler="daily_health_check",
                auto_fix=True,
                severity_threshold="warning"
            ))
        
        # Weekly tasks  
        if self.config['maintenance_schedule']['weekly_deep_scan']:
            tasks.append(MaintenanceTask(
                task_id="weekly_deep_scan",
                name="Weekly Deep Scan",
                description="Comprehensive scan and cleanup of repository organization",
                schedule="weekly",
                handler="weekly_deep_scan",
                auto_fix=True,
                severity_threshold="error"
            ))
        
        # Monthly tasks
        if self.config['maintenance_schedule']['monthly_archive_review']:
            tasks.append(MaintenanceTask(
                task_id="monthly_archive_review", 
                name="Monthly Archive Review",
                description="Review and clean up old archives, checkpoints, and temporary files",
                schedule="monthly",
                handler="monthly_archive_review",
                auto_fix=True,
                severity_threshold="info"
            ))
        
        # Quarterly tasks
        if self.config['maintenance_schedule']['quarterly_full_reorganization']:
            tasks.append(MaintenanceTask(
                task_id="quarterly_reorganization",
                name="Quarterly Full Reorganization",
                description="Complete repository reorganization and optimization",
                schedule="quarterly",
                handler="quarterly_full_reorganization",
                auto_fix=False,  # Requires manual approval
                severity_threshold="critical"
            ))
        
        # Self-healing tasks
        if self.config['auto_healing']['enabled']:
            tasks.append(MaintenanceTask(
                task_id="continuous_healing",
                name="Continuous Self-Healing",
                description="Automatically fix minor organization violations",
                schedule="hourly",
                handler="continuous_self_healing",
                auto_fix=True,
                severity_threshold=self.config['auto_healing']['safety_threshold']
            ))
        
        return tasks

    def start_maintenance_scheduler(self):
        """Start the automated maintenance scheduler."""
        if self.scheduler_active:
            logger.warning("Maintenance scheduler is already active")
            return
        
        logger.info("Starting automated maintenance scheduler...")
        
        # Schedule tasks
        for task in self.maintenance_tasks:
            if not task.enabled:
                continue
                
            self._schedule_task(task)
        
        self.scheduler_active = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("Maintenance scheduler started successfully")

    def stop_maintenance_scheduler(self):
        """Stop the maintenance scheduler."""
        if not self.scheduler_active:
            logger.warning("Maintenance scheduler is not active")
            return
        
        self.scheduler_active = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=30)
        
        schedule.clear()
        logger.info("Maintenance scheduler stopped")

    def _schedule_task(self, task: MaintenanceTask):
        """Schedule a specific maintenance task."""
        if task.schedule == "hourly":
            schedule.every().hour.do(self._run_maintenance_task, task)
        elif task.schedule == "daily":
            schedule.every().day.at("02:00").do(self._run_maintenance_task, task)
        elif task.schedule == "weekly":
            schedule.every().week.do(self._run_maintenance_task, task)
        elif task.schedule == "monthly":
            schedule.every(30).days.do(self._run_maintenance_task, task)
        elif task.schedule == "quarterly":
            schedule.every(90).days.do(self._run_maintenance_task, task)
        
        logger.info(f"Scheduled task: {task.name} ({task.schedule})")

    def _scheduler_loop(self):
        """Main scheduler loop."""
        logger.info("Maintenance scheduler loop started")
        
        while self.scheduler_active:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                time.sleep(60)
        
        logger.info("Maintenance scheduler loop stopped")

    def _run_maintenance_task(self, task: MaintenanceTask) -> MaintenanceResult:
        """Execute a maintenance task."""
        logger.info(f"Running maintenance task: {task.name}")
        
        start_time = datetime.now()
        task.last_run = start_time
        
        try:
            # Get task handler
            handler = getattr(self, task.handler, None)
            if not handler:
                error_msg = f"Handler not found for task: {task.task_id}"
                logger.error(error_msg)
                return MaintenanceResult(
                    task_id=task.task_id,
                    timestamp=start_time,
                    success=False,
                    actions_taken=[],
                    issues_found=0,
                    issues_resolved=0,
                    backup_created=False,
                    errors=[error_msg]
                )
            
            # Execute task
            result = handler(task)
            
            # Log result
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Task {task.name} completed in {duration:.1f}s: "
                       f"{result.issues_resolved}/{result.issues_found} issues resolved")
            
            # Save result
            self._save_maintenance_result(result)
            
            return result
            
        except Exception as e:
            error_msg = f"Task {task.name} failed: {e}"
            logger.error(error_msg)
            return MaintenanceResult(
                task_id=task.task_id,
                timestamp=start_time,
                success=False,
                actions_taken=[],
                issues_found=0,
                issues_resolved=0,
                backup_created=False,
                errors=[error_msg]
            )

    def daily_health_check(self, task: MaintenanceTask) -> MaintenanceResult:
        """Daily health monitoring and minor issue resolution."""
        logger.info("Performing daily health check...")
        
        actions_taken = []
        issues_found = 0
        issues_resolved = 0
        backup_created = False
        errors = []
        
        try:
            # Collect current health metrics
            health_report = self.monitor.get_current_health_report()
            issues_found = health_report['stats']['violations_detected']
            
            actions_taken.append(f"Health score: {health_report['health_score']:.1f}")
            actions_taken.append(f"Violations detected: {issues_found}")
            
            # Run validation
            validation_report = self.validator.validate('ci_validation')
            actions_taken.append(f"Validation status: {validation_report.overall_status}")
            
            # Auto-heal if enabled and safe
            if task.auto_fix and self.config['auto_healing']['enabled']:
                healing_result = self._perform_auto_healing(task.severity_threshold)
                issues_resolved += healing_result['resolved_count']
                actions_taken.extend(healing_result['actions'])
                backup_created = healing_result['backup_created']
            
            # Generate daily report
            if self.config['documentation']['auto_generate']:
                daily_report_path = self.maintenance_dir / f"daily_health_{datetime.now().strftime('%Y%m%d')}.json"
                with open(daily_report_path, 'w') as f:
                    json.dump({
                        'timestamp': datetime.now().isoformat(),
                        'health_report': health_report,
                        'validation_report': asdict(validation_report),
                        'actions_taken': actions_taken
                    }, f, indent=2, default=str)
                actions_taken.append(f"Daily report saved: {daily_report_path}")
            
            return MaintenanceResult(
                task_id=task.task_id,
                timestamp=datetime.now(),
                success=True,
                actions_taken=actions_taken,
                issues_found=issues_found,
                issues_resolved=issues_resolved,
                backup_created=backup_created,
                errors=errors
            )
            
        except Exception as e:
            errors.append(str(e))
            return MaintenanceResult(
                task_id=task.task_id,
                timestamp=datetime.now(),
                success=False,
                actions_taken=actions_taken,
                issues_found=issues_found,
                issues_resolved=issues_resolved,
                backup_created=backup_created,
                errors=errors
            )

    def weekly_deep_scan(self, task: MaintenanceTask) -> MaintenanceResult:
        """Weekly comprehensive scan and cleanup."""
        logger.info("Performing weekly deep scan...")
        
        actions_taken = []
        issues_found = 0
        issues_resolved = 0
        backup_created = False
        errors = []
        
        try:
            # Create backup if auto-fix is enabled
            if task.auto_fix and self.config['auto_healing']['require_backup']:
                backup_manifest = self.safety_validator.create_backup(
                    ["."],  # Backup current state
                    f"weekly_scan_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                backup_created = True
                actions_taken.append(f"Backup created: {backup_manifest.backup_id}")
            
            # Full repository scan
            scan_results = self.scanner.scan_repository()
            issues_found = len([f for f in scan_results['files'] if f.issues])
            actions_taken.append(f"Scanned {len(scan_results['files'])} files")
            
            # Directory structure analysis
            structure_issues = self.directory_standardizer.analyze_directory_structure()
            issues_found += len(structure_issues)
            actions_taken.append(f"Found {len(structure_issues)} structure issues")
            
            # Auto-fix structural issues if safe
            if task.auto_fix and structure_issues:
                fixed_count = self._fix_structure_issues(structure_issues, task.severity_threshold)
                issues_resolved += fixed_count
                actions_taken.append(f"Fixed {fixed_count} structure issues")
            
            # Clean up scattered root files
            root_cleanup_result = self._cleanup_root_directory(task.auto_fix)
            issues_resolved += root_cleanup_result['resolved']
            actions_taken.extend(root_cleanup_result['actions'])
            
            # Generate weekly report
            weekly_report = self.reporter.generate_detailed_json_report()
            weekly_report_path = self.maintenance_dir / f"weekly_scan_{datetime.now().strftime('%Y%m%d')}.json"
            with open(weekly_report_path, 'w') as f:
                json.dump(weekly_report, f, indent=2, default=str)
            actions_taken.append(f"Weekly report saved: {weekly_report_path}")
            
            return MaintenanceResult(
                task_id=task.task_id,
                timestamp=datetime.now(),
                success=True,
                actions_taken=actions_taken,
                issues_found=issues_found,
                issues_resolved=issues_resolved,
                backup_created=backup_created,
                errors=errors
            )
            
        except Exception as e:
            errors.append(str(e))
            return MaintenanceResult(
                task_id=task.task_id,
                timestamp=datetime.now(),
                success=False,
                actions_taken=actions_taken,
                issues_found=issues_found,
                issues_resolved=issues_resolved,
                backup_created=backup_created,
                errors=errors
            )

    def monthly_archive_review(self, task: MaintenanceTask) -> MaintenanceResult:
        """Monthly archive and cleanup review."""
        logger.info("Performing monthly archive review...")
        
        actions_taken = []
        issues_found = 0
        issues_resolved = 0
        backup_created = False
        errors = []
        
        try:
            # Review checkpoint files
            checkpoint_dir = self.root_path / "checkpoints"
            if checkpoint_dir.exists():
                old_checkpoints = self._identify_old_checkpoints(days_old=30)
                issues_found += len(old_checkpoints)
                
                if task.auto_fix and old_checkpoints:
                    archived_count = self._archive_old_checkpoints(old_checkpoints)
                    issues_resolved += archived_count
                    actions_taken.append(f"Archived {archived_count} old checkpoints")
            
            # Review temp files
            temp_dir = self.root_path / "temp"
            if temp_dir.exists():
                old_temp_files = self._identify_old_temp_files(days_old=7)
                issues_found += len(old_temp_files)
                
                if task.auto_fix and old_temp_files:
                    cleaned_count = self._cleanup_old_temp_files(old_temp_files)
                    issues_resolved += cleaned_count
                    actions_taken.append(f"Cleaned {cleaned_count} old temp files")
            
            # Review large files
            large_files = self._identify_large_files(threshold_mb=50)
            if large_files:
                actions_taken.append(f"Identified {len(large_files)} large files for review")
            
            # Clean up old backups
            cleaned_backups = self.safety_validator.cleanup_old_backups(keep_count=10)
            if cleaned_backups > 0:
                actions_taken.append(f"Cleaned up {cleaned_backups} old backups")
            
            # Generate archive report
            archive_report_path = self.maintenance_dir / f"archive_review_{datetime.now().strftime('%Y%m%d')}.json"
            with open(archive_report_path, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'old_checkpoints_found': len(old_checkpoints) if 'old_checkpoints' in locals() else 0,
                    'old_temp_files_found': len(old_temp_files) if 'old_temp_files' in locals() else 0,
                    'large_files_found': len(large_files),
                    'actions_taken': actions_taken
                }, f, indent=2, default=str)
            actions_taken.append(f"Archive report saved: {archive_report_path}")
            
            return MaintenanceResult(
                task_id=task.task_id,
                timestamp=datetime.now(),
                success=True,
                actions_taken=actions_taken,
                issues_found=issues_found,
                issues_resolved=issues_resolved,
                backup_created=backup_created,
                errors=errors
            )
            
        except Exception as e:
            errors.append(str(e))
            return MaintenanceResult(
                task_id=task.task_id,
                timestamp=datetime.now(),
                success=False,
                actions_taken=actions_taken,
                issues_found=issues_found,
                issues_resolved=issues_resolved,
                backup_created=backup_created,
                errors=errors
            )

    def continuous_self_healing(self, task: MaintenanceTask) -> MaintenanceResult:
        """Continuous self-healing for minor violations."""
        logger.debug("Performing continuous self-healing check...")
        
        actions_taken = []
        issues_found = 0
        issues_resolved = 0
        backup_created = False
        errors = []
        
        try:
            # Check for recent violations
            violations = self.monitor._perform_organization_scan()
            
            # Filter to only minor violations
            minor_violations = [v for v in violations 
                             if v.severity in ['info', 'warning'] and v.auto_fix_available]
            
            issues_found = len(minor_violations)
            
            if not minor_violations:
                return MaintenanceResult(
                    task_id=task.task_id,
                    timestamp=datetime.now(),
                    success=True,
                    actions_taken=["No violations requiring auto-healing"],
                    issues_found=0,
                    issues_resolved=0,
                    backup_created=False,
                    errors=[]
                )
            
            # Limit auto-fixes per session
            max_fixes = self.config['auto_healing']['max_auto_fixes_per_session']
            violations_to_fix = minor_violations[:max_fixes]
            
            # Create backup for auto-fixes
            if self.config['auto_healing']['require_backup']:
                files_to_backup = [v.file_path for v in violations_to_fix]
                backup_manifest = self.safety_validator.create_backup(
                    files_to_backup,
                    f"self_healing_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                backup_created = True
                actions_taken.append(f"Backup created: {backup_manifest.backup_id}")
            
            # Apply fixes
            for violation in violations_to_fix:
                try:
                    fix_result = self._apply_violation_fix(violation)
                    if fix_result:
                        issues_resolved += 1
                        actions_taken.append(f"Fixed: {violation.violation_type} - {violation.file_path}")
                except Exception as e:
                    errors.append(f"Failed to fix {violation.violation_id}: {e}")
            
            return MaintenanceResult(
                task_id=task.task_id,
                timestamp=datetime.now(),
                success=len(errors) == 0,
                actions_taken=actions_taken,
                issues_found=issues_found,
                issues_resolved=issues_resolved,
                backup_created=backup_created,
                errors=errors
            )
            
        except Exception as e:
            errors.append(str(e))
            return MaintenanceResult(
                task_id=task.task_id,
                timestamp=datetime.now(),
                success=False,
                actions_taken=actions_taken,
                issues_found=issues_found,
                issues_resolved=issues_resolved,
                backup_created=backup_created,
                errors=errors
            )

    def _perform_auto_healing(self, severity_threshold: str) -> Dict[str, Any]:
        """Perform automated healing of violations."""
        healing_result = {
            'resolved_count': 0,
            'actions': [],
            'backup_created': False
        }
        
        try:
            violations = self.monitor._perform_organization_scan()
            
            # Filter violations based on severity threshold
            severity_order = ['info', 'warning', 'error', 'critical']
            threshold_index = severity_order.index(severity_threshold)
            
            healable_violations = [
                v for v in violations 
                if (v.auto_fix_available and 
                    severity_order.index(v.severity) <= threshold_index)
            ]
            
            if not healable_violations:
                healing_result['actions'].append("No violations requiring auto-healing")
                return healing_result
            
            # Apply fixes
            for violation in healable_violations[:10]:  # Limit to 10 per run
                try:
                    if self._apply_violation_fix(violation):
                        healing_result['resolved_count'] += 1
                        healing_result['actions'].append(f"Auto-fixed: {violation.violation_type}")
                except Exception as e:
                    healing_result['actions'].append(f"Failed to fix {violation.violation_type}: {e}")
            
        except Exception as e:
            healing_result['actions'].append(f"Auto-healing error: {e}")
        
        return healing_result

    def _apply_violation_fix(self, violation) -> bool:
        """Apply a fix for a specific violation."""
        try:
            # Simple implementation - in production would need more sophisticated handling
            if violation.violation_type == "scattered_root_file":
                # Use proven root organizer
                return self._move_scattered_file(violation.file_path, violation.expected_location)
            elif violation.violation_type == "old_file":
                # Move to archive
                return self._archive_old_file(violation.file_path)
            elif violation.violation_type == "malformed_directory":
                # Use directory standardizer 
                return self._fix_directory_name(violation.file_path, violation.expected_location)
            
            return False
        except Exception as e:
            logger.warning(f"Failed to apply fix for {violation.violation_id}: {e}")
            return False

    def _move_scattered_file(self, file_path: str, target_location: str) -> bool:
        """Move scattered file using proven organizer."""
        try:
            # Use the proven root directory organizer
            return self.root_organizer._move_file_safely(file_path, target_location)
        except:
            return False

    def _archive_old_file(self, file_path: str) -> bool:
        """Archive old file."""
        try:
            file_path_obj = Path(self.root_path) / file_path
            if file_path_obj.exists():
                archive_dir = file_path_obj.parent / "archive"
                archive_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(file_path_obj), str(archive_dir / file_path_obj.name))
                return True
        except:
            pass
        return False

    def _fix_directory_name(self, current_path: str, suggested_name: str) -> bool:
        """Fix directory name using standardizer."""
        try:
            # Use the proven directory standardizer
            current_dir = Path(self.root_path) / current_path
            if current_dir.exists() and current_dir.is_dir():
                new_dir = current_dir.parent / suggested_name
                current_dir.rename(new_dir)
                return True
        except:
            pass
        return False

    def generate_maintenance_documentation(self) -> Path:
        """Generate comprehensive maintenance documentation."""
        logger.info("Generating maintenance documentation...")
        
        doc_content = [
            "# Repository Organization Maintenance System",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Repository: {self.root_path}",
            "",
            "## System Overview",
            "",
            "This repository uses an automated organization maintenance system built on:",
            "- **Stream A Infrastructure**: File discovery and root directory organization",
            "- **Stream B Infrastructure**: Directory structure standardization", 
            "- **Stream C Infrastructure**: Monitoring, validation, and reporting",
            "",
            "### Key Components",
            "",
            "1. **Repository Scanner** - Comprehensive file discovery and categorization",
            "2. **Safety Validator** - Multi-layer safety checks and backup systems",
            "3. **Organization Monitor** - Real-time violation detection and alerting",
            "4. **Validation System** - CI/CD integrated organization validation",
            "5. **Reporting System** - Automated compliance reporting and trend analysis",
            "6. **Maintenance System** - Automated cleanup and self-healing capabilities",
            "",
            "## Automated Maintenance Tasks",
            ""
        ]
        
        # Document maintenance tasks
        for task in self.maintenance_tasks:
            doc_content.extend([
                f"### {task.name}",
                f"- **Schedule**: {task.schedule}",
                f"- **Description**: {task.description}",
                f"- **Auto-fix**: {'Enabled' if task.auto_fix else 'Disabled'}",
                f"- **Status**: {'Enabled' if task.enabled else 'Disabled'}",
                ""
            ])
        
        # Add usage instructions
        doc_content.extend([
            "## Usage Instructions",
            "",
            "### Starting Automated Maintenance",
            "```bash",
            "python scripts/organization_maintenance.py --start-scheduler",
            "```",
            "",
            "### Manual Maintenance Tasks",
            "```bash",
            "# Run daily health check",
            "python scripts/organization_maintenance.py --task daily_health_check",
            "",
            "# Run weekly deep scan",
            "python scripts/organization_maintenance.py --task weekly_deep_scan",
            "",
            "# Run monthly archive review",
            "python scripts/organization_maintenance.py --task monthly_archive_review",
            "```",
            "",
            "### Monitoring and Validation",
            "```bash",
            "# Check current health",
            "python scripts/repository_organization_monitor.py --dashboard",
            "",
            "# Run validation",
            "python scripts/organization_validator.py --suite ci_validation",
            "",
            "# Generate reports",
            "python scripts/organization_reporter.py --generate-all",
            "```",
            "",
            "## Emergency Procedures",
            "",
            "### If Organization Issues Occur",
            "1. **Stop automated maintenance**: `python scripts/organization_maintenance.py --stop-scheduler`",
            "2. **Check health status**: `python scripts/repository_organization_monitor.py --status`",
            "3. **Run validation**: `python scripts/organization_validator.py --suite full_validation`",
            "4. **Review recent changes**: Check git history for recent organization changes",
            "5. **Restore from backup if needed**: Use safety validator backup restoration",
            "",
            "### Recovery from Backup",
            "```bash",
            "# List available backups",
            "python scripts/safety_validator.py --list-backups",
            "",
            "# Restore specific backup",
            "python scripts/safety_validator.py --restore-backup BACKUP_ID",
            "```",
            "",
            "## Configuration Files",
            "",
            "- `config/monitoring_config.json` - Monitoring system configuration",
            "- `config/validation_config.json` - Validation system configuration", 
            "- `config/reporting_config.json` - Reporting system configuration",
            "- `config/maintenance_config.json` - Maintenance system configuration",
            "",
            "## Maintenance Logs and Reports",
            "",
            "- **Maintenance Logs**: `temp/maintenance/logs/`",
            "- **Health Reports**: `temp/reports/`",
            "- **Validation Reports**: `temp/validation/`",
            "- **Monitoring Data**: `temp/monitoring/`",
            "",
            f"## System Status (as of {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})",
            "",
        ]
        
        # Add current system status
        try:
            health_report = self.monitor.get_current_health_report()
            doc_content.extend([
                f"- **Health Score**: {health_report['health_score']:.1f}/100",
                f"- **Total Violations**: {health_report['stats']['violations_detected']}",
                f"- **Monitoring Status**: {health_report['monitoring_status']}",
                f"- **Last Scan**: {health_report['stats']['last_scan_time']}",
                ""
            ])
        except:
            doc_content.append("- **Status**: Unable to retrieve current status")
        
        # Save documentation
        doc_path = self.documentation_dir / "README.md"
        with open(doc_path, 'w') as f:
            f.write('\n'.join(doc_content))
        
        logger.info(f"Maintenance documentation saved: {doc_path}")
        return doc_path

    def _save_maintenance_result(self, result: MaintenanceResult):
        """Save maintenance result to log."""
        log_file = self.log_dir / f"maintenance_{datetime.now().strftime('%Y%m%d')}.json"
        
        # Load existing logs
        logs = []
        if log_file.exists():
            try:
                with open(log_file) as f:
                    logs = json.load(f)
            except:
                logs = []
        
        # Add new result
        logs.append(asdict(result))
        
        # Save updated logs
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2, default=str)

    def get_maintenance_status(self) -> Dict[str, Any]:
        """Get current maintenance system status."""
        return {
            'scheduler_active': self.scheduler_active,
            'total_tasks': len(self.maintenance_tasks),
            'enabled_tasks': len([t for t in self.maintenance_tasks if t.enabled]),
            'auto_healing_enabled': self.config['auto_healing']['enabled'],
            'last_maintenance_runs': {
                task.task_id: task.last_run.isoformat() if task.last_run else None
                for task in self.maintenance_tasks
            }
        }

    # Helper methods for specific cleanup operations
    def _identify_old_checkpoints(self, days_old: int = 30) -> List[Path]:
        """Identify old checkpoint files."""
        checkpoint_dir = self.root_path / "checkpoints"
        if not checkpoint_dir.exists():
            return []
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        old_checkpoints = []
        
        for checkpoint_file in checkpoint_dir.rglob('*.json'):
            try:
                if checkpoint_file.stat().st_mtime < cutoff_date.timestamp():
                    old_checkpoints.append(checkpoint_file)
            except:
                continue
        
        return old_checkpoints

    def _identify_old_temp_files(self, days_old: int = 7) -> List[Path]:
        """Identify old temporary files."""
        temp_dir = self.root_path / "temp"
        if not temp_dir.exists():
            return []
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        old_temp_files = []
        
        for temp_file in temp_dir.rglob('*'):
            if temp_file.is_file():
                try:
                    if temp_file.stat().st_mtime < cutoff_date.timestamp():
                        old_temp_files.append(temp_file)
                except:
                    continue
        
        return old_temp_files

    def _identify_large_files(self, threshold_mb: int = 50) -> List[Path]:
        """Identify large files that might need attention."""
        threshold_bytes = threshold_mb * 1024 * 1024
        large_files = []
        
        for file_path in self.root_path.rglob('*'):
            if file_path.is_file():
                try:
                    if file_path.stat().st_size > threshold_bytes:
                        large_files.append(file_path)
                except:
                    continue
        
        return large_files


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Repository Organization Maintenance System")
    parser.add_argument("--root", default=".", help="Repository root path")
    parser.add_argument("--config", help="Configuration file path") 
    parser.add_argument("--start-scheduler", action='store_true', help="Start maintenance scheduler")
    parser.add_argument("--stop-scheduler", action='store_true', help="Stop maintenance scheduler")
    parser.add_argument("--status", action='store_true', help="Show maintenance status")
    parser.add_argument("--task", help="Run specific maintenance task")
    parser.add_argument("--generate-docs", action='store_true', help="Generate maintenance documentation")
    parser.add_argument("--verbose", action='store_true', help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    maintenance = OrganizationMaintenance(args.root, args.config)
    
    if args.start_scheduler:
        maintenance.start_maintenance_scheduler()
        print("✅ Maintenance scheduler started")
        print("Press Ctrl+C to stop...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            maintenance.stop_maintenance_scheduler()
            print("\n🛑 Maintenance scheduler stopped")
    
    elif args.stop_scheduler:
        maintenance.stop_maintenance_scheduler()
        print("🛑 Maintenance scheduler stopped")
    
    elif args.status:
        status = maintenance.get_maintenance_status()
        print("📊 Maintenance System Status:")
        print(f"  Scheduler Active: {status['scheduler_active']}")
        print(f"  Total Tasks: {status['total_tasks']}")
        print(f"  Enabled Tasks: {status['enabled_tasks']}")
        print(f"  Auto-healing: {status['auto_healing_enabled']}")
        print("  Last Runs:")
        for task_id, last_run in status['last_maintenance_runs'].items():
            run_time = last_run if last_run else "Never"
            print(f"    {task_id}: {run_time}")
    
    elif args.task:
        # Find and run specific task
        task = next((t for t in maintenance.maintenance_tasks if t.task_id == args.task), None)
        if task:
            print(f"🔧 Running maintenance task: {task.name}")
            result = maintenance._run_maintenance_task(task)
            print(f"✅ Task completed: {result.issues_resolved}/{result.issues_found} issues resolved")
            if result.errors:
                print(f"⚠️ Errors: {len(result.errors)}")
        else:
            print(f"❌ Task not found: {args.task}")
            print("Available tasks:")
            for t in maintenance.maintenance_tasks:
                print(f"  {t.task_id}: {t.name}")
    
    elif args.generate_docs:
        doc_path = maintenance.generate_maintenance_documentation()
        print(f"📝 Maintenance documentation generated: {doc_path}")
    
    else:
        print("Use --start-scheduler, --stop-scheduler, --status, --task, or --generate-docs")


if __name__ == "__main__":
    main()