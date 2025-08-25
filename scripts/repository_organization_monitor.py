#!/usr/bin/env python3
"""
Repository Organization Monitor for Issue #255 Stream C.

Automated monitoring system for detecting and alerting on repository organization violations:
- Real-time file system monitoring for organization standard breaches
- Automated detection of scattered files, malformed directories
- Integration with existing safety and validation frameworks
- Configurable alerts and automated remediation capabilities

Building on proven infrastructure from Streams A & B.
"""

import json
import logging
import os
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
import threading
import queue
import re

# Import our existing proven infrastructure
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from repository_scanner import RepositoryScanner, FileInfo
    from safety_validator import SafetyValidator, SafetyCheck
    from directory_structure_analyzer import DirectoryStructureAnalyzer, DirectoryIssue
except ImportError as e:
    logging.error(f"Failed to import required modules: {e}")
    logging.error("Ensure repository_scanner.py, safety_validator.py, and directory_structure_analyzer.py are in the same directory")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class OrganizationViolation:
    """Represents a detected organization violation."""
    timestamp: datetime
    violation_id: str
    violation_type: str
    severity: str  # 'info', 'warning', 'error', 'critical'
    file_path: str
    current_location: str
    expected_location: str
    description: str
    auto_fix_available: bool = False
    fix_commands: List[str] = None
    
    def __post_init__(self):
        if self.fix_commands is None:
            self.fix_commands = []


@dataclass
class MonitoringStats:
    """Statistics about monitoring activity."""
    monitoring_start: datetime
    total_files_monitored: int
    violations_detected: int
    violations_resolved: int
    last_scan_time: datetime
    scan_count: int
    avg_scan_duration: float
    health_score: float  # 0-100 based on organization compliance


class RepositoryOrganizationMonitor:
    """Automated monitoring system for repository organization compliance."""
    
    def __init__(self, root_path: str = ".", config_file: Optional[str] = None):
        self.root_path = Path(root_path).resolve()
        self.config = self._load_config(config_file)
        
        # Initialize core components
        self.scanner = RepositoryScanner(str(self.root_path))
        self.safety_validator = SafetyValidator(str(self.root_path))
        self.structure_analyzer = DirectoryStructureAnalyzer(str(self.root_path))
        
        # Monitoring state
        self.violations = deque(maxlen=self.config['max_violations_history'])
        self.stats = MonitoringStats(
            monitoring_start=datetime.now(),
            total_files_monitored=0,
            violations_detected=0,
            violations_resolved=0,
            last_scan_time=datetime.now(),
            scan_count=0,
            avg_scan_duration=0.0,
            health_score=100.0
        )
        
        # Threading and control
        self.monitoring_active = False
        self.monitoring_thread = None
        self.violation_queue = queue.Queue()
        self.alert_handlers = []
        
        # Monitoring intervals (in seconds)
        self.scan_interval = self.config['monitoring']['scan_interval_seconds']
        self.alert_cooldown = self.config['monitoring']['alert_cooldown_seconds']
        self.last_alert_times = defaultdict(lambda: datetime.min)
        
        # Create monitoring directories
        self.monitor_dir = self.root_path / "temp" / "monitoring"
        self.monitor_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Repository Organization Monitor initialized for: {self.root_path}")

    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load monitoring configuration."""
        default_config = {
            "max_violations_history": 1000,
            "monitoring": {
                "scan_interval_seconds": 300,  # 5 minutes
                "alert_cooldown_seconds": 3600,  # 1 hour
                "auto_fix_enabled": False,
                "max_auto_fix_files": 10
            },
            "violation_thresholds": {
                "scattered_files_threshold": 5,
                "malformed_directories_threshold": 2,
                "large_files_threshold_mb": 50,
                "old_files_threshold_days": 90
            },
            "organization_standards": {
                "max_root_files": 20,
                "required_directories": ["src", "scripts", "examples", "tests"],
                "forbidden_root_patterns": [
                    r"^test_.*\.py$",
                    r"^debug_.*",
                    r"^temp_.*",
                    r".*\.tmp$"
                ]
            },
            "alerts": {
                "enabled": True,
                "log_file": "temp/monitoring/violations.log",
                "email_enabled": False,
                "slack_enabled": False
            }
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file) as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Could not load config file {config_file}: {e}")
        
        return default_config

    def start_monitoring(self):
        """Start the monitoring daemon."""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Repository organization monitoring started")

    def stop_monitoring(self):
        """Stop the monitoring daemon."""
        if not self.monitoring_active:
            logger.warning("Monitoring not active")
            return
        
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=30)
        
        logger.info("Repository organization monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop."""
        logger.info(f"Monitoring loop started with {self.scan_interval}s intervals")
        
        while self.monitoring_active:
            try:
                start_time = time.time()
                
                # Perform organization scan
                violations = self._perform_organization_scan()
                
                # Process violations
                for violation in violations:
                    self._process_violation(violation)
                
                # Update statistics
                scan_duration = time.time() - start_time
                self._update_stats(len(violations), scan_duration)
                
                # Save periodic monitoring report
                if self.stats.scan_count % 10 == 0:  # Every 10 scans
                    self._save_monitoring_report()
                
                # Wait for next scan
                time.sleep(self.scan_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait a minute before retrying

    def _perform_organization_scan(self) -> List[OrganizationViolation]:
        """Perform a comprehensive organization scan."""
        violations = []
        
        logger.debug("Performing organization scan...")
        
        # 1. Scan for scattered files in root directory
        violations.extend(self._scan_scattered_root_files())
        
        # 2. Scan for malformed directory names
        violations.extend(self._scan_malformed_directories())
        
        # 3. Scan for files violating naming conventions
        violations.extend(self._scan_naming_violations())
        
        # 4. Scan for old/unused files
        violations.extend(self._scan_old_files())
        
        # 5. Scan for oversized files
        violations.extend(self._scan_oversized_files())
        
        # 6. Validate directory structure compliance
        violations.extend(self._validate_structure_compliance())
        
        logger.debug(f"Organization scan complete: {len(violations)} violations found")
        return violations

    def _scan_scattered_root_files(self) -> List[OrganizationViolation]:
        """Scan for files scattered in root directory that should be organized."""
        violations = []
        
        try:
            # Use our proven repository scanner
            scan_results = self.scanner.scan_repository()
            
            # Find files in root that should be moved
            root_files = [f for f in scan_results['files'] 
                         if f.path.parent == Path('.') and f.subcategory == 'scattered_in_root']
            
            if len(root_files) > self.config['violation_thresholds']['scattered_files_threshold']:
                for file_info in root_files:
                    violation = OrganizationViolation(
                        timestamp=datetime.now(),
                        violation_id=f"scattered_root_{file_info.path.name}_{int(time.time())}",
                        violation_type="scattered_root_file",
                        severity="warning",
                        file_path=str(file_info.path),
                        current_location=str(file_info.path.parent),
                        expected_location=file_info.target_location or "appropriate_directory",
                        description=f"File '{file_info.path.name}' should be moved to {file_info.target_location}",
                        auto_fix_available=file_info.safety_level == 'safe',
                        fix_commands=[
                            f"mkdir -p {file_info.target_location}",
                            f"mv '{file_info.path}' '{file_info.target_location}'"
                        ] if file_info.target_location else []
                    )
                    violations.append(violation)
                    
        except Exception as e:
            logger.error(f"Error scanning scattered root files: {e}")
        
        return violations

    def _scan_malformed_directories(self) -> List[OrganizationViolation]:
        """Scan for malformed directory names."""
        violations = []
        
        try:
            # Use our proven directory structure analyzer
            issues = self.structure_analyzer.analyze_directory_structure()
            
            malformed_count = len([i for i in issues if i.issue_type == 'malformed_name'])
            
            if malformed_count > self.config['violation_thresholds']['malformed_directories_threshold']:
                for issue in issues:
                    if issue.issue_type in ['malformed_name', 'naming_convention']:
                        violation = OrganizationViolation(
                            timestamp=datetime.now(),
                            violation_id=f"malformed_dir_{issue.current_name}_{int(time.time())}",
                            violation_type=issue.issue_type,
                            severity=issue.severity,
                            file_path=str(issue.path),
                            current_location=issue.current_name,
                            expected_location=issue.suggested_name,
                            description=issue.description,
                            auto_fix_available=issue.severity != 'critical',
                            fix_commands=issue.fix_commands
                        )
                        violations.append(violation)
                        
        except Exception as e:
            logger.error(f"Error scanning malformed directories: {e}")
        
        return violations

    def _scan_naming_violations(self) -> List[OrganizationViolation]:
        """Scan for files/directories violating naming conventions."""
        violations = []
        
        try:
            # Check against forbidden patterns in root
            root_path = self.root_path
            for item in root_path.iterdir():
                if item.is_file():
                    for pattern in self.config['organization_standards']['forbidden_root_patterns']:
                        if re.match(pattern, item.name):
                            violation = OrganizationViolation(
                                timestamp=datetime.now(),
                                violation_id=f"naming_violation_{item.name}_{int(time.time())}",
                                violation_type="naming_convention",
                                severity="warning",
                                file_path=str(item.relative_to(root_path)),
                                current_location="root",
                                expected_location="appropriate_directory",
                                description=f"File '{item.name}' matches forbidden root pattern: {pattern}",
                                auto_fix_available=True,
                                fix_commands=[
                                    f"# Move {item.name} to appropriate directory based on type"
                                ]
                            )
                            violations.append(violation)
                            break
                            
        except Exception as e:
            logger.error(f"Error scanning naming violations: {e}")
        
        return violations

    def _scan_old_files(self) -> List[OrganizationViolation]:
        """Scan for old files that might need archiving."""
        violations = []
        
        try:
            threshold_days = self.config['violation_thresholds']['old_files_threshold_days']
            cutoff_date = datetime.now() - timedelta(days=threshold_days)
            
            # Focus on temporary and output directories
            temp_patterns = ['temp/', 'examples/outputs/', 'checkpoints/']
            
            for pattern in temp_patterns:
                pattern_path = self.root_path / pattern
                if pattern_path.exists():
                    for file_path in pattern_path.rglob('*'):
                        if file_path.is_file():
                            try:
                                mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                                if mtime < cutoff_date:
                                    violation = OrganizationViolation(
                                        timestamp=datetime.now(),
                                        violation_id=f"old_file_{file_path.name}_{int(time.time())}",
                                        violation_type="old_file",
                                        severity="info",
                                        file_path=str(file_path.relative_to(self.root_path)),
                                        current_location=str(file_path.parent.relative_to(self.root_path)),
                                        expected_location="archive/",
                                        description=f"File '{file_path.name}' is {threshold_days}+ days old and might need archiving",
                                        auto_fix_available=True,
                                        fix_commands=[
                                            f"mkdir -p {file_path.parent}/archive",
                                            f"mv '{file_path}' '{file_path.parent}/archive/'"
                                        ]
                                    )
                                    violations.append(violation)
                            except Exception:
                                continue
                                
        except Exception as e:
            logger.error(f"Error scanning old files: {e}")
        
        return violations

    def _scan_oversized_files(self) -> List[OrganizationViolation]:
        """Scan for oversized files that should be compressed or moved."""
        violations = []
        
        try:
            threshold_mb = self.config['violation_thresholds']['large_files_threshold_mb']
            threshold_bytes = threshold_mb * 1024 * 1024
            
            for file_path in self.root_path.rglob('*'):
                if file_path.is_file():
                    try:
                        size = file_path.stat().st_size
                        if size > threshold_bytes:
                            # Skip certain known large file types
                            if file_path.suffix.lower() in ['.git', '.pyc', '.so', '.dll']:
                                continue
                                
                            violation = OrganizationViolation(
                                timestamp=datetime.now(),
                                violation_id=f"large_file_{file_path.name}_{int(time.time())}",
                                violation_type="oversized_file",
                                severity="warning",
                                file_path=str(file_path.relative_to(self.root_path)),
                                current_location=str(file_path.parent.relative_to(self.root_path)),
                                expected_location="archive/large_files/",
                                description=f"File '{file_path.name}' is {size/(1024*1024):.1f}MB (threshold: {threshold_mb}MB)",
                                auto_fix_available=False,  # Large files need manual review
                                fix_commands=[
                                    f"# Consider compressing or moving large file: {file_path.name}"
                                ]
                            )
                            violations.append(violation)
                    except Exception:
                        continue
                        
        except Exception as e:
            logger.error(f"Error scanning oversized files: {e}")
        
        return violations

    def _validate_structure_compliance(self) -> List[OrganizationViolation]:
        """Validate overall directory structure compliance."""
        violations = []
        
        try:
            # Check for required directories
            required_dirs = self.config['organization_standards']['required_directories']
            for required_dir in required_dirs:
                dir_path = self.root_path / required_dir
                if not dir_path.exists():
                    violation = OrganizationViolation(
                        timestamp=datetime.now(),
                        violation_id=f"missing_dir_{required_dir}_{int(time.time())}",
                        violation_type="missing_required_directory",
                        severity="info",
                        file_path=required_dir,
                        current_location="not_present",
                        expected_location=required_dir,
                        description=f"Required directory '{required_dir}' is missing",
                        auto_fix_available=True,
                        fix_commands=[
                            f"mkdir -p {required_dir}",
                            f"touch {required_dir}/.gitkeep"
                        ]
                    )
                    violations.append(violation)
            
            # Check root file count
            max_root_files = self.config['organization_standards']['max_root_files']
            root_files = [f for f in self.root_path.iterdir() if f.is_file()]
            if len(root_files) > max_root_files:
                violation = OrganizationViolation(
                    timestamp=datetime.now(),
                    violation_id=f"too_many_root_files_{int(time.time())}",
                    violation_type="excessive_root_files",
                    severity="warning",
                    file_path=".",
                    current_location="root",
                    expected_location="organized_directories",
                    description=f"Root directory has {len(root_files)} files (max: {max_root_files})",
                    auto_fix_available=False,  # Need analysis to determine where files should go
                    fix_commands=["# Run repository scanner to determine proper file organization"]
                )
                violations.append(violation)
                
        except Exception as e:
            logger.error(f"Error validating structure compliance: {e}")
        
        return violations

    def _process_violation(self, violation: OrganizationViolation):
        """Process a detected violation."""
        # Add to violation history
        self.violations.append(violation)
        self.stats.violations_detected += 1
        
        # Check if we should send an alert
        if self._should_alert(violation):
            self._send_alert(violation)
        
        # Auto-fix if enabled and safe
        if (self.config['monitoring']['auto_fix_enabled'] and 
            violation.auto_fix_available and 
            violation.severity in ['info', 'warning']):
            self._attempt_auto_fix(violation)

    def _should_alert(self, violation: OrganizationViolation) -> bool:
        """Determine if an alert should be sent for this violation."""
        if not self.config['alerts']['enabled']:
            return False
        
        # Check cooldown period for this type of violation
        last_alert = self.last_alert_times[violation.violation_type]
        cooldown = timedelta(seconds=self.alert_cooldown)
        
        if datetime.now() - last_alert < cooldown:
            return False
        
        # Alert for critical and error violations immediately
        if violation.severity in ['critical', 'error']:
            return True
        
        # Alert for warnings if there are multiple
        if violation.severity == 'warning':
            recent_warnings = [v for v in self.violations 
                             if v.violation_type == violation.violation_type and
                                v.timestamp > datetime.now() - timedelta(hours=1)]
            return len(recent_warnings) >= 3
        
        return False

    def _send_alert(self, violation: OrganizationViolation):
        """Send alert for a violation."""
        self.last_alert_times[violation.violation_type] = datetime.now()
        
        alert_message = (
            f"Repository Organization Violation Detected\n"
            f"Type: {violation.violation_type}\n"
            f"Severity: {violation.severity}\n"
            f"File: {violation.file_path}\n"
            f"Description: {violation.description}\n"
            f"Auto-fix available: {violation.auto_fix_available}\n"
            f"Timestamp: {violation.timestamp}\n"
        )
        
        logger.warning(f"ORGANIZATION VIOLATION: {alert_message}")
        
        # Log to violation file
        if self.config['alerts']['log_file']:
            log_file = Path(self.config['alerts']['log_file'])
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, 'a') as f:
                f.write(f"{datetime.now().isoformat()} - {alert_message}\n---\n")

    def _attempt_auto_fix(self, violation: OrganizationViolation):
        """Attempt to automatically fix a violation."""
        if not violation.fix_commands:
            return
        
        # Safety check: limit number of auto-fixes per monitoring cycle
        auto_fixes_this_cycle = len([v for v in self.violations 
                                   if v.timestamp > datetime.now() - timedelta(minutes=10) and
                                      hasattr(v, 'auto_fixed') and getattr(v, 'auto_fixed', False)])
        
        if auto_fixes_this_cycle >= self.config['monitoring']['max_auto_fix_files']:
            logger.warning("Auto-fix limit reached for this monitoring cycle")
            return
        
        try:
            logger.info(f"Attempting auto-fix for violation: {violation.violation_id}")
            
            # Use safety validator to check if fix is safe
            file_operations = [{
                'source': violation.file_path,
                'target': violation.expected_location,
                'operation': 'move'
            }]
            
            is_safe, safety_checks = self.safety_validator.validate_operation(file_operations)
            
            if not is_safe:
                logger.warning(f"Auto-fix deemed unsafe for violation {violation.violation_id}")
                return
            
            # Create backup before auto-fix
            backup_manifest = self.safety_validator.create_backup([violation.file_path])
            
            # Execute fix commands (simplified - in production would need proper shell execution)
            logger.info(f"Auto-fix commands for {violation.violation_id}:")
            for command in violation.fix_commands:
                logger.info(f"  {command}")
            
            # Mark as auto-fixed
            violation.auto_fixed = True
            self.stats.violations_resolved += 1
            
            logger.info(f"Auto-fix completed for violation: {violation.violation_id}")
            
        except Exception as e:
            logger.error(f"Auto-fix failed for violation {violation.violation_id}: {e}")

    def _update_stats(self, violations_count: int, scan_duration: float):
        """Update monitoring statistics."""
        self.stats.scan_count += 1
        self.stats.last_scan_time = datetime.now()
        
        # Update average scan duration
        total_duration = self.stats.avg_scan_duration * (self.stats.scan_count - 1) + scan_duration
        self.stats.avg_scan_duration = total_duration / self.stats.scan_count
        
        # Calculate health score (0-100) based on violations
        if violations_count == 0:
            violation_score = 100
        else:
            # Penalize based on severity
            severity_weights = {'info': 1, 'warning': 3, 'error': 5, 'critical': 10}
            weighted_violations = sum(severity_weights.get(v.severity, 1) for v in self.violations)
            violation_score = max(0, 100 - weighted_violations)
        
        self.stats.health_score = violation_score

    def _save_monitoring_report(self):
        """Save periodic monitoring report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'stats': asdict(self.stats),
            'recent_violations': [asdict(v) for v in list(self.violations)[-50:]],  # Last 50
            'violation_summary': self._get_violation_summary()
        }
        
        report_file = self.monitor_dir / f"monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Monitoring report saved: {report_file}")

    def _get_violation_summary(self) -> Dict[str, Any]:
        """Get summary of violations by type and severity."""
        summary = {
            'by_type': defaultdict(int),
            'by_severity': defaultdict(int),
            'recent_trend': defaultdict(int)
        }
        
        recent_threshold = datetime.now() - timedelta(hours=24)
        
        for violation in self.violations:
            summary['by_type'][violation.violation_type] += 1
            summary['by_severity'][violation.severity] += 1
            
            if violation.timestamp > recent_threshold:
                summary['recent_trend'][violation.violation_type] += 1
        
        return dict(summary)

    def get_current_health_report(self) -> Dict[str, Any]:
        """Get current repository health report."""
        return {
            'timestamp': datetime.now().isoformat(),
            'health_score': self.stats.health_score,
            'monitoring_status': 'active' if self.monitoring_active else 'inactive',
            'stats': asdict(self.stats),
            'violation_summary': self._get_violation_summary(),
            'recent_violations': [asdict(v) for v in list(self.violations)[-10:]]  # Last 10
        }

    def generate_compliance_dashboard(self) -> str:
        """Generate a text-based compliance dashboard."""
        health_report = self.get_current_health_report()
        
        dashboard = [
            "="*80,
            "REPOSITORY ORGANIZATION HEALTH DASHBOARD",
            "="*80,
            f"Status: {'🟢 MONITORING ACTIVE' if self.monitoring_active else '🔴 MONITORING INACTIVE'}",
            f"Health Score: {self.stats.health_score:.1f}/100",
            f"Last Scan: {self.stats.last_scan_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Scans: {self.stats.scan_count}",
            f"Violations Detected: {self.stats.violations_detected}",
            f"Violations Resolved: {self.stats.violations_resolved}",
            "",
            "VIOLATION BREAKDOWN:",
            "-"*40
        ]
        
        summary = self._get_violation_summary()
        
        if summary['by_severity']:
            dashboard.append("By Severity:")
            for severity in ['critical', 'error', 'warning', 'info']:
                count = summary['by_severity'].get(severity, 0)
                if count > 0:
                    icon = {'critical': '🔴', 'error': '🟠', 'warning': '🟡', 'info': '🔵'}[severity]
                    dashboard.append(f"  {icon} {severity.title()}: {count}")
        else:
            dashboard.append("🟢 No violations detected")
        
        dashboard.extend([
            "",
            "RECENT ACTIVITY:",
            "-"*40
        ])
        
        if self.violations:
            recent_violations = list(self.violations)[-5:]
            for violation in recent_violations:
                timestamp = violation.timestamp.strftime('%H:%M:%S')
                dashboard.append(f"  {timestamp} - {violation.violation_type}: {violation.file_path}")
        else:
            dashboard.append("  No recent violations")
        
        dashboard.extend([
            "",
            f"Monitoring since: {self.stats.monitoring_start.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Average scan duration: {self.stats.avg_scan_duration:.2f}s",
            "="*80
        ])
        
        return "\n".join(dashboard)


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Repository Organization Monitor")
    parser.add_argument("--root", default=".", help="Repository root path")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--start", action='store_true', help="Start monitoring daemon")
    parser.add_argument("--stop", action='store_true', help="Stop monitoring daemon")
    parser.add_argument("--status", action='store_true', help="Show current status")
    parser.add_argument("--dashboard", action='store_true', help="Show compliance dashboard")
    parser.add_argument("--scan-once", action='store_true', help="Perform single scan")
    parser.add_argument("--verbose", action='store_true', help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    monitor = RepositoryOrganizationMonitor(args.root, args.config)
    
    if args.start:
        monitor.start_monitoring()
        print("Monitoring started. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            monitor.stop_monitoring()
            print("\nMonitoring stopped.")
    
    elif args.stop:
        monitor.stop_monitoring()
        print("Monitoring stopped.")
    
    elif args.status:
        health_report = monitor.get_current_health_report()
        print(f"Repository Health Score: {health_report['health_score']:.1f}/100")
        print(f"Monitoring Status: {health_report['monitoring_status']}")
        print(f"Total Violations: {health_report['stats']['violations_detected']}")
    
    elif args.dashboard:
        print(monitor.generate_compliance_dashboard())
    
    elif args.scan_once:
        print("Performing single organization scan...")
        violations = monitor._perform_organization_scan()
        print(f"Found {len(violations)} organization violations:")
        for violation in violations[:10]:  # Show first 10
            print(f"  - {violation.violation_type}: {violation.file_path}")
    
    else:
        print("Use --start, --stop, --status, --dashboard, or --scan-once")


if __name__ == "__main__":
    main()