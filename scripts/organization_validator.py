#!/usr/bin/env python3
"""
Organization Validator for Issue #255 Stream C.

CI/CD integrated validation system for repository organization standards:
- Pre-commit hooks for organization validation
- Automated test suite for organization standards
- Integration with existing validation workflows
- Fail-fast validation for CI/CD pipelines
- Comprehensive reporting for compliance tracking

Building on proven infrastructure from Streams A & B and monitoring system.
"""

import json
import logging
import sys
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import tempfile
import os

# Import our existing proven infrastructure
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from repository_scanner import RepositoryScanner
    from safety_validator import SafetyValidator
    from directory_structure_analyzer import DirectoryStructureAnalyzer
    from repository_organization_monitor import RepositoryOrganizationMonitor
except ImportError as e:
    logging.error(f"Failed to import required modules: {e}")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Results of organization validation."""
    test_name: str
    passed: bool
    severity: str  # 'info', 'warning', 'error', 'critical'
    message: str
    details: Dict[str, Any] = None
    fix_suggestions: List[str] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}
        if self.fix_suggestions is None:
            self.fix_suggestions = []


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    timestamp: datetime
    repository_path: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    overall_status: str  # 'PASS', 'FAIL', 'WARNING'
    validation_results: List[ValidationResult]
    summary: Dict[str, Any]
    ci_exit_code: int


class OrganizationValidator:
    """CI/CD integrated organization validation system."""
    
    def __init__(self, root_path: str = ".", config_file: Optional[str] = None, strict_mode: bool = False):
        self.root_path = Path(root_path).resolve()
        self.strict_mode = strict_mode  # Fail on warnings in CI/CD
        self.config = self._load_validation_config(config_file)
        
        # Initialize core components
        self.scanner = RepositoryScanner(str(self.root_path))
        self.safety_validator = SafetyValidator(str(self.root_path))
        self.structure_analyzer = DirectoryStructureAnalyzer(str(self.root_path))
        self.monitor = RepositoryOrganizationMonitor(str(self.root_path))
        
        # Test registry
        self.validation_tests = {
            'root_directory_cleanliness': self._test_root_directory_cleanliness,
            'directory_naming_conventions': self._test_directory_naming_conventions,
            'file_organization': self._test_file_organization,
            'structure_compliance': self._test_structure_compliance,
            'naming_standards': self._test_naming_standards,
            'file_size_compliance': self._test_file_size_compliance,
            'archive_organization': self._test_archive_organization,
            'critical_file_protection': self._test_critical_file_protection,
            'git_integration_health': self._test_git_integration_health,
            'monitoring_readiness': self._test_monitoring_readiness
        }
        
        logger.info(f"Organization Validator initialized for: {self.root_path}")
        logger.info(f"Strict mode: {'ON' if self.strict_mode else 'OFF'}")

    def _load_validation_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load validation configuration."""
        default_config = {
            "validation_rules": {
                "max_root_files": 20,
                "max_root_scattered_files": 5,
                "max_malformed_directories": 0,
                "max_naming_violations": 0,
                "max_large_files_mb": 100,
                "required_directories": ["src", "scripts", "examples", "tests"],
                "forbidden_root_extensions": [".tmp", ".bak", ".orig"],
                "required_gitignore_patterns": ["*.pyc", "__pycache__", ".DS_Store"]
            },
            "ci_cd_settings": {
                "fail_on_warnings": False,
                "generate_junit_xml": True,
                "create_github_annotations": True,
                "report_output_file": "temp/validation_report.json"
            },
            "test_suites": {
                "pre_commit": [
                    "root_directory_cleanliness",
                    "critical_file_protection",
                    "naming_standards"
                ],
                "ci_validation": [
                    "root_directory_cleanliness",
                    "directory_naming_conventions", 
                    "file_organization",
                    "structure_compliance",
                    "naming_standards"
                ],
                "full_validation": "all"
            }
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file) as f:
                    user_config = json.load(f)
                    # Merge configs recursively
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

    def validate(self, test_suite: str = "ci_validation") -> ValidationReport:
        """Run validation test suite."""
        logger.info(f"Starting organization validation: {test_suite}")
        
        # Determine which tests to run
        if test_suite == "all":
            tests_to_run = list(self.validation_tests.keys())
        else:
            tests_to_run = self.config['test_suites'].get(test_suite, [])
            if not tests_to_run:
                raise ValueError(f"Unknown test suite: {test_suite}")
        
        # Run validation tests
        validation_results = []
        for test_name in tests_to_run:
            if test_name in self.validation_tests:
                logger.info(f"Running test: {test_name}")
                try:
                    result = self.validation_tests[test_name]()
                    validation_results.append(result)
                except Exception as e:
                    logger.error(f"Test {test_name} failed with exception: {e}")
                    validation_results.append(ValidationResult(
                        test_name=test_name,
                        passed=False,
                        severity="error",
                        message=f"Test execution failed: {e}"
                    ))
        
        # Compile report
        report = self._compile_validation_report(validation_results)
        
        # Save report
        self._save_validation_report(report)
        
        logger.info(f"Validation complete: {report.overall_status}")
        return report

    def _test_root_directory_cleanliness(self) -> ValidationResult:
        """Test that root directory is clean and organized."""
        try:
            scan_results = self.scanner.scan_repository()
            
            # Count scattered root files
            scattered_files = [f for f in scan_results['files'] 
                             if f.path.parent == Path('.') and f.subcategory == 'scattered_in_root']
            
            max_scattered = self.config['validation_rules']['max_root_scattered_files']
            
            if len(scattered_files) > max_scattered:
                return ValidationResult(
                    test_name="root_directory_cleanliness",
                    passed=False,
                    severity="error",
                    message=f"Root directory has {len(scattered_files)} scattered files (max: {max_scattered})",
                    details={
                        "scattered_files": [str(f.path) for f in scattered_files],
                        "threshold": max_scattered
                    },
                    fix_suggestions=[
                        "Run repository scanner to identify proper locations for scattered files",
                        "Use root_directory_organizer.py to move files to appropriate directories",
                        f"Move {len(scattered_files)} files to their designated locations"
                    ]
                )
            
            # Check total root file count
            root_files = [f for f in scan_results['files'] if f.path.parent == Path('.')]
            max_root_files = self.config['validation_rules']['max_root_files']
            
            if len(root_files) > max_root_files:
                return ValidationResult(
                    test_name="root_directory_cleanliness",
                    passed=False,
                    severity="warning",
                    message=f"Root directory has {len(root_files)} files (recommended max: {max_root_files})",
                    details={"root_file_count": len(root_files), "threshold": max_root_files},
                    fix_suggestions=[
                        "Review root files and move non-essential files to appropriate subdirectories"
                    ]
                )
            
            return ValidationResult(
                test_name="root_directory_cleanliness",
                passed=True,
                severity="info", 
                message=f"Root directory is clean: {len(root_files)} files, {len(scattered_files)} scattered"
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="root_directory_cleanliness",
                passed=False,
                severity="error",
                message=f"Failed to test root directory cleanliness: {e}"
            )

    def _test_directory_naming_conventions(self) -> ValidationResult:
        """Test directory naming conventions."""
        try:
            issues = self.structure_analyzer.analyze_directory_structure()
            
            malformed_dirs = [i for i in issues if i.issue_type in ['malformed_name', 'naming_convention']]
            max_malformed = self.config['validation_rules']['max_malformed_directories']
            
            if len(malformed_dirs) > max_malformed:
                return ValidationResult(
                    test_name="directory_naming_conventions",
                    passed=False,
                    severity="critical" if any(i.severity == 'critical' for i in malformed_dirs) else "error",
                    message=f"Found {len(malformed_dirs)} directory naming violations (max: {max_malformed})",
                    details={
                        "violations": [
                            {"path": str(i.path), "current": i.current_name, "suggested": i.suggested_name}
                            for i in malformed_dirs
                        ]
                    },
                    fix_suggestions=[
                        "Run directory_structure_standardizer.py to fix naming issues",
                        f"Rename {len(malformed_dirs)} directories to follow snake_case convention"
                    ]
                )
            
            return ValidationResult(
                test_name="directory_naming_conventions", 
                passed=True,
                severity="info",
                message="All directories follow naming conventions"
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="directory_naming_conventions",
                passed=False,
                severity="error",
                message=f"Failed to test directory naming: {e}"
            )

    def _test_file_organization(self) -> ValidationResult:
        """Test that files are properly organized."""
        try:
            scan_results = self.scanner.scan_repository()
            
            # Count mislocated files
            mislocated_files = [f for f in scan_results['files'] if f.subcategory == 'mislocated']
            
            if mislocated_files:
                return ValidationResult(
                    test_name="file_organization",
                    passed=False,
                    severity="warning",
                    message=f"Found {len(mislocated_files)} mislocated files",
                    details={
                        "mislocated_files": [
                            {"file": str(f.path), "current_location": str(f.path.parent), "target": f.target_location}
                            for f in mislocated_files
                        ]
                    },
                    fix_suggestions=[
                        "Use repository organizer to move files to appropriate locations",
                        "Review file categorization and update organization rules if needed"
                    ]
                )
            
            return ValidationResult(
                test_name="file_organization",
                passed=True,
                severity="info",
                message="Files are properly organized"
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="file_organization",
                passed=False,
                severity="error",
                message=f"Failed to test file organization: {e}"
            )

    def _test_structure_compliance(self) -> ValidationResult:
        """Test overall directory structure compliance."""
        try:
            required_dirs = self.config['validation_rules']['required_directories']
            missing_dirs = []
            
            for required_dir in required_dirs:
                dir_path = self.root_path / required_dir
                if not dir_path.exists():
                    missing_dirs.append(required_dir)
            
            if missing_dirs:
                return ValidationResult(
                    test_name="structure_compliance",
                    passed=False,
                    severity="warning",
                    message=f"Missing required directories: {', '.join(missing_dirs)}",
                    details={"missing_directories": missing_dirs},
                    fix_suggestions=[
                        f"Create missing directories: {', '.join(missing_dirs)}",
                        "mkdir -p " + " ".join(missing_dirs)
                    ]
                )
            
            return ValidationResult(
                test_name="structure_compliance",
                passed=True,
                severity="info",
                message="Directory structure complies with standards"
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="structure_compliance", 
                passed=False,
                severity="error",
                message=f"Failed to test structure compliance: {e}"
            )

    def _test_naming_standards(self) -> ValidationResult:
        """Test file naming standards compliance."""
        try:
            violations = []
            forbidden_extensions = self.config['validation_rules']['forbidden_root_extensions']
            
            # Check root directory for forbidden file extensions
            for item in self.root_path.iterdir():
                if item.is_file():
                    for forbidden_ext in forbidden_extensions:
                        if item.suffix == forbidden_ext:
                            violations.append(f"File with forbidden extension in root: {item.name}")
            
            max_violations = self.config['validation_rules']['max_naming_violations']
            
            if len(violations) > max_violations:
                return ValidationResult(
                    test_name="naming_standards",
                    passed=False,
                    severity="error",
                    message=f"Found {len(violations)} naming standard violations (max: {max_violations})",
                    details={"violations": violations},
                    fix_suggestions=[
                        "Remove or relocate files with forbidden extensions",
                        "Review and update file naming conventions"
                    ]
                )
            
            return ValidationResult(
                test_name="naming_standards",
                passed=True,
                severity="info",
                message="File naming standards are compliant"
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="naming_standards",
                passed=False,
                severity="error",
                message=f"Failed to test naming standards: {e}"
            )

    def _test_file_size_compliance(self) -> ValidationResult:
        """Test file size compliance."""
        try:
            large_files = []
            max_size_mb = self.config['validation_rules']['max_large_files_mb']
            max_size_bytes = max_size_mb * 1024 * 1024
            
            # Scan for large files (excluding certain directories)
            exclude_dirs = {'.git', '__pycache__', 'node_modules', 'venv'}
            
            for file_path in self.root_path.rglob('*'):
                if file_path.is_file():
                    # Skip excluded directories
                    if any(part in exclude_dirs for part in file_path.parts):
                        continue
                    
                    try:
                        size = file_path.stat().st_size
                        if size > max_size_bytes:
                            size_mb = size / (1024 * 1024)
                            large_files.append({
                                "file": str(file_path.relative_to(self.root_path)),
                                "size_mb": round(size_mb, 1)
                            })
                    except Exception:
                        continue
            
            if large_files:
                total_size = sum(f["size_mb"] for f in large_files)
                return ValidationResult(
                    test_name="file_size_compliance",
                    passed=False,
                    severity="warning",
                    message=f"Found {len(large_files)} large files totaling {total_size:.1f}MB",
                    details={"large_files": large_files, "threshold_mb": max_size_mb},
                    fix_suggestions=[
                        "Review large files for archiving or compression opportunities",
                        "Consider using Git LFS for large binary files",
                        "Move large files to appropriate storage locations"
                    ]
                )
            
            return ValidationResult(
                test_name="file_size_compliance",
                passed=True,
                severity="info", 
                message="File sizes are within acceptable limits"
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="file_size_compliance",
                passed=False,
                severity="error",
                message=f"Failed to test file size compliance: {e}"
            )

    def _test_archive_organization(self) -> ValidationResult:
        """Test that archive directories are properly organized."""
        try:
            issues = []
            
            # Check if timestamped files are in archive directories
            for file_path in self.root_path.rglob('*'):
                if file_path.is_file():
                    # Simple check for timestamped files
                    if any(char.isdigit() for char in file_path.name) and \
                       any(pattern in file_path.name.lower() for pattern in ['2024', '2025', 'report', 'output']):
                        
                        # Check if it's in an archive directory
                        parent_names = [p.name.lower() for p in file_path.parents]
                        if not any(archive_name in parent_names for archive_name in ['archive', 'history', 'old']):
                            issues.append(str(file_path.relative_to(self.root_path)))
            
            if issues and len(issues) > 10:  # Only flag if many timestamped files are unarchived
                return ValidationResult(
                    test_name="archive_organization",
                    passed=False,
                    severity="info",
                    message=f"Found {len(issues)} timestamped files that could be archived",
                    details={"unarchived_files": issues[:20]},  # Show first 20
                    fix_suggestions=[
                        "Create archive subdirectories for old timestamped files",
                        "Move historical reports and outputs to archive directories"
                    ]
                )
            
            return ValidationResult(
                test_name="archive_organization",
                passed=True,
                severity="info",
                message="Archive organization is acceptable"
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="archive_organization",
                passed=False,
                severity="error",
                message=f"Failed to test archive organization: {e}"
            )

    def _test_critical_file_protection(self) -> ValidationResult:
        """Test that critical files are protected and in correct locations."""
        try:
            critical_files = ['pyproject.toml', 'README.md', '.gitignore', 'LICENSE']
            issues = []
            
            for critical_file in critical_files:
                file_path = self.root_path / critical_file
                if not file_path.exists():
                    issues.append(f"Missing critical file: {critical_file}")
                elif critical_file != 'LICENSE' and file_path.parent != self.root_path:
                    issues.append(f"Critical file not in root: {critical_file}")
            
            if issues:
                return ValidationResult(
                    test_name="critical_file_protection",
                    passed=False,
                    severity="error",
                    message=f"Critical file protection issues: {len(issues)} problems",
                    details={"issues": issues},
                    fix_suggestions=[
                        "Ensure all critical files are present in root directory",
                        "Create missing critical files as needed"
                    ]
                )
            
            return ValidationResult(
                test_name="critical_file_protection", 
                passed=True,
                severity="info",
                message="Critical files are properly protected"
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="critical_file_protection",
                passed=False,
                severity="error",
                message=f"Failed to test critical file protection: {e}"
            )

    def _test_git_integration_health(self) -> ValidationResult:
        """Test git repository health for organization tracking."""
        try:
            # Check if we're in a git repository
            try:
                result = subprocess.run(['git', 'rev-parse', '--git-dir'], 
                                      capture_output=True, text=True, cwd=self.root_path)
                if result.returncode != 0:
                    return ValidationResult(
                        test_name="git_integration_health",
                        passed=True,
                        severity="info",
                        message="Not a git repository - git integration tests skipped"
                    )
            except FileNotFoundError:
                return ValidationResult(
                    test_name="git_integration_health",
                    passed=True,
                    severity="info", 
                    message="Git not available - git integration tests skipped"
                )
            
            issues = []
            
            # Check for .gitignore patterns
            gitignore_path = self.root_path / '.gitignore'
            if gitignore_path.exists():
                required_patterns = self.config['validation_rules']['required_gitignore_patterns']
                with open(gitignore_path) as f:
                    gitignore_content = f.read()
                
                for pattern in required_patterns:
                    if pattern not in gitignore_content:
                        issues.append(f"Missing .gitignore pattern: {pattern}")
            else:
                issues.append("Missing .gitignore file")
            
            # Check for excessive uncommitted files
            result = subprocess.run(['git', 'status', '--porcelain'], 
                                  capture_output=True, text=True, cwd=self.root_path)
            uncommitted_files = result.stdout.strip().split('\n') if result.stdout.strip() else []
            
            if len(uncommitted_files) > 50:
                issues.append(f"Excessive uncommitted files: {len(uncommitted_files)}")
            
            if issues:
                return ValidationResult(
                    test_name="git_integration_health",
                    passed=False,
                    severity="warning",
                    message=f"Git integration issues found: {len(issues)} problems",
                    details={"issues": issues},
                    fix_suggestions=[
                        "Update .gitignore with required patterns",
                        "Commit or stash uncommitted changes",
                        "Review git repository configuration"
                    ]
                )
            
            return ValidationResult(
                test_name="git_integration_health",
                passed=True,
                severity="info",
                message="Git integration is healthy"
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="git_integration_health",
                passed=False,
                severity="warning",
                message=f"Could not test git integration health: {e}"
            )

    def _test_monitoring_readiness(self) -> ValidationResult:
        """Test that monitoring system can operate effectively."""
        try:
            # Test monitor initialization
            monitor = RepositoryOrganizationMonitor(str(self.root_path))
            
            # Test basic monitoring capabilities
            health_report = monitor.get_current_health_report()
            
            # Check if monitoring directories exist
            monitor_dir = self.root_path / "temp" / "monitoring"
            if not monitor_dir.exists():
                return ValidationResult(
                    test_name="monitoring_readiness",
                    passed=False,
                    severity="warning",
                    message="Monitoring directories not initialized",
                    fix_suggestions=[
                        "Initialize monitoring system: python scripts/repository_organization_monitor.py --status"
                    ]
                )
            
            # Check health score
            if health_report['health_score'] < 80:
                return ValidationResult(
                    test_name="monitoring_readiness",
                    passed=False,
                    severity="warning",
                    message=f"Low repository health score: {health_report['health_score']:.1f}/100",
                    details={"health_score": health_report['health_score']},
                    fix_suggestions=[
                        "Address organization violations to improve health score",
                        "Run full repository scan to identify issues"
                    ]
                )
            
            return ValidationResult(
                test_name="monitoring_readiness",
                passed=True,
                severity="info",
                message=f"Monitoring system ready, health score: {health_report['health_score']:.1f}/100"
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="monitoring_readiness",
                passed=False,
                severity="error",
                message=f"Failed to test monitoring readiness: {e}"
            )

    def _compile_validation_report(self, validation_results: List[ValidationResult]) -> ValidationReport:
        """Compile comprehensive validation report."""
        total_tests = len(validation_results)
        passed_tests = sum(1 for r in validation_results if r.passed)
        failed_tests = total_tests - passed_tests
        
        # Determine overall status
        has_critical = any(r.severity == 'critical' and not r.passed for r in validation_results)
        has_errors = any(r.severity == 'error' and not r.passed for r in validation_results) 
        has_warnings = any(r.severity == 'warning' and not r.passed for r in validation_results)
        
        if has_critical or has_errors:
            overall_status = "FAIL"
            ci_exit_code = 1
        elif has_warnings and self.strict_mode:
            overall_status = "FAIL"
            ci_exit_code = 1
        elif has_warnings:
            overall_status = "WARNING"
            ci_exit_code = 0
        else:
            overall_status = "PASS"
            ci_exit_code = 0
        
        # Create summary
        summary = {
            "by_severity": {
                "critical": len([r for r in validation_results if r.severity == 'critical' and not r.passed]),
                "error": len([r for r in validation_results if r.severity == 'error' and not r.passed]),
                "warning": len([r for r in validation_results if r.severity == 'warning' and not r.passed]),
                "info": len([r for r in validation_results if r.severity == 'info'])
            },
            "test_coverage": {
                "root_cleanliness": any(r.test_name == 'root_directory_cleanliness' for r in validation_results),
                "naming_conventions": any(r.test_name == 'directory_naming_conventions' for r in validation_results),
                "file_organization": any(r.test_name == 'file_organization' for r in validation_results),
                "structure_compliance": any(r.test_name == 'structure_compliance' for r in validation_results)
            }
        }
        
        return ValidationReport(
            timestamp=datetime.now(),
            repository_path=str(self.root_path),
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            overall_status=overall_status,
            validation_results=validation_results,
            summary=summary,
            ci_exit_code=ci_exit_code
        )

    def _save_validation_report(self, report: ValidationReport):
        """Save validation report to file."""
        output_file = self.config['ci_cd_settings']['report_output_file']
        
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to dict for JSON serialization
            report_dict = asdict(report)
            
            with open(output_path, 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)
            
            logger.info(f"Validation report saved: {output_path}")

    def create_pre_commit_hook(self, hook_path: Optional[str] = None) -> Path:
        """Create pre-commit hook for organization validation."""
        if hook_path is None:
            git_dir = self.root_path / '.git'
            if git_dir.exists():
                hook_path = git_dir / 'hooks' / 'pre-commit'
            else:
                hook_path = self.root_path / 'pre-commit-organization-validator'
        else:
            hook_path = Path(hook_path)
        
        hook_script = f'''#!/bin/bash
# Organization validation pre-commit hook
# Generated by OrganizationValidator

echo "Running repository organization validation..."

python "{self.root_path / 'scripts/organization_validator.py'}" --suite pre_commit --strict

if [ $? -ne 0 ]; then
    echo "❌ Organization validation failed!"
    echo "Please fix organization issues before committing."
    echo "Run: python scripts/organization_validator.py --suite pre_commit"
    exit 1
fi

echo "✅ Organization validation passed!"
exit 0
'''
        
        hook_path.parent.mkdir(parents=True, exist_ok=True)
        with open(hook_path, 'w') as f:
            f.write(hook_script)
        
        # Make executable
        hook_path.chmod(0o755)
        
        logger.info(f"Pre-commit hook created: {hook_path}")
        return hook_path

    def generate_junit_xml(self, report: ValidationReport, output_path: Optional[str] = None) -> Path:
        """Generate JUnit XML report for CI/CD integration."""
        if output_path is None:
            output_path = self.root_path / "temp" / "junit_validation_results.xml"
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        xml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<testsuite name="OrganizationValidation" tests="{report.total_tests}" '
            f'failures="{report.failed_tests}" time="0" timestamp="{report.timestamp.isoformat()}">',
        ]
        
        for result in report.validation_results:
            if result.passed:
                xml_lines.append(f'  <testcase name="{result.test_name}" classname="OrganizationValidator"/>')
            else:
                xml_lines.extend([
                    f'  <testcase name="{result.test_name}" classname="OrganizationValidator">',
                    f'    <failure message="{result.message}" type="{result.severity}">',
                    f'      {result.message}',
                    '    </failure>',
                    '  </testcase>'
                ])
        
        xml_lines.append('</testsuite>')
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(xml_lines))
        
        logger.info(f"JUnit XML report saved: {output_path}")
        return output_path


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Repository Organization Validator")
    parser.add_argument("--root", default=".", help="Repository root path")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--suite", default="ci_validation", 
                       choices=["pre_commit", "ci_validation", "full_validation", "all"],
                       help="Test suite to run")
    parser.add_argument("--strict", action='store_true', help="Fail on warnings (CI mode)")
    parser.add_argument("--create-hook", action='store_true', help="Create pre-commit hook")
    parser.add_argument("--junit-xml", help="Generate JUnit XML report to specified path")
    parser.add_argument("--verbose", action='store_true', help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    validator = OrganizationValidator(args.root, args.config, args.strict)
    
    if args.create_hook:
        hook_path = validator.create_pre_commit_hook()
        print(f"Pre-commit hook created: {hook_path}")
        return 0
    
    # Run validation
    report = validator.validate(args.suite)
    
    # Generate JUnit XML if requested
    if args.junit_xml:
        validator.generate_junit_xml(report, args.junit_xml)
    
    # Print results
    print(f"\n{'='*80}")
    print(f"ORGANIZATION VALIDATION RESULTS")
    print(f"{'='*80}")
    print(f"Overall Status: {report.overall_status}")
    print(f"Tests: {report.passed_tests}/{report.total_tests} passed")
    
    if report.failed_tests > 0:
        print(f"\nFAILED TESTS:")
        print("-" * 40)
        for result in report.validation_results:
            if not result.passed:
                print(f"❌ {result.test_name}: {result.message}")
                if result.fix_suggestions:
                    for suggestion in result.fix_suggestions[:2]:  # Show first 2 suggestions
                        print(f"   💡 {suggestion}")
    
    # Summary by severity
    if any(report.summary['by_severity'].values()):
        print(f"\nISSUES BY SEVERITY:")
        print("-" * 40)
        for severity in ['critical', 'error', 'warning', 'info']:
            count = report.summary['by_severity'][severity]
            if count > 0:
                icons = {'critical': '🔴', 'error': '🟠', 'warning': '🟡', 'info': '🔵'}
                print(f"{icons[severity]} {severity.title()}: {count}")
    
    print(f"\nValidation completed: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Report saved to: {validator.config['ci_cd_settings']['report_output_file']}")
    print(f"{'='*80}")
    
    return report.ci_exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)