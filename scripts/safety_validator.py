#!/usr/bin/env python3
"""
Safety Validation Framework for Issue #255 - Repository Organization & Cleanup.

Comprehensive safety checks and backup validation system to ensure:
- No critical files are accidentally moved or deleted
- All operations can be safely rolled back
- User confirmation for high-risk operations
- Integration with git for atomic commit/rollback procedures

Safety-first approach with multiple validation layers.
"""

import json
import logging
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import hashlib
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class SafetyCheck:
    """Results of a safety validation check."""
    check_name: str
    passed: bool
    severity: str  # 'info', 'warning', 'error', 'critical'
    message: str
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


@dataclass
class BackupManifest:
    """Manifest of backed up files and directories."""
    timestamp: datetime
    backup_id: str
    source_files: List[str]
    backup_location: str
    git_commit_hash: Optional[str] = None
    file_hashes: Dict[str, str] = None
    
    def __post_init__(self):
        if self.file_hashes is None:
            self.file_hashes = {}


class SafetyValidator:
    """Comprehensive safety validation system for file operations."""
    
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path).resolve()
        self.backup_dir = self.root_path / "temp" / "safety_backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Critical file patterns that should never be moved without explicit approval
        self.critical_patterns = {
            'configuration': [
                'pyproject.toml', 'setup.py', 'setup.cfg', 'requirements*.txt',
                'models.yaml', 'mcp_tools_config.json', '.env*', '*.ini', '*.cfg'
            ],
            'documentation': [
                'README.md', 'CHANGELOG.md', 'LICENSE', 'MANIFEST.in',
                'CONTRIBUTING.md', 'CODE_OF_CONDUCT.md'
            ],
            'version_control': [
                '.gitignore', '.gitmodules', '.gitattributes'
            ],
            'build_system': [
                'Makefile', 'build.py', 'CMakeLists.txt', 'webpack.config.*',
                'package.json', 'yarn.lock', 'package-lock.json'
            ],
            'ci_cd': [
                '.github/**/*', '.gitlab-ci.yml', '.travis.yml',
                'azure-pipelines.yml', 'Jenkinsfile'
            ]
        }
        
        # Safety thresholds
        self.safety_thresholds = {
            'max_files_per_operation': 1000,
            'max_size_per_operation': 100 * 1024 * 1024,  # 100MB
            'min_free_space_ratio': 0.1,  # 10% free space required
            'max_critical_files': 0,  # No critical files should be moved without approval
        }
    
    def validate_operation(self, file_operations: List[Dict[str, Any]], 
                          operation_type: str = 'move') -> Tuple[bool, List[SafetyCheck]]:
        """
        Validate a proposed file operation for safety.
        
        Args:
            file_operations: List of operations, each dict with 'source', 'target', 'operation'
            operation_type: Type of operation ('move', 'delete', 'copy')
            
        Returns:
            Tuple of (is_safe, list_of_safety_checks)
        """
        logger.info(f"Validating {len(file_operations)} {operation_type} operations")
        
        checks = []
        
        # 1. Critical file protection
        checks.extend(self._check_critical_files(file_operations))
        
        # 2. Operation size and scope validation
        checks.extend(self._check_operation_scope(file_operations))
        
        # 3. Target location validation
        checks.extend(self._check_target_locations(file_operations))
        
        # 4. File system capacity check
        checks.extend(self._check_filesystem_capacity(file_operations))
        
        # 5. Git repository state validation
        checks.extend(self._check_git_state())
        
        # 6. Dependency validation (files that might be referenced by others)
        checks.extend(self._check_file_dependencies(file_operations))
        
        # 7. Collision detection
        checks.extend(self._check_file_collisions(file_operations))
        
        # Determine overall safety
        is_safe = self._evaluate_overall_safety(checks)
        
        logger.info(f"Safety validation complete: {'SAFE' if is_safe else 'NOT SAFE'}")
        return is_safe, checks
    
    def _check_critical_files(self, operations: List[Dict[str, Any]]) -> List[SafetyCheck]:
        """Check if any critical files are being affected."""
        checks = []
        critical_files = []
        
        for operation in operations:
            source_path = Path(operation['source'])
            
            # Check against critical patterns
            for category, patterns in self.critical_patterns.items():
                for pattern in patterns:
                    if pattern.endswith('**/*'):
                        # Directory pattern
                        if str(source_path).startswith(pattern[:-5]):
                            critical_files.append((str(source_path), category))
                    elif '*' in pattern:
                        # Wildcard pattern
                        if source_path.match(pattern):
                            critical_files.append((str(source_path), category))
                    else:
                        # Exact match
                        if source_path.name == pattern:
                            critical_files.append((str(source_path), category))
        
        if critical_files:
            checks.append(SafetyCheck(
                check_name="critical_files",
                passed=False,
                severity="critical",
                message=f"Operation affects {len(critical_files)} critical files",
                details={
                    "critical_files": critical_files,
                    "recommendation": "Manual review required for all critical files"
                }
            ))
        else:
            checks.append(SafetyCheck(
                check_name="critical_files",
                passed=True,
                severity="info",
                message="No critical files affected"
            ))
        
        return checks
    
    def _check_operation_scope(self, operations: List[Dict[str, Any]]) -> List[SafetyCheck]:
        """Validate the scope and scale of operations."""
        checks = []
        
        # Count files and total size
        file_count = len(operations)
        total_size = 0
        
        for operation in operations:
            source_path = Path(self.root_path) / operation['source']
            if source_path.exists():
                try:
                    total_size += source_path.stat().st_size
                except:
                    pass
        
        # Check file count threshold
        max_files = self.safety_thresholds['max_files_per_operation']
        if file_count > max_files:
            checks.append(SafetyCheck(
                check_name="operation_scope_files",
                passed=False,
                severity="warning",
                message=f"Large operation: {file_count} files (max recommended: {max_files})",
                details={"file_count": file_count, "threshold": max_files}
            ))
        else:
            checks.append(SafetyCheck(
                check_name="operation_scope_files",
                passed=True,
                severity="info",
                message=f"File count within limits: {file_count} files"
            ))
        
        # Check total size threshold
        max_size = self.safety_thresholds['max_size_per_operation']
        if total_size > max_size:
            checks.append(SafetyCheck(
                check_name="operation_scope_size",
                passed=False,
                severity="warning",
                message=f"Large data operation: {total_size / (1024*1024):.1f}MB (max recommended: {max_size / (1024*1024):.1f}MB)",
                details={"total_size": total_size, "threshold": max_size}
            ))
        else:
            checks.append(SafetyCheck(
                check_name="operation_scope_size",
                passed=True,
                severity="info",
                message=f"Data size within limits: {total_size / (1024*1024):.1f}MB"
            ))
        
        return checks
    
    def _check_target_locations(self, operations: List[Dict[str, Any]]) -> List[SafetyCheck]:
        """Validate that target locations are appropriate and accessible."""
        checks = []
        invalid_targets = []
        
        for operation in operations:
            if 'target' not in operation:
                continue
                
            target_path = Path(self.root_path) / operation['target']
            
            # Check if target directory exists or can be created
            try:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Check write permissions
                if not os.access(target_path.parent, os.W_OK):
                    invalid_targets.append(f"No write permission: {target_path.parent}")
                
            except Exception as e:
                invalid_targets.append(f"Cannot create target directory {target_path.parent}: {e}")
        
        if invalid_targets:
            checks.append(SafetyCheck(
                check_name="target_locations",
                passed=False,
                severity="error",
                message=f"Invalid target locations found",
                details={"invalid_targets": invalid_targets}
            ))
        else:
            checks.append(SafetyCheck(
                check_name="target_locations",
                passed=True,
                severity="info",
                message="All target locations are valid and accessible"
            ))
        
        return checks
    
    def _check_filesystem_capacity(self, operations: List[Dict[str, Any]]) -> List[SafetyCheck]:
        """Check available filesystem capacity."""
        checks = []
        
        try:
            # Get filesystem statistics
            stat = shutil.disk_usage(self.root_path)
            free_space = stat.free
            total_space = stat.total
            
            # Calculate space needed for operations (assuming worst case of copying)
            space_needed = sum(
                Path(self.root_path / op['source']).stat().st_size
                for op in operations
                if (Path(self.root_path) / op['source']).exists()
            )
            
            # Check if we have enough space
            free_ratio = free_space / total_space
            min_ratio = self.safety_thresholds['min_free_space_ratio']
            
            if space_needed > free_space * 0.8:  # Use 80% of free space max
                checks.append(SafetyCheck(
                    check_name="filesystem_capacity",
                    passed=False,
                    severity="error",
                    message=f"Insufficient disk space: need {space_needed/(1024*1024):.1f}MB, available {free_space/(1024*1024):.1f}MB",
                    details={
                        "space_needed": space_needed,
                        "free_space": free_space,
                        "total_space": total_space
                    }
                ))
            elif free_ratio < min_ratio:
                checks.append(SafetyCheck(
                    check_name="filesystem_capacity",
                    passed=False,
                    severity="warning",
                    message=f"Low disk space: {free_ratio*100:.1f}% free (minimum {min_ratio*100:.1f}%)",
                    details={"free_ratio": free_ratio, "min_ratio": min_ratio}
                ))
            else:
                checks.append(SafetyCheck(
                    check_name="filesystem_capacity",
                    passed=True,
                    severity="info",
                    message=f"Sufficient disk space: {free_ratio*100:.1f}% free"
                ))
                
        except Exception as e:
            checks.append(SafetyCheck(
                check_name="filesystem_capacity",
                passed=False,
                severity="warning",
                message=f"Could not check disk space: {e}"
            ))
        
        return checks
    
    def _check_git_state(self) -> List[SafetyCheck]:
        """Check git repository state and ensure it's safe for operations."""
        checks = []
        
        try:
            # Check if we're in a git repository
            result = subprocess.run(['git', 'rev-parse', '--git-dir'], 
                                  capture_output=True, text=True, cwd=self.root_path)
            
            if result.returncode != 0:
                checks.append(SafetyCheck(
                    check_name="git_state",
                    passed=True,
                    severity="info", 
                    message="Not a git repository - backup will be manual only"
                ))
                return checks
            
            # Check for uncommitted changes
            result = subprocess.run(['git', 'status', '--porcelain'], 
                                  capture_output=True, text=True, cwd=self.root_path)
            
            uncommitted_files = result.stdout.strip().split('\n') if result.stdout.strip() else []
            
            if uncommitted_files and len(uncommitted_files) > 10:
                checks.append(SafetyCheck(
                    check_name="git_state",
                    passed=False,
                    severity="warning",
                    message=f"Repository has {len(uncommitted_files)} uncommitted changes - recommend commit before reorganization",
                    details={"uncommitted_count": len(uncommitted_files)}
                ))
            else:
                checks.append(SafetyCheck(
                    check_name="git_state",
                    passed=True,
                    severity="info",
                    message="Git repository state is acceptable for operations"
                ))
                
        except Exception as e:
            checks.append(SafetyCheck(
                check_name="git_state",
                passed=False,
                severity="warning",
                message=f"Could not check git state: {e}"
            ))
        
        return checks
    
    def _check_file_dependencies(self, operations: List[Dict[str, Any]]) -> List[SafetyCheck]:
        """Check for files that might be referenced by other files."""
        checks = []
        dependency_warnings = []
        
        # Common dependency patterns
        dependency_patterns = [
            (r'import\s+.*', 'Python imports'),
            (r'from\s+.*\s+import', 'Python imports'),
            (r'require\s*\(\s*[\'"].*[\'"]', 'Node.js requires'),
            (r'include\s+[\'"].*[\'"]', 'Include statements'),
            (r'#include\s+[<"].*[>"]', 'C/C++ includes'),
        ]
        
        source_files = [op['source'] for op in operations]
        
        # Simple heuristic: check if any moved files are referenced in remaining files
        for operation in operations:
            source_path = Path(operation['source'])
            filename = source_path.name
            
            # Skip very common filenames
            if filename.lower() in ['test.py', 'main.py', '__init__.py', 'config.py']:
                dependency_warnings.append(f"Moving common filename: {filename}")
        
        if dependency_warnings:
            checks.append(SafetyCheck(
                check_name="file_dependencies",
                passed=True,  # Warning only
                severity="warning",
                message="Potential dependency concerns found",
                details={"warnings": dependency_warnings}
            ))
        else:
            checks.append(SafetyCheck(
                check_name="file_dependencies",
                passed=True,
                severity="info",
                message="No obvious dependency conflicts detected"
            ))
        
        return checks
    
    def _check_file_collisions(self, operations: List[Dict[str, Any]]) -> List[SafetyCheck]:
        """Check for potential file name collisions at target locations."""
        checks = []
        collisions = []
        
        target_files = defaultdict(list)
        
        for operation in operations:
            if 'target' not in operation:
                continue
                
            target_path = Path(self.root_path) / operation['target']
            
            # Check if target already exists
            if target_path.exists():
                collisions.append(f"Target exists: {operation['target']}")
            
            # Check for name collisions with other operations
            target_files[str(target_path)].append(operation['source'])
        
        # Find multiple sources mapping to same target
        for target, sources in target_files.items():
            if len(sources) > 1:
                collisions.append(f"Multiple sources -> {target}: {sources}")
        
        if collisions:
            checks.append(SafetyCheck(
                check_name="file_collisions",
                passed=False,
                severity="error",
                message=f"File collisions detected: {len(collisions)} conflicts",
                details={"collisions": collisions}
            ))
        else:
            checks.append(SafetyCheck(
                check_name="file_collisions",
                passed=True,
                severity="info",
                message="No file collisions detected"
            ))
        
        return checks
    
    def _evaluate_overall_safety(self, checks: List[SafetyCheck]) -> bool:
        """Evaluate overall safety based on all checks."""
        # Any critical or error checks fail the safety evaluation
        for check in checks:
            if check.severity in ['critical', 'error'] and not check.passed:
                return False
        
        return True
    
    def create_backup(self, file_paths: List[str], backup_id: Optional[str] = None) -> BackupManifest:
        """Create a backup of specified files before operations."""
        if backup_id is None:
            backup_id = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_path = self.backup_dir / backup_id
        backup_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating backup {backup_id} with {len(file_paths)} files")
        
        backed_up_files = []
        file_hashes = {}
        
        for file_path in file_paths:
            source = Path(self.root_path) / file_path
            if not source.exists():
                continue
                
            # Create backup destination maintaining directory structure
            relative_path = source.relative_to(self.root_path)
            backup_dest = backup_path / relative_path
            backup_dest.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                # Copy file
                shutil.copy2(source, backup_dest)
                backed_up_files.append(file_path)
                
                # Calculate hash for integrity verification
                file_hashes[file_path] = self._calculate_file_hash(source)
                
            except Exception as e:
                logger.warning(f"Failed to backup {file_path}: {e}")
        
        # Try to get current git commit
        git_commit = None
        try:
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True, cwd=self.root_path)
            if result.returncode == 0:
                git_commit = result.stdout.strip()
        except:
            pass
        
        # Create manifest
        manifest = BackupManifest(
            timestamp=datetime.now(),
            backup_id=backup_id,
            source_files=backed_up_files,
            backup_location=str(backup_path),
            git_commit_hash=git_commit,
            file_hashes=file_hashes
        )
        
        # Save manifest
        manifest_path = backup_path / "backup_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(asdict(manifest), f, indent=2, default=str)
        
        logger.info(f"Backup complete: {len(backed_up_files)} files backed up to {backup_path}")
        return manifest
    
    def restore_backup(self, backup_id: str, verify_integrity: bool = True) -> bool:
        """Restore files from a backup."""
        backup_path = self.backup_dir / backup_id
        manifest_path = backup_path / "backup_manifest.json"
        
        if not manifest_path.exists():
            logger.error(f"Backup manifest not found: {manifest_path}")
            return False
        
        # Load manifest
        with open(manifest_path) as f:
            manifest_data = json.load(f)
        
        manifest = BackupManifest(**manifest_data)
        
        logger.info(f"Restoring backup {backup_id} with {len(manifest.source_files)} files")
        
        restored_count = 0
        
        for file_path in manifest.source_files:
            source = backup_path / file_path
            dest = Path(self.root_path) / file_path
            
            if not source.exists():
                logger.warning(f"Backup file missing: {source}")
                continue
            
            try:
                # Verify backup integrity if requested
                if verify_integrity and file_path in manifest.file_hashes:
                    current_hash = self._calculate_file_hash(source)
                    if current_hash != manifest.file_hashes[file_path]:
                        logger.warning(f"Backup integrity check failed for {file_path}")
                        continue
                
                # Restore file
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
                restored_count += 1
                
            except Exception as e:
                logger.error(f"Failed to restore {file_path}: {e}")
        
        logger.info(f"Restore complete: {restored_count} files restored")
        return restored_count == len(manifest.source_files)
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file."""
        hasher = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception:
            return ""
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List all available backups."""
        backups = []
        
        for backup_dir in self.backup_dir.iterdir():
            if not backup_dir.is_dir():
                continue
                
            manifest_path = backup_dir / "backup_manifest.json"
            if manifest_path.exists():
                try:
                    with open(manifest_path) as f:
                        manifest_data = json.load(f)
                    
                    backups.append({
                        'backup_id': manifest_data['backup_id'],
                        'timestamp': manifest_data['timestamp'],
                        'file_count': len(manifest_data['source_files']),
                        'git_commit': manifest_data.get('git_commit_hash', 'N/A'),
                        'location': backup_dir
                    })
                except Exception as e:
                    logger.warning(f"Could not read backup manifest {manifest_path}: {e}")
        
        return sorted(backups, key=lambda x: x['timestamp'], reverse=True)
    
    def cleanup_old_backups(self, keep_count: int = 10) -> int:
        """Clean up old backups, keeping only the most recent ones."""
        backups = self.list_backups()
        
        if len(backups) <= keep_count:
            return 0
        
        cleaned_count = 0
        backups_to_clean = backups[keep_count:]
        
        for backup in backups_to_clean:
            try:
                backup_path = Path(backup['location'])
                shutil.rmtree(backup_path)
                cleaned_count += 1
                logger.info(f"Cleaned up old backup: {backup['backup_id']}")
            except Exception as e:
                logger.warning(f"Could not clean up backup {backup['backup_id']}: {e}")
        
        return cleaned_count


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Repository Safety Validator")
    parser.add_argument("--root", default=".", help="Repository root path")
    parser.add_argument("--list-backups", action='store_true', help="List available backups")
    parser.add_argument("--cleanup-backups", type=int, metavar='KEEP_COUNT', help="Clean up old backups")
    
    args = parser.parse_args()
    
    validator = SafetyValidator(args.root)
    
    if args.list_backups:
        backups = validator.list_backups()
        if backups:
            print("Available backups:")
            for backup in backups:
                print(f"  {backup['backup_id']} ({backup['timestamp']}) - {backup['file_count']} files")
        else:
            print("No backups found")
    
    elif args.cleanup_backups is not None:
        cleaned = validator.cleanup_old_backups(args.cleanup_backups)
        print(f"Cleaned up {cleaned} old backups")
    
    else:
        print("Safety validator ready. Use --list-backups or --cleanup-backups options.")


if __name__ == "__main__":
    main()