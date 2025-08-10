"""Intelligent Dependency Manager - Issue #206 Task 1.3

Automatic dependency detection, validation, and secure installation with
vulnerability scanning, integrity checking, and secure package management.
"""

import ast
import re
import logging
import hashlib
import time
import asyncio
import json
import tempfile
import shutil
import subprocess
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import aiofiles
import aiohttp
from packaging import version as pkg_version

from .policy_engine import Vulnerability, ThreatLevel
from .docker_manager import SecureContainer

logger = logging.getLogger(__name__)


class DependencySource(Enum):
    """Sources for dependency detection."""
    IMPORTS = "imports"
    REQUIREMENTS_TXT = "requirements_txt"
    PACKAGE_JSON = "package_json" 
    SETUP_PY = "setup_py"
    PYPROJECT_TOML = "pyproject_toml"
    MANUAL = "manual"


class PackageEcosystem(Enum):
    """Package ecosystem types."""
    PYPI = "pypi"
    NPM = "npm"
    CONDA = "conda"
    APT = "apt"
    YUM = "yum"
    UNKNOWN = "unknown"


class InstallationStatus(Enum):
    """Package installation status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


@dataclass
class PackageInfo:
    """Information about a package dependency."""
    name: str
    version: Optional[str] = None
    ecosystem: PackageEcosystem = PackageEcosystem.PYPI
    source: DependencySource = DependencySource.MANUAL
    required: bool = True
    extras: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Normalize package name."""
        self.name = self.name.lower().replace('_', '-')
    
    @property
    def full_name(self) -> str:
        """Get full package name with version."""
        if self.version:
            return f"{self.name}=={self.version}"
        return self.name
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'name': self.name,
            'version': self.version,
            'ecosystem': self.ecosystem.value,
            'source': self.source.value,
            'required': self.required,
            'extras': self.extras,
            'constraints': self.constraints,
            'full_name': self.full_name
        }


@dataclass
class DependencyAnalysis:
    """Results of dependency analysis."""
    direct_dependencies: List[PackageInfo] = field(default_factory=list)
    transitive_dependencies: List[PackageInfo] = field(default_factory=list)
    vulnerabilities: List[Vulnerability] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    security_score: float = 10.0
    installation_plan: List[str] = field(default_factory=list)
    estimated_install_time: float = 0.0
    estimated_size_mb: float = 0.0
    analysis_timestamp: float = field(default_factory=time.time)
    
    @property
    def all_dependencies(self) -> List[PackageInfo]:
        """Get all dependencies (direct + transitive)."""
        return self.direct_dependencies + self.transitive_dependencies
    
    @property
    def total_packages(self) -> int:
        """Get total number of packages."""
        return len(self.direct_dependencies) + len(self.transitive_dependencies)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'direct_dependencies': [d.to_dict() for d in self.direct_dependencies],
            'transitive_dependencies': [d.to_dict() for d in self.transitive_dependencies],
            'vulnerabilities': [v.to_dict() for v in self.vulnerabilities],
            'conflicts': self.conflicts,
            'security_score': self.security_score,
            'installation_plan': self.installation_plan,
            'estimated_install_time': self.estimated_install_time,
            'estimated_size_mb': self.estimated_size_mb,
            'total_packages': self.total_packages,
            'analysis_timestamp': self.analysis_timestamp
        }


@dataclass
class InstallationResult:
    """Result of package installation."""
    package_info: PackageInfo
    status: InstallationStatus
    installed_version: Optional[str] = None
    install_time: float = 0.0
    size_mb: float = 0.0
    stdout: str = ""
    stderr: str = ""
    error_message: Optional[str] = None
    dependencies_installed: List[str] = field(default_factory=list)
    
    @property
    def success(self) -> bool:
        """Check if installation was successful."""
        return self.status == InstallationStatus.SUCCESS
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'package_info': self.package_info.to_dict(),
            'status': self.status.value,
            'installed_version': self.installed_version,
            'install_time': self.install_time,
            'size_mb': self.size_mb,
            'stdout': self.stdout,
            'stderr': self.stderr,
            'error_message': self.error_message,
            'dependencies_installed': self.dependencies_installed,
            'success': self.success
        }


@dataclass
class InstallationSummary:
    """Summary of batch installation results."""
    total_packages: int
    successful_installs: int
    failed_installs: int
    skipped_installs: int
    blocked_installs: int
    total_time: float
    total_size_mb: float
    results: List[InstallationResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate installation success rate."""
        if self.total_packages == 0:
            return 100.0
        return (self.successful_installs / self.total_packages) * 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'total_packages': self.total_packages,
            'successful_installs': self.successful_installs,
            'failed_installs': self.failed_installs,
            'skipped_installs': self.skipped_installs,
            'blocked_installs': self.blocked_installs,
            'total_time': self.total_time,
            'total_size_mb': self.total_size_mb,
            'success_rate': self.success_rate,
            'results': [r.to_dict() for r in self.results],
            'errors': self.errors
        }


class DependencyExtractor:
    """Extract dependencies from various sources."""
    
    def __init__(self):
        # Standard library modules that should not be installed
        self.stdlib_modules = {
            'os', 'sys', 'time', 'datetime', 'json', 're', 'math', 'random',
            'collections', 'itertools', 'functools', 'operator', 'pathlib',
            'urllib', 'http', 'socket', 'threading', 'multiprocessing',
            'subprocess', 'logging', 'argparse', 'configparser', 'tempfile',
            'shutil', 'glob', 'pickle', 'csv', 'xml', 'html', 'email',
            'base64', 'hashlib', 'hmac', 'secrets', 'uuid', 'decimal',
            'fractions', 'statistics', 'unittest', 'doctest', 'pdb',
            'trace', 'traceback', 'warnings', 'contextlib', 'copy',
            'types', 'inspect', 'gc', 'weakref', 'abc', 'atexit'
        }
        
        # Common package name mappings (import name -> package name)
        self.package_mappings = {
            'cv2': 'opencv-python',
            'PIL': 'Pillow',
            'sklearn': 'scikit-learn',
            'serial': 'pyserial',
            'yaml': 'PyYAML',
            'bs4': 'beautifulsoup4',
            'MySQLdb': 'mysqlclient',
            'psycopg2': 'psycopg2-binary',
            'pycurl': 'pycurl',
            'lxml': 'lxml',
            'dateutil': 'python-dateutil',
            'jwt': 'PyJWT',
            'crypto': 'pycryptodome',
            'Crypto': 'pycryptodome'
        }
    
    def extract_from_python_code(self, code: str) -> List[PackageInfo]:
        """Extract dependencies from Python code using AST analysis."""
        dependencies = []
        
        try:
            # Parse code using AST
            tree = ast.parse(code)
            
            # Extract import statements
            imports = set()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split('.')[0])
            
            # Filter out standard library modules
            third_party_imports = imports - self.stdlib_modules
            
            # Convert to PackageInfo objects
            for import_name in third_party_imports:
                package_name = self.package_mappings.get(import_name, import_name)
                package_info = PackageInfo(
                    name=package_name,
                    ecosystem=PackageEcosystem.PYPI,
                    source=DependencySource.IMPORTS
                )
                dependencies.append(package_info)
            
        except SyntaxError as e:
            logger.warning(f"Syntax error in Python code: {e}")
        except Exception as e:
            logger.error(f"Error extracting Python dependencies: {e}")
        
        return dependencies
    
    def extract_from_requirements_txt(self, content: str) -> List[PackageInfo]:
        """Extract dependencies from requirements.txt content."""
        dependencies = []
        
        for line in content.splitlines():
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Skip -e and -f flags for now
            if line.startswith('-'):
                continue
            
            # Parse package specification
            package_info = self._parse_package_specification(line)
            if package_info:
                package_info.source = DependencySource.REQUIREMENTS_TXT
                dependencies.append(package_info)
        
        return dependencies
    
    def extract_from_package_json(self, content: str) -> List[PackageInfo]:
        """Extract dependencies from package.json content."""
        dependencies = []
        
        try:
            data = json.loads(content)
            
            # Extract dependencies and devDependencies
            for dep_type in ['dependencies', 'devDependencies']:
                deps = data.get(dep_type, {})
                for name, version_spec in deps.items():
                    package_info = PackageInfo(
                        name=name,
                        version=version_spec.lstrip('^~>=<'),
                        ecosystem=PackageEcosystem.NPM,
                        source=DependencySource.PACKAGE_JSON,
                        required=(dep_type == 'dependencies')
                    )
                    dependencies.append(package_info)
                    
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid package.json format: {e}")
        except Exception as e:
            logger.error(f"Error extracting NPM dependencies: {e}")
        
        return dependencies
    
    def extract_from_setup_py(self, content: str) -> List[PackageInfo]:
        """Extract dependencies from setup.py content."""
        dependencies = []
        
        # Look for install_requires in setup.py
        install_requires_pattern = r'install_requires\s*=\s*\[(.*?)\]'
        match = re.search(install_requires_pattern, content, re.DOTALL)
        
        if match:
            requirements_text = match.group(1)
            # Extract quoted strings
            for req_match in re.finditer(r'[\'"]([^\'"]+)[\'"]', requirements_text):
                requirement = req_match.group(1)
                package_info = self._parse_package_specification(requirement)
                if package_info:
                    package_info.source = DependencySource.SETUP_PY
                    dependencies.append(package_info)
        
        return dependencies
    
    def _parse_package_specification(self, spec: str) -> Optional[PackageInfo]:
        """Parse package specification string."""
        # Remove extra whitespace
        spec = spec.strip()
        
        if not spec:
            return None
        
        # Handle version specifiers
        version_pattern = r'^([a-zA-Z0-9_.-]+)([><=!~]+.+)?$'
        match = re.match(version_pattern, spec)
        
        if match:
            name = match.group(1)
            version_spec = match.group(2)
            
            # Extract version if it's an exact match
            version = None
            if version_spec and version_spec.startswith('=='):
                version = version_spec[2:]
            
            return PackageInfo(
                name=name,
                version=version,
                ecosystem=PackageEcosystem.PYPI,
                constraints=[version_spec] if version_spec else []
            )
        
        return None


class PackageValidator:
    """Validate packages for security and integrity."""
    
    def __init__(self):
        # Known malicious packages
        self.malicious_packages = {
            PackageEcosystem.PYPI: {
                'requests-malicious', 'urllib3-fake', 'numpy-evil',
                'setup-tools', 'python3-pip', 'pip-tools-fake',
                'colorama-fake', 'pyyaml-bad', 'setuptools-fake'
            },
            PackageEcosystem.NPM: {
                'cross-env-malicious', 'event-stream-bad', 'eslint-scope-fake',
                'babel-core-fake', 'lodash-fake', 'express-fake'
            }
        }
        
        # Typosquatting detection patterns
        self.legitimate_packages = {
            PackageEcosystem.PYPI: {
                'requests', 'urllib3', 'numpy', 'pandas', 'matplotlib',
                'scipy', 'scikit-learn', 'pillow', 'django', 'flask',
                'click', 'jinja2', 'werkzeug', 'setuptools', 'pip'
            },
            PackageEcosystem.NPM: {
                'express', 'lodash', 'react', 'angular', 'vue',
                'webpack', 'babel', 'eslint', 'jest', 'mocha'
            }
        }
        
        # Package size limits (MB)
        self.size_limits = {
            PackageEcosystem.PYPI: 500,  # 500MB max for Python packages
            PackageEcosystem.NPM: 100,   # 100MB max for NPM packages
        }
    
    async def validate_package(self, package_info: PackageInfo) -> List[str]:
        """Validate single package for security issues."""
        issues = []
        
        # Check for known malicious packages
        malicious_set = self.malicious_packages.get(package_info.ecosystem, set())
        if package_info.name.lower() in malicious_set:
            issues.append(f"Package {package_info.name} is known to be malicious")
        
        # Check for typosquatting
        if self._is_likely_typosquatting(package_info):
            issues.append(f"Package {package_info.name} may be typosquatting a legitimate package")
        
        # Check package existence and metadata
        try:
            metadata = await self._get_package_metadata(package_info)
            if not metadata:
                issues.append(f"Package {package_info.name} not found in registry")
            else:
                # Validate package metadata
                metadata_issues = self._validate_package_metadata(metadata, package_info)
                issues.extend(metadata_issues)
                
        except Exception as e:
            issues.append(f"Failed to validate package {package_info.name}: {e}")
        
        return issues
    
    def _is_likely_typosquatting(self, package_info: PackageInfo) -> bool:
        """Check if package name is likely typosquatting."""
        legitimate_set = self.legitimate_packages.get(package_info.ecosystem, set())
        package_name = package_info.name.lower()
        
        for legitimate_name in legitimate_set:
            # Check for common typosquatting patterns
            if self._is_similar_name(package_name, legitimate_name):
                return True
        
        return False
    
    def _is_similar_name(self, name1: str, name2: str) -> bool:
        """Check if two names are suspiciously similar."""
        if name1 == name2:
            return False
        
        # Check for character substitutions (e.g., 'o' -> '0')
        substitutions = {'o': '0', 'i': '1', 'l': '1', 'e': '3', 's': '5'}
        
        # Check forward substitution
        modified_name = name2
        for original, substitute in substitutions.items():
            modified_name = modified_name.replace(original, substitute)
        if name1 == modified_name:
            return True
        
        # Check character omission/addition
        if len(name1) == len(name2) - 1 and name2.replace(name1, '') in name2:
            return True
        if len(name1) == len(name2) + 1 and name1.replace(name2, '') in name1:
            return True
        
        # Check character transposition (adjacent characters swapped)
        for i in range(len(name2) - 1):
            swapped = list(name2)
            swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
            if ''.join(swapped) == name1:
                return True
        
        return False
    
    async def _get_package_metadata(self, package_info: PackageInfo) -> Optional[Dict[str, Any]]:
        """Get package metadata from registry."""
        try:
            if package_info.ecosystem == PackageEcosystem.PYPI:
                url = f"https://pypi.org/pypi/{package_info.name}/json"
            elif package_info.ecosystem == PackageEcosystem.NPM:
                url = f"https://registry.npmjs.org/{package_info.name}"
            else:
                return None
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        return await response.json()
                    return None
                    
        except Exception as e:
            logger.warning(f"Failed to get metadata for {package_info.name}: {e}")
            return None
    
    def _validate_package_metadata(self, metadata: Dict[str, Any], package_info: PackageInfo) -> List[str]:
        """Validate package metadata for suspicious characteristics."""
        issues = []
        
        try:
            if package_info.ecosystem == PackageEcosystem.PYPI:
                info = metadata.get('info', {})
                
                # Check for suspicious characteristics
                if not info.get('description', '').strip():
                    issues.append("Package has empty description")
                
                if not info.get('home_page', '').strip():
                    issues.append("Package has no home page")
                
                # Check creation date (packages created very recently might be suspicious)
                releases = metadata.get('releases', {})
                if releases:
                    latest_version = max(releases.keys(), key=lambda v: pkg_version.parse(v) if pkg_version.parse(v) else pkg_version.parse('0.0.0'))
                    release_info = releases.get(latest_version, [])
                    if release_info:
                        upload_time = release_info[0].get('upload_time')
                        if upload_time:
                            # Check if package was uploaded very recently (last 7 days)
                            from datetime import datetime, timedelta
                            upload_date = datetime.fromisoformat(upload_time.replace('Z', '+00:00'))
                            if datetime.now().replace(tzinfo=upload_date.tzinfo) - upload_date < timedelta(days=7):
                                issues.append("Package was uploaded very recently (potential supply chain attack)")
            
            elif package_info.ecosystem == PackageEcosystem.NPM:
                # Check NPM package characteristics
                if not metadata.get('description', '').strip():
                    issues.append("Package has empty description")
                
                if not metadata.get('repository', {}).get('url', '').strip():
                    issues.append("Package has no repository URL")
        
        except Exception as e:
            logger.warning(f"Error validating metadata for {package_info.name}: {e}")
        
        return issues


class SecurePackageInstaller:
    """Secure package installation with sandboxing and validation."""
    
    def __init__(self):
        self.installation_stats = {
            'total_installs': 0,
            'successful_installs': 0,
            'failed_installs': 0,
            'blocked_installs': 0,
            'total_install_time': 0.0
        }
    
    async def install_package(
        self,
        package_info: PackageInfo,
        container: SecureContainer,
        validate: bool = True,
        timeout: int = 300
    ) -> InstallationResult:
        """Install single package in secure container."""
        
        self.installation_stats['total_installs'] += 1
        start_time = time.time()
        
        result = InstallationResult(
            package_info=package_info,
            status=InstallationStatus.PENDING
        )
        
        try:
            # Validate package before installation if requested
            if validate:
                validator = PackageValidator()
                validation_issues = await validator.validate_package(package_info)
                
                if validation_issues:
                    result.status = InstallationStatus.BLOCKED
                    result.error_message = f"Package blocked due to security issues: {'; '.join(validation_issues)}"
                    self.installation_stats['blocked_installs'] += 1
                    return result
            
            result.status = InstallationStatus.IN_PROGRESS
            
            # Install package based on ecosystem
            if package_info.ecosystem == PackageEcosystem.PYPI:
                install_result = await self._install_python_package(package_info, container, timeout)
            elif package_info.ecosystem == PackageEcosystem.NPM:
                install_result = await self._install_npm_package(package_info, container, timeout)
            else:
                result.status = InstallationStatus.FAILED
                result.error_message = f"Unsupported ecosystem: {package_info.ecosystem}"
                self.installation_stats['failed_installs'] += 1
                return result
            
            # Update result with installation details
            result.status = InstallationStatus.SUCCESS if install_result['success'] else InstallationStatus.FAILED
            result.stdout = install_result.get('stdout', '')
            result.stderr = install_result.get('stderr', '')
            result.error_message = install_result.get('error')
            result.installed_version = install_result.get('version')
            result.dependencies_installed = install_result.get('dependencies', [])
            
            if result.success:
                self.installation_stats['successful_installs'] += 1
            else:
                self.installation_stats['failed_installs'] += 1
            
        except Exception as e:
            result.status = InstallationStatus.FAILED
            result.error_message = str(e)
            self.installation_stats['failed_installs'] += 1
            logger.error(f"Installation error for {package_info.name}: {e}")
        
        finally:
            result.install_time = time.time() - start_time
            self.installation_stats['total_install_time'] += result.install_time
        
        return result
    
    async def _install_python_package(
        self,
        package_info: PackageInfo,
        container: SecureContainer,
        timeout: int
    ) -> Dict[str, Any]:
        """Install Python package using pip."""
        
        # Build pip install command
        package_spec = package_info.full_name
        if package_info.extras:
            extras_str = ','.join(package_info.extras)
            package_spec = f"{package_info.name}[{extras_str}]"
            if package_info.version:
                package_spec += f"=={package_info.version}"
        
        install_command = [
            'pip', 'install', '--no-cache-dir', '--disable-pip-version-check',
            '--quiet', package_spec
        ]
        
        # Execute installation
        from .docker_manager import EnhancedDockerManager
        docker_manager = EnhancedDockerManager()
        
        try:
            exec_result = await docker_manager.execute_in_container(
                container,
                ' '.join(install_command),
                timeout=timeout
            )
            
            if exec_result['success']:
                # Get installed version
                version_result = await docker_manager.execute_in_container(
                    container,
                    f'pip show {package_info.name}',
                    timeout=10
                )
                
                installed_version = None
                if version_result['success']:
                    # Parse version from pip show output
                    for line in version_result['output'].splitlines():
                        if line.startswith('Version:'):
                            installed_version = line.split(':', 1)[1].strip()
                            break
                
                return {
                    'success': True,
                    'stdout': exec_result['output'],
                    'stderr': exec_result['error'],
                    'version': installed_version,
                    'dependencies': []  # pip doesn't easily provide dependency list
                }
            else:
                return {
                    'success': False,
                    'stdout': exec_result['output'],
                    'stderr': exec_result['error'],
                    'error': f"pip install failed: {exec_result['error']}"
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Installation failed: {e}",
                'stdout': '',
                'stderr': str(e)
            }
    
    async def _install_npm_package(
        self,
        package_info: PackageInfo,
        container: SecureContainer,
        timeout: int
    ) -> Dict[str, Any]:
        """Install NPM package."""
        
        # Build npm install command
        package_spec = package_info.full_name
        install_command = ['npm', 'install', '--no-save', '--quiet', package_spec]
        
        # Execute installation
        from .docker_manager import EnhancedDockerManager
        docker_manager = EnhancedDockerManager()
        
        try:
            exec_result = await docker_manager.execute_in_container(
                container,
                ' '.join(install_command),
                timeout=timeout
            )
            
            if exec_result['success']:
                return {
                    'success': True,
                    'stdout': exec_result['output'],
                    'stderr': exec_result['error'],
                    'version': package_info.version,
                    'dependencies': []
                }
            else:
                return {
                    'success': False,
                    'stdout': exec_result['output'],
                    'stderr': exec_result['error'],
                    'error': f"npm install failed: {exec_result['error']}"
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f"NPM installation failed: {e}",
                'stdout': '',
                'stderr': str(e)
            }
    
    async def install_packages_batch(
        self,
        packages: List[PackageInfo],
        container: SecureContainer,
        validate: bool = True,
        max_concurrent: int = 3
    ) -> InstallationSummary:
        """Install multiple packages with concurrency control."""
        
        start_time = time.time()
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def install_with_semaphore(pkg: PackageInfo) -> InstallationResult:
            async with semaphore:
                return await self.install_package(pkg, container, validate)
        
        # Install packages concurrently
        tasks = [install_with_semaphore(pkg) for pkg in packages]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        installation_results = []
        errors = []
        successful = 0
        failed = 0
        skipped = 0
        blocked = 0
        total_size = 0.0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_msg = f"Installation failed for {packages[i].name}: {result}"
                errors.append(error_msg)
                failed += 1
                # Create failed result
                failed_result = InstallationResult(
                    package_info=packages[i],
                    status=InstallationStatus.FAILED,
                    error_message=str(result)
                )
                installation_results.append(failed_result)
            else:
                installation_results.append(result)
                total_size += result.size_mb
                
                if result.status == InstallationStatus.SUCCESS:
                    successful += 1
                elif result.status == InstallationStatus.FAILED:
                    failed += 1
                elif result.status == InstallationStatus.SKIPPED:
                    skipped += 1
                elif result.status == InstallationStatus.BLOCKED:
                    blocked += 1
        
        total_time = time.time() - start_time
        
        return InstallationSummary(
            total_packages=len(packages),
            successful_installs=successful,
            failed_installs=failed,
            skipped_installs=skipped,
            blocked_installs=blocked,
            total_time=total_time,
            total_size_mb=total_size,
            results=installation_results,
            errors=errors
        )
    
    def get_installation_stats(self) -> Dict[str, Any]:
        """Get installation statistics."""
        stats = self.installation_stats.copy()
        
        if stats['total_installs'] > 0:
            stats['success_rate'] = (stats['successful_installs'] / stats['total_installs']) * 100
            stats['average_install_time'] = stats['total_install_time'] / stats['total_installs']
        else:
            stats['success_rate'] = 0.0
            stats['average_install_time'] = 0.0
        
        return stats


class IntelligentDependencyManager:
    """Main dependency manager orchestrating all dependency operations."""
    
    def __init__(self, enable_validation: bool = True, enable_caching: bool = True):
        self.extractor = DependencyExtractor()
        self.validator = PackageValidator()
        self.installer = SecurePackageInstaller()
        self.enable_validation = enable_validation
        self.enable_caching = enable_caching
        
        # Cache for analysis results
        self.analysis_cache: Dict[str, DependencyAnalysis] = {}
        
        # Manager statistics
        self.stats = {
            'analyses_performed': 0,
            'cache_hits': 0,
            'packages_installed': 0,
            'security_blocks': 0
        }
        
        logger.info("IntelligentDependencyManager initialized")
    
    async def analyze_code_dependencies(
        self,
        code: str,
        language: str = 'python',
        additional_requirements: Optional[str] = None
    ) -> DependencyAnalysis:
        """Analyze code for dependencies and create installation plan."""
        
        self.stats['analyses_performed'] += 1
        
        # Check cache first
        cache_key = hashlib.sha256(f"{code}{language}{additional_requirements or ''}".encode()).hexdigest()
        if self.enable_caching and cache_key in self.analysis_cache:
            self.stats['cache_hits'] += 1
            return self.analysis_cache[cache_key]
        
        analysis = DependencyAnalysis()
        
        try:
            # Extract dependencies from code
            if language.lower() == 'python':
                code_deps = self.extractor.extract_from_python_code(code)
            else:
                # For non-Python languages, create empty list for now
                code_deps = []
            
            analysis.direct_dependencies.extend(code_deps)
            
            # Extract from additional requirements if provided
            if additional_requirements:
                if language.lower() == 'python':
                    req_deps = self.extractor.extract_from_requirements_txt(additional_requirements)
                elif language.lower() == 'javascript':
                    req_deps = self.extractor.extract_from_package_json(additional_requirements)
                else:
                    req_deps = []
                
                analysis.direct_dependencies.extend(req_deps)
            
            # Remove duplicates
            unique_deps = {}
            for dep in analysis.direct_dependencies:
                if dep.name not in unique_deps:
                    unique_deps[dep.name] = dep
            analysis.direct_dependencies = list(unique_deps.values())
            
            # Validate dependencies if enabled
            if self.enable_validation:
                all_vulnerabilities = []
                for dep in analysis.direct_dependencies:
                    validation_issues = await self.validator.validate_package(dep)
                    if validation_issues:
                        # Convert issues to vulnerabilities
                        for issue in validation_issues:
                            from .policy_engine import Vulnerability
                            vuln = Vulnerability(
                                package_name=dep.name,
                                version=dep.version or 'unknown',
                                vulnerability_id='VALIDATION_ISSUE',
                                severity=ThreatLevel.HIGH,
                                description=issue
                            )
                            all_vulnerabilities.append(vuln)
                
                analysis.vulnerabilities = all_vulnerabilities
            
            # Calculate security score
            analysis.security_score = self._calculate_security_score(analysis)
            
            # Generate installation plan
            analysis.installation_plan = self._generate_installation_plan(analysis)
            
            # Estimate installation metrics
            analysis.estimated_install_time = len(analysis.direct_dependencies) * 10.0  # 10 seconds per package estimate
            analysis.estimated_size_mb = len(analysis.direct_dependencies) * 20.0  # 20MB per package estimate
            
            # Cache the result
            if self.enable_caching:
                self.analysis_cache[cache_key] = analysis
            
            logger.info(f"Dependency analysis complete: {len(analysis.direct_dependencies)} packages, "
                       f"security score: {analysis.security_score:.1f}")
            
        except Exception as e:
            logger.error(f"Dependency analysis failed: {e}")
            analysis.conflicts.append(f"Analysis error: {e}")
        
        return analysis
    
    async def install_dependencies_securely(
        self,
        dependencies: List[PackageInfo],
        container: SecureContainer,
        validate: bool = None
    ) -> InstallationSummary:
        """Install dependencies in secure container."""
        
        if validate is None:
            validate = self.enable_validation
        
        # Install packages
        summary = await self.installer.install_packages_batch(
            dependencies, container, validate=validate
        )
        
        # Update statistics
        self.stats['packages_installed'] += summary.successful_installs
        self.stats['security_blocks'] += summary.blocked_installs
        
        logger.info(f"Package installation complete: {summary.successful_installs}/{summary.total_packages} successful")
        
        return summary
    
    async def auto_resolve_and_install(
        self,
        code: str,
        container: SecureContainer,
        language: str = 'python',
        additional_requirements: Optional[str] = None
    ) -> Tuple[DependencyAnalysis, InstallationSummary]:
        """Automatically resolve dependencies and install them."""
        
        # Analyze dependencies
        analysis = await self.analyze_code_dependencies(code, language, additional_requirements)
        
        # Install dependencies
        summary = await self.install_dependencies_securely(
            analysis.direct_dependencies, container
        )
        
        return analysis, summary
    
    def _calculate_security_score(self, analysis: DependencyAnalysis) -> float:
        """Calculate security score based on vulnerabilities and packages."""
        base_score = 10.0
        
        # Deduct points for vulnerabilities
        for vuln in analysis.vulnerabilities:
            if vuln.severity == ThreatLevel.CRITICAL:
                base_score -= 3.0
            elif vuln.severity == ThreatLevel.HIGH:
                base_score -= 2.0
            elif vuln.severity == ThreatLevel.MEDIUM:
                base_score -= 1.0
            elif vuln.severity == ThreatLevel.LOW:
                base_score -= 0.5
        
        # Slight deduction for number of dependencies (larger attack surface)
        base_score -= len(analysis.direct_dependencies) * 0.1
        
        return max(0.0, min(10.0, base_score))
    
    def _generate_installation_plan(self, analysis: DependencyAnalysis) -> List[str]:
        """Generate step-by-step installation plan."""
        plan = []
        
        if not analysis.direct_dependencies:
            plan.append("No dependencies to install")
            return plan
        
        plan.append(f"Install {len(analysis.direct_dependencies)} packages:")
        
        for dep in analysis.direct_dependencies:
            plan.append(f"  - {dep.full_name} ({dep.ecosystem.value})")
        
        if analysis.vulnerabilities:
            plan.append(f"Warning: {len(analysis.vulnerabilities)} security issues detected")
        
        return plan
    
    def get_manager_statistics(self) -> Dict[str, Any]:
        """Get comprehensive manager statistics."""
        return {
            'manager_stats': self.stats.copy(),
            'installer_stats': self.installer.get_installation_stats(),
            'cache_size': len(self.analysis_cache),
            'cache_enabled': self.enable_caching,
            'validation_enabled': self.enable_validation
        }
    
    def clear_cache(self):
        """Clear analysis cache."""
        self.analysis_cache.clear()
        logger.info("Dependency analysis cache cleared")


# Utility functions
async def analyze_dependencies(
    code: str,
    language: str = 'python',
    requirements: Optional[str] = None
) -> DependencyAnalysis:
    """Convenience function for dependency analysis."""
    manager = IntelligentDependencyManager()
    return await manager.analyze_code_dependencies(code, language, requirements)


async def install_requirements_safely(
    requirements: str,
    container: SecureContainer,
    language: str = 'python'
) -> InstallationSummary:
    """Safely install requirements in container."""
    manager = IntelligentDependencyManager()
    
    # Parse requirements
    extractor = DependencyExtractor()
    if language.lower() == 'python':
        dependencies = extractor.extract_from_requirements_txt(requirements)
    else:
        dependencies = []
    
    # Install securely
    return await manager.install_dependencies_securely(dependencies, container)


# Export main classes
__all__ = [
    'IntelligentDependencyManager',
    'DependencyExtractor',
    'PackageValidator', 
    'SecurePackageInstaller',
    'DependencyAnalysis',
    'PackageInfo',
    'InstallationResult',
    'InstallationSummary',
    'DependencySource',
    'PackageEcosystem',
    'InstallationStatus',
    'analyze_dependencies',
    'install_requirements_safely'
]