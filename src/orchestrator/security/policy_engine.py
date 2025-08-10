"""Advanced Security Policy Engine - Issue #206 Task 1.2

Comprehensive security analysis and policy enforcement with threat detection,
code analysis, vulnerability scanning, and dynamic policy generation.
"""

import ast
import re
import logging
import hashlib
import time
import asyncio
import json
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import requests
import aiohttp
from urllib.parse import urlparse

from .docker_manager import ResourceLimits, SecurityConfig, SandboxingLevel

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ViolationType(Enum):
    """Types of security violations."""
    DANGEROUS_IMPORT = "dangerous_import"
    DANGEROUS_BUILTIN = "dangerous_builtin"
    SYSTEM_ACCESS = "system_access"
    NETWORK_ACCESS = "network_access"
    FILE_SYSTEM_ACCESS = "file_system_access"
    SUBPROCESS_EXECUTION = "subprocess_execution"
    EVAL_EXECUTION = "eval_execution"
    MALICIOUS_PATTERN = "malicious_pattern"
    RESOURCE_ABUSE = "resource_abuse"
    VULNERABILITY_RISK = "vulnerability_risk"


class CodeLanguage(Enum):
    """Supported code languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    BASH = "bash"
    SHELL = "shell"
    UNKNOWN = "unknown"


@dataclass
class SecurityViolation:
    """Represents a security policy violation."""
    violation_type: ViolationType
    severity: ThreatLevel
    description: str
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    recommendation: Optional[str] = None
    auto_fix: Optional[str] = None
    confidence_score: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'type': self.violation_type.value,
            'severity': self.severity.value,
            'description': self.description,
            'line_number': self.line_number,
            'code_snippet': self.code_snippet,
            'recommendation': self.recommendation,
            'auto_fix': self.auto_fix,
            'confidence_score': self.confidence_score
        }


@dataclass
class Vulnerability:
    """Represents a dependency vulnerability."""
    package_name: str
    version: str
    vulnerability_id: str
    severity: ThreatLevel
    description: str
    cvss_score: Optional[float] = None
    fixed_versions: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'package_name': self.package_name,
            'version': self.version,
            'vulnerability_id': self.vulnerability_id,
            'severity': self.severity.value,
            'description': self.description,
            'cvss_score': self.cvss_score,
            'fixed_versions': self.fixed_versions,
            'references': self.references
        }


@dataclass
class CodeAnalysis:
    """Results of static code analysis."""
    language: CodeLanguage
    imports: List[str] = field(default_factory=list)
    builtin_functions: List[str] = field(default_factory=list)
    system_calls: List[str] = field(default_factory=list)
    network_operations: List[str] = field(default_factory=list)
    file_operations: List[str] = field(default_factory=list)
    subprocess_calls: List[str] = field(default_factory=list)
    eval_calls: List[str] = field(default_factory=list)
    dangerous_patterns: List[str] = field(default_factory=list)
    estimated_complexity: int = 1
    estimated_runtime: float = 1.0
    estimated_memory_mb: int = 64
    lines_of_code: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'language': self.language.value,
            'imports': self.imports,
            'builtin_functions': self.builtin_functions,
            'system_calls': self.system_calls,
            'network_operations': self.network_operations,
            'file_operations': self.file_operations,
            'subprocess_calls': self.subprocess_calls,
            'eval_calls': self.eval_calls,
            'dangerous_patterns': self.dangerous_patterns,
            'estimated_complexity': self.estimated_complexity,
            'estimated_runtime': self.estimated_runtime,
            'estimated_memory_mb': self.estimated_memory_mb,
            'lines_of_code': self.lines_of_code
        }


@dataclass
class SecurityAssessment:
    """Comprehensive security assessment result."""
    code_analysis: CodeAnalysis
    violations: List[SecurityViolation] = field(default_factory=list)
    vulnerabilities: List[Vulnerability] = field(default_factory=list)
    risk_score: float = 0.0
    threat_level: ThreatLevel = ThreatLevel.LOW
    sandboxing_level: SandboxingLevel = SandboxingLevel.DIRECT
    resource_requirements: Optional[ResourceLimits] = None
    security_config: Optional[SecurityConfig] = None
    recommendations: List[str] = field(default_factory=list)
    auto_fixes: List[str] = field(default_factory=list)
    execution_allowed: bool = True
    analysis_timestamp: float = field(default_factory=time.time)
    
    def add_violation(self, violation: SecurityViolation):
        """Add security violation and update risk score."""
        self.violations.append(violation)
        self._recalculate_risk_score()
    
    def add_vulnerability(self, vulnerability: Vulnerability):
        """Add vulnerability and update risk score."""
        self.vulnerabilities.append(vulnerability)
        self._recalculate_risk_score()
    
    def _recalculate_risk_score(self):
        """Recalculate risk score based on violations and vulnerabilities."""
        score = 0.0
        
        # Score from violations
        for violation in self.violations:
            if violation.severity == ThreatLevel.LOW:
                score += 0.1 * violation.confidence_score
            elif violation.severity == ThreatLevel.MEDIUM:
                score += 0.3 * violation.confidence_score
            elif violation.severity == ThreatLevel.HIGH:
                score += 0.6 * violation.confidence_score
            elif violation.severity == ThreatLevel.CRITICAL:
                score += 1.0 * violation.confidence_score
        
        # Score from vulnerabilities
        for vuln in self.vulnerabilities:
            if vuln.severity == ThreatLevel.LOW:
                score += 0.2
            elif vuln.severity == ThreatLevel.MEDIUM:
                score += 0.4
            elif vuln.severity == ThreatLevel.HIGH:
                score += 0.7
            elif vuln.severity == ThreatLevel.CRITICAL:
                score += 1.0
        
        self.risk_score = min(score, 10.0)  # Cap at 10.0
        
        # Determine threat level
        if self.risk_score < 1.0:
            self.threat_level = ThreatLevel.LOW
        elif self.risk_score < 3.0:
            self.threat_level = ThreatLevel.MEDIUM
        elif self.risk_score < 6.0:
            self.threat_level = ThreatLevel.HIGH
        else:
            self.threat_level = ThreatLevel.CRITICAL
        
        # Determine sandboxing level
        if self.risk_score < 0.5:
            self.sandboxing_level = SandboxingLevel.DIRECT
        elif self.risk_score < 2.0:
            self.sandboxing_level = SandboxingLevel.SANDBOXED
        elif self.risk_score < 7.0:
            self.sandboxing_level = SandboxingLevel.ISOLATED
        else:
            self.sandboxing_level = SandboxingLevel.BLOCKED
            self.execution_allowed = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'code_analysis': self.code_analysis.to_dict(),
            'violations': [v.to_dict() for v in self.violations],
            'vulnerabilities': [v.to_dict() for v in self.vulnerabilities],
            'risk_score': self.risk_score,
            'threat_level': self.threat_level.value,
            'sandboxing_level': self.sandboxing_level.value,
            'resource_requirements': self.resource_requirements.__dict__ if self.resource_requirements else None,
            'security_config': self.security_config.__dict__ if self.security_config else None,
            'recommendations': self.recommendations,
            'auto_fixes': self.auto_fixes,
            'execution_allowed': self.execution_allowed,
            'analysis_timestamp': self.analysis_timestamp
        }


class StaticCodeAnalyzer:
    """Static code analysis for security threats and patterns."""
    
    def __init__(self):
        # Dangerous patterns for different languages
        self.dangerous_patterns = {
            CodeLanguage.PYTHON: {
                'imports': {
                    # System access
                    'os', 'sys', 'subprocess', 'commands', 'platform',
                    # Network access
                    'socket', 'urllib', 'urllib2', 'requests', 'http', 'ftplib',
                    'telnetlib', 'smtplib', 'poplib', 'imaplib', 'nntplib',
                    # File system
                    'shutil', 'tempfile', 'glob', 'pathlib', 'fileinput',
                    # Process control
                    'multiprocessing', 'threading', 'concurrent', 'asyncio.subprocess',
                    # Low-level access
                    'ctypes', 'mmap', 'marshal', 'pickle', 'shelve',
                    # Code execution
                    'importlib', 'pkgutil', 'runpy', 'code', 'codeop'
                },
                'builtins': {
                    'eval', 'exec', 'compile', '__import__', 'open',
                    'input', 'raw_input', 'getattr', 'setattr', 'hasattr',
                    'globals', 'locals', 'vars', 'dir'
                },
                'patterns': [
                    r'os\.system\s*\(',
                    r'os\.popen\s*\(',
                    r'os\.spawn\w*\s*\(',
                    r'subprocess\.',
                    r'eval\s*\(',
                    r'exec\s*\(',
                    r'compile\s*\(',
                    r'__import__\s*\(',
                    r'getattr\s*\(',
                    r'setattr\s*\(',
                    r'open\s*\(',
                    r'file\s*\(',
                    r'execfile\s*\(',
                    r'reload\s*\(',
                    r'input\s*\(',
                    r'raw_input\s*\(',
                ]
            },
            CodeLanguage.JAVASCRIPT: {
                'imports': {
                    'child_process', 'fs', 'path', 'os', 'cluster', 'worker_threads',
                    'http', 'https', 'net', 'dgram', 'dns', 'url',
                    'vm', 'repl', 'module', 'process'
                },
                'builtins': {
                    'eval', 'Function', 'setTimeout', 'setInterval',
                    'require', 'import', 'process', 'global', 'Buffer'
                },
                'patterns': [
                    r'eval\s*\(',
                    r'Function\s*\(',
                    r'setTimeout\s*\(',
                    r'setInterval\s*\(',
                    r'require\s*\(',
                    r'process\.',
                    r'global\.',
                    r'Buffer\.',
                    r'child_process',
                    r'fs\.',
                    r'execSync\s*\(',
                    r'spawn\s*\(',
                    r'exec\s*\('
                ]
            },
            CodeLanguage.BASH: {
                'patterns': [
                    r'\$\(',
                    r'`[^`]*`',
                    r'eval\s+',
                    r'exec\s+',
                    r'sh\s+',
                    r'bash\s+',
                    r'sudo\s+',
                    r'rm\s+',
                    r'mv\s+',
                    r'cp\s+',
                    r'chmod\s+',
                    r'chown\s+',
                    r'wget\s+',
                    r'curl\s+',
                    r'nc\s+',
                    r'netcat\s+',
                    r'>&',
                    r'\|\s*sh',
                    r'\|\s*bash'
                ]
            }
        }
    
    def detect_language(self, code: str) -> CodeLanguage:
        """Detect programming language from code."""
        code_lower = code.lower().strip()
        
        # Python indicators
        if any(pattern in code for pattern in ['import ', 'def ', 'class ', 'print(', 'if __name__']):
            return CodeLanguage.PYTHON
        
        # JavaScript indicators
        if any(pattern in code for pattern in ['function ', 'var ', 'let ', 'const ', 'require(', 'module.exports']):
            return CodeLanguage.JAVASCRIPT
        
        # Bash indicators
        if any(pattern in code_lower for pattern in ['#!/bin/bash', '#!/bin/sh', 'echo ', 'if [', 'for ']):
            return CodeLanguage.BASH
        
        # Shell command indicators
        if any(pattern in code_lower for pattern in ['ls ', 'cd ', 'mkdir ', 'rm ', 'mv ', 'cp ', 'grep ']):
            return CodeLanguage.SHELL
        
        return CodeLanguage.UNKNOWN
    
    def analyze_python_code(self, code: str) -> CodeAnalysis:
        """Analyze Python code using AST parsing."""
        analysis = CodeAnalysis(language=CodeLanguage.PYTHON)
        
        try:
            # Parse code using AST
            tree = ast.parse(code)
            analysis.lines_of_code = len(code.splitlines())
            
            # Analyze AST nodes
            for node in ast.walk(tree):
                # Import statements
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis.imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        analysis.imports.append(node.module)
                
                # Function calls
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        analysis.builtin_functions.append(func_name)
                        
                        # Categorize function calls
                        if func_name in ['eval', 'exec', 'compile']:
                            analysis.eval_calls.append(func_name)
                        elif func_name in ['open', 'file']:
                            analysis.file_operations.append(func_name)
                    
                    elif isinstance(node.func, ast.Attribute):
                        attr_name = node.func.attr
                        if isinstance(node.func.value, ast.Name):
                            module_name = node.func.value.id
                            full_name = f"{module_name}.{attr_name}"
                            
                            # Categorize module calls
                            if module_name in ['os', 'sys']:
                                analysis.system_calls.append(full_name)
                            elif module_name in ['subprocess']:
                                analysis.subprocess_calls.append(full_name)
                            elif module_name in ['requests', 'urllib', 'socket']:
                                analysis.network_operations.append(full_name)
                            elif module_name in ['shutil', 'pathlib']:
                                analysis.file_operations.append(full_name)
            
            # Pattern-based analysis for missed cases
            self._analyze_patterns(code, analysis)
            
            # Estimate complexity and resource usage
            analysis.estimated_complexity = self._estimate_complexity(tree)
            analysis.estimated_runtime = self._estimate_runtime(analysis)
            analysis.estimated_memory_mb = self._estimate_memory(analysis)
            
        except SyntaxError as e:
            logger.warning(f"Syntax error in Python code analysis: {e}")
            analysis.dangerous_patterns.append(f"Syntax error: {e}")
        
        except Exception as e:
            logger.error(f"Error analyzing Python code: {e}")
            analysis.dangerous_patterns.append(f"Analysis error: {e}")
        
        return analysis
    
    def analyze_javascript_code(self, code: str) -> CodeAnalysis:
        """Analyze JavaScript code using pattern matching."""
        analysis = CodeAnalysis(language=CodeLanguage.JAVASCRIPT)
        analysis.lines_of_code = len(code.splitlines())
        
        # Extract require/import statements
        require_pattern = r'require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)'
        import_pattern = r'import\s+.*?from\s+[\'"]([^\'"]+)[\'"]'
        
        for match in re.finditer(require_pattern, code):
            analysis.imports.append(match.group(1))
        
        for match in re.finditer(import_pattern, code):
            analysis.imports.append(match.group(1))
        
        # Pattern-based analysis
        self._analyze_patterns(code, analysis)
        
        # Estimate resource usage
        analysis.estimated_complexity = len(code.splitlines())
        analysis.estimated_runtime = analysis.estimated_complexity * 0.1
        analysis.estimated_memory_mb = max(32, analysis.estimated_complexity * 2)
        
        return analysis
    
    def analyze_bash_code(self, code: str) -> CodeAnalysis:
        """Analyze Bash/Shell code using pattern matching."""
        analysis = CodeAnalysis(language=CodeLanguage.BASH)
        analysis.lines_of_code = len(code.splitlines())
        
        # Pattern-based analysis
        self._analyze_patterns(code, analysis)
        
        # Estimate resource usage
        analysis.estimated_complexity = len(code.splitlines())
        analysis.estimated_runtime = analysis.estimated_complexity * 0.05
        analysis.estimated_memory_mb = max(16, analysis.estimated_complexity)
        
        return analysis
    
    def _analyze_patterns(self, code: str, analysis: CodeAnalysis):
        """Analyze code for dangerous patterns using regex."""
        patterns = self.dangerous_patterns.get(analysis.language, {}).get('patterns', [])
        
        for pattern in patterns:
            matches = re.finditer(pattern, code, re.IGNORECASE)
            for match in matches:
                analysis.dangerous_patterns.append(match.group(0))
    
    def _estimate_complexity(self, tree: ast.AST) -> int:
        """Estimate code complexity from AST."""
        complexity = 1
        
        for node in ast.walk(tree):
            # Control structures add complexity
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try)):
                complexity += 1
            elif isinstance(node, ast.FunctionDef):
                complexity += 2
            elif isinstance(node, ast.ClassDef):
                complexity += 3
        
        return complexity
    
    def _estimate_runtime(self, analysis: CodeAnalysis) -> float:
        """Estimate runtime in seconds based on analysis."""
        base_time = analysis.lines_of_code * 0.01  # 10ms per line base
        
        # Add time for different operations
        base_time += len(analysis.imports) * 0.1  # Import overhead
        base_time += len(analysis.network_operations) * 1.0  # Network calls
        base_time += len(analysis.file_operations) * 0.1  # File operations
        base_time += len(analysis.subprocess_calls) * 0.5  # Subprocess calls
        base_time += analysis.estimated_complexity * 0.05  # Complexity overhead
        
        return max(0.1, base_time)  # Minimum 100ms
    
    def _estimate_memory(self, analysis: CodeAnalysis) -> int:
        """Estimate memory usage in MB based on analysis."""
        base_memory = 32  # Base Python interpreter memory
        
        # Add memory for different operations
        base_memory += len(analysis.imports) * 5  # Import overhead
        base_memory += len(analysis.network_operations) * 10  # Network buffers
        base_memory += len(analysis.file_operations) * 5  # File buffers
        base_memory += analysis.estimated_complexity * 2  # Complexity overhead
        
        return max(32, min(base_memory, 1024))  # Cap at 1GB


class DependencyValidator:
    """Validates dependencies against vulnerability databases."""
    
    def __init__(self):
        self.vulnerability_databases = [
            {
                'name': 'OSV',
                'url': 'https://api.osv.dev/v1/query',
                'method': 'POST'
            },
            {
                'name': 'Snyk',
                'url': 'https://api.snyk.io/v1/vuln',
                'method': 'GET',
                'requires_auth': True
            }
        ]
        
        # Known malicious packages (simplified list)
        self.known_malicious_packages = {
            'python': {
                'requests-malicious', 'numpy-evil', 'urllib3-fake',
                'setup-tools', 'python3-pip', 'pip-tools-fake'
            },
            'javascript': {
                'cross-env-malicious', 'event-stream-bad', 'eslint-scope-fake',
                'babel-core-fake', 'lodash-fake'
            }
        }
        
        # Typosquatting patterns
        self.typosquatting_targets = {
            'python': {
                'requests': ['request', 'requsts', 'reqeusts'],
                'urllib3': ['urllib', 'urlib3', 'urllib4'],
                'numpy': ['numpy1', 'nummpy', 'numpi'],
                'pandas': ['panda', 'pandass', 'panads']
            },
            'javascript': {
                'lodash': ['lodsh', 'lodas', 'lodash-es'],
                'express': ['expresss', 'expres', 'express-js'],
                'react': ['recat', 'react-dom-fake', 'reactjs']
            }
        }
    
    async def validate_dependencies(
        self,
        dependencies: List[str],
        language: str = 'python'
    ) -> List[Vulnerability]:
        """Validate list of dependencies against security databases."""
        vulnerabilities = []
        
        for dependency in dependencies:
            # Check for known malicious packages
            if await self._is_known_malicious(dependency, language):
                vulnerability = Vulnerability(
                    package_name=dependency,
                    version='unknown',
                    vulnerability_id='MALICIOUS_PACKAGE',
                    severity=ThreatLevel.CRITICAL,
                    description=f'Package {dependency} is known to be malicious'
                )
                vulnerabilities.append(vulnerability)
                continue
            
            # Check for typosquatting
            if await self._is_typosquatting(dependency, language):
                vulnerability = Vulnerability(
                    package_name=dependency,
                    version='unknown',
                    vulnerability_id='TYPOSQUATTING_ATTEMPT',
                    severity=ThreatLevel.HIGH,
                    description=f'Package {dependency} appears to be typosquatting a legitimate package'
                )
                vulnerabilities.append(vulnerability)
                continue
            
            # Query vulnerability databases
            db_vulnerabilities = await self._query_vulnerability_databases(dependency)
            vulnerabilities.extend(db_vulnerabilities)
        
        return vulnerabilities
    
    async def _is_known_malicious(self, package_name: str, language: str) -> bool:
        """Check if package is in known malicious list."""
        malicious_set = self.known_malicious_packages.get(language, set())
        return package_name.lower() in malicious_set
    
    async def _is_typosquatting(self, package_name: str, language: str) -> bool:
        """Check if package name appears to be typosquatting."""
        targets = self.typosquatting_targets.get(language, {})
        
        for legitimate_name, typos in targets.items():
            if package_name.lower() in typos:
                return True
        
        return False
    
    async def _query_vulnerability_databases(self, package_name: str) -> List[Vulnerability]:
        """Query vulnerability databases for package vulnerabilities."""
        vulnerabilities = []
        
        # Query OSV database (open source)
        try:
            osv_vulnerabilities = await self._query_osv(package_name)
            vulnerabilities.extend(osv_vulnerabilities)
        except Exception as e:
            logger.warning(f"Error querying OSV database: {e}")
        
        return vulnerabilities
    
    async def _query_osv(self, package_name: str) -> List[Vulnerability]:
        """Query OSV vulnerability database."""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    'package': {
                        'name': package_name,
                        'ecosystem': 'PyPI'  # Default to PyPI
                    }
                }
                
                async with session.post(
                    'https://api.osv.dev/v1/query',
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_osv_response(data, package_name)
                    else:
                        logger.warning(f"OSV query failed with status {response.status}")
                        return []
        
        except Exception as e:
            logger.warning(f"Error querying OSV for {package_name}: {e}")
            return []
    
    def _parse_osv_response(self, data: Dict[str, Any], package_name: str) -> List[Vulnerability]:
        """Parse OSV API response into Vulnerability objects."""
        vulnerabilities = []
        
        vulns = data.get('vulns', [])
        for vuln_data in vulns:
            try:
                severity = self._parse_severity(vuln_data.get('severity', []))
                
                vulnerability = Vulnerability(
                    package_name=package_name,
                    version='unknown',
                    vulnerability_id=vuln_data.get('id', 'UNKNOWN'),
                    severity=severity,
                    description=vuln_data.get('summary', 'No description available'),
                    references=vuln_data.get('references', [])
                )
                
                vulnerabilities.append(vulnerability)
                
            except Exception as e:
                logger.warning(f"Error parsing OSV vulnerability data: {e}")
        
        return vulnerabilities
    
    def _parse_severity(self, severity_data: List[Dict[str, Any]]) -> ThreatLevel:
        """Parse severity information from vulnerability data."""
        if not severity_data:
            return ThreatLevel.MEDIUM
        
        # Look for CVSS score
        for severity_item in severity_data:
            if severity_item.get('type') == 'CVSS_V3':
                score = severity_item.get('score', 0.0)
                if score >= 9.0:
                    return ThreatLevel.CRITICAL
                elif score >= 7.0:
                    return ThreatLevel.HIGH
                elif score >= 4.0:
                    return ThreatLevel.MEDIUM
                else:
                    return ThreatLevel.LOW
        
        return ThreatLevel.MEDIUM


class ThreatDetector:
    """Advanced threat detection using pattern matching and heuristics."""
    
    def __init__(self):
        self.threat_patterns = {
            # Code injection patterns
            'code_injection': [
                r'eval\s*\(\s*input\s*\(',
                r'exec\s*\(\s*input\s*\(',
                r'exec\s*\(\s*raw_input\s*\(',
                r'compile\s*\(\s*input\s*\(',
                r'__import__\s*\(\s*input\s*\('
            ],
            
            # Command injection patterns
            'command_injection': [
                r'os\.system\s*\(\s*.*input',
                r'subprocess\..*\(\s*.*input',
                r'os\.popen\s*\(\s*.*input',
                r'commands\..*\(\s*.*input'
            ],
            
            # File system attacks
            'file_system_attack': [
                r'\.\./',  # Directory traversal
                r'\/etc\/passwd',
                r'\/etc\/shadow',
                r'\/proc\/',
                r'\/sys\/',
                r'open\s*\(\s*[\'"]\/[^\'\"]*[\'"]',
                r'rm\s+-rf\s+\/',
                r'rmdir\s+\/'
            ],
            
            # Network attacks
            'network_attack': [
                r'socket\.socket\s*\(',
                r'bind\s*\(\s*\([\'\"]\s*0\.0\.0\.0',
                r'connect\s*\(\s*\([\'\"]\s*\d+\.\d+\.\d+\.\d+',
                r'nc\s+-l\s+\d+',
                r'netcat\s+-l\s+\d+'
            ],
            
            # Resource exhaustion
            'resource_exhaustion': [
                r'while\s+True\s*:',
                r'for\s+.*\s+in\s+range\s*\(\s*\d{6,}',  # Large loops
                r'[\[\{]\s*\d+\s*[\]\}]\s*\*\s*\d{6,}',  # Large data structures
                r'open\s*\(\s*[\'"]\/dev\/zero[\'"]',
                r'malloc\s*\(\s*\d{9,}'  # Large memory allocation
            ],
            
            # Obfuscation attempts
            'obfuscation': [
                r'base64\.b64decode',
                r'codecs\.decode',
                r'bytes\.fromhex',
                r'chr\s*\(\s*\d+\s*\)',
                r'ord\s*\(\s*.*\s*\)',
                r'\.decode\s*\(\s*[\'"]rot13[\'"]'
            ]
        }
        
        # Suspicious string patterns
        self.suspicious_strings = [
            '/bin/sh', '/bin/bash', 'cmd.exe', 'powershell',
            'wget', 'curl', 'nc ', 'netcat', 'telnet',
            'reverse shell', 'backdoor', 'rootkit',
            'password', 'passwd', 'credentials', 'token',
            'exploit', 'payload', 'shellcode'
        ]
    
    def detect_threats(self, code: str, analysis: CodeAnalysis) -> List[SecurityViolation]:
        """Detect security threats in code."""
        violations = []
        
        # Pattern-based threat detection
        for threat_type, patterns in self.threat_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, code, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    line_number = code[:match.start()].count('\n') + 1
                    violation = SecurityViolation(
                        violation_type=ViolationType.MALICIOUS_PATTERN,
                        severity=self._get_threat_severity(threat_type),
                        description=f"{threat_type.replace('_', ' ').title()} pattern detected",
                        line_number=line_number,
                        code_snippet=match.group(0),
                        recommendation=f"Remove or sanitize {threat_type.replace('_', ' ')} pattern",
                        confidence_score=0.8
                    )
                    violations.append(violation)
        
        # Suspicious string detection
        for suspicious_string in self.suspicious_strings:
            if suspicious_string.lower() in code.lower():
                violation = SecurityViolation(
                    violation_type=ViolationType.MALICIOUS_PATTERN,
                    severity=ThreatLevel.MEDIUM,
                    description=f"Suspicious string detected: {suspicious_string}",
                    recommendation="Review usage of suspicious string",
                    confidence_score=0.6
                )
                violations.append(violation)
        
        # Import-based threat detection
        violations.extend(self._analyze_imports(analysis))
        
        # Builtin function threat detection
        violations.extend(self._analyze_builtins(analysis))
        
        # System call threat detection
        violations.extend(self._analyze_system_calls(analysis))
        
        return violations
    
    def _get_threat_severity(self, threat_type: str) -> ThreatLevel:
        """Get severity level for threat type."""
        severity_map = {
            'code_injection': ThreatLevel.CRITICAL,
            'command_injection': ThreatLevel.CRITICAL,
            'file_system_attack': ThreatLevel.HIGH,
            'network_attack': ThreatLevel.HIGH,
            'resource_exhaustion': ThreatLevel.MEDIUM,
            'obfuscation': ThreatLevel.MEDIUM
        }
        return severity_map.get(threat_type, ThreatLevel.LOW)
    
    def _analyze_imports(self, analysis: CodeAnalysis) -> List[SecurityViolation]:
        """Analyze imports for security threats."""
        violations = []
        
        dangerous_imports = {
            'os': ThreatLevel.HIGH,
            'sys': ThreatLevel.HIGH,
            'subprocess': ThreatLevel.CRITICAL,
            'commands': ThreatLevel.CRITICAL,
            'socket': ThreatLevel.HIGH,
            'urllib': ThreatLevel.MEDIUM,
            'requests': ThreatLevel.MEDIUM,
            'ctypes': ThreatLevel.HIGH,
            'marshal': ThreatLevel.HIGH,
            'pickle': ThreatLevel.MEDIUM,
            'tempfile': ThreatLevel.MEDIUM,
            'shutil': ThreatLevel.MEDIUM
        }
        
        for import_name in analysis.imports:
            base_import = import_name.split('.')[0]
            if base_import in dangerous_imports:
                severity = dangerous_imports[base_import]
                violation = SecurityViolation(
                    violation_type=ViolationType.DANGEROUS_IMPORT,
                    severity=severity,
                    description=f"Dangerous import detected: {import_name}",
                    recommendation=f"Consider removing import {import_name} or using sandboxed execution",
                    confidence_score=0.9
                )
                violations.append(violation)
        
        return violations
    
    def _analyze_builtins(self, analysis: CodeAnalysis) -> List[SecurityViolation]:
        """Analyze builtin functions for security threats."""
        violations = []
        
        dangerous_builtins = {
            'eval': ThreatLevel.CRITICAL,
            'exec': ThreatLevel.CRITICAL,
            'compile': ThreatLevel.HIGH,
            '__import__': ThreatLevel.HIGH,
            'open': ThreatLevel.MEDIUM,
            'input': ThreatLevel.LOW,
            'getattr': ThreatLevel.MEDIUM,
            'setattr': ThreatLevel.MEDIUM
        }
        
        for builtin_name in analysis.builtin_functions:
            if builtin_name in dangerous_builtins:
                severity = dangerous_builtins[builtin_name]
                violation = SecurityViolation(
                    violation_type=ViolationType.DANGEROUS_BUILTIN,
                    severity=severity,
                    description=f"Dangerous builtin function used: {builtin_name}",
                    recommendation=f"Avoid using {builtin_name} or implement input validation",
                    confidence_score=0.8
                )
                violations.append(violation)
        
        return violations
    
    def _analyze_system_calls(self, analysis: CodeAnalysis) -> List[SecurityViolation]:
        """Analyze system calls for security threats."""
        violations = []
        
        for system_call in analysis.system_calls:
            violation = SecurityViolation(
                violation_type=ViolationType.SYSTEM_ACCESS,
                severity=ThreatLevel.HIGH,
                description=f"System call detected: {system_call}",
                recommendation=f"Review necessity of system call {system_call}",
                confidence_score=0.7
            )
            violations.append(violation)
        
        for subprocess_call in analysis.subprocess_calls:
            violation = SecurityViolation(
                violation_type=ViolationType.SUBPROCESS_EXECUTION,
                severity=ThreatLevel.CRITICAL,
                description=f"Subprocess execution detected: {subprocess_call}",
                recommendation=f"Remove or sandbox subprocess call {subprocess_call}",
                confidence_score=0.9
            )
            violations.append(violation)
        
        return violations


class SecurityPolicyEngine:
    """Advanced security policy engine with comprehensive threat analysis."""
    
    def __init__(self):
        self.code_analyzer = StaticCodeAnalyzer()
        self.dependency_validator = DependencyValidator()
        self.threat_detector = ThreatDetector()
        
        # Policy engine statistics
        self.stats = {
            'assessments_performed': 0,
            'threats_detected': 0,
            'executions_blocked': 0,
            'vulnerabilities_found': 0
        }
    
    async def evaluate_execution_request(
        self,
        code: str,
        dependencies: Optional[List[str]] = None,
        language: Optional[str] = None
    ) -> SecurityAssessment:
        """Perform comprehensive security assessment of execution request."""
        
        self.stats['assessments_performed'] += 1
        
        # Detect language if not provided
        if language is None:
            detected_language = self.code_analyzer.detect_language(code)
        else:
            detected_language = CodeLanguage(language.lower())
        
        # Perform static code analysis
        if detected_language == CodeLanguage.PYTHON:
            code_analysis = self.code_analyzer.analyze_python_code(code)
        elif detected_language == CodeLanguage.JAVASCRIPT:
            code_analysis = self.code_analyzer.analyze_javascript_code(code)
        elif detected_language in [CodeLanguage.BASH, CodeLanguage.SHELL]:
            code_analysis = self.code_analyzer.analyze_bash_code(code)
        else:
            # Unknown language - create basic analysis
            code_analysis = CodeAnalysis(language=detected_language)
            code_analysis.lines_of_code = len(code.splitlines())
        
        # Create security assessment
        assessment = SecurityAssessment(code_analysis=code_analysis)
        
        # Detect security threats
        violations = self.threat_detector.detect_threats(code, code_analysis)
        for violation in violations:
            assessment.add_violation(violation)
        
        if violations:
            self.stats['threats_detected'] += len(violations)
        
        # Validate dependencies if provided
        if dependencies:
            language_str = detected_language.value if detected_language != CodeLanguage.UNKNOWN else 'python'
            vulnerabilities = await self.dependency_validator.validate_dependencies(
                dependencies, language_str
            )
            for vulnerability in vulnerabilities:
                assessment.add_vulnerability(vulnerability)
            
            if vulnerabilities:
                self.stats['vulnerabilities_found'] += len(vulnerabilities)
        
        # Generate recommendations and security configuration
        assessment.recommendations = self._generate_recommendations(assessment)
        assessment.auto_fixes = self._generate_auto_fixes(assessment)
        assessment.resource_requirements = self._determine_resource_requirements(assessment)
        assessment.security_config = self._determine_security_config(assessment)
        
        # Check if execution should be blocked
        if assessment.sandboxing_level == SandboxingLevel.BLOCKED:
            self.stats['executions_blocked'] += 1
        
        logger.info(f"Security assessment completed: Risk={assessment.risk_score:.2f}, "
                   f"Level={assessment.threat_level.value}, Sandbox={assessment.sandboxing_level.value}")
        
        return assessment
    
    def _generate_recommendations(self, assessment: SecurityAssessment) -> List[str]:
        """Generate security recommendations based on assessment."""
        recommendations = []
        
        # General recommendations based on threat level
        if assessment.threat_level == ThreatLevel.CRITICAL:
            recommendations.append("Code contains critical security threats - execution should be blocked")
            recommendations.append("Review all imports and system calls for necessity")
            recommendations.append("Consider rewriting code to avoid dangerous operations")
        
        elif assessment.threat_level == ThreatLevel.HIGH:
            recommendations.append("Code requires isolated sandboxed execution")
            recommendations.append("Implement strict resource limits and monitoring")
            recommendations.append("Review dangerous function calls and imports")
        
        elif assessment.threat_level == ThreatLevel.MEDIUM:
            recommendations.append("Code should run in standard sandbox environment")
            recommendations.append("Monitor resource usage during execution")
            recommendations.append("Validate all inputs and outputs")
        
        # Specific recommendations based on violations
        violation_types = set(v.violation_type for v in assessment.violations)
        
        if ViolationType.DANGEROUS_IMPORT in violation_types:
            recommendations.append("Remove unnecessary dangerous imports")
            recommendations.append("Use safer alternatives for system operations")
        
        if ViolationType.DANGEROUS_BUILTIN in violation_types:
            recommendations.append("Avoid using eval/exec functions")
            recommendations.append("Implement input validation for dynamic code")
        
        if ViolationType.SUBPROCESS_EXECUTION in violation_types:
            recommendations.append("Replace subprocess calls with safer alternatives")
            recommendations.append("Implement strict command validation if subprocess is necessary")
        
        # Recommendations based on vulnerabilities
        if assessment.vulnerabilities:
            recommendations.append("Update vulnerable dependencies to fixed versions")
            recommendations.append("Remove unused dependencies to reduce attack surface")
        
        return recommendations
    
    def _generate_auto_fixes(self, assessment: SecurityAssessment) -> List[str]:
        """Generate automatic fixes for common security issues."""
        auto_fixes = []
        
        # Auto-fixes for specific violations
        for violation in assessment.violations:
            if violation.auto_fix:
                auto_fixes.append(violation.auto_fix)
            elif violation.violation_type == ViolationType.DANGEROUS_IMPORT:
                auto_fixes.append(f"Consider removing: {violation.code_snippet}")
            elif violation.violation_type == ViolationType.DANGEROUS_BUILTIN:
                auto_fixes.append(f"Replace {violation.code_snippet} with safer alternative")
        
        # Auto-fixes for vulnerabilities
        for vulnerability in assessment.vulnerabilities:
            if vulnerability.fixed_versions:
                fixed_version = vulnerability.fixed_versions[0]
                auto_fixes.append(f"Update {vulnerability.package_name} to version {fixed_version}")
        
        return auto_fixes
    
    def _determine_resource_requirements(self, assessment: SecurityAssessment) -> ResourceLimits:
        """Determine appropriate resource limits based on assessment."""
        analysis = assessment.code_analysis
        
        # Base limits
        memory_mb = max(64, analysis.estimated_memory_mb)
        cpu_cores = 0.5
        timeout = max(10, int(analysis.estimated_runtime * 3))  # 3x estimated runtime
        
        # Adjust based on threat level
        if assessment.threat_level == ThreatLevel.CRITICAL:
            memory_mb = min(memory_mb, 128)  # Cap at 128MB
            cpu_cores = 0.25  # Limit CPU
            timeout = min(timeout, 15)  # Cap at 15 seconds
        elif assessment.threat_level == ThreatLevel.HIGH:
            memory_mb = min(memory_mb, 256)  # Cap at 256MB
            cpu_cores = 0.5
            timeout = min(timeout, 30)  # Cap at 30 seconds
        elif assessment.threat_level == ThreatLevel.MEDIUM:
            memory_mb = min(memory_mb, 512)  # Cap at 512MB
            cpu_cores = 0.75
            timeout = min(timeout, 60)  # Cap at 60 seconds
        
        return ResourceLimits(
            memory_mb=memory_mb,
            cpu_cores=cpu_cores,
            execution_timeout=timeout,
            cpu_quota=int(cpu_cores * 100000),
            pids_limit=50 if assessment.threat_level in [ThreatLevel.CRITICAL, ThreatLevel.HIGH] else 100,
            disk_limit_mb=min(500, memory_mb * 2)
        )
    
    def _determine_security_config(self, assessment: SecurityAssessment) -> SecurityConfig:
        """Determine appropriate security configuration based on assessment."""
        
        # Start with base configuration
        config = SecurityConfig()
        
        # Adjust based on threat level
        if assessment.threat_level == ThreatLevel.CRITICAL:
            config = SecurityConfig(
                read_only_root=True,
                no_new_privileges=True,
                drop_all_capabilities=True,
                allowed_capabilities=[],  # No capabilities
                user_namespace=True,
                network_isolation=True
            )
        elif assessment.threat_level == ThreatLevel.HIGH:
            config = SecurityConfig(
                read_only_root=True,
                no_new_privileges=True,
                drop_all_capabilities=True,
                allowed_capabilities=['SETUID', 'SETGID'],
                user_namespace=True,
                network_isolation=True
            )
        
        # Adjust based on specific violations
        for violation in assessment.violations:
            if violation.violation_type == ViolationType.NETWORK_ACCESS:
                config.network_isolation = True
            elif violation.violation_type == ViolationType.FILE_SYSTEM_ACCESS:
                config.read_only_root = True
        
        return config
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get policy engine statistics."""
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset policy engine statistics."""
        self.stats = {
            'assessments_performed': 0,
            'threats_detected': 0,
            'executions_blocked': 0,
            'vulnerabilities_found': 0
        }


# Utility functions
async def analyze_code_security(
    code: str,
    dependencies: Optional[List[str]] = None,
    language: Optional[str] = None
) -> SecurityAssessment:
    """Convenience function for security analysis."""
    engine = SecurityPolicyEngine()
    return await engine.evaluate_execution_request(code, dependencies, language)


def create_security_assessment_report(assessment: SecurityAssessment) -> str:
    """Generate human-readable security assessment report."""
    lines = []
    lines.append(f"Security Assessment Report")
    lines.append("=" * 50)
    lines.append(f"Language: {assessment.code_analysis.language.value}")
    lines.append(f"Risk Score: {assessment.risk_score:.2f}/10")
    lines.append(f"Threat Level: {assessment.threat_level.value.upper()}")
    lines.append(f"Sandboxing Level: {assessment.sandboxing_level.value.upper()}")
    lines.append(f"Execution Allowed: {'YES' if assessment.execution_allowed else 'NO'}")
    lines.append("")
    
    if assessment.violations:
        lines.append(f"Security Violations ({len(assessment.violations)}):")
        for i, violation in enumerate(assessment.violations, 1):
            lines.append(f"{i}. {violation.description} (Severity: {violation.severity.value})")
        lines.append("")
    
    if assessment.vulnerabilities:
        lines.append(f"Dependency Vulnerabilities ({len(assessment.vulnerabilities)}):")
        for i, vuln in enumerate(assessment.vulnerabilities, 1):
            lines.append(f"{i}. {vuln.package_name}: {vuln.description} (Severity: {vuln.severity.value})")
        lines.append("")
    
    if assessment.recommendations:
        lines.append("Recommendations:")
        for i, rec in enumerate(assessment.recommendations, 1):
            lines.append(f"{i}. {rec}")
        lines.append("")
    
    return "\n".join(lines)


# Export main classes
__all__ = [
    'SecurityPolicyEngine',
    'StaticCodeAnalyzer', 
    'DependencyValidator',
    'ThreatDetector',
    'SecurityAssessment',
    'SecurityViolation',
    'Vulnerability',
    'CodeAnalysis',
    'ThreatLevel',
    'ViolationType',
    'CodeLanguage',
    'analyze_code_security',
    'create_security_assessment_report'
]