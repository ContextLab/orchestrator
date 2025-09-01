"""Enhanced Tool Registry System - Issue #312

Comprehensive tool registry system providing:
- Tool discovery and management
- Version management and compatibility checking
- Extensible registry design for easy tool addition
- Security considerations for tool registration
- Centralized coordination for automatic setup and dependency management
"""

import logging
from typing import Dict, List, Any, Optional, Set, Union, Tuple, Type, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
from packaging import version
import asyncio
import json
import hashlib
from pathlib import Path

from .base import Tool as OrchestratorTool, ToolParameter
from .universal_registry import (
    UniversalToolRegistry, ToolSource, ToolCategory, ToolMetadata,
    ToolExecutionResult, get_universal_registry
)
from .discovery import ToolDiscoveryEngine, ToolMatch

logger = logging.getLogger(__name__)


class RegistrationStatus(Enum):
    """Status of tool registration."""
    PENDING = "pending"
    REGISTERED = "registered"
    DEPRECATED = "deprecated"
    BLOCKED = "blocked"
    MIGRATED = "migrated"


class SecurityLevel(Enum):
    """Security levels for tool execution."""
    STRICT = "strict"      # Maximum security, sandboxed execution
    MODERATE = "moderate"  # Default security, input validation
    PERMISSIVE = "permissive"  # Minimal restrictions
    TRUSTED = "trusted"    # System tools, no restrictions


class InstallationStatus(Enum):
    """Installation status for tools."""
    AVAILABLE = "available"        # Ready to use
    NEEDS_INSTALL = "needs_install"  # Requires installation
    INSTALLING = "installing"      # Currently being installed
    FAILED = "failed"             # Installation failed
    UNAVAILABLE = "unavailable"   # Cannot be installed


@dataclass
class VersionInfo:
    """Version information for tools."""
    major: int
    minor: int
    patch: int
    pre_release: Optional[str] = None
    build_metadata: Optional[str] = None
    
    def __str__(self) -> str:
        """String representation of version."""
        version_str = f"{self.major}.{self.minor}.{self.patch}"
        if self.pre_release:
            version_str += f"-{self.pre_release}"
        if self.build_metadata:
            version_str += f"+{self.build_metadata}"
        return version_str
    
    def __lt__(self, other: 'VersionInfo') -> bool:
        """Compare versions for sorting."""
        return version.parse(str(self)) < version.parse(str(other))
    
    def __le__(self, other: 'VersionInfo') -> bool:
        return version.parse(str(self)) <= version.parse(str(other))
    
    def __gt__(self, other: 'VersionInfo') -> bool:
        return version.parse(str(self)) > version.parse(str(other))
    
    def __ge__(self, other: 'VersionInfo') -> bool:
        return version.parse(str(self)) >= version.parse(str(other))
    
    def __eq__(self, other: 'VersionInfo') -> bool:
        return version.parse(str(self)) == version.parse(str(other))
    
    @classmethod
    def parse(cls, version_string: str) -> 'VersionInfo':
        """Parse version string into VersionInfo."""
        parsed = version.parse(version_string)
        return cls(
            major=parsed.major,
            minor=parsed.minor,
            patch=parsed.micro,
            pre_release=str(parsed.pre) if parsed.pre else None,
            build_metadata=str(parsed.local) if parsed.local else None
        )


@dataclass
class CompatibilityRequirement:
    """Compatibility requirement for tools."""
    name: str
    min_version: Optional[VersionInfo] = None
    max_version: Optional[VersionInfo] = None
    required: bool = True
    reason: str = ""
    
    def is_compatible(self, available_version: VersionInfo) -> bool:
        """Check if available version meets requirement."""
        if self.min_version and available_version < self.min_version:
            return False
        if self.max_version and available_version > self.max_version:
            return False
        return True


@dataclass
class SecurityPolicy:
    """Security policy for tool execution."""
    level: SecurityLevel
    allowed_operations: List[str] = field(default_factory=list)
    blocked_operations: List[str] = field(default_factory=list)
    sandboxed: bool = False
    network_access: bool = True
    file_system_access: bool = True
    max_execution_time: Optional[int] = None  # seconds
    max_memory_usage: Optional[int] = None    # MB
    environment_variables: Dict[str, str] = field(default_factory=dict)


@dataclass  
class InstallationRequirement:
    """Installation requirement for tools."""
    package_manager: str  # pip, npm, apt, etc.
    package_name: str
    version_spec: Optional[str] = None
    install_command: Optional[str] = None
    post_install_command: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    environment_setup: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnhancedToolMetadata(ToolMetadata):
    """Enhanced metadata extending the base ToolMetadata."""
    
    # Versioning information
    version_info: VersionInfo = field(default_factory=lambda: VersionInfo(1, 0, 0))
    compatibility_requirements: List[CompatibilityRequirement] = field(default_factory=list)
    
    # Registration information
    registration_status: RegistrationStatus = RegistrationStatus.PENDING
    registration_time: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    deprecation_notice: Optional[str] = None
    
    # Security information
    security_policy: Optional[SecurityPolicy] = None
    security_hash: Optional[str] = None
    
    # Installation information
    installation_status: InstallationStatus = InstallationStatus.AVAILABLE
    installation_requirements: List[InstallationRequirement] = field(default_factory=list)
    
    # Extensibility information
    plugin_interface: Optional[str] = None
    extension_points: List[str] = field(default_factory=list)
    configuration_schema: Optional[Dict[str, Any]] = None
    
    # Tool relationships
    provides: List[str] = field(default_factory=list)  # What capabilities this tool provides
    requires: List[str] = field(default_factory=list)  # What capabilities this tool needs
    conflicts_with: List[str] = field(default_factory=list)  # Tools that conflict
    supersedes: List[str] = field(default_factory=list)  # Tools this replaces
    
    # Performance metrics
    avg_execution_time: float = 0.0
    success_rate: float = 0.0
    usage_count: int = 0
    
    def generate_security_hash(self) -> str:
        """Generate security hash for tool integrity."""
        # Create hash from critical metadata
        hash_data = {
            "name": self.name,
            "version": str(self.version_info),
            "source": self.source.value,
            "security_level": self.security_level
        }
        hash_str = json.dumps(hash_data, sort_keys=True)
        return hashlib.sha256(hash_str.encode()).hexdigest()[:16]


class EnhancedToolRegistry(UniversalToolRegistry):
    """Enhanced tool registry with versioning, compatibility checking, and extensible design.
    
    Features:
    - Comprehensive tool discovery and management
    - Version management and compatibility checking
    - Extensible registry design for easy tool addition
    - Security considerations for tool registration
    - Installation requirement management
    - Tool relationship tracking
    - Performance monitoring
    """
    
    def __init__(self):
        super().__init__()
        
        # Enhanced metadata storage
        self.enhanced_metadata: Dict[str, EnhancedToolMetadata] = {}
        
        # Version management
        self.version_registry: Dict[str, Dict[str, VersionInfo]] = {}  # tool_name -> {version -> VersionInfo}
        self.compatibility_cache: Dict[str, Dict[str, bool]] = {}  # cache compatibility checks
        
        # Security management
        self.security_policies: Dict[SecurityLevel, SecurityPolicy] = self._create_default_security_policies()
        self.blocked_tools: Set[str] = set()
        
        # Installation management
        self.installation_tracker: Dict[str, InstallationStatus] = {}
        self.installation_callbacks: Dict[str, List[Callable]] = {}
        
        # Extensibility support
        self.plugin_interfaces: Dict[str, Type] = {}
        self.extension_registry: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.performance_metrics: Dict[str, Dict[str, float]] = {}
        
        # Tool discovery engine
        self.discovery_engine = ToolDiscoveryEngine(self)
        
        logger.info("Enhanced Tool Registry initialized")
    
    def _create_default_security_policies(self) -> Dict[SecurityLevel, SecurityPolicy]:
        """Create default security policies for different levels."""
        return {
            SecurityLevel.STRICT: SecurityPolicy(
                level=SecurityLevel.STRICT,
                allowed_operations=["read", "analyze", "validate"],
                blocked_operations=["execute", "network", "file_write"],
                sandboxed=True,
                network_access=False,
                file_system_access=False,
                max_execution_time=30,
                max_memory_usage=256
            ),
            SecurityLevel.MODERATE: SecurityPolicy(
                level=SecurityLevel.MODERATE,
                allowed_operations=["read", "write", "analyze", "process"],
                blocked_operations=["system_call", "admin_access"],
                sandboxed=False,
                network_access=True,
                file_system_access=True,
                max_execution_time=300,
                max_memory_usage=1024
            ),
            SecurityLevel.PERMISSIVE: SecurityPolicy(
                level=SecurityLevel.PERMISSIVE,
                allowed_operations=["*"],
                blocked_operations=["admin_access"],
                sandboxed=False,
                network_access=True,
                file_system_access=True,
                max_execution_time=1800,
                max_memory_usage=4096
            ),
            SecurityLevel.TRUSTED: SecurityPolicy(
                level=SecurityLevel.TRUSTED,
                allowed_operations=["*"],
                blocked_operations=[],
                sandboxed=False,
                network_access=True,
                file_system_access=True,
                max_execution_time=None,
                max_memory_usage=None
            )
        }
    
    # Enhanced registration methods
    
    def register_tool_enhanced(
        self,
        tool: OrchestratorTool,
        version_info: Optional[VersionInfo] = None,
        category: ToolCategory = ToolCategory.CUSTOM,
        security_level: SecurityLevel = SecurityLevel.MODERATE,
        installation_requirements: Optional[List[InstallationRequirement]] = None,
        compatibility_requirements: Optional[List[CompatibilityRequirement]] = None,
        provides: Optional[List[str]] = None,
        requires: Optional[List[str]] = None,
        conflicts_with: Optional[List[str]] = None,
        **kwargs
    ) -> bool:
        """Enhanced tool registration with full metadata support."""
        
        # Check if tool already exists
        if tool.name in self.enhanced_metadata:
            existing = self.enhanced_metadata[tool.name]
            if existing.registration_status == RegistrationStatus.BLOCKED:
                logger.error(f"Tool {tool.name} is blocked from registration")
                return False
        
        # Create enhanced metadata
        metadata = EnhancedToolMetadata(
            name=tool.name,
            source=ToolSource.ORCHESTRATOR,
            category=category,
            description=tool.description,
            version_info=version_info or VersionInfo(1, 0, 0),
            compatibility_requirements=compatibility_requirements or [],
            security_policy=self.security_policies[security_level],
            installation_requirements=installation_requirements or [],
            provides=provides or [],
            requires=requires or [],
            conflicts_with=conflicts_with or [],
            registration_time=datetime.now(),
            registration_status=RegistrationStatus.PENDING,
            **kwargs
        )
        
        # Generate security hash
        metadata.security_hash = metadata.generate_security_hash()
        
        # Validate compatibility requirements
        if not self._validate_compatibility_requirements(tool.name, metadata):
            logger.error(f"Compatibility validation failed for tool {tool.name}")
            return False
        
        # Check for conflicts
        conflicts = self._check_tool_conflicts(tool.name, metadata)
        if conflicts:
            logger.warning(f"Tool {tool.name} has conflicts: {conflicts}")
            # Could be handled differently based on policy
        
        # Register with base registry
        self.register_orchestrator_tool(tool, category, security_level=security_level.value)
        
        # Store enhanced metadata
        self.enhanced_metadata[tool.name] = metadata
        
        # Update version registry
        if tool.name not in self.version_registry:
            self.version_registry[tool.name] = {}
        self.version_registry[tool.name][str(metadata.version_info)] = metadata.version_info
        
        # Mark as registered
        metadata.registration_status = RegistrationStatus.REGISTERED
        metadata.last_updated = datetime.now()
        
        # Set installation status
        self.installation_tracker[tool.name] = self._determine_installation_status(metadata)
        
        logger.info(f"Enhanced tool registration completed: {tool.name} v{metadata.version_info}")
        return True
    
    def register_plugin_interface(self, interface_name: str, interface_class: Type) -> None:
        """Register a plugin interface for extensibility."""
        self.plugin_interfaces[interface_name] = interface_class
        logger.info(f"Registered plugin interface: {interface_name}")
    
    def register_extension(self, tool_name: str, extension_name: str, extension_config: Dict[str, Any]) -> bool:
        """Register an extension for a tool."""
        if tool_name not in self.enhanced_metadata:
            logger.error(f"Cannot register extension for unknown tool: {tool_name}")
            return False
        
        if tool_name not in self.extension_registry:
            self.extension_registry[tool_name] = {}
        
        self.extension_registry[tool_name][extension_name] = extension_config
        logger.info(f"Registered extension {extension_name} for tool {tool_name}")
        return True
    
    # Version management methods
    
    def get_tool_versions(self, tool_name: str) -> List[VersionInfo]:
        """Get all available versions of a tool."""
        if tool_name not in self.version_registry:
            return []
        return sorted(self.version_registry[tool_name].values())
    
    def get_latest_version(self, tool_name: str) -> Optional[VersionInfo]:
        """Get the latest version of a tool."""
        versions = self.get_tool_versions(tool_name)
        return versions[-1] if versions else None
    
    def is_version_compatible(self, tool_name: str, required_version: str) -> bool:
        """Check if a specific version is compatible with current system."""
        if tool_name not in self.enhanced_metadata:
            return False
        
        # Use cached result if available
        cache_key = f"{tool_name}:{required_version}"
        if cache_key in self.compatibility_cache:
            return self.compatibility_cache[cache_key]
        
        metadata = self.enhanced_metadata[tool_name]
        required = VersionInfo.parse(required_version)
        
        # Check compatibility requirements
        for req in metadata.compatibility_requirements:
            if not req.is_compatible(required):
                self.compatibility_cache[cache_key] = False
                return False
        
        self.compatibility_cache[cache_key] = True
        return True
    
    def upgrade_tool(self, tool_name: str, target_version: Optional[str] = None) -> bool:
        """Upgrade a tool to a specific or latest version."""
        if tool_name not in self.enhanced_metadata:
            logger.error(f"Cannot upgrade unknown tool: {tool_name}")
            return False
        
        current_metadata = self.enhanced_metadata[tool_name]
        
        if target_version:
            if not self.is_version_compatible(tool_name, target_version):
                logger.error(f"Target version {target_version} is not compatible for {tool_name}")
                return False
            target = VersionInfo.parse(target_version)
        else:
            target = self.get_latest_version(tool_name)
            if not target:
                logger.error(f"No versions available for tool: {tool_name}")
                return False
        
        if current_metadata.version_info >= target:
            logger.info(f"Tool {tool_name} is already at or above target version {target}")
            return True
        
        # Perform upgrade (placeholder - would need actual upgrade logic)
        logger.info(f"Upgrading {tool_name} from {current_metadata.version_info} to {target}")
        current_metadata.version_info = target
        current_metadata.last_updated = datetime.now()
        
        return True
    
    def deprecate_tool(self, tool_name: str, reason: str, replacement: Optional[str] = None) -> bool:
        """Deprecate a tool."""
        if tool_name not in self.enhanced_metadata:
            return False
        
        metadata = self.enhanced_metadata[tool_name]
        metadata.registration_status = RegistrationStatus.DEPRECATED
        metadata.deprecation_notice = reason
        if replacement:
            metadata.supersedes = [replacement]
        metadata.last_updated = datetime.now()
        
        logger.info(f"Deprecated tool {tool_name}: {reason}")
        return True
    
    # Compatibility checking methods
    
    def _validate_compatibility_requirements(self, tool_name: str, metadata: EnhancedToolMetadata) -> bool:
        """Validate all compatibility requirements for a tool."""
        for req in metadata.compatibility_requirements:
            if req.required:
                # Check if required tool/capability is available
                if req.name not in self.enhanced_metadata:
                    logger.error(f"Required dependency {req.name} not available for {tool_name}")
                    return False
                
                # Check version compatibility
                dep_metadata = self.enhanced_metadata[req.name]
                if not req.is_compatible(dep_metadata.version_info):
                    logger.error(f"Version compatibility failed: {tool_name} requires {req.name} {req.min_version}-{req.max_version}")
                    return False
        
        return True
    
    def _check_tool_conflicts(self, tool_name: str, metadata: EnhancedToolMetadata) -> List[str]:
        """Check for conflicts with existing tools."""
        conflicts = []
        
        for conflict_name in metadata.conflicts_with:
            if conflict_name in self.enhanced_metadata:
                existing = self.enhanced_metadata[conflict_name]
                if existing.registration_status == RegistrationStatus.REGISTERED:
                    conflicts.append(conflict_name)
        
        return conflicts
    
    def check_system_compatibility(self) -> Dict[str, Any]:
        """Check compatibility of all registered tools with current system."""
        compatibility_report = {
            "compatible": [],
            "incompatible": [],
            "warnings": [],
            "missing_dependencies": []
        }
        
        for tool_name, metadata in self.enhanced_metadata.items():
            if metadata.registration_status != RegistrationStatus.REGISTERED:
                continue
            
            # Check compatibility requirements
            missing_deps = []
            for req in metadata.compatibility_requirements:
                if req.required and req.name not in self.enhanced_metadata:
                    missing_deps.append(req.name)
            
            if missing_deps:
                compatibility_report["missing_dependencies"].append({
                    "tool": tool_name,
                    "missing": missing_deps
                })
                compatibility_report["incompatible"].append(tool_name)
            else:
                compatibility_report["compatible"].append(tool_name)
        
        return compatibility_report
    
    # Installation management methods
    
    def _determine_installation_status(self, metadata: EnhancedToolMetadata) -> InstallationStatus:
        """Determine the installation status of a tool."""
        if not metadata.installation_requirements:
            return InstallationStatus.AVAILABLE
        
        # Check if all requirements are satisfied
        for req in metadata.installation_requirements:
            # Placeholder - would implement actual installation checking
            # Could check if packages are installed, commands are available, etc.
            pass
        
        return InstallationStatus.AVAILABLE
    
    def install_tool_requirements(self, tool_name: str) -> bool:
        """Install requirements for a tool."""
        if tool_name not in self.enhanced_metadata:
            return False
        
        metadata = self.enhanced_metadata[tool_name]
        if not metadata.installation_requirements:
            return True
        
        self.installation_tracker[tool_name] = InstallationStatus.INSTALLING
        
        try:
            for req in metadata.installation_requirements:
                logger.info(f"Installing requirement: {req.package_name}")
                # Placeholder - would implement actual installation logic
                # Could use subprocess to run pip, npm, etc.
                pass
            
            self.installation_tracker[tool_name] = InstallationStatus.AVAILABLE
            metadata.installation_status = InstallationStatus.AVAILABLE
            
            # Notify callbacks
            if tool_name in self.installation_callbacks:
                for callback in self.installation_callbacks[tool_name]:
                    try:
                        callback(tool_name, InstallationStatus.AVAILABLE)
                    except Exception as e:
                        logger.error(f"Installation callback failed: {e}")
            
            return True
        
        except Exception as e:
            logger.error(f"Installation failed for {tool_name}: {e}")
            self.installation_tracker[tool_name] = InstallationStatus.FAILED
            metadata.installation_status = InstallationStatus.FAILED
            return False
    
    def register_installation_callback(self, tool_name: str, callback: Callable) -> None:
        """Register callback for installation completion."""
        if tool_name not in self.installation_callbacks:
            self.installation_callbacks[tool_name] = []
        self.installation_callbacks[tool_name].append(callback)
    
    # Discovery and query methods
    
    def discover_tools_advanced(
        self,
        action_description: Optional[str] = None,
        category: Optional[ToolCategory] = None,
        version_constraints: Optional[Dict[str, str]] = None,
        security_level: Optional[SecurityLevel] = None,
        installation_status: Optional[InstallationStatus] = None,
        capabilities: Optional[List[str]] = None,
        exclude_deprecated: bool = True
    ) -> List[Dict[str, Any]]:
        """Advanced tool discovery with multiple filtering options."""
        
        candidates = []
        
        # Start with all registered tools
        for tool_name, metadata in self.enhanced_metadata.items():
            
            # Skip deprecated tools if requested
            if exclude_deprecated and metadata.registration_status == RegistrationStatus.DEPRECATED:
                continue
            
            # Filter by category
            if category and metadata.category != category:
                continue
            
            # Filter by security level
            if security_level and metadata.security_policy.level != security_level:
                continue
            
            # Filter by installation status
            if installation_status:
                current_status = self.installation_tracker.get(tool_name, InstallationStatus.AVAILABLE)
                if current_status != installation_status:
                    continue
            
            # Filter by version constraints
            if version_constraints and tool_name in version_constraints:
                constraint = version_constraints[tool_name]
                if not self.is_version_compatible(tool_name, constraint):
                    continue
            
            # Filter by capabilities
            if capabilities:
                provided_capabilities = set(metadata.provides)
                required_capabilities = set(capabilities)
                if not required_capabilities.issubset(provided_capabilities):
                    continue
            
            # Build result
            tool_info = {
                "name": tool_name,
                "metadata": asdict(metadata),
                "installation_status": self.installation_tracker.get(tool_name, InstallationStatus.AVAILABLE).value,
                "extensions": self.extension_registry.get(tool_name, {}),
                "performance": self.performance_metrics.get(tool_name, {})
            }
            
            candidates.append(tool_info)
        
        # If action description provided, use discovery engine to rank
        if action_description:
            matches = self.discovery_engine.discover_tools_for_action(action_description)
            match_scores = {match.tool_name: match.confidence for match in matches}
            
            # Sort by discovery confidence
            candidates.sort(key=lambda x: match_scores.get(x["name"], 0), reverse=True)
        
        return candidates
    
    def get_tool_chain_enhanced(
        self,
        action_description: str,
        constraints: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Get enhanced tool chain with dependency resolution."""
        
        # Get basic tool chain from discovery engine
        basic_chain = self.discovery_engine.get_tool_chain_for_action(action_description)
        
        enhanced_chain = []
        for match in basic_chain:
            tool_name = match.tool_name
            
            if tool_name not in self.enhanced_metadata:
                continue
            
            metadata = self.enhanced_metadata[tool_name]
            
            # Resolve dependencies
            dependencies = []
            for dep_name in metadata.requires:
                if dep_name in self.enhanced_metadata:
                    dependencies.append({
                        "name": dep_name,
                        "metadata": asdict(self.enhanced_metadata[dep_name]),
                        "required": True
                    })
            
            enhanced_chain.append({
                "tool": match,
                "metadata": asdict(metadata),
                "dependencies": dependencies,
                "installation_status": self.installation_tracker.get(tool_name, InstallationStatus.AVAILABLE).value
            })
        
        return enhanced_chain
    
    # Security methods
    
    def validate_security_policy(self, tool_name: str, operation: str) -> bool:
        """Validate if an operation is allowed by tool's security policy."""
        if tool_name not in self.enhanced_metadata:
            return False
        
        metadata = self.enhanced_metadata[tool_name]
        if not metadata.security_policy:
            return True  # No policy means allowed
        
        policy = metadata.security_policy
        
        # Check blocked operations
        if operation in policy.blocked_operations:
            return False
        
        # Check allowed operations
        if policy.allowed_operations and "*" not in policy.allowed_operations:
            return operation in policy.allowed_operations
        
        return True
    
    def block_tool(self, tool_name: str, reason: str) -> bool:
        """Block a tool from execution."""
        if tool_name not in self.enhanced_metadata:
            return False
        
        self.blocked_tools.add(tool_name)
        metadata = self.enhanced_metadata[tool_name]
        metadata.registration_status = RegistrationStatus.BLOCKED
        metadata.last_updated = datetime.now()
        
        logger.warning(f"Blocked tool {tool_name}: {reason}")
        return True
    
    def unblock_tool(self, tool_name: str) -> bool:
        """Unblock a previously blocked tool."""
        if tool_name in self.blocked_tools:
            self.blocked_tools.remove(tool_name)
            if tool_name in self.enhanced_metadata:
                metadata = self.enhanced_metadata[tool_name]
                metadata.registration_status = RegistrationStatus.REGISTERED
                metadata.last_updated = datetime.now()
            logger.info(f"Unblocked tool {tool_name}")
            return True
        return False
    
    # Performance monitoring methods
    
    def update_performance_metrics(self, tool_name: str, execution_time: float, success: bool) -> None:
        """Update performance metrics for a tool."""
        if tool_name not in self.enhanced_metadata:
            return
        
        metadata = self.enhanced_metadata[tool_name]
        
        # Update metadata metrics
        metadata.usage_count += 1
        metadata.avg_execution_time = (
            (metadata.avg_execution_time * (metadata.usage_count - 1) + execution_time) / metadata.usage_count
        )
        
        if success:
            metadata.success_rate = (
                (metadata.success_rate * (metadata.usage_count - 1) + 1.0) / metadata.usage_count
            )
        else:
            metadata.success_rate = (
                (metadata.success_rate * (metadata.usage_count - 1) + 0.0) / metadata.usage_count
            )
        
        # Update performance metrics dictionary
        if tool_name not in self.performance_metrics:
            self.performance_metrics[tool_name] = {}
        
        self.performance_metrics[tool_name].update({
            "avg_execution_time": metadata.avg_execution_time,
            "success_rate": metadata.success_rate,
            "usage_count": metadata.usage_count,
            "last_updated": datetime.now().isoformat()
        })
    
    # Enhanced execution with monitoring
    
    async def execute_tool_monitored(
        self,
        tool_name: str,
        **kwargs
    ) -> ToolExecutionResult:
        """Execute tool with enhanced monitoring and security checks."""
        
        # Security validation
        if tool_name in self.blocked_tools:
            return ToolExecutionResult(
                success=False,
                output=None,
                source=ToolSource.ORCHESTRATOR,
                execution_time=0.0,
                tool_name=tool_name,
                error="Tool is blocked from execution"
            )
        
        # Check installation status
        install_status = self.installation_tracker.get(tool_name, InstallationStatus.AVAILABLE)
        if install_status != InstallationStatus.AVAILABLE:
            return ToolExecutionResult(
                success=False,
                output=None,
                source=ToolSource.ORCHESTRATOR,
                execution_time=0.0,
                tool_name=tool_name,
                error=f"Tool not available for execution: {install_status.value}"
            )
        
        # Execute with monitoring
        result = await self.execute_tool_enhanced(tool_name, **kwargs)
        
        # Update performance metrics
        self.update_performance_metrics(tool_name, result.execution_time, result.success)
        
        return result
    
    # Export and import methods
    
    def export_registry_state(self) -> Dict[str, Any]:
        """Export complete registry state."""
        return {
            "enhanced_metadata": {
                name: asdict(metadata) for name, metadata in self.enhanced_metadata.items()
            },
            "version_registry": {
                name: {v: asdict(version_info) for v, version_info in versions.items()}
                for name, versions in self.version_registry.items()
            },
            "installation_tracker": {
                name: status.value for name, status in self.installation_tracker.items()
            },
            "performance_metrics": self.performance_metrics,
            "extension_registry": self.extension_registry,
            "blocked_tools": list(self.blocked_tools),
            "export_timestamp": datetime.now().isoformat()
        }
    
    def import_registry_state(self, state: Dict[str, Any]) -> bool:
        """Import registry state from export."""
        try:
            # Import enhanced metadata
            for name, metadata_dict in state.get("enhanced_metadata", {}).items():
                # Convert back to dataclass (simplified)
                self.enhanced_metadata[name] = EnhancedToolMetadata(**metadata_dict)
            
            # Import version registry
            for name, versions in state.get("version_registry", {}).items():
                self.version_registry[name] = {
                    v: VersionInfo(**version_dict) for v, version_dict in versions.items()
                }
            
            # Import other state
            self.installation_tracker = {
                name: InstallationStatus(status) 
                for name, status in state.get("installation_tracker", {}).items()
            }
            self.performance_metrics = state.get("performance_metrics", {})
            self.extension_registry = state.get("extension_registry", {})
            self.blocked_tools = set(state.get("blocked_tools", []))
            
            logger.info("Registry state imported successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import registry state: {e}")
            return False


# Global enhanced registry instance
enhanced_registry = EnhancedToolRegistry()


def get_enhanced_registry() -> EnhancedToolRegistry:
    """Get the global enhanced registry instance."""
    return enhanced_registry


# Convenience functions for common operations

def register_tool_simple(
    tool: OrchestratorTool,
    version: str = "1.0.0",
    category: str = "custom",
    security_level: str = "moderate"
) -> bool:
    """Simple tool registration for basic use cases."""
    return enhanced_registry.register_tool_enhanced(
        tool=tool,
        version_info=VersionInfo.parse(version),
        category=ToolCategory(category.lower()),
        security_level=SecurityLevel(security_level.upper())
    )


def discover_tools_for_action(action: str) -> List[str]:
    """Simple tool discovery for an action."""
    tools = enhanced_registry.discover_tools_advanced(action_description=action)
    return [tool["name"] for tool in tools]


def check_tool_compatibility(tool_name: str, required_version: str) -> bool:
    """Check if a tool version is compatible."""
    return enhanced_registry.is_version_compatible(tool_name, required_version)


__all__ = [
    "EnhancedToolRegistry",
    "EnhancedToolMetadata", 
    "VersionInfo",
    "CompatibilityRequirement",
    "SecurityPolicy",
    "InstallationRequirement",
    "RegistrationStatus",
    "SecurityLevel",
    "InstallationStatus",
    "enhanced_registry",
    "get_enhanced_registry",
    "register_tool_simple",
    "discover_tools_for_action", 
    "check_tool_compatibility"
]