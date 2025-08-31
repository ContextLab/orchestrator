"""Tool library for Orchestrator framework."""

from .base import Tool, ToolRegistry
from .data_tools import DataProcessingTool
from .validation import ValidationTool  # Import from new validation module
from .report_tools import PDFCompilerTool, ReportGeneratorTool
from .system_tools import FileSystemTool, TerminalTool
from .web_tools import HeadlessBrowserTool, WebSearchTool
from .llm_tools import TaskDelegationTool, MultiModelRoutingTool, PromptOptimizationTool
from .code_execution import PythonExecutorTool
from .checkpoint_tool import CheckpointTool
from .visualization_tools import VisualizationTool

# Enhanced registry system - Issue #312
from .registry import (
    EnhancedToolRegistry,
    EnhancedToolMetadata,
    VersionInfo,
    CompatibilityRequirement,
    SecurityPolicy,
    InstallationRequirement,
    RegistrationStatus,
    SecurityLevel,
    InstallationStatus,
    enhanced_registry,
    get_enhanced_registry,
    register_tool_simple,
    discover_tools_for_action,
    check_tool_compatibility
)

# Universal registry system
from .universal_registry import (
    UniversalToolRegistry,
    ToolSource,
    ToolCategory,
    ToolMetadata,
    ToolExecutionResult,
    universal_registry,
    get_universal_registry
)

# Discovery engine
from .discovery import (
    ToolDiscoveryEngine,
    ToolMatch
)

# Setup and installation system - Issue #312 Stream B
from .setup import (
    SetupSystem,
    PlatformDetector,
    ConfigurationManager,
    SetupConfiguration,
    PlatformInfo,
    InstallationResult,
    Platform,
    PackageManager,
    setup_system,
    get_setup_system,
    setup_tool,
    setup_tools,
    check_tool_availability
)

from .installers import (
    PackageInstaller,
    PipInstaller,
    CondaInstaller,
    NpmInstaller,
    AptInstaller,
    HomebrewInstaller,
    ChocolateyInstaller,
    WingetInstaller,
    PackageInstallerFactory,
    ConcurrentInstaller,
    PackageInfo,
    InstallationEnvironment
)

__all__ = [
    # Base tools and registry
    "Tool",
    "ToolRegistry",
    "HeadlessBrowserTool",
    "WebSearchTool",
    "TerminalTool",
    "FileSystemTool",
    "DataProcessingTool",
    "ValidationTool",
    "ReportGeneratorTool",
    "PDFCompilerTool",
    "TaskDelegationTool",
    "MultiModelRoutingTool",
    "PromptOptimizationTool",
    "PythonExecutorTool",
    "CheckpointTool",
    "VisualizationTool",
    
    # Enhanced registry system - Issue #312
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
    "check_tool_compatibility",
    
    # Universal registry system
    "UniversalToolRegistry",
    "ToolSource",
    "ToolCategory",
    "ToolMetadata",
    "ToolExecutionResult",
    "universal_registry",
    "get_universal_registry",
    
    # Discovery engine
    "ToolDiscoveryEngine",
    "ToolMatch",
    
    # Setup and installation system - Issue #312 Stream B
    "SetupSystem",
    "PlatformDetector",
    "ConfigurationManager",
    "SetupConfiguration",
    "PlatformInfo",
    "InstallationResult",
    "Platform",
    "PackageManager",
    "setup_system",
    "get_setup_system",
    "setup_tool",
    "setup_tools",
    "check_tool_availability",
    
    # Package installers
    "PackageInstaller",
    "PipInstaller",
    "CondaInstaller",
    "NpmInstaller",
    "AptInstaller",
    "HomebrewInstaller",
    "ChocolateyInstaller",
    "WingetInstaller",
    "PackageInstallerFactory",
    "ConcurrentInstaller",
    "PackageInfo",
    "InstallationEnvironment",
]
