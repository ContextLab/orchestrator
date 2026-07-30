"""Tool library for Orchestrator framework.

Tool implementations are imported lazily (PEP 562). Importing this package no
longer pulls in heavy optional dependencies such as ``requests``, ``bs4``,
``PIL``, ``matplotlib`` or ``anthropic``; each tool module is only loaded when
one of its names is first accessed.

This keeps ``import orchestrator`` lightweight while preserving the historical
API: ``from orchestrator.tools import WebSearchTool`` still works, but it now
raises the underlying ``ImportError`` only if that tool's optional dependency is
genuinely missing.
"""

from .._lazy import lazy_exports

# Map of exported name -> submodule that defines it. Accessing a name imports
# only that submodule.
_EXPORTS: dict[str, str] = {
    # Base tools and registry
    "Tool": "base",
    "ToolRegistry": "base",
    "default_registry": "base",
    # Concrete tools
    "DataProcessingTool": "data_tools",
    "ValidationTool": "validation",
    "ReportGeneratorTool": "report_tools",
    "PDFCompilerTool": "report_tools",
    "FileSystemTool": "system_tools",
    "TerminalTool": "system_tools",
    "HeadlessBrowserTool": "web_tools",
    "WebSearchTool": "web_tools",
    "TaskDelegationTool": "llm_tools",
    "MultiModelRoutingTool": "llm_tools",
    "PromptOptimizationTool": "llm_tools",
    "PythonExecutorTool": "code_execution",
    "CheckpointTool": "checkpoint_tool",
    "VisualizationTool": "visualization_tools",
    # Enhanced registry system - Issue #312
    "EnhancedToolRegistry": "registry",
    "EnhancedToolMetadata": "registry",
    "VersionInfo": "registry",
    "CompatibilityRequirement": "registry",
    "SecurityPolicy": "registry",
    "InstallationRequirement": "registry",
    "RegistrationStatus": "registry",
    "SecurityLevel": "registry",
    "InstallationStatus": "registry",
    "enhanced_registry": "registry",
    "get_enhanced_registry": "registry",
    "register_tool_simple": "registry",
    "discover_tools_for_action": "registry",
    "check_tool_compatibility": "registry",
    # Universal registry system
    "UniversalToolRegistry": "universal_registry",
    "ToolSource": "universal_registry",
    "ToolCategory": "universal_registry",
    "ToolMetadata": "universal_registry",
    "ToolExecutionResult": "universal_registry",
    "universal_registry": "universal_registry",
    "get_universal_registry": "universal_registry",
    # Discovery engine
    "ToolDiscoveryEngine": "discovery",
    "ToolMatch": "discovery",
    # Setup and installation system - Issue #312 Stream B
    "SetupSystem": "setup",
    "PlatformDetector": "setup",
    "ConfigurationManager": "setup",
    "SetupConfiguration": "setup",
    "PlatformInfo": "setup",
    "InstallationResult": "setup",
    "Platform": "setup",
    "PackageManager": "setup",
    "setup_system": "setup",
    "get_setup_system": "setup",
    "setup_tool": "setup",
    "setup_tools": "setup",
    "check_tool_availability": "setup",
    # Package installers
    "PackageInstaller": "installers",
    "PipInstaller": "installers",
    "CondaInstaller": "installers",
    "NpmInstaller": "installers",
    "AptInstaller": "installers",
    "HomebrewInstaller": "installers",
    "ChocolateyInstaller": "installers",
    "WingetInstaller": "installers",
    "PackageInstallerFactory": "installers",
    "ConcurrentInstaller": "installers",
    "PackageInfo": "installers",
    "InstallationEnvironment": "installers",
}

__all__ = sorted(_EXPORTS)
__getattr__, __dir__ = lazy_exports(
    __name__, {k: f".{v}" for k, v in _EXPORTS.items()}, globals()
)
