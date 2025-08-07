"""Security Layer - Phase 3 Advanced Features

Security capabilities for the orchestrator including:
- LangChain Sandbox with Docker isolation
- Secure code execution with resource limits
- Security policy enforcement
- Dependency management and validation
- Container hardening and isolation
"""

from .langchain_sandbox import (
    LangChainSandbox,
    SandboxType,
    SecurityPolicy,
    ExecutionResult,
    SandboxConfig,
    SecurePythonExecutor,
    create_secure_sandbox,
    execute_code_safely
)

__all__ = [
    "LangChainSandbox",
    "SandboxType",
    "SecurityPolicy",
    "ExecutionResult",
    "SandboxConfig",
    "SecurePythonExecutor",
    "create_secure_sandbox",
    "execute_code_safely"
]