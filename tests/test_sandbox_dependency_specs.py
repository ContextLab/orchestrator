"""Tests for sandbox dependency-specifier handling.

Dependency strings arrive from pipeline content, which is untrusted. They used
to be interpolated directly into generated source::

    f"import subprocess; subprocess.check_call(['pip', 'install', '{dep}'])"
    f"require('child_process').execSync('npm install {dep}');"

so a crafted specifier ran arbitrary code inside the container. These tests
are adversarial: the interesting cases are the ones that try to break out of
the specifier and into the surrounding program.
"""

import json

import pytest

from orchestrator.security.langchain_sandbox import (
    DependencySpecError,
    SandboxConfig,
    SandboxType,
    SecurityPolicy,
    _NPM_SPEC_RE,
    _PYTHON_SPEC_RE,
    _validate_dependencies,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Specifiers that must be accepted -- refusing these would break real users
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "spec",
    [
        "requests",
        "requests==2.31.0",
        "requests>=2.0",
        "requests>=2.0,<3.0",
        "python-dateutil",
        "ruamel.yaml",
        "uvicorn[standard]",
        "uvicorn[standard]==0.29.0",
        "numpy~=1.26",
        "pkg!=1.0",
        "Django===4.2",
    ],
)
def test_valid_python_specs_are_accepted(spec):
    assert _validate_dependencies([spec], _PYTHON_SPEC_RE, "python") == [spec]


@pytest.mark.parametrize(
    "spec",
    ["express", "express@4.18.2", "@types/node", "@types/node@20", "lodash.merge"],
)
def test_valid_npm_specs_are_accepted(spec):
    assert _validate_dependencies([spec], _NPM_SPEC_RE, "npm") == [spec]


# ---------------------------------------------------------------------------
# Injection attempts that must be refused
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "spec",
    [
        # Break out of the quoted string and into the generated Python.
        "x']); import os; os.system('id')  #",
        "x'); __import__('os').system('id'); ('",
        'x"]); import os; os.system("id")  #',
        # Newline-based statement injection.
        "requests\nimport os",
        "requests\n; print(1)",
        # Installing from a URL/VCS/path executes attacker-controlled setup.py.
        "https://example.com/evil.tar.gz",
        "git+https://example.com/evil.git",
        "./local/evil",
        "/abs/evil",
        "evil @ https://example.com/evil.whl",
        # pip option injection.
        "--index-url=https://evil.example.com/simple",
        "-e .",
        # Shell metacharacters.
        "requests; id",
        "requests && id",
        "requests|id",
        "requests`id`",
        "requests$(id)",
        # Environment markers are a needless extra grammar.
        "requests; python_version<'3.9'",
        # Empty / whitespace.
        "",
        "   ",
    ],
)
def test_python_injection_specs_are_refused(spec):
    with pytest.raises(DependencySpecError):
        _validate_dependencies([spec], _PYTHON_SPEC_RE, "python")


@pytest.mark.parametrize(
    "spec",
    [
        "x'); require('child_process').execSync('id'); ('",
        "express\n; require('fs')",
        "express; id",
        "express && id",
        "express`id`",
        "express$(id)",
        "https://example.com/evil.tgz",
        "git+https://example.com/evil.git",
        "../evil",
        "--registry=https://evil.example.com",
        "",
    ],
)
def test_npm_injection_specs_are_refused(spec):
    with pytest.raises(DependencySpecError):
        _validate_dependencies([spec], _NPM_SPEC_RE, "npm")


def test_non_string_dependency_is_refused():
    with pytest.raises(DependencySpecError):
        _validate_dependencies([{"pkg": "evil"}], _PYTHON_SPEC_RE, "python")


def test_dependency_count_is_bounded():
    with pytest.raises(DependencySpecError, match="maximum"):
        _validate_dependencies(["pkg"] * 500, _PYTHON_SPEC_RE, "python")


def test_validation_fails_closed_rather_than_dropping():
    """A rejected specifier must raise, not be silently skipped.

    Silently dropping it would run the user's code without its dependency and
    surface as a confusing ImportError far from the real cause.
    """
    with pytest.raises(DependencySpecError):
        _validate_dependencies(["requests", "evil; id"], _PYTHON_SPEC_RE, "python")


# ---------------------------------------------------------------------------
# End-to-end: the generated script must not contain injected text
# ---------------------------------------------------------------------------

def _sandbox():
    from orchestrator.security.langchain_sandbox import LangChainSandbox

    return LangChainSandbox.__new__(LangChainSandbox)


def test_generated_python_script_passes_specs_as_data_not_source():
    config = SandboxConfig(
        sandbox_type=SandboxType.PYTHON,
        security_policy=SecurityPolicy.MODERATE,
    )
    script = _sandbox()._prepare_python_script(
        "print('hi')", config, ["requests==2.31.0", "uvicorn[standard]"]
    )
    # The specifiers travel as a JSON payload decoded at runtime, so they are
    # data. Nothing is spliced into an executable string literal.
    assert "json.loads(" in script
    assert json.dumps(["requests==2.31.0", "uvicorn[standard]"]) in script
    assert "pip" in script
    # A shell is never involved.
    assert "shell=True" not in script


def test_generated_python_script_refuses_injection_end_to_end():
    config = SandboxConfig(
        sandbox_type=SandboxType.PYTHON,
        security_policy=SecurityPolicy.MODERATE,
    )
    with pytest.raises(DependencySpecError):
        _sandbox()._prepare_python_script(
            "print('hi')", config, ["x']); import os; os.system('id')  #"]
        )


def test_generated_javascript_script_uses_argument_vector():
    config = SandboxConfig(
        sandbox_type=SandboxType.PYTHON,
        security_policy=SecurityPolicy.MODERATE,
    )
    script = _sandbox()._prepare_javascript_script(
        "console.log(1)", config, ["express@4.18.2"]
    )
    # execFileSync takes an argv array; execSync takes a shell command string.
    assert "execFileSync" in script
    assert "execSync(" not in script.replace("execFileSync(", "")
    assert json.dumps(["express@4.18.2"]) in script


def test_generated_javascript_script_refuses_injection_end_to_end():
    config = SandboxConfig(
        sandbox_type=SandboxType.PYTHON,
        security_policy=SecurityPolicy.MODERATE,
    )
    with pytest.raises(DependencySpecError):
        _sandbox()._prepare_javascript_script(
            "console.log(1)", config, ["x'); require('child_process').execSync('id'); ('"]
        )


def test_no_dependencies_produces_no_install_step():
    config = SandboxConfig(
        sandbox_type=SandboxType.PYTHON,
        security_policy=SecurityPolicy.MODERATE,
    )
    script = _sandbox()._prepare_python_script("print('hi')", config, None)
    assert "pip" not in script
