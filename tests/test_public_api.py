"""Contract tests for the lazy public API surface.

The package resolves public names lazily (PEP 562). Two properties must hold:

1. **Every advertised name resolves.** A typo in the name -> module map would
   otherwise turn into an AttributeError only for the unlucky user who touches
   that name.
2. **Resolution is order-independent.** Importing a name *first*, in a fresh
   interpreter, must work. A circular import inside the package can make a name
   resolve only after something else has been imported, which is exactly the
   kind of defect that hides during development (where everything is warm) and
   surfaces for a user whose first line is the unlucky one.
"""

import subprocess
import sys
from pathlib import Path

import pytest

import orchestrator

pytestmark = pytest.mark.contract

SRC = str(Path(__file__).parent.parent / "src")


def test_all_is_non_trivial():
    assert len(orchestrator.__all__) > 40, "public surface unexpectedly small"


@pytest.mark.parametrize("name", sorted(orchestrator.__all__))
def test_every_public_name_resolves(name):
    """Warm resolution: the name -> module map has no dangling entries."""
    assert getattr(orchestrator, name) is not None


def test_unknown_name_raises_attribute_error():
    with pytest.raises(AttributeError, match="has no attribute"):
        orchestrator.definitely_not_a_real_name


def test_dir_includes_public_names():
    listed = set(dir(orchestrator))
    assert set(orchestrator.__all__) <= listed


@pytest.mark.parametrize(
    "name",
    [
        # These five previously failed on a COLD import: resolving them ran
        # control_flow/__init__, which reached compiler/__init__, which imported
        # control_flow_compiler, which imported back into the partially
        # initialized control_flow package. They only worked if something had
        # already loaded `compiler` first.
        "ConditionalHandler",
        "ForLoopHandler",
        "WhileLoopHandler",
        "DynamicFlowHandler",
        "ControlFlowAutoResolver",
        # Spot-check the main entry points too.
        "Orchestrator",
        "YAMLCompiler",
        "ControlFlowCompiler",
        "Pipeline",
        "Task",
    ],
)
def test_name_imports_cold_in_a_fresh_interpreter(name):
    """Each name must import first, in a process that imported nothing else."""
    result = subprocess.run(
        [sys.executable, "-c", f"from orchestrator import {name}"],
        capture_output=True,
        text=True,
        timeout=120,
        env={"PYTHONPATH": SRC, "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, (
        f"cold `from orchestrator import {name}` failed - this usually means a "
        f"circular import inside the package:\n{result.stderr}"
    )
