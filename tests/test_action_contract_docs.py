"""The documented action vocabulary must match the registry.

`docs/actions.md` is generated from `orchestrator.core.actions`. Generating it
is only useful if regeneration is enforced -- otherwise the committed file
drifts and the documentation quietly starts describing a vocabulary the code no
longer has, which is how this project ended up advertising ten actions with no
handler in the first place.
"""

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.contract]

ROOT = Path(__file__).resolve().parent.parent
GENERATOR = ROOT / "scripts" / "generate_action_docs.py"
DOC = ROOT / "docs" / "actions.md"


def test_the_generated_action_docs_are_committed_and_current():
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, (
        f"docs/actions.md is out of date. Run:\n"
        f"    python scripts/generate_action_docs.py\n\n"
        f"{result.stdout}\n{result.stderr}"
    )


def test_every_registered_action_appears_in_the_documentation():
    """Belt and braces: the check above compares bytes, this checks intent."""
    from orchestrator.core.actions import ACTION_FAMILIES, ACTION_SPECS

    text = DOC.read_text()
    for spec in ACTION_SPECS:
        assert f"`{spec.name}`" in text, f"{spec.name} is undocumented"
        for alias in spec.aliases:
            assert f"`{alias}`" in text, f"alias {alias} is undocumented"
    for family in ACTION_FAMILIES:
        assert f"`{family.name}`" in text, f"family {family.name} is undocumented"


def test_the_exit_code_boundary_is_documented():
    """#153's runtime boundary versus #241's compile-time one.

    Both are failures, they are not the same failure, and the difference is
    visible to anyone scripting the CLI.
    """
    text = DOC.read_text()

    assert "**2**" in text and "**1**" in text
    assert "compile time" in text
    assert "before any side effect" in text
