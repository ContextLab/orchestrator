"""The catalogue report says what the validator said, and says it the same way twice.

A dashboard nobody can trust is worse than no dashboard: it gets read once,
found wrong, and then ignored while still occupying a CI slot. Two properties
carry that trust.

**Determinism.** The first attempt grouped failures by the first line
mentioning "error", which picked up log timestamps -- so every run produced a
different set of groups -- and echoed YAML source, so one file was grouped
under `id: legacy_tool_usage`, a line from its own body.

**The gate is a list of names, not a count.** The headline number moves for
reasons that are not regressions: repairing an `enhanced/` twin moves it by
two, retiring a legacy file moves it by one. Gating on it would punish
ordinary progress and reward deleting the hard examples.
"""

import importlib.util
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

REPO = Path(__file__).resolve().parent.parent
SUPPORTED_LIST = REPO / "scripts" / "catalogue_validating.txt"


def _module():
    """Load the script by path; `scripts/` is not an importable package."""
    spec = importlib.util.spec_from_file_location(
        "catalogue_report", REPO / "scripts" / "catalogue_report.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


report = _module()


# ---------------------------------------------------------------------------
# The signature is what the validator said
# ---------------------------------------------------------------------------

def test_the_validators_own_finding_is_used():
    output = """
Validation ✗ FAILED (STRICT)
Failed to compile YAML: Pipeline validation failed:
  - Undefined variable: 'ghost'
  - Undefined task reference: 'rows'
"""
    assert report.first_error(output) == "Undefined variable: '...'"


def test_a_yaml_list_item_in_echoed_source_is_not_a_finding():
    """`  - id: legacy_tool_usage` is a line of the pipeline, not a complaint.

    Matching any `  - ...` line grouped one file under its own body text.
    """
    output = """
Reading pipeline:
  - id: legacy_tool_usage
    tool: nonexistent
2026-08-03 19:52:19 - orchestrator.validation.validation_report - ERROR - Validation error in tool: Tool 'nonexistent' not found in registry
"""
    assert report.first_error(output) == "Tool '...' not found in registry"


def test_a_timestamp_never_reaches_a_signature():
    """Two runs of the same failure must group together.

    Timestamps in the signature meant every run reported a different set of
    groups, each of size one.
    """
    template = (
        "%s - orchestrator.validation.validation_report - ERROR - "
        "Validation error in template: Undefined variable: 'x'"
    )
    first = report.first_error(template % "2026-08-03 19:52:19")
    second = report.first_error(template % "2026-08-04 07:15:02")

    assert first == second, f"{first!r} != {second!r}; a run stamp leaked in"
    assert "2026" not in first


def test_a_file_that_does_not_parse_is_reported_as_such():
    """The parser echoes source, and a pipeline whose prose contains the word
    "error" would otherwise be grouped by its own text."""
    output = """
Failed to compile YAML: Invalid YAML:
  in "<unicode string>", line 117, column 5
117:     error_handling: continue
yaml.scanner.ScannerError: mapping values are not allowed here
"""
    assert report.first_error(output) == "YAML does not parse"


def test_counts_inside_a_message_are_normalised():
    """"Schema validation failed: 56 errors" and "...: 4272 errors" are one
    kind of problem; leaving the count in split it across nine rows."""
    def sig(n):
        return report.first_error(
            f"validation failed:\n  - Compilation failed: "
            f"Schema validation failed: {n} errors"
        )

    assert sig(56) == sig(4272)
    assert "N errors" in sig(56)


def test_the_specifics_of_a_name_are_normalised():
    def sig(name):
        return report.first_error(
            f"validation failed:\n  - Tool '{name}' not found in registry"
        )

    assert sig("debug") == sig("count_words")


def test_output_with_no_error_at_all_is_labelled_not_guessed():
    assert report.first_error("everything went fine\n") == "(no error reported)"


# ---------------------------------------------------------------------------
# The gate is a list of names
# ---------------------------------------------------------------------------

def test_a_file_leaving_the_list_is_a_regression():
    regressed, gained = report.compare(["a.yaml", "b.yaml"], ["a.yaml"])
    assert regressed == ["b.yaml"]
    assert gained == []


def test_a_newly_validating_file_is_not_a_regression():
    """Failing here would mean every repaired example breaks the build until
    someone updates a list, which teaches people to stop repairing examples."""
    regressed, gained = report.compare(["a.yaml"], ["a.yaml", "c.yaml"])
    assert regressed == []
    assert gained == ["c.yaml"]


def test_a_swap_of_equal_size_is_still_a_regression():
    """The count is unchanged here, which is exactly why the count is not the
    contract."""
    regressed, gained = report.compare(["a.yaml", "b.yaml"], ["a.yaml", "c.yaml"])
    assert regressed == ["b.yaml"]
    assert gained == ["c.yaml"]


# ---------------------------------------------------------------------------
# The committed list describes this repository
# ---------------------------------------------------------------------------

def test_every_listed_example_exists():
    """A stale entry would fail CI forever with a confusing message."""
    missing = [
        line for line in SUPPORTED_LIST.read_text().splitlines()
        if line.strip() and not line.startswith("#")
        and not (REPO / line.strip()).exists()
    ]
    assert not missing, f"listed but not present: {missing}"


def test_the_list_is_not_empty():
    assert report.load_supported(), (
        "an empty supported list would make the regression gate vacuous"
    )
