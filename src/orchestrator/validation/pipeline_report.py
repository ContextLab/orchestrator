"""Validating a pipeline, with its findings, from Python.

`orchestrator validate` has reported structured findings since #467. The
Python API had no equivalent: `PipelineAPI.validate_yaml` returns a bare
`bool`, so a caller embedding the orchestrator could learn *that* a pipeline
was rejected and nothing about why -- and could not see warnings at all, which
is where "this reference could not be checked" lives.

The findings were not missing, only private: `cli._reportable_issues` and
`cli._issue_payload` built them for the JSON output and nothing else could
reach them. A second implementation for the API would be a second thing to
drift; this module is the one implementation, and the CLI formats what it
returns.

The payload field names are interface. Anything consuming a finding should not
have to parse the human message to learn which step or reference is at fault.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass(frozen=True)
class Finding:
    """One validation finding, in stable machine-readable fields."""

    code: str
    severity: str
    category: Optional[str] = None
    step: Optional[str] = None
    parameter_path: Optional[str] = None
    referenced_step: Optional[str] = None
    referenced_field: Optional[str] = None
    message: str = ""
    suggestions: Tuple[str, ...] = ()

    @property
    def is_error(self) -> bool:
        return self.severity == "error"

    @property
    def is_warning(self) -> bool:
        return self.severity == "warning"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "category": self.category,
            "step": self.step,
            "parameter_path": self.parameter_path,
            "referenced_step": self.referenced_step,
            "referenced_field": self.referenced_field,
            "message": self.message,
            "suggestions": list(self.suggestions),
        }


@dataclass(frozen=True)
class PipelineValidation:
    """The result of validating one pipeline document."""

    valid: bool
    findings: Tuple[Finding, ...] = ()
    pipeline_id: Optional[str] = None
    #: Step id -> its dependencies, so a caller can check the graph the
    #: compiler actually built rather than re-deriving it.
    tasks: Dict[str, List[str]] = field(default_factory=dict)
    #: The exception that stopped compilation, if one did.
    error: Optional[str] = None

    @property
    def errors(self) -> Tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.is_error)

    @property
    def warnings(self) -> Tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.is_warning)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "pipeline": self.pipeline_id,
            "tasks": self.tasks,
            "findings": [f.as_dict() for f in self.findings],
            **({"error": self.error} if self.error is not None else {}),
        }


def _findings_from(compiler) -> Tuple[Finding, ...]:
    """Findings a successful validation should still tell the caller about.

    The compiler has always collected these; nothing displayed them. A
    pipeline could report valid while carrying a warning saying a reference
    could not be checked -- and then fail at run time on exactly that
    reference (#465).

    Informational findings are excluded: "tool is available" and "execution
    order computed" are not things anyone needs told.
    """
    report = getattr(compiler, "validation_report", None)
    if report is None:
        return ()
    return tuple(
        _finding(issue)
        for issue in report.issues
        if issue.is_error or issue.is_warning
    )


def _finding(issue) -> Finding:
    metadata = issue.metadata or {}
    severity = issue.severity
    return Finding(
        code=issue.code,
        severity=getattr(severity, "value", severity),
        category=issue.category,
        step=metadata.get("step", issue.component),
        parameter_path=metadata.get("parameter_path", issue.path),
        referenced_step=metadata.get("referenced_step"),
        referenced_field=metadata.get("referenced_field"),
        message=issue.message,
        suggestions=tuple(issue.suggestions or ()),
    )


async def validate_pipeline_text(yaml_text: str) -> PipelineValidation:
    """Compile `yaml_text` without running it and report what was found.

    Compilation failure is a result, not an exception: a caller validating
    user input wants the findings alongside the failure, and raising would
    discard them.
    """
    from ..compiler.yaml_compiler import YAMLCompiler

    compiler = YAMLCompiler()
    try:
        pipeline = await compiler.compile(yaml_text, {})
    except Exception as exc:
        return PipelineValidation(
            valid=False,
            findings=_findings_from(compiler),
            error=f"{type(exc).__name__}: {exc}",
        )

    tasks = getattr(pipeline, "tasks", {}) or {}
    return PipelineValidation(
        valid=True,
        findings=_findings_from(compiler),
        pipeline_id=getattr(pipeline, "id", None),
        tasks={
            task_id: list(getattr(task, "dependencies", []) or [])
            for task_id, task in tasks.items()
        },
    )


def validate_pipeline_file(path: Union[str, Path]) -> PipelineValidation:
    """`validate_pipeline_text` for a file on disk, without an event loop.

    Provided because the common case -- checking a pipeline someone wrote --
    is synchronous, and requiring callers to build an event loop to learn
    whether a file is valid is why the boolean survived so long.
    """
    return asyncio.run(validate_pipeline_text(Path(path).read_text()))
