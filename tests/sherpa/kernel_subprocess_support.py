"""Support for kernel fault-injection tests run through REAL subprocesses."""

from __future__ import annotations

import json
import os
import signal
import sys
from pathlib import Path


def build_spec(problem_json: str):
    from sherpa.ir import ProblemSpec

    return ProblemSpec(**json.loads(problem_json))


def build_engine(workspace: Path):
    from sherpa.capabilities import (
        Capability,
        CapabilityContext,
        CapabilityRegistry,
        CapabilitySpec,
    )
    from sherpa.ir import Authority
    from sherpa.kernel import Engine

    class AppendLine(Capability):
        spec = CapabilitySpec(
            name="demo.append_line",
            input_schema={
                "type": "object",
                "required": ["file", "line"],
                "properties": {"file": {"type": "string"}, "line": {"type": "string"}},
            },
            output_schema={"type": "object"},
            authority_required=Authority(fs_write=("**",)),
        )

        def run(self, inputs: dict, ctx: CapabilityContext) -> dict:
            p = ctx.workspace / inputs["file"]
            with open(p, "a", encoding="utf-8") as fh:
                fh.write(inputs["line"] + "\n")
            return {"appended": inputs["line"], "file": inputs["file"]}

        def probe(self, ctx: CapabilityContext) -> bytes:
            canary = ctx.workspace / ".probe_append"
            canary.write_text("x", encoding="utf-8")
            canary.unlink()
            return b"append probe ok"

    reg = CapabilityRegistry()
    reg.register(AppendLine())
    return Engine(workspace, registry=reg)


def kill_self_after(n_events: int) -> None:
    """Arm the engine's env-based hook; verify it is set for honesty."""
    os.environ["SHERPA_KILL_AFTER_EVENTS"] = str(n_events)
    assert int(os.environ["SHERPA_KILL_AFTER_EVENTS"]) == n_events
