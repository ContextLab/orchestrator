"""Engine construction for scenario A subprocess fault injection."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def build_engine_a(workspace: Path):
    from sherpa.capabilities import (
        Capability,
        CapabilityContext,
        CapabilityRegistry,
        CapabilitySpec,
        assert_fs_access,
    )
    from sherpa.kernel import Engine

    class Append(Capability):
        spec = CapabilitySpec(
            name="demo.append_line",
            input_schema={"type": "object", "required": ["file", "line"],
                          "properties": {"file": {"type": "string"},
                                         "line": {"type": "string"}}},
            output_schema={"type": "object"},
            requires=("fs_write",),
        )

        def run(self, inputs: dict, ctx: CapabilityContext) -> dict:
            p = assert_fs_access(inputs["file"], "fs_write", ctx, self.spec.name)
            with open(p, "a", encoding="utf-8") as fh:
                fh.write(inputs["line"] + "\n")
            return {"appended": inputs["line"]}

        def probe(self, ctx: CapabilityContext) -> bytes:
            return b"append ok"

    reg = CapabilityRegistry()
    reg.register(Append())
    # This process exists to be SIGKILLed mid-run, so it opts in explicitly.
    return Engine(workspace, registry=reg, fault_injection=True)


if __name__ == "__main__":  # pragma: no cover - invoked via python -c in scenarios
    print(json.dumps({"ok": True}))
    sys.exit(0)
