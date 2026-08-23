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
    )
    from sherpa.ir import Authority
    from sherpa.kernel import Engine

    class Append(Capability):
        spec = CapabilitySpec(
            name="demo.append_line",
            input_schema={"type": "object", "required": ["file", "line"],
                          "properties": {"file": {"type": "string"},
                                         "line": {"type": "string"}}},
            output_schema={"type": "object"},
            authority_required=Authority(fs_write=("**",)),
        )

        def run(self, inputs: dict, ctx: CapabilityContext) -> dict:
            p = ctx.workspace / inputs["file"]
            with open(p, "a", encoding="utf-8") as fh:
                fh.write(inputs["line"] + "\n")
            return {"appended": inputs["line"]}

        def probe(self, ctx: CapabilityContext) -> bytes:
            return b"append ok"

    reg = CapabilityRegistry()
    reg.register(Append())
    return Engine(workspace, registry=reg)


if __name__ == "__main__":  # pragma: no cover - invoked via python -c in scenarios
    print(json.dumps({"ok": True}))
    sys.exit(0)
