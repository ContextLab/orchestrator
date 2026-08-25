"""Admission control: an `atomic` step is a checked claim, not a label (#492 §3).

Before any ``InvokeCapability`` leaf executes, the checker verifies the named
capability exists, the resolved inputs type-check against its declared schema,
the required authority is granted, and — decisively — *executable evidence*
exists that the capability can satisfy this step (``Capability.probe`` runs
now; its bytes are hashed into the blob store). Rejected atomic claims are
reclassified for decomposition or escalated, never silently run. Every check
logs one ``admission_checked`` event; metrics consume these to measure
overclaim rate and corrected branching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from sherpa.events import Event
from sherpa.ir import Authority, InvokeCapability

if TYPE_CHECKING:
    from sherpa.capabilities import CapabilityContext, CapabilityRegistry
    from sherpa.store import BlobStore, Store


@dataclass
class AdmissionPolicy:
    reclassify_on_probe_fail: bool = True


class AdmissionVerdict(BaseModel):
    decision: Literal["admitted", "reclassify_decompose", "escalate"]
    reasons: list[str] = field(default_factory=list)
    io_compatible: bool = False
    evidence_sha: str | None = None
    probe_ok: bool = False


def _type_ok(value: Any, spec: dict) -> bool:  # noqa: PLR0911 - explicit whitelist
    t = spec.get("type")
    if t is None:
        return True
    if t == "string":
        return isinstance(value, str)
    if t == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if t == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if t == "boolean":
        return isinstance(value, bool)
    if t == "null":
        return value is None
    if t == "array":
        return isinstance(value, list) and all(_type_ok(v, spec.get("items", {})) for v in value)
    if t == "object":
        if not isinstance(value, dict):
            return False
        props = spec.get("properties", {})
        for req in spec.get("required", []):
            if req not in value:
                return False
        for key, sub in props.items():
            if key in value and not _type_ok(value[key], sub):
                return False
        return True
    return False


def check_io(value: Any, schema: dict) -> tuple[bool, list[str]]:
    """Fail-closed subset validator (no jsonschema dependency)."""
    errors: list[str] = []
    if not _type_ok(value, schema or {}):
        errors.append(f"value does not match schema {schema.get('type', 'any')}")
    return (not errors), errors


class AdmissionChecker:
    """Independent gate between plans and execution."""

    def __init__(
        self,
        store: "Store",
        registry: "CapabilityRegistry",
        blob: "BlobStore",
        policy: AdmissionPolicy | None = None,
    ) -> None:
        self.store = store
        self.registry = registry
        self.blob = blob
        self.policy = policy or AdmissionPolicy()

    def check(
        self,
        step: InvokeCapability,
        resolved_inputs: dict,
        granted: Authority,
        ctx: "CapabilityContext",
    ) -> AdmissionVerdict:
        decision = "admitted"
        reasons: list[str] = []
        io_compatible = False
        evidence_sha: str | None = None
        probe_ok = False

        try:
            cap = self.registry.get(step.capability)
        except KeyError:
            cap = None
            decision = "escalate"
            reasons.append(f"capability {step.capability!r} is not registered")

        if cap is not None:
            io_ok, io_errors = check_io(resolved_inputs, cap.spec.input_schema)
            if not io_ok:
                decision = "escalate"
                reasons.extend(io_errors)
            else:
                io_compatible = True

            try:
                # Both gates: the dimensions this capability uses must be
                # granted at all, and any pattern-shaped requirement (e.g.
                # subprocess_allow) must be covered. The per-resource check
                # happens later, inside run_capability.
                from sherpa.capabilities import assert_requires

                assert_requires(cap.spec, granted, step.capability)
                assert_authority_granted(cap.spec.authority_required, granted)
            except PermissionError as exc:
                decision = "escalate"
                reasons.append(str(exc))

            if decision == "admitted":
                # executable evidence: run the capability's probe now.
                try:
                    evidence = cap.probe(ctx)
                    evidence_sha = self.blob.put_bytes(evidence)
                    probe_ok = True
                except Exception as exc:  # noqa: BLE001 - boundary of executable evidence
                    probe_ok = False
                    if self.policy.reclassify_on_probe_fail:
                        decision = "reclassify_decompose"
                        reasons.append(
                            f"probe failed ({type(exc).__name__}: {exc}); atomic claim reclassified for decomposition"
                        )
                    else:
                        decision = "escalate"
                        reasons.append(f"probe failed ({type(exc).__name__})")

        verdict = AdmissionVerdict(
            decision=decision,
            reasons=reasons,
            io_compatible=io_compatible,
            evidence_sha=evidence_sha,
            probe_ok=probe_ok,
        )
        self.store.append(
            Event(
                kind="admission_checked",
                run_id=ctx.run_id,
                node_key=ctx.node_key,
                payload={
                    "capability": step.capability,
                    "decision": verdict.decision,
                    "io_compatible": verdict.io_compatible,
                    "probe_ok": verdict.probe_ok,
                    "evidence_sha": verdict.evidence_sha,
                    "reasons": verdict.reasons,
                    "atomic_claimed": step.atomic_claim,
                },
            )
        )
        return verdict


def assert_authority_granted(required: Authority, granted: Authority) -> None:
    """Pattern-level delegation check, shared with delegation and execution.

    This used to be a third, divergent implementation that compared the
    capability's declared *pattern* against grants, so a properly scoped grant
    such as ``fs_read=("src/**",)`` refused every builtin while execution would
    have allowed it -- the gate and the executor answered different questions.
    """
    from sherpa.authority import authority_covers, missing_powers

    if not authority_covers(granted, required):
        missing = missing_powers(granted, required)
        raise PermissionError(f"authority not granted: {', '.join(missing)}")
