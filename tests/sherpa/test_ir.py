"""Hermetic tests for sherpa.ir (issue #492 typed plan IR)."""

from __future__ import annotations

import pytest

from sherpa.ir import (
    Authority,
    Branch,
    Budgets,
    Decompose,
    Fail,
    InvokeCapability,
    Parallel,
    Plan,
    Return,
    While,
    estimate_depth,
    validate_plan,
)

pytestmark = [pytest.mark.unit]


def _cap(node_id: str, capability: str = "fs.read_file") -> InvokeCapability:
    return InvokeCapability(kind="invoke_capability", id=node_id, capability=capability)


def _plan(**kw: object) -> Plan:
    defaults: dict = {
        "id": "p1",
        "authority": Authority(fs_read=("data/",)),
        "budgets": Budgets(max_fanout=3, max_depth=5),
        "root": [_cap("a")],
    }
    defaults.update(kw)
    return Plan(**defaults)


class TestAuthority:
    def test_subset_allowed(self) -> None:
        parent = Authority(fs_read=("data/**",), fs_write=("out/",))
        child = Authority(fs_read=("data/x.csv",), fs_write=("out/a.txt",))
        assert parent.allows(child)

    def test_bare_prefix_covers_deeper(self) -> None:
        parent = Authority(fs_read=("data/",))
        assert parent.allows(Authority(fs_read=("data/sub/deep/file.txt",)))

    def test_outside_denied(self) -> None:
        parent = Authority(fs_write=("out/",))
        assert not parent.allows(Authority(fs_write=("etc/passwd",)))

    def test_empty_child_always_allowed(self) -> None:
        assert Authority().allows(Authority())

    def test_new_domain_denied(self) -> None:
        parent = Authority(net_domains=("example.com",))
        assert not parent.allows(Authority(net_domains=("evil.net",)))


class TestValidatePlan:
    def test_valid_minimal(self) -> None:
        assert validate_plan(_plan(), registry_names={"fs.read_file"}) == []

    def test_duplicate_id_rejected(self) -> None:
        plan = _plan(root=[_cap("a"), _cap("a")])
        codes = [e.code for e in validate_plan(plan)]
        assert "duplicate_id" in codes

    def test_nested_duplicate_caught(self) -> None:
        inner = While(kind="while", id="loop", guard="x < 3", max_iterations=2, body=[_cap("a")])
        plan = _plan(root=[_cap("a"), inner])
        assert any(e.code == "duplicate_id" for e in validate_plan(plan))

    def test_unknown_capability_when_registry_given(self) -> None:
        plan = _plan()
        errs = validate_plan(plan, registry_names={"other.cap"})
        assert any(e.code == "unknown_capability" for e in errs)
        assert validate_plan(plan, registry_names=None) == []

    def test_parallel_fanout_exceeded(self) -> None:
        branches = [[_cap(f"n{i}")] for i in range(4)]
        plan = _plan(root=[Parallel(kind="parallel", id="par", branches=branches)])
        assert any(e.code == "fanout_exceeded" for e in validate_plan(plan))

    def test_while_missing_max_iterations_is_schema_error(self) -> None:
        with pytest.raises(Exception):
            While(kind="while", id="w", guard="True", max_iterations=0, body=[_cap("a")])

    def test_depth_exceeded(self) -> None:
        deep: object = _cap("leaf")
        for i in range(6):
            deep = While(kind="while", id=f"w{i}", guard="False", max_iterations=1, body=[deep])
        plan = _plan(root=[deep], budgets=Budgets(max_depth=3, max_fanout=4))
        assert any(e.code == "depth_exceeded" for e in validate_plan(plan))

    def test_decompose_cap_over_budget(self) -> None:
        plan = _plan(
            root=[
                Decompose(
                    kind="decompose",
                    id="d",
                    subgoal="do it",
                    fanout_cap=99,
                )
            ]
        )
        assert any(e.code == "fanout_exceeded" for e in validate_plan(plan))

    def test_fail_and_return_nodes_validate(self) -> None:
        plan = _plan(root=[_cap("a"), Fail(kind="fail", id="f", reason="nope"), Return(kind="return", id="r")])
        assert validate_plan(plan) == []


class TestEstimateDepth:
    def test_flat_is_one(self) -> None:
        assert estimate_depth(_plan()) == 1

    def test_nested_structures_count(self) -> None:
        node = Branch(
            kind="branch",
            id="b",
            cases=[
                __import__("sherpa.ir", fromlist=["BranchCase"]).BranchCase(when=None, body=[_cap("x")])
            ],
        )
        plan = _plan(root=[node])
        assert estimate_depth(plan) == 2
