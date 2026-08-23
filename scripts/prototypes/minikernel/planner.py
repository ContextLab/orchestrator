"""Planners: the thing that decides `f`.

`f` -- the fraction of emitted steps a planner calls AMBIGUOUS rather than
atomic -- is the load-bearing parameter of the whole design. With mean fan-out
`b`, recursion is finite in expectation iff `m = b*f < 1`. Every review of #485
so far has had to GUESS `f`. This module exists so it can be measured.

Two implementations behind one interface:

  StubPlanner  deterministic, seeded, no network. Used by the tests and by the
               ablation sweeps, where we want to *set* f and observe the system.

  LLMPlanner   a real model, asked to decompose a real problem against the real
               capability library, emitting strict JSON. Used to *measure* f --
               and, critically, to measure f as a function of library size,
               which is the claim that the library is the termination mechanism.

Note what `f` is NOT: a property of the model alone. It is a property of
(model, problem distribution, library contents). Any measured value must be
quoted with all three.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

from .ir import Authority, Plan, Step


@dataclass
class PlanDraft:
    plan: Plan
    rationale: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    raw: str = ""
    model: str = "stub"

    @property
    def fan_out(self) -> int:
        return len(self.plan.steps)

    @property
    def n_ambiguous(self) -> int:
        return sum(1 for s in self.plan.steps if s.kind == "decompose")

    @property
    def f(self) -> float:
        return self.n_ambiguous / self.fan_out if self.fan_out else 0.0


class Planner(Protocol):
    def decompose(self, problem: str, signature: str,
                  capabilities: dict[str, str], depth: int) -> PlanDraft: ...


# --------------------------------------------------------------- stub planner


@dataclass
class StubPlanner:
    """Deterministic planner with a *settable* ambiguity rate.

    Used to drive the system through regimes we cannot afford to reach with a
    real model (m > 1 with thousands of nodes).
    """

    seed: int = 0
    b: int = 5
    f: float = 0.2
    capability_pool: tuple[str, ...] = ("echo@1",)

    def _rng_for(self, problem: str, depth: int) -> random.Random:
        """Deterministic in (problem, depth).

        FIX (found by scenario S7): the first version carried one RNG stream
        across calls, so asking the same question twice produced different
        plans. No real planner is memoryless like that, and it made the
        library's benefit unmeasurable -- mission 5 drew a worse plan than
        mission 1 for reasons that had nothing to do with the library. A
        planner must be a function of its inputs for a cache over it to mean
        anything, which is also why the runtime content-addresses plan inputs.
        """
        import hashlib

        h = hashlib.sha256(f"{self.seed}|{problem}|{depth}".encode()).digest()
        return random.Random(int.from_bytes(h[:8], "big"))

    def decompose(self, problem: str, signature: str,
                  capabilities: dict[str, str], depth: int) -> PlanDraft:
        rng = self._rng_for(problem, depth)
        usable = [r for r, m in capabilities.items() if m in ("trusted", "candidate")]
        usable = usable or list(self.capability_pool)
        n = max(2, min(self.b, 12))
        steps: list[Step] = []
        for i in range(n):
            if rng.random() < self.f:
                steps.append(
                    Step(id=f"s{i}", kind="decompose",
                         problem=f"{problem} :: part {i} (depth {depth})",
                         outputs=("y",), output_schema="str")
                )
            else:
                steps.append(
                    Step(id=f"s{i}", kind="capability",
                         ref=rng.choice(usable),
                         args={"text": f"{problem}#{i}"}, output_schema="str")
                )
        return PlanDraft(Plan(id=f"plan-{abs(hash(problem)) % 10**8}",
                              problem=problem, steps=tuple(steps),
                              signature=signature),
                         rationale="stub", tokens_in=0, tokens_out=0)


# ---------------------------------------------------------------- LLM planner

SYSTEM = """You are the decomposition planner of a recursive problem-solving \
runtime. Break the given problem into a SHORT pipeline of concrete steps \
(aim for 3-7, hard maximum 10).

Each step is exactly one of:
  {"id": "s1", "kind": "capability", "ref": "<one of the AVAILABLE CAPABILITIES>",
   "args": {...}, "output_schema": "<type>"}
      -- use this when the step can be done RIGHT NOW by that capability, with
         at most trivial adaptation of its output.
  {"id": "s2", "kind": "decompose", "problem": "<what happens at this step>",
   "outputs": ["name"], "output_schema": "<type>"}
      -- use this ONLY when you do not know how to do the step with the
         available capabilities and it must itself be broken down further.

Be honest and be economical: marking a step "decompose" costs a whole recursive \
subtree, so only do it when the step genuinely needs one. If an available \
capability does the job, use it.

Reply with ONLY a JSON object: {"steps": [...], "rationale": "<one sentence>"}"""


class LLMPlanner:
    """OpenAI-compatible chat completions over stdlib urllib. No SDK, no mocks."""

    def __init__(self, model: str, api_key: str,
                 base_url: str = "https://api.openai.com/v1",
                 max_retries: int = 3, temperature: float | None = None):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.temperature = temperature
        self.calls = 0
        self.tokens_in = 0
        self.tokens_out = 0

    def _chat(self, messages: list[dict[str, str]],
              max_tokens: int = 1600) -> tuple[str, int, int]:
        payload: dict[str, Any] = {"model": self.model, "messages": messages,
                                   "max_completion_tokens": max_tokens}
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        last: Exception | None = None
        for attempt in range(self.max_retries):
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=json.dumps(payload).encode(),
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    body = json.load(resp)
                self.calls += 1
                usage = body.get("usage", {})
                ti = int(usage.get("prompt_tokens", 0))
                to = int(usage.get("completion_tokens", 0))
                self.tokens_in += ti
                self.tokens_out += to
                return body["choices"][0]["message"]["content"] or "", ti, to
            except urllib.error.HTTPError as exc:
                detail = exc.read()[:300].decode("utf8", "ignore")
                last = RuntimeError(f"HTTP {exc.code}: {detail}")
                if exc.code in (429, 500, 502, 503, 529):
                    time.sleep(2 ** attempt)
                    continue
                raise last
            except Exception as exc:
                last = exc
                time.sleep(2 ** attempt)
        raise RuntimeError(f"chat failed after {self.max_retries} attempts: {last}")

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        text = text.strip()
        fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
        if fence:
            text = fence.group(1).strip()
        start = text.find("{")
        if start < 0:
            raise ValueError(f"no JSON object in reply: {text[:200]!r}")
        depth, end = 0, None
        for i, ch in enumerate(text[start:], start):
            depth += (ch == "{") - (ch == "}")
            if depth == 0:
                end = i + 1
                break
        if end is None:
            raise ValueError(f"unterminated JSON in reply: {text[:200]!r}")
        return json.loads(text[start:end])

    def decompose(self, problem: str, signature: str,
                  capabilities: dict[str, str], depth: int) -> PlanDraft:
        usable = sorted(r for r, m in capabilities.items()
                        if m in ("trusted", "candidate"))
        user = (
            f"PROBLEM: {problem}\n"
            f"REQUIRED SIGNATURE: {signature}\n"
            f"RECURSION DEPTH: {depth}\n"
            f"AVAILABLE CAPABILITIES ({len(usable)}):\n"
            + ("\n".join(f"  - {r}" for r in usable) if usable else "  (none)")
        )
        content, ti, to = self._chat(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        )
        data = self._extract_json(content)
        steps: list[Step] = []
        for i, raw in enumerate(data.get("steps", [])):
            kind = raw.get("kind")
            sid = str(raw.get("id") or f"s{i}")
            if kind == "capability":
                steps.append(Step(id=sid, kind="capability", ref=raw.get("ref"),
                                  args=raw.get("args") or {},
                                  output_schema=str(raw.get("output_schema") or "any")))
            elif kind == "decompose":
                outs = tuple(raw.get("outputs") or ("y",))
                steps.append(Step(id=sid, kind="decompose",
                                  problem=str(raw.get("problem") or problem),
                                  outputs=outs,
                                  output_schema=str(raw.get("output_schema") or "any")))
            else:
                # An unrecognised kind is NOT silently coerced -- it is recorded
                # as a validator-visible error by emitting an invalid step.
                steps.append(Step(id=sid, kind="capability",
                                  ref=str(raw.get("ref") or f"<unknown-kind:{kind}>"),
                                  output_schema=""))
        return PlanDraft(
            Plan(id=f"plan-{abs(hash((problem, self.model))) % 10**8}",
                 problem=problem, steps=tuple(steps), signature=signature),
            rationale=str(data.get("rationale", "")),
            tokens_in=ti, tokens_out=to, raw=content, model=self.model,
        )


def load_env_key(name: str = "OPENAI_API_KEY") -> str | None:
    if os.environ.get(name):
        return os.environ[name]
    path = os.path.expanduser("~/.orchestrator/.env")
    if not os.path.exists(path):
        return None
    for line in open(path):
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == name:
            return v.strip().strip('"').strip("'")
    return None
