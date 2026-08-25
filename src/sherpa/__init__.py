"""sherpa: experimental evidence-driven recursive agentic runtime (#492).

Public API::

    from sherpa import Engine, ProblemSpec

    engine = Engine(workspace)
    result = engine.run(problem)          # RunResult(status=..., outputs=...)
    resumed = engine.resume(result.run_id)

See docs/adr/0003-sherpa-recursive-agentic-runtime-mvp.md and issue #492.
Nothing here imports the supported `orchestrator` path except the lazy live
provider adapters in `sherpa.channel`.
"""

from sherpa.ir import Authority, Budgets, Plan, ProblemSpec, validate_plan
from sherpa.kernel import Engine, RunResult

__version__ = "0.1.0"

__all__ = [
    "Authority",
    "Budgets",
    "Engine",
    "Plan",
    "ProblemSpec",
    "RunResult",
    "validate_plan",
    "__version__",
]
