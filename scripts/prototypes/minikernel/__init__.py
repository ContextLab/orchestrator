"""A deliberately small, runnable kernel for the #485 redesign.

Not a product. A schematic you can execute: every mechanism the issue proposes
is present in its cheapest honest form, so that its claims can be tested rather
than argued about. See scripts/prototypes/README.md.
"""

from .capabilities import (BugReport, BugTracker, Capability,
                           CapabilityRegistry, SeparationOfDuty)
from .ir import Authority, Budget, Plan, Step, validate
from .library import SolvedProblemLibrary
from .planner import LLMPlanner, PlanDraft, StubPlanner, load_env_key
from .review import (ArtifactVersion, Criterion, Finding, InsightPool,
                     ReviewBoard)
from .runtime import Crash, Message, MessageBus, NodeResult, Runtime, RunStats
from .store import Store

__all__ = [
    "Authority", "Budget", "Plan", "Step", "validate", "Store",
    "Capability", "CapabilityRegistry", "BugReport", "BugTracker",
    "SeparationOfDuty", "SolvedProblemLibrary", "StubPlanner", "LLMPlanner",
    "PlanDraft", "load_env_key", "ReviewBoard", "Criterion", "Finding",
    "ArtifactVersion", "InsightPool", "Runtime", "RunStats", "NodeResult",
    "Message", "MessageBus", "Crash",
]
