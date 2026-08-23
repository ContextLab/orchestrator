#!/usr/bin/env python3
"""What does #485's shared-scratchpad semaphore cost the fleet?

#485 specifies: "before executing a step, wait for the lock to become
available. then grab the lock. then read recent messages and consider whether
it alters the current plans ... then write down your plans. then release the
lock."

The "consider" is an LLM call, so it sits *inside* the critical section. This
is a discrete-event simulation of that, and of the same fleet with the model
call moved outside the lock.

It also models the second-order problem: the scratchpad is an append-only log,
not a static document, so the context-recursion tree over it is rebuilt
constantly unless segments are sealed.

**Findings (report section 2).**

* With the model call inside the lock, fleet throughput saturates at
  ``3600 / critical_section`` agent-steps per hour -- 450/hr at 8s -- no matter
  how many agents run. At 32 agents, 72% of every step's wall clock is lock
  wait. Adding agents past ~16 buys latency and cost and no throughput.
* Moving the call outside (the lock then guards only an append and a sequence
  bump) gives ~61x the throughput at 512 agents, and the read path needs no
  lock at all.
* Re-summarising the log on every read is quadratic in log length: 2.0e12
  tokens at 100k notes, against 5.0e7 for sealed immutable segments.

**Assumptions** in ASSUMPTIONS below. The structural conclusions -- a hard
throughput ceiling of 1/critical-section, and quadratic-vs-linear churn -- hold
for any values; the specific numbers do not.
"""

from __future__ import annotations

import heapq
import random
import statistics

# --- assumptions -----------------------------------------------------------
ASSUMPTIONS = {
    "critical_section_locked_s": 8.0,  # read tail + LLM decide + write
    "critical_section_append_s": 0.03,  # append + version bump only
    "useful_work_s": 60.0,  # per agent step, outside the lock
    "horizon_s": 3600.0,  # simulated hour
    "context_tokens": 200_000,
    "tail_frac": 0.05,  # #485's "direct read" share
    "seal_frac": 0.25,  # sealed segment size, as a share of context
    "tokens_per_note": 400,
    "r": 0.2,  # summary compression, per level
}


def simulate_lock(
    n_agents: int,
    critical_section_s: float,
    work_s: float,
    horizon_s: float = ASSUMPTIONS["horizon_s"],
    seed: int = 0,
) -> dict:
    """Agents loop: [wait for lock -> critical section -> release] -> work.

    One global lock, exponential service and work times. Service that would run
    past the horizon is truncated so utilisation cannot exceed 100% and
    throughput cannot exceed the ``horizon / critical_section`` ceiling.
    """
    rng = random.Random(seed)
    events: list[tuple[float, int]] = []
    for agent in range(n_agents):
        heapq.heappush(events, (rng.random() * work_s, agent))

    lock_free_at, busy, completed = 0.0, 0.0, 0
    waits: list[float] = []
    while events:
        now, agent = heapq.heappop(events)
        if now > horizon_s:
            break
        start = max(now, lock_free_at)
        if start > horizon_s:
            break
        waits.append(start - now)
        duration = rng.expovariate(1 / critical_section_s)
        lock_free_at = start + duration
        busy += min(lock_free_at, horizon_s) - start
        completed += 1
        heapq.heappush(events, (lock_free_at + rng.expovariate(1 / work_s), agent))

    return {
        "steps": completed,
        "steps_per_hour": completed * 3600.0 / horizon_s,
        "mean_wait": statistics.mean(waits) if waits else 0.0,
        "p95_wait": sorted(waits)[int(0.95 * len(waits))] if waits else 0.0,
        "utilisation": busy / horizon_s,
    }


def throughput_ceiling(critical_section_s: float, horizon_s: float = 3600.0) -> float:
    """Steps per hour a single global lock can ever admit."""
    return horizon_s / critical_section_s


def churn(
    n_notes: int,
    context: int = ASSUMPTIONS["context_tokens"],
    tail_frac: float = ASSUMPTIONS["tail_frac"],
    tokens_per_note: int = ASSUMPTIONS["tokens_per_note"],
    seal_frac: float = ASSUMPTIONS["seal_frac"],
    r: float = ASSUMPTIONS["r"],
) -> dict:
    """Summarisation tokens for an append-only log, naive vs sealed segments.

    naive: every read re-summarises the whole non-tail body.
    sealed: a segment is summarised once when it fills, then never again.
    """
    log_tokens = n_notes * tokens_per_note
    naive = sum(
        max(0.0, i * tokens_per_note - context * tail_frac) for i in range(n_notes)
    )
    sealed, current = 0.0, float(log_tokens)
    while current > context * seal_frac:
        sealed += current
        current *= r
    return {"log_tokens": log_tokens, "naive": naive, "sealed": sealed}


def main() -> None:
    locked = ASSUMPTIONS["critical_section_locked_s"]
    work = ASSUMPTIONS["useful_work_s"]

    print("== global scratchpad lock, LLM call INSIDE the critical section ==")
    print(f"critical section = {locked}s; useful work = {work}s/step")
    hdr = (
        f"{'agents':>7} {'steps/hr':>9} {'lock util':>10} "
        f"{'mean wait':>11} {'p95 wait':>10} {'wait/step':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for n in (2, 4, 8, 16, 32, 64, 128):
        r = simulate_lock(n, locked, work)
        share = r["mean_wait"] / (r["mean_wait"] + work + locked)
        print(
            f"{n:>7} {r['steps']:>9} {r['utilisation']:>9.0%} "
            f"{r['mean_wait']:>10.1f}s {r['p95_wait']:>9.1f}s {share:>9.0%}"
        )
    print()
    print(
        f"theoretical ceiling with a {locked}s critical section: "
        f"{throughput_ceiling(locked):.0f} agent-steps/hour, regardless of fleet size"
    )

    append = ASSUMPTIONS["critical_section_append_s"]
    print()
    print("== same fleet, LLM call moved OUTSIDE the lock ==")
    print(f"lock now guards an append + version bump only (~{append * 1000:.0f}ms)")
    print(hdr)
    print("-" * len(hdr))
    for n in (2, 8, 32, 128, 512):
        r = simulate_lock(n, append, work + locked)
        share = r["mean_wait"] / (r["mean_wait"] + work + locked)
        print(
            f"{n:>7} {r['steps']:>9} {r['utilisation']:>9.0%} "
            f"{r['mean_wait']:>10.3f}s {r['p95_wait']:>9.3f}s {share:>9.1%}"
        )

    print()
    print("== summary churn (append-only log, not a static document) ==")
    print("Agents append notes; every reader needs summaries of all but the tail.")
    hdr2 = (
        f"{'notes':>8} {'log tokens':>11} {'naive re-sum tokens':>21} "
        f"{'sealed-segment tokens':>23} {'ratio':>13}"
    )
    print(hdr2)
    print("-" * len(hdr2))
    for n_notes in (100, 1_000, 10_000, 100_000):
        c = churn(n_notes)
        ratio = (
            f"{c['naive'] / c['sealed']:>6.0f}x" if c["sealed"] else "fits directly"
        )
        print(
            f"{n_notes:>8,} {c['log_tokens']:>11,} {int(c['naive']):>21,} "
            f"{int(c['sealed']):>23,} {ratio:>13}"
        )


if __name__ == "__main__":
    main()
