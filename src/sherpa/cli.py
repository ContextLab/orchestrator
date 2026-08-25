"""CLI entry point: python -m sherpa {run,resume,status,export-trace}."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from sherpa.ir import ProblemSpec
from sherpa.kernel import Engine


@click.group()
def main() -> None:
    """sherpa — experimental recursive agentic runtime (issue #492)."""


@main.command()
@click.argument("problem", type=click.Path(exists=True, path_type=Path))
@click.option("--workspace", "-w", type=click.Path(path_type=Path), default=Path(".sherpa_ws"))
@click.option("--channel", type=click.Choice(["recorded", "live"]), default="recorded")
def run(problem: Path, workspace: Path, channel: str) -> None:
    """Run a problem spec (JSON) to a loud terminal state."""
    spec = ProblemSpec(**json.loads(problem.read_text(encoding="utf-8")))
    engine = Engine(workspace, channel_policy=channel)
    try:
        result = engine.run(spec)
        _emit(result.model_dump())
        sys.exit(_exit_code(result.status))
    finally:
        engine.close()


@main.command()
@click.argument("run_id")
@click.option("--workspace", "-w", type=click.Path(path_type=Path), default=Path(".sherpa_ws"))
def resume(run_id: str, workspace: Path) -> None:
    """Resume an interrupted/blocked run by replaying its log."""
    engine = Engine(workspace)
    try:
        result = engine.resume(run_id)
        _emit(result.model_dump())
        sys.exit(_exit_code(result.status))
    finally:
        engine.close()


@main.command()
@click.argument("run_id")
@click.option("--workspace", "-w", type=click.Path(path_type=Path), default=Path(".sherpa_ws"))
def status(run_id: str, workspace: Path) -> None:
    """Print the durable projection of a run."""
    engine = Engine(workspace)
    try:
        _emit(engine.status(run_id))
    finally:
        engine.close()


@main.command()
@click.argument("run_id")
@click.argument("out", type=click.Path(path_type=Path))
@click.option("--workspace", "-w", type=click.Path(path_type=Path), default=Path(".sherpa_ws"))
def export_trace(run_id: str, out: Path, workspace: Path) -> None:
    """Export events + projections + metrics for a run."""
    engine = Engine(workspace)
    try:
        path = engine.export_trace(run_id, out)
        click.echo(f"wrote {path}")
    finally:
        engine.close()


def _emit(data: dict) -> None:
    click.echo(json.dumps(data, indent=2, sort_keys=True))


def _exit_code(status: str) -> int:
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    main()
