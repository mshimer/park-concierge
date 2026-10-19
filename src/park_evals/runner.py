"""Load scenarios and run the agent against them."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from langchain_core.language_models import BaseChatModel
from mcp.client.stdio import StdioServerParameters
from mcp.server.mcpserver import MCPServer

from park_agent.graph import AgentRun, run_agent
from park_core.config import settings

log = logging.getLogger(__name__)

SCENARIO_DIR = Path(__file__).resolve().parent / "scenarios"


@dataclass(frozen=True)
class Scenario:
    """One eval case: what to ask, what must happen, and what good looks like."""

    id: str
    prompt: str
    rubric: list[str]
    must_call: list[str]
    must_call_any: list[str]
    must_not_call: list[str]
    min_score: int

    @classmethod
    def from_file(cls, path: Path) -> "Scenario":
        data: dict[str, Any] = yaml.safe_load(path.read_text())
        missing = {"id", "prompt", "rubric"} - data.keys()
        if missing:
            raise ValueError(f"{path.name} is missing {sorted(missing)}")
        return cls(
            id=data["id"],
            prompt=data["prompt"],
            rubric=list(data["rubric"]),
            must_call=list(data.get("must_call", [])),
            must_call_any=list(data.get("must_call_any", [])),
            must_not_call=list(data.get("must_not_call", [])),
            min_score=int(data.get("min_score", 4)),
        )


def load_scenarios(directory: Path | None = None) -> list[Scenario]:
    """Every scenario on disk, in a stable order so parametrized test ids do not shuffle."""
    return [Scenario.from_file(p) for p in sorted((directory or SCENARIO_DIR).glob("*.yaml"))]


async def run_scenario(
    scenario: Scenario,
    *,
    server: MCPServer | StdioServerParameters | str,
    model: BaseChatModel | None = None,
) -> AgentRun:
    """Run one scenario. `server` is explicit - see run_agent's docstring for why."""
    if model is None:
        from park_core.models import make_chat_model

        model = make_chat_model(settings.agent_model)
    log.info("scenario %s", scenario.id)
    return await run_agent(scenario.prompt, server=server, model=model)


def check_trace(scenario: Scenario, run: AgentRun) -> list[str]:
    """Deterministic checks on the tool trace. Returns a list of failures, empty when clean.

    Runs before the judge on purpose: a tool that was never wired up should fail loudly and for
    free, not surface later as a puzzling rubric score.

    `must_call` names tools that must each be called. `must_call_any` is satisfied by any one of
    its tools - use it when the requirement is "consulted the knowledge base" rather than "used
    this exact tool", so a scenario does not fail because the agent reasonably picked
    parks_similar_to over parks_search.
    """
    failures = []
    for name in scenario.must_call:
        if not run.trace.called(name):
            failures.append(f"expected a call to {name}; agent called {run.trace.names or 'nothing'}")
    if scenario.must_call_any and not any(run.trace.called(n) for n in scenario.must_call_any):
        failures.append(
            f"expected a call to one of {scenario.must_call_any}; "
            f"agent called {run.trace.names or 'nothing'}"
        )
    for name in scenario.must_not_call:
        if run.trace.called(name):
            failures.append(f"did not expect a call to {name}")
    if run.hit_step_limit:
        failures.append(f"agent hit the {len(run.trace.calls)}-step limit without settling")
    return failures
