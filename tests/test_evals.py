"""The eval suite: run each scenario, assert the tool trace, then let a judge grade the answer.

Marked `eval` because it spends tokens on two models. Run it with:

    uv run pytest -m eval -v

It needs a vector store built with real embeddings (`uv run park-ingest`) and an OPENAI_API_KEY.
Postgres still comes from a throwaway container, so the agent is reading real rows over real SQL.
"""

from collections.abc import AsyncIterator

import httpx
import pytest
from langchain_openai import ChatOpenAI

from park_core.chroma import default_embeddings
from park_core.config import settings
from park_core.models import make_chat_model
from park_evals.judge import default_judge, judge_answer
from park_evals.runner import Scenario, check_trace, load_scenarios, run_scenario
from park_mcp.context import ServerContext
from park_mcp.server import build_server

pytestmark = [pytest.mark.anyio, pytest.mark.eval]

SCENARIOS = load_scenarios()


@pytest.fixture
def eval_ctx(pg_dsn: str, real_chroma_dir: str) -> ServerContext:
    """Container Postgres plus the real, OpenAI-embedded vector store.

    Function-scoped for the same reason as the model fixtures: it holds an OpenAI client, and
    anyio runs each test on a fresh event loop. Opening Chroma again per test is a local file
    read, so this costs nothing.
    """
    return ServerContext.from_settings(
        pg_dsn=pg_dsn, chroma_dir=real_chroma_dir, embeddings=default_embeddings()
    )


# CONFUSING-BIT WARNING: three separate traps conspire here, and all of them surface as the same
# "RuntimeError: Event loop is closed" - two from inside the llm node, one at teardown.
#
#   1. A ChatOpenAI holds an httpx async client bound to the event loop it first ran on, and anyio
#      gives every test a fresh loop. Hence function scope, not session scope.
#   2. Function scope alone is not enough: langchain_openai memoises the default async client in a
#      process-wide LRU cache, so a newly constructed model cheerfully picks the dead one back up.
#   3. Even with a fresh client per test, langchain's wrapper closes it from a finaliser. Garbage
#      collection happens whenever it happens - typically during a *later* test - and closing an
#      httpx transport on a dead loop raises, failing a test whose assertions all passed.
#
# So the client is created and closed explicitly, inside the test's own loop, and handed to both
# models. Nothing is left for a finaliser to trip over.
TIMEOUT = httpx.Timeout(60.0)


@pytest.fixture
async def http_client() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        yield client


@pytest.fixture
def agent_model(http_client: httpx.AsyncClient) -> ChatOpenAI:
    return make_chat_model(
        settings.agent_model,
        timeout=TIMEOUT,
        http_async_client=http_client,
    )


@pytest.fixture
def judge_model(http_client: httpx.AsyncClient) -> ChatOpenAI:
    return default_judge(timeout=TIMEOUT, http_async_client=http_client)


def test_scenarios_are_loadable() -> None:
    """Not an LLM test, but it belongs with the suite it guards."""
    assert len(SCENARIOS) == 5
    assert len({s.id for s in SCENARIOS}) == len(SCENARIOS)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.id)
async def test_scenario(
    scenario: Scenario,
    eval_ctx: ServerContext,
    agent_model: ChatOpenAI,
    judge_model: ChatOpenAI,
) -> None:
    run = await run_scenario(scenario, server=build_server(None, eval_ctx), model=agent_model)

    # Deterministic first, and cheap: a tool that was never wired up should fail here, plainly,
    # rather than showing up later as a puzzling rubric score we pay a judge to discover.
    failures = check_trace(scenario, run)
    assert not failures, "\n".join(failures)

    verdict = judge_answer(
        scenario.prompt,
        run.answer,
        run.trace.retrieved,
        scenario.rubric,
        model=judge_model,
    )

    # Printed so `-v -s` shows the grading, not just a pass or fail.
    print(f"\n[{scenario.id}] score={verdict.score}/5 passed={verdict.passed}\n  {verdict.reason}")
    assert verdict.passed, verdict.reason
    assert verdict.score >= scenario.min_score, f"{verdict.score} < {scenario.min_score}: {verdict.reason}"
