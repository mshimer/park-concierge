"""LangSmith tracing: one place that decides whether tracing is on, and flushes it at exit.

LangChain and LangGraph trace themselves when `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` are in
the environment - no callback wiring needed. This module adds the three things that are not
automatic: a default project name, a startup status line, and a flush before a short-lived CLI exits.
"""

import logging
import os

log = logging.getLogger(__name__)

DEFAULT_PROJECT = "park-concierge"


def tracing_enabled() -> bool:
    """True when LangSmith will actually record runs from this process."""
    flag = os.getenv("LANGSMITH_TRACING", "").lower() in {"true", "1", "yes"}
    return flag and bool(os.getenv("LANGSMITH_API_KEY"))


def setup_tracing() -> bool:
    """Call once, after load_dotenv() and before the first model call. Returns whether tracing is on.

    NOTE: LangSmith reads its settings from the environment lazily and caches them, so this must
    run before the first traced call - which is why the CLI calls it straight after load_dotenv().
    """
    if os.getenv("LANGSMITH_TRACING", "").lower() in {"true", "1", "yes"} and not os.getenv(
        "LANGSMITH_API_KEY"
    ):
        log.warning("LANGSMITH_TRACING is set but LANGSMITH_API_KEY is missing; tracing is off")
        os.environ["LANGSMITH_TRACING"] = "false"
        return False
    if not tracing_enabled():
        log.info("LangSmith tracing is off (set LANGSMITH_TRACING=true to enable)")
        return False
    os.environ.setdefault("LANGSMITH_PROJECT", DEFAULT_PROJECT)
    log.info(
        "LangSmith tracing is on -> project %r at %s",
        os.environ["LANGSMITH_PROJECT"],
        os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
    )
    return True


def flush_traces() -> None:
    """Block until queued traces are uploaded.

    NOTE: runs are uploaded by a background thread. A one-shot CLI that exits right after the
    answer can lose the tail of its trace, so flush before returning from main().
    """
    if not tracing_enabled():
        return
    from langchain_core.tracers.langchain import wait_for_all_tracers
    from langsmith import Client

    wait_for_all_tracers()
    Client().flush()
