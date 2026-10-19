"""Ask the concierge a question.

    uv run park-agent "Customer 1042 is planning a trip in March. Where should they go?"
    uv run park-agent                 # REPL
    uv run park-agent --inprocess ... # skip the subprocess, useful when debugging a tool

By default the agent talks to the MCP server over stdio, as a real subprocess with a real
JSON-RPC boundary - which is the point of the demo. `--inprocess` wires the same server object
straight into the client instead.
"""

import logging
import os
import sys

import anyio
from dotenv import load_dotenv
from mcp.client.stdio import StdioServerParameters, get_default_environment

from park_core.config import PROJECT_ROOT, settings
from park_core.tracing import flush_traces, setup_tracing

from .graph import run_agent

log = logging.getLogger(__name__)


# NOTE: the stdio client does not hand the child process your whole environment. It forwards a
# small safe allowlist (HOME, PATH, SHELL, TERM, USER, LOGNAME) and drops everything else, so
# without this the server subprocess starts with no OPENAI_API_KEY and no PG_DSN and parks_search
# fails with a credentials error that looks like it came from the agent.
FORWARDED_ENV = ("OPENAI_API_KEY", "PG_DSN", "CHROMA_DIR", "CHROMA_COLLECTION", "EMBED_MODEL")


def _server(inprocess: bool):
    if inprocess:
        from park_mcp.server import build_server

        return build_server()
    env = get_default_environment()
    env.update({k: v for k in FORWARDED_ENV if (v := os.getenv(k))})
    return StdioServerParameters(
        command="uv", args=["run", "park-mcp"], cwd=str(PROJECT_ROOT), env=env
    )


async def _ask(question: str, inprocess: bool) -> None:
    from park_core.models import make_chat_model

    # temperature 0 where the model allows it: recommendations should not wander between runs.
    model = make_chat_model(settings.agent_model)
    run = await run_agent(question, server=_server(inprocess), model=model)

    print("\n=== TOOLS CALLED ===")
    for name, args in run.trace.calls:
        print(f"  {name}({', '.join(f'{k}={v!r}' for k, v in args.items())})")
    print("\n=== ANSWER ===")
    print(run.answer)
    if run.hit_step_limit:
        print(f"\n(note: stopped at the {len(run.trace.calls)}-step limit)", file=sys.stderr)


async def _repl(inprocess: bool) -> None:
    print("=== PARK CONCIERGE === (blank line or 'exit' to quit)")
    while True:
        question = input("\nQuestion: ").strip()
        if not question or question.lower() in {"exit", "quit"}:
            break
        await _ask(question, inprocess)


def main() -> None:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    setup_tracing()  # after logging is configured so its status line shows; before any model call
    args = [a for a in sys.argv[1:] if a != "--inprocess"]
    inprocess = "--inprocess" in sys.argv[1:]

    try:
        if args:
            anyio.run(_ask, " ".join(args), inprocess)
        else:
            anyio.run(_repl, inprocess)
    finally:
        flush_traces()


if __name__ == "__main__":
    main()
