import os
from dataclasses import dataclass, field
from pathlib import Path

# src/park_core/config.py -> src/park_core -> src -> <project root>
PROJECT_ROOT = Path(__file__).resolve().parents[2] # ../..


def _env(name: str, default: str):
    return field(default_factory=lambda: os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    """Process-wide defaults. Read from the environment once, at import.

    NOTE: nothing downstream should reach for `settings` at call time. The MCP toolsets take an
    explicit `ServerContext` instead, because under testcontainers the Postgres DSN is only known
    *after* this module has already been imported - see park_mcp.server.ServerContext.
    """

    pg_dsn: str = _env("PG_DSN", "postgresql://postgres:postgres@localhost:5433/parks")
   
    # Default is absolute, resolved from __file__: `uv run` from a subdirectory must not silently
    # create a second, empty chroma dir next to the cwd.
   
    chroma_dir: str = _env("CHROMA_DIR", str(PROJECT_ROOT / "data" / "chroma"))
   
    collection: str = _env("CHROMA_COLLECTION", "parks")
   
    embed_model: str = _env("EMBED_MODEL", "text-embedding-3-small")
   
    agent_model: str = _env("AGENT_MODEL", "gpt-4o")
   
    # Deliberately a different (and cheaper) model than the agent: a judge that shares the
    # agent's weights tends to rubber-stamp its own reasoning.
   
    judge_model: str = _env("JUDGE_MODEL", "gpt-4o-mini")
   
    nps_api_key: str = _env("NPS_API_KEY", "")

    @property
    def snapshot_path(self) -> Path:
        return PROJECT_ROOT / "data" / "parks_snapshot.json"


settings = Settings()
