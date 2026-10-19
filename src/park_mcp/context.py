"""What the toolsets are handed, and the annotation they all share.

Lives in its own module so neither `server` nor `toolsets` has to import the other at module
level - `server` imports `toolsets` to register them, so the dependency can only run one way.
"""

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings
from mcp.types import ToolAnnotations

from park_core.chroma import default_embeddings, open_collection
from park_core.config import settings

# Every tool in this server reads; none of them writes, and none of them reaches the open internet.
READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)


@dataclass(frozen=True)
class ServerContext:
    """Everything the toolsets need, passed in rather than read from the environment.

    NOTE: this is the whole reason the tools are closures (see toolsets/).
    `park_core.config.settings` is frozen at import time, which is fine in production but useless
    under testcontainers - the container's port, and therefore its DSN, is only known once a pytest
    fixture has started it, long after `park_mcp` was imported. A tool body that read `settings`
    would make the tests silently talk to the developer's own database instead of the container,
    and still pass. `monkeypatch.setenv` cannot fix that either. So the DSN travels as data.
    """

    pg_dsn: str
    chroma_dir: str
    collection: str
    snapshot_path: Path
    # None means "build the real OpenAI embeddings when something first needs them". Resolving it
    # lazily rather than here is what lets `park-mcp` start, list its tools and answer
    # parks_get/parks_alerts with no OPENAI_API_KEY at all - only the two tools that actually
    # embed a query need one.
    embeddings: Embeddings | None = None

    @classmethod
    def from_settings(cls, **overrides: Any) -> "ServerContext":
        """Production defaults, with any field overridable - which is what tests do."""
        base: dict[str, Any] = {
            "pg_dsn": settings.pg_dsn,
            "chroma_dir": settings.chroma_dir,
            "collection": settings.collection,
            "snapshot_path": settings.snapshot_path,
        }
        base.update(overrides)
        return cls(**base)

    @cached_property
    def store(self) -> Any:
        """The Chroma collection. Opened once per server, lazily."""
        return open_collection(self.chroma_dir, self.collection, self.embeddings or default_embeddings())

    @cached_property
    def parks_by_code(self) -> dict[str, dict[str, Any]]:
        """Structured park records, keyed by NPS code.

        JUMP-OFF POINT: alerts, designations and elevations are structured facts, not things you
        want to retrieve by cosine similarity, so `parks_get` and `parks_alerts` read them straight
        from the snapshot. The vector store stays responsible for what it is actually good at -
        `parks_search` and `parks_similar_to`. Moving these into Postgres (or pgvector) would be
        the natural next step.
        """
        parks = json.loads(self.snapshot_path.read_text())
        elevations_path = self.snapshot_path.parent / "park_elevations.json"
        elevations = json.loads(elevations_path.read_text()) if elevations_path.exists() else {}
        for park in parks:
            park.update(elevations.get(park["park_code"], {}))
        return {p["park_code"]: p for p in parks}
