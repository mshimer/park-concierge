"""Fixtures: a throwaway Postgres, an offline Chroma collection, and a scriptable model.

The default suite must run with no OPENAI_API_KEY and spend nothing. That shapes two choices:
the vector store is built with `HashEmbeddings`, and anything needing a real model is either
scripted or marked `eval`.
"""

import pathlib
from collections.abc import Iterator

import psycopg
import pytest
from testcontainers.community.postgres import PostgresContainer

from park_core.chroma import HashEmbeddings
from park_core.config import PROJECT_ROOT
from park_core.ingest import ingest
from park_mcp.context import ServerContext

INIT_SQL = PROJECT_ROOT / "db" / "init.sql"


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    """Declared once here rather than in every async test module."""
    return "asyncio"


@pytest.fixture(scope="session")
def pg_dsn() -> Iterator[str]:
    """A real, throwaway Postgres with the demo schema applied."""
    with PostgresContainer("postgres:17", driver=None) as pg:
        # GOTCHA: get_connection_url() defaults to the psycopg2 driver and returns
        # postgresql+psycopg2://..., which psycopg3 rejects outright. driver=None gives plain
        # postgresql://.
        dsn = pg.get_connection_url()
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(INIT_SQL.read_text())
        yield dsn


@pytest.fixture(scope="session")
def embeddings() -> HashEmbeddings:
    return HashEmbeddings()


@pytest.fixture(scope="session")
def chroma_dir(tmp_path_factory: pytest.TempPathFactory, embeddings: HashEmbeddings) -> str:
    """The parks collection, embedded offline. Built once per session."""
    directory = str(tmp_path_factory.mktemp("chroma"))
    ingest(offline=True, chroma_dir=directory, embeddings=embeddings)
    return directory


@pytest.fixture(scope="session")
def ctx(pg_dsn: str, chroma_dir: str, embeddings: HashEmbeddings) -> ServerContext:
    """A ServerContext wired to the container and the offline collection.

    This is the payoff for making the toolsets closures over a context: the DSN below was only
    knowable after the container started, long after park_mcp was imported.
    """
    return ServerContext.from_settings(
        pg_dsn=pg_dsn, chroma_dir=chroma_dir, embeddings=embeddings
    )


@pytest.fixture(scope="session")
def real_chroma_dir() -> str:
    """The persisted collection built by `uv run park-ingest`, for the eval suite only."""
    directory = PROJECT_ROOT / "data" / "chroma"
    if not any(directory.glob("*.sqlite3")):
        pytest.skip("no vector store; run `uv run park-ingest` first")
    return str(directory)
