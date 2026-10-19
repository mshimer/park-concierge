"""Customer database tools - the "query an internal data source" half of the demo."""

import logging
from typing import Any

import psycopg
from mcp.server.mcpserver import MCPServer
from psycopg.rows import dict_row

from ..context import READ_ONLY, ServerContext

log = logging.getLogger(__name__)
MAX_ROWS = 100


def register(mcp: MCPServer, ctx: ServerContext) -> None:
    """Register the customer tools against `ctx`.

    The tool functions are defined *inside* register() on purpose, for two reasons:

      1. `@mcp.tool()` is bound to a server instance, so `mcp` has to exist at `def` time. A
         module-level server singleton would break `build_server`, because importlib caches the
         module and a second build would register nothing.
      2. Closing over `ctx` is what lets a test point these tools at a throwaway container.

    (Note `@mcp.tool()` needs the parentheses - bare `@mcp.tool` raises TypeError.)
    """
    def _query(sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        # read_only makes Postgres itself reject writes, so no tool here can mutate data even if
        # the SQL is wrong or the model gets creative.
        with psycopg.connect(ctx.pg_dsn, row_factory=dict_row) as conn:
            conn.read_only = True
            with conn.cursor() as cur:
                cur.execute(sql, params)  # type: ignore[arg-type]
                return cur.fetchmany(MAX_ROWS)

    @mcp.tool(annotations=READ_ONLY)
    def customer_lookup(customer_id: int) -> dict[str, Any] | None:
        """Look up one park-pass customer by id.

        Returns their pass tier, home state, party size, stated accessibility needs and any maximum
        elevation they can manage. Returns null if no such customer exists - do not invent one.
        """
        rows = _query(
            """SELECT id, name, email, pass_tier, home_state, party_size,
                      accessibility_needs, max_elevation_ft
               FROM customer WHERE id = %s""",
            (customer_id,),
        )
        return rows[0] if rows else None

    @mcp.tool(annotations=READ_ONLY)
    def customer_find_by_email(email: str) -> dict[str, Any] | None:
        """Look up one park-pass customer by email address. Returns null if there is no match."""
        rows = _query(
            """SELECT id, name, email, pass_tier, home_state, party_size,
                      accessibility_needs, max_elevation_ft
               FROM customer WHERE lower(email) = lower(%s)""",
            (email,),
        )
        return rows[0] if rows else None

    @mcp.tool(annotations=READ_ONLY)
    def customer_visits(customer_id: int, limit: int = 20) -> list[dict[str, Any]]:
        """List a customer's past park visits, newest first, with the rating they gave.

        `park_code` matches the codes used by the parks_* tools.
        """
        return _query(
            """SELECT park_code, visited_on, rating
               FROM visit WHERE customer_id = %s
               ORDER BY visited_on DESC LIMIT %s""",
            (customer_id, min(limit, MAX_ROWS)),
        )
