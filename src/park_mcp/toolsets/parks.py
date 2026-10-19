"""Park knowledge-base tools - the RAG half of the demo.

`parks_search` and `parks_similar_to` go through the Chroma vector store. `parks_get` and
`parks_alerts` read structured fields from the snapshot (see ServerContext.parks_by_code for why).
"""

import logging
from typing import Any

from mcp.server.mcpserver import MCPServer

from park_core.chroma import park_document

from ..context import READ_ONLY, ServerContext

log = logging.getLogger(__name__)
MAX_K = 10


def _where(state: str | None, max_elevation_ft: int | None) -> dict[str, Any] | None:
    """Build a Chroma metadata filter.

    JUMP-OFF POINT: soft preference belongs in the query text, hard constraints belong here. The
    embedding has no idea that 5000 is less than 7860, so a query that merely mentions "under
    5000 ft" cheerfully returns alpine parks whose prose sounds appealing. Every tool that
    retrieves parks takes the same filter - a constraint that only one retrieval path can enforce
    is not a constraint, it is a coin flip over which tool the model happens to pick.

    Chroma allows one top-level operator, so multiple conditions have to be wrapped in `$and`.
    """
    conditions: list[dict[str, Any]] = []
    if state:
        conditions.append({"states": {"$eq": state}})
    if max_elevation_ft is not None:
        # elevation_ft_min: the park has *some* terrain at or below the visitor's ceiling.
        conditions.append({"elevation_ft_min": {"$lte": max_elevation_ft}})
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def register(mcp: MCPServer, ctx: ServerContext) -> None:
    """Register the park tools against `ctx`. See customers.py for why these are closures."""

    def _hits_to_parks(docs: list[Any], exclude: str | None = None) -> list[dict[str, Any]]:
        """Collapse chunk hits into one entry per park, keeping best-match order.

        A park can be split across several chunks, so a raw similarity search happily returns the
        same park three times. Dedupe here rather than making the model reason about it.
        """
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for doc in docs:
            code = doc.metadata["park_code"]
            if code in seen or code == exclude:
                continue
            seen.add(code)
            park = ctx.parks_by_code.get(code, {})
            out.append(
                {
                    "park_code": code,
                    "name": doc.metadata["name"],
                    "states": doc.metadata["states"],
                    "designation": doc.metadata["designation"],
                    "matched_text": doc.page_content,
                    "elevation_ft_min": park.get("elevation_ft_min"),
                    "elevation_ft_max": park.get("elevation_ft_max"),
                    "alert_count": len(park.get("alerts", [])),
                }
            )
        return out

    @mcp.tool(annotations=READ_ONLY)
    def parks_search(
        query: str,
        k: int = 5,
        state: str | None = None,
        max_elevation_ft: int | None = None,
    ) -> list[dict[str, Any]]:
        """Search the park knowledge base by meaning, not keywords.

        Describe what the visitor wants - terrain, climate, season, activities, accessibility - and
        this returns the closest matching parks with the text that matched. These results are the
        only parks you may recommend.

        `state` restricts to a two-letter state code. `max_elevation_ft` keeps only parks that have
        terrain at or below that elevation - pass the customer's limit here rather than describing
        it in the query, because a hard ceiling is a filter, not a matter of similarity.
        """
        k = max(1, min(k, MAX_K))
        # Over-fetch: several chunks can belong to one park, and dedupe shrinks the list.
        docs = ctx.store.similarity_search(query, k=k * 3, filter=_where(state, max_elevation_ft))
        return _hits_to_parks(docs)[:k]

    @mcp.tool(annotations=READ_ONLY)
    def parks_get(park_code: str) -> dict[str, Any] | None:
        """Full record for one park: description, activities, topics, elevation range, weather.

        `park_code` is the NPS code, e.g. 'zion'. Returns null for an unknown code.
        """
        return ctx.parks_by_code.get(park_code.lower())

    @mcp.tool(annotations=READ_ONLY)
    def parks_alerts(park_code: str) -> list[dict[str, Any]]:
        """Current alerts for one park: closures, road status, hazards, reservation requirements.

        Always check this before recommending a park, and tell the visitor about anything that
        would affect their trip. Returns an empty list when there are no active alerts.
        """
        park = ctx.parks_by_code.get(park_code.lower())
        return list(park.get("alerts", [])) if park else []

    @mcp.tool(annotations=READ_ONLY)
    def parks_similar_to(
        park_code: str,
        k: int = 5,
        state: str | None = None,
        max_elevation_ft: int | None = None,
    ) -> list[dict[str, Any]]:
        """Find the parks most like a given one - "they loved Zion, where else?".

        Embeds the named park's own description and returns its nearest neighbours in the vector
        store, excluding itself. Useful for recommending against a customer's visit history.

        Takes the same `state` and `max_elevation_ft` filters as parks_search, and for the same
        reason: pass the customer's elevation limit here so the suggestions respect it.
        """
        k = max(1, min(k, MAX_K))
        park = ctx.parks_by_code.get(park_code.lower())
        if park is None:
            return []
        docs = ctx.store.similarity_search(
            park_document(park), k=(k + 1) * 3, filter=_where(state, max_elevation_ft)
        )
        return _hits_to_parks(docs, exclude=park["park_code"])[:k]
