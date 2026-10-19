"""Shared foundation: settings, the Chroma access contract, and the NPS ingest.

`park_mcp` serves tools over the stores; `park_agent` drives the loop; both need to agree on
*where* the vectors live and *which* embedding model wrote them. That agreement lives here so the
two cannot drift apart.
"""
