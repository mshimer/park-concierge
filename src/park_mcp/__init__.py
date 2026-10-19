"""MCP server exposing the two data sources as tools.

`customers` reads Postgres, `parks` reads the Chroma vector store plus the structured park
snapshot. Every tool is read-only.
"""
