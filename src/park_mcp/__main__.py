"""Run the server: `uv run park-mcp` (stdio) or `uv run park-mcp streamable-http`."""

import logging
import sys

from dotenv import load_dotenv

from .server import build_server


def main() -> None:
    load_dotenv()
    # stderr, never stdout: on stdio transport stdout *is* the JSON-RPC channel, and a stray
    # print or log line there corrupts the stream and the client just sees a parse error.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    mcp = build_server()
    match transport:
        case "stdio":
            mcp.run("stdio")
        case "streamable-http":
            print("park-mcp on http://127.0.0.1:8050/mcp", file=sys.stderr)
            mcp.run("streamable-http", host="127.0.0.1", port=8050)
        case _:
            raise SystemExit(f"unknown transport {transport!r}; use stdio or streamable-http")


if __name__ == "__main__":
    main()
