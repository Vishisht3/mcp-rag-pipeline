"""
mcp_server/__main__.py
Entrypoint for running the RAG MCP server as a standalone process.

Usage:
    python -m mcp_server [--config config/phase4.yaml]            # transport from config (default stdio)
    python -m mcp_server --config config/phase4.yaml --http        # force streamable-http
    python -m mcp_server --config config/phase4.yaml --http --port 9000
"""
from __future__ import annotations

import argparse

from dotenv import load_dotenv

from config.loader import load_config
from mcp_server.server import create_server


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG Pipeline MCP Server")
    parser.add_argument(
        "--config",
        default="config/phase4.yaml",
        help="Path to config YAML (default: config/phase4.yaml)",
    )
    transport_group = parser.add_mutually_exclusive_group()
    transport_group.add_argument(
        "--http",
        action="store_true",
        help="Run with streamable-http transport (overrides mcp.server.transport)",
    )
    transport_group.add_argument(
        "--stdio",
        action="store_true",
        help="Run with stdio transport (overrides mcp.server.transport)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="HTTP port override (implies --http; default: mcp.server.http_port)",
    )
    return parser


def resolve_transport(args: argparse.Namespace, cfg) -> tuple[str, int]:
    """Return (transport, port) after applying CLI overrides on top of config."""
    if args.port is not None and not (1 <= args.port <= 65535):
        raise SystemExit(f"--port must be between 1 and 65535 (got {args.port})")
    if args.stdio and args.port is not None:
        raise SystemExit("--port cannot be combined with --stdio")

    if args.http or args.port is not None:
        transport = "streamable-http"
    elif args.stdio:
        transport = "stdio"
    else:
        transport = cfg.mcp.server.transport

    port = args.port if args.port is not None else cfg.mcp.server.http_port
    return transport, port


def main(argv: list[str] | None = None):
    load_dotenv()

    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args.config)
    transport, port = resolve_transport(args, cfg)

    server = create_server(config_path=args.config)

    try:
        if transport == "streamable-http":
            server.run(transport="streamable-http", port=port)
        else:
            server.run(transport="stdio")
    finally:
        pipeline = getattr(server, "_pipeline", None)
        if pipeline is not None and hasattr(pipeline.vector_store, "close"):
            pipeline.vector_store.close()


if __name__ == "__main__":
    main()
