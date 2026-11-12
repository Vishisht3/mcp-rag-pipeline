"""
agent/mcp_client.py
MCP Client Manager: connects to external MCP servers (MCP SDK v2) and translates tool schemas.
"""
from __future__ import annotations

import json
import logging
import os
import re
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from config.loader import MCPServerConnectionConfig

load_dotenv()

logger = logging.getLogger(__name__)


def _substitute_env_vars(env_dict: Dict[str, str], server_name: str = "") -> Dict[str, str]:
    """
    Substitute ${VAR} patterns with os.environ values.
    Raises ValueError if any referenced environment variable is unset or empty.
    """
    res: Dict[str, str] = {}
    pattern = re.compile(r"\$\{([A-Za-z0-9_]+)\}")
    missing: List[str] = []

    for k, v in env_dict.items():
        def repl(match):
            var_name = match.group(1)
            val = os.getenv(var_name)
            if val is None or not val.strip():
                missing.append(var_name)
                return ""
            return val

        res[k] = pattern.sub(repl, v)

    if missing:
        srv_label = f" for MCP server '{server_name}'" if server_name else ""
        unique_missing = list(dict.fromkeys(missing))
        raise ValueError(
            f"Missing required environment variable(s){srv_label}: {', '.join(unique_missing)}"
        )

    return res


def _extract_input_schema(tool: Any, server_name: str) -> Dict[str, Any]:
    """Extract and validate MCP v2 tool.input_schema."""
    schema = getattr(tool, "input_schema", None)
    if schema is None:
        schema = getattr(tool, "inputSchema", None)
    if not isinstance(schema, dict):
        raise ValueError(
            f"Tool '{getattr(tool, 'name', '<unknown>')}' from server '{server_name}' "
            f"has invalid or missing input_schema: {schema!r}"
        )
    return schema


class MCPClientManager:
    def __init__(self, server_configs: List[MCPServerConnectionConfig]):
        seen_names = set()
        for cfg in server_configs:
            if cfg.name in seen_names:
                raise ValueError(f"Duplicate MCP server name: '{cfg.name}'")
            seen_names.add(cfg.name)

        self.server_configs = [c for c in server_configs if c.enabled]
        self._sessions: Dict[str, Any] = {}
        self._exit_stack: Optional[AsyncExitStack] = None
        self._tool_map: Dict[str, Dict[str, Any]] = {}
        self.connection_status: Dict[str, Dict[str, Any]] = {}

    def get_connection_status(self) -> Dict[str, Dict[str, Any]]:
        """Return the connection status report for all configured MCP servers."""
        return dict(self.connection_status)

    async def connect_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Connects to all enabled MCP servers using MCP SDK v2 protocol negotiation
        and populates their tool schemas. Returns a per-server connection status map.
        """
        try:
            from mcp import Client, ClientSession, StdioServerParameters
            from mcp.client._probe import negotiate_auto
            from mcp.client.stdio import stdio_client
            from mcp.client.streamable_http import streamable_http_client
        except ImportError as exc:
            err_msg = f"mcp>=2.3.0 package is required for external MCP tools: {exc}"
            logger.error(err_msg)
            for s_cfg in self.server_configs:
                self.connection_status[s_cfg.name] = {
                    "connected": False,
                    "tools": [],
                    "error": err_msg,
                }
            raise ImportError(err_msg) from exc

        if self._exit_stack is not None:
            await self.close_all()

        self._exit_stack = AsyncExitStack()
        self.connection_status = {}

        for s_cfg in self.server_configs:
            try:
                if s_cfg.transport == "stdio":
                    if not s_cfg.command:
                        raise ValueError(
                            f"MCP server '{s_cfg.name}' has transport 'stdio' but no command specified."
                        )
                    resolved_env = _substitute_env_vars(s_cfg.env, server_name=s_cfg.name)
                    env = {**os.environ, **resolved_env}
                    params = StdioServerParameters(
                        command=s_cfg.command,
                        args=s_cfg.args,
                        env=env,
                    )
                    transport_cm = stdio_client(params)
                elif s_cfg.transport == "streamable-http":
                    if not s_cfg.url:
                        raise ValueError(
                            f"MCP server '{s_cfg.name}' has transport 'streamable-http' but no url specified."
                        )
                    transport_cm = streamable_http_client(s_cfg.url)
                else:
                    raise ValueError(
                        f"Unknown transport '{s_cfg.transport}' for MCP server '{s_cfg.name}'."
                    )

                streams = await self._exit_stack.enter_async_context(transport_cm)
                read_stream, write_stream = streams[0], streams[1]
                session = await self._exit_stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
                )

                # Negotiate modern 2026-07-28 protocol via discovery with automatic fallback to legacy initialize()
                if hasattr(session, "send_discover"):
                    await negotiate_auto(session)
                else:
                    await session.initialize()

                self._sessions[s_cfg.name] = session

                tools_response = await session.list_tools()
                exposed_tools: List[str] = []
                for tool in tools_response.tools:
                    if s_cfg.tool_allowlist and tool.name not in s_cfg.tool_allowlist:
                        continue
                    input_schema = _extract_input_schema(tool, server_name=s_cfg.name)
                    namespaced_name = f"{s_cfg.name}__{tool.name}"
                    self._tool_map[namespaced_name] = {
                        "server": s_cfg.name,
                        "tool_name": tool.name,
                        "raw_tool": tool,
                        "schema": {
                            "type": "function",
                            "function": {
                                "name": namespaced_name,
                                "description": f"[{s_cfg.name}] {tool.description or ''}".strip(),
                                "parameters": input_schema,
                            },
                        },
                    }
                    exposed_tools.append(namespaced_name)

                self.connection_status[s_cfg.name] = {
                    "connected": True,
                    "tools": exposed_tools,
                    "error": None,
                }
            except Exception as e:
                logger.warning(f"Failed to connect to MCP server '{s_cfg.name}': {e}")
                self.connection_status[s_cfg.name] = {
                    "connected": False,
                    "tools": [],
                    "error": str(e),
                }

        return dict(self.connection_status)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Returns all discovered tool schemas for OpenRouter/OpenAI function calling."""
        return [item["schema"] for item in self._tool_map.values()]

    async def call_tool(self, namespaced_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Calls a namespaced MCP tool and returns result or error dict."""
        if namespaced_name not in self._tool_map:
            return {"error": f"Tool '{namespaced_name}' not found.", "is_error": True}

        info = self._tool_map[namespaced_name]
        server_name = info["server"]
        tool_name = info["tool_name"]
        session = self._sessions.get(server_name)

        if not session:
            return {
                "error": f"Session for server '{server_name}' is not connected.",
                "is_error": True,
            }

        try:
            res = await session.call_tool(tool_name, arguments)
            is_error = bool(getattr(res, "is_error", getattr(res, "isError", False)))
            structured_content = getattr(
                res, "structured_content", getattr(res, "structuredContent", None)
            )

            text_outputs: List[str] = []
            if hasattr(res, "content") and res.content:
                for item in res.content:
                    if hasattr(item, "text") and item.text is not None:
                        text_outputs.append(str(item.text))
                    elif isinstance(item, dict) and "text" in item:
                        text_outputs.append(str(item["text"]))
                    else:
                        text_outputs.append(str(item))

            if structured_content is not None:
                structured_str = (
                    json.dumps(structured_content)
                    if not isinstance(structured_content, str)
                    else structured_content
                )
                if not text_outputs:
                    output_str = structured_str
                else:
                    output_str = "\n".join(text_outputs)
            elif text_outputs:
                output_str = "\n".join(text_outputs)
            else:
                output_str = str(res)

            payload: Dict[str, Any] = {
                "result": output_str,
                "is_error": is_error,
            }
            if structured_content is not None:
                payload["structured_content"] = structured_content
            if is_error:
                payload["error"] = output_str or f"Tool '{namespaced_name}' returned an error."

            return payload
        except Exception as e:
            return {"error": f"MCP tool call failed: {str(e)}", "is_error": True}

    async def close_all(self) -> None:
        if self._exit_stack:
            try:
                await self._exit_stack.aclose()
            finally:
                self._exit_stack = None
        self._sessions.clear()
        self._tool_map.clear()
