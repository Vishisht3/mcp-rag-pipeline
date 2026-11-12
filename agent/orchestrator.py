"""
agent/orchestrator.py
Agentic RAG orchestrator: manages the LLM tool-calling loop between local RAG and external MCP tools.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent.mcp_client import MCPClientManager
from agent.tools import LOCAL_SEARCH_TOOL_SCHEMA, run_local_search
from config.loader import PipelineConfig, create_llm_client, get_default_model
from store.vector_store import RetrievedChunk

URL_REGEX = re.compile(r"https?://[^\s<>\"'\)\]\},]+")


@dataclass
class AgenticAnswer:
    question: str
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    citations: List[str]
    tool_trace: List[Dict[str, Any]]
    web_sources: List[Dict[str, Any]] = field(default_factory=list)
    mcp_status: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def pretty_print(self) -> None:
        """Print a formatted answer and tool trace to stdout."""
        try:
            from rich.console import Console
            from rich.panel import Panel

            console = Console()
            console.print(Panel(self.answer, title="Agentic Answer", border_style="cyan"))
            console.print("\n[bold]Tool Trace:[/bold]")
            for step in self.tool_trace:
                console.print(f"  • Tool: [bold green]{step['tool']}[/bold green] | Args: {step['args']}")
            if self.retrieved_chunks:
                console.print("\n[bold]Local Chunks Used:[/bold]")
                for chunk in self.retrieved_chunks:
                    console.print(f"  • [{chunk.citation_id}] {chunk.source} (score={chunk.score:.4f})")
            if self.web_sources:
                console.print("\n[bold]Web Sources Used:[/bold]")
                for i, src in enumerate(self.web_sources, 1):
                    console.print(f"  • [web:{i}] {src.get('url', 'N/A')}")
        except ImportError:
            print(f"Answer:\n{self.answer}\n")
            print("Tool Trace:", self.tool_trace)


class AgenticRAGPipeline:
    def __init__(
        self,
        retriever: Any,
        mcp_manager: MCPClientManager,
        cfg: PipelineConfig,
    ):
        self.retriever = retriever
        self.mcp_manager = mcp_manager
        self.cfg = cfg
        self._llm = self._init_llm()

    def _init_llm(self):
        return create_llm_client(self.cfg)

    async def _call_llm_async(self, **kwargs: Any) -> Any:
        """Invoke LLM chat completion without blocking the async event loop."""
        create_fn = self._llm.chat.completions.create
        if inspect.iscoroutinefunction(create_fn):
            return await create_fn(**kwargs)
        return await asyncio.to_thread(create_fn, **kwargs)

    def answer(self, question: str, model: Optional[str] = None) -> AgenticAnswer:
        """
        Synchronous entry point that runs the async agentic loop.
        Safely handles callers that already have a running event loop in the current thread.
        """
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is not None and running_loop.is_running():
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self._async_answer(question, model=model))
                return future.result()

        return asyncio.run(self._async_answer(question, model=model))

    async def aanswer(self, question: str, model: Optional[str] = None) -> AgenticAnswer:
        """Async entry point for async frameworks and MCP server handlers."""
        return await self._async_answer(question, model=model)

    async def _async_answer(self, question: str, model: Optional[str] = None) -> AgenticAnswer:
        if self._llm is None:
            raise RuntimeError(
                "LLM client not initialised. Check OPENROUTER_API_KEY (or OPENAI_API_KEY)."
            )

        effective_model = model or get_default_model(self.cfg)

        agent_cfg = self.cfg.agent
        strategy = agent_cfg.strategy
        max_iterations = agent_cfg.max_tool_iterations
        score_threshold = agent_cfg.confidence_threshold

        mcp_status: Dict[str, Dict[str, Any]] = {}
        try:
            conn_res = await self.mcp_manager.connect_all()
            if isinstance(conn_res, dict):
                mcp_status = conn_res
            elif hasattr(self.mcp_manager, "get_connection_status"):
                status_cand = self.mcp_manager.get_connection_status()
                if isinstance(status_cand, dict):
                    mcp_status = status_cand

            external_tools = self.mcp_manager.get_tool_schemas()
            local_tool = LOCAL_SEARCH_TOOL_SCHEMA

            if strategy in ("rag_only", "rag_then_fallback"):
                available_tools = [local_tool]
            else:
                available_tools = [local_tool] + external_tools

            system_prompt = self.cfg.prompts.agent_system or (
                "You are a helpful research assistant. Prefer search_local_documents first. "
                "Use external tools if local context is insufficient or missing."
            )

            messages: List[Dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]

            tool_trace: List[Dict[str, Any]] = []
            retrieved_chunks: List[RetrievedChunk] = []
            web_sources: List[Dict[str, Any]] = []
            citations: List[str] = []

            iteration = 0
            final_answer_text = ""

            while iteration < max_iterations:
                iteration += 1

                kwargs: Dict[str, Any] = {
                    "model": effective_model,
                    "messages": messages,
                    "temperature": 0.0,
                }
                if available_tools:
                    kwargs["tools"] = available_tools
                    if strategy == "rag_then_fallback" and iteration == 1:
                        kwargs["tool_choice"] = {
                            "type": "function",
                            "function": {"name": "search_local_documents"},
                        }
                    else:
                        kwargs["tool_choice"] = "auto"

                response = await self._call_llm_async(**kwargs)
                message = response.choices[0].message
                tool_calls = getattr(message, "tool_calls", None)

                if not tool_calls:
                    final_answer_text = message.content or ""
                    break

                msg_dict: Dict[str, Any] = {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in tool_calls
                    ],
                }
                messages.append(msg_dict)

                for tc in tool_calls:
                    tool_name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments)
                    except Exception:
                        args = {"query": tc.function.arguments}

                    if tool_name == "search_local_documents":
                        res = run_local_search(
                            self.retriever,
                            args.get("query", question),
                            score_threshold=score_threshold,
                        )
                        raw_res = res.pop("_raw_result", None)
                        if raw_res is not None:
                            retrieved_chunks.extend(raw_res.chunks)
                            citations.extend(raw_res.citations)

                        tool_result_content = json.dumps(res)
                        summary = f"Found {len(res['chunks'])} chunks (top_score={res['top_score']:.4f})"
                        is_err = False

                        if strategy == "rag_then_fallback" and res.get("below_confidence_threshold"):
                            for ext_t in external_tools:
                                if ext_t not in available_tools:
                                    available_tools.append(ext_t)

                    else:
                        res = await self.mcp_manager.call_tool(tool_name, args)
                        is_err = bool(res.get("is_error") or "error" in res)
                        summary = str(res.get("error") if is_err else res.get("result", ""))[:200]

                        new_web_refs = []
                        if not is_err and "result" in res:
                            raw_urls = URL_REGEX.findall(str(res["result"]))
                            for url in raw_urls:
                                url = url.rstrip(".,;:!?)\\]}")
                                existing_idx = next(
                                    (i + 1 for i, ws in enumerate(web_sources) if ws.get("url") == url),
                                    None,
                                )
                                if existing_idx is None:
                                    web_sources.append({"url": url})
                                    existing_idx = len(web_sources)
                                new_web_refs.append(f"[web:{existing_idx}] {url}")

                        if new_web_refs:
                            res["web_citations"] = new_web_refs
                        tool_result_content = json.dumps(res)

                    tool_trace.append({
                        "iteration": iteration,
                        "tool": tool_name,
                        "args": args,
                        "is_error": is_err,
                        "result_summary": summary,
                    })

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result_content,
                    })

            if not final_answer_text and messages:
                try:
                    synth_resp = await self._call_llm_async(
                        model=effective_model,
                        messages=messages,
                        temperature=0.0,
                    )
                    final_answer_text = synth_resp.choices[0].message.content or ""
                except Exception:
                    final_answer_text = "Reached maximum tool iterations without complete answer."

            if hasattr(self.retriever, "enforcer") and self.retriever.enforcer is not None:
                enforcer = self.retriever.enforcer
                if hasattr(enforcer, "enforce_agentic"):
                    def _retry_agentic(prev_ans: str) -> str:
                        valid_local = [f"[{c.citation_id}]" for c in retrieved_chunks]
                        valid_web = [f"[web:{i+1}] ({ws['url']})" for i, ws in enumerate(web_sources)]
                        retry_msg = (
                            f"Your previous answer was missing valid inline citations.\n"
                            f"Valid local citations: {valid_local}\n"
                            f"Valid web citations: {valid_web}\n"
                            f"Rewrite the answer ensuring every factual claim is cited:\n{prev_ans}"
                        )
                        # Runs inside a worker thread (see asyncio.to_thread below), so a
                        # blocking client call here does not stall the event loop.
                        r = self._llm.chat.completions.create(
                            model=effective_model,
                            messages=messages + [{"role": "user", "content": retry_msg}],
                            temperature=0.0,
                        )
                        return r.choices[0].message.content or prev_ans

                    check = await asyncio.to_thread(
                        enforcer.enforce_agentic,
                        final_answer_text,
                        retrieved_chunks,
                        web_sources,
                        retry_fn=_retry_agentic,
                    )
                    final_answer_text = check.answer
                else:
                    check = await asyncio.to_thread(
                        enforcer.check_agentic, final_answer_text, retrieved_chunks, web_sources
                    )
                    if not check.is_valid:
                        final_answer_text = check.answer

            return AgenticAnswer(
                question=question,
                answer=final_answer_text,
                retrieved_chunks=retrieved_chunks,
                citations=list(dict.fromkeys(citations)),
                tool_trace=tool_trace,
                web_sources=web_sources,
                mcp_status=mcp_status,
            )
        finally:
            await self.mcp_manager.close_all()
