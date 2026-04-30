"""LangGraph agent graph for the Smart Travel Planner.

Nodes (in execution order):
  parse_intent      → cheap LLM extracts TripPreference from raw query
  check_sufficient  → gate: if not enough info, return NeedsMoreInfo immediately
  predict_styles    → heavy LLM predicts 1-2 travel styles from TripPreference
  rag_retrieve      → pgvector similarity search filtered by predicted styles
  live_conditions   → Open-Meteo weather for top RAG destinations
  synthesise        → heavy LLM writes the final travel recommendation

The graph is built once (build_graph()) and reused across requests.
"""

from __future__ import annotations

import functools
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import structlog
from google import genai
from google.genai import types
from langgraph.graph import END, StateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from agent.prompts import SYNTHESIS_PROMPT
from agent.router import _STRONG_FALLBACK_CHAIN, _call_with_fallback, _make_retry, parse_intent, predict_styles
from agent.tools.live_conditions import live_conditions
from agent.tools.rag_retrieve import rag_retrieve
from schemas.agent import AgentState, NeedsMoreInfoResponse, ToolCallRecord
from schemas.tools import LiveConditionsInput, RAGRetrieveInput, ToolError

log = structlog.get_logger()


# ── Node implementations ───────────────────────────────────────────────────────

async def node_parse_intent(state: AgentState, *, cheap_client: genai.Client, cheap_model: str) -> dict:
    t0 = time.monotonic()
    preferences = await parse_intent(cheap_client, cheap_model, state.query)
    duration_ms = int((time.monotonic() - t0) * 1000)
    log.info("node.parse_intent.done", duration_ms=duration_ms, sufficient=preferences.is_sufficient())
    return {"preferences": preferences}


async def node_predict_styles(
    state: AgentState,
    *,
    strong_client: genai.Client,
    strong_model: str,
) -> dict:
    t0 = time.monotonic()
    styles = await predict_styles(strong_client, strong_model, state.preferences)
    duration_ms = int((time.monotonic() - t0) * 1000)
    log.info("node.predict_styles.done", styles=styles, duration_ms=duration_ms)
    return {"styles_wanted": styles}


_MAX_DESTINATIONS_PER_STYLE = 2
_CHUNKS_PER_DESTINATION = 3


async def node_rag_retrieve(
    state: AgentState,
    *,
    db: AsyncSession,
    cheap_client: genai.Client,
    embedding_model: str,
    top_k: int,
) -> dict:
    # rag_chunks_by_style maps style → list of chunks (already capped to
    # _MAX_DESTINATIONS_PER_STYLE destinations, _CHUNKS_PER_DESTINATION chunks each).
    # We still keep the flat lists for backwards compat with live_conditions node.
    all_chunks: list[str] = []
    all_sources: list[str] = []
    rag_chunks_by_style: dict[str, list[str]] = {}
    dest_metadata: dict[str, Any] = {}  # destination name → {image_url, source_url}
    tool_calls: list[ToolCallRecord] = list(state.tool_calls)
    per_style_results: dict[str, Any] = {}

    t0_total = time.monotonic()
    for style in state.styles_wanted:
        t0 = time.monotonic()
        # Fetch enough chunks to ensure we can pick 2 distinct destinations
        fetch_k = _MAX_DESTINATIONS_PER_STYLE * _CHUNKS_PER_DESTINATION * 2
        inp = RAGRetrieveInput(query=state.query, travel_style=style, top_k=fetch_k)
        result = await rag_retrieve(inp, db, cheap_client, embedding_model)
        duration_ms = int((time.monotonic() - t0) * 1000)

        if isinstance(result, ToolError):
            log.warning("node.rag_retrieve.tool_error", style=style, error=result.error)
            per_style_results[style] = {"error": result.error}
            rag_chunks_by_style[style] = []
            continue

        # Cap to _MAX_DESTINATIONS_PER_STYLE distinct destinations,
        # keeping up to _CHUNKS_PER_DESTINATION chunks each.
        dest_seen: dict[str, int] = {}  # destination → chunks already kept
        style_chunks: list[str] = []
        style_sources: list[str] = []
        global_seen = set(all_chunks)
        for chunk, src, img_url, src_url in zip(
            result.chunks,
            result.source_documents,
            result.image_urls or ([None] * len(result.chunks)),
            result.source_urls or ([None] * len(result.chunks)),
        ):
            dest_count = dest_seen.get(src, 0)
            if len(dest_seen) >= _MAX_DESTINATIONS_PER_STYLE and src not in dest_seen:
                continue  # already at dest cap, skip new destinations
            if dest_count >= _CHUNKS_PER_DESTINATION:
                continue  # already have enough chunks for this destination
            if chunk in global_seen:
                continue
            style_chunks.append(chunk)
            style_sources.append(src)
            dest_seen[src] = dest_count + 1
            global_seen.add(chunk)
            # Store first-seen image/source URL per destination (dict deduplicates)
            if src not in dest_metadata:
                dest_metadata[src] = {"image_url": img_url, "source_url": src_url}

        all_chunks.extend(style_chunks)
        all_sources.extend(style_sources)
        rag_chunks_by_style[style] = style_chunks
        per_style_results[style] = {
            "destinations": list(dest_seen.keys()),
            "chunks": len(style_chunks),
        }

    total_ms = int((time.monotonic() - t0_total) * 1000)
    tool_calls.append(ToolCallRecord(
        tool_name="rag_retrieve",
        input={"query": state.query, "styles": state.styles_wanted, "top_k": top_k},
        output={"per_style": per_style_results, "total_chunks": len(all_chunks)},
        duration_ms=total_ms,
    ))

    log.info("node.rag_retrieve.done", chunks=len(all_chunks), sources=len(set(all_sources)))
    return {
        "rag_chunks": all_chunks,
        "rag_sources": all_sources,
        "rag_chunks_by_style": rag_chunks_by_style,
        "destination_metadata": dest_metadata,
        "tool_calls": tool_calls,
    }


async def node_live_conditions(state: AgentState) -> dict:
    # Run live conditions for the top 3 unique source destinations
    unique_destinations = list(dict.fromkeys(state.rag_sources))[:3]
    travel_month = (state.preferences.travel_month or 7) if state.preferences else 7

    tool_calls: list[ToolCallRecord] = list(state.tool_calls)
    conditions: dict[str, Any] = {}

    for dest in unique_destinations:
        t0 = time.monotonic()
        inp = LiveConditionsInput(destination_name=dest, travel_month=travel_month)
        result = await live_conditions(inp)
        duration_ms = int((time.monotonic() - t0) * 1000)

        tool_calls.append(ToolCallRecord(
            tool_name="live_conditions",
            input=inp.model_dump(),
            output=result.model_dump(),
            duration_ms=duration_ms,
        ))

        if isinstance(result, ToolError):
            log.warning("node.live_conditions.tool_error", dest=dest, error=result.error)
            conditions[dest] = {"error": result.error}
        else:
            conditions[dest] = result.model_dump()

    log.info("node.live_conditions.done", destinations=list(conditions.keys()))
    return {"live_conditions": conditions, "tool_calls": tool_calls}


_MONTH_NAMES = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def _build_synthesis_prompt(state: AgentState) -> str:
    """Build the synthesis prompt from agent state. Used by both batch and streaming paths."""
    by_style = getattr(state, "rag_chunks_by_style", None) or {}
    if by_style:
        rag_sections = []
        for style in state.styles_wanted:
            chunks = by_style.get(style, [])
            if chunks:
                rag_sections.append(f"[Style: {style}]\n" + "\n\n".join(chunks))
        rag_text = "\n\n---\n\n".join(rag_sections) if rag_sections else "No destination knowledge retrieved."
    else:
        rag_text = "\n\n".join(state.rag_chunks[:15]) if state.rag_chunks else "No destination knowledge retrieved."

    if state.live_conditions:
        cond_lines = []
        for dest, cond in state.live_conditions.items():
            if "error" in cond:
                cond_lines.append(f"{dest}: data unavailable ({cond['error']})")
            else:
                cond_lines.append(
                    f"{dest}: avg temp {cond.get('avg_temp_c', 'N/A')}°C, "
                    f"precip {cond.get('precipitation_mm', 'N/A')} mm — {cond.get('weather_summary', '')}"
                )
        live_text = "\n".join(cond_lines)
    else:
        live_text = "No live conditions data available."

    month_num = state.preferences.travel_month if state.preferences else None
    month_name = _MONTH_NAMES[month_num] if month_num else "your travel month"

    return SYNTHESIS_PROMPT.format(
        preferences=state.preferences.model_dump_json(indent=2) if state.preferences else "{}",
        styles=", ".join(state.styles_wanted),
        rag_chunks=rag_text,
        live_conditions=live_text,
        travel_month=month_name,
    )


async def node_synthesise(
    state: AgentState,
    *,
    strong_client: genai.Client,
    strong_model: str,
) -> dict:
    prompt = _build_synthesis_prompt(state)

    t0 = time.monotonic()
    response, _ = await _call_with_fallback(
        strong_client,
        strong_model,
        prompt,
        types.GenerateContentConfig(temperature=0.7),
        _STRONG_FALLBACK_CHAIN,
    )
    duration_ms = int((time.monotonic() - t0) * 1000)

    answer = response.text or "Sorry, I could not generate a recommendation. Please try again."

    usage = getattr(response, "usage_metadata", None)
    token_usage = dict(state.token_usage)
    if usage:
        token_usage["synthesis"] = {
            "prompt_tokens": getattr(usage, "prompt_token_count", None),
            "output_tokens": getattr(usage, "candidates_token_count", None),
        }

    log.info("node.synthesise.done", answer_len=len(answer), duration_ms=duration_ms)
    return {"answer": answer, "token_usage": token_usage}


# ── Routing ────────────────────────────────────────────────────────────────────

def _route_after_parse(state: AgentState) -> str:
    """Gate: go to predict_styles only if we have enough info."""
    if state.preferences and state.preferences.is_sufficient():
        return "predict_styles"
    return "insufficient"


# ── Graph builder ──────────────────────────────────────────────────────────────

async def run_agent(
    query: str,
    cheap_client: genai.Client,
    strong_client: genai.Client,
    cheap_model: str,
    strong_model: str,
    db: AsyncSession,
    embedding_model: str,
    top_k: int = 5,
) -> AgentState | NeedsMoreInfoResponse:
    """Run the agent graph and return the final AgentState.

    Returns NeedsMoreInfoResponse when the user's query lacks enough information
    for the agent to proceed — the caller streams this back to the client.
    """
    initial = AgentState(query=query, run_id=uuid.uuid4())

    graph = StateGraph(AgentState)

    graph.add_node(
        "parse_intent",
        functools.partial(node_parse_intent, cheap_client=cheap_client, cheap_model=cheap_model),
    )
    graph.add_node(
        "predict_styles",
        functools.partial(node_predict_styles, strong_client=strong_client, strong_model=strong_model),
    )
    graph.add_node(
        "rag_retrieve",
        functools.partial(
            node_rag_retrieve, db=db, cheap_client=cheap_client,
            embedding_model=embedding_model, top_k=top_k,
        ),
    )
    graph.add_node("live_conditions", node_live_conditions)
    graph.add_node(
        "synthesise",
        functools.partial(node_synthesise, strong_client=strong_client, strong_model=strong_model),
    )

    graph.set_entry_point("parse_intent")

    graph.add_conditional_edges(
        "parse_intent",
        _route_after_parse,
        {"predict_styles": "predict_styles", "insufficient": END},
    )

    graph.add_edge("predict_styles", "rag_retrieve")
    graph.add_edge("rag_retrieve", "live_conditions")
    graph.add_edge("live_conditions", "synthesise")
    graph.add_edge("synthesise", END)

    compiled = graph.compile()
    final_state_dict = await compiled.ainvoke(initial.model_dump())
    final = AgentState.model_validate(final_state_dict)

    # Graph stopped at "insufficient" — answer will be empty
    if not final.answer and final.preferences and not final.preferences.is_sufficient():
        from agent.prompts import CLARIFICATION_PROMPT
        missing = final.preferences.missing_fields()
        message = CLARIFICATION_PROMPT.format(missing_fields=", ".join(missing))
        return NeedsMoreInfoResponse(
            message=message,
            missing_fields=missing,
            partial_preferences=final.preferences,
        )

    return final


async def _run_pre_synthesis_graph(
    query: str,
    cheap_client: genai.Client,
    strong_client: genai.Client,
    cheap_model: str,
    strong_model: str,
    db: AsyncSession,
    embedding_model: str,
    top_k: int,
) -> AgentState | NeedsMoreInfoResponse:
    """Run all graph nodes except synthesis. Returns intermediate AgentState."""
    initial = AgentState(query=query, run_id=uuid.uuid4())

    graph = StateGraph(AgentState)
    graph.add_node(
        "parse_intent",
        functools.partial(node_parse_intent, cheap_client=cheap_client, cheap_model=cheap_model),
    )
    graph.add_node(
        "predict_styles",
        functools.partial(node_predict_styles, strong_client=strong_client, strong_model=strong_model),
    )
    graph.add_node(
        "rag_retrieve",
        functools.partial(
            node_rag_retrieve, db=db, cheap_client=cheap_client,
            embedding_model=embedding_model, top_k=top_k,
        ),
    )
    graph.add_node("live_conditions", node_live_conditions)

    graph.set_entry_point("parse_intent")
    graph.add_conditional_edges(
        "parse_intent",
        _route_after_parse,
        {"predict_styles": "predict_styles", "insufficient": END},
    )
    graph.add_edge("predict_styles", "rag_retrieve")
    graph.add_edge("rag_retrieve", "live_conditions")
    graph.add_edge("live_conditions", END)

    compiled = graph.compile()
    state_dict = await compiled.ainvoke(initial.model_dump())
    state = AgentState.model_validate(state_dict)

    if not state.styles_wanted and state.preferences and not state.preferences.is_sufficient():
        from agent.prompts import CLARIFICATION_PROMPT
        missing = state.preferences.missing_fields()
        message = CLARIFICATION_PROMPT.format(missing_fields=", ".join(missing))
        return NeedsMoreInfoResponse(
            message=message,
            missing_fields=missing,
            partial_preferences=state.preferences,
        )

    return state


async def run_agent_stream(
    query: str,
    cheap_client: genai.Client,
    strong_client: genai.Client,
    cheap_model: str,
    strong_model: str,
    db: AsyncSession,
    embedding_model: str,
    top_k: int = 5,
) -> AsyncIterator[str]:
    """Run the agent and stream SSE events.

    Yields raw SSE-formatted strings:
      data: {"type": "tool_call", ...}\n\n
      data: {"type": "token", "text": "..."}\n\n
      data: {"type": "done", "run": {...}}\n\n
      data: {"type": "error", "message": "..."}\n\n
    """
    import json

    def _sse(payload: dict) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    try:
        pre = await _run_pre_synthesis_graph(
            query, cheap_client, strong_client, cheap_model, strong_model,
            db, embedding_model, top_k,
        )
    except Exception as exc:
        log.error("stream.pre_synthesis.error", error=str(exc))
        yield _sse({"type": "error", "message": str(exc)})
        return

    if isinstance(pre, NeedsMoreInfoResponse):
        yield _sse({"type": "needs_more_info", "message": pre.message, "missing_fields": pre.missing_fields})
        return

    # Emit tool_call events so the frontend can display the trace live
    for tc in pre.tool_calls:
        yield _sse({
            "type": "tool_call",
            "tool_name": tc.tool_name,
            "input": tc.input,
            "output": tc.output,
            "duration_ms": tc.duration_ms,
        })

    # Build synthesis prompt and stream the response
    prompt = _build_synthesis_prompt(pre)
    config = types.GenerateContentConfig(temperature=0.7)

    chain = [strong_model] + [m for m in _STRONG_FALLBACK_CHAIN if m != strong_model]
    answer_parts: list[str] = []
    streamed = False

    for candidate in chain:
        try:
            async for attempt in _make_retry():
                with attempt:
                    stream = await strong_client.aio.models.generate_content_stream(
                        model=candidate,
                        contents=prompt,
                        config=config,
                    )
                    async for chunk in stream:
                        text = getattr(chunk, "text", None) or ""
                        if text:
                            answer_parts.append(text)
                            yield _sse({"type": "token", "text": text})
                    streamed = True
            if streamed:
                break
        except Exception as exc:
            from google.genai import errors as genai_errors
            code = getattr(exc, "code", None)
            if isinstance(exc, (genai_errors.ServerError, genai_errors.ClientError)) and code in (429, 500, 503):
                log.warning("stream.synthesis.model_unavailable", model=candidate, code=code)
                continue
            log.error("stream.synthesis.error", model=candidate, error=str(exc))
            yield _sse({"type": "error", "message": str(exc)})
            return

    if not answer_parts:
        fallback = "Sorry, I could not generate a recommendation. Please try again."
        answer_parts.append(fallback)
        yield _sse({"type": "token", "text": fallback})

    full_answer = "".join(answer_parts)
    log.info("stream.synthesis.done", answer_len=len(full_answer))

    dest_meta_list = [
        {"name": name, "image_url": meta.get("image_url"), "source_url": meta.get("source_url")}
        for name, meta in pre.destination_metadata.items()
    ]

    yield _sse({
        "type": "done",
        "run_id": str(pre.run_id),
        "answer": full_answer,
        "styles_predicted": pre.styles_wanted,
        "destination_metadata": dest_meta_list,
        "tool_calls": [tc.model_dump() for tc in pre.tool_calls],
        "token_usage": pre.token_usage,
    })
