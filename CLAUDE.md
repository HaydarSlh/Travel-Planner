# Smart Travel Planner — Project Reference

This file is the project's authoritative source for architecture decisions and engineering rules. Every code change should be consistent with what's written here. Update this file when a decision changes — never leave stale rules.

---

## 1. What this project is

A full-stack AI travel planner. A user asks "I have two weeks in July, $1,500, want somewhere warm and not touristy with hiking — where should I go?" — and the system answers with a real plan, fires a webhook with the result, and persists everything behind a JWT-auth'd account.

It composes four AI building blocks plus production-grade engineering:

- **ML classifier** that labels destinations with one of six travel styles (Adventure / Relaxation / Culture / Budget / Luxury / Family)
- **RAG** over destination prose stored in pgvector
- **LangGraph agent** with three tools, two-model routing, and Pydantic-validated boundaries
- **React + FastAPI** stack with auth, persistence, and a real Discord/Slack webhook on completion

---

## 2. Architecture

```
                        ┌─────────────┐
                        │  React UI   │  Vite + React, SSE consumer
                        └──────┬──────┘
                               │ /api/*
                        ┌──────┴──────┐
                        │   FastAPI   │  async, Depends() everywhere
                        └──────┬──────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   LangGraph  │      │  Postgres +  │      │   Webhook    │
│    Agent     │◄────►│   pgvector   │      │ (Discord etc.)│
└──────┬───────┘      └──────────────┘      └──────────────┘
       │
       ├── Tool 1: rag_retrieve         (pgvector similarity + SQL filter)
       ├── Tool 2: classify_destination (joblib Pipeline + Open-Meteo for unseen)
       └── Tool 3: live_conditions      (Open-Meteo + Amadeus, TTL-cached)
```

---

## 3. Agent Flow

```
1. User free-text query
2. Cheap LLM (gemini-1.5-flash) → TripPreference (typed Pydantic)
3. Cheap LLM → rewrites query for vector search
4. rag_retrieve tool: SQL filter (style + cost) + pgvector similarity → 3-5 candidates
5. classify_destination tool (CONDITIONAL): fires only when
     (a) candidate is borderline,
     (b) user named a destination not in DB, or
     (c) confidence check on top pick
6. live_conditions tool: top 1-3 candidates only — weather + flights
7. Heavy LLM (gemini-2.5-flash) → synthesizes; MUST reconcile RAG vs live data,
   not concatenate
8. Stream response (SSE) to user
9. Background: persist AgentRun + ToolCalls; fire webhook
```

The synthesis node's contract: if RAG content and live API content disagree, the response surfaces both and recommends a resolution. "Concatenation" is a failure mode.

---

## 4. Model Routing

| Slot | Model | Used for |
|------|-------|----------|
| Cheap | `gemini-1.5-flash` | Intent parsing, tool argument extraction, RAG query rewriting |
| Heavy | `gemini-2.5-flash` | Final synthesis with cross-tool reconciliation |

SDK: `google-genai` (the unified Gemini SDK). Async via `client.aio.models.generate_content(...)`. Structured output via `response_mime_type="application/json"` + `response_schema=PydanticModel`.

Token usage is logged per step (structlog). Per-query cost goes in the README.

---

## 5. Schemas

### TripPreference (output of cheap LLM intent parse)

| Field | Type | Notes |
|-------|------|-------|
| `budget_usd` | `float \| None` | None when unspecified |
| `duration_days` | `int \| None` (gt=0) | "2 weeks" → 14 |
| `travel_month` | `int \| None` (1-12) | None when unspecified |
| `climate_pref` | `Literal["warm","cold","mild","any"]` | default `"any"` |
| `style_keywords` | `list[str]` | free-form: "hiking", "not touristy" |
| `group_type` | `Literal["solo","couple","family","group","unknown"]` | default `"unknown"` |

### DestinationFeatures (ML classifier input)

12 features, source-locked:

| Feature | Type | Source |
|---------|------|--------|
| `climate_zone` | categorical | Köppen raster |
| `avg_temp_peak_season_c` | numeric | Open-Meteo API |
| `peak_season_length_months` | numeric | judgment |
| `terrain_primary` | categorical | Wikipedia + judgment |
| `coastal_access` | binary 0/1 | map + judgment |
| `unesco_sites_count` | numeric | UNESCO CSV |
| `outdoor_activity_score` | ordinal 1-3 | judgment |
| `daily_cost_bucket` | ordinal 1-4 | BudgetYourTrip |
| `accommodation_range` | categorical | Booking / Wikivoyage |
| `visa_difficulty` | ordinal 1-3 | Passport Index |
| `english_prevalence` | ordinal 1-3 | EF EPI index |
| `tourism_maturity` | categorical | judgment |

Target: `travel_style` (one of the six styles). Single-label classification.

### Classifier role at runtime
- **Offline**: every destination in the DB has a hand-label (training target) and a stored predicted label.
- **Online tool**: fires when the agent reasons about a specific destination — input is `destination_name`; tool internally looks up features in the DB or fetches them live (Open-Meteo + UNESCO) for unseen destinations.

---

## 6. Engineering Standards (binding rules)

These come from the bootcamp's Engineering Standards companion guide. Every file written for this project must satisfy them.

### 6.1 Async All the Way Down

Every I/O hop in the request path is async. **Never** put `requests`, `time.sleep`, `joblib.load(...)` (per request), or any blocking call in an async route or tool.

```python
# ✓ correct
async with httpx.AsyncClient(timeout=10.0) as http:
    weather, flights = await asyncio.gather(
        http.get(WEATHER_URL),
        http.get(FLIGHTS_URL),
    )
response = await client.aio.models.generate_content(...)
```

CPU-bound work (heavy ML inference) goes through `asyncio.to_thread()`. The classifier's `predict()` is fast enough to run inline.

### 6.2 Dependency Injection — `Depends()`

Routes declare what they need. FastAPI wires it. Never instantiate clients inside route handlers.

```python
@router.post("/agent/query")
async def run_query(
    body: AgentQueryRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    cheap: genai.Client = Depends(get_cheap_llm),
):
    ...
```

In tests: `app.dependency_overrides[get_cheap_llm] = lambda: FakeClient()`.

### 6.3 Singletons in Lifespan

Engine, session factory, ML model, both Gemini clients — built once in the `@asynccontextmanager` lifespan handler, attached to `app.state`, disposed on shutdown.

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.engine = create_engine(settings)
    app.state.session_factory = make_session_factory(app.state.engine)
    app.state.cheap_client = build_gemini_client(settings.gemini_api_key)
    app.state.strong_client = build_gemini_client(settings.gemini_api_key)
    yield
    await app.state.engine.dispose()
```

Loading the joblib model on every request is a bug, not a style choice.

### 6.4 Caching

- `@lru_cache(maxsize=1)` on `get_settings()` — pure, deterministic, expensive (file IO + parsing).
- TTL caches (`cachetools.TTLCache`) on live tool responses where staleness within a window is acceptable. Document the TTL: "weather cached 10 minutes — same answer is fine for the same city in that window."
- A lock around the cache miss path to avoid the thundering herd.

### 6.5 Configuration — pydantic-settings

A single `Settings` class in `core/config.py`. `extra="forbid"` so a typo in `.env` raises at startup. **No `os.getenv` outside of this file.**

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="forbid")
    gemini_api_key: str = Field(..., min_length=1)
    database_url: str
    cheap_model: str = "gemini-1.5-flash"
    strong_model: str = "gemini-2.5-flash"
    ...
```

### 6.6 Pydantic at Every External Boundary

Every place data crosses into our process:

- HTTP request bodies → Pydantic on the route signature
- Tool inputs / outputs → Pydantic on the function signature
- LLM structured output → `response_schema=PydanticModel`
- Webhook payloads → Pydantic before the POST fires

Validate at the edge, **trust types inside.** No defensive `if isinstance(...)` 12 levels deep.

### 6.7 Errors, Retries, Failure Isolation

Three layers, all required:

1. **Timeouts** on every external call.
2. **Tenacity retries** with exponential backoff, retrying only transient errors (`httpx.TimeoutException`, `httpx.NetworkError`, 5xx). Never retry 4xx.
3. **Tool failures returned as `ToolError`** — a Pydantic model the LLM can read — never raised into the agent loop.

```python
async def live_conditions(city: str) -> WeatherResult | ToolError:
    try:
        return await fetch_weather(city)
    except httpx.HTTPStatusError as e:
        return ToolError(error=f"weather API {e.response.status_code}", retryable=False)
    except (httpx.TimeoutException, httpx.NetworkError) as e:
        return ToolError(error=f"weather unreachable: {e}", retryable=True)
```

Webhook failure is logged via structlog and swallowed — it must not affect the user-facing response.

### 6.8 Code Hygiene

- Project laid out by concern (already done — see file map below).
- `structlog` JSON logging only. No `print()`. No bare `logging.info(f"...")` either.
- `ruff` runs in pre-commit. Configuration in `pyproject.toml`:
  - `select = ["E","F","I","B","UP","ASYNC","S"]`
  - `line-length = 100`
  - `target-version = "py312"`

### 6.9 Tests — Critical Path

Three classes of tests, all wired to GitHub Actions:

1. **Pydantic schemas** — valid + invalid cases (`pytest.raises(ValidationError)`).
2. **Tools in isolation** — mock the LLM and external APIs (`monkeypatch`).
3. **One end-to-end** — full agent flow with all I/O mocked, asserts the right tools fired and the response shape is valid.

Test runtime stays under 10 seconds locally.

---

## 7. File Map

```
backend/
├── core/
│   ├── config.py        ← Settings (pydantic-settings, extra="forbid")
│   ├── security.py      ← bcrypt + JWT
│   ├── deps.py          ← Depends() functions for everything shared
│   └── webhook.py       ← async webhook delivery (later step)
├── schemas/
│   ├── auth.py          ← Register/Login/Token request-response schemas
│   ├── trip.py          ← TripPreference (cheap-LLM output)
│   ├── agent.py         ← AgentQueryRequest, AgentRunResponse, ToolCallRecord
│   ├── tools.py         ← Three tool I/O schemas + ToolError
│   └── webhook.py       ← Outbound webhook payload schema
├── agent/
│   ├── graph.py         ← LangGraph StateGraph (later step)
│   ├── prompts.py       ← System prompts for cheap and strong models
│   ├── router.py        ← Gemini client factory + parse_intent
│   └── tools/
│       ├── rag_retrieve.py
│       ├── classify_destination.py
│       └── live_conditions.py
├── db/
│   ├── engine.py        ← create_engine, make_session_factory (no globals)
│   ├── models.py        ← Six ORM tables
│   └── migrations/      ← Alembic
├── ml/
│   ├── notebook.ipynb   ← EDA + classifier training
│   ├── train.py         ← CLI: fits the winning Pipeline, saves model.joblib
│   ├── destinations.csv ← Hand-labeled dataset (~150 rows)
│   ├── results.csv      ← Experiment log
│   └── model.joblib     ← Saved Pipeline (loaded once via lifespan)
├── rag/
│   ├── loader.py        ← Wikivoyage / tourism-board scraping
│   ├── chunker.py       ← Configurable size + overlap
│   ├── embedder.py      ← Gemini text-embedding-004
│   └── store.py         ← Insert + similarity_search via pgvector
├── routers/
│   ├── auth.py          ← /auth/register, /auth/login
│   ├── agent.py         ← /agent/query (SSE), /agent/history
│   └── health.py        ← /health
├── tests/
│   ├── conftest.py
│   ├── test_schemas.py
│   ├── test_auth.py
│   ├── test_tools.py
│   └── test_agent.py
├── main.py              ← FastAPI app factory + lifespan
├── pyproject.toml       ← uv-managed (NEVER pip)
├── alembic.ini
└── Dockerfile

frontend/                ← Vite + React, SSE consumer (later step)
docker-compose.yml       ← postgres (pgvector) + backend + frontend
```

---

## 8. How to run (current state)

```bash
# install deps via uv (NEVER pip)
cd backend
uv sync

# start postgres (must be pgvector-enabled)
docker compose up postgres -d

# apply migrations
uv run alembic upgrade head

# run the API
uv run uvicorn main:app --reload

# tests
uv run pytest -q

# lint
uv run ruff check .
```

---

## 9. Hard Rules (do not violate)

- **uv only.** Never `pip install`. Never edit `requirements.txt` (it doesn't exist).
- **No globals.** Module-level `engine = create_async_engine(...)` is forbidden. Use lifespan + DI.
- **No `os.getenv`** outside `core/config.py`.
- **No `print()`.** Use `structlog`.
- **No `requests`** anywhere. Use `httpx.AsyncClient`.
- **No blocking I/O** in async functions. No `time.sleep`, no `joblib.load` per request.
- **No magic strings** for model names, URLs, keys. Settings only.
- **Validate at the edge.** Pydantic models on every route, every tool, every external boundary. No `if not isinstance(...)` defensive checks deep in business logic.
- **Tools return `ToolError`** on failure. They do not raise into the agent loop.
- **Webhook failure** is logged and swallowed. It does not affect the API response.
