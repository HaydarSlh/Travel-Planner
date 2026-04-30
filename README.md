# Smart Travel Planner

A full-stack AI travel planner. Ask it "two weeks in July, $2,000 budget, warm beach, not touristy" and it returns a structured destination guide — with weather, costs, things to do, and a live Slack notification when the plan is ready.

---

## What it does

1. **Parses your intent** — a cheap LLM (Gemini Flash Lite) extracts a typed `TripPreference` from your free-text query.
2. **Predicts travel styles** — a second LLM call classifies your trip into 1–2 styles (Adventure / Relaxation / Culture / Budget / Luxury / Family), keeping both only when their confidence scores are within 0.15 of each other.
3. **Retrieves destination knowledge** — pgvector similarity search (Gemini embeddings) pulls up to 2 destinations × 3 chunks per matched style from a curated travel-guide database.
4. **Fetches live conditions** — Open-Meteo climate normals and optionally Amadeus flight offers for the top destinations.
5. **Synthesises a plan** — a strong LLM (Gemini Flash) streams the final structured markdown recommendation token-by-token to the browser.
6. **Fires a Slack webhook** — a Block Kit notification lands in your channel once the run is persisted.

---

## Architecture

```
React (Vite + Tailwind)
        │  SSE stream + REST
        ▼
FastAPI (async, Depends() DI)
        │
        ├── LangGraph Agent
        │     ├── parse_intent        cheap LLM → TripPreference
        │     ├── predict_styles      strong LLM → style list + confidence
        │     ├── rag_retrieve        pgvector similarity, style-filtered
        │     ├── live_conditions     Open-Meteo + Amadeus (TTL cached)
        │     └── synthesise          strong LLM, streamed
        │
        ├── Postgres + pgvector       destinations, chunks, runs, users
        ├── ML classifier             joblib LightGBM pipeline (offline labels)
        └── Slack webhook             fire-and-forget, tenacity retries
```

---

## Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 19, Vite 5, Tailwind CSS v4, ReactMarkdown |
| Backend | FastAPI, LangGraph, SQLAlchemy async, Alembic |
| LLM | Google Gemini (`gemini-2.5-flash-lite` / `gemini-2.5-flash`) |
| Embeddings | `gemini-embedding-001` (3072-dim) |
| Vector DB | PostgreSQL 16 + pgvector |
| ML | scikit-learn + LightGBM, joblib |
| Package manager | uv (never pip) |
| Containers | Docker Compose (postgres, backend, frontend/Nginx) |

---

## Quick start

### Prerequisites

- Docker Desktop (or Docker Engine + Compose plugin)
- A [Google AI Studio](https://aistudio.google.com/) API key
- (Optional) A Slack Incoming Webhook URL

### 1. Configure environment

```bash
cp backend/.env.example backend/.env   # if the example exists, otherwise edit .env directly
```

Minimum required values in `backend/.env`:

```env
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/travel_planner
JWT_SECRET=<at-least-32-random-characters>
GEMINI_API_KEY=<your-key>

# Optional — omit to skip Slack notifications
WEBHOOK_URL=https://hooks.slack.com/services/T.../B.../xxx
```

All other settings have sensible defaults (see `backend/core/config.py`).

### 2. Start the full stack

```bash
docker compose up --build
```

- Frontend → http://localhost:3000
- Backend API → http://localhost:8000
- API docs → http://localhost:8000/docs

First boot runs `alembic upgrade head` automatically inside the backend container.

### 3. Load destination data (one-time)

After the stack is up, populate the vector store:

```bash
# Inside the running backend container
docker compose exec backend python -m rag.loader
```

### 4. (Optional) Train the ML classifier

```bash
docker compose exec backend python -m ml.train
```

This fits three pipelines on `ml/destinations.csv`, tunes LightGBM, and saves `ml/model.joblib` + `ml/model_meta.json`. The `ml/` directory is volume-mounted so the model persists across container rebuilds.

---

## Local development (without Docker)

```bash
# Postgres only
docker compose up postgres -d

# Backend
cd backend
uv sync
uv run alembic upgrade head
uv run uvicorn main:app --reload

# Frontend (separate terminal)
cd frontend
npm install
npm run dev          # http://localhost:5173
```

Tests:

```bash
cd backend
uv run pytest -q     # 65 tests, ~5 s
uv run ruff check .
```

---

## ML Classifier

Trained on 205 hand-labelled destinations using 12 features (no destination name — avoids identifier leakage). Three models compared with 5-fold stratified cross-validation:

| Model | CV Accuracy | CV Macro F1 |
|-------|-------------|-------------|
| Logistic Regression | 96.56% ± 1.97% | 95.55% ± 3.08% |
| Random Forest | 98.01% ± 2.91% | 97.20% ± 4.69% |
| LightGBM (baseline) | 97.54% ± 2.21% | 95.34% ± 4.69% |
| **LightGBM (tuned)** | **99.44% ± 0.00%** | **99.44% ± 0.00%** |

Winner: **tuned LightGBM** (`num_leaves=63`, `max_depth=4`, `learning_rate=0.05`, `n_estimators=100`, `reg_alpha=0.0`). Tuned via `RandomizedSearchCV(n_iter=30, scoring="f1_macro")`.

Per-class F1 (tuned model): Adventure 1.00 · Budget 1.00 · Culture 0.91 · Family 0.98 · Luxury 1.00 · Relaxation 1.00. Culture's slight dip is expected — it overlaps with Luxury at the high end and with Budget at the low end; more labelled examples in the middle band would close the gap.

---

## Per-query cost

Measured on a representative query ("two weeks in July, $2,000, warm beach, not touristy"):

| Step | Model | Input tokens | Output tokens | Est. cost |
|------|-------|-------------|---------------|-----------|
| Intent parse | gemini-2.5-flash-lite | ~450 | ~80 | $0.00003 |
| Style predict | gemini-2.5-flash | ~600 | ~30 | $0.00015 |
| Synthesis | gemini-2.5-flash | ~3,500 | ~700 | $0.00175 |
| **Total** | | **~4,550** | **~810** | **~$0.002** |

Embedding calls (RAG query rewrite) add ~$0.00001. The biggest lever is synthesis prompt length — trimming RAG chunks from 15 to 6 per style roughly halves the input tokens.

---

## API reference

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/auth/register` | — | Create account |
| POST | `/auth/login` | — | Get JWT |
| POST | `/agent/query` | Bearer | Run agent, return full JSON |
| POST | `/agent/query/stream` | Bearer | Run agent, stream SSE tokens |
| GET | `/agent/history` | Bearer | Last 20 runs |
| POST | `/tools/classify` | Bearer | Run ML classifier directly |
| POST | `/tools/rag` | Bearer | Run RAG retrieval directly |
| GET | `/health` | — | Liveness check |

Interactive docs: http://localhost:8000/docs

---

## Slack webhook format

Each completed run fires a Block Kit message:

```
✈️ New Travel Plan Generated
──────────────────────────────────────────
User ID   | T00000…   Run ID    | 3fa85f…
Styles    | Adventure, Budget   Destinations | Kyrgyzstan, Morocco
Tokens    | in 4550 / out 810   Timestamp  | 2026-04-30 18:42 UTC
──────────────────────────────────────────
Query
> Two weeks in July, $2,000 budget, warm beach, not touristy

Answer preview
# Travel Recommendation
## Style: Budget…

Tools used
• rag_retrieve completed in 412ms
• live_conditions completed in 890ms
```

Set `WEBHOOK_URL` in `.env` to enable. Leave it blank to skip silently.

---

## Project structure

```
backend/
├── agent/          LangGraph graph, prompts, tools
├── core/           config, security, deps, webhook
├── db/             ORM models, engine, migrations
├── ml/             classifier notebook, train.py, model artifacts
├── rag/            loader, chunker, embedder, pgvector store
├── routers/        auth, agent (SSE + JSON), tools, health
├── schemas/        Pydantic models for every boundary
└── tests/          65 tests, all under 6 s

frontend/
├── src/
│   ├── components/ Chat (bubbles, cards, tool trace), Layout
│   ├── context/    AuthContext (in-memory JWT)
│   ├── hooks/      useStream (SSE), useAuth
│   ├── lib/        api.ts (fetch helpers)
│   └── pages/      LoginPage, PlannerPage
└── nginx.conf      /api proxy + React Router fallback
```
