"""FastAPI application entry point.

Lifespan handler is the only place singletons are constructed:
  - the async DB engine + session factory
  - the cheap Gemini client (intent parse, RAG rewrite)
  - the strong Gemini client (final synthesis)

Everything is attached to `app.state` and exposed through dependencies in
`core.deps` — there are no module-level globals for shared resources.
"""

from __future__ import annotations

import logging
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agent.router import build_gemini_client
from core.config import get_settings
from db.engine import create_engine, make_session_factory
from routers import agent as agent_router
from routers import auth as auth_router
from routers import health as health_router
from routers import tools as tools_router


def _configure_logging() -> None:
    """Configure structlog to emit JSON, suitable for shipping to any log backend."""
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(level=logging.INFO, format="%(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN201 - FastAPI's expected signature
    """Build singletons on startup, dispose on shutdown."""
    _configure_logging()
    log = structlog.get_logger()

    settings = get_settings()
    log.info("lifespan.startup.begin")

    engine = create_engine(settings)
    session_factory = make_session_factory(engine)
    cheap_client = build_gemini_client(settings.gemini_api_key)
    strong_client = build_gemini_client(settings.gemini_api_key)

    # Load ML classifier if the model file exists (skipped in test environments
    # where train.py hasn't been run yet — get_classifier will return None).
    classifier = None
    model_file = Path(settings.model_path)
    if model_file.exists():
        classifier = joblib.load(model_file)
        log.info("lifespan.classifier.loaded", path=str(model_file))
    else:
        log.warning("lifespan.classifier.missing", path=str(model_file))

    app.state.settings = settings
    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.cheap_client = cheap_client
    app.state.strong_client = strong_client
    app.state.classifier = classifier

    log.info(
        "lifespan.startup.ready",
        cheap_model=settings.cheap_model,
        strong_model=settings.strong_model,
        classifier_loaded=classifier is not None,
    )

    try:
        yield
    finally:
        log.info("lifespan.shutdown.begin")
        await engine.dispose()
        log.info("lifespan.shutdown.complete")


def create_app() -> FastAPI:
    """Application factory — also useful for tests that want a fresh app."""
    settings = get_settings()
    app = FastAPI(
        title="Smart Travel Planner",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        tb = traceback.format_exc()
        log = structlog.get_logger()
        log.error(
            "unhandled_exception",
            path=str(request.url.path),
            method=request.method,
            error=str(exc),
            traceback=tb,
        )
        return JSONResponse(
            status_code=500,
            content={
                "detail": str(exc),
                "type": type(exc).__name__,
                "traceback": tb,
            },
        )

    app.include_router(health_router.router)
    app.include_router(auth_router.router)
    app.include_router(agent_router.router)
    app.include_router(tools_router.router)

    return app


app = create_app()
