import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import models as _models  # noqa: F401  — registers every model on Base.metadata
from app.config import settings
from app.core.errors import AppError, log_and_convert
from app.models.schemas import HealthResponse
from app.routers import admin, admin_audit, chat as chat_router
from app.security.owner_bootstrap import reconcile_owner_allowlist, warn_if_no_owner

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

_log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Reconcile the owner allowlist against existing accounts on every boot.

    This is what makes setting `OWNER_CLERK_USER_IDS` *after* the owner has
    already signed up recover the situation: the promotion no longer depends on
    the allowlist having been correct at the moment of first sign-in. Failure
    here is logged and does not prevent the app from serving, because refusing to
    boot over a bootstrap detail would turn a lockout into an outage.
    """
    from app.db.database import SessionLocal

    try:
        db = SessionLocal()
        try:
            promoted = reconcile_owner_allowlist(db)
            if promoted:
                _log.warning("owner_reconciliation promoted=%d", promoted)
            warn_if_no_owner(db)
        finally:
            db.close()
    except Exception:
        _log.error("owner_reconciliation_failed", exc_info=True)
    yield


# `docs_url` / `openapi_url` / `redoc_url` are off deliberately. The generated
# schema is a complete map of the admin surface — every path, method and payload
# shape — served unauthenticated. It is a reconnaissance aid and nothing else in
# production; the schema is still reachable in-process for tests via
# `app.openapi()`.
app = FastAPI(
    title=settings.app_name,
    lifespan=lifespan,
    docs_url=None,
    openapi_url=None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(AppError)
async def app_error_handler(_request: Request, exc: AppError) -> JSONResponse:
    """Render the taxonomy's errors, and nothing else, to the client."""
    client = log_and_convert(exc, context="request")
    headers = {}
    retry_after = getattr(exc, "retry_after", None)
    if retry_after:
        headers["Retry-After"] = str(retry_after)
    return JSONResponse(
        status_code=client.status_code, content=client.payload(), headers=headers
    )


app.include_router(chat_router.router)
app.include_router(admin.router)
app.include_router(admin_audit.router)


@app.get("/healthz", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
async def healthcheck() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/healthz")
@app.post("/health")
async def healthcheck_post() -> HealthResponse:
    return HealthResponse(status="ok")
