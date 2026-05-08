from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.models import chat, index_registry  # noqa: F401 — registers with Base.metadata
from app.models.schemas import HealthResponse
from app.routers import admin, chat as chat_router

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.cors_allow_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router.router)
app.include_router(admin.router)


@app.get("/healthz", response_model=HealthResponse)
async def healthcheck() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/healthz")
async def healthcheck_post() -> HealthResponse:
    return HealthResponse(status="ok")
