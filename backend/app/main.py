from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
import app.models.chat  # noqa: F401
import app.models.index_registry  # noqa: F401
from app.models.schemas import HealthResponse
from app.routers import admin, chat as chat_router

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router.router)
app.include_router(admin.router)


@app.get("/healthz", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
async def healthcheck() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/healthz")
@app.post("/health")
async def healthcheck_post() -> HealthResponse:
    return HealthResponse(status="ok")
