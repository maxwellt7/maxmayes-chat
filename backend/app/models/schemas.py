import uuid

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


# The user's message is passed to the optimizer, the router, the verifier and
# the synthesizer, so its length is multiplied four times over before any
# retrieved context is added. Without a bound, a single request can be made to
# cost dollars rather than cents. 4000 characters is roughly 1000 tokens — far
# more than any real question, and cheap enough that the ceiling in
# `app.services.spend_guard` is the binding constraint rather than this.
MAX_CHAT_MESSAGE_CHARS = 4000


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_CHAT_MESSAGE_CHARS)
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()), max_length=255)


class ChatResponse(BaseModel):
    request_id: str
    reply: str


class IndexRegistryCreate(BaseModel):
    index_name: str
    project_id: str
    api_key_env_var: str
    dimension: int
    embedding_model: str
    metric: str
    domain_description: str
    sample_queries: list[str]
    namespaces: dict = {}
    is_active: bool = True


class IndexRegistryUpdate(BaseModel):
    index_name: str | None = None
    project_id: str | None = None
    api_key_env_var: str | None = None
    dimension: int | None = None
    embedding_model: str | None = None
    metric: str | None = None
    domain_description: str | None = None
    sample_queries: list[str] | None = None
    namespaces: dict | None = None
    is_active: bool | None = None


class IndexRegistryResponse(BaseModel):
    id: uuid.UUID
    index_name: str
    project_id: str
    api_key_env_var: str
    dimension: int
    embedding_model: str
    metric: str
    domain_description: str
    sample_queries: list[str]
    namespaces: dict
    is_active: bool

    model_config = {"from_attributes": True}


class DiscoveredIndex(BaseModel):
    index_name: str
    project_id: str
    dimension: int
    metric: str
    already_in_registry: bool


class AutoDescribeResult(BaseModel):
    domain_description: str
    sample_queries: list[str]
