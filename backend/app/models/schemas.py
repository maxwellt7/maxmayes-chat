import uuid

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


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
