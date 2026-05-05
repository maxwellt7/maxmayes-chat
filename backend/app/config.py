from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "chat-with-my-pinecone-backend"
    app_env: str = "development"
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/postgres"
    cors_allow_origin: str = "http://localhost:3000"

    clerk_secret_key: str = ""

    pinecone_api_key_1: str = ""
    pinecone_api_key_2: str = ""
    pinecone_api_key_3: str = ""

    openai_api_key: str = ""
    cohere_api_key: str = ""
    anthropic_api_key: str = ""

    voice_index_name: str = "max-copywriting-voice"
    voice_index_project: str = "1"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


settings = Settings()


def normalized_database_url() -> str:
    raw = settings.database_url.strip()
    if raw.startswith("postgresql://"):
        # Railway Postgres URLs are often emitted without SQLAlchemy driver suffix.
        return raw.replace("postgresql://", "postgresql+psycopg://", 1)
    return raw
