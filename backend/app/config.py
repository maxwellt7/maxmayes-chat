from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "chat-with-my-pinecone-backend"
    app_env: str = "development"
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/postgres"
    cors_allow_origin: str = "http://localhost:3000"

    @property
    def cors_origins(self) -> list[str]:
        return [
            o.strip()
            for o in self.cors_allow_origin.split(",")
            if o.strip()
        ]

    clerk_secret_key: str = ""

    # Clerk instance issuer, e.g. "https://clerk.example.com" or
    # "https://your-app.clerk.accounts.dev". The JWKS URL is derived from it
    # unless one is supplied explicitly.
    clerk_issuer: str = ""
    clerk_jwks_url: str = ""
    # Optional PEM public key for fully networkless verification. Takes
    # precedence over the JWKS URL when set.
    clerk_jwt_public_key: str = ""
    # Comma-separated frontend origins permitted to mint tokens for this API.
    # Checked against the token's `azp` claim.
    clerk_authorized_parties: str = ""

    # Bootstrap allowlist for the owner role, comma-separated Clerk user IDs
    # and/or email addresses. Accounts matching these are promoted to `owner`
    # on first sight; everyone else defaults to `member`.
    owner_clerk_user_ids: str = ""
    owner_emails: str = ""

    @property
    def resolved_clerk_jwks_url(self) -> str:
        if self.clerk_jwks_url:
            return self.clerk_jwks_url
        if self.clerk_issuer:
            return f"{self.clerk_issuer.rstrip('/')}/.well-known/jwks.json"
        return ""

    @property
    def authorized_parties(self) -> list[str]:
        return [p.strip() for p in self.clerk_authorized_parties.split(",") if p.strip()]

    @property
    def owner_user_id_allowlist(self) -> set[str]:
        return {v.strip() for v in self.owner_clerk_user_ids.split(",") if v.strip()}

    @property
    def owner_email_allowlist(self) -> set[str]:
        return {
            v.strip().lower() for v in self.owner_emails.split(",") if v.strip()
        }

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() in {"production", "prod"}

    pinecone_api_key_1: str = ""
    pinecone_api_key_2: str = ""
    pinecone_api_key_3: str = ""

    openai_api_key: str = ""
    cohere_api_key: str = ""
    anthropic_api_key: str = ""

    voice_index_name: str = "max-copywriting-voice"
    voice_index_project: str = "1"
    voice_index_dimension: int = 1536

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
