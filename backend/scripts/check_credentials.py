"""Verify every configured credential actually works.

Prints a pass/fail line per provider and never echoes a secret. Run with:

    ./.venv/bin/python scripts/check_credentials.py
"""
from __future__ import annotations

import os
import socket
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

OK = "PASS"
BAD = "FAIL"
SKIP = "SKIP"

results: list[tuple[str, str, str]] = []


def record(name: str, status: str, detail: str) -> None:
    results.append((name, status, detail))
    print(f"{status:4}  {name:22}  {detail}", flush=True)


def check_openai() -> None:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        return record("OPENAI_API_KEY", SKIP, "not set")
    try:
        from openai import OpenAI

        client = OpenAI(api_key=key, timeout=30)
        vec = client.embeddings.create(
            model="text-embedding-3-small", input="connectivity probe"
        )
        dim = len(vec.data[0].embedding)
        record("OPENAI_API_KEY", OK, f"embeddings work, dim={dim}")
    except Exception as exc:
        record("OPENAI_API_KEY", BAD, f"{type(exc).__name__}: {str(exc)[:160]}")


def check_cohere() -> None:
    key = os.getenv("COHERE_API_KEY", "")
    if not key:
        return record("COHERE_API_KEY", SKIP, "not set")
    try:
        import cohere

        client = cohere.ClientV2(api_key=key)
        resp = client.embed(
            texts=["connectivity probe"],
            model="embed-english-v3.0",
            input_type="search_query",
            embedding_types=["float"],
        )
        dim = len(resp.embeddings.float_[0])
        record("COHERE_API_KEY", OK, f"embeddings work, dim={dim}")
    except Exception as exc:
        record("COHERE_API_KEY", BAD, f"{type(exc).__name__}: {str(exc)[:160]}")


def check_anthropic() -> None:
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        return record(
            "ANTHROPIC_API_KEY", SKIP, "not set - synthesis cannot run without it"
        )
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=key, timeout=30)
        client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=8,
            messages=[{"role": "user", "content": "say ok"}],
        )
        record("ANTHROPIC_API_KEY", OK, "messages API works")
    except Exception as exc:
        record("ANTHROPIC_API_KEY", BAD, f"{type(exc).__name__}: {str(exc)[:160]}")


def check_pinecone() -> None:
    for slot in ("1", "2", "3"):
        key = os.getenv(f"PINECONE_API_KEY_{slot}", "")
        name = f"PINECONE_API_KEY_{slot}"
        if not key:
            record(name, SKIP, "not set")
            continue
        try:
            from pinecone import Pinecone

            pc = Pinecone(api_key=key)
            names = [i["name"] for i in pc.list_indexes()]
            preview = ", ".join(sorted(names)[:6])
            more = f" (+{len(names) - 6} more)" if len(names) > 6 else ""
            record(name, OK, f"{len(names)} indexes: {preview}{more}")
        except Exception as exc:
            record(name, BAD, f"{type(exc).__name__}: {str(exc)[:160]}")


def check_clerk() -> None:
    # httpx rather than urllib: api.clerk.com sits behind Cloudflare, which
    # rejects the default Python-urllib user agent with error 1010 and looks
    # exactly like an auth failure.
    import httpx

    issuer = os.getenv("CLERK_ISSUER", "")
    if issuer:
        url = f"{issuer.rstrip('/')}/.well-known/jwks.json"
        try:
            resp = httpx.get(url, timeout=15)
            n = len(resp.json().get("keys", []))
            record("CLERK_ISSUER (JWKS)", OK if n else BAD, f"{n} signing key(s)")
        except Exception as exc:
            record("CLERK_ISSUER (JWKS)", BAD, f"{type(exc).__name__}: {str(exc)[:160]}")
    else:
        record("CLERK_ISSUER (JWKS)", SKIP, "not set")

    secret = os.getenv("CLERK_SECRET_KEY", "")
    if not secret:
        return record("CLERK_SECRET_KEY", SKIP, "not set")
    try:
        resp = httpx.get(
            "https://api.clerk.com/v1/users?limit=100",
            headers={"Authorization": f"Bearer {secret}"},
            timeout=20,
        )
        resp.raise_for_status()
        users = resp.json()
        record("CLERK_SECRET_KEY", OK, f"backend API works, {len(users)} user(s)")
    except Exception as exc:
        record("CLERK_SECRET_KEY", BAD, f"{type(exc).__name__}: {str(exc)[:160]}")


def check_database() -> None:
    url = os.getenv("DATABASE_URL", "")
    if not url:
        return record("DATABASE_URL", SKIP, "not set")

    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = parsed.hostname or ""
    port = parsed.port or 5432

    try:
        socket.getaddrinfo(host, port)
    except socket.gaierror:
        return record(
            "DATABASE_URL",
            BAD,
            f"host '{host}' does not resolve - this is a Railway-internal "
            f"address, reachable only from inside Railway",
        )

    try:
        import psycopg

        driver_url = url.replace("postgresql+psycopg://", "postgresql://", 1)
        with psycopg.connect(driver_url, connect_timeout=15) as conn:
            with conn.cursor() as cur:
                cur.execute("select current_database(), version()")
                db, version = cur.fetchone()
        record("DATABASE_URL", OK, f"connected to '{db}' ({version.split(',')[0]})")
    except Exception as exc:
        record("DATABASE_URL", BAD, f"{type(exc).__name__}: {str(exc)[:160]}")


if __name__ == "__main__":
    print("Checking credentials (no secret values are printed)\n")
    check_openai()
    check_cohere()
    check_anthropic()
    check_pinecone()
    check_clerk()
    check_database()

    failed = [r for r in results if r[1] == BAD]
    skipped = [r for r in results if r[1] == SKIP]
    print(
        f"\n{len(results) - len(failed) - len(skipped)} passed, "
        f"{len(failed)} failed, {len(skipped)} not set"
    )
    sys.exit(1 if failed else 0)
