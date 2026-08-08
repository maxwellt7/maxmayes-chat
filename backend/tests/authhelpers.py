"""Helpers for testing the real auth path end to end.

An RSA keypair is generated in-process and its public half is served as a JWKS
document from a loopback HTTP server. The app is pointed at that URL through the
same `CLERK_JWKS_URL` setting production uses, so tests exercise key fetching,
`kid` selection, signature verification, issuer and `azp` checks — the whole
chain — with no Clerk credentials and no network.

This is a stronger test than one run against a real Clerk instance: a real
instance will not mint an `alg:none` token, a token signed by the wrong key, or a
token that expired an hour ago. Those have to be forged locally, which is exactly
what an attacker does.
"""
from __future__ import annotations

import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

ISSUER = "https://test-instance.clerk.accounts.dev"
FRONTEND = "https://chat.example.test"


def b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _b64url_uint(value: int) -> str:
    length = (value.bit_length() + 7) // 8
    return b64url(value.to_bytes(length, "big"))


class KeyPair:
    """An RSA keypair plus the JWKS entry that describes its public half."""

    def __init__(self, kid: str) -> None:
        self.kid = kid
        self._key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.private_pem = self._key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode()
        self.public_pem = (
            self._key.public_key()
            .public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            .decode()
        )

    def jwk(self) -> dict:
        numbers = self._key.public_key().public_numbers()
        return {
            "kty": "RSA",
            "use": "sig",
            "alg": "RS256",
            "kid": self.kid,
            "n": _b64url_uint(numbers.n),
            "e": _b64url_uint(numbers.e),
        }

    def sign(self, claims: dict, algorithm: str = "RS256") -> str:
        return jwt.encode(
            claims, self.private_pem, algorithm=algorithm, headers={"kid": self.kid}
        )


class JWKSServer:
    """Serves a JWKS document at `/.well-known/jwks.json` on a loopback port."""

    def __init__(self, keys: list[KeyPair]) -> None:
        body = json.dumps({"keys": [k.jwk() for k in keys]}).encode()

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 — stdlib naming
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args):
                pass

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> JWKSServer:
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._server.shutdown()
        self._server.server_close()

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/.well-known/jwks.json"


def claims(
    subject: str = "user_legit",
    *,
    issuer: str = ISSUER,
    azp: str | None = FRONTEND,
    lifetime: int = 60,
    issued_offset: int = 0,
    **extra,
) -> dict:
    now = int(time.time()) + issued_offset
    payload = {
        "sub": subject,
        "iss": issuer,
        "sid": "sess_test",
        "iat": now,
        "nbf": now,
        "exp": now + lifetime,
    }
    if azp is not None:
        payload["azp"] = azp
    payload.update(extra)
    return payload


def unsigned_token(payload: dict) -> str:
    """A `header.payload.` token with `alg: none` — the original exploit shape."""
    header = b64url(json.dumps({"alg": "none", "typ": "JWT"}).encode())
    body = b64url(json.dumps(payload).encode())
    return f"{header}.{body}."


def shaped_but_unsigned_token(payload: dict) -> str:
    """`hdr.<base64 payload>.sig` — literally what the deleted middleware, and
    the tests that certified it, accepted as proof of identity."""
    return f"hdr.{b64url(json.dumps(payload).encode())}.sig"


def bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}
