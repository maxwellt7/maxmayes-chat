"""Cryptographic verification of Clerk session tokens.

Clerk signs session JWTs with RS256 using the instance's private key. The
matching public key is published as a JWKS document. Verification is therefore
networkless after the first key fetch, and fully offline when a PEM is supplied
directly via ``CLERK_JWT_PUBLIC_KEY``.

Nothing in this module trusts any part of a token before the signature checks
out. That is the whole point of it existing.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import jwt
from jwt import PyJWKClient

_log = logging.getLogger(__name__)

# Clerk signs session tokens with RS256. Accepting anything else — in particular
# `none` or an HMAC algorithm — would let a caller sign tokens with a key we
# have published.
_ALLOWED_ALGORITHMS = ["RS256"]

# Tolerance for clock drift between Clerk's servers and ours, in seconds.
# Clerk's own SDKs default to 5s.
_CLOCK_SKEW_SECONDS = 5

# Clerk session tokens are short-lived (60s) and refreshed by the frontend SDK.
# The JWKS itself rotates rarely; an hour of caching is safe and keeps us off
# the network on the hot path.
_JWKS_CACHE_SECONDS = 3600


class TokenVerificationError(Exception):
    """Raised whenever a token cannot be trusted, for any reason.

    Deliberately carries no provider detail: the message is surfaced to
    callers, and distinguishing "expired" from "bad signature" from "unknown
    key" is useful to an attacker probing the endpoint.
    """


@dataclass(frozen=True)
class VerifiedToken:
    """The claims of a token whose signature has been verified."""

    subject: str
    session_id: str | None
    issuer: str | None
    authorized_party: str | None
    expires_at: int | None
    claims: dict[str, Any] = field(default_factory=dict)


class ClerkTokenVerifier:
    """Verifies Clerk session tokens against a JWKS or a static public key.

    Thread-safe. A single instance is shared across the process; the underlying
    ``PyJWKClient`` maintains its own key cache.
    """

    def __init__(
        self,
        *,
        jwks_url: str | None = None,
        public_key_pem: str | None = None,
        issuer: str | None = None,
        authorized_parties: list[str] | None = None,
    ) -> None:
        if not jwks_url and not public_key_pem:
            raise ValueError(
                "ClerkTokenVerifier requires either a JWKS URL or a public key PEM"
            )
        self._jwks_url = jwks_url
        self._public_key_pem = public_key_pem
        self._issuer = issuer
        self._authorized_parties = authorized_parties or []
        self._lock = threading.Lock()
        self._jwk_client: PyJWKClient | None = None
        self._jwk_client_created_at = 0.0

    def _get_signing_key(self, token: str) -> Any:
        if self._public_key_pem:
            return self._public_key_pem

        with self._lock:
            expired = (
                time.monotonic() - self._jwk_client_created_at > _JWKS_CACHE_SECONDS
            )
            if self._jwk_client is None or expired:
                self._jwk_client = PyJWKClient(
                    self._jwks_url,  # type: ignore[arg-type]
                    cache_keys=True,
                    lifespan=_JWKS_CACHE_SECONDS,
                )
                self._jwk_client_created_at = time.monotonic()
            client = self._jwk_client

        return client.get_signing_key_from_jwt(token).key

    def verify(self, token: str) -> VerifiedToken:
        """Verify a token and return its claims, or raise TokenVerificationError."""
        try:
            signing_key = self._get_signing_key(token)
        except Exception as exc:
            # A JWKS fetch failure is an availability problem on our side, not a
            # caller error, so it is worth logging loudly. We still refuse the
            # request: failing open here would defeat the entire module.
            _log.error("Unable to resolve Clerk signing key: %s", exc)
            raise TokenVerificationError("Unable to verify token") from exc

        try:
            claims = jwt.decode(
                token,
                signing_key,
                algorithms=_ALLOWED_ALGORITHMS,
                issuer=self._issuer if self._issuer else None,
                options={
                    "require": ["exp", "sub"],
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_nbf": True,
                    "verify_iat": True,
                    "verify_iss": bool(self._issuer),
                    # Clerk session tokens carry no `aud` claim by default.
                    "verify_aud": False,
                },
                leeway=_CLOCK_SKEW_SECONDS,
            )
        except jwt.PyJWTError as exc:
            _log.info("Rejected token: %s", type(exc).__name__)
            raise TokenVerificationError("Invalid or expired token") from exc

        # `azp` identifies the frontend origin the token was minted for. Checking
        # it prevents a token issued to some other application on the same Clerk
        # instance from being replayed against this API.
        azp = claims.get("azp")
        if self._authorized_parties and azp is not None:
            if azp not in self._authorized_parties:
                _log.warning("Rejected token with unauthorized azp: %s", azp)
                raise TokenVerificationError("Invalid or expired token")

        subject = claims.get("sub")
        if not isinstance(subject, str) or not subject:
            raise TokenVerificationError("Invalid or expired token")

        return VerifiedToken(
            subject=subject,
            session_id=claims.get("sid"),
            issuer=claims.get("iss"),
            authorized_party=azp,
            expires_at=claims.get("exp"),
            claims=claims,
        )
